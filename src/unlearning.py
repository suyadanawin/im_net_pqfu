import os
from copy import deepcopy
from typing import Dict, List, Tuple

from src.fl_client import FLClient
from src.fl_server import fedavg_aggregate
from src.metrics import (
    add_round_column,
    last_layer_distance,
    model_l2_distance,
    model_update_norm,
    per_layer_l2_distance,
)
from src.models import get_model
from src.utils import evaluate_model, print_round_summary, save_checkpoint


def run_warmstart_unlearning(
    train_dataset,
    val_dataset,
    client_to_indices: Dict[int, List[int]],
    forget_client_id: int,
    full_model_state: Dict,
    oracle_round_states: Dict[int, Dict],
    oracle_final_state: Dict,
    config: dict,
    device: str
) -> Tuple[object, Dict, List[Dict], List[Dict], List[Dict]]:
    remaining_clients = {cid: idxs for cid, idxs in client_to_indices.items() if cid != forget_client_id}

    model_cfg = {
        "name": config["model"]["name"],
        "num_classes": config["dataset"]["num_classes"],
        "pretrained": config["model"]["pretrained"]
    }

    unlearn_train_cfg = {
        "local_epochs": config["unlearning"]["local_epochs"],
        "batch_size": config["unlearning"]["batch_size"],
        "lr": config["unlearning"]["lr"],
        "momentum": config["unlearning"]["momentum"],
        "weight_decay": config["unlearning"]["weight_decay"]
    }

    global_model = get_model(
        model_name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"]
    )
    global_model.load_state_dict(deepcopy(full_model_state))
    global_model.to(device)

    round_metrics = []
    per_layer_rows = []
    last_layer_rows = []

    best_acc = -1.0
    best_round = -1
    best_state = None

    prev_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    initial_dist_to_oracle_final = model_l2_distance(prev_state, oracle_final_state)

    for round_idx in range(1, config["unlearning"]["correction_rounds"] + 1):
        global_state = deepcopy(global_model.state_dict())
        client_results = []

        for client_id, train_indices in remaining_clients.items():
            client = FLClient(
                client_id=client_id,
                train_dataset=train_dataset,
                train_indices=train_indices,
                model_cfg=model_cfg,
                train_cfg=unlearn_train_cfg,
                dataloader_cfg=config["dataloader"],
                device=device
            )
            result = client.train(global_model_state=global_state)
            client_results.append(result)

        aggregated_state = fedavg_aggregate(client_results)
        global_model.load_state_dict(aggregated_state)

        current_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        val_metrics = evaluate_model(
            model=global_model,
            dataset=val_dataset,
            batch_size=config["unlearning"]["eval_batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
            device=device
        )

        print_round_summary("Warm-Start Unlearning", round_idx, client_results, val_metrics)

        avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
        avg_client_acc = sum(r["train_acc"] for r in client_results) / len(client_results)

        oracle_round_state = oracle_round_states.get(round_idx, oracle_final_state)

        dist_unlearn_to_oracle_round = model_l2_distance(current_state, oracle_round_state)
        dist_unlearn_to_oracle_final = model_l2_distance(current_state, oracle_final_state)
        dist_unlearn_to_full = model_l2_distance(current_state, full_model_state)
        unlearn_update_norm = model_update_norm(prev_state, current_state)

        if initial_dist_to_oracle_final > 0:
            distance_reduction_ratio = (
                (initial_dist_to_oracle_final - dist_unlearn_to_oracle_final) / initial_dist_to_oracle_final
            )
        else:
            distance_reduction_ratio = 0.0

        row = {
            "round": round_idx,
            "avg_client_train_loss": avg_client_loss,
            "avg_client_train_acc": avg_client_acc,
            "unlearn_val_loss": val_metrics["loss"],
            "unlearn_val_acc": val_metrics["acc"],
            "unlearn_to_oracle_round_l2": dist_unlearn_to_oracle_round,
            "unlearn_to_oracle_final_l2": dist_unlearn_to_oracle_final,
            "unlearn_to_full_l2": dist_unlearn_to_full,
            "unlearn_update_norm": unlearn_update_norm,
            "distance_reduction_ratio_to_oracle_final": distance_reduction_ratio,
        }
        round_metrics.append(row)

        if config["tracking"]["save_per_layer_every_round"]:
            rows1 = per_layer_l2_distance(current_state, oracle_round_state)
            rows2 = per_layer_l2_distance(current_state, oracle_final_state)
            rows3 = per_layer_l2_distance(current_state, full_model_state)
            per_layer_rows.extend(add_round_column(rows1, round_idx, "unlearn_vs_oracle_round"))
            per_layer_rows.extend(add_round_column(rows2, round_idx, "unlearn_vs_oracle_final"))
            per_layer_rows.extend(add_round_column(rows3, round_idx, "unlearn_vs_full"))

        if config["tracking"]["save_last_layer_every_round"]:
            last1 = last_layer_distance(current_state, oracle_round_state)
            last1["round"] = round_idx
            last1["tag"] = "unlearn_vs_oracle_round"
            last_layer_rows.append(last1)

            last2 = last_layer_distance(current_state, oracle_final_state)
            last2["round"] = round_idx
            last2["tag"] = "unlearn_vs_oracle_final"
            last_layer_rows.append(last2)

            last3 = last_layer_distance(current_state, full_model_state)
            last3["round"] = round_idx
            last3["tag"] = "unlearn_vs_full"
            last_layer_rows.append(last3)

        if config["unlearning"]["save_every_round"]:
            save_checkpoint(
                {
                    "round": round_idx,
                    "model_state_dict": current_state,
                    "val_metrics": val_metrics,
                    "forget_client_id": forget_client_id
                },
                os.path.join(config["paths"]["ckpt_dir"], f"unlearn_round_{round_idx}.pt")
            )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_round = round_idx
            best_state = {k: v.clone() for k, v in current_state.items()}
            save_checkpoint(
                {
                    "round": round_idx,
                    "model_state_dict": current_state,
                    "val_metrics": val_metrics,
                    "forget_client_id": forget_client_id
                },
                os.path.join(config["paths"]["ckpt_dir"], "unlearn_best.pt")
            )

        prev_state = {k: v.clone() for k, v in current_state.items()}

    final_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    save_checkpoint(
        {
            "round": config["unlearning"]["correction_rounds"],
            "model_state_dict": final_state,
            "best_val_acc": best_acc,
            "best_round": best_round,
            "forget_client_id": forget_client_id
        },
        os.path.join(config["paths"]["ckpt_dir"], "unlearn_final.pt")
    )

    summary = {
        "forget_client_id": forget_client_id,
        "best_val_acc": best_acc,
        "best_round": best_round,
        "correction_rounds": config["unlearning"]["correction_rounds"],
        "initial_dist_to_oracle_final": initial_dist_to_oracle_final,
    }

    return global_model, summary, round_metrics, per_layer_rows, last_layer_rows