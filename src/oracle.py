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


def build_remaining_clients(client_to_indices: Dict[int, List[int]], forget_client_id: int) -> Dict[int, List[int]]:
    return {cid: idxs for cid, idxs in client_to_indices.items() if cid != forget_client_id}


def run_oracle_retraining(
    train_dataset,
    val_dataset,
    client_to_indices: Dict[int, List[int]],
    forget_client_id: int,
    full_model_state: Dict,
    config: dict,
    device: str
) -> Tuple[object, Dict, List[Dict], List[Dict], List[Dict]]:
    remaining_clients = build_remaining_clients(client_to_indices, forget_client_id)

    model_cfg = {
        "name": config["model"]["name"],
        "num_classes": config["dataset"]["num_classes"],
        "pretrained": config["model"]["pretrained"]
    }

    global_model = get_model(
        model_name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"]
    )
    global_model.to(device)

    round_metrics = []
    per_layer_rows = []
    last_layer_rows = []

    best_acc = -1.0
    best_round = -1
    best_state = None

    prev_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    # Build oracle-specific config by overriding FL training settings
    oracle_config = deepcopy(config)
    oracle_config["fl"]["local_epochs"] = config["oracle"]["local_epochs"]
    oracle_config["fl"]["batch_size"] = config["oracle"]["batch_size"]
    oracle_config["fl"]["lr"] = config["oracle"]["lr"]
    oracle_config["fl"]["momentum"] = config["oracle"]["momentum"]
    oracle_config["fl"]["weight_decay"] = config["oracle"]["weight_decay"]

    for round_idx in range(1, config["oracle"]["rounds"] + 1):
        global_state = deepcopy(global_model.state_dict())
        client_results = []

        for client_id, train_indices in remaining_clients.items():
            client = FLClient(
                client_id=client_id,
                train_dataset=train_dataset,
                train_indices=train_indices,
                config=oracle_config,
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
            batch_size=config["oracle"]["eval_batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
            device=device
        )

        print_round_summary("Oracle", round_idx, client_results, val_metrics)

        avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
        avg_client_acc = sum(r["train_acc"] for r in client_results) / len(client_results)

        dist_oracle_to_full = model_l2_distance(current_state, full_model_state)
        oracle_update_norm = model_update_norm(prev_state, current_state)

        row = {
            "round": round_idx,
            "avg_client_train_loss": avg_client_loss,
            "avg_client_train_acc": avg_client_acc,
            "oracle_val_loss": val_metrics["loss"],
            "oracle_val_acc": val_metrics["acc"],
            "oracle_to_full_l2": dist_oracle_to_full,
            "oracle_update_norm": oracle_update_norm,
        }
        round_metrics.append(row)

        if config["tracking"]["save_per_layer_every_round"]:
            rows = per_layer_l2_distance(current_state, full_model_state)
            per_layer_rows.extend(add_round_column(rows, round_idx, "oracle_vs_full"))

        if config["tracking"]["save_last_layer_every_round"]:
            last = last_layer_distance(current_state, full_model_state)
            last["round"] = round_idx
            last["tag"] = "oracle_vs_full"
            last_layer_rows.append(last)

        if config["oracle"]["save_every_round"]:
            save_checkpoint(
                {
                    "round": round_idx,
                    "model_state_dict": current_state,
                    "val_metrics": val_metrics,
                    "forget_client_id": forget_client_id
                },
                os.path.join(config["paths"]["ckpt_dir"], f"oracle_round_{round_idx}.pt")
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
                os.path.join(config["paths"]["ckpt_dir"], "oracle_best.pt")
            )

        prev_state = {k: v.clone() for k, v in current_state.items()}

    final_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    if best_state is not None:
        global_model.load_state_dict(best_state)

    save_checkpoint(
        {
            "round": config["oracle"]["rounds"],
            "model_state_dict": final_state,
            "best_val_acc": best_acc,
            "best_round": best_round,
            "forget_client_id": forget_client_id
        },
        os.path.join(config["paths"]["ckpt_dir"], "oracle_final.pt")
    )

    summary = {
        "forget_client_id": forget_client_id,
        "best_val_acc": best_acc,
        "best_round": best_round,
        "num_rounds": config["oracle"]["rounds"]
    }

    return global_model, summary, round_metrics, per_layer_rows, last_layer_rows