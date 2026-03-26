import os

import torch

from src.datasets import get_tiny_imagenet_datasets, sanity_check_tiny_imagenet
from src.fl_partition import load_client_indices
from src.unlearning import run_warmstart_unlearning
from src.utils import (
    elapsed_seconds,
    load_checkpoint,
    load_yaml,
    now,
    prepare_output_dirs,
    save_json,
    save_metrics_csv,
    set_seed,
)
from src.metrics import save_rows_csv


def load_oracle_round_states(config: dict):
    states = {}
    for round_idx in range(1, config["oracle"]["rounds"] + 1):
        path = os.path.join(config["paths"]["ckpt_dir"], f"oracle_round_{round_idx}.pt")
        if os.path.exists(path):
            ckpt = load_checkpoint(path, map_location="cpu")
            states[round_idx] = ckpt["model_state_dict"]
    return states


def main():
    config = load_yaml("config_phase3.yaml")
    set_seed(config["seed"])
    prepare_output_dirs(config)

    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    train_dataset, val_dataset = get_tiny_imagenet_datasets(
        data_root=config["paths"]["data_root"],
        image_size=config["dataset"]["image_size"]
    )
    sanity_check_tiny_imagenet(train_dataset, val_dataset, num_classes=config["dataset"]["num_classes"])

    client_to_indices = load_client_indices(config["paths"]["client_indices_path"])
    forget_client_id = config["forgetting"]["forget_client_id"]

    full_ckpt = load_checkpoint(config["paths"]["full_model_ckpt"], map_location="cpu")
    full_model_state = full_ckpt["model_state_dict"]

    oracle_final_ckpt = load_checkpoint(
        os.path.join(config["paths"]["ckpt_dir"], "oracle_final.pt"),
        map_location="cpu"
    )
    oracle_final_state = oracle_final_ckpt["model_state_dict"]
    oracle_round_states = load_oracle_round_states(config)

    print("=" * 80)
    print("Phase 3 - Warm-Start Unlearning")
    print("=" * 80)
    print(f"Forget client id: {forget_client_id}")

    start = now()
    unlearn_model, unlearn_summary, unlearn_round_metrics, unlearn_per_layer_rows, unlearn_last_layer_rows = run_warmstart_unlearning(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        client_to_indices=client_to_indices,
        forget_client_id=forget_client_id,
        full_model_state=full_model_state,
        oracle_round_states=oracle_round_states,
        oracle_final_state=oracle_final_state,
        config=config,
        device=device
    )
    end = now()

    unlearn_summary["unlearning_latency_seconds"] = elapsed_seconds(start, end)

    save_metrics_csv(
        unlearn_round_metrics,
        os.path.join(config["paths"]["metrics_dir"], "unlearn_round_metrics.csv")
    )
    save_json(
        unlearn_summary,
        os.path.join(config["paths"]["metrics_dir"], "unlearn_summary.json")
    )
    save_rows_csv(
        unlearn_per_layer_rows,
        os.path.join(config["paths"]["metrics_dir"], "unlearn_per_layer_roundwise.csv")
    )
    save_rows_csv(
        unlearn_last_layer_rows,
        os.path.join(config["paths"]["metrics_dir"], "unlearn_last_layer_roundwise.csv")
    )

    print("Warm-start unlearning complete.")
    print(f"Unlearning latency: {unlearn_summary['unlearning_latency_seconds']:.4f} seconds")


if __name__ == "__main__":
    main()