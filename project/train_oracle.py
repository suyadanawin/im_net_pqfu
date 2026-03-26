import os

import torch

from src.datasets import get_tiny_imagenet_datasets, sanity_check_tiny_imagenet
from src.fl_partition import load_client_indices
from src.oracle import run_oracle_retraining
from src.utils import (
    load_checkpoint,
    load_yaml,
    prepare_output_dirs,
    save_json,
    save_metrics_csv,
    save_yaml,
    set_seed,
)
from src.metrics import save_rows_csv


def main():
    config = load_yaml("project/oracle.yaml")
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

    save_yaml(config, os.path.join(config["paths"]["log_dir"], "phase3_used_config.yaml"))

    print("=" * 80)
    print("Phase 3 - Oracle Retraining")
    print("=" * 80)
    print(f"Forget client id: {forget_client_id}")

    oracle_model, oracle_summary, oracle_round_metrics, oracle_per_layer_rows, oracle_last_layer_rows = run_oracle_retraining(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        client_to_indices=client_to_indices,
        forget_client_id=forget_client_id,
        full_model_state=full_model_state,
        config=config,
        device=device
    )
    
    save_metrics_csv(
        oracle_round_metrics,
        os.path.join(config["paths"]["metrics_dir"], "oracle_round_metrics.csv")
    )
    save_json(
        oracle_summary,
        os.path.join(config["paths"]["metrics_dir"], "oracle_summary.json")
    )
    save_rows_csv(
        oracle_per_layer_rows,
        os.path.join(config["paths"]["metrics_dir"], "oracle_per_layer_roundwise.csv")
    )
    save_rows_csv(
        oracle_last_layer_rows,
        os.path.join(config["paths"]["metrics_dir"], "oracle_last_layer_roundwise.csv")
    )
    torch.save(
    {
        "model_state_dict": oracle_model.state_dict(),
        "best_val_acc": oracle_summary.get("best_val_acc"),
        "best_round": oracle_summary.get("best_round"),
        "forget_client_id": forget_client_id,
    },
    os.path.join(config["paths"]["ckpt_dir"], "oracle_final.pt")
)
    print("Oracle retraining complete.")


if __name__ == "__main__":
    main()