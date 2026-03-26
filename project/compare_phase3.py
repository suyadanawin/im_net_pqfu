import os

import torch

from src.datasets import get_tiny_imagenet_datasets, sanity_check_tiny_imagenet
from src.metrics import (
    last_layer_distance,
    model_l2_distance,
    per_layer_l2_distance,
    save_rows_csv,
)
from src.models import get_model
from src.utils import (
    evaluate_model,
    load_checkpoint,
    load_json,
    load_yaml,
    prepare_output_dirs,
    save_json,
    save_metrics_csv,
    set_seed,
)


def build_model_from_ckpt(ckpt_path: str, config: dict, device: str):
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = get_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt["model_state_dict"]


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

    full_model_path = config["paths"]["full_model_ckpt"]
    oracle_model_path = os.path.join(config["paths"]["ckpt_dir"], "oracle_best.pt") \
        if config["comparison"]["use_best_oracle_for_final_eval"] else os.path.join(config["paths"]["ckpt_dir"], "oracle_final.pt")
    unlearn_model_path = os.path.join(config["paths"]["ckpt_dir"], "unlearn_best.pt") \
        if config["comparison"]["use_best_unlearned_for_final_eval"] else os.path.join(config["paths"]["ckpt_dir"], "unlearn_final.pt")

    full_model, full_state = build_model_from_ckpt(full_model_path, config, device)
    oracle_model, oracle_state = build_model_from_ckpt(oracle_model_path, config, device)
    unlearn_model, unlearn_state = build_model_from_ckpt(unlearn_model_path, config, device)

    full_metrics = evaluate_model(
        model=full_model,
        dataset=val_dataset,
        batch_size=config["phase2_fl"]["eval_batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        device=device
    )
    oracle_metrics = evaluate_model(
        model=oracle_model,
        dataset=val_dataset,
        batch_size=config["oracle"]["eval_batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        device=device
    )
    unlearn_metrics = evaluate_model(
        model=unlearn_model,
        dataset=val_dataset,
        batch_size=config["unlearning"]["eval_batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        device=device
    )

    dist_full_oracle = model_l2_distance(full_state, oracle_state)
    dist_unlearn_oracle = model_l2_distance(unlearn_state, oracle_state)
    dist_unlearn_full = model_l2_distance(unlearn_state, full_state)

    reduction_ratio = None
    if dist_full_oracle > 0:
        reduction_ratio = (dist_full_oracle - dist_unlearn_oracle) / dist_full_oracle

    per_layer_full_oracle = per_layer_l2_distance(full_state, oracle_state)
    per_layer_unlearn_oracle = per_layer_l2_distance(unlearn_state, oracle_state)
    per_layer_unlearn_full = per_layer_l2_distance(unlearn_state, full_state)

    save_rows_csv(
        per_layer_full_oracle,
        os.path.join(config["paths"]["metrics_dir"], "per_layer_distance_full_vs_oracle.csv")
    )
    save_rows_csv(
        per_layer_unlearn_oracle,
        os.path.join(config["paths"]["metrics_dir"], "per_layer_distance_unlearned_vs_oracle.csv")
    )
    save_rows_csv(
        per_layer_unlearn_full,
        os.path.join(config["paths"]["metrics_dir"], "per_layer_distance_unlearned_vs_full.csv")
    )

    last_full_oracle = last_layer_distance(full_state, oracle_state)
    last_unlearn_oracle = last_layer_distance(unlearn_state, oracle_state)
    last_unlearn_full = last_layer_distance(unlearn_state, full_state)

    comparison = {
        "full_model": {
            "val_loss": full_metrics["loss"],
            "val_acc": full_metrics["acc"],
        },
        "oracle_model": {
            "val_loss": oracle_metrics["loss"],
            "val_acc": oracle_metrics["acc"],
        },
        "unlearned_model": {
            "val_loss": unlearn_metrics["loss"],
            "val_acc": unlearn_metrics["acc"],
        },
        "oracle_proximity": {
            "l2_w_full_minus_w_oracle": dist_full_oracle,
            "l2_w_unlearned_minus_w_oracle": dist_unlearn_oracle,
            "l2_w_unlearned_minus_w_full": dist_unlearn_full,
            "distance_reduction_ratio_toward_oracle": reduction_ratio,
            "last_layer_full_vs_oracle": last_full_oracle,
            "last_layer_unlearned_vs_oracle": last_unlearn_oracle,
            "last_layer_unlearned_vs_full": last_unlearn_full
        },
        "theory_note": config["theory_note"]
    }

    save_json(
        comparison,
        os.path.join(config["paths"]["metrics_dir"], "phase3_comparison.json")
    )

    summary_table = [
        {
            "model_name": "full_model",
            "val_loss": full_metrics["loss"],
            "val_acc": full_metrics["acc"],
            "distance_to_oracle": dist_full_oracle
        },
        {
            "model_name": "oracle_model",
            "val_loss": oracle_metrics["loss"],
            "val_acc": oracle_metrics["acc"],
            "distance_to_oracle": 0.0
        },
        {
            "model_name": "unlearned_model",
            "val_loss": unlearn_metrics["loss"],
            "val_acc": unlearn_metrics["acc"],
            "distance_to_oracle": dist_unlearn_oracle
        }
    ]
    save_metrics_csv(summary_table, os.path.join(config["paths"]["metrics_dir"], "phase3_summary_table.csv"))

    print("=" * 80)
    print("Phase 3 Comparison")
    print("=" * 80)
    print(f"Full model val acc:      {full_metrics['acc']:.2f}%")
    print(f"Oracle model val acc:    {oracle_metrics['acc']:.2f}%")
    print(f"Unlearned model val acc: {unlearn_metrics['acc']:.2f}%")
    print("-" * 80)
    print(f"||w_full - w_oracle||:      {dist_full_oracle:.6f}")
    print(f"||w_unlearned - w_oracle||: {dist_unlearn_oracle:.6f}")
    print(f"Reduction ratio:            {reduction_ratio}")
    print("-" * 80)
    print(f"Saved comparison metrics to: {config['paths']['metrics_dir']}")


if __name__ == "__main__":
    main()