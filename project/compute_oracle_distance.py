import csv
import os

from src.phase3_utils import (
    load_config, set_seed, ensure_dirs, get_device,
    build_model, load_checkpoint, model_l2_distance,
    per_layer_l2_distance, last_layer_l2_distance
)


def load_named_model(ckpt_path, config, device):
    model = build_model(
        config["model"]["name"],
        config["dataset"]["num_classes"],
        config["model"]["pretrained"]
    ).to(device)
    return load_checkpoint(model, ckpt_path, device)


def main():
    config = load_config("project/config3.yaml")
    set_seed(config["seed"])
    device = get_device(config["device"])

    ensure_dirs([config["paths"]["metrics_dir"]])

    full_model = load_named_model(config["paths"]["full_model_ckpt"], config, device)
    oracle_model = load_named_model(config["paths"]["oracle_ckpt"], config, device)
    unlearn_model = load_named_model(config["paths"]["unlearn_ckpt"], config, device)

    summary_rows = [
        {
            "pair": "full_vs_oracle",
            "full_model_l2": model_l2_distance(full_model, oracle_model),
            "last_layer_l2": last_layer_l2_distance(full_model, oracle_model)
        },
        {
            "pair": "unlearn_vs_oracle",
            "full_model_l2": model_l2_distance(unlearn_model, oracle_model),
            "last_layer_l2": last_layer_l2_distance(unlearn_model, oracle_model)
        }
    ]

    with open(os.path.join(config["paths"]["metrics_dir"], "distance_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "full_model_l2", "last_layer_l2"])
        writer.writeheader()
        writer.writerows(summary_rows)

    per_layer = per_layer_l2_distance(unlearn_model, oracle_model)
    with open(os.path.join(config["paths"]["metrics_dir"], "unlearn_per_layer_roundwise.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "l2_distance"])
        writer.writeheader()
        writer.writerows(per_layer)

    print("Distance metrics saved.")


if __name__ == "__main__":
    main()