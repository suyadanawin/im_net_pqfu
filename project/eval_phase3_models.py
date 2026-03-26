import csv
import os
from torch.utils.data import DataLoader

from src.phase3_utils import (
    load_config, set_seed, ensure_dirs, get_device,
    get_datasets, build_model, load_checkpoint, evaluate
)


def eval_one(label, ckpt_path, config, val_loader, device):
    model = build_model(
        config["model"]["name"],
        config["dataset"]["num_classes"],
        config["model"]["pretrained"]
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device)
    val_loss, val_acc = evaluate(model, val_loader, device)

    return {
        "model": label,
        "val_loss": val_loss,
        "val_acc": val_acc
    }


def main():
    config = load_config("project/config3.yaml")
    set_seed(config["seed"])
    device = get_device(config["device"])

    ensure_dirs([config["paths"]["metrics_dir"]])

    _, val_dataset = get_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )

    results = []
    results.append(eval_one("full", config["paths"]["full_model_ckpt"], config, val_loader, device))
    results.append(eval_one("oracle", config["paths"]["oracle_ckpt"], config, val_loader, device))
    results.append(eval_one("unlearn", config["paths"]["unlearn_ckpt"], config, val_loader, device))

    out_csv = os.path.join(config["paths"]["metrics_dir"], "phase3_model_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()