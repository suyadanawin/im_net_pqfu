import os
import csv
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.phase3_utils import (
    load_config, set_seed, ensure_dirs, get_device,
    get_datasets, build_model, load_json, get_client_subsets,
    get_retain_dataset, load_checkpoint, save_checkpoint,
    train_one_epoch, evaluate, save_json
)


def main():
    config = load_config("project/config3.yaml")
    set_seed(config["seed"])
    device = get_device(config["device"])

    ensure_dirs([
        config["paths"]["ckpt_dir"],
        config["paths"]["metrics_dir"],
        config["paths"]["plots_dir"],
    ])

    train_dataset, val_dataset = get_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"]
    )

    client_indices = load_json(config["paths"]["client_indices_path"])
    client_subsets = get_client_subsets(train_dataset, client_indices)

    forget_client_id = config["forgetting"]["forget_client_id"]
    retain_dataset = get_retain_dataset(client_subsets, forget_client_id)

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=config["fl"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )

    model = build_model(
        config["model"]["name"],
        config["dataset"]["num_classes"],
        config["model"]["pretrained"]
    ).to(device)

    model = load_checkpoint(model, config["paths"]["full_model_ckpt"], device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["fl"]["lr"],
        momentum=config["fl"]["momentum"],
        weight_decay=config["fl"]["weight_decay"]
    )

    correction_rounds = config["unlearning"]["correction_rounds"]
    finetune_epochs = config["unlearning"]["finetune_epochs"]

    best_val_acc = -1.0
    best_round = -1
    rows = []

    for round_idx in range(1, correction_rounds + 1):
        train_loss, train_acc = 0.0, 0.0

        for _ in range(finetune_epochs):
            train_loss, train_acc = train_one_epoch(model, retain_loader, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, device)

        row = {
            "round": round_idx,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        rows.append(row)

        print(
            f"[Unlearn Round {round_idx}/{correction_rounds}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_round = round_idx
            save_checkpoint(model, os.path.join(config["paths"]["ckpt_dir"], "unlearn_best.pt"))

    save_checkpoint(model, os.path.join(config["paths"]["ckpt_dir"], "unlearn_final.pt"))

    csv_path = os.path.join(config["paths"]["metrics_dir"], "unlearn_round_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "forget_client_id": forget_client_id,
        "correction_rounds": correction_rounds,
        "best_val_acc": best_val_acc,
        "best_round": best_round
    }
    save_json(summary, os.path.join(config["paths"]["metrics_dir"], "unlearn_summary.json"))

    print("Saved unlearning model and metrics successfully.")


if __name__ == "__main__":
    main()