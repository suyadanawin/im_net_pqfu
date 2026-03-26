import csv
import os
from torch.utils.data import DataLoader

from src.phase3_utils import (
    load_config, set_seed, ensure_dirs, get_device,
    get_datasets, load_json, get_client_subsets, get_retain_dataset,
    get_forget_dataset, build_model, load_checkpoint, evaluate
)


def eval_model_on_loader(name, ckpt_path, loader, config, device):
    model = build_model(
        config["model"]["name"],
        config["dataset"]["num_classes"],
        config["model"]["pretrained"]
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    loss, acc = evaluate(model, loader, device)
    return {"model": name, "loss": loss, "acc": acc}


def main():
    config = load_config("project/config3.yaml")
    set_seed(config["seed"])
    device = get_device(config["device"])
    ensure_dirs([config["paths"]["metrics_dir"]])

    train_dataset, _ = get_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"]
    )

    client_indices = load_json(config["paths"]["client_indices_path"])
    client_subsets = get_client_subsets(train_dataset, client_indices)

    forget_client_id = config["forgetting"]["forget_client_id"]
    forget_dataset = get_forget_dataset(client_subsets, forget_client_id)
    retain_dataset = get_retain_dataset(client_subsets, forget_client_id)

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )

    rows = []
    for label, ckpt_path in [
        ("full", config["paths"]["full_model_ckpt"]),
        ("oracle", config["paths"]["oracle_ckpt"]),
        ("unlearn", config["paths"]["unlearn_ckpt"]),
    ]:
        r1 = eval_model_on_loader(label, ckpt_path, forget_loader, config, device)
        r1["split"] = "forget"
        rows.append(r1)

        r2 = eval_model_on_loader(label, ckpt_path, retain_loader, config, device)
        r2["split"] = "retain"
        rows.append(r2)

    out_csv = os.path.join(config["paths"]["metrics_dir"], "forget_retain_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split", "loss", "acc"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()