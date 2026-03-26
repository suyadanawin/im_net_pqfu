import csv
import os
from statistics import mean
from torch.utils.data import DataLoader

from src.phase3_utils import (
    load_config, set_seed, ensure_dirs, get_device,
    get_datasets, load_json, get_client_subsets, get_retain_dataset,
    get_forget_dataset, build_model, load_checkpoint, get_confidence_scores
)


def mia_attack(member_scores, nonmember_scores, threshold):
    tp = sum(s >= threshold for s in member_scores)
    fn = sum(s < threshold for s in member_scores)
    tn = sum(s < threshold for s in nonmember_scores)
    fp = sum(s >= threshold for s in nonmember_scores)

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    return {
        "threshold": threshold,
        "member_mean_conf": mean(member_scores) if member_scores else 0.0,
        "nonmember_mean_conf": mean(nonmember_scores) if nonmember_scores else 0.0,
        "mia_acc": acc
    }


def run_for_model(label, ckpt_path, member_loader, nonmember_loader, config, device):
    model = build_model(
        config["model"]["name"],
        config["dataset"]["num_classes"],
        config["model"]["pretrained"]
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device)

    member_scores = get_confidence_scores(model, member_loader, device)
    nonmember_scores = get_confidence_scores(model, nonmember_loader, device)

    result = mia_attack(member_scores, nonmember_scores, config["mia"]["threshold"])
    result["model"] = label
    return result


def main():
    config = load_config("project/config3.yaml")
    set_seed(config["seed"])
    device = get_device(config["device"])
    ensure_dirs([config["paths"]["metrics_dir"]])

    train_dataset, val_dataset = get_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"]
    )

    client_indices = load_json(config["paths"]["client_indices_path"])
    client_subsets = get_client_subsets(train_dataset, client_indices)
    forget_client_id = config["forgetting"]["forget_client_id"]

    member_dataset = get_forget_dataset(client_subsets, forget_client_id)
    nonmember_dataset = val_dataset

    member_loader = DataLoader(
        member_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )
    nonmember_loader = DataLoader(
        nonmember_dataset,
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
        rows.append(run_for_model(label, ckpt_path, member_loader, nonmember_loader, config, device))

    out_csv = os.path.join(config["paths"]["metrics_dir"], "mia_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "threshold", "member_mean_conf", "nonmember_mean_conf", "mia_acc"]
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()