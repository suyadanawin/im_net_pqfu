import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.datasets import build_tiny_imagenet_datasets, dirichlet_partition_indices
from src.models import build_model
from src.utils import ensure_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_flexible_state_dict(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
        if "state_dict" in obj:
            return obj["state_dict"]
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_or_create_partitions(train_dataset, config):
    partition_file = config["paths"].get("partition_file", None)
    if partition_file and os.path.exists(partition_file):
        with open(partition_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        client_indices = [loaded[str(i)] for i in range(config["fl"]["num_clients"])]
        return client_indices

    client_indices = dirichlet_partition_indices(
        labels=np.array(train_dataset.targets),
        num_clients=config["fl"]["num_clients"],
        alpha=config["fl"]["dirichlet_alpha"],
        seed=config["seed"],
    )
    return client_indices


def load_model(config, checkpoint_path, device):
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    ).to(device)
    model.load_state_dict(load_flexible_state_dict(checkpoint_path, device))
    model.eval()
    return model


@torch.no_grad()
def collect_confidences(model, loader, device):
    confs = []
    for x, _ in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        max_conf = probs.max(dim=1).values
        confs.extend(max_conf.detach().cpu().tolist())
    return np.array(confs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    ensure_dir(config["paths"]["mia_root"])

    forget_client_id = config["unlearning"]["forget_client_id"]

    train_dataset, val_dataset = build_tiny_imagenet_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"],
    )

    client_indices = load_or_create_partitions(train_dataset, config)
    member_indices = client_indices[forget_client_id]

    max_member = min(len(member_indices), config["mia"]["max_member_samples"])
    member_indices = member_indices[:max_member]

    max_nonmember = min(len(val_dataset), config["mia"]["max_nonmember_samples"])
    nonmember_indices = list(range(max_nonmember))

    member_loader = DataLoader(
        Subset(train_dataset, member_indices),
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    nonmember_loader = DataLoader(
        Subset(val_dataset, nonmember_indices),
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    model_paths = {
        "full_ckks": config["paths"]["base_ckks_checkpoint"],
        "oracle_ckks": os.path.join(config["paths"]["oracle_root"], "checkpoints", "oracle_best_ckks.pt"),
        "unlearn_ckks": os.path.join(config["paths"]["unlearn_root"], "checkpoints", "ckks_unlearn_best.pt"),
    }

    rows = []
    for model_name, path in model_paths.items():
        model = load_model(config, path, device)

        member_conf = collect_confidences(model, member_loader, device)
        nonmember_conf = collect_confidences(model, nonmember_loader, device)

        threshold = float((member_conf.mean() + nonmember_conf.mean()) / 2.0)

        member_pred = (member_conf >= threshold).astype(int)
        nonmember_pred = (nonmember_conf >= threshold).astype(int)

        member_acc = member_pred.mean()
        nonmember_acc = (1 - nonmember_pred).mean()
        mia_acc = float((member_acc + nonmember_acc) / 2.0)

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "member_mean_conf": float(member_conf.mean()),
            "nonmember_mean_conf": float(nonmember_conf.mean()),
            "member_acc": float(member_acc),
            "nonmember_acc": float(nonmember_acc),
            "mia_acc": mia_acc,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config["paths"]["mia_root"], "mia_ckks_unlearning_summary.csv")
    df.to_csv(out_csv, index=False)

    print(df)
    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()