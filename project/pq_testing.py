import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.datasets import build_tiny_imagenet_datasets, dirichlet_partition_indices
from src.models import build_model
from src.utils import ensure_dir, save_json


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


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * total_correct / total


def state_l2_distance(sd1, sd2):
    total = 0.0
    for k in sd1.keys():
        diff = (sd1[k].float().cpu() - sd2[k].float().cpu()).view(-1)
        total += torch.sum(diff * diff).item()
    return total ** 0.5


def final_layer_l2_distance(sd1, sd2):
    candidate_keys = [k for k in sd1.keys() if ("fc" in k or "classifier" in k)]
    if len(candidate_keys) == 0:
        return None

    total = 0.0
    for k in candidate_keys:
        diff = (sd1[k].float().cpu() - sd2[k].float().cpu()).view(-1)
        total += torch.sum(diff * diff).item()
    return total ** 0.5


def load_model(config, checkpoint_path, device):
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    ).to(device)
    model.load_state_dict(load_flexible_state_dict(checkpoint_path, device))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    eval_root = config["paths"]["eval_root"]
    ensure_dir(eval_root)

    forget_client_id = config["unlearning"]["forget_client_id"]

    train_dataset, val_dataset = build_tiny_imagenet_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"],
    )

    client_indices = load_or_create_partitions(train_dataset, config)

    forget_subset = Subset(train_dataset, client_indices[forget_client_id])
    retain_subsets = [
        Subset(train_dataset, idxs)
        for cid, idxs in enumerate(client_indices)
        if cid != forget_client_id
    ]
    retain_dataset = ConcatDataset(retain_subsets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )
    forget_loader = DataLoader(
        forget_subset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    full_ckks_path = config["paths"]["base_ckks_checkpoint"]
    oracle_ckks_path = os.path.join(config["paths"]["oracle_root"], "checkpoints", "oracle_best_ckks.pt")
    unlearn_ckks_path = os.path.join(config["paths"]["unlearn_root"], "checkpoints", "ckks_unlearn_best.pt")

    # Always required
    full_model = load_model(config, full_ckks_path, device)
    unlearn_model = load_model(config, unlearn_ckks_path, device)

    # Oracle is optional (e.g. pq_8192_light has oracle.rounds = 0)
    oracle_model = None
    oracle_available = (
        config.get("oracle", {}).get("rounds", 0) > 0
        and os.path.exists(oracle_ckks_path)
    )
    if oracle_available:
        oracle_model = load_model(config, oracle_ckks_path, device)

    models = {
        "full_ckks": full_model,
        "unlearn_ckks": unlearn_model,
    }
    if oracle_model is not None:
        models["oracle_ckks"] = oracle_model

    # Keep order consistent
    ordered_names = ["full_ckks", "oracle_ckks", "unlearn_ckks"]
    rows = []

    for name in ordered_names:
        if name not in models:
            continue
        model = models[name]

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        forget_loss, forget_acc = evaluate_model(model, forget_loader, device)
        retain_loss, retain_acc = evaluate_model(model, retain_loader, device)

        rows.append({
            "model": name,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "forget_loss": forget_loss,
            "forget_acc": forget_acc,
            "retain_loss": retain_loss,
            "retain_acc": retain_acc,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(eval_root, "ckks_unlearning_model_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Distance summary
    full_sd = full_model.state_dict()
    unlearn_sd = unlearn_model.state_dict()

    if oracle_model is not None:
        oracle_sd = oracle_model.state_dict()
        distance_summary = {
            "full_vs_oracle_l2": state_l2_distance(full_sd, oracle_sd),
            "unlearn_vs_oracle_l2": state_l2_distance(unlearn_sd, oracle_sd),
            "full_vs_oracle_last_layer_l2": final_layer_l2_distance(full_sd, oracle_sd),
            "unlearn_vs_oracle_last_layer_l2": final_layer_l2_distance(unlearn_sd, oracle_sd),
            "oracle_used": True,
        }
    else:
        distance_summary = {
            "full_vs_oracle_l2": None,
            "unlearn_vs_oracle_l2": None,
            "full_vs_oracle_last_layer_l2": None,
            "unlearn_vs_oracle_last_layer_l2": None,
            "oracle_used": False,
            "note": "Oracle skipped because oracle.rounds = 0 or oracle checkpoint is missing.",
        }

    save_json(distance_summary, os.path.join(eval_root, "ckks_distance_summary.json"))

    print(df)
    print(distance_summary)
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()