import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


"""def prepare_output_dirs(config: dict):
    ensure_dir(config["paths"]["output_root"])
    ensure_dir(config["paths"]["ckpt_dir"])
    ensure_dir(config["paths"]["log_dir"])
    ensure_dir(config["paths"]["metrics_dir"])
    ensure_dir(config["paths"]["stats_dir"])"""

def prepare_output_dirs(config: dict):
    paths = config["paths"]

    if "output_root" not in paths:
        raise KeyError(
            "Missing 'paths.output_root' in config_phase2.yaml. "
            "Please add output_root under paths."
        )

    ensure_dir(paths["output_root"])
    ensure_dir(paths["ckpt_dir"])
    ensure_dir(paths["log_dir"])
    ensure_dir(paths["metrics_dir"])
    ensure_dir(paths["stats_dir"])


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_metrics_csv(metrics: List[Dict], path: str):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def accuracy_from_logits(logits, labels, return_count=False):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    if return_count:
        return correct
    return correct / labels.size(0)


def evaluate_model(model, dataset, batch_size: int, num_workers: int, pin_memory: bool, device: str):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size_now = labels.size(0)
            total_loss += loss.item() * batch_size_now
            total_correct += accuracy_from_logits(outputs, labels, return_count=True)
            total_samples += batch_size_now

            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

    metrics = {
        "val_loss": total_loss / total_samples,
        "val_acc": 100.0 * total_correct / total_samples
    }
    return metrics


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def print_round_summary(round_idx: int, client_results: List[Dict], val_metrics: Dict):
    avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
    avg_client_acc = sum(r["train_acc"] for r in client_results) / len(client_results)

    print("=" * 80)
    print(f"Round {round_idx} Summary")
    print("=" * 80)
    print(f"Avg client train loss: {avg_client_loss:.4f}")
    print(f"Avg client train acc:  {avg_client_acc:.2f}%")
    print(f"Global val loss:       {val_metrics['val_loss']:.4f}")
    print(f"Global val acc:        {val_metrics['val_acc']:.2f}%")
    print("=" * 80)