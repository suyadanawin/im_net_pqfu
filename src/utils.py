import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_output_dirs(config: dict):
    if "phase3_root" in config["paths"]:
        ensure_dir(config["paths"]["phase3_root"])
    if "output_root" in config["paths"]:
        ensure_dir(config["paths"]["output_root"])
    if "ckpt_dir" in config["paths"]:
        ensure_dir(config["paths"]["ckpt_dir"])
    if "log_dir" in config["paths"]:
        ensure_dir(config["paths"]["log_dir"])
    if "metrics_dir" in config["paths"]:
        ensure_dir(config["paths"]["metrics_dir"])
    if "stats_dir" in config["paths"]:
        ensure_dir(config["paths"]["stats_dir"])
    if "plots_dir" in config["paths"]:
        ensure_dir(config["paths"]["plots_dir"])


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_csv(metrics: List[Dict], path: str):
    pd.DataFrame(metrics).to_csv(path, index=False)


def append_metrics_csv(rows: List[Dict], path: str):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
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

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples
    }


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def print_round_summary(title: str, round_idx: int, client_results: List[Dict], val_metrics: Dict):
    avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
    avg_client_acc = sum(r["train_acc"] for r in client_results) / len(client_results)

    print("=" * 80)
    print(f"{title} Round {round_idx} Summary")
    print("=" * 80)
    print(f"Avg client train loss: {avg_client_loss:.4f}")
    print(f"Avg client train acc:  {avg_client_acc:.2f}%")
    print(f"Global val loss:       {val_metrics['loss']:.4f}")
    print(f"Global val acc:        {val_metrics['acc']:.2f}%")
    print("=" * 80)


def now():
    return time.perf_counter()


def elapsed_seconds(start_time: float, end_time: float) -> float:
    return float(end_time - start_time)