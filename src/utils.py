import os
import random
import time
import json
from typing import Dict, Any

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More reproducible, slightly slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    if device_name.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare_output_dirs(output_root: str, log_dir_name: str, ckpt_dir_name: str):
    log_dir = os.path.join(output_root, log_dir_name)
    ckpt_dir = os.path.join(output_root, ckpt_dir_name)
    ensure_dir(output_root)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    return log_dir, ckpt_dir


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        return 100.0 * correct / total


def save_checkpoint(state: Dict[str, Any], checkpoint_path: str) -> None:
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)

    return model, optimizer, scheduler, start_epoch, best_val_acc


def append_log(log_file: str, text: str) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")