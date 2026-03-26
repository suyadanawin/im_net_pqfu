import os
import json
import yaml
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset

# IMPORTANT: use the same dataset + model code as your original training
from src.datasets import get_tiny_imagenet_datasets
from src.models import get_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# DATA
# Use the SAME dataset loader as Phase 1 / Phase 2 / Oracle
# =========================================================
def get_datasets(data_root: str, image_size: int = 64):
    return get_tiny_imagenet_datasets(
        data_root=data_root,
        image_size=image_size
    )


# =========================================================
# MODEL
# Use the SAME model builder as the original training code
# =========================================================
def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    return get_model(model_name, num_classes=num_classes)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


def get_client_subsets(train_dataset, client_indices: Dict[str, List[int]]):
    client_subsets = {}
    for client_id, indices in client_indices.items():
        client_subsets[int(client_id)] = Subset(train_dataset, indices)
    return client_subsets


def get_retain_dataset(client_subsets: Dict[int, Subset], forget_client_id: int):
    keep_datasets = [ds for cid, ds in client_subsets.items() if cid != forget_client_id]
    return ConcatDataset(keep_datasets)


def get_forget_dataset(client_subsets: Dict[int, Subset], forget_client_id: int):
    return client_subsets[forget_client_id]


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Missing keys count: {len(missing)}")
    print(f"Unexpected keys count: {len(unexpected)}")
    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])

    return model


def save_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    round_idx: int = None,
    val_metrics: dict = None,
    config: dict = None
):
    checkpoint = {
        "model_state_dict": model.state_dict()
    }

    if round_idx is not None:
        checkpoint["round"] = round_idx
    if val_metrics is not None:
        checkpoint["val_metrics"] = val_metrics
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, ckpt_path)


def model_l2_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    sq_sum = 0.0
    with torch.no_grad():
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            sq_sum += torch.sum((p1.detach().cpu() - p2.detach().cpu()) ** 2).item()
    return sq_sum ** 0.5


def per_layer_l2_distance(model_a: nn.Module, model_b: nn.Module):
    rows = []
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert n1 == n2
            dist = torch.norm(p1.detach().cpu() - p2.detach().cpu(), p=2).item()
            rows.append({"layer": n1, "l2_distance": dist})
    return rows


def last_layer_l2_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    a_params = dict(model_a.named_parameters())
    b_params = dict(model_b.named_parameters())
    keys = [k for k in a_params.keys() if k.startswith("fc.")]
    sq_sum = 0.0
    with torch.no_grad():
        for k in keys:
            sq_sum += torch.sum((a_params[k].detach().cpu() - b_params[k].detach().cpu()) ** 2).item()
    return sq_sum ** 0.5


def get_confidence_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            scores.extend(max_probs.detach().cpu().tolist())
    return scores