import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets import (
    fix_tiny_imagenet_val_folder,
    sanity_check_tiny_imagenet,
    build_dataloaders,
)
from src.models import build_resnet18_tiny_imagenet
from src.utils import load_config, get_device, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, total=len(loader), desc="Evaluating")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        running_loss = total_loss / total_samples
        running_acc = 100.0 * total_correct / total_samples

        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc": f"{running_acc:.2f}%"
        })

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Tiny-ImageNet-200 Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_root = cfg["paths"]["data_root"]
    device = get_device(cfg["train"]["device"])
    image_size = cfg["dataset"]["image_size"]
    num_workers = cfg["dataset"]["num_workers"]
    pin_memory = cfg["dataset"]["pin_memory"]
    num_classes = cfg["dataset"]["num_classes"]
    eval_batch_size = cfg["eval"]["batch_size"]

    print("[INFO] Starting evaluation")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    fix_tiny_imagenet_val_folder(data_root)
    sanity_check_tiny_imagenet(data_root)

    _, val_loader = build_dataloaders(
        data_root=data_root,
        image_size=image_size,
        train_batch_size=cfg["train"]["batch_size"],
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = build_resnet18_tiny_imagenet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device
    )

    print("[RESULT] Validation results")
    print(f"[RESULT] Loss: {val_loss:.4f}")
    print(f"[RESULT] Accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main()