import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets import (
    fix_tiny_imagenet_val_folder,
    sanity_check_tiny_imagenet,
    build_dataloaders,
    print_dataset_summary,
)
from src.models import build_resnet18_tiny_imagenet
from src.utils import (
    load_config,
    set_seed,
    get_device,
    prepare_output_dirs,
    AverageMeter,
    accuracy,
    save_checkpoint,
    load_checkpoint,
    append_log,
    timestamp,
    save_json,
)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, print_freq=20):
    model.train()

    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")
    for batch_idx, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

        if batch_idx % print_freq == 0:
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2f}%"
            })

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")

    pbar = tqdm(loader, total=len(loader), desc=f"Val Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_acc = accuracy(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}%"
        })

    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Tiny-ImageNet-200 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_root = cfg["paths"]["data_root"]
    output_root = cfg["paths"]["output_root"]

    num_classes = cfg["dataset"]["num_classes"]
    image_size = cfg["dataset"]["image_size"]
    num_workers = cfg["dataset"]["num_workers"]
    pin_memory = cfg["dataset"]["pin_memory"]

    seed = cfg["train"]["seed"]
    device = get_device(cfg["train"]["device"])
    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    learning_rate = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    momentum = cfg["train"]["momentum"]
    optimizer_name = cfg["train"]["optimizer"].lower()
    scheduler_name = cfg["train"]["scheduler"].lower()
    step_size = cfg["train"]["step_size"]
    gamma = cfg["train"]["gamma"]
    print_freq = cfg["train"]["print_freq"]
    save_every = cfg["train"]["save_every"]

    eval_batch_size = cfg["eval"]["batch_size"]

    resume = cfg["checkpoint"]["resume"]
    resume_path = cfg["checkpoint"]["resume_path"]

    log_dir, ckpt_dir = prepare_output_dirs(
        output_root=output_root,
        log_dir_name=cfg["logging"]["log_dir"],
        ckpt_dir_name=cfg["logging"]["checkpoint_dir"]
    )

    run_id = timestamp()
    log_file = os.path.join(log_dir, f"train_{run_id}.log")
    metrics_file = os.path.join(log_dir, f"metrics_{run_id}.json")

    set_seed(seed)

    print("[INFO] Starting Phase 1 Tiny-ImageNet-200 training")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Outputs: {output_root}")

    append_log(log_file, f"Run ID: {run_id}")
    append_log(log_file, f"Device: {device}")
    append_log(log_file, f"Config: {cfg}")

    # 1) Fix val folder
    fix_tiny_imagenet_val_folder(data_root)

    # 2) Dataset sanity checks
    sanity_check_tiny_imagenet(data_root)

    # 3) Build loaders
    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        image_size=image_size,
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print_dataset_summary(train_loader, val_loader)

    # 4) Build model
    model = build_resnet18_tiny_imagenet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    start_epoch = 1
    best_val_acc = 0.0

    if resume:
        if not resume_path:
            raise ValueError("resume is true, but resume_path is empty.")
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model, optimizer, scheduler, start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            print_freq=print_freq
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_log = (
            f"Epoch [{epoch}/{epochs}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
        )
        print(epoch_log)
        append_log(log_file, epoch_log)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "config": cfg,
        }

        if epoch % save_every == 0:
            epoch_ckpt_path = os.path.join(ckpt_dir, f"phase1_epoch_{epoch}.pt")
            save_checkpoint(checkpoint_state, epoch_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_state["best_val_acc"] = best_val_acc
            best_ckpt_path = os.path.join(ckpt_dir, "phase1_best.pt")
            save_checkpoint(checkpoint_state, best_ckpt_path)
            print(f"[INFO] New best checkpoint saved: {best_ckpt_path}")

    final_ckpt_path = os.path.join(ckpt_dir, "phase1_last.pt")
    save_checkpoint({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "config": cfg,
    }, final_ckpt_path)

    save_json(history, metrics_file)

    print("[INFO] Training completed.")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"[INFO] Final checkpoint: {final_ckpt_path}")
    print(f"[INFO] Metrics file: {metrics_file}")
    print(f"[INFO] Log file: {log_file}")


if __name__ == "__main__":
    main()