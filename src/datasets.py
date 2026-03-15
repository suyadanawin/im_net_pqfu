import os
import shutil
from typing import Tuple

from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_train_transforms(image_size: int = 64):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        ),
    ])


def get_val_transforms(image_size: int = 64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        ),
    ])


def fix_tiny_imagenet_val_folder(data_root: str) -> None:
    """
    Tiny-ImageNet validation images are usually stored in:
      val/images/*.JPEG
    with labels in:
      val/val_annotations.txt

    This function reorganizes them into:
      val/<class_name>/*.JPEG

    It is safe to run multiple times.
    """
    val_dir = os.path.join(data_root, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"val_annotations.txt not found: {annotations_path}")

    # If images_dir does not exist, assume folder has already been reorganized
    if not os.path.isdir(images_dir):
        print("[INFO] Validation folder already appears fixed. Skipping val reorganization.")
        return

    print("[INFO] Fixing Tiny-ImageNet validation folder structure...")

    with open(annotations_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    moved_count = 0
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        image_name, class_name = parts[0], parts[1]
        src_path = os.path.join(images_dir, image_name)
        class_dir = os.path.join(val_dir, class_name)
        dst_path = os.path.join(class_dir, image_name)

        os.makedirs(class_dir, exist_ok=True)

        if os.path.isfile(src_path) and not os.path.isfile(dst_path):
            shutil.move(src_path, dst_path)
            moved_count += 1

    # Remove empty images dir if everything moved
    if os.path.isdir(images_dir) and len(os.listdir(images_dir)) == 0:
        os.rmdir(images_dir)

    print(f"[INFO] Validation folder fix complete. Moved {moved_count} images.")


def sanity_check_tiny_imagenet(data_root: str) -> None:
    """
    Basic sanity checks for dataset structure.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    wnids_path = os.path.join(data_root, "wnids.txt")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val directory not found: {val_dir}")
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(f"wnids.txt not found: {wnids_path}")

    train_classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    val_classes = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ])

    with open(wnids_path, "r", encoding="utf-8") as f:
        wnids = sorted([line.strip() for line in f if line.strip()])

    if len(wnids) != 200:
        raise ValueError(f"Expected 200 classes in wnids.txt, found {len(wnids)}")

    if len(train_classes) != 200:
        raise ValueError(f"Expected 200 train class folders, found {len(train_classes)}")

    if len(val_classes) != 200:
        raise ValueError(f"Expected 200 val class folders after fixing, found {len(val_classes)}")

    if train_classes != wnids:
        raise ValueError("Train classes do not match wnids.txt")

    if val_classes != wnids:
        raise ValueError("Val classes do not match wnids.txt")

    print("[INFO] Dataset sanity check passed.")
    print(f"[INFO] Number of classes: {len(wnids)}")


def build_datasets(data_root: str, image_size: int = 64):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size=image_size)
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms(image_size=image_size)
    )

    return train_dataset, val_dataset


def build_dataloaders(
    data_root: str,
    image_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(data_root, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def print_dataset_summary(train_loader: DataLoader, val_loader: DataLoader) -> None:
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    print("[INFO] Dataset summary")
    print(f"       Train samples: {len(train_dataset)}")
    print(f"       Val samples:   {len(val_dataset)}")
    print(f"       Num classes:   {len(train_dataset.classes)}")

    x, y = train_dataset[0]
    print(f"[INFO] One training sample tensor shape: {tuple(x.shape)}")
    print(f"[INFO] One training sample label index: {y}")