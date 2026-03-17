import os
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNetTrainDataset(Dataset):
    """
    Tiny-ImageNet-200 training dataset loader.

    Expected structure:
    tiny-imagenet-200/
        train/
            n01443537/
                images/
                    *.JPEG
            ...
        wnids.txt
    """

    def __init__(self, root: str, transform: Callable = None):
        self.root = root
        self.transform = transform

        self.wnids_path = os.path.join(root, "wnids.txt")
        self.train_dir = os.path.join(root, "train")

        if not os.path.exists(self.wnids_path):
            raise FileNotFoundError(f"wnids.txt not found at: {self.wnids_path}")
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"train directory not found at: {self.train_dir}")

        self.class_to_idx = self._load_class_to_idx()
        self.samples, self.targets = self._build_samples()

    def _load_class_to_idx(self) -> Dict[str, int]:
        with open(self.wnids_path, "r") as f:
            wnids = [line.strip() for line in f.readlines()]
        return {wnid: idx for idx, wnid in enumerate(wnids)}

    def _build_samples(self) -> Tuple[List[Tuple[str, int]], List[int]]:
        samples = []
        targets = []

        for wnid, class_idx in self.class_to_idx.items():
            img_dir = os.path.join(self.train_dir, wnid, "images")
            if not os.path.exists(img_dir):
                continue

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.lower().endswith((".jpeg", ".jpg", ".png")):
                    img_path = os.path.join(img_dir, file_name)
                    samples.append((img_path, class_idx))
                    targets.append(class_idx)

        if len(samples) == 0:
            raise RuntimeError("No training images found in Tiny-ImageNet train split.")

        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class TinyImageNetValDataset(Dataset):
    """
    Tiny-ImageNet-200 validation dataset loader.

    Supports BOTH formats:

    1. Original format:
       val/
         images/
         val_annotations.txt

    2. Reorganized class-folder format:
       val/
         n01443537/
           images/
             *.JPEG
         n01629819/
           images/
             *.JPEG
         ...
    """

    def __init__(self, root: str, transform: Callable = None):
        self.root = root
        self.transform = transform

        self.wnids_path = os.path.join(root, "wnids.txt")
        self.val_dir = os.path.join(root, "val")
        self.val_images_dir = os.path.join(self.val_dir, "images")
        self.val_annotations_path = os.path.join(self.val_dir, "val_annotations.txt")

        if not os.path.exists(self.wnids_path):
            raise FileNotFoundError(f"wnids.txt not found at: {self.wnids_path}")
        if not os.path.exists(self.val_dir):
            raise FileNotFoundError(f"val directory not found at: {self.val_dir}")

        self.class_to_idx = self._load_class_to_idx()
        self.samples, self.targets = self._build_samples()

    def _load_class_to_idx(self) -> Dict[str, int]:
        with open(self.wnids_path, "r") as f:
            wnids = [line.strip() for line in f.readlines()]
        return {wnid: idx for idx, wnid in enumerate(wnids)}

    def _build_samples(self) -> Tuple[List[Tuple[str, int]], List[int]]:
        # Case 1: original Tiny-ImageNet validation format
        if os.path.exists(self.val_images_dir) and os.path.exists(self.val_annotations_path):
            return self._build_from_original_val_format()

        # Case 2: reorganized class-folder validation format
        return self._build_from_class_folder_format()

    def _build_from_original_val_format(self) -> Tuple[List[Tuple[str, int]], List[int]]:
        samples = []
        targets = []

        with open(self.val_annotations_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            file_name = parts[0]
            wnid = parts[1]

            if wnid not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[wnid]
            img_path = os.path.join(self.val_images_dir, file_name)

            if os.path.exists(img_path):
                samples.append((img_path, class_idx))
                targets.append(class_idx)

        if len(samples) == 0:
            raise RuntimeError("No validation images found in original Tiny-ImageNet val format.")

        print("Validation loader: detected original val/images format.")
        return samples, targets

    def _build_from_class_folder_format(self) -> Tuple[List[Tuple[str, int]], List[int]]:
        samples = []
        targets = []

        for wnid, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.val_dir, wnid)

            # Supports both:
            # val/<wnid>/images/*.JPEG
            # val/<wnid>/*.JPEG
            images_dir = os.path.join(class_dir, "images")

            candidate_dirs = []
            if os.path.exists(images_dir):
                candidate_dirs.append(images_dir)
            if os.path.exists(class_dir):
                candidate_dirs.append(class_dir)

            for candidate_dir in candidate_dirs:
                for file_name in sorted(os.listdir(candidate_dir)):
                    file_path = os.path.join(candidate_dir, file_name)
                    if os.path.isfile(file_path) and file_name.lower().endswith((".jpeg", ".jpg", ".png")):
                        samples.append((file_path, class_idx))
                        targets.append(class_idx)

        if len(samples) == 0:
            raise RuntimeError("No validation images found in reorganized class-folder val format.")

        print("Validation loader: detected reorganized val/<class>/images format.")
        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_tiny_imagenet_transforms(image_size: int = 64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        ),
    ])

    return train_transform, val_transform


def get_tiny_imagenet_datasets(data_root: str, image_size: int = 64):
    train_transform, val_transform = build_tiny_imagenet_transforms(image_size=image_size)

    train_dataset = TinyImageNetTrainDataset(
        root=data_root,
        transform=train_transform
    )

    val_dataset = TinyImageNetValDataset(
        root=data_root,
        transform=val_transform
    )

    return train_dataset, val_dataset


def sanity_check_tiny_imagenet(train_dataset, val_dataset, num_classes: int = 200):
    print("=" * 60)
    print("Tiny-ImageNet Sanity Check")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Train classes: {len(set(train_dataset.targets))}")
    print(f"Val classes:   {len(set(val_dataset.targets))}")

    if len(set(train_dataset.targets)) != num_classes:
        raise ValueError(
            f"Expected {num_classes} train classes, got {len(set(train_dataset.targets))}"
        )

    if len(set(val_dataset.targets)) != num_classes:
        raise ValueError(
            f"Expected {num_classes} val classes, got {len(set(val_dataset.targets))}"
        )

    print("Sanity check passed.")
    print("=" * 60)