"""

PyTorch Dataset and DataLoader utilities for CGR image classification.

"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Class definitions

# Binary classification: resistant vs non-resistant
CLASSES_BINARY = ["non_resistant", "resistant"]

# Label map: folder name -> integer label
BINARY_LABEL_MAP = {
    "resistant":     1,
    "non_resistant": 0,
}

# Alias for any code that references CLASSES_MULTICLASS
CLASSES_MULTICLASS = CLASSES_BINARY


# Transforms

# ImageNet normalization stats (used since models are ImageNet pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Return appropriate transforms for train/val/test splits.

    Training: random flips, small rotations, color jitter for augmentation.
    Val/Test: only resize and normalize (no augmentation).

    Horizontal flips of CGR images correspond to A-T swaps,
    which is biologically meaningful (complement strand reading direction).
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# Dataset class

class CGRDataset(Dataset):
    """
    Dataset for CGR image classification.

    Directory structure expected:
        data_dir/
            beta_lactam/      *.png
            tetracycline/     *.png
            ...
            non_resistant/    *.png

    Args:
        file_paths: list of (image_path, label_int) tuples
        transform:  torchvision transform pipeline
        task:       "binary" or "multiclass"
    """

    def __init__(
        self,
        file_paths: list[tuple[Path, int]],
        transform: Optional[transforms.Compose] = None,
        task: str = "multiclass",
    ):
        self.samples = file_paths
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_labels(self) -> list[int]:
        """Return list of all labels (for computing class weights)."""
        return [label for _, label in self.samples]


# Data loading utilities

def collect_samples(data_dir: Path, task: str) -> list[tuple[Path, int]]:
    """
    Scan data_dir for PNG images and return (path, label) pairs.
    Always uses binary labels: resistant=1, non_resistant=0.
    """
    samples = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in BINARY_LABEL_MAP:
            continue

        label = BINARY_LABEL_MAP[class_name]
        png_files = list(class_dir.glob("*.png"))

        if not png_files:
            print(f"  WARNING: No images found in {class_dir}")
            continue

        for img_path in png_files:
            samples.append((img_path, label))

    return samples


def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for handling class imbalance.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str | Path,
    task: str = "multiclass",
    batch_size: int = 32,
    image_size: int = 224,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train/val/test DataLoaders with stratified splitting.

    Args:
        data_dir:              directory with per-class image subdirs
        task:                  "binary" or "multiclass"
        batch_size:            samples per batch
        image_size:            resize target (default 224)
        val_size:              fraction for validation (default 0.15)
        test_size:             fraction for test (default 0.15)
        seed:                  random seed for reproducibility
        num_workers:           DataLoader workers
        use_weighted_sampler:  balance classes in training batches

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = Path(data_dir)
    samples = collect_samples(data_dir, task)

    if not samples:
        raise ValueError(f"No samples found in {data_dir} for task='{task}'")

    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]
    n_classes = 2 if task == "binary" else len(CLASSES_MULTICLASS)

    print(f"\nDataset summary ({task}):")
    class_names = CLASSES_BINARY if task == "binary" else CLASSES_MULTICLASS
    label_counts = np.bincount(labels, minlength=n_classes)
    for i, name in enumerate(class_names):
        print(f"  {name:<25} {label_counts[i]:>5}")
    print(f"  {'TOTAL':<25} {len(samples):>5}\n")

    # Stratified 70/15/15 split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=seed
    )
    val_fraction = val_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_fraction),
        stratify=temp_labels,
        random_state=seed
    )

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Build datasets
    train_dataset = CGRDataset(
        list(zip(train_paths, train_labels)),
        transform=get_transforms("train", image_size),
        task=task
    )
    val_dataset = CGRDataset(
        list(zip(val_paths, val_labels)),
        transform=get_transforms("val", image_size),
        task=task
    )
    test_dataset = CGRDataset(
        list(zip(test_paths, test_labels)),
        transform=get_transforms("test", image_size),
        task=task
    )

    # Weighted sampler for training (handles class imbalance)
    sampler = None
    if use_weighted_sampler:
        class_weights = compute_class_weights(train_labels, n_classes)
        sample_weights = [class_weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, class_names
