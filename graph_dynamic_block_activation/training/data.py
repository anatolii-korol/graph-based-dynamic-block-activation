"""
CIFAR-10/100 data loaders with the standard augmentation pipeline.

Pipeline (train)
----------------
  RandomCrop(32, padding=4, padding_mode=reflect)
  RandomHorizontalFlip()
  ToTensor()
  Normalize(mean=CIFAR_*_MEAN, std=CIFAR_*_STD)

Pipeline (val/test)
-------------------
  ToTensor()
  Normalize(...)

The augmentation choices are standard for CIFAR ResNet papers since
He et al. 2016. We make no modifications.

Train/val split
---------------
We carve a stratified validation split out of the official training
set (default 10%). Stratification preserves the original class
distribution in both halves so that validation accuracy is a good
predictor of test accuracy.

drop_last behavior
------------------
The train loader uses `drop_last=True` to skip the trailing partial
batch. This is the standard practice from torchvision reference scripts
and serves three purposes:

  1. Keeps BatchNorm running statistics consistent across iterations
     (same sample count -> same variance scaling).
  2. Prevents fp16 overflow in AMP training when a small final batch
     has unusually large activation variance.
  3. Required by some training techniques (MixUp, distributed sync) that
     assume uniform batch sizes.

The fraction discarded is ~0.16% per epoch at batch_size=128 — well
below the noise floor across random seeds. Val/test loaders use
`drop_last=False` because we want every sample evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ..constants import (
    CIFAR10_NORMALIZATION_MEAN,
    CIFAR10_NORMALIZATION_STD,
    CIFAR100_NORMALIZATION_MEAN,
    CIFAR100_NORMALIZATION_STD,
    CIFAR_NATIVE_RESOLUTION,
    CIFAR_RANDOM_CROP_PADDING,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataBundle:
    """
    Container for the four data artifacts produced by `build_loaders`.

    Attributes
    ----------
    train, val, test
        The underlying Dataset / Subset objects, exposed in case the
        caller wants to do something beyond looping (e.g. compute class
        weights).
    train_loader, val_loader, test_loader
        Configured DataLoader instances ready for the training loop.
    num_classes
        Convenience field — 10 for CIFAR-10, 100 for CIFAR-100.
    """

    train: Dataset
    val: Dataset
    test: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int


# ─────────────────────────────────────────────────────────────────────────────
# Normalization parameters
# ─────────────────────────────────────────────────────────────────────────────

def _normalization_params(dataset: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (mean, std) channel statistics for the named CIFAR variant."""
    name = dataset.lower()
    if name == "cifar10":
        return CIFAR10_NORMALIZATION_MEAN, CIFAR10_NORMALIZATION_STD
    if name == "cifar100":
        return CIFAR100_NORMALIZATION_MEAN, CIFAR100_NORMALIZATION_STD
    raise ValueError(f"Unsupported dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(
    dataset: str,
    *,
    image_size: int = CIFAR_NATIVE_RESOLUTION,
    augment: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build the (train, test) torchvision transform pipelines.

    Parameters
    ----------
    dataset
        "cifar10" or "cifar100".
    image_size
        Output resolution. Defaults to 32 (native CIFAR). Pass a larger
        value to upsample for ImageNet-style backbones (e.g. 224 for a
        full-resolution ResNet stem).
    augment
        If True, the train pipeline applies random crop + horizontal
        flip; if False, train and test pipelines are identical.

    Returns
    -------
    (train_transform, test_transform) — torchvision Compose objects.
    """
    mean, std = _normalization_params(dataset)

    # Train pipeline
    train_ops: list = []
    if augment:
        train_ops.extend([
            transforms.RandomCrop(
                CIFAR_NATIVE_RESOLUTION,
                padding=CIFAR_RANDOM_CROP_PADDING,
            ),
            transforms.RandomHorizontalFlip(),
        ])
    if image_size != CIFAR_NATIVE_RESOLUTION:
        # Upsample AFTER the 32x32 crop/flip to keep augmentation
        # operating in the native pixel grid.
        train_ops.append(transforms.Resize((image_size, image_size)))
    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test pipeline — no augmentation
    test_ops: list = []
    if image_size != CIFAR_NATIVE_RESOLUTION:
        test_ops.append(transforms.Resize((image_size, image_size)))
    test_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    dataset: str,
    root: str,
    *,
    train: bool,
    transform,
) -> Dataset:
    """Construct one CIFAR dataset, downloading if necessary."""
    name = dataset.lower()
    if name == "cifar10":
        return datasets.CIFAR10(
            root=root, train=train, transform=transform, download=True,
        )
    if name == "cifar100":
        return datasets.CIFAR100(
            root=root, train=train, transform=transform, download=True,
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Stratified train/val split
# ─────────────────────────────────────────────────────────────────────────────

def _targets_array(ds: Dataset) -> np.ndarray:
    """Extract the integer label array from a CIFAR-like dataset."""
    if hasattr(ds, "targets"):
        return np.asarray(getattr(ds, "targets"))
    raise ValueError(
        f"Dataset {type(ds).__name__} does not expose a `targets` attribute"
    )


def stratified_split_indices(
    targets: Sequence[int],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Split sample indices into train/val groups, preserving per-class
    proportions.

    For each class, we randomly assign ceil(N_c * val_ratio) samples to
    validation and the rest to training, where N_c is the number of
    samples of that class.

    Parameters
    ----------
    targets
        Integer label array, length = total dataset size.
    val_ratio
        Fraction of samples (per class) to use for validation.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    (train_indices, val_indices) — two disjoint lists of integers that
    together cover [0, len(targets)).
    """
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio}")

    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    # Final shuffle so the train loader doesn't see classes in order.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    dataset: str,
    *,
    root: str,
    batch_size: int,
    num_workers: int,
    image_size: int = CIFAR_NATIVE_RESOLUTION,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
) -> DataBundle:
    """
    One-call constructor for train, val, and test loaders.

    Parameters
    ----------
    dataset
        "cifar10" or "cifar100".
    root
        Filesystem path where the dataset is stored (downloaded if absent).
    batch_size
        Batch size for all three loaders. Val and test do not honor
        `drop_last`.
    num_workers
        Number of data-loading worker processes per loader.
    image_size
        Image resolution after transforms (default 32 = native CIFAR).
    val_ratio
        Fraction of the official training set carved out for validation.
    seed
        Seed for the stratified split.
    augment
        Whether to apply train-time augmentation (RandomCrop + Flip).

    Returns
    -------
    DataBundle with all six artifacts populated.
    """
    train_transform, test_transform = build_transforms(
        dataset, image_size=image_size, augment=augment,
    )

    # Two views of the training data: one with augmentation (for train
    # split) and one without (for val split). They share the underlying
    # image files but apply different transforms at load time.
    train_full_aug = build_dataset(
        dataset, root, train=True, transform=train_transform,
    )
    train_full_noaug = build_dataset(
        dataset, root, train=True, transform=test_transform,
    )
    test_dataset = build_dataset(
        dataset, root, train=False, transform=test_transform,
    )

    # Stratified split using only the augmented view's labels (same
    # underlying data, so labels match).
    targets = _targets_array(train_full_aug)
    train_idx, val_idx = stratified_split_indices(
        targets, val_ratio=val_ratio, seed=seed,
    )

    train_dataset = Subset(train_full_aug, train_idx)
    val_dataset = Subset(train_full_noaug, val_idx)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last=True for BatchNorm/AMP stability — see module docstring.
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = 10 if dataset.lower() == "cifar10" else 100

    return DataBundle(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
    )
