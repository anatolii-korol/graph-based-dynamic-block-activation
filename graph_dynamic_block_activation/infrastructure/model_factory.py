"""
Factory for building ResNet models from configuration.

This module centralizes the logic for instantiating torchvision ResNets
with the CIFAR-style stem (3x3 conv, no maxpool) when the dataset is
CIFAR. Other parts of the codebase should call `build_model()` rather
than reaching into torchvision directly.

The factory also supports loading weights from a checkpoint, optionally
applying Stochastic Depth wrappers if the checkpoint was trained with SD
(detected automatically from state_dict key naming).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

from ..training.stochastic_depth import apply_stochastic_depth
from .checkpoint import (
    CheckpointMetadata,
    detect_stochastic_depth_keys,
    load_checkpoint,
)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture registry
# ─────────────────────────────────────────────────────────────────────────────

# Map from model name string to (torchvision factory, expected_fc_in_features).
# We use a dict rather than a chain of if/elif to keep the function below
# minimal and to make adding new architectures a one-line change.
_RESNET_FACTORIES = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
    "resnet101": tvm.resnet101,
}


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR stem patch
# ─────────────────────────────────────────────────────────────────────────────

def _apply_cifar_stem(model: nn.Module) -> None:
    """
    Replace the ImageNet stem (7x7 conv stride 2 + maxpool) with the
    CIFAR-style stem (3x3 conv stride 1, no maxpool).

    Why
    ---
    The original ImageNet stem aggressively downsamples 224x224 input
    to 56x56. Applying it to 32x32 CIFAR images would yield 8x8 feature
    maps before stage 1 even starts, leaving almost no spatial
    information for the deeper stages.

    The CIFAR-adapted stem (He et al. 2016, Section 4.2) preserves the
    full 32x32 resolution into stage 1, which then naturally downsamples
    through stages 2-4.
    """
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    model.maxpool = nn.Identity()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    *,
    num_classes: int,
    cifar_stem: bool = True,
) -> nn.Module:
    """
    Construct a ResNet variant ready for training or inference.

    Parameters
    ----------
    model_name
        Architecture identifier. See `_RESNET_FACTORIES` for the
        supported set.
    num_classes
        Output dimensionality of the final FC layer.
    cifar_stem
        If True (default), replace the ImageNet stem with the
        CIFAR-style 3x3 conv. Set to False when training on
        upsampled CIFAR (image_size >= 96) or actual ImageNet.

    Returns
    -------
    nn.Module ready to use; not yet on any device.

    Raises
    ------
    KeyError
        If `model_name` is not in the registry.
    """
    factory = _RESNET_FACTORIES.get(model_name)
    if factory is None:
        raise KeyError(
            f"Unknown model {model_name!r}. "
            f"Supported: {sorted(_RESNET_FACTORIES.keys())}"
        )

    # Build with random weights — pre-trained ImageNet weights are NOT
    # used in our protocol because the CIFAR stem patch makes them
    # geometrically incompatible.
    model = factory(weights=None)

    if cifar_stem:
        _apply_cifar_stem(model)

    # Replace the FC layer to match num_classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    cifar_stem: bool = True,
) -> tuple[nn.Module, CheckpointMetadata]:
    """
    Build a model and load weights from a checkpoint file in one step.

    The function:
      1. Loads the checkpoint payload and inspects metadata.
      2. Builds a matching architecture via `build_model()`.
      3. If the checkpoint was trained with Stochastic Depth (detected
         via state_dict key naming), wraps the blocks with
         `StochasticDepthBlock` before loading weights — otherwise the
         state_dict load would fail with shape mismatch.
      4. Loads the weights into the model.

    Parameters
    ----------
    checkpoint_path
        Path to the .pt file.
    map_location
        Device for tensor loading. "cpu" is safest; move to GPU later.
    cifar_stem
        Whether to apply the CIFAR stem. Should match how the checkpoint
        was trained.

    Returns
    -------
    Tuple of (loaded_model, metadata). The metadata fields can be used
    to reconstruct training configuration or verify compatibility.

    Raises
    ------
    ValueError
        If the metadata is missing fields needed to build the model
        and they cannot be inferred from the state_dict.
    """
    state_dict, metadata = load_checkpoint(checkpoint_path, map_location=map_location)

    if metadata.model_name is None:
        raise ValueError(
            "Checkpoint metadata is missing model_name. Cannot infer "
            "architecture; please re-save the checkpoint with metadata."
        )
    if metadata.num_classes is None:
        raise ValueError(
            "Checkpoint metadata is missing num_classes."
        )

    model = build_model(
        model_name=metadata.model_name,
        num_classes=metadata.num_classes,
        cifar_stem=cifar_stem,
    )

    # If the checkpoint was trained with Stochastic Depth, the saved
    # parameter names include the `block.` prefix introduced by
    # StochasticDepthBlock. We must wrap the model the same way before
    # loading, otherwise load_state_dict will fail with key mismatches.
    if detect_stochastic_depth_keys(state_dict):
        p_max = metadata.stochastic_depth_p_max
        if p_max is None:
            # Metadata didn't record the exact value but keys say SD was
            # used. Fall back to a reasonable default — the actual drop
            # probability is irrelevant at inference time (SD is a no-op
            # in eval mode).
            p_max = 0.1
        apply_stochastic_depth(model, p_max=p_max)

    model.load_state_dict(state_dict)
    return model, metadata
