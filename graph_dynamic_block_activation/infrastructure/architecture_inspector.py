"""
Architecture inspector: convert a live PyTorch ResNet into a `BlockGraph`.

This is an adapter at the boundary between the framework-agnostic domain
layer (`domain.architecture`) and the PyTorch world. Its single
responsibility is to look at a torchvision ResNet object and produce
the corresponding `BlockGraph` so the rest of the GDBA algorithm can
operate on the framework-free representation.

Why a separate inspector
------------------------
The domain layer must not depend on PyTorch (see `domain.architecture`
docstring). But somewhere we need to translate between the two worlds.
Following the Adapter pattern, that translation lives in `infrastructure`,
and it has exactly one job: count blocks per stage.

"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from ..constants import RESNET_STAGE_NAMES
from ..domain.architecture import BlockGraph, build_block_graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def inspect_resnet(
    model: nn.Module,
    *,
    stage_names: Sequence[str] = RESNET_STAGE_NAMES,
) -> BlockGraph:
    """
    Build a `BlockGraph` from a torchvision ResNet model.

    The function walks the four canonical stage attributes ("layer1" ..
    "layer4" by default) and counts the residual blocks in each.

    Parameters
    ----------
    model
        A torchvision ResNet or any nn.Module exposing the named stages
        as iterable nn.Sequential containers.
    stage_names
        Names of the stage attributes to look at. Defaults to the
        torchvision convention ("layer1", "layer2", "layer3", "layer4").
        Customize this argument if you adapt a non-torchvision model.

    Returns
    -------
    BlockGraph
        Domain-level description of the model's block topology.

    Raises
    ------
    AttributeError
        If `model` is missing one of the expected stage attributes.
    ValueError
        If a stage exists but is empty (zero blocks), which indicates
        either a malformed model or wrong `stage_names`.
    """
    stage_specs: list[tuple[str, int]] = []

    for stage_name in stage_names:
        stage = getattr(model, stage_name, None)
        if stage is None:
            raise AttributeError(
                f"Model has no stage attribute named {stage_name!r}. "
                f"Either the model is not a ResNet variant or you need "
                f"to pass a custom `stage_names` argument."
            )

        # nn.Sequential supports len(); also handles ModuleList.
        try:
            num_blocks = len(stage)
        except TypeError as exc:
            raise TypeError(
                f"Stage {stage_name!r} of type {type(stage).__name__} "
                f"does not support len(); expected nn.Sequential or "
                f"nn.ModuleList."
            ) from exc

        if num_blocks < 1:
            raise ValueError(
                f"Stage {stage_name!r} contains zero blocks; cannot build "
                f"a meaningful BlockGraph."
            )

        stage_specs.append((stage_name, num_blocks))

    return build_block_graph(stage_specs)


def get_block_module(
    model: nn.Module,
    block_id: str,
) -> nn.Module:
    """
    Look up a single residual block by its domain-level identifier.

    Domain block IDs have the form "stage_name.index" (e.g. "layer2.3").
    This helper splits the ID and walks the model to find the actual
    nn.Module object, which is what infrastructure components (hooks,
    forward wrappers) actually operate on.

    Parameters
    ----------
    model
        The PyTorch model.
    block_id
        Block identifier in "stage.index" form.

    Returns
    -------
    nn.Module
        The block module (BasicBlock, Bottleneck, or a wrapper around
        them like StochasticDepthBlock).

    Raises
    ------
    KeyError
        If the block_id cannot be resolved.
    """
    if "." not in block_id:
        raise KeyError(
            f"Invalid block_id {block_id!r}: expected 'stage.index' form"
        )

    stage_name, index_str = block_id.split(".", maxsplit=1)
    stage = getattr(model, stage_name, None)
    if stage is None:
        raise KeyError(f"Model has no stage named {stage_name!r}")

    try:
        index = int(index_str)
    except ValueError as exc:
        raise KeyError(
            f"Block index in {block_id!r} is not an integer"
        ) from exc

    try:
        return stage[index]
    except (IndexError, KeyError) as exc:
        raise KeyError(
            f"Stage {stage_name!r} has no block at index {index}"
        ) from exc
