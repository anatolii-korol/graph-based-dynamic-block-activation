"""
PyTorch wrapper that applies block-level gates to a ResNet forward pass.

A `GatedResNet` takes an existing torchvision ResNet (already loaded with
weights) and adds the ability to skip individual residual blocks at
inference time according to a gate map. Entry blocks of each stage are
always run regardless of the gate value, because skipping them would
break the shape contract between stages.

This wrapper deliberately implements **hard gating only**: a block is
either fully executed or fully skipped.

Design notes
------------
The forward pass of a residual ResNet is straightforward:

    x = stem(x)                       # conv1 + bn1 + relu + maxpool
    for stage in stages:
        for block in stage:
            x = block(x)              # already does x = x + F(x) internally
    x = avgpool(x)
    x = flatten(x)
    return fc(x)

We replicate this exactly, but interleave a gate check before each block
call. When `gate == 0`, we simply skip the call — this means the residual
connection collapses to the identity branch `x = x`, which is the
intended behavior of "block off" in a residual network. No special
handling of the residual connection is needed because the block itself
is what implements it; skipping the block leaves `x` unchanged.

Memory and performance
----------------------
This is a thin Python wrapper over an existing `nn.Module`. It does not
copy weights, allocate new buffers, or trace anything. The only overhead
is the dictionary lookup for each gate, which is in the nanosecond range
and negligible compared to a single convolution.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from ..domain.architecture import BlockGraph, BlockId

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

GateMap = Mapping[BlockId, int]
"""Binary gate map produced by `domain.selection.TopKSelector`."""


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GatedResNet(nn.Module):
    """
    A ResNet wrapper that runs only the blocks marked active by a gate map.

    The wrapper does NOT modify the underlying model's parameters or
    state. It only intercepts the forward pass to decide which residual
    blocks to call. Entry blocks (the first block of each stage) are
    always called regardless of their gate value, because they perform
    spatial down-sampling and channel expansion.

    Use `set_gates()` to update the active block set; this is fast (just
    a dictionary copy). Use `clear_gates()` to revert to "all blocks
    active" behavior.

    Example
    -------
        graph = inspect_resnet(model)
        wrapper = GatedResNet(model, graph)
        wrapper.set_gates({"layer1.1": 1, "layer1.2": 0, ...})
        with torch.no_grad():
            logits = wrapper(x)

    Parameters
    ----------
    backbone
        A torchvision ResNet (or compatible nn.Module) already loaded
        with trained weights.
    graph
        The corresponding domain BlockGraph. Used to know which blocks
        exist, which are entry blocks, and the iteration order.
    """

    def __init__(self, backbone: nn.Module, graph: BlockGraph) -> None:
        super().__init__()
        self._backbone = backbone
        self._graph = graph
        # Cache the set of entry block IDs for O(1) lookup in forward().
        self._entry_block_ids: frozenset[BlockId] = graph.entry_block_ids
        # Cache stage iteration order: list of (stage_name, [block_ids]).
        self._stage_iter_order = tuple(
            (stage.name, stage.block_ids) for stage in graph.stages
        )
        # Active set of *gated-off* block IDs. Empty = full network runs.
        # Using a set rather than the full gate map for cheaper membership
        # tests in the forward path.
        self._gated_off: frozenset[BlockId] = frozenset()

    # ── Property exposure (read-only) ─────────────────────────────────────

    @property
    def backbone(self) -> nn.Module:
        """The underlying (unmodified) ResNet model."""
        return self._backbone

    @property
    def graph(self) -> BlockGraph:
        """The block graph this wrapper was built for."""
        return self._graph

    @property
    def gated_off_blocks(self) -> frozenset[BlockId]:
        """The set of blocks currently disabled."""
        return self._gated_off

    @property
    def num_active_blocks(self) -> int:
        """Number of blocks currently active (entry + non-entry)."""
        return self._graph.num_blocks - len(self._gated_off)

    # ── Gate management ──────────────────────────────────────────────────

    def set_gates(self, gates: GateMap) -> None:
        """
        Replace the current gate map.

        Entry blocks are always kept active regardless of what `gates`
        says; an explicit `0` for an entry block is silently ignored.
        This is a safety net: the domain selector already respects this
        rule, but a defense-in-depth check at the wrapper boundary
        prevents user errors from crashing the forward pass.

        Parameters
        ----------
        gates
            Map from block_id to gate value. Missing block_ids default
            to "active" (1).
        """
        gated_off_set: set[BlockId] = set()
        for block_id in self._graph.all_block_ids:
            if block_id in self._entry_block_ids:
                continue  # entry blocks are always on
            value = gates.get(block_id, 1)
            if int(value) == 0:
                gated_off_set.add(block_id)
        self._gated_off = frozenset(gated_off_set)

    def clear_gates(self) -> None:
        """Revert to "all blocks active" behavior."""
        self._gated_off = frozenset()

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the gated forward pass.

        The structure mirrors torchvision.models.resnet.ResNet.forward
        exactly except for the per-block gating logic in the stage loops.
        """
        m = self._backbone

        # Stem (always run): initial conv, BN, ReLU, and optional maxpool.
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        # CIFAR variant of ResNet sets maxpool to nn.Identity() — we
        # call it either way; for nn.Identity the call is a no-op.
        if hasattr(m, "maxpool"):
            x = m.maxpool(x)

        # Residual stages with gating.
        for stage_name, block_ids in self._stage_iter_order:
            stage_module = getattr(m, stage_name)
            for idx, block_id in enumerate(block_ids):
                # Entry blocks are always run — see set_gates() for the
                # rationale.
                if block_id in self._entry_block_ids:
                    x = stage_module[idx](x)
                    continue
                # Non-entry: skip if gated off, else run.
                if block_id in self._gated_off:
                    continue  # skip: x unchanged (identity through residual)
                x = stage_module[idx](x)

        # Head (always run): adaptive pooling, flatten, fully-connected.
        if hasattr(m, "avgpool"):
            x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return m.fc(x)
