"""
Analytical FLOPs counter with gate-aware effective-FLOPs reporting.

This module computes the floating-point operation count of a ResNet
forward pass by analyzing each module's input shape and applying closed-
form FLOPs formulas for Conv2d, BatchNorm2d, Linear, etc. The analytical
approach is preferred over runtime hooks for two reasons:

  1. **Determinism**: analytical FLOPs depend only on architecture and
     input shape, not on which kernels happen to be called. Two
     independent runs of the same model on the same shape always agree.

  2. **Gate awareness**: by computing FLOPs *per block*, we can sum
     only the blocks that are actually active, giving exact "effective
     FLOPs" for any gate map without re-running the model.

The implementation is split into two phases:

  - Tracing: a single forward pass records the input shape arriving at
    each residual block. Hooks are removed immediately after.

  - Accounting: per-block FLOPs are computed by walking the BasicBlock /
    Bottleneck structure (conv1, bn1, conv2, bn2, [conv3, bn3], downsample)
    and applying analytical formulas to each submodule.

Conventions
-----------
* FLOPs include both multiplications and additions of multiply-accumulate
  operations: 1 MAC = 2 FLOPs. This matches the convention used by
  `fvcore` (in MACs) when we multiply by 2.
* BatchNorm contributes 4 FLOPs per element (subtract mean, divide by std,
  scale, bias). This is an approximation: fused implementations are
  cheaper, but the proportional contribution is small (< 1% of total).
* Activations (ReLU, etc.) contribute 1 FLOP per element.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from ..domain.architecture import BlockGraph, BlockId


# ─────────────────────────────────────────────────────────────────────────────
# Per-module FLOPs formulas
# ─────────────────────────────────────────────────────────────────────────────

def _conv2d_flops(conv: nn.Conv2d, input_shape: torch.Size) -> float:
    """
    FLOPs for a Conv2d call with the given input shape.

    For input (N, C_in, H_in, W_in) and a Conv2d producing
    (N, C_out, H_out, W_out), the number of multiply-accumulate ops is:

        N * C_out * H_out * W_out * (C_in / groups) * kH * kW

    Each MAC is 2 FLOPs (one multiplication + one addition).
    """
    n = int(input_shape[0])
    kh, kw = (conv.kernel_size if isinstance(conv.kernel_size, tuple)
              else (conv.kernel_size, conv.kernel_size))
    sh, sw = (conv.stride if isinstance(conv.stride, tuple)
              else (conv.stride, conv.stride))
    ph, pw = (conv.padding if isinstance(conv.padding, tuple)
              else (conv.padding, conv.padding))
    dh, dw = (conv.dilation if isinstance(conv.dilation, tuple)
              else (conv.dilation, conv.dilation))

    h_in, w_in = int(input_shape[2]), int(input_shape[3])
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    macs = n * conv.out_channels * h_out * w_out * (conv.in_channels // conv.groups) * kh * kw
    return 2.0 * macs


def _conv2d_output_shape(conv: nn.Conv2d, input_shape: torch.Size) -> torch.Size:
    """Compute the spatial output shape of a Conv2d. Mirrors _conv2d_flops."""
    n = int(input_shape[0])
    kh, kw = (conv.kernel_size if isinstance(conv.kernel_size, tuple)
              else (conv.kernel_size, conv.kernel_size))
    sh, sw = (conv.stride if isinstance(conv.stride, tuple)
              else (conv.stride, conv.stride))
    ph, pw = (conv.padding if isinstance(conv.padding, tuple)
              else (conv.padding, conv.padding))
    dh, dw = (conv.dilation if isinstance(conv.dilation, tuple)
              else (conv.dilation, conv.dilation))

    h_in, w_in = int(input_shape[2]), int(input_shape[3])
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    return torch.Size([n, conv.out_channels, h_out, w_out])


def _bn2d_flops(bn: nn.BatchNorm2d, input_shape: torch.Size) -> float:
    """BatchNorm: 4 FLOPs per element (sub-mean, div-std, scale, shift)."""
    n, c, h, w = [int(s) for s in input_shape]
    return 4.0 * n * c * h * w


def _linear_flops(linear: nn.Linear, input_shape: torch.Size) -> float:
    """Linear: 2 FLOPs per MAC; batch is product of leading dims."""
    batch = 1
    for s in input_shape[:-1]:
        batch *= int(s)
    return 2.0 * batch * linear.in_features * linear.out_features


# ─────────────────────────────────────────────────────────────────────────────
# Block-level FLOPs (residual blocks)
# ─────────────────────────────────────────────────────────────────────────────

def _residual_block_flops(block: nn.Module, input_shape: torch.Size) -> float:
    """
    Sum FLOPs for one BasicBlock / Bottleneck given its input shape.

    Walks conv1->bn1 [->relu] ->conv2->bn2 [->conv3->bn3 for Bottleneck]
    plus the optional downsample branch. The wrapping `StochasticDepthBlock`
    (which our trainer adds) is transparently handled via attribute
    delegation; `block.conv1` etc. resolve to the inner BasicBlock.
    """
    total = 0.0
    current_shape = input_shape

    # Main residual branch: conv1, conv2, optional conv3
    for i in (1, 2, 3):
        conv = getattr(block, f"conv{i}", None)
        bn = getattr(block, f"bn{i}", None)
        if conv is None:
            break
        total += _conv2d_flops(conv, current_shape)
        current_shape = _conv2d_output_shape(conv, current_shape)
        if bn is not None:
            total += _bn2d_flops(bn, current_shape)

    # Downsample branch (only present on entry blocks of stages 2-4 and
    # the entry of stage 1 when channels change).
    downsample = getattr(block, "downsample", None)
    if downsample is not None:
        ds_shape = input_shape
        for sub in downsample:
            if isinstance(sub, nn.Conv2d):
                total += _conv2d_flops(sub, ds_shape)
                ds_shape = _conv2d_output_shape(sub, ds_shape)
            elif isinstance(sub, nn.BatchNorm2d):
                total += _bn2d_flops(sub, ds_shape)

    return total


def _stem_and_head_flops(model: nn.Module, sample_input: torch.Tensor) -> float:
    """
    FLOPs of everything outside the four residual stages: stem + head.

    Stem: conv1 + bn1 + relu [+ maxpool, if not Identity].
    Head: avgpool + fc.

    These are always executed regardless of gate state.
    """
    total = 0.0

    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        total += _conv2d_flops(model.conv1, sample_input.shape)
        stem_shape = _conv2d_output_shape(model.conv1, sample_input.shape)
        if hasattr(model, "bn1") and isinstance(model.bn1, nn.BatchNorm2d):
            total += _bn2d_flops(model.bn1, stem_shape)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        n = int(sample_input.shape[0])
        total += _linear_flops(
            model.fc, torch.Size([n, model.fc.in_features])
        )

    return total


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Distinguished key for stem + head FLOPs in the per-block dict.
STEM_AND_HEAD_KEY: str = "__stem_head__"


def compute_block_flops(
    model: nn.Module,
    graph: BlockGraph,
    sample_input: torch.Tensor,
) -> dict[str, float]:
    """
    Analytical FLOPs per residual block, plus a stem-and-head entry.

    The returned dict contains:
      - One key per block ID (e.g. "layer1.0") with that block's FLOPs.
      - One special key `STEM_AND_HEAD_KEY` = `"__stem_head__"` with the
        cumulative FLOPs of stem + head (always-executed parts).

    Phase 1 (tracing): a single forward pass records the input shape that
    arrives at each residual block. Hooks are removed before returning.

    Phase 2 (accounting): walks each block and applies analytical
    formulas. No further model interaction.

    Parameters
    ----------
    model
        The ResNet to analyze. Must be in eval mode for stable shape
        tracing.
    graph
        Domain block graph; provides the block IDs to look up.
    sample_input
        A representative input tensor (same shape as inference inputs).
        Batch size matters: FLOPs scale linearly with N.

    Returns
    -------
    Dict mapping block_id (or "__stem_head__") to FLOPs as a float.
    """
    # ── Phase 1: trace input shapes ──────────────────────────────────────
    input_shapes: dict[BlockId, torch.Size] = {}
    handles = []

    def _make_shape_hook(block_id: str):
        def _hook(_module, inputs, _output):
            if inputs and isinstance(inputs[0], torch.Tensor):
                input_shapes[block_id] = inputs[0].shape
        return _hook

    block_id_set = set(graph.all_block_ids)
    for name, module in model.named_modules():
        if name in block_id_set:
            handles.append(module.register_forward_hook(_make_shape_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # ── Phase 2: per-block FLOPs ─────────────────────────────────────────
    flops: dict[str, float] = {}
    for stage in graph.stages:
        stage_module = getattr(model, stage.name)
        for idx, block_id in enumerate(stage.block_ids):
            input_shape = input_shapes.get(block_id)
            if input_shape is None:
                flops[block_id] = 0.0
                continue
            flops[block_id] = _residual_block_flops(
                stage_module[idx], input_shape
            )

    flops[STEM_AND_HEAD_KEY] = _stem_and_head_flops(model, sample_input)
    return flops


def compute_effective_flops(
    block_flops: Mapping[str, float],
    gates: Mapping[BlockId, int],
    graph: BlockGraph,
) -> float:
    """
    Sum the FLOPs of all *active* blocks plus stem + head.

    The gate semantic is: gate[bid] == 0 means the block is skipped and
    contributes zero FLOPs. Entry blocks are always counted regardless
    of their gate value (the wrapper enforces this in the forward pass,
    and we mirror it here).

    Parameters
    ----------
    block_flops
        Output of `compute_block_flops`.
    gates
        Gate map produced by `domain.selection.TopKSelector.select`.
        Missing block IDs default to "active" (1).
    graph
        Domain graph; needed to identify entry blocks.

    Returns
    -------
    Total effective FLOPs.
    """
    entries = graph.entry_block_ids
    total = block_flops.get(STEM_AND_HEAD_KEY, 0.0)

    for block_id in graph.all_block_ids:
        if block_id in entries:
            # Entry blocks always run.
            total += block_flops.get(block_id, 0.0)
            continue
        gate = gates.get(block_id, 1)
        if int(gate) == 0:
            continue
        total += block_flops.get(block_id, 0.0)

    return total
