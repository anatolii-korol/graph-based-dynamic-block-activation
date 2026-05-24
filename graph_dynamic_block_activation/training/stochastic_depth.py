"""
Stochastic Depth (Huang et al. 2016): per-block dropout for residual networks.

During training, this module wraps each residual block of a ResNet so
that, with some probability, the residual branch F(x) is dropped while
the skip connection x is preserved:

    H_l = ReLU(b_l * F(x_{l-1}) + id(x_{l-1})),    b_l ~ Bernoulli(1 - p_l)

At eval time the block runs in full (no dropping), so the wrapper has
zero inference overhead.

Why this matters for GDBA
------------------------
GDBA disables residual blocks at inference time. A model trained without
stochastic depth has never seen any block "missing" during training, so
its internal feature distributions are tightly coupled to the presence
of every block. Disabling blocks at inference under those conditions
catastrophically degrades accuracy (we measured a drop from 87.8% to
34.5% on CIFAR-10 at top-k = 0.5).

Stochastic depth solves this: during training, the model learns to
produce sensible outputs even when arbitrary subsets of blocks are
absent. This is the empirical prerequisite that makes GDBA practical.

Drop probability schedule
-------------------------
We use the *linear* schedule from the original paper:

    p_l = (l / (L - 1)) * p_max,    l = 0, 1, ..., L - 1

where l is the block's depth index and L is the total number of blocks.
The deepest block has drop probability p_max; the shallowest has 0.
Empirically this works better than a uniform p_max for all blocks
because deeper blocks are more redundant.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from ..constants import DEFAULT_STOCHASTIC_DEPTH_P_MAX


# ─────────────────────────────────────────────────────────────────────────────
# Block wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDepthBlock(nn.Module):
    """
    Wrap a torchvision residual block (BasicBlock or Bottleneck) so that
    its residual branch F(x) is stochastically dropped during training.

    Design choice: rather than monkey-patching the inner block's forward
    method, we inline the forward logic of BasicBlock / Bottleneck here
    and inject the StochasticDepth call between F(x) and the residual
    add. This keeps the wrapper transparent to PyTorch (a plain nn.Module
    composition) and means our forward EXACTLY matches torchvision's
    when drop_prob = 0 or during eval.

    Attribute delegation
    --------------------
    Other parts of GDBA (flops_counter, BlockActivationTracer) expect
    standard BasicBlock attributes (conv1, bn1, downsample, ...) to be
    accessible on the wrapper. We implement this via `__getattr__`,
    which Python only calls when normal attribute lookup fails, so
    there is no overhead in the common case.

    AMP safety
    ----------
    Under PyTorch AMP autocast, conv/BN outputs are fp16. torchvision's
    StochasticDepth divides by survival_rate (~0.9), which can push
    large fp16 activations past the fp16 max (~65504), producing
    Inf/NaN that propagate through the skip-connection add and corrupt
    downstream blocks. We protect against this by running StochasticDepth
    under autocast=False (i.e. in fp32) and casting back if needed.
    """

    def __init__(
        self,
        block: nn.Module,
        drop_prob: float = 0.0,
        mode: str = "row",
    ) -> None:
        super().__init__()
        self.block = block
        self.drop_prob = float(drop_prob)
        self.stochastic_depth = StochasticDepth(p=self.drop_prob, mode=mode)

    def __getattr__(self, name: str):
        """
        Delegate attribute lookups to the inner block for standard ResNet
        block attributes (conv1, bn1, downsample, ...). This lets FLOPs
        counters and tracers see through the wrapper without changes.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "block":
                # The block attribute itself failed to resolve — propagate
                # the error rather than recursing.
                raise
            block = self._modules.get("block")
            if block is not None and hasattr(block, name):
                return getattr(block, name)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Replicate the torchvision residual-block forward pass exactly,
        # splitting F(x) out so we can apply stochastic depth to it
        # before the identity add.
        b = self.block
        out = b.conv1(x)
        out = b.bn1(out)
        out = b.relu(out)

        out = b.conv2(out)
        out = b.bn2(out)

        # Bottleneck has a 3rd conv-BN pair after a ReLU.
        if hasattr(b, "conv3") and b.conv3 is not None:
            out = b.relu(out)
            out = b.conv3(out)
            out = b.bn3(out)

        # Downsample branch — only present on entry blocks of stages
        # 2..4 (and stage 1 if the input channel count differs).
        if b.downsample is not None:
            identity = b.downsample(x)

        # Apply stochastic depth in fp32 to avoid AMP overflow.
        if self.drop_prob > 0.0 and self.training:
            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
            except (AttributeError, TypeError):
                # Fallback for older PyTorch versions.
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            with autocast_ctx:
                out = self.stochastic_depth(out.float())
            # Preserve downstream dtype if AMP is active elsewhere.
            if identity.dtype != out.dtype:
                out = out.to(identity.dtype)

        out = out + identity
        out = b.relu(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Application helper
# ─────────────────────────────────────────────────────────────────────────────

def _linear_drop_schedule(n_blocks: int, p_max: float) -> list[float]:
    """
    Compute per-block drop probabilities under the linear schedule.

    Returns a list of length `n_blocks` where element l is the drop
    probability for the l-th block in forward order:

        p_l = (l / (n_blocks - 1)) * p_max,   l = 0, 1, ..., n_blocks-1

    Special case: a single block (n_blocks = 1) gets probability 0.
    """
    if n_blocks <= 1:
        return [0.0] * n_blocks
    return [(l / (n_blocks - 1)) * p_max for l in range(n_blocks)]


def apply_stochastic_depth(
    model: nn.Module,
    *,
    p_max: float = DEFAULT_STOCHASTIC_DEPTH_P_MAX,
    schedule: str = "linear",
    protect_entry_blocks: bool = True,
    mode: str = "row",
    stage_names: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
) -> dict[str, float]:
    """
    Wrap every residual block of `model` with `StochasticDepthBlock`,
    in place, using the requested per-block drop schedule.

    Parameters
    ----------
    model
        torchvision ResNet-family model to modify.
    p_max
        Maximum drop probability for the deepest block. Typical values:
          0.1 for CIFAR ResNet-18/34 (paper default)
          0.2 for ImageNet ResNet-50
          0.3 for very deep networks (ResNet-152+)
    schedule
        "linear" (recommended) or "uniform".
    protect_entry_blocks
        If True, entry blocks (layerN.0 — those that change spatial
        dimensions via downsample) keep drop_prob = 0. This mirrors the
        GDBA convention (entry blocks are forced active at inference).
        Strongly recommended unless you have a specific reason to drop them.
    mode
        StochasticDepth mode:
          "row" samples per-batch-item (stronger regularization, default)
          "batch" samples once per batch (weaker)
    stage_names
        Names of the stage attributes on `model`. Defaults to the
        torchvision ResNet convention.

    Returns
    -------
    Dict mapping each modified block name (e.g. "layer1.0") to the
    drop probability that was assigned to it. Useful for logging and
    reproducibility.

    Raises
    ------
    ValueError
        If `schedule` is not one of the recognized strategies.
    """
    # Walk all stages, collecting blocks in forward (depth) order.
    ordered_blocks: list[tuple[str, str, int, nn.Module]] = []
    for stage_name in stage_names:
        stage = getattr(model, stage_name, None)
        if stage is None:
            continue
        for idx, block in enumerate(stage):
            full_name = f"{stage_name}.{idx}"
            ordered_blocks.append((full_name, stage_name, idx, block))

    n = len(ordered_blocks)

    # Build the schedule.
    if schedule == "linear":
        probs = _linear_drop_schedule(n, p_max)
    elif schedule == "uniform":
        probs = [p_max] * n
    else:
        raise ValueError(f"Unknown stochastic-depth schedule: {schedule!r}")

    # Apply the wrappers in place.
    assigned: dict[str, float] = {}
    for (full_name, stage_name, idx, block), p in zip(ordered_blocks, probs):
        is_entry = (idx == 0)
        effective_p = 0.0 if (protect_entry_blocks and is_entry) else float(p)

        wrapped = StochasticDepthBlock(
            block, drop_prob=effective_p, mode=mode,
        )
        getattr(model, stage_name)[idx] = wrapped
        assigned[full_name] = effective_p

    return assigned


def zero_init_last_bn(model: nn.Module) -> int:
    """
    Initialize the last BatchNorm's gamma (weight) of every residual
    block to zero. This is the "zero-init BN" trick from Goyal et al.
    2017 (Appendix A).

    Why
    ---
    At initialization, F(x) = BN(...) is meant to be small relative to
    the identity branch. With default BN init (gamma = 1), the residual
    contribution at step 0 is large, which destabilizes very deep
    networks (ResNet-50+). Setting gamma = 0 makes F(x) exactly zero
    at init, so each block starts as the identity H(x) = ReLU(0 + x) = x,
    and the network can gradually learn non-trivial residuals as
    training progresses.

    The fix is harmless for shallow networks (ResNet-18/34 converge
    either way) but eliminates training failures on deep ones. We turn
    it on by default in the trainer.

    This function:
      - works on both BasicBlock (last BN = bn2) and Bottleneck (bn3).
      - transparently handles StochasticDepthBlock-wrapped blocks. When
        an SD wrapper is encountered, we operate on its inner block and
        SKIP the inner block on the subsequent iteration (Python's
        `model.modules()` walks the tree depth-first, so the wrapper is
        visited before its inner submodule). Without this guard, each
        BN would be zeroed twice — harmless but inflates the returned
        count and confuses callers.

    Returns
    -------
    Count of residual blocks whose final BN was zeroed (each block
    counted exactly once, regardless of wrapping).
    """
    from torchvision.models.resnet import BasicBlock, Bottleneck

    count = 0
    # Track which inner BasicBlock/Bottleneck instances we have already
    # processed via their SD wrapper, so we skip them on the second visit.
    already_seen: set[int] = set()

    for m in model.modules():
        # Case 1: SD wrapper — operate on its inner block, then mark it.
        inner = getattr(m, "block", None)
        if isinstance(m, nn.Module) and isinstance(inner, (BasicBlock, Bottleneck)) and inner is not m:
            if isinstance(inner, Bottleneck):
                nn.init.zeros_(inner.bn3.weight)
            else:  # BasicBlock
                nn.init.zeros_(inner.bn2.weight)
            already_seen.add(id(inner))
            count += 1
            continue

        # Case 2: bare residual block (no wrapper).
        if isinstance(m, Bottleneck):
            if id(m) in already_seen:
                continue
            nn.init.zeros_(m.bn3.weight)
            count += 1
        elif isinstance(m, BasicBlock):
            if id(m) in already_seen:
                continue
            nn.init.zeros_(m.bn2.weight)
            count += 1

    return count


def get_stochastic_depth_state(
    model: nn.Module,
    stage_names: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
) -> dict[str, float]:
    """
    Introspect the drop probabilities currently applied to a model.

    Useful for verifying that `apply_stochastic_depth` did what was
    expected, and for logging the SD configuration into checkpoint
    metadata.

    Returns
    -------
    Dict from block name to drop probability. Blocks that are NOT
    wrapped (e.g. plain BasicBlock) are omitted from the result.
    """
    out: dict[str, float] = {}
    for stage_name in stage_names:
        stage = getattr(model, stage_name, None)
        if stage is None:
            continue
        for idx, block in enumerate(stage):
            if isinstance(block, StochasticDepthBlock):
                out[f"{stage_name}.{idx}"] = block.drop_prob
    return out
