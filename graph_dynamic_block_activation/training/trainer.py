"""
Training loop for ResNet classifiers with optional Stochastic Depth.

This module implements the standard SGD-with-momentum training protocol
that produced the checkpoints used in the paper. Key features:

  - Cosine learning-rate schedule with linear warmup
  - Stochastic Depth (Huang et al. 2016) applied before training
  - Zero-init residual BN (Goyal et al. 2017) for deep-network stability
  - Automatic Mixed Precision (AMP) on CUDA
  - Best-checkpoint saving by validation accuracy
  - Early stopping on validation accuracy plateau

Training is *not* required to use GDBA — GDBA is a zero-shot method
applied at inference. However, GDBA only works well on models that were
trained with Stochastic Depth, so this trainer is the recommended way
to produce GDBA-ready checkpoints.

Numerical stability notes
-------------------------
Deep ResNets (50+) on small datasets occasionally produce NaN losses
during AMP training, typically caused by activation overflow in fp16.
The trainer detects these batches and skips the optimizer step rather
than letting NaN poison the parameter update. A small number of skipped
batches per epoch (< 1%) is normal and does not affect final accuracy;
a large number indicates a learning-rate or scaling problem that the
user should address.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from ..constants import (
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_STOCHASTIC_DEPTH_P_MAX,
    DEFAULT_WEIGHT_DECAY,
)
from ..infrastructure.checkpoint import CheckpointMetadata, save_checkpoint
from .data import DataBundle
from .stochastic_depth import apply_stochastic_depth, zero_init_last_bn


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingConfig:
    """
    Hyperparameters for one training run.

    Defaults come from `constants.py` and reflect the paper's protocol.
    All fields are documented so a reviewer can audit each choice.

    SGD hyperparameters
    -------------------
    learning_rate
        Initial LR. The cosine schedule decays this to 0 over `epochs`.
    momentum
        SGD momentum. 0.9 is standard for CIFAR / ImageNet vision.
    weight_decay
        L2 regularization strength. 5e-4 is the CIFAR ResNet default.

    Schedule
    --------
    epochs
        Total training epochs (including warmup).
    warmup_epochs
        Linear warmup from 0 to learning_rate over this many epochs at
        the start. Crucial for deep networks (ResNet-50+): without
        warmup, large initial gradients destabilize training.
    label_smoothing
        Smoothing factor for cross-entropy. 0.1 reduces overfitting.

    Stochastic Depth
    ----------------
    stochastic_depth_p_max
        Maximum drop probability for the deepest block. 0 disables SD.
    stochastic_depth_schedule
        "linear" (paper default) or "uniform".
    stochastic_depth_protect_entry
        If True (recommended), entry blocks are not dropped — keeps the
        shape contract intact between stages.
    stochastic_depth_mode
        "row" (per-sample sampling, stronger) or "batch".

    Stabilization
    -------------
    zero_init_residual_bn
        Apply Goyal et al. 2017 zero-init BN to deep residual blocks.

    Mixed precision
    ---------------
    use_amp
        Enable AMP (fp16) training on CUDA. ~2x speedup, no accuracy
        impact in our experiments.
    grad_clip_norm
        Gradient norm clipping threshold. 0 disables. Recommended
        value: 1.0 for very deep networks.

    Other
    -----
    patience
        Early-stopping patience in epochs. 0 disables.
    """

    epochs: int
    learning_rate: float = DEFAULT_LEARNING_RATE
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_epochs: int = 5
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING

    stochastic_depth_p_max: float = DEFAULT_STOCHASTIC_DEPTH_P_MAX
    stochastic_depth_schedule: str = "linear"
    stochastic_depth_protect_entry: bool = True
    stochastic_depth_mode: str = "row"

    zero_init_residual_bn: bool = True
    use_amp: bool = True
    grad_clip_norm: float = 0.0
    patience: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError(
                f"warmup_epochs must be in [0, epochs); "
                f"got warmup={self.warmup_epochs}, epochs={self.epochs}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


# ─────────────────────────────────────────────────────────────────────────────
# Training result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpochRecord:
    """Per-epoch metrics; one is appended to `TrainingResult.history`."""

    epoch: int
    train_loss: float
    train_top1_acc: float
    val_top1_acc: float
    val_top5_acc: float
    learning_rate: float
    epoch_time_s: float
    skipped_batches: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one completed training run."""

    best_checkpoint_path: Path
    best_val_top1_acc: float
    best_epoch: int
    final_test_top1_acc: float
    final_test_top5_acc: float
    history: list[EpochRecord] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & scheduler builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> Optimizer:
    """SGD with momentum and weight decay."""
    return SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )


def _build_scheduler(
    optimizer: Optimizer,
    cfg: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Linear warmup followed by cosine annealing.

    During the first `warmup_epochs`, the LR grows linearly from
    learning_rate / 100 to learning_rate. After that, it follows a
    cosine curve from learning_rate down to 0 by the final epoch.

    If warmup_epochs == 0, only the cosine schedule is used.
    """
    if cfg.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    main_phase = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, main_phase],
        milestones=[cfg.warmup_epochs],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EpochStats:
    """Mutable accumulator used inside one epoch; collapsed to EpochRecord."""

    total_loss: float = 0.0
    total_samples: int = 0
    total_correct_top1: int = 0
    skipped_batches: int = 0


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: TrainingConfig,
    use_amp: bool,
) -> _EpochStats:
    """
    Run one training epoch. Returns aggregated stats for logging.

    The AMP code path follows the standard PyTorch recipe:
      1. autocast forward and loss
      2. scaler.scale(loss).backward()
      3. optional grad clipping (requires scaler.unscale_)
      4. scaler.step(optimizer)
      5. scaler.update()

    NaN/Inf losses (from fp16 overflow) skip the optimizer step rather
    than corrupting parameters. The skip count is logged.
    """
    model.train()
    stats = _EpochStats()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass under AMP autocast if enabled.
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        # NaN/Inf guard: skip this batch entirely if the loss is invalid.
        # This protects against fp16 overflow without forcing the user
        # to babysit the scaler manually.
        if not torch.isfinite(loss):
            stats.skipped_batches += 1
            continue

        # Backward with gradient scaling (no-op when use_amp=False).
        scaler.scale(loss).backward()

        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        # Bookkeeping for the epoch summary.
        bs = y.size(0)
        stats.total_loss += float(loss.item()) * bs
        stats.total_samples += bs
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            stats.total_correct_top1 += int((pred == y).sum().item())

    return stats


@torch.inference_mode()
def _validate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (top1, top5) accuracy on the loader."""
    model.eval()
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        bs = y.size(0)
        total += bs

        # Top-1
        pred = logits.argmax(dim=1)
        correct_top1 += int((pred == y).sum().item())

        # Top-5
        _, top5 = logits.topk(5, dim=1)
        correct_top5 += int(top5.eq(y.view(-1, 1)).any(dim=1).sum().item())

    return correct_top1 / max(total, 1), correct_top5 / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    *,
    model: nn.Module,
    data: DataBundle,
    cfg: TrainingConfig,
    output_dir: str | Path,
    device: torch.device,
    checkpoint_metadata: Optional[CheckpointMetadata] = None,
) -> TrainingResult:
    """
    Train a model end-to-end and save the best checkpoint by validation
    accuracy.

    Parameters
    ----------
    model
        The model to train. Should be on CPU initially — this function
        applies Stochastic Depth (if requested) and then moves to device.
    data
        Constructed by `build_loaders()`.
    cfg
        TrainingConfig with all hyperparameters.
    output_dir
        Directory where `checkpoint.pt` is saved.
    device
        Device to train on.
    checkpoint_metadata
        Optional metadata to embed in the saved checkpoint. The trainer
        fills in the fields it knows about (best_top1, epoch) on top of
        what the caller provides.

    Returns
    -------
    TrainingResult with the path to the best checkpoint and per-epoch
    history.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"

    # ── Apply Stochastic Depth ─────────────────────────────────────────────
    if cfg.stochastic_depth_p_max > 0.0:
        assigned = apply_stochastic_depth(
            model,
            p_max=cfg.stochastic_depth_p_max,
            schedule=cfg.stochastic_depth_schedule,
            protect_entry_blocks=cfg.stochastic_depth_protect_entry,
            mode=cfg.stochastic_depth_mode,
        )
        # apply_stochastic_depth replaces modules in-place; the new
        # StochasticDepthBlock wrappers are created on CPU and must be
        # moved to device along with the rest of the model.
        print(f"[trainer] Stochastic Depth applied "
              f"(p_max={cfg.stochastic_depth_p_max}, "
              f"schedule={cfg.stochastic_depth_schedule}); "
              f"max drop_prob = {max(assigned.values()):.3f}")

    # ── Zero-init residual BN ──────────────────────────────────────────────
    if cfg.zero_init_residual_bn:
        n_zeroed = zero_init_last_bn(model)
        print(f"[trainer] Zero-init residual BN: zeroed gamma in "
              f"{n_zeroed} BN layers.")

    model = model.to(device)

    # ── Build optimization machinery ───────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # ── Training state ─────────────────────────────────────────────────────
    best_val_top1 = -1.0
    best_epoch = 0
    history: list[EpochRecord] = []
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        train_stats = _train_one_epoch(
            model=model, loader=data.train_loader,
            optimizer=optimizer, criterion=criterion, scaler=scaler,
            device=device, cfg=cfg, use_amp=use_amp,
        )
        scheduler.step()

        val_top1, val_top5 = _validate(model, data.val_loader, device)

        train_loss = train_stats.total_loss / max(train_stats.total_samples, 1)
        train_top1 = train_stats.total_correct_top1 / max(train_stats.total_samples, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_start

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_top1_acc=train_top1,
            val_top1_acc=val_top1,
            val_top5_acc=val_top5,
            learning_rate=current_lr,
            epoch_time_s=epoch_time,
            skipped_batches=train_stats.skipped_batches,
        )
        history.append(record)

        skip_msg = ""
        if train_stats.skipped_batches > 0:
            skip_msg = f" | skipped={train_stats.skipped_batches}"
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"loss={train_loss:.4f} train_top1={train_top1:.4f} "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f} | "
            f"lr={current_lr:.5f} | {epoch_time:.1f}s{skip_msg}"
        )

        # ── Save best checkpoint ─────────────────────────────────────
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            epochs_without_improvement = 0

            meta = checkpoint_metadata or CheckpointMetadata()
            # Update mutable parts with current info.
            meta = CheckpointMetadata(
                model_name=meta.model_name,
                num_classes=meta.num_classes,
                stochastic_depth_p_max=cfg.stochastic_depth_p_max if cfg.stochastic_depth_p_max > 0 else None,
                epoch=epoch,
                best_top1=best_val_top1,
                extra=dict(meta.extra),
            )
            save_checkpoint(checkpoint_path, model, meta)
        else:
            epochs_without_improvement += 1
            if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
                print(f"[trainer] Early stopping: no improvement "
                      f"for {cfg.patience} epochs.")
                break

    # ── Final test evaluation on the best checkpoint ─────────────────────
    # Reload the best weights to evaluate on the held-out test set.
    from ..infrastructure.checkpoint import load_checkpoint
    best_state, _ = load_checkpoint(checkpoint_path, map_location=device)

    # The saved state_dict has block.* wrapper keys if SD was used;
    # since we still have the SD-wrapped model in memory, we can load
    # straight back.
    model.load_state_dict(best_state)
    test_top1, test_top5 = _validate(model, data.test_loader, device)

    print(f"[trainer] Final test: top1={test_top1:.4f} top5={test_top5:.4f} "
          f"(best epoch {best_epoch}, best val {best_val_top1:.4f})")

    return TrainingResult(
        best_checkpoint_path=checkpoint_path,
        best_val_top1_acc=best_val_top1,
        best_epoch=best_epoch,
        final_test_top1_acc=test_top1,
        final_test_top5_acc=test_top5,
        history=history,
    )
