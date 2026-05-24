"""
Command-line entry point: train a ResNet classifier on CIFAR.

Usage
-----
    python -m graph_dynamic_block_activation.cli.train \
        --model resnet18 \
        --dataset cifar100 \
        --epochs 50 \
        --stochastic-depth-p 0.1 \
        --output-dir ./checkpoints/r18_c100

The script writes the best-validation checkpoint to
`<checkpoints-dir>/checkpoint.pt` and a JSON training log to
`<checkpoints-dir>/training_result.json`.

This file is intentionally a thin wrapper. All real logic lives in
`training.trainer.train_model()` and `training.data.build_loaders()`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..infrastructure.checkpoint import CheckpointMetadata
from ..infrastructure.model_factory import build_model
from ..training.data import build_loaders
from ..training.trainer import TrainingConfig, train_model


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Arguments are grouped into three categories matching our three
    configuration dataclasses: experiment-level (model, dataset),
    training-level (epochs, learning rate), and execution-level
    (output path, seed).
    """
    p = argparse.ArgumentParser(
        prog="graph_dynamic_block_activation.cli.train",
        description="Train a CIFAR classifier with optional Stochastic Depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment configuration ────────────────────────────────────────
    exp = p.add_argument_group("experiment")
    exp.add_argument(
        "--model", required=True,
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    exp.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100"],
    )
    exp.add_argument("--data-root", default="./data")
    exp.add_argument("--image-size", type=int, default=32)
    exp.add_argument("--batch-size", type=int, default=128)
    exp.add_argument("--num-workers", type=int, default=4)
    exp.add_argument("--val-ratio", type=float, default=0.1)
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ── Training hyperparameters ────────────────────────────────────────
    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--warmup-epochs", type=int, default=5)
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--grad-clip-norm", type=float, default=0.0)
    train.add_argument("--patience", type=int, default=0)
    train.add_argument(
        "--no-amp", dest="use_amp", action="store_false",
        help="Disable automatic mixed precision (fp16) training.",
    )
    train.set_defaults(use_amp=True)
    train.add_argument(
        "--no-zero-init-bn", dest="zero_init_residual_bn",
        action="store_false",
        help="Disable zero-init of last residual BN (Goyal et al. 2017).",
    )
    train.set_defaults(zero_init_residual_bn=True)

    # ── Stochastic Depth ────────────────────────────────────────────────
    sd = p.add_argument_group("stochastic depth")
    sd.add_argument(
        "--stochastic-depth-p", type=float, default=0.1,
        help="Maximum drop probability for the deepest block. "
             "Set to 0 to disable Stochastic Depth entirely.",
    )
    sd.add_argument(
        "--stochastic-depth-schedule",
        choices=["linear", "uniform"], default="linear",
    )
    sd.add_argument(
        "--no-protect-entry", dest="stochastic_depth_protect_entry",
        action="store_false",
        help="Allow entry blocks (layerN.0) to be dropped by SD.",
    )
    sd.set_defaults(stochastic_depth_protect_entry=True)

    # ── Output ──────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir", required=True,
        help="Directory for checkpoint.pt and training_result.json.",
    )
    out.add_argument("--run-name", default="default_run")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Build experiment config (validates dataset/num_classes consistency) ─
    num_classes = 10 if args.dataset == "cifar10" else 100
    exp_cfg = ExperimentConfig(
        model=args.model,
        dataset=args.dataset,
        num_classes=num_classes,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    # ── Build training config ──────────────────────────────────────────
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        stochastic_depth_p_max=args.stochastic_depth_p,
        stochastic_depth_schedule=args.stochastic_depth_schedule,
        stochastic_depth_protect_entry=args.stochastic_depth_protect_entry,
        zero_init_residual_bn=args.zero_init_residual_bn,
        use_amp=args.use_amp,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
    )

    # ── Build data, model, run training ─────────────────────────────────
    print(f"[train] Building loaders for {exp_cfg.dataset} ({exp_cfg.num_classes} classes)")
    data = build_loaders(
        dataset=exp_cfg.dataset,
        root=exp_cfg.data_root,
        batch_size=exp_cfg.batch_size,
        num_workers=exp_cfg.num_workers,
        image_size=exp_cfg.image_size,
        val_ratio=exp_cfg.val_ratio,
        seed=exp_cfg.seed,
        augment=True,
    )

    print(f"[train] Building {exp_cfg.model}")
    model = build_model(
        model_name=exp_cfg.model,
        num_classes=exp_cfg.num_classes,
        cifar_stem=(exp_cfg.image_size <= 48),
    )

    device = torch.device(exp_cfg.device if torch.cuda.is_available() else "cpu")
    output_path = Path(exp_cfg.output_dir) / exp_cfg.run_name

    checkpoint_metadata = CheckpointMetadata(
        model_name=exp_cfg.model,
        num_classes=exp_cfg.num_classes,
        stochastic_depth_p_max=(
            train_cfg.stochastic_depth_p_max
            if train_cfg.stochastic_depth_p_max > 0 else None
        ),
        extra={
            "dataset": exp_cfg.dataset,
            "image_size": exp_cfg.image_size,
            "seed": exp_cfg.seed,
            "run_name": exp_cfg.run_name,
        },
    )

    result = train_model(
        model=model,
        data=data,
        cfg=train_cfg,
        output_dir=output_path,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
    )

    # ── Persist a JSON summary alongside the checkpoint ─────────────────
    summary_path = output_path / "training_result.json"
    summary = {
        "experiment": asdict(exp_cfg),
        "training": asdict(train_cfg),
        "best_val_top1": result.best_val_top1_acc,
        "best_epoch": result.best_epoch,
        "final_test_top1": result.final_test_top1_acc,
        "final_test_top5": result.final_test_top5_acc,
        "checkpoint_path": str(result.best_checkpoint_path),
        "history": [asdict(r) for r in result.history],
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
