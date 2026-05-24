"""
Command-line entry point: evaluate GDBA on a checkpoint at one ratio.

Usage
-----
    python -m graph_dynamic_block_activation.cli.run_gdba \
        --checkpoint ./checkpoints/r18_c100/checkpoint.pt \
        --top-k-ratio 0.5 \
        --min-keep-per-stage 0 \
        --output ./outputs/r18_c100/gdba_r05.json

Produces a JSON file with all accuracy, FLOPs, latency, and energy
metrics for that single (model, ratio, min_keep) configuration.

For Pareto-curve construction (multiple ratios), use
`measure_metrics.py` instead, which loops over a list of ratios in
one process and amortizes data loading.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from ..application.benchmark import benchmark_gdba
from ..application.controller import GDBAController
from ..domain.importance import ImportanceWeights
from ..infrastructure.model_factory import build_model_from_checkpoint
from ..training.data import build_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_dynamic_block_activation.cli.run_gdba",
        description="Evaluate GDBA on a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoint and data ─────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt file produced by `train.py`.",
    )
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-batches", type=int, default=None,
                   help="Optional cap for fast smoke tests.")

    # ── GDBA parameters ──────────────────────────────────────────────────
    gdba = p.add_argument_group("GDBA")
    gdba.add_argument("--top-k-ratio", type=float, required=True,
                     help="Fraction of non-entry blocks to keep active.")
    gdba.add_argument("--min-keep-per-stage", type=int, default=0,
                     help="Minimum active blocks per stage "
                          "(entry block counts toward the total).")
    gdba.add_argument("--refresh-interval", type=int, default=4)
    gdba.add_argument("--alpha", type=float, default=0.35)
    gdba.add_argument("--beta", type=float, default=0.30)
    gdba.add_argument("--gamma", type=float, default=0.15)
    gdba.add_argument("--delta", type=float, default=0.10)
    gdba.add_argument("--epsilon", type=float, default=0.10)

    # ── Output ──────────────────────────────────────────────────────────
    p.add_argument("--output", required=True,
                   help="Destination JSON file for the result.")
    p.add_argument("--warmup-iterations", type=int, default=10)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model from checkpoint ──────────────────────────────────────
    print(f"[run_gdba] Loading {args.checkpoint}")
    model, metadata = build_model_from_checkpoint(
        args.checkpoint,
        map_location=device,
        cifar_stem=(args.image_size <= 48),
    )
    model = model.to(device).eval()

    if metadata.model_name is None or metadata.num_classes is None:
        raise RuntimeError(
            "Checkpoint metadata is incomplete. The checkpoint must be "
            "produced by the new train.py to include model_name and "
            "num_classes."
        )

    # ── Build test loader ───────────────────────────────────────────────
    dataset = metadata.extra.get("dataset")
    if dataset is None:
        # Fallback heuristic: 10 classes => cifar10, 100 => cifar100.
        dataset = "cifar10" if metadata.num_classes == 10 else "cifar100"

    print(f"[run_gdba] Building test loader for {dataset}")
    data = build_loaders(
        dataset=dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=0.1,
        seed=42,
        augment=False,
    )

    # ── Build controller ────────────────────────────────────────────────
    weights = ImportanceWeights(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon,
    )
    controller = GDBAController.build(
        backbone=model,
        top_k_ratio=args.top_k_ratio,
        min_keep_per_stage=args.min_keep_per_stage,
        weights=weights,
        refresh_interval=args.refresh_interval,
    ).to(device)

    # ── Run benchmark ───────────────────────────────────────────────────
    print(f"[run_gdba] Running benchmark at r={args.top_k_ratio}, "
          f"m={args.min_keep_per_stage}, refresh={args.refresh_interval}")
    config_snapshot = {
        "checkpoint": args.checkpoint,
        "model": metadata.model_name,
        "dataset": dataset,
        "num_classes": metadata.num_classes,
        "top_k_ratio": args.top_k_ratio,
        "min_keep_per_stage": args.min_keep_per_stage,
        "refresh_interval": args.refresh_interval,
        "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
        "delta": args.delta, "epsilon": args.epsilon,
        "batch_size": args.batch_size,
    }
    result = benchmark_gdba(
        controller=controller,
        loader=data.test_loader,
        device=device,
        warmup_iterations=args.warmup_iterations,
        max_batches=args.max_batches,
        config_snapshot=config_snapshot,
    )

    # ── Write result ────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result.to_run_dict(), f, indent=2)
    print(f"[run_gdba] top1={result.accuracy.top1_acc:.4f} "
          f"top5={result.accuracy.top5_acc:.4f} "
          f"latency/sample={result.latency_s_per_sample*1000:.3f} ms")
    print(f"[run_gdba] Result written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
