"""
Command-line entry point: sweep top-k ratios for Pareto-curve generation.

Usage
-----
    python -m graph_dynamic_block_activation.cli.measure_metrics \
        --checkpoint ./checkpoints/r18_c100/checkpoint.pt \
        --top-k-ratios 0.1 0.3 0.5 0.7 0.9 1.0 \
        --output-dir ./outputs/r18_c100/sweep

Produces one JSON file per ratio plus a `summary_all.json` aggregating
the headline metrics across all ratios. This is the data source for
all Pareto-style tables and figures in the paper.

The sweep loads the model and data exactly once and reuses them across
all ratios, which is roughly N times faster than running run_gdba.py
N times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ..application.benchmark import (
    BenchmarkResult,
    benchmark_baseline,
    benchmark_gdba,
)
from ..application.controller import GDBAController
from ..domain.importance import ImportanceWeights
from ..infrastructure.model_factory import build_model_from_checkpoint
from ..training.data import build_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_dynamic_block_activation.cli.measure_metrics",
        description="Sweep GDBA across multiple top-k ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-batches", type=int, default=None)

    # Sweep parameters
    p.add_argument(
        "--top-k-ratios", type=float, nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        help="Ratios to sweep, space-separated.",
    )
    p.add_argument("--min-keep-per-stage", type=int, default=0)
    p.add_argument("--refresh-interval", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--epsilon", type=float, default=0.10)

    p.add_argument(
        "--no-baseline", dest="run_baseline", action="store_false",
        help="Skip baseline (no-gating) measurement.",
    )
    p.set_defaults(run_baseline=True)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--warmup-iterations", type=int, default=10)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _summarize(result: BenchmarkResult) -> dict:
    """
    Reduce a BenchmarkResult to a flat dict suitable for a multi-row
    summary table. Keeps only the fields used by the paper figures.
    """
    return {
        "top1_acc": result.accuracy.top1_acc,
        "top5_acc": result.accuracy.top5_acc,
        "total_samples": result.accuracy.total_samples,
        "latency_s_per_sample": result.latency_s_per_sample,
        "energy_j_per_sample": result.energy_j_per_sample,
        "flops_per_sample": result.flops_per_sample,
        "refresh_count": result.refresh_count,
        "inference_time_s": result.inference.total_time_s,
        "scoring_time_s": result.scoring.total_time_s,
        "inference_energy_j": result.inference.total_energy_j,
        "scoring_energy_j": result.scoring.total_energy_j,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + data once ──────────────────────────────────────────
    print(f"[measure_metrics] Loading {args.checkpoint}")
    model, metadata = build_model_from_checkpoint(
        args.checkpoint, map_location=device,
        cifar_stem=(args.image_size <= 48),
    )
    model = model.to(device).eval()

    dataset = metadata.extra.get("dataset")
    if dataset is None:
        dataset = "cifar10" if metadata.num_classes == 10 else "cifar100"

    print(f"[measure_metrics] Building test loader for {dataset}")
    data = build_loaders(
        dataset=dataset, root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, val_ratio=0.1, seed=42, augment=False,
    )

    summary: dict[str, dict] = {}

    # ── Baseline first (optional) ───────────────────────────────────────
    if args.run_baseline:
        print("[measure_metrics] Measuring baseline (no gating)")
        baseline_result = benchmark_baseline(
            model=model, loader=data.test_loader, device=device,
            warmup_iterations=args.warmup_iterations,
            max_batches=args.max_batches,
            config_snapshot={"phase": "baseline"},
        )
        summary["baseline"] = _summarize(baseline_result)
        with (output_dir / "baseline.json").open("w") as f:
            json.dump(baseline_result.to_measurement_dict(), f, indent=2)
        print(f"  baseline top1={baseline_result.accuracy.top1_acc:.4f}")

    # ── GDBA sweep ───────────────────────────────────────────────────────
    weights = ImportanceWeights(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon,
    )

    for ratio in args.top_k_ratios:
        key = f"r{ratio:g}"  # e.g. "r0.5", "r1"
        print(f"[measure_metrics] Sweep point: {key}")

        controller = GDBAController.build(
            backbone=model,
            top_k_ratio=ratio,
            min_keep_per_stage=args.min_keep_per_stage,
            weights=weights,
            refresh_interval=args.refresh_interval,
        ).to(device)

        config_snapshot = {
            "checkpoint": args.checkpoint,
            "model": metadata.model_name,
            "dataset": dataset,
            "top_k_ratio": ratio,
            "min_keep_per_stage": args.min_keep_per_stage,
            "refresh_interval": args.refresh_interval,
            "batch_size": args.batch_size,
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "delta": args.delta, "epsilon": args.epsilon,
        }
        result = benchmark_gdba(
            controller=controller, loader=data.test_loader, device=device,
            warmup_iterations=args.warmup_iterations,
            max_batches=args.max_batches,
            config_snapshot=config_snapshot,
        )

        summary[key] = _summarize(result)
        with (output_dir / f"{key}.json").open("w") as f:
            json.dump(result.to_measurement_dict(), f, indent=2)
        print(f"  {key} top1={result.accuracy.top1_acc:.4f} "
              f"latency/sample={result.latency_s_per_sample*1000:.3f} ms")

    # ── Aggregated summary ──────────────────────────────────────────────
    with (output_dir / "summary_all.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[measure_metrics] Summary written to {output_dir / 'summary_all.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
