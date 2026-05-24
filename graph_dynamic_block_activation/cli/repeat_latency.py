"""
repeat_latency.py — latency measurement.

Single latency measurements vary 0.5-1.0% between runs due to GPU
thermal state, cuDNN algorithm selection, and OS scheduler jitter.
Reporting a single number is misleading; reporting `mean ± std (n=K)`
across multiple independent runs is the standard.

This CLI runs the same latency measurement N times in the same process
and reports cross-run statistics:

  - mean_of_means    : average warm_latency across all runs
  - std_of_means     : std-dev between runs (the number to put in the paper)
  - cv_pct           : coefficient of variation = std/mean * 100%
  - 95% CI           : bootstrap confidence interval
  - per-run details  : individual measurements for reference

Usage (PowerShell):
    python -m graph_dynamic_block_activation.cli.repeat_latency `
        --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt `
        --top-k-ratio 0.9 --min-keep-per-stage 2 `
        --num-runs 5 --warm-iterations 100 `
        --output ./outputs/latency_r0.9.json

Report e.g.:
    latency = 0.917 ± 0.003 ms/sample  (mean ± 1 std, n=5)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch

from ..application.controller import GDBAController
from ..application.benchmark import _measure_latency_stats
from ..domain.importance import ImportanceWeights
from ..infrastructure.model_factory import build_model_from_checkpoint
from ..training.data import build_loaders


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_dynamic_block_activation.cli.repeat_latency",
        description="Run the latency measurement N times for paper-grade error bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # GDBA parameters
    p.add_argument("--top-k-ratio", type=float, default=0.9)
    p.add_argument("--min-keep-per-stage", type=int, default=2)
    p.add_argument("--refresh-interval", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--epsilon", type=float, default=0.10)

    # Measurement config
    p.add_argument("--num-runs", type=int, default=5,
                   help="Independent latency measurements to perform.")
    p.add_argument("--warm-iterations", type=int, default=100,
                   help="Warm iterations per run (same as benchmark_gdba).")
    p.add_argument("--inter-run-cooldown-s", type=float, default=2.0,
                   help="Idle seconds between runs to let GPU temperature stabilise.")
    p.add_argument("--baseline", action="store_true",
                   help="Measure the bare un-gated model instead of GDBA.")
    p.add_argument("--output", required=True)
    return p


def _bootstrap_ci_95(values: list[float], n_resamples: int = 1000,
                     seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    import random
    rng = random.Random(seed)
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return (lo, hi)


def _summary_stats(values: list[float]) -> dict:
    """Mean, std (sample), min, max, CV%."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv_pct": 0.0}
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    cv = (std / mean * 100.0) if mean > 0 else 0.0
    return {
        "mean": mean, "std": std,
        "min": min(values), "max": max(values),
        "cv_pct": cv,
        "n": n,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"Repeated latency measurement: n={args.num_runs} runs")
    print(f"  warm_iterations per run: {args.warm_iterations}")
    print(f"  inter-run cooldown:      {args.inter_run_cooldown_s}s")
    print("=" * 70)

    # ── Load model + data once ───────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, metadata = build_model_from_checkpoint(
        args.checkpoint, map_location=device,
    )
    model = model.to(device).eval()

    dataset = metadata.extra.get("dataset")
    if dataset is None:
        dataset = "cifar10" if metadata.num_classes == 10 else "cifar100"
    data = build_loaders(
        dataset=dataset, root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, val_ratio=0.1, seed=args.seed, augment=False,
    )

    # Build forward function — either bare model or gated wrapper.
    if args.baseline:
        forward_fn = model
        mode = "baseline (un-gated)"
    else:
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
        # One scoring + inference pass to fix the gates.
        sample_iter = iter(data.test_loader)
        x0, y0 = next(sample_iter)
        x0 = x0.to(device); y0 = y0.to(device)
        _ = controller.step(x0, batch_y=y0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_fn = controller.wrapper
        mode = f"GDBA r={args.top_k_ratio}, m={args.min_keep_per_stage}"
    print(f"Mode: {mode}\n")

    # Reusable sample batch (same across all runs for fair comparison).
    sample_x = next(iter(data.test_loader))[0].to(device)
    batch_size = int(sample_x.size(0))

    # ── Run N times ──────────────────────────────────────────────────────
    import time
    per_run = []
    for run_idx in range(args.num_runs):
        # Inter-run cooldown (skip before the first run).
        if run_idx > 0 and args.inter_run_cooldown_s > 0:
            time.sleep(args.inter_run_cooldown_s)

        stats = _measure_latency_stats(
            forward_fn=forward_fn,
            sample_batch=sample_x,
            device=device,
            n_warm_iterations=args.warm_iterations,
        )
        per_run.append({
            "run": run_idx + 1,
            "warm_mean_s": stats.warm_latency_mean_s,
            "warm_std_s": stats.warm_latency_std_s,
            "warm_p50_s": stats.warm_latency_p50_s,
            "warm_p99_s": stats.warm_latency_p99_s,
            "warm_per_sample_s": stats.warm_latency_per_sample_s,
            "throughput_sps": stats.throughput_samples_per_sec,
        })
        print(
            f"  Run {run_idx + 1}/{args.num_runs}:  "
            f"mean={stats.warm_latency_mean_s * 1000:.3f}ms  "
            f"per_sample={stats.warm_latency_per_sample_s * 1e6:.1f}us  "
            f"thr={stats.throughput_samples_per_sec:.1f}"
        )

    # ── Cross-run aggregates ─────────────────────────────────────────────
    means = [r["warm_mean_s"] for r in per_run]
    per_samples = [r["warm_per_sample_s"] for r in per_run]
    throughputs = [r["throughput_sps"] for r in per_run]

    agg = {
        "warm_mean_s": _summary_stats(means),
        "warm_per_sample_s": _summary_stats(per_samples),
        "throughput_samples_per_sec": _summary_stats(throughputs),
    }
    ci_mean = _bootstrap_ci_95(means, seed=args.seed)
    ci_per_sample = _bootstrap_ci_95(per_samples, seed=args.seed)
    agg["warm_mean_s"]["ci95_lo"] = ci_mean[0]
    agg["warm_mean_s"]["ci95_hi"] = ci_mean[1]
    agg["warm_per_sample_s"]["ci95_lo"] = ci_per_sample[0]
    agg["warm_per_sample_s"]["ci95_hi"] = ci_per_sample[1]

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"AGGREGATE (n={args.num_runs} runs, batch_size={batch_size})")
    print("=" * 70)
    m = agg["warm_per_sample_s"]
    print(f"  per-sample latency:")
    print(f"    mean ± std : {m['mean'] * 1e6:.2f} ± {m['std'] * 1e6:.2f} us")
    print(f"    CV         : {m['cv_pct']:.2f}%")
    print(f"    range      : [{m['min'] * 1e6:.2f}, {m['max'] * 1e6:.2f}] us")
    print(f"    95% CI     : [{m['ci95_lo'] * 1e6:.2f}, {m['ci95_hi'] * 1e6:.2f}] us")
    t = agg["throughput_samples_per_sec"]
    print(f"  throughput:")
    print(f"    mean ± std : {t['mean']:.1f} ± {t['std']:.1f} samples/sec")
    print(f"    CV         : {t['cv_pct']:.2f}%")
    print()
    paper_str = (
        f"latency = {m['mean'] * 1000:.4f} ± {m['std'] * 1000:.4f} ms/sample "
        f"(n={args.num_runs})"
    )
    print(f"  For the paper: {paper_str}")

    # ── Save ─────────────────────────────────────────────────────────────
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "model": metadata.model_name,
            "dataset": dataset,
            "num_classes": metadata.num_classes,
            "mode": mode,
            "batch_size": batch_size,
            "num_runs": args.num_runs,
            "warm_iterations_per_run": args.warm_iterations,
            "inter_run_cooldown_s": args.inter_run_cooldown_s,
            "device": str(device),
            "seed": args.seed,
        },
        "per_run": per_run,
        "aggregate": agg,
        "paper_string": paper_str,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
