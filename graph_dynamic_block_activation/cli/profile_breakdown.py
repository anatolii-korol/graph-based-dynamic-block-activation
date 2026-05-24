"""
profile_breakdown.py — operator-category profiler for GDBA inference.

Produces the per-category time breakdown used in the paper's Section 6
(Table 8.bis): how much of inference time is spent on convolution vs
BatchNorm vs activations vs other ops. Output is a single JSON with
the breakdown plus a config snapshot.

Usage (PowerShell):
    python -m graph_dynamic_block_activation.cli.profile_breakdown `
        --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt `
        --top-k-ratio 0.5 `
        --min-keep-per-stage 2 `
        --batch-size 128 `
        --output ./outputs/profile_breakdown_r50_c100_r05.json

The operator categorisation logic (CATEGORIES, categorize_op,
profile_breakdown) is preserved verbatim — only the model-loading and
controller-construction code changes to use the new infrastructure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

# API imports
from ..application.controller import GDBAController
from ..domain.importance import ImportanceWeights
from ..infrastructure.model_factory import build_model_from_checkpoint
from ..training.data import build_loaders


# ──────────────────────────────────────────────────────────────────────────
# Operation categorisation
# ──────────────────────────────────────────────────────────────────────────

CATEGORIES: List[Tuple[str, List[str]]] = [
    ("__wrapper__", [r"^model_inference$"]),
    ("convolution", [
        r"^aten::conv\d?d$",
        r"^aten::convolution",
        r"^aten::_convolution",
        r"^aten::cudnn_convolution",
        r"^aten::miopen_convolution",
        r"^aten::mkldnn_convolution",
        r"^aten::thnn_conv",
    ]),
    ("batch_norm", [
        r"^aten::batch_norm",
        r"^aten::native_batch_norm",
        r"^aten::cudnn_batch_norm",
        r"^aten::miopen_batch_norm",
        r"^aten::_batch_norm",
    ]),
    ("activation", [
        r"^aten::relu",
        r"^aten::silu",
        r"^aten::gelu",
        r"^aten::hardswish",
        r"^aten::hardsigmoid",
        r"^aten::leaky_relu",
        r"^aten::threshold",
        r"^aten::clamp",
    ]),
    ("elementwise_add", [
        r"^aten::add$",
        r"^aten::add_$",
        r"^aten::add\.Tensor",
        r"^aten::iadd",
    ]),
    ("pooling", [
        r"^aten::adaptive_avg_pool",
        r"^aten::avg_pool",
        r"^aten::max_pool",
        r"^aten::adaptive_max_pool",
    ]),
    ("linear", [
        r"^aten::linear$",
        r"^aten::matmul",
        r"^aten::mm$",
        r"^aten::bmm",
        r"^aten::addmm",
    ]),
    ("reshape_mem", [
        r"^aten::view",
        r"^aten::reshape",
        r"^aten::flatten",
        r"^aten::contiguous",
        r"^aten::permute",
        r"^aten::transpose",
        r"^aten::squeeze",
        r"^aten::unsqueeze",
        r"^aten::expand",
    ]),
    ("copy_cast", [
        r"^aten::to$",
        r"^aten::_to_copy",
        r"^aten::copy_",
        r"^aten::clone",
    ]),
    ("normalize", [
        r"^aten::div",
        r"^aten::mul",
        r"^aten::sub",
        r"^aten::neg",
    ]),
]


def categorize_op(op_name: str) -> str:
    """Return category name for given op name. Uses first matching pattern."""
    for cat_name, patterns in CATEGORIES:
        for pat in patterns:
            if re.match(pat, op_name, re.IGNORECASE):
                return cat_name
    return "other"


# ──────────────────────────────────────────────────────────────────────────
# Profiler with full breakdown (unchanged from the original)
# ──────────────────────────────────────────────────────────────────────────

def profile_breakdown(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_iters: int = 20,
) -> Dict:
    """
    Run PyTorch Profiler and produce a full category breakdown.

    The key insight: aten::conv2d, aten::convolution, aten::_convolution,
    aten::cudnn_convolution are 4 abstraction layers of the same op.
    To avoid double-counting we use SELF time which only counts time
    inside the op itself (not nested wrapper ops).
    """
    model.eval()
    activities = [ProfilerActivity.CPU]
    if sample_input.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_input)
    if sample_input.is_cuda:
        torch.cuda.synchronize()

    # Profile
    with profile(activities=activities, record_shapes=False) as prof:
        with torch.no_grad():
            for _ in range(num_iters):
                with record_function("model_inference"):
                    _ = model(sample_input)
                if sample_input.is_cuda:
                    torch.cuda.synchronize()

    def _get_total_time_us(evt) -> float:
        """Total time (includes nested ops) — used for the wrapper."""
        for attr in ("device_time_total", "cuda_time_total"):
            if hasattr(evt, attr):
                val = getattr(evt, attr)
                if val is not None and val > 0:
                    return float(val)
        return float(getattr(evt, "cpu_time_total", 0.0))

    def _get_self_time_us(evt) -> float:
        """Self time (excludes nested ops) — used for category breakdown."""
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(evt, attr):
                val = getattr(evt, attr)
                if val is not None:
                    return float(val)
        return float(getattr(evt, "self_cpu_time_total", 0.0))

    use_cuda = sample_input.is_cuda

    # Total time from the model_inference wrapper
    total_time_us = 0.0
    for evt in prof.key_averages():
        if evt.key == "model_inference":
            total_time_us = _get_total_time_us(evt) / num_iters
            break

    # Top-5 operators (compatible with original format)
    sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
    top_ops = []
    for evt in prof.key_averages():
        time_us = _get_total_time_us(evt) / num_iters
        top_ops.append({
            "name": evt.key,
            "time_us": time_us,
            "calls": int(evt.count) // max(1, num_iters),
        })
    top_ops.sort(key=lambda x: x["time_us"], reverse=True)

    # Category breakdown using SELF time
    category_data: Dict[str, Dict] = {}
    for evt in prof.key_averages():
        cat = categorize_op(evt.key)
        if cat == "__wrapper__":
            continue
        self_time_us = _get_self_time_us(evt) / num_iters
        if self_time_us <= 0:
            continue
        if cat not in category_data:
            category_data[cat] = {"time_us": 0.0, "n_ops": 0, "ops": []}
        category_data[cat]["time_us"] += self_time_us
        category_data[cat]["n_ops"] += 1
        category_data[cat]["ops"].append(evt.key)

    breakdown_total_us = sum(c["time_us"] for c in category_data.values())

    breakdown = []
    for cat, data in sorted(category_data.items(),
                            key=lambda x: -x[1]["time_us"]):
        share = (
            data["time_us"] / breakdown_total_us * 100
            if breakdown_total_us > 0 else 0.0
        )
        breakdown.append({
            "category": cat,
            "time_us": round(data["time_us"], 1),
            "share_pct": round(share, 2),
            "n_unique_ops": data["n_ops"],
            "example_ops": data["ops"][:3],
        })

    # Convolution share (clean number for the paper)
    conv_self_us = sum(
        _get_self_time_us(evt) / num_iters
        for evt in prof.key_averages()
        if categorize_op(evt.key) == "convolution"
    )
    conv_share_of_breakdown = (
        conv_self_us / breakdown_total_us * 100
        if breakdown_total_us > 0 else 0.0
    )
    conv_share_of_total = (
        conv_self_us / total_time_us * 100
        if total_time_us > 0 else 0.0
    )

    return {
        "profiler_available": True,
        "profiler_iters": num_iters,
        "sort_key": sort_key,
        "top_operators": top_ops[:5],
        "category_breakdown": breakdown,
        "total_time_us_per_iter": round(total_time_us, 1),
        "breakdown_sum_us_per_iter": round(breakdown_total_us, 1),
        "convolution_self_time_us": round(conv_self_us, 1),
        "convolution_share_of_breakdown_pct": round(conv_share_of_breakdown, 2),
        "convolution_share_of_total_pct": round(conv_share_of_total, 2),
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_dynamic_block_activation.cli.profile_breakdown",
        description="PyTorch-profiler operator-category breakdown for GDBA inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoint + data
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # GDBA parameters
    p.add_argument("--top-k-ratio", type=float, default=0.5)
    p.add_argument("--min-keep-per-stage", type=int, default=2)
    p.add_argument("--refresh-interval", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--epsilon", type=float, default=0.10)

    # Profiler config
    p.add_argument("--num-iters", type=int, default=20,
                   help="Number of profiled iterations.")
    p.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=== Profile Breakdown ===")
    print(f"r={args.top_k_ratio}, m={args.min_keep_per_stage}")
    print(f"Output: {args.output}")

    # ── [1/4] Load model ─────────────────────────────────────────────────
    print("[1/4] Loading checkpoint...")
    model, metadata = build_model_from_checkpoint(
        args.checkpoint, map_location=device,
    )
    model = model.to(device).eval()

    # ── [2/4] Build data loader ──────────────────────────────────────────
    print("[2/4] Building data loader...")
    dataset = metadata.extra.get("dataset")
    if dataset is None:
        dataset = "cifar10" if metadata.num_classes == 10 else "cifar100"
    data = build_loaders(
        dataset=dataset, root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, val_ratio=0.1, seed=args.seed, augment=False,
    )

    # ── [3/4] Build controller and fix gates ─────────────────────────────
    print("[3/4] Building GDBA controller and fixing gates...")
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

    sample_x = next(iter(data.test_loader))[0].to(device)
    sample_y = next(iter(data.test_loader))[1].to(device)

    # Run one scoring + inference pass to fix the gates. After this,
    # subsequent forwards measure steady-state inference cost (the
    # profiler should not see any scoring overhead).
    _ = controller.step(sample_x, batch_y=sample_y)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── [4/4] Profile ────────────────────────────────────────────────────
    print(f"[4/4] Profiling ({args.num_iters} iterations)...")
    breakdown = profile_breakdown(
        controller.wrapper, sample_x, num_iters=args.num_iters,
    )

    # ── Assemble result ──────────────────────────────────────────────────
    result = {
        "config": {
            "model": metadata.model_name,
            "dataset": dataset,
            "num_classes": metadata.num_classes,
            "checkpoint": args.checkpoint,
            "top_k_ratio": args.top_k_ratio,
            "min_keep_per_stage": args.min_keep_per_stage,
            "refresh_interval": args.refresh_interval,
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "delta": args.delta, "epsilon": args.epsilon,
            "batch_size": args.batch_size,
            "num_iters_profiled": args.num_iters,
            "device": str(device),
            "seed": args.seed,
        },
        "profiler": breakdown,
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(
        f"Total time (model_inference) per iteration: "
        f"{breakdown['total_time_us_per_iter']:.1f} us "
        f"({breakdown['total_time_us_per_iter'] / 1000:.3f} ms)"
    )
    print(
        f"Breakdown sum (sum of self-times): "
        f"{breakdown['breakdown_sum_us_per_iter']:.1f} us"
    )
    print()
    print(f"{'Category':<18} | {'Time (us)':<12} | {'Share':<8} | Ops")
    print("-" * 72)
    for entry in breakdown["category_breakdown"]:
        ops_preview = ", ".join(entry["example_ops"][:2])
        print(
            f"{entry['category']:<18} | {entry['time_us']:>9.1f}    | "
            f"{entry['share_pct']:>5.2f}%  | "
            f"{entry['n_unique_ops']} ops: {ops_preview}"
        )
    print()
    print(
        f"Convolution self time:        "
        f"{breakdown['convolution_self_time_us']:.1f} us"
    )
    print(
        f"Convolution share (of total): "
        f"{breakdown['convolution_share_of_total_pct']:.1f}%"
    )
    print(
        f"Convolution share (of breakdown): "
        f"{breakdown['convolution_share_of_breakdown_pct']:.1f}%"
    )
    print()
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
