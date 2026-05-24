"""
Tests for the benchmark module.

Covers:
  1. BenchmarkResult assembly and derived properties.
  2. benchmark_gdba() runs end-to-end on a small loader.
  3. benchmark_baseline() runs end-to-end.
  4. Separation of scoring vs inference budgets (scoring cost != 0 for GDBA).
  5. Determinism on identical inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ..application.benchmark import (  # noqa: E402
    AccuracyMetrics,
    BenchmarkResult,
    CostMetrics,
    benchmark_baseline,
    benchmark_gdba,
)
from ..application.controller import GDBAController  # noqa: E402


def _make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    torch.manual_seed(42)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def _make_synthetic_loader(
    n_batches: int = 5,
    batch_size: int = 4,
    num_classes: int = 10,
) -> DataLoader:
    """Random tensors with random labels — enough to exercise the code path."""
    torch.manual_seed(0)
    total = n_batches * batch_size
    xs = torch.randn(total, 3, 32, 32)
    ys = torch.randint(0, num_classes, (total,))
    dataset = TensorDataset(xs, ys)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_result_derived_properties():
    """total_*, per_sample_* are computed correctly from raw fields."""
    result = BenchmarkResult(
        accuracy=AccuracyMetrics(top1_acc=0.7, top5_acc=0.9, total_samples=100),
        inference=CostMetrics(
            total_time_s=2.0, total_energy_j=10.0,
            energy_available=True, total_flops=1e10,
        ),
        scoring=CostMetrics(
            total_time_s=0.5, total_energy_j=2.0,
            energy_available=True, total_flops=2e9,
        ),
        refresh_count=3,
    )
    assert result.total_time_s == 2.5
    assert result.total_energy_j == 12.0
    assert result.total_flops == 1.2e10
    assert result.latency_s_per_sample == 0.02
    assert result.energy_j_per_sample == 0.1
    assert result.flops_per_sample == 1e8


def test_benchmark_result_serialization():
    """to_dict() produces a JSON-serializable structure."""
    import json
    result = BenchmarkResult(
        accuracy=AccuracyMetrics(top1_acc=0.5, top5_acc=0.8, total_samples=10),
        inference=CostMetrics(1.0, 5.0, True, 1e9),
        scoring=CostMetrics(0.2, 1.0, True, 5e8),
        refresh_count=2,
        config_snapshot={"ratio": 0.5},
    )
    d = result.to_dict()
    # Must be JSON-serializable
    s = json.dumps(d)
    assert isinstance(s, str)
    assert d["top1_acc"] == 0.5
    assert d["refresh_count"] == 2
    assert d["config_snapshot"] == {"ratio": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_gdba
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_gdba_runs_end_to_end():
    """End-to-end smoke test: benchmark_gdba returns sensible metrics."""
    model = _make_cifar_resnet18(num_classes=10)
    device = torch.device("cpu")
    controller = GDBAController.build(
        model, top_k_ratio=0.5, refresh_interval=2,
    )
    loader = _make_synthetic_loader(n_batches=6, batch_size=4)

    result = benchmark_gdba(
        controller=controller,
        loader=loader,
        device=device,
        warmup_iterations=2,
    )

    # Sanity on accuracy
    assert 0.0 <= result.accuracy.top1_acc <= 1.0
    assert 0.0 <= result.accuracy.top5_acc <= 1.0
    assert result.accuracy.total_samples == 6 * 4

    # Inference metrics must be positive
    assert result.inference.total_time_s > 0.0
    assert result.inference.total_flops > 0.0

    # Scoring must have happened (refresh_interval=2, 6 batches -> 3 refreshes)
    assert result.refresh_count == 3
    assert result.scoring.total_time_s > 0.0
    assert result.scoring.total_flops > 0.0


def test_benchmark_gdba_no_refresh_with_huge_interval():
    """If refresh_interval > total batches, only the first batch refreshes."""
    model = _make_cifar_resnet18(num_classes=10)
    device = torch.device("cpu")
    controller = GDBAController.build(
        model, top_k_ratio=0.5, refresh_interval=100,
    )
    loader = _make_synthetic_loader(n_batches=5, batch_size=4)

    result = benchmark_gdba(
        controller=controller,
        loader=loader,
        device=device,
        warmup_iterations=1,
    )
    assert result.refresh_count == 1  # only step 0 refreshes


def test_benchmark_gdba_max_batches_cap():
    """`max_batches` halts the loop early."""
    model = _make_cifar_resnet18(num_classes=10)
    device = torch.device("cpu")
    controller = GDBAController.build(model, top_k_ratio=0.5)
    loader = _make_synthetic_loader(n_batches=10, batch_size=4)

    result = benchmark_gdba(
        controller=controller, loader=loader, device=device,
        warmup_iterations=1, max_batches=3,
    )
    assert result.accuracy.total_samples == 3 * 4


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_baseline_runs_end_to_end():
    """Baseline has zero scoring cost and a positive inference time."""
    model = _make_cifar_resnet18(num_classes=10)
    device = torch.device("cpu")
    loader = _make_synthetic_loader(n_batches=4, batch_size=4)

    result = benchmark_baseline(
        model=model, loader=loader, device=device, warmup_iterations=2,
    )
    assert result.refresh_count == 0
    assert result.scoring.total_time_s == 0.0
    assert result.scoring.total_flops == 0.0
    assert result.inference.total_time_s > 0.0
    assert 0.0 <= result.accuracy.top1_acc <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_gdba_accuracy_is_deterministic():
    """
    Same model, same data, same config -> identical accuracy on repeated
    runs. Timing/energy will vary, but accuracy must not.
    """
    model_a = _make_cifar_resnet18(num_classes=10)
    model_b = _make_cifar_resnet18(num_classes=10)  # same seed -> same weights

    device = torch.device("cpu")
    loader = _make_synthetic_loader(n_batches=5, batch_size=4)

    ctrl_a = GDBAController.build(model_a, top_k_ratio=0.5, refresh_interval=2)
    ctrl_b = GDBAController.build(model_b, top_k_ratio=0.5, refresh_interval=2)

    r_a = benchmark_gdba(controller=ctrl_a, loader=loader, device=device, warmup_iterations=1)
    r_b = benchmark_gdba(controller=ctrl_b, loader=loader, device=device, warmup_iterations=1)

    assert abs(r_a.accuracy.top1_acc - r_b.accuracy.top1_acc) < 1e-9
    assert abs(r_a.accuracy.top5_acc - r_b.accuracy.top5_acc) < 1e-9
    assert r_a.refresh_count == r_b.refresh_count
    # FLOPs should be deterministic (analytical)
    assert abs(r_a.inference.total_flops - r_b.inference.total_flops) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_benchmark_result_derived_properties,
        test_benchmark_result_serialization,
        test_benchmark_gdba_runs_end_to_end,
        test_benchmark_gdba_no_refresh_with_huge_interval,
        test_benchmark_gdba_max_batches_cap,
        test_benchmark_baseline_runs_end_to_end,
        test_benchmark_gdba_accuracy_is_deterministic,
    ]
    passed = failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
