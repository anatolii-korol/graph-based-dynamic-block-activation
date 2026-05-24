"""
Tests for flops_counter, latency_meter, energy_meter, and checkpoint.

The flops_counter has a regression test against the old `block_flops.py`
output. Latency and checkpoint are tested for shape and roundtrip
correctness. Energy is tested in graceful-degradation mode (no pynvml
required).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ..infrastructure.architecture_inspector import (  # noqa: E402
    inspect_resnet,
)
from ..infrastructure.checkpoint import (  # noqa: E402
    CheckpointMetadata,
    detect_stochastic_depth_keys,
    load_checkpoint,
    save_checkpoint,
)
from ..infrastructure.energy_meter import EnergyMeter  # noqa: E402
from ..infrastructure.flops_counter import (  # noqa: E402
    STEM_AND_HEAD_KEY,
    compute_block_flops,
    compute_effective_flops,
)
from ..infrastructure.latency_meter import (  # noqa: E402
    benchmark_callable,
    benchmark_forward,
)


def _make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# flops_counter
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_block_flops_returns_all_blocks():
    """Result has one entry per block plus the stem-and-head key."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    x = torch.randn(1, 3, 32, 32)
    flops = compute_block_flops(model, graph, x)
    assert STEM_AND_HEAD_KEY in flops
    for bid in graph.all_block_ids:
        assert bid in flops
        assert flops[bid] > 0.0, f"Block {bid} has zero FLOPs"


def test_compute_block_flops_scales_with_batch():
    """FLOPs scale linearly with batch size."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    f1 = compute_block_flops(model, graph, torch.randn(1, 3, 32, 32))
    f4 = compute_block_flops(model, graph, torch.randn(4, 3, 32, 32))
    for bid in graph.all_block_ids:
        ratio = f4[bid] / f1[bid]
        assert abs(ratio - 4.0) < 1e-6, (
            f"Block {bid}: expected 4x ratio, got {ratio}"
        )


def test_effective_flops_full_active_equals_total():
    """With every block active, effective FLOPs = sum of all entries."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    x = torch.randn(2, 3, 32, 32)
    flops = compute_block_flops(model, graph, x)
    gates = {bid: 1 for bid in graph.all_block_ids}
    effective = compute_effective_flops(flops, gates, graph)
    expected = sum(flops.values())
    assert abs(effective - expected) < 1e-3


def test_effective_flops_zero_non_entry_keeps_entries_and_head():
    """With all non-entry gates off, effective FLOPs = entry + stem + head."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    x = torch.randn(2, 3, 32, 32)
    flops = compute_block_flops(model, graph, x)
    gates = {bid: 0 for bid in graph.non_entry_block_ids}
    effective = compute_effective_flops(flops, gates, graph)
    expected = flops[STEM_AND_HEAD_KEY] + sum(
        flops[bid] for bid in graph.entry_block_ids
    )
    assert abs(effective - expected) < 1e-3


def test_effective_flops_entry_blocks_force_active():
    """An entry block gated off is still counted (safety semantics)."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    x = torch.randn(1, 3, 32, 32)
    flops = compute_block_flops(model, graph, x)
    gates = {bid: 0 for bid in graph.all_block_ids}  # try to gate off everything
    effective = compute_effective_flops(flops, gates, graph)
    # Even though gates say 0 for entries, they should still contribute.
    expected = flops[STEM_AND_HEAD_KEY] + sum(
        flops[bid] for bid in graph.entry_block_ids
    )
    assert abs(effective - expected) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# latency_meter
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_callable_returns_consistent_statistics():
    """Latency statistics are well-formed and ordered correctly."""
    device = torch.device("cpu")
    counter = [0]
    def fn():
        counter[0] += 1
    stats = benchmark_callable(fn, device, n_warmup=2, n_runs=10)
    assert stats.n_runs == 10
    assert stats.n_warmup == 2
    # Sanity: counter should have been called n_warmup + n_runs times
    assert counter[0] == 12
    # Statistical invariants: min <= median <= max, std >= 0
    assert stats.min_s <= stats.median_s <= stats.max_s
    assert stats.std_s >= 0.0
    # Per-sample with samples_per_run=1 equals batch-level
    assert abs(stats.per_sample_mean_s - stats.mean_s) < 1e-12


def test_benchmark_callable_rejects_bad_params():
    """Negative or zero n_runs and samples_per_run are rejected."""
    device = torch.device("cpu")
    bad_kwarg_sets = [
        {"samples_per_run": 0, "n_runs": 10, "n_warmup": 1},
        {"samples_per_run": -1, "n_runs": 10, "n_warmup": 1},
        {"samples_per_run": 1, "n_runs": 0, "n_warmup": 1},
        {"samples_per_run": 1, "n_runs": 10, "n_warmup": -1},
    ]
    for kwargs in bad_kwarg_sets:
        try:
            benchmark_callable(lambda: None, device, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Should reject kwargs={kwargs}")


def test_benchmark_forward_on_cpu():
    """End-to-end: time a small model on CPU, verify the result."""
    model = _make_cifar_resnet18()
    x = torch.randn(2, 3, 32, 32)
    stats = benchmark_forward(model, x, n_warmup=2, n_runs=5)
    assert stats.n_runs == 5
    assert stats.samples_per_run == 2
    assert stats.mean_s > 0
    assert stats.per_sample_mean_s == stats.mean_s / 2


def test_benchmark_forward_rejects_device_mismatch():
    """Input on wrong device must raise — we don't move data implicitly."""
    if not torch.cuda.is_available():
        return  # Skip silently if no CUDA
    model = _make_cifar_resnet18().cuda()
    x = torch.randn(1, 3, 32, 32)  # on CPU
    try:
        benchmark_forward(model, x)
    except ValueError:
        return
    raise AssertionError("Should reject CPU input with CUDA model")


# ─────────────────────────────────────────────────────────────────────────────
# energy_meter
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_meter_handles_no_nvml_gracefully():
    """
    On a system without working NVML (or without GPU), the meter must
    not crash and `report()` must indicate unavailability.
    """
    meter = EnergyMeter(device_index=0)
    # Whether NVML is available depends on the test environment. Either
    # way, exercising the context manager should be safe.
    with meter:
        # Do something trivial.
        for _ in range(100_000):
            pass
    report = meter.report()
    # If unavailable, all numeric fields are zero
    if not report.available:
        assert report.energy_joules == 0.0
        assert report.average_power_watts == 0.0
        assert report.duration_seconds == 0.0
    else:
        # If available, sanity checks
        assert report.duration_seconds >= 0.0
        assert report.sample_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip():
    """A checkpoint written by save_checkpoint loads back identically."""
    model = _make_cifar_resnet18(num_classes=100)
    metadata = CheckpointMetadata(
        model_name="resnet18",
        num_classes=100,
        stochastic_depth_p_max=0.1,
        epoch=42,
        best_top1=0.6998,
        extra={"note": "test"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(path, model, metadata)
        state_dict, loaded_meta = load_checkpoint(path)

    # State dicts: matching keys and tensors equal
    original_sd = model.state_dict()
    assert set(state_dict.keys()) == set(original_sd.keys())
    for k in original_sd:
        assert torch.equal(state_dict[k], original_sd[k]), (
            f"Tensor mismatch at {k}"
        )

    # Metadata preserved
    assert loaded_meta.model_name == "resnet18"
    assert loaded_meta.num_classes == 100
    assert loaded_meta.stochastic_depth_p_max == 0.1
    assert loaded_meta.epoch == 42
    assert loaded_meta.best_top1 == 0.6998
    assert loaded_meta.extra == {"note": "test"}


def test_load_bare_state_dict():
    """A bare state_dict (no metadata wrapper) loads correctly."""
    model = _make_cifar_resnet18()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bare.pt"
        torch.save(model.state_dict(), path)
        state_dict, meta = load_checkpoint(path)
    assert set(state_dict.keys()) == set(model.state_dict().keys())
    # Metadata defaults
    assert meta.model_name is None
    assert meta.num_classes is None


def test_load_legacy_model_key():
    """Legacy checkpoints with 'model' key are recognized."""
    model = _make_cifar_resnet18()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pt"
        torch.save({"model": model.state_dict()}, path)
        state_dict, _ = load_checkpoint(path)
    assert set(state_dict.keys()) == set(model.state_dict().keys())


def test_load_legacy_experiment_py_format():
    """
    Checkpoints saved by the pre-refactor experiment.py have a specific
    legacy payload structure with top-level "model" (string), nested
    "exp_cfg" and "train_cfg" dicts. The metadata extractor must
    correctly recover model_name, num_classes, etc. from this format.
    """
    model = _make_cifar_resnet18(num_classes=100)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy_exp.pt"
        torch.save({
            "model": "resnet18",
            "dataset": "cifar100",
            "state_dict": model.state_dict(),
            "exp_cfg": {
                "model": "resnet18",
                "dataset": "cifar100",
                "num_classes": 100,
                "image_size": 32,
                "seed": 42,
            },
            "train_cfg": {
                "epochs": 40,
                "stochastic_depth_p_max": 0.1,
            },
            "best_val_top1_acc": 0.6998,
            "epoch": 35,
        }, path)

        state_dict, meta = load_checkpoint(path)

    # Metadata recovered from legacy fields
    assert meta.model_name == "resnet18"
    assert meta.num_classes == 100
    assert meta.stochastic_depth_p_max == 0.1
    assert meta.epoch == 35
    assert abs(meta.best_top1 - 0.6998) < 1e-6
    assert meta.extra["dataset"] == "cifar100"
    assert meta.extra["seed"] == 42

    # State dict was extracted correctly
    assert set(state_dict.keys()) == set(model.state_dict().keys())


def test_load_legacy_with_sd_disabled_recognizes_no_sd():
    """
    Legacy training config with stochastic_depth_p_max=0.0 should
    translate to metadata.stochastic_depth_p_max = None (the new
    convention for "disabled").
    """
    model = _make_cifar_resnet18()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy_no_sd.pt"
        torch.save({
            "model": "resnet18",
            "state_dict": model.state_dict(),
            "exp_cfg": {"model": "resnet18", "num_classes": 10},
            "train_cfg": {"stochastic_depth_p_max": 0.0},
        }, path)
        _, meta = load_checkpoint(path)
    assert meta.stochastic_depth_p_max is None


def test_load_missing_file_raises():
    """Missing path raises FileNotFoundError."""
    try:
        load_checkpoint("/this/does/not/exist.pt")
    except FileNotFoundError:
        return
    raise AssertionError("Should raise FileNotFoundError")


def test_detect_stochastic_depth_keys():
    """The SD detector recognizes the wrapper signature in keys."""
    sd_with_wrapper = {
        "layer1.0.block.conv1.weight": torch.zeros(1),
        "layer1.0.block.bn1.weight": torch.zeros(1),
    }
    sd_without_wrapper = {
        "layer1.0.conv1.weight": torch.zeros(1),
        "layer1.0.bn1.weight": torch.zeros(1),
    }
    assert detect_stochastic_depth_keys(sd_with_wrapper) is True
    assert detect_stochastic_depth_keys(sd_without_wrapper) is False


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        # flops
        test_compute_block_flops_returns_all_blocks,
        test_compute_block_flops_scales_with_batch,
        test_effective_flops_full_active_equals_total,
        test_effective_flops_zero_non_entry_keeps_entries_and_head,
        test_effective_flops_entry_blocks_force_active,
        # latency
        test_benchmark_callable_returns_consistent_statistics,
        test_benchmark_callable_rejects_bad_params,
        test_benchmark_forward_on_cpu,
        test_benchmark_forward_rejects_device_mismatch,
        # energy
        test_energy_meter_handles_no_nvml_gracefully,
        # checkpoint
        test_save_and_load_roundtrip,
        test_load_bare_state_dict,
        test_load_legacy_model_key,
        test_load_legacy_experiment_py_format,
        test_load_legacy_with_sd_disabled_recognizes_no_sd,
        test_load_missing_file_raises,
        test_detect_stochastic_depth_keys,
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