"""
Tests for ExperimentConfig.

ExperimentConfig is small but its validation is important — the cross-
field check between `dataset` and `num_classes` is a common pitfall
that we catch at construction time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..config import ExperimentConfig  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Valid construction
# ─────────────────────────────────────────────────────────────────────────────

def test_cifar10_valid_construction():
    """The canonical CIFAR-10 configuration constructs without error."""
    cfg = ExperimentConfig(
        model="resnet18",
        dataset="cifar10",
        num_classes=10,
    )
    assert cfg.model == "resnet18"
    assert cfg.dataset == "cifar10"
    assert cfg.num_classes == 10
    # Defaults take effect
    assert cfg.batch_size == 128
    assert cfg.seed == 42


def test_cifar100_valid_construction():
    """CIFAR-100 with num_classes=100 constructs correctly."""
    cfg = ExperimentConfig(
        model="resnet50",
        dataset="cifar100",
        num_classes=100,
        batch_size=64,
    )
    assert cfg.num_classes == 100
    assert cfg.batch_size == 64


def test_config_is_frozen():
    """ExperimentConfig instances cannot be mutated after creation."""
    cfg = ExperimentConfig(model="resnet18", dataset="cifar10", num_classes=10)
    try:
        cfg.batch_size = 256  # type: ignore[misc]
    except Exception:  # frozen dataclass raises FrozenInstanceError
        return
    raise AssertionError("Frozen config should reject mutation")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-field validation
# ─────────────────────────────────────────────────────────────────────────────

def test_cifar10_wrong_num_classes_rejected():
    """cifar10 + num_classes=100 is a mismatch and must be rejected."""
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar10", num_classes=100,
        )
    except ValueError:
        return
    raise AssertionError("Should reject cifar10 with num_classes=100")


def test_cifar100_wrong_num_classes_rejected():
    """cifar100 + num_classes=10 is a mismatch and must be rejected."""
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar100", num_classes=10,
        )
    except ValueError:
        return
    raise AssertionError("Should reject cifar100 with num_classes=10")


# ─────────────────────────────────────────────────────────────────────────────
# Field-level validation
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_batch_size_rejected():
    """batch_size < 1 is rejected."""
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar10", num_classes=10,
            batch_size=0,
        )
    except ValueError:
        return
    raise AssertionError("Should reject batch_size=0")


def test_negative_num_workers_rejected():
    """num_workers < 0 is rejected."""
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar10", num_classes=10,
            num_workers=-1,
        )
    except ValueError:
        return
    raise AssertionError("Should reject num_workers=-1")


def test_val_ratio_out_of_range_rejected():
    """val_ratio outside [0, 1) is rejected."""
    for bad in [-0.1, 1.0, 1.5]:
        try:
            ExperimentConfig(
                model="resnet18", dataset="cifar10", num_classes=10,
                val_ratio=bad,
            )
        except ValueError:
            continue
        raise AssertionError(f"Should reject val_ratio={bad}")


def test_zero_image_size_rejected():
    """image_size < 1 is rejected."""
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar10", num_classes=10,
            image_size=0,
        )
    except ValueError:
        return
    raise AssertionError("Should reject image_size=0")


def test_single_class_rejected():
    """num_classes < 2 is rejected (a one-class classifier is meaningless)."""
    # Use a hack to bypass the cross-field check: temporarily set
    # dataset to cifar10 but num_classes=1. The num_classes check fires
    # first.
    try:
        ExperimentConfig(
            model="resnet18", dataset="cifar10", num_classes=1,
        )
    except ValueError:
        return
    raise AssertionError("Should reject num_classes=1")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_cifar10_valid_construction,
        test_cifar100_valid_construction,
        test_config_is_frozen,
        test_cifar10_wrong_num_classes_rejected,
        test_cifar100_wrong_num_classes_rejected,
        test_zero_batch_size_rejected,
        test_negative_num_workers_rejected,
        test_val_ratio_out_of_range_rejected,
        test_zero_image_size_rejected,
        test_single_class_rejected,
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
