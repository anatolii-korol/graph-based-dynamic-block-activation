"""
Tests for the training subsystem (stochastic_depth, data, trainer).

These tests deliberately use:
  - Small models (ResNet-18 only),
  - Synthetic data loaders (TensorDataset, not real CIFAR),
  - Very short runs (2-3 epochs, ~10 batches).

The goal is to verify behavioral correctness — Stochastic Depth wires
itself in, the trainer's loop runs end-to-end, checkpoints are written
— not to measure final accuracy. Accuracy benchmarks happen on full
runs outside the test suite.

Real CIFAR download is avoided to keep the test suite hermetic and fast
(< 30 s on CPU). The data.py module is exercised by a separate test that
mocks the dataset directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ..training.data import (  # noqa: E402
    DataBundle,
    _normalization_params,
    build_transforms,
    stratified_split_indices,
)
from ..training.stochastic_depth import (  # noqa: E402
    StochasticDepthBlock,
    apply_stochastic_depth,
    get_stochastic_depth_state,
    zero_init_last_bn,
)
from ..training.trainer import (  # noqa: E402
    EpochRecord,
    TrainingConfig,
    TrainingResult,
    train_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with the CIFAR stem (3x3 conv, no maxpool)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _make_synthetic_bundle(
    num_classes: int = 10,
    n_train: int = 40,
    n_val: int = 16,
    n_test: int = 16,
    batch_size: int = 8,
) -> DataBundle:
    """
    Build a minimal DataBundle around random tensors.

    Used in lieu of a real CIFAR download for unit tests. The label
    distribution is uniformly random; the trainer cannot learn anything
    meaningful, but it WILL run all code paths.
    """
    torch.manual_seed(0)

    def _ds(n: int) -> TensorDataset:
        x = torch.randn(n, 3, 32, 32)
        y = torch.randint(0, num_classes, (n,))
        return TensorDataset(x, y)

    train_ds = _ds(n_train)
    val_ds = _ds(n_val)
    test_ds = _ds(n_test)

    return DataBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        num_classes=num_classes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# stochastic_depth
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_stochastic_depth_wraps_all_blocks():
    """Every residual block becomes a StochasticDepthBlock."""
    model = _make_cifar_resnet18()
    assigned = apply_stochastic_depth(model, p_max=0.1)

    # ResNet-18 has 4 stages of 2 blocks each = 8 blocks
    assert len(assigned) == 8

    # All blocks are now wrapped
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(model, stage_name)
        for block in stage:
            assert isinstance(block, StochasticDepthBlock), (
                f"Block in {stage_name} not wrapped"
            )


def test_linear_schedule_assigns_correct_probabilities():
    """Drop probabilities follow p_l = (l / (L-1)) * p_max."""
    model = _make_cifar_resnet18()
    assigned = apply_stochastic_depth(
        model, p_max=0.1, protect_entry_blocks=False,
    )

    # 8 blocks: l = 0, 1, ..., 7. p_l should range from 0 to 0.1.
    # Order is layer1.0, layer1.1, layer2.0, ..., layer4.1
    expected_keys = [
        f"layer{s}.{i}" for s in (1, 2, 3, 4) for i in (0, 1)
    ]
    probs_in_order = [assigned[k] for k in expected_keys]

    assert probs_in_order[0] == 0.0  # shallowest
    assert abs(probs_in_order[-1] - 0.1) < 1e-9  # deepest
    # Monotonically non-decreasing
    for a, b in zip(probs_in_order[:-1], probs_in_order[1:]):
        assert b >= a, f"Schedule not monotonic: {probs_in_order}"


def test_protect_entry_blocks_keeps_layer_zero_at_zero():
    """With protect_entry_blocks=True, every layerN.0 has drop_prob=0."""
    model = _make_cifar_resnet18()
    assigned = apply_stochastic_depth(
        model, p_max=0.5, protect_entry_blocks=True,
    )
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        assert assigned[f"{stage_name}.0"] == 0.0, (
            f"{stage_name}.0 should be protected but got "
            f"drop_prob={assigned[f'{stage_name}.0']}"
        )


def test_unknown_schedule_rejected():
    """An unrecognized schedule string raises ValueError."""
    model = _make_cifar_resnet18()
    try:
        apply_stochastic_depth(model, schedule="cosine")
    except ValueError:
        return
    raise AssertionError("Should have rejected unknown schedule")


def test_wrapped_block_eval_matches_unwrapped():
    """
    At eval time, StochasticDepthBlock should produce IDENTICAL output
    to the unwrapped block (because stochastic_depth is a no-op outside
    training mode).
    """
    torch.manual_seed(0)
    model_a = _make_cifar_resnet18()
    torch.manual_seed(0)
    model_b = _make_cifar_resnet18()
    apply_stochastic_depth(model_b, p_max=0.5)

    model_a.eval()
    model_b.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-6), (
        f"Wrapped model differs at eval; max diff "
        f"{(out_a - out_b).abs().max().item()}"
    )


def test_wrapped_block_supports_resnet_attributes():
    """
    The wrapper must delegate standard BasicBlock attributes (conv1,
    bn1, etc.) via __getattr__, so FLOPs counters and tracers continue
    to work.
    """
    model = _make_cifar_resnet18()
    apply_stochastic_depth(model, p_max=0.1)

    wrapped_block = model.layer1[0]
    # These are the attributes a FLOPs counter would look for.
    for attr in ("conv1", "bn1", "conv2", "bn2", "downsample"):
        assert hasattr(wrapped_block, attr), (
            f"Wrapper missing attribute {attr!r}"
        )


def test_zero_init_last_bn_zeros_correct_layers():
    """
    Zero-init should zero gamma in bn2 (BasicBlock) or bn3 (Bottleneck).
    For ResNet-18, only bn2 should be zeroed.
    """
    model = _make_cifar_resnet18()
    # Set all BN gammas to 1 for an obvious starting point.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)

    n_zeroed = zero_init_last_bn(model)
    assert n_zeroed == 8  # 4 stages * 2 BasicBlocks each

    # Check that exactly bn2 of each block is zeroed.
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(model, stage_name):
            assert float(block.bn2.weight.abs().sum()) == 0.0, (
                f"{stage_name}.bn2 should be zeroed"
            )
            # bn1 should remain ones.
            assert float(block.bn1.weight.abs().sum()) > 0.0


def test_zero_init_works_through_stochastic_depth_wrapper():
    """zero_init_last_bn must also reach through SD wrappers."""
    model = _make_cifar_resnet18()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)

    apply_stochastic_depth(model, p_max=0.1)
    n_zeroed = zero_init_last_bn(model)
    assert n_zeroed == 8


def test_get_stochastic_depth_state_returns_assigned_probs():
    """The introspection function returns the same probs we assigned."""
    model = _make_cifar_resnet18()
    assigned = apply_stochastic_depth(model, p_max=0.1)
    introspected = get_stochastic_depth_state(model)
    assert assigned == introspected


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────

def test_normalization_params_for_cifar10():
    """CIFAR-10 mean/std have expected channel counts and values."""
    mean, std = _normalization_params("cifar10")
    assert len(mean) == 3
    assert len(std) == 3
    # CIFAR-10 means are around 0.45-0.50
    assert all(0.4 < m < 0.55 for m in mean)


def test_normalization_params_for_cifar100():
    """CIFAR-100 mean/std have expected channel counts."""
    mean, std = _normalization_params("cifar100")
    assert len(mean) == 3
    assert len(std) == 3


def test_normalization_params_rejects_unknown_dataset():
    """An unknown dataset name raises ValueError."""
    try:
        _normalization_params("imagenet")
    except ValueError:
        return
    raise AssertionError("Should have rejected unknown dataset")


def test_stratified_split_preserves_class_proportions():
    """
    Stratified split should put approximately val_ratio fraction of
    each class into the val set.
    """
    import numpy as np
    # 100 samples, 10 classes, 10 samples per class.
    targets = np.array([c for c in range(10) for _ in range(10)])
    train_idx, val_idx = stratified_split_indices(
        targets, val_ratio=0.2, seed=42,
    )

    # 20 samples to val total (2 per class).
    assert len(val_idx) == 20
    assert len(train_idx) == 80

    # Per-class breakdown
    val_targets = targets[val_idx]
    for c in range(10):
        n_c = int((val_targets == c).sum())
        assert n_c == 2, f"Class {c} has {n_c} val samples, expected 2"


def test_stratified_split_rejects_bad_ratio():
    """val_ratio outside [0, 1) is rejected."""
    import numpy as np
    targets = np.array([0, 1, 2, 0, 1, 2])
    for bad in [-0.1, 1.0, 1.5]:
        try:
            stratified_split_indices(targets, val_ratio=bad, seed=0)
        except ValueError:
            continue
        raise AssertionError(f"Should reject val_ratio={bad}")


def test_build_transforms_train_has_augmentation():
    """With augment=True, the train pipeline contains RandomCrop/Flip."""
    from torchvision import transforms

    train_tf, test_tf = build_transforms("cifar10", augment=True)

    # Inspect the composition for the augmentation ops.
    train_op_types = [type(op).__name__ for op in train_tf.transforms]
    assert "RandomCrop" in train_op_types
    assert "RandomHorizontalFlip" in train_op_types

    # Test pipeline has no augmentation.
    test_op_types = [type(op).__name__ for op in test_tf.transforms]
    assert "RandomCrop" not in test_op_types
    assert "RandomHorizontalFlip" not in test_op_types


def test_build_transforms_no_augmentation():
    """With augment=False, train pipeline is identical to test pipeline."""
    train_tf, test_tf = build_transforms("cifar10", augment=False)
    train_op_types = [type(op).__name__ for op in train_tf.transforms]
    test_op_types = [type(op).__name__ for op in test_tf.transforms]
    assert train_op_types == test_op_types


# ─────────────────────────────────────────────────────────────────────────────
# TrainingConfig validation
# ─────────────────────────────────────────────────────────────────────────────

def test_training_config_rejects_zero_epochs():
    """epochs < 1 is rejected at construction."""
    try:
        TrainingConfig(epochs=0)
    except ValueError:
        return
    raise AssertionError("Should reject epochs=0")


def test_training_config_rejects_warmup_too_long():
    """warmup_epochs >= epochs is rejected."""
    try:
        TrainingConfig(epochs=5, warmup_epochs=5)
    except ValueError:
        return
    raise AssertionError("Should reject warmup_epochs == epochs")


def test_training_config_rejects_negative_warmup():
    """warmup_epochs < 0 is rejected."""
    try:
        TrainingConfig(epochs=5, warmup_epochs=-1)
    except ValueError:
        return
    raise AssertionError("Should reject negative warmup_epochs")


def test_training_config_rejects_zero_lr():
    """Non-positive learning rate is rejected."""
    try:
        TrainingConfig(epochs=5, learning_rate=0.0)
    except ValueError:
        return
    raise AssertionError("Should reject zero learning_rate")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer end-to-end (synthetic data, 2 epochs)
# ─────────────────────────────────────────────────────────────────────────────

def test_trainer_runs_end_to_end():
    """
    Smoke test: train for 2 epochs on synthetic data, verify a
    checkpoint is written and the TrainingResult is well-formed.
    """
    model = _make_cifar_resnet18(num_classes=10)
    bundle = _make_synthetic_bundle(num_classes=10)
    cfg = TrainingConfig(
        epochs=2,
        warmup_epochs=0,
        learning_rate=0.01,
        stochastic_depth_p_max=0.0,  # disable for the smoke test
        use_amp=False,  # CPU run
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_model(
            model=model,
            data=bundle,
            cfg=cfg,
            output_dir=tmpdir,
            device=torch.device("cpu"),
        )

        # Checkpoint was written
        assert result.best_checkpoint_path.exists(), (
            f"Checkpoint missing at {result.best_checkpoint_path}"
        )

        # History has one record per epoch
        assert len(result.history) == 2, f"history has {len(result.history)} records"
        for record in result.history:
            assert isinstance(record, EpochRecord), f"got {type(record)}"
            assert 0.0 <= record.train_top1_acc <= 1.0, f"train_top1={record.train_top1_acc}"
            assert 0.0 <= record.val_top1_acc <= 1.0, f"val_top1={record.val_top1_acc}"
            # Note: LR can be 0 on the final epoch of cosine annealing.
            assert record.learning_rate >= 0.0, f"lr={record.learning_rate}"

        # The first epoch should have a strictly positive LR.
        assert result.history[0].learning_rate > 0.0, (
            f"First-epoch LR is non-positive: {result.history[0].learning_rate}"
        )

        # Final test metrics are present
        assert 0.0 <= result.final_test_top1_acc <= 1.0, (
            f"final_top1={result.final_test_top1_acc}"
        )
        assert 0.0 <= result.final_test_top5_acc <= 1.0, (
            f"final_top5={result.final_test_top5_acc}"
        )


def test_trainer_with_stochastic_depth():
    """
    Train for 2 epochs WITH Stochastic Depth enabled. Verify that the
    model contains SD-wrapped blocks after training.
    """
    model = _make_cifar_resnet18(num_classes=10)
    bundle = _make_synthetic_bundle(num_classes=10)
    cfg = TrainingConfig(
        epochs=2,
        warmup_epochs=0,
        learning_rate=0.01,
        stochastic_depth_p_max=0.1,
        use_amp=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_model(
            model=model,
            data=bundle,
            cfg=cfg,
            output_dir=tmpdir,
            device=torch.device("cpu"),
        )

    # After training, model should be SD-wrapped.
    state = get_stochastic_depth_state(model)
    assert len(state) == 8  # ResNet-18 has 8 blocks


def test_trainer_with_warmup_and_cosine():
    """
    Train with both warmup and cosine annealing active. Verify that
    the learning rate changes over epochs (decreasing from warmup peak).
    """
    model = _make_cifar_resnet18(num_classes=10)
    bundle = _make_synthetic_bundle(num_classes=10)
    cfg = TrainingConfig(
        epochs=4,
        warmup_epochs=2,
        learning_rate=0.05,
        stochastic_depth_p_max=0.0,
        use_amp=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_model(
            model=model, data=bundle, cfg=cfg,
            output_dir=tmpdir, device=torch.device("cpu"),
        )

    # LR should change across epochs (not be constant).
    lrs = [r.learning_rate for r in result.history]
    assert len(set(lrs)) > 1, f"LR was constant: {lrs}"


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        # stochastic_depth
        test_apply_stochastic_depth_wraps_all_blocks,
        test_linear_schedule_assigns_correct_probabilities,
        test_protect_entry_blocks_keeps_layer_zero_at_zero,
        test_unknown_schedule_rejected,
        test_wrapped_block_eval_matches_unwrapped,
        test_wrapped_block_supports_resnet_attributes,
        test_zero_init_last_bn_zeros_correct_layers,
        test_zero_init_works_through_stochastic_depth_wrapper,
        test_get_stochastic_depth_state_returns_assigned_probs,
        # data
        test_normalization_params_for_cifar10,
        test_normalization_params_for_cifar100,
        test_normalization_params_rejects_unknown_dataset,
        test_stratified_split_preserves_class_proportions,
        test_stratified_split_rejects_bad_ratio,
        test_build_transforms_train_has_augmentation,
        test_build_transforms_no_augmentation,
        # config validation
        test_training_config_rejects_zero_epochs,
        test_training_config_rejects_warmup_too_long,
        test_training_config_rejects_negative_warmup,
        test_training_config_rejects_zero_lr,
        # trainer end-to-end
        test_trainer_runs_end_to_end,
        test_trainer_with_stochastic_depth,
        test_trainer_with_warmup_and_cosine,
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
