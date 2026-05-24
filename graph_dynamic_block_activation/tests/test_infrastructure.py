"""
Integration tests for the infrastructure layer.

These tests instantiate a real torchvision ResNet, build the wrapper, and
verify:

  1. `inspect_resnet()` produces a BlockGraph identical to one built
     from the canonical specification.
  2. `GatedResNet` with NO gates produces the same logits as the bare
     ResNet (the wrapper is transparent when nothing is gated).
  3. `GatedResNet` with all-zero gates collapses each stage to just its
     entry block (verifies the gate logic actually fires).
  4. Activation tracer captures non-empty tensors and produces finite,
     non-negative magnitudes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ..domain.architecture import build_resnet18_graph  # noqa: E402
from ..infrastructure.activation_tracer import (  # noqa: E402
    BlockActivationTracer,
)
from ..infrastructure.architecture_inspector import (  # noqa: E402
    get_block_module,
    inspect_resnet,
)
from ..infrastructure.pytorch_wrapper import GatedResNet  # noqa: E402


def _make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a CIFAR-style ResNet-18 (3x3 stem, no maxpool, custom fc)."""
    model = resnet18(weights=None)
    # CIFAR stem patch (matches `models.py:_patch_resnet_cifar`).
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def _random_cifar_batch(batch_size: int = 4) -> torch.Tensor:
    """Random batch with CIFAR-10 shape (B, 3, 32, 32)."""
    torch.manual_seed(0)
    return torch.randn(batch_size, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Inspector
# ─────────────────────────────────────────────────────────────────────────────

def test_inspect_resnet18_matches_canonical():
    """`inspect_resnet` on a real ResNet-18 yields the canonical graph."""
    model = _make_cifar_resnet18()
    inspected = inspect_resnet(model)
    canonical = build_resnet18_graph()

    assert inspected.num_blocks == canonical.num_blocks
    assert inspected.num_stages == canonical.num_stages
    assert inspected.entry_block_ids == canonical.entry_block_ids
    assert inspected.all_block_ids == canonical.all_block_ids


def test_inspect_rejects_missing_stage():
    """If a stage attribute is missing, inspector raises AttributeError."""
    fake = nn.Module()
    fake.layer1 = nn.Sequential(nn.Identity())
    # Missing layer2, layer3, layer4
    try:
        inspect_resnet(fake)
    except AttributeError:
        return
    raise AssertionError("Should have raised AttributeError")


def test_get_block_module_returns_correct_object():
    """`get_block_module` resolves a domain ID to the actual nn.Module."""
    model = _make_cifar_resnet18()
    block = get_block_module(model, "layer2.1")
    # Should be the second item of layer2.
    assert block is model.layer2[1]


def test_get_block_module_rejects_bad_id():
    """Malformed or non-existent block IDs raise KeyError."""
    model = _make_cifar_resnet18()
    for bad_id in ["layer99.0", "layer1.99", "no_dot", "layer1.abc"]:
        try:
            get_block_module(model, bad_id)
        except KeyError:
            continue
        raise AssertionError(f"Should reject id={bad_id!r}")


# ─────────────────────────────────────────────────────────────────────────────
# GatedResNet — transparency
# ─────────────────────────────────────────────────────────────────────────────

def test_wrapper_with_no_gates_matches_bare_model():
    """
    GatedResNet without any gates set must produce IDENTICAL output to
    the raw torchvision model. This proves the wrapper has no
    accidental side effects.
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)

    x = _random_cifar_batch()
    with torch.no_grad():
        bare_output = model(x)
        wrapped_output = wrapper(x)

    assert torch.allclose(bare_output, wrapped_output, atol=1e-6), (
        f"Wrapper changed output! Max diff: "
        f"{(bare_output - wrapped_output).abs().max().item()}"
    )


def test_wrapper_with_all_active_gates_matches_bare_model():
    """
    Setting all gates to 1 explicitly is equivalent to no gating.
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)
    wrapper.set_gates({bid: 1 for bid in graph.all_block_ids})

    x = _random_cifar_batch()
    with torch.no_grad():
        bare_output = model(x)
        wrapped_output = wrapper(x)

    assert torch.allclose(bare_output, wrapped_output, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# GatedResNet — actual gating
# ─────────────────────────────────────────────────────────────────────────────

def test_wrapper_skips_gated_blocks():
    """
    With all non-entry blocks gated off, the wrapper should produce
    different output than the bare model (we are actually skipping work).
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)
    # Gate off everything except entry blocks
    wrapper.set_gates({bid: 0 for bid in graph.non_entry_block_ids})

    x = _random_cifar_batch()
    with torch.no_grad():
        bare_output = model(x)
        wrapped_output = wrapper(x)

    # If gating worked, outputs differ substantially
    diff = (bare_output - wrapped_output).abs().max().item()
    assert diff > 1e-3, (
        f"Outputs are too similar (max diff {diff}); "
        f"gating may not have fired"
    )


def test_wrapper_entry_blocks_cannot_be_gated_off():
    """
    Even if the user passes a gate value of 0 for an entry block, the
    wrapper must still execute it (safety net).
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)

    # Explicitly try to gate off everything including entry blocks.
    bad_gates = {bid: 0 for bid in graph.all_block_ids}
    wrapper.set_gates(bad_gates)

    # All entry block IDs must NOT appear in the gated_off set.
    for entry_id in graph.entry_block_ids:
        assert entry_id not in wrapper.gated_off_blocks, (
            f"Entry block {entry_id} was incorrectly gated off"
        )

    # Forward pass must still succeed (shape contract preserved).
    x = _random_cifar_batch()
    with torch.no_grad():
        out = wrapper(x)
    assert out.shape[0] == x.shape[0]
    assert torch.isfinite(out).all()


def test_wrapper_clear_gates_restores_full_model():
    """After clear_gates(), the wrapper output matches the bare model."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)

    wrapper.set_gates({bid: 0 for bid in graph.non_entry_block_ids})
    wrapper.clear_gates()

    x = _random_cifar_batch()
    with torch.no_grad():
        bare = model(x)
        wrapped = wrapper(x)
    assert torch.allclose(bare, wrapped, atol=1e-6)


def test_wrapper_num_active_blocks():
    """`num_active_blocks` reports correct count after gating."""
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    wrapper = GatedResNet(model, graph)

    assert wrapper.num_active_blocks == graph.num_blocks  # default: all
    # Gate off half the non-entry blocks
    non_entry = list(graph.non_entry_block_ids)
    to_gate = non_entry[: len(non_entry) // 2]
    gates = {bid: (0 if bid in to_gate else 1) for bid in graph.all_block_ids}
    wrapper.set_gates(gates)
    assert wrapper.num_active_blocks == graph.num_blocks - len(to_gate)


# ─────────────────────────────────────────────────────────────────────────────
# Activation tracer
# ─────────────────────────────────────────────────────────────────────────────

def test_tracer_captures_activations():
    """
    Tracer captures non-empty tensors and computes finite, non-negative
    magnitudes for every requested block.
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    block_ids = list(graph.all_block_ids)

    x = _random_cifar_batch()
    x.requires_grad_(False)

    with BlockActivationTracer(model, block_ids) as tracer:
        out = model(x)
        # Compute a simple loss and backward for saliency
        loss = out.sum()
        model.zero_grad()
        loss.backward()

    magnitudes = tracer.compute_activation_magnitudes()
    saliencies = tracer.compute_saliencies()

    assert set(magnitudes.keys()) == set(block_ids)
    assert set(saliencies.keys()) == set(block_ids)
    for bid, val in magnitudes.items():
        assert val >= 0.0 and val < float("inf"), (
            f"Bad magnitude for {bid}: {val}"
        )
    # At least some saliencies should be non-zero (loss is non-constant).
    assert any(v > 0.0 for v in saliencies.values())


def test_tracer_cleans_up_hooks_after_exit():
    """
    After the with-block exits, the model has no leftover hooks. Detect
    by checking that the model's _forward_hooks dicts are empty for the
    blocks we hooked.
    """
    model = _make_cifar_resnet18()
    graph = inspect_resnet(model)
    block_ids = list(graph.all_block_ids)

    with BlockActivationTracer(model, block_ids):
        pass

    for name, module in model.named_modules():
        if name in block_ids:
            assert not module._forward_hooks, (
                f"Block {name} has leftover hooks: {module._forward_hooks}"
            )


def test_tracer_is_not_reentrant():
    """A single tracer instance cannot be entered twice."""
    model = _make_cifar_resnet18()
    tracer = BlockActivationTracer(model, ["layer1.0"])
    with tracer:
        try:
            with tracer:
                pass
        except RuntimeError:
            return
    raise AssertionError("Re-entry should have raised RuntimeError")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_inspect_resnet18_matches_canonical,
        test_inspect_rejects_missing_stage,
        test_get_block_module_returns_correct_object,
        test_get_block_module_rejects_bad_id,
        test_wrapper_with_no_gates_matches_bare_model,
        test_wrapper_with_all_active_gates_matches_bare_model,
        test_wrapper_skips_gated_blocks,
        test_wrapper_entry_blocks_cannot_be_gated_off,
        test_wrapper_clear_gates_restores_full_model,
        test_wrapper_num_active_blocks,
        test_tracer_captures_activations,
        test_tracer_cleans_up_hooks_after_exit,
        test_tracer_is_not_reentrant,
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
