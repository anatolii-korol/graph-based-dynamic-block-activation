"""
Integration tests for the GDBAController.

Covers:
  1. Construction via build() classmethod produces a working controller.
  2. step() refreshes on step 0 and at the configured interval.
  3. step() reuses gates between refreshes.
  4. forward() returns logits matching the wrapper's direct output.
  5. reset() clears state correctly.
  6. End-to-end: scoring + selection + gated forward produces deterministic
     output for the same input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ..application.controller import (  # noqa: E402
    GDBAController,
    GDBAStepResult,
)
from ..application.scoring import (  # noqa: E402
    GraphCentralityCache,
    compute_importance_scores,
    entropy_loss,
)
from ..domain.architecture import build_resnet18_graph  # noqa: E402
from ..domain.importance import ImportanceWeights  # noqa: E402


def _make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    torch.manual_seed(42)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def _random_batch(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Centrality cache
# ─────────────────────────────────────────────────────────────────────────────

def test_centrality_cache_populates_all_blocks():
    """Cache contains one value per block for each centrality measure."""
    graph = build_resnet18_graph()
    cache = GraphCentralityCache(graph)
    for bid in graph.all_block_ids:
        assert bid in cache.degree
        assert bid in cache.eigenvector
        assert bid in cache.pagerank
        # All values are finite
        assert 0.0 <= cache.degree[bid] <= 1.0
        assert 0.0 <= cache.eigenvector[bid] <= 1.0
        assert 0.0 <= cache.pagerank[bid] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_importance_scores
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_importance_scores_produces_valid_breakdown():
    """Score pipeline returns finite, non-negative scores for all blocks."""
    model = _make_cifar_resnet18()
    graph = build_resnet18_graph()
    cache = GraphCentralityCache(graph)
    batch_x = _random_batch()

    breakdown = compute_importance_scores(
        model=model,
        graph=graph,
        centrality_cache=cache,
        batch_x=batch_x,
    )

    assert set(breakdown.final.keys()) == set(graph.all_block_ids)
    for bid, score in breakdown.final.items():
        assert 0.0 <= score <= 1.0 + 1e-9, (
            f"Score for {bid} out of [0,1]: {score}"
        )


def test_compute_importance_scores_deterministic():
    """Same model, same batch -> identical scores."""
    model = _make_cifar_resnet18()
    graph = build_resnet18_graph()
    cache = GraphCentralityCache(graph)
    batch_x = _random_batch()

    b1 = compute_importance_scores(
        model=model, graph=graph, centrality_cache=cache, batch_x=batch_x
    )
    b2 = compute_importance_scores(
        model=model, graph=graph, centrality_cache=cache, batch_x=batch_x
    )
    for bid in graph.all_block_ids:
        assert abs(b1.final[bid] - b2.final[bid]) < 1e-9, (
            f"Non-determinism at {bid}: {b1.final[bid]} vs {b2.final[bid]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Controller construction
# ─────────────────────────────────────────────────────────────────────────────

def test_controller_build_basic():
    """build() with defaults yields a usable controller."""
    model = _make_cifar_resnet18()
    controller = GDBAController.build(model, top_k_ratio=0.5)
    assert controller.step_count == 0
    assert controller.last_selection is None
    assert controller.graph.num_blocks == 8  # ResNet-18


def test_controller_rejects_bad_refresh_interval():
    """refresh_interval < 1 is rejected."""
    model = _make_cifar_resnet18()
    try:
        GDBAController.build(model, top_k_ratio=0.5, refresh_interval=0)
    except ValueError:
        return
    raise AssertionError("Should reject refresh_interval=0")


# ─────────────────────────────────────────────────────────────────────────────
# Controller step()
# ─────────────────────────────────────────────────────────────────────────────

def test_step_refreshes_on_first_call():
    """The very first step always triggers a refresh."""
    model = _make_cifar_resnet18()
    controller = GDBAController.build(model, top_k_ratio=0.5)
    result = controller.step(_random_batch())
    assert result.refreshed is True
    assert result.score_breakdown is not None
    assert result.selection is not None
    assert controller.last_selection is not None


def test_step_uses_refresh_interval():
    """
    With refresh_interval=4, refreshes happen on steps 0, 4, 8, ...
    """
    model = _make_cifar_resnet18()
    controller = GDBAController.build(
        model, top_k_ratio=0.5, refresh_interval=4
    )

    refreshed_steps = []
    for i in range(10):
        result = controller.step(_random_batch(seed=i))
        if result.refreshed:
            refreshed_steps.append(i)

    assert refreshed_steps == [0, 4, 8], (
        f"Expected refreshes at [0, 4, 8], got {refreshed_steps}"
    )


def test_step_reuses_gates_between_refreshes():
    """
    On non-refresh steps, the gate state remains unchanged.
    """
    model = _make_cifar_resnet18()
    controller = GDBAController.build(
        model, top_k_ratio=0.5, refresh_interval=4
    )

    # Step 0: refresh and capture gates
    controller.step(_random_batch(seed=0))
    gates_step0 = frozenset(controller.wrapper.gated_off_blocks)

    # Steps 1, 2, 3: no refresh, gates should not change
    for i in range(1, 4):
        controller.step(_random_batch(seed=i))
        assert frozenset(controller.wrapper.gated_off_blocks) == gates_step0, (
            f"Gates changed at non-refresh step {i}"
        )


def test_step_top_k_ratio_one_keeps_all_blocks():
    """top_k_ratio=1.0 yields all blocks active after refresh."""
    model = _make_cifar_resnet18()
    controller = GDBAController.build(model, top_k_ratio=1.0)
    result = controller.step(_random_batch())
    assert controller.wrapper.num_active_blocks == controller.graph.num_blocks
    assert len(controller.wrapper.gated_off_blocks) == 0


def test_step_top_k_ratio_low_gates_some_blocks():
    """Aggressive ratio gates non-entry blocks (but keeps entries)."""
    model = _make_cifar_resnet18()
    controller = GDBAController.build(
        model, top_k_ratio=0.1, min_keep_per_stage=0
    )
    controller.step(_random_batch())
    # On ResNet-18 with 4 non-entry blocks, ratio=0.1 -> ceil(0.4)=1 kept.
    # So 3 non-entry blocks should be gated off.
    assert len(controller.wrapper.gated_off_blocks) > 0
    # No entry block is ever gated off
    for entry_id in controller.graph.entry_block_ids:
        assert entry_id not in controller.wrapper.gated_off_blocks


# ─────────────────────────────────────────────────────────────────────────────
# forward() vs step()
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_returns_same_logits_as_step():
    """
    forward(x) is equivalent to step(x).logits — same code path.
    """
    model = _make_cifar_resnet18()
    controller_a = GDBAController.build(model, top_k_ratio=0.5)
    controller_b = GDBAController.build(_make_cifar_resnet18(), top_k_ratio=0.5)

    x = _random_batch()
    out_step = controller_a.step(x).logits
    out_forward = controller_b(x)
    assert torch.allclose(out_step, out_forward, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_clears_state():
    """reset() returns the controller to its initial state."""
    model = _make_cifar_resnet18()
    controller = GDBAController.build(model, top_k_ratio=0.5)
    controller.step(_random_batch())
    assert controller.step_count == 1
    assert controller.last_selection is not None
    assert len(controller.wrapper.gated_off_blocks) >= 0

    controller.reset()
    assert controller.step_count == 0
    assert controller.last_selection is None
    assert len(controller.wrapper.gated_off_blocks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_determinism():
    """
    Two controllers built from identical models and run on the same
    batches must produce identical logits.
    """
    model_a = _make_cifar_resnet18()
    model_b = _make_cifar_resnet18()  # Same seed -> same weights

    # Sanity: identical weights
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p1, p2)

    ctrl_a = GDBAController.build(model_a, top_k_ratio=0.5)
    ctrl_b = GDBAController.build(model_b, top_k_ratio=0.5)

    for i in range(5):
        x = _random_batch(seed=i)
        out_a = ctrl_a(x)
        out_b = ctrl_b(x)
        assert torch.allclose(out_a, out_b, atol=1e-6), (
            f"Mismatch at step {i}: max diff "
            f"{(out_a - out_b).abs().max().item()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_centrality_cache_populates_all_blocks,
        test_compute_importance_scores_produces_valid_breakdown,
        test_compute_importance_scores_deterministic,
        test_controller_build_basic,
        test_controller_rejects_bad_refresh_interval,
        test_step_refreshes_on_first_call,
        test_step_uses_refresh_interval,
        test_step_reuses_gates_between_refreshes,
        test_step_top_k_ratio_one_keeps_all_blocks,
        test_step_top_k_ratio_low_gates_some_blocks,
        test_forward_returns_same_logits_as_step,
        test_reset_clears_state,
        test_end_to_end_determinism,
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
