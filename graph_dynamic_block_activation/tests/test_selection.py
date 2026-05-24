"""
Validation tests for the top-k block selector.

Tests cover:
  1. Basic top-k selection (ranking, ceiling rounding).
  2. Entry-block force-active constraint.
  3. Per-stage minimum constraint.
  4. Edge cases: ratio=0, ratio=1, min_keep larger than stage size.
  5. Regression: identical gates to original `selectors.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..domain.architecture import (  # noqa: E402
    build_block_graph,
    build_resnet18_graph,
    build_resnet50_graph,
)
from ..domain.selection import TopKSelector, _k_from_ratio  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_scores(graph) -> dict:
    """All blocks score 0.5 — useful for testing constraints in isolation."""
    return {bid: 0.5 for bid in graph.all_block_ids}


def _descending_scores(graph) -> dict:
    """
    Scores that decrease linearly with block index: layer1.0 = highest,
    layer4.2 = lowest. Useful for verifying that top-k picks the highest.
    """
    n = graph.num_blocks
    return {bid: 1.0 - (i / n) for i, bid in enumerate(graph.all_block_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# _k_from_ratio
# ─────────────────────────────────────────────────────────────────────────────

def test_k_from_ratio_basic():
    """Standard ratios produce expected ceiling-rounded counts."""
    assert _k_from_ratio(12, 1.0) == 12
    assert _k_from_ratio(12, 0.5) == 6
    assert _k_from_ratio(12, 0.7) == 9  # ceil(8.4) = 9
    assert _k_from_ratio(12, 0.1) == 2  # ceil(1.2) = 2
    assert _k_from_ratio(12, 0.0) == 0


def test_k_from_ratio_clamps():
    """Edge cases at the boundaries are handled correctly."""
    # Empty set
    assert _k_from_ratio(0, 0.5) == 0
    # Single eligible block
    assert _k_from_ratio(1, 0.1) == 1  # ceil(0.1) = 1
    assert _k_from_ratio(1, 0.0) == 0


def test_k_from_ratio_rejects_bad_input():
    """Invalid ratios raise ValueError."""
    for bad in [-0.1, 1.1, 2.0]:
        try:
            _k_from_ratio(12, bad)
        except ValueError:
            continue
        raise AssertionError(f"Should reject ratio={bad}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry blocks are always active
# ─────────────────────────────────────────────────────────────────────────────

def test_entry_blocks_always_active_at_ratio_zero():
    """Even with ratio=0, all entry blocks remain active."""
    graph = build_resnet50_graph()
    selector = TopKSelector(top_k_ratio=0.0, min_keep_per_stage=0)
    result = selector.select(_uniform_scores(graph), graph)

    for entry_id in graph.entry_block_ids:
        assert result.gates[entry_id] == 1, f"Entry block {entry_id} gated off"

    # Non-entry blocks should all be gated off with ratio=0 and min=0
    for bid in graph.non_entry_block_ids:
        assert result.gates[bid] == 0, f"Non-entry {bid} unexpectedly active"


def test_entry_blocks_always_active_at_ratio_one():
    """At ratio=1, everything (entry + non-entry) is active."""
    graph = build_resnet50_graph()
    selector = TopKSelector(top_k_ratio=1.0, min_keep_per_stage=0)
    result = selector.select(_uniform_scores(graph), graph)

    for bid in graph.all_block_ids:
        assert result.gates[bid] == 1, f"Block {bid} not active at ratio=1"


# ─────────────────────────────────────────────────────────────────────────────
# Top-k respects score ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_top_k_picks_highest_scoring_blocks():
    """At ratio=0.5, the 6 highest-scoring non-entry blocks should be kept."""
    graph = build_resnet50_graph()
    scores = _descending_scores(graph)  # layer1.0 = highest, layer4.2 = lowest
    selector = TopKSelector(top_k_ratio=0.5, min_keep_per_stage=0)
    result = selector.select(scores, graph)

    # ratio=0.5, num_eligible=12 -> k=6
    assert result.requested_k == 6

    # The 6 highest-scoring NON-ENTRY blocks. Entry blocks are at indices
    # 0, 3, 7, 13 (layer1.0, layer2.0, layer3.0, layer4.0).
    # Non-entry ordered by descending score: layer1.1 (idx 1), layer1.2 (idx 2),
    # layer2.1 (idx 4), layer2.2 (idx 5), layer2.3 (idx 6), layer3.1 (idx 8),
    # layer3.2, layer3.3, layer3.4, layer3.5, layer4.1, layer4.2
    expected_top6_non_entry = ["layer1.1", "layer1.2", "layer2.1", "layer2.2",
                                "layer2.3", "layer3.1"]
    for bid in expected_top6_non_entry:
        assert result.gates[bid] == 1, f"Expected {bid} to be active"


# ─────────────────────────────────────────────────────────────────────────────
# Ceiling rounding behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_ratio_07_rounds_up():
    """At ratio=0.7 on ResNet-50 (12 eligible), k = ceil(8.4) = 9."""
    graph = build_resnet50_graph()
    selector = TopKSelector(top_k_ratio=0.7, min_keep_per_stage=0)
    result = selector.select(_descending_scores(graph), graph)
    assert result.requested_k == 9
    # 4 entry + 9 selected = 13 active blocks
    assert result.num_active_blocks == 13


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage minimum constraint
# ─────────────────────────────────────────────────────────────────────────────

def test_min_keep_per_stage_promotes_blocks():
    """
    On ResNet-50 with min_keep=2 and a score distribution that puts all
    top-k picks in layer3, the selector must promote non-entry blocks in
    the starved layers until each stage has at least 2 active blocks
    *total* (entry block counts toward the minimum).
    """
    graph = build_resnet50_graph()
    # All layer3 non-entry blocks score 1.0, everything else scores 0.0
    scores = {}
    for bid in graph.all_block_ids:
        scores[bid] = 1.0 if bid.startswith("layer3.") and bid != "layer3.0" else 0.0

    selector = TopKSelector(top_k_ratio=0.3, min_keep_per_stage=2)
    result = selector.select(scores, graph)

    # ratio=0.3, num_eligible=12 -> requested_k = ceil(3.6) = 4
    assert result.requested_k == 4

    # Each stage must have at least min(min_keep, stage_size) ACTIVE
    # blocks in total (entry block included). For ResNet-50 every stage
    # has at least 2 blocks total, so the effective minimum is 2.
    for stage in graph.stages:
        active_in_stage = sum(
            1 for bid in stage.block_ids if result.gates[bid] == 1
        )
        expected_min = min(2, len(stage.block_ids))
        assert active_in_stage >= expected_min, (
            f"Stage {stage.name} has only {active_in_stage} active blocks, "
            f"expected at least {expected_min}"
        )

    # Forced promotions should have happened in layers 1, 2, 4 (since
    # all top-k picks were concentrated in layer3, and entry blocks of
    # other stages already count for 1; each starved stage needs 1 more).
    assert result.forced_active_count >= 3


def test_resnet18_min_keep_2_keeps_everything():
    """
    On ResNet-18 each stage has only 2 blocks (1 entry + 1 non-entry).
    With min_keep_per_stage=2, but only 1 non-entry per stage, the
    effective minimum is min(2, 1) = 1, so every non-entry must be active.
    """
    graph = build_resnet18_graph()
    selector = TopKSelector(top_k_ratio=0.1, min_keep_per_stage=2)
    result = selector.select(_uniform_scores(graph), graph)

    # All blocks (entry and non-entry) should be active because each
    # stage's single non-entry block is forced active by the per-stage min.
    for bid in graph.all_block_ids:
        assert result.gates[bid] == 1, f"Block {bid} unexpectedly gated"


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_rejects_bad_ratio():
    """top_k_ratio outside [0, 1] is rejected."""
    for bad in [-0.1, 1.1]:
        try:
            TopKSelector(top_k_ratio=bad)
        except ValueError:
            continue
        raise AssertionError(f"Should reject top_k_ratio={bad}")


def test_rejects_negative_min_keep():
    """Negative min_keep_per_stage is rejected."""
    try:
        TopKSelector(top_k_ratio=0.5, min_keep_per_stage=-1)
    except ValueError:
        return
    raise AssertionError("Should reject negative min_keep_per_stage")


def test_missing_scores_raises():
    """If scores omit eligible blocks, select() raises ValueError."""
    graph = build_resnet50_graph()
    selector = TopKSelector(top_k_ratio=0.5)
    # Drop the score for one eligible block
    scores = {bid: 0.5 for bid in graph.all_block_ids if bid != "layer2.1"}
    try:
        selector.select(scores, graph)
    except ValueError:
        return
    raise AssertionError("Should raise when eligible block has no score")


# ─────────────────────────────────────────────────────────────────────────────
# Regression: identical gates to old StaticSelector
# ─────────────────────────────────────────────────────────────────────────────

def _old_selector_gates(scores, graph, ratio, min_keep):
    """
    Bit-exact replica of the OLD StaticSelector.compute_gates() logic from
    selectors.py, paper-mode only (no soft gating, no EMA, no stage_weights,
    ceil k-selection).

    Note: the original code counts the entry block toward the per-stage
    minimum (since entry blocks are part of the 'selected' set from the
    start). This test reproduces that semantic exactly.
    """
    import math as _math

    entry_blocks = set(graph.entry_block_ids)
    non_protected = [
        bid for bid in graph.all_block_ids if bid not in entry_blocks
    ]
    n = len(non_protected)
    k = max(0, min(n, _math.ceil(ratio * n)))

    # Sort by score descending; break ties by name for determinism
    ranked = sorted(non_protected, key=lambda b: (-scores[b], b))
    selected = set(entry_blocks) | set(ranked[:k])

    # Per-stage minimum (counts entry/protected blocks toward kept)
    if min_keep > 0:
        for stage in graph.stages:
            stage_names = list(stage.block_ids)
            kept = [x for x in stage_names if x in selected]
            if len(kept) < min_keep:
                unsel = [x for x in stage_names if x not in selected]
                ranked_stage = sorted(unsel, key=lambda x: (-scores[x], x))
                selected.update(ranked_stage[: (min_keep - len(kept))])

    return {
        bid: (1 if bid in selected else 0)
        for bid in graph.all_block_ids
    }


def test_regression_resnet50_uniform_scores():
    """Equivalence with old code under uniform scores at multiple ratios."""
    graph = build_resnet50_graph()
    scores = _uniform_scores(graph)
    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for min_keep in [0, 1, 2]:
            new_sel = TopKSelector(top_k_ratio=ratio, min_keep_per_stage=min_keep)
            new_result = new_sel.select(scores, graph)
            old_gates = _old_selector_gates(scores, graph, ratio, min_keep)
            assert new_result.gates == old_gates, (
                f"Mismatch at ratio={ratio}, min_keep={min_keep}"
            )


def test_regression_resnet50_descending_scores():
    """Equivalence with old code under descending scores at multiple ratios."""
    graph = build_resnet50_graph()
    scores = _descending_scores(graph)
    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for min_keep in [0, 1, 2]:
            new_sel = TopKSelector(top_k_ratio=ratio, min_keep_per_stage=min_keep)
            new_result = new_sel.select(scores, graph)
            old_gates = _old_selector_gates(scores, graph, ratio, min_keep)
            assert new_result.gates == old_gates, (
                f"Mismatch at ratio={ratio}, min_keep={min_keep}"
            )


def test_regression_resnet18_descending_scores():
    """Equivalence on the smaller ResNet-18."""
    graph = build_resnet18_graph()
    scores = _descending_scores(graph)
    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for min_keep in [0, 1, 2]:
            new_sel = TopKSelector(top_k_ratio=ratio, min_keep_per_stage=min_keep)
            new_result = new_sel.select(scores, graph)
            old_gates = _old_selector_gates(scores, graph, ratio, min_keep)
            assert new_result.gates == old_gates, (
                f"Mismatch at ratio={ratio}, min_keep={min_keep}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_k_from_ratio_basic,
        test_k_from_ratio_clamps,
        test_k_from_ratio_rejects_bad_input,
        test_entry_blocks_always_active_at_ratio_zero,
        test_entry_blocks_always_active_at_ratio_one,
        test_top_k_picks_highest_scoring_blocks,
        test_ratio_07_rounds_up,
        test_min_keep_per_stage_promotes_blocks,
        test_resnet18_min_keep_2_keeps_everything,
        test_rejects_bad_ratio,
        test_rejects_negative_min_keep,
        test_missing_scores_raises,
        test_regression_resnet50_uniform_scores,
        test_regression_resnet50_descending_scores,
        test_regression_resnet18_descending_scores,
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
