"""
Validation tests for the architecture (block topology) module.

Tests cover:
  1. Block and stage descriptor invariants.
  2. Graph construction from stage specifications.
  3. Adjacency matrix correctness for canonical ResNets.
  4. Edge cases (empty specs, duplicate names, zero-block stages).
  5. Regression: identical adjacency to original `graph.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..constants import SEQUENTIAL_EDGE_WEIGHT, SKIP_EDGE_WEIGHT  # noqa: E402
from ..domain.architecture import (  # noqa: E402
    BlockDescriptor,
    StageDescriptor,
    build_block_graph,
    build_resnet18_graph,
    build_resnet50_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# BlockDescriptor invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_block_descriptor_entry_at_index_zero():
    """The entry block must be at index 0; index 0 must be entry."""
    # Valid: index 0, marked as entry
    b = BlockDescriptor("layer1.0", "layer1", 0, is_entry_block=True)
    assert b.is_entry_block

    # Valid: non-zero index, not entry
    b = BlockDescriptor("layer1.3", "layer1", 3, is_entry_block=False)
    assert not b.is_entry_block

    # Invalid: index 0 but marked as non-entry — should raise
    try:
        BlockDescriptor("layer1.0", "layer1", 0, is_entry_block=False)
    except ValueError:
        return
    raise AssertionError("Should have rejected index-0 non-entry block")


# ─────────────────────────────────────────────────────────────────────────────
# StageDescriptor invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_descriptor_basic():
    """Stages expose entry block, total, and non-entry count correctly."""
    stage = StageDescriptor(
        name="layer1", block_ids=("layer1.0", "layer1.1", "layer1.2")
    )
    assert stage.entry_block == "layer1.0"
    assert stage.num_blocks == 3
    assert stage.num_non_entry_blocks == 2


def test_stage_descriptor_empty_rejected():
    """A stage must contain at least one block."""
    try:
        StageDescriptor(name="layer1", block_ids=())
    except ValueError:
        return
    raise AssertionError("Should have rejected empty stage")


# ─────────────────────────────────────────────────────────────────────────────
# build_block_graph: basic construction
# ─────────────────────────────────────────────────────────────────────────────

def test_build_graph_minimal():
    """Single stage with one block should build successfully."""
    graph = build_block_graph([("layer1", 1)])
    assert graph.num_blocks == 1
    assert graph.num_stages == 1
    assert graph.entry_block_ids == {"layer1.0"}
    assert graph.non_entry_block_ids == ()
    assert graph.adjacency.shape == (1, 1)
    assert graph.adjacency[0, 0] == 0.0  # No self-loop


def test_build_graph_resnet18_structure():
    """ResNet-18 has 8 blocks, 4 stages, 4 entry blocks."""
    graph = build_resnet18_graph()
    assert graph.num_blocks == 8
    assert graph.num_stages == 4
    assert graph.entry_block_ids == {
        "layer1.0", "layer2.0", "layer3.0", "layer4.0",
    }
    assert len(graph.non_entry_block_ids) == 4


def test_build_graph_resnet50_structure():
    """ResNet-50 has 16 blocks across 4 stages with 3+4+6+3 distribution."""
    graph = build_resnet50_graph()
    assert graph.num_blocks == 16
    assert graph.num_stages == 4

    expected_per_stage = {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}
    for stage in graph.stages:
        assert len(stage.block_ids) == expected_per_stage[stage.name], (
            f"Stage {stage.name}: expected {expected_per_stage[stage.name]} blocks, "
            f"got {len(stage.block_ids)}"
        )

    # 4 entry blocks + 12 non-entry blocks
    assert len(graph.entry_block_ids) == 4
    assert len(graph.non_entry_block_ids) == 12


# ─────────────────────────────────────────────────────────────────────────────
# build_block_graph: input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_build_graph_empty_specs_rejected():
    """Empty stage list should raise."""
    try:
        build_block_graph([])
    except ValueError:
        return
    raise AssertionError("Should have rejected empty specs")


def test_build_graph_duplicate_stage_names_rejected():
    """Same stage name twice should raise."""
    try:
        build_block_graph([("layer1", 2), ("layer1", 3)])
    except ValueError:
        return
    raise AssertionError("Should have rejected duplicate stage names")


def test_build_graph_zero_blocks_rejected():
    """A stage with zero blocks is meaningless and should be rejected."""
    try:
        build_block_graph([("layer1", 0)])
    except ValueError:
        return
    raise AssertionError("Should have rejected zero-block stage")


def test_build_graph_empty_stage_name_rejected():
    """Empty stage name is rejected."""
    try:
        build_block_graph([("", 3)])
    except ValueError:
        return
    raise AssertionError("Should have rejected empty stage name")


# ─────────────────────────────────────────────────────────────────────────────
# Adjacency matrix structure
# ─────────────────────────────────────────────────────────────────────────────

def test_adjacency_sequential_edges_resnet50():
    """Every block i has a sequential edge to block i+1 (weight 1.0)."""
    graph = build_resnet50_graph()
    n = graph.num_blocks
    for i in range(n - 1):
        assert graph.adjacency[i, i + 1] == SEQUENTIAL_EDGE_WEIGHT, (
            f"Missing sequential edge {i} -> {i+1}"
        )


def test_adjacency_skip_edges_within_stage():
    """
    Within each stage, edge (i+1, i) exists with weight SKIP_EDGE_WEIGHT.
    These do not cross stage boundaries.
    """
    graph = build_resnet50_graph()
    # layer2 has 4 blocks at indices 3..6. Backward edges within: 4->3, 5->4, 6->5.
    layer2_indices = [3, 4, 5, 6]
    for a, b in zip(layer2_indices[1:], layer2_indices[:-1]):
        assert graph.adjacency[a, b] == SKIP_EDGE_WEIGHT, (
            f"Missing within-stage skip edge {a} -> {b}"
        )


def test_adjacency_no_cross_stage_skip():
    """
    Backward edges do NOT cross stage boundaries (different spatial
    resolutions break the skip path).
    """
    graph = build_resnet50_graph()
    # layer1 ends at index 2, layer2 starts at index 3.
    # There must be NO backward skip edge 3 -> 2.
    # But the forward sequential edge 2 -> 3 must still exist.
    assert graph.adjacency[3, 2] == 0.0, "Spurious cross-stage backward skip edge"
    assert graph.adjacency[2, 3] == SEQUENTIAL_EDGE_WEIGHT, (
        "Missing cross-stage forward edge"
    )


def test_adjacency_no_self_loops():
    """No block should have an edge to itself."""
    graph = build_resnet50_graph()
    diagonal = np.diag(graph.adjacency)
    assert np.all(diagonal == 0.0), f"Self-loops present: {np.where(diagonal != 0)}"


# ─────────────────────────────────────────────────────────────────────────────
# Lookup methods
# ─────────────────────────────────────────────────────────────────────────────

def test_index_of():
    """index_of returns the correct row in the adjacency matrix."""
    graph = build_resnet50_graph()
    assert graph.index_of("layer1.0") == 0
    assert graph.index_of("layer4.2") == 15

    try:
        graph.index_of("layer99.0")
    except KeyError:
        return
    raise AssertionError("Should have raised KeyError for unknown block")


def test_stage_of():
    """stage_of returns the correct stage for a given block."""
    graph = build_resnet50_graph()
    stage = graph.stage_of("layer2.1")
    assert stage.name == "layer2"
    assert "layer2.1" in stage.block_ids


def test_blocks_in_stage():
    """blocks_in_stage returns all block IDs in order."""
    graph = build_resnet50_graph()
    layer1_blocks = graph.blocks_in_stage("layer1")
    assert layer1_blocks == ("layer1.0", "layer1.1", "layer1.2")


# ─────────────────────────────────────────────────────────────────────────────
# Regression test: identical adjacency to original graph.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_old_graph_adjacency_resnet50() -> np.ndarray:
    """
    Reconstruct the adjacency matrix that the original `graph.py` would
    produce for ResNet-50. This is the *exact same construction logic*
    extracted from `build_resnet_block_graph()` in the old code.
    """
    n = 16
    adj = np.zeros((n, n), dtype=np.float64)
    # Sequential
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    # Within-stage skip
    stages = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15]]
    for stage_indices in stages:
        for a, b in zip(stage_indices[:-1], stage_indices[1:]):
            adj[a, b] = 1.0
            adj[b, a] = 0.25
    # Stage transitions
    for prev, nxt in zip(stages[:-1], stages[1:]):
        adj[prev[-1], nxt[0]] = 1.0
    return adj


def test_regression_resnet50_adjacency_matches_old_code():
    """The new graph builder must produce a bit-identical adjacency matrix."""
    old_adj = _build_old_graph_adjacency_resnet50()
    new_graph = build_resnet50_graph()
    assert np.array_equal(old_adj, new_graph.adjacency), (
        f"Max difference: {np.abs(old_adj - new_graph.adjacency).max()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        test_block_descriptor_entry_at_index_zero,
        test_stage_descriptor_basic,
        test_stage_descriptor_empty_rejected,
        test_build_graph_minimal,
        test_build_graph_resnet18_structure,
        test_build_graph_resnet50_structure,
        test_build_graph_empty_specs_rejected,
        test_build_graph_duplicate_stage_names_rejected,
        test_build_graph_zero_blocks_rejected,
        test_build_graph_empty_stage_name_rejected,
        test_adjacency_sequential_edges_resnet50,
        test_adjacency_skip_edges_within_stage,
        test_adjacency_no_cross_stage_skip,
        test_adjacency_no_self_loops,
        test_index_of,
        test_stage_of,
        test_blocks_in_stage,
        test_regression_resnet50_adjacency_matches_old_code,
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
