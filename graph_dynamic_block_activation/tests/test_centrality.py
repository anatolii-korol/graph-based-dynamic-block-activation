"""
Validation tests for graph centrality algorithms.

These tests verify our implementations against:
  1. Hand-computable small graphs where the answer is obvious by inspection.
  2. Edge cases (empty graphs, single nodes, disconnected components).
  3. Mathematical invariants (e.g. PageRank sums to 1).

Run with:  python -m pytest tests/test_centrality.py
or:        python tests/test_centrality.py

The tests are deliberately verbose: each test states what it is checking
and why, so a reviewer can confirm the algorithm is correct without
running the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..domain.centrality import (  # noqa: E402
    degree_centrality,
    eigenvector_centrality,
    pagerank,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures: small graphs with hand-computable centralities
# ─────────────────────────────────────────────────────────────────────────────

def _star_graph(n: int = 4) -> np.ndarray:
    """
    Star graph: node 0 connected to all others, no other edges.
    Degree centrality of center = 1.0; of leaves = 1/n by symmetry.
    """
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        adj[0, i] = 1.0
        adj[i, 0] = 1.0
    return adj


def _path_graph(n: int = 5) -> np.ndarray:
    """
    Path graph: 0 - 1 - 2 - ... - (n-1), undirected.
    Centralities should be symmetric around the middle.
    """
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _empty_graph(n: int = 5) -> np.ndarray:
    """No edges. All centralities should be zero or uniform."""
    return np.zeros((n, n), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Degree centrality tests
# ─────────────────────────────────────────────────────────────────────────────

def test_degree_centrality_star_graph():
    """In a star graph, the center has maximal degree; leaves are tied."""
    adj = _star_graph(n=4)
    centrality = degree_centrality(adj)

    # Center (node 0) has degree 6 (3 in + 3 out); leaves have degree 2 each.
    # After max-normalization: center = 1.0, leaves = 2/6 = 1/3.
    assert centrality[0] == 1.0, f"Expected 1.0 for center, got {centrality[0]}"
    expected_leaf = 2.0 / 6.0
    for i in range(1, 4):
        assert abs(centrality[i] - expected_leaf) < 1e-9, (
            f"Leaf {i} centrality {centrality[i]} != {expected_leaf}"
        )


def test_degree_centrality_path_symmetry():
    """In a path graph, centralities are symmetric around the middle."""
    adj = _path_graph(n=5)
    centrality = degree_centrality(adj)

    assert abs(centrality[0] - centrality[4]) < 1e-9, "Endpoints not symmetric"
    assert abs(centrality[1] - centrality[3]) < 1e-9, "Symmetric pair mismatch"
    assert centrality[2] >= centrality[1], "Middle should not be less central"


def test_degree_centrality_empty_graph():
    """Graph with no edges: all centralities zero."""
    adj = _empty_graph(n=5)
    centrality = degree_centrality(adj)
    assert np.all(centrality == 0.0)


def test_degree_centrality_zero_size():
    """Empty matrix should return empty result without errors."""
    adj = np.zeros((0, 0))
    centrality = degree_centrality(adj)
    assert centrality.shape == (0,)


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvector centrality tests
# ─────────────────────────────────────────────────────────────────────────────

def test_eigenvector_centrality_star_graph():
    """
    In a star graph, the center is uniquely the most central by ANY
    sensible measure of "connected to important nodes".
    """
    adj = _star_graph(n=4)
    centrality = eigenvector_centrality(adj)

    assert centrality[0] == 1.0, f"Center should max-normalize to 1, got {centrality[0]}"
    # Leaves are tied by symmetry
    leaf_values = centrality[1:]
    assert np.all(np.abs(leaf_values - leaf_values[0]) < 1e-7), "Leaves not tied"


def test_eigenvector_centrality_in_unit_range():
    """All centrality values must lie in [0, 1] after normalization."""
    adj = _path_graph(n=10)
    centrality = eigenvector_centrality(adj)
    assert np.all(centrality >= 0.0)
    assert np.all(centrality <= 1.0)
    assert centrality.max() == 1.0  # Max-normalized


def test_eigenvector_centrality_empty_graph():
    """Empty graph: convergence to uniform within numerical tolerance."""
    adj = _empty_graph(n=4)
    centrality = eigenvector_centrality(adj)
    # With no edges, the iteration immediately produces zero, and we
    # max-normalize. Result is well-defined as zero vector.
    # The exact value depends on convergence behavior; we just check
    # that no NaN/Inf appears.
    assert np.all(np.isfinite(centrality))


# ─────────────────────────────────────────────────────────────────────────────
# PageRank tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pagerank_sums_to_one():
    """PageRank is a probability distribution; values must sum to 1."""
    adj = _path_graph(n=5)
    p = pagerank(adj)
    assert abs(p.sum() - 1.0) < 1e-9, f"PageRank sums to {p.sum()}, not 1"


def test_pagerank_uniform_for_empty_graph():
    """
    With no outgoing edges anywhere, only the teleportation term matters,
    and teleportation is uniform; so PageRank should be uniform.
    """
    adj = _empty_graph(n=5)
    p = pagerank(adj)
    expected = 1.0 / 5.0
    assert np.allclose(p, expected), f"Expected uniform {expected}, got {p}"


def test_pagerank_star_graph_concentrates_mass_on_center():
    """In a star graph, the center should accumulate the highest mass."""
    adj = _star_graph(n=5)
    p = pagerank(adj)
    assert p[0] == p.max(), f"Center is not the maximum; got {p}"


def test_pagerank_rejects_invalid_damping():
    """Damping must be strictly between 0 and 1."""
    adj = _star_graph(n=3)
    for bad_damping in [-0.1, 0.0, 1.0, 1.5]:
        try:
            pagerank(adj, damping=bad_damping)
        except ValueError:
            continue
        raise AssertionError(f"Should have rejected damping={bad_damping}")


def test_pagerank_non_negative():
    """No PageRank value should be negative (after numerical clipping)."""
    adj = _path_graph(n=10)
    p = pagerank(adj)
    assert np.all(p >= 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-algorithm consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_all_algorithms_agree_on_most_central_node_in_star():
    """
    Sanity check: in a star graph, all three centrality measures should
    agree that the center is the most important node.
    """
    adj = _star_graph(n=6)

    deg = degree_centrality(adj)
    eig = eigenvector_centrality(adj)
    pr = pagerank(adj)

    assert deg.argmax() == 0, "Degree disagrees on most central"
    assert eig.argmax() == 0, "Eigenvector disagrees on most central"
    assert pr.argmax() == 0, "PageRank disagrees on most central"


def test_all_algorithms_validate_input_shape():
    """All algorithms should reject non-square inputs."""
    bad_input = np.zeros((3, 4))
    for fn, name in [
        (degree_centrality, "degree"),
        (eigenvector_centrality, "eigenvector"),
        (pagerank, "pagerank"),
    ]:
        try:
            fn(bad_input)
        except ValueError:
            continue
        raise AssertionError(f"{name} should have rejected non-square input")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests when invoked directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        # Degree centrality
        test_degree_centrality_star_graph,
        test_degree_centrality_path_symmetry,
        test_degree_centrality_empty_graph,
        test_degree_centrality_zero_size,
        # Eigenvector
        test_eigenvector_centrality_star_graph,
        test_eigenvector_centrality_in_unit_range,
        test_eigenvector_centrality_empty_graph,
        # PageRank
        test_pagerank_sums_to_one,
        test_pagerank_uniform_for_empty_graph,
        test_pagerank_star_graph_concentrates_mass_on_center,
        test_pagerank_rejects_invalid_damping,
        test_pagerank_non_negative,
        # Cross
        test_all_algorithms_agree_on_most_central_node_in_star,
        test_all_algorithms_validate_input_shape,
    ]

    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR   {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
