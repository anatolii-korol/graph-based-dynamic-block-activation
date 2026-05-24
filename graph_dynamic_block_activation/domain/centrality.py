"""
Pure graph centrality algorithms.

This module contains the mathematical primitives for computing node
importance in a directed weighted graph. It deliberately has *no*
dependencies on PyTorch, neural network architectures, or any other
domain-specific machinery — the inputs and outputs are plain NumPy arrays.

The separation matters for two reasons:
  1. Testability: each algorithm can be validated on canonical small
     graphs from the literature (e.g. Zachary's karate club, the Florentine
     marriage network) without spinning up a neural network.
  2. Reusability: the same algorithms apply to any directed graph; we
     simply happen to use them on residual-block graphs.

Three centrality measures are implemented, each capturing a different
notion of "importance":

  - degree_centrality      : count of incident edges (local connectivity)
  - eigenvector_centrality : importance weighted by neighbours' importance
                             (Bonacich 1987)
  - pagerank               : importance via random-walk stationary
                             distribution (Page & Brin 1998)

All three return values normalized so they can be combined linearly with
data-dependent signals (activations, saliency) in the final importance
score (Equation (1) in the paper).

References:
  Bonacich, P. (1987). Power and Centrality: A Family of Measures.
    American Journal of Sociology, 92(5), 1170-1182.
  Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank
    Citation Ranking: Bringing Order to the Web. Stanford InfoLab.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    PAGERANK_DAMPING_FACTOR,
    POWER_ITERATION_MAX_ITER,
    POWER_ITERATION_TOLERANCE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases for documentation purposes.
# ─────────────────────────────────────────────────────────────────────────────

# A directed weighted adjacency matrix. adj[i, j] > 0 iff there is an edge
# from node i to node j; the value is the edge weight.
AdjacencyMatrix = np.ndarray  # shape: (n_nodes, n_nodes), dtype: float64

# A vector of per-node centrality scores in [0, 1]. The scaling convention
# differs per algorithm (max-normalized for degree/eigenvector,
# sum-normalized for PageRank) — see each function's docstring.
CentralityVector = np.ndarray  # shape: (n_nodes,), dtype: float64


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_adjacency(adj: AdjacencyMatrix) -> None:
    """
    Validate that the input is a square 2D matrix suitable for centrality
    computation. Raises ValueError on malformed input.

    We deliberately do *not* check that values are non-negative — the
    caller is trusted on this front, and signed adjacency matrices have
    legitimate uses (e.g. signed networks).
    """
    if adj.ndim != 2:
        raise ValueError(
            f"Adjacency matrix must be 2D, got shape {adj.shape}"
        )
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square, got shape {adj.shape}"
        )


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Row-stochastic transformation: divide each row by its sum.

    Rows that sum to zero are left as zero rows (rather than dividing by
    zero). This handles "sink" nodes — vertices with no outgoing edges —
    in a numerically safe way; PageRank then handles them via teleportation.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Use a copy to avoid mutating the input, and replace zero sums with 1
    # so that division yields zero rows (since the row was already zero).
    safe_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return matrix / safe_sums


def _max_normalize(values: np.ndarray) -> np.ndarray:
    """
    Scale a non-negative vector by its maximum so that the largest value
    becomes 1. If the maximum is zero, returns a zero vector (avoiding
    division by zero).

    This normalization preserves relative ordering and ratios, which is
    what we need for combining different centrality measures.
    """
    abs_values = np.abs(values)
    max_value = abs_values.max() if abs_values.size else 0.0
    if max_value == 0.0:
        return np.zeros_like(values)
    return abs_values / max_value


# ─────────────────────────────────────────────────────────────────────────────
# Centrality 1: Degree
# ─────────────────────────────────────────────────────────────────────────────

def degree_centrality(adj: AdjacencyMatrix) -> CentralityVector:
    """
    Compute degree centrality on a directed weighted graph.

    For node i, degree centrality is the sum of weights of all incident
    edges, both incoming and outgoing:

        C_deg(i) = sum_j adj[i, j] + sum_j adj[j, i]

    This counts each undirected pair once in each direction; it is sometimes
    called "total degree centrality" to distinguish from in-degree or
    out-degree alone. The result is then max-normalized to [0, 1].

    Interpretation
    --------------
    Degree centrality is the simplest measure: a node is important if it
    has many connections. It captures local connectivity but ignores the
    global structure — a node with three connections to three isolated
    leaves scores the same as a node with three connections to three hubs.

    Args:
        adj: (n, n) directed adjacency matrix, floating-point.

    Returns:
        (n,) vector of degree centrality scores in [0, 1].
    """
    _validate_adjacency(adj)
    out_degree = adj.sum(axis=1)  # weighted out-degree
    in_degree = adj.sum(axis=0)   # weighted in-degree
    total_degree = out_degree + in_degree
    return _max_normalize(total_degree)


# ─────────────────────────────────────────────────────────────────────────────
# Centrality 2: Eigenvector
# ─────────────────────────────────────────────────────────────────────────────

def eigenvector_centrality(
    adj: AdjacencyMatrix,
    *,
    max_iterations: int = POWER_ITERATION_MAX_ITER,
    tolerance: float = POWER_ITERATION_TOLERANCE,
) -> CentralityVector:
    """
    Compute eigenvector centrality via power iteration on the symmetrized
    adjacency matrix.

    The intuition: a node is important if it is connected to other
    important nodes. Mathematically, eigenvector centrality is the
    eigenvector x associated with the largest eigenvalue lambda of the
    adjacency matrix:

        A x = lambda * x

    For a directed graph, we symmetrize first (A_sym = A + A^T) so that
    the principal eigenvalue is real and the corresponding eigenvector is
    non-negative (Perron-Frobenius theorem). This is the standard treatment
    when applying eigenvector centrality to directed graphs.

    Algorithm: power iteration
    --------------------------
    Starting from x_0 = (1/n, 1/n, ..., 1/n), iterate

        x_{k+1} = A_sym x_k / ||A_sym x_k||_2

    until ||x_{k+1} - x_k||_1 < tolerance, or max_iterations is reached.

    Power iteration converges to the dominant eigenvector at a rate
    governed by the ratio of the second to the first eigenvalue. For our
    block graphs (8-16 nodes, well-connected), convergence typically
    occurs within 30-50 iterations.

    Numerical notes
    ---------------
    The L2-normalization at each step prevents overflow and underflow.
    The final result is post-processed with `abs()` (since power iteration
    can converge to either the +x or -x eigenvector) and max-normalized
    to align with the convention of degree_centrality.

    Args:
        adj: (n, n) directed adjacency matrix.
        max_iterations: hard cap on iterations (default from constants).
        tolerance: L1 convergence threshold (default from constants).

    Returns:
        (n,) vector of eigenvector-centrality scores in [0, 1].
    """
    _validate_adjacency(adj)
    n = adj.shape[0]

    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Symmetrize: A_sym[i, j] = adj[i, j] + adj[j, i]. This makes the
    # spectrum real and the principal eigenvector non-negative.
    symmetric_adj = adj + adj.T

    # Initialize with a uniform vector. Any non-zero vector with a non-zero
    # component along the principal eigenvector would converge; uniform is
    # simplest and avoids accidentally starting orthogonal to the target.
    x = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iterations):
        x_next = symmetric_adj @ x
        x_next_norm = np.linalg.norm(x_next, ord=2)

        # If the matrix-vector product collapsed to zero, the graph has no
        # edges (or x became orthogonal to all eigenvectors with non-zero
        # eigenvalue). Return the current iterate.
        if x_next_norm == 0.0:
            break

        x_next = x_next / x_next_norm

        # Convergence check in L1 norm (more conservative than L2 for
        # detecting tail oscillations).
        if np.linalg.norm(x_next - x, ord=1) < tolerance:
            x = x_next
            break

        x = x_next

    return _max_normalize(x)


# ─────────────────────────────────────────────────────────────────────────────
# Centrality 3: PageRank
# ─────────────────────────────────────────────────────────────────────────────

def pagerank(
    adj: AdjacencyMatrix,
    *,
    damping: float = PAGERANK_DAMPING_FACTOR,
    max_iterations: int = POWER_ITERATION_MAX_ITER,
    tolerance: float = POWER_ITERATION_TOLERANCE,
) -> CentralityVector:
    """
    Compute PageRank: the stationary distribution of a random walk that, at
    each step, follows an outgoing edge with probability `damping` or
    teleports to a uniformly random node with probability (1 - damping).

    Mathematically, PageRank solves the fixed-point equation

        p = (1 - damping) * (1/n) * 1_n + damping * P^T p

    where P is the row-stochastic transition matrix (each row sums to 1,
    or to 0 for sink nodes), 1_n is the all-ones vector, and 1/n is the
    uniform teleportation distribution. The result p is normalized to
    sum to 1, like a probability distribution.

    Algorithm: power iteration (modified for teleportation)
    -------------------------------------------------------
    Starting from a uniform p_0, iterate

        p_{k+1} = (1 - damping) / n + damping * P^T p_k

    until ||p_{k+1} - p_k||_1 < tolerance.

    Convergence is guaranteed by the contraction-mapping property: the
    teleportation term ensures that the iteration is a strict contraction
    with rate `damping`, so convergence is exponential at rate
    -ln(damping) ~ 0.16 per iteration for damping=0.85.

    Why the standard damping=0.85
    -----------------------------
    See `constants.PAGERANK_DAMPING_FACTOR` for details. Briefly: this is
    the value adopted in the original Google PageRank paper, validated
    across diverse web graphs. Smaller damping yields more uniform
    distributions; larger damping concentrates mass on hubs.

    Args:
        adj: (n, n) directed adjacency matrix.
        damping: random-walk continuation probability in (0, 1).
        max_iterations: convergence iteration cap.
        tolerance: L1 convergence threshold.

    Returns:
        (n,) probability vector, sum = 1, all entries in [0, 1].
    """
    _validate_adjacency(adj)
    if not (0.0 < damping < 1.0):
        raise ValueError(
            f"PageRank damping must be strictly between 0 and 1; got {damping}"
        )

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Convert adjacency to a transition matrix where rows sum to 1.
    # We work on a float64 copy to avoid mutating the input.
    transition = _row_normalize(adj.astype(np.float64, copy=True))

    # Initialize with uniform mass and define the teleport distribution.
    p = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)
    teleport_factor = (1.0 - damping)

    for _ in range(max_iterations):
        # transition.T @ p sums probability mass flowing INTO each node
        # along outgoing edges (since transition[i, j] is the probability
        # of i -> j, transition[i, j] * p[i] is mass flowing from i to j;
        # summing over i gives total inflow into j).
        p_next = teleport_factor * teleport + damping * (transition.T @ p)

        if np.linalg.norm(p_next - p, ord=1) < tolerance:
            p = p_next
            break

        p = p_next

    # Clip any small negative values from numerical error and renormalize.
    # Without renormalization the result might sum to slightly less than 1
    # because clipping removes negative noise.
    p = np.maximum(p, 0.0)
    total = p.sum()
    if total > 0.0:
        p = p / total

    return p
