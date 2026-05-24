"""
Pure-domain implementation of the GDBA importance formula.

This module computes the per-block importance score from Equation (1):

    Score(v_i) = alpha * activation(v_i)
               + beta  * saliency(v_i)
               + gamma * degree_centrality(v_i)
               + delta * eigenvector_centrality(v_i)
               + epsilon * pagerank(v_i)

It deliberately works on plain dictionaries of floats — the caller is
responsible for supplying the raw signals (activation magnitudes,
saliencies, centralities). This keeps the formula testable in isolation
from PyTorch model tracing and gradient computation.

Two design points are worth highlighting:

  1. **Normalization is pluggable.** The raw signals have wildly different
     scales: activations may span [1e-3, 1e+2], saliencies span many
     decades, centralities are in [0, 1]. To make them comparable, we
     normalize each signal to [0, 1] *within* a normalization scope —
     either globally across all blocks, or per-stage (separately within
     each layer1/layer2/.../ group). Global normalization is the default
     used in all reported experiments; it allows blocks across stages
     to compete on the same scale.

  2. **Non-linear pre-transformation.** Before normalization, we apply
     sqrt() to activations and log() to saliencies. These compress
     outliers that would otherwise saturate the min-max mapping. The
     concrete transformations are baked into this module because they
     are mathematically inseparable from the importance metric — but the
     module also exposes a no-transform code path for ablations.

The result of `compute_block_scores()` is a dictionary `{block_id: score}`
ready to be passed to a top-k selector (see `domain.selection`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from ..constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    EPSILON_LOG,
    EPSILON_NORMALIZATION,
)
from .architecture import BlockGraph, BlockId

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases for clarity
# ─────────────────────────────────────────────────────────────────────────────

# Map from block identifier to a scalar value. The five different signal
# types (activation, saliency, three centralities) all share this shape.
SignalMap = Mapping[BlockId, float]

# Normalization scope: either treat the whole graph as one set ("global")
# or normalize each stage independently ("per_stage"). The latter is the
# default because it prevents inter-stage scale imbalances from biasing
# the ranking.
NormalizationScope = Literal["global", "per_stage"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImportanceWeights:
    """
    Coefficients of the linear combination in the importance formula.

    The five weights correspond to the five signals in Equation (1). They
    must sum to 1.0 so that the final score remains in a bounded range
    after each signal is min-max normalized to [0, 1].

    Defaults come from `constants.py` and reflect the choice motivated in
    the paper: 66% weight on data-dependent signals (alpha + beta),
    34% on graph-structural signals (gamma + delta + epsilon).

    Attributes
    ----------
    alpha   : weight for activation magnitude
    beta    : weight for saliency (|gradient * activation|)
    gamma   : weight for degree centrality
    delta   : weight for eigenvector centrality
    epsilon : weight for PageRank
    """

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    gamma: float = DEFAULT_GAMMA
    delta: float = DEFAULT_DELTA
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        # Allow generous tolerance for floating-point construction of
        # weights from external configs.
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Importance weights must sum to 1.0; got {total:.6f} "
                f"(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
                f"delta={self.delta}, epsilon={self.epsilon})"
            )
        for name, value in self.as_dict().items():
            if value < 0.0:
                raise ValueError(
                    f"Importance weight {name!r} must be non-negative, got {value}"
                )

    def as_dict(self) -> dict[str, float]:
        """Return the weights as a name-indexed dict."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "epsilon": self.epsilon,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _min_max_normalize(values: Mapping[BlockId, float]) -> dict[BlockId, float]:
    """
    Map values to [0, 1] by min-max normalization, treating constants as 0.

    For a vector x = (x_1, ..., x_n) with min m and max M, the result is
    (x_i - m) / (M - m). If M == m, all values are identical; we return
    zeros uniformly (rather than NaN from 0/0).

    The "constant -> 0" convention is consistent across all centrality
    algorithms in `domain.centrality` and ensures the combined formula
    is total: it never produces NaN regardless of input distribution.
    """
    if not values:
        return {}

    arr = np.asarray(list(values.values()), dtype=np.float64)
    min_value = float(arr.min())
    max_value = float(arr.max())
    spread = max_value - min_value

    if spread < EPSILON_NORMALIZATION:
        return {k: 0.0 for k in values}

    return {k: float((v - min_value) / spread) for k, v in values.items()}


def normalize_global(values: SignalMap) -> dict[BlockId, float]:
    """
    Normalize values to [0, 1] across the entire block set.
    
    This is the default normalization used in all reported experiments
    and the empirically validated choice for ResNet-50/CIFAR-100. 
    Allows blocks across different stages to compete on the same scale,
    so structurally important stages (e.g. layer3 in ResNet-50, which 
    has the highest saliency) receive more representatives in the top-k.
    """
    return _min_max_normalize(values)


def normalize_per_stage(
    values: SignalMap,
    graph: BlockGraph,
) -> dict[BlockId, float]:
    """
    Normalize values to [0, 1] independently within each stage.

    Alternative scope; NOT used in reported experiments. Empirically 
    measured to underperform global normalization on ResNet-50 / 
    CIFAR-100 r=0.5 m=2 by 5.6 p.p. top-1 (64.25 % vs 58.66 %), because 
    per-stage scaling forces selection of weak blocks in shallow stages 
    to satisfy min_keep_per_stage, at the expense of structurally 
    important blocks in deeper stages.

    Parameters
    ----------
    values
        Raw signal map; must contain all blocks of every stage in `graph`.
    graph
        Block graph providing the stage structure.

    Returns
    -------
    Map with the same keys, values in [0, 1].
    """
    result: dict[BlockId, float] = {}
    for stage in graph.stages:
        stage_values = {bid: values[bid] for bid in stage.block_ids if bid in values}
        normalized = _min_max_normalize(stage_values)
        result.update(normalized)
    return result


def _normalize(
    values: SignalMap,
    scope: NormalizationScope,
    graph: BlockGraph,
) -> dict[BlockId, float]:
    """Dispatch helper for the two normalization scopes."""
    if scope == "global":
        return normalize_global(values)
    if scope == "per_stage":
        return normalize_per_stage(values, graph)
    raise ValueError(
        f"Unknown normalization scope: {scope!r}. "
        f"Expected 'global' or 'per_stage'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-transformations
# ─────────────────────────────────────────────────────────────────────────────
#
# Raw activation magnitudes and saliencies span many orders of magnitude
# within a typical ResNet. Direct min-max normalization on such a long-
# tailed distribution would compress most blocks against 0 while the
# single largest entry sits at 1.
#
# To redistribute the dynamic range more uniformly, we apply non-linear
# pre-transformations before normalization:
#
#   - Activation:  sqrt(x)
#     Sqrt is a moderate compressor — turns a 100x ratio into 10x.
#     We use it for activations because the magnitudes are positive and
#     the underlying distribution is approximately log-normal.
#
#   - Saliency:    log(x + epsilon)
#     Saliencies span more decades than activations (they include both
#     gradient and activation magnitudes). log() compresses harder and
#     handles the rare exact-zero saliencies via the small offset.
#
# Both transformations are monotonic and preserve ranking. We could
# equivalently use rank-based normalization, but smoothness is preferable
# for downstream stability of the combined score.
# ─────────────────────────────────────────────────────────────────────────────

def transform_activation(value: float) -> float:
    """
    Compress activation magnitudes by taking the square root.

    Activations are non-negative by construction (they are absolute
    values of feature-map outputs). Sqrt of a non-negative real is
    well-defined; no offset is needed.
    """
    # Guard against negative inputs that would yield NaN. Such inputs
    # would indicate a bug upstream, but we clip to 0 for robustness.
    return float(np.sqrt(max(value, 0.0)))


def transform_saliency(value: float) -> float:
    """
    Compress saliencies by taking the natural logarithm with a small offset.

    The offset `EPSILON_LOG` (1e-8) handles exact-zero saliencies that
    occur when a block's gradient vanishes. The resulting value can be
    negative (log of a small number), which is fine — the subsequent
    min-max normalization handles any range.
    """
    return float(np.log(max(value, 0.0) + EPSILON_LOG))


# ─────────────────────────────────────────────────────────────────────────────
# Main importance computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScoreBreakdown:
    """
    Detailed breakdown of how each block's score was assembled.

    Useful for diagnostics, ablations, and rendering tables of the form
    "block X scored 0.73 because its saliency was very high (0.91)
    although its degree centrality was low (0.12)".

    All fields are post-normalization values in [0, 1] except `final`,
    which is the linear combination weighted by `ImportanceWeights`.

    Attributes
    ----------
    activation, saliency, degree, eigenvector, pagerank
        Normalized per-block signal values.
    final
        Combined score = alpha*A + beta*S + gamma*deg + delta*eig + eps*pr.
    weights
        The `ImportanceWeights` used to produce `final`; stored for
        reproducibility of downstream logs.
    """

    activation: dict[BlockId, float]
    saliency: dict[BlockId, float]
    degree: dict[BlockId, float]
    eigenvector: dict[BlockId, float]
    pagerank: dict[BlockId, float]
    final: dict[BlockId, float]
    weights: ImportanceWeights


def compute_block_scores(
    *,
    raw_activation: SignalMap,
    raw_saliency: SignalMap,
    degree_centrality: SignalMap,
    eigenvector_centrality: SignalMap,
    pagerank: SignalMap,
    graph: BlockGraph,
    weights: ImportanceWeights = ImportanceWeights(),
    normalization_scope: NormalizationScope = "global",
) -> ScoreBreakdown:
    """
    Compute the GDBA importance score (Equation (1)) for every block.

    Pipeline
    --------
    1. Pre-transform raw activations and saliencies (sqrt and log
       respectively) to compress their dynamic range.
    2. Normalize all five signals to [0, 1] under the chosen scope
       (`global` or `per_stage`).
    3. Take the linear combination with the supplied weights.

    The function is *pure*: it has no side effects, no random sampling,
    and depends only on its arguments. Given identical inputs, it returns
    identical outputs to within floating-point determinism.

    Parameters
    ----------
    raw_activation
        Per-block mean-absolute activation magnitude (output of a forward
        pass; supplied by the infrastructure layer).
    raw_saliency
        Per-block saliency, |gradient * activation| averaged over batch.
    degree_centrality
    eigenvector_centrality
    pagerank
        Per-block graph centralities (output of `domain.centrality`).
    graph
        Block topology, used by per-stage normalization.
    weights
        Coefficients of the linear combination; defaults are paper values.
    normalization_scope
        "global" or "per_stage". Where "global" used in all reported experiments

    Returns
    -------
    ScoreBreakdown
        All intermediate (normalized) signals plus the final combined
        score, suitable for downstream top-k selection or logging.

    Raises
    ------
    ValueError
        If the raw signal maps and graph blocks disagree on the set of
        block IDs.
    """
    # ── Sanity check: same blocks everywhere ─────────────────────────────
    block_set = set(graph.all_block_ids)
    for name, sig in [
        ("raw_activation", raw_activation),
        ("raw_saliency", raw_saliency),
        ("degree_centrality", degree_centrality),
        ("eigenvector_centrality", eigenvector_centrality),
        ("pagerank", pagerank),
    ]:
        missing = block_set - set(sig.keys())
        if missing:
            raise ValueError(
                f"Signal {name!r} is missing values for blocks: {sorted(missing)}"
            )

    # ── Step 1: pre-transform raw signals ────────────────────────────────
    transformed_act = {bid: transform_activation(v) for bid, v in raw_activation.items()}
    transformed_sal = {bid: transform_saliency(v) for bid, v in raw_saliency.items()}

    # ── Step 2: normalize all five signals ───────────────────────────────
    norm_act = _normalize(transformed_act, normalization_scope, graph)
    norm_sal = _normalize(transformed_sal, normalization_scope, graph)
    norm_deg = _normalize(degree_centrality, normalization_scope, graph)
    norm_eig = _normalize(eigenvector_centrality, normalization_scope, graph)
    norm_pr = _normalize(pagerank, normalization_scope, graph)

    # ── Step 3: weighted linear combination ──────────────────────────────
    final: dict[BlockId, float] = {}
    for block_id in graph.all_block_ids:
        final[block_id] = (
            weights.alpha * norm_act[block_id]
            + weights.beta * norm_sal[block_id]
            + weights.gamma * norm_deg[block_id]
            + weights.delta * norm_eig[block_id]
            + weights.epsilon * norm_pr[block_id]
        )

    return ScoreBreakdown(
        activation=norm_act,
        saliency=norm_sal,
        degree=norm_deg,
        eigenvector=norm_eig,
        pagerank=norm_pr,
        final=final,
        weights=weights,
    )
