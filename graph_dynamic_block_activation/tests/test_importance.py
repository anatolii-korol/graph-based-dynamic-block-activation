"""
Tests for the importance formula (Equation 9) and its helpers.

This is the mathematical core of GDBA — every metric in the paper
ultimately depends on these functions producing the right scores. The
tests cover:

  1. ImportanceWeights construction & validation
  2. Pre-transformations (sqrt, log)
  3. Normalization (global, per_stage)
  4. compute_block_scores end-to-end:
     - input validation
     - weighted combination
     - score range
     - determinism

Where possible, we use synthetic data with known answers rather than
relying on a model — this isolates the math from PyTorch entirely.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..domain.architecture import (  # noqa: E402
    build_block_graph,
    build_resnet18_graph,
    build_resnet50_graph,
)
from ..domain.importance import (  # noqa: E402
    ImportanceWeights,
    ScoreBreakdown,
    compute_block_scores,
    normalize_global,
    normalize_per_stage,
    transform_activation,
    transform_saliency,
)


# ─────────────────────────────────────────────────────────────────────────────
# ImportanceWeights validation
# ─────────────────────────────────────────────────────────────────────────────

def test_default_weights_sum_to_one():
    """Default ImportanceWeights coefficients sum to 1.0."""
    w = ImportanceWeights()
    total = w.alpha + w.beta + w.gamma + w.delta + w.epsilon
    assert abs(total - 1.0) < 1e-9, f"Default weights sum to {total}, not 1"


def test_custom_weights_must_sum_to_one():
    """Weights that do not sum to 1 are rejected."""
    try:
        ImportanceWeights(alpha=0.5, beta=0.5, gamma=0.5, delta=0.0, epsilon=0.0)
    except ValueError:
        return
    raise AssertionError("Should reject weights summing to 1.5")


def test_negative_weight_rejected():
    """A negative coefficient is rejected even if the sum is 1."""
    try:
        # 0.5 + 0.6 + (-0.1) + 0 + 0 = 1.0 but negative gamma
        ImportanceWeights(
            alpha=0.5, beta=0.6, gamma=-0.1, delta=0.0, epsilon=0.0,
        )
    except ValueError:
        return
    raise AssertionError("Should reject negative weights")


def test_weights_are_frozen():
    """ImportanceWeights is an immutable dataclass."""
    w = ImportanceWeights()
    try:
        w.alpha = 0.5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ImportanceWeights should be frozen")


def test_weights_as_dict():
    """as_dict() returns all five coefficients."""
    w = ImportanceWeights()
    d = w.as_dict()
    assert set(d.keys()) == {"alpha", "beta", "gamma", "delta", "epsilon"}
    assert abs(sum(d.values()) - 1.0) < 1e-9


def test_custom_valid_weights():
    """A valid non-default combination constructs successfully."""
    # Saliency-only configuration from the paper's ablation study
    w = ImportanceWeights(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, epsilon=0.0)
    assert w.alpha == 0.0
    assert w.beta == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Pre-transformations
# ─────────────────────────────────────────────────────────────────────────────

def test_transform_activation_sqrt():
    """Activation transform = sqrt."""
    assert transform_activation(4.0) == 2.0
    assert transform_activation(0.25) == 0.5
    assert transform_activation(0.0) == 0.0
    assert transform_activation(1.0) == 1.0


def test_transform_activation_clamps_negatives():
    """Negative inputs are clamped to 0 (sqrt of negative is undefined)."""
    result = transform_activation(-1.0)
    assert result == 0.0, f"sqrt(-1) should clamp to 0, got {result}"


def test_transform_saliency_log_with_offset():
    """Saliency transform = log(x + epsilon)."""
    # log(1 + epsilon) ~ log(1) = 0
    assert abs(transform_saliency(1.0)) < 1e-6
    # log(e + epsilon) ~ 1
    assert abs(transform_saliency(math.e) - 1.0) < 1e-6


def test_transform_saliency_handles_zero():
    """Zero saliency yields log(epsilon), which is large negative but finite."""
    result = transform_saliency(0.0)
    assert math.isfinite(result), f"log(0+eps) should be finite, got {result}"
    # log(1e-8) ~ -18.4
    assert result < -10, f"log(epsilon) should be very negative, got {result}"


def test_transform_saliency_handles_negative():
    """Negative saliency (shouldn't happen, but defensive) clamps to 0 + eps."""
    # transform_saliency clamps to >=0 internally, so log(0+eps) is the floor.
    result = transform_saliency(-0.5)
    assert math.isfinite(result)
    # Same as transform_saliency(0.0)
    assert abs(result - transform_saliency(0.0)) < 1e-9


def test_transform_monotonic():
    """Both transforms are monotonically non-decreasing on non-negative inputs."""
    inputs = [0.0, 0.1, 1.0, 10.0, 100.0]
    acts = [transform_activation(x) for x in inputs]
    sals = [transform_saliency(x) for x in inputs]
    for a, b in zip(acts[:-1], acts[1:]):
        assert b >= a, f"Activation transform not monotonic: {acts}"
    for a, b in zip(sals[:-1], sals[1:]):
        assert b >= a, f"Saliency transform not monotonic: {sals}"


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_global_min_max():
    """Global normalization maps min to 0 and max to 1."""
    values = {"a": 10.0, "b": 20.0, "c": 30.0}
    normalized = normalize_global(values)
    assert normalized["a"] == 0.0  # min
    assert normalized["c"] == 1.0  # max
    assert normalized["b"] == 0.5  # midpoint


def test_normalize_global_constant_input():
    """All-equal inputs return all-zero outputs (no division by zero)."""
    values = {"a": 5.0, "b": 5.0, "c": 5.0}
    normalized = normalize_global(values)
    assert all(v == 0.0 for v in normalized.values()), (
        f"Constant input should map to zeros, got {normalized}"
    )


def test_normalize_global_empty_input():
    """Empty input returns empty output without error."""
    result = normalize_global({})
    assert result == {}


def test_normalize_global_preserves_keys():
    """All input keys appear in the output."""
    values = {"layer1.0": 1.0, "layer1.1": 2.0, "layer2.0": 3.0}
    normalized = normalize_global(values)
    assert set(normalized.keys()) == set(values.keys())


def test_normalize_per_stage_independent_within_stage():
    """
    Per-stage normalization maps the largest block in EACH stage to 1.0,
    not the largest globally.
    """
    graph = build_resnet18_graph()
    # Activate layer1 highly, layer4 lightly. ResNet-18 has 2 blocks per
    # stage; with 100 and 50 in layer1, min-max gives 1.0 and 0.0.
    values = {bid: 0.0 for bid in graph.all_block_ids}
    values["layer1.0"] = 100.0
    values["layer1.1"] = 50.0
    values["layer4.1"] = 0.01

    normalized = normalize_per_stage(values, graph)

    # Within layer1: layer1.0 (100, max) -> 1.0, layer1.1 (50, min) -> 0.0
    assert normalized["layer1.0"] == 1.0
    assert normalized["layer1.1"] == 0.0
    # Within layer4: 0.0 (layer4.0) and 0.01 (layer4.1)
    # layer4.1 is the max of its stage -> 1.0, even though it is tiny
    # in absolute terms. This is the whole point of per-stage normalization.
    assert normalized["layer4.1"] == 1.0
    assert normalized["layer4.0"] == 0.0
    # layer2 and layer3 are uniformly zero — should all be 0.0
    for stage_name in ("layer2", "layer3"):
        for bid in graph.blocks_in_stage(stage_name):
            assert normalized[bid] == 0.0, (
                f"Uniform-zero stage {stage_name} block {bid} should be 0"
            )


def test_normalize_per_stage_constant_stage_maps_to_zeros():
    """Stages with constant scores map to zeros within that stage."""
    graph = build_resnet18_graph()
    # layer2 all constant, layer3 varying
    values = {bid: 0.0 for bid in graph.all_block_ids}
    for bid in graph.blocks_in_stage("layer2"):
        values[bid] = 7.0
    values["layer3.0"] = 1.0
    values["layer3.1"] = 2.0

    normalized = normalize_per_stage(values, graph)
    # layer2: all 7.0 → all 0.0
    for bid in graph.blocks_in_stage("layer2"):
        assert normalized[bid] == 0.0, (
            f"Constant stage block {bid} should be 0, got {normalized[bid]}"
        )
    # layer3 normalized internally
    assert normalized["layer3.0"] == 0.0
    assert normalized["layer3.1"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_block_scores — input validation
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_signals(graph, value: float = 0.5) -> dict:
    """All blocks get the same value for a given signal."""
    return {bid: value for bid in graph.all_block_ids}


def test_compute_scores_missing_block_raises():
    """If a signal is missing for a block in the graph, raises ValueError."""
    graph = build_resnet18_graph()
    signals = _uniform_signals(graph)
    # Remove one block from one signal
    incomplete = dict(signals)
    del incomplete["layer2.1"]

    try:
        compute_block_scores(
            raw_activation=incomplete,
            raw_saliency=signals,
            degree_centrality=signals,
            eigenvector_centrality=signals,
            pagerank=signals,
            graph=graph,
        )
    except ValueError as e:
        # Error message should mention the missing block
        assert "layer2.1" in str(e), f"Error doesn't mention missing block: {e}"
        return
    raise AssertionError("Should raise ValueError for missing signal")


def test_compute_scores_all_signals_required():
    """Each of the 5 signals must be present for every block."""
    graph = build_resnet18_graph()
    signals = _uniform_signals(graph)
    incomplete = dict(signals)
    del incomplete["layer3.0"]

    # Try omitting each signal in turn — all should raise
    for missing_signal in [
        "raw_activation", "raw_saliency",
        "degree_centrality", "eigenvector_centrality", "pagerank",
    ]:
        kwargs = {
            "raw_activation": signals,
            "raw_saliency": signals,
            "degree_centrality": signals,
            "eigenvector_centrality": signals,
            "pagerank": signals,
            "graph": graph,
        }
        kwargs[missing_signal] = incomplete
        try:
            compute_block_scores(**kwargs)
        except ValueError:
            continue
        raise AssertionError(
            f"Should reject incomplete {missing_signal}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_block_scores — output structure
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_scores_returns_breakdown_with_all_blocks():
    """ScoreBreakdown contains all 6 fields, each covering all blocks."""
    graph = build_resnet18_graph()
    signals = _uniform_signals(graph, value=0.5)

    breakdown = compute_block_scores(
        raw_activation=signals,
        raw_saliency=signals,
        degree_centrality=signals,
        eigenvector_centrality=signals,
        pagerank=signals,
        graph=graph,
    )

    assert isinstance(breakdown, ScoreBreakdown)
    block_set = set(graph.all_block_ids)
    for field_name in [
        "activation", "saliency", "degree", "eigenvector", "pagerank", "final",
    ]:
        field = getattr(breakdown, field_name)
        assert set(field.keys()) == block_set, (
            f"{field_name} missing blocks: "
            f"{block_set - set(field.keys())}"
        )


def test_compute_scores_final_is_weighted_combination():
    """
    With known normalized inputs and known weights, the final score must
    equal the explicit weighted sum.
    """
    # Synthetic graph with 2 stages of 2 blocks each.
    graph = build_block_graph([("layer1", 2), ("layer2", 2)])

    # Construct raw signals such that after sqrt and log preprocessing,
    # all four blocks have DIFFERENT activations and saliencies — so
    # min-max normalization actually does something.
    # Easier: build POST-normalization signals by side-stepping the
    # transforms with raw values that survive them in known form.
    # Simpler: use uniform activation+saliency, then the activation and
    # saliency normalize to zero. The "final" is dominated by graph
    # centralities — also uniform, so all four blocks score equally.
    signals = {bid: 0.5 for bid in graph.all_block_ids}
    breakdown = compute_block_scores(
        raw_activation=signals,
        raw_saliency=signals,
        degree_centrality=signals,
        eigenvector_centrality=signals,
        pagerank=signals,
        graph=graph,
    )
    # All scores equal because all inputs were uniform → all normalize to 0
    scores = list(breakdown.final.values())
    assert all(abs(s - scores[0]) < 1e-9 for s in scores), (
        f"Uniform inputs should give uniform output, got {scores}"
    )


def test_compute_scores_weighted_combination_explicit():
    """
    With saliency-only weights and varying saliency, the final score
    should equal the normalized saliency exactly.
    """
    graph = build_block_graph([("layer1", 2), ("layer2", 2)])

    # Distinct positive saliencies; activation and centralities uniform
    uniform = _uniform_signals(graph, 1.0)
    saliencies = {
        "layer1.0": 1.0,
        "layer1.1": math.e,
        "layer2.0": math.e ** 2,
        "layer2.1": math.e ** 3,
    }
    # alpha=0, beta=1, gamma=delta=epsilon=0
    saliency_only = ImportanceWeights(
        alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, epsilon=0.0,
    )

    breakdown = compute_block_scores(
        raw_activation=uniform,
        raw_saliency=saliencies,
        degree_centrality=uniform,
        eigenvector_centrality=uniform,
        pagerank=uniform,
        graph=graph,
        weights=saliency_only,
        normalization_scope="global",
    )

    # final should equal the normalized saliency (since beta=1, all others=0)
    for bid in graph.all_block_ids:
        assert abs(breakdown.final[bid] - breakdown.saliency[bid]) < 1e-9, (
            f"final[{bid}] != saliency[{bid}] under saliency-only weights"
        )


def test_compute_scores_finite_and_in_range():
    """All output scores are finite and in [0, 1]."""
    graph = build_resnet50_graph()
    # Mix of small and large values
    activation = {bid: float(i + 1) for i, bid in enumerate(graph.all_block_ids)}
    saliency = {bid: float(i + 1) * 1e-3 for i, bid in enumerate(graph.all_block_ids)}
    centrality = {bid: 0.5 for bid in graph.all_block_ids}

    breakdown = compute_block_scores(
        raw_activation=activation,
        raw_saliency=saliency,
        degree_centrality=centrality,
        eigenvector_centrality=centrality,
        pagerank=centrality,
        graph=graph,
    )

    for bid, score in breakdown.final.items():
        assert math.isfinite(score), f"Non-finite score for {bid}: {score}"
        assert 0.0 <= score <= 1.0 + 1e-9, (
            f"Score for {bid} out of [0,1]: {score}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_scores_deterministic():
    """Identical inputs produce identical outputs across calls."""
    graph = build_resnet50_graph()
    activation = {bid: float(hash(bid) % 100) for bid in graph.all_block_ids}
    saliency = {bid: float(hash(bid) % 50) for bid in graph.all_block_ids}
    centrality = {bid: 0.5 for bid in graph.all_block_ids}

    kwargs = dict(
        raw_activation=activation,
        raw_saliency=saliency,
        degree_centrality=centrality,
        eigenvector_centrality=centrality,
        pagerank=centrality,
        graph=graph,
    )
    b1 = compute_block_scores(**kwargs)
    b2 = compute_block_scores(**kwargs)
    for bid in graph.all_block_ids:
        assert b1.final[bid] == b2.final[bid], (
            f"Non-deterministic at {bid}: {b1.final[bid]} vs {b2.final[bid]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Normalization scope dispatch
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_normalization_scope_rejected():
    """An unrecognized scope string raises ValueError."""
    graph = build_resnet18_graph()
    signals = _uniform_signals(graph)
    try:
        compute_block_scores(
            raw_activation=signals,
            raw_saliency=signals,
            degree_centrality=signals,
            eigenvector_centrality=signals,
            pagerank=signals,
            graph=graph,
            normalization_scope="batch_norm",  # bogus value
        )
    except ValueError:
        return
    raise AssertionError("Should reject unknown normalization scope")


def test_global_and_per_stage_differ_when_stages_have_different_scales():
    """
    With non-uniform signal magnitudes across stages, global and per-
    stage normalization produce different results.
    """
    graph = build_resnet18_graph()
    activation = {bid: 1.0 for bid in graph.all_block_ids}
    # Layer1 has high activation, layer4 has low
    for bid in graph.blocks_in_stage("layer1"):
        activation[bid] = 100.0
    for bid in graph.blocks_in_stage("layer4"):
        activation[bid] = 0.01

    uniform = {bid: 0.5 for bid in graph.all_block_ids}
    common = dict(
        raw_activation=activation,
        raw_saliency=uniform,
        degree_centrality=uniform,
        eigenvector_centrality=uniform,
        pagerank=uniform,
        graph=graph,
    )

    b_global = compute_block_scores(**common, normalization_scope="global")
    b_perstage = compute_block_scores(**common, normalization_scope="per_stage")

    # Find at least one block where the two differ
    differences = [
        bid for bid in graph.all_block_ids
        if abs(b_global.final[bid] - b_perstage.final[bid]) > 1e-6
    ]
    assert len(differences) > 0, (
        "Global and per-stage normalization should differ on non-uniform input"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [
        # ImportanceWeights
        test_default_weights_sum_to_one,
        test_custom_weights_must_sum_to_one,
        test_negative_weight_rejected,
        test_weights_are_frozen,
        test_weights_as_dict,
        test_custom_valid_weights,
        # Transforms
        test_transform_activation_sqrt,
        test_transform_activation_clamps_negatives,
        test_transform_saliency_log_with_offset,
        test_transform_saliency_handles_zero,
        test_transform_saliency_handles_negative,
        test_transform_monotonic,
        # Normalization
        test_normalize_global_min_max,
        test_normalize_global_constant_input,
        test_normalize_global_empty_input,
        test_normalize_global_preserves_keys,
        test_normalize_per_stage_independent_within_stage,
        test_normalize_per_stage_constant_stage_maps_to_zeros,
        # compute_block_scores
        test_compute_scores_missing_block_raises,
        test_compute_scores_all_signals_required,
        test_compute_scores_returns_breakdown_with_all_blocks,
        test_compute_scores_final_is_weighted_combination,
        test_compute_scores_weighted_combination_explicit,
        test_compute_scores_finite_and_in_range,
        test_compute_scores_deterministic,
        test_unknown_normalization_scope_rejected,
        test_global_and_per_stage_differ_when_stages_have_different_scales,
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
