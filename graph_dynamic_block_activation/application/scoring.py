"""
High-level scoring orchestration.

The score computation pipeline has four conceptual steps:

  1. Run a forward + backward pass with hooks installed to capture each
     residual block's output and gradient.

  2. Reduce the captured tensors to two scalars per block: activation
     magnitude (mean abs) and saliency (mean abs of output * grad).

  3. Compute the three graph centralities for the architecture's block
     graph. These do not depend on data and are normally cached.

  4. Combine the five signals via the importance formula (Equation (1)).

This module ties those four steps together behind a single function,
`compute_importance_scores()`. The function lives in the *application*
layer because it crosses the domain/infrastructure boundary — it uses
PyTorch (via `BlockActivationTracer`) and pure math (via `domain.*`).

By keeping the orchestration logic here rather than inside the domain
layer, the domain remains framework-free (and trivially unit-testable),
while infrastructure components remain single-purpose adapters.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..domain.architecture import BlockGraph, BlockId
from ..domain.centrality import (
    degree_centrality,
    eigenvector_centrality,
    pagerank,
)
from ..domain.importance import (
    ImportanceWeights,
    NormalizationScope,
    ScoreBreakdown,
    compute_block_scores,
)
from ..infrastructure.activation_tracer import BlockActivationTracer


# ─────────────────────────────────────────────────────────────────────────────
# Centrality cache
# ─────────────────────────────────────────────────────────────────────────────

class GraphCentralityCache:
    """
    Pre-computes and caches the three graph centralities for a BlockGraph.

    Centralities depend only on the adjacency matrix, not on data, so
    they are constant for the lifetime of an experiment. We compute them
    once at construction time and return cached dictionaries thereafter.

    Why a class rather than a module-level dict
    -------------------------------------------
    Multiple GDBA experiments may run in the same Python process with
    different graphs (e.g. ResNet-18 and ResNet-50 side by side). Each
    experiment instantiates its own cache, keyed by the BlockGraph
    identity. This avoids accidental cross-contamination.
    """

    def __init__(self, graph: BlockGraph) -> None:
        self._graph = graph

        # Compute once. These are NumPy arrays indexed by adjacency
        # matrix row; convert to {block_id: value} for the importance
        # formula's signature.
        deg = degree_centrality(graph.adjacency)
        eig = eigenvector_centrality(graph.adjacency)
        pr = pagerank(graph.adjacency)

        self._degree: dict[BlockId, float] = {
            graph.blocks[i].block_id: float(deg[i])
            for i in range(graph.num_blocks)
        }
        self._eigenvector: dict[BlockId, float] = {
            graph.blocks[i].block_id: float(eig[i])
            for i in range(graph.num_blocks)
        }
        self._pagerank: dict[BlockId, float] = {
            graph.blocks[i].block_id: float(pr[i])
            for i in range(graph.num_blocks)
        }

    @property
    def degree(self) -> dict[BlockId, float]:
        return self._degree

    @property
    def eigenvector(self) -> dict[BlockId, float]:
        return self._eigenvector

    @property
    def pagerank(self) -> dict[BlockId, float]:
        return self._pagerank


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions for the scoring backward pass
# ─────────────────────────────────────────────────────────────────────────────
#
# The saliency signal is computed from gradients of a target loss with
# respect to each block's output. We support two loss families:
#
#   - Entropy loss (default): -sum(p * log(p)) of the softmax output.
#     Use when ground-truth labels are unavailable at scoring time
#     (the realistic case for zero-shot inference).
#
#   - Cross-entropy loss: standard supervised loss. Use only when
#     ground-truth labels are available — typically only in ablation
#     studies, NOT for the deployed algorithm.

def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of the softmax distribution over the batch.

        L = E_n[-sum_c p(c|x_n) * log(p(c|x_n))]

    The minus sign makes this a non-negative quantity. Lower entropy
    means the model is more confident; higher entropy means it is more
    uncertain. As a scoring target, we use entropy directly because the
    *magnitude* of its gradient with respect to each block's output is
    exactly the saliency signal we want.
    """
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@torch.enable_grad()
def compute_importance_scores(
    *,
    model: nn.Module,
    graph: BlockGraph,
    centrality_cache: GraphCentralityCache,
    batch_x: torch.Tensor,
    batch_y: Optional[torch.Tensor] = None,
    weights: ImportanceWeights = ImportanceWeights(),
    normalization_scope: NormalizationScope = "global",
    loss_fn: Callable[[torch.Tensor], torch.Tensor] = entropy_loss,
) -> ScoreBreakdown:
    """
    Compute the GDBA importance score for every block on a given batch.

    Pipeline
    --------
    1. Install hooks on every block via BlockActivationTracer.
    2. Run forward pass to obtain logits.
    3. Compute the scoring loss (default: entropy of softmax outputs).
       If `batch_y` is provided AND `loss_fn` is `cross_entropy_loss`,
       supervised CE is used instead.
    4. Run backward to populate gradients of block outputs.
    5. Reduce captured tensors to per-block activation magnitudes and
       saliencies.
    6. Combine those with the cached centralities through the importance
       formula.

    The function uses `@torch.enable_grad()` AND explicitly disables
    `torch.inference_mode()` if the caller has it active. inference_mode
    is strictly stronger than no_grad — it marks tensors as having no
    autograd metadata at all, preventing `.backward()` even with
    enable_grad active. We must exit inference_mode here for the
    backward pass to succeed.

    Parameters
    ----------
    model
        The (possibly wrapped) ResNet model to score.
    graph
        Domain block graph for the model.
    centrality_cache
        Pre-computed graph centralities. Pass the same instance for the
        lifetime of an experiment.
    batch_x
        Input batch tensor, already on the model's device.
    batch_y
        Ground-truth labels (used only with supervised loss functions).
        Default `None` is correct for entropy-based scoring.
    weights
        Coefficients of the importance formula.
    normalization_scope
        "global" (default, used in all reported experiments) or "per_stage".
    loss_fn
        Callable taking logits and returning a scalar loss tensor.
        Default is `entropy_loss`; pass a custom function for ablations.

    Returns
    -------
    ScoreBreakdown with all five normalized signals plus the combined
    final score, ready to be passed to a TopKSelector.
    """
    # Explicitly exit any active inference_mode for the duration of this
    # function. Backward propagation requires autograd metadata that
    # inference_mode strips from new tensors, so even @torch.enable_grad()
    # is insufficient on its own when called from within an
    # inference_mode block.
    with torch.inference_mode(False):
        return _compute_importance_scores_impl(
            model=model,
            graph=graph,
            centrality_cache=centrality_cache,
            batch_x=batch_x,
            batch_y=batch_y,
            weights=weights,
            normalization_scope=normalization_scope,
            loss_fn=loss_fn,
        )


def _compute_importance_scores_impl(
    *,
    model: nn.Module,
    graph: BlockGraph,
    centrality_cache: GraphCentralityCache,
    batch_x: torch.Tensor,
    batch_y: Optional[torch.Tensor],
    weights: ImportanceWeights,
    normalization_scope: NormalizationScope,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> ScoreBreakdown:
    """Internal: actual scoring logic, called from inside a no-inference-mode block."""
    model.eval()

    # Step 1-4: gather activation and gradient signals.
    block_ids = list(graph.all_block_ids)

    # If the input batch was created under inference_mode, it lacks
    # autograd metadata. We clone it into a normal (non-inference) tensor
    # so the forward pass through the model produces grad-tracking
    # activations.
    if batch_x.is_inference():
        batch_x = batch_x.clone()

    # Reset any stale gradients from previous calls.
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None

    # Hooks must be attached to the BACKBONE (the underlying ResNet),
    # not to a GatedResNet wrapper. The block IDs in the graph
    # (e.g. "layer1.0") refer to module paths on the bare ResNet —
    # if we pass the wrapper, `named_modules()` yields paths like
    # "_backbone.layer1.0" which never match the block IDs.
    #
    # CRITICAL: the forward pass for scoring must also go through the
    # bare BACKBONE, not the wrapper. Otherwise scoring sees the model
    # WITH the previous refresh's gates applied, which corrupts the
    # activation/saliency signals (blocks that were gated off produce
    # identity outputs, so the scorer thinks they have "low" activation
    # when in reality those blocks are just being skipped). For correct
    # scoring, the forward pass must always go through the un-gated
    # backbone, regardless of which gates are currently set on the wrapper.
    hook_target = getattr(model, "backbone", model)

    with BlockActivationTracer(hook_target, block_ids) as tracer:
        logits = hook_target(batch_x)
        loss = loss_fn(logits)
        loss.backward()

        raw_activation = tracer.compute_activation_magnitudes()
        raw_saliency = tracer.compute_saliencies()

    # Step 5-6: combine via the importance formula.
    return compute_block_scores(
        raw_activation=raw_activation,
        raw_saliency=raw_saliency,
        degree_centrality=centrality_cache.degree,
        eigenvector_centrality=centrality_cache.eigenvector,
        pagerank=centrality_cache.pagerank,
        graph=graph,
        weights=weights,
        normalization_scope=normalization_scope,
    )
