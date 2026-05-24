"""
GDBA controller — top-level orchestrator for inference with adaptive gating.

The controller owns the full GDBA inference loop and exposes a simple
interface to the application that uses it:

    controller = GDBAController.build(backbone, top_k_ratio=0.5)
    logits = controller(batch_x)  # automatically scores and gates

Internally, the controller manages four pieces of state:

  1. The wrapped model (`GatedResNet`) that actually runs inference.
  2. A pre-built BlockGraph and centrality cache (computed once).
  3. A TopKSelector with the configured ratio and per-stage minimum.
  4. A refresh counter that triggers re-scoring every K batches.

Refresh semantics
-----------------
Scoring requires a forward + backward pass, which is roughly twice as
expensive as plain inference. Refreshing every K batches amortizes this
cost: on batches where no refresh is due, we run only the gated forward
pass, with no scoring overhead.

The first call to `forward()` always triggers a refresh, since no gates
have been set yet. After that, refreshes happen every `refresh_interval`
batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..constants import (
    DEFAULT_MIN_KEEP_PER_STAGE,
    DEFAULT_REFRESH_INTERVAL,
)
from ..domain.architecture import BlockGraph
from ..domain.importance import (
    ImportanceWeights,
    NormalizationScope,
    ScoreBreakdown,
)
from ..domain.selection import SelectionResult, TopKSelector
from ..infrastructure.architecture_inspector import inspect_resnet
from ..infrastructure.pytorch_wrapper import GatedResNet
from .scoring import GraphCentralityCache, compute_importance_scores


# ─────────────────────────────────────────────────────────────────────────────
# Per-call diagnostic record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GDBAStepResult:
    """
    Diagnostic snapshot of one inference call.

    Returned by `controller.step()` (the verbose entry point) so the
    caller can log scoring decisions, plot per-batch gate activity, or
    compute statistics across an epoch. The plain `forward()` method
    discards this information.

    Attributes
    ----------
    logits
        The model output tensor.
    refreshed
        True iff the scoring + selection pipeline ran on this call.
        False means the gates from a previous call were reused.
    score_breakdown
        Full importance breakdown if refreshed, else None.
    selection
        Selector output (gates + diagnostics) if refreshed, else None.
    """

    logits: torch.Tensor
    refreshed: bool
    score_breakdown: Optional[ScoreBreakdown]
    selection: Optional[SelectionResult]


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class GDBAController(nn.Module):
    """
    Top-level orchestrator for GDBA inference.

    Inherits from `nn.Module` so it can be moved to a device, used inside
    other modules, and integrated with PyTorch's `state_dict` machinery
    (though it adds no learnable parameters of its own).

    Construction
    ------------
    Use the `build()` classmethod for the common case:

        controller = GDBAController.build(
            backbone=model,
            top_k_ratio=0.5,
            min_keep_per_stage=2,
        )

    For full control (custom graph, custom weights), use the constructor
    directly.

    State managed
    -------------
    - `wrapper`     : the GatedResNet wrapping the user-supplied model.
    - `_selector`   : TopKSelector configured with the chosen ratio.
    - `_cache`      : GraphCentralityCache; computed once.
    - `_step_count` : monotonic counter; triggers refresh modulo K.

    Thread safety
    -------------
    Not thread-safe. The step counter and current gate state are mutable
    object attributes; concurrent calls from multiple threads would race.
    For multi-stream inference, instantiate one controller per stream.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        graph: BlockGraph,
        selector: TopKSelector,
        weights: ImportanceWeights = ImportanceWeights(),
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        normalization_scope: NormalizationScope = "global",
    ) -> None:
        super().__init__()

        if refresh_interval < 1:
            raise ValueError(
                f"refresh_interval must be >= 1, got {refresh_interval}"
            )

        # Wrap the backbone so we can apply gates during forward.
        self.wrapper = GatedResNet(backbone, graph)

        # Domain components.
        self._graph = graph
        self._selector = selector
        self._weights = weights
        self._normalization_scope = normalization_scope

        # Pre-compute centralities once.
        self._cache = GraphCentralityCache(graph)

        # Refresh scheduling.
        self._refresh_interval = int(refresh_interval)
        self._step_count = 0
        # Most recent selection result, for diagnostics. None until the
        # first refresh.
        self._last_selection: Optional[SelectionResult] = None

    # ── Convenience constructor ──────────────────────────────────────────

    @classmethod
    def build(
        cls,
        backbone: nn.Module,
        *,
        top_k_ratio: float,
        min_keep_per_stage: int = DEFAULT_MIN_KEEP_PER_STAGE,
        weights: Optional[ImportanceWeights] = None,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        normalization_scope: NormalizationScope = "global",
    ) -> "GDBAController":
        """
        Build a controller with sensible defaults.

        Parameters
        ----------
        backbone
            ResNet model (already loaded with weights) to wrap.
        top_k_ratio
            Fraction of non-entry blocks to keep active.
        min_keep_per_stage
            Per-stage minimum (counting entry block toward the total).
        weights
            Importance formula coefficients. Default = paper values.
        refresh_interval
            Re-score gates every K batches.
        normalization_scope
            "global" (default) or "per_stage".
        """
        graph = inspect_resnet(backbone)
        selector = TopKSelector(
            top_k_ratio=top_k_ratio,
            min_keep_per_stage=min_keep_per_stage,
        )
        return cls(
            backbone=backbone,
            graph=graph,
            selector=selector,
            weights=weights or ImportanceWeights(),
            refresh_interval=refresh_interval,
            normalization_scope=normalization_scope,
        )

    # ── Read-only state inspection ───────────────────────────────────────

    @property
    def graph(self) -> BlockGraph:
        return self._graph

    @property
    def step_count(self) -> int:
        """Number of forward calls since construction (or last reset)."""
        return self._step_count

    @property
    def last_selection(self) -> Optional[SelectionResult]:
        """Selection result from the most recent refresh, or None."""
        return self._last_selection

    @property
    def refresh_interval(self) -> int:
        return self._refresh_interval

    # ── State management ─────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset the step counter and clear active gates.

        Useful between distinct evaluation passes (e.g. baseline vs
        gated) when you want guaranteed reproducibility regardless of
        the call history.
        """
        self._step_count = 0
        self._last_selection = None
        self.wrapper.clear_gates()

    # ── Forward pass ─────────────────────────────────────────────────────

    def step(
        self,
        batch_x: torch.Tensor,
        batch_y: Optional[torch.Tensor] = None,
    ) -> GDBAStepResult:
        """
        Run one inference step with full diagnostics.

        Behavior:
          - On step 0 and every `refresh_interval`-th step thereafter,
            re-score blocks and update the gates.
          - On other steps, reuse the gates from the most recent refresh.

        Always runs the wrapped forward pass.

        Parameters
        ----------
        batch_x
            Input tensor on the model's device.
        batch_y
            Ignored for entropy-based scoring. Reserved
            for ablation studies that use supervised loss.

        Returns
        -------
        GDBAStepResult with logits and (if a refresh happened) the score
        breakdown and selection result.
        """
        # Decide whether to refresh.
        should_refresh = (self._step_count % self._refresh_interval) == 0

        breakdown: Optional[ScoreBreakdown] = None
        selection: Optional[SelectionResult] = None

        if should_refresh:
            breakdown = compute_importance_scores(
                model=self.wrapper,
                graph=self._graph,
                centrality_cache=self._cache,
                batch_x=batch_x,
                batch_y=batch_y,
                weights=self._weights,
                normalization_scope=self._normalization_scope,
            )
            selection = self._selector.select(breakdown.final, self._graph)
            self.wrapper.set_gates(selection.gates)
            self._last_selection = selection

        # Always run the (possibly gated) forward pass for the final
        # logits returned to the caller.
        with torch.inference_mode():
            logits = self.wrapper(batch_x)

        self._step_count += 1

        return GDBAStepResult(
            logits=logits,
            refreshed=should_refresh,
            score_breakdown=breakdown,
            selection=selection,
        )

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        """
        Plain inference call returning only logits.

        Equivalent to `step(batch_x).logits` but allows the controller
        to be used as a drop-in replacement for the wrapped model in
        existing code:

            controller = GDBAController.build(model, top_k_ratio=0.5)
            for batch_x, _ in loader:
                logits = controller(batch_x)   # like model(batch_x)
        """
        return self.step(batch_x).logits
