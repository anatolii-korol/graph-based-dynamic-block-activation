"""
Pure-domain top-k block selector.

Given a `ScoreBreakdown` from `domain.importance`, this module selects
which blocks should remain *active* during inference. The selection
respects two architectural constraints from the paper:

  1. **Entry blocks are forced active.** The first block of each stage
     (`layer1.0`, `layer2.0`, ...) performs spatial down-sampling and
     channel expansion. Disabling it breaks the shape contract between
     stages, so it never participates in the top-k decision.

  2. **Minimum k blocks per stage.** Each stage must keep at least `m`
     blocks active *in total* (counting both entry and non-entry blocks),
     regardless of the global top-k ratio. The entry block, being always
     active, contributes 1 to this count. This prevents the algorithm
     from "vacating" an entire stage at low top-k ratios.

     Concretely: with `min_keep_per_stage = 2`, every stage must have
     at least 2 active blocks. Since the entry block is always one of
     them, this means at least 1 *non-entry* block must also be active
     in each stage. This semantic matches the experimental measurements
     reported in the paper.

The selector is *deterministic* given the same scores and graph: pure
top-k with ceiling rounding, no random sampling, no EMA, no stochastic
round-up.

Why ceiling rounding
--------------------
When the requested fraction r yields a non-integer number of blocks
(e.g. 0.5 * 12 = 6.0, but 0.7 * 12 = 8.4), we have to round to an
integer. Three obvious choices:

  - floor:  always round down (k = 8 for r=0.7) - more aggressive
  - round:  nearest integer (k = 8 for r=0.7)   - inconsistent
  - ceil:   always round up   (k = 9 for r=0.7) - more conservative

The paper uses **ceil** because it preserves the property "at most r
fraction of blocks pruned" rather than "exactly r fraction". This
guarantees that the user-supplied bound is never exceeded on the
permissive side, which matters when r is an upper-bound budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..constants import DEFAULT_MIN_KEEP_PER_STAGE
from .architecture import BlockGraph, BlockId

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# A gate map: for each block ID, whether it is active (1) or gated off (0).
# We use a plain dict of ints rather than a Tensor to keep the domain
# layer framework-agnostic. The infrastructure layer converts this to a
# PyTorch buffer when wiring up the wrapper.
GateMap = dict[BlockId, int]

# A per-block score map, typically the `final` field of a `ScoreBreakdown`.
ScoreMap = Mapping[BlockId, float]


# ─────────────────────────────────────────────────────────────────────────────
# Selection result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelectionResult:
    """
    Outcome of a top-k selection pass over the non-entry blocks.

    Attributes
    ----------
    gates
        Per-block binary mask: 1 = active, 0 = gated off. Entry blocks
        are always 1; non-entry blocks may be 0.
    requested_k
        How many non-entry blocks the user *asked* to keep (after applying
        the ratio and the ceiling rounding).
    effective_k
        How many non-entry blocks are actually kept after applying the
        per-stage minimum. Always >= requested_k.
    forced_active_count
        Number of blocks promoted from "gated off" to "active" because
        of the per-stage minimum constraint. Equal to effective_k - actual
        top-k size where this would have been below the minimum. Useful
        for diagnostics: a high value means the per-stage constraint is
        biting harder than the global ratio.
    """

    gates: GateMap
    requested_k: int
    effective_k: int
    forced_active_count: int

    @property
    def num_active_blocks(self) -> int:
        """Total number of blocks left active (including entry blocks)."""
        return sum(self.gates.values())

    @property
    def num_gated_blocks(self) -> int:
        """Number of blocks gated off."""
        return len(self.gates) - self.num_active_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _k_from_ratio(num_eligible: int, ratio: float) -> int:
    """
    Convert a fractional ratio into an integer top-k count, ceiling round.

    Examples
    --------
        ratio=1.0, num=12  ->  12
        ratio=0.7, num=12  ->  ceil(8.4) = 9
        ratio=0.5, num=12  ->  6
        ratio=0.1, num=12  ->  ceil(1.2) = 2
        ratio=0.0, num=12  ->  0
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Ratio must be in [0, 1]; got {ratio}")
    if num_eligible < 0:
        raise ValueError(f"num_eligible must be non-negative; got {num_eligible}")
    k = math.ceil(ratio * num_eligible)
    # Clamp into the valid range. ceil(0 * n) = 0, ceil(1 * n) = n, both
    # of which are already in [0, n], so this is defensive only.
    return max(0, min(num_eligible, k))


def _rank_blocks_descending(scores: ScoreMap) -> list[BlockId]:
    """
    Return block IDs sorted by score, highest first.

    Ties are broken by the natural ordering of block IDs (alphabetical),
    which makes the function fully deterministic. Python's sort is
    stable, so we can chain the tie-break key into the sort key.
    """
    return sorted(scores.keys(), key=lambda bid: (-scores[bid], bid))


# ─────────────────────────────────────────────────────────────────────────────
# The selector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TopKSelector:
    """
    Deterministic top-k selector with architectural safety constraints.

    Parameters
    ----------
    top_k_ratio
        Fraction of *non-entry* blocks to keep, in [0, 1]. The number of
        kept blocks is `ceil(ratio * num_non_entry_blocks)`.
    min_keep_per_stage
        Minimum number of non-entry blocks that must remain active in
        each stage, regardless of the global ratio. See
        `constants.DEFAULT_MIN_KEEP_PER_STAGE`.

    Methods
    -------
    select(scores, graph)
        Compute the gate map and selection diagnostics.

    Why a class rather than a function
    -----------------------------------
    The selector holds policy parameters (`top_k_ratio`, `min_keep_per_stage`)
    that are fixed for the duration of an experiment. Wrapping them in a
    class makes the policy injectable into the controller and lets us
    swap selectors without changing the controller code (Strategy pattern).
    """

    top_k_ratio: float
    min_keep_per_stage: int = DEFAULT_MIN_KEEP_PER_STAGE

    def __post_init__(self) -> None:
        if not (0.0 <= self.top_k_ratio <= 1.0):
            raise ValueError(
                f"top_k_ratio must be in [0, 1]; got {self.top_k_ratio}"
            )
        if self.min_keep_per_stage < 0:
            raise ValueError(
                f"min_keep_per_stage must be non-negative; got {self.min_keep_per_stage}"
            )

    # ── Main entry point ─────────────────────────────────────────────────

    def select(self, scores: ScoreMap, graph: BlockGraph) -> SelectionResult:
        """
        Compute the gate map for the given scores and graph.

        Algorithm
        ---------
        1. Build the eligible block set: all non-entry blocks. Entry blocks
           are unconditionally active.

        2. Rank eligible blocks by score (descending).

        3. Pick the top `k = ceil(ratio * num_eligible)` ranked blocks
           as the initial active set.

        4. Apply the per-stage minimum: for any stage where fewer than
           `min_keep_per_stage` *total* blocks (entry + non-entry) are
           active, promote additional non-entry blocks (in score order
           within that stage) until the minimum is met. The entry block
           counts as 1 toward the minimum, since it is always active.

        5. Emit gate values 1 for every active block (entry + selected
           + force-promoted), 0 for the rest.

        Parameters
        ----------
        scores
            Final importance scores from `compute_block_scores`. Must
            contain at least all eligible (non-entry) blocks; entry blocks
            may be present but are ignored.
        graph
            Block graph providing stage structure and the eligible set.

        Returns
        -------
        SelectionResult with the gate map and bookkeeping fields.
        """
        eligible_ids = list(graph.non_entry_block_ids)
        entry_ids = set(graph.entry_block_ids)
        num_eligible = len(eligible_ids)

        # ── Step 1: validate inputs ───────────────────────────────────────
        missing = set(eligible_ids) - set(scores.keys())
        if missing:
            raise ValueError(
                f"Scores missing for non-entry blocks: {sorted(missing)}"
            )

        # ── Step 2: compute the requested k ──────────────────────────────
        requested_k = _k_from_ratio(num_eligible, self.top_k_ratio)

        # ── Step 3: rank eligible blocks and pick initial top-k ──────────
        eligible_scores = {bid: scores[bid] for bid in eligible_ids}
        ranked = _rank_blocks_descending(eligible_scores)
        active_set: set[BlockId] = set(ranked[:requested_k])

        # ── Step 4: enforce per-stage minimum ─────────────────────────────
        # The original paper's measurements were produced with a semantic
        # in which the entry block COUNTS toward the per-stage minimum.
        # We preserve that semantic here so reported numbers reproduce
        # bit-identically.
        forced_active_count = 0
        if self.min_keep_per_stage > 0:
            for stage in graph.stages:
                # All blocks of this stage (entry + non-entry).
                all_stage_blocks = list(stage.block_ids)
                # Currently active in this stage = entry (always) + any
                # non-entry that made it into the top-k.
                currently_active_in_stage = sum(
                    1 for bid in all_stage_blocks
                    if bid in entry_ids or bid in active_set
                )
                # If the stage falls short of the minimum, promote
                # additional non-entry blocks (since entry is already on).
                stage_capacity = len(all_stage_blocks)
                target = min(self.min_keep_per_stage, stage_capacity)
                shortfall = max(0, target - currently_active_in_stage)
                if shortfall > 0:
                    # Rank non-entry blocks of this stage by score, descending.
                    stage_non_entry = [
                        bid for bid in stage.block_ids if bid not in entry_ids
                    ]
                    stage_ranked = _rank_blocks_descending(
                        {bid: scores[bid] for bid in stage_non_entry}
                    )
                    promoted = 0
                    for bid in stage_ranked:
                        if bid not in active_set:
                            active_set.add(bid)
                            promoted += 1
                            forced_active_count += 1
                            if promoted >= shortfall:
                                break

        effective_k = len(active_set)

        # ── Step 5: emit the full gate map (entry + active + gated) ──────
        gates: GateMap = {}
        for bid in graph.all_block_ids:
            if bid in entry_ids or bid in active_set:
                gates[bid] = 1
            else:
                gates[bid] = 0

        return SelectionResult(
            gates=gates,
            requested_k=requested_k,
            effective_k=effective_k,
            forced_active_count=forced_active_count,
        )
