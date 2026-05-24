"""
Pure-domain description of ResNet block topology.

This module captures *what* a ResNet looks like as a directed graph of
residual blocks, without depending on any specific deep-learning framework.
A concrete `nn.Module` is converted into a `BlockGraph` once, by an
infrastructure adapter (see `infrastructure.pytorch_inspector`); from that
point on, the GDBA algorithm works exclusively with the framework-agnostic
representation.

Design rationale
----------------
Separating the topology description from the live model has three benefits:

  1. *Testability* — a `BlockGraph` for ResNet-50 can be built by hand or
     from a YAML file, allowing unit tests to exercise scoring and
     selection logic without instantiating a 25-million-parameter model.

  2. *Architecture independence* — adding support for MobileNet,
     EfficientNet, or a custom architecture means writing a new inspector,
     not modifying the algorithm. The dependency direction is correct:
     algorithm depends on `BlockGraph`, inspectors depend on `BlockGraph`.

  3. *Reviewability* — the graph used by the algorithm becomes an
     explicit, inspectable object that can be serialized, logged, and
     diffed across experiments.

Domain vocabulary
-----------------
- *Block*: A residual unit, e.g. one `BasicBlock` or `Bottleneck`.
  Identified by a string name like "layer1.0".

- *Stage*: A contiguous group of blocks that share the same spatial
  resolution. Standard ResNet has four stages (layer1...layer4).

- *Entry block*: The first block of a stage. It always performs spatial
  down-sampling (stride > 1 or via a 1x1 down-sample conv) and changes
  the number of channels, so it cannot be replaced by an identity skip.
  In GDBA, entry blocks are *forced active* regardless of their score.

- *Block graph*: A directed weighted graph over the set of blocks. The
  weight of edge (u, v) encodes how strongly information flows from
  block u to block v (sequential = 1.0, skip = 0.25).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from ..constants import (
    SEQUENTIAL_EDGE_WEIGHT,
    SKIP_EDGE_WEIGHT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Block and stage descriptors
# ─────────────────────────────────────────────────────────────────────────────

# A unique block identifier. Strings keep the representation portable and
# debuggable; the convention used throughout this codebase is "stage.index"
# e.g. "layer2.3" for the fourth block of the second stage.
BlockId = str


@dataclass(frozen=True)
class BlockDescriptor:
    """
    Immutable description of a single residual block within a network.

    The fields are deliberately minimal: GDBA only needs to know what block
    exists, which stage it belongs to, and whether it is the entry block
    of that stage. Number of parameters, expansion factor, channel count,
    etc. live elsewhere (computed by FLOPs counters when needed).

    Attributes
    ----------
    block_id
        Globally unique identifier, e.g. "layer1.0".
    stage_name
        Name of the parent stage, e.g. "layer1". Multiple blocks share a
        stage_name.
    index_in_stage
        Zero-based position of this block within its stage. The entry
        block always has index 0.
    is_entry_block
        True iff this is the first block of its stage. Derived from
        index_in_stage but stored explicitly for clarity at use sites.
    """

    block_id: BlockId
    stage_name: str
    index_in_stage: int
    is_entry_block: bool

    def __post_init__(self) -> None:
        # Internal consistency: is_entry_block must agree with index 0.
        # We allow caller-provided is_entry to support architectures where
        # the entry block might differ from index 0 (rare, but possible).
        # If it does diverge, treat as a programmer error.
        if self.index_in_stage == 0 and not self.is_entry_block:
            raise ValueError(
                f"Block {self.block_id!r} is at index 0 but not marked as "
                f"entry block; this is almost certainly a bug."
            )


@dataclass(frozen=True)
class StageDescriptor:
    """
    Immutable description of a stage (a group of blocks at one spatial
    resolution).

    Attributes
    ----------
    name
        Stage identifier, e.g. "layer1".
    block_ids
        Ordered list of block IDs within this stage. Order matters: the
        graph construction relies on it to wire up the sequential edges.
    """

    name: str
    block_ids: tuple[BlockId, ...]

    def __post_init__(self) -> None:
        if len(self.block_ids) == 0:
            raise ValueError(f"Stage {self.name!r} has no blocks")

    @property
    def entry_block(self) -> BlockId:
        """The first block of the stage; always the entry block."""
        return self.block_ids[0]

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def num_non_entry_blocks(self) -> int:
        """
        Number of blocks eligible for gating (all blocks except the entry).
        This is the search space size for top-k selection within the stage.
        """
        return len(self.block_ids) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Block graph (the central domain object)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlockGraph:
    """
    Complete graph representation of a ResNet's residual-block topology.

    A `BlockGraph` consists of an ordered list of stages (each carrying
    its own blocks), plus a precomputed adjacency matrix that encodes
    both sequential connections within stages, skip connections within
    stages, and the stage-to-stage transitions.

    The class is frozen — immutable — because it represents fixed
    architecture, and GDBA's behavior must be deterministic given a
    network. Construction goes through `build_block_graph()`, which is
    the only place where the adjacency matrix is computed.

    Attributes
    ----------
    stages
        Ordered tuple of `StageDescriptor`s. The order defines the
        forward-pass sequence.
    blocks
        Ordered tuple of `BlockDescriptor`s, flattened across stages.
        Element i corresponds to row/column i in the adjacency matrix.
    adjacency
        Directed weighted adjacency matrix. `adjacency[i, j]` is the
        edge weight from block i to block j (0 means no edge).

    Edge convention
    ---------------
    The graph is *directed*. Three kinds of edges exist:

      1. Sequential edges (within a stage): block i -> block i+1, weight
         `SEQUENTIAL_EDGE_WEIGHT` (1.0). Reflects the principal
         information flow.

      2. Skip edges (within a stage): block i+1 -> block i, weight
         `SKIP_EDGE_WEIGHT` (0.25). Reflects the residual short-circuit
         that propagates information backwards via the identity branch.
         Lower weight encodes the empirical observation that skip
         connections carry less of the "useful" signal than direct
         sequential flow.

      3. Stage-transition edges: last block of stage k -> first block of
         stage k+1, weight `SEQUENTIAL_EDGE_WEIGHT` (1.0). Crosses spatial-
         resolution boundaries.
    """

    stages: tuple[StageDescriptor, ...]
    blocks: tuple[BlockDescriptor, ...]
    adjacency: np.ndarray  # (n_blocks, n_blocks), float64
    _block_id_to_index: Mapping[BlockId, int] = field(repr=False)

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def all_block_ids(self) -> tuple[BlockId, ...]:
        return tuple(b.block_id for b in self.blocks)

    @property
    def entry_block_ids(self) -> frozenset[BlockId]:
        """
        Set of block IDs that GDBA must keep active regardless of score.

        These are the entry blocks of each stage (which perform spatial
        down-sampling and channel expansion). Disabling them would break
        the shape contract between stages.
        """
        return frozenset(s.entry_block for s in self.stages)

    @property
    def non_entry_block_ids(self) -> tuple[BlockId, ...]:
        """
        Ordered tuple of blocks eligible for gating.

        This is the search space for top-k selection. The order matches
        the forward-pass order and is determined by the original
        block order.
        """
        entries = self.entry_block_ids
        return tuple(b.block_id for b in self.blocks if b.block_id not in entries)

    def index_of(self, block_id: BlockId) -> int:
        """
        Look up the row/column index of a block in the adjacency matrix.

        Raises
        ------
        KeyError
            If the block ID is not part of this graph.
        """
        try:
            return self._block_id_to_index[block_id]
        except KeyError:
            raise KeyError(f"Unknown block ID: {block_id!r}") from None

    def stage_of(self, block_id: BlockId) -> StageDescriptor:
        """Return the stage descriptor containing the given block."""
        for stage in self.stages:
            if block_id in stage.block_ids:
                return stage
        raise KeyError(f"Block {block_id!r} does not belong to any stage")

    def blocks_in_stage(self, stage_name: str) -> tuple[BlockId, ...]:
        """Return all block IDs in the named stage, in forward order."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.block_ids
        raise KeyError(f"Unknown stage: {stage_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_block_graph(stage_specs: Iterable[tuple[str, int]]) -> BlockGraph:
    """
    Construct a `BlockGraph` from a stage specification.

    The specification is an iterable of `(stage_name, num_blocks)` pairs,
    given in forward-pass order. For example, ResNet-50 on CIFAR is:

        [("layer1", 3), ("layer2", 4), ("layer3", 6), ("layer4", 3)]

    Edge weights are pulled from `constants.SEQUENTIAL_EDGE_WEIGHT` and
    `constants.SKIP_EDGE_WEIGHT`; no other tunables affect graph topology.

    Parameters
    ----------
    stage_specs
        Iterable of (stage_name, num_blocks) tuples. Stage names should be
        non-empty and unique; num_blocks must be at least 1.

    Returns
    -------
    BlockGraph
        Fully constructed, frozen graph ready for centrality computation.

    Raises
    ------
    ValueError
        If the specification is empty, has duplicate stage names, or has
        any stage with zero blocks.
    """
    # --- Validate input upfront so the rest of the function is total -----
    specs = list(stage_specs)
    if not specs:
        raise ValueError("stage_specs must contain at least one stage")

    seen_names: set[str] = set()
    for name, n in specs:
        if not name:
            raise ValueError("Stage name must be non-empty")
        if name in seen_names:
            raise ValueError(f"Duplicate stage name: {name!r}")
        seen_names.add(name)
        if n < 1:
            raise ValueError(f"Stage {name!r} must have at least 1 block, got {n}")

    # --- Build the per-stage descriptors and the flat block list ---------
    stages: list[StageDescriptor] = []
    blocks: list[BlockDescriptor] = []

    for stage_name, n_blocks in specs:
        stage_block_ids = tuple(f"{stage_name}.{i}" for i in range(n_blocks))
        stages.append(StageDescriptor(name=stage_name, block_ids=stage_block_ids))
        for i, bid in enumerate(stage_block_ids):
            blocks.append(BlockDescriptor(
                block_id=bid,
                stage_name=stage_name,
                index_in_stage=i,
                is_entry_block=(i == 0),
            ))

    # --- Build the index lookup table ------------------------------------
    block_id_to_index: dict[BlockId, int] = {
        b.block_id: i for i, b in enumerate(blocks)
    }

    # --- Build the adjacency matrix --------------------------------------
    n = len(blocks)
    adjacency = np.zeros((n, n), dtype=np.float64)

    # 1. Sequential edges (block i -> block i+1) across the entire network.
    #    These are the principal forward edges in the directed graph.
    for i in range(n - 1):
        adjacency[i, i + 1] = SEQUENTIAL_EDGE_WEIGHT

    # 2. Within-stage skip edges (block i+1 -> block i).
    #    The forward sequential edge i -> i+1 already has weight 1.0
    #    from step 1; here we add the backward skip edge with the smaller
    #    weight 0.25. Only edges within the same stage are added — stage
    #    boundaries break the skip pattern.
    for stage in stages:
        for u, v in zip(stage.block_ids[:-1], stage.block_ids[1:]):
            iu = block_id_to_index[u]
            iv = block_id_to_index[v]
            # Forward edge already set in step 1; explicitly set again
            # for clarity/idempotency.
            adjacency[iu, iv] = SEQUENTIAL_EDGE_WEIGHT
            # Backward (skip) edge.
            adjacency[iv, iu] = SKIP_EDGE_WEIGHT

    # 3. Stage-transition edges (last block of stage k -> first of k+1).
    #    These were already set as sequential edges in step 1, but we
    #    re-set them here for documentation; their weight is the standard
    #    1.0. No backward skip across stages because of resolution change.
    for prev_stage, next_stage in zip(stages[:-1], stages[1:]):
        u = block_id_to_index[prev_stage.block_ids[-1]]
        v = block_id_to_index[next_stage.block_ids[0]]
        adjacency[u, v] = SEQUENTIAL_EDGE_WEIGHT

    return BlockGraph(
        stages=tuple(stages),
        blocks=tuple(blocks),
        adjacency=adjacency,
        _block_id_to_index=block_id_to_index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Predefined ResNet topologies
# ─────────────────────────────────────────────────────────────────────────────

# Block counts per stage are an architectural fact of each ResNet variant.
# Source: He et al. 2016, Table 1.

RESNET18_STAGE_SPECS: tuple[tuple[str, int], ...] = (
    ("layer1", 2),
    ("layer2", 2),
    ("layer3", 2),
    ("layer4", 2),
)
"""ResNet-18: 4 stages, 2 blocks each = 8 blocks total."""

RESNET34_STAGE_SPECS: tuple[tuple[str, int], ...] = (
    ("layer1", 3),
    ("layer2", 4),
    ("layer3", 6),
    ("layer4", 3),
)
"""ResNet-34: same depth distribution as ResNet-50 but BasicBlocks."""

RESNET50_STAGE_SPECS: tuple[tuple[str, int], ...] = (
    ("layer1", 3),
    ("layer2", 4),
    ("layer3", 6),
    ("layer4", 3),
)
"""ResNet-50: 4 stages with 3, 4, 6, 3 Bottlenecks = 16 blocks total."""

RESNET101_STAGE_SPECS: tuple[tuple[str, int], ...] = (
    ("layer1", 3),
    ("layer2", 4),
    ("layer3", 23),
    ("layer4", 3),
)
"""ResNet-101: deeper layer3 = 33 blocks total."""


def build_resnet18_graph() -> BlockGraph:
    """Convenience constructor for the canonical ResNet-18 block graph."""
    return build_block_graph(RESNET18_STAGE_SPECS)


def build_resnet50_graph() -> BlockGraph:
    """Convenience constructor for the canonical ResNet-50 block graph."""
    return build_block_graph(RESNET50_STAGE_SPECS)
