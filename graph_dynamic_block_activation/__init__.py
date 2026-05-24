"""
GDBA (Graph-Based Dynamic Block Activation) — reference implementation.

This package implements the inference-time block-gating method described
in the paper. The codebase follows Clean Architecture with four layers:

    constants/                Magic-number-free configuration
    domain/                   Pure business logic, no framework deps
    infrastructure/           PyTorch / NVML / file-system adapters
    application/              Orchestration, benchmarks
    training/                 Training loop, Stochastic Depth
    cli/                      Command-line entry points

Public API exported below. Everything else is internal.
"""

from __future__ import annotations

# Top-level configuration
from .config import ExperimentConfig

# Domain (pure logic)
from .domain.architecture import (
    BlockDescriptor,
    BlockGraph,
    BlockId,
    StageDescriptor,
    build_block_graph,
    build_resnet18_graph,
    build_resnet50_graph,
)
from .domain.importance import (
    ImportanceWeights,
    ScoreBreakdown,
    compute_block_scores,
)
from .domain.selection import (
    SelectionResult,
    TopKSelector,
)

# Infrastructure (framework adapters)
from .infrastructure.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from .infrastructure.model_factory import (
    build_model,
    build_model_from_checkpoint,
)

# Application (orchestration)
from .application.benchmark import (
    AccuracyMetrics,
    BenchmarkResult,
    CostMetrics,
    benchmark_baseline,
    benchmark_gdba,
)
from .application.controller import GDBAController, GDBAStepResult

# Training (used when producing checkpoints)
from .training.trainer import (
    EpochRecord,
    TrainingConfig,
    TrainingResult,
    train_model,
)

__all__ = [
    "ExperimentConfig",
    "TrainingConfig",
    "ImportanceWeights",
    "BlockDescriptor",
    "BlockGraph",
    "BlockId",
    "StageDescriptor",
    "build_block_graph",
    "build_resnet18_graph",
    "build_resnet50_graph",
    "ScoreBreakdown",
    "compute_block_scores",
    "SelectionResult",
    "TopKSelector",
    "build_model",
    "build_model_from_checkpoint",
    "CheckpointMetadata",
    "load_checkpoint",
    "save_checkpoint",
    "AccuracyMetrics",
    "BenchmarkResult",
    "CostMetrics",
    "benchmark_baseline",
    "benchmark_gdba",
    "GDBAController",
    "GDBAStepResult",
    "EpochRecord",
    "TrainingResult",
    "train_model",
]

__version__ = "1.0.0"
