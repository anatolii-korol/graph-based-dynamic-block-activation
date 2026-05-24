"""
Paper-grade benchmark protocol for GDBA inference.

This module implements the measurement methodology described in Section 5
of the paper. A single call to `benchmark_gdba()` produces all the
numbers required for the result tables: top-1/top-5 accuracy, effective
FLOPs, wall-clock latency, and GPU energy — separated cleanly between
the *scoring* pass and the *inference* pass.

Why separate scoring from inference
-----------------------------------
GDBA needs occasional scoring passes (forward + backward) to update the
gate map. Conflating their cost with inference would either:

  - hide the scoring overhead (if we only timed inference), or
  - inflate the apparent inference cost (if we summed them together).

Neither is acceptable for honest reporting. The benchmark therefore
tracks two independent budgets and reports the sum and the breakdown.

Why analytical FLOPs
--------------------
A hook-based FLOPs counter has two serious problems:

  1. It includes the scoring backward pass, double-counting against
     inference FLOPs.
  2. It is non-deterministic across runs because the hook activations
     depend on which CUDA kernels are dispatched.

Instead, we use `compute_block_flops` once on a probe batch to obtain
per-block, per-sample FLOPs analytically. At each inference batch, we
sum the contributions of the currently active blocks and multiply by
the actual batch size. This handles partial trailing batches correctly.

Why warmup
----------
The first 10 inference calls on a CUDA device are dominated by:
  - cuDNN algorithm autotuning (first call to each conv shape),
  - CUDA context initialization,
  - GPU clock ramp from idle.

These costs are real but one-time, so they belong in the warmup phase,
not in the timed measurements.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..constants import DEFAULT_LATENCY_WARMUP_ITERATIONS
from ..infrastructure.energy_meter import EnergyMeter
from ..infrastructure.flops_counter import (
    compute_block_flops,
    compute_effective_flops,
)
from ..infrastructure.latency_meter import _CUDAEventTimer
from .controller import GDBAController


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AccuracyMetrics:
    """Top-1 and top-5 accuracy plus the sample count they were computed on."""

    top1_acc: float
    top5_acc: float
    total_samples: int


@dataclass(frozen=True)
class PerClassMetrics:
    """
    Per-class accuracy breakdown computed by accumulating the confusion-
    matrix diagonal alongside the main inference loop.

    Attributes
    ----------
    per_class_accuracy
        Dict mapping class label (with optional name from CIFAR-100
        index→name map) to accuracy in [0, 1].
    accuracy_mean, accuracy_std, accuracy_min, accuracy_max
        Aggregate statistics over the per-class values.
    worst_5, best_5
        Lists of (class_index, accuracy, support_count) for the five
        worst- and best-classified classes.
    confusion_diagonal_sum
        Sum of correctly-classified samples (== total top-1 hits).
    confusion_total
        Total samples seen (== total_samples in AccuracyMetrics).
    """

    per_class_accuracy: dict[int, float]
    per_class_accuracy_named: dict[str, float]
    accuracy_mean: float
    accuracy_std: float
    accuracy_min: float
    accuracy_max: float
    worst_5: list[tuple[int, float, int]]
    best_5: list[tuple[int, float, int]]
    confusion_diagonal_sum: int
    confusion_total: int


@dataclass(frozen=True)
class LatencyStats:
    """
    Detailed latency distribution measured by repeated forward passes
    over the same batch. The "cold" measurement is the very first run
    after warmup; "warm" measurements are repeated samples that show
    steady-state behaviour.

    All times in seconds. Throughput is samples/sec under warm conditions.
    """

    cold_latency_s: float
    cold_latency_per_sample_s: float
    warm_latency_mean_s: float
    warm_latency_std_s: float
    warm_latency_p50_s: float
    warm_latency_p90_s: float
    warm_latency_p99_s: float
    warm_latency_min_s: float
    warm_latency_max_s: float
    warm_latency_per_sample_s: float
    throughput_samples_per_sec: float
    cold_warm_overhead_ratio: float
    n_warm_iterations: int


@dataclass(frozen=True)
class CostMetrics:
    """
    Cost figures for one phase of the benchmark (either "inference" or
    "scoring"). All time and energy values are totals across the entire
    benchmark run; per-sample values are derived for reporting.

    Attributes
    ----------
    total_time_s
        Wall-clock time accumulated by all calls in this phase.
    total_energy_j
        GPU energy in joules. 0.0 if NVML was unavailable.
    energy_available
        False if any sample in this phase failed to acquire NVML data.
    total_flops
        Floating-point operations summed across all calls. For inference,
        uses analytical effective FLOPs (gate-aware); for scoring, uses
        a 3x-of-forward heuristic (forward + backward + hooks).
    """

    total_time_s: float
    total_energy_j: float
    energy_available: bool
    total_flops: float


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Complete output of one benchmark run.

    The `inference` and `scoring` fields hold separate cost budgets; the
    `accuracy` field holds classification quality. The optional
    `per_class` and `latency` fields hold extended diagnostics produced
    by `benchmark_gdba` (None for baseline runs).
    """

    accuracy: AccuracyMetrics
    inference: CostMetrics
    scoring: CostMetrics
    refresh_count: int
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # Extended diagnostics (optional — may be None for baseline)
    per_class: Optional[PerClassMetrics] = None
    latency: Optional[LatencyStats] = None
    # Snapshots of the final batch's selection state (for debugging /
    # reproducibility):
    gates_last_batch: dict[str, int] = field(default_factory=dict)
    scores_last_batch: dict[str, float] = field(default_factory=dict)
    # Subgraph stability: fraction of refreshes that produced the same
    # active-block set as the previous refresh. 1.0 means gates never
    # changed; lower values indicate scoring is sensitive to data.
    subgraph_stability: float = 0.0

    # ── Derived metrics for reporting ────────────────────────────────────

    @property
    def total_time_s(self) -> float:
        """Combined scoring + inference time."""
        return self.inference.total_time_s + self.scoring.total_time_s

    @property
    def total_energy_j(self) -> float:
        """Combined scoring + inference energy."""
        return self.inference.total_energy_j + self.scoring.total_energy_j

    @property
    def total_flops(self) -> float:
        """Combined scoring + inference FLOPs."""
        return self.inference.total_flops + self.scoring.total_flops

    @property
    def latency_s_per_sample(self) -> float:
        n = self.accuracy.total_samples
        return self.inference.total_time_s / n if n > 0 else 0.0

    @property
    def energy_j_per_sample(self) -> float:
        n = self.accuracy.total_samples
        return self.inference.total_energy_j / n if n > 0 else 0.0

    @property
    def flops_per_sample(self) -> float:
        n = self.accuracy.total_samples
        return self.inference.total_flops / n if n > 0 else 0.0

    @property
    def scoring_overhead_per_sample_s(self) -> float:
        """
        Average extra time per sample attributable to GDBA scoring passes.
        Equal to total scoring time divided by total samples — represents
        the per-sample tax of GDBA over a hypothetical no-scoring baseline.
        """
        n = self.accuracy.total_samples
        return self.scoring.total_time_s / n if n > 0 else 0.0

    # ── Serialization ────────────────────────────────────────────────────
    #
    # Two CLI entry points consume this result and expect DIFFERENT
    # JSON layouts:
    #
    #   - cli.run_gdba         -> to_run_dict()
    #         Emits run-level cost totals + 'rsa_run' block (scoring
    #         overhead, last-batch gates/scores, subgraph stability).
    #         NO 'measurement' block.
    #
    #   - cli.measure_metrics -> to_measurement_dict()
    #         Emits 'measurement' block with per-class breakdown +
    #         latency distribution. NO 'rsa_run' block, no inference/
    #         scoring cost totals.
    #
    # The legacy `to_dict()` method returns BOTH sections; kept for
    # backwards compatibility with scripts that consumed the older
    # combined format.

    def _common_top_level(self) -> dict[str, Any]:
        """Top-level fields shared between both serializers (accuracy +
        refresh_count + config_snapshot)."""
        return {
            "accuracy": asdict(self.accuracy),
            "refresh_count": self.refresh_count,
            "config_snapshot": dict(self.config_snapshot),
        }

    def _rsa_run_block(self) -> dict[str, Any]:
        """The GDBA-mechanism diagnostics section."""
        return {
            "scoring_time_s_total": self.scoring.total_time_s,
            "scoring_overhead_per_sample_s": self.scoring_overhead_per_sample_s,
            "gates_last_batch": dict(self.gates_last_batch),
            "scores_last_batch": dict(self.scores_last_batch),
            "subgraph_stability": self.subgraph_stability,
        }

    def _measurement_block(self) -> dict[str, Any]:
        """The observed-outcomes section: per-class + latency."""
        m: dict[str, Any] = {}
        if self.per_class is not None:
            num_classes = len(self.per_class.per_class_accuracy)
            names_by_index: dict[int, str] = {}
            for idx, name in enumerate(self.per_class.per_class_accuracy_named.keys()):
                names_by_index[idx] = name
            full_records = [
                {
                    "id": class_id,
                    "name": names_by_index.get(class_id, str(class_id)),
                    "accuracy": self.per_class.per_class_accuracy.get(class_id, 0.0),
                }
                for class_id in range(num_classes)
            ]
            m["per_class"] = {
                "top1_acc": self.accuracy.top1_acc,
                "top5_acc": self.accuracy.top5_acc,
                "samples": self.accuracy.total_samples,
                "refresh_count": self.refresh_count,
                "subgraph_stability": self.subgraph_stability,
                "per_class_accuracy_mean": self.per_class.accuracy_mean,
                "per_class_accuracy_std": self.per_class.accuracy_std,
                "per_class_accuracy_min": self.per_class.accuracy_min,
                "per_class_accuracy_max": self.per_class.accuracy_max,
                "worst_5_classes": [list(t) for t in self.per_class.worst_5],
                "best_5_classes": [list(t) for t in self.per_class.best_5],
                "confusion_matrix_diagonal_sum": self.per_class.confusion_diagonal_sum,
                "confusion_matrix_total": self.per_class.confusion_total,
                "per_class_accuracy_full": full_records,
            }
        if self.latency is not None:
            m["latency"] = asdict(self.latency)
        return m

    def to_run_dict(self) -> dict[str, Any]:
        """
        JSON payload for `cli.run_gdba` output.

        Includes cost totals (inference / scoring) and the 'rsa_run'
        diagnostics block. Does NOT include the 'measurement' section
        (no per-class breakdown, no latency distribution).
        """
        out = self._common_top_level()
        out["inference"] = asdict(self.inference)
        out["scoring"] = asdict(self.scoring)
        out["top1_acc"] = self.accuracy.top1_acc
        out["top5_acc"] = self.accuracy.top5_acc
        out["latency_s_per_sample"] = self.latency_s_per_sample
        out["energy_j_per_sample"] = self.energy_j_per_sample
        out["flops_per_sample"] = self.flops_per_sample
        out["rsa_run"] = self._rsa_run_block()
        return out

    def to_measurement_dict(self) -> dict[str, Any]:
        """
        JSON payload for `cli.measure_metrics` output.

        Includes only the 'measurement' section with per-class accuracy
        breakdown and latency distribution. Does NOT include inference/
        scoring cost totals or the 'rsa_run' diagnostics block.
        """
        out = self._common_top_level()
        out["measurement"] = self._measurement_block()
        return out

    def to_dict(self) -> dict[str, Any]:
        """
        Combined payload (legacy). Includes BOTH 'rsa_run' and
        'measurement' sections alongside the top-level fields.
        Equivalent to merging `to_run_dict()` and `to_measurement_dict()`.
        Use the explicit serializers for new code.
        """
        out = self.to_run_dict()
        out["measurement"] = self._measurement_block()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _accuracy_topk(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> float:
    """
    Fraction of `target` labels appearing among the top-k predictions.

    Returns a float in [0, 1]. Computed on CPU/GPU according to the
    tensors' device — no implicit transfers.
    """
    if logits.size(0) == 0:
        return 0.0
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return float(correct.any(dim=1).float().mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-100 class names (for human-readable per-class output)
# ─────────────────────────────────────────────────────────────────────────────

# Loaded lazily from torchvision when first requested. The classes
# attribute on a `datasets.CIFAR100` instance returns the canonical
# 100-element list (e.g. classes[48] == "motorcycle"). We cache the
# result so subsequent benchmark calls don't re-download or re-import.
_CIFAR100_CLASS_NAMES_CACHE: Optional[list[str]] = None


def _cifar100_class_names() -> list[str]:
    """
    Return the canonical CIFAR-100 class-name list, ordered by class
    index. Caches the result process-wide.
    """
    global _CIFAR100_CLASS_NAMES_CACHE
    if _CIFAR100_CLASS_NAMES_CACHE is not None:
        return _CIFAR100_CLASS_NAMES_CACHE
    # Hard-coded fallback list so this works even without re-downloading
    # the dataset. Matches torchvision.datasets.CIFAR100().classes.
    _CIFAR100_CLASS_NAMES_CACHE = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
        "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
        "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
        "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
        "flatfish", "forest", "fox", "girl", "hamster", "house",
        "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle",
        "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
        "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
        "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
        "table", "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe", "whale",
        "willow_tree", "wolf", "woman", "worm",
    ]
    return _CIFAR100_CLASS_NAMES_CACHE


def _cifar10_class_names() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


def _resolve_class_names(num_classes: int) -> list[str]:
    """Return a class-name list matching `num_classes`, or numeric labels."""
    if num_classes == 100:
        return _cifar100_class_names()
    if num_classes == 10:
        return _cifar10_class_names()
    return [str(i) for i in range(num_classes)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-class accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _PerClassAccumulator:
    """
    Streaming accumulator for per-class accuracy + confusion-diagonal.

    Maintained alongside the main inference loop with O(num_classes)
    memory and O(batch) update cost. Internal counters are kept on the
    GPU when available to avoid a per-batch transfer.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = int(num_classes)
        self.device = device
        # support[c] = number of samples seen with true class c
        self.support = torch.zeros(num_classes, dtype=torch.long, device=device)
        # correct[c] = number of those that were correctly classified
        self.correct = torch.zeros(num_classes, dtype=torch.long, device=device)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        with torch.inference_mode():
            pred = logits.argmax(dim=1)
            hits = (pred == target).long()
            # Scatter-add into per-class counters
            self.support.index_add_(
                0, target, torch.ones_like(target, dtype=torch.long)
            )
            self.correct.index_add_(0, target, hits)

    def finalize(self, class_names: Sequence[str]) -> PerClassMetrics:
        support_cpu = self.support.cpu().tolist()
        correct_cpu = self.correct.cpu().tolist()
        total = int(sum(support_cpu))
        diag_sum = int(sum(correct_cpu))

        per_class_acc: dict[int, float] = {}
        per_class_named: dict[str, float] = {}
        for c in range(self.num_classes):
            n = support_cpu[c]
            acc = (correct_cpu[c] / n) if n > 0 else 0.0
            per_class_acc[c] = float(acc)
            name = (
                class_names[c] if c < len(class_names) else str(c)
            )
            per_class_named[name] = float(acc)

        accs = [v for v in per_class_acc.values()]
        if accs:
            mean = float(sum(accs) / len(accs))
            var = float(sum((a - mean) ** 2 for a in accs) / len(accs))
            std = math.sqrt(var)
            acc_min = float(min(accs))
            acc_max = float(max(accs))
        else:
            mean = std = acc_min = acc_max = 0.0

        # Rank classes by accuracy ascending (worst) / descending (best)
        ranked = sorted(
            (
                (c, per_class_acc[c], support_cpu[c])
                for c in range(self.num_classes)
            ),
            key=lambda t: t[1],
        )
        worst_5 = [(c, a, n) for c, a, n in ranked[:5]]
        best_5 = [(c, a, n) for c, a, n in ranked[-5:][::-1]]

        return PerClassMetrics(
            per_class_accuracy=per_class_acc,
            per_class_accuracy_named=per_class_named,
            accuracy_mean=mean,
            accuracy_std=std,
            accuracy_min=acc_min,
            accuracy_max=acc_max,
            worst_5=worst_5,
            best_5=best_5,
            confusion_diagonal_sum=diag_sum,
            confusion_total=total,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Detailed latency stats
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


@torch.inference_mode()
def _measure_latency_stats(
    forward_fn,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warm_iterations: int = 100,
) -> LatencyStats:
    """
    Time `forward_fn(sample_batch)` repeatedly to characterise the
    distribution of inference latency.

    Sequence:
      1. One cold run (timed separately).
      2. `n_warm_iterations` warm runs (timed individually).

    Returns aggregate statistics.
    """
    batch_size = int(sample_batch.size(0))

    # Cold run
    with _CUDAEventTimer(device) as cold_timer:
        _ = forward_fn(sample_batch)
    cold = cold_timer.elapsed_s

    # Warm runs
    warm_times: list[float] = []
    for _ in range(n_warm_iterations):
        with _CUDAEventTimer(device) as t:
            _ = forward_fn(sample_batch)
        warm_times.append(t.elapsed_s)

    if not warm_times:
        return LatencyStats(
            cold_latency_s=cold,
            cold_latency_per_sample_s=cold / max(batch_size, 1),
            warm_latency_mean_s=0.0, warm_latency_std_s=0.0,
            warm_latency_p50_s=0.0, warm_latency_p90_s=0.0,
            warm_latency_p99_s=0.0, warm_latency_min_s=0.0,
            warm_latency_max_s=0.0, warm_latency_per_sample_s=0.0,
            throughput_samples_per_sec=0.0,
            cold_warm_overhead_ratio=1.0,
            n_warm_iterations=0,
        )

    warm_sorted = sorted(warm_times)
    n = len(warm_times)
    mean = sum(warm_times) / n
    variance = sum((t - mean) ** 2 for t in warm_times) / max(n - 1, 1)
    std = variance ** 0.5

    return LatencyStats(
        cold_latency_s=cold,
        cold_latency_per_sample_s=cold / max(batch_size, 1),
        warm_latency_mean_s=mean,
        warm_latency_std_s=std,
        warm_latency_p50_s=_percentile(warm_sorted, 0.50),
        warm_latency_p90_s=_percentile(warm_sorted, 0.90),
        warm_latency_p99_s=_percentile(warm_sorted, 0.99),
        warm_latency_min_s=warm_sorted[0],
        warm_latency_max_s=warm_sorted[-1],
        warm_latency_per_sample_s=mean / max(batch_size, 1),
        throughput_samples_per_sec=batch_size / mean if mean > 0 else 0.0,
        cold_warm_overhead_ratio=cold / mean if mean > 0 else 1.0,
        n_warm_iterations=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probe-based analytical FLOPs setup
# ─────────────────────────────────────────────────────────────────────────────

def _measure_per_sample_block_flops(
    backbone: nn.Module,
    graph,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run one probe batch through the bare backbone and reduce the
    resulting block-level FLOPs to *per-sample* values.

    Per-sample units are necessary because test loaders typically have a
    trailing partial batch (10000 = 39 * 256 + 16); using the probe
    batch's raw FLOPs verbatim would over-count by ~2.4% at high
    top-k ratio.

    Returns
    -------
    Dict mapping each block ID (and `"__stem_head__"`) to its per-sample
    FLOPs.
    """
    sample_x, _ = next(iter(loader))
    probe_x = sample_x.to(device)
    raw = compute_block_flops(backbone, graph, probe_x)
    probe_batch_size = float(probe_x.size(0))
    return {k: v / probe_batch_size for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gdba(
    *,
    controller: GDBAController,
    loader: DataLoader,
    device: torch.device,
    warmup_iterations: int = DEFAULT_LATENCY_WARMUP_ITERATIONS,
    max_batches: Optional[int] = None,
    config_snapshot: Optional[dict[str, Any]] = None,
    num_classes: Optional[int] = None,
    collect_per_class: bool = True,
    collect_latency_stats: bool = True,
    latency_warm_iterations: int = 100,
) -> BenchmarkResult:
    """
    Run a full benchmark pass over `loader` and return all measured
    metrics.

    Protocol
    --------
    1. Probe batch -> per-sample block FLOPs (analytical).
    2. Warmup: `warmup_iterations` forwards, discarded.
    3. Main loop:
       - On refresh steps: time the scoring pass separately (forward +
         backward + selector), counting the cost against `scoring`.
       - On every step: time the gated forward pass against `inference`.
    4. Detailed latency stats (cold + warm-iteration distribution) on the
       first batch if `collect_latency_stats` is True.

    Parameters
    ----------
    controller
        The configured GDBAController to evaluate.
    loader
        Validation / test data loader. Yields (x, y) tuples.
    device
        Device used for timing. Must match the controller's device.
    warmup_iterations
        Discarded warmup forwards before timing begins.
    max_batches
        Optional cap on batches processed; useful for smoke tests.
    config_snapshot
        Optional dict of experiment configuration to embed in the
        result for reproducibility.
    num_classes
        Number of classes for per-class breakdown. If None, inferred
        from the first batch's logits.
    collect_per_class
        If True, accumulate a per-class accuracy breakdown.
    collect_latency_stats
        If True, run a detailed cold/warm-iteration latency study on the
        first batch (uses `latency_warm_iterations` warm iterations).
    latency_warm_iterations
        Number of warm iterations for the latency-distribution study.

    Returns
    -------
    BenchmarkResult with separated scoring/inference metrics + optional
    per-class accuracy and detailed latency stats.
    """
    # Make sure the controller is in eval mode and starts fresh.
    controller.eval()
    controller.reset()

    # ── Step 1: probe-based per-sample FLOPs ──────────────────────────────
    # `compute_block_flops` expects a bare ResNet (with `layer1`...`layer4`
    # attributes); we pass the unwrapped backbone, not the GatedResNet
    # wrapper. The trace will see the same residual blocks regardless of
    # whether we go through the wrapper or the backbone directly, so the
    # per-block shapes (and therefore FLOPs) are identical.
    per_sample_flops = _measure_per_sample_block_flops(
        controller.wrapper.backbone, controller.graph, loader, device,
    )

    # ── Step 2: warmup ────────────────────────────────────────────────────
    # Warmup uses the controller's full forward path (which may include
    # a first-call refresh). Timing is discarded.
    warm_iter = iter(loader)
    for _ in range(max(0, warmup_iterations)):
        try:
            wx, _ = next(warm_iter)
        except StopIteration:
            break
        with torch.inference_mode():
            _ = controller(wx.to(device, non_blocking=True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Reset the step counter so the first real batch triggers a refresh.
    controller.reset()

    # ── Step 3: main evaluation loop ──────────────────────────────────────
    total_samples = 0
    correct_top1 = 0.0
    correct_top5 = 0.0

    inf_time_s = 0.0
    inf_energy_j = 0.0
    inf_flops = 0.0
    inf_energy_ok = True

    score_time_s = 0.0
    score_energy_j = 0.0
    score_flops = 0.0
    score_energy_ok = True

    refresh_count = 0
    # Pre-compute the total per-sample FLOPs of the FULL network — used
    # below to estimate scoring cost as 3x of one forward pass.
    full_flops_per_sample = sum(per_sample_flops.values())

    # Extended-metrics state
    per_class_acc: Optional[_PerClassAccumulator] = None
    # Hold a single sample batch + scores/gates from the final refresh
    # for latency-distribution measurement and JSON reporting.
    last_sample_batch: Optional[torch.Tensor] = None
    last_scores: dict[str, float] = {}
    last_gates_dict: dict[str, int] = {}
    # Subgraph stability: average Jaccard similarity |A∩B| / |A∪B| between
    # the active block sets of consecutive refreshes:
    #
    #     a = {k for k,v in current_gates.items() if v}
    #     b = {k for k,v in previous_gates.items() if v}
    #     stability_scores.append(len(a & b) / max(len(a | b), 1))
    #     ...
    #     subgraph_stability = mean(stability_scores)  # or 1.0 if empty
    #
    # 1.0  = identical active set every time
    # 0.95 = small overlap drop (e.g. one block swapped in/out)
    # 0.0  = completely disjoint active sets
    stability_scores: list[float] = []
    previous_active_set: Optional[frozenset] = None

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_n = int(y.size(0))

        # Decide if this step triggers a scoring refresh.
        will_refresh = (
            controller.step_count % controller.refresh_interval == 0
        )

        if will_refresh:
            # ── Scoring pass: measured separately ─────────────────────────
            # We call the lower-level scoring functions directly here
            # because the controller's step() bundles scoring with the
            # subsequent inference forward into one call. Splitting lets
            # us assign separate timer/energy contexts to each.
            from .scoring import compute_importance_scores

            with EnergyMeter() as score_meter:
                with _CUDAEventTimer(device) as score_timer:
                    breakdown = compute_importance_scores(
                        model=controller.wrapper,
                        graph=controller.graph,
                        centrality_cache=controller._cache,  # noqa: SLF001
                        batch_x=x,
                        batch_y=y,
                        weights=controller._weights,  # noqa: SLF001
                        normalization_scope=controller._normalization_scope,  # noqa: SLF001
                    )
                    selection = controller._selector.select(  # noqa: SLF001
                        breakdown.final, controller.graph,
                    )
                    controller.wrapper.set_gates(selection.gates)
            score_report = score_meter.report()

            score_time_s += score_timer.elapsed_s
            score_energy_j += score_report.energy_joules
            score_energy_ok = score_energy_ok and score_report.available
            # Scoring cost in FLOPs: forward + backward + hook overhead.
            # Empirically ~3x of one forward.
            score_flops += 3.0 * full_flops_per_sample * batch_n

            controller._last_selection = selection  # noqa: SLF001
            refresh_count += 1

            # Subgraph stability bookkeeping: append Jaccard similarity
            # between current and previous active sets. One value per
            # refresh transition; averaged at the end.
            current_active_set = frozenset(
                bid for bid, g in selection.gates.items() if g == 1
            )
            if previous_active_set is not None:
                union = current_active_set | previous_active_set
                inter = current_active_set & previous_active_set
                jaccard = len(inter) / max(len(union), 1)
                stability_scores.append(jaccard)
            previous_active_set = current_active_set

            # Snapshot scores + gates for JSON output
            last_scores = dict(breakdown.final)
            last_gates_dict = dict(selection.gates)

        # ── Inference pass (always run) ───────────────────────────────────
        with EnergyMeter() as inf_meter:
            with _CUDAEventTimer(device) as inf_timer:
                with torch.inference_mode():
                    logits = controller.wrapper(x)
        inf_report = inf_meter.report()

        inf_time_s += inf_timer.elapsed_s
        inf_energy_j += inf_report.energy_joules
        inf_energy_ok = inf_energy_ok and inf_report.available

        # Effective FLOPs from analytical per-sample table, scaled by
        # the actual sample count (handles trailing batches correctly).
        current_gates = {
            bid: (0 if bid in controller.wrapper.gated_off_blocks else 1)
            for bid in controller.graph.all_block_ids
        }
        inf_flops += compute_effective_flops(
            per_sample_flops, current_gates, controller.graph,
        ) * batch_n

        # Accuracy bookkeeping
        correct_top1 += _accuracy_topk(logits, y, k=1) * batch_n
        correct_top5 += _accuracy_topk(logits, y, k=5) * batch_n
        total_samples += batch_n

        # Per-class accumulation: lazy-init on the first batch once we
        # know the number of classes (or use the explicit `num_classes`).
        if collect_per_class:
            if per_class_acc is None:
                k = (
                    num_classes
                    if num_classes is not None
                    else int(logits.size(1))
                )
                per_class_acc = _PerClassAccumulator(k, device)
            per_class_acc.update(logits, y)

        # Keep a sample batch for the post-loop latency study.
        if last_sample_batch is None:
            last_sample_batch = x.detach()

        # Increment step count manually (we bypassed controller.step()).
        controller._step_count += 1  # noqa: SLF001

    # ── Step 4: detailed latency stats (optional) ─────────────────────────
    latency_stats: Optional[LatencyStats] = None
    if collect_latency_stats and last_sample_batch is not None:
        def _gated_forward(x: torch.Tensor) -> torch.Tensor:
            return controller.wrapper(x)
        latency_stats = _measure_latency_stats(
            forward_fn=_gated_forward,
            sample_batch=last_sample_batch,
            device=device,
            n_warm_iterations=latency_warm_iterations,
        )

    # ── Step 5: subgraph stability ────────────────────────────────────────
    # Average Jaccard similarity between active block sets of consecutive
    # refreshes: empty -> 1.0, otherwise sum(jaccards) / count.
    if stability_scores:
        subgraph_stability = sum(stability_scores) / len(stability_scores)
    else:
        subgraph_stability = 1.0

    # ── Step 6: per-class metrics ─────────────────────────────────────────
    per_class_result: Optional[PerClassMetrics] = None
    if per_class_acc is not None:
        class_names = _resolve_class_names(per_class_acc.num_classes)
        per_class_result = per_class_acc.finalize(class_names)

    # ── Assemble the result ──────────────────────────────────────────────
    accuracy = AccuracyMetrics(
        top1_acc=correct_top1 / max(total_samples, 1),
        top5_acc=correct_top5 / max(total_samples, 1),
        total_samples=total_samples,
    )
    inference = CostMetrics(
        total_time_s=inf_time_s,
        total_energy_j=inf_energy_j,
        energy_available=inf_energy_ok,
        total_flops=inf_flops,
    )
    scoring = CostMetrics(
        total_time_s=score_time_s,
        total_energy_j=score_energy_j,
        energy_available=score_energy_ok,
        total_flops=score_flops,
    )

    return BenchmarkResult(
        accuracy=accuracy,
        inference=inference,
        scoring=scoring,
        refresh_count=refresh_count,
        config_snapshot=dict(config_snapshot or {}),
        per_class=per_class_result,
        latency=latency_stats,
        gates_last_batch=last_gates_dict,
        scores_last_batch=last_scores,
        subgraph_stability=subgraph_stability,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (no gating)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_baseline(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_iterations: int = DEFAULT_LATENCY_WARMUP_ITERATIONS,
    max_batches: Optional[int] = None,
    config_snapshot: Optional[dict[str, Any]] = None,
) -> BenchmarkResult:
    """
    Benchmark the bare (un-gated) model for use as a reference point.

    Returns a `BenchmarkResult` with `scoring` set to zeros (baseline has
    no scoring pass) and `inference` containing the full-model metrics.

    Parameters
    ----------
    model
        The trained model, in eval mode. Should already be on `device`.
    loader, device, warmup_iterations, max_batches, config_snapshot
        Same meaning as in `benchmark_gdba`.
    """
    model.eval()

    # Warmup
    warm_iter = iter(loader)
    for _ in range(max(0, warmup_iterations)):
        try:
            wx, _ = next(warm_iter)
        except StopIteration:
            break
        with torch.inference_mode():
            _ = model(wx.to(device, non_blocking=True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

     # FLOPs of the bare model cannot be computed analytically without
    # the BlockGraph that maps to the architecture. The baseline FLOPs
    # equal the controller-equivalent at top_k_ratio=1.0, which the
    # caller can derive from a separate GDBA run if needed.
    #
    # For the baseline we leave total_flops unfilled and let the caller
    # compute it separately if needed. This keeps the function tightly
    # scoped to model-execution metrics.

    total_samples = 0
    correct_top1 = 0.0
    correct_top5 = 0.0
    inf_time_s = 0.0
    inf_energy_j = 0.0
    inf_energy_ok = True

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_n = int(y.size(0))

        with EnergyMeter() as meter:
            with _CUDAEventTimer(device) as timer:
                with torch.inference_mode():
                    logits = model(x)
        report = meter.report()

        inf_time_s += timer.elapsed_s
        inf_energy_j += report.energy_joules
        inf_energy_ok = inf_energy_ok and report.available

        correct_top1 += _accuracy_topk(logits, y, k=1) * batch_n
        correct_top5 += _accuracy_topk(logits, y, k=5) * batch_n
        total_samples += batch_n

    accuracy = AccuracyMetrics(
        top1_acc=correct_top1 / max(total_samples, 1),
        top5_acc=correct_top5 / max(total_samples, 1),
        total_samples=total_samples,
    )
    inference = CostMetrics(
        total_time_s=inf_time_s,
        total_energy_j=inf_energy_j,
        energy_available=inf_energy_ok,
        total_flops=0.0,  # not measured here; see docstring
    )
    scoring = CostMetrics(
        total_time_s=0.0,
        total_energy_j=0.0,
        energy_available=True,
        total_flops=0.0,
    )

    return BenchmarkResult(
        accuracy=accuracy,
        inference=inference,
        scoring=scoring,
        refresh_count=0,
        config_snapshot=dict(config_snapshot or {}),
    )
