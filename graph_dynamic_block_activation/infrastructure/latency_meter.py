"""
Wall-clock latency measurement for inference workloads.

This module times forward passes accurately on both CPU and CUDA devices
and reports a battery of statistics: mean, std, median, p90, p99, min,
max.

Why so many statistics
----------------------
A single mean latency hides important information:

  - Tail latency (p99) matters for real-time applications: a service
    that averages 1 ms but spikes to 10 ms 1% of the time may miss SLAs.
  - Std measures run-to-run variability — large std on CUDA usually
    indicates thermal throttling or memory contention.
  - Cold vs warm distinction is captured by separating warmup from
    timed runs (cold runs are discarded entirely).

Why CUDA events instead of `time.perf_counter()` on GPU
--------------------------------------------------------
CUDA is asynchronous: a Python-level call to `model(x)` returns
immediately after queuing the kernels, not after they complete. Wall-
clock timing on the host would therefore measure *kernel-enqueue* time,
which is dominated by Python overhead and unrelated to actual GPU work.

Solution: `torch.cuda.Event(enable_timing=True)`. Two events are
recorded around the operation; `synchronize()` then `elapsed_time()`
returns true elapsed GPU time in milliseconds. We convert to seconds
for consistency with `time.perf_counter()` (used on CPU).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..constants import (
    DEFAULT_LATENCY_TIMED_RUNS,
    DEFAULT_LATENCY_WARMUP_ITERATIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatencyStatistics:
    """
    Summary statistics for a batch of latency measurements.

    All times are in seconds. `per_sample_*` fields divide by the batch
    size so the user can compare across different batch sizes.

    Attributes
    ----------
    mean_s, std_s, median_s, p90_s, p99_s, min_s, max_s
        Standard descriptive statistics in seconds (per batch).
    n_warmup, n_runs
        Bookkeeping: how many warmups were discarded and how many runs
        contributed to the statistics.
    samples_per_run
        Batch size; used to derive per-sample numbers.
    per_sample_mean_s, per_sample_std_s
        Mean and std divided by `samples_per_run`.
    raw_times_s
        The full sequence of timed run durations (optional; for plotting
        latency histograms or computing additional percentiles).
    """

    mean_s: float
    std_s: float
    median_s: float
    p90_s: float
    p99_s: float
    min_s: float
    max_s: float
    n_warmup: int
    n_runs: int
    samples_per_run: int
    per_sample_mean_s: float
    per_sample_std_s: float
    raw_times_s: tuple[float, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level timer
# ─────────────────────────────────────────────────────────────────────────────

class _CUDAEventTimer:
    """
    Context-manager that measures elapsed time using CUDA Events when on
    GPU, and `time.perf_counter()` on CPU. Result is in seconds.

    For CUDA, both events are recorded on the default stream. We
    `synchronize()` after the stop event to ensure all kernels have
    completed before reading elapsed_time().
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.elapsed_s: float = 0.0
        self._start_event = None
        self._stop_event = None
        self._t0 = 0.0

    def __enter__(self) -> "_CUDAEventTimer":
        if self._use_cuda:
            # Make sure any prior work on the device has finished before
            # we record the start event. Otherwise we'd attribute the
            # tail of the previous batch to the current measurement.
            torch.cuda.synchronize(self._device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._stop_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._use_cuda:
            self._stop_event.record()
            # Block until the stop event actually completes.
            torch.cuda.synchronize(self._device)
            # elapsed_time is in milliseconds.
            self.elapsed_s = self._start_event.elapsed_time(self._stop_event) / 1000.0
        else:
            self.elapsed_s = time.perf_counter() - self._t0
        return False  # do not suppress exceptions


def _percentile(sorted_values: list[float], q: float) -> float:
    """
    Compute the q-th percentile (q in [0, 1]) of a sorted list using
    linear interpolation between neighbors.

    This implementation avoids the NumPy dependency for this single
    operation — keeps the latency module a thin layer over PyTorch.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def benchmark_callable(
    fn: Callable[[], object],
    device: torch.device,
    *,
    samples_per_run: int = 1,
    n_warmup: int = DEFAULT_LATENCY_WARMUP_ITERATIONS,
    n_runs: int = DEFAULT_LATENCY_TIMED_RUNS,
    keep_raw_times: bool = True,
) -> LatencyStatistics:
    """
    Time a parameter-less callable `fn` repeatedly.

    The callable should perform the exact operation under measurement
    (typically a single model forward) and nothing else. The caller is
    responsible for binding the input batch via closure.

    Sequence
    --------
    1. `n_warmup` warmup calls; discarded entirely.
    2. CUDA synchronize.
    3. `n_runs` timed calls, each wrapped in a CUDA-event timer.
    4. Statistics computed from the `n_runs` measurements.

    Parameters
    ----------
    fn
        Zero-argument callable that runs one forward pass.
    device
        Device on which the work happens. Determines the timer kind.
    samples_per_run
        Batch size — used only for the per-sample fields. Does NOT
        change what `fn` does.
    n_warmup, n_runs
        Number of warmup and timed iterations. Defaults from constants.
    keep_raw_times
        If True (default), include the full timing sequence in the
        result for downstream analysis.

    Returns
    -------
    LatencyStatistics with all summary fields populated.
    """
    if samples_per_run <= 0:
        raise ValueError("samples_per_run must be positive")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative")

    # Warmup
    for _ in range(n_warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed
    timings: list[float] = []
    for _ in range(n_runs):
        with _CUDAEventTimer(device) as timer:
            _ = fn()
        timings.append(timer.elapsed_s)

    # Statistics
    n = len(timings)
    mean = sum(timings) / n
    variance = sum((t - mean) ** 2 for t in timings) / max(n - 1, 1)
    std = variance ** 0.5
    sorted_times = sorted(timings)

    return LatencyStatistics(
        mean_s=mean,
        std_s=std,
        median_s=_percentile(sorted_times, 0.50),
        p90_s=_percentile(sorted_times, 0.90),
        p99_s=_percentile(sorted_times, 0.99),
        min_s=sorted_times[0],
        max_s=sorted_times[-1],
        n_warmup=n_warmup,
        n_runs=n_runs,
        samples_per_run=samples_per_run,
        per_sample_mean_s=mean / samples_per_run,
        per_sample_std_s=std / samples_per_run,
        raw_times_s=tuple(timings) if keep_raw_times else (),
    )


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    *,
    n_warmup: int = DEFAULT_LATENCY_WARMUP_ITERATIONS,
    n_runs: int = DEFAULT_LATENCY_TIMED_RUNS,
) -> LatencyStatistics:
    """
    Convenience wrapper: time `model(sample_input)` repeatedly.

    The input MUST already be on the same device as the model — we
    deliberately do not move data here, so the timer measures only the
    forward pass, not host-to-device copies.
    """
    if sample_input.device != next(model.parameters()).device:
        raise ValueError(
            "sample_input device must match model device; "
            "move the input before calling benchmark_forward()."
        )

    model.eval()
    return benchmark_callable(
        fn=lambda: model(sample_input),
        device=sample_input.device,
        samples_per_run=int(sample_input.size(0)),
        n_warmup=n_warmup,
        n_runs=n_runs,
    )
