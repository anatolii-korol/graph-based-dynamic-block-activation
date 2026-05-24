"""
GPU energy measurement via NVIDIA Management Library (NVML).

This module samples GPU power draw at fixed rate from a background
thread, then integrates the resulting power-versus-time curve via the
trapezoidal rule to obtain total energy in joules. The class is a
context manager:

    with EnergyMeter(device_index=0) as meter:
        run_some_workload()
    print(meter.report().energy_joules)

Method comparison
-----------------
The experimental code used `subprocess.check_output(["nvidia-smi", ...])`
which spawns a process per sample — slow and bursty. This implementation
uses `pynvml` directly, which is the same library `nvidia-smi` calls
under the hood, but stays in-process. Per-sample overhead drops from
~10 ms (subprocess) to <100 us (direct C call), allowing reliable
sampling up to ~50 Hz.

Sampling caveats
----------------
NVML's `nvmlDeviceGetPowerUsage` is itself rate-limited by the driver
to roughly 10-20 Hz on consumer GPUs. Calling it faster than that just
returns the same value, so 10 Hz is a sensible default. The integration
error from trapezoidal rule at 10 Hz is dominated by the NVML
quantization noise (~1% on consumer cards), so reported energies are
"approximate" — useful for *relative* comparisons across runs, less
suitable as absolute metrological quantities.

Graceful degradation
--------------------
If `pynvml` is not installed or the device cannot be queried, the meter
silently records `available=False` in the report rather than raising.
This lets the rest of the pipeline (latency, accuracy) still produce
results on systems without GPU energy telemetry.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..constants import NVML_SAMPLING_HZ


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnergyReport:
    """
    Outcome of an energy-monitoring session.

    Attributes
    ----------
    energy_joules
        Total energy consumed during the measurement window. Computed as
        the trapezoidal integral of power(t).
    average_power_watts
        Mean power = energy / duration. Useful for sanity checks: TDP
        of the GPU bounds this value above.
    duration_seconds
        Wall-clock duration of the measurement window.
    sample_count
        Number of power samples collected. With sampling rate 10 Hz and
        a 1-second workload, we expect ~10 samples.
    available
        False if NVML could not be used (library missing or query
        failed). In that case all numeric fields are 0.0 and should be
        ignored.
    """

    energy_joules: float
    average_power_watts: float
    duration_seconds: float
    sample_count: int
    available: bool


# ─────────────────────────────────────────────────────────────────────────────
# Meter
# ─────────────────────────────────────────────────────────────────────────────

class EnergyMeter:
    """
    Context-manager that samples GPU power in a background thread and
    integrates to total energy on exit.

    Concurrency
    -----------
    The sampler runs in a daemon thread to avoid blocking the main
    Python interpreter doing real work. We use `threading.Event` for
    clean shutdown — the sampler checks the flag between samples.

    Failure modes
    -------------
    If pynvml is unavailable at construction time, the meter records
    `available=False` and `__enter__`/`__exit__` become no-ops. This
    keeps callers simple — no try/except scattered through application
    code.

    Parameters
    ----------
    device_index
        GPU index to query (0 for first / only GPU).
    sample_hz
        Sampling frequency. Default from constants.
    """

    def __init__(
        self,
        device_index: int = 0,
        sample_hz: float = NVML_SAMPLING_HZ,
    ) -> None:
        self._device_index = int(device_index)
        self._sample_period_s = 1.0 / float(sample_hz)
        self._samples: list[tuple[float, float]] = []  # (timestamp, power_watts)
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available, self._device_handle = self._try_init_nvml()

    # ── NVML initialization ─────────────────────────────────────────────

    def _try_init_nvml(self) -> tuple[bool, object]:
        """
        Attempt to load pynvml and obtain a device handle. Returns
        (available, handle). On any failure, returns (False, None) and
        the meter degrades to a no-op.
        """
        try:
            import pynvml  # type: ignore[import]
        except ImportError:
            return False, None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            # Probe once to confirm the device responds.
            pynvml.nvmlDeviceGetPowerUsage(handle)
            self._pynvml = pynvml
            return True, handle
        except Exception:  # noqa: BLE001
            return False, None

    # ── Sampling thread ─────────────────────────────────────────────────

    def _sample_power_milliwatts(self) -> float:
        """
        Return current power draw in milliwatts. NVML returns mW, we
        keep it that way and convert to W at integration time.
        """
        return float(self._pynvml.nvmlDeviceGetPowerUsage(self._device_handle))

    def _run_sampler(self) -> None:
        """Background sampling loop. Stops when _stop_flag is set."""
        while not self._stop_flag.is_set():
            try:
                power_mw = self._sample_power_milliwatts()
                t = time.perf_counter()
                power_w = power_mw / 1000.0
                self._samples.append((t, power_w))
            except Exception:  # noqa: BLE001
                # A transient NVML hiccup should not crash the meter;
                # mark as unavailable so the report indicates incomplete
                # data.
                self._available = False
                break
            # Sleep until next sample. Event.wait honors the stop flag,
            # so shutdown is responsive.
            self._stop_flag.wait(self._sample_period_s)

    # ── Context-manager protocol ────────────────────────────────────────

    def __enter__(self) -> "EnergyMeter":
        if not self._available:
            return self
        self._samples.clear()
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_sampler, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._thread is not None:
            self._stop_flag.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        return False

    # ── Reporting ────────────────────────────────────────────────────────

    def report(self) -> EnergyReport:
        """
        Compute total energy from the collected samples via trapezoidal
        integration.

        At least two samples are needed; with fewer, returns a report
        marked `available=False`.
        """
        if not self._available or len(self._samples) < 2:
            return EnergyReport(
                energy_joules=0.0,
                average_power_watts=0.0,
                duration_seconds=0.0,
                sample_count=len(self._samples),
                available=False,
            )

        timestamps = [t for t, _ in self._samples]
        powers = [p for _, p in self._samples]

        # Trapezoidal integration: energy = sum((p_i + p_{i+1})/2 * dt_i)
        energy = 0.0
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            energy += 0.5 * (powers[i] + powers[i - 1]) * dt
        duration = timestamps[-1] - timestamps[0]
        avg_power = energy / duration if duration > 0 else 0.0

        return EnergyReport(
            energy_joules=energy,
            average_power_watts=avg_power,
            duration_seconds=duration,
            sample_count=len(self._samples),
            available=True,
        )
