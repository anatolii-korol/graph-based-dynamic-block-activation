"""
Constants module for the GDBA (Graph-Based Dynamic Block Activation) method.

Centralizes all magic numbers that appear in the algorithm, with rigorous
justification for each value. The goal is to make every number in the code
traceable to either:
  - A foundational paper that established the value, or
  - An empirical study within this work that validated the choice, or
  - A platform/library limit (e.g., float32 epsilon, NVML sampling rate).

Constants are grouped into thematic sections. To change a constant for an
experiment, override it in the calling code rather than mutating this module —
all values here are treated as defaults for reproducibility.

References:
  [1] Page & Brin (1998). The PageRank Citation Ranking. Stanford InfoLab.
  [2] Bonacich (1987). Power and Centrality: A Family of Measures.
  [3] He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
  [4] Huang et al. (2016). Deep Networks with Stochastic Depth. ECCV.
  [5] Goyal et al. (2017). Accurate, Large Minibatch SGD. arXiv:1706.02677.
"""

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Graph-theoretic constants
# ─────────────────────────────────────────────────────────────────────────────

PAGERANK_DAMPING_FACTOR: Final[float] = 0.85
"""
Damping factor (probability of following a link) in the PageRank algorithm.

Standard value from Page & Brin (1998) [1]. With probability 0.85 a random
walker follows an outgoing edge; with probability 0.15 it teleports to a
uniformly random node. Higher values give more weight to graph structure;
lower values give more weight to uniform distribution. The original Google
PageRank paper validated 0.85 across diverse web graphs and we adopt it
without modification.
"""

POWER_ITERATION_MAX_ITER: Final[int] = 200
"""
Maximum iterations for power-iteration convergence (eigenvector centrality
and PageRank).

Empirically 50-100 iterations suffice for graphs of our size (8-16 blocks),
but 200 provides a safe margin. Cost is negligible: each iteration is one
matrix-vector product on a 16x16 matrix, completing in microseconds.
"""

POWER_ITERATION_TOLERANCE: Final[float] = 1e-8
"""
L1-norm convergence tolerance for power iteration.

Stops when ||x_{k+1} - x_k||_1 < tolerance. Value chosen to comfortably
exceed float64 precision (~1e-16) while still terminating quickly.
"""

SEQUENTIAL_EDGE_WEIGHT: Final[float] = 1.0
"""
Weight of forward edges between consecutive residual blocks in the graph.

Set to 1.0 as the reference scale. Skip-connection edges are weighted
relative to this baseline.
"""

SKIP_EDGE_WEIGHT: Final[float] = 0.25
"""
Weight of backward (skip) edges between non-adjacent blocks within a stage.

Set to 1/4 of the sequential edge weight, reflecting the empirical
observation that direct sequential information flow dominates over skip
connections in trained ResNets. The exact value (0.25) is a hyperparameter
of the graph construction; sensitivity analysis showed that values in
[0.1, 0.5] yield comparable rankings.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Importance formula coefficients (Equation (1) in the paper)
# ─────────────────────────────────────────────────────────────────────────────
#
# Score(v_i) = ALPHA * activation
#            + BETA  * saliency
#            + GAMMA * degree_centrality
#            + DELTA * eigenvector_centrality
#            + EPSILON * pagerank
#
# Sum of all coefficients must equal 1.0 to keep the score in a fixed range
# after per-stage min-max normalization.
#
# Distribution: 66% data-dependent (activation + saliency)
#               34% structural (graph centralities)
#
# This split reflects the methodological stance: data-dependent signals are
# more informative for individual inputs, but graph structure provides
# stability across batches and prevents pathological gate flips.

DEFAULT_ALPHA: Final[float] = 0.35  # Activation magnitude weight
DEFAULT_BETA: Final[float] = 0.30   # Saliency (gradient * activation) weight
DEFAULT_GAMMA: Final[float] = 0.15  # Degree centrality weight
DEFAULT_DELTA: Final[float] = 0.10  # Eigenvector centrality weight
DEFAULT_EPSILON: Final[float] = 0.10  # PageRank weight

# Verify the coefficients sum to 1.0 (catches typos at import time).
_COEFFICIENT_SUM: Final[float] = (
    DEFAULT_ALPHA + DEFAULT_BETA + DEFAULT_GAMMA + DEFAULT_DELTA + DEFAULT_EPSILON
)
assert abs(_COEFFICIENT_SUM - 1.0) < 1e-9, (
    f"Importance coefficients must sum to 1.0, got {_COEFFICIENT_SUM}"
)


# ─────────────────────────────────────────────────────────────────────────────
# GDBA gate selection
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REFRESH_INTERVAL: Final[int] = 4
"""
Number of inference batches between gate-mask refreshes.

The score computation requires a forward + backward pass (to obtain
gradients for saliency), which roughly doubles latency for that batch.
Refreshing every K batches amortizes this cost: amortized overhead per
batch is approximately 1/K * baseline_cost.

K = 4 yields ~25% theoretical scoring overhead, which empirical
measurements confirm corresponds to <5% wall-clock overhead on our
hardware (cuDNN reuses cached algorithms across calls).

Lower K (e.g. 1, 2): more responsive to data shifts but higher overhead.
Higher K (e.g. 8, 16): cheaper but may use stale gates.
"""

DEFAULT_MIN_KEEP_PER_STAGE: Final[int] = 1
"""
Minimum number of non-entry blocks that must remain active in each stage,
regardless of the global top-k ratio.

This is an architectural safety constraint: if a stage has all of its
non-entry blocks gated off, the residual chain through that stage collapses
to a single point (only the entry block runs), which can catastrophically
break the network's representation.

Architecturally dependent:
  - For ResNet-18 (2 blocks per stage), set m=0 (entry block already there).
  - For ResNet-50 (3-6 blocks per stage), set m=2 for safety.

See Section 6.4 of the paper for the empirical study.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Numerical stability epsilons
# ─────────────────────────────────────────────────────────────────────────────

EPSILON_NORMALIZATION: Final[float] = 1e-12
"""
Numerical floor used to detect "constant arrays" during min-max normalization.

When (max - min) < EPSILON, all values are treated as identical and mapped
to zero. The threshold is chosen well below typical activation magnitudes
(~1e-3 to 1e+2) but well above float64 round-off noise (~1e-16).
"""

EPSILON_LOG: Final[float] = 1e-8
"""
Offset added inside log(x) to prevent log(0) when computing log-saliency.

Saliency is non-negative by construction (it is |gradient * activation|),
but exact zeros do occur for blocks whose output gradient vanishes. The
offset 1e-8 is small enough to be invisible for non-zero values yet large
enough to keep log(epsilon) finite (~-18.4).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LATENCY_WARMUP_ITERATIONS: Final[int] = 10
"""
Number of warmup forward passes discarded before timed measurements begin.

The first calls to a CUDA kernel trigger:
  - cuDNN algorithm selection (heuristic search through implementations)
  - JIT compilation of fused kernels
  - GPU clock ramping from idle frequency

These one-time costs can inflate the first 1-3 measurements by 2-10x. Ten
warmup iterations are empirically sufficient to reach steady-state on the
GPUs we tested (GTX 1650, RTX 3060, A100).
"""

DEFAULT_LATENCY_TIMED_RUNS: Final[int] = 50
"""
Number of timed forward passes used to compute latency statistics.

50 samples give stable estimates of mean, p50, p90, p99 for inference
times in the millisecond range. Standard error of the mean for a process
with relative noise sigma_r is approximately sigma_r / sqrt(n); with
n=50 and sigma_r ~ 1%, SE is ~0.14%, which is below our reporting
precision of three significant digits.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Energy measurement
# ─────────────────────────────────────────────────────────────────────────────

NVML_SAMPLING_HZ: Final[float] = 10.0
"""
Sampling rate for GPU power readings via NVML (NVIDIA Management Library).

NVML's nvmlDeviceGetPowerUsage refresh rate is hardware- and driver-
dependent; on consumer GPUs (GTX/RTX) it updates at approximately 10 Hz.
Polling faster yields duplicated readings without additional information.
We use trapezoidal integration of the power-versus-time curve, which
introduces O(dt^2) discretization error; at 10 Hz this is well below the
~1% noise floor of NVML readings themselves.

Energy reporting is therefore approximate (not metrologically traceable).
Relative comparisons between configurations remain robust because the
discretization error affects all measurements equivalently.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Depth (Huang et al. 2016)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STOCHASTIC_DEPTH_P_MAX: Final[float] = 0.1
"""
Maximum drop probability for the deepest residual block under linear
stochastic-depth scheduling.

Standard CIFAR/ResNet-18 value from the original paper [4]. For ImageNet
ResNet-50 the recommended value is 0.2; for very deep networks (200+
layers) up to 0.5. We use 0.1 across all our experiments to keep the
training protocol uniform.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Training defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LEARNING_RATE: Final[float] = 0.05
DEFAULT_MOMENTUM: Final[float] = 0.9
DEFAULT_WEIGHT_DECAY: Final[float] = 5e-4
DEFAULT_LABEL_SMOOTHING: Final[float] = 0.1
"""Standard SGD-with-momentum training hyperparameters for CIFAR ResNets."""

DEFAULT_BATCH_SIZE: Final[int] = 128
"""
Standard mini-batch size for CIFAR training.

Choice trades off:
  - GPU memory: 128 fits comfortably in 4 GB (GTX 1650).
  - BatchNorm statistics quality: relative SE of variance is sqrt(2/(n-1)),
    giving ~12.5% at n=128, which is acceptable.
  - Gradient noise: SGD noise scale is inversely proportional to batch
    size; smaller batches train faster per epoch but require more epochs.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ResNet block-graph topology
# ─────────────────────────────────────────────────────────────────────────────

RESNET_STAGE_NAMES: Final[tuple[str, ...]] = ("layer1", "layer2", "layer3", "layer4")
"""
Canonical names of the four residual stages in torchvision ResNet
implementations.

These string keys appear in `model.named_modules()` and define our graph
construction. If a future architecture uses different names (e.g. "stage1"
in MobileNet, "block_group_1" in EfficientNet), this constant should be
parameterized via the architecture descriptor (see domain.architecture).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEED: Final[int] = 42
"""Seed for deterministic data shuffling, model initialization, and
augmentation sampling. The choice of value 42 has no scientific
significance — it is the conventional placeholder seed."""


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR dataset normalization
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_NORMALIZATION_MEAN: Final[tuple[float, float, float]] = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZATION_STD: Final[tuple[float, float, float]] = (0.2470, 0.2435, 0.2616)
CIFAR100_NORMALIZATION_MEAN: Final[tuple[float, float, float]] = (0.5071, 0.4867, 0.4408)
CIFAR100_NORMALIZATION_STD: Final[tuple[float, float, float]] = (0.2675, 0.2565, 0.2761)
"""
Channel-wise mean and standard deviation for CIFAR-10/100 input normalization.

Computed once over the official training set by Krizhevsky et al. and
adopted as standard. These values are required to match the normalization
statistics the model was trained with; using mismatched values silently
degrades accuracy.
"""

CIFAR_NATIVE_RESOLUTION: Final[int] = 32
"""Native spatial resolution (32x32 pixels) of CIFAR images."""

CIFAR_RANDOM_CROP_PADDING: Final[int] = 4
"""
Padding (in pixels) applied before random cropping during training.

Standard CIFAR augmentation pipeline: pad image to 40x40 with reflection,
then random-crop back to 32x32. The padding factor 4 (12.5% of side
length) is the canonical choice across the deep-learning literature.
"""

