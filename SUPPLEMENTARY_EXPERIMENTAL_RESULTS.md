# Supplementary Experimental Results

This supplementary document presents extended experimental results obtained during the evaluation of the proposed Graph-Based Dynamic Block Activation method that were not included in the main paper due to space limitations. While the article reports the key results required to support the principal findings, the present document provides additional measurements and detailed reproducibility information for a more complete analysis of the method.

The document includes extended Pareto-sweep results, operator-category profiling, repeated latency measurements with error bars, additional results for different model and dataset configurations, per-class accuracy analysis, and energy-efficiency estimates. These materials are intended to complement the experimental section of the paper and to improve the transparency and reproducibility of the reported evaluation.

## 1. Pareto Sweeps

A single `measure_metrics` run per (model, dataset) configuration sweeps ten top-k ratios and produces all the accuracy, FLOPs, and per-class data needed for the Pareto-style analysis. Latency in these JSONs is from a single timed pass; the paper-grade latency with error bars comes from and replaces those values in the final article.

```bash
# ResNet-18 / CIFAR-100 — m = 0
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r18_c100_sd01_final/checkpoint.pt \
    --top-k-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --min-keep-per-stage 0 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r18_c100_sd01_final/pareto

# ResNet-18 / CIFAR-10 — m = 0
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r18_c10_sd01_final/checkpoint.pt \
    --top-k-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --min-keep-per-stage 0 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r18_c10_sd01_final/pareto
```

Each command writes one `r<ratio>.json` per ratio plus a `summary_all.json`.

**Note on the ξ-metric (ξ = Accuracy / Energy).** The ξ-metric values are computed by post-processing the `summary_all.json` outputs of these sweeps. For the two ResNet-18 configurations on a sub-millisecond inference budget, NVML cannot directly measure per-batch energy; the energy values are derived analytically via `E = FLOPs × k`, with the calibration coefficient `k` fitted against the ResNet-50 measurement. This is documented in section 6.10 of the paper.

---

## 2. GPU Operator-Category Breakdown

This section reports the time spent in each operator category (convolution, BatchNorm, activation, element-wise add, other) at r = 0.5, averaged over **five independent profiler runs**. Run the profiler five times and aggregate:

```bash
for i in 1 2 3 4 5; do
    python -m graph_dynamic_block_activation.cli.profile_breakdown \
        --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
        --top-k-ratio 0.5 \
        --min-keep-per-stage 2 \
        --batch-size 128 \
        --num-iters 20 \
        --output ./outputs/r50_c100_sd01_final/profile/run${i}.json
done
```

PowerShell equivalent:

```powershell
for ($i = 1; $i -le 5; $i++) {
    python -m graph_dynamic_block_activation.cli.profile_breakdown `
        --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt `
        --top-k-ratio 0.5 `
        --min-keep-per-stage 2 `
        --batch-size 128 `
        --num-iters 20 `
        --output "./outputs/r50_c100_sd01_final/profile/run${i}.json"
}
```

Expected mean ± std across the five runs:

| Category             | Time / iter (μ ± σ, μs) | Share (μ ± σ, %)  |
|----------------------|-------------------------|-------------------|
| Convolution          | 96 022 ± 118            | 66.22 ± 0.10      |
| Batch Normalization  | 21 182 ± 92             | 14.61 ± 0.04      |
| Activation (ReLU)    | 14 002 ± 54             | 9.66 ± 0.02       |
| Element-wise (add)   | 12 044 ± 30             | 8.31 ± 0.02       |
| Other                | 1 751 ± 66              | 1.20 ± 0.05       |
| **Total / iter**     | **145 527 ± 362**       | **100**           |

The category-share standard deviation is ≤ 0.10 percentage points, which is evidence that the breakdown is stable across runs.

---

## 3. Per-Sample Latency with Error Bars

Single latency measurements vary 0.5–1.0% run-to-run due to GPU thermal state, cuDNN algorithm selection, and OS scheduler jitter. The paper reports `mean ± std (n = 5)` per ratio rather than a single number; the `repeat_latency` CLI produces this directly.

For every (model, dataset, ratio) point, run the following with the appropriate `--top-k-ratio` and `--min-keep-per-stage`:

```powershell
python -m graph_dynamic_block_activation.cli.repeat_latency `
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt `
    --top-k-ratio 0.5 --min-keep-per-stage 2 `
    --num-runs 5 --warm-iterations 100 `
    --inter-run-cooldown-s 2.0 `
    --output ./outputs/r50_c100_sd01_final/latency/r0.5.json
```

Sweep all ten ratios per configuration with a PowerShell loop:

```powershell
$CKPT = "./checkpoints/r50_c100_sd01_final/checkpoint.pt"
$OUTDIR = "./outputs/r50_c100_sd01_final/latency"

foreach ($r in 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) {
    $rname = $r.ToString().Replace(".", "_")
    python -m graph_dynamic_block_activation.cli.repeat_latency `
        --checkpoint $CKPT `
        --top-k-ratio $r --min-keep-per-stage 2 `
        --num-runs 5 --warm-iterations 100 `
        --output "$OUTDIR/r${rname}.json"
}
```

Each JSON contains a `per_run` array of five measurements and an `aggregate` block with `mean`, `std`, `cv_pct`, and a bootstrap 95% confidence interval. The `paper_string` field at the top level is a ready-to-quote line in the format the article uses.

For the ResNet-18 configurations, repeat the loop with the corresponding checkpoint and `--min-keep-per-stage 0`.

Coefficient of variation between runs typically stays in the 0.2–0.6% range on a GTX 1650. Anything above ~1.5% indicates thermal throttling and warrants increasing `--inter-run-cooldown-s` or running on a cooler machine.

---

## 4. Single-Point Measurement with run_gdba.py

`measure_metrics.py` is optimised for sweeping multiple ratios in one process. For one-off measurements — debugging a particular config, running an ablation by hand, or testing on a new machine — the simpler `run_gdba.py` script measures a single (model, ratio, min_keep) point:

```bash
python -m graph_dynamic_block_activation.cli.run_gdba \
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
    --top-k-ratio 0.5 \
    --min-keep-per-stage 2 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output ./outputs/r50_c100_sd01_final/single_r05.json
```

The output JSON has the same schema as one entry in a `measure_metrics` sweep, so it can be consumed by the same downstream analysis scripts.

Use cases for `run_gdba.py` over `measure_metrics.py`:

**Single ablation point** — running a variant at one ratio rather than the full 0.3 / 0.5 / 0.7 sweep:

```bash
python -m graph_dynamic_block_activation.cli.run_gdba \
    --checkpoint $CKPT \
    --top-k-ratio 0.5 --min-keep-per-stage 2 \
    --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --epsilon 0.0 \
    --output ./outputs/r50_c100/saliency_only_r05.json
```

**Smoke test on a new machine** — verify the pipeline runs end-to-end before committing to a full sweep:

```bash
python -m graph_dynamic_block_activation.cli.run_gdba \
    --checkpoint $CKPT --top-k-ratio 1.0 \
    --max-batches 5 \
    --output /tmp/smoke.json
```

**Reproducing a single regressed measurement** — re-run a configuration whose JSON file got corrupted or lost.

For paper-grade results, prefer `measure_metrics.py`: it amortises model loading across the sweep and writes a `summary_all.json` with all ratios in one file, which is what the analysis notebooks expect. For paper-grade *latency* values specifically, use `repeat_latency.py` from Section 3 — single-pass latency is not sufficient.

---

## 5. Full Pareto Curve — ResNet-50 / CIFAR-100 (m = 2)

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
    --top-k-ratios 0.1 0.3 0.5 0.7 0.8 0.9 1.0 \
    --min-keep-per-stage 2 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r50_c100_sd01_final/pareto
```

| r   | Top-1   | Δ Acc., p.p. | Latency, ms (μ ± σ, n=5) | Energy, mJ | ΔFLOP   |
|-----|---------|--------------|--------------------------|------------|---------|
| 1.0 | 70.74 % | 0.00         | 1.394 ± 0.003            | 69.57      | 0.0 %   |
| 0.9 | 70.69 % | −0.05        | 1.319 ± 0.003            | 65.80      | −5.5 %  |
| 0.8 | 69.41 % | −1.33        | 1.263 ± 0.003            | 62.59      | −10.9 % |
| 0.7 | 69.27 % | −1.47        | 1.263 ± 0.003            | 62.84      | −11.2 % |
| 0.6 | 66.91 % | −3.83        | 1.151 ± 0.003            | 56.97      | −16.5 % |
| 0.5 | 64.25 % | −6.49        | 1.094 ± 0.003            | 54.00      | −22.0 % |
| 0.3 | 51.54 % | −19.20       | 0.981 ± 0.003            | 48.82      | −32.9 % |
| 0.1 | 46.46 % | −24.28       | 0.915 ± 0.005            | 45.08      | −38.4 % |

---

## 6. Effective FLOPs and Scoring Overhead

Detailed view of effective FLOPs, active blocks, warm latency, scoring overhead, and subgraph stability per ratio. Produced by the Section 5 Pareto sweep — read `flops_per_sample`, `scoring_overhead_per_sample_s`, and `subgraph_stability` from each `r<ratio>.json`.

| r   | Eff. FLOPs, GF | Active Blocks | Warm Lat., ms (μ ± σ) | Scoring, ms | Stability |
|-----|----------------|---------------|------------------------|-------------|-----------|
| 1.0 | 2.610          | 16/16         | 1.394 ± 0.003          | 10.72       | 1.000     |
| 0.9 | 2.467          | 15/16         | 1.319 ± 0.003          | 10.81       | 1.000     |
| 0.8 | 2.324          | 14/16         | 1.263 ± 0.003          | 10.30       | 1.000     |
| 0.7 | 2.317          | 14/16         | 1.263 ± 0.003          | 10.33       | 0.992     |
| 0.5 | 2.037          | 12/16         | 1.094 ± 0.003          | 10.78       | 1.000     |
| 0.3 | 1.751          | 10/16         | 0.981 ± 0.003          | 10.78       | 0.981     |
| 0.1 | 1.607          |  9/16         | 0.915 ± 0.005          | 10.25       | 0.979     |

Scoring overhead is constant (~10.5 ms) across all ratios because scoring always runs the full model to compute saliency gradients. At K = 4 the amortised overhead per sample is ~2.7 ms.

---

## 7. Selected Results — ResNet-18 / CIFAR-100 (m = 0)

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r18_c100_sd01_final/checkpoint.pt \
    --top-k-ratios 0.2 0.5 0.6 0.7 0.9 1.0 \
    --min-keep-per-stage 0 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r18_c100_sd01_final/pareto
```

| r   | Top-1   | Δ Acc., p.p. | Latency, ms (μ ± σ, n=5) | Energy, mJ |
|-----|---------|--------------|--------------------------|------------|
| 1.0 | 69.98 % | 0.00         | 0.412 ± 0.003            | 20.03      |
| 0.9 | 69.98 % | 0.00         | 0.411 ± 0.003            | 20.29      |
| 0.7 | 69.99 % | +0.01        | 0.343 ± 0.004            | 16.53      |
| 0.6 | 69.99 % | +0.01        | 0.342 ± 0.003            | 16.47      |
| 0.5 | 66.04 % | −3.94        | 0.307 ± 0.004            | 15.03      |
| 0.2 | 64.06 % | −5.92        | 0.254 ± 0.004            | 12.33      |

Energy values are derived analytically via E = FLOPs × k (NVML sampling period exceeds ResNet-18 inference time). See Limitations §6.10 of the paper.

At r = 0.7: −17.5% energy, −17.3% latency, Δtop-1 = +0.01 p.p.  
At r = 0.9: no gain (⌈0.9·4⌉ = 4 — all non-entry blocks remain active).

---

## 8. Selected Results — ResNet-18 / CIFAR-10 (m = 0)

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r18_c10_sd01_final/checkpoint.pt \
    --top-k-ratios 0.2 0.5 0.7 1.0 \
    --min-keep-per-stage 0 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r18_c10_sd01_final/pareto
```

| r   | Top-1   | Δ Acc., p.p. | Latency, ms (μ ± σ, n=5) | Energy, mJ |
|-----|---------|--------------|--------------------------|------------|
| 1.0 | 90.51 % | 0.00         | 0.408 ± 0.001            | —          |
| 0.7 | 90.37 % | −0.14        | 0.342 ± 0.004            | —          |
| 0.5 | 87.77 % | −2.74        | 0.308 ± 0.004            | —          |
| 0.2 | 81.96 % | −8.55        | 0.265 ± 0.004            | —          |

The energy column is dashes: no NVML calibration is available for this configuration (see Limitations §6.10 of the paper).

---

## 9. Per-Class Accuracy Analysis

ResNet-50 / CIFAR-100, r = 0.5, m = 2. Produced by the Section 5 Pareto sweep — `per_class_accuracy` field in `r0.5.json`.

| Highest-Accuracy Classes | Top-1 | Lowest-Accuracy Classes | Top-1 |
|--------------------------|-------|-------------------------|-------|
| Motorcycle               | 93 %  | Seal                    | 28 %  |
| Road                     | 90 %  | Lizard                  | 30 %  |
| Wardrobe                 | 89 %  | Man                     | 35 %  |
| Orange                   | 86 %  | Squirrel                | 38 %  |
| Sunflower                | 86 %  | Bowl                    | 38 %  |

Classes with distinctive geometric or colour features (motorcycle, orange, sunflower) retain accuracy because identification is possible from early-stage activations. Fine-grained classes (seal, lizard) degrade because late blocks responsible for subtle discrimination are excluded first at aggressive pruning.

---

## 10. Energy-Efficiency Metric ξ

ξ = Accuracy / Energy [% / mJ] = Accuracy [%] × 1000 / Energy [mJ], reported as 1/J.

| Model / Dataset        | r   | Top-1   | Energy, mJ | ξ, 1/J | Δξ      |
|------------------------|-----|---------|------------|--------|---------|
| ResNet-50 / CIFAR-100  | 1.0 | 70.74 % | 69.57      | 10.16  | —       |
| ResNet-50 / CIFAR-100  | 0.9 | 70.69 % | 65.80      | 10.74  | +5.7 %  |
| ResNet-18 / CIFAR-10   | 1.0 | 90.51 % | 20.06      | 45.11  | —       |
| ResNet-18 / CIFAR-10   | 0.7 | 90.37 % | 16.64      | 54.30  | +20.4 % |
| ResNet-18 / CIFAR-100  | 1.0 | 69.98 % | 20.03      | 34.93  | —       |
| ResNet-18 / CIFAR-100  | 0.7 | 69.99 % | 16.53      | 42.34  | +21.2 % |

At r = 0.7 for ResNet-18: ξ improves by 20.4–21.2% versus full inference. ResNet-18 energy values are derived analytically (E = FLOPs × k).

---

## 11. Determinism Notes

The package sets seeds for Python's `random`, NumPy, and PyTorch CPU and CUDA generators wherever applicable. However, full bit-level reproducibility on CUDA is **not** guaranteed because:

- cuDNN heuristic algorithm selection may choose different implementations on different runs (controlled by `torch.backends.cudnn.deterministic`, which is set to `False` for speed in benchmarks).
- Reduction order in atomic CUDA operations is non-deterministic.

In practice, this means:

- Accuracy numbers reproduce to **3–4 decimal places** run-to-run.
- FLOPs are **exactly** reproducible (analytical, not measured).
- Latency varies by **0.2–1.0%** run-to-run on the same hardware (this is exactly why the paper uses `mean ± std (n = 5)` from Section 3).
- Energy varies by **2–5%** run-to-run when NVML can sample at all; for sub-millisecond ResNet-18 inference NVML samples too slowly and energy is reported via the FLOPs-based analytical model (see §6.10 of the paper).

For exact bit-level comparison against reference values, see the regression tests in `tests/`. Those use deterministic, single-threaded CPU code paths and confirm **0.00e+00 diff** on all algorithmic components.

---

## Notes on Reproducibility

- **Accuracy:** ≤ 0.05 p.p. run-to-run variation.
- **FLOPs:** exact (analytical, not measured).
- **Latency:** 0.2–1.0% variation; use `repeat_latency.py` for μ ± σ.
- **Energy:** 2–5% variation (NVML noise). ResNet-18 uses the analytical model.
- **Hardware:** GTX 1650 (4 GB VRAM), Intel i7-10750H, Windows 11. Accuracy and FLOPs are hardware-independent; latency and energy are not.
