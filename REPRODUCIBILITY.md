# Reproducibility guide


This document lists the exact commands used to produce every numerical
result reported in the paper. It is limited to the experiments and
tables included in the final article. Extended measurements and
supporting experimental results that were obtained during the study but
not fully included in the paper are documented separately in
[SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md](SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md).

All commands assume:

- CWD is the parent of the `graph_dynamic_block_activation/` directory.
- A Python environment with PyTorch ≥ 2.0, torchvision, numpy installed.
- For energy reporting, `pynvml` is installed and the test machine has an
  NVIDIA GPU. CPU-only runs will report `energy_available=false` but
  produce correct accuracy and FLOPs numbers.
- Output goes to `./outputs/` (created automatically).

The reference hardware is an Intel i7-10750H laptop with a GTX 1650
(4 GB VRAM), 16 GB RAM, Windows 11. Other machines will produce
identical accuracy and FLOPs but different latency / energy figures.

**Table numbering below follows the paper.** The five tables in the
final article are:

| #  | Content                                                              | Section in this guide |
|----|----------------------------------------------------------------------|-----------------------|
| 1  | ResNet-50 stage structure (static; no measurement needed)            | —                     |
| 2  | Ablation study of importance-formula components                      | §3                    |
| 3  | Base accuracies (r = 1.0) + Pareto curve, ResNet-50 / CIFAR-100     | §1, §2                |
| 4  | m = 0 vs m = 2 comparison, ResNet-50 / CIFAR-100                    | §4                    |
| 5  | Energy-efficiency metric ξ = Accuracy / Energy                       | §2 (post-processing)  |

Section §1 covers training the three checkpoints. Section §2 runs the
main Pareto sweep. Section §3 covers ablation. Section §4 covers the
m-parameter comparison. Sections §5–§6 cover determinism notes and
pitfalls.

## 1. Train the three reference checkpoints

These commands produce the checkpoints referenced in Tables 3–5.

### ResNet-18 on CIFAR-10

```bash
python -m graph_dynamic_block_activation.cli.train \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 40 \
    --learning-rate 0.05 \
    --warmup-epochs 5 \
    --stochastic-depth-p 0.1 \
    --batch-size 128 \
    --seed 42 \
    --output-dir ./outputs \
    --run-name r18_c10_sd01_final
```

Expected final test top-1: 90.51 % (± 0.10 across seeds).
Wall time: ~25 minutes on a GTX 1650.

### ResNet-18 on CIFAR-100

```bash
python -m graph_dynamic_block_activation.cli.train \
    --model resnet18 \
    --dataset cifar100 \
    --epochs 40 \
    --learning-rate 0.05 \
    --warmup-epochs 5 \
    --stochastic-depth-p 0.1 \
    --batch-size 128 \
    --seed 42 \
    --output-dir ./outputs \
    --run-name r18_c100_sd01_final
```

Expected final test top-1: 69.98 %.
Wall time: ~25 minutes on a GTX 1650.

### ResNet-50 on CIFAR-100

ResNet-50 is deeper and benefits from the zero-init residual BN trick
(on by default) plus a small batch size (the 4 GB VRAM constraint).

```bash
python -m graph_dynamic_block_activation.cli.train \
    --model resnet50 \
    --dataset cifar100 \
    --epochs 50 \
    --learning-rate 0.05 \
    --warmup-epochs 5 \
    --stochastic-depth-p 0.1 \
    --batch-size 64 \
    --seed 42 \
    --output-dir ./outputs \
    --run-name r50_c100_sd01_final
```

Expected final test top-1: 70.74 %.
Wall time: ~3 hours on a GTX 1650.

These three checkpoints provide the baseline row (r = 1.0) of Table 3:

| Model / dataset       | Top-1   | Top-5   |
|-----------------------|---------|---------|
| ResNet-18 / CIFAR-10  | 90.51 % | 99.67 % |
| ResNet-18 / CIFAR-100 | 69.98 % | 90.07 % |
| ResNet-50 / CIFAR-100 | 70.74 % | 90.48 % |

## 2. Pareto sweep — ResNet-50 / CIFAR-100 (Table 3, Table 5)

The Pareto curve in Table 3 uses the ResNet-50 / CIFAR-100 checkpoint
swept over ten top-k ratios. The ξ-metric in Table 5 is derived by
post-processing the same sweep output.

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
    --top-k-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --min-keep-per-stage 2 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output-dir ./outputs/r50_c100_sd01_final/pareto
```

The command writes one `r<ratio>.json` per ratio plus a
`summary_all.json` aggregating all ten points. The Pareto curve in
Table 3 uses columns top1_acc, flops_per_sample, and
latency_s_per_sample from this summary.

**Note on Table 5 (ξ = Accuracy / Energy).** The ξ-metric values are
computed by post-processing the `summary_all.json` output above.
For ResNet-18 configurations, NVML cannot directly measure per-batch
energy at sub-millisecond inference times; the energy values are
therefore derived analytically via `E = FLOPs × k`, with the
calibration coefficient `k` fitted against the ResNet-50 measurement.
This is documented in the Limitations section of the paper.

## 3. Importance-formula ablation (Table 2)

Table 2 measures the contribution of each component in the importance
formula by zeroing the corresponding coefficients. Four variants ×
three ratios = 12 measurements on ResNet-50 / CIFAR-100. The full
formula baseline row is already covered by the §2 sweep.

```bash
CKPT=./checkpoints/r50_c100_sd01_final/checkpoint.pt
OUTDIR=./outputs/r50_c100_sd01_final/ablation

# Variant 1: only activation (alpha = 1)
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint $CKPT \
    --top-k-ratios 0.3 0.5 0.7 --min-keep-per-stage 2 \
    --alpha 1.0 --beta 0.0 --gamma 0.0 --delta 0.0 --epsilon 0.0 \
    --no-baseline \
    --output-dir $OUTDIR/only_act

# Variant 2: only saliency (beta = 1)
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint $CKPT \
    --top-k-ratios 0.3 0.5 0.7 --min-keep-per-stage 2 \
    --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --epsilon 0.0 \
    --no-baseline \
    --output-dir $OUTDIR/only_sal

# Variant 3: only graph centralities. Note: coefficients must sum to
# exactly 1.0 (within 1e-6). Use 0.34 + 0.33 + 0.33 = 1.00.
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint $CKPT \
    --top-k-ratios 0.3 0.5 0.7 --min-keep-per-stage 2 \
    --alpha 0.0 --beta 0.0 --gamma 0.34 --delta 0.33 --epsilon 0.33 \
    --no-baseline \
    --output-dir $OUTDIR/only_graph

# Variant 4: without graph centralities (alpha = beta = 0.5)
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint $CKPT \
    --top-k-ratios 0.3 0.5 0.7 --min-keep-per-stage 2 \
    --alpha 0.5 --beta 0.5 --gamma 0.0 --delta 0.0 --epsilon 0.0 \
    --no-baseline \
    --output-dir $OUTDIR/no_graph
```

### Expected results (Table 2 of the paper)

| Variant         | r = 0.3 | r = 0.5 | r = 0.7 |
|-----------------|---------|---------|---------|
| Full formula    | 51.64 % | 64.25 % | 69.41 % |
| Only activation | 45.86 % | 51.03 % | 58.76 % |
| Only saliency   | 60.63 % | 65.93 % | 67.74 % |
| Only graph      | 56.24 % | 60.51 % | 69.26 % |
| No graph        | 47.78 % | 51.23 % | 69.41 % |

Reproductions on different hardware should match within 0.05 p.p.

## 4. m = 0 vs m = 2 comparison (Table 4)

Table 4 contrasts the minimum-per-stage parameter on ResNet-50 /
CIFAR-100 at five ratios. The m = 2 column is already captured by the
sweep in §2; only the m = 0 column needs an extra run:

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
    --top-k-ratios 0.1 0.3 0.5 0.7 0.9 \
    --min-keep-per-stage 0 \
    --refresh-interval 4 \
    --batch-size 128 \
    --no-baseline \
    --output-dir ./outputs/r50_c100_sd01_final/m0_sweep
```

Expected accuracies (top-1, %):

| r   | m = 0   | m = 2   | Δ (m=2 − m=0) |
|-----|---------|---------|---------------|
| 0.1 | 22.99   | 46.46   | +23.47 p.p.   |
| 0.3 | 43.71   | 51.54   | +7.83 p.p.    |
| 0.5 | 58.34   | 64.25   | +5.91 p.p.    |
| 0.7 | 69.03   | 69.27   | +0.24 p.p.    |
| 0.9 | 70.69   | 70.69   | 0.00 p.p.     |

## 5. Single-point measurement with run_gdba.py

`measure_metrics.py` is optimised for sweeping multiple ratios in one
process. For one-off measurements — debugging a particular config,
running an ablation by hand, or testing on a new machine — the simpler
`run_gdba.py` script measures a single (model, ratio, min_keep) point:

```bash
python -m graph_dynamic_block_activation.cli.run_gdba \
    --checkpoint ./checkpoints/r50_c100_sd01_final/checkpoint.pt \
    --top-k-ratio 0.5 \
    --min-keep-per-stage 2 \
    --refresh-interval 4 \
    --batch-size 128 \
    --output ./outputs/r50_c100_sd01_final/single_r05.json
```

Use cases for `run_gdba.py` over `measure_metrics.py`:

- **Single ablation point** — running a Table 2 variant at one ratio:

  ```bash
  python -m graph_dynamic_block_activation.cli.run_gdba \
      --checkpoint $CKPT \
      --top-k-ratio 0.5 --min-keep-per-stage 2 \
      --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --epsilon 0.0 \
      --output ./outputs/r50_c100/saliency_only_r05.json
  ```

- **Smoke test on a new machine** — verify the pipeline end-to-end:

  ```bash
  python -m graph_dynamic_block_activation.cli.run_gdba \
      --checkpoint $CKPT --top-k-ratio 1.0 \
      --max-batches 5 \
      --output /tmp/smoke.json
  ```

- **Reproducing a single regressed measurement** — re-run a
  configuration whose JSON file got corrupted or lost.

## 6. Determinism notes

The package sets seeds for Python's `random`, NumPy, and PyTorch CPU
and CUDA generators wherever applicable. However, full bit-level
reproducibility on CUDA is *not* guaranteed because:

- cuDNN heuristic algorithm selection may choose different
  implementations on different runs (`torch.backends.cudnn.deterministic`
  is set to `False` for speed in benchmarks).
- Reduction order in atomic CUDA operations is non-deterministic.

In practice:

- Accuracy numbers reproduce to **3–4 decimal places** run-to-run.
- FLOPs are **exactly** reproducible (analytical, not measured).
- Latency varies by **0.2–1.0 %** run-to-run on the same hardware.
- Energy varies by **2–5 %** run-to-run because NVML sampling is noisy.

## 7. Common pitfalls

**"Checkpoint metadata is missing model_name"** — the checkpoint was
saved by old code without the new metadata fields. Either re-save it
with the new trainer, or load via the lower-level `load_checkpoint()`
API and build the model manually.

**"Energy not available"** — `pynvml` is not installed or the GPU
doesn't support power telemetry. Accuracy and FLOPs are still reported
correctly; only the energy fields will be zero.

**Latency much higher than expected** — first run after boot includes
cuDNN autotuning and GPU clock ramp. The benchmark warmup
(`--warmup-iterations 10`, default) handles this within a single
process.

**Run-to-run scores_last_batch values differ slightly** —
`scores_last_batch` is a snapshot of the most recent scoring pass, not
a deterministic fingerprint. Its activation- and saliency-dependent
components vary with cuDNN algorithm selection at the 8th decimal
place. The decisions made from these scores (`gates_last_batch` and
`subgraph_stability`) are stable, which is what the paper relies on.

**"NaN loss during training"** — happens occasionally with deep ResNets
under AMP fp16. The trainer skips affected batches automatically
(logged as `skipped=N`). If `N > 5 %` of batches per epoch, lower the
learning rate or disable AMP with `--no-amp`.

## 8. Supplementary Experimental Results

The experiments documented in this guide correspond to the numerical
results reported in the final article. Additional measurements obtained
during the evaluation, including expanded Pareto analyses, operator-
category profiling, repeated latency measurements, per-class accuracy
analysis, and supporting energy-efficiency results, are provided in
[SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md](SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md).

These supplementary materials are intended to complement the article
and provide a more complete view of the experimental evaluation of the
proposed method.
