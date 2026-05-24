# GDBA: Graph-Based Dynamic Block Activation in ResNets

Zero-shot inference-time block gating for ResNet-family models.

This is the reference implementation accompanying the paper
*Dynamic Activation of Neural Network Blocks Based on Graph Representation
for Energy-Efficient Inference* (Anatoliy Korol, 2026).

## Quick start

### Install

The code depends on PyTorch ≥ 2.0, torchvision, numpy, and optionally
`pynvml` for GPU-energy reporting. A minimal environment:

```bash
pip install torch torchvision numpy
pip install pynvml      # optional: NVIDIA GPU energy reporting
```

### Inference (most common use)

```python
import torch
from graph_dynamic_block_activation import GDBAController, build_model_from_checkpoint

# Load a trained checkpoint
model, metadata = build_model_from_checkpoint("checkpoint.pt")
model = model.cuda().eval()

# Wrap it for GDBA inference
controller = GDBAController.build(
    backbone=model,
    top_k_ratio=0.5,             # keep 50% of non-entry blocks
    min_keep_per_stage=2,        # safety: at least 2 active blocks per stage
).cuda()

# Inference — drop-in replacement for model(x)
with torch.no_grad():
    for x, _ in loader:
        logits = controller(x.cuda())
```

That's it. `controller` is an `nn.Module`. It scores blocks every 4
batches (configurable via `refresh_interval`) and uses the resulting
gates on the intervening batches.

### Train a GDBA-ready checkpoint

GDBA itself is zero-shot, but it only works well on models trained with
Stochastic Depth. The package provides a trainer that handles this
automatically:

```bash
python -m graph_dynamic_block_activation.cli.train \
    --model resnet50 \
    --dataset cifar100 \
    --epochs 50 \
    --stochastic-depth-p 0.1 \
    --output-dir ./outputs/r50_c100
```

### Benchmark

For a Pareto sweep across multiple ratios (the common case for paper
tables):

```bash
python -m graph_dynamic_block_activation.cli.measure_metrics \
    --checkpoint ./checkpoints/r50_c100/checkpoint.pt \
    --top-k-ratios 0.1 0.3 0.5 0.7 0.9 1.0 \
    --output-dir ./outputs/r50_c100/sweep
```

For a single point (useful for ablations or debugging one config):

```bash
python -m graph_dynamic_block_activation.cli.run_gdba \
    --checkpoint ./checkpoints/r50_c100/checkpoint.pt \
    --top-k-ratio 0.5 \
    --min-keep-per-stage 2 \
    --output ./outputs/r50_c100/r05.json
```

Both produce JSON files with top-1, top-5, FLOPs, latency, and energy
metrics, separated cleanly between the scoring pass and the inference
pass. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact
commands to reproduce each table reported in the paper. Extended
measurements and supporting experimental results that were not fully
included in the article are provided in
[SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md](SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md).

---

## Architecture

The codebase follows Clean Architecture with one-way dependencies:

```
┌──────────────────────────────────────────────────────────────┐
│ cli/                  Thin argparse wrappers, no logic        │
├──────────────────────────────────────────────────────────────┤
│ application/          Orchestration: controller, benchmark    │
├──────────────────────────────────────────────────────────────┤
│ infrastructure/       PyTorch / NVML / filesystem adapters    │
├──────────────────────────────────────────────────────────────┤
│ domain/               Pure business logic, no framework deps  │
└──────────────────────────────────────────────────────────────┘
```

Higher layers may import from lower; the reverse is forbidden. The
`domain/` layer in particular has zero PyTorch dependencies, so its
algorithms can be unit-tested with plain NumPy arrays.

### Module map

| Layer | Module | Responsibility |
|---|---|---|
| domain | `centrality.py` | Graph algorithms (degree, eigenvector, PageRank) |
| domain | `architecture.py` | `BlockGraph` — framework-agnostic topology |
| domain | `importance.py` | Importance formula (Equation 9) |
| domain | `selection.py` | `TopKSelector` with per-stage minimum |
| infrastructure | `architecture_inspector.py` | `nn.Module` → `BlockGraph` adapter |
| infrastructure | `activation_tracer.py` | Forward/backward hooks for scoring |
| infrastructure | `pytorch_wrapper.py` | `GatedResNet` (hard-gated inference) |
| infrastructure | `flops_counter.py` | Analytical, gate-aware FLOPs |
| infrastructure | `latency_meter.py` | CUDA-event-based wall clock |
| infrastructure | `energy_meter.py` | NVML power sampling |
| infrastructure | `checkpoint.py` | Self-describing checkpoint I/O |
| infrastructure | `model_factory.py` | ResNet builder + checkpoint loader |
| application | `scoring.py` | Score computation pipeline |
| application | `controller.py` | `GDBAController` orchestrator |
| application | `benchmark.py` | Paper-grade measurement protocol |
| training | `stochastic_depth.py` | `StochasticDepthBlock` + zero-init BN |
| training | `data.py` | CIFAR loaders with stratified split |
| training | `trainer.py` | SGD + warmup + cosine schedule |

### Configuration is distributed

After the refactoring, there is no megaclass holding all parameters.
Instead, each component owns the parameters it actually uses:

```python
# environment (where to run)
ExperimentConfig(model="resnet50", dataset="cifar100", num_classes=100)

# training hyperparameters
TrainingConfig(epochs=50, learning_rate=0.05, stochastic_depth_p_max=0.1)

# GDBA importance formula
ImportanceWeights(alpha=0.35, beta=0.30, gamma=0.15, delta=0.10, epsilon=0.10)

# GDBA selector behavior
TopKSelector(top_k_ratio=0.5, min_keep_per_stage=2)

# GDBA inference orchestration
GDBAController.build(backbone=model, top_k_ratio=0.5, refresh_interval=4)
```

All five are `@dataclass(frozen=True)` with `__post_init__` validation.

---

## Method summary

For background, see Sections 3–5 of the paper. In brief:

1. **Build a directed weighted graph** over the residual blocks of a
   ResNet. Forward sequential edges have weight 1; backward within-stage
   skip edges have weight 0.25; stage transitions have weight 1.

2. **Compute three centralities** on this graph: degree, eigenvector,
   PageRank. These describe each block's structural importance and are
   constant for a given architecture.

3. **At inference**, every K batches:
   - Run a forward + backward pass to capture each block's output
     magnitude (activation) and `|output × gradient|` (saliency).
   - Combine the five signals via the importance formula:

         Score(v) = α · A(v) + β · S(v) + γ · C_deg(v) + δ · C_eig(v) + ε · C_pr(v)

   - Select the top-`k = ceil(r · n_non_entry)` blocks.
   - Force entry blocks always active (shape contract).
   - Force at least `m` blocks active per stage (safety).

4. **Use the resulting gate map** on the next K-1 batches; on the K-th
   batch, refresh.

---

## Reproducibility

Every algorithmic choice in this codebase has at least one corresponding
regression test in `tests/` that compares against the original
ungdba implementation. The full test suite (128 tests) runs in
under 2 minutes on CPU:

```bash
cd graph_dynamic_block_activation
for t in centrality architecture selection infrastructure infrastructure_2 \
         controller benchmark training config; do
    python tests/test_$t.py
done
```

All tests pass with **bit-identical** numerical agreement to the
original code on:

- Graph centralities (max diff 0.00e+00 on ResNet-50 adjacency)
- Adjacency matrix construction (bit-identical)
- Block selection across 54 (ratio × min_keep) configurations
- `GatedResNet` forward pass across 5 gate configurations
- FLOPs counter across 3 gate configurations
- End-to-end controller across 8 step refreshes

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact commands to
reproduce each numerical result reported in the paper, including
hardware notes, random seeds, and expected output ranges. Additional
experimental measurements, expanded result tables, latency analyses,
operator-category profiling, and supporting reproducibility details are
available in
[SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md](SUPPLEMENTARY_EXPERIMENTAL_RESULTS.md).

---

## Code originality and similarity to related repositories

GDBA touches several well-trodden areas: pruning, residual gating,
test-time adaptation, stochastic depth, and analytical FLOPs counting.
This section documents which parts of the implementation are original
work and which intersect with public reference implementations, broken
down by source repository. The comparison is **technical**, not legal:
the question is whether code in this repo would look like a derivative
of an existing implementation to a reader who knows both.

Three labels are used:

- **HIGH similarity** — code structure or specific implementation
  choices clearly track the referenced source; attribution recommended.
- **MEDIUM similarity** — same algorithmic idea, but the implementation
  was written from the equations/description rather than adapted from
  the reference code.
- **Likely independent** — implementation follows from the standard
  PyTorch / NumPy / textbook treatment of a topic that the referenced
  repo also covers, with no detectable structural inheritance.

### NVlabs/Taylor_pruning (CVPR 2019, Molchanov et al.)

**Likely independent.** The conceptual link is the use of a first-order
Taylor approximation to estimate the loss change when zeroing a tensor.
In Taylor_pruning the criterion is applied per-filter (channels of a
Conv2d) via gates attached after each BN, computed inside a stateful
`pytorch_pruning` engine across a long fine-tuning loop.

In this repo (`infrastructure/activation_tracer.py` and
`domain/importance.py`) the criterion is applied per-block (entire
residual blocks of a ResNet) as a one-shot statistic `mean |a · grad|`
read from forward hooks, with no fine-tuning loop, no per-filter gate
machinery, and no persistent pruning state. The Taylor identity is
standard and a few lines long; the surrounding architecture is
materially different and not adapted from Taylor_pruning's code.

Attribution: the paper cites Molchanov et al. [16] for the Taylor-based
importance estimation used in the saliency formulation.

### DequanWang/tent (ICLR 2021, Wang et al.)

**MEDIUM similarity (conceptual only).** Tent minimises softmax
entropy of the test batch and uses its gradient to update
channel-wise affine BN parameters. This repo uses the **same entropy
loss** (`application/scoring.py::entropy_loss`) but feeds its gradient
into the saliency statistic — block parameters are **never updated**,
the model stays frozen at inference. The shared element is exactly one
expression: `-sum(p * log(p))`.

The implementation differs structurally: Tent ships a `Tent` nn.Module
wrapping an optimiser, with `configure_model` and `collect_params`
helpers for BN-affine parameters. None of that has an analogue here —
GDBA has no optimiser at inference, no parameter selection, no
model-configuration step.

Attribution: this conceptual relationship is documented here for
transparency. Tent is not currently included as a numbered reference in
the paper bibliography.

### VainF/Torch-Pruning / DepGraph (CVPR 2023, Fang et al.)

**Likely independent.** Torch-Pruning is a general structural-pruning
framework with a `DependencyGraph` that traces parameter coupling
across a model, plus various importance criteria as plug-in classes.
Our `domain/architecture.py::BlockGraph` is also a graph, but it
represents **block-level dataflow in ResNets specifically** (16 or 8
residual blocks across 4 stages, not a parameter dependency graph), is
built manually from `RESNET_STAGE_NAMES`, and uses NumPy adjacency
matrices rather than nn.Module pointers.

The analytical FLOPs counter in `infrastructure/flops_counter.py` is
also independent of Torch-Pruning's accounting: closed-form formulas
for `Conv2d`, `BatchNorm2d`, `Linear` (MACs × 2) are textbook and
appear in every FLOPs library; the structure here (a tracing forward
pass that records input shapes, then a per-block accounting pass with
`STEM_AND_HEAD_KEY`) does not mirror Torch-Pruning's API or internals.

Attribution: the paper cites DepGraph [9]. The existing citation is sufficient.

### KaimingHe/deep-residual-networks (CVPR 2016)

**Likely independent.** This repo never reimplements a ResNet:
torchvision's `resnet18` / `resnet50` are loaded directly in
`infrastructure/model_factory.py`. The only architectural intervention
is the **CIFAR stem patch** (3×3 stride-1 conv, identity maxpool) in
`_apply_cifar_stem` — six lines that follow the recipe described in
Section 4.2 of the original ResNet paper. The patch itself is one of
the canonical adjustments for running torchvision ResNets on 32×32
inputs and is documented in dozens of independent implementations.

Attribution: the paper cites ResNet [13]. The existing citation is
sufficient; the CIFAR-stem trick is not exclusive to one implementation.

### Tushar-N/blockdrop (CVPR 2018, Wu et al.)

**MEDIUM similarity (conceptual only).** BlockDrop trains a separate
policy network to predict a Bernoulli mask over residual blocks per
input via REINFORCE. The mask is then applied at inference: skipped
blocks are bypassed via the identity branch of the residual connection,
exactly as in `pytorch_wrapper.py::GatedResNet.forward`.

The forward-pass mechanic is the same — **skip the residual branch,
keep the identity**, which is how residual blocks work and is one
line of code (`continue` in the stage loop). That single mechanic is
shared by BlockDrop, SkipNet, this work, and every other paper in the
dynamic-inference family.

The selection mechanism, however, is completely different:
- BlockDrop: per-input Bernoulli policy network, RL-trained.
- This repo: graph-aware importance scoring, deterministic top-k.

No policy network exists in this repo; no REINFORCE loss; no per-input
mask prediction. There is no code inheritance, only a shared
inference-time gating concept.

Attribution: the paper cites BlockDrop [21]. Existing citation is
sufficient.

### ucbdrive/skipnet (ECCV 2018, Wang et al.)

**MEDIUM similarity (conceptual only).** SkipNet adds learned per-block
gating networks (RNN-based) that decide which residual blocks to skip
per sample. As with BlockDrop, the **forward-time skip semantic** is
the same standard "do nothing through the residual branch", and that's
the entire overlap.

This repo has no learned gates, no RNN, no per-sample decisions, no
training-time gradient through the gating decisions. Gates are produced
by a closed-form importance score and a top-k cutoff.

Attribution: the paper cites SkipNet [20]. Existing citation is
sufficient.

### yueatsprograms/Stochastic_Depth (ECCV 2016, Huang et al.)

**MEDIUM similarity.** Both implementations realise the same algorithm
from Huang et al. 2016, so the conceptual content is identical by
construction. The Torch (Lua) original applied a Bernoulli mask to the
residual branch via a custom `ResidualDrop` module; this repo
implements the same idea in PyTorch via `StochasticDepthBlock`
(`training/stochastic_depth.py`) wrapping torchvision's `BasicBlock` /
`Bottleneck`.

The wrapper specifically:
- inlines the forward of the wrapped block (conv1 → bn1 → relu →
  conv2 → bn2 → optional conv3/bn3 + downsample) rather than calling
  it, so a Bernoulli mask can be inserted between F(x) and the
  identity-add;
- uses `torchvision.ops.StochasticDepth` for the Bernoulli draw;
- applies an autocast-disabled fp32 region to avoid AMP overflow.

Of these, the first is forced by the equation; the second is a stdlib
call; the third is an idiom from torchvision's training references. No
code is copied from the Lua reference — the language and framework
differ, and the surrounding integration with our wrapper / hook system
is GDBA-specific.

Attribution: the experimental protocol relies on Stochastic Depth. The
final bibliography number for Huang et al. should be verified before a
numbered cross-reference is stated here.

### MobileNets (Howard et al., 2017)

**Likely independent.** No overlap. This repo implements dynamic block
selection for ResNet-family models and does not implement MobileNet
depthwise-separable convolutions or its architecture. The shared element
is only the broader research objective of efficient inference.

Attribution: the paper cites MobileNets [14] as related work on efficient
convolutional neural networks.

### Summary

| Repository                                | Similarity              | Action          |
|-------------------------------------------|-------------------------|-----------------|
| NVlabs/Taylor_pruning                     | Likely independent      | None (cited)    |
| DequanWang/tent                           | MEDIUM (conceptual)     | None (cited)    |
| VainF/Torch-Pruning                       | Likely independent      | None            |
| KaimingHe/deep-residual-networks          | Likely independent      | None (cited)    |
| Tushar-N/blockdrop                        | MEDIUM (conceptual)     | None (cited)    |
| ucbdrive/skipnet                          | MEDIUM (conceptual)     | None (cited)    |
| yueatsprograms/Stochastic_Depth           | MEDIUM (same algorithm) | None (cited)    |
| MobileNets (Howard et al.)                 | Likely independent      | None (cited)    |

No HIGH-similarity matches were found. The code in this repository is
either independently implemented or is one of the standard PyTorch
idioms (residual-block forward pass, hook-based output capture,
analytical Conv2d FLOPs, min-max normalisation, power iteration) whose
shape is dictated by the framework or the underlying equation, not by
any particular upstream codebase.

The current paper bibliography covers the main related references used
here: DepGraph [9], ResNet [13], MobileNets [14], Taylor-based importance
estimation [16], SkipNet [20], and BlockDrop [21]. Stochastic Depth is
used in the experimental protocol, but its final bibliography number
should be verified in the article. Tent is documented in this README as
a conceptual comparison and is not currently included as a numbered
reference in the paper bibliography. No additional license notices are
required.

This project includes ideas and partial implementation concepts
inspired by the following projects:

- Torch-Pruning (MIT License)
- Deep Residual Networks (MIT License)

Original authors retain copyright to their respective works.

---

## License

This project is licensed under the Apache License 2.0.

```bibtex
@article{antkorlgdba,
  title   = {Dynamic Activation of Neural Network Blocks Based on
             Graph Representation for Energy-Efficient Inference},
  author  = {Korol, Anatoliy Valeriyovych},
  journal = {Nauka ì tehnìka sʹogodnì},
  issn = {2786-6025},
  url = {https://perspectives.pp.ua/index.php/nts/index},
  publication_year = {2026},
}
```

---

## Disclaimer

This project is provided for research and educational purposes only.
The implementation is provided "AS IS", without warranty of any kind.
The authors do not guarantee suitability for production or
safety-critical systems.

Performance and energy-efficiency results may vary depending on:

- hardware configuration,
- datasets,
- inference settings,
- pruning levels,
- model architecture.

Users are responsible for validating the method in their own
environment.

---

## Acknowledgements

This work was inspired by prior research and open-source projects,
including:

- Taylor Pruning
- Tent
- Torch-Pruning
- Deep Residual Networks
- BlockDrop
- SkipNet
- Stochastic Depth
- MobileNets

We thank the authors of these works for advancing research in
efficient deep learning systems.