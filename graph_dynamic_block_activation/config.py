"""
Top-level experiment configuration.

After the refactoring, configuration is *distributed* across layers
rather than concentrated in a single megaclass:

  - Per-experiment dataset/model/device settings ........ ExperimentConfig (here)
  - Training-loop hyperparameters ........................ TrainingConfig
                                                           (training.trainer)
  - GDBA importance formula coefficients .................. ImportanceWeights
                                                           (domain.importance)
  - GDBA selector settings ................................ TopKSelector args
                                                           (domain.selection)
  - GDBA controller settings .............................. GDBAController.build()
                                                           args (application.controller)

This decomposition reflects the single-responsibility principle: each
component owns the parameters it actually uses.

What remains here
-----------------
`ExperimentConfig` is the only top-level config because it describes
the *environment* of an experiment — which dataset, which model
architecture, what device — rather than algorithm behavior. CLI scripts
populate it from command-line arguments and pass it down to the data
and model factories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

# Names we explicitly support. Restricting to a Literal makes typos
# fail at import time rather than producing cryptic runtime errors.
DatasetName = Literal["cifar10", "cifar100"]
ModelName = Literal["resnet18", "resnet34", "resnet50", "resnet101"]
DeviceSpec = Literal["cpu", "cuda"]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Environment settings for a single experiment run.

    Used by CLI scripts to bundle command-line arguments before passing
    them to the data and model factories. Frozen so that experiment
    parameters can't be mutated mid-run by accident.

    Fields
    ------
    model
        Architecture identifier. See ModelName for the supported set.
    dataset
        Dataset identifier (cifar10 or cifar100).
    data_root
        Filesystem path where dataset files are stored. The data
        factory downloads them here if absent.
    num_classes
        Output dimensionality of the classifier. Should match the
        dataset: 10 for CIFAR-10, 100 for CIFAR-100.
    image_size
        Input image resolution in pixels. 32 is native CIFAR; larger
        values upsample for ImageNet-style backbones.
    device
        Device specification ("cpu" or "cuda"). Use `select_device()`
        from utils to convert to a torch.device.
    batch_size
        Mini-batch size for data loaders.
    num_workers
        Number of worker processes per DataLoader.
    seed
        RNG seed for stratified split, model init, augmentation.
    val_ratio
        Fraction of the training set used as validation.
    output_dir
        Directory where checkpoints, logs, and result JSONs are saved.
    run_name
        Subdirectory name within output_dir for this specific run.
    """

    model: ModelName
    dataset: DatasetName
    num_classes: int

    data_root: str = "./data"
    image_size: int = 32

    device: DeviceSpec = "cuda"
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 4
    seed: int = DEFAULT_SEED
    val_ratio: float = 0.1

    output_dir: str = "./outputs"
    run_name: str = "default_run"

    def __post_init__(self) -> None:
        # Cross-field sanity checks.
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {self.num_classes}"
            )
        if self.dataset == "cifar10" and self.num_classes != 10:
            raise ValueError(
                f"cifar10 dataset has 10 classes; got num_classes={self.num_classes}"
            )
        if self.dataset == "cifar100" and self.num_classes != 100:
            raise ValueError(
                f"cifar100 dataset has 100 classes; got num_classes={self.num_classes}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be >= 0, got {self.num_workers}"
            )
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError(
                f"val_ratio must be in [0, 1); got {self.val_ratio}"
            )
        if self.image_size < 1:
            raise ValueError(
                f"image_size must be >= 1, got {self.image_size}"
            )
