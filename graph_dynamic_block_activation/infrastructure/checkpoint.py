"""
Checkpoint repository: load and save model weights from disk.

This module isolates the disk I/O for trained models from the rest of
the pipeline. Loading a checkpoint requires:

  1. Reading the file (torch.load).
  2. Inspecting its structure (some are `state_dict` only, some wrap it
     under "model", "state_dict", or similar keys).
  3. Detecting whether the model was trained with Stochastic Depth (in
     which case the saved keys have a `block.` prefix that must be
     accounted for).
  4. Building a matching architecture and loading the weights.

Keeping all of this behind a single function (`load_checkpoint`) means
the rest of the code never has to think about checkpoint format quirks.

Save format
-----------
We use a self-describing dict:

    {
        "state_dict": <model.state_dict()>,
        "metadata": {
            "model_name": "resnet50",
            "num_classes": 100,
            "stochastic_depth_p_max": 0.1,
            "epoch": 50,
            "best_top1": 0.7074,
        }
    }

`metadata` is optional but recommended — without it, loading must guess
the architecture from the state-dict keys alone, which is brittle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Side-band information about a saved model.

    All fields are optional; older checkpoints without metadata get a
    default-constructed instance.

    Attributes
    ----------
    model_name
        Architecture identifier (e.g. "resnet18", "resnet50").
    num_classes
        Output dimensionality of the final fc layer.
    stochastic_depth_p_max
        If the model was trained with Stochastic Depth, the deepest-block
        drop probability. None means SD was disabled.
    epoch, best_top1
        Optional bookkeeping fields.
    extra
        Catch-all for any additional metadata keys the trainer wanted to
        record.
    """

    model_name: Optional[str] = None
    num_classes: Optional[int] = None
    stochastic_depth_p_max: Optional[float] = None
    epoch: Optional[int] = None
    best_top1: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
) -> None:
    """
    Persist a model's `state_dict` plus metadata to disk.

    Parameters
    ----------
    path
        Destination file. Parent directories are created if needed.
    model
        Model whose weights to save.
    metadata
        Side-band info; stored alongside the state_dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "model_name": metadata.model_name,
            "num_classes": metadata.num_classes,
            "stochastic_depth_p_max": metadata.stochastic_depth_p_max,
            "epoch": metadata.epoch,
            "best_top1": metadata.best_top1,
            "extra": dict(metadata.extra),
        },
    }
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """
    Normalize the many ways a checkpoint might wrap its state_dict.

    Common patterns we accept:
      - Bare `state_dict` (dict of name->Tensor).
      - {"state_dict": <state_dict>, "metadata": {...}}  (our format)
      - {"model": <state_dict>, ...}                       (older code)
      - {"model_state_dict": <state_dict>, ...}            (third-party)
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint payload must be a dict, got {type(payload).__name__}"
        )

    # Heuristic: if any top-level value is a Tensor, treat the whole dict
    # as a bare state_dict.
    if any(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload  # type: ignore[return-value]

    for key in ("state_dict", "model", "model_state_dict"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]

    raise ValueError(
        "Could not find a state_dict in the checkpoint. Inspected keys: "
        f"{list(payload.keys())}"
    )


def _extract_metadata(payload: Any) -> CheckpointMetadata:
    """
    Extract metadata from the payload, returning defaults if absent.

    Supports three payload conventions:

      1. New format (this codebase): metadata under a `"metadata"` key.
      2. Legacy format (pre-refactor `experiment.py`): top-level keys
         `"model"`, `"dataset"`, `"exp_cfg"`, `"train_cfg"` etc.
      3. Bare state_dict with no metadata at all.

    Format 2 is detected by the presence of a top-level `"model"` key
    whose value is a STRING (architecture name) rather than a dict
    (which would mean it's a state_dict).
    """
    if not isinstance(payload, dict):
        return CheckpointMetadata()

    # ── Case 1: new format with explicit "metadata" block ─────────────────
    raw = payload.get("metadata")
    if isinstance(raw, dict):
        return CheckpointMetadata(
            model_name=raw.get("model_name"),
            num_classes=raw.get("num_classes"),
            stochastic_depth_p_max=raw.get("stochastic_depth_p_max"),
            epoch=raw.get("epoch"),
            best_top1=raw.get("best_top1"),
            extra=dict(raw.get("extra", {})),
        )

    # ── Case 2: legacy format ─────────────────────
    # The old code put the architecture name at the top level under
    # "model" (as a string), the dataset under "dataset", and nested
    # configs under "exp_cfg" and "train_cfg".
    legacy_model = payload.get("model")
    if isinstance(legacy_model, str):
        # Legacy payload detected. Extract fields where available.
        exp_cfg = payload.get("exp_cfg") if isinstance(payload.get("exp_cfg"), dict) else {}
        train_cfg = payload.get("train_cfg") if isinstance(payload.get("train_cfg"), dict) else {}

        extra: dict[str, Any] = {}
        if "dataset" in payload:
            extra["dataset"] = payload["dataset"]
        if "image_size" in exp_cfg:
            extra["image_size"] = exp_cfg["image_size"]
        if "seed" in exp_cfg:
            extra["seed"] = exp_cfg["seed"]

        # stochastic_depth_p_max can live under train_cfg in the legacy format
        sd_p = train_cfg.get("stochastic_depth_p_max")
        # Older experiment stored 0.0 meaning "disabled" — translate to None
        # for consistency with the new convention.
        if sd_p == 0.0:
            sd_p = None

        return CheckpointMetadata(
            model_name=legacy_model,
            num_classes=exp_cfg.get("num_classes"),
            stochastic_depth_p_max=sd_p,
            epoch=payload.get("epoch"),
            best_top1=payload.get("best_val_top1_acc"),
            extra=extra,
        )

    # ── Case 3: no recoverable metadata ──────────────────────────────────
    return CheckpointMetadata()


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata]:
    """
    Load weights and metadata from a checkpoint file.

    The function does NOT construct a model — that is the caller's
    responsibility, since the right architecture depends on metadata
    that is only available after the load. The typical pattern is:

        state_dict, meta = load_checkpoint("model.pt")
        model = build_model(meta.model_name, meta.num_classes)
        if meta.stochastic_depth_p_max is not None:
            apply_stochastic_depth(model, p_max=meta.stochastic_depth_p_max)
        model.load_state_dict(state_dict)

    Parameters
    ----------
    path
        Source file.
    map_location
        Device to map tensors to during load. Use "cpu" for safety, then
        move the model to GPU after construction.

    Returns
    -------
    Tuple of (state_dict, metadata).

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the payload structure cannot be interpreted as a checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False is needed because we save the full payload dict
    # (not just tensors). Trade-off: only load files you trust.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    state_dict = _extract_state_dict(payload)
    metadata = _extract_metadata(payload)
    return state_dict, metadata


def detect_stochastic_depth_keys(state_dict: dict[str, torch.Tensor]) -> bool:
    """
    Heuristic detector: was this model trained with `StochasticDepthBlock`?

    The wrapper prefixes inner block parameter names with `block.`, so
    keys like `layer1.0.block.conv1.weight` are a tell-tale sign. We
    use this to decide whether to wrap blocks before calling
    `model.load_state_dict()`.

    Returns
    -------
    True iff at least one key contains `.block.` (the wrapper signature).
    """
    return any(".block." in key for key in state_dict.keys())