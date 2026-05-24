"""
Activation tracer: PyTorch hook machinery for collecting per-block
output tensors and their gradients.

The score computation in `domain.importance` needs two data-dependent
signals for every residual block:

  - **Activation magnitude** — the mean absolute value of the block's
    output tensor on a forward pass.
  - **Saliency** — the mean absolute value of (output * grad), where grad
    is the gradient of a target loss with respect to that output.

Both values are simple statistics over tensors, but obtaining the
tensors themselves requires PyTorch-specific machinery: forward hooks
to capture the outputs, `retain_grad()` to keep gradients available
after backward, and careful book-keeping to remove the hooks afterwards
(an un-removed hook would leak memory and slow down later inference).

This module encapsulates all that machinery behind a single context
manager, so the application layer can write:

    with BlockActivationTracer(model, block_ids) as tracer:
        logits = model(batch)
        loss = entropy_loss(logits)
        loss.backward()
    activations = tracer.compute_activation_magnitudes()
    saliencies = tracer.compute_saliencies()

The tracer guarantees cleanup even if an exception is raised inside the
`with` block.
"""

from __future__ import annotations

from types import TracebackType
from typing import Iterable

import torch
import torch.nn as nn

from ..domain.architecture import BlockId


# ─────────────────────────────────────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────────────────────────────────────

class BlockActivationTracer:
    """
    Context manager that captures the output tensors of named blocks
    during a forward pass and (optionally) their gradients after backward.

    Usage
    -----
        with BlockActivationTracer(model, block_ids=["layer1.0", ...]) as t:
            logits = model(x)
            loss.backward()        # only needed if you want saliency
        magnitudes = t.compute_activation_magnitudes()
        saliencies = t.compute_saliencies()

    Implementation
    --------------
    On `__enter__` the tracer walks the model's named submodules and
    registers a single forward hook on each block of interest. The hook
    calls `retain_grad()` on the output tensor and stores a reference
    to it.

    `retain_grad()` is critical: PyTorch normally discards gradients of
    non-leaf tensors after backward, which is exactly what block outputs
    are. Without `retain_grad()`, `output.grad` would be `None` even
    after a backward pass.

    On `__exit__` all hooks are removed and the references are cleared,
    freeing the GPU memory held by the cached tensors.

    The class is NOT re-entrant: a single instance can be used for one
    `with` block. After exit, attempting to read the magnitudes raises
    a `RuntimeError`.

    Parameters
    ----------
    model
        The network containing the blocks.
    block_ids
        Iterable of dotted block identifiers (e.g. "layer2.3") to hook.
    """

    def __init__(self, model: nn.Module, block_ids: Iterable[BlockId]) -> None:
        self._model = model
        self._block_ids = tuple(block_ids)
        self._block_id_set = set(self._block_ids)

        # Populated on __enter__, cleared on __exit__.
        self._captured_outputs: dict[BlockId, torch.Tensor] = {}
        self._hook_handles: list = []
        self._entered = False
        self._exited = False

    # ── Context-manager protocol ──────────────────────────────────────────

    def __enter__(self) -> "BlockActivationTracer":
        if self._entered:
            raise RuntimeError(
                "BlockActivationTracer is not re-entrant; create a new "
                "instance for each forward pass."
            )
        self._entered = True
        self._captured_outputs.clear()

        # Walk named_modules() once; this is O(depth) and yields each
        # submodule's dotted path, which we compare against the requested
        # block IDs.
        for name, module in self._model.named_modules():
            if name in self._block_id_set:
                handle = module.register_forward_hook(self._make_hook(name))
                self._hook_handles.append(handle)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        # Always remove hooks, even on exception, to avoid leaks.
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._exited = True
        # Returning False lets exceptions propagate normally.
        return False

    # ── Hook factory ──────────────────────────────────────────────────────

    def _make_hook(self, block_id: BlockId):
        """
        Create a closure that captures the block_id and stores the output
        tensor when the hook fires.
        """
        def _hook(_module, _inputs, output):
            # Some block types may return tuples (rare for residual blocks
            # but possible after wrapping). Take the first tensor element.
            if isinstance(output, (tuple, list)):
                if not output or not isinstance(output[0], torch.Tensor):
                    return
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return

            # `retain_grad()` keeps `.grad` populated after backward for
            # this non-leaf tensor. Required for saliency.
            if output.requires_grad:
                output.retain_grad()

            self._captured_outputs[block_id] = output

        return _hook

    # ── Statistics computation ────────────────────────────────────────────

    def compute_activation_magnitudes(self) -> dict[BlockId, float]:
        """
        Return per-block mean absolute activation magnitude.

        For each captured output tensor, returns

            mean_{batch, channels, spatial} |output|

        as a Python float. Blocks for which no tensor was captured (i.e.
        the block didn't run or wasn't reached) get value 0.0.

        Returns
        -------
        Dict from block_id to a non-negative float.
        """
        self._require_exited_or_inside()

        magnitudes: dict[BlockId, float] = {}
        for block_id in self._block_ids:
            tensor = self._captured_outputs.get(block_id)
            if tensor is None:
                magnitudes[block_id] = 0.0
                continue
            magnitudes[block_id] = float(tensor.detach().abs().mean().item())

        return magnitudes

    def compute_saliencies(self) -> dict[BlockId, float]:
        """
        Return per-block mean absolute (output * grad), the saliency proxy.

        This is the first-order Taylor approximation of how much the loss
        would change if the block's output were zeroed. Requires a
        backward pass to have been called before this method.

        For each captured output tensor `a` with gradient `g`:

            mean_{batch, channels, spatial} |a * g|

        Blocks without a gradient (e.g. their output didn't influence the
        loss) get value 0.0.

        Returns
        -------
        Dict from block_id to a non-negative float.
        """
        self._require_exited_or_inside()

        saliencies: dict[BlockId, float] = {}
        for block_id in self._block_ids:
            tensor = self._captured_outputs.get(block_id)
            if tensor is None:
                saliencies[block_id] = 0.0
                continue
            grad = tensor.grad
            if grad is None:
                saliencies[block_id] = 0.0
                continue
            saliencies[block_id] = float(
                (tensor.detach() * grad.detach()).abs().mean().item()
            )

        return saliencies

    # ── Internal guards ───────────────────────────────────────────────────

    def _require_exited_or_inside(self) -> None:
        """Allow reading stats either inside the with-block or after."""
        if not self._entered:
            raise RuntimeError(
                "BlockActivationTracer not yet entered; use `with` first."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers for callers who don't want a context manager directly
# ─────────────────────────────────────────────────────────────────────────────

def collect_block_signals(
    model: nn.Module,
    block_ids: Iterable[BlockId],
    forward_with_backward,
) -> tuple[dict[BlockId, float], dict[BlockId, float]]:
    """
    One-shot helper: run a forward+backward pass and return both
    activation magnitudes and saliencies.

    Parameters
    ----------
    model
        Network whose blocks to trace.
    block_ids
        Block identifiers to capture.
    forward_with_backward
        Callable that does the forward and backward pass. It receives no
        arguments; the caller closes over the input batch, loss function,
        and any other context. Must call `.backward()` on a loss tensor.

    Returns
    -------
    (activation_magnitudes, saliencies) — two dicts indexed by block_id.

    Example
    -------
        def run():
            logits = model(batch)
            loss = entropy_loss(logits)
            model.zero_grad()
            loss.backward()
        a, s = collect_block_signals(model, ids, run)
    """
    with BlockActivationTracer(model, block_ids) as tracer:
        forward_with_backward()
        magnitudes = tracer.compute_activation_magnitudes()
        saliencies = tracer.compute_saliencies()

    return magnitudes, saliencies
