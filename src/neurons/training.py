"""One optimizer update on a contiguous stream, with user-owned state and loss."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


def train_chunk(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    state: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    *,
    max_grad_norm: float | None = None,
    **forward_kwargs,
) -> tuple[Tensor, Tensor]:
    """Run one truncated-BPTT block. Pass the returned state into the next call.

    The model returns (sequence output, final state). loss_fn can include a
    task readout; put its trainable parameters in the same optimizer. The caller
    selects the loss, optimizer, chunk length and model.train()/eval() mode.
    """
    if inputs.ndim == 0 or inputs.size(0) == 0:
        raise ValueError("A training chunk must contain at least one time step")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and nonnegative")
    optimizer.zero_grad(set_to_none=True)
    output, next_state = model(inputs, state.detach(), **forward_kwargs)
    loss = loss_fn(output)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar loss")
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    if max_grad_norm is not None:
        parameters = (p for group in optimizer.param_groups for p in group["params"])
        nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), next_state.detach()
