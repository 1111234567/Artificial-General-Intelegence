"""Scalar fast synapses with receipt-gated writes and read-before-write predictions.

Sequences use keys [..., time, width] and targets/masks [..., time]. A boolean
mask gates the receipt's write, independently of whether a prediction is read
at that position. Query keys must be finite, including at hidden receipts.
"""

from typing import NamedTuple

import torch
from torch import Tensor


class RLSState(NamedTuple):
    weights: Tensor
    inverse_covariance: Tensor


def read_fast_weights(weights: Tensor, keys: Tensor, *, prior: float | Tensor = 0.) -> Tensor:
    """Read a scalar from the current weights without changing them."""
    return prior + (weights * keys).sum(dim=-1)


def _observed(keys: Tensor, targets: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if keys.ndim < 1 or keys.shape[-1] == 0 or targets.shape != keys.shape[:-1]:
        raise ValueError("targets must have one scalar per nonempty key")
    targets = targets.to(dtype=keys.dtype)
    if mask is None:
        return keys, targets
    if mask.dtype != torch.bool or mask.shape != targets.shape:
        raise ValueError("mask must have one boolean per target")
    # Mask inputs before products and solves, including hidden NaN placeholders.
    return torch.where(mask.unsqueeze(-1), keys, 0), torch.where(mask, targets, 0)


def _ridge_identity(keys: Tensor, ridge: float | Tensor, size: int) -> tuple[Tensor, Tensor]:
    value = torch.as_tensor(ridge, dtype=keys.dtype, device=keys.device)
    if value.ndim or not (torch.isfinite(value) & (value > 0)):
        raise ValueError("ridge must be a positive finite scalar")
    return torch.eye(size, dtype=keys.dtype, device=keys.device), value


def initial_rls_state(example_key: Tensor, ridge: float | Tensor) -> RLSState:
    """Birth state with weights zero and inverse covariance I / ridge."""
    if example_key.ndim < 1 or example_key.shape[-1] == 0:
        raise ValueError("example_key must have a nonempty width axis")
    width = example_key.shape[-1]
    eye, value = _ridge_identity(example_key, ridge, width)
    inverse = (eye / value).expand(*example_key.shape, width).clone()
    return RLSState(torch.zeros_like(example_key), inverse)


def rls_update(
    state: RLSState, key: Tensor, target: Tensor, *, mask: Tensor | None = None,
) -> RLSState:
    """Apply one newly received sample to each lane. Read before calling this.

    For delayed feedback, supply the write key saved at the event. The residual
    is computed with the weights present when the receipt is applied.
    """
    key, target = _observed(key, target, mask)
    if state.weights.shape != key.shape or state.inverse_covariance.shape != (*key.shape, key.shape[-1]):
        raise ValueError("state and write keys must have equal batch and width axes")
    projected = (state.inverse_covariance @ key.unsqueeze(-1)).squeeze(-1)
    direction = projected / (1. + (projected * key).sum(dim=-1, keepdim=True))
    error = target - read_fast_weights(state.weights, key)
    covariance = state.inverse_covariance - direction.unsqueeze(-1) * projected.unsqueeze(-2)
    weights = state.weights + error.unsqueeze(-1) * direction
    return RLSState(weights, covariance)


def rls_prefix_predictions(
    keys: Tensor, targets: Tensor, ridge: float | Tensor, *,
    queries: Tensor | None = None, mask: Tensor | None = None,
) -> Tensor:
    """Use a single Cholesky factor to read all zero-state RLS prefixes.

    Query t uses only receipts strictly before t. Queries default to the write
    keys and can instead be supplied independently with the same shape.
    """
    if keys.ndim < 2:
        raise ValueError("keys must have shape [..., time, width]")
    writes, labels = _observed(keys, targets, mask)
    if queries is not None and queries.shape != keys.shape:
        raise ValueError("queries must have the same shape as keys")
    eye, value = _ridge_identity(keys, ridge, keys.shape[-2])
    if keys.shape[-2] == 0:
        return labels
    lower = torch.linalg.cholesky(writes @ writes.transpose(-1, -2) + value * eye)
    innovations = torch.linalg.solve_triangular(lower, labels.unsqueeze(-1), upper=False)
    if queries is None and mask is None:
        # The off-diagonal of L already gives the projected same-key queries.
        projected = lower
    else:
        reads = keys if queries is None else queries
        cross = writes @ reads.transpose(-1, -2)
        projected = torch.linalg.solve_triangular(lower, cross, upper=False).transpose(-1, -2)
    return (projected.tril(-1) @ innovations).squeeze(-1)


def delta_step(
    weights: Tensor, key: Tensor, target: Tensor, *, rate: float | Tensor,
    prior: float | Tensor = 0., mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Read from one normalized key before applying its scalar delta update."""
    if weights.shape != key.shape:
        raise ValueError("weights and write keys must have the same shape")
    write, label = _observed(key, target, mask)
    prediction = read_fast_weights(weights, key, prior=prior)
    write_prediction = prediction if mask is None else read_fast_weights(weights, write, prior=prior)
    next_weights = weights + (rate * (label - write_prediction)).unsqueeze(-1) * write
    return prediction, next_weights


def delta_prefix_predictions(
    keys: Tensor, targets: Tensor, *, rate: float | Tensor, prior: float | Tensor = 0.,
    mask: Tensor | None = None, weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Read before writing, using the strictly lower delta-system solve.

    Keys are already normalized by the caller, as in ring_metric. Pass the
    terminal weights into the next chunk to continue learning.
    """
    if keys.ndim < 2:
        raise ValueError("keys must have shape [..., time, width]")
    writes, labels = _observed(keys, targets, mask)
    state_shape = (*keys.shape[:-2], keys.shape[-1])
    if weights is None:
        weights = keys.new_zeros(state_shape)
    if weights.shape != state_shape:
        raise ValueError("weights must have shape [..., width]")
    ticks = keys.shape[-2]
    if not ticks:
        return labels, weights
    base = read_fast_weights(weights.unsqueeze(-2), writes, prior=prior)
    gram = (writes @ writes.transpose(-1, -2)).tril(-1) * rate
    identity = torch.eye(ticks, dtype=keys.dtype, device=keys.device)
    errors = torch.linalg.solve_triangular(gram + identity, (labels - base).unsqueeze(-1),
                                          upper=False, unitriangular=True)
    updates = (rate * errors) * writes
    after = weights.unsqueeze(-2) + updates.cumsum(dim=-2)
    before = torch.cat((weights.unsqueeze(-2), after[..., :-1, :]), dim=-2)
    predictions = read_fast_weights(before, keys, prior=prior)
    return predictions, after[..., -1, :]
