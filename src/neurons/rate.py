"""Trainable discrete and elapsed-time rate dynamics for PyTorch streams."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def projected_tanh_step(
    drive: Tensor, state: Tensor, recurrent: Tensor, gate: Tensor | float | None = None,
) -> Tensor:
    """Apply an already projected input and a prepared [receiver, sender] matrix.

    A supplied gate is the update amount in ``lerp(previous, candidate, gate)``.
    The caller may compute it from observations or share it across leading axes.
    """
    candidate = (drive + F.linear(state, recurrent)).tanh()
    return candidate if gate is None else torch.lerp(state, candidate, gate)


def heun_rate_flow(
    state: Tensor, recurrent: Tensor, elapsed: float, max_step: float = .25,
) -> Tensor:
    """Advance dh/dt = -h + W tanh(h) for one autonomous interval.

    ``recurrent`` is [receiver, sender] and may be reused across any number of
    observations under the same parameters. Elapsed time is a task scalar.
    """
    elapsed, max_step = float(elapsed), float(max_step)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    steps = max(1, math.ceil(elapsed / max_step))
    delta = elapsed / steps
    for _ in range(steps):
        first = -state + F.linear(state.tanh(), recurrent)
        trial = state + delta * first
        second = -trial + F.linear(trial.tanh(), recurrent)
        state = state + .5 * delta * (first + second)
    return state


def _chunk_state(inputs: Tensor, state: Tensor | None, input_size: int, recurrent: Tensor) -> Tensor:
    if inputs.ndim != 3 or inputs.shape[2] != input_size:
        raise ValueError("inputs must have shape [time, batch, input_size]")
    shape = (inputs.shape[1], recurrent.shape[0])
    if state is None:
        return recurrent.new_zeros(shape)
    if state.shape != shape:
        raise ValueError("state must have shape [batch, hidden_size]")
    return state


class TanhRNN(nn.Module):
    """Elman recurrence with an optional fixed or learned leak.

    Input chunks and output activities are time-first. The last activity is
    the next chunk's state. Forward keeps its autograd history for BPTT.
    """

    def __init__(
        self, input_size: int, hidden_size: int, *, leak: float = 1.,
        learnable_leak: bool = False, device=None, dtype=None,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size) < 1:
            raise ValueError("input_size and hidden_size must be positive")
        if not 0. < leak <= 1. or (learnable_leak and leak == 1.):
            raise ValueError("leak must be in (0, 1], or in (0, 1) when learned")
        self.input_size, self.hidden_size = input_size, hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.recurrent = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
                                      / math.sqrt(hidden_size))
        self.leak = float(leak)
        self.register_parameter("leak_logit", None)
        if learnable_leak:
            logit = math.log(leak) - math.log1p(-leak)
            self.leak_logit = nn.Parameter(self.recurrent.new_full((hidden_size,), logit))

    def initial_state(self, batch_size: int) -> Tensor:
        return self.recurrent.new_zeros(batch_size, self.hidden_size)

    def _gate(self) -> Tensor | float | None:
        if self.leak_logit is not None:
            return self.leak_logit.sigmoid()
        return None if self.leak == 1. else self.leak

    def step(self, inputs: Tensor, state: Tensor) -> Tensor:
        """Advance one sample, with optional leading batch axes."""
        return projected_tanh_step(self.input_projection(inputs), state, self.recurrent, self._gate())

    def forward(self, inputs: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        state = _chunk_state(inputs, state, self.input_size, self.recurrent)
        if not len(inputs):
            return state.new_empty((0, *state.shape)), state
        # Project the full chunk in one linear operation, as in ScoutPulseNet.
        drives = self.input_projection(inputs)
        gate = self._gate()
        activities = []
        for drive in drives.unbind():
            state = projected_tanh_step(drive, state, self.recurrent, gate)
            activities.append(state)
        return torch.stack(activities), state


class PulseRateRNN(nn.Module):
    """Heun rate dynamics with one input pulse after each elapsed interval.

    Return tanh(h) features [time, batch, hidden_size] and the raw terminal h.
    Supply elapsed times per tick or one common interval for the whole chunk.
    ``train_recurrent=False`` keeps W fixed while the input pulse map trains.
    """

    def __init__(
        self, input_size: int, hidden_size: int, *, max_step: float = .25,
        train_recurrent: bool = True, device=None, dtype=None,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size) < 1:
            raise ValueError("input_size and hidden_size must be positive")
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("max_step must be positive and finite")
        self.input_size, self.hidden_size = input_size, hidden_size
        self.max_step = float(max_step)
        self.encoder = nn.Linear(input_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.recurrent = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
                                      * (1.2 / math.sqrt(hidden_size)), requires_grad=train_recurrent)

    def initial_state(self, batch_size: int) -> Tensor:
        return self.recurrent.new_zeros(batch_size, self.hidden_size)

    def flow(self, state: Tensor, elapsed: float) -> Tensor:
        return heun_rate_flow(state, self.recurrent, elapsed, self.max_step)

    def step(self, inputs: Tensor, state: Tensor, elapsed: float) -> tuple[Tensor, Tensor]:
        state = self.flow(state, elapsed) + self.encoder(inputs)
        return state.tanh(), state

    def forward(
        self, inputs: Tensor, state: Tensor | None = None, *, elapsed: float | Sequence[float],
    ) -> tuple[Tensor, Tensor]:
        state = _chunk_state(inputs, state, self.input_size, self.recurrent)
        ticks = len(inputs)
        if isinstance(elapsed, Real):
            elapsed = (elapsed,) * ticks
        if len(elapsed) != ticks:
            raise ValueError("elapsed must provide one interval per tick")
        if ticks == 0:
            return state.new_empty((0, *state.shape)), state
        features = []
        for tick, dt in zip(inputs.unbind(), elapsed, strict=True):
            output, state = self.step(tick, state, dt)
            features.append(output)
        return torch.stack(features), state
