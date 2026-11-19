"""Minimal persistent state models with explicit elapsed time.

Continuous variants integrate dx/dt = -x + W tanh(x), then apply a single
observation impulse x <- x + B u. They differ only in whether W is trained.
There are no target firing rates, memory slots, protected units or reset gates.
This is not an implementation of Predictive Alignment: training uses BPTT.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class DynamicsModel(nn.Module):
    def __init__(self, input_size: int = 48, width: int = 128,
                 n_actions: int = 11, kind: str = "learned",
                 max_step: float = .25) -> None:
        super().__init__()
        if kind not in {"reservoir", "learned", "gru"}:
            raise ValueError(f"unknown model kind: {kind}")
        if input_size < 1 or width < 1 or n_actions < 2 or max_step <= 0:
            raise ValueError("invalid model dimensions or solver step")
        self.input_size, self.width, self.n_actions = input_size, width, n_actions
        self.kind, self.max_step = kind, float(max_step)
        # Shared initialization of encoder/head before allocating recurrent core.
        self.encoder = nn.Linear(input_size, width, bias=False)
        self.head = nn.Linear(width, 2 * n_actions)
        if kind == "gru":
            # The baseline receives identical events plus actual elapsed time.
            # Its state is updated at each common observation/action opportunity.
            self.core = nn.GRUCell(width + 1, width)
        else:
            self.recurrent = nn.Parameter(
                torch.randn(width, width) * (1.2 / math.sqrt(width)),
                requires_grad=(kind == "learned"),
            )

    def initial_state(self, batch_size: int,
                      device: str | torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.width,
                           device=device or self.encoder.weight.device,
                           dtype=self.encoder.weight.dtype)

    def flow(self, state: torch.Tensor, dt: float,
             max_step: float | None = None) -> torch.Tensor:
        """Heun integration of autonomous flow, with no repeated event input.

        Time constants use task time units (tau=1). Halving max_step changes
        numerical accuracy only; it creates no additional actions/observations.
        """
        if self.kind == "gru":
            raise ValueError("GRU has discrete updates, not an autonomous ODE")
        limit = self.max_step if max_step is None else float(max_step)
        elapsed = float(dt)
        if not math.isfinite(elapsed) or elapsed < 0 or limit <= 0 or not math.isfinite(limit):
            raise ValueError("elapsed time must be finite/nonnegative; solver step positive")
        n = max(1, math.ceil(elapsed / limit))
        delta = elapsed / n
        for _ in range(n):
            slope = -state + torch.nn.functional.linear(state.tanh(), self.recurrent)
            trial = state + delta * slope
            slope2 = -trial + torch.nn.functional.linear(trial.tanh(), self.recurrent)
            state = state + (.5 * delta) * (slope + slope2)
        return state

    def step(self, obs: torch.Tensor, state: torch.Tensor, dt: float,
             max_step: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or obs.shape[-1] != self.input_size:
            raise ValueError("observation shape must be [batch,input_size]")
        if state.shape != (obs.shape[0], self.width):
            raise ValueError("state shape must be [batch,width]")
        if not math.isfinite(float(dt)) or dt < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        if self.kind == "gru":
            elapsed = obs.new_full((obs.shape[0], 1), float(dt))
            state = self.core(torch.cat((self.encoder(obs), elapsed), -1), state)
            readout = state
        else:
            state = self.flow(state, dt, max_step=max_step) + self.encoder(obs)
            readout = state.tanh()
        return self.head(readout).view(-1, 2, self.n_actions), state

    def parameter_report(self) -> dict[str, int | str | float | None]:
        return {"kind": self.kind, "state_scalars": self.width,
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "max_solver_step": self.max_step,
                "time_constant": None if self.kind == "gru" else 1.0}
