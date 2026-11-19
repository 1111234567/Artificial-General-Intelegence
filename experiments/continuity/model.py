"""The previous minimal dynamics, generalized to several concurrent jobs.

Only the runner decides whether to reset state. Observation pulses are injected
once, after autonomous elapsed-time integration. No task timers or memory slots
are supplied to this network.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class AgentModel(nn.Module):
    def __init__(self, input_size=96, n_jobs=4, n_actions=11, width=128,
                 kind="learned", max_step=.25):
        super().__init__()
        if kind not in {"learned", "reservoir", "gru"}:
            raise ValueError(kind)
        if min(input_size, n_jobs, n_actions, width) <= 0 or not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("positive finite model dimensions/solver step required")
        self.input_size, self.n_jobs, self.n_actions = input_size, n_jobs, n_actions
        self.width, self.kind, self.max_step = width, kind, float(max_step)
        self.encoder = nn.Linear(input_size, width, bias=False)
        self.head = nn.Linear(width, n_jobs * n_actions)
        if kind == "gru":
            self.core = nn.GRUCell(width + 1, width)
        else:
            self.recurrent = nn.Parameter(torch.randn(width, width) * (1.2 / math.sqrt(width)),
                                          requires_grad=kind == "learned")

    def initial_state(self, batch_size, device=None):
        return self.encoder.weight.new_zeros(batch_size, self.width,
                                            device=device or self.encoder.weight.device)

    def flow(self, state, dt, max_step=None):
        if self.kind == "gru":
            raise ValueError("GRU has no autonomous ODE flow")
        limit = self.max_step if max_step is None else float(max_step)
        elapsed = float(dt)
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid time increment or solver step")
        n = max(1, math.ceil(elapsed / limit))
        delta = elapsed / n
        for _ in range(n):
            first = -state + torch.nn.functional.linear(state.tanh(), self.recurrent)
            trial = state + delta * first
            second = -trial + torch.nn.functional.linear(trial.tanh(), self.recurrent)
            state = state + .5 * delta * (first + second)
        return state

    def step(self, obs, state, dt, max_step=None):
        if obs.ndim != 2 or obs.shape[-1] != self.input_size:
            raise ValueError("invalid observation shape")
        if state.shape != (len(obs), self.width):
            raise ValueError("invalid hidden-state shape")
        if not math.isfinite(float(dt)) or dt < 0:
            raise ValueError("invalid elapsed time")
        if self.kind == "gru":
            interval = obs.new_full((len(obs), 1), float(dt))
            state = self.core(torch.cat((self.encoder(obs), interval), -1), state)
            features = state
        else:
            state = self.flow(state, dt, max_step) + self.encoder(obs)
            features = state.tanh()
        return self.head(features).reshape(-1, self.n_jobs, self.n_actions), state

    def parameter_report(self):
        return {"kind": self.kind, "input_size": self.input_size,
                "n_jobs": self.n_jobs, "n_actions": self.n_actions,
                "state_scalars": self.width,
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "max_solver_step": self.max_step,
                "time_constant": None if self.kind == "gru" else 1.0}

    def config(self):
        return {name: getattr(self, name) for name in (
            "input_size", "n_jobs", "n_actions", "width", "kind", "max_step")}
