"""Matched two-bank continuous fields with different nonlinear organization.

Write the persistent state as h=(x,y), with equally sized banks, and partition
the one recurrent Parameter into [[A,B],[C,D]].  The three vector fields are:

plain:
    dx = -x + A tanh(x) + B tanh(y)
    dy = -y + D tanh(y) + C tanh(x)
parallel:
    dx = -x + A tanh(x) + tanh(B tanh(y))
    dy = -y + D tanh(y) + tanh(C tanh(x))
coupled:
    dx = -x + A tanh(x + B tanh(y))
    dy = -y + D tanh(y + C tanh(x))

The banks have no assigned task, memory, or timing roles.  All modes have the
same persistent state, parameters, and matrix MAC budget.  Parallel and coupled
also have the same number of tanh evaluations.  Coupled combines the two banks
inside a further nonlinearity; parallel keeps their contributions separable.
Zero cross-bank blocks B=C=0 give all three modes the same birth function.

This changes the nonlinear transition, not the number of graph-theoretic cycles
in an otherwise complete directed state graph.  The original observation pulse,
readout, leak, physical time, and Heun integration convention are retained.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from experiments.continuity.model import AgentModel


FIELD_MODES = ("plain", "parallel", "coupled")


class OrganizationModel(AgentModel):
    def __init__(self, input_size=112, n_jobs=4, n_actions=11, width=128,
                 kind="learned", max_step=.25, field_mode="plain"):
        if field_mode not in FIELD_MODES:
            raise ValueError("unknown organization field mode")
        if kind != "learned":
            raise ValueError("organization fields require learned recurrence")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0 or width % 2:
            raise ValueError("organization fields require a positive even width")
        super().__init__(input_size=input_size, n_jobs=n_jobs, n_actions=n_actions,
                         width=width, kind=kind, max_step=max_step)
        self.field_mode = field_mode

    def vector_field(self, state):
        if state.ndim < 1 or state.shape[-1] != self.width:
            raise ValueError("invalid hidden-state shape")
        half = self.width // 2
        x, y = state.split(half, dim=-1)
        rx, ry = x.tanh(), y.tanh()
        a = self.recurrent[:half, :half]
        b = self.recurrent[:half, half:]
        c = self.recurrent[half:, :half]
        d = self.recurrent[half:, half:]
        cross_x, cross_y = F.linear(ry, b), F.linear(rx, c)
        if self.field_mode == "coupled":
            dx = -x + F.linear((x + cross_x).tanh(), a)
            dy = -y + F.linear((y + cross_y).tanh(), d)
        else:
            if self.field_mode == "parallel":
                cross_x, cross_y = cross_x.tanh(), cross_y.tanh()
            dx = -x + F.linear(rx, a) + cross_x
            dy = -y + F.linear(ry, d) + cross_y
        return torch.cat((dx, dy), dim=-1)

    def flow(self, state, dt, max_step=None):
        # The frozen parent's Heun implementation embeds its original field.
        # Keep that integration scheme while evaluating this selected field.
        limit = self.max_step if max_step is None else float(max_step)
        elapsed = float(dt)
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid time increment or solver step")
        n = max(1, math.ceil(elapsed / limit))
        delta = elapsed / n
        for _ in range(n):
            first = self.vector_field(state)
            trial = state + delta * first
            second = self.vector_field(trial)
            state = state + .5 * delta * (first + second)
        return state

    def config(self):
        return {**super().config(), "field_mode": self.field_mode}

    def parameter_report(self):
        report = super().parameter_report()
        report.update(
            field_mode=self.field_mode,
            bank_count=2,
            bank_width=self.width // 2,
            persistent_state_scalars=self.width,
            recurrent_parameters=self.recurrent.numel(),
            recurrent_mac_per_field=self.width ** 2,
            recurrent_matrix_products_per_field=4,
            tanh_scalars_per_field=self.width * (1 if self.field_mode == "plain" else 2),
            field_evaluations_per_heun_substep=2,
            bank_roles="Symmetric banks; no assigned memory, timer, job, or family roles.",
            topology="Same available complete directed state graph; different nonlinear transition organization.",
            formula={
                "plain": "dx=-x+A tanh(x)+B tanh(y); dy=-y+D tanh(y)+C tanh(x)",
                "parallel": "dx=-x+A tanh(x)+tanh(B tanh(y)); dy=-y+D tanh(y)+tanh(C tanh(x))",
                "coupled": "dx=-x+A tanh(x+B tanh(y)); dy=-y+D tanh(y+C tanh(x))",
            }[self.field_mode],
        )
        return report
