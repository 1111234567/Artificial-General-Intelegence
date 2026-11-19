"""Function-preserving extensions of the existing dense nonlinear recurrent field.

The directed graph remains complete in every arm.  The interaction arm adds
recipient/source products inside the vector field, rather than new state layers.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.continuity.model import AgentModel


ARMS = ("baseline", "source_square", "interaction")


class LoopModel(AgentModel):
    @classmethod
    def attach(cls, model: AgentModel, arm: str):
        """Reuse every inherited Parameter object; allocating zeros draws no RNG."""
        if arm not in ARMS:
            raise ValueError("unknown loop arm")
        if type(model) is not AgentModel or model.kind != "learned":
            raise ValueError("an unchanged learned AgentModel is required")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in ("input_size", "n_jobs", "n_actions", "width", "kind", "max_step"):
            setattr(result, name, getattr(model, name))
        result.arm = arm
        result.encoder, result.head = model.encoder, model.head
        result.recurrent = model.recurrent
        if arm != "baseline":
            result.branch = nn.Parameter(torch.zeros_like(model.recurrent))
        result.train(model.training)
        return result

    def branch_forcing(self, state):
        return self._branch_from_rate(state.tanh())

    def _branch_from_rate(self, rate):
        if self.arm == "baseline":
            return torch.zeros_like(rate)
        if self.arm == "source_square":
            return F.linear(rate.square(), self.branch)
        return rate * F.linear(rate, self.branch)

    def vector_field(self, state):
        rate = state.tanh()
        original = -state + F.linear(rate, self.recurrent)
        return original if self.arm == "baseline" else original + self._branch_from_rate(rate)

    def flow(self, state, dt, max_step=None):
        if self.arm == "baseline":
            # Preserve the original autograd graph construction and floating
            # point accumulation order, as well as its forward function.
            return super().flow(state, dt, max_step)
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
        return {**super().config(), "arm": self.arm}

    def parameter_report(self):
        result = super().parameter_report()
        extended = self.arm != "baseline"
        result.update(arm=self.arm, recurrent_parameters=self.recurrent.numel(),
            branch_parameters=self.width ** 2 if extended else 0,
            recurrent_mac_per_field=self.width ** 2 * (2 if extended else 1),
            recurrent_matrix_products_per_field=2 if extended else 1,
            tanh_scalars_per_field=self.width,
            branch_scalar_products_per_field=self.width if extended else 0,
            persistent_state_scalars=self.width,
            topology="The same complete directed state graph in all arms.",
            formula={"baseline": "-h + W tanh(h)",
                "source_square": "-h + W r + V (r*r), r=tanh(h)",
                "interaction": "-h + W r + r*(V r), r=tanh(h)"}[self.arm])
        return result

    @torch.no_grad()
    def branch_metrics(self, state):
        rate = state.tanh()
        rates64 = rate.double()
        old = F.linear(rate, self.recurrent)
        added = self.branch_forcing(state)
        rms = lambda value: float(value.double().square().mean().sqrt())
        # E_ij[r_i*r_j] and E_ij[(r_i*r_j)^2], computed without B*H*H storage.
        sums = rates64.sum(-1)
        sum_squares = rates64.square().sum(-1)
        pair_mean = float((sums.square() / self.width ** 2).mean())
        pair_rms = float((sum_squares.square() / self.width ** 2).mean().sqrt())
        off_count = self.width * (self.width - 1)
        off_mean = None if not off_count else float(((sums.square() - sum_squares) / off_count).mean())
        off_rms = None if not off_count else float(((sum_squares.square() - rates64.pow(4).sum(-1))
                                                    / off_count).mean().clamp_min(0).sqrt())
        return {"inherited_forcing_rms": rms(old), "branch_forcing_rms": rms(added),
            "rate_mean": float(rates64.mean()), "rate_rms": rms(rate),
            "source_square_mean": float(rates64.square().mean()),
            "source_square_rms": rms(rate.square()),
            "recipient_source_product_mean": pair_mean,
            "recipient_source_product_rms": pair_rms,
            "off_diagonal_recipient_source_product_mean": off_mean,
            "off_diagonal_recipient_source_product_rms": off_rms,
            "saturated_fraction_abs_rate_ge_095": float((rate.abs() >= .95).double().mean()),
            "branch_weight_l2": 0. if self.arm == "baseline" else float(self.branch.double().norm())}
