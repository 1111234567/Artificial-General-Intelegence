"""Optional directed support fusion over the prepared reward CUDA backend.

The receipt reductions and rule-weight division retain the reference Torch
operations. The fused kernel preserves every FP16 storage rounding point in
the support drive, gate, and newborn weight. Matrix scattering remains in
the inherited support-tick path; reward rates and reward writes are unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import EDGE_COOLDOWN, WEIGHT_LIMIT
from .uncapped_directed_reward_cuda_prepared import _FusedPreparedPolicy
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE


_THREADS = 128


@lru_cache(maxsize=None)
def _support_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_support_update",), device=device)


class _FusedSupportPolicy(_FusedPreparedPolicy):
    """Fuse support updates while retaining the prepared reward backend."""

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._support = (
            _support_kernels(device) if device.type == "cuda" and mode != "frozen" else None
        )
        super().__init__(birth, actors, device, mode)

    def _change_support(self, coactivity, difference, receipt):
        if self._support is None:
            return super()._change_support(coactivity, difference, receipt)

        # Keep the original reduction shapes/order, including the rounded
        # half products before each FP32-accumulating Torch sum. Combining
        # the two directions of this reduction could select a different
        # CUDA reduction path. The smaller division also stays in Torch.
        birth_coefficient = (receipt * self.rule[:, 0, None, 5:]).sum(-1)
        gate_coefficient = (receipt * self.rule[:, 1, None, 5:]).sum(-1)
        weight_scale = self.rule[:, :, 1] / WEIGHT_LIMIT
        self._support["directed_support_update"](
            (self.population * self.actors,), (_THREADS,),
            coactivity, difference, self.node_trace, self.u, self.v,
            self.rule, weight_scale, birth_coefficient, gate_coefficient,
            self.birth_positive, self.birth_negative,
            self.g, self.w, self.eligibility, self.mask, self.last_change,
            self.births, self.deaths, self.edge_count, self.edge_ticks,
            self.pair_count, self.actors, self.nodes, self.rule.size(-1),
            self.step, self.step - self.edge_span_start, EDGE_COOLDOWN,
            GATE_RATE, BIRTH_THRESHOLD, DEATH_THRESHOLD,
        )
        self.edge_span_start = self.step
