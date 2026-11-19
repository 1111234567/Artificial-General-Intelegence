"""Optional support fusion that also writes the complete recurrent matrix.

Support arithmetic and the Torch receipt reductions match ``_FusedSupportPolicy``.
Each arc's final weight is scattered before that same CUDA kernel finishes,
including unchanged support arcs whose weights may have changed through reward.
Only the corresponding in-act matrix refresh is skipped; reset and explicit
matrix refreshes retain the inherited behavior. This module selects no runner.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import EDGE_COOLDOWN, WEIGHT_LIMIT
from .uncapped_directed_support_fused import _FusedSupportPolicy
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE


_THREADS = 128


@lru_cache(maxsize=None)
def _scattered_support_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_support_scattered_update",), device=device)


class _ScatteredSupportPolicy(_FusedSupportPolicy):
    """Write support-tick matrix entries with the final per-arc support state."""

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._scattered_support = (
            _scattered_support_kernels(device)
            if device.type == "cuda" and mode != "frozen" else None
        )
        # Initialize before super: the base constructor calls reset(), which
        # dynamically dispatches _update_matrix(). Step zero always scatters.
        self._support_scatter_in_act = False
        self._support_scatter_step = None
        super().__init__(birth, actors, device, mode)

    def act(self, obs):
        previous = self._support_scatter_in_act
        self._support_scatter_in_act = True
        self._support_scatter_step = None
        try:
            return super().act(obs)
        finally:
            self._support_scatter_in_act = previous
            self._support_scatter_step = None

    def _change_support(self, coactivity, difference, receipt):
        if self._scattered_support is None:
            return super()._change_support(coactivity, difference, receipt)

        # Retain the original rounded half products and Torch reduction order,
        # and the independently rounded rule-weight division.
        birth_coefficient = (receipt * self.rule[:, 0, None, 5:]).sum(-1)
        gate_coefficient = (receipt * self.rule[:, 1, None, 5:]).sum(-1)
        weight_scale = self.rule[:, :, 1] / WEIGHT_LIMIT
        self._scattered_support["directed_support_scattered_update"](
            (self.population * self.actors,), (_THREADS,),
            coactivity, difference, self.node_trace, self.u, self.v,
            self.rule, weight_scale, birth_coefficient, gate_coefficient,
            self.birth_positive, self.birth_negative,
            self.g, self.w, self.matrix, self.arc_indices,
            self.eligibility, self.mask, self.last_change,
            self.births, self.deaths, self.edge_count, self.edge_ticks,
            self.pair_count, self.actors, self.nodes, self.rule.size(-1),
            self.step, self.step - self.edge_span_start, EDGE_COOLDOWN,
            GATE_RATE, BIRTH_THRESHOLD, DEATH_THRESHOLD,
        )
        self.edge_span_start = self.step
        self._support_scatter_step = self.step

    def _update_matrix(self):
        if (self.step and self._support_scatter_in_act
                and self._support_scatter_step == self.step):
            return
        super()._update_matrix()
