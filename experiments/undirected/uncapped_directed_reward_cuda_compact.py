"""Optional active-pair reward work lists for the prepared CUDA policy.

Support still evaluates every possible arc. Reset and support changes rebuild
a fixed-capacity list of pairs having at least one active direction; ordinary
eligibility/reward updates then visit only those pairs. The original-shaped
half delta scratch and Torch reduction retain the diagnostic rounding/order.

Compose this cooperative mixin before node/support policies, for example
``class CompactCombined(_CompactRewardPolicy, _CombinedPolicy): pass``.
This module does not select a backend or change the production runner.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import GATE_INTERVAL, WEIGHT_LIMIT
from .uncapped_directed_reward_cuda_prepared import _FusedPreparedPolicy


_THREADS = 128


@lru_cache(maxsize=None)
def _compact_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, (
        "directed_compact_build_pairs", "directed_compact_eligibility_only",
        "directed_compact_reward_update",
    ), device=device)


class _CompactRewardPolicy(_FusedPreparedPolicy):
    """Visit only active endpoint pairs while keeping full per-arc state."""

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._compact = _compact_kernels(device) if device.type == "cuda" and mode == "full" else None
        self._active_pairs = self._active_pair_counts = None
        if self._compact is not None:
            # The base constructor dispatches reset() before returning. These
            # persistent buffers must already exist when that reset rebuilds.
            population, nodes = birth["node_bias"].shape
            pairs = nodes * (nodes - 1) // 2
            self._active_pairs = torch.empty(
                (population, actors, pairs), device=device, dtype=torch.int32,
            )
            self._active_pair_counts = torch.empty(
                (population, actors), device=device, dtype=torch.int32,
            )
        super().__init__(birth, actors, device, mode)

    def _rebuild_active_pairs(self):
        if self._compact is not None:
            self._compact["directed_compact_build_pairs"](
                (self.population * self.actors,), (_THREADS,), self.mask,
                self._active_pairs, self._active_pair_counts, self.pair_count,
            )

    def reset(self, birth):
        super().reset(birth)
        self._rebuild_active_pairs()

    def _change_support(self, coactivity, difference, receipt):
        super()._change_support(coactivity, difference, receipt)
        # Retirement zeroes eligibility; births start from that zero. Rebuild
        # only after both directional masks have their final support-tick state.
        self._rebuild_active_pairs()

    def _change_compact_weights(self, features, event, direct, scatter):
        grid, block = (self.population * self.actors,), (_THREADS,)
        if not self._receipt_possible():
            self._compact["directed_compact_eligibility_only"](
                grid, block, features, self.u, self.v, self.mask,
                self.eligibility, self._active_pairs, self._active_pair_counts,
                self.pair_count, self.nodes, direct,
            )
            return

        # Each compact entry writes deltas at the original two arc positions.
        # All absent positions must be zero before the unchanged Torch sum.
        self.proposal.zero_()
        self._compact["directed_compact_reward_update"](
            grid, block, features, self.u, self.v, self.mask,
            self.eligibility, self.w, self.matrix, self.arc_indices,
            self.reward, self.reward_rate, event, self.attempted_writes,
            self.weight_updates, self.proposal,
            self._active_pairs, self._active_pair_counts,
            self.pair_count, self.actors, self.nodes, direct, scatter,
            WEIGHT_LIMIT,
        )
        self.weight_change.add_(self.proposal.sum((-2, -1), dtype=torch.float32))

    def _change_prepared_weights(self, local_activity, event):
        if self._compact is None:
            return super()._change_prepared_weights(local_activity, event)
        self._change_compact_weights(local_activity, event, 1, 1)

    def _change_weights(self, coactivity, event):
        if self._compact is None:
            return super()._change_weights(coactivity, event)
        # Support ticks still use the original rounded coactivity tensor. The
        # subsequent support gate owns scattering after admitting/retiring arcs.
        self._change_compact_weights(
            coactivity, event, 0, int(self.step % GATE_INTERVAL != 0),
        )
