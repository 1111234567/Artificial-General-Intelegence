"""CUDA reward updates for the directed evaluator's FP16 arc state."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import GATE_INTERVAL, WEIGHT_LIMIT
from .uncapped_directed_reward import _DirectedRewardPolicy


_THREADS = 128


@lru_cache(maxsize=1)
def _kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_eligibility_only", "directed_reward_update"), device=device)


class _FusedRewardPolicy(_DirectedRewardPolicy):
    """Fuse writes and counters while preserving the support gate and GEMM."""

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._fused = _kernels(device) if device.type == "cuda" and mode == "full" else None
        super().__init__(birth, actors, device, mode)

    def _receipt_possible(self):
        # A caller with a validated tape may override this to skip reward
        # diagnostics on ticks that cannot carry the previous action's receipt.
        return True

    def _change_weights(self, coactivity, event):
        if self._fused is None:
            return super()._change_weights(coactivity, event)

        grid, block = (self.population * self.actors,), (_THREADS,)
        if not self._receipt_possible():
            self._fused["directed_eligibility_only"](
                grid, block, coactivity, self.mask, self.eligibility, self.pair_count,
            )
            return

        # The base proposal buffer is private scratch. Reuse it for rounded
        # per-arc deltas, with the existing torch.sum to preserve FP32 order.
        self._fused["directed_reward_update"](
            grid, block, coactivity, self.mask, self.eligibility, self.w, self.matrix,
            self.arc_indices, self.reward, self.reward_rate, event, self.attempted_writes,
            self.weight_updates, self.proposal, self.pair_count, self.actors, self.nodes,
            int(self.step % GATE_INTERVAL != 0), WEIGHT_LIMIT,
        )
        self.weight_change.add_(self.proposal.sum((-2, -1), dtype=torch.float32))

    def _update_matrix(self):
        if self._fused is None or self.step % GATE_INTERVAL == 0:
            # Step zero is reset. On a support tick the unchanged gate may
            # admit/retire arcs after the fused reward update, so scatter here.
            super()._update_matrix()
