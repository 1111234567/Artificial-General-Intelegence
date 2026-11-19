"""Optional direct-local-activity CUDA updates; the original backend is unchanged.

Support ticks retain the original PyTorch feature preparation. Other full-mode
ticks keep PyTorch's local-activity tanh and round the endpoint product inside
the eligibility/reward kernel, avoiding both index_selects and the large
coactivity tensor. This backend is intended for numerical parity validation
before selecting it for a run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import ACTIONS, GATE_INTERVAL, INPUTS, JOBS, WEIGHT_LIMIT
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_directed_reward_fused_run import _FusedTrainingPolicy
from .uncapped_reward_precision import _ACK, _REJ


_THREADS = 128


@lru_cache(maxsize=1)
def _prepared_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, (
        "directed_prepare_trace", "directed_prepared_eligibility_only",
        "directed_prepared_reward_update",
    ), device=device)


class _FusedPreparedPolicy(_FusedTrainingPolicy):
    """Training-only static receipt skip plus direct endpoint-product loading."""

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._prepared = _prepared_kernels(device) if device.type == "cuda" and mode == "full" else None
        super().__init__(birth, actors, device, mode)

    def _prepare_trace(self, local_activity):
        count = self.population * self.actors * self.nodes
        self._prepared["directed_prepare_trace"](
            ((count + _THREADS - 1) // _THREADS,), (_THREADS,),
            local_activity, self.node_trace, count,
        )

    def _change_prepared_weights(self, local_activity, event):
        grid, block = (self.population * self.actors,), (_THREADS,)
        if not self._receipt_possible():
            self._prepared["directed_prepared_eligibility_only"](
                grid, block, local_activity, self.u, self.v, self.mask,
                self.eligibility, self.pair_count, self.nodes,
            )
            return
        self._prepared["directed_prepared_reward_update"](
            grid, block, local_activity, self.u, self.v, self.mask,
            self.eligibility, self.w, self.matrix, self.arc_indices,
            self.reward, self.reward_rate, event, self.attempted_writes,
            self.weight_updates, self.proposal, self.pair_count, self.actors,
            self.nodes, 1, WEIGHT_LIMIT,
        )
        # Keep the existing rounded half deltas and PyTorch FP32 reduction.
        self.weight_change.add_(self.proposal.sum((-2, -1), dtype=torch.float32))

    def act(self, obs):
        if self._prepared is None or not self.step or self.step % GATE_INTERVAL == 0:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype)
        port = obs[..., :FEATURES_PER_SLOT]
        self.reward.copy_((port[..., _ACK] - port[..., _REJ]) * self.pending)
        event = self.reward.ne(0)
        self.reward_events.add_(event)
        # Retaining these two Torch operations preserves the original half
        # rounding and CUDA tanhf path. Only node-trace update is fused here.
        local_activity = (4. * self.h).tanh()
        self._prepare_trace(local_activity)
        self._change_prepared_weights(local_activity, event)
        self._update_matrix()

        self.tried.logical_and_(~port[..., NEW].bool())
        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        self.h.mul_(.75).add_(current.reshape_as(self.h) + self.bias, alpha=.25)
        self.h[..., :INPUTS].add_(obs, alpha=.25)
        action_ports = self.h[..., INPUTS:INPUTS + JOBS * ACTIONS]
        actions = action_ports.reshape(self.population, self.actors, JOBS, ACTIONS).argmax(-1)
        primary = actions[..., 0]
        self.pending.copy_(port[..., READY].bool() & ~self.tried
                           & primary.ge(FIRST_FINISH) & primary.lt(N_ACTIONS))
        self.tried.logical_or_(self.pending)
        torch.tanh(self.h, out=self.activity)
        self.step += 1
        return actions


class _FusedPreparedTorchTracePolicy(_FusedPreparedPolicy):
    """Diagnostic variant isolating endpoint gathers from trace preparation."""

    def _prepare_trace(self, local_activity):
        self.node_trace.mul_(.95).add_(local_activity, alpha=.05)
