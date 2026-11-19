"""Fuse CUDA actor preparation with encoded eligibility and reward updates."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from .evolved_plasticity import GATE_INTERVAL, INPUTS, JOBS, WEIGHT_LIMIT
from .uncapped_directed_reward_encoded_pairs import _encoded_source
from .uncapped_private_symbol_fused_run import _FusedPolicy
from .uncapped_reward_precision import _ACK, _REJ


_THREADS = 128


@lru_cache(maxsize=None)
def _prepare_reward_kernel(device):
    definitions = f"#define NODE_INPUTS {INPUTS}\n#define NODE_ACK {_ACK}\n#define NODE_REJ {_REJ}\n"
    source = definitions + _encoded_source() + "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    names = "directed_reward_prepare_update", "directed_reward_prepare_update_and_sum"
    return compile_kernels(source, names, device=device)


class _RewardPrepareMixin:
    """Use one CTA per actor for FP16 node preparation and compact arc updates."""

    _reward_prepare_threads = _THREADS
    _fuse_reward_diagnostic = False

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        self._reward_prepare = None
        if (getattr(self, "_encoded_build", None) is not None and self._node is not None
                and self._fuse_prepare_tanh and self.nodes == 96 and self.pair_count == 4560):
            kernels = _prepare_reward_kernel(self.device)
            name = ("directed_reward_prepare_update_and_sum" if self._fuse_reward_diagnostic
                    else "directed_reward_prepare_update")
            self._reward_prepare = kernels[name]

    def _prepare_and_change_weights(self, obs, *, save_local=True):
        grid = (self.population * self.actors,)
        local, event = self._node_local_activity, self._node_event
        if self._reward_prepare is None:
            self._node["directed_node_prepare"](
                grid, (_THREADS,), self.h, obs, self.pending, self.reward, event,
                self.reward_events, self.node_trace, local, self.nodes, int(self._fuse_prepare_tanh),
            )
            if not self._fuse_prepare_tanh:
                local = (4. * self.h).tanh()
                self._prepare_trace(local)
            self._change_prepared_weights(local, event)
            return local

        possible = int(self._receipt_possible())
        self._reward_prepare(
            grid, (self._reward_prepare_threads,), self.h, obs, self.pending, self.reward, event,
            self.reward_events, self.node_trace, local, self.eligibility, self.w, self.matrix,
            self.reward_rate, self.attempted_writes, self.weight_updates, self.proposal,
            self._sparse_nonfinite, self._active_pairs, self._active_pair_counts,
            self.mask, self.weight_change, self.actors, possible, int(save_local), WEIGHT_LIMIT,
        )
        if possible and not self._fuse_reward_diagnostic:
            self._sparse["directed_sparse_diagnostic_sum"](
                grid, (32, 16), self.proposal, self.mask, self.w, event,
                self._sparse_nonfinite, self.weight_change,
            )
        return local


class RewardPreparePolicy(_RewardPrepareMixin, _FusedPolicy):
    """Exercise the fused preparation on ordinary training ticks."""

    def act(self, obs):
        if self._reward_prepare is None or not self.step or self.step % GATE_INTERVAL == 0:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype).contiguous()
        self._prepare_and_change_weights(obs, save_local=False)
        self._update_matrix()

        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        actions = torch.empty((self.population, self.actors, JOBS), device=self.device, dtype=torch.long)
        self._node["directed_node_finish"](
            (self.population * self.actors,), (_THREADS,), current, self.bias, obs, self.h, self.activity,
            self.tried, self.pending, actions, self.nodes, self.actors, int(self._fuse_activity_tanh),
        )
        if not self._fuse_activity_tanh:
            torch.tanh(self.h, out=self.activity)
        self.step += 1
        return actions
