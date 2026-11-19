"""Optional FP16 node-transition fusion after the unchanged Torch CUDA bmm.

Full-mode non-support ticks also fuse the receipt counters, rounded local
activity and node trace. Reset, support ticks, CPU, and the two ablations keep
the existing prepared policy. This module does not select a backend for runs.

``_FusedNodeTorchTanhPolicy`` keeps both tanh calls in PyTorch for platforms
whose libdevice implementation fails the required bitwise parity check.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import ACTIONS, GATE_INTERVAL, INPUTS, JOBS
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_directed_reward_cuda_prepared import _FusedPreparedPolicy
from .uncapped_reward_precision import _ACK, _REJ


_THREADS = 128


@lru_cache(maxsize=1)
def _node_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    constants = {
        "NODE_INPUTS": INPUTS,
        "NODE_JOBS": JOBS,
        "NODE_ACTIONS": ACTIONS,
        "NODE_NEW": NEW,
        "NODE_READY": READY,
        "NODE_ACK": _ACK,
        "NODE_REJ": _REJ,
        "NODE_FIRST_FINISH": FIRST_FINISH,
        "NODE_N_ACTIONS": N_ACTIONS,
    }
    source = "\n".join(f"#define {name} {value}" for name, value in constants.items())
    source += "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_node_prepare", "directed_node_finish"), device=device)


class _FusedNodePolicy(_FusedPreparedPolicy):
    """Fuse the small pre/post-GEMM operations on ordinary full-mode ticks.

    The class attributes allow the two transcendental substitutions to be
    isolated during validation without changing the original constructor API.
    All kernels are compiled and scratch tensors allocated before capture.
    """

    _fuse_prepare_tanh = True
    _fuse_activity_tanh = True

    def __init__(self, birth, actors, device, mode):
        device = torch.device(device)
        self._node = _node_kernels(device) if device.type == "cuda" and mode == "full" else None
        super().__init__(birth, actors, device, mode)
        if self._node is not None:
            self._node_local_activity = torch.empty_like(self.h)
            self._node_event = torch.empty_like(self.pending)

    def act(self, obs):
        if self._node is None or not self.step or self.step % GATE_INTERVAL == 0:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        # The public API accepts strided observations. CUDA kernels use a
        # compact frame, while the common world path is already contiguous.
        obs = obs.to(device=self.device, dtype=self.h.dtype).contiguous()
        grid, block = (self.population * self.actors,), (_THREADS,)
        local_activity = self._node_local_activity
        event = self._node_event
        self._node["directed_node_prepare"](
            grid, block, self.h, obs, self.pending, self.reward, event,
            self.reward_events, self.node_trace, local_activity, self.nodes,
            int(self._fuse_prepare_tanh),
        )
        if not self._fuse_prepare_tanh:
            local_activity = (4. * self.h).tanh()
            self._prepare_trace(local_activity)
        self._change_prepared_weights(local_activity, event)
        self._update_matrix()

        # Keep Torch's FP32-accumulating FP16 bmm and its output rounding.
        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        # A fresh result preserves act()'s ordinary ownership semantics for
        # callers that retain earlier actions, including during graph capture.
        actions = torch.empty((self.population, self.actors, JOBS), device=self.device, dtype=torch.long)
        self._node["directed_node_finish"](
            grid, block, current, self.bias, obs, self.h, self.activity,
            self.tried, self.pending, actions, self.nodes, self.actors,
            int(self._fuse_activity_tanh),
        )
        if not self._fuse_activity_tanh:
            torch.tanh(self.h, out=self.activity)
        self.step += 1
        return actions


class _FusedNodeTorchTanhPolicy(_FusedNodePolicy):
    """Fallback retaining PyTorch's local-activity and recurrent tanh kernels."""

    _fuse_prepare_tanh = False
    _fuse_activity_tanh = False
