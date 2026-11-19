"""Fuse the directed FP16 matrix-vector product into the node transition."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from .evolved_plasticity import ACTIONS, INPUTS, JOBS
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_directed_reward_sparse_reduction import _replace_once


def transition_source():
    source = Path(__file__).with_name("uncapped_directed_node_fused.cu").read_text(encoding="utf-8")
    node_helpers, kernels = source.split('extern "C"', maxsplit=1)
    finish = 'extern "C"' + kernels.split('extern "C"', maxsplit=1)[1]
    matvec = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    math = matvec[matvec.index("__device__"):matvec.index('extern "C"')]
    finish = _replace_once(finish, "void directed_node_finish(", "void directed_tensor_transition(")
    finish = _replace_once(finish, "const half_bits* __restrict__ current,",
                           "const unsigned* __restrict__ matrix,")
    finish = _replace_once(finish, "int nodes, int actors, int fuse_tanh", "int nodes, int actors")
    finish = _replace_once(finish, "__shared__ half_bits action_ports[NODE_JOBS * NODE_ACTIONS];",
                           "__shared__ half_bits action_ports[NODE_JOBS * NODE_ACTIONS];\n"
                           "    __shared__ half_bits current[96];\n"
                           "    multiply_rows(matrix + actor * 96 * 48,\n"
                           "                  reinterpret_cast<const unsigned*>(activity) + actor * 48,\n"
                           "                  current, threadIdx.x / 32, threadIdx.x % 32);\n"
                           "    __syncthreads();")
    finish = _replace_once(finish, "from_half(current[i])", "from_half(current[node])")
    finish = _replace_once(finish, "if (fuse_tanh) activity[i]", "activity[i]")
    constants = {"INPUTS": INPUTS, "JOBS": JOBS, "ACTIONS": ACTIONS, "NEW": NEW, "READY": READY,
                 "FIRST_FINISH": FIRST_FINISH, "N_ACTIONS": N_ACTIONS}
    return "\n".join(f"#define NODE_{name} {value}" for name, value in constants.items()) + "\n" + node_helpers + math + finish


@lru_cache(maxsize=1)
def _kernels(device):
    return compile_kernels(transition_source(), ("directed_tensor_transition",), device=device)


class _TensorTransitionMixin:
    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        self._tensor_transition = None
        if self._sparse is not None and self._node is not None and self._fuse_activity_tanh:
            self._tensor_transition = _kernels(device)["directed_tensor_transition"]

    def _tensor_step(self, obs):
        actions = torch.empty((self.population, self.actors, JOBS), device=self.device, dtype=torch.long)
        self._tensor_transition(
            (self.population * self.actors,), (192,), self.matrix, self.bias, obs, self.h, self.activity,
            self.tried, self.pending, actions, self.nodes, self.actors,
        )
        return actions
