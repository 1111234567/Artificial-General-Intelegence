"""Fuse full-mode support tick preparation and refresh only changed pair lists."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from .evolved_plasticity import EDGE_COOLDOWN, GATE_INTERVAL, JOBS, WEIGHT_LIMIT
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_selective import _selective_support_source
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE


def support_source():
    source = _selective_support_source()
    source = _replace_once(source, "void directed_support_selective_update(", "void directed_support_tick(")
    source = _replace_once(source, "const half_bits* __restrict__ coactivity,\n    const half_bits* __restrict__ difference,",
                           "const half_bits* __restrict__ local_activity,")
    source = _replace_once(source, "long long* __restrict__ edge_ticks,",
                           "long long* __restrict__ edge_ticks,\n    bool* __restrict__ topology_changed,")
    source = _replace_once(source, "const float active = from_half(coactivity[activity_base + pair]);\n"
                           "        const float distinct = from_half(difference[activity_base + pair]);\n"
                           "        const float trace = round_half(\n"
                           "            from_half(node_trace[node_base + u[pair]]) *\n"
                           "            from_half(node_trace[node_base + v[pair]]));",
                           "const int first = u[pair], second = v[pair];\n"
                           "        const float a = from_half(local_activity[node_base + first]);\n"
                           "        const float b = from_half(local_activity[node_base + second]);\n"
                           "        const float active = round_half(a * b);\n"
                           "        const float distinct = fabsf(round_half(a - b));\n"
                           "        const float trace = round_half(\n"
                           "            from_half(node_trace[node_base + first]) *\n"
                           "            from_half(node_trace[node_base + second]));")
    return _replace_once(source, "edge_count[actor] = prior_count + admitted - retired;",
                         "edge_count[actor] = prior_count + admitted - retired;\n"
                         "            topology_changed[actor] = admitted != 0 || retired != 0;")


def rebuild_source():
    source = Path(__file__).with_name("uncapped_directed_reward_cuda_compact.cu").read_text(encoding="utf-8")
    builder = source[source.index('extern "C"'):source.index("template<bool direct>")]
    builder = _replace_once(builder, "void directed_compact_build_pairs(", "void directed_rebuild_changed_pairs(")
    builder = _replace_once(builder, "int pairs", "int pairs, const bool* __restrict__ topology_changed")
    return "using offset_t = unsigned long long;\n" + _replace_once(
        builder, "const unsigned actor = blockIdx.x;", "const unsigned actor = blockIdx.x;\n"
        "    if (!topology_changed[actor]) return;",
    )


@lru_cache(maxsize=1)
def _kernels(device):
    return compile_kernels(support_source(), ("directed_support_tick",), device=device)["directed_support_tick"]


@lru_cache(maxsize=1)
def _builders(device):
    return compile_kernels(rebuild_source(), ("directed_rebuild_changed_pairs",), device=device)["directed_rebuild_changed_pairs"]


class _SupportTickPolicy:
    """Cooperative mixin for the selective, compact and fused-node backend."""

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        self._support_tick = None
        if self._sparse is not None and self._node is not None and self._fuse_prepare_tanh and self._fuse_activity_tanh:
            self._support_tick = _kernels(self.device)
            self._changed_pairs = torch.empty_like(self.pending)
            self._build_changed = _builders(self.device)

    def _rebuild_changed_pairs(self, changed):
        self._build_changed((self.population * self.actors,), (128,), self.mask,
                            self._active_pairs, self._active_pair_counts, self.pair_count, changed)

    def act(self, obs):
        if self._support_tick is None or not self.step or self.step % GATE_INTERVAL:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype).contiguous()
        grid = (self.population * self.actors,)
        block = (128,)
        local, event = self._node_local_activity, self._node_event
        self._node["directed_node_prepare"](
            grid, block, self.h, obs, self.pending, self.reward, event,
            self.reward_events, self.node_trace, local, self.nodes, 1,
        )
        self._change_prepared_weights(local, event)
        receipt = self._receipt(obs)
        birth_coefficient = (receipt * self.rule[:, 0, None, 5:]).sum(-1)
        gate_coefficient = (receipt * self.rule[:, 1, None, 5:]).sum(-1)
        weight_scale = self.rule[:, :, 1] / WEIGHT_LIMIT
        self._support_tick(
            grid, block, local, self.node_trace, self.u, self.v, self.rule,
            weight_scale, birth_coefficient, gate_coefficient, self.birth_positive, self.birth_negative,
            self.g, self.w, self.matrix, self.arc_indices, self.eligibility, self.mask, self.last_change,
            self.births, self.deaths, self.edge_count, self.edge_ticks, self._changed_pairs,
            self.pair_count, self.actors, self.nodes, self.rule.size(-1),
            self.step, self.step - self.edge_span_start, EDGE_COOLDOWN, GATE_RATE, BIRTH_THRESHOLD, DEATH_THRESHOLD,
        )
        self.edge_span_start = self.step
        self._rebuild_changed_pairs(self._changed_pairs)

        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        actions = torch.empty((self.population, self.actors, JOBS), device=self.device, dtype=torch.long)
        self._node["directed_node_finish"](
            grid, block, current, self.bias, obs, self.h, self.activity,
            self.tried, self.pending, actions, self.nodes, self.actors, 1,
        )
        self.step += 1
        return actions
