"""Optional 96-node support update with fused receipt and active-list refresh."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from .evolved_plasticity import EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, JOBS, WEIGHT_LIMIT, GraphBatch
from .evolved_plasticity_world import FEATURES_PER_SLOT
from .uncapped_directed_reward_encoded_pairs import _NODE_BITS, _PAIR_BITS
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE
from .uncapped_private_symbol_fused_run import _FusedPolicy
from .uncapped_reward_precision import _ACK, _REJ


_THREADS = 128
_REFERENCE_RECEIPT = GraphBatch._receipt


def support_tile_source():
    constants = {
        "ENCODED_NODE_BITS": _NODE_BITS,
        "ENCODED_PAIR_BITS": _PAIR_BITS,
        "SUPPORT_JOB_FEATURES": FEATURES_PER_SLOT,
    }
    definitions = "\n".join(f"#define {name} {value}" for name, value in constants.items())
    return definitions + "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _support_tile_kernel(device):
    name = "directed_support_tile"
    return compile_kernels(support_tile_source(), (name,), device=device)[name]


class _SupportTileMixin:
    """Combine support, short Torch receipt reductions and encoded-list refresh."""

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        self._tiled_support = None
        if (self._support_tick is not None and self._encoded_build is not None
                and self.nodes == 96 and self.pair_count == 4560
                and self.rule.size(-1) == 9 and self.rule.is_contiguous()
                and type(self)._receipt is _REFERENCE_RECEIPT
                and (INPUTS, JOBS, FEATURES_PER_SLOT, _ACK, _REJ) == (56, 2, 28, 22, 23)):
            self._tiled_support = _support_tile_kernel(torch.device(device))
            self._refresh_support_scale()

    def _refresh_support_scale(self):
        # The rule is fixed for a life. CUDA Graph reset still updates the
        # scale from the newly transferred birth parameters on each replay.
        self._support_weight_scale = self.rule[:, :, 1] / WEIGHT_LIMIT

    def reset(self, birth):
        super().reset(birth)
        if getattr(self, "_tiled_support", None) is not None:
            self._refresh_support_scale()

    def _update_tiled_support(self, local, obs):
        """Run after node preparation and sparse reward updates on a support tick."""
        self._tiled_support(
            (self.population * self.actors,), (_THREADS,),
            local, self.node_trace, self._encoded_metadata, obs, self.rule, self._support_weight_scale,
            self.birth_positive, self.birth_negative, self.g, self.w, self.matrix,
            self.arc_indices, self.eligibility, self.mask, self.last_change,
            self.births, self.deaths, self.edge_count, self.edge_ticks,
            self._active_pairs, self._active_pair_counts,
            self.actors, self.step, self.step - self.edge_span_start, EDGE_COOLDOWN,
            GATE_RATE, BIRTH_THRESHOLD, DEATH_THRESHOLD,
        )
        self.edge_span_start = self.step

    def act(self, obs):
        if self._tiled_support is None or not self.step or self.step % GATE_INTERVAL:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype).contiguous()
        grid, block = (self.population * self.actors,), (_THREADS,)
        local, event = self._node_local_activity, self._node_event
        self._node["directed_node_prepare"](
            grid, block, self.h, obs, self.pending, self.reward, event,
            self.reward_events, self.node_trace, local, self.nodes, 1,
        )
        self._change_prepared_weights(local, event)
        self._update_tiled_support(local, obs)

        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        actions = torch.empty((self.population, self.actors, JOBS), device=self.device, dtype=torch.long)
        self._node["directed_node_finish"](
            grid, block, current, self.bias, obs, self.h, self.activity,
            self.tried, self.pending, actions, self.nodes, self.actors, 1,
        )
        self.step += 1
        return actions


class SupportTilePolicy(_SupportTileMixin, _FusedPolicy):
    """Standalone candidate with the inherited recurrent matvec and reward path."""
