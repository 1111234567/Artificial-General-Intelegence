"""Rebuild active metadata only after an actual support change."""

from functools import lru_cache

import torch

from ._cuda_nvrtc import compile_kernels
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_cooldown import _SupportCooldownMixin, support_cooldown_source
from .uncapped_directed_support_table import _table_support


def support_rebuild_source(threads):
    source = support_cooldown_source(threads)
    source = _replace_once(source, "void directed_support_cooldown(", "void directed_support_rebuild(")
    source = _replace_once(source, "    __shared__ unsigned forward_masks[tiles], reverse_masks[tiles];\n", "")
    source = _replace_once(source, "unsigned admitted = 0, retired = 0, active_count = 0;",
                           "unsigned admitted = 0, retired = 0;")
    source = _replace_once(source, "        bool forward = false, reverse = false;\n", "")
    source = _replace_once(source, "                if (direction == 0) forward = present;\n"
                           "                else reverse = present;\n", "")
    source = _replace_once(source, "        const unsigned forward_mask = __ballot_sync(all_lanes, forward);\n"
                           "        const unsigned reverse_mask = __ballot_sync(all_lanes, reverse);\n"
                           "        if (lane == 0) {\n"
                           "            forward_masks[tile] = forward_mask;\n"
                           "            reverse_masks[tile] = reverse_mask;\n"
                           "            active_count += __popc(forward_mask | reverse_mask);\n"
                           "        }\n", "")
    source = _replace_once(source, "birth_warps[warps], death_warps[warps], counts[warps]",
                           "birth_warps[warps], death_warps[warps]")
    source = _replace_once(source, "        counts[warp] = active_count;\n", "")
    start = source.index("        if (changed)", source.index("edge_count[actor] = prior_count"))
    finish = source.index('\nextern "C"', start)
    rebuild = """    }
    if (!changed) return;

    __shared__ unsigned active_count;
    if (threadIdx.x == 0) active_count = 0;
    __syncthreads();
    const unsigned lane_bit = 1u << lane;
    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int pair = tile * warp_size + lane;
        const bool forward = pair < pairs && mask[arc_base + pair];
        const bool reverse = pair < pairs && mask[arc_base + pairs + pair];
        const unsigned present = __ballot_sync(all_lanes, forward || reverse);
        unsigned cursor = 0;
        if (lane == 0 && present) cursor = atomicAdd(&active_count, __popc(present));
        cursor = __shfl_sync(all_lanes, cursor, 0);
        if (forward || reverse) {
            const unsigned direction = (forward ? forward_bit : 0u) | (reverse ? reverse_bit : 0u);
            const unsigned rank = __popc(present & (lane_bit - 1u));
            active_pairs[pair_base + cursor + rank] = metadata[pair] | direction;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) active_pair_counts[actor] = active_count;
}
"""
    return source[:start] + rebuild + source[finish:]


@lru_cache(maxsize=2)
def _kernel(device, threads):
    name = "directed_support_rebuild"
    return compile_kernels(support_rebuild_source(threads), (name,), device=device)[name]


class _SupportRebuildMixin(_SupportCooldownMixin):
    """Move pair compaction out of actor ticks with unchanged topology."""

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._tiled_support is not None:
            _, table, ready = _table_support(self.g.device)
            threads = self._support_cooldown_threads
            kernel = _kernel(self.g.device, threads)
            torch.cuda.current_stream(self.g.device).wait_event(ready)

            def launch(grid, _block, *arguments):
                kernel(grid, (threads,), table, *arguments)

            self._tiled_support = launch
