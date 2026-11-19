"""Optional support tiles using an exact CUDA tanh table over all Half inputs."""

from functools import lru_cache

import torch

from ._cuda_nvrtc import compile_kernels
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_tile import _SupportTileMixin, support_tile_source
from .uncapped_private_symbol_fused_run import _FusedPolicy


_TABLE_SIZE = 1 << 16
_TABLE_INIT_THREADS = 256
_OLD_DRIVE_FINAL = """    drive = round_half(__fmaf_rn(difference, rule[4], drive));
    return round_half(tanhf(drive));"""
_NEW_DRIVE_FINAL = """    const half_bits input = to_half(__fmaf_rn(difference, rule[4], drive));
    return from_half(__ldg(table + input));"""

_INIT_TANH_TABLE = """
extern "C" __global__ void directed_support_tanh_table(half_bits* table) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    table[index] = to_half(tanhf(from_half(static_cast<half_bits>(index))));
}
"""


def support_table_source():
    """Replace the repeated per-edge libdevice expression by a Half lookup."""
    source = support_tile_source()
    source = _replace_once(source, "void directed_support_tile(", "void directed_support_tile_table(")
    source = _replace_once(source, "    const half_bits* __restrict__ local_activity,",
                           "    const half_bits* __restrict__ table,\n"
                           "    const half_bits* __restrict__ local_activity,")
    source = _replace_once(source, "    const float* rule, float weight, float coactivity,",
                           "    const half_bits* table, const float* rule, float weight, float coactivity,")
    source = _replace_once(source, _OLD_DRIVE_FINAL, _NEW_DRIVE_FINAL)
    source = _replace_once(source, "arc_drive(support_rule", "arc_drive(table, support_rule", count=2)
    return source + _INIT_TANH_TABLE


@lru_cache(maxsize=None)
def _table_support(device):
    names = "directed_support_tile_table", "directed_support_tanh_table"
    kernels = compile_kernels(support_table_source(), names, device=device)
    table = torch.empty(_TABLE_SIZE, dtype=torch.float16, device=device)
    kernels[names[1]]((_TABLE_SIZE // _TABLE_INIT_THREADS,), (_TABLE_INIT_THREADS,), table)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return kernels[names[0]], table, ready


class _SupportTableMixin(_SupportTileMixin):
    """Use like the support tile mixin. Table initialization precedes capture."""

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._tiled_support is not None:
            kernel, table, ready = _table_support(self.g.device)
            torch.cuda.current_stream(self.g.device).wait_event(ready)

            def launch(grid, block, *arguments):
                kernel(grid, block, table, *arguments)

            self._tiled_support = launch


class SupportTablePolicy(_SupportTableMixin, _FusedPolicy):
    """Standalone support table candidate for the same 96-node policy."""
