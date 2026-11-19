"""Optional support kernel that shares gate drives for bitwise equal twins."""

from functools import lru_cache

import torch

from ._cuda_nvrtc import compile_kernels
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_table import _table_support, support_table_source
from .uncapped_directed_support_tile import _SupportTileMixin
from .uncapped_directed_support_wide import support_wide_source
from .uncapped_private_symbol_fused_run import _FusedPolicy


_NAME = "directed_support_reuse"
_ORIGINAL = """            #pragma unroll
            for (int direction = 0; direction < 2; ++direction) {
                const int arc = direction * pairs + pair;
                const offset_t edge = arc_base + arc;
                const float weight = from_half(weights[edge]);
                const float target = arc_drive(table, support_rule + 5, weight, active, distinct, trace);"""
_COMMON_TARGET = """            const half_bits forward_bits = weights[arc_base + pair];
            const half_bits reverse_bits = weights[arc_base + pairs + pair];
            const float forward_weight = from_half(forward_bits);
            const float forward_target = arc_drive(table, support_rule + 5, forward_weight, active, distinct, trace);
            #pragma unroll
            for (int direction = 0; direction < 2; ++direction) {
                const int arc = direction * pairs + pair;
                const offset_t edge = arc_base + arc;
                const float weight = direction == 0 ? forward_weight : from_half(reverse_bits);
                const float target = direction == 0 || forward_bits == reverse_bits
                    ? forward_target : arc_drive(table, support_rule + 5, weight, active, distinct, trace);"""


def support_reuse_source(threads=256):
    if threads == 128:
        source, entry = support_table_source(), "directed_support_tile_table"
    elif threads == 256:
        source, entry = support_wide_source(), "directed_support_wide"
    else:
        raise ValueError("support reuse requires either 128 or 256 threads")
    source = _replace_once(source, f"void {entry}(", f"void {_NAME}(")
    # Both directions share all other inputs. Keep each gate's old value,
    # readiness, topology and birth update even when their gate target agrees.
    return _replace_once(source, _ORIGINAL, _COMMON_TARGET)


@lru_cache(maxsize=None)
def _reuse_kernel(device, threads):
    return compile_kernels(support_reuse_source(threads), (_NAME,), device=device)[_NAME]


class _SupportReuseMixin(_SupportTileMixin):
    """Use one target evaluation when the incoming Half weight bits match."""

    _support_reuse_threads = 256

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._tiled_support is not None:
            _, table, ready = _table_support(self.g.device)
            threads = self._support_reuse_threads
            kernel = _reuse_kernel(self.g.device, threads)
            torch.cuda.current_stream(self.g.device).wait_event(ready)

            def launch(grid, _block, *arguments):
                kernel(grid, (threads,), table, *arguments)

            self._tiled_support = launch


class SupportReusePolicy(_SupportReuseMixin, _FusedPolicy):
    """Standalone shared-drive candidate with a 256-thread block."""
