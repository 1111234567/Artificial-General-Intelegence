"""Optional 256-thread launch of the exact support table kernel."""

from functools import lru_cache

import torch

from ._cuda_nvrtc import compile_kernels
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_table import _table_support, support_table_source
from .uncapped_directed_support_tile import _SupportTileMixin
from .uncapped_private_symbol_fused_run import _FusedPolicy


_WIDE_THREADS = 256
_NAME = "directed_support_wide"


def support_wide_source():
    source = support_table_source()
    source = _replace_once(source, "void directed_support_tile_table(", f"void {_NAME}(")
    source = _replace_once(source, "constexpr int threads = 128;", f"constexpr int threads = {_WIDE_THREADS};")
    return _replace_once(
        source, "        if (changed) active_pair_counts[actor] = counts[0] + counts[1] + counts[2] + counts[3];",
        """        if (changed) {
            unsigned total = 0;
            #pragma unroll
            for (int i = 0; i < warps; ++i) total += counts[i];
            active_pair_counts[actor] = total;
        }""",
    )


@lru_cache(maxsize=None)
def _wide_kernel(device):
    return compile_kernels(support_wide_source(), (_NAME,), device=device)[_NAME]


class _SupportWideMixin(_SupportTileMixin):
    """Expose the tiled support hook with a wider per-actor block."""

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._tiled_support is not None:
            _, table, ready = _table_support(self.g.device)
            kernel = _wide_kernel(self.g.device)
            torch.cuda.current_stream(self.g.device).wait_event(ready)

            def launch(grid, _block, *arguments):
                kernel(grid, (_WIDE_THREADS,), table, *arguments)

            self._tiled_support = launch


class SupportWidePolicy(_SupportWideMixin, _FusedPolicy):
    """Standalone wide support table candidate."""
