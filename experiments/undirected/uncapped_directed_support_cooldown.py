"""Optional support kernel that reads cooldown only at a transition threshold."""

from functools import lru_cache

import torch

from ._cuda_nvrtc import compile_kernels
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_reuse import support_reuse_source
from .uncapped_directed_support_table import _table_support
from .uncapped_directed_support_tile import _SupportTileMixin
from .uncapped_private_symbol_fused_run import _FusedPolicy


_NAME = "directed_support_cooldown"
_UNCONDITIONAL = """                const bool ready = last_change[edge] <= step - cooldown;
                bool present = mask[edge];
                const float gate = from_half(next_gate);
                if (ready && present && gate <= retire_at) {"""
_CONDITIONAL = """                bool present = mask[edge];
                const float gate = from_half(next_gate);
                const bool near_transition = present ? gate <= retire_at : gate >= admit_at;
                const bool transition = near_transition && last_change[edge] <= step - cooldown;
                if (transition && present) {"""


def support_cooldown_source(threads=256):
    source = support_reuse_source(threads)
    source = _replace_once(source, "void directed_support_reuse(", f"void {_NAME}(")
    source = _replace_once(source, _UNCONDITIONAL, _CONDITIONAL)
    return _replace_once(source, "} else if (ready && !present && gate >= admit_at) {",
                         "} else if (transition) {")


@lru_cache(maxsize=None)
def _cooldown_kernel(device, threads):
    return compile_kernels(support_cooldown_source(threads), (_NAME,), device=device)[_NAME]


class _SupportCooldownMixin(_SupportTileMixin):
    """Skip cooldown traffic for arcs whose rounded gates fail their thresholds."""

    _support_cooldown_threads = 256

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._tiled_support is not None:
            _, table, ready = _table_support(self.g.device)
            threads = self._support_cooldown_threads
            kernel = _cooldown_kernel(self.g.device, threads)
            torch.cuda.current_stream(self.g.device).wait_event(ready)

            def launch(grid, _block, *arguments):
                kernel(grid, (threads,), table, *arguments)

            self._tiled_support = launch


class SupportCooldownPolicy(_SupportCooldownMixin, _FusedPolicy):
    """Standalone shared-drive and conditional-cooldown candidate."""
