"""Optional compact reward updates with selective support matrix scattering.

Reward updates always scatter changed weights, including on support ticks.
The support kernel therefore only writes matrix entries that it admits or
retires, rather than scanning unchanged entries into the recurrent matrix.
All support arithmetic comes from the existing scattered kernel, with guarded
source replacements so an upstream change cannot silently change this variant.
This module does not select a training backend or modify any running process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .uncapped_directed_node_fused import _FusedNodePolicy
from .uncapped_directed_reward_cuda_compact import _CompactRewardPolicy
from .uncapped_directed_support_scattered import _ScatteredSupportPolicy


_KERNEL_NAME = "directed_support_selective_update"
_ORIGINAL_NAME = "directed_support_scattered_update"
_OLD_SCATTER = """            // Reward updates suppress their scatter on support ticks. Write
            // EVERY final weight, including arcs with unchanged support.
            matrix[matrix_base + arc_indices[arc]] = weight_bits;"""
_SELECTIVE_SCATTER = """            // Reward already scattered weight changes, even on support ticks.
            // Retirement must write +0; birth writes its rounded new weight.
            if (retire || admit) {
                matrix[matrix_base + arc_indices[arc]] = weight_bits;
            }"""


def _selective_support_source():
    """Return the existing support arithmetic with only its scatter guarded."""
    source = Path(__file__).with_name(
        "uncapped_directed_support_scattered.cu",
    ).read_text(encoding="utf-8")
    for old, new in ((f"void {_ORIGINAL_NAME}(", f"void {_KERNEL_NAME}("),
                     (_OLD_SCATTER, _SELECTIVE_SCATTER)):
        if source.count(old) != 1:
            raise RuntimeError("the scattered support source no longer matches the selective variant")
        source = source.replace(old, new, 1)
    return source


@lru_cache(maxsize=None)
def _selective_support_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    return compile_kernels(_selective_support_source(), (_KERNEL_NAME,), device=device)


class _SelectiveSupportPolicy(_ScatteredSupportPolicy):
    """Scatter support changes and retain inherited reset/refresh behavior.

    Combine with a reward policy that scatters changes on support ticks too;
    ``_SelectivePolicy`` below supplies that requirement for compact updates.
    The inherited constructor remains cooperative, warming its original
    support kernels before this optional kernel is selected, all before graph
    capture. No per-tick allocation or dispatch is added by the selection.
    """

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        if self._scattered_support is not None:
            kernels = _selective_support_kernels(torch.device(device))
            # Reuse the inherited launch ABI and in-act refresh suppression.
            self._scattered_support = {_ORIGINAL_NAME: kernels[_KERNEL_NAME]}


class _SelectivePolicy(_CompactRewardPolicy, _FusedNodePolicy, _SelectiveSupportPolicy):
    """Compact/node policy whose support matrix needs only topology writes."""

    def _change_weights(self, coactivity, event):
        if self._compact is None:
            return super()._change_weights(coactivity, event)
        # Support still uses the original rounded coactivity. Scatter reward
        # changes before support so unchanged-support entries remain current.
        self._change_compact_weights(coactivity, event, 0, 1)
