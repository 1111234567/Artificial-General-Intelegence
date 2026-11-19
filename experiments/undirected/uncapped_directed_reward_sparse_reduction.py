"""Optional event-sparse, original-tree FP32 weight-change diagnostics.

The original compact backend is untouched. This mixin specializes only the
installed Torch 2.11.0/cu130 sum configuration for contiguous 9120-arc FP16
rows on sm_89. It needs GPU bitwise/throughput validation before selection.
Unknown configurations retain the unchanged compact backend.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity import WEIGHT_LIMIT
from .uncapped_directed_reward_cuda_compact import _CompactRewardPolicy, _THREADS


def _replace_once(source, old, new, *, count=1):
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f"sparse diagnostic source guard: expected {count}, found {actual}: {old!r}")
    return source.replace(old, new)


def _sparse_source():
    # Reuse identical eligibility, reward rounding, scatter and integer-counter
    # work. The only new data is a small per-actor nonfinite flag, computed while
    # the updater already reads each active old weight.
    compact = Path(__file__).with_name("uncapped_directed_reward_cuda_compact.cu")
    source = compact.read_text(encoding="utf-8")
    source = _replace_once(source, "void directed_compact_reward_update(",
                           "void directed_sparse_reward_update(")
    source = _replace_once(
        source, "half_bits* deltas, unsigned& attempted, unsigned& changed\n",
        "half_bits* deltas, unsigned& attempted, unsigned& changed, unsigned& nonfinite\n",
    )
    source = _replace_once(
        source, "const int* active_pairs, int count, unsigned& attempted, unsigned& changed\n",
        "const int* active_pairs, int count, unsigned& attempted, unsigned& changed, unsigned& nonfinite\n",
    )
    source = _replace_once(
        source, "deltas, attempted, changed);",
        "deltas, attempted, changed, nonfinite);", count=2,
    )
    source = _replace_once(
        source, "deltas, active_pairs, count, attempted, changed);",
        "deltas, active_pairs, count, attempted, changed, nonfinite);", count=4,
    )
    source = _replace_once(
        source, "const half_bits prior = weights[edge];",
        "const half_bits prior = weights[edge];\n"
        "    nonfinite += (prior & 0x7c00u) == 0x7c00u;",
    )
    source = _replace_once(
        source, "unsigned attempted = 0, changed = 0;",
        "unsigned attempted = 0, changed = 0, nonfinite = 0;",
    )
    source = _replace_once(
        source, "half_bits* __restrict__ deltas,\n    const int* __restrict__ active_pairs,",
        "half_bits* __restrict__ deltas,\n"
        "    bool* __restrict__ nonfinite_flags,\n    const int* __restrict__ active_pairs,",
    )
    source = _replace_once(
        source, "__shared__ unsigned attempted_warps[32], changed_warps[32];",
        "__shared__ unsigned attempted_warps[32], changed_warps[32], nonfinite_warps[32];",
    )
    source = _replace_once(
        source, "changed = warp_sum(changed);\n    const unsigned lane",
        "changed = warp_sum(changed);\n    nonfinite = warp_sum(nonfinite);\n    const unsigned lane",
    )
    source = _replace_once(
        source, "changed_warps[warp] = changed;",
        "changed_warps[warp] = changed;\n        nonfinite_warps[warp] = nonfinite;",
    )
    source = _replace_once(
        source, "changed = warp_sum(lane < blockDim.x / 32 ? changed_warps[lane] : 0u);",
        "changed = warp_sum(lane < blockDim.x / 32 ? changed_warps[lane] : 0u);\n"
        "        nonfinite = warp_sum(lane < blockDim.x / 32 ? nonfinite_warps[lane] : 0u);",
    )
    source = _replace_once(
        source, "if (lane == 0) {\n            if (attempted)",
        "if (lane == 0) {\n            nonfinite_flags[actor] = nonfinite != 0;\n            if (attempted)",
    )
    return source + "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _sparse_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    return compile_kernels(_sparse_source(), (
        "directed_sparse_reward_update", "directed_sparse_diagnostic_sum",
    ), device=device)


class _SparseDiagnosticPolicy(_CompactRewardPolicy):
    """Retain the original reduction tree while skipping actors with no write."""

    def __init__(self, birth, actors, device, mode):
        self._sparse = None
        self._sparse_nonfinite = None
        super().__init__(birth, actors, device, mode)
        device = torch.device(device)
        compatible = (
            self._compact is not None
            and self.nodes == 96 and self.pair_count == 4560
            and self.population * self.actors >= 16
            and self.proposal.dtype == torch.float16
            and self.proposal.is_contiguous() and self.mask.is_contiguous()
            and str(torch.__version__) == "2.11.0+cu130"
            and torch.version.cuda == "13.0"
            and torch.cuda.get_device_capability(device) == (8, 9)
        )
        if compatible:
            self._sparse = _sparse_kernels(device)
            self._sparse_nonfinite = torch.empty_like(self.pending)

    def _change_compact_weights(self, features, event, direct, scatter):
        if self._sparse is None or not self._receipt_possible():
            return super()._change_compact_weights(features, event, direct, scatter)

        grid = (self.population * self.actors,)
        # All present deltas are overwritten. The reducer reads absent slots
        # as positive zero, so stale scratch never contributes to the sum.
        self._sparse["directed_sparse_reward_update"](
            grid, (_THREADS,), features, self.u, self.v, self.mask,
            self.eligibility, self.w, self.matrix, self.arc_indices,
            self.reward, self.reward_rate, event, self.attempted_writes,
            self.weight_updates, self.proposal, self._sparse_nonfinite,
            self._active_pairs, self._active_pair_counts,
            self.pair_count, self.actors, self.nodes, direct, scatter,
            WEIGHT_LIMIT,
        )
        self._sparse["directed_sparse_diagnostic_sum"](
            grid, (32, 16), self.proposal, self.mask, self.w, event,
            self._sparse_nonfinite, self.weight_change,
        )
