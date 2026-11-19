"""Optional encoded active pairs for the production private-symbol policy."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from .uncapped_directed_reward_cuda_compact import _THREADS
from .uncapped_directed_reward_sparse_reduction import _replace_once, _sparse_source
from .uncapped_private_symbol_fast_run import _TunedPolicy


_PAIR_BITS = 13
_NODE_BITS = 7


def _encoded_source():
    """Keep the production eligibility/reward arithmetic and diagnostic tree."""
    source = _sparse_source()
    source = _replace_once(source, "void directed_compact_eligibility_only(",
                           "void directed_encoded_eligibility_only(")
    source = _replace_once(source, "void directed_sparse_reward_update(",
                           "void directed_encoded_reward_update(")
    source = _replace_once(
        source,
        "const half_bits* features, const long long* u, const long long* v,\n"
        "    offset_t activity_base, offset_t node_base, int pair\n",
        "const half_bits* features, offset_t activity_base, offset_t node_base, int pair,\n"
        "    unsigned metadata\n",
    )
    source = _replace_once(source, "node_base + u[pair]", "node_base + ((metadata >> u_shift) & node_mask)")
    source = _replace_once(source, "node_base + v[pair]", "node_base + ((metadata >> v_shift) & node_mask)")
    source = _replace_once(
        source, "const int pair = active_pairs[activity_base + index];",
        "const unsigned metadata = active_pairs[activity_base + index];\n"
        "        const int pair = metadata & pair_mask;", count=2,
    )
    source = _replace_once(
        source, "pair_activity<direct>(features, u, v, activity_base, node_base, pair)",
        "pair_activity<direct>(features, activity_base, node_base, pair, metadata)", count=2,
    )
    for edge, direction in (("first", "forward"), ("second", "reverse")):
        source = _replace_once(source, f"mask[{edge}]", f"(metadata & {direction}_bit) != 0", count=2)
    builder = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    constants = f"#define ENCODED_PAIR_BITS {_PAIR_BITS}\n#define ENCODED_NODE_BITS {_NODE_BITS}\n"
    return constants + builder + "\n" + source


@lru_cache(maxsize=None)
def _encoded_kernels(device):
    from ._cuda_nvrtc import compile_kernels

    return compile_kernels(_encoded_source(), (
        "directed_encoded_build_pairs", "directed_encoded_eligibility_only", "directed_encoded_reward_update",
    ), device=device)


class _EncodedPairMixin:
    """Pack immutable endpoints and current direction bits in the int32 list."""

    def __init__(self, birth, actors, device, mode):
        self._encoded_build = None
        super().__init__(birth, actors, device, mode)
        if (self._sparse is None or self.nodes > (1 << _NODE_BITS)
                or self.pair_count > (1 << _PAIR_BITS)):
            return

        kernels = _encoded_kernels(torch.device(device))
        self._encoded_metadata = (
            torch.arange(self.pair_count, dtype=torch.int32, device=device)
            | (self.u.to(dtype=torch.int32) << _PAIR_BITS)
            | (self.v.to(dtype=torch.int32) << (_PAIR_BITS + _NODE_BITS))
        )
        # Per-instance launch mappings leave the cached production kernels intact.
        self._compact = {
            **self._compact,
            "directed_compact_eligibility_only": kernels["directed_encoded_eligibility_only"],
        }
        self._sparse = {
            **self._sparse,
            "directed_sparse_reward_update": kernels["directed_encoded_reward_update"],
        }
        self._encoded_build = kernels["directed_encoded_build_pairs"]
        self._rebuild_active_pairs()

    def _rebuild_encoded_pairs(self, changed):
        self._encoded_build(
            (self.population * self.actors,), (_THREADS,), self.mask,
            self._encoded_metadata, changed, self._active_pairs, self._active_pair_counts,
            self.pair_count,
        )

    def _rebuild_active_pairs(self):
        if self._encoded_build is None:
            return super()._rebuild_active_pairs()
        self._rebuild_encoded_pairs(ctypes.c_void_p())

    def _rebuild_changed_pairs(self, changed):
        if self._encoded_build is not None:
            self._rebuild_encoded_pairs(changed)
            return
        inherited = getattr(super(), "_rebuild_changed_pairs", None)
        if inherited is not None:
            inherited(changed)
        else:
            self._rebuild_active_pairs()


class EncodedPairPolicy(_EncodedPairMixin, _TunedPolicy):
    """Private-symbol policy using encoded active-pair metadata on supported CUDA."""
