"""Opt-in CUDA arc specialization for the 96-node local-rule policy."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from .local_rule_policy import (
    PLASTIC_FIELDS, PLASTIC_SIZE, _ARC_THREADS, _CONSTANTS, _COUNTS, _TOTALS,
    make_policy as _make_policy,
)


_KERNELS = {
    1: "fast_local_arcs_one_tile",
    2: "fast_local_arcs_two_tiles",
    4: "fast_local_arcs_four_tiles",
}
_STATS_SIZE = 6 * (_ARC_THREADS // 32)


@lru_cache(maxsize=None)
def _kernels(device):
    constants = {**_CONSTANTS, **{field.upper(): i for i, field in enumerate(PLASTIC_FIELDS)},
                 **{f"COUNT_{name.upper()}": i for i, name in enumerate(_COUNTS)},
                 **{f"TOTAL_{name.upper()}": i for i, name in enumerate(_TOTALS)},
                 "PLASTIC_SIZE": PLASTIC_SIZE, "ARC_THREADS": _ARC_THREADS, "TENSOR_TRANSITION": 1}
    definitions = "\n".join(f"#define {name} {value!r}f" if isinstance(value, float)
                            else f"#define {name} {value}" for name, value in constants.items())
    tensor_source = Path(__file__).with_name("uncapped_directed_matvec_tensor.cu").read_text(encoding="utf-8")
    tensor_helpers = tensor_source[:tensor_source.index('extern "C"')]
    original_source = Path(__file__).with_name("local_rule_policy.cu").read_text(encoding="utf-8")
    local_helpers = original_source[:original_source.index('extern "C"')]
    arc_source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    finish_source = original_source[original_source.index('extern "C" __global__ void local_rule_finish('):]
    # The original finish still handles all actor dynamics. The tiled version
    # commits warp partials here so capture retains three policy launches.
    finish_source = finish_source.replace("void local_rule_finish(", "void fast_local_arcs_finish(", 1)
    signature = "    int nodes\n) {\n    const unsigned actor = blockIdx.x;"
    if finish_source.count(signature) != 1:
        raise RuntimeError("the local rule finish signature changed")
    finish_source = finish_source.replace(signature, """    int nodes,
    const int* __restrict__ partial,
    long long* __restrict__ counts,
    float* __restrict__ totals
) {
    const unsigned actor = blockIdx.x;
    fast_commit_arc_stats(partial + static_cast<offset_t>(actor) * fast_stat_stride,
                          counts, totals, gridDim.x, actor);""", 1)
    source = "\n".join((definitions, tensor_helpers, local_helpers, arc_source, finish_source))
    return compile_kernels(source, (*_KERNELS.values(), "fast_local_arcs_finish"), device=device)


def install(policy, *, tiles=1):
    """Install the chosen arc kernel before capturing the policy's CUDA graph.

    Returns the same policy. Other node counts and non-CUDA policies continue
    using their existing backend. Use a fresh policy to compare tile settings.
    """
    if tiles not in _KERNELS:
        raise ValueError("expected 1, 2, or 4 arc tiles")
    if getattr(policy, "_cuda", None) is None or not policy._tensor_transition:
        return policy
    kernels = _kernels(policy.device)
    backend = dict(policy._cuda)
    arc_kernel = kernels[_KERNELS[tiles]]
    # The single-CTA specialization keeps its partials in shared memory, so
    # its final pointer argument is unused.
    partial = (policy._counts if tiles == 1 else
               torch.empty((policy._batch, _STATS_SIZE), device=policy.device, dtype=torch.int32))

    def launch_arcs(grid, block, *args):
        arc_kernel((grid[0], tiles), (block[0] // tiles,), *args, partial)

    backend["local_rule_arcs"] = launch_arcs
    if tiles != 1:
        finish = kernels["fast_local_arcs_finish"]
        counts, totals = policy._counts, policy._totals

        def launch_finish(grid, block, *args):
            finish(grid, block, *args, partial, counts, totals)

        backend["local_rule_finish"] = launch_finish
    policy._cuda = backend
    return policy


def make_policy(birth, actors, device, mode, *, backend="auto", tiles=1):
    """Create the original policy or the specialized CUDA local-rule policy."""
    return install(_make_policy(birth, actors, device, mode, backend=backend), tiles=tiles)
