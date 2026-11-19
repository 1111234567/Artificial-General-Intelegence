"""Experimental warp compaction for sparse 96-node local-rule policies."""

from functools import lru_cache
from pathlib import Path
from textwrap import dedent, indent

from ._cuda_nvrtc import compile_kernels
from .local_rule_policy import PLASTIC_FIELDS, PLASTIC_SIZE, _ARC_THREADS, _CONSTANTS, _COUNTS, _TOTALS


def _section(source, start, stop):
    begin = source.index(start)
    return source[begin:source.index(stop, begin)].rstrip()


@lru_cache(maxsize=None)
def _kernel(device):
    constants = {**_CONSTANTS, **{field.upper(): i for i, field in enumerate(PLASTIC_FIELDS)},
                 **{f"COUNT_{name.upper()}": i for i, name in enumerate(_COUNTS)},
                 **{f"TOTAL_{name.upper()}": i for i, name in enumerate(_TOTALS)},
                 "PLASTIC_SIZE": PLASTIC_SIZE, "ARC_THREADS": _ARC_THREADS, "TENSOR_TRANSITION": 0}
    definitions = "\n".join(f"#define {name} {value!r}f" if isinstance(value, float)
                            else f"#define {name} {value}" for name, value in constants.items())
    original = Path(__file__).with_name("local_rule_policy.cu").read_text(encoding="utf-8")
    local_helpers = original[:original.index('extern "C"')]
    candidate = Path(__file__).with_name("fast_local_arcs.cu").read_text(encoding="utf-8")
    stats_helpers = candidate[:candidate.index("template<int tiles>")]
    # Reuse the single-tile formulas without changing its compiled source.
    # The active pass reloads immutable node features and the gate just written
    # by the same warp, then deposits each float contribution in its old slot.
    cache = _section(candidate, "    const offset_t node_base", "    for (int arc = first_arc")
    cache = (cache.replace("ARC_THREADS / tiles", "ARC_THREADS")
                  .replace("blockIdx.y * tile_threads + thread", "thread")
                  .replace("tiles == 1 ? fast_stat_stride : 1", "fast_stat_stride"))
    gate = _section(candidate, "        const offset_t i = arc_base", "        if (!live &&")
    features = _section(gate, "        const offset_t i", "        const float legacy_support")
    update = _section(candidate, "        float new_weight;", "\n    }\n\n    int* warp_stats")
    update = update.replace("change += fabsf(intrinsic - visible);",
                            "warp_deltas[slot] = __float_as_int(fabsf(intrinsic - visible));")
    compact = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    for label, fragment, spaces in (
        ("CACHE", cache, 4),
        ("GATE", gate, 16),
        ("UPDATE", features + "\n        const float next_gate = gate[i];\n" + update, 12),
    ):
        marker = " " * spaces + f"/* {label} */"
        compact = compact.replace(marker, indent(dedent(fragment), " " * spaces), 1)
    kernel = "fast_local_arcs_compact"
    source = "\n".join((definitions, local_helpers, stats_helpers, compact))
    return compile_kernels(source, kernel, device=device)[kernel]


def install(policy):
    """Install the compact arc kernel before CUDA graph capture."""
    if getattr(policy, "_cuda", None) is None or not policy._tensor_transition:
        return policy
    policy._cuda = {**policy._cuda, "local_rule_arcs": _kernel(policy.device)}
    return policy
