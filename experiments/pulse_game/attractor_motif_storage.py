"""Intervene on passive readout state while leaving every future spike intact."""

import argparse
from pathlib import Path
import json

import numpy as np

from .attractor_motif import GAP, choices, vector
from .attractor_motif_find import experiment
from .spike_analysis import estimates


ANCHOR = 18
SEEDS = tuple(range(8_903_000_001, 8_903_000_065))


def mean_before_read(enc, after_state, trace):
    mean = after_state.copy()
    for t in range(ANCHOR + 1):
        mean += enc.mean_rate * (trace[t] - mean)
    return mean


def propagate(enc, trace, mean, recorded_spikes):
    trace, mean = trace.copy(), mean.copy()
    for t in range(ANCHOR + 1, GAP + 1):
        trace = enc.trace_decay * trace + (1. - enc.trace_decay) * recorded_spikes[t]
        mean += enc.mean_rate * (trace - mean)
    return ((trace - mean) @ enc.proj.T).reshape(-1, 8, enc.proj.shape[0])


def pool(x):
    return np.repeat(x.reshape(-1, 8, x.shape[-1]).mean(axis=1), 8, axis=0)


def run(genes, birth):
    enc, fork, w, readable, core = experiment(genes, birth, SEEDS)
    traces, activity = (readable["state"][k] for k in ("trace", "spike"))
    a = traces[ANCHOR]
    m = mean_before_read(enc, fork.start.mean, traces)
    direct = readable["address"]
    np.testing.assert_array_equal(propagate(enc, a, m, activity), direct)
    index = np.arange(len(fork.labels)) // 8 * 8 + (fork.labels ^ 1)
    swapped_goals = fork.goals[:, np.arange(8) ^ 1]
    base = choices(w, readable, fork)
    interventions = {"pool_trace": (pool(a), m), "pool_mean": (a, pool(m)), "pool_both": (pool(a), pool(m)),
                     "swap_trace": (a[index], m), "swap_mean": (a, m[index]), "swap_both": (a[index], m[index])}
    registry = {}
    for label, (trace, mean) in interventions.items():
        f = propagate(enc, trace, mean, activity)
        z = choices(w, {"address": f}, fork)
        registry[label] = {"actual": estimates((z == fork.goals).mean(1)),
                           "actual_minus_baseline": estimates(((z == fork.goals).astype(float) - (base == fork.goals)).mean(1)),
                           "donor": estimates((z == swapped_goals).mean(1)),
                           "donor_minus_baseline": estimates(((z == swapped_goals).astype(float) - (base == swapped_goals)).mean(1)),
                           "max_difference_from_entire_donor_readout": float(np.abs(f - direct[:, np.arange(8) ^ 1]).max())}
    outside_anchor = activity[ANCHOR + 1:GAP + 1].sum(axis=(0, 2)).reshape(-1, 8)
    return {"prefix_control": core, "baseline_donor_correct": estimates((base == swapped_goals).mean(1)),
            "fraction_no_spike_after_anchor": float((outside_anchor == 0).mean()),
            "last_output_states": registry}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    o = p.parse_args()
    genes = vector()
    all_rows = {str(birth): run(genes, birth) for birth in (43, 151)}
    o.artifact.write_text(json.dumps({"snapshot_after_tick": ANCHOR, "test_worlds": SEEDS, "graphs": all_rows},
                                     separators=(",", ":")) + "\n", encoding="utf-8")
    print({b: {k: round(v["actual"]["mean"], 4) for k, v in result["last_output_states"].items()}
           for b, result in all_rows.items()}, flush=True)


if __name__ == "__main__":
    main()
