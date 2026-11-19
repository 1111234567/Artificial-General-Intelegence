"""Same-state physical offer interventions and whole-life budget changes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import active_symmetry_agent as p, active_symmetry_world as w
from .active_symmetry_train import assessment, average_world
from .spike_analysis import estimates


def actual_first(panel, params, credits, fees, given):
    n = len(panel.seeds)
    positioned = np.arange(n)
    archetype = replace(panel, horizon=np.full(n, credits, dtype=np.int16), costs=np.full((n, 2), fees))
    newborn = np.zeros((n, 2))
    affect = archetype.arena[:, 0]
    newborn[positioned, affect] = given
    offer = w.Offer(affect, archetype.costs, archetype.horizon, archetype.probe_side[:, 0])
    option, _, likelihood = p.decide(p.Gate("learned", params), offer, newborn, newborn,
                                     newborn, positioned, 0, np.zeros(n))
    _, moved, arrived = w.reach_gate(archetype, positioned, 0, option)
    assert arrived.all()
    return moved.astype(float), likelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=192)
    args = parser.parse_args()
    sets = [json.loads(k.read_text(encoding="utf-8")) for k in args.checkpoints]
    seeds = tuple(range(args.start, args.start + args.worlds))
    for collection in sets:
        if set(seeds) & (set(collection["training_worlds"]) | set(collection["development_worlds"])):
            raise ValueError("Closed-loop controls must remain held out")
    origin = w.build(seeds)
    per_config = {}
    for days in (4, 8, 14):
        for cost in (min(w.COSTS), max(w.COSTS)):
            life = replace(origin, horizon=np.full(args.worlds, days, dtype=np.int16), costs=np.full_like(origin.costs, cost))
            trials = []
            for r in sets:
                trials.append(assessment(p.lifetime(life, p.Gate("learned", np.asarray(r["selected_params"])), 211), life))
            per_config[(days, cost)] = average_world(trials)
    paired = {}
    for label, a, b in (
        ("cost_at_4", (4, .10), (4, .72)), ("cost_at_8", (8, .10), (8, .72)),
        ("cost_at_14", (14, .10), (14, .72)), ("extra_days_low_cost", (14, .10), (4, .10)),
        ("extra_days_high_cost", (14, .72), (4, .72)),
    ):
        paired[label] = {k: estimates(per_config[a][k] - per_config[b][k]) for k in per_config[a]}
    physical = {}
    for label, a, b in (
        ("low_minus_high_when_4_unknown", (4, .10, 0.), (4, .72, 0.)),
        ("low_minus_high_when_14_unknown", (14, .10, 0.), (14, .72, 0.)),
        ("many_minus_few_when_low_unknown", (14, .10, 0.), (4, .10, 0.)),
        ("many_minus_few_when_high_unknown", (14, .72, 0.), (4, .72, 0.)),
        ("unknown_minus_known_same_low_long_offer", (14, .10, 0.), (14, .10, 1.5)),
    ):
        cells = []
        for r in sets:
            params = np.asarray(r["selected_params"])
            x, px = actual_first(origin, params, *a)
            y, py = actual_first(origin, params, *b)
            cells.append((x-y, px-py, x, y))
        sample = np.mean(cells, axis=0)
        physical[label] = {"physical_probe_difference": estimates(sample[0]), "propensity_difference": estimates(sample[1]),
                           "first_offer_physical_probe": estimates(sample[2]), "second_offer_physical_probe": estimates(sample[3])}
    out = {"world_seeds": seeds, "actual_budget_pairs": paired, "actual_gate_effects_given_same_body": physical,
           "whole_life": {str(key): {k: estimates(v) for k, v in sub.items()} for key, sub in per_config.items()}}
    args.artifact.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
