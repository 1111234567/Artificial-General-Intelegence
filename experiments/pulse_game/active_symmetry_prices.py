"""Physically verify fee and horizon response across prespecified offer states."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import active_symmetry_agent as p, active_symmetry_world as w
from .spike_analysis import estimates


CREDITS = (1, 2, 4, 6, 8, 10, 14)
GAINS = (0., .45, 1., 1.5)


def physically(panel, weights, cost, horizon, current_conf):
    n = len(panel.seeds)
    ids = np.arange(n)
    present = panel.arena[:, 0]
    cap = np.full_like(panel.horizon, horizon)
    fees = np.full_like(panel.costs, .36)
    fees[ids, present] = cost
    combo = replace(panel, horizon=cap, costs=fees)
    g = np.full((n, 2), .50)
    g[ids, present] = current_conf
    offer = w.Offer(present, fees, cap, panel.probe_side[:, 0])
    did, _, odds = p.decide(p.Gate("learned", weights), offer, g, g, g, ids, 0, np.zeros(n))
    _, probe, arrives = w.reach_gate(combo, ids, 0, did)
    if not arrives.all():
        raise ValueError("A physical offer required a gate contact")
    return probe.astype(np.float64), odds


def effect(panel, settings):
    cases = []
    means, wrong = [], []
    for horizon in CREDITS:
        for stored in GAINS:
            outcomes = []
            for model in settings:
                parameters = np.asarray(model["selected_params"])
                a, pa = physically(panel, parameters, .10, horizon, stored)
                b, pb = physically(panel, parameters, .72, horizon, stored)
                outcomes.append((a-b, pa-pb, a, b, (a < b).astype(float)))
            contrast, slope, low, high, reversed_ = np.mean(outcomes, axis=0)
            means.append(contrast)
            wrong.append(reversed_)
            cases.append({"remaining": horizon, "confidence": stored, "low_minus_high_probe": estimates(contrast),
                          "mean_propensity_difference": estimates(slope), "low_fee": estimates(low),
                          "high_fee": estimates(high), "reversed_case_rate": float(reversed_.mean())})
    return {"cells": cases, "average_physical_effect": estimates(np.mean(means, axis=0)),
            "wrong_direction_fraction": estimates(np.mean(wrong, axis=0)),
            "episodes_compare_physically": len(cases) * len(settings) * 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--v1", type=Path, nargs="+", required=True)
    parser.add_argument("--v2", type=Path, nargs="+", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=192)
    args = parser.parse_args()
    groups = {"original": [json.loads(f.read_text(encoding="utf-8")) for f in args.v1],
              "paired": [json.loads(f.read_text(encoding="utf-8")) for f in args.v2]}
    seeds = tuple(range(args.start, args.start + args.n))
    for lines in groups.values():
        for x in lines:
            if set(seeds) & (set(x["training_worlds"]) | set(x["development_worlds"])):
                raise ValueError("A held offer used a training world")
    world = w.build(seeds)
    report = {name: effect(world, models) for name, models in groups.items()}
    args.artifact.write_text(json.dumps({"held_worlds": seeds, "results": report}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
