"""Probe a frozen two-step actor with changed geometry or transition noise."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_distinct import fruit_score, make_distinct_tape
from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_neural import TwoStepNetPolicy
from experiments.pulse_game.twostep_neural_train import load
from experiments.pulse_game.twostep_world import make_tape


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--seed-start", type=int, required=True)
    cli.add_argument("--lives", type=int, default=256)
    cli.add_argument("--policy-seed", type=int, default=460359)
    args = cli.parse_args()
    torch.set_num_threads(2)
    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for left, right in metadata["world_ranges"]:
        used.update(range(left, right))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        cli.error("probe lives must be held out")
    distinct = metadata.get("distinct_fruit", False)
    make = make_distinct_tape if distinct else make_tape
    standard = make(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    entry = standard.entry_left.copy()
    fruits = standard.fruit_left.copy()
    entry[:, standard.probe] ^= 1
    fruits[:, standard.probe] ^= 1
    scenarios = {
        "reference": standard,
        "probe_sides_rearranged": replace(standard, entry_left=entry, fruit_left=fruits),
        "hard_geometry": make(seeds, rare_probability=0.2, distance_jitter=0.30, wind_scale=0.16),
        "rare_35_percent": make(seeds, rare_probability=0.35, distance_jitter=0.22, wind_scale=0.12),
        "more_acquisition": make(seeds, acquisition=64, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12),
        "mirrored": replace(standard, entry_left=1 - standard.entry_left, fruit_left=1 - standard.fruit_left, distances=standard.distances[..., ::-1].copy(), wind=-standard.wind),
    }
    results = {}
    traces = {}
    factory = lambda: TwoStepNetPolicy(model, seed=args.policy_seed, mean_visits=distinct)
    for name, tape in scenarios.items():
        result, a, b = paired(factory, tape)
        if distinct:
            result["conditional_fruit_stay"] = fruit_score(tape, a, reversed=False)
            result["conditional_fruit_reverse"] = fruit_score(tape, b, reversed=True)
        results[name] = result
        traces[name] = (a, b)
    parity = {}
    for label, a, b in zip(("stay", "reversed"), traces["reference"], traces["mirrored"]):
        parity[label] = {
            name: float(np.mean(getattr(a, name) == getattr(b, name)))
            for name in ("entry_id", "entry_tick", "room_id", "fruit_id", "fruit_tick", "reward")
        }
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "scenarios": results, "mirror_parity": parity}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
