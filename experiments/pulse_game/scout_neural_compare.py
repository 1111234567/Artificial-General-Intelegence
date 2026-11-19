"""Pair frozen scout checkpoints over the same held-out trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.scout_neural import ScoutNetPolicy
from experiments.pulse_game.scout_neural_train import load, measure, tape_for, used_world_seeds
from experiments.pulse_game.scout_search import _paired
from experiments.pulse_game.scout_world import ScoutOraclePolicy, rollout_scout
from experiments.pulse_game.world import MOVEMENT_TICKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=642264)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seeds = range(args.seed_start, args.seed_start + args.lives)
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)) or "blind_oracle" in names:
        parser.error("model names must be distinct and may not use blind_oracle")
    tape = tape_for(seeds)
    outcomes = {}
    for name, path in args.model:
        model, metadata = load(Path(path))
        if used_world_seeds(metadata).intersection(seeds):
            parser.error(f"{name} saw at least one evaluation life")
        policy = ScoutNetPolicy(model, seed=args.policy_seed)
        outcomes[name] = rollout_scout(policy, tape)
    outcomes["blind_oracle"] = rollout_scout(ScoutOraclePolicy(tape, probe=False), tape, sensor="none")

    reference = outcomes[names[0]]
    reference_cost = np.where(reference.timed_out, MOVEMENT_TICKS, reference.latency_ticks).mean(axis=1)
    comparisons = {}
    for name, outcome in outcomes.items():
        cost = np.where(outcome.timed_out, MOVEMENT_TICKS, outcome.latency_ticks).mean(axis=1)
        comparisons[name] = {
            "accuracy": _paired(reference.score - outcome.score),
            "late_accuracy": _paired(reference.correct[:, -16:].mean(axis=1) - outcome.correct[:, -16:].mean(axis=1)),
            "mean_ticks": _paired(reference_cost - cost),
        }
    print(json.dumps({
        "world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "reference": names[0],
        "conditions": {name: measure(outcome, tape) for name, outcome in outcomes.items()},
        "paired_reference_minus_condition": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
