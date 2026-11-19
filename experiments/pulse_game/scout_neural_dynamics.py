"""Evaluate whether a frozen scout actor needs its within-trial recurrent state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.scout_neural import ScoutNetPolicy
from experiments.pulse_game.scout_neural_train import load, measure, tape_for, used_world_seeds
from experiments.pulse_game.scout_search import _paired
from experiments.pulse_game.scout_world import rollout_scout


class CutEdges(ScoutNetPolicy):
    def reset(self, n: int) -> None:
        super().reset(n)
        self.weights = torch.zeros_like(self.weights)


class DropActivity(ScoutNetPolicy):
    def act(self, obs: np.ndarray) -> np.ndarray:
        self.activity.zero_()
        return super().act(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=822976)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, meta = load(args.model)
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used_world_seeds(meta).intersection(seeds):
        parser.error("test lives overlap training or development")
    tape = tape_for(seeds)
    outcomes = {}
    reference = None
    for label, policy in (("intact", ScoutNetPolicy), ("cut_recurrent_edges", CutEdges), ("drop_across_tick_activity", DropActivity)):
        result = rollout_scout(policy(model, seed=args.policy_seed), tape)
        if reference is None:
            reference = result
        outcomes[label] = {**measure(result, tape), "paired_intact_minus_case_accuracy": _paired(reference.score - result.score)}
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": outcomes}, indent=2))


if __name__ == "__main__":
    main()
