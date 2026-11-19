"""Causal interventions on the motor recurrence and visit memory at the first probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_distinct import make_distinct_tape
from experiments.pulse_game.twostep_eval import _accuracy, _paired_diff, paired
from experiments.pulse_game.twostep_neural import TwoStepNetPolicy
from experiments.pulse_game.twostep_neural_train import load
from experiments.pulse_game.twostep_world import ENTRANCE_STAGE, VISIT, make_tape
from experiments.pulse_game.world import TRIAL_START


class ProbeIntervention(TwoStepNetPolicy):
    def __init__(self, *args, lesion: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lesion = lesion

    def reset(self, n: int) -> None:
        super().reset(n)
        self.visited = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.visited |= bool(obs[0, VISIT])
        if self.visited and obs[0, ENTRANCE_STAGE]:
            if self.lesion == "cut_edges":
                intact = self.weights
                self.weights = torch.zeros_like(intact)
                pulses = super().act(obs)
                self.weights = intact
                return pulses
            if self.lesion == "drop_activity" or (self.lesion == "reset_boundary" and obs[0, TRIAL_START]):
                self.activity.zero_()
        return super().act(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=520046)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, meta = load(args.model)
    used = set(range(*meta["development"]))
    for start, stop in meta["world_ranges"]:
        used.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("probe worlds must be held out")
    distinct = meta.get("distinct_fruit", False)
    build = make_distinct_tape if distinct else make_tape
    tape = build(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    conditions = {}
    reference = None
    for lesion in ("intact", "cut_edges", "reset_boundary", "drop_activity", "freeze_visit_writes"):
        result, a, b = paired(lambda: ProbeIntervention(
            model, seed=args.policy_seed, mean_visits=distinct, lesion=lesion,
            write_visit_values=lesion != "freeze_visit_writes",
        ), tape)
        current = _accuracy(a, b)
        if reference is None:
            reference = current
        result["paired_intact_minus_case"] = _paired_diff(reference - current)
        conditions[lesion] = result
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": conditions}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
