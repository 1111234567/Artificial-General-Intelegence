"""Move the reward-free interval between revaluation and first entrance choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_neural import TwoStepNetPolicy
from experiments.pulse_game.twostep_neural_train import load
from experiments.pulse_game.twostep_world import ENTRANCE_STAGE, VISIT, make_tape
from experiments.pulse_game.world import TRIAL_START, Policy


class QuietBeforeProbe:
    """Advance the controller during an input-silent interval with no movement slots."""

    def __init__(self, inner: Policy, delay: int) -> None:
        self.inner = inner
        self.delay = delay

    def reset(self, n: int) -> None:
        self.inner.reset(n)
        self.visits_seen = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.delay and self.visits_seen and obs[0, TRIAL_START] and obs[0, ENTRANCE_STAGE]:
            quiet = np.zeros_like(obs)
            for _ in range(self.delay):
                self.inner.act(quiet)
        self.visits_seen |= bool(obs[0, VISIT])
        return self.inner.act(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=502927)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for left, right in metadata["world_ranges"]:
        used.update(range(left, right))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds) or metadata.get("distinct_fruit", False):
        parser.error("provide new worlds for a core two-step checkpoint")
    tape = make_tape(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    reports = {}
    for wait in (0, 32, 128, 512, 2048):
        results, _, _ = paired(lambda: QuietBeforeProbe(TwoStepNetPolicy(model, seed=args.policy_seed), wait), tape)
        reports[str(wait)] = results
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "delay_ticks": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
