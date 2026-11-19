"""Compare actual-reward objectives on matched independent two-step lives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.pulse_game.twostep_eval import _accuracy, _paired_diff, paired
from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_latent_train import load
from experiments.pulse_game.twostep_world import make_tape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--lives", default=256, type=int)
    parser.add_argument("--policy-seed", default=801227, type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seeds = range(args.seed_start, args.seed_start + args.lives)
    arms = {}
    for name, path in args.model:
        if name in arms:
            parser.error("model names must be unique")
        model, info = load(Path(path))
        seen = set(range(*info["development"]))
        for start, stop in info["world_ranges"]:
            seen.update(range(start, stop))
        if info.get("distinct_fruit", False) or seen.intersection(seeds):
            parser.error(f"{name} needs novel core-task worlds")
        arms[name] = model, info

    results = {}
    for rare in (0.0, 0.2):
        tape = make_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        comparison = {}
        base = None
        for name, (model, info) in arms.items():
            policy = lambda: LatentPolicy(model, seed=args.policy_seed, erase_stage_tags=info.get("erase_stage_tags", False))
            record, a, b = paired(policy, tape)
            per_life = _accuracy(a, b)
            if base is None:
                base = per_life
            record["paired_reference_minus_this"] = _paired_diff(base - per_life)
            comparison[name] = record
        results[f"rare_{rare}"] = comparison
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "reference": next(iter(arms)), "conditions": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
