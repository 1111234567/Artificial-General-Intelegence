"""Check a frozen scout policy across body geometry and mirrored physics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from experiments.pulse_game.scout_neural import ScoutNetPolicy
from experiments.pulse_game.scout_neural_train import load, measure, used_world_seeds
from experiments.pulse_game.scout_world import make_scout_tape, rollout_scout
from experiments.pulse_game.world import swap_sides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=884255)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load(args.model)
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used_world_seeds(metadata).intersection(seeds):
        parser.error("transfer seeds must be held out")

    nominal = make_scout_tape(seeds)
    standard = swap_sides(make_scout_tape(seeds, distance_jitter=0.22, wind_scale=0.12), first_trial=32)
    harder = swap_sides(make_scout_tape(seeds, distance_jitter=0.28, wind_scale=0.122), first_trial=32)
    beacon = harder.scout_side.copy()
    beacon[:, 32:] = 1 - beacon[:, 32:]
    harder = replace(harder, scout_side=beacon)
    mirrored = replace(
        standard, left_id=1 - standard.left_id, distances=standard.distances[..., ::-1].copy(),
        wind=-standard.wind, scout_side=1 - standard.scout_side,
    )
    tapes = {"nominal": nominal, "standard": standard, "harder_and_beacon_moved": harder, "mirrored": mirrored}
    results = {key: rollout_scout(ScoutNetPolicy(model, seed=args.policy_seed), tape) for key, tape in tapes.items()}
    base, twin = results["standard"], results["mirrored"]
    mirror_pair = {
        "same_contact_identity": float((base.contact_id == twin.contact_id).mean()),
        "same_contact_tick": float((base.contact_tick == twin.contact_tick).mean()),
        "same_probe_tick": float((base.scout_tick == twin.scout_tick).mean()),
        "opposite_direction_on_contact": float((1 - base.contact_side[~base.timed_out] == twin.contact_side[~base.timed_out]).mean()),
    }
    print(json.dumps({
        "world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed,
        "conditions": {mode: measure(result, tapes[mode]) for mode, result in results.items()}, "mirror_pairs": mirror_pair,
    }, indent=2))


if __name__ == "__main__":
    main()
