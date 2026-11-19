"""Pair tagged, masked, and trained-without-tags two-step actors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.pulse_game.twostep_eval import _accuracy, _paired_diff, paired
from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_latent_train import load
from experiments.pulse_game.twostep_world import VISIT, make_tape


class ProbeMask(LatentPolicy):
    def public_frame(self, obs):
        self.erase_stage_tags = self.visited and not bool(obs[0, VISIT])
        return super().public_frame(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tagged", type=Path, required=True)
    parser.add_argument("--trained-masked", type=Path)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=487221)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seeds = range(args.seed_start, args.seed_start + args.lives)

    def heldout(path: Path):
        model, metadata = load(path)
        known = set(range(*metadata["development"]))
        for left, right in metadata["world_ranges"]:
            known.update(range(left, right))
        if known.intersection(seeds):
            parser.error("evaluation worlds overlap training or development")
        return model, metadata

    tagged, history = heldout(args.tagged)
    arms = {
        "tags_train_and_probe": lambda: LatentPolicy(tagged, seed=args.policy_seed),
        "mask_at_evaluation": lambda: LatentPolicy(tagged, seed=args.policy_seed, erase_stage_tags=True),
        "mask_only_at_probe": lambda: ProbeMask(tagged, seed=args.policy_seed),
    }
    same_training_budget = None
    if args.trained_masked:
        masked, record = heldout(args.trained_masked)
        if not record.get("erase_stage_tags", False):
            parser.error("masked checkpoint must be trained without stage tags")
        arms["mask_train_and_probe"] = lambda: LatentPolicy(masked, seed=args.policy_seed, erase_stage_tags=True)
        same_training_budget = history["world_ranges"] == record["world_ranges"] and history["model_seed"] == record["model_seed"]
    evaluations = {}
    for rare in (0.0, 0.2):
        tape = make_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        before = None
        results = {}
        for label, policy in arms.items():
            profile, a, b = paired(policy, tape)
            correct = _accuracy(a, b)
            if before is None:
                before = correct
            profile["relative_loss_from_unmasked"] = _paired_diff(before - correct)
            results[label] = profile
        evaluations[f"rare_{rare}"] = results
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "matched_train_worlds_and_initial_seed": same_training_budget, "conditions": evaluations}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
