"""Compare dynamic revaluation controllers on the same independent worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_eval import _paired_diff
from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_rewire import CASES, CASE_NAMES, condition, make_rewire_tape, profile, run
from experiments.pulse_game.twostep_rewire_train import load as load_typed
from experiments.pulse_game.twostep_untyped import UntypedPolicy, UntypedWorld, load as load_untyped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=3, action="append", metavar=("NAME", "PATH", "TYPE"), required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=620724)
    args = parser.parse_args()
    torch.set_num_threads(2)
    worlds = range(args.seed_start, args.seed_start + args.lives)
    loaded = {}
    for name, path, kind in args.model:
        if kind not in ("typed", "untyped") or name in loaded:
            parser.error("each model needs a unique name and a type of typed or untyped")
        model, metadata = (load_untyped if kind == "untyped" else load_typed)(Path(path))
        known = set(range(*metadata["development"]))
        for begin, end in metadata["world_ranges"]:
            known.update(range(begin, end))
        if known.intersection(worlds):
            parser.error(f"{name} overlaps training or development")
        loaded[name] = model, metadata, kind

    evaluation = {}
    for rare in (0.0, 0.2):
        tape = make_rewire_tape(worlds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        results = {}
        base_score = None
        for name, (model, metadata, kind) in loaded.items():
            arms, correct = {}, []
            for case, (route, value) in zip(CASE_NAMES, CASES):
                trial = condition(tape, route=route, value=value)
                if kind == "untyped":
                    out = run(lambda: UntypedPolicy(model, seed=args.policy_seed), trial, world_type=UntypedWorld)
                else:
                    out = run(lambda: LatentPolicy(model, seed=args.policy_seed), trial, erase_stage_tags=metadata["erase_stage_tags"])
                arms[case] = profile(trial, out)
                correct.append(out.effective_entry[:, trial.probe])
            outcome = np.stack(correct)
            per_life = outcome.mean(axis=0)
            if base_score is None:
                base_score = per_life
            results[name] = {"balanced_accuracy": float(per_life.mean()), "all_four_correct": float(outcome.all(axis=0).mean()),
                             "reference_minus_condition": _paired_diff(base_score - per_life), "cells": arms}
        evaluation[f"rare_{rare}"] = results
    print(json.dumps({"world_seeds": [worlds.start, worlds.stop - 1], "policy_seed": args.policy_seed, "reference": next(iter(loaded)), "conditions": evaluation}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
