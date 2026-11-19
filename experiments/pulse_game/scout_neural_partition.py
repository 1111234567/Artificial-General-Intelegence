"""Train and confirm on disjoint hidden table patterns with matched cue marginals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.scout_neural import ScoutNetPolicy, ScoutPulseNet
from experiments.pulse_game.scout_neural_train import (
    MODEL_SEED,
    POLICY_SEED,
    Examples,
    behavior_clone,
    collect,
    evaluate,
    load,
    measure,
    receipt_audit,
    save,
    tape_for,
    used_world_seeds,
)
from experiments.pulse_game.scout_world import ScoutTape, rollout_scout
from experiments.pulse_game.world import assert_unseen, make_tape


TRAIN_STARTS = (1550000, 1560000, 1570000)
DEV_START = 1580000


def table_filtered(start: int, n: int, parity: int, *, nominal: bool = False) -> ScoutTape:
    candidates = make_tape(range(start, start + n * 4), trials=1)
    take = candidates.good_map[:, 0].sum(axis=1) % 2 == parity
    selected = tuple(np.asarray(candidates.seeds)[take][:n].tolist())
    if len(selected) < n:
        raise RuntimeError("candidate seed block could not supply enough maps")
    return tape_for(selected, nominal=nominal)


def train(output: Path) -> None:
    torch.manual_seed(MODEL_SEED)
    model = ScoutPulseNet()
    training = [table_filtered(lo, 256, 0) for lo in TRAIN_STARTS]
    development = table_filtered(DEV_START, 128, 0)
    meta = {
        "model_seed": MODEL_SEED, "teacher_lives": 256, "teacher_start": TRAIN_STARTS[0], "agent_dagger_starts": list(TRAIN_STARTS[1:]),
        "world_seed_rosters": {"teacher": training[0].seeds, "dagger_0": training[1].seeds, "dagger_1": training[2].seeds, "development": development.seeds},
        "map_parity": 0, "map_codes": sorted({int(code) for tape in training for code in tape.map_codes}),
    }
    initial = collect(training[0], seed=POLICY_SEED)
    losses = behavior_clone(model, initial, epochs=8, batch=192, seed=MODEL_SEED, lr=0.002)
    print(json.dumps({"stage": "clone", "loss": losses, "development": measure(rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development), development)}), flush=True)
    for index, tape in enumerate(training[1:]):
        extra = collect(tape, seed=POLICY_SEED, model=model, teacher_mix=0.05 - 0.02 * index)
        group = Examples(torch.cat((initial.inputs, extra.inputs)), torch.cat((initial.actions, extra.actions)))
        losses = behavior_clone(model, group, epochs=4, batch=192, seed=MODEL_SEED + index + 1, lr=0.0009)
        print(json.dumps({"stage": "dagger", "index": index, "loss": losses, "development": measure(rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development), development)}), flush=True)
    save(output, model, meta)


def confirm(model_path: Path, start: int, n: int, *, seed: int) -> dict:
    model, meta = load(model_path)
    trained_parity = meta.get("map_parity")
    if trained_parity not in (0, 1):
        raise ValueError("confirmation requires a parity-filtered training checkpoint")
    tape = table_filtered(start, n, 1 - trained_parity)
    evaluation_codes = set(tape.map_codes)
    if evaluation_codes.intersection(meta["map_codes"]):
        raise ValueError("training and evaluation share complete A preference patterns")
    if used_world_seeds(meta).intersection(tape.seeds):
        raise ValueError("training and evaluation share life seeds")
    for seeds in meta["world_seed_rosters"].values():
        assert_unseen(make_tape(seeds, trials=1), tape, disjoint_maps=True)
    return {
        "training_map_codes": meta["map_codes"], "evaluation_map_codes": sorted(int(code) for code in evaluation_codes),
        "evaluation_seed_candidates_start": start, "evaluation_seeds": list(tape.seeds), "policy_seed": seed,
        "results": evaluate(model, tape, seed=seed), "receipt_audit": receipt_audit(model, tape, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    fit = command.add_parser("train")
    fit.add_argument("--output", type=Path, required=True)
    check = command.add_parser("confirm")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--seed-start", type=int, required=True)
    check.add_argument("--lives", type=int, default=256)
    check.add_argument("--policy-seed", type=int, default=631450)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        train(args.output)
    else:
        print(json.dumps(confirm(args.model, args.seed_start, args.lives, seed=args.policy_seed), indent=2))


if __name__ == "__main__":
    main()
