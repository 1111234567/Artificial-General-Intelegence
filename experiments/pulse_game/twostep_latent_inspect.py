"""Decode held-out graph and payoff information from the frozen fast banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_latent_train import load
from experiments.pulse_game.twostep_world import ENTRANCE_STAGE, VISIT, make_tape, rollout, set_reversal
from experiments.pulse_game.world import TRIAL_START


class ReadBank(LatentPolicy):
    def reset(self, n: int) -> None:
        super().reset(n)
        self.before = None
        self.after = None

    def logits(self, obs: np.ndarray) -> torch.Tensor:
        current = super().logits(obs)
        if obs[0, TRIAL_START]:
            snap = bool(obs[0, VISIT]) and self.before is None
            final = self.visited and bool(obs[0, ENTRANCE_STAGE])
            if snap or final:
                route, value = self.model.memory.recall(self.memory, self.assoc)
                pair = route.numpy(), value.numpy()
                if snap:
                    self.before = pair
                else:
                    self.after = pair
        return current


def payoff_pairs(tape, good_room: np.ndarray) -> np.ndarray:
    selected = 2 * good_room + tape.good_fruit
    return (np.arange(4)[None, :] == selected[:, None]).astype(np.float64)


def collect(model, seeds: range, *, seed: int) -> dict:
    tape = make_tape(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    all_route, all_value, true_route, true_value = [], [], [], []
    for source in (set_reversal(tape, False), tape):
        policy = ReadBank(model, seed=seed)
        rollout(policy, source)
        for banks, known in ((policy.before, source.initial_good_room), (policy.after, source.final_good_room)):
            route, value = banks
            all_route.append(route)
            all_value.append(value)
            true_route.append(source.common_room)
            true_value.append(payoff_pairs(source, known))
    return {
        "route": (np.concatenate(all_route), np.concatenate(true_route)),
        "value": (np.concatenate(all_value), np.concatenate(true_value)),
    }


def _matrix(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x.reshape(-1, x.shape[-1]).astype(np.float64), np.ones(x.shape[0] * x.shape[1])))


def readout(data: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray]) -> dict:
    learned, y = data
    observed, expected = test
    design = _matrix(learned)
    ridge = 0.001 * np.eye(design.shape[-1])
    weights = np.linalg.solve(design.T @ design + ridge, design.T @ y.ravel())
    scores = (_matrix(observed) @ weights).reshape(expected.shape)
    binary = scores >= 0.5
    truth = expected.astype(bool)
    return {
        "balanced_accuracy": float(0.5 * (binary[truth].mean() + (~binary[~truth]).mean())),
        "accuracy": float((binary == truth).mean()),
        "true_positive_rate": float(binary[truth].mean()),
        "true_negative_rate": float((~binary[~truth]).mean()),
        "top_positive_slot_accuracy": float((scores.argmax(axis=1) == truth.argmax(axis=1)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-start", type=int, required=True)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, meta = load(args.model)
    used = set(range(*meta["development"]))
    for lo, hi in meta["world_ranges"]:
        used.update(range(lo, hi))
    fit = range(args.fit_start, args.fit_start + args.lives)
    test = range(args.test_start, args.test_start + args.lives)
    if meta.get("distinct_fruit", False) or used.intersection(fit) or used.intersection(test) or set(fit).intersection(test):
        parser.error("provide two independent cohorts for a frozen core model")
    source, destination = collect(model, fit, seed=8761), collect(model, test, seed=8762)
    response = {role: readout(source[role], destination[role]) for role in ("route", "value")}
    print(json.dumps({"probe_fit": [fit.start, fit.stop - 1], "probe_test": [test.start, test.stop - 1], "frozen_memories": response}, indent=2))


if __name__ == "__main__":
    main()
