"""Measure how learned route content changes after actual unrewarded arrivals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_latent_inspect import _matrix
from experiments.pulse_game.twostep_neural import PULSES
from experiments.pulse_game.twostep_rewire import CASES, RewireTape, RewireWorld, condition, make_rewire_tape
from experiments.pulse_game.twostep_rewire_train import load as load_typed
from experiments.pulse_game.twostep_untyped import UntypedPolicy, UntypedWorld, load as load_untyped


@dataclass
class RecallTrace:
    routes: np.ndarray
    values: np.ndarray
    made_entry: np.ndarray


@torch.no_grad()
def traces(factory: Callable, tape: RewireTape, world: RewireWorld) -> RecallTrace:
    policy = factory()
    policy.reset(tape.n)
    routes, values = [], []
    obs = world.observe()
    while obs is not None:
        logits = policy.logits(obs)
        if world.tick_index == 0:
            route, payoff = policy.model.memory.recall(policy.memory, policy.assoc)
            routes.append(route.numpy())
            values.append(payoff.numpy())
        obs = world.step(PULSES[logits.argmax(dim=1).numpy()])
    return RecallTrace(np.stack(routes, axis=1), np.stack(values, axis=1), world.result().entry_id >= 0)


def _fit(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = _matrix(samples)
    return np.linalg.solve(design.T @ design + 0.001 * np.eye(design.shape[-1]), design.T @ target.ravel())


def _targets(tape: RewireTape) -> tuple[np.ndarray, np.ndarray]:
    room = tape.common_room ^ tape.reverse_route[:, None]
    pair = 2 * tape.final_good_room + tape.good_fruit
    return room, np.arange(4)[None, :] == pair[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--untyped", action="store_true")
    parser.add_argument("--fit-start", type=int, required=True)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, meta = (load_untyped if args.untyped else load_typed)(args.model)
    used = set(range(*meta["development"]))
    for low, high in meta["world_ranges"]:
        used.update(range(low, high))
    fit_seeds = range(args.fit_start, args.fit_start + args.lives)
    test_seeds = range(args.test_start, args.test_start + args.lives)
    if used.intersection(fit_seeds) or used.intersection(test_seeds) or set(fit_seeds).intersection(test_seeds):
        parser.error("diagnostic calibration and test worlds must be held out and disjoint")

    def record(seed_set, *, policy_seed: int):
        base = make_rewire_tape(seed_set, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
        result = []
        for route, value in CASES:
            tape = condition(base, route=route, value=value)
            build = UntypedWorld(tape) if args.untyped else RewireWorld(tape, erase_stage_tags=meta["erase_stage_tags"])
            policy = lambda: (UntypedPolicy if args.untyped else LatentPolicy)(model, seed=policy_seed)
            result.append((tape, traces(policy, tape, build)))
        return result

    fit = record(fit_seeds, policy_seed=63031)
    test = record(test_seeds, policy_seed=63032)
    w_route = _fit(np.concatenate([s.routes[:, -1] for _, s in fit]), np.concatenate([_targets(t)[0] for t, _ in fit]))
    w_value = _fit(np.concatenate([s.values[:, -1] for _, s in fit]), np.concatenate([_targets(t)[1] for t, _ in fit]))
    response = {"final_transition_accuracy": [], "final_payoff_top_slot": []}
    timing_slots: dict[int, list[bool]] = {k: [] for k in (0, 1, 2, 3, 4, 6, 8, 10, 12)}
    for tape, s in test:
        expected_room, expected_pair = _targets(tape)
        guessed = (_matrix(s.routes.reshape(-1, 2, s.routes.shape[-1])) @ w_route).reshape(tape.n, tape.episodes, 2)
        guessed_e0 = guessed[..., 0] > guessed[..., 1]
        output_payoff = (_matrix(s.values[:, -1]) @ w_value).reshape(tape.n, 4)
        response["final_transition_accuracy"].extend((guessed_e0[:, -1] == expected_room[:, 0]).tolist())
        response["final_payoff_top_slot"].extend((output_payoff.argmax(axis=1) == expected_pair.argmax(axis=1)).tolist())
        if tape.reverse_route.any():
            for row in range(tape.n):
                contacts = 0
                for tick in range(tape.switch_at[row], tape.acquisition + 1):
                    if contacts in timing_slots:
                        timing_slots[contacts].append(bool(guessed_e0[row, tick] == expected_room[row, 0]))
                    contacts += int(s.made_entry[row, tick])
    summary = {key: float(np.mean(values)) for key, values in response.items()}
    count_curves = {str(k): {"n": len(v), "decoded_new_transition": float(np.mean(v)) if v else None} for k, v in timing_slots.items()}
    print(json.dumps({"fit_seeds": [fit_seeds.start, fit_seeds.stop - 1], "test_seeds": [test_seeds.start, test_seeds.stop - 1], **summary, "after_observed_route_contacts": count_curves}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
