"""A conditional-fruit protocol for choices after rare room transitions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np

from experiments.pulse_game.twostep_agent import DirectedRoutePolicy
from experiments.pulse_game.twostep_eval import evaluate_revaluation, paired
from experiments.pulse_game.twostep_world import TwoStepRollout, TwoStepTape, make_tape


@dataclass(frozen=True)
class DistinctFruitTape(TwoStepTape):
    final_rich_fruit: np.ndarray
    low_bonus: np.ndarray

    def preferred_fruit(self, rows: np.ndarray, episode: int, room: np.ndarray) -> np.ndarray:
        richest = self.initial_good_room if episode < self.acquisition else self.final_good_room
        rich_fruit = self.good_fruit if episode < self.acquisition else self.final_rich_fruit
        return rich_fruit[rows] ^ (room != richest[rows])

    def positive(self, rows: np.ndarray, episode: int, room: np.ndarray, fruit: np.ndarray) -> np.ndarray:
        richest = self.initial_good_room if episode < self.acquisition else self.final_good_room
        return (fruit == self.preferred_fruit(rows, episode, room)) & ((room == richest[rows]) | self.low_bonus[rows, episode])


def make_distinct_tape(seeds: Sequence[int], *, visits: int = 16, **kwargs) -> DistinctFruitTape:
    """The locally preferred fruit pays 1 or 1/4 across the two rooms.

    In both branches the rich fruit changes identity after acquisition.
    In the reversal branch the room roles also reverse. A simultaneous
    permutation of hidden forced-room labels and room roles thus leaves
    the room-blind joint distribution of fruit identities and signs intact.
    Balanced forced visits expose the low-quality payoff once per four
    exposures of each pair, with random order and real contact required.
    """
    if visits % 16:
        raise ValueError("distinct-fruit visits must balance sixteen exposures")
    base = make_tape(seeds, visits=visits, **kwargs)
    bonus = np.zeros((base.n, base.episodes), dtype=bool)
    visit_range = np.arange(base.acquisition, base.probe)
    for row, seed in enumerate(base.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x9EC72939]))
        bonus[row] = rng.random(base.episodes) < 0.25
        bonus[row, visit_range] = False
        for room in (0, 1):
            for fruit in (0, 1):
                pair = visit_range[(base.forced_room[row, visit_range] == room) & (base.offered_fruit[row, visit_range] == fruit)]
                bonus[row, rng.choice(pair, visits // 16, replace=False)] = True
    return DistinctFruitTape(**vars(base), final_rich_fruit=1 - base.good_fruit, low_bonus=bonus)


def fruit_score(tape: DistinctFruitTape, result: TwoStepRollout, *, reversed: bool) -> dict[str, float | int | None]:
    episode = tape.probe
    arrived = result.room_id[:, episode]
    rich_fruit = tape.final_rich_fruit
    richest = 1 - tape.initial_good_room if reversed else tape.initial_good_room
    poor = (arrived >= 0) & (arrived != richest)
    rich = arrived == richest
    wanted = rich_fruit ^ poor
    chosen = result.fruit_id[:, episode] == wanted
    return {
        "optimal_fruit_all_lives": float((chosen & (arrived >= 0)).mean()),
        "rich_arrivals": int(rich.sum()), "poor_arrivals": int(poor.sum()),
        "optimal_fruit_in_rich_room": float(chosen[rich].mean()) if rich.any() else None,
        "optimal_fruit_in_poor_room": float(chosen[poor].mean()) if poor.any() else None,
    }


def main() -> None:
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--seed-start", type=int, default=2320000)
    options.add_argument("--lives", type=int, default=256)
    options.add_argument("--policy-seed", type=int, default=778032)
    args = options.parse_args()
    tape = make_distinct_tape(range(args.seed_start, args.seed_start + args.lives), rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    graph = lambda: DirectedRoutePolicy(args.policy_seed, mean_visits=True)
    controls = {
        "room_conditioned": graph,
        "global_fruit_identity": lambda: DirectedRoutePolicy(args.policy_seed, mean_visits=True, global_probe_fruit=True),
    }
    report = {}
    for label, policy in controls.items():
        pair, stay, reverse = paired(policy, tape)
        pair["fruit_stay"] = fruit_score(tape, stay, reversed=False)
        pair["fruit_reverse"] = fruit_score(tape, reverse, reversed=True)
        report[label] = pair
    report["value_revaluation_controls"] = evaluate_revaluation(graph, tape, seed=args.policy_seed)
    print(json.dumps({"world_seeds": [tape.seeds[0], tape.seeds[-1]], "policy_seed": args.policy_seed, "conditions": report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
