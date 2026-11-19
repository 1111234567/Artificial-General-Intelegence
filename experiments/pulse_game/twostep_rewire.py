"""Revalue a learned two-step route after a single hidden transition reversal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from typing import Callable, Sequence

import numpy as np

from experiments.pulse_game.twostep_agent import DirectedRoutePolicy, RouteMemory
from experiments.pulse_game.twostep_world import (
    AVAILABLE, ENTRANCE_STAGE, ROOM_ID, ROOM_STAGE, STAGE_START, VISIT,
    TwoStepRollout, TwoStepTape, TwoStepWorld, make_tape,
)
from experiments.pulse_game.world import LEFT_DISTANCE, LEFT_ID, Policy, RIGHT_DISTANCE, RIGHT_ID


CASES = ((False, False), (True, False), (False, True), (True, True))
CASE_NAMES = ("stay", "route_only", "value_only", "route_and_value")


@dataclass(frozen=True)
class RewireTape(TwoStepTape):
    ordinary: int
    survey: int
    switch_at: np.ndarray
    reverse_route: np.ndarray

    @property
    def final_entrance(self) -> np.ndarray:
        return super().final_entrance ^ self.reverse_route


def make_rewire_tape(
    seeds: Sequence[int], *, ordinary: int = 32, survey: int = 16, route_reverse: bool = True,
    value_reverse: bool = True, **kwargs,
) -> RewireTape:
    if ordinary < 1 or survey < 8:
        raise ValueError("allow an acquisition period and at least eight survey attempts")
    base = make_tape(seeds, acquisition=ordinary + survey, reversal=value_reverse, **kwargs)
    times = np.asarray([
        ordinary + np.random.default_rng(np.random.SeedSequence([seed, 0xDA278282])).integers(2, min(8, survey - 3))
        for seed in base.seeds
    ])
    return RewireTape(**vars(base), ordinary=ordinary, survey=survey, switch_at=times, reverse_route=np.full(base.n, route_reverse))


def condition(tape: RewireTape, *, route: bool, value: bool) -> RewireTape:
    final = 1 - tape.initial_good_room if value else tape.initial_good_room
    return replace(tape, reverse_route=np.full(tape.n, route), final_good_room=final)


class RewireWorld(TwoStepWorld):
    """Survey offers actual routes and their destinations, with closed fruit access."""

    def __init__(
        self, tape: RewireTape, *, hide_survey_room: bool = False, hide_visit_room: bool = False,
        erase_stage_tags: bool = False, record: bool = False,
    ) -> None:
        self.hide_survey_room = hide_survey_room
        self.erase_stage_tags = erase_stage_tags
        super().__init__(tape, hide_rooms="visits" if hide_visit_room else "never", record=record)

    @property
    def in_survey(self) -> bool:
        return self.tape.ordinary <= self.episode_index < self.tape.acquisition

    def _load_stage(self) -> None:
        super()._load_stage()
        if self.stage != 1:
            return
        if not self.is_visit:
            actual = self._room_id[:, self.episode_index]
            switched = self.tape.reverse_route & (self.episode_index >= self.tape.switch_at) & self._present
            actual[switched] ^= 1
            self._background[:, ROOM_ID] = 0
            if not (self.in_survey and self.hide_survey_room):
                self._background[self._present, ROOM_ID] = np.eye(2, dtype=np.float32)[actual[self._present]]
        if self.in_survey:
            self._available = np.zeros((self.tape.n, 2), dtype=bool)
            self._background[:, AVAILABLE] = 0
            self._background[:, LEFT_ID] = 0
            self._background[:, RIGHT_ID] = 0
            self._background[:, LEFT_DISTANCE:RIGHT_DISTANCE + 1] = 0

    def _active(self) -> np.ndarray:
        return super()._active() & self._available.any(axis=1)

    def observe(self) -> np.ndarray:
        obs = super().observe()
        if self.erase_stage_tags:
            obs[:, [ENTRANCE_STAGE, ROOM_STAGE, STAGE_START, VISIT]] = 0
        return obs


class AdaptiveTransitions(RouteMemory):
    """Replace whole-life averaging by change-tolerant belief updating."""

    def __init__(self, n: int, *, hazard: float = 0.045, estimated_noise: float = 0.20) -> None:
        super().__init__(n)
        self.posterior = np.full(n, 0.5, dtype=np.float32)
        self.hazard = hazard
        self.noise = estimated_noise

    def observe(self, obs: np.ndarray) -> None:
        super().observe(obs)
        bits = (obs[:, ROOM_ID].sum(axis=1) == 1) & (obs[:, STAGE_START] > 0) & (self.entry >= 0)
        event = np.flatnonzero(bits)
        aligned = self.entry[event] == self.room[event]
        predicted = self.hazard + (1 - 2 * self.hazard) * self.posterior[event]
        likely = np.where(aligned, 1 - self.noise, self.noise)
        num = likely * predicted
        self.posterior[event] = num / (num + (1 - likely) * (1 - predicted))

    def probabilities(self) -> np.ndarray:
        p = self.posterior
        return np.stack((p, 1 - p, 1 - p, p), axis=1).reshape(-1, 2, 2)


class AdaptiveRoutePolicy(DirectedRoutePolicy):
    def reset(self, n: int) -> None:
        super().reset(n)
        self.memory = AdaptiveTransitions(n)


def run(new_policy: Callable[[], Policy], tape: RewireTape, *, world_type: type[RewireWorld] = RewireWorld, **world_options) -> TwoStepRollout:
    world = world_type(tape, **world_options)
    policy = new_policy()
    policy.reset(tape.n)
    obs = world.observe()
    while obs is not None:
        obs = world.step(policy.act(obs))
    return world.result()


def profile(tape: RewireTape, result: TwoStepRollout) -> dict:
    did_route = result.entry_id[:, tape.ordinary:tape.acquisition] >= 0
    after = np.arange(tape.ordinary, tape.acquisition) >= tape.switch_at[:, None]
    return {
        "correct_first_entry": result.probe_accuracy,
        "first_entry_contact": float((result.probe_entry >= 0).mean()),
        "first_fruit_contact": float((result.fruit_id[:, tape.probe] >= 0).mean()),
        "probe_terminal_success": float((result.reward[:, tape.probe] > 0).mean()),
        "ordinary_old_entry_last_8": float(result.effective_entry[:, max(0, tape.ordinary - 8):tape.ordinary].mean()),
        "survey_contact_rate": float(did_route.mean()),
        "survey_contacts_after_hidden_change": float((did_route & after).sum(axis=1).mean()),
        "no_survey_or_probe_payoff_display": bool(not result.shown[:, tape.ordinary:tape.acquisition].any() and not result.shown[:, tape.probe].any()),
        "all_survey_fruit_closed": bool((result.fruit_id[:, tape.ordinary:tape.acquisition] < 0).all()),
        "all_value_visits_skip_entry": bool((result.entry_id[:, tape.acquisition:tape.probe] < 0).all()),
    }


def evaluate(new_policy: Callable[[], Policy], tape: RewireTape, **world_options) -> dict:
    results = {}
    correct = []
    chosen = []
    for name, (route, value) in zip(CASE_NAMES, CASES):
        scene = condition(tape, route=route, value=value)
        solved = run(new_policy, scene, **world_options)
        results[name] = profile(scene, solved)
        correct.append(solved.effective_entry[:, tape.probe])
        chosen.append(solved.probe_entry)
    entries = np.stack(chosen)
    distinctive = (entries >= 0).all(axis=0) & (entries[0] == entries[3]) & (entries[1] == entries[2]) & (entries[0] != entries[1])
    per_case = np.stack(correct)
    return {"cells": results, "balanced_accuracy": float(per_case.mean()), "all_four_correct": float(per_case.all(axis=0).mean()), "xor_signature": float(distinctive.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=3160000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=262281)
    args = parser.parse_args()
    conditions = {}
    seeds = range(args.seed_start, args.seed_start + args.lives)
    for rare in (0.0, 0.2):
        tape = make_rewire_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        adaptive = lambda: AdaptiveRoutePolicy(args.policy_seed)
        conditions[f"rare_{rare}"] = {
            "adaptive": evaluate(adaptive, tape),
            "cumulative": evaluate(lambda: DirectedRoutePolicy(args.policy_seed), tape),
            "hide_survey_rooms": evaluate(adaptive, tape, hide_survey_room=True),
            "hide_value_visit_rooms": evaluate(adaptive, tape, hide_visit_room=True),
        }
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": conditions}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
