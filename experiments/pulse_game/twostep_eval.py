"""Paired first-choice readout after revaluation without an entry action."""

from __future__ import annotations

import argparse
import json
from typing import Callable

import numpy as np

from experiments.pulse_game.twostep_agent import CachedTerminalPolicy, DirectedRoutePolicy, OraclePolicy
from experiments.pulse_game.twostep_world import TwoStepRollout, TwoStepTape, make_tape, make_visit_replay, rollout, set_reversal
from experiments.pulse_game.world import Policy, REWARD_NEG, REWARD_POS


def _paired_diff(values: np.ndarray, *, seed: int = 606470) -> dict[str, float | list[float]]:
    draw = np.random.default_rng(seed).integers(len(values), size=(2500, len(values)))
    interval = np.quantile(values[draw].mean(axis=1), (0.025, 0.975))
    return {"mean": float(values.mean()), "ci95": interval.tolist()}


def _episode_stats(tape: TwoStepTape, result: TwoStepRollout) -> dict[str, float]:
    tail = slice(max(0, tape.acquisition - 8), tape.acquisition)
    return {
        "old_start_accuracy_last_8": float(result.effective_entry[:, tail].mean()),
        "old_terminal_success_last_8": float((result.reward[:, tail] > 0).mean()),
        "probe_optimal_entrance": result.probe_accuracy,
        "probe_terminal_success": float((result.reward[:, tape.probe] > 0).mean()),
        "probe_entrance_contact": float((result.probe_entry >= 0).mean()),
        "probe_fruit_contact": float((result.fruit_id[:, tape.probe] >= 0).mean()),
        "both_pivotal_visit_contacts": float(result.visits_confirmed.mean()),
        "visit_context_without_entry": float((result.entry_id[:, tape.acquisition:tape.probe] < 0).mean()),
    }


def paired(
    new_policy: Callable[[], Policy],
    tape: TwoStepTape,
    *,
    record: bool = False,
    **world_options,
) -> tuple[dict, TwoStepRollout, TwoStepRollout]:
    stay_tape = set_reversal(tape, False)
    reverse_tape = set_reversal(tape, True)
    stay = rollout(new_policy(), stay_tape, record=record, **world_options)
    reverse = rollout(new_policy(), reverse_tape, record=record, **world_options)
    directionally_correct = stay.effective_entry[:, tape.probe] & reverse.effective_entry[:, tape.probe]
    changed = (stay.probe_entry != reverse.probe_entry) & (stay.probe_entry >= 0) & (reverse.probe_entry >= 0)
    zero_probe_feedback = bool(not np.any(stay.shown[:, tape.probe]) and not np.any(reverse.shown[:, tape.probe]))
    old_known = stay.effective_entry[:, tape.acquisition - 1]
    prefix = slice(0, tape.acquisition)
    visits = slice(tape.acquisition, tape.probe)
    report = {
        "stay": _episode_stats(stay_tape, stay), "reversed": _episode_stats(reverse_tape, reverse),
        "balanced_optimal_entrance": (stay.probe_accuracy + reverse.probe_accuracy) / 2,
        "changed_first_entrance": float(changed.mean()),
        "changed_to_correct_new_entrance": float(directionally_correct.mean()),
        "changed_when_old_entry_learned": float(directionally_correct[old_known].mean()) if old_known.any() else None,
        "identical_acquisition": bool(np.array_equal(stay.entry_id[:, prefix], reverse.entry_id[:, prefix]) and np.array_equal(stay.shown[:, prefix], reverse.shown[:, prefix])),
        "no_probe_reward_observed": zero_probe_feedback,
    }
    if record:
        public = [i for i in range(stay.obs.shape[-1]) if i not in (REWARD_POS, REWARD_NEG)]
        report["identical_acquisition_observation_and_pulses"] = bool(
            np.array_equal(stay.obs[:, prefix], reverse.obs[:, prefix]) and np.array_equal(stay.pulses[:, prefix], reverse.pulses[:, prefix])
        )
        report["identical_visit_physics_barring_sign"] = bool(
            np.array_equal(stay.obs[:, visits][..., public], reverse.obs[:, visits][..., public]) and np.array_equal(stay.pulses[:, visits], reverse.pulses[:, visits])
        )
        report["identical_initial_probe_observations"] = bool(np.array_equal(stay.obs[:, tape.probe, 0], reverse.obs[:, tape.probe, 0]))
    return report, stay, reverse


def _accuracy(a: TwoStepRollout, b: TwoStepRollout) -> np.ndarray:
    return 0.5 * (a.effective_entry[:, -1].astype(np.float32) + b.effective_entry[:, -1])


def evaluate_revaluation(new_policy: Callable[[], Policy], tape: TwoStepTape, *, seed: int, record: bool = False) -> dict:
    tests = {}
    true, stay, reverse = paired(new_policy, tape, record=record)
    actual = _accuracy(stay, reverse)
    replay = make_visit_replay(reverse, seed=seed)
    tests["own_feedback"] = true
    for label, settings in (("room_identity_hidden_during_visits", {"hide_rooms": "visits"}), ("peer_reward_on_own_visit_contact", {"visit_replay": replay})):
        report, a, b = paired(new_policy, tape, **settings)
        report["paired_true_minus_condition"] = _paired_diff(actual - _accuracy(a, b))
        tests[label] = report
    comparison = lambda: DirectedRoutePolicy(seed, cached_probe=True)
    tests["cached_entrance_control"], _, _ = paired(comparison, tape)
    tests["separately_acquired_terminal_cache"], _, _ = paired(lambda: CachedTerminalPolicy(seed), tape)
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=2150000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=99227)
    parser.add_argument("--record-proof", action="store_true")
    args = parser.parse_args()
    conditions = {}
    seeds = range(args.seed_start, args.seed_start + args.lives)
    for rare in (0.0, 0.2):
        tape = make_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        result = evaluate_revaluation(lambda: DirectedRoutePolicy(args.policy_seed), tape, seed=args.policy_seed, record=args.record_proof)
        oracle = rollout(OraclePolicy(tape), tape)
        result["oracle"] = _episode_stats(tape, oracle)
        conditions[f"rare_{rare}"] = result
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": conditions}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
