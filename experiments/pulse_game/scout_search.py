"""Calibrate local scout policies on training lives and run held-out controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from typing import Sequence

import numpy as np

from experiments.pulse_game.scout_agent import LocalScoutPolicy, ScoutConfig
from experiments.pulse_game.scout_world import (
    SCOUT_HIT,
    ScoutOraclePolicy,
    ScoutRollout,
    ScoutTape,
    ScoutWorld,
    make_context_replay,
    make_scout_tape,
    rollout_scout,
)
from experiments.pulse_game.world import CTX, MOVEMENT_TICKS, assert_unseen, swap_sides


TRAIN_SEEDS = range(470000, 470128)
DEVELOPMENT_SEEDS = range(490000, 490128)
POLICY_SEED = 77819


def _tape(seeds: Sequence[int], *, perturb: bool = True) -> ScoutTape:
    if perturb:
        tape = make_scout_tape(seeds, distance_jitter=0.22, wind_scale=0.12)
        return swap_sides(tape, first_trial=tape.trials // 2)
    return make_scout_tape(seeds)


def _cost(result: ScoutRollout) -> np.ndarray:
    return np.where(result.timed_out, MOVEMENT_TICKS, result.latency_ticks)


def _paired(diff: np.ndarray, *, seed: int = 817) -> dict[str, float | list[float]]:
    indices = np.random.default_rng(seed).integers(len(diff), size=(2500, len(diff)))
    bootstrap = diff[indices].mean(axis=1)
    return {"mean_difference": float(diff.mean()), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}


def _conditions(tape: ScoutTape) -> tuple[np.ndarray, np.ndarray]:
    sensitive = tape.sensitive[np.arange(tape.n)[:, None], tape.cue]
    correct_side = (tape.left_id != tape.good_id).astype(np.int8)
    return sensitive, sensitive & (tape.scout_side != correct_side)


def _metrics(result: ScoutRollout, mandatory: np.ndarray, intents: np.ndarray | None = None) -> dict[str, float]:
    sensitive = result.sensitive_trial
    late = np.arange(result.correct.shape[1]) >= result.correct.shape[1] - 16
    metrics = {
        "success": result.accuracy,
        "success_late": float(result.correct[:, late].mean()),
        "contact": result.contact_rate,
        "timeout": float(result.timed_out.mean()),
        "crossed": result.probe_rate,
        "opposite_contact": result.reverse_rate,
        "mandatory_success": float(result.correct[mandatory].mean()),
        "mandatory_cross_then_correct": float((result.reversed & result.correct)[mandatory].mean()),
        "mandatory_cross_then_correct_late": float((result.reversed & result.correct)[mandatory & late].mean()),
        "mean_ticks": float(_cost(result).mean()),
        "mean_ticks_insensitive": float(_cost(result)[~sensitive].mean()),
        "mean_ticks_sensitive": float(_cost(result)[sensitive].mean()),
        "mean_ticks_late": float(_cost(result)[:, late].mean()),
    }
    if intents is not None:
        metrics.update(
            intent=float(intents.mean()),
            intent_early=float(intents[:, :16].mean()),
            intent_late=float(intents[:, late].mean()),
            intent_sensitive=float(intents[sensitive].mean()),
            intent_insensitive=float(intents[~sensitive].mean()),
            intent_sensitive_late=float(intents[sensitive & late].mean()),
            intent_insensitive_late=float(intents[~sensitive & late].mean()),
        )
    return metrics


def _candidates() -> tuple[list[ScoutConfig], list[ScoutConfig]]:
    base = ScoutConfig()
    adaptive = [base]
    adaptive += [replace(base, novelty=novelty, probe_cost=cost) for novelty in (0.08, 0.11, 0.15) for cost in (0.035, 0.06, 0.08)]
    adaptive += [replace(base, memory_error=error) for error in (0.02, 0.08)]
    actor = [base]
    actor += [replace(base, actor_ucb=ucb, value_step=step) for ucb in (0.12, 0.24, 0.4) for step in (0.18, 0.35, 0.5)]
    return adaptive, actor


@dataclass(frozen=True)
class FrozenScout:
    config: ScoutConfig
    actor_config: ScoutConfig
    script: np.ndarray
    actor_script: np.ndarray


def frozen_scout() -> FrozenScout:
    """Return settings and per-trial script counts frozen on TRAIN_SEEDS."""
    adaptive_counts = (
        63, 59, 53, 55, 49, 50, 57, 54, 58, 64, 49, 57, 54, 59, 46, 58,
        74, 62, 57, 55, 69, 61, 67, 59, 63, 67, 65, 70, 58, 63, 67, 66,
        57, 71, 61, 60, 69, 65, 65, 54, 63, 60, 64, 61, 72, 59, 68, 79,
        64, 78, 58, 65, 65, 68, 61, 64, 62, 68, 68, 69, 68, 58, 67, 79,
    )
    actor_counts = (
        0, 15, 24, 42, 44, 40, 54, 63, 64, 52, 66, 65, 62, 81, 75, 72,
        74, 70, 84, 79, 87, 87, 81, 70, 75, 75, 80, 87, 79, 86, 88, 78,
        82, 92, 87, 79, 85, 91, 85, 88, 80, 89, 83, 79, 93, 83, 88, 94,
        91, 92, 88, 84, 81, 98, 84, 94, 88, 92, 84, 85, 88, 81, 92, 94,
    )
    return FrozenScout(
        replace(ScoutConfig(), probe_cost=0.08, novelty=0.08),
        replace(ScoutConfig(), value_step=0.35, actor_ucb=0.24),
        np.asarray(adaptive_counts) / len(TRAIN_SEEDS),
        np.asarray(actor_counts) / len(TRAIN_SEEDS),
    )


def calibrate(training: ScoutTape) -> tuple[FrozenScout, list[dict]]:
    """Choose common parameters from training lives. All memories reset at evaluation."""
    adaptive_candidates, actor_candidates = _candidates()
    evaluations = []
    evaluated = {}
    for strategy, candidates in (("adaptive", adaptive_candidates), ("reward_actor", actor_candidates)):
        by_accuracy = []
        for config in candidates:
            policy = LocalScoutPolicy(config, strategy=strategy, seed=POLICY_SEED)
            result = rollout_scout(policy, training)
            record = {"strategy": strategy, "config": asdict(config), "accuracy": result.accuracy, "ticks": float(_cost(result).mean())}
            evaluations.append(record)
            by_accuracy.append((record, config, policy.intents))
        highest = max(record["accuracy"] for record, _, _ in by_accuracy)
        viable = [entry for entry in by_accuracy if highest - entry[0]["accuracy"] <= 0.0025]
        evaluated[strategy] = min(viable, key=lambda entry: (entry[0]["ticks"], -entry[0]["accuracy"]))

    _, config, training_intents = evaluated["adaptive"]
    _, actor_config, actor_intents = evaluated["reward_actor"]
    frozen = FrozenScout(config, actor_config, training_intents.mean(axis=0), actor_intents.mean(axis=0))
    return frozen, evaluations


def causal_receipt_audit(tape: ScoutTape, *, config: ScoutConfig, seed: int = POLICY_SEED) -> dict[str, float | int]:
    """Flip only the observed context at a late-trial beacon receipt."""
    world = ScoutWorld(tape)
    policy = LocalScoutPolicy(config, seed=seed)
    policy.reset(tape.n)
    _, required = _conditions(tape)
    eligible = 0
    responded = 0
    would_turn = 0
    flipped_choice = 0
    obs = world.observe()
    while obs is not None:
        trial = world.trial_index
        matched = np.flatnonzero(required[:, trial] & (obs[:, SCOUT_HIT] == 1)) if trial >= tape.trials - 16 else []
        if len(matched):
            # Fork the same local learner state before either version of the pulse.
            alternate = deepcopy(policy)
            changed = obs.copy()
            changed[matched, CTX] = obs[matched, CTX][:, ::-1]
            alternative = alternate.act(changed)
            pulses = policy.act(obs)
            good_side = 1 - tape.scout_side[matched, trial]
            turns = pulses[matched, good_side] & ~pulses[matched, 1 - good_side]
            counterpattern = ~alternative[matched, good_side] & alternative[matched, 1 - good_side]
            responded += len(matched)
            would_turn += int(turns.sum())
            flipped_choice += int((turns & counterpattern).sum())
        else:
            pulses = policy.act(obs)
        if world.tick_index == 0 and trial >= tape.trials - 16:
            eligible += int(required[:, trial].sum())
        obs = world.step(pulses)
    return {
        "eligible": eligible,
        "physically_receipted": responded,
        "turns_on_real_receipt": would_turn,
        "turned_only_with_real_context": flipped_choice,
    }


def _contrast(a: ScoutRollout, b: ScoutRollout) -> dict:
    return {
        "success": _paired(a.score - b.score),
        "success_late": _paired(a.correct[:, -16:].mean(axis=1) - b.correct[:, -16:].mean(axis=1)),
        "ticks": _paired(_cost(a).mean(axis=1) - _cost(b).mean(axis=1)),
    }


def evaluate(tape: ScoutTape, frozen: FrozenScout, *, seed: int = POLICY_SEED) -> dict:
    config = frozen.config
    adaptive = LocalScoutPolicy(config, seed=seed)
    true = rollout_scout(adaptive, tape)
    _, mandatory = _conditions(tape)
    replay = make_context_replay(true, seed=67751)
    actor = LocalScoutPolicy(frozen.actor_config, seed=seed, strategy="reward_actor")
    actor_true = rollout_scout(actor, tape)
    actor_replay = make_context_replay(actor_true, seed=67751)
    controls = {
        "all_probe": (LocalScoutPolicy(config, seed=seed, strategy="all_probe"), {}),
        "never_probe": (LocalScoutPolicy(config, seed=seed, strategy="never_probe"), {}),
        "passive": (LocalScoutPolicy(config, seed=seed, strategy="passive"), {}),
        "fixed_schedule": (LocalScoutPolicy(config, seed=seed, strategy="fixed_schedule", probe_schedule=frozen.script), {}),
        "replayed_sensor": (LocalScoutPolicy(config, seed=seed), {"sensor": "replay", "replay": replay}),
        "closed_sensor": (LocalScoutPolicy(config, seed=seed), {"sensor": "none"}),
        "frozen_synapses": (LocalScoutPolicy(config, seed=seed, learn=False), {}),
        "no_exploration_critic": (LocalScoutPolicy(replace(config, actor_weight=0.0), seed=seed), {}),
        "actor_fixed_schedule": (
            LocalScoutPolicy(frozen.actor_config, seed=seed, strategy="fixed_schedule", probe_schedule=frozen.actor_script), {}
        ),
        "actor_closed_sensor": (LocalScoutPolicy(frozen.actor_config, seed=seed, strategy="reward_actor"), {"sensor": "none"}),
        "actor_replayed_sensor": (
            LocalScoutPolicy(frozen.actor_config, seed=seed, strategy="reward_actor"), {"sensor": "replay", "replay": actor_replay}
        ),
    }
    results = {"adaptive": _metrics(true, mandatory, adaptive.intents), "reward_actor": _metrics(actor_true, mandatory, actor.intents)}
    contrasts = {"reward_actor": _contrast(true, actor_true)}
    actor_contrasts = {}
    for name, (policy, settings) in controls.items():
        result = rollout_scout(policy, tape, **settings)
        results[name] = _metrics(result, mandatory, policy.intents)
        if name.startswith("actor_"):
            actor_contrasts[name.removeprefix("actor_")] = _contrast(actor_true, result)
        else:
            contrasts[name] = _contrast(true, result)
    for name, probe in (("oracle", True), ("blind_oracle", False)):
        oracle = rollout_scout(ScoutOraclePolicy(tape, probe=probe), tape, sensor="true" if probe else "none")
        results[name] = _metrics(oracle, mandatory)
    return {
        "results": results,
        "adaptive_minus_control": contrasts,
        "reward_actor_minus_control": actor_contrasts,
        "causal_receipt": causal_receipt_audit(tape, config=config, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=DEVELOPMENT_SEEDS.start)
    parser.add_argument("--lives", type=int, default=len(DEVELOPMENT_SEEDS))
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--recalibrate", action="store_true", help="reproduce the training sweep before evaluation")
    args = parser.parse_args()
    if args.lives < 2:
        parser.error("at least two lives are required for deranged replay")
    training = _tape(TRAIN_SEEDS)
    tape = _tape(range(args.seed_start, args.seed_start + args.lives), perturb=not args.nominal)
    assert_unseen(training, tape)
    frozen = frozen_scout()
    sweep = None
    if args.recalibrate:
        reproduced, sweep = calibrate(training)
        if reproduced.config != frozen.config or reproduced.actor_config != frozen.actor_config:
            raise RuntimeError("calibration did not reproduce the frozen configurations")
        np.testing.assert_array_equal(reproduced.script, frozen.script)
        np.testing.assert_array_equal(reproduced.actor_script, frozen.actor_script)
    report = evaluate(tape, frozen)
    report["training_seeds"] = [TRAIN_SEEDS.start, TRAIN_SEEDS.stop - 1]
    report["evaluation_seeds"] = [tape.seeds[0], tape.seeds[-1]]
    report["perturbed"] = not args.nominal
    report["frozen"] = {
        "config": asdict(frozen.config), "actor_config": asdict(frozen.actor_config),
        "script": frozen.script.tolist(), "actor_script": frozen.actor_script.tolist(),
    }
    if sweep is not None:
        report["calibration"] = sweep
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
