"""Closed-loop and matched-trajectory tests for lifetime consolidation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import BatchConsolidatingLIF, Consolidation
from .spike_policy import SpikeConfig
from .world import REWARD_NEG, REWARD_POS, TRIAL_START, YokeSchedule, make_tape, rollout


ACQUIRE = 48
STOP = 96
ASSESS = 64
DISCOVERY_CODES = (3, 5, 10, 12)
CONFIRM_CODES = (6, 9)


def curriculum(seeds, *, codes, phases=2):
    """Two staged cue banks and a quiet randomized revisit of both banks."""
    tape = make_tape(seeds, ACQUIRE * phases + ASSESS, map_codes=codes)
    cue = tape.cue.copy()
    for phase in range(phases):
        start = ACQUIRE * phase
        cue[:, start:start + ACQUIRE] = cue[:, start:start + ACQUIRE] % 2 + 2 * (phase % 2)
    good = tape.good_map[np.arange(tape.n)[:, None], tape.ctx, cue]
    return replace(tape, cue=cue, good_id=good)


def behavior(tape, mechanism, config, seed):
    actions = [np.random.SeedSequence((s, 0x434F4E53)) for s in tape.seeds]
    return BatchConsolidatingLIF(mechanism, config, network_seed=seed, action_seeds=actions)


class TrialSchedule:
    """Evaluator intervention on public outcomes and replay choices."""

    def __init__(self, policy, *, quiet_from=STOP, forced=None, budget=None):
        self.policy = policy
        self.quiet_from = quiet_from
        self.forced = forced
        self.budget = budget

    def reset(self, n):
        self.trial = -1
        self.policy.reset(n)

    def act(self, obs):
        if obs[0, TRIAL_START]:
            self.trial += 1
            for row, agent in enumerate(self.policy.agents):
                if self.forced is not None and self.trial < self.forced.shape[1]:
                    agent.forced_identity = int(self.forced[row, self.trial])
                else:
                    agent.forced_identity = None
                if self.budget is not None and self.trial < self.budget.shape[1]:
                    agent.write_budget = float(self.budget[row, self.trial])
                else:
                    agent.write_budget = None
        if self.trial >= self.quiet_from:
            obs = obs.copy()
            obs[:, REWARD_POS:REWARD_NEG + 1] = 0
        return self.policy.act(obs)


def row_scores(tape, result, *, stop):
    score = result.correct
    eval_old = tape.cue[:, stop:] < 2
    out = {
        "pre16": score[:, :16].mean(axis=1),
        "old_learning": score[:, ACQUIRE - 16:ACQUIRE].mean(axis=1),
        "new_learning": score[:, STOP - 16:STOP].mean(axis=1),
        "old_revisit": (score[:, stop:] & eval_old).sum(axis=1) / eval_old.sum(axis=1),
        "new_revisit": (score[:, stop:] & ~eval_old).sum(axis=1) / (~eval_old).sum(axis=1),
        "joint_revisit": score[:, stop:].mean(axis=1),
        "contact_rate": (result.contact_id >= 0).mean(axis=1),
    }
    return {key: value.tolist() for key, value in out.items()}


def execute(name, tape, mechanism, config, seed, *, quiet_from=STOP, yoke=None, forced=None, budget=None, record=False):
    stop = tape.trials - ASSESS
    controller = behavior(tape, mechanism, config, seed)
    policy = TrialSchedule(controller, quiet_from=quiet_from, forced=forced, budget=budget)
    result = rollout(policy, tape, feedback="true" if yoke is None else "yoked", yoke=yoke, record=record)
    row = {"name": name, "network_seed": seed, "mechanism": asdict(mechanism), "config": asdict(config),
           "quiet_from": quiet_from, "feedback": "true" if yoke is None else "yoked", "forced": forced is not None,
           "budget_matched": budget is not None, "scores": row_scores(tape, result, stop=stop),
           "weights": [agent.statistics() for agent in controller.agents],
           "write_norms": [agent.write_norms for agent in controller.agents],
           "correct": result.correct.tolist(), "contact_id": result.contact_id.tolist()}
    scores = row["scores"]
    print(f"{seed:02d} {name:28s} old train {np.mean(scores['old_learning']):.3f} "
          f"new train {np.mean(scores['new_learning']):.3f} "
          f"old/new test {np.mean(scores['old_revisit']):.3f} / {np.mean(scores['new_revisit']):.3f}", flush=True)
    return row, result


def discoveries(tape, seeds):
    base = SpikeConfig(use_residual=True)
    programs = [
        ("ordinary", Consolidation(), base),
        ("protect-12", Consolidation(kind="protect", strength=12), base),
        ("protect-60", Consolidation(kind="protect", strength=60), base),
        ("activity-diagonal", Consolidation(kind="diagonal"), base),
        ("activity-permuted", Consolidation(kind="permuted"), base),
    ]
    for rate in (.015, .055, .2, 1.):
        programs.extend((
            (f"decorrelate-length-{rate}", Consolidation(kind="decorrelated", covariance_rate=rate), base),
            (f"decorrelate-raw-{rate}", Consolidation(kind="decorrelated", covariance_rate=rate, direction="raw"),
             replace(base, learning_rate=.24)),
        ))
    programs.extend((
        ("decorrelate-mixed", Consolidation(kind="decorrelated", covariance_rate=.2, mixing=.80), base),
        ("decorrelate-refresh", Consolidation(kind="decorrelated", covariance_rate=.2, refresh=.01), base),
    ))
    return [execute(name, tape, mechanism, config, seed)[0]
            for seed in seeds for name, mechanism, config in programs]


def paired_yoke(tape, donor):
    if tape.n % 2:
        raise ValueError("yoked conditions need an even life count")
    order = np.arange(tape.n) ^ 1
    return YokeSchedule(donor.observed_reward[order].copy(), order, tuple(tape.seeds[i] for i in order))


def matched_motion(tape, seeds, mechanism, config):
    forced = np.stack([np.random.default_rng(np.random.SeedSequence((s, 761288))).integers(2, size=STOP)
                       for s in tape.seeds])
    results = []
    for seed in seeds:
        candidate, donor = execute("candidate-same-motion", tape, mechanism, config, seed, forced=forced, record=True)
        budget = np.asarray(candidate["write_norms"])
        if budget.shape != forced.shape:
            raise ValueError("exact write match requires one update per exposed trial and life")
        results.append(candidate)
        shadows = (
            ("ordinary-same-motion", Consolidation(), SpikeConfig(use_residual=True), None),
            ("ordinary-budget-matched", Consolidation(), SpikeConfig(use_residual=True), budget),
            ("permuted-budget-matched", replace(mechanism, kind="permuted"), config, budget),
            ("diag-budget-matched", replace(mechanism, kind="diagonal"), config, budget),
        )
        for name, training, base, target in shadows:
            row, replay = execute(name, tape, training, base, seed, forced=forced, budget=target, record=True)
            np.testing.assert_array_equal(replay.pulses[:, :STOP], donor.pulses[:, :STOP])
            np.testing.assert_array_equal(replay.obs[:, :STOP], donor.obs[:, :STOP])
            if target is not None:
                error = np.abs(np.asarray(row["write_norms"]) - budget)
                row["write_budget_match_max_abs"] = float(error.max())
                row["write_budget_match_mean_abs"] = float(error.mean())
            results.append(row)
    return results


def confirm(tape, seeds, mechanism, config):
    base = SpikeConfig(use_residual=True)
    results = []
    for seed in seeds:
        result, donor = execute("candidate", tape, mechanism, config, seed)
        results.append(result)
        for label, rule, cfg, cutoff, yoke in (
            ("ordinary", Consolidation(), base, STOP, None),
            ("candidate-stop-before-new", mechanism, config, ACQUIRE, None),
            ("ordinary-stop-before-new", Consolidation(), base, ACQUIRE, None),
            ("candidate-yoked", mechanism, config, STOP, paired_yoke(tape, donor)),
            ("candidate-no-feedback", mechanism, config, 0, None),
            ("candidate-inverted", mechanism, replace(config, feedback="inverted"), STOP, None),
        ):
            results.append(execute(label, tape, rule, cfg, seed, quiet_from=cutoff, yoke=yoke)[0])
    return results


def paired_analysis(results):
    names = dict.fromkeys(row["name"] for row in results)
    raw = {name: {metric: np.mean([row["scores"][metric] for row in results if row["name"] == name], axis=0)
                  for metric in ("old_learning", "new_learning", "old_revisit", "new_revisit", "joint_revisit")}
           for name in names}
    refs = {
        "candidate": ("ordinary", "candidate-stop-before-new", "candidate-yoked", "candidate-no-feedback"),
        "candidate-same-motion": ("ordinary-same-motion", "ordinary-budget-matched", "permuted-budget-matched",
                                  "diag-budget-matched"),
    }
    diffs = {}
    for candidate, comparators in refs.items():
        if candidate not in names:
            continue
        for reference in comparators:
            this = raw[candidate]
            if reference == "candidate-yoked":
                diffs[reference] = {key: estimates((value - raw[reference][key]).reshape(-1, 2).mean(axis=1))
                                    for key, value in this.items()}
            else:
                diffs[reference] = {key: estimates(value - raw[reference][key]) for key, value in this.items()}
    return {"per_arm": {name: {key: estimates(value) for key, value in scores.items()} for name, scores in raw.items()},
            "pairwise_candidate_minus": diffs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discover", "confirm", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--network-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    codes = DISCOVERY_CODES if args.stage == "discover" else CONFIRM_CODES
    tape = curriculum(list(range(args.start, args.start + args.n)), codes=codes)
    if args.stage == "discover":
        results = discoveries(tape, args.network_seeds)
    else:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        mechanism = Consolidation(**frozen["mechanism"])
        config = SpikeConfig(**frozen["config"])
        runner = confirm if args.stage == "confirm" else matched_motion
        results = runner(tape, args.network_seeds, mechanism, config)
    report = {"stage": args.stage, "life_seeds": tape.seeds, "map_codes": tape.map_codes.tolist(),
              "seed_schedule": args.network_seeds, "results": results, "paired_analysis": paired_analysis(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
