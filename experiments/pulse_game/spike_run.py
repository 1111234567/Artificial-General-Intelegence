"""Run matched closed-loop spiking controls and save per-life results."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .spike_policy import BatchSpikingPolicy, SpikeConfig
from .world import REWARD_NEG, REWARD_POS, TRIAL_START, make_tape, make_yoke, rollout, swap_sides


BALANCED_CODES = (3, 5, 6, 9, 10, 12)


def build_policy(config, tape, network_seed, *, quiet_from=None, silent_trials=()):
    action_seeds = [np.random.SeedSequence((seed, 0x4C49464C)) for seed in tape.seeds]
    behavior = BatchSpikingPolicy(config, network_seed=network_seed, action_seeds=action_seeds)
    if quiet_from is not None or silent_trials:
        return FeedbackMask(behavior, quiet_from, silent_trials)
    return behavior


class FeedbackMask:
    """Evaluation intervention on the public input to a continuously running life."""

    def __init__(self, policy, quiet_from, silent_trials):
        self.policy = policy
        self.quiet_from = float("inf") if quiet_from is None else quiet_from
        self.silent_trials = frozenset(silent_trials)

    @property
    def agents(self):
        return self.policy.agents

    def reset(self, n):
        self.trial = -1
        self.policy.reset(n)

    def act(self, obs):
        self.trial += int(obs[0, TRIAL_START] > 0)
        if self.trial >= self.quiet_from or self.trial in self.silent_trials:
            obs = obs.copy()
            obs[:, REWARD_POS:REWARD_NEG + 1] = 0
        return self.policy.act(obs)


def staged_tape(seeds, *, jitter=0, wind=0, review_trials=()):
    """Old pair, new pair, then revisits of the old pair, with unchanged maps."""
    tape = make_tape(seeds, 144, distance_jitter=jitter, wind_scale=wind, map_codes=BALANCED_CODES)
    # Reuse the independently drawn low cue bit, without changing context or layout.
    cue = tape.cue % 2
    cue[:, 48:96] += 2
    if review_trials:
        cue[:, review_trials] -= 2
    good_id = tape.good_map[np.arange(tape.n)[:, None], tape.ctx, cue]
    return replace(tape, cue=cue, good_id=good_id)


def summarize(result, tape, agents, *, window):
    window = min(window, result.correct.shape[1] // 2)
    visits = np.zeros_like(tape.cue, dtype=np.int32)
    pairs = []
    for row in range(tape.n):
        count = np.zeros((2, tape.good_map.shape[-1]), dtype=np.int32)
        scores = [[[] for _ in range(tape.good_map.shape[-1])] for _ in range(2)]
        for trial, (ctx, cue) in enumerate(zip(tape.ctx[row], tape.cue[row], strict=True)):
            visits[row, trial] = count[ctx, cue]
            count[ctx, cue] += 1
            scores[ctx][cue].append(bool(result.correct[row, trial]))
        matches = [[], []]
        for a, b in zip(*scores, strict=True):
            for visit, (one, two) in enumerate(zip(a, b)):
                matches[visit >= 4].append(one and two)
        pairs.append([[int(sum(part)), len(part)] for part in matches])
    visit_groups = [visits == 0, (visits > 0) & (visits < 4), (visits >= 4) & (visits < 8), visits >= 8]
    return {
        "accuracy": result.accuracy,
        "contact_rate": result.contact_rate,
        "early": result.correct[:, :window].mean(axis=1).tolist(),
        "late": result.correct[:, -window:].mean(axis=1).tolist(),
        "window_size": window,
        "rolling16": [
            result.correct[:, start:start + 16].mean(axis=1).tolist()
            for start in range(0, tape.trials, 16)
        ],
        "visit_buckets_correct_count": [
            np.stack(((result.correct & group).sum(axis=1), group.sum(axis=1)), axis=1).tolist()
            for group in visit_groups
        ],
        "opposing_context_pair_correct_count": pairs,
        "correct_by_trial": result.correct.astype(np.int8).tolist(),
        "contact_id_by_trial": result.contact_id.tolist(),
        "network_statistics": [agent.statistics() for agent in agents],
    }


def evaluate(name, config, tape, network_seed, *, feedback="true", yoke=None, quiet_from=None, silent_trials=()):
    policy = build_policy(config, tape, network_seed, quiet_from=quiet_from, silent_trials=silent_trials)
    result = rollout(policy, tape, feedback=feedback, yoke=yoke)
    row = summarize(result, tape, policy.agents, window=32)
    row["name"] = name
    row["config"] = asdict(config)
    row["network_seed"] = network_seed
    row["quiet_from"] = quiet_from
    row["silent_trials"] = list(silent_trials)
    row["world_feedback"] = feedback
    print(f"{name} (birth {network_seed}) {result.accuracy:.3f}, "
          f"{np.mean(row['early']):.3f} → {np.mean(row['late']):.3f}, contact {result.contact_rate:.3f}",
          flush=True)
    return row, result


def discover(tape, seeds):
    base = SpikeConfig()
    adjustments = (
        ("initial", {}),
        ("near-zero", {"learning_rate": 1e-5}),
        ("low", {"learning_rate": 0.025}),
        ("residual", {"use_residual": True}),
        ("residual-longtrace", {"use_residual": True, "feature_trace_decay": 0.71}),
        ("longtrace", {"feature_trace_decay": 0.71}),
        ("low-longtrace", {"learning_rate": 0.025, "feature_trace_decay": 0.71}),
        ("lowleak-residual", {"feature_leak": 0.30, "use_residual": True, "feature_trace_decay": 0.71}),
        ("engineered", {"representation": "engineered"}),
        ("engineered-low", {"representation": "engineered", "learning_rate": 0.025}),
        ("engineered-residual", {"representation": "engineered", "learning_rate": 0.025, "use_residual": True}),
        ("additive", {"representation": "additive"}),
    )
    return [
        evaluate(name, replace(base, **changes), tape, seed)[0]
        for seed in seeds
        for name, changes in adjustments
    ]


def controls(tape, seeds, config):
    results = []
    for seed in seeds:
        real, response = evaluate("live", config, tape, seed)
        results.append(real)
        yoke = make_yoke(response, seed=0xB175)
        false, _ = evaluate("yoked", config, tape, seed, feedback="yoked", yoke=yoke)
        false["donor_seeds"] = yoke.donor_seeds
        results.append(false)
        for mode in ("polarity-only", "contact-only", "inverted"):
            results.append(evaluate(mode, replace(config, feedback=mode), tape, seed)[0])
        for label, rate in (("frozen-birth", 0), ("near-zero", 1e-5)):
            results.append(evaluate(label, replace(config, learning_rate=rate), tape, seed)[0])
        cutoff = tape.trials // 2
        results.append(evaluate("frozen-after-learning", config, tape, seed, quiet_from=cutoff)[0])
        swapped = swap_sides(tape, first_trial=cutoff)
        results.append(evaluate("frozen-after-learning-swap", config, swapped, seed, quiet_from=cutoff)[0])
        results.append(evaluate("no-recurrence", replace(config, recurrence_gain=0), tape, seed)[0])
    return results


def staged(tape, review_tape, review_trials, seeds, config):
    results = []
    scenarios = (
        ("learn-old-new-and-revisit", tape, None, ()),
        ("learn-old-new-freeze-at-revisit", tape, 96, ()),
        ("learn-old-freeze-before-new", tape, 48, ()),
        ("live-review-freeze-at-revisit", review_tape, 96, ()),
        ("quiet-review-freeze-at-revisit", review_tape, 96, review_trials),
    )
    for seed in seeds:
        for name, curr, cutoff, silent in scenarios:
            row, result = evaluate(name, config, curr, seed, quiet_from=cutoff, silent_trials=silent)
            axes = {"old_last16": slice(32, 48), "new_first16": slice(48, 64),
                    "new_last16": slice(80, 96), "revisit_first16": slice(96, 112),
                    "revisit_last16": slice(128, 144)}
            for metric, phase in axes.items():
                phase_correct = result.correct[:, phase]
                if metric.startswith("new"):
                    new_trials = curr.cue[0, phase] >= 2
                    phase_correct = phase_correct[:, new_trials]
                row[metric] = phase_correct.mean(axis=1).tolist()
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discover", "controls", "staged"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--trials", type=int, default=192)
    parser.add_argument("--network-seeds", type=int, nargs="+", default=[3, 7])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--geometry-jitter", type=float, default=0.)
    parser.add_argument("--wind", type=float, default=0.)
    args = parser.parse_args()
    seeds = list(range(args.start, args.start + args.n))
    config = SpikeConfig()
    if args.config:
        config = SpikeConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    artifact_parts = {}
    if args.stage == "staged":
        tape = staged_tape(seeds, jitter=args.geometry_jitter, wind=args.wind)
        review_trials = tuple(range(70, 95, 2))
        review_tape = staged_tape(seeds, jitter=args.geometry_jitter, wind=args.wind, review_trials=review_trials)
        artifact_parts["review_trials"] = list(review_trials)
        artifact_parts["review_world_cue"] = review_tape.cue.tolist()
    else:
        tape = make_tape(seeds, args.trials, distance_jitter=args.geometry_jitter,
                         wind_scale=args.wind, map_codes=BALANCED_CODES)
    if args.stage == "discover":
        results = discover(tape, args.network_seeds)
    elif args.stage == "staged":
        results = staged(tape, review_tape, review_trials, args.network_seeds, config)
    else:
        results = controls(tape, args.network_seeds, config)
    artifact = {
        "stage": args.stage,
        "life_seeds": seeds,
        "map_codes": tape.map_codes.tolist(),
        "trials": tape.trials,
        "geometry_jitter": args.geometry_jitter,
        "wind": args.wind,
        "world_context": tape.ctx.tolist(),
        "world_cue": tape.cue.tolist(),
        "world_left_id": tape.left_id.tolist(),
        "results": results,
        **artifact_parts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
