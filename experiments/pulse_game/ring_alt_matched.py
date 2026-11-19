"""Match real bite trajectories and sign counts to compare feedback information."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_policy import ForagingSpec
from .ring_alt_run import new_policy, preference_code, world_panel
from .ring_world import NEGATIVE, NEW, POSITIVE, rollout
from .spike_analysis import estimates


def fit_schedule(tape, donor, training, mode):
    positive = donor.first_correct[:, :training]
    negative = (donor.first[:, :training] >= 0) & ~positive
    if mode == "negative-unthinned":
        return negative
    if mode == "positive-only":
        return positive
    include = np.zeros_like(positive)
    for row, seed in enumerate(tape.seeds):
        rng = np.random.default_rng(np.random.SeedSequence((seed, 74178)))
        for ctx in range(2):
            for cue in range(4):
                scope = (tape.ctx[row, :training] == ctx) & (tape.cue[row, :training] == cue)
                pos = np.flatnonzero(scope & positive[row])
                neg = np.flatnonzero(scope & negative[row])
                count = min(len(pos), len(neg))
                choices = neg if mode == "negative-matched" else pos
                include[row, rng.choice(choices, size=count, replace=False)] = True
    return include


class MatchedBites:
    """Hold the sensorimotor policy fixed while varying authentic exposed signs."""

    def __init__(self, policy, training, receipts):
        self.policy = policy
        self.training = training
        self.receipts = receipts

    def reset(self, n):
        self.actual = self.policy.spec
        self.policy.spec = replace(self.actual, scoring="blind")
        self.policy.reset(n)
        self.trial = -1

    def act(self, obs):
        if obs[0, NEW]:
            self.trial += 1
            if self.trial == self.training:
                self.policy.spec = self.actual
        frame = obs.copy()
        if self.trial < self.training:
            frame[~self.receipts[:, self.trial], POSITIVE:NEGATIVE + 1] = 0
        else:
            frame[:, POSITIVE:NEGATIVE + 1] = 0
        return self.policy.act(frame)


def experiment(tape, cfg, seeds, training):
    rows = []
    matched_signs = None
    for seed in seeds:
        donor_spec = replace(cfg, outcomes="frozen", scoring="blind")
        donor = rollout(new_policy(tape, donor_spec, seed), tape)
        modes = ("negative-unthinned", "negative-matched", "positive-matched", "positive-only")
        schedules = {mode: fit_schedule(tape, donor, training, mode) for mode in modes}
        scheduled = schedules["negative-matched"].sum(axis=1)
        np.testing.assert_array_equal(scheduled, schedules["positive-matched"].sum(axis=1))
        if matched_signs is None:
            matched_signs = scheduled.tolist()
        for mode in modes:
            spec = replace(cfg, outcomes="negative" if mode.startswith("negative") else "positive")
            actor = new_policy(tape, spec, seed)
            policy = MatchedBites(actor, training, schedules[mode])
            result = rollout(policy, tape)
            for x, y in ((result.first, donor.first), (result.first_tick, donor.first_tick), (result.motion, donor.motion)):
                np.testing.assert_array_equal(x[:, :training], y[:, :training])
            expected = schedules[mode].sum(axis=1)
            np.testing.assert_array_equal(actor.writes.sum(axis=1), expected)
            late = result.first_correct[:, training:].mean(axis=1)
            rows.append({"name": mode, "birth_seed": seed, "quiet_first": late.tolist(),
                         "training_writes": expected.tolist(), "first": result.first.tolist(),
                         "net_stats": actor.stats()})
            print(f"{seed:2d} {mode:24s} exposed signs {expected.mean():.1f}; "
                  f"quiet first bite {late.mean():.3f}", flush=True)
    avg = {name: np.mean([row["quiet_first"] for row in rows if row["name"] == name], axis=0) for name in modes}
    summary = {name: estimates(values) for name, values in avg.items()}
    summary["negative_matched_minus_positive_matched"] = estimates(avg["negative-matched"] - avg["positive-matched"])
    summary["negative_unthinned_minus_positive_only"] = estimates(avg["negative-unthinned"] - avg["positive-only"])
    return rows, summary, matched_signs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--training", type=int, default=224)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    args = parser.parse_args()
    cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    tape = world_panel(args.start, args.n, args.training + 96, panel="evaluation")
    records, summary, counts = experiment(tape, cfg, args.births, args.training)
    out = {"world_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(),
           "training_trials": args.training, "sign_count_matched_per_world": counts, "summary": summary, "rows": records}
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
