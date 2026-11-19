"""Continuous ring lives with separate quiet probes across cue-block changes."""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_policy import ForagingSpec
from .ring_alt_run import new_policy, preference_code, world_panel
from .ring_world import NEGATIVE, NEW, POSITIVE, rollout
from .spike_analysis import estimates


BLOCK = 128


def block_tape(tape, blocks, *, reverse_after=None):
    cue = tape.cue.copy()
    for block in range(blocks):
        region = cue[:, BLOCK * block:BLOCK * (block + 1)]
        region[:] = region % 2 + 2 * (block % 2)
    goal = tape.preferences[np.arange(tape.n)[:, None], tape.ctx, cue]
    if reverse_after is not None:
        flipped = cue[:, reverse_after:] ^ (cue[:, reverse_after:] < 2)
        goal[:, reverse_after:] = tape.preferences[np.arange(tape.n)[:, None], tape.ctx[:, reverse_after:], flipped]
    return replace(tape, cue=cue, goal=goal)


def probes(base, preferences):
    good = preferences[np.arange(base.n)[:, None], base.ctx, base.cue]
    return replace(base, preferences=preferences, goal=good)


class Resume:
    def __init__(self, cloned_actor):
        self.actor = cloned_actor

    def reset(self, n):
        if len(self.actor.v) != n:
            raise ValueError("probe requires the same number of neural lives")

    def act(self, obs):
        return self.actor.act(obs)


class InterruptedAssay:
    """Clone a physical life at a boundary. Test clones never reenter training."""

    def __init__(self, actor, probe, checks, last_feedback, switched=None, switch_at=None):
        self.actor = actor
        self.probe = probe
        self.checks = checks
        self.last_feedback = last_feedback
        self.switched = switched
        self.switch_at = switch_at

    def reset(self, n):
        self.actor.reset(n)
        self.trial = -1
        self.snapshots = []

    def act(self, obs):
        if obs[0, NEW]:
            self.trial += 1
            if self.trial in self.checks:
                tape = self.switched if self.switch_at is not None and self.trial > self.switch_at else self.probe
                policy = Resume(copy.deepcopy(self.actor))
                read = rollout(policy, tape, feedback="none")
                old = tape.cue < 2
                self.snapshots.append({"trained_trials": self.trial,
                                       "old": (np.sum(read.first_correct & old, axis=1) / old.sum(axis=1)).tolist(),
                                       "new": (np.sum(read.first_correct & ~old, axis=1) / (~old).sum(axis=1)).tolist(),
                                       "late_contact": np.mean(read.bites > 0, axis=1).tolist()})
        if self.trial >= self.last_feedback:
            obs = obs.copy()
            obs[:, POSITIVE:NEGATIVE + 1] = 0
        return self.actor.act(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    blocks = 4
    checks = tuple(BLOCK * step for step in range(1, blocks + 1))
    end = checks[-1]
    change = 2 * BLOCK if args.reverse else None
    raw = world_panel(args.start, args.n, end + 1, panel="evaluation")
    tape = block_tape(raw, blocks, reverse_after=change)
    fresh = world_panel(args.start + 2500, args.n, 96, panel="evaluation")
    test = probes(fresh, tape.preferences)
    switched_prefs = tape.preferences.copy()
    switched_prefs[:, :, :2] = switched_prefs[:, :, :2][:, :, ::-1]
    shifted = probes(fresh, switched_prefs)
    conditions = (("negative", cfg, end), ("negative-no-filter", replace(cfg, cov_rate=0.), end),
                  ("negative-stop-before-new", cfg, BLOCK), ("signed", replace(cfg, outcomes="signed"), end))
    rows = []
    for birth in args.births:
        for name, rule, stop in conditions:
            policy = new_policy(tape, rule, birth)
            assay = InterruptedAssay(policy, test, checks, stop, switched=shifted, switch_at=change)
            result = rollout(assay, tape)
            rows.append({"name": name, "birth_seed": birth, "checkpoints": assay.snapshots,
                         "first_correct": result.first_correct.tolist(), "network": policy.stats()})
            for snap in assay.snapshots:
                print(f"{birth:2d} {name:25s} {snap['trained_trials']} trials "
                      f"old/new {np.mean(snap['old']):.3f}/{np.mean(snap['new']):.3f}", flush=True)
    summary = {}
    for boundary in checks:
        metrics = ("old", "new", "late_contact")
        values = {name: {key: np.mean([check[key] for row in rows for check in row["checkpoints"]
                                     if row["name"] == name and check["trained_trials"] == boundary], axis=0)
                         for key in metrics} for name, _, _ in conditions}
        summary[boundary] = {name: {key: estimates(score) for key, score in group.items()} for name, group in values.items()}
        summary[boundary]["negative_minus_no_filter"] = {key: estimates(values["negative"][key] - values["negative-no-filter"][key])
                                                         for key in metrics}
    args.artifact.write_text(json.dumps({"training_seeds": tape.seeds, "probe_seeds": fresh.seeds,
                                         "map_codes": preference_code(tape.preferences).tolist(),
                                         "reversal_trial_0_based": change, "summary": summary, "rows": rows},
                                        separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
