"""One-real-error intervention with a silent same- or changed-key revisit."""

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


class OneReceipt:
    def __init__(self, policy, teach, sign, *, resume=False):
        self.policy = policy
        self.teach = teach
        self.sign = sign
        self.resume = resume

    def reset(self, n):
        if not self.resume:
            self.policy.reset(n)
        self.trial = -1

    def act(self, obs):
        self.trial += int(obs[0, NEW] > 0)
        frame = obs.copy()
        if self.trial != self.teach or self.sign == "mute":
            frame[:, POSITIVE:NEGATIVE + 1] = 0
        elif self.sign == "invert":
            frame[:, POSITIVE:NEGATIVE + 1] = frame[:, POSITIVE:NEGATIVE + 1][:, ::-1]
        return self.policy.act(frame)


def revisit_tape(base, relation):
    ctx, cue = base.ctx.copy(), base.cue.copy()
    ctx[:, -1] = ctx[:, -2] ^ (relation == "other-context")
    cue[:, -1] = (cue[:, -2] + (relation == "other-cue")) % 4
    goal = base.preferences[np.arange(base.n)[:, None], ctx, cue]
    return replace(base, ctx=ctx, cue=cue, goal=goal)


def slice_trials(base, cut):
    fields = {key: getattr(base, key)[:, cut] for key in ("ctx", "cue", "positions", "layout", "goal")}
    return replace(base, **fields)


def run(base, cfg, birth_seeds, *, factory=new_policy):
    teach = base.trials - 2
    rows = []
    reference = {}
    prefix = {}
    if teach:
        for seed in birth_seeds:
            seed_actor = factory(base, cfg, seed)
            history = slice_trials(base, slice(None, teach))
            rollout(seed_actor, history, feedback="none")
            prefix[seed] = seed_actor
    for relation in ("same", "other-context", "other-cue"):
        tape = slice_trials(revisit_tape(base, relation), slice(teach, None))
        for seed in birth_seeds:
            for name, spec, sign in (
                ("negative", cfg, "true"),
                ("muted", cfg, "mute"),
                ("inverted", cfg, "invert"),
                ("negative-no-filter", replace(cfg, cov_rate=0), "true"),
                ("positive-only", replace(cfg, outcomes="positive"), "true"),
            ):
                actor = copy.deepcopy(prefix[seed]) if teach else factory(tape, spec, seed)
                if teach:
                    actor.spec = spec
                    if not spec.cov_rate:
                        actor.lateral = None
                scope = OneReceipt(actor, 0, sign, resume=bool(teach))
                result = rollout(scope, tape)
                contact = result.first[:, 0]
                wrong = (contact >= 0) & ~result.first_correct[:, 0]
                if seed not in reference:
                    reference[seed] = (contact.copy(), result.first_tick[:, 0].copy())
                np.testing.assert_array_equal(contact, reference[seed][0])
                np.testing.assert_array_equal(result.first_tick[:, 0], reference[seed][1])
                last = result.first[:, -1]
                trained_cohorts = (actor.weights != 0).any(axis=(2, 3))
                nontasted = trained_cohorts & (np.arange(4)[None, :] != contact[:, None])
                if np.any(nontasted):
                    raise AssertionError("reward changed an untouched fruit cohort")
                repeat = (last == contact) & (last >= 0)
                context_target = (contact == tape.goal[:, -1]) & wrong
                rows.append({"birth_seed": seed, "relation": relation, "name": name,
                             "wrong": wrong.tolist(), "first_sample": contact.tolist(), "revisit": last.tolist(),
                             "repeated_sample": repeat.tolist(), "rejected_is_probe_target": context_target.tolist(),
                             "probe_correct": result.first_correct[:, -1].tolist(),
                             "write_counts": actor.writes.tolist(), "untouched_groups_written": int(nontasted.sum())})
                print(f"{seed} {relation:14s} {name:20s} repeat {repeat[wrong].mean():.3f}; "
                      f"probe {result.first_correct[wrong, -1].mean():.3f} over {wrong.sum()} real negative events", flush=True)
    summary = {}
    for relation in ("same", "other-context", "other-cue"):
        group = {}
        subset = [row for row in rows if row["relation"] == relation]
        for name in ("negative", "muted", "inverted", "negative-no-filter", "positive-only"):
            arm = [row for row in subset if row["name"] == name]
            mask = np.array([row["wrong"] for row in arm], dtype=bool)
            same = np.array([row["repeated_sample"] for row in arm], dtype=np.float64)
            other = np.array([row["probe_correct"] for row in arm], dtype=np.float64)
            target = np.array([row["rejected_is_probe_target"] for row in arm], dtype=bool)
            group[name] = {"mask": mask, "repeat": same, "probe_correct": other, "wrong_now_good": target}

        def pool(entries, metric, selector):
            valid = entries[selector]
            used = valid.any(axis=0)
            per_world = (entries[metric] * valid).sum(axis=0)[used] / valid.sum(axis=0)[used]
            return {"event_count": int(valid.sum()), "world_count": int(used.sum()),
                    **(estimates(per_world) if used.any() else {"mean": None, "ci95": None})}

        paired = {}
        base = group["negative"]
        for name, value in group.items():
            combined = {k: base[k] - value[k] if k in {"repeat", "probe_correct"} else base[k] for k in base}
            paired[name] = {"repeat": pool(combined, "repeat", "mask"),
                            "correct": pool(combined, "probe_correct", "mask"),
                            "correct_when_previous_wrong_is_target": pool(combined, "probe_correct", "wrong_now_good")}
        summary[relation] = {"arms": {name: {"repeat": pool(g, "repeat", "mask"),
                                             "correct": pool(g, "probe_correct", "mask"),
                                             "correct_when_previous_wrong_is_target": pool(g, "probe_correct", "wrong_now_good")}
                                       for name, g in group.items()}, "negative_minus_comparator": paired}
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--preexposure", type=int, default=32)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--panel", choices=("development", "evaluation"), default="evaluation")
    args = parser.parse_args()
    cfg = ForagingSpec(outcomes="negative", scoring="linear")
    if args.frozen is not None:
        cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    tape = world_panel(args.start, args.n, args.preexposure + 2, panel=args.panel)
    detail, summary = run(tape, cfg, args.births)
    args.output.write_text(json.dumps({"world_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(),
                                       "preexposure": args.preexposure, "panel": args.panel,
                                       "rows": detail, "summary": summary}) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
