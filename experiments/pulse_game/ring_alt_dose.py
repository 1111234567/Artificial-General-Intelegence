"""Causal dose response for excluding several distinct fruits with only negatives."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_matched import MatchedBites
from .ring_alt_policy import ForagingSpec
from .ring_alt_run import new_policy, preference_code, world_panel
from .ring_world import rollout
from .spike_analysis import estimates


SAMPLES = 12


def controlled_world(base, cfg, birth):
    ctx, cue = base.ctx.copy(), base.cue.copy()
    ctx[:] = ctx[:, :1]
    cue[:] = cue[:, :1]
    goal = base.preferences[np.arange(base.n)[:, None], ctx, cue]
    tape = replace(base, ctx=ctx, cue=cue, goal=goal)
    blind = replace(cfg, scoring="blind", outcomes="frozen")
    initial = rollout(new_policy(tape, blind, birth), tape, feedback="none")
    assert (initial.bites == 1).all()
    # The public physical order for teaching samples is picked independently of
    # hidden preferences. Reposition fruits at the station already reached by
    # the blind oscillator, then leave the probe layout untouched.
    plan = np.tile(base.layout[:, 0], (1, 3))
    layout = tape.layout.copy()
    for column in range(SAMPLES):
        positions = layout[:, column]
        reached = (positions == initial.first[:, column, None]).argmax(axis=1)
        required = (positions == plan[:, column, None]).argmax(axis=1)
        rows = np.arange(tape.n)
        positions[rows, reached], positions[rows, required] = positions[rows, required].copy(), positions[rows, reached].copy()
    tape = replace(tape, layout=layout)
    donor = rollout(new_policy(tape, blind, birth), tape, feedback="none")
    np.testing.assert_array_equal(donor.first[:, :SAMPLES], plan)
    np.testing.assert_array_equal(donor.first_tick, initial.first_tick)
    return tape, donor


def schedule(tape, donor, distinct, repeat=False):
    receipts = np.zeros_like(donor.first[:, :SAMPLES], dtype=bool)
    plan = donor.first[:, :4]
    wrong = plan[~donor.first_correct[:, :4]].reshape(tape.n, 3)
    which = np.arange(tape.n)
    excluded = np.zeros((tape.n, 4), dtype=bool)
    for j in range(distinct):
        item = wrong[:, 0 if repeat else j]
        slot = (plan == item[:, None]).argmax(axis=1)
        receipts[which, 4 * j + slot] = True
        excluded[which, item] = True
    assert not (receipts & donor.first_correct[:, :SAMPLES]).any()
    return receipts, excluded


def run(base, cfg, births):
    names = (("silent", 0, False), ("one-wrong-fruit", 1, False), ("two-distinct-wrong-fruits", 2, False),
             ("three-distinct-wrong-fruits", 3, False), ("same-wrong-fruit-three-times", 3, True))
    out = []
    for seed in births:
        tape, donor = controlled_world(base, cfg, seed)
        for name, dose, repeat in names:
            seen, rejected = schedule(tape, donor, dose, repeat)
            actor = new_policy(tape, cfg, seed)
            frame = MatchedBites(actor, SAMPLES, seen)
            outcome = rollout(frame, tape)
            np.testing.assert_array_equal(outcome.first[:, :SAMPLES], donor.first[:, :SAMPLES])
            np.testing.assert_array_equal(outcome.first_tick[:, :SAMPLES], donor.first_tick[:, :SAMPLES])
            np.testing.assert_array_equal(actor.writes[:, 1], np.full(tape.n, dose))
            assert not actor.writes[:, 0].any()
            chosen = outcome.first[:, -1]
            wrong_repeat = (chosen >= 0) & rejected[np.arange(tape.n), np.maximum(chosen, 0)]
            out.append({"birth": seed, "name": name, "negative_writes": dose,
                        "correct": outcome.first_correct[:, -1].tolist(), "chosen": chosen.tolist(),
                        "picked_rejected": wrong_repeat.tolist(), "rejected_groups": rejected.tolist()})
            print(f"{seed:2d} {name:28s} correct {outcome.first_correct[:, -1].mean():.3f}; "
                  f"re-bite explicitly rejected {wrong_repeat.mean():.3f}", flush=True)
    arms = {name: {key: np.mean([row[key] for row in out if row["name"] == name], axis=0)
                   for key in ("correct", "picked_rejected")} for name, _, _ in names}
    summary = {name: {key: estimates(value) for key, value in arm.items()} for name, arm in arms.items()}
    summary["three_distinct_minus_same_thrice"] = estimates(arms["three-distinct-wrong-fruits"]["correct"]
                                                            - arms["same-wrong-fruit-three-times"]["correct"])
    return out, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--n", type=int, default=384)
    args = parser.parse_args()
    cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    tape = world_panel(219001, args.n, SAMPLES + 1, panel="evaluation")
    rows, summary = run(tape, cfg, (53, 61, 73))
    report = {"world_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(),
              "teaching_trials": SAMPLES, "summary": summary, "rows": rows}
    args.artifact.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
