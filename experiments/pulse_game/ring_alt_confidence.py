"""Suppress weak negative generalizations before sensor-gated bite competition."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from .ring_alt_policy import ForagingSpec, ReceptorForager
from .ring_alt_probe import run as probe_run
from .ring_alt_run import preference_code, summarize_result, world_panel
from .ring_world import rollout
from .spike_analysis import estimates


class ConfidenceForager(ReceptorForager):
    def __init__(self, spec, *, birth_seed, action_seeds, margin):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds)
        self.margin = margin

    def _valence(self, current):
        signal = super()._valence(current)
        return np.sign(signal) * np.maximum(np.abs(signal) - self.margin, 0.)


def constructor(tape, spec, seed, margin):
    action = [np.random.SeedSequence((s, 0x414C5452)) for s in tape.seeds]
    return ConfidenceForager(spec, birth_seed=seed, action_seeds=action, margin=margin)


def selective_probe(tape, cfg, births, margin):
    def make(tape, spec, birth):
        return constructor(tape, spec, birth, margin)

    with contextlib.redirect_stdout(io.StringIO()):
        rows, report = probe_run(tape, cfg, births, factory=make)
    same = report["same"]["negative_minus_comparator"]["muted"]["repeat"]
    fear = report["other-context"]["negative_minus_comparator"]["muted"]["correct_when_previous_wrong_is_target"]
    print(f"margin {margin:.2f}: same-target repetition difference {same['mean']:+.3f}; "
          f"opposite-context correct after same fruit was rejected {fear['mean']:+.3f}", flush=True)
    return {"margin": margin, "summary": report, "rows": rows}


def paired_margins(zero, gated):
    comparisons = (
        ("same", "repeated_sample", "wrong", "same_error_tradeoff"),
        ("other-context", "probe_correct", "rejected_is_probe_target", "other_context_correct_rescue"),
    )
    output = {}
    for relation, measure, selector, name in comparisons:
        raw, thresholded = ([row for row in set_ if row["name"] == "negative" and row["relation"] == relation]
                             for set_ in (zero["rows"], gated["rows"]))
        source = np.asarray([row[selector] for row in raw], dtype=bool)
        paired = np.asarray([row[selector] for row in thresholded], dtype=bool)
        np.testing.assert_array_equal(source, paired)
        old = np.asarray([row[measure] for row in raw], dtype=np.float64)
        new = np.asarray([row[measure] for row in thresholded], dtype=np.float64)
        valid = source.any(axis=0)
        diffs = ((new - old) * source).sum(axis=0)[valid] / source.sum(axis=0)[valid]
        output[name] = {"world_count": int(valid.sum()), **estimates(diffs)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--exp", choices=("find-margin", "confirm-margin"), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    if args.exp == "find-margin":
        tape = world_panel(209001, 192, 34, panel="development")
        births = (2, 5, 8)
        margins = (0., .08, .16, .24, .32, .40, .52)
        quiet_tape = None
    else:
        criteria = json.loads(args.selection.read_text(encoding="utf-8"))
        births = tuple(criteria["births"])
        tape = world_panel(criteria["start"], criteria["worlds"], 34, panel="evaluation")
        margins = (0., criteria["margin"])
        quiet_tape = world_panel(criteria["acquisition_start"], criteria["acquisition_worlds"], 224,
                                panel="evaluation")
    records = [selective_probe(tape, cfg, births, margin) for margin in margins]
    acquisition = {}
    if quiet_tape is not None:
        arrays = []
        for margin in margins:
            scores = []
            for birth in births:
                policy = constructor(quiet_tape, cfg, birth, margin)
                result = rollout(policy, quiet_tape)
                score = summarize_result(quiet_tape, result)
                scores.append(score)
                print(f"{birth} margin {margin:.2f}: late {np.mean(score['late_first']):.3f}", flush=True)
            arrays.append(np.mean([x["late_first"] for x in scores], axis=0))
        acquisition = {"world_seeds": quiet_tape.seeds,
                       "late_first": {str(m): estimates(v) for m, v in zip(margins, arrays, strict=True)},
                       "change": estimates(arrays[-1] - arrays[0])}
    out = {"experiment": args.exp, "births": births, "world_seeds": tape.seeds,
           "map_codes": preference_code(tape.preferences).tolist(), "records": records, "learning": acquisition}
    if quiet_tape is not None:
        out["margin_minus_raw"] = paired_margins(records[0], records[-1])
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
