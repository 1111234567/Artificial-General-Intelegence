"""Probe frozen feeder-mark learners on the original one-feedback odor revisit tape."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import revisit_choice_eval as ev
from . import self_param_writer as fw


DELAYS = (4, 16)
ARMS = ("fixed", "variable")
BRANCHES = ("natural", "receipt_matched_disabled", "disabled", "skip")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rates(raw, odor, delay, pairs):
    results = {}
    for feedback in ("true", "flipped"):
        signs = [ev.score_world(raw, odor, delay, feedback, sign) for sign in (1, -1)]
        if any(result["natural_first"].sum() != 2 * pairs for result in signs):
            raise ValueError("Expected one natural first side per odor and swap")
        for branch in BRANCHES:
            for metric in ("actual", "p_correct"):
                condition = f"{feedback}.{branch}.{metric}"
                positive, negative = [result["branches"][branch][metric][result["natural_first"]].mean()
                                      for result in signs]
                results[condition] = float((positive + negative) / 2)
                if feedback == "true" and metric == "actual" and odor == 0:
                    results[f"true.{branch}.positive_repeat"] = float(positive)
                    results[f"true.{branch}.negative_switch"] = float(negative)
    for metric in ("actual", "p_correct"):
        for branch in BRANCHES:
            results[f"true_minus_flipped.{branch}.{metric}"] = (
                results[f"true.{branch}.{metric}"] - results[f"flipped.{branch}.{metric}"])
        for feedback in ("true", "flipped", "true_minus_flipped"):
            for control in ("receipt_matched_disabled", "disabled", "skip"):
                results[f"{feedback}.natural_minus_{control}.{metric}"] = (
                    results[f"{feedback}.natural.{metric}"] - results[f"{feedback}.{control}.{metric}"])
    return results


def summarize(raw_by_birth, delay_indices, pairs, bootstrap):
    result = {}
    for odor_index, odor in enumerate(ev.ODORS):
        result[odor] = {}
        for delay, delay_index in zip(DELAYS, delay_indices, strict=True):
            items = [rates(raw, odor_index, delay_index, pairs) for raw in raw_by_birth]
            result[odor][str(delay)] = {name: bootstrap([item[name] for item in items]) for name in items[0]}
    return result


def compare(first, second, bootstrap):
    return {odor: {delay: {metric: bootstrap(np.array(stat["birth_values"])
                                            - second[odor][delay][metric]["birth_values"])
                           for metric, stat in metrics.items()} for delay, metrics in depths.items()}
            for odor, depths in first.items()}


def run(root, steps):
    original_path = root / "eval.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    tape_path = root / original["trials_file"]
    births, pairs, seed = original["births"], original["pairs_per_birth"], original["holdout_seed"]
    bootstrap = ev.BirthBootstrap(births, seed, original["probability_intervals"]["draws"])
    keep_connected = json.loads((root / "run.json").read_text(encoding="utf-8"))["config"]["keep_connected"]
    report = {
        "design": "Feed feeder-mark checkpoints the original odor tape with no additional training. Each arm uses its own autonomous first action on matched stimulus and draw. Score actual second actions on each latent truth before second feedback. The main task revisits the same previously unseen odor. The novel condition presents unrelated odors on the second visit and scores the original reference slots.",
        "births": births, "pairs_per_birth": pairs, "holdout_seed": seed, "delays": DELAYS,
        "native_trials_per_birth_per_feedback_delay": 4 * pairs,
        "probability_intervals": original["probability_intervals"],
        "original_sources": {p.name: digest(p) for p in (original_path, tape_path, root / "run.json")},
        "source_sha256": {p.name: digest(p) for p in map(Path, (ev.__file__, ev.native.__file__, fw.__file__,
                                                               ev.receipt_graph.__file__, __file__))},
        "checkpoint_sha256": {}, "references": {}, "stages": {}, "contrasts": {},
        "missing_checkpoints": {},
    }
    with np.load(tape_path) as saved:
        if (saved["birth_names"].tolist() != births or saved["odor_names"].tolist() != list(ev.ODORS)
                or saved["branch_names"].tolist() != list(ev.BRANCHES)
                or saved["observed_tastes"].tolist() != list(ev.OBSERVED_TASTES)):
            raise ValueError("Saved original tape axes differ from evaluator axes")
        tape_arrays = {name: saved[name] for name in ev.Heldout.__dataclass_fields__}
        tapes = [ev.Heldout(**{name: torch.from_numpy(array[i]) for name, array in tape_arrays.items()})
                 for i in range(len(births))]
        birth_tapes = [dict((name, value[i]) for name, value in tape_arrays.items()) for i in range(len(births))]
        keys = ("motor_side", "motor_logit", "receipt_status", "gate_index", "gate_logits",
                "benefit", "first_actual_side", "first_natural_side")
        replay = {name: saved[name] for name in keys}
        ranked = saved["arm_names"].tolist().index("ranked")
        delay_indices = [saved["delays"].tolist().index(delay) for delay in DELAYS]
        for label, step in (("born", 0), ("ranked_original", max(original["steps"]))):
            slot = saved["steps"].tolist().index(step)
            raw = [tape | {name: array[ranked, slot, i] for name, array in replay.items()}
                   for i, tape in enumerate(birth_tapes)]
            scored = summarize(raw, delay_indices, pairs, bootstrap)
            for delay in DELAYS:
                measured = scored["same"][str(delay)]["true.natural.actual"]["birth_values"]
                archival = original["groups"]["ranked"][str(step)]["same"][str(delay)]["behavior"]["true"]
                if not np.allclose(measured, archival["natural_first"]["natural"]["actual"]["birth_values"],
                                   atol=1e-15, rtol=0):
                    raise ValueError("Original baseline actions no longer reproduce original summary")
            report["references"][label] = scored

    stages = report["stages"]
    for step in steps:
        folder = root / "value" if step == 4000 else root / "value" / "longitudinal"
        paths = {arm: [folder / arm / f"{birth}-{step}.npz" for birth in births] for arm in ARMS}
        missing = [str(path.relative_to(root)) for checks in paths.values() for path in checks if not path.is_file()]
        if missing:
            report["missing_checkpoints"][str(step)] = missing
            continue
        stage = stages[str(step)] = {}
        for arm, checks in paths.items():
            raw = []
            for path, tape in zip(checks, tapes, strict=True):
                with np.load(path) as checkpoint:
                    graph = fw.Graph(*(torch.from_numpy(checkpoint[key]) for key in ("w", "b", "m")))
                report["checkpoint_sha256"][str(path.relative_to(root))] = digest(path)
                raw.append(ev.evaluate(graph, tape, DELAYS, keep_connected=keep_connected))
            stage[arm] = summarize(raw, range(len(DELAYS)), pairs, bootstrap)
        report["contrasts"][str(step)] = {
            "variable_minus_fixed": compare(stage["variable"], stage["fixed"], bootstrap),
            **{f"{arm}_minus_born": compare(stage[arm], report["references"]["born"], bootstrap) for arm in ARMS},
        }
    if "4000" in stages and "12000" in stages:
        report["contrasts"]["12000_minus_4000"] = {
            arm: compare(stages["12000"][arm], stages["4000"][arm], bootstrap) for arm in ARMS}
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/revisit-choice-20261119"))
    parser.add_argument("--steps", nargs="+", type=int, default=[4000, 12000], choices=(4000, 12000))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.root, args.steps)
    destination = args.root / "value" / "reverse-transfer.json"
    destination.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
