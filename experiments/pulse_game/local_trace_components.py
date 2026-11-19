"""Replay native physical write components across entire matched support histories."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from . import local_trace_eval as main_eval, local_trace_graph as native, local_trace_train as training


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/local-trace-20261119"
OUT = ROOT / "docs/evidence/local-trace-2026-11-19/components.json"
MODES = ("active", "sham", "weights_only", "bias_only", "weights_and_bias", "topology_only")
DELAYS = {"odor": 4, "feeder": 0, "feeder_novel": 0}
METRICS = ("correct", "expected_correct", "law_negative", "law_positive", "weaker_law", "both_twins_correct")
DOSES = ("weight_changes", "bias_changes", "births", "retirements", "weight_l1", "bias_l1")
TAPE_FIELDS = ("left", "right", "right_value", "constant_value", "draw", "edit_draw", "law", "swap")


def receipt_dose(receipt):
    weights = receipt.weight_delta.flatten(1)
    bias = receipt.bias_delta
    return torch.stack((weights.count_nonzero(dim=1), bias.count_nonzero(dim=1),
                        receipt.birth >= 0, receipt.retired >= 0, weights.abs().sum(1), bias.abs().sum(1)), dim=1)


def audit(mode, before, after, dose):
    enabled = {
        "active": (True, True, True), "sham": (False, False, False),
        "weights_only": (True, False, False), "bias_only": (False, True, False),
        "weights_and_bias": (True, True, False), "topology_only": (False, False, True),
    }
    weight, bias, support = enabled[mode]
    if not weight and torch.any(dose[:, 0]):
        raise AssertionError("Unexpected tuning on surviving weight addresses")
    if not bias and (torch.any(dose[:, 1]) or not torch.equal(before.bias, after.bias)):
        raise AssertionError("Unexpected bias update")
    if not support and (torch.any(dose[:, 2:4]) or not torch.equal(before.present, after.present)):
        raise AssertionError("Unexpected structural edit")
    if not weight:
        old_and_new = before.present & after.present
        if not torch.equal(before.weights[old_and_new], after.weights[old_and_new]):
            raise AssertionError("Surviving weights changed without tuning")
    if not torch.equal(after.present, after.weights != 0):
        raise AssertionError("Support disagrees with nonzero physical weights")


@torch.no_grad()
def evaluate(graph, rule, tape, delay):
    """Run an intervention at every support commit and score a natural final choice."""
    sides, logits, doses, support_side, support_feedback = [], [], [], [], []
    for mode in MODES:
        state = native.start(graph, tape.lanes)
        dose = torch.zeros(tape.lanes, len(DOSES), dtype=graph.bias.dtype)
        chosen, earned = [], []
        for t in range(tape.contacts):
            left, right = tape.left[:, t], tape.right[:, t]
            action = native.act(rule, state, left, right, tape.draw[:, t])
            taste = main_eval.outcome(tape, action, t)
            experience = native.observe(rule, action, left, right, taste)
            commit = native.commit(rule, experience, tape.edit_draw[:, t], intervention=mode)
            actual_edits = receipt_dose(commit.receipt)
            audit(mode, state.graph, commit.state.graph, actual_edits)
            dose += actual_edits
            state = commit.state
            chosen.append(action.side.numpy())
            earned.append(taste.numpy())
        waited = native.wait(rule, state, delay)
        final = native.act(rule, waited, tape.left[:, -1], tape.right[:, -1], tape.draw[:, -1])
        sides.append(final.side.numpy())
        logits.append(final.motor_logit.numpy())
        doses.append(dose.mean(0).numpy())
        support_side.append(chosen)
        support_feedback.append(earned)
    return {
        "side": np.array(sides)[:, None], "logit": np.array(logits)[:, None],
        "contact_side": np.array(support_side), "actual_feedback": np.array(support_feedback),
        "right_value": tape.right_value.numpy(), "law": tape.law.numpy(),
        "swap": tape.swap.numpy(), "twin": tape.twin.numpy(), "dose": np.array(doses),
    }


def verify_reference(path, tape, raw, delay):
    """Confirm tape and baseline actions match the archived main evaluation exactly."""
    with np.load(path, allow_pickle=False) as saved:
        tick = saved["delays"].tolist().index(delay)
        for name in TAPE_FIELDS:
            np.testing.assert_array_equal(getattr(tape, name).numpy(), saved[name])
        for new_mode, old_mode in (("active", "intact"), ("sham", "sham")):
            src, dst = main_eval.BRANCHES.index(old_mode), MODES.index(new_mode)
            for key in ("side", "logit"):
                np.testing.assert_array_equal(raw[key][dst, 0], saved[key][src, tick])
            for key in ("contact_side", "actual_feedback"):
                np.testing.assert_array_equal(raw[key][dst], saved[key][src])


def summarize(rows, task, intervals):
    metrics, dose = {}, {}
    for i, mode in enumerate(MODES):
        by_root = [main_eval.branch_metrics(row, task, i, 0) for row in rows]
        metrics[mode] = {key: np.array([item[key].mean() for item in by_root])
                         for key in METRICS if key != "weaker_law"}
        metrics[mode]["weaker_law"] = np.minimum(metrics[mode]["law_negative"], metrics[mode]["law_positive"])
        dose[mode] = {str(birth): dict(zip(DOSES, row["dose"][i].astype(float).tolist(), strict=True))
                      for birth, row in zip(intervals.births, rows, strict=True)}

    def report(records):
        return {name: {key: intervals.report(values) for key, values in record.items()}
                for name, record in records.items()}

    def contrast(a, b):
        return {key: metrics[a][key] - metrics[b][key] for key in METRICS}

    def interaction(a, b, c):
        return {key: metrics[a][key] - metrics[b][key] - metrics[c][key] + metrics["sham"][key]
                for key in METRICS}

    return {
        "rates": report(metrics), "dose_per_life_by_birth": dose,
        "minus_sham": report({mode: contrast(mode, "sham") for mode in MODES if mode != "sham"}),
        "active_minus": report({mode: contrast("active", mode) for mode in MODES if mode not in ("active", "sham")}),
        "interaction": report({"weight_and_bias": interaction("weights_and_bias", "weights_only", "bias_only"),
                               "numeric_and_topology": interaction("active", "weights_and_bias", "topology_only")}),
    }


def run(run_path, reference, output):
    torch.set_num_threads(1)
    run_path, reference, output = map(Path, (run_path, reference, output))
    manifest = main_eval.load_manifest(run_path)
    ref_plan = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    step = manifest["config"]["updates"]
    births = manifest["births"]
    if step not in ref_plan["stages"] or births != ref_plan["births"] or ref_plan["worlds_per_birth"] != 192:
        raise ValueError("Expected the matched 192-world final main evaluation")
    seed, draws = ref_plan["seed"], ref_plan["bootstrap_draws"]
    intervals = main_eval.BirthIntervals(births, draws, seed)
    tapes = {task: main_eval.make_tape(task, seed, ref_plan["worlds_per_birth"]) for task in DELAYS}
    report = {
        "schema": 1, "checkpoint_update": step, "births": births, "seed": seed,
        "worlds_per_birth": ref_plan["worlds_per_birth"], "delays": DELAYS, "modes": MODES,
        "bootstrap_draws": draws, "interval_unit": "paired independent birth",
        "intervention_scope": "Every support commit during the same lifetime; no updates on the scored return.",
        "tape_matching": "All arms share menus, laws, layout, and draws. Delivered taste always follows the actual contact.",
        "inference_scope": "Comparisons within a frozen circuit. Components may interact and experience histories can diverge.",
        "weaker_law": "Minimum of the two laws within each birth before aggregation.",
        "reference_sha256": main_eval.digest(reference / "summary.json"),
        "source_sha256": {Path(path).name: main_eval.digest(Path(path)) for path in (
            __file__, native.__file__, native.native.__file__, training.__file__, main_eval.__file__,
        )},
        "checkpoint_sha256": {}, "exact_main_baseline_replays": 0, "results": {},
    }
    for trained_on in ("odor", "feeder"):
        graphs = []
        for birth in births:
            graph, rule, name, digest = main_eval.verified_checkpoint(training, run_path, manifest, trained_on,
                                                                     "active", birth, step)
            if ref_plan["checkpoint_sha256"]["local"].get(name) != digest:
                raise ValueError(f"Main evaluation used a different checkpoint at {name}")
            report["checkpoint_sha256"][name] = digest
            graphs.append((graph, rule))
        report["results"][trained_on] = {}
        for task, tape in tapes.items():
            rows = []
            for birth, (graph, rule) in zip(births, graphs, strict=True):
                raw = evaluate(graph, rule, tape, DELAYS[task])
                if trained_on == ("feeder" if task == "feeder_novel" else task):
                    verify_reference(reference / f"{task}-active-{birth}-{step}.npz", tape, raw, DELAYS[task])
                    report["exact_main_baseline_replays"] += 1
                rows.append(raw)
            report["results"][trained_on][task] = summarize(rows, task, intervals)
            print(f"components {trained_on} training / {task} holdout complete", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf8")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DATA / "local")
    parser.add_argument("--reference", type=Path, default=DATA / "eval-8000")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.run, args.reference, args.out)


if __name__ == "__main__":
    main()
