"""Verify frozen local-trace holdouts and extract the portable evidence summary."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STAGES = (1000, 4000, 8000)
TAPE_KEYS = ("left", "right", "right_value", "constant_value", "draw", "edit_draw",
             "legacy_choice_draw", "law", "swap", "first_mark", "twin")
ODOR_METRICS = ("correct_same", "correct_swap", "positive_repeat", "negative_switch",
                "positive_repeat_same", "positive_repeat_swap", "negative_switch_same", "negative_switch_swap")
BASE_METRICS = ("correct", "both_twins_correct", "law_negative", "law_positive")


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_hash(path, expected):
    result = fingerprint(path)
    if result != expected:
        raise ValueError(f"SHA-256 mismatch at {path}")
    return result


def json_from(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_rates(raw, task, branches, block, birth):
    """Rescore autonomous terminal actions and exact denominators on archival rows."""
    from .local_trace_eval import branch_metrics

    correct_counts = {}
    for d, delay in enumerate(raw["delays"]):
        correct_counts[str(delay)] = {}
        for i, branch in enumerate(branches):
            actual = branch_metrics(raw, task, i, d)
            expected = block[str(delay)]["branches"][branch]
            for metric, values in actual.items():
                item = expected[metric]
                denominator = item["trials_by_birth"][birth]
                check(denominator == len(values), f"{task}/{branch}/{birth}: denominator mismatch")
                check(np.isclose(item["birth_values"][birth], values.mean(), rtol=0, atol=1e-8),
                      f"{task}/{branch}/{birth}: rate mismatch in {metric}")
            if "weaker_law" in expected:
                smaller = min(actual["law_negative"].mean(), actual["law_positive"].mean())
                check(expected["weaker_law"]["birth_values"][birth] == smaller, "Weaker mapping differs")
            if branch == "intact":
                keys = BASE_METRICS + ODOR_METRICS if task == "odor" else BASE_METRICS
                correct_counts[str(delay)] = {metric: [int(actual[metric].sum()), len(actual[metric])] for metric in keys}
    return correct_counts


def verify_stage(root, step, source):
    from .local_trace_eval import audit, audit_legacy

    directory = root / f"eval-{step}"
    file = directory / "summary.json"
    stage = str(step)
    check(source["stages"] == [step], f"Unexpected stages in {file}")
    for filename, expected in source["source_sha256"].items():
        verify_hash(Path(__file__).with_name(filename), expected)
    for controller, field in (("local", "fit_manifest_sha256"), ("legacy", "legacy_fit_manifest_sha256")):
        verify_hash(root / controller / "run.json", source[field])
        for relative, expected in source["checkpoint_sha256"][controller].items():
            verify_hash(root / controller / relative, expected)
    for filename, expected in source["traces"].items():
        verify_hash(directory / filename, expected)

    matched = {}
    outcomes = {}
    for controller, prefix in (("local", ""), ("legacy", "legacy_")):
        table = source[f"{prefix}results"][stage]
        branches = source[f"{prefix}branches"]
        outcomes[controller] = {}
        for task in source["tasks"]:
            outcomes[controller][task] = {}
            for arm in source["arms"]:
                scores = outcomes[controller][task][arm] = {}
                for birth in map(str, source["births"]):
                    label = f"{task}-{arm}-{birth}-{stage}.npz"
                    name = label if controller == "local" else f"legacy-{label}"
                    with np.load(directory / name, allow_pickle=False) as dataset:
                        raw = {key: dataset[key] for key in dataset.files if key != "shuffle_draw_per_world"}
                    tape = sha256()
                    for key in TAPE_KEYS:
                        tape.update(raw[key].tobytes())
                    tape_hash = tape.hexdigest()
                    matched.setdefault(task, tape_hash)
                    check(matched[task] == tape_hash, f"Tape differs at {name}")
                    check(np.array_equal(raw["delays"], source["delays"]), f"Delays differ at {name}")
                    width = 4 if task == "odor" else 8
                    check(len(raw["twin"]) == width * source["worlds_per_birth"], f"Lane count differs at {name}")
                    actual = audit(raw) if controller == "local" else audit_legacy(raw)
                    check(actual == source[f"{prefix}audits"][stage][task][arm][birth], f"Audit differs at {name}")
                    scores[birth] = verify_rates(raw, task, branches, table[task][arm], birth)
    return {"evaluation_summary_sha256": fingerprint(file), "tape_sha256": matched,
            "file_sha256": {"raw": source["traces"], "checkpoints": source["checkpoint_sha256"],
                            "fit_manifests": {"local": source["fit_manifest_sha256"],
                                              "legacy": source["legacy_fit_manifest_sha256"]},
                            "source": source["source_sha256"]},
            "intact_birth_counts": outcomes}


def verify_intervals(source, step):
    from .local_trace_eval import BirthIntervals

    stage = str(step)
    births = list(map(str, source["births"]))
    bootstrap = BirthIntervals(births, source["bootstrap_draws"], source["seed"])

    def verifies(item, values):
        matched = bootstrap.report(values)
        # Expected-action probabilities were averaged and subtracted in float32.
        data = (item["mean"], *item["birth_values"].values(), *item["ci95"])
        replay = (matched["mean"], *matched["birth_values"].values(), *matched["ci95"])
        check(np.allclose(data, replay, rtol=0, atol=6e-8), f"Checkpoint {step} interval differs")

    def contrast(item, x, y):
        verifies(item, [x["birth_values"][b] - y["birth_values"][b] for b in births])

    for task in source["tasks"]:
        for delay in map(str, source["delays"]):
            by_controller = {controller: {arm: source[f"{prefix}results"][stage][task][arm][delay]
                                          for arm in source["arms"]}
                             for controller, prefix in (("local", ""), ("legacy", "legacy_"))}
            for arms in by_controller.values():
                for arm_data in arms.values():
                    branches = arm_data["branches"]
                    for values in branches.values():
                        for item in values.values():
                            verifies(item, [item["birth_values"][b] for b in births])
                    for branch, values in arm_data["intact_minus_branch"].items():
                        for metric, item in values.items():
                            contrast(item, branches["intact"][metric], branches[branch][metric])
            local = by_controller["local"]
            for metric, item in source["active_minus_independent"][stage][task][delay].items():
                contrast(item, local["active"]["branches"]["intact"][metric],
                         local["independent"]["branches"]["intact"][metric])
            for arm, comparisons in source["local_minus_legacy"][stage][task].items():
                for branch, values in comparisons[delay].items():
                    counterpart = "no_write" if branch == "sham" else branch
                    for metric, item in values.items():
                        contrast(item, local[arm]["branches"][branch][metric],
                                 by_controller["legacy"][arm]["branches"][counterpart][metric])


def interval(item):
    return [item["mean"], *item["ci95"]]


def reduce_metrics(metrics, keys, *, report_births=False):
    result = {name: interval(metrics[name]) for name in keys if name in metrics}
    if report_births and "weaker_law" in result:
        result["weaker_law_by_birth"] = metrics["weaker_law"]["birth_values"]
    return result


def sum_counts(by_birth):
    rows = list(by_birth.values())
    return {key: (max if "max" in key else sum)(row[key] for row in rows) for key in rows[0]}


def summarize_stage(source, step):
    stage = str(step)
    output = {"worlds_per_birth": source["worlds_per_birth"], "tasks": {}}
    for task in source["tasks"]:
        keys = BASE_METRICS + (ODOR_METRICS if task == "odor" else ("weaker_law",))
        block = output["tasks"][task] = {}
        checks = source["audits"][stage][task]
        block["write_audit"] = {}
        for arm, by_birth in checks.items():
            block["write_audit"][arm] = {
                label: sum_counts({birth: value[label] for birth, value in by_birth.items()})
                for label in ("shuffle_last_commit", "shuffle_values_last_commit", "precommit_twin_difference_lanes")
            }
            block["write_audit"][arm]["intact_counts"] = sum_counts({
                birth: value["counts_across_contacts"]["intact"] for birth, value in by_birth.items()
            })
            block["write_audit"][arm]["readout_twin_difference_lanes"] = {str(delay): sum_counts({
                birth: value["readout_twin_difference_lanes"][str(delay)] for birth, value in by_birth.items()
            }) for delay in source["delays"]}
        for delay in map(str, source["delays"]):
            row = block[delay] = {}
            for controller, prefix in (("local", ""), ("legacy", "legacy_")):
                row[controller] = {}
                for arm in source["arms"]:
                    arms = source[f"{prefix}results"][stage][task][arm][delay]
                    row[controller][arm] = {
                        "branches": {branch: reduce_metrics(values, keys, report_births=(branch == "intact"))
                                     for branch, values in arms["branches"].items()},
                        "intact_minus_branch": {branch: reduce_metrics(values, ("correct", "weaker_law"))
                                                for branch, values in arms["intact_minus_branch"].items()},
                    }
            row["local_minus_legacy"] = {arm: {
                branch: reduce_metrics(vals, ("correct", "both_twins_correct", "weaker_law"))
                for branch, vals in source["local_minus_legacy"][stage][task][arm][delay].items()
            } for arm in source["arms"]}
            row["local_active_minus_independent"] = reduce_metrics(source["active_minus_independent"][stage][task][delay], keys)
    return output


def historical(path):
    old = json_from(path)
    odor = old["autonomous_first_choice"]["ranked"]["8000"]
    marks = old["variable_law_continuation"]["12000"]["variable"]
    return {
        "interpretation": "Archived, separately trained models. The contemporary legacy fit supplies the paired comparison.",
        "intervals": "Prior odor bootstrap and prior feeder t interval, each across births.",
        "source_sha256": fingerprint(path),
        "ranked_odor_8000_correct": {delay: interval(odor[delay]["true"]) for delay in ("0", "4", "16")},
        "variable_feeder_12000": {key: interval(marks[key]) for key in ("own", "worse_law")},
        "variable_feeder_12000_weaker_law_by_birth": dict(zip(map(str, old["births"]), marks["worse_law"]["by_birth"])),
    }


def run(root, out, historical_path):
    result = {"interval_format": ["birth_mean", "bootstrap95_low", "bootstrap95_high"],
              "intact_birth_count_format": ["successes", "trials"], "stages": {}, "verification": {}}
    manifests = {controller: json_from(root / controller / "run.json") for controller in ("local", "legacy")}
    result["training_config"] = {key: value["config"] for key, value in manifests.items()}
    for step in STAGES:
        source = json_from(root / f"eval-{step}" / "summary.json")
        for field in ("births", "seed", "bootstrap_draws", "interval_unit", "branch_phase", "feeder_novel"):
            check(result.setdefault(field, source[field]) == source[field], f"Study metadata differs for {field}")
        stage = summarize_stage(source, step)
        stage["training_lives_per_birth_per_task_and_arm"] = 2 * step * manifests["local"]["config"]["pairs"]
        result["stages"][str(step)] = stage
        result["verification"][str(step)] = verify_stage(root, step, source)
        verify_intervals(source, step)
        print(f"verified and summarized step {step}", flush=True)
    result["historical_reference"] = historical(historical_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return result


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    blue, orange, gray = "#176A86", "#AA5630", "#7D807D"
    fig, (odor, marks) = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained")
    final = summary["stages"]["8000"]["tasks"]
    comparisons = (("Local trace", "local", "intact", blue), ("Sham", "local", "sham", gray),
                   ("Shuffle values", "local", "shuffle_values", gray),
                   ("Addressed writer", "legacy", "intact", orange))
    for i, (label, controller, branch, color) in enumerate(comparisons):
        offset = (i - 1.5) * .16
        points = np.array([final["odor"][str(d)][controller]["active"]["branches"][branch]["correct"]
                           for d in (0, 4, 16)]) * 100
        mean, lo, hi = points.T
        odor.errorbar(np.arange(3) + offset, mean, yerr=(mean - lo, hi - mean), fmt="o", color=color,
                      markerfacecolor="white" if i == 2 else color, capsize=3, label=label)
    odor.set(xticks=[0, 1, 2], xticklabels=["0", "4", "16"], xlabel="Neutral delay (ticks)",
             ylabel="Autonomous return correct (%)", title="A. Revisit a novel odor pair", ylim=(43, 104))
    odor.legend(frameon=False, loc="center", ncols=2, fontsize=9)

    roots = np.arange(len(summary["births"]))
    for i, (task, label, color) in enumerate((("feeder", "Familiar marks", blue),
                                              ("feeder_novel", "Novel demonstrated marks", orange))):
        values = final[task]["0"]["local"]["active"]["branches"]["intact"]["weaker_law_by_birth"]
        marks.plot(roots + (i - .5) * .17, [100 * values[str(b)] for b in summary["births"]],
                   "o", color=color, label=label, markersize=7, linestyle="none")
    marks.set(xticks=roots, xticklabels=summary["births"], xlabel="Birth seed",
              ylabel="Weaker law correct (%)", title="B. Feeder mappings within each birth", ylim=(43, 104))
    marks.legend(frameon=False, loc="upper center", ncols=2, fontsize=9)
    for ax in (odor, marks):
        ax.axhline(50, color="#9B9D98", linestyle="--", linewidth=1, zorder=0)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE1DF", linewidth=.7)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/local-trace-20261119")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/evidence/local-trace-2026-11-19/results.json")
    parser.add_argument("--historical", type=Path, default=ROOT / "docs/evidence/revisit-choice-2026-11-19/summary.json")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.out, args.historical)
    if args.figure:
        plot(result, args.figure)


if __name__ == "__main__":
    main()
