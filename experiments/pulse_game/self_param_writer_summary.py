"""Summarize raw paired writer evidence without pooling roots into false replications."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .self_param_writer_study import EVIDENCE


def grouped_receipts(records, *, condition=None, intervention="active"):
    counts = Counter()
    for record in records:
        for name, arms in record["after"].items():
            if condition is None or condition == name:
                counts.update(arms[intervention]["receipts"])
    return dict(counts)


def progress(record, condition, arm, epoch, window):
    entries = record[epoch][condition][arm]["curve"]
    if window == "first":
        entries = entries[:8]
    if window == "last":
        entries = entries[-8:]
    return mean(entries)


def paired_scores(records, epoch, conditions):
    result = {}
    for condition in conditions:
        name = condition["name"]
        windows = {}
        for window in ("all", "first", "last"):
            widths = {}
            for width, rows in records.items():
                arms = {
                    arm: {str(record["root"]): progress(record, name, arm, epoch, window)
                          for record in rows}
                    for arm in condition["arms"]
                }
                widths[width] = {
                    "mean_each_arm": {arm: mean(values.values()) for arm, values in arms.items()},
                    "each_arm_by_root": arms,
                    "active_minus_each_alternative_by_root": {
                        arm: {root: score - arms[arm][root]
                              for root, score in arms["active"].items()}
                        for arm in arms if arm != "active"
                    },
                }
            for_width = lambda width: widths[width]["each_arm_by_root"]["active"]
            widths["eight_minus_four_active_by_root"] = {
                root: score - for_width("4")[root] for root, score in for_width("8").items()
            }
            windows[window] = widths
        result[name] = windows
    return result


def main():
    protocol = EVIDENCE / "protocol.json"
    plan = json.loads(protocol.read_text(encoding="utf-8"))
    records = {}
    source = Path(__file__).resolve().parent
    inputs = [protocol, source / "self_param_writer.py", source / "self_param_writer_study.py",
              source / "self_write_transfer.py"]
    for width in (4, 8):
        rows = []
        for root in plan["roots"]:
            stem = EVIDENCE / f"writer-{width}-root-{root}"
            inputs.extend((stem.with_suffix(".npz"), stem.with_suffix(".json")))
            rows.append(json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")))
        records[str(width)] = rows
    paired = {epoch: paired_scores(records, epoch, plan["holdout"]) for epoch in ("before", "after")}
    report = {
        "sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs},
        "paired_hard_action_accuracy": paired,
        "counts_by_width_root": {width: {str(row["root"]): row["counts"] for row in rows}
                                 for width, rows in records.items()},
        "active_receipts_by_width": {width: grouped_receipts(rows) for width, rows in records.items()},
        "active_receipts_by_width_and_condition": {
            width: {condition["name"]: grouped_receipts(rows, condition=condition["name"])
                    for condition in plan["holdout"]} for width, rows in records.items()},
    }
    path = EVIDENCE / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
