"""Independent native behavior and read-only time-sidecar audit.

The frozen native state-machine scorer and independent selected-row arithmetic
are reused with new artifact paths only. No evaluator, model or learner runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.action_split import eval_audit as native
from experiments.remaining_time.eval_audit import (
    exact, independent_time_statistics, write_new,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data/teacher-timing/eval-v1"
OUT = ROOT / "data/teacher-timing/eval-audit-v1"
PLAN = OUT / "metrics_plan.json"
POLICIES = ("original", "earliest_action", "earliest_joint",
            "midpoint_action", "midpoint_joint")
EXTERNAL_FIELDS = ("actor_ids", "now", "event_kind", "event_id", "event_eligible",
                   "event_family", "event_raw", "event_version", "event_duration",
                   "event_window", "mature", "mature_id", "mature_issued",
                   "mature_duration", "mature_family")
sha, read, digest = native.sha, native.read, native.digest


def declare():
    protocol, completion = read(EVAL / "protocol.json"), read(EVAL / "completion.json")
    assert tuple(protocol["policies"]) == POLICIES
    assert protocol["seeds"] == [0, 1, 2]
    assert protocol["conditions"] == {"long": [8., 16.], "longer": [16., 32.]}
    assert completion["complete"] and completion["runs"] == 30 and completion["sidecars"] == 24
    inputs = {str((EVAL / name).relative_to(ROOT)): sha(EVAL / name)
              for name in ("protocol.json", "freeze.json", "completion.json")}
    runs = []
    for seed in protocol["seeds"]:
        for policy in protocol["policies"]:
            for condition in protocol["conditions"]:
                stem = EVAL / f"seed_{seed:03d}" / f"{policy}_{condition}"
                report = read(stem.with_suffix(".json"))
                files = [stem.with_suffix(".json"), stem.with_suffix(".npz")]
                assert sha(files[0]) == completion["files"][str(files[0].relative_to(ROOT))]
                assert sha(files[1]) == report["facts_sha256"]
                if policy != "original":
                    files.extend(stem.with_name(stem.name + "_time").with_suffix(ext)
                                 for ext in (".json", ".npz"))
                    assert sha(files[2]) == completion["time_files"][str(files[2].relative_to(ROOT))]
                    assert sha(files[3]) == read(files[2])["sidecar_sha256"]
                inputs.update({str(path.relative_to(ROOT)): sha(path) for path in files})
                runs.append(dict(seed=seed, policy=policy, condition=condition,
                                 stem=str(stem.relative_to(ROOT))))
    sources = (Path(__file__), ROOT / "experiments/action_split/eval_audit.py",
               ROOT / "experiments/remaining_time/eval_audit.py")
    write_new(PLAN, dict(format="teacher-timing-independent-metrics-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), runs=runs, inputs=inputs,
        sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
        behavior="Frozen action_split independent request state machine; only EVAL and OUT rebound",
        auxiliary="Frozen independently selected-row time arithmetic, fixed zero readiness signs, conditional content, every adjacent request boundary pair; full native mature denominator retained",
        pairing="Every raw exogenous event and maturity field must be bitwise equal across all five policies per seed and condition; full request IDs and report external hashes must agree"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for name, expected in plan[table].items():
            assert sha(ROOT / name) == expected, name
    return plan


def behavior():
    checked()
    native.EVAL, native.OUT = EVAL, OUT
    native.main()


def auxiliary():
    plan = checked()
    behavior_audit = read(OUT / "metrics.json")
    assert behavior_audit["passed"] and len(behavior_audit["runs"]) == 30
    audited = {(row["seed"], row["policy"], row["condition"]): row
               for row in behavior_audit["runs"]}
    output = []
    for run in plan["runs"]:
        if run["policy"] == "original":
            continue
        stem = ROOT / run["stem"]
        report = read(stem.with_suffix(".json"))
        assert audited[(run["seed"], run["policy"], run["condition"])]["requests_digest"] == digest(report["requests"])
        side = stem.with_name(stem.name + "_time")
        saved = read(side.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
        with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
            prediction, logits = archive["time_prediction"], archive["native_logits"]
        np.testing.assert_array_equal(logits.argmax(-1), facts["actions"])
        assert saved["facts_sha256"] == sha(stem.with_suffix(".npz"))
        assert saved["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert saved["protocol_sha256"] == report["protocol_sha256"]
        assert saved["freeze_sha256"] == report["freeze_sha256"]
        actual = independent_time_statistics(facts, prediction, logits, report["requests"])
        for key, value in actual.items():
            exact(value, saved[key], f"{run['stem']}/{key}")
        output.append(dict(**{key: run[key] for key in ("seed", "policy", "condition")},
            passed=True, groups=actual["groups"], boundary=actual["boundary"],
            mature_request_denominator=actual["mature_request_denominator"],
            paired_requests=actual["paired_requests"],
            boundary_rows_digest=digest(actual["boundary_rows"]),
            facts_sha256=sha(stem.with_suffix(".npz")),
            sidecar_sha256=sha(side.with_suffix(".npz")),
            report_sha256=sha(side.with_suffix(".json"))))
    pairing = []
    for seed in (0, 1, 2):
        for condition in ("long", "longer"):
            folder = EVAL / f"seed_{seed:03d}"
            anchor_report = read(folder / f"original_{condition}.json")
            anchor_ids = [(row["actor"], row["job"], row["rid"])
                          for row in anchor_report["requests"]]
            with np.load(folder / f"original_{condition}.npz", allow_pickle=False) as anchor:
                for policy in POLICIES[1:]:
                    report = read(folder / f"{policy}_{condition}.json")
                    assert report["external_hash"] == anchor_report["external_hash"]
                    assert [(row["actor"], row["job"], row["rid"])
                            for row in report["requests"]] == anchor_ids
                    with np.load(folder / f"{policy}_{condition}.npz", allow_pickle=False) as other:
                        for key in EXTERNAL_FIELDS:
                            np.testing.assert_array_equal(anchor[key], other[key],
                                err_msg=f"exogenous/{seed}/{condition}/{policy}/{key}")
            pairing.append(dict(seed=seed, condition=condition, policies=len(POLICIES),
                external_raw_arrays="bitwise exact", external_fields=list(EXTERNAL_FIELDS),
                external_hash=anchor_report["external_hash"],
                mature_request_ids_digest=digest(anchor_ids), mature_requests=len(anchor_ids)))
    write_new(OUT / "auxiliary_metrics.json", dict(
        format="teacher-timing-independent-auxiliary-metrics-v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        plan_sha256=sha(PLAN), behavior_audit_sha256=sha(OUT / "metrics.json"),
        runs=output, paired_exogenous_cases=pairing,
        scope="All24 sidecars: physical MAE/RMSE/bias/tick MAE sums and counts in all eight groups, fixed zero sign and conditional content counts, every adjacent boundary record and subgroup, full mature request denominators; six paired raw exogenous cases across five policies",
        interpretation="Own-policy history diagnostic only. Auxiliary row and boundary subsets never replace native mature-request outcome denominators. The complete teacher-regime contrast includes training labels and resulting public histories."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "behavior", "auxiliary", "all"))
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "behavior":
        behavior()
    elif args.command == "auxiliary":
        auxiliary()
    else:
        behavior()
        auxiliary()
