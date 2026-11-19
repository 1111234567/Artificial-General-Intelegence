"""Bind the independent native scorer and independently audit time diagnostics.

Native outcomes use the unchanged action_split request-state-machine auditor.
Auxiliary metrics are rebuilt here from finite raw rows and request-indexed
tick lists; the evaluator's time_statistics implementation is never called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from experiments.action_split import eval_audit as native
from experiments.continuity.run import state_digest

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data/remaining-time/eval-v1"
OUT = ROOT / "data/remaining-time/eval-audit-v1"
PLAN = OUT / "metrics_plan.json"
COMMON = ROOT / "data/remaining-time/common-v1"
COMMON_PLAN = OUT / "common_metrics_plan.json"
EPS = 1e-6
sha, read, digest = native.sha, native.read, native.digest


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def exact(actual, expected, name):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), name
        for key, value in actual.items():
            exact(value, expected[key], f"{name}.{key}")
    elif isinstance(actual, list):
        assert isinstance(expected, list) and len(actual) == len(expected), name
        for index, value in enumerate(actual):
            exact(value, expected[index], f"{name}[{index}]")
    else:
        assert actual == expected, f"{name}: {actual!r} != {expected!r}"


def independent_time_statistics(facts, prediction, logits, requests, warm=512):
    """Rebuild every reported row metric and adjacent-pair record independently.

    Raw signed time is computed only at selected finite started rows. The
    request denominator is supplied by the independently checked state machine,
    and remains distinct from valid-row and paired-boundary denominators.
    """
    assert prediction.shape == facts["active"].shape
    assert prediction.dtype == logits.dtype == np.float32
    assert logits.shape == (*prediction.shape, 11)
    assert np.isfinite(prediction).all() and np.isfinite(logits).all()
    take = facts["active"] & facts["started"] & np.isfinite(facts["ready"])
    take[:warm] = False
    tick, lane, job = np.nonzero(take)
    truth = facts["ready"][tick, lane, job].astype(np.float64) - facts["now"][tick]
    physical = prediction[tick, lane, job].astype(np.float64) * 16.
    error = physical - truth
    within_deadline = facts["now"][tick] <= facts["deadline"][tick, lane, job] + EPS
    expected_content = 4 * facts["version"][tick, lane, job] + facts["payload"][tick, lane, job]
    content = logits[tick, lane, job, 3:11].argmax(-1) == expected_content
    processing = truth > EPS
    slices = dict(all_valid=np.ones(len(tick), dtype=bool), processing=processing,
        legal=(~processing) & within_deadline,
        remaining_over4=truth > 4., remaining_1to4=(truth > 1.) & (truth <= 4.),
        remaining_0to1=processing & (truth <= 1.),
        near_boundary=(truth >= -.25 - EPS) & (truth <= .25 + EPS),
        overdue_beyond_training=truth < -.25 - EPS)
    groups = {}
    for name, selected in slices.items():
        errors = error[selected]
        n = int(np.count_nonzero(selected))
        absolute = float(np.abs(errors).sum())
        squared = float((errors * errors).sum())
        signed = float(errors.sum())
        predicted_ready = physical[selected] <= 0.
        actual_ready = truth[selected] <= EPS
        groups[name] = dict(rows=n, absolute_error_sum=absolute,
            squared_error_sum=squared, signed_error_sum=signed,
            mae_physical=absolute / n if n else None,
            rmse_physical=float(np.sqrt(squared / n)) if n else None,
            bias_physical=signed / n if n else None,
            mae_ticks=(absolute / n) / .25 if n else None,
            conditional_content_correct=int(content[selected].sum()),
            predicted_ready=int(predicted_ready.sum()), actual_ready=int(actual_ready.sum()),
            sign_correct=int((predicted_ready == actual_ready).sum()))
    by_request = {}
    for row, (t, l, j) in enumerate(zip(tick, lane, job)):
        key = (int(l), int(j), int(facts["rid"][t, l, j]))
        by_request.setdefault(key, []).append(row)
    pairs = []
    for request in requests:
        key = request["lane"], request["job"], request["rid"]
        rows = np.asarray(by_request.get(key, ()), dtype=np.int64)
        before = rows[processing[rows]]
        after = rows[(~processing[rows]) & within_deadline[rows]]
        if before.size == 0 or after.size == 0:
            continue
        b, a = int(before[-1]), int(after[0])
        if int(tick[b]) + 1 != int(tick[a]):
            continue
        first, second = float(physical[b]), float(physical[a])
        pairs.append(dict(actor=request["actor"], lane=request["lane"],
            job=request["job"], rid=request["rid"], family=request["family"],
            conflict=request["conflict"], ticks=[int(tick[b]), int(tick[a])],
            predicted_physical=[first, second], target_physical=[float(truth[b]), float(truth[a])],
            both_sign_correct=first > 0. and second <= 0., direction_correct=first > second,
            direction_tie=first == second, false_early=first <= 0., missed_first=second > 0.,
            first_content_correct=bool(content[a])))
    pair_groups = {}
    for name in ("all", "A_conflict", "B_conflict"):
        selected = [row for row in pairs if name == "all" or
                    (row["family"] == name[0] and row["conflict"])]
        pair_groups[name] = dict(pairs=len(selected), **{key: sum(int(row[key]) for row in selected)
            for key in ("both_sign_correct", "direction_correct", "direction_tie", "false_early",
                        "missed_first", "first_content_correct")})
    return dict(groups=groups, boundary=pair_groups, boundary_rows=pairs,
                mature_request_denominator=len(requests), paired_requests=len(pairs))


def declare():
    protocol, completion = read(EVAL / "protocol.json"), read(EVAL / "completion.json")
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
                    assert sha(files[2]) == completion["files"][str(files[2].relative_to(ROOT))]
                    assert sha(files[3]) == read(files[2])["sidecar_sha256"]
                inputs.update({str(path.relative_to(ROOT)): sha(path) for path in files})
                runs.append(dict(seed=seed, policy=policy, condition=condition,
                                 stem=str(stem.relative_to(ROOT))))
    write_new(PLAN, dict(format="remaining-time-independent-metrics-audit-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), runs=runs, inputs=inputs,
        sources={str(path.relative_to(ROOT)): sha(path) for path in
                 (Path(__file__), ROOT / "experiments/action_split/eval_audit.py")},
        behavior="Unchanged action_split independent request-state-machine scorer; only EVAL/OUT rebound",
        auxiliary="Independent selected-row physical errors, fixedzero readiness signs, conditional content and request-indexed lastillegal/firstlegal adjacentpairs; full mature denominator retained"))


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
    native_audit = read(OUT / "metrics.json")
    assert native_audit["passed"] and len(native_audit["runs"]) == 30
    audit_rows = {(row["seed"], row["policy"], row["condition"]): row for row in native_audit["runs"]}
    output = []
    for run in plan["runs"]:
        if run["policy"] == "original":
            continue
        stem = ROOT / run["stem"]
        report = read(stem.with_suffix(".json"))
        assert audit_rows[(run["seed"], run["policy"], run["condition"])]["requests_digest"] == digest(report["requests"])
        side = stem.with_name(stem.name + "_time")
        saved = read(side.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
        with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
            prediction, logits = archive["time_prediction"], archive["native_logits"]
        np.testing.assert_array_equal(logits.argmax(-1), facts["actions"])
        assert saved["facts_sha256"] == sha(stem.with_suffix(".npz"))
        assert saved["checkpoint_sha256"] == report["checkpoint_sha256"]
        actual = independent_time_statistics(facts, prediction, logits, report["requests"])
        for key, value in actual.items():
            exact(value, saved[key], f"{run['stem']}/{key}")
        output.append(dict(**{key: run[key] for key in ("seed", "policy", "condition")}, passed=True,
            groups=actual["groups"], boundary=actual["boundary"],
            mature_request_denominator=actual["mature_request_denominator"],
            paired_requests=actual["paired_requests"], boundary_rows_digest=digest(actual["boundary_rows"]),
            facts_sha256=sha(stem.with_suffix(".npz")), sidecar_sha256=sha(side.with_suffix(".npz")),
            report_sha256=sha(side.with_suffix(".json"))))
    paired = []
    for seed in (0, 1, 2):
        for condition in ("long", "longer"):
            root = EVAL / f"seed_{seed:03d}"
            with np.load(root / f"original_{condition}.npz", allow_pickle=False) as original, \
                 np.load(root / f"probe_{condition}.npz", allow_pickle=False) as probe:
                assert original.files == probe.files
                for key in original.files:
                    np.testing.assert_array_equal(original[key], probe[key], err_msg=f"probe/original/{seed}/{condition}/{key}")
            paired.append(dict(seed=seed, condition=condition, all_raw_arrays="bitwise exact"))
    write_new(OUT / "auxiliary_metrics.json", dict(format="remaining-time-independent-auxiliary-metrics-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), plan_sha256=sha(PLAN),
        behavior_audit_sha256=sha(OUT / "metrics.json"), runs=output,
        probe_original=paired,
        scope="All24 sidecars: physical MAE/RMSE/bias/tickMAE sums and counts in all8groups; conditionalcontent; zero-sign; every adjacentpair record and subgroups exact; every full mature request denominator retained. All6 probe/original rawarchives bitwise exact.",
        interpretation="Read-only diagnostics under each policy history; row and boundary subsets never replace native mature-request outcome denominators."))


def declare_common():
    protocol = read(COMMON / "protocol.json")
    freeze = read(COMMON / "histories.freeze.json")
    completion = read(COMMON / "completion.json")
    assert completion["complete"] and completion["histories"] == 6 and completion["model_replays"] == 24
    assert state_digest({key: value for key, value in protocol.items() if key != "protocol_hash"}) == protocol["protocol_hash"]
    assert state_digest({key: value for key, value in freeze.items() if key != "freeze_hash"}) == freeze["freeze_hash"]
    assert freeze["protocol_sha256"] == completion["protocol_sha256"] == sha(COMMON / "protocol.json")
    assert completion["history_freeze_sha256"] == sha(COMMON / "histories.freeze.json")
    inputs = {str((COMMON / name).relative_to(ROOT)): sha(COMMON / name)
              for name in ("protocol.json", "histories.freeze.json", "completion.json")}
    for table in (freeze["files"], completion["files"], protocol["checkpoints"]):
        for name, expected in table.items():
            assert sha(ROOT / name) == expected, name
            inputs[name] = expected
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for seed in protocol["seeds"]:
        for condition in protocol["conditions"]:
            path = EVAL / f"seed_{seed:03d}" / f"original_{condition}.json"
            inputs[str(path.relative_to(ROOT))] = sha(path)
    write_new(COMMON_PLAN, dict(format="remaining-time-independent-common-metrics-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        sources={**protocol["source_sha256"], **{str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), ROOT / "experiments/action_split/eval_audit.py")}},
        cases=[dict(seed=seed, condition=condition) for seed in protocol["seeds"] for condition in protocol["conditions"]],
        arms=protocol["arms"],
        scope="Independent teacher request-state-machine denominators and all24 common-history auxiliary metrics; teachercompletion is not tested-policy performance"))


def common():
    plan = read(COMMON_PLAN)
    for table in ("inputs", "sources"):
        for name, expected in plan[table].items():
            assert sha(ROOT / name) == expected, name
    completion = read(COMMON / "completion.json")
    outputs = []
    for case in plan["cases"]:
        seed, condition = case["seed"], case["condition"]
        path = COMMON / f"seed_{seed:03d}" / f"teacher_{condition}.npz"
        teacher = read(path.with_suffix(".json"))
        assert teacher["teacher_request_report_only"] and not teacher["autonomous_test_model_score"]
        with np.load(path, allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key != "obs"}
        state_machine = native.score(facts, seed, 512)
        for key in ("coverage", "groups", "requests", "idle_opportunities", "idle_actions"):
            exact(state_machine[key], teacher[key], f"teacher/{seed}/{condition}/{key}")
        own_original = read(EVAL / f"seed_{seed:03d}" / f"original_{condition}.json")
        assert own_original["external_hash"] == teacher["external_hash"]
        assert {(row["actor"], row["job"], row["rid"]) for row in own_original["requests"]} == {
            (row["actor"], row["job"], row["rid"]) for row in state_machine["requests"]}
        arms = {}
        for arm in plan["arms"]:
            side = path.with_name(f"{arm}_{condition}.npz")
            report = read(side.with_suffix(".json"))
            assert report["diagnostic_only"] and report["no_test_model_actions_executed"]
            assert report["facts_sha256"] == sha(path)
            assert report["raw_predictions_sha256"] == sha(side)
            assert report["teacher_request_report_sha256"] == sha(path.with_suffix(".json"))
            assert report["model_before_sha256"] == report["model_after_sha256"]
            with np.load(side, allow_pickle=False) as archive:
                prediction, logits = archive["time_prediction"], archive["native_logits"]
            actual = independent_time_statistics(facts, prediction, logits, state_machine["requests"])
            for key, value in actual.items():
                exact(value, report[key], f"common/{seed}/{condition}/{arm}/{key}")
            for key in ("groups", "boundary", "mature_request_denominator", "paired_requests"):
                exact(actual[key], completion["cases"][f"{seed}:{condition}"][arm][key], "common manifest summary")
            arms[arm] = dict(passed=True, groups=actual["groups"], boundary=actual["boundary"],
                mature_request_denominator=actual["mature_request_denominator"],
                paired_requests=actual["paired_requests"], boundary_rows_digest=digest(actual["boundary_rows"]),
                sidecar_sha256=sha(side), report_sha256=sha(side.with_suffix(".json")))
        assert len({row["mature_request_denominator"] for row in arms.values()}) == 1
        assert len({row["paired_requests"] for row in arms.values()}) == 1
        assert len({row["groups"]["all_valid"]["rows"] for row in arms.values()}) == 1
        outputs.append(dict(seed=seed, condition=condition, passed=True,
            teacher_facts_sha256=sha(path), teacher_request_report_sha256=sha(path.with_suffix(".json")),
            teacher_request_count=len(state_machine["requests"]),
            matched_autonomous_external_hash_and_request_ids=True,
            teacher_request_digest=digest(state_machine["requests"]), arms=arms))
    write_new(OUT / "common_history_metrics.json", dict(format="remaining-time-independent-common-metrics-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), plan_sha256=sha(COMMON_PLAN), cases=outputs,
        histories=6, auxiliary_replays=24,
        scope="Independent6teacherrequeststatemachines andall24common-history rawprediction derivedmetrics: all8groups, everyboundarypair, fixedsign/content counts, fullmaturedenominators andmanifestsummary exact",
        interpretation="Only diagnostic comparison on identical actual teacher observations. Teacher outcome counts are denominator checks and are not tested-policy autonomous performance."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "behavior", "auxiliary", "all", "declare-common", "common"))
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "behavior":
        behavior()
    elif args.command == "auxiliary":
        auxiliary()
    elif args.command == "declare-common":
        declare_common()
    elif args.command == "common":
        common()
    else:
        behavior()
        auxiliary()
