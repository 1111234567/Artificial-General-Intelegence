"""Reproduce selected final-stage diagnostics from strictly audited inputs.

This analysis does not train, evaluate, or alter frozen runtime/evidence inputs.
Requires ijson to verify raw bindings without materializing the full report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import ijson


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SELECTORS = [dict(name=name, architecture="fixed_shrink", schedule="mixed",
    learning="history_replay", coverage=coverage, condition=condition,
    tick=172032, family=0, steps=2048, cohort="primary")
    for name, coverage, condition in (("narrow_short", "narrow", "short"),
        ("wide_short", "wide", "short"), ("wide_long_both", "wide", "long_both"))]
PHASE_FIELDS = ("opportunities", "model_finish_attempts", "executed_finish_attempts",
    "finish_choice_correct", "payload_choice_correct", "version_choice_correct")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(values):
    if [row["seed"] for row in values] != [0, 1, 2]:
        raise ValueError("exactly three trained seeds are required")
    numbers = [row["value"] for row in values]
    return dict(n=3, mean=statistics.mean(numbers), sample_sd=statistics.stdev(numbers),
        per_seed=values)


def matches(row, selector):
    return all(str(row[key]) == str(selector[key]) for key in (
        "architecture", "schedule", "learning", "coverage", "condition", "tick", "family", "steps"))


def aggregate_bindings(path, required):
    """Read only small header fields and input bindings, before large tables."""
    header = {}
    found = {}
    with Path(path).open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix in ("format", "complete", "audit.status", "audit.checks") and event in (
                    "string", "boolean", "number"):
                header[prefix] = value
            if prefix.startswith("inputs.") and event == "string":
                key = prefix.removeprefix("inputs.")
                if key in required:
                    found[key] = value
            if prefix == "inputs" and event == "end_map":
                break
    if header.get("complete") is not True or header.get("audit.status") != "passed":
        raise ValueError("strict aggregate did not pass")
    if set(found) != required:
        raise ValueError("raw files are missing from the strict aggregate bindings")
    return header, found


def build(aggregate, raw):
    aggregate, raw = Path(aggregate).resolve(), Path(raw).resolve()
    artifacts_path = aggregate / "artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    content_path = aggregate / "content_by_phase_age_per_seed.csv"
    report_path = aggregate / "report.json"
    for path in (content_path, report_path):
        if sha(path) != artifacts[path.name]:
            raise ValueError(f"aggregate artifact changed: {path}")
    raw_paths = {str(raw / "evaluation" / f"seed_{seed:03d}__fixed_shrink__mixed__history_replay__{coverage}.json")
        for seed in (0, 1, 2) for coverage in ("narrow", "wide")}
    header, bindings = aggregate_bindings(report_path, raw_paths)
    inputs = {str(artifacts_path): sha(artifacts_path), str(content_path): sha(content_path),
        str(report_path): artifacts[report_path.name], str(Path(__file__).resolve()): sha(__file__)}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError(f"raw input changed: {path}")
        inputs[path] = expected

    phases = {}
    with content_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["cohort"] != "primary" or row["phase"] != "legal_finish":
                continue
            for selector in SELECTORS:
                if not matches(row, selector):
                    continue
                seed = int(row["seed"])
                target = phases.setdefault((selector["name"], seed),
                    dict(age_rows=0, **{key: 0 for key in PHASE_FIELDS}))
                target["age_rows"] += 1
                for field in PHASE_FIELDS:
                    target[field] += int(row[field])

    selected = {selector["name"]: [] for selector in SELECTORS}
    for path in sorted(raw_paths):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        branch = value["branch"]
        for selector in SELECTORS:
            if any(branch[key] != selector[key] for key in ("architecture", "schedule", "learning", "coverage")):
                continue
            records = [record for record in value["records"] if all(
                record["probe"][key] == selector[key] for key in ("tick", "family", "condition", "steps"))]
            if len(records) != 1:
                raise ValueError("selector is missing or ambiguous")
            record, seed = records[0], branch["seed"]
            if record["parameter_updates"] or record["intervention"] != "none" or record["action_policy"] != "model":
                raise ValueError("diagnostic source is not a frozen autonomous model fork")
            detail = record["request_diagnosis"]["primary"]
            n = detail["requests"]
            if n != record["primary"]["matured"] or n <= 0:
                raise ValueError("request denominator differs from the audited mature cohort")
            phase = phases[(selector["name"], seed)]
            if phase["model_finish_attempts"] != phase["executed_finish_attempts"] or phase["opportunities"] <= 0:
                raise ValueError("legal-phase model action accounting disagrees")
            metrics = {}

            def ratio(name, numerator, denominator, unit="percent"):
                if denominator <= 0:
                    raise ValueError(f"undefined selected metric: {name}")
                metrics[name] = dict(numerator=numerator, denominator=denominator,
                    value=(100 if unit == "percent" else 1) * numerator / denominator, unit=unit)

            for name, field in (("ordinary_completion", "completed"), ("strict_completion", "clean_completed"),
                    ("began", "began"), ("recorded_or_inherited_error", "with_recorded_or_inherited_error"),
                    ("uncompleted_no_begin", "uncompleted_no_begin"),
                    ("uncompleted_despite_legal", "uncompleted_despite_legal_opportunity")):
                ratio(name, detail[field], n)
            ratio("completed_after_error", detail["completed"] - detail["clean_completed"], n)
            for cause in ("wrong_payload", "early", "stale_version"):
                ratio("first_error_" + cause, detail["first_error_causes"].get(cause, 0), n)
            ratio("active_finish_attempts_per_matured_request", detail["active_finish_attempts"], n,
                unit="attempts_per_request")
            for name, field in (("legal_finish_action", "model_finish_attempts"),
                    ("legal_conditional_finish_readout", "finish_choice_correct"),
                    ("legal_conditional_payload_readout", "payload_choice_correct"),
                    ("legal_conditional_version_readout", "version_choice_correct")):
                ratio(name, phase[field], phase["opportunities"])
            selected[selector["name"]].append(dict(seed=seed, branch=branch["id"], raw_input=path,
                probe=record["probe"], mature_primary_requests=n, primary_censored=record["censored"]["primary"]["count"],
                primary_request_diagnosis=detail, pooled_primary_legal_phase=phase, metrics=metrics))
        del value

    groups = []
    for selector in SELECTORS:
        rows = sorted(selected[selector["name"]], key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != [0, 1, 2]:
            raise ValueError("incomplete selected seed set")
        summaries = {name: summary([dict(seed=row["seed"], value=row["metrics"][name]["value"])
            for row in rows]) | dict(unit=rows[0]["metrics"][name]["unit"]) for name in rows[0]["metrics"]}
        groups.append(dict(selector=selector, per_seed=rows, summary=summaries))
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"input changed during diagnosis: {path}")
    return dict(format="lifelong-selected-diagnosis-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        strict_aggregate_header=header, inputs_sha256=inputs, selectors=SELECTORS, groups=groups,
        method="Select exact final2048tick frozen probes. Sum legal_finish count numerators and denominators over age bins within each trained seed, compute that seed's rates, then report the equal-weight mean and sample SD of seeds0,1,2. Request metrics use each seed's mature primary request count. No request-level pseudoreplication.",
        limitations=[
            "Request outcomes and first-error causes concern mature final-effective primary requests; right-censored requests are reported separately.",
            "The legal-phase table covers policy-dependent active primary opportunities, including possibly right-censored requests; its denominator is not the mature-request denominator.",
            "Conditional FINISH/payload/version readouts use the existing eight-FINISH head; they do not measure all information recoverable from hidden state and do not establish a chance baseline.",
            "First-error causes are the evaluator's exclusive attribution; full error patterns and raw counts are retained, and later errors can coexist.",
            "Low illegal idle action frequency alone does not establish successful active control. Wide probes show acquisition failure before withdrawal, so their floor scores do not demonstrate memory retention.",
            "Shrink strength, temporal supervision density, old source hidden state with finite current-weight warmup, and capacity are candidate explanations requiring separate interventions; this descriptive analysis identifies none as the unique cause.",
            "Three independent trained seeds remain the statistical sample size; no new behavior runs or parameter training are performed."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=HERE / "aggregate-v1")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/lifelong/main-v1")
    parser.add_argument("--output", type=Path, default=HERE / "diagnosis-v1.json")
    args = parser.parse_args()
    result = build(args.aggregate, args.raw)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), selectors=len(result["groups"]),
        seeds=3, input_files=len(result["inputs_sha256"]))))


if __name__ == "__main__":
    main()
