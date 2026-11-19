"""Aggregate frozen continuity studies without loading or advancing a learner.

Rates are summarized across seeds using the sample standard deviation. Request
counts and actor-level outcomes remain in summary.json; teacher trajectories
contribute only to the separately labelled conditional readout diagnostic.

Example::

    .venv/Scripts/python -m experiments.continuity.report \
        --root data/continuity/main-20261027-v1 \
        --evaluations data/continuity/evaluation-A \
        --adapt-root data/continuity/adapt-20261027-v1 \
        --adapt-evaluations data/continuity/evaluation-B \
        --manifest docs/evidence/continuity-2026-10-27/main-study-v1.json \
        --output docs/evidence/continuity-2026-10-27/aggregate

Use --allow-incomplete for an explicitly incomplete progress report. Integrity
mismatches still fail. --self-test checks statistics and deliberate mismatches,
then renders clearly labelled synthetic figures for visual verification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = "continuity-aggregate-v1"
A_VARIANTS = ("c64", "c128", "c256", "reset128", "reservoir128", "gru128", "online128")
B_MODES = ("frozen", "current", "current_repeat", "replay")
GRADIENT_PASSES = dict(frozen=0, current=1, current_repeat=2, replay=2)
DEFAULT_CONDITIONS = ("iid", "long_wait", "longer", "sparse", "conflict")
MODEL_METRICS = ("primary_clean_completion", "primary_completion",
                 "boundary_clean_completion", "boundary_completion", "finish_precision",
                 "false_actions_per_1000")
CORE_SOURCES = ("world.py", "model.py", "run.py", "test_run.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Each input is one seed, never an actor or an individual request."""
    records = list(values)
    numeric = [float(item["value"]) for item in records if item["value"] is not None]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("summary contains a nonfinite seed value")
    return {"n": len(numeric), "n_total": len(records),
            "n_undefined": len(records) - len(numeric),
            "mean": statistics.mean(numeric) if numeric else None,
            "sample_sd": statistics.stdev(numeric) if len(numeric) > 1 else None,
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "per_seed": deepcopy(records)}


def _run_key(item: dict[str, Any], evaluation: bool = False) -> tuple[str, int, str, str]:
    config = item["training_config"] if evaluation else item["config"]
    variant = item["training_variant"] if evaluation else item["variant"]
    stage = item["training_stage"] if evaluation else item["stage"]
    return stage, int(config["seed"]), variant["name"], item.get("adaptation_mode") or ""


def _run_id(key: tuple[str, int, str, str]) -> str:
    stage, seed, variant, mode = key
    return f"{stage}/seed_{seed:03d}/{mode or variant}"


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["condition"], record["control"], record["action_policy"]


def _expand_paths(inputs: Iterable[str | Path], pattern: str) -> list[Path]:
    found = set()
    for item in inputs:
        text = str(item)
        matches = [Path(value) for value in glob.glob(text, recursive=True)] if any(
            token in text for token in ("*", "?", "[")) else [Path(text)]
        for path in matches:
            if path.is_dir():
                found.update(child.resolve() for child in path.rglob(pattern) if child.is_file())
            elif path.is_file():
                found.add(path.resolve())
    return sorted(found)


def load_training_reports(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in _expand_paths(inputs, "report.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") == "continuity-trainer-v1":
            value["_provenance"] = {"path": str(path), "sha256": file_sha256(path)}
            records.append(value)
    return records


def load_evaluations(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in _expand_paths(inputs, "*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != "continuity-evaluation-v1":
            continue
        entries = value.get("evaluations", [value] if "records" in value else [])
        provenance = {"path": str(path), "sha256": file_sha256(path)}
        for entry in entries:
            entry["_provenance"] = deepcopy(provenance)
            records.append(entry)
    return records


class Audit:
    def __init__(self, allow_incomplete: bool):
        self.allow_incomplete = allow_incomplete
        self.checks: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []

    def problem(self, code: str, message: str, details: Any = None, *, missing: bool = False) -> None:
        self.issues.append({"level": "warning" if missing and self.allow_incomplete else "error",
                            "code": code, "message": message, "details": details})

    def equal(self, code: str, label: str, values: dict[str, Any]) -> None:
        if not values:
            return
        first = next(iter(values.values()))
        passed = first is not None and all(value == first for value in values.values())
        self.checks.append({"code": code, "label": label, "passed": passed, "values": values})
        if not passed:
            self.problem(code, label, values)

    def require(self, code: str, label: str, passed: bool, details: Any = None) -> None:
        self.checks.append({"code": code, "label": label, "passed": bool(passed), "details": details})
        if not passed:
            self.problem(code, label, details)

    def result(self) -> dict[str, Any]:
        errors = sum(item["level"] == "error" for item in self.issues)
        missing = any(item["code"].startswith("MISSING_") for item in self.issues)
        return {"status": "failed" if errors else "incomplete" if missing else "passed",
                "error_count": errors, "warning_count": len(self.issues) - errors,
                "check_count": len(self.checks), "issues": self.issues, "checks": self.checks}


def _metric_values(record: dict[str, Any]) -> dict[str, float | None]:
    return {"primary_clean_completion": record["primary"]["clean_completion_rate"],
            "primary_completion": record["primary"]["completion_rate"],
            "boundary_clean_completion": record["boundary_crossing"]["clean_completion_rate"],
            "boundary_completion": record["boundary_crossing"]["completion_rate"],
            "finish_precision": record.get("post_fork_finish_precision"),
            "false_actions_per_1000": record.get("post_fork_false_actions_per_1000_opportunities")}


def _raw_counts(record: dict[str, Any]) -> dict[str, Any]:
    return {"primary": {key: record["primary"][key] for key in
                        ("matured", "completed", "clean_completed")},
            "boundary_crossing": {key: record["boundary_crossing"][key] for key in
                                   ("matured", "completed", "clean_completed")},
            "post_fork_counters": deepcopy(record["post_fork_counters"]),
            "censored": deepcopy(record["censored"])}


def _audit_cohorts(audit: Audit, run_id: str, record: dict[str, Any]) -> None:
    label = f"{run_id}/{record['condition']}/{record['control']}/{record['action_policy']}"
    for name in ("primary", "boundary_crossing", "inherited", "carryover_contract"):
        cohort = record[name]
        matured, completed, clean = (cohort[key] for key in ("matured", "completed", "clean_completed"))
        valid_counts = all(isinstance(value, int) and not isinstance(value, bool)
                           for value in (matured, completed, clean)) and 0 <= clean <= completed <= matured
        audit.require("COHORT_COUNTS", f"{label}/{name} ordered counts", valid_counts,
                      dict(matured=matured, completed=completed, clean_completed=clean))
        for rate, numerator in (("completion_rate", completed), ("clean_completion_rate", clean)):
            expected = numerator / matured if matured else None
            actual = cohort.get(rate)
            passed = actual is None if expected is None else isinstance(actual, (int, float)) and (
                math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12))
            audit.require("RATE_DENOMINATOR", f"{label}/{name}/{rate}", passed,
                          dict(reported=actual, expected=expected))
        if "per_actor" in cohort:
            for count_name in ("matured", "completed", "clean_completed"):
                values = cohort["per_actor"][count_name]
                audit.require("ACTOR_COUNTS", f"{label}/{name}/{count_name} actor sum",
                              len(values) == record["batch_size"] and sum(values) == cohort[count_name])
    primary, carryover = record["primary"], record["carryover_contract"]
    audit.require("CENSOR_ACCOUNTING", f"{label} primary request conservation",
                  primary["matured"] + record["censored"]["primary"]["count"] ==
                  record["primary_eligible_issued"])
    audit.require("CENSOR_ACCOUNTING", f"{label} prior-contract request conservation",
                  carryover["matured"] + record["censored"]["carryover_contract"]["count"] ==
                  record["carryover_contract_eligible_issued"])
    audit.require("CENSOR_ACCOUNTING", f"{label} inherited request conservation",
                  record["inherited"]["matured"] + record["censored"]["inherited"]["count"] ==
                  record["inherited"]["initial_eligible_pending"])
    audit.require("CENSOR_ACCOUNTING", f"{label} all new eligible requests accounted for",
                  record["primary_eligible_issued"] + record["carryover_contract_eligible_issued"] ==
                  record["post_fork_counters"]["eligible"])
    audit.require("BOUNDARY_SUBSET", f"{label} boundary cohort is within primary cohort",
                  all(record["boundary_crossing"][name] <= record["primary"][name]
                      for name in ("matured", "completed", "clean_completed")))
    expected_precision = record["post_fork_counters"]["completed_all"] / record[
        "post_fork_counters"]["finish_attempted"] if record["post_fork_counters"]["finish_attempted"] else None
    actual_precision = record.get("post_fork_finish_precision")
    audit.require("FINISH_PRECISION", f"{label} FINISH precision denominator",
                  actual_precision is None if expected_precision is None else isinstance(
                      actual_precision, (int, float)) and math.isclose(
                          actual_precision, expected_precision, rel_tol=0, abs_tol=1e-12))


def _compact_training(item: dict[str, Any]) -> dict[str, Any]:
    # Full training traces remain in the SHA-256-addressed source report.
    return {key: deepcopy(value) for key, value in item.items() if key not in ("trace", "curve")}


def _audit_evaluation_spec(audit: Audit, run_id: str, evaluation: dict[str, Any],
                           manifest: dict[str, Any]) -> None:
    arguments = evaluation.get("evaluation_args")
    audit.require("EVALUATION_ARGUMENTS", f"{run_id}/evaluation arguments are recorded",
                  isinstance(arguments, dict))
    arguments = arguments if isinstance(arguments, dict) else {}
    declared_conditions = arguments.get("conditions")
    actual_conditions = sorted({record["condition"] for record in evaluation["records"]})
    valid_conditions = isinstance(declared_conditions, list) and all(
        isinstance(value, str) for value in declared_conditions) and len(set(declared_conditions)) == len(declared_conditions)
    audit.require("EVALUATION_ARGUMENTS", f"{run_id}/declared conditions match the records",
                  valid_conditions and sorted(declared_conditions) == actual_conditions,
                  {"arguments": declared_conditions, "records": actual_conditions})
    probe_flag = arguments.get("content_probe")
    audit.require("EVALUATION_CONTENT_PROBE", f"{run_id}/probe argument is a boolean", isinstance(probe_flag, bool))
    if "content_probe" in manifest:
        audit.equal("EVALUATION_CONTENT_PROBE", f"{run_id}/frozen probe selection",
                    {"manifest": manifest["content_probe"], "arguments": probe_flag})
    for condition in actual_conditions:
        expected_records = [(condition, "continuous", "model")]
        if condition == "iid":
            expected_records.append((condition, "hidden_reset", "model"))
        if probe_flag is True:
            expected_records.append((condition, "continuous", "teacher"))
        actual_records = [_record_key(record) for record in evaluation["records"] if record["condition"] == condition]
        audit.equal("EVALUATION_CONTENT_PROBE", f"{run_id}/{condition}/argument-selected record set",
                    {"arguments": sorted(expected_records), "records": sorted(actual_records)})
    for record in evaluation["records"]:
        label = f"{run_id}/{record['condition']}/{record['control']}/{record['action_policy']}"
        for record_field, argument_field, manifest_field in (
                ("steps", "steps", "eval_steps"), ("dt", "dt", "dt"),
                ("boundary_interval", "reset_interval", "reset_control_period_ticks")):
            values = {"arguments": arguments.get(argument_field), "record": record.get(record_field)}
            audit.equal("EVALUATION_ARGUMENTS", f"{label}/{record_field} agrees with evaluation arguments", values)
            if manifest_field in manifest:
                audit.equal("EVALUATION_BUDGET", f"{label}/{record_field} agrees with frozen study",
                            {"manifest": manifest[manifest_field], **values})


def aggregate(training_reports: Iterable[dict[str, Any]], evaluations: Iterable[dict[str, Any]], *,
              manifest: dict[str, Any] | None = None, seeds: Iterable[int] = (0, 1, 2),
              allow_incomplete: bool = False, include_b: bool | None = None,
              verify_checkpoint_files: bool = False) -> dict[str, Any]:
    """Pure JSON aggregation. Does not import torch or open a training checkpoint."""
    manifest = deepcopy(manifest or {})
    seed_list = list(dict.fromkeys(map(int, seeds)))
    audit = Audit(allow_incomplete)
    reports: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for item in training_reports:
        key = _run_key(item)
        if key in reports:
            audit.problem("DUPLICATE_TRAINING_RUN", _run_id(key),
                          [reports[key].get("_provenance"), item.get("_provenance")])
        else:
            reports[key] = item
        if key[0] not in {"A", "B"} or key[1] not in seed_list or (
                key[0] == "A" and key[2] not in A_VARIANTS) or (
                key[0] == "B" and (key[2] != "online128" or key[3] not in B_MODES)):
            audit.problem("UNEXPECTED_RUN", "run is outside the declared study", _run_id(key))
    evaluation_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evaluations:
        key = _run_key(item, True)
        if key[0] not in {"A", "B"} or key[1] not in seed_list or (
                key[0] == "A" and key[2] not in A_VARIANTS) or (
                key[0] == "B" and (key[2] != "online128" or key[3] not in B_MODES)):
            audit.problem("UNEXPECTED_EVALUATION", "evaluation is outside the declared study", _run_id(key))
            continue
        evaluation_groups[key].append(item)
    include_b = bool(any(key[0] == "B" for key in reports) or
                     any(key[0] == "B" for key in evaluation_groups)) if include_b is None else include_b
    expected = [("A", seed, variant, "") for seed in seed_list for variant in A_VARIANTS]
    if include_b:
        expected += [("B", seed, "online128", mode) for seed in seed_list for mode in B_MODES]
    for key in expected:
        if key not in reports:
            audit.problem("MISSING_TRAINING_RUN", "expected training report is absent", _run_id(key), missing=True)
        if key not in evaluation_groups:
            audit.problem("MISSING_EVALUATION", "expected evaluation is absent", _run_id(key), missing=True)

    by_seed_a: dict[int, list[tuple[tuple, dict]]] = defaultdict(list)
    by_seed_b: dict[int, list[tuple[tuple, dict]]] = defaultdict(list)
    for key, item in reports.items():
        (by_seed_a if key[0] == "A" else by_seed_b)[key[1]].append((key, item))
        run_id = _run_id(key)
        audit.require("WORLD_CONTINUITY", f"{run_id} has no world reset", item.get("world_resets") == 0)
        if manifest:
            for name, digest in manifest.get("source_hashes", {}).items():
                audit.equal("FROZEN_SOURCES", f"{run_id}/{name}",
                            {"manifest": digest, "report": item.get("source_hashes", {}).get(name)})
            for name, expected_value in (("batch_size", manifest.get("batch_size")),
                                         ("n_jobs", manifest.get("n_jobs")),
                                         ("tbptt", manifest.get("tbptt")),
                                         ("lr", manifest.get("lr")),
                                         ("wait_weight", manifest.get("wait_weight")),
                                         ("train_dts", manifest.get("train_dts"))):
                if expected_value is not None:
                    audit.equal("FROZEN_CONFIG", f"{run_id}/{name}",
                                {"manifest": expected_value, "report": item["config"].get(name)})
            expected_updates = manifest.get(f"{key[0]}_updates")
            if expected_updates is not None:
                audit.equal("TRAINING_BUDGET", f"{run_id}/stage updates",
                            {"manifest": expected_updates, "report": item.get("stage_updates")})
                if manifest.get("tbptt") is not None:
                    audit.equal("TRAINING_BUDGET", f"{run_id}/stage ticks",
                                {"expected": expected_updates * manifest["tbptt"],
                                 "report": item.get("stage_ticks")})
                    lifetime_updates = expected_updates + (manifest.get("A_updates", 0) if key[0] == "B" else 0)
                    audit.equal("TRAINING_BUDGET", f"{run_id}/lifetime ticks",
                                {"expected": lifetime_updates * manifest["tbptt"], "report": item.get("ticks")})
                if key[0] == "A":
                    audit.equal("TRAINING_BUDGET", f"{run_id}/optimizer updates",
                                {"expected": expected_updates, "report": item.get("stage_gradient_updates")})
            expected_model = manifest.get("parameters", {}).get(key[2])
            if expected_model:
                audit.equal("PARAMETER_COUNT", f"{run_id}/model parameters",
                            {"manifest": expected_model, "report": item.get("model")})
        reset_period = item["variant"].get("reset_every", 0)
        expected_resets = (item["ticks"] - 1) // reset_period if reset_period else 0
        audit.equal("HIDDEN_RESET_CONTROL", f"{run_id}/only the declared state cuts",
                    {"expected": expected_resets, "report": item.get("hidden_resets")})
        if key[0] == "B":
            origin = item.get("adaptation_origin") or {}
            if "B_durations" in manifest:
                audit.equal("B_DURATION_REGIME", f"{run_id}/frozen adaptation duration range",
                            {"manifest": manifest["B_durations"], "origin": origin.get("duration_range_after")})
            if key[3] == "frozen":
                audit.equal("B_FROZEN_PARAMETERS", f"{run_id}/parameter preservation",
                            {"fork": origin.get("model_hash"), "after": item.get("final_parameter_hash")})
            audit.equal("B_GRADIENT_BUDGET", f"{run_id}/optimizer updates",
                        {"expected": item.get("stage_updates", 0) * GRADIENT_PASSES.get(key[3], -1),
                         "actual": item.get("stage_gradient_updates")})
            audit.require("B_REPLAY_PAST_ONLY", f"{run_id}/finite A-only replay",
                          all(0 <= start < end <= origin.get("tick", -1)
                              for start, end in item.get("replay", {}).get("stored_tick_ranges", []))
                          and item.get("replay", {}).get("stored_blocks", 0) <=
                          item.get("replay", {}).get("capacity_blocks", -1), item.get("replay"))
            a = reports.get(("A", key[1], "online128", ""))
            if a:
                for origin_field, a_field in (("model_hash", "final_parameter_hash"),
                                               ("hidden_hash", "hidden_hash"),
                                               ("world_hash", "world_hash"),
                                               ("action_hash", "action_hash"),
                                               ("data_hash", "data_hash"), ("tick", "ticks"),
                                               ("update", "updates")):
                    audit.equal("B_A_ORIGIN", f"{run_id}/{origin_field} comes from online A",
                                {"A": a.get(a_field), "B_origin": origin.get(origin_field)})
    for seed, items in by_seed_a.items():
        label = f"A/seed_{seed:03d}"
        for field in ("event_hash", "data_seed", "ticks"):
            audit.equal("A_EXOGENOUS_PAIRING", f"{label}/{field}",
                        {_run_id(key): item.get(field) for key, item in items})
        audit.equal("A_INITIALIZATION", f"{label}/same-width learned initialization",
                    {_run_id(key): item.get("initial_parameter_hash") for key, item in items
                     if key[2] in {"c128", "reset128", "online128"}})
        audit.equal("A_TEACHER_ACTIONS", f"{label}/teacher action trace",
                    {_run_id(key): item.get("action_hash") for key, item in items
                     if item["variant"]["policy"] == "teacher"})
    for seed, items in by_seed_b.items():
        label = f"B/seed_{seed:03d}"
        audit.equal("B_FORK_PAIRING", f"{label}/complete adaptation origin",
                    {_run_id(key): item.get("adaptation_origin") for key, item in items})
        for field in ("event_hash", "data_seed", "ticks"):
            audit.equal("B_EXOGENOUS_PAIRING", f"{label}/{field}",
                        {_run_id(key): item.get(field) for key, item in items})
        audit.equal("B_FINITE_REPLAY", f"{label}/same finite A experience reservoir",
                    {_run_id(key): item.get("replay") for key, item in items})

    raw_records, raw_evaluations = [], []
    records_by_run: dict[tuple, dict[tuple, dict]] = {}
    cohort_pairs: dict[tuple, dict[str, Any]] = defaultdict(dict)
    teacher_pairs: dict[tuple, dict[str, Any]] = defaultdict(dict)
    for key, entries in evaluation_groups.items():
        run_id = _run_id(key)
        source_report = reports.get(key)
        if source_report is None:
            audit.problem("MISSING_TRAINING_RUN", "evaluation has no matching report", run_id, missing=True)
        merged: dict[tuple, dict] = {}
        first_checkpoint = entries[0].get("checkpoint_sha256")
        for item in entries:
            _audit_evaluation_spec(audit, run_id, item, manifest)
            audit.equal("EVALUATION_CHECKPOINT", f"{run_id}/same checkpoint across evaluation files",
                        {"reference": first_checkpoint, "current": item.get("checkpoint_sha256")})
            if source_report:
                for report_field, evaluation_field in (("updates", "training_updates"),
                                                         ("ticks", "training_ticks"),
                                                         ("config", "training_config"),
                                                         ("variant", "training_variant"), ("model", "model"),
                                                         ("source_hashes", "checkpoint_source_hashes")):
                    report_value, evaluation_value = source_report.get(report_field), item.get(evaluation_field)
                    # from_checkpoint(device=...) records the selected compute
                    # device in its config. All other saved settings must match.
                    if report_field == "config":
                        report_value = {name: value for name, value in report_value.items() if name != "device"}
                        evaluation_value = {name: value for name, value in evaluation_value.items() if name != "device"}
                    audit.equal("REPORT_EVALUATION_LINK", f"{run_id}/{report_field}",
                                {"report": report_value, "evaluation": evaluation_value})
            if manifest:
                for name, digest in manifest.get("source_hashes", {}).items():
                    if name in {"world.py", "model.py", "run.py"}:
                        audit.equal("FROZEN_EVALUATOR_SOURCES", f"{run_id}/{name}",
                                    {"manifest": digest, "evaluation": item.get("source_hashes", {}).get(name)})
                if manifest.get("evaluator_sha256"):
                    audit.equal("FROZEN_EVALUATOR_SOURCES", f"{run_id}/evaluate.py",
                                {"manifest": manifest["evaluator_sha256"],
                                 "evaluation": item.get("source_hashes", {}).get("evaluate.py")})
            if verify_checkpoint_files:
                checkpoint = Path(item["checkpoint"])
                audit.require("CHECKPOINT_FILE", f"{run_id}/checkpoint exists", checkpoint.is_file(), str(checkpoint))
                if checkpoint.is_file():
                    audit.equal("CHECKPOINT_FILE", f"{run_id}/checkpoint SHA-256",
                                {"evaluated": item["checkpoint_sha256"], "current": file_sha256(checkpoint)})
            raw_evaluations.append({name: deepcopy(value) for name, value in item.items() if name != "records"})
            for record in item["records"]:
                record_key = _record_key(record)
                if record_key in merged:
                    audit.problem("DUPLICATE_EVALUATION_RECORD", f"{run_id}/{record_key}", item.get("_provenance"))
                    continue
                merged[record_key] = record
                if record["action_policy"] not in {"model", "teacher"}:
                    audit.problem("ACTION_POLICY", f"{run_id}/unknown action policy", record["action_policy"])
                if record["action_policy"] == "teacher" and "content_probe" not in record:
                    audit.problem("ACTION_POLICY", f"{run_id}/teacher record has no diagnostic readout probe")
                if record["action_policy"] == "model" and "content_probe" in record:
                    audit.problem("ACTION_POLICY", f"{run_id}/teacher diagnostic mislabelled as autonomous")
                audit.require("FROZEN_EVALUATION", f"{run_id}/{record_key}/no learning or world resets",
                              record.get("parameter_updates") == 0 and record.get("world_resets") == 0)
                if source_report:
                    audit.equal("EVALUATION_PARAMETERS", f"{run_id}/{record_key}/same frozen parameters",
                                {"trained": source_report.get("final_parameter_hash"),
                                 "evaluation": record.get("parameter_hash")})
                    audit.equal("EVALUATION_WORLD_FORK", f"{run_id}/{record_key}/complete saved world",
                                {"trained": source_report.get("world_hash"),
                                 "evaluation": record.get("start_world_state_hash")})
                _audit_cohorts(audit, run_id, record)
                # Exact pairing is within a stage, seed and condition. Different
                # duration regimes and A-vs-B evaluation forks need not have the
                # same request schedule. Inherited/prior-contract cohorts are
                # intentionally excluded from comparisons across controllers.
                paired_key = key[0], key[1], record["condition"]
                paired_id = f"{run_id}/{record['control']}/{record['action_policy']}"
                for cohort in ("primary", "boundary_crossing"):
                    cohort_pairs[(*paired_key, cohort, "fingerprint")][paired_id] = record[cohort]["request_fingerprint"]
                    cohort_pairs[(*paired_key, cohort, "matured")][paired_id] = record[cohort]["matured"]
                    if "per_actor" in record[cohort]:
                        cohort_pairs[(*paired_key, cohort, "per_actor_matured")][paired_id] = record[cohort]["per_actor"]["matured"]
                cohort_pairs[(*paired_key, "event_hash")][paired_id] = record.get("end_event_hash")
                if record["action_policy"] == "teacher":
                    # Inherited active/started state can differ between an
                    # own-action controller and teacher training. Only identical
                    # complete starting worlds imply identical teacher actions.
                    teacher_pairs[(*paired_key, record.get("start_world_state_hash"))][paired_id] = record.get("action_hash")
                    probe = record["content_probe"]
                    expected_accuracy = probe["correct"] / probe["count"] if probe["count"] else None
                    audit.require("PROBE_COUNTS", f"{paired_id}/conditional readout denominator",
                                  0 <= probe["correct"] <= probe["count"] and
                                  (probe["accuracy"] is None if expected_accuracy is None else
                                   math.isclose(probe["accuracy"], expected_accuracy, rel_tol=0, abs_tol=1e-12)))
                raw_records.append({"run_id": run_id, "stage": key[0], "seed": key[1],
                                    "variant": key[2], "mode": key[3] or None,
                                    "checkpoint": item["checkpoint"],
                                    "checkpoint_sha256": item["checkpoint_sha256"],
                                    "evaluation_source": deepcopy(item.get("_provenance")),
                                    "record": deepcopy(record)})
        records_by_run[key] = merged
        conditions = manifest.get("eval_conditions", DEFAULT_CONDITIONS) if key[0] == "A" else manifest.get(
            "B_eval_conditions", ("iid", "long_wait", "longer"))
        for condition in conditions:
            expected_record_keys = [(condition, "continuous", "model")]
            if condition == "iid":
                expected_record_keys.append((condition, "hidden_reset", "model"))
            if manifest.get("content_probe", True):
                expected_record_keys.append((condition, "continuous", "teacher"))
            for record_key in expected_record_keys:
                if record_key not in merged:
                    audit.problem("MISSING_EVALUATION_RECORD", f"{run_id}/{record_key}", missing=True)
    for key, values in cohort_pairs.items():
        audit.equal("COHORT_PAIRING", "/".join(map(str, key)), values)
    for key, values in teacher_pairs.items():
        audit.equal("EVALUATION_TEACHER_ACTIONS", "/".join(map(str, key)), values)

    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    probes: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_records:
        record = raw["record"]
        group = raw["stage"], raw["mode"] or raw["variant"], record["condition"], record["control"]
        if record["action_policy"] == "model":
            grouped[group].append({"seed": raw["seed"], "run_id": raw["run_id"],
                                   "metrics": _metric_values(record), "counts": _raw_counts(record),
                                   "primary_request_fingerprint": record["primary"]["request_fingerprint"],
                                   "boundary_request_fingerprint": record["boundary_crossing"]["request_fingerprint"]})
        elif record["action_policy"] == "teacher":
            probes[group].append({"seed": raw["seed"], "run_id": raw["run_id"],
                                  **deepcopy(record["content_probe"])})
    model_summary = []
    for group, rows in sorted(grouped.items()):
        stage, name, condition, control = group
        model_summary.append({"stage": stage, "name": name, "condition": condition, "control": control,
                              "action_policy": "model", "seeds": sorted(row["seed"] for row in rows),
                              "metrics": {metric: summarize({"seed": row["seed"], "value": row["metrics"][metric]}
                                                            for row in rows) for metric in MODEL_METRICS},
                              "per_seed_counts": [{key: value for key, value in row.items() if key != "metrics"}
                                                  for row in rows]})
    probe_summary = []
    for group, rows in sorted(probes.items()):
        stage, name, condition, control = group
        probe_summary.append({"stage": stage, "name": name, "condition": condition, "control": control,
                              "action_policy": "teacher", "metric": "conditional_finish_readout",
                              "accuracy": summarize({"seed": row["seed"], "value": row["accuracy"]} for row in rows),
                              "per_seed_counts": rows})

    parameter_table = []
    for name in A_VARIANTS:
        selected = [(key, item) for key, item in reports.items() if key[0] == "A" and key[2] == name]
        if not selected:
            continue
        audit.equal("PARAMETERS_ACROSS_SEEDS", name, {_run_id(key): item["model"] for key, item in selected})
        representative = selected[0][1]
        parameter_table.append({"variant": name, "policy": representative["variant"]["policy"],
                                "hidden_reset_period": representative["variant"]["reset_every"],
                                **deepcopy(representative["model"]),
                                "seeds": sorted(key[1] for key, _ in selected)})

    def find_record(stage: str, seed: int, name: str, condition: str, control: str = "continuous") -> dict | None:
        run = (stage, seed, name if stage == "A" else "online128", "" if stage == "A" else name)
        return records_by_run.get(run, {}).get((condition, control, "model"))

    paired_comparisons = []

    def add_comparison(label: str, left: tuple, right: tuple, conditions: Iterable[str]) -> None:
        for condition in conditions:
            for metric in MODEL_METRICS:
                values = []
                for seed in seed_list:
                    a = find_record(left[0], seed, left[1], condition, left[2])
                    b = find_record(right[0], seed, right[1], condition, right[2])
                    if a is None or b is None:
                        continue
                    av, bv = _metric_values(a)[metric], _metric_values(b)[metric]
                    values.append({"seed": seed, "left": av, "right": bv,
                                   "value": None if av is None or bv is None else bv - av})
                if values:
                    paired_comparisons.append({"comparison": label, "condition": condition, "metric": metric,
                                               "direction": "right minus left", "left": list(left), "right": list(right),
                                               "difference": summarize(values)})
    add_comparison("evaluation hidden reset of c128", ("A", "c128", "continuous"),
                   ("A", "c128", "hidden_reset"), ("iid",))
    add_comparison("training reset128 versus continuous c128", ("A", "c128", "continuous"),
                   ("A", "reset128", "continuous"), manifest.get("eval_conditions", DEFAULT_CONDITIONS))
    if include_b:
        add_comparison("replay versus current_repeat (two gradient passes)", ("B", "current_repeat", "continuous"),
                       ("B", "replay", "continuous"), manifest.get("B_eval_conditions", ("iid", "long_wait", "longer")))
        for mode in B_MODES[1:]:
            add_comparison(f"{mode} versus frozen continuation", ("B", "frozen", "continuous"),
                           ("B", mode, "continuous"), ("iid", "long_wait", "longer"))

    adaptation_tradeoff = []
    for mode in B_MODES if include_b else ():
        rows = []
        for seed in seed_list:
            before_a = find_record("A", seed, "online128", "iid")
            before_b = find_record("A", seed, "online128", "long_wait")
            after_a = find_record("B", seed, mode, "iid")
            after_b = find_record("B", seed, mode, "long_wait")
            if not all(record is not None for record in (before_a, before_b, after_a, after_b)):
                continue
            values = {name: record["primary"]["clean_completion_rate"] for name, record in
                      (("before_A", before_a), ("before_B", before_b), ("after_A", after_a), ("after_B", after_b))}
            values["retention_change"] = None if values["before_A"] is None or values["after_A"] is None else (
                values["after_A"] - values["before_A"])
            values["adaptation_gain"] = None if values["before_B"] is None or values["after_B"] is None else (
                values["after_B"] - values["before_B"])
            rows.append({"seed": seed, **values})
        if rows:
            adaptation_tradeoff.append({"mode": mode, "gradient_passes_per_live_chunk": GRADIENT_PASSES[mode],
                                        "replay_extra_no_grad_burnin_ticks": manifest.get("replay_burnin_ticks", 64)
                                        if mode == "replay" else 0,
                                        "metrics": {name: summarize({"seed": row["seed"], "value": row[name]} for row in rows)
                                                    for name in ("before_A", "before_B", "after_A", "after_B",
                                                                 "retention_change", "adaptation_gain")},
                                        "per_seed": rows})

    return {"format": FORMAT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": seed_list, "includes_B": include_b, "manifest": manifest,
            "audit": audit.result(), "model_summary": model_summary,
            "conditional_readout_summary": probe_summary, "parameters": parameter_table,
            "paired_comparisons": paired_comparisons, "adaptation_tradeoff": adaptation_tradeoff,
            "raw_training_reports": [_compact_training(item) for _, item in sorted(reports.items())],
            "raw_evaluation_metadata": raw_evaluations, "raw_seed_records": raw_records,
            "interpretation": {
                "statistics": "unweighted mean and sample SD of seed-level rates; actors and requests are not independent seeds",
                "primary": "model actions only; eligible post-fork evaluation-regime requests with matured deadlines",
                "boundary": "primary requests crossing absolute-tick cuts under issued_at < cut_time < issued_at + duration",
                "finish_precision": "accepted FINISH / attempted FINISH after the fork, including inherited and censored work",
                "readout_probe": "teacher receipts and timing; conditional argmax over existing FINISH logits; separate from autonomous success",
                "B_pre_post": "functional behavior of inherited weights plus numerical state on different future request cohorts; A pre-B long_wait shares an exogenous prefix with subsequent B training and is not an independent held-out knowledge baseline",
                "B_primary_comparison": "paired learned-arm versus frozen-B endpoint contrasts control the passage of experience; results combine parameter and live-state effects, not pure knowledge changes",
                "B_compute": "current_repeat and replay have two gradient passes and equal supervised ticks; replay adds no-grad burn-in and is not FLOP-matched",
                "training_counters": "lifetime counters persist across A and B; evaluation primary cohort counts determine task rates",
                "runtime": "concurrent workers share a GPU; elapsed seconds are not isolated throughput benchmarks",
                "undefined_rates": "zero denominators remain null and are excluded from the numeric seed mean; n and n_undefined are explicit",
                "data_hash": "not required equal: reset metadata and model-action feedback change this digest",
                "scope": "no claims of unlimited temporal credit, autonomous reward-only learning or lifelong generalization"}}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _stat_text(stat: dict[str, Any], percentage: bool = True) -> str:
    scale = 100 if percentage else 1
    if stat["mean"] is None:
        return "undefined"
    mean = stat["mean"] * scale
    sd = stat["sample_sd"]
    return f"{mean:.2f} ± {sd * scale:.2f}" if sd is not None else f"{mean:.2f} (n={stat['n']})"


def write_tables(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for item in report["model_summary"]:
        for name, stat in item["metrics"].items():
            model_rows.append({key: item[key] for key in ("stage", "name", "condition", "control", "action_policy")} |
                              {"metric": name} | {key: value for key, value in stat.items() if key != "per_seed"})
    _write_csv(output / "model_summary.csv", model_rows)
    _write_csv(output / "parameters.csv", report["parameters"])
    _write_csv(output / "conditional_readout.csv", [
        {key: item[key] for key in ("stage", "name", "condition", "control", "action_policy")} |
        {key: value for key, value in item["accuracy"].items() if key != "per_seed"}
        for item in report["conditional_readout_summary"]])
    _write_csv(output / "paired_comparisons.csv", [
        {key: item[key] for key in ("comparison", "condition", "metric", "direction", "left", "right")} |
        {key: value for key, value in item["difference"].items() if key != "per_seed"}
        for item in report["paired_comparisons"]])
    _write_csv(output / "adaptation_per_seed.csv", [
        {"mode": item["mode"], "gradient_passes_per_live_chunk": item["gradient_passes_per_live_chunk"], **row}
        for item in report["adaptation_tradeoff"] for row in item["per_seed"]])
    raw_rows = []
    for raw in report["raw_seed_records"]:
        record = raw["record"]
        raw_rows.append({key: raw[key] for key in ("run_id", "stage", "seed", "variant", "mode", "checkpoint_sha256")} |
                        {key: record[key] for key in ("condition", "control", "action_policy")} |
                        _metric_values(record) | {
                            "primary_matured": record["primary"]["matured"],
                            "primary_clean_completed": record["primary"]["clean_completed"],
                            "boundary_matured": record["boundary_crossing"]["matured"],
                            "boundary_clean_completed": record["boundary_crossing"]["clean_completed"],
                            "finish_attempted": record["post_fork_counters"]["finish_attempted"],
                            "completed_all": record["post_fork_counters"]["completed_all"],
                            "probe_count": record.get("content_probe", {}).get("count"),
                            "probe_correct": record.get("content_probe", {}).get("correct"),
                            "probe_accuracy": record.get("content_probe", {}).get("accuracy"),
                            "primary_request_fingerprint": record["primary"]["request_fingerprint"],
                            "boundary_request_fingerprint": record["boundary_crossing"]["request_fingerprint"],
                            "end_event_hash": record["end_event_hash"]})
    _write_csv(output / "seed_records.csv", raw_rows)
    lines = ["# Continuity study aggregation", "", f"Audit: **{report['audit']['status']}**; "
             f"{report['audit']['error_count']} errors, {report['audit']['warning_count']} warnings.", "",
             "Cells show percentages, mean ± sample SD across seeds. Autonomous scores use model actions. "
             "Conditional teacher readout appears separately. Raw counts, fingerprints and input SHA-256 values are in summary.json.", "",
             "| Stage | Controller | Condition | Control | Clean completion | Boundary clean | FINISH precision | n |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for row in report["model_summary"]:
        metrics = row["metrics"]
        lines.append(f"| {row['stage']} | {row['name']} | {row['condition']} | {row['control']} | "
                     f"{_stat_text(metrics['primary_clean_completion'])} | {_stat_text(metrics['boundary_clean_completion'])} | "
                     f"{_stat_text(metrics['finish_precision'])} | {metrics['primary_clean_completion']['n']} |")
    lines += ["", "## Parameters", "", "| Controller | State scalars | Total parameters | Trainable parameters | Training actions |",
              "|---|---:|---:|---:|---|"]
    for row in report["parameters"]:
        lines.append(f"| {row['variant']} | {row['state_scalars']} | {row['total_parameters']:,} | "
                     f"{row['trainable_parameters']:,} | {row['policy']} |")
    if report["adaptation_tradeoff"]:
        lines += ["", "## Adaptation", "", "| Arm | Gradients / live chunk | A before | A after | A change | B before | B after | B gain |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["adaptation_tradeoff"]:
            m = row["metrics"]
            lines.append(f"| {row['mode']} | {row['gradient_passes_per_live_chunk']} | " + " | ".join(
                _stat_text(m[name]) for name in ("before_A", "after_A", "retention_change", "before_B", "after_B", "adaptation_gain")) + " |")
    lines += ["", "## Conditional FINISH readout", "",
              "Teacher timing and receipts; argmax over the eight existing FINISH logits. These are diagnostic readout rates.", "",
              "| Stage | Controller | Condition | Readout accuracy | n |", "|---|---|---|---:|---:|"]
    for row in report["conditional_readout_summary"]:
        lines.append(f"| {row['stage']} | {row['name']} | {row['condition']} | {_stat_text(row['accuracy'])} | {row['accuracy']['n']} |")
    lines += ["", "Current-repeat and replay each use two gradient passes. Replay adds a 64-tick no-grad burn-in; forward FLOPs differ. "
              "A/B pre/post differences describe functional behavior of weights plus numerical state on different future requests. "
              "The A pre-B long-wait evaluation shares an exogenous prefix with B training. Paired contrasts against the frozen-B endpoint "
              "are the primary adaptation comparison; these also combine parameter and live-state effects. "
              "Training counters are lifetime totals; evaluation cohorts determine these rates.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(report: dict[str, Any], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    synthetic = bool(report.get("synthetic"))
    prefix = "SYNTHETIC CHECK — " if synthetic else ""
    rows = {(row["stage"], row["name"], row["condition"], row["control"]): row for row in report["model_summary"]}
    probes = {(row["stage"], row["name"], row["condition"]): row for row in report["conditional_readout_summary"]}
    parameters = {row["variant"]: row for row in report["parameters"]}
    files = []

    def save(fig, name: str) -> None:
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=175, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)

    candidates = [("c128", "continuous", "c128\ncontinuous"),
                  ("c128", "hidden_reset", "c128\neval reset / 512"),
                  ("reset128", "continuous", "reset128\ntrain reset / 512"),
                  ("online128", "continuous", "online128\ncontinuous")]
    selected = [(name, control, label, rows[("A", name, "iid", control)])
                for name, control, label in candidates if ("A", name, "iid", control) in rows]
    if selected:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
        colors = ["#287b8e", "#d77b35", "#a95d5d", "#6676a8"]
        for axis, metric, title in zip(axes, ("primary_clean_completion", "boundary_clean_completion"),
                                       ("All primary requests", "Requests spanning a 512-tick boundary")):
            for index, (_, _, _, row) in enumerate(selected):
                stat = row["metrics"][metric]
                if stat["mean"] is None:
                    continue
                axis.bar(index, stat["mean"] * 100, width=0.6, color=colors[index], alpha=0.65,
                         yerr=0 if stat["sample_sd"] is None else stat["sample_sd"] * 100,
                         capsize=4, error_kw={"linewidth": 1.4})
                valid = [item for item in stat["per_seed"] if item["value"] is not None]
                offsets = np.linspace(-0.12, 0.12, len(valid))
                axis.scatter(index + offsets, [item["value"] * 100 for item in valid],
                             s=24, color="#172d3b", zorder=4)
            axis.set_xticks(range(len(selected)), [item[2] for item in selected])
            axis.set_ylim(0, 105)
            axis.yaxis.set_major_formatter(ticker.PercentFormatter())
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Clean completion of matured requests")
        fig.suptitle(prefix + "State continuity: IID autonomous control", fontsize=13)
        fig.supxlabel("Dots: seeds. Bars: mean ± sample SD. Post-fork evaluation-regime requests only.", fontsize=9)
        save(fig, "continuity_boundary")

    width_names = [name for name in ("c64", "c128", "c256") if name in parameters]
    if width_names:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
        conditions = [("iid", "IID", "#287b8e"), ("long_wait", "Duration 4–8", "#d77b35"),
                      ("longer", "Duration 8–16", "#8d6296")]
        for condition, label, color in conditions:
            for axis, probe in zip(axes, (False, True)):
                points = []
                for name in width_names:
                    item = probes.get(("A", name, condition)) if probe else rows.get(("A", name, condition, "continuous"))
                    if item:
                        stat = item["accuracy"] if probe else item["metrics"]["primary_clean_completion"]
                        if stat["mean"] is not None:
                            points.append((parameters[name]["total_parameters"], stat))
                if points:
                    axis.errorbar([x for x, _ in points], [stat["mean"] * 100 for _, stat in points],
                                  yerr=[(stat["sample_sd"] or 0) * 100 for _, stat in points],
                                  marker="o", capsize=4, color=color, linewidth=1.8, label=label)
                    for x, stat in points:
                        valid = [item for item in stat["per_seed"] if item["value"] is not None]
                        axis.scatter([x * (1 + offset) for offset in np.linspace(-0.035, 0.035, len(valid))],
                                     [item["value"] * 100 for item in valid], color=color, s=16, alpha=0.6)
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xticks([parameters[name]["total_parameters"] for name in width_names],
                           [f"{parameters[name]['state_scalars']} states\n{parameters[name]['total_parameters']:,} params"
                            for name in width_names])
            axis.xaxis.set_minor_locator(ticker.NullLocator())
            axis.set_ylim(0, 105)
            axis.yaxis.set_major_formatter(ticker.PercentFormatter())
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_title("Autonomous clean completion")
        axes[0].set_ylabel("Rate")
        axes[1].set_title("Conditional FINISH readout (teacher)")
        axes[1].axhline(12.5, linestyle=":", color="#888888", linewidth=1, label="1/8 FINISH classes")
        axes[0].legend(loc="best", fontsize=9)
        axes[1].legend(loc="best", fontsize=9)
        fig.suptitle(prefix + "Parameter scaling under the same training budget", fontsize=13)
        fig.supxlabel("Mean ± sample SD across seeds. Teacher readout excludes WAIT/BEGIN competition.", fontsize=9)
        save(fig, "parameter_scaling")

    tradeoffs = report["adaptation_tradeoff"]
    if tradeoffs:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
        colors = dict(frozen="#7a8088", current="#287b8e", current_repeat="#d77b35", replay="#8d6296")
        labels = dict(frozen="Frozen (0 updates)", current="Current (1 update)",
                      current_repeat="Current repeat (2 updates)", replay="Current + past (2 updates)")
        for row in tradeoffs:
            mode, metrics = row["mode"], row["metrics"]
            for axis, xname, yname in ((axes[0], "after_A", "after_B"),
                                       (axes[1], "retention_change", "adaptation_gain")):
                x, y = metrics[xname], metrics[yname]
                if x["mean"] is None or y["mean"] is None:
                    continue
                axis.errorbar(x["mean"] * 100, y["mean"] * 100,
                              xerr=(x["sample_sd"] or 0) * 100, yerr=(y["sample_sd"] or 0) * 100,
                              color=colors[mode], marker="o", markersize=8, capsize=4,
                              linewidth=1.5, label=labels[mode], zorder=4)
                valid = [item for item in row["per_seed"] if item[xname] is not None and item[yname] is not None]
                axis.scatter([item[xname] * 100 for item in valid], [item[yname] * 100 for item in valid],
                             s=22, color=colors[mode], alpha=0.55, zorder=3)
        before = tradeoffs[0]["metrics"]
        if before["before_A"]["mean"] is not None and before["before_B"]["mean"] is not None:
            axes[0].scatter(before["before_A"]["mean"] * 100, before["before_B"]["mean"] * 100,
                            marker="*", s=140, color="#172d3b", label="Before adaptation", zorder=5)
        axes[0].set(xlim=(-2, 102), ylim=(-2, 102), xlabel="Old A: IID clean completion (%)",
                    ylabel="New B: duration 4–8 clean completion (%)", title="After continued experience")
        axes[0].legend(loc="best", fontsize=8)
        axes[1].axhline(0, color="#999999", linewidth=1)
        axes[1].axvline(0, color="#999999", linewidth=1)
        axes[1].set(xlabel="Old-A change from before adaptation (pp)",
                    ylabel="New-B gain from before adaptation (pp)", title="Pre/post functional behavior")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.suptitle(prefix + "Adaptation and retention", fontsize=13)
        fig.supxlabel("Mean ± sample SD across seeds. Changes combine weights and live state; frozen B is the primary comparison.\n"
                      "Replay adds 64 no-grad burn-in ticks; forward FLOPs differ.", fontsize=9)
        save(fig, "adaptation_tradeoff")
    return files


def write_outputs(report: dict[str, Any], output: Path, plots: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report["reporter"] = {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))}
    write_tables(report, output)
    report["figures"] = make_plots(report, output / "figures") if plots else []
    atomic_json(output / "audit.json", report["audit"])
    atomic_json(output / "summary.json", report)


def _synthetic_fixture() -> tuple[list[dict], list[dict], dict]:
    """Fictional outcomes for deterministic reporter tests and layout checks."""
    h = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    parameters = {
        "c64": (64, 13100, 13100, "learned"), "c128": (128, 34348, 34348, "learned"),
        "c256": (256, 101420, 101420, "learned"), "reset128": (128, 34348, 34348, "learned"),
        "reservoir128": (128, 34348, 17964, "reservoir"), "gru128": (128, 117420, 117420, "gru"),
        "online128": (128, 34348, 34348, "learned")}
    manifest = {"version": "synthetic-fixture-not-experimental-evidence", "seeds": [0, 1, 2],
                "A_updates": 2048, "B_updates": 1024, "batch_size": 4, "n_jobs": 4, "tbptt": 64,
                "lr": 0.001, "wait_weight": 0.1, "train_dts": [0.25],
                "eval_steps": 2048, "dt": 0.25, "reset_control_period_ticks": 512,
                "content_probe": True, "B_durations": [4, 8],
                "eval_conditions": list(DEFAULT_CONDITIONS), "B_eval_conditions": ["iid", "long_wait", "longer"],
                "source_hashes": {name: h(name) for name in CORE_SOURCES}, "evaluator_sha256": h("evaluate.py")}
    model_reports = {name: dict(kind=kind, input_size=96, n_jobs=4, n_actions=11, state_scalars=width,
                                total_parameters=total, trainable_parameters=trainable,
                                max_solver_step=0.25, time_constant=None if kind == "gru" else 1.0)
                     for name, (width, total, trainable, kind) in parameters.items()}
    manifest["parameters"] = deepcopy(model_reports)
    reports, evaluations = [], []

    def cohort(seed: int, condition: str, name: str, rate: float, matured: int) -> dict:
        completed = int(rate * matured)
        split = lambda total: [total // 4 + int(index < total % 4) for index in range(4)]
        return dict(matured=matured, completed=completed, clean_completed=completed,
                    completion_rate=completed / matured if matured else None,
                    clean_completion_rate=completed / matured if matured else None,
                    request_fingerprint=h((seed, condition, name)),
                    per_actor={"matured": split(matured), "completed": split(completed),
                               "clean_completed": split(completed)})

    def evaluation(report: dict, base_rate: float) -> dict:
        seed, stage = report["config"]["seed"], report["stage"]
        name = report["adaptation_mode"] or report["variant"]["name"]
        records = []
        for condition in manifest["eval_conditions"] if stage == "A" else manifest["B_eval_conditions"]:
            for control, policy in [("continuous", "model")] + (
                    [("hidden_reset", "model")] if condition == "iid" else []) + [("continuous", "teacher")]:
                rate = min(0.99, max(0.02, base_rate - {"iid": 0, "long_wait": 0.25, "longer": 0.45,
                                                      "sparse": 0.08, "conflict": 0.12}[condition] + (seed - 1) * 0.025))
                if control == "hidden_reset":
                    rate -= 0.04
                if policy == "teacher":
                    rate = 1.0
                primary = cohort(seed, condition, "primary", rate, 1000 + seed * 100)
                boundary = cohort(seed, condition, "boundary", max(0, rate - (0.25 if control == "hidden_reset" else 0.05)),
                                  100 + seed * 10)
                inherited = cohort(seed, condition, "inherited", 0.5, 20)
                inherited.update(initial_eligible_pending=22, initial_active=24, initial_started=12)
                carryover = cohort(seed, condition, "carryover", rate, 10)
                count = primary["completed"]
                attempted = max(500, count + 50)
                record = dict(format="continuity-evaluation-v1", condition=condition, control=control,
                              action_policy=policy, world_resets=0, parameter_updates=0,
                              steps=2048, dt=0.25, batch_size=4, n_jobs=4, active_jobs=4,
                              actor_ids=[0, 1, 2, 3], fork_tick=report["ticks"] - 1,
                              end_tick=report["ticks"] + 2047, boundary_interval=512,
                              primary=primary, boundary_crossing=boundary, inherited=inherited,
                              carryover_contract=carryover, primary_eligible_issued=primary["matured"] + 5,
                              carryover_contract_eligible_issued=12,
                              censored={"primary": dict(count=5, completed=2, clean_completed=2),
                                        "inherited": dict(count=2, completed=1, clean_completed=1),
                                        "carryover_contract": dict(count=2, completed=1, clean_completed=1)},
                              post_fork_counters=dict(completed_all=count, finish_attempted=attempted,
                                                      false_action=20, eligible=primary["matured"] + 17),
                              post_fork_finish_precision=count / attempted,
                              post_fork_false_actions_per_1000_opportunities=20 * 1000 / (2048 * 4 * 4),
                              end_event_hash=h((stage, seed, condition, "future-events")),
                              start_world_state_hash=report["world_hash"],
                              parameter_hash=report["final_parameter_hash"],
                              action_hash=h((stage, seed, condition, "teacher" if policy == "teacher" else name, control)))
                if policy == "teacher":
                    correct = int((min(0.98, base_rate + 0.1) + (seed - 1) * 0.01) * 1000)
                    record["content_probe"] = dict(count=1000, correct=correct, accuracy=correct / 1000,
                                                   fingerprint=h((stage, seed, condition, name, "probe")))
                records.append(record)
        return dict(format="continuity-evaluation-v1", checkpoint=f"synthetic/{stage}/{seed}/{name}.pt",
                    checkpoint_sha256=h((stage, seed, name, "checkpoint")), checkpoint_state_hash=h((stage, seed, name, "state")),
                    training_variant=deepcopy(report["variant"]), training_config=deepcopy(report["config"]),
                    training_stage=stage, adaptation_mode=report["adaptation_mode"], training_updates=report["updates"],
                    training_ticks=report["ticks"], model=deepcopy(report["model"]), records=records,
                    evaluation_args=dict(conditions=deepcopy(manifest["eval_conditions"] if stage == "A" else manifest["B_eval_conditions"]),
                                         steps=2048, dt=0.25, reset_interval=512, content_probe=True, device="synthetic"),
                    checkpoint_source_hashes=deepcopy(manifest["source_hashes"]),
                    source_hashes=manifest["source_hashes"] | {"evaluate.py": manifest["evaluator_sha256"]},
                    _provenance=dict(path="synthetic-evaluation.json", sha256=h("synthetic-evaluations")))

    for seed in range(3):
        a_online = None
        for index, name in enumerate(A_VARIANTS):
            width, _, _, kind = parameters[name]
            policy = "online" if name == "online128" else "teacher"
            report = dict(format="continuity-trainer-v1", variant=dict(name=name, kind=kind, width=width, policy=policy,
                                                                       reset_every=512 if name == "reset128" else 0),
                          config=dict(seed=seed, batch_size=4, tbptt=64, lr=0.001, wait_weight=0.1,
                                      n_jobs=4, train_dts=[0.25]), model=deepcopy(model_reports[name]),
                          stage="A", adaptation_mode=None, updates=2048, stage_updates=2048, ticks=131072,
                          stage_ticks=131072, gradient_updates=2048, stage_gradient_updates=2048, world_resets=0,
                          hidden_resets=255 if name == "reset128" else 0,
                          initial_parameter_hash=h((seed, width, "initial")),
                          final_parameter_hash=h((seed, name, "trained")), hidden_hash=h((seed, name, "hidden")),
                          world_hash=h((seed, policy, "world")), event_hash=h((seed, "A-events")),
                          data_seed=seed + 1000, action_hash=h((seed, policy, "actions")), data_hash=h((seed, name, "data")),
                          adaptation_origin=None, source_hashes=deepcopy(manifest["source_hashes"]),
                          replay=dict(capacity_blocks=32, stored_blocks=32, complete_blocks_seen=1024,
                                      stored_tick_ranges=[[index * 128, (index + 1) * 128] for index in range(32)]),
                          _provenance=dict(path=f"synthetic/A/{seed}/{name}/report.json", sha256=h((seed, name, "report"))))
            reports.append(report)
            base = dict(c64=0.60, c128=0.77, c256=0.86, reset128=0.69, reservoir128=0.55, gru128=0.68, online128=0.80)[name]
            evaluations.append(evaluation(report, base))
            if name == "online128":
                a_online = report
        for mode in B_MODES:
            report = deepcopy(a_online)
            report.update(stage="B", adaptation_mode=mode, updates=3072, stage_updates=1024,
                          ticks=196608, stage_ticks=65536,
                          gradient_updates=2048 + GRADIENT_PASSES[mode] * 1024,
                          stage_gradient_updates=GRADIENT_PASSES[mode] * 1024,
                          final_parameter_hash=a_online["final_parameter_hash"] if mode == "frozen" else h((seed, mode, "adapted")),
                          event_hash=h((seed, "B-events")), world_hash=h((seed, mode, "B-world")),
                          adaptation_origin=dict(model_hash=a_online["final_parameter_hash"], optimizer_hash=h((seed, "Adam")),
                                                 hidden_hash=a_online["hidden_hash"], world_hash=a_online["world_hash"],
                                                 replay_hash=h((seed, "replay")), action_hash=a_online["action_hash"],
                                                 data_hash=a_online["data_hash"], tick=131072, update=2048,
                                                 duration_range_before=[1, 4], duration_range_after=[4, 8]))
            reports.append(report)
            evaluations.append(evaluation(report, dict(frozen=0.78, current=0.82, current_repeat=0.85, replay=0.88)[mode]))
    return reports, evaluations, manifest


def self_test(output: Path, plots: bool = True) -> dict[str, Any]:
    values = summarize([dict(seed=0, value=1), dict(seed=1, value=2), dict(seed=2, value=3), dict(seed=3, value=None)])
    assert values["mean"] == 2 and values["sample_sd"] == 1 and values["n"] == 3 and values["n_undefined"] == 1
    assert summarize([dict(seed=0, value=0)])["sample_sd"] is None
    assert summarize([dict(seed=0, value=None)])["mean"] is None
    training, evaluations, manifest = _synthetic_fixture()
    report = aggregate(training, evaluations, manifest=manifest)
    assert report["audit"]["status"] == "passed", report["audit"]["issues"]
    assert len(report["model_summary"]) == 7 * 6 + 4 * 4
    assert len(report["conditional_readout_summary"]) == 7 * 5 + 4 * 3
    assert len(report["adaptation_tradeoff"]) == 4
    row = next(item for item in report["model_summary"] if (item["stage"], item["name"], item["condition"], item["control"])
               == ("A", "c128", "iid", "continuous"))
    assert 0.7 < row["metrics"]["primary_clean_completion"]["mean"] < 0.85
    different_inheritance = deepcopy(evaluations)
    for evaluation in different_inheritance:
        evaluation["training_config"]["device"] = "cpu-evaluation-device"
        if evaluation["training_variant"]["name"] == "online128":
            for record in evaluation["records"]:
                if record["action_policy"] == "teacher":
                    record["action_hash"] = hashlib.sha256(
                        (evaluation["checkpoint_sha256"] + record["condition"]).encode()).hexdigest()
    assert aggregate(training, different_inheritance, manifest=manifest)["audit"]["status"] == "passed"
    cases = []
    bad = deepcopy(training)
    bad[0]["event_hash"] = "wrong"
    cases.append(("A_EXOGENOUS_PAIRING", "different A exogenous streams", bad, evaluations))
    bad = deepcopy(training)
    next(item for item in bad if item["stage"] == "B" and item["adaptation_mode"] == "frozen")["final_parameter_hash"] = "wrong"
    cases.append(("B_FROZEN_PARAMETERS", "frozen B weights changed", bad, evaluations))
    bad = deepcopy(training)
    next(item for item in bad if item["stage"] == "B" and item["adaptation_mode"] == "replay")["adaptation_origin"]["hidden_hash"] = "wrong"
    cases.append(("B_FORK_PAIRING", "B arms have different inherited hidden state", bad, evaluations))
    bad_evaluations = deepcopy(evaluations)
    bad_evaluations[0]["records"][0]["primary"]["request_fingerprint"] = "wrong"
    cases.append(("COHORT_PAIRING", "different primary request masks", training, bad_evaluations))
    for record_field, argument_field, wrong_value in (("steps", "steps", 1024), ("dt", "dt", 0.5),
                                                     ("boundary_interval", "reset_interval", 256)):
        bad_evaluations = deepcopy(evaluations)
        bad_evaluations[0]["evaluation_args"][argument_field] = wrong_value
        for record in bad_evaluations[0]["records"]:
            record[record_field] = wrong_value
        cases.append(("EVALUATION_BUDGET", f"all records and arguments agree on wrong {record_field}",
                      training, bad_evaluations))
    bad_evaluations = deepcopy(evaluations)
    bad_evaluations[0]["records"][0]["steps"] = 1024
    cases.append(("EVALUATION_ARGUMENTS", "one record disagrees with its declared evaluation steps", training, bad_evaluations))
    bad_evaluations = deepcopy(evaluations)
    bad_evaluations[0]["evaluation_args"]["conditions"].append("undeclared-regime")
    cases.append(("EVALUATION_ARGUMENTS", "declared conditions disagree with actual records", training, bad_evaluations))
    bad_evaluations = deepcopy(evaluations)
    bad_evaluations[0]["evaluation_args"]["content_probe"] = False
    bad_evaluations[0]["records"] = [record for record in bad_evaluations[0]["records"] if record["action_policy"] == "model"]
    cases.append(("EVALUATION_CONTENT_PROBE", "probe arguments and records agree but violate frozen probe selection",
                  training, bad_evaluations))
    bad_evaluations = deepcopy(evaluations)
    bad_evaluations[0]["evaluation_args"]["content_probe"] = False
    cases.append(("EVALUATION_CONTENT_PROBE", "probe argument disagrees with teacher records", training, bad_evaluations))
    bad = deepcopy(training)
    for item in bad:
        if item["stage"] == "B":
            item["adaptation_origin"]["duration_range_after"] = [5, 9]
    cases.append(("B_DURATION_REGIME", "all B arms share the same wrong duration regime", bad, evaluations))
    for code, description, t, e in cases:
        failed = aggregate(t, e, manifest=manifest)
        assert any(issue["code"] == code and issue["level"] == "error" for issue in failed["audit"]["issues"]), description
    incomplete = aggregate(training[1:], evaluations[1:], manifest=manifest, allow_incomplete=True)
    assert incomplete["audit"]["status"] == "incomplete"
    report["synthetic"] = True
    report["self_test"] = {"passed": True, "deliberate_mismatches_detected": [
        {"code": code, "case": description} for code, description, _, _ in cases],
                           "note": "All outcomes are fictional; these files only test reporting and figure layout."}
    write_outputs(report, output, plots)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "continuity" / "main-20261027-v1")
    parser.add_argument("--evaluations", nargs="+", default=[])
    parser.add_argument("--adapt-root", type=Path)
    parser.add_argument("--adapt-evaluations", nargs="+", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-checkpoint-files", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        output = args.output or ROOT / "data" / "continuity" / "report-selftest"
        report = self_test(output, plots=not args.no_plots)
        print(json.dumps({"self_test": "passed", "synthetic": True, "output": str(output),
                          "figures": len(report["figures"])}))
        return 0
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    seeds = args.seeds or manifest.get("seeds", [0, 1, 2])
    training_roots = [args.root] + ([args.adapt_root] if args.adapt_root else [])
    report = aggregate(load_training_reports(training_roots),
                       load_evaluations(args.evaluations + args.adapt_evaluations),
                       manifest=manifest, seeds=seeds, allow_incomplete=args.allow_incomplete,
                       include_b=bool(args.adapt_root or args.adapt_evaluations),
                       verify_checkpoint_files=args.verify_checkpoint_files)
    report["input_selection"] = {"training_roots": [str(path.resolve()) for path in training_roots],
                                 "evaluation_inputs": args.evaluations + args.adapt_evaluations,
                                 "manifest": str(args.manifest.resolve()) if args.manifest else None,
                                 "manifest_sha256": file_sha256(args.manifest) if args.manifest else None}
    output = args.output or ROOT / "docs" / "evidence" / "continuity-2026-10-27" / "aggregate"
    write_outputs(report, output, plots=not args.no_plots)
    print(json.dumps({"audit": report["audit"]["status"], "errors": report["audit"]["error_count"],
                      "warnings": report["audit"]["warning_count"], "output": str(output),
                      "A_runs": sum(item["stage"] == "A" for item in report["raw_training_reports"]),
                      "B_runs": sum(item["stage"] == "B" for item in report["raw_training_reports"])}))
    return int(report["audit"]["error_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
