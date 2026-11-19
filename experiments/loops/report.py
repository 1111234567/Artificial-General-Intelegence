"""Strict finite-audit aggregation, without loading or advancing a learner.

The trained seed is the replicate. Fixed mask draws are averaged within a
trained seed before the three trained seeds enter a mean and sample SD.
Missing, duplicate, unpaired or corrupted inputs are fatal; no partial scores
are published. The unchanged own-action observer's denominator audit is reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any

from experiments.integration.report import _audit_record


FORMAT_VERSION = "loops-existing-audit-summary-v1"
MODES = ("A", "bp64", "bp_replay")
SEEDS = (0, 1, 2)
MASK_SEEDS = (11, 29, 47)
CONDITIONS = ("iid", "long_sparse", "longer")
INTERVENTIONS = ("none", "diagonal_only", "dag_cut", "random_cut",
                 "dag_norm_matched", "random_norm_matched", "dag_quarter", "random_quarter")
COHORTS = ("primary", "inherited", "carryover_contract")
METRICS = ("strict_clean_completion", "completion", "idle_action_rate",
           "idle_actions_per_1000_idle", "idle_actions_per_1000_total")
PAIRS = (("dag_norm_matched", "random_norm_matched"), ("dag_quarter", "random_quarter"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _integer(value: Any, minimum=0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value: Any, minimum=None, maximum=None) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (minimum is None or value >= minimum) and (maximum is None or value <= maximum))


def _close(left, right, *, tolerance=2e-6):
    return _finite(left) and _finite(right) and math.isclose(left, right, rel_tol=tolerance, abs_tol=1e-10)


class AuditError(ValueError):
    def __init__(self, code, where, details=None, *, checks=0):
        self.audit = {"status": "failed", "checks": checks,
                      "errors": [{"code": code, "where": where, "details": details}]}
        super().__init__(f"{code}: {where}" + (f" ({details})" if details is not None else ""))


class _Checks:
    def __init__(self):
        self.counts = Counter()

    def require(self, code, where, passed, details=None, **_):
        self.counts[code] += 1
        if not passed:
            raise AuditError(code, where, details, checks=sum(self.counts.values()))

    def equal(self, code, where, actual, expected):
        self.require(code, where, actual == expected, {"actual": actual, "expected": expected})

    def result(self):
        return {"status": "passed", "checks": sum(self.counts.values()), "errors": [],
                "checks_by_code": dict(sorted(self.counts.items()))}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                           parse_constant=reject_constant)
    except (OSError, ValueError) as error:
        raise AuditError("READ_JSON", str(path), str(error)) from error
    if not isinstance(value, dict):
        raise AuditError("JSON_OBJECT", str(path))
    return value


def _verify_files(files, audit, code):
    audit.require(code, "file map", isinstance(files, dict) and bool(files))
    for name, expected in files.items():
        path = Path(name)
        audit.require(code, name, path.is_absolute() and path.is_file() and _hash(expected))
        audit.equal(code, name, file_sha256(path), expected)


def _execution_status(root, protocol, protocol_hash, plan, sources, audit):
    """Recognize only the coordinator's root status file, never a raw result."""
    path = root / "execution.json"
    if not path.exists():
        return None, {}
    status = _read_json(path)
    audit.require("EXECUTION_STATUS_SCHEMA", str(path), isinstance(status.get("plan"), str)
                  and _hash(status.get("plan_digest")) and isinstance(status.get("jobs"), dict)
                  and status.get("complete") is True)
    manifest_path = Path(status["plan"]).resolve()
    manifest = _read_json(manifest_path)
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit.equal("EXECUTION_MANIFEST_DIGEST", str(path), status["plan_digest"], canonical(manifest))
    expected_ids = {f"{identifier}_mask_{mask}" for identifier in sources for mask in MASK_SEEDS}
    audit.equal("EXECUTION_JOB_COVERAGE", str(path), set(status["jobs"]), expected_ids)
    jobs = manifest.get("jobs", [])
    audit.equal("EXECUTION_MANIFEST_JOBS", str(path), len(jobs), 27)
    audit.equal("EXECUTION_MANIFEST_JOBS", str(path), {job.get("id") for job in jobs}, expected_ids)
    for name, expected in {**plan["sources"], str(protocol): protocol_hash}.items():
        audit.equal("EXECUTION_FROZEN_INPUT", name, manifest.get("file_hashes", {}).get(name), expected)
    for job in jobs:
        row = status["jobs"][job["id"]]
        audit.equal("EXECUTION_COMMAND_DIGEST", job["id"], row.get("command_digest"), canonical(job["command"]))
        audit.equal("EXECUTION_SUCCESS", job["id"], row.get("returncode"), 0)
        audit.equal("EXECUTION_OUTPUTS", job["id"], row.get("missing_outputs"), [])
        audit.require("EXECUTION_OUTPUTS", job["id"], all(Path(name).is_file() for name in job.get("outputs", [])))
    files = {str(path): file_sha256(path), str(manifest_path): file_sha256(manifest_path)}
    return {"path": str(path), "sha256": files[str(path)], "plan": str(manifest_path),
            "plan_sha256": files[str(manifest_path)], "complete": True, "jobs": 27}, files


def _plan(plan, audit):
    for field, expected in (("format", "loops-existing-audit-protocol-v1"),
            ("mask_seeds", list(MASK_SEEDS)), ("conditions", list(CONDITIONS)),
            ("steps", 1024), ("dt", .25), ("state_samples", 16),
            ("device", "cpu"), ("parameter_updates", 0)):
        audit.equal("PROTOCOL", field, plan.get(field), expected)
    for field in ("steps", "state_samples", "parameter_updates"):
        audit.require("PROTOCOL_INTEGER", field, _integer(plan.get(field)))
    checkpoints = plan.get("checkpoints")
    audit.require("SOURCE_COUNT", "protocol", isinstance(checkpoints, list) and len(checkpoints) == 9)
    sources = {}
    for item in checkpoints:
        audit.require("SOURCE_SCHEMA", "protocol", isinstance(item, dict))
        seed, mode, identifier = item.get("seed"), item.get("mode"), item.get("id")
        audit.require("SOURCE_ID", str(identifier), _integer(seed) and seed in SEEDS and mode in MODES
                      and identifier == f"seed_{seed:03d}_{mode}" and identifier not in sources)
        audit.require("SOURCE_PATH", identifier, isinstance(item.get("path"), str) and _hash(item.get("sha256")))
        sources[identifier] = item
    audit.equal("SOURCE_COVERAGE", "protocol", set(sources),
                {f"seed_{seed:03d}_{mode}" for seed in SEEDS for mode in MODES})
    _verify_files(plan.get("sources"), audit, "FROZEN_SOURCE_SHA256")
    _verify_files({item["path"]: item["sha256"] for item in sources.values()}, audit, "CHECKPOINT_SHA256")
    return sources


def _metadata(metadata, mask_seed, audit, where):
    audit.equal("MASK_FORMAT", where, metadata.get("format"), "loops-weight-interventions-v1")
    audit.equal("MASK_SEED", where, metadata.get("seed"), mask_seed)
    width = metadata.get("width")
    audit.require("MASK_WIDTH", where, _integer(width, 1))
    audit.equal("MASK_ORDER", where, sorted(metadata.get("node_order", [])), list(range(width)))
    audit.require("MASK_SOURCE_HASH", where, _hash(metadata.get("source_weight_sha256")))
    variants = metadata.get("interventions", {})
    audit.equal("MASK_INTERVENTIONS", where, set(variants), set(INTERVENTIONS))
    for name, row in variants.items():
        label = f"{where}/{name}"
        for field in ("weight_sha256", "offdiagonal_mask_sha256"):
            audit.require("MASK_HASH", f"{label}/{field}", _hash(row.get(field)))
        count = row.get("retained_offdiagonal_edges")
        audit.require("MASK_COUNTS", label, _integer(count) and count <= width * (width-1))
        for field in ("retained_offdiagonal_per_target", "retained_offdiagonal_per_source"):
            vector = row.get(field, [])
            audit.require("MASK_COUNTS", f"{label}/{field}", len(vector) == width and all(
                _integer(value) and value < width for value in vector) and sum(vector) == count)
        norms = row.get("offdiagonal_row_l2", [])
        audit.require("MASK_NORMS", label, len(norms) == width and all(_finite(value, 0) for value in norms))
        partial = name.endswith("quarter")
        scales = row.get("row_scales")
        audit.require("MASK_SCALES", label, scales is None if partial else isinstance(scales, list)
                      and len(scales) == width and all(_finite(value, 0, 1) for value in scales))
        audit.equal("DIAGONAL_PRESERVED", label, row.get("diagonal_preserved"), True)
        audit.equal("DAG_CONSTRUCTION", label, row.get("distinct_node_acyclic_by_construction"),
                    name in ("diagonal_only", "dag_cut", "dag_norm_matched"))
        audit.equal("INTERPOLATION_ALPHA", label, row.get("interpolation_alpha"), .25 if partial else None)
        endpoint = "dag_norm_matched" if name == "dag_quarter" else "random_norm_matched" if partial else None
        audit.equal("INTERPOLATION_ENDPOINT", label, row.get("interpolation_endpoint"), endpoint)
        audit.equal("NO_INTERPOLATED_NORM_CLAIM", label, row.get("interpolated_row_l2_matching_claimed"), False)
    audit.equal("SOURCE_WEIGHT_HASH", where, variants["none"]["weight_sha256"], metadata["source_weight_sha256"])
    audit.equal("DIAGONAL_EDGE_COUNT", where, variants["diagonal_only"]["retained_offdiagonal_edges"], 0)
    target = metadata.get("common_target_offdiagonal_row_l2", [])
    audit.require("MATCHED_TARGET", where, len(target) == width and all(_finite(value, 0) for value in target))
    for left, right in (("dag_cut", "random_cut"), ("dag_norm_matched", "random_norm_matched")):
        audit.equal("MATCHED_INCOMING_COUNTS", where, variants[left]["retained_offdiagonal_per_target"],
                    variants[right]["retained_offdiagonal_per_target"])
    for index, expected in enumerate(target):
        audit.require("MATCHED_TARGET_MINIMUM", where, _close(expected, min(
            variants["dag_cut"]["offdiagonal_row_l2"][index], variants["random_cut"]["offdiagonal_row_l2"][index])))
    for name, raw in (("dag_norm_matched", "dag_cut"), ("random_norm_matched", "random_cut")):
        audit.equal("MATCHED_SUPPORT", where, variants[name]["offdiagonal_mask_sha256"],
                    variants[raw]["offdiagonal_mask_sha256"])
        audit.equal("MATCHED_COUNTS", where, variants[name]["retained_offdiagonal_per_target"],
                    variants[raw]["retained_offdiagonal_per_target"])
        audit.require("MATCHED_NORMS", where, all(_close(value, expected) for value, expected in zip(
            variants[name]["offdiagonal_row_l2"], target)))
        audit.require("NO_AMPLIFICATION", where, all(value <= original + 1e-10 for value, original in zip(
            variants[name]["offdiagonal_row_l2"], variants[raw]["offdiagonal_row_l2"])))
    for name in ("dag_quarter", "random_quarter"):
        audit.equal("QUARTER_SUPPORT", where, variants[name]["offdiagonal_mask_sha256"],
                    variants["none"]["offdiagonal_mask_sha256"])
    return width


def _structure(value, width, plan, audit, where):
    audit.require("STRUCTURE_SCHEMA", where, isinstance(value, dict))
    weight, state = value.get("weight", {}), value.get("state", {})
    audit.equal("WEIGHT_STRUCTURE_FORMAT", where, weight.get("format"), "loops-weight-structure-v1")
    audit.equal("STATE_STRUCTURE_FORMAT", where, state.get("format"), "loops-instantaneous-states-v1")
    audit.equal("STRUCTURE_WIDTH", where, weight.get("width"), width)
    audit.equal("STRUCTURE_WIDTH", where, state.get("width"), width)
    audit.equal("STRUCTURE_SAMPLE_COUNT", where, state.get("sample_count"), plan["state_samples"])
    audit.equal("STRUCTURE_SAMPLE_COUNT", where, len(state.get("samples", [])), plan["state_samples"])
    audit.equal("STRUCTURE_SAMPLE_INDEX", where, [row.get("sample_index") for row in state["samples"]],
                list(range(plan["state_samples"])))
    for field in ("weight", "offdiagonal", "graph", "conventions"):
        audit.require("STRUCTURE_FIELD", f"{where}/{field}", isinstance(weight.get(field), dict))
    audit.require("STRUCTURE_AGGREGATE", where, isinstance(state.get("aggregate"), dict)
                  and bool(state["aggregate"]) and isinstance(value.get("sampling"), str))


def _record(record, width, plan, audit, where):
    for field in ("batch_size", "n_jobs", "active_jobs"):
        audit.require("RECORD_DIMENSIONS", f"{where}/{field}", _integer(record.get(field), 1))
    _audit_record(record, audit, where, {"eval_steps": plan["steps"], "dt": plan["dt"],
                  "batch_size": record["batch_size"], "n_jobs": record["n_jobs"]})
    for field in ("world_resets", "hidden_resets", "parameter_updates"):
        audit.require("NO_RESETS_OR_TRAINING", f"{where}/{field}", _integer(record.get(field)) and record[field] == 0)
    for field in ("action_hash", "model_action_hash", "observation_hash", "raw_observation_hash",
            "first_observation_hash", "first_model_observation_hash", "end_world_state_hash",
            "end_hidden_hash", "graph_parameter_hash"):
        audit.require("FORK_HASH", f"{where}/{field}", _hash(record.get(field)))
    for field in ("action_hash", "observation_hash", "first_observation_hash", "parameter_hash"):
        other = {"action_hash": "model_action_hash", "observation_hash": "raw_observation_hash",
                 "first_observation_hash": "first_model_observation_hash", "parameter_hash": "graph_parameter_hash"}[field]
        audit.equal("OWN_ACTION_ONLY", f"{where}/{field}", record.get(field), record.get(other))
    audit.equal("NO_ACTION_ORACLE", where, record.get("intervention"), "none")
    audit.equal("NO_ACTION_ORACLE", where, record.get("action_policy"), "model")
    for field in ("model_action_changes", "observation_ablation_opportunities"):
        audit.equal("NO_ACTION_ORACLE", f"{where}/{field}", record["opportunities"].get(field), 0)
    audit.equal("MODEL_WIDTH", where, record.get("model", {}).get("width"), width)
    total = record["opportunities"]
    audit.require("IDLE_ACTION_SUBSET", where, total["inactive_actions"] <= total["inactive_opportunities"])
    audit.equal("DISABLED_OPPORTUNITIES", where, total["disabled_slot_opportunities"],
                record["steps"] * record["batch_size"] * (record["n_jobs"] - record["active_jobs"]))
    for cohort in COHORTS:
        row, censored = record[cohort], record["censored"][cohort]
        vectors = row["per_actor"]
        for field in ("matured", "completed", "clean_completed"):
            audit.require("ACTOR_NONNEGATIVE", f"{where}/{cohort}/{field}",
                          all(_integer(value) for value in vectors[field]))
        audit.require("ACTOR_SUBSET", f"{where}/{cohort}", all(0 <= clean <= completed <= matured
            for matured, completed, clean in zip(vectors["matured"], vectors["completed"], vectors["clean_completed"])))
        audit.require("CENSORED_COUNTS", f"{where}/{cohort}", all(_integer(censored.get(field))
            for field in ("count", "completed", "clean_completed")) and
            0 <= censored["clean_completed"] <= censored["completed"] <= censored["count"])
    for row in record["inactive_history"]:
        audit.require("IDLE_HISTORY_SUBSET", where, _integer(row["opportunities"]) and _integer(row["actions"])
                      and row["actions"] <= row["opportunities"])
        if "action_rate_per_idle_opportunity" in row:
            expected = row["actions"] / row["opportunities"] if row["opportunities"] else None
            audit.require("IDLE_HISTORY_RATE", where, row["action_rate_per_idle_opportunity"] is None
                          if expected is None else _close(row["action_rate_per_idle_opportunity"], expected))


def _paired(record):
    fields = ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash", "end_scheduler_hash",
              "start_event_hash", "end_event_hash", "first_observation_hash", "actor_ids", "case_ids",
              "fork_tick", "fork_time", "end_tick", "end_time", "steps", "dt", "batch_size", "n_jobs",
              "active_jobs", "evaluation_regime_id", "regime", "primary_eligible_issued",
              "carryover_contract_eligible_issued", "initial_eligible_pending")
    return {**{field: record[field] for field in fields}, "cohorts": {
        name: {"matured": record[name]["matured"], "per_actor_matured": record[name]["per_actor"]["matured"],
               "request_fingerprint": record[name]["request_fingerprint"],
               "censored": record["censored"][name]["count"]} for name in COHORTS}}


def _metrics(record):
    return dict(strict_clean_completion=record["primary"]["clean_completion_rate"],
                completion=record["primary"]["completion_rate"],
                idle_action_rate=record["inactive_actions_per_idle_opportunity"],
                idle_actions_per_1000_idle=record["inactive_actions_per_1000_idle_opportunities"],
                idle_actions_per_1000_total=record["inactive_actions_per_1000_total_opportunities"])


def _summarize(per_seed):
    defined = [row["value"] for row in per_seed if row["value"] is not None]
    return {"n": len(defined), "n_total": len(per_seed), "n_undefined": len(per_seed)-len(defined),
            "mean": statistics.mean(defined) if defined else None,
            "sample_sd": statistics.stdev(defined) if len(defined) > 1 else None, "per_seed": per_seed}


def _statistic(records, mode, condition, left, metric, right=None):
    masks = MASK_SEEDS[:1] if left in ("none", "diagonal_only") else MASK_SEEDS
    rows = []
    for seed in SEEDS:
        values = []
        for mask in masks:
            value = _metrics(records[(seed, mode, mask, condition, left)])[metric]
            if right is not None:
                other_mask = MASK_SEEDS[0] if right in ("none", "diagonal_only") else mask
                other = _metrics(records[(seed, mode, other_mask, condition, right)])[metric]
                value = value-other if value is not None and other is not None else None
            values.append(value)
        # An undefined draw makes that trained-seed statistic undefined; do not
        # silently change the fixed mask weighting by dropping that draw.
        value = statistics.mean(values) if all(item is not None for item in values) else None
        rows.append({"seed": seed, "mask_count": len(masks), "mask_seeds": list(masks),
                     "mask_values": values, "value": value})
    return _summarize(rows)


def build_report(protocol: str | Path, root: str | Path) -> dict[str, Any]:
    """Validate every expected input before returning a complete aggregate."""
    protocol, root = Path(protocol).resolve(), Path(root).resolve()
    audit = _Checks()
    analysis_files = {str(path): file_sha256(path) for path in (
        Path(__file__).resolve(), Path(_audit_record.__code__.co_filename).resolve())}
    plan = _read_json(protocol)
    protocol_hash = file_sha256(protocol)
    sources = _plan(plan, audit)
    audit.require("RAW_DIRECTORY", str(root), root.is_dir())
    execution_status, execution_files = _execution_status(root, protocol, protocol_hash, plan, sources, audit)
    paths = sorted(path for path in root.rglob("*.json") if path.is_file() and path != root / "execution.json")
    audit.equal("JOB_COUNT", str(root), len(paths), 27)
    jobs, records, structures = {}, {}, []
    paired, boundaries, parameters, source_weights, source_states = {}, {}, {}, {}, {}
    raw_hashes = {}
    for path in paths:
        where = str(path)
        raw_hash = file_sha256(path)
        item = _read_json(path)
        audit.equal("RAW_SHA256_STABLE", where, file_sha256(path), raw_hash)
        raw_hashes[where] = raw_hash
        audit.equal("RAW_FORMAT", where, item.get("format"), "loops-existing-audit-v1")
        audit.equal("PROTOCOL_SHA256", where, item.get("protocol_sha256"), protocol_hash)
        audit.require("PROTOCOL_LINK", where, isinstance(item.get("protocol"), str)
                      and Path(item["protocol"]).resolve() == protocol)
        source = item.get("source", {})
        audit.require("RAW_SOURCE_SCHEMA", where, isinstance(source, dict))
        audit.require("RAW_RECORDS_SCHEMA", where, isinstance(item.get("records"), list))
        identifier, mask = source.get("id"), item.get("mask_seed")
        audit.require("RAW_SOURCE_ID", where, identifier in sources)
        audit.equal("RAW_SOURCE_LINK", where, source, sources[identifier])
        audit.require("RAW_MASK_SEED", where, _integer(mask) and mask in MASK_SEEDS)
        job_key = (identifier, mask)
        audit.require("DUPLICATE_JOB", where, job_key not in jobs, job_key)
        audit.equal("RAW_FROZEN_SOURCES", where, item.get("source_files"), plan["sources"])
        audit.equal("CHECKPOINT_UNCHANGED", where, item.get("checkpoint_bytes_unchanged"), True)
        before, after = item.get("source_semantic_hash_before"), item.get("source_semantic_hash_after")
        audit.require("SOURCE_STATE_HASH", where, _hash(before) and before == after)
        if identifier in source_states:
            audit.equal("SOURCE_STATE_ACROSS_MASKS", where, before, source_states[identifier])
        source_states[identifier] = before
        try:
            metadata = item["intervention_metadata"]
            width = _metadata(metadata, mask, audit, where)
            if identifier in source_weights:
                audit.equal("SOURCE_WEIGHT_ACROSS_MASKS", where, metadata["source_weight_sha256"], source_weights[identifier])
            source_weights[identifier] = metadata["source_weight_sha256"]
            if mask == MASK_SEEDS[0]:
                _structure(item.get("structure"), width, plan, audit, where)
                value = deepcopy(item["structure"])
                value["state"].pop("samples", None)
                structures.append({"source": deepcopy(source), "mask_seed": mask,
                                   "raw_path": where, "raw_sha256": raw_hash, **value})
            else:
                audit.equal("STRUCTURE_FIRST_DRAW_ONLY", where, item.get("structure"), None)
            names = INTERVENTIONS if mask == MASK_SEEDS[0] else INTERVENTIONS[2:]
            expected = {(condition, name) for condition in CONDITIONS for name in names}
            seen = set()
            for record in item["records"]:
                condition, name = record.get("condition"), record.get("graph_intervention")
                local_key = (condition, name)
                label = f"{where}/{condition}/{name}"
                audit.require("RECORD_EXPECTED", label, local_key in expected)
                audit.require("DUPLICATE_RECORD", label, local_key not in seen)
                seen.add(local_key)
                audit.equal("RECORD_MASK_SEED", label, record.get("mask_seed"), mask)
                audit.equal("GRAPH_POLICY", label, record.get("graph_action_policy"),
                            "unmodified_model" if name == "none" else "parameter_intervention_counterfactual")
                _record(record, width, plan, audit, label)
                pairing_key = (identifier, condition)
                identity = _paired(record)
                if pairing_key in paired:
                    audit.equal("EXOGENOUS_PAIRING_ACROSS_MASKS", label, identity, paired[pairing_key])
                paired[pairing_key] = identity
                boundary = {field: record[field] for field in ("start_world_state_hash", "start_hidden_hash",
                    "start_scheduler_hash", "start_event_hash", "fork_tick", "fork_time", "actor_ids", "case_ids")}
                if identifier in boundaries:
                    audit.equal("COMMON_INHERITED_BOUNDARY", label, boundary, boundaries[identifier])
                boundaries[identifier] = boundary
                parameter_key = (identifier, mask, name)
                hashes = (record["parameter_hash"], record["source_continuation_hash"])
                if parameter_key in parameters:
                    audit.equal("PARAMETERS_ACROSS_CONDITIONS", label, hashes, parameters[parameter_key])
                parameters[parameter_key] = hashes
                records[(source["seed"], source["mode"], mask, condition, name)] = record
            audit.equal("RECORD_COVERAGE", where, seen, expected)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise AuditError("MALFORMED_RAW_SCHEMA", where, str(error), checks=sum(audit.counts.values())) from error
        jobs[job_key] = {"path": where, "sha256": raw_hash, "source_id": identifier, "mask_seed": mask,
                         "forks": len(item["records"]), "source_semantic_hash_before": before,
                         "source_semantic_hash_after": after}
    audit.equal("JOB_COVERAGE", str(root), set(jobs), {(identifier, mask) for identifier in sources for mask in MASK_SEEDS})
    audit.equal("FORK_COUNT", str(root), len(records), 540)
    audit.equal("STRUCTURAL_SUMMARY_COUNT", str(root), len(structures), 9)
    tables, contrasts = [], []
    for mode in MODES:
        for condition in CONDITIONS:
            for name in INTERVENTIONS:
                tables.append({"mode": mode, "condition": condition, "intervention": name,
                    "metrics": {metric: _statistic(records, mode, condition, name, metric) for metric in METRICS}})
            for left, right in (*PAIRS, *((name, "none") for name in INTERVENTIONS if name != "none")):
                contrasts.append({"mode": mode, "condition": condition, "left": left, "right": right,
                    "contrast": f"{left}-minus-{right}",
                    "metrics": {metric: _statistic(records, mode, condition, left, metric, right) for metric in METRICS}})
    # Recheck all input bytes at the final boundary, including the 27 raw files.
    audit.equal("PROTOCOL_UNCHANGED", str(protocol), file_sha256(protocol), protocol_hash)
    _verify_files(plan["sources"], audit, "FROZEN_SOURCE_SHA256_FINAL")
    _verify_files({item["path"]: item["sha256"] for item in sources.values()}, audit, "CHECKPOINT_SHA256_FINAL")
    _verify_files(raw_hashes, audit, "RAW_SHA256_FINAL")
    if execution_files:
        _verify_files(execution_files, audit, "EXECUTION_SHA256_FINAL")
    _verify_files(analysis_files, audit, "ANALYSIS_SOURCE_SHA256_FINAL")
    return {"format": FORMAT_VERSION, "complete": True, "audit": audit.result(),
        "counts": {"sources": 9, "jobs": 27, "forks": 540, "structural_summaries": 9},
        "protocol": str(protocol), "protocol_sha256": protocol_hash, "raw_root": str(root),
        "analysis_source_files": analysis_files,
        "execution_status": execution_status,
        "sources": list(sources.values()), "source_files": deepcopy(plan["sources"]),
        "raw_results": sorted(jobs.values(), key=lambda row: (row["source_id"], row["mask_seed"])),
        "structural_summaries": sorted(structures, key=lambda row: row["source"]["id"]),
        "tables": tables, "paired_differences": contrasts,
        "statistics": {"replicate": "trained seed", "trained_seeds": list(SEEDS), "mask_seeds": list(MASK_SEEDS),
            "rule": "Average fixed mask draws within each trained seed, then mean and sample SD across three trained seeds.",
            "baseline": "none and diagonal_only are run once per trained seed; the same none baseline is used for all paired mask differences.",
            "undefined": "Any undefined mask rate makes that seed's mask mean undefined; n and n_undefined are explicit.",
            "sd": "Sample SD uses denominator n-1 across trained seeds; actors and masks are not independent training replicates.",
            "difference": "left minus right, paired within trained seed and mask before mask averaging"},
        "definitions": {"strict_clean_completion": "primary clean_completed / primary matured",
            "completion": "primary completed / primary matured",
            "idle_action_rate": "inactive_actions / inactive_opportunities (enabled slots only)",
            "idle_actions_per_1000_idle": "1000 * inactive_actions / inactive_opportunities",
            "idle_actions_per_1000_total": "1000 * inactive_actions / all enabled opportunities"},
        "interpretation": deepcopy(plan.get("interpretation", {})),
        "limitations": ["Structural directed cycles establish possible feedback, not a causal share of behavior.",
            "Frozen-state weighted closed walks can revisit neurons and repeat shorter cycles; they do not count simple cycles or measure multi-tick feedback.",
            "DAG/random deletions match incoming counts and matched endpoints also match row L2 norms; outgoing degree, spectrum and state distributions remain unmatched.",
            "Quarter interventions retain cycles and their interpolated row norms need not match.",
            "Parameter intervention scores are counterfactuals. Only none is the unmodified existing model.",
            "Three trained seeds provide descriptive means and sample SD, not a definitive causal estimate."]}


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _number(summary, scale=1):
    if summary["mean"] is None:
        return "undefined"
    sd = "undefined" if summary["sample_sd"] is None else f"{scale * summary['sample_sd']:.4f}"
    return f"{scale * summary['mean']:.4f} ± {sd} (n={summary['n']})"


def write_artifacts(report, output):
    output = Path(output).resolve()
    raw = Path(report["raw_root"])
    if output == raw or raw in output.parents:
        raise ValueError("output must be outside the raw-input directory")
    _atomic_text(output / "report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for filename, rows, dimensions in (("table.csv", report["tables"], ("mode", "condition", "intervention")),
            ("paired_differences.csv", report["paired_differences"], ("mode", "condition", "left", "right", "contrast"))):
        stream = io.StringIO(newline="")
        fields = [*dimensions, "metric", "n", "n_total", "n_undefined", "mean", "sample_sd"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for metric, summary in row["metrics"].items():
                writer.writerow({**{key: row[key] for key in dimensions}, "metric": metric,
                                 **{key: summary[key] for key in fields if key not in (*dimensions, "metric")}})
        _atomic_text(output / filename, stream.getvalue())
    text = ["# Existing-model loop audit", "", "Complete: 9 existing checkpoints, 27 raw files, 540 frozen forks.",
        f"Protocol SHA256: `{report['protocol_sha256']}`.", "",
        "Every file, checkpoint, frozen source, inherited boundary, exogenous cohort and rate denominator passed validation.",
        "Masks are averaged within trained seed; displayed mean ± sample SD uses three trained seeds.",
        "Strict and completion columns are percentages. Idle actions are per 1,000 idle opportunities.", "",
        "| Mode | Condition | Intervention | Strict clean completion | Completion | Idle actions |",
        "|---|---|---|---:|---:|---:|"]
    for row in report["tables"]:
        values = [_number(row["metrics"][name], scale) for name, scale in (
            ("strict_clean_completion", 100), ("completion", 100), ("idle_actions_per_1000_idle", 1))]
        text.append(f"| {row['mode']} | {row['condition']} | {row['intervention']} | " + " | ".join(values) + " |")
    text += ["", "## Paired differences", "", "Differences are left minus right, paired before mask averaging.",
        "The completion differences are percentage points; idle differences are per 1,000 idle opportunities.", "",
        "| Mode | Condition | Left | Right | Strict difference | Completion difference | Idle difference |",
        "|---|---|---|---|---:|---:|---:|"]
    for row in report["paired_differences"]:
        values = [_number(row["metrics"][name], scale) for name, scale in (
            ("strict_clean_completion", 100), ("completion", 100), ("idle_actions_per_1000_idle", 1))]
        text.append(f"| {row['mode']} | {row['condition']} | {row['left']} | {row['right']} | " + " | ".join(values) + " |")
    text += ["", "## Structural summaries", "", "The nine first-draw summaries are retained in report.json; individual state samples remain in the linked raw files.",
        "| Source | Width | Exact off-diagonal edges | Cyclic SCC sizes | Guaranteed simple-cycle lengths |",
        "|---|---:|---:|---|---|"]
    for row in report["structural_summaries"]:
        graph = row["weight"]["graph"]["exact"]
        text.append(f"| {row['source']['id']} | {row['weight']['width']} | {graph['offdiagonal_edges']} | "
                    f"{graph['multineuron_cyclic_scc_sizes']} | {graph['simple_cycle_lengths_guaranteed']} |")
    text += ["", "## Scope of interpretation", "", *[f"- {value}" for value in report["limitations"]],
        "", "## Raw provenance", "", "| Raw result | SHA256 |", "|---|---|"]
    text += [f"| [{Path(row['path']).name}](<{Path(row['path']).as_posix()}>) | `{row['sha256']}` |" for row in report["raw_results"]]
    _atomic_text(output / "report.md", "\n".join(text) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.protocol, args.root)
        write_artifacts(report, args.output)
    except (AuditError, ValueError) as error:
        parser.exit(1, f"Loop audit rejected: {error}\n")
    print(f"Validated 9 sources / 27 jobs / 540 forks; saved {args.output.resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
