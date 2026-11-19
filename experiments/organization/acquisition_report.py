"""Strict paired audit of zero-shrink acquisition and existing native controls.

Only the new nine lifetimes are new training. Old checkpoints and frozen probe
records are immutable borrowed evidence, never regenerated evaluation runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import asdict
import gc
import json
import multiprocessing
import os
from pathlib import Path

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong import report as legacy
from experiments.lifelong import run as native
from experiments.loops.report import AuditError, _Checks, _read_json, _hash, _object
from experiments.organization import acquisition as execution

FORMAT = "organization-acquisition-summary-v1"
METRICS = ("strict_completion", "completion", "finish_precision", "own_state_content",
    "own_state_legal_content", "first_legal_content", "inactive_per_1000_idle",
    "inactive_per_1000_total", "first_error_fraction", "cue_content", "matched_cue_content",
    "matched_legal_content", "matched_loss_given_cue_correct", "uncompleted_no_begin",
    "uncompleted_despite_legal")
PROTOCOL_FIELDS = set("""format created_utc sources raw_root branches variants seeds native_template
    target_ticks checkpoint_ticks latest_interval_ticks training_device evaluation_device threads dt
    evaluation_specs baseline interpretation""".split())
MARKER_FIELDS = set("""format branch protocol_sha256 target stage_ticks checkpoint_sha256 checkpoints
    hidden_resets initial_parameter_hash birth_hash baseline_id baseline_files""".split())
ENTRY_FIELDS = {"path", "sha256", "semantic_hash", "initial_parameter_hash", "birth_hash"}
EVALUATION_FIELDS = {"format", "branch", "protocol_sha256", "source_unchanged", "states", "records", "baseline_id", "birth_hash"}
BASELINE_ENTRY_FIELDS = {"checkpoint", "checkpoint_sha256", "semantic_hash", "cpu_semantic_hash", "initial_parameter_hash", "birth_hash"}
GROUP = (*execution.KEYS, "tick", "phase", "family", "condition", "steps")


def _fields(value, fields, audit, where):
    legacy._fields(value, fields, audit, where)


def _input(path, inputs, audit, expected=None):
    path, sha = legacy._input(path, inputs, audit)
    if expected is not None:
        audit.equal("ACQUISITION_INPUT_SHA256", str(path), sha, expected)
    return path


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate CSV columns")
        return list(reader)


def _read_inputs_object(path):
    """Extract the old pretty-printed top-level binding map without loading scores."""
    lines, started = [], False
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not started:
                if line.rstrip() == '  "inputs": {':
                    lines, started = ["{"], True
            elif line.rstrip() in ("  },", "  }"):
                lines.append("}")
                return json.loads("".join(lines), object_pairs_hook=_object)
            else:
                lines.append(line)
    raise ValueError("old aggregate has no complete top-level input bindings")


def _phase(plan, tick):
    return dict(zip(plan["checkpoint_ticks"][1:], ("main_acquisition", "post_withdrawal", "after_return")))[tick]


def _identity(item):
    return {key: item[key] for key in ("id", "seed", *execution.KEYS)}


def _spec_key(spec):
    return tuple(spec[key] for key in ("tick", "family", "condition", "steps"))


def _config(state, item, audit, where, *, shrink):
    expected = dict(item["native_config"], shrink_step=shrink)
    audit.equal("FULL_ACQUISITION_CONFIG", where, legacy.normalize(state["config"]), expected)
    audit.equal("ACTUAL_ACQUISITION_SHRINK", where, state["operator"]["config"]["shrink_step"], shrink)


def _protocol(path, root, audit, inputs, mechanical):
    path = _input(path, inputs, audit)
    plan = _read_json(path)
    _fields(plan, PROTOCOL_FIELDS, audit, str(path))
    audit.equal("ACQUISITION_RAW_ROOT", str(path), plan["raw_root"], str(root))
    audit.equal("THREE_TRAINING_SEEDS", str(path), plan["seeds"], [0, 1, 2])
    audit.equal("DECLARED_VARIANTS", str(path), plan["variants"], list(execution.VARIANTS))
    if not mechanical:
        audit.equal("FORMAL_NATIVE_DEFAULTS", str(path), plan["native_template"],
                    legacy.normalize(asdict(native.NativeConfig(device="cuda"))))
        audit.equal("FORMAL_SNAPSHOT_TICKS", str(path), plan["checkpoint_ticks"], [0, 131072, 163840, 172032])
        audit.equal("FORMAL_PROBE_HORIZONS", str(path), {s["steps"] for s in plan["evaluation_specs"]}, {2048})
        audit.equal("FORMAL_PERIODIC_LATEST", str(path), plan["latest_interval_ticks"], 32768)
    for item in plan["branches"]:
        checked, branch, sha = execution.read_plan(path, item["id"])
        audit.equal("EXECUTION_PROTOCOL_READER", item["id"], checked, plan)
        audit.equal("EXPLICIT_EXECUTION_BRANCH", item["id"], branch, item)
    audit.equal("EXPLICIT_EXECUTION_INVENTORY", str(path), plan["sources"], execution.execution_files())
    for source, digest in plan["sources"].items():
        _input(source, inputs, audit, digest)
    for kind in ("training", "evaluation", "baseline"):
        manifest_path = _input(path.parent / f"acquisition-{kind}-commands-v1.json", inputs, audit)
        manifest = _read_json(manifest_path)
        _fields(manifest, {"cwd", "file_hashes", "jobs"}, audit, str(manifest_path))
        audit.equal("COMMAND_FILE_BINDINGS", str(manifest_path), manifest["file_hashes"],
            {**plan["sources"], **plan["baseline"]["files"], str(path): sha})
        audit.equal("COMMAND_CWD", str(manifest_path), manifest["cwd"], str(execution.ROOT))
        audit.equal("COMMAND_JOBS", str(manifest_path), [j["id"] for j in manifest["jobs"]],
                    [] if kind == "baseline" else [b["id"] for b in plan["branches"]])
    return plan, sha


def _baseline(plan, inputs, audit, mechanical):
    bound = plan["baseline"]
    _fields(bound, {"files", "branches", "shrink_step", "note"}, audit, "baseline")
    audit.equal("BASELINE_SHRINK", "baseline", bound["shrink_step"], 2e-4)
    files = {}
    for name, sha in bound["files"].items():
        p = _input(name, inputs, audit, sha)
        audit.require("BASELINE_FILE_NAMES_UNIQUE", str(p), p.name not in files)
        files[p.name] = p
    audit.equal("BASELINE_FILE_INVENTORY", "baseline", set(files),
        {"completion.json", "artifacts.json", "checkpoints.csv", "matched_memory_per_seed.csv", "per_seed.csv", "supplement_per_seed.csv"})
    evidence = files["completion.json"].parent
    if not mechanical:
        audit.equal("ORIGINAL_BASELINE_ROOT", "baseline", evidence, execution.OLD_EVIDENCE)
    for name, p in files.items():
        expected = evidence / name if name == "completion.json" else evidence / "aggregate-v1" / name
        audit.equal("BASELINE_EVIDENCE_LOCATION", name, p, expected)
    done, artifacts = _read_json(files["completion.json"]), _read_json(files["artifacts.json"])
    audit.require("OLD_STRICT_AUDIT_PASSED", str(evidence), done.get("complete") is True
        and done.get("audit", {}).get("status") == "passed" and done["audit"].get("errors") == [])
    audit.equal("OLD_AGGREGATE_ARTIFACTS", str(evidence), artifacts, done["aggregate_artifacts"])
    for name in ("checkpoints.csv", "matched_memory_per_seed.csv", "per_seed.csv", "supplement_per_seed.csv"):
        audit.equal("OLD_AUDITED_CSV_SHA256", name, inputs[str(files[name])], artifacts[name])
    old_report = _input(evidence / "aggregate-v1/report.json", inputs, audit, artifacts["report.json"])
    old_inputs = _read_inputs_object(old_report)
    audit.require("OLD_RAW_INPUT_MAP", str(old_report), bool(old_inputs)
                  and all(_hash(h) and Path(p).is_absolute() for p, h in old_inputs.items()))
    old_ids = {item["baseline_id"] for item in plan["branches"]}
    audit.equal("OLD_BRANCH_INVENTORY", "baseline", set(bound["branches"]), old_ids)
    wanted_ticks = set(plan["checkpoint_ticks"])
    checkpoints = {identifier: {} for identifier in old_ids}
    for row in _csv_rows(files["checkpoints.csv"]):
        if row["id"] not in old_ids or int(row["tick"]) not in wanted_ticks:
            continue
        tick, identifier = int(row["tick"]), row["id"]
        audit.require("UNIQUE_OLD_CHECKPOINT_ROW", identifier, tick not in checkpoints[identifier])
        item = next(b for b in plan["branches"] if b["baseline_id"] == identifier)
        for key in ("seed", *execution.KEYS):
            audit.equal("OLD_CHECKPOINT_BRANCH", f"{identifier}/{key}", int(row[key]) if key == "seed" else row[key], item[key])
        entry = bound["branches"][identifier][str(tick)]
        _fields(entry, BASELINE_ENTRY_FIELDS, audit, f"{identifier}/{tick}")
        for key, value in entry.items():
            audit.equal("OLD_CHECKPOINT_CSV_BINDING", f"{identifier}/{tick}/{key}", row[key], value)
        cp = _input(entry["checkpoint"], inputs, audit, entry["checkpoint_sha256"])
        audit.equal("OLD_CHECKPOINT_RAW_AUDIT_BINDING", str(cp), old_inputs.get(str(cp)), entry["checkpoint_sha256"])
        decoded = {key: json.loads(value) if value and value[:1] in ("{", "[") else value for key, value in row.items()}
        for key in ("tick", "updates", "seed", "world_tick", "total_parameters"):
            decoded[key] = int(decoded[key])
        decoded["world_time"] = float(decoded["world_time"])
        checkpoints[identifier][tick] = decoded
        if tick == 0:
            with _preserve_process_state():
                state = torch.load(cp, map_location="cpu", weights_only=False)
                _config(state, item, audit, str(cp), shrink=2e-4)
                born = native.NativeTrainer.from_state_dict(state, device="cpu", strict_sources=True)
                audit.equal("OLD_REAL_BIRTH_HASH", identifier, state_digest(born.birth), entry["birth_hash"])
                audit.equal("OLD_REAL_INITIAL_PARAMETERS", identifier, born.initial_parameter_hash, entry["initial_parameter_hash"])
                audit.equal("OLD_REAL_ZERO_PARAMETERS", identifier, state_digest(state["model"]), entry["initial_parameter_hash"])
    for identifier, cps in checkpoints.items():
        audit.equal("OLD_CHECKPOINT_TICKS", identifier, set(cps), wanted_ticks)
        for tick, cp in cps.items():
            audit.equal("OLD_UNCHANGED_BIRTH", f"{identifier}/{tick}", cp["birth_hash"], cps[0]["birth_hash"])
            audit.equal("OLD_UNCHANGED_INITIAL_PARAMETERS", f"{identifier}/{tick}", cp["initial_parameter_hash"], cps[0]["initial_parameter_hash"])
    return checkpoints, files["matched_memory_per_seed.csv"], old_inputs


def _new_checkpoints(plan, item, sha, inputs, audit):
    folder = Path(item["output"])
    marker_path = _input(folder / "launch-complete.json", inputs, audit)
    marker = _read_json(marker_path)
    _fields(marker, MARKER_FIELDS, audit, str(marker_path))
    for key, expected in (("format", "lifelong-launch-complete-v1"), ("branch", item),
            ("protocol_sha256", sha), ("target", plan["target_ticks"]), ("stage_ticks", plan["target_ticks"]),
            ("hidden_resets", 0), ("baseline_id", item["baseline_id"]), ("baseline_files", plan["baseline"]["files"])):
        audit.equal("NEW_COMPLETION_BINDING", f"{item['id']}/{key}", marker[key], expected)
    start_path = _input(folder.with_name(folder.name + ".launch-start.json"), inputs, audit)
    audit.equal("NEW_START_BINDING", item["id"], _read_json(start_path),
        dict(format="organization-acquisition-start-v1", branch=item, protocol_sha256=sha))
    ticks = plan["checkpoint_ticks"]
    audit.equal("NEW_SNAPSHOT_SET", item["id"], set(marker["checkpoints"]), {str(t) for t in ticks})
    audit.equal("NEW_SNAPSHOT_FILE_SET", item["id"], {p.resolve() for p in (folder / "checkpoints").glob("*.pt")},
                {folder / "checkpoints" / f"tick_{tick:06d}.pt" for tick in ticks})
    checkpoints, previous = {}, None
    for tick in ticks:
        entry = marker["checkpoints"][str(tick)]
        _fields(entry, ENTRY_FIELDS, audit, f"{item['id']}/{tick}")
        cp = _input(folder / "checkpoints" / f"tick_{tick:06d}.pt", inputs, audit, entry["sha256"])
        audit.equal("NEW_SNAPSHOT_PATH", str(cp), entry["path"], str(cp))
        state = torch.load(cp, map_location="cpu", weights_only=False)
        _config(state, item, audit, str(cp), shrink=0.)
        result = legacy.audit_checkpoint_payload(state, plan=None, branch=item, tick=tick, audit=audit, where=str(cp))
        for key in ("semantic_hash", "birth_hash", "initial_parameter_hash"):
            audit.equal("NEW_SNAPSHOT_BINDING", f"{cp}/{key}", result[key], entry[key])
        old = plan["baseline"]["branches"][item["baseline_id"]]["0"]
        audit.equal("PAIRED_BIRTH_HASH", str(cp), result["birth_hash"], old["birth_hash"])
        audit.equal("PAIRED_INITIAL_PARAMETERS", str(cp), result["initial_parameter_hash"], old["initial_parameter_hash"])
        if previous is not None:
            audit.equal("NEW_TRACE_PREFIX", str(cp), legacy.json_hash(state["trace"][:previous["updates"]]), previous["trace_json_hash"])
        result.update(checkpoint=str(cp), checkpoint_sha256=entry["sha256"])
        if tick == plan["target_ticks"]:
            latest = _input(folder / "latest.pt", inputs, audit, marker["checkpoint_sha256"])
            with _preserve_process_state():
                latest_state = torch.load(latest, map_location="cpu", weights_only=False)
                _config(latest_state, item, audit, str(latest), shrink=0.)
                restored = native.NativeTrainer.from_state_dict(latest_state, device="cpu", strict_sources=True)
                audit.equal("LATEST_FULL_RESTORED_STATE", str(latest), execution.normalized_hash(restored, plan["training_device"]), result["semantic_hash"])
            expected_report = deepcopy(result["report"])
            expected_report["config"]["device"] = state["config"]["device"]
            expected_report["runtime"] = state["runtime"]
            report_path = _input(folder / "report.json", inputs, audit)
            audit.equal("SAVED_NATIVE_REPORT", str(report_path), legacy.json_hash(_read_json(report_path)), legacy.json_hash(expected_report))
            trace_path = _input(folder / "trace.jsonl", inputs, audit)
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            audit.require("TRACE_JSONL_LINES", str(trace_path), all(line.strip() for line in lines))
            audit.equal("SAVED_FULL_TRACE", str(trace_path), legacy.json_hash([json.loads(line) for line in lines]), result["trace_json_hash"])
            for namespace in ("sources", "dependencies"):
                for name, text in state[namespace].items():
                    saved = _input(folder / namespace / name, inputs, audit)
                    audit.equal("SAVED_NATIVE_SOURCE", str(saved), saved.read_text(encoding="utf-8"), text)
            for key in ("birth_hash", "initial_parameter_hash"):
                audit.equal("COMPLETION_BIRTH_BINDING", item["id"], marker[key], result[key])
        result.pop("report")
        result.pop("checks")
        checkpoints[tick] = result
        previous = result
        del state
        gc.collect()
    return checkpoints


def _row(item, spec, metrics, record, cp, plan, *, old=False):
    return _identity(item) | {k: spec[k] for k in ("tick", "family", "condition", "steps")} | dict(
        phase=_phase(plan, spec["tick"]), shrink_step=2e-4 if old else 0.,
        baseline_id=item["baseline_id"],
        evidence="borrowed_original_frozen_probe" if old else "new_frozen_probe",
        within_training_support=legacy.within_training_support(item["coverage"], spec["condition"]),
        metrics=metrics, primary=record["primary"], primary_censored=record["censored"]["primary"]["count"],
        actual_family_exposures=record["actual_family_exposures"], source_family_recency=cp["family_recency"],
        source_clock=legacy.family_clock_metrics(cp, spec["family"]))


def _denominator(row, record):
    return {k: row[k] for k in ("id", "seed", *GROUP, "shrink_step", "evidence", "baseline_id")} | dict(
        primary=record["primary"], censored=record["censored"], opportunities=record["opportunities"],
        post_fork_counters=record["post_fork_counters"], request_diagnosis=record["request_diagnosis"],
        content_by_cohort_phase_age=record["content_by_cohort_phase_age"], inactive_history=record["inactive_history"])


def _new_evaluation(plan, item, sha, cps, inputs, audit):
    path = _input(Path(plan["raw_root"]) / "evaluation" / f"{item['id']}.json", inputs, audit)
    value = _read_json(path)
    _fields(value, EVALUATION_FIELDS, audit, str(path))
    for key, expected in (("format", "organization-acquisition-evaluation-v1"), ("branch", item),
            ("protocol_sha256", sha), ("source_unchanged", True), ("baseline_id", item["baseline_id"]),
            ("birth_hash", cps[0]["birth_hash"])):
        audit.equal("NEW_EVALUATION_BINDING", f"{path}/{key}", value[key], expected)
    audit.equal("NEW_EVALUATED_TICKS", str(path), set(value["states"]), {str(t) for t in plan["checkpoint_ticks"][1:]})
    for tick in plan["checkpoint_ticks"][1:]:
        cp, saved = cps[tick], value["states"][str(tick)]
        expected = dict(checkpoint=cp["checkpoint"], checkpoint_sha256=cp["checkpoint_sha256"],
            state_hash=cp["cpu_semantic_hash"], training_device_normalized_state_hash=cp["semantic_hash"], family_recency=cp["family_recency"])
        audit.equal("NEW_EVALUATION_STATE", f"{path}/{tick}", saved, legacy.normalize(expected))
    specs = plan["evaluation_specs"]
    audit.equal("NEW_FORK_COUNT", str(path), len(value["records"]), len(specs))
    stub = dict(sources=plan["sources"], dt=plan["dt"], batch_size=item["native_config"]["batch_size"])
    rows, denoms = [], []
    for record, spec in zip(value["records"], specs):
        metrics = legacy.audit_native_record(record, cps[spec["tick"]], plan=stub, spec=spec, audit=audit, where=f"{path}/{_spec_key(spec)}")
        audit.equal("METRIC_INVENTORY", str(path), set(metrics), set(METRICS))
        row = _row(item, spec, metrics, record, cps[spec["tick"]], plan)
        rows.append(row)
        denoms.append(_denominator(row, record))
    return rows, denoms


def _baseline_evaluation(plan, checkpoints, csv_path, old_inputs, inputs, audit):
    by_id = {b["baseline_id"]: b for b in plan["branches"]}
    desired = {_spec_key(s) for s in plan["evaluation_specs"]}
    selected = {}
    for row in _csv_rows(csv_path):
        if row["id"] not in by_id or int(row["tick"]) not in plan["checkpoint_ticks"][1:] or row["condition"] not in ("short", "long_both"):
            continue
        identifier = row["id"]
        spec = dict(tick=int(row["tick"]), family=int(row["family"]), condition=row["condition"], steps=int(row["steps"]))
        key = (identifier, *_spec_key(spec))
        audit.require("UNIQUE_BORROWED_PROBE", str(key), key not in selected)
        audit.require("EXPECTED_BORROWED_PROBE", str(key), _spec_key(spec) in desired)
        item = by_id[identifier]
        for name in ("seed", *execution.KEYS):
            audit.equal("BORROWED_CSV_BRANCH", f"{key}/{name}", int(row[name]) if name == "seed" else row[name], item[name])
        selected[key] = row
    audit.equal("EXACT_BORROWED_PROBES", str(csv_path), set(selected),
                {(identifier, *spec) for identifier in by_id for spec in desired})
    rows, denoms = [], []
    for identifier, item in by_id.items():
        paths = [Path(p) for p in old_inputs if Path(p).name == f"{identifier}.json" and Path(p).parent.name == "evaluation"]
        records = {}
        for path in paths:
            _input(path, inputs, audit, old_inputs[str(path)])
            value = _read_json(path)
            origin = "primary" if value["format"] == "lifelong-evaluation-v1" else "supplement" if value["format"] == "lifelong-coverage-evaluation-v1" else None
            audit.require("BORROWED_RAW_EVALUATION_FORMAT", str(path), origin is not None)
            audit.equal("BORROWED_RAW_BRANCH_ID", str(path), value["branch"]["id"], identifier)
            audit.equal("BORROWED_RAW_SOURCE_UNCHANGED", str(path), value["source_unchanged"], True)
            for record in value["records"]:
                if _spec_key(record["probe"]) in desired:
                    key = (identifier, *_spec_key(record["probe"]))
                    audit.require("UNIQUE_BORROWED_RAW_PROBE", str(key), key not in records)
                    records[key] = (record, origin)
        audit.equal("BORROWED_RAW_COVERAGE", identifier, set(records), {(identifier, *key) for key in desired})
        for spec in plan["evaluation_specs"]:
            key = (identifier, *_spec_key(spec))
            record, origin = records[key]
            source = selected[key]
            cp = checkpoints[identifier][spec["tick"]]
            stub = dict(sources=plan["sources"], dt=plan["dt"], batch_size=item["native_config"]["batch_size"])
            metrics = legacy.audit_native_record(record, cp, plan=stub, spec=record["probe"], audit=audit, where=str(key))
            for name in METRICS:
                borrowed = None if source[name] == "" else float(source[name])
                audit.require("BORROWED_CSV_METRIC", f"{key}/{name}", legacy._same(metrics[name], borrowed))
            for name, actual in (("primary", record["primary"]), ("actual_family_exposures", record["actual_family_exposures"]),
                                 ("source_family_recency", cp["family_recency"]), ("source_clock", legacy.family_clock_metrics(cp, spec["family"]))):
                audit.equal("BORROWED_CSV_POPULATION", f"{key}/{name}", json.loads(source[name]), legacy.normalize(actual))
            audit.equal("BORROWED_CSV_CENSORED", str(key), int(source["primary_censored"]), record["censored"]["primary"]["count"])
            audit.equal("BORROWED_CSV_ORIGIN", str(key), source["probe_origin"], origin)
            audit.equal("BORROWED_CSV_PURPOSE", str(key), source["purpose"], record["probe"]["purpose"])
            audit.equal("BORROWED_CSV_ROLE", str(key), source["query_role"], "withheld_family" if spec["family"] == 0 else "other_family")
            audit.equal("BORROWED_CSV_TRAINING_SUPPORT", str(key), source["within_training_support"],
                        str(legacy.within_training_support(item["coverage"], spec["condition"])))
            row = _row(item, spec, metrics, record, cp, plan, old=True)
            row["baseline_id"] = identifier
            rows.append(row)
            denoms.append(_denominator(row, record))
    return rows, denoms


def paired_rows(new, old):
    def index(rows):
        result = {}
        for row in rows:
            key = (row["seed"], *[row[f] for f in GROUP])
            if key in result:
                raise ValueError("duplicate seed/probe in paired evidence")
            result[key] = row
        return result
    left, right = index(new), index(old)
    if set(left) != set(right):
        raise ValueError("missing or foreign paired seed/probe")
    result = []
    for key, row in left.items():
        baseline = right[key]
        if set(row["metrics"]) != set(baseline["metrics"]):
            raise ValueError("paired metric inventory mismatch")
        result.append({k: row[k] for k in ("seed", *GROUP)} | dict(contrast="shrink0_minus_shrink2e-4",
            new_primary=row["primary"], baseline_primary=baseline["primary"],
            new_censored=row["primary_censored"], baseline_censored=baseline["primary_censored"],
            metrics={name: legacy._difference(row["metrics"][name], baseline["metrics"][name]) for name in row["metrics"]}))
    return result


def summaries(rows, *, paired=False):
    fields = GROUP + (("contrast",) if paired else ("shrink_step",))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        if sorted(row["seed"] for row in group) != [0, 1, 2]:
            raise ValueError("statistical group requires exactly seeds 0,1,2")
        result.append(dict(zip(fields, key)) | dict(metrics={name: legacy.summarize_seed_values([
            dict(seed=row["seed"], value=row["metrics"][name]) for row in group]) for name in group[0]["metrics"]}))
    return result


@contextmanager
def _single_thread_environment():
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    before = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _branch_worker(job):
    """One complete branch; CPU models, with the caller's CUDA RNG visibility."""
    plan, item, sha, analysis_sources = job
    audit, inputs = _Checks(), {}
    try:
        torch.set_num_threads(1)
        for path, expected in analysis_sources.items():
            audit.equal("WORKER_ANALYSIS_SOURCE_UNCHANGED", path, legacy.file_sha256(path), expected)
        checkpoints = _new_checkpoints(plan, item, sha, inputs, audit)
        for path, expected in inputs.items():
            audit.equal("WORKER_INPUT_UNCHANGED", path, legacy.file_sha256(path), expected)
        for path, expected in analysis_sources.items():
            audit.equal("WORKER_ANALYSIS_SOURCE_UNCHANGED", path, legacy.file_sha256(path), expected)
        return dict(branch_id=item["id"], checkpoints=checkpoints, inputs=inputs,
                    analysis_sources=analysis_sources, audit=audit.result())
    except AuditError as error:
        return dict(branch_id=item["id"], failure=error.audit, audit=audit.result())
    except Exception as error:
        # Malformed/foreign checkpoints and unexpected worker failures must never
        # turn into missing branches or partial-score artifacts.
        return dict(branch_id=item["id"], failure=dict(errors=[dict(code="ACQUISITION_WORKER_REJECTED",
                    where=item["id"], details=f"{type(error).__name__}: {error}")]), audit=audit.result())


def _new_branch_audits(plan, sha, workers, inputs, audit, analysis_sources):
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError("checkpoint workers must be an integer from one to six")
    jobs = [(plan, item, sha, analysis_sources) for item in plan["branches"]]
    pool = None
    branches = {}
    with _single_thread_environment():
        try:
            if workers == 1:
                results = map(_branch_worker, jobs)
            else:
                # Spawn avoids inheriting a live numerical runtime on platforms
                # where the default process start method would otherwise fork.
                pool = ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"))
                results = pool.map(_branch_worker, jobs)
            for item, result in zip(plan["branches"], results):
                audit.counts.update(result["audit"]["checks_by_code"])
                if "failure" in result:
                    error = result["failure"]["errors"][0]
                    raise AuditError(error["code"], error["where"], error.get("details"),
                                     checks=sum(audit.counts.values()))
                _fields(result, {"branch_id", "checkpoints", "inputs", "analysis_sources", "audit"},
                        audit, f"worker/{item['id']}")
                audit.equal("WORKER_BRANCH_BINDING", item["id"], result["branch_id"], item["id"])
                audit.equal("WORKER_ANALYSIS_BINDING", item["id"], result["analysis_sources"], analysis_sources)
                audit.equal("WORKER_SNAPSHOT_INVENTORY", item["id"], set(result["checkpoints"]), set(plan["checkpoint_ticks"]))
                for path, digest in result["inputs"].items():
                    if path in inputs:
                        audit.equal("WORKER_SHARED_INPUT_BINDING", path, inputs[path], digest)
                    inputs[path] = digest
                branches[item["id"]] = result["checkpoints"]
            audit.equal("WORKER_COMPLETE_BRANCH_INVENTORY", "new checkpoints", set(branches),
                        {item["id"] for item in plan["branches"]})
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    return branches


def build_report(protocol, root, *, mechanical=False, workers=3):
    """mechanical permits small declared fixtures; the CLI always uses formal budgets."""
    torch.set_num_threads(1)
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError("checkpoint workers must be an integer from one to six")
    audit, inputs, root = _Checks(), {}, Path(root).resolve()
    analysis_sources = {str(p.resolve()): legacy.file_sha256(p) for p in (
        Path(__file__), Path(__file__).with_name("test_acquisition_report.py"),
        Path(legacy.__file__), Path(legacy.audit_native_record.__code__.co_filename),
        Path(_Checks.result.__code__.co_filename), Path(legacy._legacy_audit_record.__code__.co_filename))}
    try:
        plan, sha = _protocol(protocol, root, audit, inputs, mechanical)
        old_cps, old_csv, old_inputs = _baseline(plan, inputs, audit, mechanical)
        audit.equal("EXACT_NEW_EVALUATION_FILES", str(root), {p.resolve() for p in (root / "evaluation").glob("*.json")},
                    {root / "evaluation" / f"{b['id']}.json" for b in plan["branches"]})
        new_branches = _new_branch_audits(plan, sha, workers, inputs, audit, analysis_sources)
        new_rows, new_denoms, cp_rows, old_cp_rows, births, exposure_work = [], [], [], [], [], []
        for item in plan["branches"]:
            cps = new_branches[item["id"]]
            old = old_cps[item["baseline_id"]]
            for tick, cp in cps.items():
                audit.equal("MATCHED_WORK_BUDGET", f"{item['id']}/{tick}", cp["work"], old[tick]["work"])
                audit.equal("MATCHED_TOTAL_PARAMETERS", f"{item['id']}/{tick}", cp["total_parameters"], old[tick]["total_parameters"])
                audit.equal("MATCHED_EXOGENOUS_TRAINING", f"{item['id']}/{tick}",
                    (cp["event_sequence_hash"], cp["scheduler_hash"], cp["event_hash"]),
                    (old[tick]["event_sequence_hash"], old[tick]["scheduler_hash"], old[tick]["event_hash"]))
                cp_rows.append(_identity(item) | dict(shrink_step=0., **cp))
                old_cp_rows.append(_identity(item) | dict(shrink_step=2e-4, **old[tick]))
                for shrink, source in ((0., cp), (2e-4, old[tick])):
                    exposure_work.append(_identity(item) | dict(tick=tick, shrink_step=shrink,
                        structure=source["structure"], work=source["work"], actual_family_exposures=source["actual_family_exposures"],
                        family_recency=source["family_recency"], auxiliary_stats=source["auxiliary_stats"]))
            births.append(_identity(item) | dict(baseline_id=item["baseline_id"], equal=True,
                birth_hash=cps[0]["birth_hash"], initial_parameter_hash=cps[0]["initial_parameter_hash"]))
            rows, denoms = _new_evaluation(plan, item, sha, cps, inputs, audit)
            new_rows.extend(rows)
            new_denoms.extend(denoms)
        old_rows, old_denoms = _baseline_evaluation(plan, old_cps, old_csv, old_inputs, inputs, audit)
        count = len(plan["branches"]) * len(plan["evaluation_specs"])
        audit.equal("EXACT_NEW_FORKS", str(root), len(new_rows), count)
        audit.equal("EXACT_BORROWED_FORKS", str(root), len(old_rows), count)
        audit.equal("EXACT_NEW_FULL_CHECKPOINTS", str(root), len(cp_rows), 36)
        paired = paired_rows(new_rows, old_rows)
        for path, expected in list(inputs.items()):
            audit.equal("FINAL_INPUT_UNCHANGED", path, legacy.file_sha256(path), expected)
        for path, expected in analysis_sources.items():
            audit.equal("ANALYSIS_SOURCE_UNCHANGED", path, legacy.file_sha256(path), expected)
        return dict(format=FORMAT, complete=True, protocol=str(Path(protocol).resolve()), protocol_sha256=sha,
            raw_root=str(root), audit=audit.result(), inputs=inputs, analysis_sources=analysis_sources,
            counts=dict(new_training_runs=9, independent_training_seeds=3, new_full_checkpoints=36,
                old_bound_checkpoints=36, old_real_birth_restorations=9, new_evaluation_forks=count,
                borrowed_old_evaluation_forks=count, additional_baseline_training_runs=0),
            per_seed=new_rows, baseline_per_seed=old_rows, paired_per_seed=paired,
            summary=summaries(new_rows + old_rows), paired_summary=summaries(paired, paired=True),
            checkpoints=cp_rows, baseline_checkpoints=old_cp_rows,
            training=[row for row in cp_rows if row["tick"] == plan["target_ticks"]],
            structure_exposure_work=exposure_work, score_denominators=new_denoms + old_denoms, birth_equality=births,
            interpretation=dict(order="Assess main_acquisition first, then post_withdrawal retention and after_return performance.",
                contrast="Within the same seed, architecture, coverage, family, condition, checkpoint and horizon: new shrink0 minus borrowed shrink2e-4.",
                replicate="Three independently trained seeds; forks, actors, families and ticks do not add replicates.",
                denominators="Request maturity, right censoring, idle opportunities, conditional content and cue/first-legal populations stay separate.",
                borrowing="Old 216 frozen probe records are re-audited and matched to immutable audited CSV values; no old training or evaluation is rerun.",
                limits=["A low score before withdrawal is acquisition failure, not evidence of forgetting an acquired skill.",
                    "Narrow long_both is outside its training timing support.",
                    "Shrink also changes capacity and subsequent own-action trajectories; this is a whole-update-policy contrast, not pure topology.",
                    "Paired frozen forks retain different inherited hidden/world states and matured populations."]))
    except AuditError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as error:
        raise AuditError("ACQUISITION_REPORT_REJECTED", str(root), str(error), checks=sum(audit.counts.values())) from error


def write_artifacts(report, output):
    output = Path(output).resolve()
    if report.get("complete") is not True or report.get("audit", {}).get("status") != "passed":
        raise ValueError("only a complete passed acquisition audit can produce artifacts")
    if output.exists():
        raise FileExistsError("acquisition results require a new output directory")
    output.mkdir(parents=True)
    legacy._atomic_json(output / "report.json", report)
    files = {"report.json": legacy.file_sha256(output / "report.json")}
    for name in ("per_seed", "baseline_per_seed", "paired_per_seed", "summary", "paired_summary", "checkpoints",
                 "baseline_checkpoints", "training", "structure_exposure_work", "score_denominators", "birth_equality"):
        rows = []
        for row in report[name]:
            base = {k: v for k, v in row.items() if k != "metrics"}
            if name in ("summary", "paired_summary"):
                rows.extend(base | dict(metric=metric, **summary) for metric, summary in row["metrics"].items())
            else:
                rows.append(base | row.get("metrics", {}))
        legacy._csv(output / f"{name}.csv", rows)
        files[f"{name}.csv"] = legacy.file_sha256(output / f"{name}.csv")
    legacy._atomic_json(output / "artifacts.json", files)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 7),
                        help="parallel complete-branch checkpoint auditors (1 is serial; default 3)")
    args = parser.parse_args(argv)
    try:
        result = build_report(args.protocol, args.root, workers=args.workers)
        write_artifacts(result, args.output)
    except (AuditError, ValueError, OSError) as error:
        parser.exit(1, f"Acquisition audit rejected: {error}\n")
    print(json.dumps(dict(counts=result["counts"], audit=result["audit"], checkpoint_workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
