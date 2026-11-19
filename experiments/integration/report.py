"""Audit and summarize the new integration study without loading a learner.

One seed is one statistical replicate. Autonomous scores use only intervention
``none``; oracle/ablation forks enter a separate diagnostic delta table. Raw
training traces and observer tables stay in SHA256-addressed input files rather
than being duplicated in the aggregate. Missing data may be reported explicitly
with --allow-incomplete; integrity mismatches are never relaxed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


FORMAT_VERSION = "integration-aggregate-v1"
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("frozen", "bp64", "repeat64", "replay64", "accum128", "bp128", "coact_live",
        "coact_mix", "coact_mix_shuffled", "coact_only_mix", "coverage_long_dense", "coverage_short_sparse")
CONDITIONS = ("iid", "long_wait", "longer", "sparse", "long_sparse")
COHORTS = ("primary", "inherited", "carryover_contract")
PHASES = ("unstarted_feasible", "unstarted_missed_start", "processing", "legal_finish")
INTERVENTIONS = ("none", "finish_content_oracle", "timing_oracle",
                 "completion_receipt_ablation", "cancel_command_ablation")
CORE_METRICS = ("strict_completion", "completion", "finish_precision", "own_state_content",
                "own_state_legal_content", "first_legal_content", "inactive_per_1000_idle",
                "inactive_per_1000_total", "first_error_fraction", "cue_content",
                "matched_cue_content", "matched_legal_content", "matched_loss_given_cue_correct")
REGIMES = {
    "iid": ([1., 4.], [.5, 2.]), "long_wait": ([4., 8.], [.5, 2.]),
    "longer": ([8., 16.], [.5, 2.]), "sparse": ([1., 4.], [4., 12.]),
    "long_sparse": ([4., 8.], [4., 12.]),
}
CONTRASTS = (("bp64", "frozen"), ("repeat64", "bp64"), ("replay64", "repeat64"),
             ("accum128", "bp64"), ("bp128", "accum128"), ("coact_live", "bp64"),
             ("coact_mix", "coact_live"), ("coact_mix_shuffled", "coact_mix"),
             ("coact_only_mix", "frozen"), ("coverage_long_dense", "bp64"),
             ("coverage_short_sparse", "bp64"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_text_sha256(path: Path) -> str:
    # Runner snapshots hash read_text() strings; Windows writes may use CRLF.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def semantic_record_digest(record: dict[str, Any]) -> str:
    return json_digest({key: value for key, value in record.items()
                        if key not in {"seconds", "wall_seconds", "elapsed_wall_seconds"}})


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(values)
    numeric = [float(row["value"]) for row in rows if row["value"] is not None]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("nonfinite per-seed statistic")
    return {"n": len(numeric), "n_total": len(rows), "n_undefined": len(rows) - len(numeric),
            "mean": statistics.mean(numeric) if numeric else None,
            "sample_sd": statistics.stdev(numeric) if len(numeric) > 1 else None,
            "per_seed": rows}


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _same_number(actual: Any, expected: float | None, *, rel_tol=1e-10) -> bool:
    return actual is None if expected is None else isinstance(actual, (int, float)) and not isinstance(
        actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-12)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


class Audit:
    """Keep counts of successful checks and details only for failed checks."""
    def __init__(self, allow_incomplete=False):
        self.allow_incomplete = allow_incomplete
        self.counts: Counter = Counter()
        self.issues: list[dict[str, Any]] = []

    def require(self, code: str, where: str, passed: bool, details=None, *, missing=False):
        self.counts[code] += 1
        if not passed:
            self.issues.append({"level": "warning" if missing and self.allow_incomplete else "error",
                                "code": code, "where": where, "details": details})

    def equal(self, code: str, where: str, actual: Any, expected: Any):
        self.require(code, where, actual == expected, {"actual": actual, "expected": expected})

    def result(self):
        errors = sum(item["level"] == "error" for item in self.issues)
        return {"status": "failed" if errors else "incomplete" if self.issues else "passed",
                "error_count": errors, "warning_count": len(self.issues) - errors,
                "check_count": sum(self.counts.values()), "checks_by_code": dict(sorted(self.counts.items())),
                "issues": self.issues}


def _paths(inputs: Iterable[str | Path], pattern: str) -> list[Path]:
    found = set()
    for item in inputs:
        value = str(item)
        matches = [Path(path) for path in glob.glob(value, recursive=True)] if any(
            x in value for x in ("*", "?", "[")) else [Path(value)]
        for path in matches:
            if path.is_dir():
                found.update(child.resolve() for child in path.rglob(pattern) if child.is_file())
            elif path.is_file():
                found.add(path.resolve())
    return sorted(found)


def load_reports(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    output = []
    for path in _paths(inputs, "report.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("format") == "integration-trainer-v1":
            item["_provenance"] = {"path": str(path), "sha256": file_sha256(path)}
            output.append(item)
    return output


def load_evaluations(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    output = []
    for path in _paths(inputs, "*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") not in {"integration-own-trace-v1", "integration-own-trace-v2"}:
            continue
        for item in value.get("evaluations", [value] if "records" in value else []):
            item["_provenance"] = {"path": str(path), "sha256": file_sha256(path)}
            output.append(item)
    return output


def _arm_spec(mode: str, manifest: dict[str, Any]) -> dict[str, Any]:
    live = 128 if mode in {"accum128", "bp128"} else 64
    spec = {"mode": mode, "live_ticks": live, "gradient_horizon": 128 if mode == "bp128" else 64,
            "duration_range": [1., 4.] if mode == "coverage_short_sparse" else [4., 8.],
            "gap_range": [.5, 2.] if mode == "coverage_long_dense" else [4., 12.],
            "local_lr": manifest.get("local_lr", .001), "local_tau": manifest.get("local_tau", 16.)}
    spec.update(manifest.get("arm_configs", {}).get(mode, {}))
    return spec


def _arm_flags(mode: str) -> dict[str, Any]:
    local = mode.startswith("coact_")
    return {"current_gradient": mode not in {"frozen", "coact_only_mix"},
            "extra_gradient": "current" if mode == "repeat64" else "replay" if mode == "replay64" else None,
            "local_source": "live" if mode == "coact_live" else "mix" if local else None,
            "shuffled": mode == "coact_mix_shuffled"}


def _regime_group(mode: str) -> str:
    return mode if mode.startswith("coverage_") else "main"


def _source_maps(manifest):
    sources = manifest.get("source_hashes", {})
    return (sources.get("continuity", manifest.get("continuity_source_hashes", {})),
            sources.get("integration", manifest.get("integration_source_hashes", {})),
            sources.get("diagnose", manifest.get("diagnostic_source_hashes", {})))


def _audit_training(item, manifest, audit, *, verify_files=False):
    seed, mode = int(item["config"]["seed"]), item["integration_config"]["mode"]
    label, config = f"seed_{seed:03d}/{mode}", item["config"]
    spec, flags = _arm_spec(mode, manifest), _arm_flags(mode)
    ticks, batch, jobs, width = (manifest.get("stage_ticks", 65536), manifest.get("batch_size", 64),
                                  manifest.get("n_jobs", 4), manifest.get("width", 128))
    audit.equal("RUN_FORMAT", label, item.get("format"), "integration-trainer-v1")
    audit.equal("ARM_FLAGS", label, item.get("arm"), flags)
    for key, value in spec.items():
        audit.equal("ARM_CONFIG", f"{label}/{key}", item["integration_config"].get(key), value)
    for key, value in (("batch_size", batch), ("n_jobs", jobs), ("train_dts", [manifest.get("dt", .25)]),
                       ("replay_capacity", 32), ("replay_block_ticks", 128), ("replay_burnin", 64)):
        audit.equal("TRAINING_CONFIG", f"{label}/{key}", config.get(key), value)
    for key in ("lr", "wait_weight", "weight_decay", "clip", "max_step"):
        if key in manifest:
            audit.equal("TRAINING_CONFIG", f"{label}/{key}", config.get(key), manifest[key])
    for key, value in (("kind", "learned"), ("state_scalars", width), ("input_size", jobs * 24),
                       ("n_jobs", jobs), ("n_actions", 11),
                       ("total_parameters", manifest.get("total_parameters", 34348))):
        audit.equal("PARAMETER_COUNT", f"{label}/{key}", item["model"].get(key), value)
    audit.require("NO_RESETS", label, item.get("world_resets") == item.get("hidden_resets") == 0)
    audit.require("SOURCE_VARIANT", label, item["variant"].get("policy") == "online" and
                  item["variant"].get("kind") == "learned" and item["variant"].get("width") == width
                  and item["variant"].get("reset_every") == 0 and item.get("stage") == "B")
    audit.equal("ARM_NAME", label, item.get("adaptation_mode"), mode)
    updates = ticks // spec["live_ticks"]
    passes = int(flags["current_gradient"]) + int(flags["extra_gradient"] is not None)
    for key, value in (("stage_ticks", ticks), ("stage_updates", updates),
                       ("stage_gradient_updates", updates * passes),
                       ("local_updates", updates if flags["local_source"] else 0),
                       ("stage_batch_observations", ticks * batch),
                       ("stage_action_labels_observed", ticks * batch * jobs)):
        audit.equal("TRAINING_BUDGET", f"{label}/{key}", item.get(key), value)
    origin = item.get("origin", {})
    for field in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash"):
        audit.require("ORIGIN_HASH", f"{label}/{field}", _hash(origin.get(field)))
    audit.equal("LIFETIME_BUDGET", f"{label}/ticks", item.get("ticks"), origin.get("tick", -1) + ticks)
    audit.equal("LIFETIME_BUDGET", f"{label}/updates", item.get("updates"), origin.get("update", -1) + updates)
    if "origin_ticks" in manifest:
        audit.equal("ORIGIN_BUDGET", label, origin.get("tick"), manifest["origin_ticks"])
    if "origin_updates" in manifest:
        audit.equal("ORIGIN_BUDGET", f"{label}/updates", origin.get("update"), manifest["origin_updates"])
    expected_origin = manifest.get("source_checkpoints", {}).get(str(seed))
    if expected_origin:
        audit.equal("PROTOCOL_ORIGIN", f"{label}/sha256", origin.get("checkpoint_sha256"), expected_origin.get("sha256"))
        audit.equal("PROTOCOL_ORIGIN", f"{label}/path", str(Path(origin.get("checkpoint", "")).resolve()),
                    str(Path(expected_origin["path"]).resolve()))
    if mode == "frozen":
        audit.equal("FROZEN_WEIGHTS", label, item.get("final_parameter_hash"), origin.get("model_hash"))
    has_old = flags["extra_gradient"] == "replay" or flags["local_source"] is not None
    old_samples = updates if has_old else 0
    supervised = (ticks if flags["current_gradient"] else 0) + (
        ticks if flags["extra_gradient"] == "current" else old_samples * 64 if flags["extra_gradient"] == "replay" else 0)
    expected_work = dict(live_forward_ticks=ticks,
                         current_reprocess_forward_ticks=ticks if mode == "repeat64" else 0,
                         old_forward_ticks=old_samples * 128, old_burnin_ticks=old_samples * 64,
                         supervised_ticks=supervised, backward_horizon_ticks=supervised, old_samples=old_samples,
                         local_contributing_ticks=(ticks + old_samples * 64 if flags["local_source"] == "mix"
                                                   else ticks if flags["local_source"] else 0))
    audit.equal("WORK_BUDGET", label, item.get("work"), expected_work)
    audit.equal("PARAMETER_UPDATES", label, item.get("parameter_updates"), {
        "adam_parameters": item["model"]["total_parameters"] if flags["current_gradient"] else 0,
        "local_parameters": width**2 if flags["local_source"] else 0})
    replay = item.get("replay", {})
    audit.require("OLD_EXPERIENCE", label, 0 < replay.get("stored_blocks", 0) <= replay.get("capacity_blocks", 0) == 32
                  and replay.get("block_ticks") == 128 and all(0 <= lo < hi <= origin.get("tick", -1)
                  for lo, hi in replay.get("stored_tick_ranges", [])))
    continuity_sources, integration_sources, _ = _source_maps(manifest)
    for expected, actual, code in ((continuity_sources, item.get("source_hashes", {}), "CONTINUITY_SOURCES"),
                                  (integration_sources, item.get("integration_source_hashes", {}), "INTEGRATION_SOURCES")):
        for name, digest in expected.items():
            audit.equal(code, f"{label}/{name}", actual.get(name), digest)
    audit.equal("INHERITED_SOURCES", label, origin.get("inherited_source_hashes"), item.get("source_hashes"))
    for name, value in item.get("update_norms", {}).items():
        audit.require("UPDATE_NORMS", f"{label}/{name}", isinstance(value, (int, float)) and math.isfinite(value) and value >= 0)
        if (name.startswith("bp_") and not passes) or (name.startswith("local_") and not flags["local_source"]):
            audit.equal("UNUSED_UPDATE_NORMS", f"{label}/{name}", value, 0.)
    for namespace in ("live", "replay"):
        local = item.get(f"local_{namespace}")
        if not flags["local_source"]:
            audit.equal("LOCAL_STATE", f"{label}/{namespace}", local, None)
            continue
        audit.require("LOCAL_STATE", f"{label}/{namespace}", isinstance(local, dict))
        if not isinstance(local, dict):
            continue
        for field, value in (("lr", spec["local_lr"]), ("tau", spec["local_tau"]), ("shuffled", flags["shuffled"])):
            audit.equal("LOCAL_CONFIG", f"{label}/{namespace}/{field}", local["config"].get(field), value)
        for field, value in (("observed_ticks", ticks if namespace == "live" else old_samples * 128),
                             ("plasticity_ticks", ticks if namespace == "live" else old_samples * 64),
                             ("local_updates", updates if namespace == "live" else 0),
                             ("matrix_writes", updates if namespace == "live" and spec["local_lr"] else 0),
                             ("pending_ticks", 0), ("pending_elapsed", 0.),
                             ("history_restarts", 0 if namespace == "live" else old_samples),
                             ("trace_bytes", batch * width * 4),
                             ("buffer_bytes", (batch * width + width**2 + width) * 4)):
            audit.equal("LOCAL_BUDGET", f"{label}/{namespace}/{field}", local.get(field), value)
    trace = [point for point in item.get("trace", []) if point.get("stage") == "B"]
    audit.equal("TRACE_LENGTH", label, len(trace), updates)
    for index, point in enumerate(trace):
        where = f"{label}/chunk_{index + 1}"
        audit.equal("TRACE_BUDGET", where, point.get("stage_ticks"), (index + 1) * spec["live_ticks"])
        audit.equal("TRACE_GRADIENT", where, point.get("gradient_horizon_this_chunk"),
                    spec["gradient_horizon"] if flags["current_gradient"] else 0)
        audit.equal("TRACE_PASSES", where, point.get("gradient_updates_this_chunk"), passes)
        audit.equal("TRACE_LOCAL", where, point.get("local_updates_this_chunk"), int(bool(flags["local_source"])))
        if has_old:
            block_index, start = point.get("old_block_index"), point.get("replay_start_tick")
            ranges = replay.get("stored_tick_ranges", [])
            audit.require("TRACE_OLD_SOURCE", where, isinstance(block_index, int) and not isinstance(block_index, bool)
                          and 0 <= block_index < len(ranges) and start == ranges[block_index][0])
        if flags["local_source"]:
            local = point.get("local") or {}
            audit.equal("TRACE_MIXTURE", where, local.get("source_weights"),
                        [.5, .5] if flags["local_source"] == "mix" else [1.])
            audit.require("TRACE_LOCAL_NORM", where, _same_number(local.get("delta_frob"),
                          local.get("lr", -1) * local.get("direction_frob", -1), rel_tol=2e-6))
    for name, fields in (("bp_recurrent", ("bp_recurrent_delta_frob", "extra_bp_recurrent_delta_frob")),
                         ("bp_all", ("bp_all_delta_frob", "extra_bp_all_delta_frob")),
                         ("local_recurrent", ())):
        values = [float(point["local"]["delta_frob"]) for point in trace if point.get("local")] if not fields else [
            float(point[field]) for point in trace for field in fields if point.get(field) is not None]
        for suffix, expected in (("frob_sum", sum(values)), ("frob_squared_sum", sum(x*x for x in values))):
            audit.require("NORM_TRACE_TOTAL", f"{label}/{name}_{suffix}",
                          _same_number(item.get("update_norms", {}).get(f"{name}_{suffix}"), expected))
    if verify_files and item.get("_provenance"):
        folder = Path(item["_provenance"]["path"]).parent
        for prefix, sources in (("continuity", item.get("source_hashes", {})),
                                ("integration", item.get("integration_source_hashes", {}))):
            for name, digest in sources.items():
                path = folder / "sources" / prefix / name
                audit.require("SOURCE_SNAPSHOT", f"{label}/{prefix}/{name}", path.is_file() and source_text_sha256(path) == digest)


def _audit_record(record, audit, where, manifest):
    intervention = record.get("intervention")
    audit.require("INTERVENTION", where, intervention in INTERVENTIONS)
    audit.equal("ACTION_POLICY", where, record.get("action_policy"),
                "model" if intervention == "none" else "diagnostic_counterfactual")
    audit.require("FROZEN_FORK", where, record.get("world_resets") == record.get("hidden_resets") == record.get("parameter_updates") == 0)
    audit.equal("EVALUATION_FORMAT", where, record.get("format"), manifest.get("evaluation_format", "integration-own-trace-v2"))
    for field, value in (("steps", manifest.get("eval_steps", 2048)), ("dt", manifest.get("dt", .25)),
                         ("batch_size", manifest.get("batch_size", 64)), ("n_jobs", manifest.get("n_jobs", 4)),
                         ("active_jobs", manifest.get("n_jobs", 4))):
        audit.equal("EVALUATION_BUDGET", f"{where}/{field}", record.get(field), value)
    for field in ("start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash",
                  "start_world_state_hash", "start_hidden_hash", "parameter_hash", "source_continuation_hash"):
        audit.require("FORK_HASH", f"{where}/{field}", _hash(record.get(field)))
    expected_regime = REGIMES.get(record["condition"])
    if expected_regime:
        for field, value in zip(("duration_range", "gap_range"), expected_regime):
            audit.equal("EVALUATION_REGIME", f"{where}/{field}", record.get("regime", {}).get(field), value)
        for field, value in (("window_range", [1.5, 2.5]), ("cancel", .2), ("replace", .3)):
            audit.equal("EVALUATION_REGIME", f"{where}/{field}", record.get("regime", {}).get(field), value)
    audit.equal("FORK_TICK_BUDGET", where, record.get("end_tick"), record["fork_tick"] + record["steps"])
    audit.require("FORK_TIME_BUDGET", where, _same_number(record.get("end_time"), record["fork_time"] + record["steps"] * record["dt"]))
    audit.require("ACTOR_IDENTITY", where, len(record.get("actor_ids", [])) == len(set(record.get("actor_ids", []))) == record["batch_size"]
                  and len(record.get("case_ids", [])) == len(set(record.get("case_ids", []))) == record["batch_size"])
    for cohort in COHORTS:
        values = record[cohort]
        n, completed, clean = (values[x] for x in ("matured", "completed", "clean_completed"))
        audit.require("COHORT_COUNTS", f"{where}/{cohort}", all(isinstance(x, int) and not isinstance(x, bool)
                      for x in (n, completed, clean)) and 0 <= clean <= completed <= n)
        for field, numerator in (("completion_rate", completed), ("clean_completion_rate", clean)):
            audit.require("RATE_DENOMINATOR", f"{where}/{cohort}/{field}", _same_number(values.get(field), _rate(numerator, n)))
        audit.require("COHORT_FINGERPRINT", f"{where}/{cohort}", _hash(values.get("request_fingerprint")))
        for field in ("matured", "completed", "clean_completed"):
            actors = values.get("per_actor", {}).get(field, [])
            audit.require("ACTOR_COUNTS", f"{where}/{cohort}/{field}", len(actors) == record["batch_size"] and sum(actors) == values[field])
        issued_field = {"primary": "primary_eligible_issued", "inherited": "initial_eligible_pending",
                        "carryover_contract": "carryover_contract_eligible_issued"}[cohort]
        audit.equal("CENSOR_ACCOUNTING", f"{where}/{cohort}", n + record["censored"][cohort]["count"], record[issued_field])
        detail = record.get("request_diagnosis", {}).get(cohort, {})
        for source, target in (("requests", "matured"), ("completed", "completed"), ("clean_completed", "clean_completed")):
            audit.equal("REQUEST_DIAGNOSIS", f"{where}/{cohort}/{source}", detail.get(source), values[target])
        errors = detail.get("with_recorded_or_inherited_error", -1)
        audit.require("FIRST_ERROR_PARTITION", f"{where}/{cohort}", 0 <= errors <= n and
                      sum(detail.get("first_error_causes", {}).values()) == errors == sum(detail.get("first_error_patterns", {}).values()))
        failures = sum(detail.get(key, -n-1) for key in ("uncompleted_no_begin",
                       "uncompleted_began_but_no_legal_opportunity", "uncompleted_despite_legal_opportunity"))
        audit.equal("FAILURE_PARTITION", f"{where}/{cohort}", failures, n - completed)
        audit.require("FIRST_LEGAL", f"{where}/{cohort}", 0 <= detail.get("first_observed_legal_content_correct", -1)
                      <= detail.get("first_observed_legal_count", -1) <= n)
        if record.get("format") == "integration-own-trace-v2":
            required = ("cue_observed_count", "cue_content_correct_count", "cue_legal_pair_count", "cue_legal_both_correct",
                        "cue_legal_lost", "cue_legal_gained", "cue_legal_both_wrong", "cue_legal_cue_correct",
                        "cue_legal_legal_correct", "cue_without_legal_count", "legal_without_cue_count", "neither_cue_nor_legal_count")
            audit.require("CUE_PAIR_COUNTS", f"{where}/{cohort}", all(isinstance(detail.get(key), int) and
                          not isinstance(detail.get(key), bool) and detail[key] >= 0 for key in required))
            if all(key in detail for key in required):
                pairs = detail["cue_legal_pair_count"]
                audit.equal("CUE_PAIR_PARTITION", f"{where}/{cohort}/pairs", pairs,
                            sum(detail[key] for key in ("cue_legal_both_correct", "cue_legal_lost", "cue_legal_gained", "cue_legal_both_wrong")))
                audit.equal("CUE_PAIR_PARTITION", f"{where}/{cohort}/cue_correct", detail["cue_legal_cue_correct"],
                            detail["cue_legal_both_correct"] + detail["cue_legal_lost"])
                audit.equal("CUE_PAIR_PARTITION", f"{where}/{cohort}/legal_correct", detail["cue_legal_legal_correct"],
                            detail["cue_legal_both_correct"] + detail["cue_legal_gained"])
                audit.equal("CUE_EXCLUSIONS", f"{where}/{cohort}/cue", detail["cue_observed_count"], pairs + detail["cue_without_legal_count"])
                audit.equal("CUE_EXCLUSIONS", f"{where}/{cohort}/legal", detail["first_observed_legal_count"], pairs + detail["legal_without_cue_count"])
                audit.equal("CUE_EXCLUSIONS", f"{where}/{cohort}/all", n, pairs + sum(detail[key] for key in (
                    "cue_without_legal_count", "legal_without_cue_count", "neither_cue_nor_legal_count")))
                audit.require("CUE_CORRECT_SUBSET", f"{where}/{cohort}", detail["cue_legal_cue_correct"] <= detail["cue_content_correct_count"]
                              <= detail["cue_legal_cue_correct"] + detail["cue_without_legal_count"] and
                              detail["cue_legal_legal_correct"] <= detail["first_observed_legal_content_correct"]
                              <= detail["cue_legal_legal_correct"] + detail["legal_without_cue_count"])
                for field, numerator, denominator in (("cue_content_accuracy", detail["cue_content_correct_count"], detail["cue_observed_count"]),
                    ("cue_legal_cue_accuracy", detail["cue_legal_cue_correct"], pairs),
                    ("cue_legal_legal_accuracy", detail["cue_legal_legal_correct"], pairs),
                    ("cue_legal_accuracy_given_cue_correct", detail["cue_legal_both_correct"], detail["cue_legal_cue_correct"])):
                    audit.require("CUE_RATE_DENOMINATOR", f"{where}/{cohort}/{field}", _same_number(detail.get(field), _rate(numerator, denominator)))
    total = record["opportunities"]
    audit.require("NONNEGATIVE_OPPORTUNITIES", where, all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in total.values()))
    counters = record["post_fork_counters"]
    audit.require("FINISH_PRECISION", where, _same_number(record.get("post_fork_finish_precision"),
                  _rate(counters["completed_all"], counters["finish_attempted"])))
    audit.equal("OPPORTUNITIES", where, total["opportunities"], record["steps"] * record["batch_size"] * record["active_jobs"])
    audit.equal("OPPORTUNITY_PARTITION", where, total["active_opportunities"] + total["inactive_opportunities"], total["opportunities"])
    audit.equal("INACTIVE_PARTITION", where, total["inactive_begin_actions"] + total["inactive_finish_actions"], total["inactive_actions"])
    audit.equal("INACTIVE_COUNTER", where, total["inactive_actions"], record["post_fork_counters"]["false_action"] - total["disabled_slot_actions"])
    for field, num, den in (("inactive_actions_per_idle_opportunity", total["inactive_actions"], total["inactive_opportunities"]),
                            ("inactive_actions_per_1000_idle_opportunities", 1000 * total["inactive_actions"], total["inactive_opportunities"]),
                            ("inactive_actions_per_1000_total_opportunities", 1000 * total["inactive_actions"], total["opportunities"])):
        audit.require("INACTIVE_DENOMINATOR", f"{where}/{field}", _same_number(record.get(field), _rate(num, den)))
    for field, totalfield in (("opportunities", "inactive_opportunities"), ("actions", "inactive_actions")):
        audit.equal("IDLE_HISTORY", f"{where}/{field}", sum(row[field] for row in record["inactive_history"]), total[totalfield])
    rows = record["content_by_cohort_phase_age"]
    audit.equal("CONTENT_OPPORTUNITIES", where, sum(row["opportunities"] for row in rows), total["active_opportunities"])
    for row in rows:
        audit.require("CONTENT_COUNTS", where, row["cohort"] in (*COHORTS, "ineligible") and row["phase"] in PHASES
                      and 0 <= row["finish_choice_correct"] <= row["opportunities"] and 0 <= row["first_legal_choice_correct"] <= row["first_legal_opportunities"])
        audit.require("CONTENT_DENOMINATOR", where, _same_number(row.get("conditional_finish_accuracy"),
                      _rate(row["finish_choice_correct"], row["opportunities"])))
        audit.require("FIRST_LEGAL_CONTENT_DENOMINATOR", where, _same_number(row.get("first_legal_conditional_accuracy"),
                      _rate(row["first_legal_choice_correct"], row["first_legal_opportunities"])))
    if intervention == "none":
        audit.equal("UNMODIFIED_ACTIONS", where, record.get("action_hash"), record.get("model_action_hash"))
        audit.equal("UNMODIFIED_OBSERVATIONS", where, record.get("observation_hash"), record.get("raw_observation_hash"))


def _metrics(record):
    content = [row for row in record["content_by_cohort_phase_age"] if row["cohort"] == "primary"]
    legal = [row for row in content if row["phase"] == "legal_finish"]
    detail = record["request_diagnosis"]["primary"]
    return dict(strict_completion=record["primary"]["clean_completion_rate"], completion=record["primary"]["completion_rate"],
                finish_precision=record.get("post_fork_finish_precision"),
                own_state_content=_rate(sum(row["finish_choice_correct"] for row in content), sum(row["opportunities"] for row in content)),
                own_state_legal_content=_rate(sum(row["finish_choice_correct"] for row in legal), sum(row["opportunities"] for row in legal)),
                first_legal_content=_rate(detail["first_observed_legal_content_correct"], detail["first_observed_legal_count"]),
                inactive_per_1000_idle=record["inactive_actions_per_1000_idle_opportunities"],
                inactive_per_1000_total=record["inactive_actions_per_1000_total_opportunities"],
                first_error_fraction=_rate(detail["with_recorded_or_inherited_error"], detail["requests"]),
                cue_content=detail.get("cue_content_accuracy"), matched_cue_content=detail.get("cue_legal_cue_accuracy"),
                matched_legal_content=detail.get("cue_legal_legal_accuracy"),
                matched_loss_given_cue_correct=_rate(detail.get("cue_legal_lost", 0), detail.get("cue_legal_cue_correct", 0)))


def _paired_fields(record):
    fields = ("start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash",
              "actor_ids", "case_ids", "fork_tick", "fork_time", "steps", "dt")
    return {**{field: record[field] for field in fields},
            "cohort_fingerprints": {name: record[name]["request_fingerprint"] for name in COHORTS}}


def aggregate(training_reports: Iterable[dict[str, Any]], evaluations: Iterable[dict[str, Any]], *,
              manifest: dict[str, Any] | None = None, allow_incomplete=False,
              verify_checkpoint_files=False) -> dict[str, Any]:
    manifest = deepcopy(manifest or {})
    seeds, modes = manifest.get("seeds", [0, 1, 2]), manifest.get("arms", list(ARMS))
    conditions = manifest.get("conditions", list(CONDITIONS))
    audit, reports, forks = Audit(allow_incomplete), {}, {}
    audit.require("PROTOCOL_SEEDS", "manifest", bool(seeds) and len(seeds) == len(set(seeds)))
    audit.require("PROTOCOL_ARMS", "manifest", bool(modes) and len(modes) == len(set(modes)) and all(mode in ARMS for mode in modes))
    audit.require("PROTOCOL_CONDITIONS", "manifest", bool(conditions) and len(conditions) == len(set(conditions)) and all(condition in CONDITIONS for condition in conditions))
    if verify_checkpoint_files:
        for name, digest in manifest.get("source_file_hashes", {}).items():
            path = ROOT / name
            audit.require("PROTOCOL_SOURCE_FILE", name, path.is_file() and file_sha256(path) == digest)
    declared_runs = {(row["seed"], row["mode"]): row for row in manifest.get("runs", [])}
    if "runs" in manifest:
        audit.equal("PROTOCOL_RUNS", "manifest", sorted(declared_runs), sorted((seed, mode) for seed in seeds for mode in modes))
        audit.equal("PROTOCOL_RUN_UNIQUENESS", "manifest", len(declared_runs), len(manifest["runs"]))
    inputs = {}
    for item in training_reports:
        key = (int(item["config"]["seed"]), item["integration_config"]["mode"])
        audit.require("EXPECTED_RUN", str(key), key[0] in seeds and key[1] in modes)
        audit.require("DUPLICATE_RUN", str(key), key not in reports)
        reports[key] = item
        _audit_training(item, manifest, audit, verify_files=verify_checkpoint_files)
        if key in declared_runs:
            declared = declared_runs[key]
            audit.equal("PROTOCOL_RUN_CONFIG", str(key), item["integration_config"], declared["integration_config"])
            if item.get("_provenance"):
                audit.equal("PROTOCOL_RUN_PATH", str(key), str(Path(item["_provenance"]["path"]).parent.resolve()), str(Path(declared["output"]).resolve()))
        if item.get("_provenance"):
            inputs[item["_provenance"]["path"]] = item["_provenance"]["sha256"]
    checkpoint_links, evaluation_sources = {}, defaultdict(set)
    for evaluation in evaluations:
        key = (int(evaluation["training_config"]["seed"]), evaluation.get("adaptation_mode"))
        label = f"seed_{key[0]:03d}/{key[1]}"
        audit.require("EXPECTED_EVALUATION", label, key[0] in seeds and key[1] in modes)
        item = reports.get(key)
        audit.require("MISSING_TRAINING_RUN", label, item is not None, missing=True)
        if not item:
            continue
        if evaluation.get("_provenance"):
            inputs[evaluation["_provenance"]["path"]] = evaluation["_provenance"]["sha256"]
            evaluation_sources[key].add(evaluation["_provenance"]["path"])
        for left, right in (("updates", "training_updates"), ("ticks", "training_ticks"), ("variant", "training_variant"),
                            ("model", "model"), ("stage", "training_stage"), ("integration_source_hashes", "checkpoint_source_hashes")):
            audit.equal("REPORT_EVALUATION_LINK", f"{label}/{left}", evaluation.get(right), item.get(left))
        audit.equal("REPORT_EVALUATION_CONFIG", label,
                    {k:v for k,v in evaluation["training_config"].items() if k != "device"},
                    {k:v for k,v in item["config"].items() if k != "device"})
        args = evaluation.get("evaluation_args", {})
        audit.equal("EVALUATION_FORMAT", label, evaluation.get("format"), manifest.get("evaluation_format", "integration-own-trace-v2"))
        audit.equal("STRICT_EVALUATION_SOURCES", label, args.get("strict_sources"), True)
        _, _, expected_sources = _source_maps(manifest)
        for name, digest in expected_sources.items():
            audit.equal("DIAGNOSTIC_SOURCES", f"{label}/{name}", evaluation.get("source_hashes", {}).get(name), digest)
        link = {name: evaluation.get(name) for name in ("checkpoint", "checkpoint_sha256", "checkpoint_state_hash")}
        audit.require("CHECKPOINT_HASH", label, _hash(link.get("checkpoint_sha256")) and _hash(link.get("checkpoint_state_hash")))
        if key in checkpoint_links:
            audit.equal("EVALUATION_CHECKPOINT_PAIR", label, link, checkpoint_links[key])
        else:
            checkpoint_links[key] = link
            if verify_checkpoint_files:
                path = Path(evaluation["checkpoint"])
                audit.require("CHECKPOINT_FILE", label, path.is_file() and file_sha256(path) == evaluation["checkpoint_sha256"])
                if item.get("_provenance"):
                    audit.equal("FINAL_CHECKPOINT_PATH", label, str(path.resolve()),
                                str((Path(item["_provenance"]["path"]).parent / "checkpoint.pt").resolve()))
        declared = {(condition, intervention) for condition in args.get("conditions", []) for intervention in args.get("interventions", [])}
        actual = {(row["condition"], row["intervention"]) for row in evaluation["records"]}
        audit.equal("EVALUATION_ARGUMENTS", label, sorted(actual), sorted(declared))
        audit.equal("EVALUATION_RECORD_UNIQUENESS", label, len(actual), len(evaluation["records"]))
        for row in evaluation["records"]:
            record_key = (*key, row["condition"], row["intervention"])
            where = "/".join(map(str, record_key))
            _audit_record(row, audit, where, manifest)
            audit.require("EXPECTED_CONDITION", where, row["condition"] in conditions)
            audit.equal("PARAMETER_LINK", where, row.get("parameter_hash"), item.get("final_parameter_hash"))
            audit.equal("WORLD_LINK", where, row.get("start_world_state_hash"), item.get("world_hash"))
            audit.equal("EVENT_LINK", where, row.get("start_event_hash"), item.get("event_hash"))
            for field in ("steps", "dt", "age_edges"):
                audit.equal("EVALUATION_ARGUMENTS", f"{where}/{field}", row.get(field), args.get(field))
            if record_key in forks:
                audit.equal("DUPLICATE_FORK_CONTENT", where, semantic_record_digest(row), semantic_record_digest(forks[record_key]))
            else:
                forks[record_key] = row
    for seed in seeds:
        origin = None
        train_groups = defaultdict(list)
        for mode in modes:
            key, label = (seed, mode), f"seed_{seed:03d}/{mode}"
            audit.require("MISSING_TRAINING_RUN", label, key in reports, missing=True)
            for condition in conditions:
                audit.require("MISSING_BASELINE_FORK", f"{label}/{condition}", (*key, condition, "none") in forks, missing=True)
            if mode in manifest.get("diagnostic_arms", []):
                for condition, interventions in manifest.get("diagnostic_interventions", {}).items():
                    for intervention in interventions:
                        audit.require("MISSING_DIAGNOSTIC_FORK", f"{label}/{condition}/{intervention}",
                                      (*key, condition, intervention) in forks, missing=True)
            if key not in reports:
                continue
            item = reports[key]
            if origin is None:
                origin = item["origin"]
            audit.equal("COMPLETE_ORIGIN_PAIRING", label, item["origin"], origin)
            own = [row for (fork_seed, fork_mode, _, intervention), row in forks.items() if fork_seed == seed and fork_mode == mode and intervention == "none"]
            for row in own[1:]:
                for field in ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash", "start_event_hash", "parameter_hash", "source_continuation_hash", "fork_tick", "fork_time"):
                    audit.equal("CONDITION_START_PAIRING", f"{label}/{row['condition']}/{field}", row.get(field), own[0].get(field))
            train_groups[_regime_group(mode)].append(item)
        if verify_checkpoint_files and origin:
            path = Path(origin.get("checkpoint", ""))
            audit.require("ORIGIN_CHECKPOINT_FILE", f"seed_{seed:03d}", path.is_file() and file_sha256(path) == origin.get("checkpoint_sha256"))
            declared = manifest.get("source_checkpoints", {}).get(str(seed))
            if declared:
                report_path = path.parent / "report.json"
                valid = report_path.is_file() and file_sha256(report_path) == declared.get("report_sha256")
                audit.require("ORIGIN_REPORT_FILE", f"seed_{seed:03d}", valid)
                if valid:
                    source = json.loads(report_path.read_text(encoding="utf-8"))
                    for origin_field, source_field in (("model_hash", "final_parameter_hash"), ("world_hash", "world_hash"),
                                                       ("tick", "ticks"), ("update", "updates")):
                        audit.equal("ORIGIN_REPORT_LINK", f"{seed}/{origin_field}", origin.get(origin_field), source.get(source_field))
                    audit.require("ORIGIN_ONLINE128", f"seed_{seed:03d}", source.get("variant", {}).get("name") == "online128"
                                  and source.get("variant", {}).get("policy") == "online" and source.get("stage") == "A"
                                  and source.get("config", {}).get("seed") == seed)
        for group, items in train_groups.items():
            reference = items[0]
            for item in items[1:]:
                mode = item["integration_config"]["mode"]
                for field in ("ticks", "event_hash", "data_seed", "replay"):
                    audit.equal("TRAINING_EVENT_PAIRING", f"{seed}/{group}/{mode}/{field}", item.get(field), reference.get(field))
            for condition in conditions:
                rows = [(item["integration_config"]["mode"], forks.get((seed, item["integration_config"]["mode"], condition, "none"))) for item in items]
                rows = [(mode, row) for mode, row in rows if row]
                for mode, row in rows[1:]:
                    audit.equal("EVALUATION_EVENT_COHORT_PAIRING", f"{seed}/{group}/{mode}/{condition}", _paired_fields(row), _paired_fields(rows[0][1]))
        replay_rows = [reports[(seed, mode)] for mode in modes if (seed, mode) in reports and
                       (mode.startswith("coact_") or mode == "replay64")]
        selections = [[(p.get("old_block_index"), p.get("replay_start_tick")) for p in item.get("trace", []) if p.get("stage") == "B"] for item in replay_rows]
        for item, selection in zip(replay_rows[1:], selections[1:]):
            audit.equal("OLD_SAMPLE_PAIRING", f"{seed}/{item['integration_config']['mode']}", selection, selections[0])
    per_seed, content_rows, error_rows, idle_rows, diagnostic_rows = [], [], [], [], []
    for (seed, mode, condition, intervention), row in sorted(forks.items()):
        base = forks.get((seed, mode, condition, "none"))
        if intervention != "none":
            if "diagnostic_arms" in manifest:
                audit.require("EXPECTED_DIAGNOSTIC_FORK", f"{seed}/{mode}/{condition}/{intervention}",
                              mode in manifest["diagnostic_arms"] and intervention in manifest.get(
                                  "diagnostic_interventions", {}).get(condition, []))
            audit.require("MISSING_PAIRED_BASELINE", f"{seed}/{mode}/{condition}/{intervention}", base is not None, missing=True)
            if base:
                audit.equal("INTERVENTION_EXTERNAL_PAIRING", f"{seed}/{mode}/{condition}/{intervention}", _paired_fields(row), _paired_fields(base))
                for field in ("start_world_state_hash", "start_hidden_hash", "parameter_hash", "source_continuation_hash"):
                    audit.equal("INTERVENTION_STATE_PAIRING", f"{seed}/{mode}/{condition}/{intervention}/{field}", row.get(field), base.get(field))
                audit.equal("INTERVENTION_ACTION_LINK", f"{seed}/{mode}/{condition}/{intervention}", row.get("paired_baseline_action_hash"), base.get("action_hash"))
                after, before = _metrics(row), _metrics(base)
                diagnostic_rows.append(dict(seed=seed, arm=mode, condition=condition, intervention=intervention,
                    classification="diagnostic_counterfactual_only", metrics={name: None if after[name] is None or before[name] is None
                    else after[name] - before[name] for name in CORE_METRICS},
                    primary_matured=row["primary"]["matured"], action_changes=row["opportunities"]["model_action_changes"]))
            continue
        detail, primary = row["request_diagnosis"]["primary"], row["primary"]
        failure_counts = {"clean_completed": primary["clean_completed"], "completed_after_error": primary["completed"] - primary["clean_completed"],
                          **{name: detail[name] for name in ("uncompleted_no_begin", "uncompleted_began_but_no_legal_opportunity", "uncompleted_despite_legal_opportunity")}}
        per_seed.append(dict(seed=seed, arm=mode, condition=condition, metrics=_metrics(row),
            primary={key: primary[key] for key in ("matured", "completed", "clean_completed", "request_fingerprint")},
            censored=deepcopy(row["censored"]), opportunities=deepcopy(row["opportunities"]),
            first_error_causes=deepcopy(detail["first_error_causes"]), first_error_patterns=deepcopy(detail["first_error_patterns"]),
            paired_cue_legal={key: value for key, value in detail.items() if key.startswith("cue_") or key in ("legal_without_cue_count", "neither_cue_nor_legal_count")},
            failure_counts=failure_counts, first_legal_count=detail["first_observed_legal_count"],
            first_legal_correct=detail["first_observed_legal_content_correct"],
            observed_first_error_age_sum=detail["observed_first_error_age_sum"],
            observed_first_error_age_count=detail["observed_first_error_age_count"]))
        for content in row["content_by_cohort_phase_age"]:
            if content["cohort"] == "primary":
                content_rows.append({"seed": seed, "arm": mode, "condition": condition, **deepcopy(content)})
        for cause, count in detail["first_error_causes"].items():
            error_rows.append(dict(seed=seed, arm=mode, condition=condition, cause=cause, count=count, matured=primary["matured"]))
        for idle in row["inactive_history"]:
            idle_rows.append({"seed": seed, "arm": mode, "condition": condition, **deepcopy(idle)})
    lookup = {(row["seed"], row["arm"], row["condition"]): row for row in per_seed}
    summaries, first_error_summary, failure_summary = [], [], []
    for mode in modes:
        for condition in conditions:
            rows = [lookup[(seed, mode, condition)] for seed in seeds if (seed, mode, condition) in lookup]
            summaries.append(dict(arm=mode, condition=condition, metrics={name: summarize(
                {"seed": row["seed"], "value": row["metrics"][name]} for row in rows) for name in CORE_METRICS}))
            for cause in sorted({cause for row in rows for cause in row["first_error_causes"]}):
                first_error_summary.append(dict(arm=mode, condition=condition, cause=cause, metrics={
                    "count": summarize({"seed": row["seed"], "value": row["first_error_causes"].get(cause, 0)} for row in rows),
                    "fraction_of_matured": summarize({"seed": row["seed"], "value": _rate(row["first_error_causes"].get(cause, 0), row["primary"]["matured"])} for row in rows)}))
            for phase in ("clean_completed", "completed_after_error", "uncompleted_no_begin",
                          "uncompleted_began_but_no_legal_opportunity", "uncompleted_despite_legal_opportunity"):
                failure_summary.append(dict(arm=mode, condition=condition, phase=phase, metrics={
                    "count": summarize({"seed": row["seed"], "value": row["failure_counts"][phase]} for row in rows),
                    "fraction_of_matured": summarize({"seed": row["seed"], "value": _rate(row["failure_counts"][phase], row["primary"]["matured"])} for row in rows)}))
    comparisons = []
    for after, before in CONTRASTS:
        if after not in modes or before not in modes:
            continue
        for condition in conditions:
            rows = [(seed, lookup.get((seed, after, condition)), lookup.get((seed, before, condition))) for seed in seeds]
            rows = [(seed, a, b) for seed, a, b in rows if a and b]
            comparisons.append(dict(arm=after, reference=before, condition=condition,
                event_cohort_paired=_regime_group(after) == _regime_group(before),
                metrics={name: summarize({"seed": seed, "value": None if a["metrics"][name] is None or b["metrics"][name] is None
                           else a["metrics"][name] - b["metrics"][name]} for seed, a, b in rows) for name in CORE_METRICS}))
    diagnostics = []
    for mode, condition, intervention in sorted({(row["arm"], row["condition"], row["intervention"]) for row in diagnostic_rows}):
        selected = [row for row in diagnostic_rows if (row["arm"], row["condition"], row["intervention"]) == (mode, condition, intervention)]
        diagnostics.append(dict(arm=mode, condition=condition, intervention=intervention, classification="diagnostic_counterfactual_only",
            metric_deltas={name: summarize({"seed": row["seed"], "value": row["metrics"][name]} for row in selected) for name in CORE_METRICS}))
    training = []
    for key, item in sorted(reports.items()):
        local = [item[name] for name in ("local_live", "local_replay") if item.get(name)]
        bank = item["replay"]["stored_blocks"]
        batch, width, jobs = item["config"]["batch_size"], item["model"]["state_scalars"], item["config"]["n_jobs"]
        training.append(dict(seed=key[0], arm=key[1], integration_config=item["integration_config"], arm_flags=item["arm"],
            stage_ticks=item["stage_ticks"], stage_updates=item["stage_updates"], stage_gradient_updates=item["stage_gradient_updates"],
            local_updates=item["local_updates"], work=item["work"], update_norms=item["update_norms"],
            total_parameters=item["model"]["total_parameters"], parameter_updates=item["parameter_updates"],
            old_bank_tensor_bytes_derived=bank * (128 * batch * (jobs * 24 * 4 + jobs * 8) + batch * width * 4),
            local_tensor_and_rng_bytes=sum(value["buffer_bytes"] + value["private_rng_bytes"] for value in local),
            origin=item["origin"], checkpoint=checkpoint_links.get(key), report_provenance=item.get("_provenance"),
            evaluation_files=sorted(evaluation_sources[key])))
    return {"format": FORMAT_VERSION, "audit": audit.result(), "manifest": manifest,
            "inputs": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.items())],
            "training": training, "per_seed": per_seed, "summary": summaries, "comparisons": comparisons,
            "content_by_phase_age": content_rows, "first_error_counts": error_rows, "inactive_history": idle_rows,
            "first_error_summary": first_error_summary, "failure_summary": failure_summary,
            "diagnostic_per_seed": diagnostic_rows, "diagnostic_deltas": diagnostics,
            "definitions": {"statistical_unit": "one independently trained seed; sample SD, never request-level pseudo-replication",
                "strict_completion": "clean_completed / matured primary requests; right-censored requests excluded",
                "own_state_content": "existing 8-FINISH argmax on active primary opportunities; opportunities are policy-dependent and may include not-yet-matured requests",
                "first_legal_content": "first observed legal opportunity per matured primary request with such an opportunity",
                "matched_loss_given_cue_correct": "lost / (both_correct + lost), on matured requests with both observed cue and first legal readout; no-cue/no-legal/neither counts remain separate. Existing-head readability, not full encoding information.",
                "inactive_rates": "same inactive-action numerator, per1000 idle enabled-slot opportunities and per1000 all enabled-slot opportunities separately",
                "diagnostic_deltas": "intervention minus matched baseline; counterfactual trajectories, never autonomous controller scores",
                "coverage_comparisons": "paired by training seed/origin, but training distribution and event/cohort exposures differ",
                "pure_coactivity": "no label-based updates during this continuation; inherits supervised initial weights; unequal information to BP",
                "storage": "derived old-bank tensor bytes assume float32 observations/states and int64 targets; metadata/optimizer/autograd storage excluded",
                "update_norms": "BP uses realized post-Adam parameter differences. Local delta norms describe the additive float32 update before addition-rounding; call counts and norms do not imply matched FLOPs.",
                "trace": "live EMA persists across chunks/resume; historical zero-start plus64tick burn-in is approximate, not derivative eligibility"}}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _long_summaries(rows, field="metrics", *, diagnostic=False):
    output = []
    for row in rows:
        for metric, value in row[field].items():
            output.append({**{key: item for key, item in row.items() if key != field},
                           "metric": metric, **{key: value[key] for key in ("n", "n_total", "n_undefined", "mean", "sample_sd")},
                           **({"classification": "diagnostic_counterfactual_only"} if diagnostic else {})})
    return output


def render_figures(result: dict[str, Any], folder: Path, *, synthetic=False) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    manifest = result["manifest"]
    modes, conditions = manifest.get("arms", list(ARMS)), manifest.get("conditions", list(CONDITIONS))
    lookup = {(row["arm"], row["condition"]): row for row in result["summary"]}
    seed_lookup = {(row["seed"], row["arm"], row["condition"]): row for row in result["per_seed"]}
    seeds = manifest.get("seeds", [0, 1, 2])
    prefix = "SYNTHETIC LAYOUT CHECK - " if synthetic else ""
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.4, len(modes) * .43)), constrained_layout=True)
    for ax, metric, title in zip(axes, ("strict_completion", "completion"), ("No-error completion", "Ordinary completion")):
        data = np.full((len(modes), len(conditions)), np.nan)
        for i, mode in enumerate(modes):
            for j, condition in enumerate(conditions):
                value = lookup.get((mode, condition), {}).get("metrics", {}).get(metric, {})
                if value.get("mean") is not None:
                    data[i, j] = 100 * value["mean"]
        heat = ax.imshow(np.ma.masked_invalid(data), vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for i, mode in enumerate(modes):
            for j, condition in enumerate(conditions):
                value = lookup.get((mode, condition), {}).get("metrics", {}).get(metric, {})
                mean, sd = value.get("mean"), value.get("sample_sd")
                if mean is not None:
                    label = f"{100*mean:.1f}" + (f"\n+/- {100*sd:.1f}" if sd is not None else "")
                    ax.text(j, i, label, ha="center", va="center", fontsize=7,
                            color="white" if mean < .56 else "black")
        ax.set_xticks(range(len(conditions)), [name.replace("_", "\n") for name in conditions])
        ax.set_yticks(range(len(modes)), modes)
        ax.set_title(title + " (%)")
    fig.colorbar(heat, ax=axes, shrink=.85, label="Mean across training seeds (%)")
    fig.suptitle(prefix + "Autonomous controller scores; mean +/- sample SD")
    path = folder / "control_conditions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    selected = [mode for mode in ("frozen", "bp64", "repeat64", "replay64", "coact_live", "coact_mix", "coact_mix_shuffled", "coact_only_mix") if mode in modes]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.1), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for i, mode in enumerate(selected):
        old = lookup.get((mode, "iid"), {}).get("metrics", {}).get("strict_completion", {})
        new = lookup.get((mode, "long_sparse"), {}).get("metrics", {}).get("strict_completion", {})
        if old.get("mean") is None or new.get("mean") is None:
            continue
        axes[0].errorbar(100*old["mean"], 100*new["mean"], xerr=100*(old.get("sample_sd") or 0),
                         yerr=100*(new.get("sample_sd") or 0), marker="o", capsize=3, color=colors(i), label=mode)
    axes[0].set(xlim=(-2, 102), ylim=(-2, 102), xlabel="IID no-error completion (%)",
                ylabel="Long + sparse no-error completion (%)", title="Retention and adaptation at the same lifetime")
    if axes[0].lines:
        axes[0].legend(fontsize=8, loc="best")
    training = {(row["seed"], row["arm"]): row for row in result["training"]}
    for offset, label, norm_field, count_field in ((-.18, "BP recurrent change / BP step", "bp_recurrent_frob_sum", "stage_gradient_updates"),
                                                  (.18, "Local additive delta / application", "local_recurrent_frob_sum", "local_updates")):
        means, deviations = [], []
        for mode in selected:
            values = [_rate(training[(seed, mode)]["update_norms"][norm_field], training[(seed, mode)][count_field])
                      for seed in seeds if (seed, mode) in training]
            values = [value for value in values if value is not None]
            means.append(statistics.mean(values) if values else 0)
            deviations.append(statistics.stdev(values) if len(values) > 1 else 0)
        axes[1].bar(np.arange(len(selected)) + offset, means, .34, yerr=deviations, capsize=2, label=label)
    axes[1].set_xticks(range(len(selected)), selected, rotation=48, ha="right")
    axes[1].set(ylabel="Recurrent update magnitude (Frobenius norm)", title="Update scale audit (different compute budgets)")
    axes[1].legend(fontsize=8)
    fig.suptitle(prefix + "Local activity updates, historical participation and shuffled control")
    path = folder / "local_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, grid = plt.subplots(2, 2, figsize=(16.4, 11.5), constrained_layout=True)
    axes = list(grid.flat)
    categories = ("clean_completed", "completed_after_error", "uncompleted_no_begin",
                  "uncompleted_began_but_no_legal_opportunity", "uncompleted_despite_legal_opportunity")
    labels = ("Clean completion", "Completed after error", "No accepted BEGIN", "Began, no legal opportunity", "Had legal opportunity, incomplete")
    left = np.zeros(len(selected))
    for index, (category, label) in enumerate(zip(categories, labels)):
        values = []
        for mode in selected:
            per_seed_values = [_rate(seed_lookup[(seed, mode, "long_sparse")]["failure_counts"][category],
                                    seed_lookup[(seed, mode, "long_sparse")]["primary"]["matured"])
                               for seed in seeds if (seed, mode, "long_sparse") in seed_lookup]
            values.append(100*statistics.mean([value for value in per_seed_values if value is not None]) if any(value is not None for value in per_seed_values) else 0)
        axes[0].barh(selected, values, left=left, label=label, color=colors(index))
        left += values
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 100), xlabel="Matured primary requests (%)", title="Long + sparse request outcomes")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.1), fontsize=7)
    causes = ("early", "stale_version", "duplicate_begin", "before_begin", "wrong_payload", "late", "other")
    left = np.zeros(len(selected))
    for index, cause in enumerate(causes):
        values = []
        for mode in selected:
            ratios = []
            for seed in seeds:
                row = seed_lookup.get((seed, mode, "long_sparse"))
                if row:
                    count = row["first_error_causes"].get(cause, 0) if cause != "other" else sum(
                        value for key, value in row["first_error_causes"].items() if key not in causes[:-1])
                    value = _rate(count, row["primary"]["matured"])
                    if value is not None:
                        ratios.append(value)
            values.append(100*statistics.mean(ratios) if ratios else 0)
        axes[1].barh(selected, values, left=left, label=cause, color=colors(index))
        left += values
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 100), xlabel="Matured primary requests (%)", title="First rejected attempt (fixed cause priority)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncols=2, fontsize=7)
    rescue = [row for row in result["diagnostic_deltas"] if row["condition"] == "long_sparse" and
              row["intervention"] in {"finish_content_oracle", "timing_oracle"}]
    if rescue:
        labels = [row["arm"] + " / " + ("content" if row["intervention"] == "finish_content_oracle" else "timing") for row in rescue]
        means = [100*row["metric_deltas"]["strict_completion"]["mean"] if row["metric_deltas"]["strict_completion"]["mean"] is not None else 0 for row in rescue]
        deviations = [100*(row["metric_deltas"]["strict_completion"].get("sample_sd") or 0) for row in rescue]
        axes[2].barh(labels, means, xerr=deviations, capsize=3, color=["#E6A23C" if row["intervention"] == "finish_content_oracle" else "#4C78A8" for row in rescue])
        axes[2].invert_yaxis()
        axes[2].axvline(0, color="black", linewidth=.8)
        axes[2].set(xlabel="Change in no-error completion (percentage points)", title="Diagnostic counterfactual rescue only")
    else:
        axes[2].text(.5, .5, "No prespecified rescue results available", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()
    paired, paired_means, paired_sd = [], [], []
    for mode in selected:
        value = lookup.get((mode, "long_sparse"), {}).get("metrics", {}).get("matched_loss_given_cue_correct", {})
        if value.get("mean") is not None:
            paired.append(mode)
            paired_means.append(100*value["mean"])
            paired_sd.append(100*(value.get("sample_sd") or 0))
    if paired:
        axes[3].barh(paired, paired_means, xerr=paired_sd, capsize=3, color="#B279A2")
        axes[3].invert_yaxis()
        axes[3].set(xlim=(0, 100), xlabel="Lost / cue-correct matched requests (%)",
                    title="Own-head loss: matched cue to first legal opportunity")
        axes[3].text(.5, -.13, "No-cue / no-legal / neither exclusions are reported separately.\nReadability of the existing head does not measure all encoded information.",
                     ha="center", va="top", fontsize=8, transform=axes[3].transAxes)
    else:
        axes[3].text(.5, .5, "No matched cue / first-legal observations", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_axis_off()
    fig.suptitle(prefix + "Error phases and paired rescue diagnostics; oracle scores are not autonomous performance")
    path = folder / "error_phases_rescue.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def write_artifacts(result: dict[str, Any], folder: str | Path, *, figures=True, synthetic=False) -> dict[str, str]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for row in result["per_seed"]:
        seed_rows.append({key: row[key] for key in ("seed", "arm", "condition")} | row["metrics"] |
                         {"primary_"+key: row["primary"][key] for key in ("matured", "completed", "clean_completed")} |
                         {key: row[key] for key in ("first_legal_count", "first_legal_correct", "failure_counts", "first_error_causes", "first_error_patterns", "paired_cue_legal")})
    budget_rows = []
    for row in result["training"]:
        budget_rows.append({key: row[key] for key in ("seed", "arm", "stage_ticks", "stage_updates", "stage_gradient_updates", "local_updates",
                                                     "total_parameters", "old_bank_tensor_bytes_derived", "local_tensor_and_rng_bytes")} |
                          {key: row["integration_config"][key] for key in ("live_ticks", "gradient_horizon", "duration_range", "gap_range", "local_lr", "local_tau")} |
                          row["arm_flags"] | row["work"] | row["update_norms"])
    tables = {
        "seed_metrics.csv": seed_rows,
        "metric_summary.csv": _long_summaries(result["summary"]),
        "budgets.csv": budget_rows,
        "paired_comparisons.csv": _long_summaries(result["comparisons"]),
        "diagnostic_delta_summary.csv": _long_summaries(result["diagnostic_deltas"], "metric_deltas", diagnostic=True),
        "diagnostic_per_seed.csv": result["diagnostic_per_seed"],
        "own_state_content_by_phase_age.csv": result["content_by_phase_age"],
        "first_error_counts.csv": result["first_error_counts"],
        "first_error_summary.csv": _long_summaries(result["first_error_summary"]),
        "failure_summary.csv": _long_summaries(result["failure_summary"]),
        "inactive_history.csv": result["inactive_history"],
    }
    output, table_files = {}, []
    for name, rows in tables.items():
        path = folder / name
        _write_csv(path, rows)
        output[name] = str(path.resolve())
        table_files.append({"path": name, "sha256": file_sha256(path), "rows": len(rows)})
    compact = {key: value for key, value in result.items() if key not in
               ("content_by_phase_age", "first_error_counts", "inactive_history")}
    compact["table_files"] = table_files
    compact["synthetic_layout_check"] = synthetic
    if figures:
        paths = render_figures(result, folder / "figures", synthetic=synthetic)
        compact["figures"] = [{"path": str(path.relative_to(folder)), "sha256": file_sha256(path)} for path in paths]
    atomic_json(folder / "summary.json", compact)
    atomic_json(folder / "audit.json", result["audit"])
    return {**output, "summary.json": str((folder / "summary.json").resolve()), "audit.json": str((folder / "audit.json").resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True, help="Training root(s) containing report.json")
    parser.add_argument("--evaluations", nargs="+", required=True, help="Primary and diagnostic JSON files/directories")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-checkpoint-files", action="store_true", help="Skip disk snapshot/checkpoint byte verification; content audits still apply")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = aggregate(load_reports(args.root), load_evaluations(args.evaluations), manifest=manifest,
                       allow_incomplete=args.allow_incomplete, verify_checkpoint_files=not args.skip_checkpoint_files)
    result["manifest_provenance"] = {"path": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest)}
    if result["audit"]["error_count"]:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "audit.json", result["audit"])
        print(json.dumps({"audit": result["audit"]["status"], "errors": result["audit"]["error_count"],
                          "path": str((args.output / "audit.json").resolve())}))
        return 1
    write_artifacts(result, args.output, figures=not args.no_figures)
    print(json.dumps({"audit": result["audit"]["status"], "runs": len(result["training"]),
                      "baseline_seed_conditions": len(result["per_seed"]), "diagnostic_forks": len(result["diagnostic_per_seed"]),
                      "summary": str((args.output / "summary.json").resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
