"""Strict audit and paired summaries of the sparse-B factor continuation study.

Only complete, source-bound continuations are published. Checkpoint workers use
independent processes, preserve their RNG state and return compact audit data.
The trained seed, never an evaluation opportunity, is the statistical replicate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.integration.diagnose import IDLE_CAUSES, _age_labels
from experiments.integration.report import _audit_record, _metrics
from experiments.loops.extension_report import (
    _adam, _exogenous, _expected_work, _input, _normalized_semantic_hash,
    _snapshot_hashes, _summaries, json_hash, normalize,
)
from experiments.loops.report import (
    AuditError, _Checks, _atomic_text, _close, _finite, _hash, _integer,
    _read_json, _verify_files, file_sha256,
)
from experiments.rewiring.report import _jsonl
from experiments.rewiring.plasticity import tensor_hash
from experiments.rewiring.run import PARAMETER_NAMES, RewireTrainer


FORMAT_VERSION = "factors-study-summary-v1"
SEEDS = (0, 1, 2)
MODES = ("dense", "dense_shrink", "fixed", "fixed_shrink", "retire", "grow")
IDLE_LAMBDAS = (0., .1, 1.)
CONDITIONS = ("iid", "long_wait", "longer", "sparse", "long_sparse")
STRUCTURAL_CONTRASTS = (("dense_shrink", "dense"), ("fixed_shrink", "fixed"),
    ("fixed", "dense"), ("fixed_shrink", "dense_shrink"), ("retire", "fixed_shrink"), ("grow", "retire"))
COHORTS = ("primary", "inherited", "carryover_contract")
BUCKETS = ("idle", "active_wait", "legal_begin", "legal_finish", "disabled")
IDLE_COUNT_FIELDS = ("opportunities", "actions", "begin_actions", "finish_actions",
    "finish_matches_retired_content", "known_retired_content_opportunities",
    "conditional_finish_matches_retired_content", "model_wait_preferences",
    "model_finish_preferences", "finish_preferred_to_wait", "retired_exact_finish_preferred_to_wait",
    "retired_content_readout_with_wait")


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _difference(left, right):
    return left - right if left is not None and right is not None else None


def _loss_close(actual, expected):
    # The differentiated objective is float32; raw diagnostics independently
    # recompute log-softmax in float64. A relative-only check fails as the idle
    # NLL approaches zero even when the absolute discrepancy is only a few e-9.
    return (_finite(actual) and _finite(expected)
        and abs(actual-expected) <= max(2e-7, 3e-6 * max(abs(actual), abs(expected))))


def _diagnostic_metrics(record):
    result = _metrics(record)
    counts = record["opportunities"]
    detail = record["request_diagnosis"]["primary"]
    result.update(idle_finish_per_1000_idle=_rate(1000 * counts["inactive_finish_actions"], counts["inactive_opportunities"]),
        idle_begin_per_1000_idle=_rate(1000 * counts["inactive_begin_actions"], counts["inactive_opportunities"]),
        idle_opportunity_fraction=_rate(counts["inactive_opportunities"], counts["opportunities"]),
        uncompleted_no_begin_fraction=_rate(detail["uncompleted_no_begin"], detail["requests"]),
        uncompleted_no_legal_opportunity_fraction=_rate(detail["uncompleted_began_but_no_legal_opportunity"], detail["requests"]),
        uncompleted_despite_legal_fraction=_rate(detail["uncompleted_despite_legal_opportunity"], detail["requests"]),
        active_finish_attempt_rate=_rate(counts["active_finish_attempts"], counts["active_opportunities"]),
        illegally_timed_finish_attempt_rate=_rate(counts["illegally_timed_finish_attempts"], counts["active_opportunities"]))
    return result


def _idle_metrics(row):
    return dict(idle_action_rate=_rate(row["actions"], row["opportunities"]),
        idle_finish_rate=_rate(row["finish_actions"], row["opportunities"]),
        idle_begin_rate=_rate(row["begin_actions"], row["opportunities"]),
        idle_error_finish_fraction=_rate(row["finish_actions"], row["actions"]),
        idle_error_begin_fraction=_rate(row["begin_actions"], row["actions"]),
        finish_matches_retired_content_rate=_rate(row["finish_matches_retired_content"], row["finish_actions"]),
        retained_content_readout_rate=_rate(row["conditional_finish_matches_retired_content"], row["known_retired_content_opportunities"]),
        model_wait_preference_rate=_rate(row["model_wait_preferences"], row["opportunities"]),
        model_finish_preference_rate=_rate(row["model_finish_preferences"], row["opportunities"]),
        finish_preferred_to_wait_rate=_rate(row["finish_preferred_to_wait"], row["opportunities"]),
        retired_exact_finish_preferred_to_wait_rate=_rate(row["retired_exact_finish_preferred_to_wait"], row["known_retired_content_opportunities"]),
        retired_content_readout_with_wait_rate=_rate(row["retired_content_readout_with_wait"], row["known_retired_content_opportunities"]),
        mean_finish_minus_wait_logit=_rate(row["finish_minus_wait_logit_sum"], row["opportunities"]))


def _idle_history(record, audit, where):
    """Check disjoint cause/age populations and retain zero-denominator cells."""
    labels = tuple(_age_labels(tuple(record["age_edges"]))) + ("unknown",)
    expected = {(cause, label) for cause in IDLE_CAUSES for label in labels}
    found = {}
    for row in record["inactive_history"]:
        key = row.get("cause"), row.get("time_since_terminal_bin")
        audit.require("IDLE_HISTORY_CELL", where, key in expected and key not in found)
        audit.require("IDLE_HISTORY_COUNTS", f"{where}/{key}",
            all(_integer(row.get(field)) for field in IDLE_COUNT_FIELDS))
        n, known = row["opportunities"], row["known_retired_content_opportunities"]
        audit.require("IDLE_HISTORY_SUBSETS", f"{where}/{key}",
            row["actions"] == row["begin_actions"] + row["finish_actions"] <= n
            and row["finish_matches_retired_content"] <= min(row["finish_actions"], known)
            and known <= n
            and all(row[field] <= n for field in ("model_wait_preferences", "model_finish_preferences", "finish_preferred_to_wait"))
            and all(row[field] <= known for field in ("conditional_finish_matches_retired_content",
                "retired_exact_finish_preferred_to_wait", "retired_content_readout_with_wait")))
        audit.require("IDLE_HISTORY_MARGIN", f"{where}/{key}", _finite(row.get("finish_minus_wait_logit_sum")))
        for field, expected_value in (("action_rate_per_idle_opportunity", _rate(row["actions"], n)),
                ("actions_per_1000_total_opportunities", _rate(1000 * row["actions"], record["opportunities"]["opportunities"])),
                ("mean_finish_minus_wait_logit", _rate(row["finish_minus_wait_logit_sum"], n))):
            audit.require("IDLE_HISTORY_DENOMINATOR", f"{where}/{key}/{field}", row.get(field) == expected_value
                if expected_value is None else _close(row.get(field), expected_value, tolerance=1e-10))
        found[key] = row
    for field, totalfield in (("opportunities", "inactive_opportunities"), ("actions", "inactive_actions"),
                             ("begin_actions", "inactive_begin_actions"), ("finish_actions", "inactive_finish_actions"),
                             ("finish_matches_retired_content", "inactive_finish_matches_retired_content")):
        audit.equal("IDLE_HISTORY_TOTAL", f"{where}/{field}", sum(row[field] for row in found.values()), record["opportunities"][totalfield])
    output = []
    for cause, label in sorted(expected):
        row = found.get((cause, label), {**dict.fromkeys(IDLE_COUNT_FIELDS, 0), "finish_minus_wait_logit_sum": 0.})
        output.append({"cause": cause, "time_since_terminal_bin": label,
            **{field: row[field] for field in (*IDLE_COUNT_FIELDS, "finish_minus_wait_logit_sum")}, "metrics": _idle_metrics(row)})
    return output


def _diagnostics(value, audit, where, *, opportunities=None):
    """Recompute every loss denominator from raw before-action bucket totals."""
    for name in ("base_weight_mass", "base_weighted_ce_sum", "idle_nll_sum"):
        audit.require("LOSS_FINITE", f"{where}/{name}", _finite(value.get(name), 0))
    audit.require("IDLE_LOSS_COUNT", where, _integer(value.get("idle_count")))
    buckets = value["buckets"]
    audit.equal("LOSS_BUCKETS", where, set(buckets), set(BUCKETS))
    for name, row in buckets.items():
        for field in ("count", "actual_nonwait_count"):
            audit.require("LOSS_BUCKET_COUNT", f"{where}/{name}/{field}", _integer(row.get(field)))
        audit.require("LOSS_BUCKET_SUBSET", f"{where}/{name}", row["actual_nonwait_count"] <= row["count"])
        for field in ("weight_mass", "weighted_ce_sum", "target_nll_sum", "p_wrong_sum"):
            audit.require("LOSS_BUCKET_FINITE", f"{where}/{name}/{field}", _finite(row.get(field), 0))
        for field in ("target_margin_sum", "wait_margin_sum"):
            audit.require("LOSS_BUCKET_FINITE", f"{where}/{name}/{field}", _finite(row.get(field)))
        audit.require("LOSS_PROBABILITY", f"{where}/{name}", row["p_wrong_sum"] <= row["count"] + 1e-8)
        weight = .1 if name in ("idle", "active_wait", "disabled") else 1.
        audit.require("LOSS_BUCKET_WEIGHT", f"{where}/{name}", _close(row["weight_mass"], row["count"] * weight, tolerance=3e-8))
        audit.require("LOSS_BUCKET_WEIGHTED_NLL", f"{where}/{name}",
            _close(row["weighted_ce_sum"], row["target_nll_sum"] * weight, tolerance=3e-8))
    audit.equal("IDLE_BEFORE_ACTION_COUNT", where, value["idle_count"], buckets["idle"]["count"])
    audit.require("IDLE_NLL_DENOMINATOR", where, _close(value["idle_nll_sum"], buckets["idle"]["target_nll_sum"], tolerance=1e-10))
    for total, field in (("base_weight_mass", "weight_mass"), ("base_weighted_ce_sum", "weighted_ce_sum")):
        audit.require("BASE_LOSS_DENOMINATOR", f"{where}/{total}", _close(value[total], sum(row[field] for row in buckets.values()), tolerance=1e-9))
    if opportunities is not None:
        audit.equal("LOSS_ALL_OPPORTUNITIES", where, sum(row["count"] for row in buckets.values()), opportunities)
    return dict(base_weighted_ce=_rate(value["base_weighted_ce_sum"], value["base_weight_mass"]),
        idle_nll_mean=_rate(value["idle_nll_sum"], value["idle_count"]),
        idle_count=value["idle_count"], base_weight_mass=value["base_weight_mass"],
        buckets={name: {**row, "mean_nll": _rate(row["target_nll_sum"], row["count"]),
            "mean_p_wrong": _rate(row["p_wrong_sum"], row["count"]),
            "actual_nonwait_rate": _rate(row["actual_nonwait_count"], row["count"]),
            "mean_target_margin": _rate(row["target_margin_sum"], row["count"]),
            "mean_wait_margin": _rate(row["wait_margin_sum"], row["count"])} for name, row in buckets.items()})


def _merge_audit(audit, result):
    audit.require("WORKER_AUDIT_PASSED", "checkpoint worker", result["status"] == "passed" and not result["errors"])
    audit.counts.update(result["checks_by_code"])


def _worker_init():
    torch.set_num_threads(1)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _map_workers(function, jobs, workers):
    if not _integer(workers, 1) or workers > 6:
        raise ValueError("workers must be an integer from one through six")
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        return list(executor.map(function, jobs))


def _source(path, training_device, audit, inputs):
    path, digest = _input(path, inputs, audit, "SOURCE_CHECKPOINT")
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
        for namespace, texts, hashes in (("continuity", trainer.base.sources, trainer.base.source_hashes),
                ("credit", trainer.sources, trainer.source_hashes), ("credit-dependencies", trainer.dependencies, trainer.dependency_hashes),
                ("rewiring", trainer.rewire_sources, trainer.rewire_source_hashes),
                ("rewiring-dependencies", trainer.rewire_dependencies, trainer.rewire_dependency_hashes)):
            _snapshot_hashes(texts, hashes, audit, f"{path}/{namespace}")
        value = trainer.semantic_state()
        semantic_hash = _normalized_semantic_hash(value, training_device)
        credit, base = value["credit"], value["credit"]["continuation"]
        width = base["variant"]["width"]
        operator = trainer.operator.summary()
        audit.require("SOURCE_SPARSE", str(path), 0 < operator["active"] < width * width)
        audit.equal("SOURCE_ARM", str(path), value["rewire_config"],
            dict(arm="retire", shrink_step=.0002, window_updates=8, structural_seed=91000 + base["config"]["seed"]))
        audit.require("SOURCE_B_BOUNDARY", str(path), trainer.stage_ticks == value["phase_ticks"]
            and not value["phase_returned"] and value["credit"]["continuation"]["hidden_resets"] == 0)
        audit.equal("SOURCE_EXACT_SUPPORT", str(path), operator["active"], int(base["model"]["recurrent"].ne(0).sum()))
        adam = _adam(base["optimizer"], base["model"], [PARAMETER_NAMES], audit, str(path))
        config = normalize(base["config"])
        config["device"] = training_device
        result = dict(path=str(path), sha256=digest, semantic_hash=semantic_hash,
            actual_cpu_semantic_hash=state_digest(value), config=config, variant=normalize(base["variant"]),
            tick=base["ticks"], update=base["updates"], gradient_updates=base["gradient_updates"],
            stage_ticks=trainer.stage_ticks, stage_updates=trainer.stage_updates,
            phase_start_tick=base["phase_start_tick"], phase_start_update=base["phase_start_update"],
            phase_start_gradient=base["phase_start_gradient"], world_tick=base["world"]["tick"], world_now=base["world"]["now"],
            world_spec=normalize(base["world"]["spec"]), world_regime=normalize(base["world"]["regime"]),
            world_regime_history_hash=state_digest(base["world"]["regime_history"]),
            model_config=trainer.model.config(), batch_size=base["config"]["batch_size"], n_jobs=base["config"]["n_jobs"],
            model_shapes={name: list(tensor.shape) for name, tensor in base["model"].items()},
            optimizer_groups=normalize(base["optimizer"]["param_groups"]), adam=adam,
            hashes={name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng")},
            trace_count=len(base["trace"]), trace_json_hash=json_hash(base["trace"]),
            trace_event_sequence_hash=json_hash([row["event_hash"] for row in base["trace"]]),
            action_hash=base["action_hash"], data_hash=base["data_hash"], event_hash=base["world"]["event_hash"],
            initial_parameter_hash=base["initial_parameter_hash"], data_seed=base["data_seed"],
            replay_rng_hash=state_digest(credit["sampler_rng"]), work=deepcopy(credit["work"]),
            update_norms=deepcopy(credit["update_norms"]), credit_config=normalize(credit["credit_config"]),
            credit_origin=normalize(credit["origin"]), regime_changes=normalize(credit["regime_changes"]),
            inherited_rewiring_hashes={key: state_digest(value[key]) for key in
                ("format", "rewire_config", "origin", "optimizer_parameter_names", "phase_ticks", "phase_returned", "return_regime", "structural_windows")},
            return_regime=normalize(value["return_regime"]),
            operator_hash=state_digest(value["rewiring"]), operator_hashes={key: state_digest(item) for key, item in value["rewiring"].items()},
            inherited_rewiring_origin_sha256=value["origin"]["checkpoint_sha256"],
            structure=operator, inherited_source_hashes=deepcopy(trainer.rewire_source_hashes),
            inherited_dependency_hashes=deepcopy(trainer.rewire_dependency_hashes),
            stats_hash=state_digest(base["stats"]), stats_count=int(base["stats"]["confusion"].sum()), seconds=trainer.seconds)
        result["hashes"].update(hidden_fork=state_digest({"hidden": base["hidden"]}),
            scheduler=state_digest(base["world"]["scheduler"]),
            source_continuation=state_digest({"model": base["model"], "world": base["world"], "hidden": base["hidden"]}))
        audit.equal("SOURCE_BYTES_STABLE", str(path), file_sha256(path), digest)
        return result


def _binding(checkpoint):
    return {key: checkpoint[key] for key in ("checkpoint", "checkpoint_sha256", "state_hash", "training_device_normalized_state_hash")}


def _record(record, cp, plan, audit, where):
    _audit_record(record, audit, where, dict(eval_steps=plan["evaluation_steps"], dt=plan["dt"],
        batch_size=cp["batch_size"], n_jobs=cp["n_jobs"]))
    for field in ("world_resets", "hidden_resets", "parameter_updates"):
        audit.require("FROZEN_FORK", f"{where}/{field}", _integer(record.get(field)) and record[field] == 0)
    audit.equal("OWN_ACTION_POLICY", where, (record.get("intervention"), record.get("action_policy")), ("none", "model"))
    for field in ("model_action_changes", "observation_ablation_opportunities"):
        audit.equal("NO_INFERENCE_GATE", f"{where}/{field}", record["opportunities"].get(field), 0)
    for field, expected in (("model", cp["model_config"]), ("parameter_hash", cp["hashes"]["model"]),
            ("source_continuation_hash", cp["hashes"]["source_continuation"]),
            ("start_world_state_hash", cp["hashes"]["world"]), ("start_hidden_hash", cp["hashes"]["hidden_fork"]),
            ("start_scheduler_hash", cp["hashes"]["scheduler"]), ("start_event_hash", cp["event_hash"]),
            ("fork_tick", cp["fork_tick"]), ("fork_time", cp["fork_time"])):
        audit.equal("ACTUAL_EVALUATION_BOUNDARY", f"{where}/{field}", record.get(field), expected)
    for field in ("action_hash", "model_action_hash", "observation_hash", "raw_observation_hash", "first_observation_hash",
                  "first_model_observation_hash", "end_world_state_hash", "end_hidden_hash"):
        audit.require("EVALUATION_HASH", f"{where}/{field}", _hash(record.get(field)))
    audit.equal("UNMODIFIED_FIRST_OBSERVATION", where, record["first_observation_hash"], record["first_model_observation_hash"])
    total = record["opportunities"]
    audit.equal("DISABLED_OPPORTUNITIES", where, total["disabled_slot_opportunities"],
        record["steps"] * record["batch_size"] * (record["n_jobs"] - record["active_jobs"]))
    for cohort in COHORTS:
        row, censored = record[cohort], record["censored"][cohort]
        audit.require("ACTOR_SUBSETS", f"{where}/{cohort}", all(_integer(number)
            for field in ("matured", "completed", "clean_completed") for number in row["per_actor"][field])
            and all(0 <= clean <= complete <= mature for mature, complete, clean in
                zip(row["per_actor"]["matured"], row["per_actor"]["completed"], row["per_actor"]["clean_completed"])))
        audit.require("CENSORED_SUBSETS", f"{where}/{cohort}", all(_integer(censored.get(field))
            for field in ("count", "completed", "clean_completed")) and 0 <= censored["clean_completed"] <= censored["completed"] <= censored["count"])
    return _idle_history(record, audit, where)


def _trace(value, origin, item, plan, audit):
    base, credit = value["credit"]["continuation"], value["credit"]
    where, chunk, ticks = item["id"], plan["chunk_ticks"], plan["target_ticks"]
    prefix, width = origin["trace_count"], origin["variant"]["width"]
    trace = base["trace"]
    audit.equal("TRACE_LENGTH", where, len(trace), prefix + ticks // chunk)
    audit.equal("INHERITED_TRACE_PREFIX", where, json_hash(trace[:prefix]), origin["trace_json_hash"])
    active = origin["structure"]["active_per_row"][:]
    birth_rows, death_rows = origin["structure"]["cumulative_births_per_row"][:], origin["structure"]["cumulative_deaths_per_row"][:]
    cumulative_births, cumulative_deaths = origin["structure"]["cumulative_births"], origin["structure"]["cumulative_deaths"]
    factor_survivors, factor_ever_born = origin["structure"]["active"], 0
    dense_allowed_hash = tensor_hash(torch.ones((width, width), dtype=torch.bool))
    action, data = origin["action_hash"], origin["data_hash"]
    events, trajectory, windows, diagnostics = [], [], [], None
    norms = deepcopy(origin["update_norms"])
    from experiments.factors.run import empty_diagnostics, add_diagnostics
    diagnostics = empty_diagnostics()
    for index, row in enumerate(trace[prefix:], 1):
        label, local_ticks = f"{where}/{index}", index * chunk
        absolute_stage = origin["stage_ticks"] + local_ticks
        absolute_update = origin["stage_updates"] + index
        for field, expected in (("stage", "credit"), ("stage_update", absolute_update), ("stage_ticks", absolute_stage),
                ("factor_ticks", local_ticks), ("factor_update", index), ("factor_mode", item["mode"]),
                ("idle_lambda", item["idle_lambda"]), ("update", origin["update"] + index),
                ("tick", origin["tick"] + local_ticks), ("world_tick", origin["world_tick"] + local_ticks),
                ("optimizer_steps_this_chunk", 1), ("gradient_updates", origin["gradient_updates"] + index), ("dt", plan["dt"]), ("old", None)):
            audit.equal("TRACE_BUDGET", f"{label}/{field}", row.get(field), expected)
        audit.require("TRACE_CLOCK", label, _close(row.get("now"), origin["world_now"] + local_ticks * plan["dt"]))
        expected_work = {name: origin["work"][name] + amount for name, amount in _expected_work(local_ticks, chunk).items()}
        audit.equal("TRACE_WORK", label, row["work"], expected_work)
        live = row["live"]
        raw, loss = live["diagnostics"], _diagnostics(live["diagnostics"], audit, label,
            opportunities=chunk * origin["batch_size"] * origin["n_jobs"])
        for field in ("actual_action_counts", "target_action_counts"):
            audit.require("TRACE_ACTION_COUNTS", f"{label}/{field}", isinstance(row.get(field), list)
                and len(row[field]) == 11 and all(_integer(number) for number in row[field])
                and sum(row[field]) == chunk * origin["batch_size"] * origin["n_jobs"])
        audit.equal("TARGET_WAIT_DENOMINATOR", label, row["target_action_counts"][0],
            sum(raw["buckets"][name]["count"] for name in ("idle", "active_wait", "disabled")))
        audit.equal("TARGET_BEGIN_DENOMINATOR", label, sum(row["target_action_counts"][1:3]), raw["buckets"]["legal_begin"]["count"])
        audit.equal("TARGET_FINISH_DENOMINATOR", label, sum(row["target_action_counts"][3:]), raw["buckets"]["legal_finish"]["count"])
        audit.equal("ACTUAL_NONWAIT_DENOMINATOR", label, sum(row["actual_action_counts"][1:]),
            sum(bucket["actual_nonwait_count"] for bucket in raw["buckets"].values()))
        audit.equal("CHUNK_DIAGNOSTIC_COUNT", label, raw["chunks"], 1)
        add_diagnostics(diagnostics, raw)
        for field, expected in (("base_weighted_ce", loss["base_weighted_ce"]),
                ("idle_nll_mean", loss["idle_nll_mean"] or 0.),
                ("idle_penalty", item["idle_lambda"] * (loss["idle_nll_mean"] or 0.))):
            audit.require("TRACE_OBJECTIVE", f"{label}/{field}", _loss_close(live.get(field), expected))
        audit.require("TRACE_TOTAL_OBJECTIVE", label, _loss_close(row["loss"], live["base_weighted_ce"] + live["idle_penalty"])
            and row["loss"] == live["weighted_loss"])
        update, event = row["parameter_update"], row["parameter_update"]["structure"]
        for name in ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm", "clip_scale", "potential_recurrent_gradient_norm"):
            audit.require("UPDATE_FINITE", f"{label}/{name}", _finite(update.get(name), 0))
            if name != "potential_recurrent_gradient_norm":
                norms[name + "_sum"] += update[name]
        audit.require("JOINT_CLIP", label, _close(update["clip_scale"], min(1., base["config"]["clip"] / (update["gradient_norm"] + 1e-6))))
        audit.equal("TRACE_NORM_KEYS", label, set(row["update_norms"]), set(norms))
        for name, expected in norms.items():
            audit.require("TRACE_UPDATE_NORMS", f"{label}/{name}", _close(row["update_norms"][name], expected, tolerance=1e-10))
        for name in ("action_chunk_hash", "data_chunk_hash", "action_hash", "data_hash", "event_hash", "live_hidden_hash"):
            audit.require("TRACE_HASH", f"{label}/{name}", _hash(row.get(name)))
        action = hashlib.sha256(bytes.fromhex(action) + bytes.fromhex(row["action_chunk_hash"])).hexdigest()
        data = hashlib.sha256(bytes.fromhex(data) + bytes.fromhex(row["data_chunk_hash"])).hexdigest()
        audit.equal("ACTION_CHAIN", label, row["action_hash"], action)
        audit.equal("DATA_CHAIN", label, row["data_hash"], data)
        events.append(row["event_hash"])
        window = absolute_update % 8 == 0
        for field, expected in (("mode", item["mode"]), ("idle_lambda", item["idle_lambda"]),
                ("factor_updates", index), ("factor_update", index), ("factor_ticks", local_ticks),
                ("updates", absolute_update), ("update", absolute_update), ("stage_update", absolute_update),
                ("stage_ticks", absolute_stage), ("updates_in_window", absolute_update % 8), ("window", window), ("nonzero_hard_cuts", 0)):
            audit.equal("STRUCTURAL_EVENT", f"{label}/{field}", event.get(field), expected)
        born, dead = event["births_per_row"], event["deaths_per_row"]
        audit.require("STRUCTURAL_ROW_COUNTS", label, all(isinstance(rows, list) and len(rows) == width
            and all(_integer(number) and number <= width for number in rows) for rows in (born, dead)))
        audit.require("EVENT_SUBSETS", label, all(b <= width-a and d <= a for a, b, d in zip(active, born, dead)))
        audit.equal("BIRTH_TOTAL", label, sum(born), event["births"])
        audit.equal("DEATH_TOTAL", label, sum(dead), event["deaths"])
        active = [a+b-d for a, b, d in zip(active, born, dead)]
        birth_rows = [a+b for a, b in zip(birth_rows, born)]
        death_rows = [a+b for a, b in zip(death_rows, dead)]
        cumulative_births += sum(born)
        cumulative_deaths += sum(dead)
        for field, expected in (("active", sum(active)), ("active_per_row", active),
                ("cumulative_births", cumulative_births), ("cumulative_deaths", cumulative_deaths),
                ("cumulative_births_per_row", birth_rows), ("cumulative_deaths_per_row", death_rows),
                ("factor_cumulative_births", cumulative_births-origin["structure"]["cumulative_births"]),
                ("factor_cumulative_deaths", cumulative_deaths-origin["structure"]["cumulative_deaths"])):
            audit.equal("CAPACITY_ACCOUNTING", f"{label}/{field}", event.get(field), expected)
        allowed = (width * width if item["mode"].startswith("dense") else
            origin["structure"]["active"] if item["mode"].startswith("fixed") else sum(active))
        audit.equal("ALLOWED_CAPACITY", label, event["allowed"], allowed)
        allowed_rows = ([width] * width if item["mode"].startswith("dense") else
            origin["structure"]["active_per_row"] if item["mode"].startswith("fixed") else active)
        audit.equal("ALLOWED_CAPACITY_ROWS", label, event["allowed_per_row"], allowed_rows)
        audit.require("ALLOWED_SUPPORT_SUBSET", label, event["active"] <= event["allowed"])
        audit.equal("SUPPORT_HASH_ALIAS", label, event["support_sha256"], event["mask_sha256"])
        audit.equal("INHERITED_INITIAL_MASK", label, event["initial_mask_sha256"], origin["structure"]["initial_mask_sha256"])
        audit.equal("FACTOR_INITIAL_ACTIVE", label, event["factor_initial_active"], origin["structure"]["active"])
        audit.require("FACTOR_LIFETIME_COUNTS", label,
            _integer(event["factor_initial_survivors"]) and _integer(event["factor_ever_born"])
            and 0 <= event["factor_initial_survivors"] <= min(factor_survivors, event["active"])
            and factor_survivors-event["factor_initial_survivors"] <= event["deaths"]
            and factor_ever_born <= event["factor_ever_born"] <= min(width * width, event["factor_cumulative_births"])
            and event["factor_ever_born"]-factor_ever_born <= event["births"])
        factor_survivors, factor_ever_born = event["factor_initial_survivors"], event["factor_ever_born"]
        if item["mode"] == "retire":
            audit.equal("RETIRE_NO_BIRTHS", label, event["births"], 0)
        if item["mode"].startswith("fixed"):
            audit.equal("FIXED_ALLOWED_HASH", label, event["allowed_mask_sha256"], origin["structure"]["mask_sha256"])
        elif item["mode"].startswith("dense"):
            audit.equal("DENSE_ALLOWED_HASH", label, event["allowed_mask_sha256"], dense_allowed_hash)
        else:
            audit.equal("DYNAMIC_ALLOWED_HASH", label, event["allowed_mask_sha256"], event["support_sha256"])
        for name in ("mask_sha256", "support_sha256", "survival_sha256", "allowed_mask_sha256"):
            audit.require("STRUCTURAL_HASH", f"{label}/{name}", _hash(event.get(name)))
        audit.require("RECURRENT_WEIGHT_NORM", label, _finite(event.get("recurrent_weight_l2"), 0))
        trajectory.append({"seed": item["seed"], "mode": item["mode"], "idle_lambda": item["idle_lambda"],
            **{key: event[key] for key in ("factor_ticks", "factor_update", "stage_ticks", "stage_update", "window",
                "allowed", "active", "births", "deaths", "factor_cumulative_births", "factor_cumulative_deaths",
                "factor_initial_survivors", "factor_ever_born", "recurrent_weight_l2")}})
        if window:
            windows.append(event)
    audit.equal("DIAGNOSTIC_ACCUMULATION", where, value["diagnostics"], diagnostics)
    audit.equal("STRUCTURAL_WINDOW_TRACE", where, value["structural_windows"], windows)
    for name, expected in norms.items():
        audit.require("FINAL_UPDATE_NORMS", f"{where}/{name}", _close(credit["update_norms"][name], expected, tolerance=1e-10))
    for name, expected in (("action_hash", action), ("data_hash", data)):
        audit.equal("FINAL_TRACE_CHAIN", f"{where}/{name}", base[name], expected)
    audit.equal("FINAL_EVENT_CHAIN", where, base["world"]["event_hash"], events[-1])
    audit.equal("FINAL_HIDDEN_TRACE", where, trace[-1]["live_hidden_hash"], state_digest(base["hidden"]))
    return dict(count=len(trace)-prefix, event_sequence_hash=json_hash(events), trajectory=trajectory,
        windows=windows, trace_json_hash=json_hash(trace))


def _checkpoint(item, origin, plan, protocol_hash, audit, inputs):
    from experiments.factors.run import FactorTrainer
    folder, where = Path(item["output"]).resolve(), item["id"]
    files = {name: _input(path, inputs, audit)[0] for name, path in {
        "checkpoint": folder / "checkpoint.pt", "report": folder / "report.json",
        "complete": folder / "launch-complete.json", "trace": folder / "training.jsonl",
        "start": folder.with_name(folder.name + ".launch-start.json")}.items()}
    marker, start, saved_report = (_read_json(files[name]) for name in ("complete", "start", "report"))
    audit.equal("START_BINDING", where, start, dict(format="factors-launch-start-v1", branch=item, protocol_sha256=protocol_hash))
    for field, expected in (("format", "factors-launch-complete-v1"), ("branch", item), ("protocol_sha256", protocol_hash),
            ("target", plan["target_ticks"]), ("stage_ticks", plan["target_ticks"]), ("hidden_resets", 0),
            ("checkpoint_sha256", inputs[str(files["checkpoint"])])):
        audit.equal("COMPLETE_BINDING", f"{where}/{field}", marker.get(field), expected)
    with _preserve_process_state():
        trainer = FactorTrainer.from_checkpoint(files["checkpoint"], device="cpu", strict_sources=True)
        for namespace, texts, hashes in (("continuity", trainer.base.sources, trainer.base.source_hashes),
                ("credit", trainer.sources, trainer.source_hashes), ("credit-dependencies", trainer.dependencies, trainer.dependency_hashes),
                ("factors", trainer.factor_sources, trainer.factor_source_hashes),
                ("factor-dependencies", trainer.factor_dependencies, trainer.factor_dependency_hashes)):
            _snapshot_hashes(texts, hashes, audit, f"{where}/{namespace}")
        value = trainer.semantic_state()
        actual_hash, normalized_hash = state_digest(value), _normalized_semantic_hash(value, plan["training_device"])
        audit.equal("ACTUAL_CHECKPOINT_SEMANTIC", where, normalized_hash, marker.get("semantic_hash"))
        base, credit = value["credit"]["continuation"], value["credit"]
        ticks, updates = plan["target_ticks"], plan["target_ticks"] // plan["chunk_ticks"]
        audit.equal("ACTUAL_FACTOR_CONFIG", where, value["factor_config"],
            {key: item[key] for key in ("mode", "idle_lambda", "shrink_step")})
        for field, expected in (("factor_start_tick", origin["tick"]), ("factor_start_update", origin["update"]),
                ("factor_start_stage_tick", origin["stage_ticks"]), ("factor_start_stage_update", origin["stage_updates"]),
                ("factor_ticks", ticks), ("factor_updates", updates)):
            audit.equal("FACTOR_OFFSETS", f"{where}/{field}", value[field], expected)
        config = normalize(base["config"])
        config["device"] = plan["training_device"]
        audit.equal("INHERITED_CONFIG", where, config, origin["config"])
        audit.equal("INHERITED_VARIANT", where, normalize(base["variant"]), origin["variant"])
        audit.equal("SOURCE_SEED", where, config["seed"], item["seed"])
        audit.equal("CREDIT_CONFIG", where, normalize(credit["credit_config"]), origin["credit_config"])
        audit.equal("CREDIT_ORIGIN", where, normalize(credit["origin"]), origin["credit_origin"])
        audit.equal("PRESERVED_REWIRING_METADATA", where,
            {key: state_digest(value["inherited_rewiring"][key]) for key in value["inherited_rewiring"]}, origin["inherited_rewiring_hashes"])
        factor_origin = value["factor_origin"]
        for field, expected in (("checkpoint", item["source"]), ("checkpoint_sha256", item["source_sha256"]),
                ("inherited_semantic_hash", origin["semantic_hash"]), ("inherited_operator_hash", origin["operator_hash"]),
                ("inherited_source_hashes", origin["inherited_source_hashes"]),
                ("inherited_dependency_hashes", origin["inherited_dependency_hashes"])):
            audit.equal("FACTOR_ORIGIN", f"{where}/{field}", factor_origin.get(field), expected)
        handoff = {name + "_hash": origin["hashes"][name] for name in ("model", "optimizer", "hidden", "world", "replay", "rng")}
        handoff.update(stats_hash=origin["stats_hash"], tick=origin["tick"], update=origin["update"],
            stage_ticks=origin["stage_ticks"], stage_updates=origin["stage_updates"], action_hash=origin["action_hash"],
            data_hash=origin["data_hash"], recurrent_nonzero=origin["structure"]["active"])
        audit.equal("COMPLETE_HANDOFF", where, factor_origin["handoff"], handoff)
        audit.equal("INHERITED_OPERATOR_STATE", where, state_digest(value["operator"]["inherited_operator"]), origin["operator_hash"])
        for field, expected in (("stage", "B"), ("adaptation_mode", "bp64"), ("hidden_resets", 0),
                ("ticks", origin["tick"] + ticks), ("updates", origin["update"] + updates),
                ("gradient_updates", origin["gradient_updates"] + updates),
                ("phase_start_tick", origin["phase_start_tick"]), ("phase_start_update", origin["phase_start_update"]),
                ("phase_start_gradient", origin["phase_start_gradient"]),
                ("initial_parameter_hash", origin["initial_parameter_hash"]), ("data_seed", origin["data_seed"])):
            audit.equal("TRAINING_CONTINUITY", f"{where}/{field}", base[field], expected)
        for name in ("replay", "rng"):
            audit.equal("INHERITED_UNUSED_STATE", f"{where}/{name}", state_digest(base[name]), origin["hashes"][name])
        audit.equal("SAMPLER_RNG", where, state_digest(credit["sampler_rng"]), origin["replay_rng_hash"])
        audit.equal("ADAM_HYPERPARAMETERS", where, normalize(base["optimizer"]["param_groups"]), origin["optimizer_groups"])
        adam = _adam(base["optimizer"], base["model"], [PARAMETER_NAMES], audit, where)
        for name, row in adam.items():
            audit.equal("ADAM_CONTINUOUS_STEP", f"{where}/{name}", row["step"], origin["adam"][name]["step"] + updates)
        for name, tensor in {**base["model"], "hidden": base["hidden"]}.items():
            audit.require("FINITE_ACTUAL_TENSOR", f"{where}/{name}", bool(torch.isfinite(tensor).all()))
        audit.equal("MODEL_SHAPES", where, {name: list(tensor.shape) for name, tensor in base["model"].items()}, origin["model_shapes"])
        audit.equal("HIDDEN_SHAPE", where, list(base["hidden"].shape), [config["batch_size"], origin["variant"]["width"]])
        world = base["world"]
        audit.equal("WORLD_SPEC", where, normalize(world["spec"]), origin["world_spec"])
        audit.equal("WORLD_TICK", where, world["tick"], origin["world_tick"] + ticks)
        audit.require("WORLD_CLOCK", where, _close(world["now"], origin["world_now"] + ticks * plan["dt"]))
        audit.equal("WORLD_ACTION_BOUNDARY", where, world["acted"], True)
        regime = plan["return_regime"]
        expected_regime = {**origin["world_regime"], "duration_range": regime["duration"], "gap_range": regime["gap"], "name": regime["name"]}
        audit.equal("LIVE_RETURN_REGIME", where, normalize(world["regime"]), expected_regime)
        audit.equal("INHERITED_REGIME_CHANGES", where, normalize(credit["regime_changes"][:-1]), origin["regime_changes"])
        change = credit["regime_changes"][-1]
        for field, expected in (("stage_tick", origin["stage_ticks"]), ("tick", origin["tick"]), ("world_tick", origin["world_tick"]), ("regime", expected_regime)):
            audit.equal("RETURN_BOUNDARY", f"{where}/{field}", normalize(change[field]), expected)
        structure = trainer.operator.summary()
        audit.equal("EXACT_SUPPORT", where, structure["active"], int(base["model"]["recurrent"].ne(0).sum()))
        audit.equal("OPERATOR_UPDATES", where, structure["updates"], origin["structure"]["updates"] + updates)
        trace = _trace(value, origin, item, plan, audit)
        for field, expected in structure.items():
            audit.equal("FINAL_OPERATOR_TRACE", f"{where}/{field}", base["trace"][-1]["parameter_update"]["structure"].get(field), expected)
        diagnostics = _diagnostics(value["diagnostics"], audit, where,
            opportunities=ticks * config["batch_size"] * config["n_jobs"])
        audit.equal("LABEL_COUNT", where, int(base["stats"]["confusion"].sum()), origin["stats_count"] + ticks * config["batch_size"] * config["n_jobs"])
        actual_report = trainer.report()
        ignored = {"config", "runtime", "runtime_history"}
        audit.equal("REPORT_KEYS", where, set(saved_report), set(actual_report))
        for field in actual_report.keys() - ignored:
            audit.equal("CHECKPOINT_REPORT", f"{where}/{field}", saved_report[field], normalize(actual_report[field]))
        audit.equal("REPORT_CONFIG", where, saved_report["config"], origin["config"])
        audit.equal("RUNTIME_HISTORY", where, saved_report["runtime_history"], normalize(trainer.runtime_history[:-1]))
        audit.require("RUNTIME_PRESENT", where, bool(saved_report["runtime_history"]))
        audit.equal("RUNTIME_LAST_TRAINING", where, saved_report["runtime"], saved_report["runtime_history"][-1])
        audit.equal("RUNTIME_DEVICE", where, saved_report["runtime"]["device"], plan["training_device"])
        for name in ("python", "platform", "torch", "numpy", "cpu_threads", "cpu_interop_threads", "deterministic_algorithms"):
            audit.require("RUNTIME_FIELD", f"{where}/{name}", name in saved_report["runtime"])
        audit.require("WALL_SECONDS", where, _finite(trainer.seconds, origin["seconds"]))
        suffix = base["trace"][origin["trace_count"]:]
        logged = [{"event": "train", **row} for index, row in enumerate(suffix, 1)
                  if index == 1 or index % config["log_every"] == 0 or index == updates]
        audit.equal("TRAINING_JSONL", where, _jsonl(files["trace"]), logged)
        for prefix, texts in (("factors", trainer.factor_sources), ("dependencies", trainer.factor_dependencies),
                              ("credit", trainer.sources), ("continuity", trainer.base.sources)):
            for name, text in texts.items():
                snapshot, _ = _input(folder / "sources" / prefix / name, inputs, audit, "SOURCE_SNAPSHOT")
                audit.equal("SOURCE_SNAPSHOT_TEXT", str(snapshot), snapshot.read_text(encoding="utf-8"), text)
        hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng")}
        hashes.update(hidden_fork=state_digest({"hidden": base["hidden"]}), scheduler=state_digest(world["scheduler"]),
            source_continuation=state_digest({"model": base["model"], "world": world, "hidden": base["hidden"]}))
        return dict(seed=item["seed"], mode=item["mode"], idle_lambda=item["idle_lambda"], branch_id=where,
            checkpoint=str(files["checkpoint"]), checkpoint_sha256=inputs[str(files["checkpoint"])],
            state_hash=actual_hash, training_device_normalized_state_hash=normalized_hash,
            factor_ticks=ticks, fork_tick=world["tick"], fork_time=world["now"], batch_size=config["batch_size"], n_jobs=config["n_jobs"],
            model_config=trainer.model.config(), hashes=hashes, adam=adam, structure=structure, diagnostics=diagnostics,
            event_hash=world["event_hash"], action_hash=base["action_hash"], data_hash=base["data_hash"],
            training_trace={key: trace[key] for key in ("count", "event_sequence_hash", "trace_json_hash")},
            capacity_trajectory=trace["trajectory"],
            operator_hashes={key: state_digest(item) for key, item in value["operator"]["operator"].items()},
            cost=dict(model_parameters=sum(tensor.numel() for tensor in base["model"].values()),
                recurrent_parameter_slots=base["model"]["recurrent"].numel(), allowed_recurrent_edges=structure["allowed"],
                effective_nonzero_recurrent_edges=structure["active"],
                live_adam_tensor_bytes=_tensor_bytes(base["optimizer"]["state"]),
                structural_tensor_bytes=_tensor_bytes(value["operator"]),
                continuation_wall_seconds=trainer.seconds-origin["seconds"], checkpoint_bytes=files["checkpoint"].stat().st_size))


def _checkpoint_worker(job):
    item, origin, plan, protocol_hash = job
    audit, inputs = _Checks(), {}
    try:
        with _preserve_process_state():
            checkpoint = _checkpoint(item, origin, plan, protocol_hash, audit, inputs)
    except AuditError as error:
        # AuditError's constructor needs more than the default exception args;
        # send plain data across the process boundary instead of pickling it.
        return dict(failure=error.audit["errors"][0], counts=dict(audit.counts))
    return dict(checkpoint=checkpoint, audit=audit.result(), inputs=inputs)


def _plan(protocol, root, audit, inputs):
    plan = _read_json(protocol)
    for field, expected in (("format", "factors-study-protocol-v1"), ("modes", list(MODES)),
            ("idle_lambdas", list(IDLE_LAMBDAS)), ("target_ticks", 65536), ("chunk_ticks", 64),
            ("source_stage_ticks", 65536), ("training_device", "cuda"), ("evaluation_device", "cpu"),
            ("conditions", list(CONDITIONS)), ("evaluation_steps", 2048), ("dt", .25),
            ("return_regime", dict(duration=[1., 4.], gap=[.5, 2.], name="rewiring-A2")),
            ("comparisons", [list(pair) for pair in STRUCTURAL_CONTRASTS])):
        audit.equal("PROTOCOL", field, plan.get(field), expected)
    _verify_files(plan["sources"], audit, "FROZEN_SOURCE_SHA256")
    required = {f"experiments/{group}/{name}" for group, names in {
        "factors": ("__init__.py", "structure.py", "run.py", "test_structure.py", "test_run.py", "study.py", "test_study.py"),
        "rewiring": ("plasticity.py", "run.py", "test_plasticity.py", "test_run.py"),
        "continuity": ("model.py", "world.py", "run.py", "evaluate.py"),
        "credit": ("run.py", "local.py"), "integration": ("run.py", "diagnose.py"),
        "loops": ("audit_run.py", "structure.py", "interventions.py"),
    }.items() for name in names}
    actual = {"/".join(Path(path).parts[-3:]) for path in plan["sources"]}
    audit.require("FROZEN_SOURCE_COVERAGE", str(protocol), required <= actual)
    inputs.update(plan["sources"])
    audit.equal("ORIGIN_COUNT", str(protocol), len(plan["origins"]), 3)
    origins = {}
    for row in plan["origins"]:
        seed = row.get("seed")
        audit.require("ORIGIN_ID", str(row.get("id")), _integer(seed) and seed in SEEDS
            and seed not in origins and row["id"] == f"seed_{seed:03d}")
        for pathfield, hashfield in (("checkpoint", "checkpoint_sha256"), ("historical_A2", "historical_A2_sha256")):
            path, digest = _input(row[pathfield], inputs, audit, "BOUND_ORIGIN")
            audit.equal("ORIGIN_ABSOLUTE_PATH", row["id"], str(path), row[pathfield])
            audit.equal("ORIGIN_CHECKPOINT_SHA256", row["id"], digest, row[hashfield])
        origins[seed] = row
    audit.equal("ORIGIN_SEED_COVERAGE", str(protocol), set(origins), set(SEEDS))
    audit.equal("DISTINCT_TRAINED_SOURCES", str(protocol), len({row["checkpoint_sha256"] for row in origins.values()}), 3)
    audit.equal("BRANCH_COUNT", str(protocol), len(plan["branches"]), 54)
    branches, labels = {}, {0.: "zero", .1: "small", 1.: "large"}
    for item in plan["branches"]:
        seed, mode, strength = item.get("seed"), item.get("mode"), item.get("idle_lambda")
        key = seed, mode, strength
        audit.require("BRANCH_ID", str(item.get("id")), _integer(seed) and seed in SEEDS
            and mode in MODES and _finite(strength) and strength in IDLE_LAMBDAS and key not in branches)
        name = f"{mode}_idle_{labels[strength]}"
        for field, expected in (("id", f"seed_{seed:03d}_{name}"), ("idle_label", labels[strength]),
                ("shrink_step", .0002), ("source", origins[seed]["checkpoint"]),
                ("source_sha256", origins[seed]["checkpoint_sha256"]), ("origin_id", origins[seed]["id"])):
            audit.equal("BRANCH_BINDING", f"{item['id']}/{field}", item.get(field), expected)
        audit.equal("BRANCH_OUTPUT", item["id"], Path(item["output"]).resolve(), root / "training" / f"seed_{seed:03d}" / name)
        branches[key] = item
    audit.equal("BRANCH_COVERAGE", str(protocol), set(branches), {(seed, mode, strength) for seed in SEEDS for mode in MODES for strength in IDLE_LAMBDAS})
    attachment, digest = _input(plan["review_attachment"]["path"], inputs, audit, "REVIEW_ATTACHMENT")
    audit.equal("REVIEW_ATTACHMENT_SHA256", str(attachment), digest, plan["review_attachment"]["sha256"])
    return plan, branches, origins


def _evaluation(path, identity, cp, plan, protocol_hash, audit, inputs, *, baseline=False):
    path, _ = _input(path, inputs, audit, "EVALUATION_FILE")
    value = _read_json(path)
    common = {"format": "factors-baseline-v1" if baseline else "factors-evaluation-v1",
        "origin" if baseline else "branch": identity, "protocol_sha256": protocol_hash,
        "source_unchanged": True, "state_hash": cp["state_hash"]}
    if not baseline:
        common.update(_binding(cp))
    for field, expected in common.items():
        audit.equal("EVALUATION_PROVENANCE", f"{identity['id']}/{field}", value.get(field), expected)
    records, history = {}, {}
    for row in value["records"]:
        condition, where = row.get("condition"), f"{identity['id']}/{row.get('condition')}"
        audit.require("EXPECTED_FORK", where, condition in CONDITIONS and condition not in records)
        history[condition] = _record(row, cp, plan, audit, where)
        records[condition] = row
    audit.equal("FORK_COVERAGE", identity["id"], set(records), set(CONDITIONS))
    return records, history


def _selected_trace(trace, template):
    """Compare historical fields explicitly, excluding only new wrapper fields."""
    def select(value, sample):
        if isinstance(sample, dict):
            return {key: select(value[key], item) for key, item in sample.items()}
        return value
    return [select(row, template) for row in trace]


def _historical(reference, current, origin, audit, inputs):
    from experiments.factors.run import FactorTrainer
    path, digest = _input(reference["historical_A2"], inputs, audit, "HISTORICAL_CHECKPOINT")
    audit.equal("HISTORICAL_SHA256", str(path), digest, reference["historical_A2_sha256"])
    with _preserve_process_state():
        old = RewireTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
        new = FactorTrainer.from_checkpoint(current["checkpoint"], device="cpu", strict_sources=True)
        left, right = old.semantic_state()["credit"]["continuation"], new.semantic_state()["credit"]["continuation"]
        audit.equal("HISTORICAL_SOURCE", str(path), old.rewire_origin["checkpoint_sha256"], origin["inherited_rewiring_origin_sha256"])
        fields = ("model", "optimizer", "hidden", "world", "replay", "rng", "stats", "action_hash", "data_hash",
                  "ticks", "updates", "gradient_updates", "hidden_resets", "phase_start_tick", "phase_start_update", "phase_start_gradient")
        equal = {name: state_digest(left[name]) == state_digest(right[name]) for name in fields}
        for name, passed in equal.items():
            audit.require("RETIRE_ZERO_HISTORICAL_CORE", f"seed_{reference['seed']}/{name}", passed)
        old_operator, new_operator = old.operator.state_dict(), new.operator.rewire.state_dict()
        operator_equal = {key: state_digest(value) == state_digest(new_operator[key]) for key, value in old_operator.items()}
        audit.require("RETIRE_ZERO_HISTORICAL_OPERATOR", str(reference["seed"]), all(operator_equal.values()), operator_equal)
        prefix = origin["trace_count"]
        old_trace, new_trace = left["trace"][prefix:], right["trace"][prefix:]
        audit.equal("HISTORICAL_TRACE_LENGTH", str(path), len(old_trace), len(new_trace))
        selected = _selected_trace(new_trace, old_trace[0])
        audit.equal("RETIRE_ZERO_HISTORICAL_TRACE", str(path), json_hash(selected), json_hash(old_trace))
        return dict(seed=reference["seed"], historical_checkpoint=str(path), historical_sha256=digest,
            current_checkpoint=current["checkpoint"], core_equal=equal, operator_equal=operator_equal,
            selected_trace_sha256=json_hash(selected), historical_trace_sha256=json_hash(old_trace),
            selected_trace_fields=normalize({key: list(value) if isinstance(value, dict) else None for key, value in old_trace[0].items()}),
            equal=True, normalization="No tensor, clock, world, action, data or operator normalization. New trace fields are excluded by selecting every historical trace field recursively.")


def _contrasts(records, history):
    contrasts, penalties, changes, idle_contrasts, idle_penalties = [], [], [], [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            for strength in IDLE_LAMBDAS:
                for left, right in STRUCTURAL_CONTRASTS:
                    a, b = (records[(seed, mode, strength, condition)] for mode in (left, right))
                    contrasts.extend(dict(seed=seed, condition=condition, idle_lambda=strength, left=left, right=right,
                        contrast=f"{left}-minus-{right}", metric=metric, value=_difference(a[metric], b[metric])) for metric in a)
                    ah, bh = (history[(seed, mode, strength, condition)] for mode in (left, right))
                    for cell in ah:
                        idle_contrasts.extend(dict(seed=seed, condition=condition, idle_lambda=strength, left=left, right=right,
                            cause=cell[0], time_since_terminal_bin=cell[1], metric=metric,
                            value=_difference(ah[cell][metric], bh[cell][metric])) for metric in ah[cell])
            for mode in MODES:
                base = records[(seed, mode, 0., condition)]
                for strength in IDLE_LAMBDAS[1:]:
                    actual = records[(seed, mode, strength, condition)]
                    penalties.extend(dict(seed=seed, condition=condition, mode=mode, idle_lambda=strength,
                        reference_idle_lambda=0., metric=metric, value=_difference(actual[metric], base[metric])) for metric in actual)
                    ah, bh = history[(seed, mode, strength, condition)], history[(seed, mode, 0., condition)]
                    for cell in ah:
                        idle_penalties.extend(dict(seed=seed, condition=condition, mode=mode, idle_lambda=strength,
                            reference_idle_lambda=0., cause=cell[0], time_since_terminal_bin=cell[1], metric=metric,
                            value=_difference(ah[cell][metric], bh[cell][metric])) for metric in ah[cell])
                for strength in IDLE_LAMBDAS:
                    actual, baseline = records[(seed, mode, strength, condition)], records[(seed, "baseline", None, condition)]
                    changes.extend(dict(seed=seed, condition=condition, mode=mode, idle_lambda=strength,
                        contrast="A2-minus-shared-B", metric=metric, value=_difference(actual[metric], baseline[metric])) for metric in actual)
    return contrasts, penalties, changes, idle_contrasts, idle_penalties


def build_report(protocol, root, *, workers=6):
    protocol, root = Path(protocol).resolve(), Path(root).resolve()
    audit, inputs = _Checks(), {}
    protocol, protocol_hash = _input(protocol, inputs, audit, "PROTOCOL_FILE")
    analysis_files = {str(path): file_sha256(path) for path in sorted({
        Path(__file__).resolve(), Path(__file__).with_name("test_report.py").resolve(),
        Path(_audit_record.__code__.co_filename).resolve(), Path(_Checks.result.__code__.co_filename).resolve(),
        Path(_summaries.__code__.co_filename).resolve()})}
    try:
        plan, branches, origin_refs = _plan(protocol, root, audit, inputs)
        audit.equal("EVALUATION_FILE_SET", str(root), set((root / "evaluation").rglob("*.json")),
            {root / "evaluation" / f"{item['id']}.json" for item in branches.values()})
        audit.equal("BASELINE_FILE_SET", str(root), set((root / "baseline").rglob("*.json")),
            {root / "baseline" / f"seed_{seed:03d}.json" for seed in SEEDS})
        audit.equal("COMPLETION_FILE_SET", str(root), set((root / "training").rglob("launch-complete.json")),
            {Path(item["output"]) / "launch-complete.json" for item in branches.values()})
        origins = {seed: _source(origin_refs[seed]["checkpoint"], plan["training_device"], audit, inputs) for seed in SEEDS}
        for seed, source in origins.items():
            audit.equal("ACTUAL_SOURCE_SEED", str(seed), source["config"]["seed"], seed)
            audit.equal("SOURCE_STAGE_TICKS", str(seed), source["stage_ticks"], plan["source_stage_ticks"])
            audit.equal("SOURCE_WIDTH", str(seed), source["variant"]["width"], 128)
            audit.equal("SOURCE_CHUNK_TICKS", str(seed), source["credit_config"]["chunk_ticks"], plan["chunk_ticks"])
            audit.equal("SOURCE_WAIT_WEIGHT", str(seed), source["config"]["wait_weight"], .1)
        ordered = [branches[(seed, mode, strength)] for seed in SEEDS for mode in MODES for strength in IDLE_LAMBDAS]
        jobs = [(item, origins[item["seed"]], plan, protocol_hash) for item in ordered]
        worker_results = _map_workers(_checkpoint_worker, jobs, workers)
        checkpoints = {}
        for item, result in zip(ordered, worker_results):
            if "failure" in result:
                audit.counts.update(result["counts"])
                failure = result["failure"]
                raise AuditError(failure["code"], failure["where"], failure.get("details"), checks=sum(audit.counts.values()))
            _merge_audit(audit, result["audit"])
            for path, digest in result["inputs"].items():
                if path in inputs:
                    audit.equal("WORKER_SHARED_INPUT", path, inputs[path], digest)
                inputs[path] = digest
            checkpoints[(item["seed"], item["mode"], item["idle_lambda"])] = result["checkpoint"]
        per_seed, baseline_per_seed, history_rows, history_metrics, baseline_history_metrics, records, history = [], [], [], [], [], {}, {}
        structural, training, costs, trajectories, exogenous = [], [], [], [], {}
        for seed in SEEDS:
            source = origins[seed]
            cp = {**source, "state_hash": source["actual_cpu_semantic_hash"], "fork_tick": source["world_tick"], "fork_time": source["world_now"]}
            source_records, source_history = _evaluation(root / "baseline" / f"seed_{seed:03d}.json", origin_refs[seed], cp,
                plan, protocol_hash, audit, inputs, baseline=True)
            for condition, row in source_records.items():
                metrics = _diagnostic_metrics(row)
                records[(seed, "baseline", None, condition)] = metrics
                baseline_per_seed.extend(dict(seed=seed, condition=condition, metric=metric, value=value) for metric, value in metrics.items())
                for cell in source_history[condition]:
                    history_rows.append(dict(seed=seed, stage="shared_B", mode=None, idle_lambda=None, condition=condition, **cell))
                    baseline_history_metrics.extend(dict(seed=seed, condition=condition,
                        cause=cell["cause"], time_since_terminal_bin=cell["time_since_terminal_bin"], metric=metric, value=value)
                        for metric, value in cell["metrics"].items())
        for item in ordered:
            seed, mode, strength = item["seed"], item["mode"], item["idle_lambda"]
            cp = checkpoints[(seed, mode, strength)]
            identity = {"scheduler": cp["hashes"]["scheduler"], "event_hash": cp["event_hash"],
                "training_events": cp["training_trace"]["event_sequence_hash"], "fork_tick": cp["fork_tick"], "fork_time": cp["fork_time"]}
            if seed in exogenous:
                audit.equal("TRAINING_EXOGENOUS_PAIRING", item["id"], identity, exogenous[seed])
            exogenous[seed] = identity
            trajectories.extend(cp.pop("capacity_trajectory"))
            for metric in ("allowed", "active", "factor_cumulative_births", "factor_cumulative_deaths", "factor_initial_survivors", "factor_ever_born", "peak_active", "min_active"):
                structural.append(dict(seed=seed, mode=mode, idle_lambda=strength, metric=metric, value=cp["structure"][metric]))
            for metric, value in cp["cost"].items():
                costs.append(dict(seed=seed, mode=mode, idle_lambda=strength, metric=metric, value=value))
            for metric in ("base_weighted_ce", "idle_nll_mean", "idle_count", "base_weight_mass"):
                training.append(dict(seed=seed, mode=mode, idle_lambda=strength, metric=metric, value=cp["diagnostics"][metric]))
            for bucket, row in cp["diagnostics"]["buckets"].items():
                for metric, value in row.items():
                    training.append(dict(seed=seed, mode=mode, idle_lambda=strength, metric=f"{bucket}/{metric}", value=value))
            evaluated, idle = _evaluation(root / "evaluation" / f"{item['id']}.json", item, cp, plan, protocol_hash, audit, inputs)
            for condition, row in evaluated.items():
                key, external = (seed, condition), _exogenous(row)
                if key in exogenous:
                    audit.equal("EVALUATION_EXOGENOUS_PAIRING", f"{item['id']}/{condition}", external, exogenous[key])
                exogenous[key] = external
                metrics = _diagnostic_metrics(row)
                records[(seed, mode, strength, condition)] = metrics
                per_seed.extend(dict(seed=seed, mode=mode, idle_lambda=strength, condition=condition, metric=metric, value=value) for metric, value in metrics.items())
                history[(seed, mode, strength, condition)] = {(cell["cause"], cell["time_since_terminal_bin"]): cell["metrics"] for cell in idle[condition]}
                for cell in idle[condition]:
                    history_rows.append(dict(seed=seed, stage="A2", mode=mode, idle_lambda=strength, condition=condition, **cell))
                    history_metrics.extend(dict(seed=seed, mode=mode, idle_lambda=strength, condition=condition,
                        cause=cell["cause"], time_since_terminal_bin=cell["time_since_terminal_bin"], metric=metric, value=value)
                        for metric, value in cell["metrics"].items())
        contrasts, penalties, changes, idle_contrasts, idle_penalties = _contrasts(records, history)
        historical = [_historical(origin_refs[seed], checkpoints[(seed, "retire", 0.)], origins[seed], audit, inputs) for seed in SEEDS]
        audit.equal("ENDPOINT_FORK_COUNT", str(root), len(per_seed) // len(_diagnostic_metrics(next(iter(evaluated.values())))), 270)
        _verify_files(inputs, audit, "FINAL_INPUT_SHA256")
        _verify_files(analysis_files, audit, "ANALYSIS_SOURCE_SHA256")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise AuditError("MALFORMED_SCHEMA", str(root), str(error), checks=sum(audit.counts.values())) from error
    return dict(format=FORMAT_VERSION, complete=True, audit=audit.result(), protocol=str(protocol), protocol_sha256=protocol_hash,
        raw_root=str(root), analysis_source_files=analysis_files, input_files=inputs,
        counts=dict(training_runs=54, trained_seeds=3, checkpoint_sources=3, endpoint_checkpoints=54,
            baseline_forks=15, endpoint_forks=270, total_forks=285), checkpoint_workers=workers,
        origins=list(origins.values()), checkpoints=list(checkpoints.values()), per_seed=per_seed,
        main_summary=_summaries(per_seed, ("mode", "idle_lambda", "condition", "metric")),
        baseline_per_seed=baseline_per_seed, shared_B_baseline=_summaries(baseline_per_seed, ("condition", "metric")),
        contrast_per_seed=contrasts, paired_structural_contrasts=_summaries(contrasts, ("idle_lambda", "condition", "left", "right", "contrast", "metric")),
        idle_penalty_per_seed=penalties, paired_idle_penalty_effects=_summaries(penalties, ("mode", "idle_lambda", "reference_idle_lambda", "condition", "metric")),
        baseline_change_per_seed=changes, paired_changes_from_B=_summaries(changes, ("mode", "idle_lambda", "condition", "contrast", "metric")),
        structural_per_seed=structural, structural_summary=_summaries(structural, ("mode", "idle_lambda", "metric")),
        training_per_seed=training, training_summary=_summaries(training, ("mode", "idle_lambda", "metric")),
        cost_per_seed=costs, cost_summary=_summaries(costs, ("mode", "idle_lambda", "metric")), capacity_trajectory=trajectories,
        idle_by_history_per_seed=history_rows,
        shared_B_idle_by_history_summary=_summaries(baseline_history_metrics, ("condition", "cause", "time_since_terminal_bin", "metric")),
        idle_by_history_summary=_summaries(history_metrics, ("mode", "idle_lambda", "condition", "cause", "time_since_terminal_bin", "metric")),
        idle_by_history_structural_contrasts=_summaries(idle_contrasts, ("idle_lambda", "condition", "left", "right", "cause", "time_since_terminal_bin", "metric")),
        idle_by_history_penalty_effects=_summaries(idle_penalties, ("mode", "idle_lambda", "reference_idle_lambda", "condition", "cause", "time_since_terminal_bin", "metric")),
        historical_retire_zero_comparison=historical,
        statistics=dict(replicate="trained seed", seeds=list(SEEDS), n_total=3,
            sample_sd="Sample standard deviation across the three seed values, with denominator n-1.",
            paired="Subtract within the same trained seed and evaluation condition before computing the mean and sample SD.",
            baseline="One shared sparse-B source and five frozen forks per seed; never counted as another seed or duplicated eighteen times.",
            undefined="Absent cause/age cells retain zero opportunities and undefined rates; n_undefined is explicit.",
            loss="Training base CE uses weighted label mass. Extra idle NLL uses the actual enabled inactive before-action count; active WAIT retains only base CE.",
            loss_scalar_tolerance="Only float32 objective scalars versus independently recomputed float64 diagnostic means use relative tolerance 3e-6 with absolute floor 2e-7. Raw counts, masks, denominators, accumulated sufficient statistics and historical reproduction retain their separate exact checks.",
            training_buckets="legal_begin and legal_finish are teacher-labelled action buckets, selected by the teacher's BEGIN/FINISH target. The teacher FINISH target follows its midpoint schedule; these buckets do not include every task-legal FINISH opportunity. The frozen observer's legal_finish phase separately uses actual world readiness and deadline.",
            support="Allowed coordinates and actual exact nonzero W entries are distinct. Births/deaths count exact support transitions, including reactivation inside fixed allowed support.",
            cost="Every branch retains dense model and Adam tensors and dense arithmetic. Nonzero edge counts do not establish measured compute savings."),
        condition_interpretation=dict(sparse_minus_iid="Changes input gap while retaining the old A duration range.",
            long_wait_minus_iid="Changes task duration while retaining the iid gap range.",
            long_sparse="Changes both duration and gap; this does not isolate either factor alone."),
        limits=deepcopy(plan.get("limits", [])) + [
            "The 65536 appended ticks train only the returning A short-gap regime. Idle penalties do not add sparse-B exposure; an unsuccessful penalty can reflect training coverage as well as loss design.",
            "Idle FINISH errors are split by observed termination cause and age, with exact retired-content matches reported separately. A false FINISH alone does not prove retained old payload caused the error."])


def write_artifacts(report, output):
    output, raw = Path(output).resolve(), Path(report["raw_root"]).resolve()
    if output == raw or raw in output.parents:
        raise ValueError("output must be outside the raw-input directory")
    _atomic_text(output / "report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name, rows in report.items():
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict) or name in ("checkpoints", "origins"):
            continue
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()} for row in rows)
        _atomic_text(output / f"{name}.csv", stream.getvalue())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("path", "sha256"))
    writer.writerows(sorted(report["input_files"].items()))
    _atomic_text(output / "input_files.csv", stream.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        result = build_report(args.protocol, args.root, workers=args.workers)
        write_artifacts(result, args.output)
    except (AuditError, ValueError, OSError) as error:
        parser.exit(1, f"Factors audit rejected: {error}\n")
    print(f"Validated 54 continuations and 285 frozen forks; saved {args.output.resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
