"""Verify and summarize the finite three-arm nonlinear-field continuation study.

Checkpoint loading always uses the current public loader inside preserved
process state. Large tensors and full training traces stay in their hashed
input files; the archive retains hashes, named Adam steps and scalar summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import Trainer, state_digest
from experiments.credit.run import CreditTrainer
from experiments.integration.report import _audit_record, _metrics
from experiments.loops.report import (
    AuditError, _Checks, _atomic_text, _close, _finite, _hash, _integer,
    _paired, _read_json, _verify_files, file_sha256,
)
from experiments.loops.train import LoopTrainer


FORMAT_VERSION = "loops-extension-summary-v1"
ARMS = ("baseline", "source_square", "interaction")
SEEDS = (0, 1, 2)
CONDITIONS = ("iid", "long_wait", "longer", "sparse", "long_sparse")
DIAGNOSTIC_CONDITIONS = ("longer", "long_sparse")
OLD_PARAMETERS = ("recurrent", "encoder.weight", "head.weight", "head.bias")
CONTRASTS = (("interaction", "source_square"), ("interaction", "baseline"), ("source_square", "baseline"))
COHORTS = ("primary", "inherited", "carryover_contract")
ROOT = Path(__file__).resolve().parents[2]


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def json_hash(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _input(path, inputs, audit, code="INPUT_FILE"):
    path = Path(path).resolve()
    audit.require(code, str(path), path.is_file())
    digest = file_sha256(path)
    inputs[str(path)] = digest
    return path, digest


def _normalized_semantic_hash(value, training_device):
    config = value["credit"]["continuation"]["config"]
    old_device = config["device"]
    try:
        config["device"] = training_device
        return state_digest(value)
    finally:
        config["device"] = old_device


def _world_without_regime_names(world):
    # The report identifies this exact, limited metadata normalization.
    value = dict(world)
    value["regime"] = {key: item for key, item in world["regime"].items() if key != "name"}
    value["regime_history"] = [{key: item for key, item in row.items() if key != "name"}
                               for row in world["regime_history"]]
    return value


def _adam(optimizer, model, groups, audit, where):
    saved_groups = optimizer.get("param_groups", [])
    audit.equal("ADAM_GROUP_NAMES", where, len(saved_groups), len(groups))
    identifiers = [index for row in saved_groups for index in row["params"]]
    names = [name for group in groups for name in group]
    audit.equal("ADAM_PARAMETER_IDS", where, identifiers, list(range(len(names))))
    audit.equal("ADAM_PARAMETER_COVERAGE", where, set(names), set(model))
    audit.equal("ADAM_STATE_COVERAGE", where, set(optimizer["state"]), set(identifiers))
    result = {}
    for index, name in zip(identifiers, names):
        moments = optimizer["state"][index]
        step = moments.get("step")
        audit.require("ADAM_STEP", f"{where}/{name}", isinstance(step, torch.Tensor)
                      and step.numel() == 1 and _finite(float(step), 0) and float(step).is_integer())
        for field in ("exp_avg", "exp_avg_sq"):
            value = moments.get(field)
            audit.require("ADAM_MOMENT", f"{where}/{name}/{field}", isinstance(value, torch.Tensor)
                          and value.shape == model[name].shape and value.dtype == model[name].dtype
                          and bool(torch.isfinite(value).all())
                          and (field != "exp_avg_sq" or bool((value >= 0).all())))
        result[name] = {"step": float(step), "moments_hash": state_digest(moments)}
    return result


def _snapshot_hashes(sources, hashes, audit, where):
    audit.equal("EMBEDDED_SOURCE_KEYS", where, set(sources), set(hashes))
    for name, text in sources.items():
        audit.require("EMBEDDED_SOURCE_TEXT", f"{where}/{name}", isinstance(text, str))
        audit.equal("EMBEDDED_SOURCE_HASH", f"{where}/{name}", hashlib.sha256(text.encode()).hexdigest(), hashes[name])


def _source_summary(path, training_device, audit, inputs):
    path, digest = _input(path, inputs, audit, "SOURCE_CHECKPOINT")
    with _preserve_process_state():
        trainer = Trainer.from_checkpoint(path, device="cpu", strict_sources=True)
        _snapshot_hashes(trainer.sources, trainer.source_hashes, audit, f"{path}/continuity")
        value = trainer.semantic_state()
        value["config"]["device"] = training_device
        audit.require("SOURCE_CONTINUITY", str(path), value["stage"] == "A" and value["hidden_resets"] == 0
                      and value["variant"]["policy"] == "online" and value["variant"]["kind"] == "learned"
                      and value["variant"]["reset_every"] == 0)
        audit.equal("SOURCE_WIDTH", str(path), value["variant"]["width"], 128)
        audit.equal("SOURCE_DT", str(path), value["config"]["train_dts"], (.25,))
        audit.equal("SOURCE_MAX_STEP", str(path), value["config"]["max_step"], .25)
        names = [list(OLD_PARAMETERS)]
        adam = _adam(value["optimizer"], value["model"], names, audit, str(path))
        result = {"path": str(path), "sha256": digest, "semantic_hash": state_digest(value),
            "tick": value["ticks"], "update": value["updates"], "gradient_updates": value["gradient_updates"],
            "config": normalize(value["config"]), "variant": normalize(value["variant"]),
            "hashes": {name: state_digest(value[name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng")},
            "model_shapes": {name: list(tensor.shape) for name, tensor in value["model"].items()},
            "optimizer_groups": normalize(value["optimizer"]["param_groups"]), "adam": adam,
            "inherited_source_hashes": deepcopy(trainer.source_hashes),
            "world_spec": normalize(value["world"]["spec"]), "world_keys": {
                name: sorted(value["world"][name]) for name in ("scheduler", "tensors", "counts")},
            "world_now": value["world"]["now"], "world_tick": value["world"]["tick"],
            "trace_count": len(value["trace"]), "trace_json_hash": json_hash(value["trace"]),
            "action_hash": value["action_hash"], "data_hash": value["data_hash"],
            "initial_parameter_hash": value["initial_parameter_hash"], "data_seed": value["data_seed"],
            "seconds": trainer.seconds}
        audit.equal("SOURCE_BYTES_STABLE", str(path), file_sha256(path), digest)
    return result


def _plan(protocol, root, audit, inputs):
    plan = _read_json(protocol)
    for field, expected in (("format", "loops-extension-protocol-v1"), ("training_device", "cuda"),
            ("evaluation_device", "cpu"), ("target_ticks", 65536), ("chunk_ticks", 64),
            ("conditions", list(CONDITIONS)), ("evaluation_steps", 2048),
            ("diagnostic_conditions", list(DIAGNOSTIC_CONDITIONS)), ("solver_refinement_max_step", .125), ("dt", .25),
            ("parameters", {"state_scalars": 128, "baseline": 34348, "source_square": 50732, "interaction": 50732})):
        audit.equal("PROTOCOL", field, plan.get(field), expected)
    _verify_files(plan.get("sources"), audit, "FROZEN_SOURCE_SHA256")
    required_sources = {f"experiments/{group}/{name}" for group, names in {
        "loops": ("__init__.py", "model.py", "train.py", "test_model.py", "test_train.py", "extension_study.py", "audit_run.py", "structure.py", "interventions.py"),
        "continuity": ("model.py", "world.py", "run.py", "evaluate.py"),
        "integration": ("run.py", "diagnose.py"),
        "credit": ("run.py", "local.py", "batch.py", "windows_runner.py", "evaluate_ready.py"),
    }.items() for name in names}
    actual_sources = {"/".join(Path(path).parts[-3:]) for path in plan["sources"]}
    audit.require("FROZEN_RUNTIME_SOURCE_COVERAGE", "protocol", required_sources <= actual_sources)
    inputs.update(plan["sources"])
    audit_path, audit_hash = _input(plan["completed_existing_audit"], inputs, audit, "PRIOR_AUDIT")
    audit.equal("PRIOR_AUDIT_SHA256", str(audit_path), audit_hash, plan["completed_existing_audit_sha256"])
    prior = _read_json(audit_path)
    audit.require("PRIOR_AUDIT_COMPLETE", str(audit_path), prior.get("complete") is True
                  and prior.get("audit", {}).get("status") == "passed" and prior["audit"].get("errors") == [])
    for key, expected in (("sources", 9), ("jobs", 27), ("forks", 540), ("structural_summaries", 9)):
        audit.equal("PRIOR_AUDIT_COUNTS", key, prior.get("counts", {}).get(key), expected)
    branches = plan.get("branches", [])
    audit.equal("BRANCH_COUNT", "protocol", len(branches), 9)
    mapping = {}
    for item in branches:
        seed, arm, identifier = item.get("seed"), item.get("arm"), item.get("id")
        audit.require("BRANCH_ID", str(identifier), _integer(seed) and seed in SEEDS and arm in ARMS
                      and identifier == f"seed_{seed:03d}_{arm}" and (seed, arm) not in mapping)
        expected_output = root / "training" / f"seed_{seed:03d}" / arm
        audit.equal("BRANCH_OUTPUT", identifier, Path(item["output"]).resolve(), expected_output)
        path, digest = _input(item["source"], inputs, audit, "BOUND_SOURCE")
        audit.equal("BOUND_SOURCE_SHA256", identifier, digest, item["source_sha256"])
        mapping[(seed, arm)] = item
    audit.equal("BRANCH_COVERAGE", "protocol", set(mapping), {(seed, arm) for seed in SEEDS for arm in ARMS})
    for seed in SEEDS:
        audit.equal("SHARED_SOURCE", str(seed), len({(mapping[(seed, arm)]["source"], mapping[(seed, arm)]["source_sha256"])
                                                   for arm in ARMS}), 1)
    audit.equal("DISTINCT_TRAINED_SOURCES", "protocol", len({mapping[(seed, "baseline")]["source_sha256"] for seed in SEEDS}), 3)
    if "historical_bp64_checkpoints" in plan:
        historical = plan["historical_bp64_checkpoints"]
        audit.equal("HISTORICAL_COUNT", "protocol", len(historical), 3)
        audit.equal("HISTORICAL_SEEDS", "protocol", sorted(row["seed"] for row in historical), list(SEEDS))
        for row in historical:
            path, digest = _input(row["path"], inputs, audit, "HISTORICAL_CHECKPOINT")
            audit.equal("HISTORICAL_SHA256", str(path), digest, row["sha256"])
    return plan, mapping


def _expected_work(ticks, chunk):
    updates = ticks // chunk
    return dict(live_forward_ticks=ticks, old_forward_ticks=0, old_warmup_ticks=0,
        live_loss_ticks=ticks, old_loss_ticks=0, optimizer_steps=updates, backward_calls=updates,
        bp_graph_ticks=ticks, manual_gradient_passes=0, old_samples=0, live_trace_resets=0, old_trace_resets=0)


def _training_trace(report, origin, plan, audit, where):
    trace = report.get("trace", [])
    prefix = origin["trace_count"]
    audit.equal("TRACE_PREFIX_LENGTH", where, len(trace), prefix + plan["target_ticks"] // plan["chunk_ticks"])
    audit.equal("TRACE_INHERITED_PREFIX", where, json_hash(trace[:prefix]), origin["trace_json_hash"])
    action, data = origin["action_hash"], origin["data_hash"]
    totals, branch_delta = defaultdict(float), 0.
    events = []
    for index, row in enumerate(trace[prefix:], 1):
        label, ticks = f"{where}/{index}", index * plan["chunk_ticks"]
        for field, expected in (("stage", "credit"), ("stage_update", index), ("stage_ticks", ticks),
                ("update", origin["update"] + index), ("tick", origin["tick"] + ticks),
                ("world_tick", origin["world_tick"] + ticks), ("optimizer_steps_this_chunk", 1),
                ("gradient_updates", origin["gradient_updates"] + index), ("old", None)):
            audit.equal("TRACE_BUDGET", f"{label}/{field}", row.get(field), expected)
        audit.require("TRACE_CLOCK", label, _close(row.get("now"), origin["world_now"] + ticks * plan["dt"]))
        audit.equal("TRACE_DT", label, row.get("dt"), plan["dt"])
        audit.equal("TRACE_WORK", label, row.get("work"), _expected_work(ticks, plan["chunk_ticks"]))
        audit.require("TRACE_LOSS", label, _finite(row.get("loss"), 0))
        update = row.get("parameter_update", {})
        for name in ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm", "branch_delta", "branch_raw_gradient_norm"):
            audit.require("UPDATE_FINITE", f"{label}/{name}", _finite(update.get(name), 0))
        audit.require("JOINT_CLIP", label, _close(update.get("clip_scale"), min(1., report["config"]["clip"] / (update["gradient_norm"] + 1e-6))))
        for name in ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm", "clip_scale"):
            totals[name + "_sum"] += update[name]
        branch_delta += update["branch_delta"]
        if report["arm"] == "baseline":
            audit.equal("NO_BASELINE_BRANCH_UPDATE", label, [update["branch_delta"], update["branch_raw_gradient_norm"]], [0., 0.])
        for name in ("action_chunk_hash", "data_chunk_hash", "action_hash", "data_hash", "event_hash", "live_hidden_hash"):
            audit.require("TRACE_HASH", f"{label}/{name}", _hash(row.get(name)))
        action = hashlib.sha256(bytes.fromhex(action) + bytes.fromhex(row["action_chunk_hash"])).hexdigest()
        data = hashlib.sha256(bytes.fromhex(data) + bytes.fromhex(row["data_chunk_hash"])).hexdigest()
        audit.equal("ACTION_CHAIN", label, row["action_hash"], action)
        audit.equal("DATA_CHAIN", label, row["data_hash"], data)
        events.append(row["event_hash"])
    audit.equal("FINAL_ACTION_CHAIN", where, report["action_hash"], action)
    audit.equal("FINAL_DATA_CHAIN", where, report["data_hash"], data)
    audit.equal("FINAL_EVENT_CHAIN", where, report["event_hash"], trace[-1]["event_hash"])
    for name, value in report["update_norms"].items():
        audit.require("UPDATE_NORM_SUM", f"{where}/{name}", _close(value, totals[name], tolerance=1e-10))
    audit.require("BRANCH_UPDATE_SUM", where, _close(report["branch_update_l2_sum"], branch_delta, tolerance=1e-10))
    return {"count": len(trace)-prefix, "event_sequence_hash": json_hash(events), "branch_update_l2_sum": branch_delta}


def _compare_scalars(actual, expected, audit, where, *, tolerance=2e-5):
    audit.equal("SCALAR_KEYS", where, set(actual), set(expected))
    for key, value in expected.items():
        audit.require("SCALAR_VALUE", f"{where}/{key}", actual[key] is None if value is None else
                      _close(actual[key], value, tolerance=tolerance))


def _audit_completed_branch(item, origin, plan, protocol_hash, audit, inputs):
    folder, identifier = Path(item["output"]).resolve(), item["id"]
    paths = {name: _input(path, inputs, audit)[0] for name, path in {
        "checkpoint": folder / "checkpoint.pt", "report": folder / "report.json",
        "complete": folder / "launch-complete.json",
        "start": folder.with_name(folder.name + ".launch-start.json")}.items()}
    report, marker, start = (_read_json(paths[name]) for name in ("report", "complete", "start"))
    expected_binding = {"format": "loops-launch-start-v1", "protocol_sha256": protocol_hash, "branch": item}
    audit.equal("START_BINDING", identifier, start, expected_binding)
    for field, expected in (("format", "loops-launch-complete-v1"), ("branch", item),
            ("protocol_sha256", protocol_hash), ("checkpoint_sha256", inputs[str(paths["checkpoint"])]),
            ("target", plan["target_ticks"]), ("stage_ticks", plan["target_ticks"]), ("hidden_resets", 0)):
        audit.equal("COMPLETE_BINDING", f"{identifier}/{field}", marker.get(field), expected)
    audit.require("MARKER_SEMANTIC_HASH", identifier, _hash(marker.get("semantic_hash")))
    with _preserve_process_state():
        # The public loader validates all named Adam groups, tensors, source
        # snapshots and the original continuous-state restoration path.
        trainer = LoopTrainer.from_checkpoint(paths["checkpoint"], device="cpu", strict_sources=True)
        for namespace, texts, hashes in (("continuity", trainer.base.sources, trainer.base.source_hashes),
                ("credit", trainer.sources, trainer.source_hashes),
                ("dependencies", trainer.dependencies, trainer.dependency_hashes),
                ("loops", trainer.loop_sources, trainer.loop_source_hashes)):
            _snapshot_hashes(texts, hashes, audit, f"{identifier}/{namespace}")
        semantic = trainer.semantic_state()
        semantic_hash = state_digest(semantic)
        normalized_hash = _normalized_semantic_hash(semantic, plan["training_device"])
        audit.equal("ACTUAL_CHECKPOINT_SEMANTIC", identifier, normalized_hash, marker["semantic_hash"])
        base, credit = semantic["credit"]["continuation"], semantic["credit"]
        for field, expected in (("format", "loops-trainer-v1"), ("arm", item["arm"]),
                ("loop_origin", semantic["origin"]), ("origin", credit["origin"]),
                ("credit_config", normalize(credit["credit_config"])), ("work", credit["work"]),
                ("update_norms", credit["update_norms"]), ("regime_changes", normalize(credit["regime_changes"])),
                ("branch_update_l2_sum", semantic["branch_update_l2_sum"]),
                ("loop_source_hashes", trainer.loop_source_hashes), ("credit_source_hashes", trainer.source_hashes),
                ("dependency_hashes", trainer.dependency_hashes), ("source_hashes", trainer.base.source_hashes),
                ("variant", normalize(base["variant"])), ("model", trainer.model.parameter_report())):
            audit.equal("CHECKPOINT_REPORT", f"{identifier}/{field}", report.get(field), expected)
        expected_config = deepcopy(origin["config"])
        audit.equal("INHERITED_TRAIN_CONFIG", identifier, report.get("config"), expected_config)
        actual_config = normalize(base["config"])
        actual_config["device"] = plan["training_device"]
        audit.equal("ACTUAL_TRAIN_CONFIG", identifier, actual_config, expected_config)
        audit.equal("ACTUAL_TRAINED_SEED", identifier, actual_config["seed"], item["seed"])
        audit.equal("CREDIT_CONFIG", identifier, normalize(credit["credit_config"]), {
            "mode": "bp64", "chunk_ticks": plan["chunk_ticks"], "duration_range": [4., 8.],
            "gap_range": [4., 12.], "regime_name": "loops-B"})
        updates, ticks = plan["target_ticks"] // plan["chunk_ticks"], plan["target_ticks"]
        for field, expected in (("stage", "B"), ("adaptation_mode", "bp64"), ("stage_ticks", ticks),
                ("stage_updates", updates), ("stage_gradient_updates", updates), ("hidden_resets", 0),
                ("world_resets", 0), ("ticks", origin["tick"] + ticks),
                ("updates", origin["update"] + updates), ("gradient_updates", origin["gradient_updates"] + updates),
                ("initial_parameter_hash", origin["initial_parameter_hash"]), ("data_seed", origin["data_seed"]),
                ("effective_updated_parameters", plan["parameters"][item["arm"]])):
            audit.equal("TRAINING_BUDGET", f"{identifier}/{field}", report.get(field), expected)
        for field in ("stage", "adaptation_mode", "hidden_resets", "ticks", "updates", "gradient_updates", "action_hash", "data_hash"):
            audit.equal("ACTUAL_CONTINUATION", f"{identifier}/{field}", base[field], report[field])
        for field, expected in (("tick", origin["tick"]), ("update", origin["update"]),
                ("semantic_hash", origin["semantic_hash"]), ("inherited_source_hashes", origin["inherited_source_hashes"]),
                *((f"{name}_hash", origin["hashes"][name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng"))):
            audit.equal("ACTUAL_ORIGIN", f"{identifier}/{field}", credit["origin"].get(field), expected)
        for origin_row in (semantic["origin"], credit["origin"]):
            audit.equal("ORIGIN_PATH", identifier, Path(origin_row.get("checkpoint", "")).resolve(), Path(item["source"]).resolve())
            audit.equal("ORIGIN_BYTES", identifier, origin_row.get("checkpoint_sha256"), item["source_sha256"])
        audit.equal("GRAFT_INHERITED_SEMANTIC", identifier, semantic["origin"].get("inherited_semantic_hash"), origin["semantic_hash"])
        audit.equal("WORK_BUDGET", identifier, report["work"], _expected_work(ticks, plan["chunk_ticks"]))
        audit.equal("REPLAY_UNCHANGED", identifier, state_digest(base["replay"]), origin["hashes"]["replay"])
        audit.equal("RNG_UNCHANGED", identifier, state_digest(base["rng"]), origin["hashes"]["rng"])
        audit.equal("TRACE_ACTUAL_CHECKPOINT", identifier, json_hash(base["trace"]), json_hash(report["trace"]))
        trace = _training_trace(report, origin, plan, audit, identifier)
        names = [list(OLD_PARAMETERS)] + ([["branch"]] if item["arm"] != "baseline" else [])
        audit.equal("NAMED_OPTIMIZER_GROUPS", identifier, semantic["optimizer_parameter_names"], names)
        adam = _adam(base["optimizer"], base["model"], names, audit, identifier)
        for name, values in adam.items():
            expected = updates if name == "branch" else origin["adam"][name]["step"] + updates
            audit.equal("ADAM_CONTINUOUS_STEP", f"{identifier}/{name}", values["step"], expected)
        for group in base["optimizer"]["param_groups"]:
            audit.equal("INHERITED_ADAM_HYPERPARAMETERS", identifier,
                normalize({key: value for key, value in group.items() if key != "params"}),
                {key: value for key, value in origin["optimizer_groups"][0].items() if key != "params"})
        model = base["model"]
        for name, tensor in {**model, "hidden": base["hidden"]}.items():
            audit.require("FINITE_ACTUAL_TENSOR", f"{identifier}/{name}", bool(torch.isfinite(tensor).all()))
        audit.equal("MODEL_PARAMETER_COUNT", identifier, sum(value.numel() for value in model.values()), plan["parameters"][item["arm"]])
        audit.equal("PERSISTENT_STATE", identifier, list(base["hidden"].shape), [report["config"]["batch_size"], 128])
        for name, shape in origin["model_shapes"].items():
            audit.equal("INHERITED_PARAMETER_SHAPE", f"{identifier}/{name}", list(model[name].shape), shape)
        world = base["world"]
        audit.equal("WORLD_IDENTITY", identifier, normalize(world["spec"]), origin["world_spec"])
        audit.equal("WORLD_TICK", identifier, world["tick"], origin["world_tick"] + ticks)
        audit.require("WORLD_CLOCK", identifier, _close(world["now"], origin["world_now"] + ticks * plan["dt"]))
        audit.equal("WORLD_ACTION_BOUNDARY", identifier, world["acted"], True)
        for name in ("scheduler", "tensors", "counts"):
            audit.equal("WORLD_STATE_KEYS", f"{identifier}/{name}", sorted(world[name]), origin["world_keys"][name])
            audit.require("WORLD_SHAPES", f"{identifier}/{name}", all(list(value.shape) == [report["config"]["batch_size"], report["config"]["n_jobs"]]
                          for value in world[name].values()))
            audit.require("WORLD_NO_NAN", f"{identifier}/{name}", all(not value.is_floating_point()
                          or not bool(torch.isnan(value).any()) for value in world[name].values()))
        hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng")}
        hashes.update(hidden_report=state_digest({"h": base["hidden"]}), hidden_fork=state_digest({"hidden": base["hidden"]}),
            scheduler=state_digest(world["scheduler"]), world_without_regime_names=state_digest(_world_without_regime_names(world)))
        for field, expected in (("final_parameter_hash", hashes["model"]), ("hidden_hash", hashes["hidden_report"]),
                ("world_hash", hashes["world"]), ("event_hash", world["event_hash"])):
            audit.equal("ACTUAL_VALUE_HASH", f"{identifier}/{field}", report.get(field), expected)
        audit.equal("FINAL_TRACE_HIDDEN", identifier, report["trace"][-1]["live_hidden_hash"], hashes["hidden"])
        audit.equal("LABEL_COUNT", identifier, int(base["stats"]["confusion"].sum()), ticks * report["config"]["batch_size"] * report["config"]["n_jobs"])
        audit.equal("RUNTIME_HISTORY", identifier, report.get("runtime_history"), normalize(trainer.runtime_history[:-1]))
        audit.require("RUNTIME_PRESENT", identifier, bool(report.get("runtime_history")))
        audit.equal("RUNTIME_LAST_TRAINING", identifier, report.get("runtime"), report["runtime_history"][-1])
        audit.equal("RUNTIME_DEVICE", identifier, report["runtime"].get("device"), plan["training_device"])
        audit.equal("WALL_SECONDS", identifier, report.get("seconds"), trainer.seconds)
        audit.require("WALL_SECONDS", identifier, _finite(report["seconds"], origin["seconds"]))
        branch_metrics = trainer.model.branch_metrics(trainer.h)
        _compare_scalars(report["branch_metrics"], branch_metrics, audit, f"{identifier}/training-branch-metrics")
        for name, text in trainer.loop_sources.items():
            path, _ = _input(folder / "sources" / name, inputs, audit, "SOURCE_SNAPSHOT")
            audit.equal("SOURCE_SNAPSHOT_TEXT", str(path), path.read_text(encoding="utf-8"), text)
            audit.equal("SOURCE_SNAPSHOT_HASH", str(path), hashlib.sha256(text.encode()).hexdigest(), trainer.loop_source_hashes[name])
        parameter_report, model_config = trainer.model.parameter_report(), trainer.model.config()
        batch = report["config"]["batch_size"]
        elapsed = report["seconds"] - origin["seconds"]
        cost = {"total_parameters": parameter_report["total_parameters"], "persistent_state_scalars": parameter_report["persistent_state_scalars"],
            "branch_parameters": parameter_report["branch_parameters"], "recurrent_mac_per_field": parameter_report["recurrent_mac_per_field"],
            "recurrent_matrix_products_per_field": parameter_report["recurrent_matrix_products_per_field"],
            "branch_scalar_products_per_field": parameter_report["branch_scalar_products_per_field"],
            "stage_wall_seconds": elapsed, "actor_ticks": ticks * batch,
            "action_labels": ticks * batch * report["config"]["n_jobs"],
            "forward_recurrent_macs": ticks * batch * 2 * parameter_report["recurrent_mac_per_field"],
            "actor_ticks_per_second": ticks * batch / elapsed if elapsed else None,
            "checkpoint_bytes": paths["checkpoint"].stat().st_size, "optimizer_steps": updates,
            "branch_update_l2_sum": semantic["branch_update_l2_sum"]}
        treatment_models = {}
        for treatment in ("none", "solver_half", "branch_zero"):
            changed = model
            config = dict(model_config)
            if treatment == "solver_half":
                config["max_step"] = plan["solver_refinement_max_step"]
            elif treatment == "branch_zero":
                if item["arm"] == "baseline":
                    continue
                changed = {**model, "branch": torch.zeros_like(model["branch"])}
            treatment_models[treatment] = {"model_config": normalize(config), "parameter_hash": state_digest(changed),
                "source_continuation_hash": state_digest({"model": changed, "world": world, "hidden": base["hidden"]})}
        result = {"seed": item["seed"], "arm": item["arm"], "branch_id": identifier,
            "checkpoint": str(paths["checkpoint"]), "checkpoint_sha256": inputs[str(paths["checkpoint"])],
            "checkpoint_state_hash": semantic_hash, "training_device_normalized_state_hash": normalized_hash,
            "hashes": hashes, "action_hash": base["action_hash"], "data_hash": base["data_hash"],
            "event_hash": world["event_hash"], "fork_tick": world["tick"], "fork_time": world["now"],
            "adam": adam, "branch_metrics": branch_metrics, "model": parameter_report, "cost": cost,
            "training_trace": trace, "runtime": report["runtime"], "runtime_sha256": json_hash(report["runtime"]),
            "treatment_models": treatment_models, "source": {"path": origin["path"], "sha256": origin["sha256"]},
            "batch_size": batch, "n_jobs": report["config"]["n_jobs"]}
        audit.equal("CHECKPOINT_LOAD_READ_ONLY", identifier, state_digest(trainer.semantic_state()), semantic_hash)
    audit.equal("CHECKPOINT_BYTES_STABLE", identifier, file_sha256(paths["checkpoint"]), inputs[str(paths["checkpoint"])])
    return result


def _exogenous(record):
    value = _paired(record)
    for field in ("start_world_state_hash", "start_hidden_hash", "first_observation_hash"):
        value.pop(field)
    return value


def _evaluation(path, item, checkpoint, plan, protocol_hash, audit, inputs):
    path, _ = _input(path, inputs, audit, "EVALUATION_FILE")
    value = _read_json(path)
    for field, expected in (("format", "loops-extension-evaluation-v1"), ("protocol_sha256", protocol_hash),
            ("branch", item), ("checkpoint", checkpoint["checkpoint"]), ("checkpoint_sha256", checkpoint["checkpoint_sha256"]),
            ("checkpoint_state_hash", checkpoint["checkpoint_state_hash"]),
            ("training_device_normalized_state_hash", checkpoint["training_device_normalized_state_hash"]),
            ("source_unchanged", True), ("model", checkpoint["model"]), ("evaluation_device", plan["evaluation_device"])):
        audit.equal("EVALUATION_PROVENANCE", f"{item['id']}/{field}", value.get(field), expected)
    _compare_scalars(value["branch_metrics"], checkpoint["branch_metrics"], audit, f"{item['id']}/evaluation-branch-metrics")
    expected = {(condition, treatment) for condition in CONDITIONS for treatment in
        (("none", "solver_half") + (("branch_zero",) if item["arm"] != "baseline" else ())
         if condition in DIAGNOSTIC_CONDITIONS else ("none",))}
    records, paired = {}, {}
    for record in value["records"]:
        condition, treatment = record.get("condition"), record.get("extension_intervention")
        key, where = (condition, treatment), f"{item['id']}/{condition}/{treatment}"
        audit.require("EXPECTED_FORK", where, key in expected)
        audit.require("DUPLICATE_FORK", where, key not in records)
        _audit_record(record, audit, where, {"eval_steps": plan["evaluation_steps"], "dt": plan["dt"],
            "batch_size": checkpoint["batch_size"], "n_jobs": checkpoint["n_jobs"]})
        audit.equal("OWN_ACTION_FORK", where, record["intervention"], "none")
        for field in ("world_resets", "hidden_resets", "parameter_updates"):
            audit.require("FROZEN_EVALUATION", f"{where}/{field}", _integer(record[field]) and record[field] == 0)
        for field in ("model_action_changes", "observation_ablation_opportunities"):
            audit.equal("NO_ORACLE", f"{where}/{field}", record["opportunities"].get(field), 0)
        expected_model = checkpoint["treatment_models"][treatment]
        for field, expected_value in (("parameter_hash", expected_model["parameter_hash"]),
                ("source_continuation_hash", expected_model["source_continuation_hash"]), ("model", expected_model["model_config"]),
                ("start_world_state_hash", checkpoint["hashes"]["world"]), ("start_hidden_hash", checkpoint["hashes"]["hidden_fork"]),
                ("start_scheduler_hash", checkpoint["hashes"]["scheduler"]), ("start_event_hash", checkpoint["event_hash"]),
                ("fork_tick", checkpoint["fork_tick"]), ("fork_time", checkpoint["fork_time"])):
            audit.equal("ACTUAL_EVALUATION_BOUNDARY", f"{where}/{field}", record.get(field), expected_value)
        for field in ("action_hash", "model_action_hash", "observation_hash", "raw_observation_hash", "first_observation_hash",
                "first_model_observation_hash", "end_world_state_hash", "end_hidden_hash"):
            audit.require("EVALUATION_HASH", f"{where}/{field}", _hash(record.get(field)))
        audit.equal("UNMODIFIED_FIRST_OBSERVATION", where, record["first_observation_hash"], record["first_model_observation_hash"])
        total = record["opportunities"]
        audit.require("IDLE_SUBSET", where, total["inactive_actions"] <= total["inactive_opportunities"])
        audit.equal("DISABLED_OPPORTUNITIES", where, total["disabled_slot_opportunities"],
            record["steps"] * record["batch_size"] * (record["n_jobs"]-record["active_jobs"]))
        for cohort in COHORTS:
            row, censored = record[cohort], record["censored"][cohort]
            vectors = row["per_actor"]
            audit.require("ACTOR_NONNEGATIVE", f"{where}/{cohort}", all(_integer(number)
                for field in ("matured", "completed", "clean_completed") for number in vectors[field]))
            audit.require("ACTOR_SUBSETS", f"{where}/{cohort}", all(0 <= clean <= complete <= mature for mature, complete, clean in
                zip(vectors["matured"], vectors["completed"], vectors["clean_completed"])))
            audit.require("CENSORED_SUBSETS", f"{where}/{cohort}", all(_integer(censored.get(field))
                for field in ("count", "completed", "clean_completed")) and 0 <= censored["clean_completed"] <= censored["completed"] <= censored["count"])
        identity = _paired(record)
        if condition in paired:
            audit.equal("DIAGNOSTIC_EXOGENOUS_PAIRING", where, identity, paired[condition])
        paired[condition] = identity
        records[key] = record
    audit.equal("FORK_COVERAGE", item["id"], set(records), expected)
    return records


def _summaries(rows, dimensions):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in dimensions)].append({"seed": row["seed"], "value": row["value"]})
    output = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in values] != list(SEEDS):
            raise AuditError("SUMMARY_SEED_COVERAGE", str(key))
        defined = [row["value"] for row in values if row["value"] is not None]
        output.append({**dict(zip(dimensions, key)), "n": len(defined), "n_total": 3, "n_undefined": 3-len(defined),
            "mean": statistics.mean(defined) if defined else None,
            "sample_sd": statistics.stdev(defined) if len(defined) > 1 else None, "per_seed": values})
    return output


def _historical(plan, checkpoints, origins, audit, inputs):
    result = []
    for reference in plan.get("historical_bp64_checkpoints", []):
        seed, path = reference["seed"], Path(reference["path"]).resolve()
        current, origin = checkpoints[(seed, "baseline")], origins[seed]
        with _preserve_process_state():
            old = CreditTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
            for namespace, texts, hashes in (("continuity", old.base.sources, old.base.source_hashes),
                    ("credit", old.sources, old.source_hashes), ("dependencies", old.dependencies, old.dependency_hashes)):
                _snapshot_hashes(texts, hashes, audit, f"historical/{seed}/{namespace}")
            value = old.semantic_state()
            base = value["continuation"]
            audit.equal("HISTORICAL_MODE", str(seed), value["credit_config"]["mode"], "bp64")
            audit.equal("HISTORICAL_SEED", str(seed), base["config"]["seed"], seed)
            audit.equal("HISTORICAL_TICKS", str(seed), old.stage_ticks, plan["target_ticks"])
            audit.equal("HISTORICAL_SOURCE_SHA256", str(seed), value["origin"].get("checkpoint_sha256"), origin["sha256"])
            old_hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world")}
            old_hashes.update(action_hash=base["action_hash"], data_hash=base["data_hash"],
                world_without_regime_names=state_digest(_world_without_regime_names(base["world"])))
            current_hashes = {**{name: current["hashes"][name] for name in ("model", "optimizer", "hidden", "world")},
                "action_hash": current["action_hash"], "data_hash": current["data_hash"],
                "world_without_regime_names": current["hashes"]["world_without_regime_names"]}
            equal = {name: old_hashes[name] == current_hashes[name] for name in old_hashes}
            result.append({"seed": seed, "historical_checkpoint": str(path), "historical_sha256": reference["sha256"],
                "new_checkpoint": current["checkpoint"], "historical_hashes": old_hashes, "new_hashes": current_hashes,
                "equal": equal, "core_reproduction_equal": all(equal[name] for name in ("model", "optimizer", "hidden", "action_hash", "data_hash")),
                "comparison_scope": "core_reproduction_equal covers model parameters, Adam, hidden values, action hash and data hash only; it does not claim full checkpoint identity.",
                "world_normalization": "Remove only regime.name and regime_history[*].name; raw world equality is reported separately."})
    return result


def build_report(protocol, root):
    protocol, root = Path(protocol).resolve(), Path(root).resolve()
    audit, inputs = _Checks(), {}
    protocol, protocol_hash = _input(protocol, inputs, audit, "PROTOCOL_FILE")
    analysis_files = {str(path): file_sha256(path) for path in (
        Path(__file__).resolve(), Path(_Checks.result.__code__.co_filename).resolve(), Path(_audit_record.__code__.co_filename).resolve())}
    plan, branches = _plan(protocol, root, audit, inputs)
    audit.equal("EVALUATION_FILE_COUNT", str(root), len(list((root / "evaluation").rglob("*.json"))), 9)
    audit.equal("TRAINING_COMPLETION_COUNT", str(root), len(list((root / "training").rglob("launch-complete.json"))), 9)
    origins = {seed: _source_summary(branches[(seed, "baseline")]["source"], plan["training_device"], audit, inputs) for seed in SEEDS}
    for seed, origin in origins.items():
        audit.equal("ACTUAL_SOURCE_SEED", str(seed), origin["config"]["seed"], seed)
    checkpoints, records, all_rows, branch_rows, cost_rows = {}, {}, [], [], []
    training_pairing, evaluation_pairing = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            item = branches[(seed, arm)]
            checkpoint = _audit_completed_branch(item, origins[seed], plan, protocol_hash, audit, inputs)
            checkpoints[(seed, arm)] = checkpoint
            identity = {"scheduler": checkpoint["hashes"]["scheduler"], "event_hash": checkpoint["event_hash"],
                "trace_event_sequence": checkpoint["training_trace"]["event_sequence_hash"], "fork_tick": checkpoint["fork_tick"], "fork_time": checkpoint["fork_time"]}
            if seed in training_pairing:
                audit.equal("TRAINING_EXOGENOUS_PAIRING", item["id"], identity, training_pairing[seed])
            training_pairing[seed] = identity
            evaluated = _evaluation(root / "evaluation" / f"{item['id']}.json", item, checkpoint, plan, protocol_hash, audit, inputs)
            for (condition, treatment), row in evaluated.items():
                records[(seed, arm, condition, treatment)] = row
                if treatment == "none":
                    identity = _exogenous(row)
                    key = (seed, condition)
                    if key in evaluation_pairing:
                        audit.equal("ARM_EXOGENOUS_PAIRING", f"{seed}/{arm}/{condition}", identity, evaluation_pairing[key])
                    evaluation_pairing[key] = identity
                    all_rows += [{"seed": seed, "arm": arm, "condition": condition, "metric": metric, "value": value}
                                 for metric, value in _metrics(row).items()]
            branch_rows += [{"seed": seed, "arm": arm, "metric": metric, "value": value} for metric, value in checkpoint["branch_metrics"].items()]
            cost_rows += [{"seed": seed, "arm": arm, "metric": metric, "value": value} for metric, value in checkpoint["cost"].items()]
    counts = Counter(key[3] for key in records)
    audit.equal("FORK_COUNTS", str(root), counts, Counter(none=45, solver_half=18, branch_zero=12))
    contrasts, diagnostics, trajectory_comparison = [], [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            for left, right in CONTRASTS:
                a, b = _metrics(records[(seed, left, condition, "none")]), _metrics(records[(seed, right, condition, "none")])
                contrasts += [{"seed": seed, "condition": condition, "left": left, "right": right,
                    "contrast": f"{left}-minus-{right}", "metric": metric,
                    "value": a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None} for metric in a]
        for arm in ARMS:
            for condition in DIAGNOSTIC_CONDITIONS:
                baseline = records[(seed, arm, condition, "none")]
                for treatment in (("solver_half",) if arm == "baseline" else ("solver_half", "branch_zero")):
                    row = records[(seed, arm, condition, treatment)]
                    a, b = _metrics(row), _metrics(baseline)
                    diagnostics += [{"seed": seed, "arm": arm, "condition": condition, "intervention": treatment,
                        "contrast": f"{treatment}-minus-none", "metric": metric,
                        "value": a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None} for metric in a]
                    trajectory_comparison.append({"seed": seed, "arm": arm, "condition": condition, "intervention": treatment,
                        "action_hash_equal": row["action_hash"] == baseline["action_hash"],
                        "end_hidden_hash_equal": row["end_hidden_hash"] == baseline["end_hidden_hash"],
                        "parameter_hash_equal": row["parameter_hash"] == baseline["parameter_hash"]})
    historical = _historical(plan, checkpoints, origins, audit, inputs)
    _verify_files(inputs, audit, "FINAL_INPUT_SHA256")
    _verify_files(analysis_files, audit, "ANALYSIS_SOURCE_SHA256")
    for checkpoint in checkpoints.values():
        checkpoint.pop("treatment_models")
    return {"format": FORMAT_VERSION, "complete": True, "audit": audit.result(),
        "counts": {"branches": 9, "training_runs": 9, "evaluation_files": 9, "forks": 75,
                   "autonomous": 45, "solver_half": 18, "branch_zero": 12},
        "protocol": str(protocol), "protocol_sha256": protocol_hash, "raw_root": str(root),
        "analysis_source_files": analysis_files, "input_files": inputs, "origins": list(origins.values()),
        "checkpoints": list(checkpoints.values()), "per_seed": all_rows,
        "main_summary": _summaries(all_rows, ("arm", "condition", "metric")),
        "contrast_per_seed": contrasts, "paired_contrasts": _summaries(contrasts, ("condition", "left", "right", "contrast", "metric")),
        "diagnostic_per_seed": diagnostics, "diagnostic_deltas": _summaries(diagnostics, ("arm", "condition", "intervention", "contrast", "metric")),
        "diagnostic_trajectories": trajectory_comparison, "branch_per_seed": branch_rows,
        "branch_summary": _summaries(branch_rows, ("arm", "metric")), "cost_per_seed": cost_rows,
        "cost_summary": _summaries(cost_rows, ("arm", "metric")), "historical_baseline_comparison": historical,
        "statistics": {"replicate": "trained seed", "seeds": list(SEEDS), "mean": "Arithmetic mean across defined trained-seed values.",
            "sample_sd": "Sample SD across trained seeds, denominator n-1; all three values and undefined counts are retained.",
            "paired": "Left minus right within the same trained seed; diagnostic treatment minus none within the same complete inherited state.",
            "undefined": "Zero-denominator rates remain null; n_total is three and n_undefined is explicit."},
        "cost_definitions": {"stage_wall_seconds": "Final saved seconds minus inherited A saved seconds; resumptions and concurrent GPU execution affect throughput.",
            "forward_recurrent_macs": "Two Heun field evaluations per .25 live tick times recurrent matrix MACs, batch and ticks; excludes encoder/head, backpropagation and optimizer arithmetic.",
            "branch_metrics": "One saved continuation boundary, recomputed on CPU; not a trajectory distribution or independent replicate."},
        "limits": deepcopy(plan.get("limits", []))}


def write_artifacts(report, output):
    output = Path(output).resolve()
    raw = Path(report["raw_root"])
    if output == raw or raw in output.parents:
        raise ValueError("output must be outside the raw-input directory")
    _atomic_text(output / "report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    names = ("per_seed", "main_summary", "contrast_per_seed", "paired_contrasts", "diagnostic_per_seed", "diagnostic_deltas",
             "branch_per_seed", "branch_summary", "cost_per_seed", "cost_summary", "diagnostic_trajectories")
    for name in names:
        rows = report[name]
        fields = list(rows[0]) if rows else []
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
        _atomic_text(output / f"{name}.csv", stream.getvalue())
    provenance = io.StringIO(newline="")
    writer = csv.writer(provenance)
    writer.writerow(("path", "sha256"))
    writer.writerows(sorted(report["input_files"].items()))
    _atomic_text(output / "input_files.csv", provenance.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        report = build_report(args.protocol, args.root)
        write_artifacts(report, args.output)
    except (AuditError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Extension audit rejected: {error}\n")
    print(f"Validated 9 training branches / 75 frozen forks; saved {args.output.resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
