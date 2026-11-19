"""Read-only audit and per-seed summaries for the frozen credit study.

One training seed is one replicate. Main frozen forks run on CPU. Original
cycle forks use CUDA; the explicit v2 revision measures all primary cycle forks
on CPU while retaining the raw CUDA training continuation. A common measurement
backend does not imply matched future cases. No Trainer is constructed or stepped.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from experiments.continuity.run import state_digest
from experiments.integration.report import (
    Audit, CORE_METRICS, _audit_record, _hash, _metrics, _paired_fields, _paths,
    _same_number, atomic_json, file_sha256, source_text_sha256, summarize,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = "credit-aggregate-v1"
PROTOCOL_SHA256 = "4e69e51b88157de0f61b1dd33dae66c83a66253c77353fecde439877f136f675"
MODES = ("frozen", "readout_only", "bp64", "bp_replay", "diag_sym", "diag_sym_replay",
         "diag_random", "diag_random_replay", "diag_reset64", "diag_instant", "diag_shuffle")
CONTRASTS = (("bp_replay", "bp64"), ("diag_sym_replay", "diag_sym"),
             ("diag_random_replay", "diag_random"), ("diag_sym", "readout_only"),
             ("diag_sym", "bp64"), ("diag_sym", "diag_reset64"),
             ("diag_sym", "diag_instant"), ("diag_sym", "diag_shuffle"),
             ("diag_random", "diag_sym"))
PARAMETERS = ("recurrent", "encoder.weight", "head.weight", "head.bias")
BASE_ANCILLARY = {"sources", "source_hashes", "runtime", "runtime_history", "seconds", "curve"}
CREDIT_ANCILLARY = {"sources", "source_hashes", "dependencies", "dependency_hashes"}


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _json_file(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_provenance"] = {"path": str(path), "sha256": file_sha256(path)}
    return value


def load_reports(inputs):
    return [value for path in _paths(inputs, "report.json")
            if (value := _json_file(path)).get("format") == "credit-trainer-v1"]


def load_evaluations(inputs):
    return [value for path in _paths(inputs, "*.json")
            if (value := _json_file(path)).get("format") == "credit-evaluation-v1" and "cycle" not in value]


def load_cycles(inputs):
    output = []
    for path in _paths(inputs, "complete.json"):
        value = _json_file(path)
        sidecar = path.parent/"measurement-complete.json"
        if sidecar.is_file():
            value["_measurement"] = _json_file(sidecar)
        output.append(value)
    return output


def load_cycle_measurement_revision(path):
    value = _json_file(path)
    required = ("format", "evaluation_device", "base_protocol_sha256", "training_device", "eval_conditions",
                "eval_steps", "dt", "adapter_source_sha256", "frozen_evaluator_sha256", "frozen_cycle_sha256", "reuse_root")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"incomplete cycle measurement revision: {missing}")
    return value


def load_protocol(path, *, expected_sha256=PROTOCOL_SHA256):
    path = Path(path).resolve()
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError("global frozen protocol hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_provenance"] = {"path": str(path), "sha256": actual}
    return value


def _expected_hash(protocol, suffix):
    matches = [digest for name, digest in protocol.get("file_hashes", {}).items()
               if str(name).replace("\\", "/").endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _semantic(state, device=None):
    result = {name: value for name, value in state.items() if name not in CREDIT_ANCILLARY}
    result["continuation"] = {name: value for name, value in state["continuation"].items() if name not in BASE_ANCILLARY}
    if device is not None:
        result["continuation"]["config"] = {**result["continuation"]["config"], "device": device}
    return result


def _model_parts(model):
    return {"model": state_digest(model), "core": state_digest({name: model[name] for name in PARAMETERS[:2]}),
            "readout": state_digest({name: model[name] for name in PARAMETERS[2:]})}


def _adam_parts(optimizer, audit, where):
    groups = optimizer.get("param_groups", [])
    audit.require("ADAM_GROUP", where, len(groups) == 1 and len(groups[0].get("params", [])) == len(PARAMETERS))
    if len(groups) != 1 or len(groups[0].get("params", [])) != len(PARAMETERS):
        return {}
    return {name: {"hash": state_digest(optimizer["state"].get(index, {})),
                   "step": float(optimizer["state"].get(index, {}).get("step", 0.))}
            for name, index in zip(PARAMETERS, groups[0]["params"])}


def _origin_state(entry, audit, inputs):
    path = Path(entry["checkpoint"])
    where = f"origin/{entry['seed']}"
    actual = file_sha256(path)
    audit.equal("ORIGIN_FILE", where, actual, entry["sha256"])
    inputs[str(path.resolve())] = actual
    state = torch.load(path, map_location="cpu", weights_only=False)
    audit.equal("ORIGIN_FORMAT", where, state.get("format"), "continuity-trainer-v1")
    semantic = {name: value for name, value in state.items() if name not in BASE_ANCILLARY}
    result = {"checkpoint": str(path.resolve()), "checkpoint_sha256": actual,
              "semantic_hash": state_digest(semantic), "tick": state["ticks"], "update": state["updates"],
              "model_hash": state_digest(state["model"]), "optimizer_hash": state_digest(state["optimizer"]),
              "hidden_hash": state_digest(state["hidden"]), "world_hash": state_digest(state["world"]),
              "replay_hash": state_digest(state["replay"]), "rng_hash": state_digest(state["rng"]),
              "inherited_source_hashes": state["source_hashes"], "parts": _model_parts(state["model"]),
              "adam": _adam_parts(state["optimizer"], audit, where), "optimizer_groups": state["optimizer"]["param_groups"],
              "world_spec": normalize(state["world"]["spec"]), "world_now": state["world"]["now"],
              "config": state["config"], "action_hash": state["action_hash"], "data_hash": state["data_hash"],
              "model_shapes": {name: list(value.shape) for name, value in state["model"].items()},
              "world_keys": {name: sorted(state["world"][name]) for name in ("scheduler", "tensors", "counts")},
              "initial_head_row_norms": state["model"]["head.weight"].T.double().norm(dim=1).tolist()}
    audit.equal("ORIGIN_IMMUTABLE", where, file_sha256(path), actual)
    return result


def expected_work(mode, protocol, ticks):
    flags = protocol["modes"][mode]
    chunk, replay = protocol["config"]["chunk_ticks"], flags["replay"]
    updates, method = ticks // chunk, flags["method"]
    old_ticks = protocol["replay"]["warmup_ticks"] + protocol["replay"]["loss_ticks"]
    return dict(live_forward_ticks=ticks, old_forward_ticks=updates*old_ticks if replay else 0,
                old_warmup_ticks=updates*protocol["replay"]["warmup_ticks"] if replay else 0,
                live_loss_ticks=ticks if method != "frozen" else 0,
                old_loss_ticks=updates*protocol["replay"]["loss_ticks"] if replay else 0,
                optimizer_steps=updates if method != "frozen" else 0,
                backward_calls=updates*(1+int(replay)) if method == "bp" else 0,
                bp_graph_ticks=ticks+(updates*old_ticks if replay else 0) if method == "bp" else 0,
                manual_gradient_passes=updates*(1+int(replay)) if method == "diag" else updates if method == "readout" else 0,
                old_samples=updates if replay else 0,
                live_trace_resets=updates if flags["reset"] == "block" else ticks if flags["reset"] == "tick" else 0,
                old_trace_resets=updates if replay and method == "diag" else 0)


def _audit_sources(report, protocol, audit, label):
    for name, digest in report.get("credit_source_hashes", {}).items():
        audit.equal("CREDIT_SOURCE", f"{label}/{name}", digest, _expected_hash(protocol, "experiments/credit/"+name))
    for name in ("run.py", "local.py"):
        audit.require("CREDIT_SOURCE_PRESENT", f"{label}/{name}", name in report.get("credit_source_hashes", {}))
    for name, digest in report.get("source_hashes", {}).items():
        audit.equal("CONTINUITY_SOURCE", f"{label}/{name}", digest, _expected_hash(protocol, "experiments/continuity/"+name))
    for name, digest in report.get("dependency_hashes", {}).items():
        audit.equal("DEPENDENCY_SOURCE", f"{label}/{name}", digest, _expected_hash(protocol, "experiments/"+name))


def _audit_report(report, protocol, audit, *, stage_ticks=None):
    seed, mode = report["config"]["seed"], report["credit_config"]["mode"]
    where, ticks = f"seed_{seed:03d}/{mode}", protocol["train_ticks"] if stage_ticks is None else stage_ticks
    flags, chunk = protocol["modes"][mode], protocol["config"]["chunk_ticks"]
    config = {**protocol["config"], "mode": mode}
    audit.equal("TRAINING_FORMAT", where, report.get("format"), "credit-trainer-v1")
    audit.equal("CREDIT_CONFIG", where, normalize(report.get("credit_config")), normalize(config))
    audit.equal("MODE_CONFIG", where, report.get("mode"), flags)
    for name, expected in (("batch_size", protocol["batch_size"]), ("n_jobs", protocol["n_jobs"]),
                           ("device", protocol["device"]), ("train_dts", [protocol["dt"]]),
                           ("lr", protocol["learning_rate"]), ("clip", protocol["gradient_clip"]),
                           ("tbptt", chunk), ("replay_block_ticks", 128), ("replay_burnin", 64), ("replay_capacity", 32)):
        audit.equal("TRAIN_CONFIG", f"{where}/{name}", report["config"].get(name), expected)
    for name, expected in (("width", protocol["width"]), ("kind", "learned"), ("policy", "online"), ("reset_every", 0)):
        audit.equal("VARIANT", f"{where}/{name}", report["variant"].get(name), expected)
    updated = protocol["updated_parameters"].get(mode, protocol["updated_parameters"]["other"])
    audit.equal("UPDATE_LOCATION", where, report.get("effective_updated_parameters"), updated)
    audit.equal("WORK", where, report.get("work"), expected_work(mode, protocol, ticks))
    updates = ticks // chunk
    for name, expected in (("stage_ticks", ticks), ("stage_updates", updates),
                           ("stage_gradient_updates", updates if flags["method"] != "frozen" else 0),
                           ("ticks", report["origin"]["tick"]+ticks), ("updates", report["origin"]["update"]+updates),
                           ("hidden_resets", 0), ("world_resets", 0)):
        audit.equal("TRAINING_BUDGET", f"{where}/{name}", report.get(name), expected)
    _audit_sources(report, protocol, audit, where)
    for name in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash", "checkpoint_sha256"):
        audit.require("ORIGIN_HASH", f"{where}/{name}", _hash(report["origin"].get(name)))
    replay = report.get("replay", {})
    audit.require("OLD_RESERVOIR", where, replay.get("stored_blocks") == replay.get("capacity_blocks") == 32
                  and replay.get("block_ticks") == 128 and len(replay.get("stored_tick_ranges", [])) == 32
                  and all(0 <= lo < hi <= report["origin"]["tick"] and hi-lo == 128 for lo, hi in replay.get("stored_tick_ranges", [])))
    trace = [row for row in report.get("trace", []) if row.get("stage") == "credit"]
    audit.equal("TRACE_LENGTH", where, len(trace), updates)
    action, data = None, None
    norms = defaultdict(float)
    for index, row in enumerate(trace):
        label = f"{where}/{index+1}"
        at = (index+1)*chunk
        for name, expected in (("stage_ticks", at), ("tick", report["origin"]["tick"]+at),
                               ("world_tick", report["origin"]["tick"]+at-1),
                               ("optimizer_steps_this_chunk", int(flags["method"] != "frozen"))):
            audit.equal("TRACE_BUDGET", f"{label}/{name}", row.get(name), expected)
        audit.require("TRACE_CLOCK", label, _same_number(row.get("now"), (row["tick"]-1)*protocol["dt"]))
        audit.equal("TRACE_WORK", label, row.get("work"), expected_work(mode, protocol, at))
        audit.require("TRACE_LOSS", label, isinstance(row.get("loss"), (int, float)) and _finite(row["loss"]))
        if flags["replay"]:
            old = row.get("old", {})
            audit.require("OLD_SAMPLE", label, isinstance(old.get("block_index"), int) and 0 <= old["block_index"] < 32
                          and 0 <= old.get("start_tick", -1) < report["origin"]["tick"])
        else:
            audit.equal("NO_OLD_PASS", label, row.get("old"), None)
        update = row.get("parameter_update")
        if flags["method"] == "frozen":
            audit.equal("FROZEN_UPDATE", label, update, None)
        elif update is not None:
            audit.require("UPDATE_FINITE", label, _finite(update) and all(update.get(name, -1) >= 0 for name in
                          ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm")))
            if flags["method"] == "readout":
                audit.equal("READOUT_CORE_UPDATE", label, [update.get("recurrent_delta"), update.get("encoder_delta")], [0., 0.])
            audit.require("JOINT_CLIP", label, _same_number(update.get("clip_scale"), min(1., protocol["gradient_clip"]/(update["gradient_norm"]+1e-6))))
            for name in ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm", "clip_scale"):
                norms[name+"_sum"] += update[name]
        else:
            audit.require("MISSING_UPDATE", label, False)
        for name in ("action_hash", "data_hash", "action_chunk_hash", "data_chunk_hash", "live_hidden_hash", "event_hash"):
            audit.require("TRACE_HASH", f"{label}/{name}", _hash(row.get(name)))
        if action is not None and all(_hash(row.get(name)) for name in ("action_chunk_hash", "data_chunk_hash")):
            audit.equal("ACTION_CHAIN", label, row["action_hash"], hashlib.sha256(bytes.fromhex(action)+bytes.fromhex(row["action_chunk_hash"])).hexdigest())
            audit.equal("DATA_CHAIN", label, row["data_hash"], hashlib.sha256(bytes.fromhex(data)+bytes.fromhex(row["data_chunk_hash"])).hexdigest())
        action, data = row.get("action_hash"), row.get("data_hash")
    for name, value in report.get("update_norms", {}).items():
        audit.require("UPDATE_NORM_TOTAL", f"{where}/{name}", _same_number(value, norms.get(name, 0.)))
    if trace:
        audit.equal("FINAL_ACTION_CHAIN", where, action, report["action_hash"])
        audit.equal("FINAL_DATA_CHAIN", where, data, report["data_hash"])
        audit.equal("FINAL_EVENT", where, trace[-1]["event_hash"], report["event_hash"])
    for namespace, source_ticks, loss_ticks, resets in (
        ("live_credit", ticks, ticks, expected_work(mode, protocol, ticks)["live_trace_resets"]),
        ("old_credit", updates*128, updates*64, updates)):
        expected = flags["method"] == "diag" and (namespace == "live_credit" or flags["replay"])
        value = report.get(namespace)
        audit.require("ELIGIBILITY_PRESENCE", f"{where}/{namespace}", (value is not None) == expected)
        if value is None:
            continue
        for name, expected_value in (("observed_ticks", source_ticks), ("loss_ticks", loss_ticks),
                                     ("pending_loss_ticks", 0), ("eligibility_resets", resets),
                                     ("reconstructed_flow_ticks", source_ticks), ("reconstructed_flow_matvecs", 2*source_ticks)):
            audit.equal("ELIGIBILITY_BUDGET", f"{where}/{namespace}/{name}", value.get(name), expected_value)
        audit.equal("ELIGIBILITY_BYTES", f"{where}/{namespace}", value.get("eligibility_bytes"),
                    protocol["batch_size"]*protocol["width"]*(protocol["width"]+4*24)*4)
        audit.require("ELIGIBILITY_FINITE", f"{where}/{namespace}", _finite(value))
    audit.require("REGIME_HISTORY", where, bool(report.get("regime_changes")) and report["regime_changes"][0]["stage_tick"] == 0)
    if stage_ticks is None:
        audit.equal("MAIN_REGIME_CHANGE_COUNT", where, len(report.get("regime_changes", [])), 1)
    return trace


def _audit_checkpoint(path, report, protocol, audit, inputs, origin, *, device="cuda", expected_sha=None):
    path, label = Path(path).resolve(), f"{report['config']['seed']}/{report['credit_config']['mode']}/{report['stage_ticks']}"
    before = file_sha256(path)
    if expected_sha is not None:
        audit.equal("CHECKPOINT_BYTES", label, before, expected_sha)
    inputs[str(path)] = before
    state = torch.load(path, map_location="cpu", weights_only=False)
    audit.equal("CHECKPOINT_FORMAT", label, state.get("format"), "credit-trainer-v1")
    base = state["continuation"]
    for name, actual, expected in (("credit_config", normalize(state["credit_config"]), report["credit_config"]),
        ("origin", state["origin"], report["origin"]), ("work", state["work"], report["work"]),
        ("update_norms", state["update_norms"], report["update_norms"]),
        ("regime_changes", normalize(state["regime_changes"]), report["regime_changes"]),
        ("ticks", base["ticks"], report["ticks"]), ("updates", base["updates"], report["updates"]),
        ("trace", normalize(base["trace"]), report["trace"])):
        audit.equal("CHECKPOINT_REPORT", f"{label}/{name}", actual, expected)
    parts = _model_parts(base["model"])
    hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world", "rng", "replay")}
    hashes.update(hidden_report=state_digest({"h": base["hidden"]}), hidden_fork=state_digest({"hidden": base["hidden"]}),
                  source_continuation=state_digest({"model": base["model"], "world": base["world"], "hidden": base["hidden"]}))
    for field, expected in (("final_parameter_hash", parts["model"]), ("hidden_hash", hashes["hidden_report"]), ("world_hash", hashes["world"])):
        audit.equal("CHECKPOINT_VALUE_HASH", f"{label}/{field}", report.get(field), expected)
    for name in ("model", "optimizer", "hidden"):
        audit.require("CHECKPOINT_FINITE", f"{label}/{name}", _finite(base[name]))
    audit.equal("HIDDEN_SHAPE", label, list(base["hidden"].shape), [protocol["batch_size"], protocol["width"]])
    audit.equal("MODEL_PARAMETER_COUNT", label, sum(value.numel() for value in base["model"].values()), report["model"]["total_parameters"])
    audit.equal("MODEL_PARAMETER_NAMES", label, sorted(base["model"]), sorted(PARAMETERS))
    audit.equal("FINAL_LIVE_HIDDEN", label, report["trace"][-1]["live_hidden_hash"], hashes["hidden"])
    for name in ("scheduler", "tensors", "counts"):
        audit.require("WORLD_ARRAY_SHAPES", f"{label}/{name}", all(list(value.shape) == [protocol["batch_size"], protocol["n_jobs"]]
                      for value in base["world"][name].values()))
        audit.require("WORLD_NO_NAN", f"{label}/{name}", all(not value.is_floating_point() or not bool(torch.isnan(value).any())
                      for value in base["world"][name].values()))
    confusion = base["stats"]["confusion"]
    audit.require("CONTINUOUS_LABEL_COUNTS", label, list(confusion.shape) == [11, 11]
                  and int(confusion.sum()) == report["stage_ticks"]*protocol["batch_size"]*protocol["n_jobs"])
    for name in ("live_credit", "old_credit"):
        audit.require("CHECKPOINT_FINITE", f"{label}/{name}", _finite(state[name]))
        if state[name] is not None:
            audit.equal("PENDING_GRADIENT", f"{label}/{name}", state[name]["pending_loss_ticks"], 0)
            audit.require("PENDING_GRADIENT", f"{label}/{name}/buffers", float(state[name]["weight_mass"]) == 0
                          and all(int(torch.count_nonzero(x)) == 0 for x in state[name]["gradients"].values()))
    flags = protocol["modes"][report["credit_config"]["mode"]]
    adam = _adam_parts(base["optimizer"], audit, label)
    if len(base["optimizer"].get("param_groups", [])) == 1:
        for name, index in zip(PARAMETERS, base["optimizer"]["param_groups"][0]["params"]):
            for moment in ("exp_avg", "exp_avg_sq"):
                value = base["optimizer"]["state"].get(index, {}).get(moment)
                audit.require("ADAM_MOMENT_SHAPE", f"{label}/{name}/{moment}", isinstance(value, torch.Tensor)
                              and value.shape == base["model"][name].shape and value.dtype == base["model"][name].dtype)
    if origin is not None:
        for name in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash", "tick", "update", "inherited_source_hashes"):
            audit.equal("FULL_ORIGIN", f"{label}/{name}", report["origin"].get(name), origin[name])
        audit.equal("OLD_BANK_UNCHANGED", label, hashes["replay"], origin["replay_hash"])
        audit.equal("GLOBAL_RNG_UNCHANGED", label, hashes["rng"], origin["rng_hash"])
        audit.equal("INHERITED_CONFIG", label, base["config"], origin["config"])
        audit.equal("PARAMETER_SHAPES", label, {name: list(value.shape) for name, value in base["model"].items()}, origin["model_shapes"])
        audit.equal("ADAM_GROUP_PRESERVED", label, base["optimizer"]["param_groups"], origin["optimizer_groups"])
        if flags["method"] == "frozen":
            audit.equal("FROZEN_MODEL", label, parts["model"], origin["model_hash"])
            audit.equal("FROZEN_ADAM", label, hashes["optimizer"], origin["optimizer_hash"])
        if flags["method"] == "readout":
            audit.equal("FIXED_RESERVOIR", label, parts["core"], origin["parts"]["core"])
            for name in PARAMETERS[:2]:
                audit.equal("FIXED_CORE_ADAM", f"{label}/{name}", adam.get(name), origin["adam"].get(name))
        for name in PARAMETERS:
            expected_steps = origin["adam"][name]["step"] + (report["stage_updates"] if flags["method"] != "frozen"
                            and (flags["method"] != "readout" or name.startswith("head.")) else 0)
            audit.equal("PARAMETER_ADAM_STEPS", f"{label}/{name}", adam.get(name, {}).get("step"), expected_steps)
        audit.equal("WORLD_IDENTITY", label, normalize(base["world"]["spec"]), origin["world_spec"])
        for name in ("scheduler", "tensors", "counts"):
            audit.equal("WORLD_STATE_KEYS", f"{label}/{name}", sorted(base["world"][name]), origin["world_keys"][name])
        audit.require("WORLD_CLOCK", label, _same_number(base["world"]["now"], origin["world_now"]+report["stage_ticks"]*protocol["dt"]))
    audit.equal("WORLD_TICK", label, base["world"]["tick"], base["ticks"]-1)
    audit.equal("WORLD_ACTION_BOUNDARY", label, base["world"]["acted"], True)
    audit.equal("LIVE_REGIME_LINK", label, normalize(base["world"]["regime"]), report["regime_changes"][-1]["regime"])
    audit.equal("LIVE_REGIME_ID_LINK", label, base["world"]["regime_id"], report["regime_changes"][-1]["regime_id"])
    for namespace in ("sources", "dependencies"):
        digest_name = "source_hashes" if namespace == "sources" else "dependency_hashes"
        for name, text in state[namespace].items():
            audit.equal("SNAPSHOT_HASH", f"{label}/{namespace}/{name}", hashlib.sha256(text.encode()).hexdigest(), state[digest_name][name])
    audit.equal("CHECKPOINT_IMMUTABLE", label, file_sha256(path), before)
    return dict(path=str(path), sha256=before, semantic_hash=state_digest(_semantic(state)),
                evaluation_semantic_hash=state_digest(_semantic(state, device)), hashes=hashes, parts=parts,
                sampler_rng_hash=state_digest(state["sampler_rng"]), base_action_hash=base["action_hash"],
                base_data_hash=base["data_hash"], adam=adam)


def _record(record, protocol, audit, where, *, steps):
    try:
        value = normalize(record)
        before = len(audit.issues)
        _audit_record(value, audit, where, {"eval_steps": steps, "dt": protocol["dt"],
                      "batch_size": protocol["batch_size"], "n_jobs": protocol["n_jobs"]})
        return value, _metrics(value) if len(audit.issues) == before else None
    except (KeyError, TypeError, ValueError, IndexError) as error:
        audit.require("RECORD_SCHEMA", where, False, repr(error))
        return record, None


def _flatten_numbers(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for name, item in value.items():
            result.update(_flatten_numbers(item, f"{prefix}.{name}" if prefix else name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        result[prefix] = value
    return result


def _audit_measurement_revision(revision, protocol, audit, inputs, verify_files):
    if revision is None:
        return "cuda"
    expected = {"format": "credit-cycle-measurement-revision-v2", "evaluation_device": "cpu",
        "base_protocol_sha256": protocol.get("_provenance", {}).get("sha256", PROTOCOL_SHA256),
        "training_device": protocol["device"], "eval_conditions": protocol["cycles"]["eval_conditions"],
        "eval_steps": protocol["cycles"]["eval_steps"], "dt": protocol["dt"],
        "frozen_evaluator_sha256": _expected_hash(protocol, "experiments/credit/evaluate.py"),
        "frozen_cycle_sha256": _expected_hash(protocol, "experiments/credit/cycle.py")}
    for name, value in expected.items():
        audit.equal("CYCLE_MEASUREMENT_REVISION", name, revision.get(name), value)
    audit.require("CYCLE_ADAPTER_HASH", "revision", _hash(revision.get("adapter_source_sha256")))
    provenance = revision.get("_provenance", {})
    audit.require("CYCLE_REVISION_FILE_HASH", "revision", _hash(provenance.get("sha256")) and bool(provenance.get("path")))
    if provenance.get("path"):
        inputs[provenance["path"]] = provenance["sha256"]
    if verify_files:
        adapter = ROOT/"experiments/credit/cycle_cpu.py"
        audit.require("CYCLE_MEASUREMENT_ADAPTER_SOURCE", str(adapter), adapter.is_file()
                      and file_sha256(adapter) == revision.get("adapter_source_sha256"))
        inputs[str(adapter.resolve())] = revision["adapter_source_sha256"]
    return "cpu"


def _audit_measurement_metadata(value, revision, protocol, audit, where, *, state_hash=None):
    if revision is None:
        audit.require("UNDECLARED_CYCLE_MEASUREMENT_REVISION", where,
                      not any(name in value for name in ("measurement_backend", "measurement_revision_sha256", "measurement_adapter_sha256")))
        return
    expected = {"measurement_backend": "cpu", "measurement_revision_sha256": revision["_provenance"]["sha256"],
                "measurement_adapter_sha256": revision["adapter_source_sha256"]}
    for name, target in expected.items():
        audit.equal("CYCLE_MEASUREMENT_METADATA", f"{where}/{name}", value.get(name), target)
    actual_path = value.get("measurement_revision")
    audit.equal("CYCLE_MEASUREMENT_REVISION_PATH", where,
                str(Path(actual_path).resolve()) if actual_path else None,
                str(Path(revision["_provenance"]["path"]).resolve()))
    if state_hash is not None:
        audit.equal("CYCLE_MEASUREMENT_TRAINING_DEVICE", where, value.get("training_device"), protocol["device"])
        for name in ("training_state_hash_before", "training_state_hash_after"):
            audit.equal("CYCLE_CPU_CALLBACK_PRESERVES_CUDA_STATE", f"{where}/{name}", value.get(name), state_hash)


def _summary(rows, keys, seeds=None):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append({"seed": row["seed"], "value": row["value"]})
    result = []
    for group, values in sorted(groups.items()):
        if seeds is not None:
            present = {row["seed"] for row in values}
            values += [{"seed": seed, "value": None} for seed in seeds if seed not in present]
        result.append({**dict(zip(keys, group)), **summarize(values)})
    return result


def aggregate(training_reports, evaluations, *, protocol, cycle_completions=None, allow_incomplete=False,
              verify_files=False, compare_previous=False, cycle_measurement_revision=None):
    protocol = normalize(protocol)
    audit, reports, forks, inputs, checkpoints = Audit(allow_incomplete), {}, {}, {}, {}
    seeds, modes = protocol["seeds"], list(protocol["modes"])
    audit.require("PROTOCOL_SEEDS", "protocol", bool(seeds) and len(set(seeds)) == len(seeds))
    audit.require("PROTOCOL_MODES", "protocol", bool(modes) and len(set(modes)) == len(modes) and set(modes) <= set(MODES))
    origin_entries = {row["seed"]: row for row in protocol["origins"]}
    audit.equal("PROTOCOL_ORIGINS", "protocol", sorted(origin_entries), sorted(seeds))
    if protocol.get("_provenance"):
        audit.equal("GLOBAL_PROTOCOL_HASH", "protocol", protocol["_provenance"]["sha256"], PROTOCOL_SHA256)
        inputs[protocol["_provenance"]["path"]] = protocol["_provenance"]["sha256"]
    if verify_files:
        for path, digest in protocol["file_hashes"].items():
            audit.require("FROZEN_PROTOCOL_INPUT", path, Path(path).is_file() and file_sha256(path) == digest)
            inputs[str(Path(path).resolve())] = digest
    cycle_device = _audit_measurement_revision(cycle_measurement_revision, protocol, audit, inputs, verify_files)
    origins = {seed: _origin_state(entry, audit, inputs) for seed, entry in origin_entries.items()} if verify_files else {}
    for item in training_reports:
        try:
            key = (int(item["config"]["seed"]), item["credit_config"]["mode"])
            audit.require("EXPECTED_RUN", str(key), key[0] in seeds and key[1] in modes)
            audit.require("DUPLICATE_RUN", str(key), key not in reports)
            if key[0] not in seeds or key[1] not in modes:
                continue
            reports[key] = item
            _audit_report(item, protocol, audit)
            origin_entry = origin_entries[key[0]]
            audit.equal("ORIGIN_CHECKPOINT_SHA", str(key), item["origin"]["checkpoint_sha256"], origin_entry["sha256"])
            if item.get("_provenance"):
                inputs[item["_provenance"]["path"]] = item["_provenance"]["sha256"]
                if verify_files:
                    folder = Path(item["_provenance"]["path"]).parent
                    checkpoints[key] = _audit_checkpoint(folder/"checkpoint.pt", item, protocol, audit, inputs, origins[key[0]], device="cpu")
                    for prefix, mapping in (("credit", item["credit_source_hashes"]), ("continuity", item["source_hashes"]),
                                            ("dependencies", item["dependency_hashes"])):
                        for name, digest in mapping.items():
                            path = folder/"sources"/prefix/name
                            audit.require("SOURCE_SNAPSHOT_FILE", str(path), path.is_file() and source_text_sha256(path) == digest)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            audit.require("TRAINING_SCHEMA", str(locals().get("key")), False, repr(error))
    evaluation_rows, state_rows, evaluation_links = [], [], {}
    for (seed, mode), item in reports.items():
        for namespace in ("update_norms", "live_credit", "old_credit"):
            value = item.get(namespace)
            if isinstance(value, dict):
                value = {name: field for name, field in value.items() if name != "config"}
            for name, number in _flatten_numbers(value, namespace).items():
                state_rows.append(dict(seed=seed, mode=mode, metric=name, value=number))
    for evaluation in evaluations:
        key = (int(evaluation["seed"]), evaluation["mode"])
        audit.require("EXPECTED_EVALUATION", str(key), key[0] in seeds and key[1] in modes)
        audit.require("EVALUATION_DUPLICATE", str(key), not any(row["seed"] == key[0] and row["mode"] == key[1] for row in evaluation_rows))
        item = reports.get(key)
        audit.require("MISSING_TRAINING_FOR_EVALUATION", str(key), item is not None, missing=True)
        if item is None:
            continue
        audit.equal("EVALUATION_FORMAT", str(key), evaluation.get("format"), "credit-evaluation-v1")
        audit.equal("EVALUATION_STAGE", str(key), evaluation.get("stage_ticks"), protocol["train_ticks"])
        audit.equal("EVALUATION_TRAINING_TICKS", str(key), evaluation.get("training_ticks"), item["ticks"])
        audit.equal("EVALUATION_CONFIG", str(key), normalize(evaluation.get("credit_config")), item["credit_config"])
        audit.equal("EVALUATOR_SOURCE", str(key), evaluation.get("source_sha256"), _expected_hash(protocol, "experiments/credit/evaluate.py"))
        audit.equal("OBSERVER_SOURCE", str(key), evaluation.get("observer_sha256"), _expected_hash(protocol, "experiments/integration/diagnose.py"))
        expected = {(name, "none") for name in protocol["evaluation"]["conditions"]}
        if key[1] in protocol["evaluation"]["intervention_modes"]:
            expected |= {(name, intervention) for name in protocol["evaluation"]["intervention_conditions"]
                         for intervention in protocol["evaluation"]["extra_interventions"]}
        actual = {(row["condition"], row["intervention"]) for row in evaluation["records"]}
        audit.equal("EVALUATION_ROSTER", str(key), sorted(actual), sorted(expected))
        audit.equal("FORK_DUPLICATES", str(key), len(actual), len(evaluation["records"]))
        cp = checkpoints.get(key)
        for name in ("checkpoint_sha256", "checkpoint_state_hash"):
            audit.require("EVALUATION_HASH", f"{key}/{name}", _hash(evaluation.get(name)))
        if cp:
            audit.equal("EVALUATION_CHECKPOINT_SHA", str(key), evaluation["checkpoint_sha256"], cp["sha256"])
            audit.equal("EVALUATION_FULL_STATE", str(key), evaluation["checkpoint_state_hash"], cp["evaluation_semantic_hash"])
            audit.equal("EVALUATION_CHECKPOINT_PATH", str(key), str(Path(evaluation["checkpoint"]).resolve()), cp["path"])
        evaluation_links[key] = {"sha256": evaluation["checkpoint_sha256"], "path": str(Path(evaluation["checkpoint"]).resolve())}
        for raw in evaluation["records"]:
            where = str((*key, raw.get("condition"), raw.get("intervention")))
            row, metrics = _record(raw, protocol, audit, where, steps=protocol["evaluation"]["steps"])
            for field, expected_value in (("parameter_hash", item["final_parameter_hash"]),
                    ("start_world_state_hash", item["world_hash"]), ("start_event_hash", item["event_hash"])):
                audit.equal("FORK_TRAINING_LINK", f"{where}/{field}", row.get(field), expected_value)
            if cp:
                audit.equal("FORK_HIDDEN_LINK", where, row.get("start_hidden_hash"), cp["hashes"]["hidden_fork"])
                audit.equal("FORK_FULL_CONTINUATION_LINK", where, row.get("source_continuation_hash"), cp["hashes"]["source_continuation"])
            record_key = (*key, row["condition"], row["intervention"])
            audit.require("UNIQUE_FORK", where, record_key not in forks)
            forks[record_key] = {"record": row, "metrics": metrics}
            if metrics is not None:
                for name, value in metrics.items():
                    evaluation_rows.append(dict(seed=key[0], mode=key[1], condition=row["condition"],
                                                intervention=row["intervention"], metric=name, value=value))
        for name, value in _flatten_numbers(evaluation.get("state_snapshot"), "snapshot").items():
            state_rows.append(dict(seed=key[0], mode=key[1], metric=name, value=value))
        if evaluation.get("_provenance"):
            inputs[evaluation["_provenance"]["path"]] = evaluation["_provenance"]["sha256"]
    for seed in seeds:
        for mode in modes:
            audit.require("MISSING_RUN", f"{seed}/{mode}", (seed, mode) in reports, missing=True)
            for condition in protocol["evaluation"]["conditions"]:
                audit.require("MISSING_BASELINE", f"{seed}/{mode}/{condition}", (seed, mode, condition, "none") in forks, missing=True)
            if mode in protocol["evaluation"]["intervention_modes"]:
                for condition in protocol["evaluation"]["intervention_conditions"]:
                    for intervention in protocol["evaluation"]["extra_interventions"]:
                        audit.require("MISSING_INTERVENTION", f"{seed}/{mode}/{condition}/{intervention}",
                                      (seed, mode, condition, intervention) in forks, missing=True)
        group = [value for (s, _), value in reports.items() if s == seed]
        for name in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash"):
            audit.require("PAIRED_ORIGIN", f"{seed}/{name}", len({row["origin"][name] for row in group}) <= 1)
        audit.require("PAIRED_TRAIN_EVENTS", str(seed), len({row["event_hash"] for row in group}) <= 1)
        samples = [tuple((point["old"]["block_index"], point["old"]["start_tick"]) for point in row["trace"]
                         if point.get("stage") == "credit" and point.get("old"))
                   for row in group if protocol["modes"][row["credit_config"]["mode"]]["replay"]]
        audit.require("PAIRED_OLD_EXPOSURES", str(seed), len(set(samples)) <= 1)
        if checkpoints:
            sampled = [value["sampler_rng_hash"] for (s, mode), value in checkpoints.items() if s == seed and protocol["modes"][mode]["replay"]]
            audit.require("PAIRED_OLD_SAMPLER", str(seed), len(set(sampled)) <= 1)
    for key, value in forks.items():
        seed, mode, condition, intervention = key
        baseline = forks.get((seed, mode, condition, "none"))
        if baseline is not None and intervention != "none":
            audit.equal("PAIRED_DIAGNOSTIC", str(key), _paired_fields(value["record"]), _paired_fields(baseline["record"]))
            audit.equal("PAIRED_BASELINE_ACTION", str(key), value["record"].get("paired_baseline_action_hash"), baseline["record"]["action_hash"])
    baseline_count = sum(key[-1] == "none" for key in forks)
    extra_count = sum(key[-1] != "none" for key in forks)
    baseline_expected = len(seeds)*len(modes)*len(protocol["evaluation"]["conditions"])
    extra_expected = len(seeds)*len(protocol["evaluation"]["intervention_modes"])*len(protocol["evaluation"]["intervention_conditions"])*len(protocol["evaluation"]["extra_interventions"])
    for name, actual_count, expected_count in (("BASELINE_FORK_COUNT", baseline_count, baseline_expected),
                                                ("EXTRA_FORK_COUNT", extra_count, extra_expected)):
        audit.require(name, "main", actual_count == expected_count, {"actual":actual_count,"expected":expected_count}, missing=actual_count < expected_count)
    contrasts, diagnostics = [], []
    for after, before in CONTRASTS:
        if after not in modes or before not in modes:
            continue
        for seed in seeds:
            for condition in protocol["evaluation"]["conditions"]:
                left, right = forks.get((seed, after, condition, "none")), forks.get((seed, before, condition, "none"))
                if not left or not right or left["metrics"] is None or right["metrics"] is None:
                    continue
                for metric in CORE_METRICS:
                    a, b = left["metrics"][metric], right["metrics"][metric]
                    contrasts.append(dict(seed=seed, after=after, before=before, condition=condition, metric=metric,
                                          value=None if a is None or b is None else a-b))
    for (seed, mode, condition, intervention), value in forks.items():
        baseline = forks.get((seed, mode, condition, "none"))
        if intervention == "none" or baseline is None or value["metrics"] is None or baseline["metrics"] is None:
            continue
        for metric in CORE_METRICS:
            a, b = value["metrics"][metric], baseline["metrics"][metric]
            diagnostics.append(dict(seed=seed, mode=mode, condition=condition, intervention=intervention, metric=metric,
                                    value=None if a is None or b is None else a-b))
    cycle_result = {"status": "not_requested"}
    if cycle_completions is not None:
        try:
            cycle_result = _aggregate_cycles(cycle_completions, protocol, reports, {**evaluation_links, **checkpoints}, origins, audit, inputs,
                                            verify_files=verify_files, measurement_revision=cycle_measurement_revision)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            audit.require("CYCLE_SCHEMA_OR_FILE", "cycles", False, repr(error))
            cycle_result = dict(status="failed", completed_runs=0, per_seed=[], summary=[], change_per_seed=[], paired_changes=[])
    previous = _compare_previous(checkpoints, audit, inputs) if compare_previous and verify_files else []
    if verify_files:
        for path, digest in inputs.items():
            audit.require("INPUT_IMMUTABLE", path, Path(path).is_file() and file_sha256(path) == digest)
    baselines = [row for row in evaluation_rows if row["intervention"] == "none"]
    backend_note = ("Main and primary cycle forks both use CPU. Main 2048-tick and cycle 512-tick estimates use different dependent future cases; a common backend does not make cases matched."
                    if cycle_device == "cpu" else "Main frozen forks use CPU; cycle forks use CUDA. Raw main-versus-cycle score changes confound backend and life history.")
    return dict(format=FORMAT_VERSION, protocol=protocol, audit=audit.result(),
                counts={"training_runs": len(reports), "baseline_forks": sum(k[-1] == "none" for k in forks),
                        "additional_diagnostic_forks": sum(k[-1] != "none" for k in forks)},
                per_seed=evaluation_rows, main_summary=_summary(baselines, ("mode", "condition", "metric"), seeds),
                contrast_per_seed=contrasts, paired_contrasts=_summary(contrasts, ("after", "before", "condition", "metric"), seeds),
                diagnostic_per_seed=diagnostics, diagnostic_deltas=_summary(diagnostics, ("mode", "condition", "intervention", "metric"), seeds),
                state_per_seed=state_rows, state_summary=_summary(state_rows, ("mode", "metric"), seeds),
                cycles=cycle_result, previous_control_comparison=previous,
                checkpoints=[{"seed": key[0], "mode": key[1], **value} for key, value in checkpoints.items()],
                inputs=inputs, notes=["One seed is one replicate; actors and requests are not independent training repeats.",
                    backend_note,
                    "Counterfactual scores are diagnostics, not autonomous performance.",
                    "Cycle changes describe familiar-regime reacquisition and retention, not new-task plasticity."])


def _aggregate_cycles(completions, protocol, reports, checkpoints, origins, audit, inputs, *, verify_files, measurement_revision=None):
    spec, found, rows, checkpoint_rows = protocol["cycles"], {}, [], []
    fork_count = 0
    protocol_sha = protocol.get("_provenance", {}).get("sha256")
    for completion in completions:
        report = completion["report"]
        key = (int(report["config"]["seed"]), report["credit_config"]["mode"])
        label = f"cycles/{key}"
        audit.require("CYCLE_ROSTER", label, key[0] in spec["seeds"] and key[1] in spec["modes"])
        audit.require("CYCLE_DUPLICATE", label, key not in found)
        found[key] = completion
        audit.equal("CYCLE_COMPLETE", label, completion.get("complete"), True)
        if protocol_sha:
            audit.equal("CYCLE_PROTOCOL", label, completion.get("protocol_sha256"), protocol_sha)
        audit.equal("CYCLE_SOURCE", label, completion.get("source_sha256"), _expected_hash(protocol, "experiments/credit/cycle.py"))
        measurement = completion.get("_measurement")
        if measurement_revision is None:
            audit.equal("UNDECLARED_CYCLE_MEASUREMENT_SIDECAR", label, measurement, None)
        else:
            audit.require("CYCLE_MEASUREMENT_SIDECAR", label, measurement is not None)
            if measurement is not None:
                audit.equal("CYCLE_MEASUREMENT_FORMAT", label, measurement.get("format"), "credit-cycle-cpu-measurement-v2")
                _audit_measurement_metadata(measurement, measurement_revision, protocol, audit, label)
                expected_complete_sha = completion.get("_provenance", {}).get("sha256")
                audit.require("CYCLE_MEASUREMENT_COMPLETION_HASHES", label, _hash(expected_complete_sha)
                              and _hash(measurement.get("source_complete_sha256")))
                audit.equal("CPU_SIDECAR_BINDS_FROZEN_COMPLETION", label, measurement.get("source_complete_sha256"), expected_complete_sha)
                for name, value in (("complete", True), ("completed_milestones", len(spec["sequence"])*len(spec["milestones"])),
                                    ("measurement_forks", len(spec["sequence"])*len(spec["milestones"])*len(spec["eval_conditions"])),
                                    ("final_stage_ticks", completion["final_stage_ticks"])):
                    audit.equal("CYCLE_MEASUREMENT_COMPLETE", f"{label}/{name}", measurement.get(name), value)
                final = completion["completed"][-1]
                for name, value in (("final_training_state_hash", final["state_hash"]),
                                    ("final_checkpoint", final["checkpoint"]), ("final_checkpoint_sha256", final["checkpoint_sha256"])):
                    audit.equal("CYCLE_MEASUREMENT_FINAL_CUDA_STATE", f"{label}/{name}", measurement.get(name), value)
                retained = sum(row.get("checkpoint_source") == "immutable_v1_prefix" for row in completion["completed"])
                audit.equal("AUXILIARY_V1_PREFIX_COUNT", label, measurement.get("retained_v1_milestones"), retained)
                if measurement.get("_provenance"):
                    inputs[measurement["_provenance"]["path"]] = measurement["_provenance"]["sha256"]
                if verify_files:
                    bootstrap = Path(measurement["bootstrap"])
                    audit.require("CYCLE_BOOTSTRAP_RECORD", str(bootstrap), bootstrap.is_file())
                    if bootstrap.is_file():
                        inputs[str(bootstrap.resolve())] = file_sha256(bootstrap)
        final_ticks = spec["start_stage_ticks"] + len(spec["sequence"])*spec["segment_ticks"]
        audit.equal("CYCLE_FINAL_TICKS", label, completion.get("final_stage_ticks"), final_ticks)
        _audit_report(report, protocol, audit, stage_ticks=final_ticks)
        main = checkpoints.get(key)
        if main:
            audit.equal("CYCLE_MAIN_ORIGIN", label, completion.get("origin_sha256"), main["sha256"])
            audit.equal("CYCLE_MAIN_ORIGIN_PATH", label, str(Path(completion["origin"]).resolve()), main["path"])
        expected = {(segment, name, offset) for segment, name in enumerate(spec["sequence"]) for offset in spec["milestones"]}
        actual = {(item["segment"], item["regime"], item["offset"]) for item in completion["completed"]}
        audit.equal("CYCLE_MILESTONE_ROSTER", label, sorted(actual), sorted(expected))
        audit.equal("CYCLE_MILESTONE_UNIQUE", label, len(actual), len(completion["completed"]))
        audit.equal("CYCLE_MILESTONE_ORDER", label, [(item["segment"], item["regime"], item["offset"]) for item in completion["completed"]],
                    [(segment, name, offset) for segment, name in enumerate(spec["sequence"]) for offset in spec["milestones"]])
        previous_trace_hash = None
        for milestone in completion["completed"]:
            segment, name, offset = milestone["segment"], milestone["regime"], milestone["offset"]
            where = f"{label}/{segment}/{offset}"
            target = spec["start_stage_ticks"] + segment*spec["segment_ticks"]+offset
            audit.equal("CYCLE_MILESTONE_TICK", where, milestone["stage_ticks"], target)
            value = _json_file(milestone["evaluation"])
            inputs[value["_provenance"]["path"]] = value["_provenance"]["sha256"]
            audit.equal("CYCLE_EVAL_METADATA", where, value.get("cycle"), milestone)
            audit.equal("CYCLE_EVAL_STAGE", where, value.get("stage_ticks"), target)
            audit.equal("CYCLE_EVAL_SEED_MODE", where, [value.get("seed"), value.get("mode")], list(key))
            audit.equal("CYCLE_EVAL_TRAINING_TICKS", where, value.get("training_ticks"), report["origin"]["tick"]+target)
            audit.equal("CYCLE_EVAL_CONFIG", where, value.get("credit_config"), report["credit_config"])
            _audit_measurement_metadata(value, measurement_revision, protocol, audit, where, state_hash=milestone["state_hash"])
            audit.equal("CYCLE_EVALUATOR_SOURCE", where, value.get("source_sha256"), _expected_hash(protocol, "experiments/credit/evaluate.py"))
            audit.equal("CYCLE_OBSERVER_SOURCE", where, value.get("observer_sha256"), _expected_hash(protocol, "experiments/integration/diagnose.py"))
            audit.equal("CYCLE_EVAL_FULL_STATE", where, value.get("checkpoint_state_hash"), milestone["state_hash"])
            audit.equal("CYCLE_EVAL_PROTOCOL", where, value.get("protocol_sha256"), completion.get("protocol_sha256"))
            audit.equal("CYCLE_EVAL_ORIGIN", where, value.get("origin_sha256"), completion.get("origin_sha256"))
            if milestone.get("checkpoint_source") == "immutable_v1_prefix":
                audit.require("AUXILIARY_V1_REFERENCE", where, measurement_revision is not None
                              and bool(milestone.get("source_evaluation"))
                              and Path(milestone["checkpoint"]).resolve().is_relative_to(Path(measurement_revision["reuse_root"]).resolve()))
                if verify_files and milestone.get("source_evaluation"):
                    old = Path(milestone["source_evaluation"]).resolve()
                    audit.require("AUXILIARY_V1_EVALUATION_RETAINED", str(old), old.is_file())
                    if old.is_file():
                        inputs[str(old)] = file_sha256(old)
            cp = None
            if verify_files:
                milestone_report = _json_file(Path(milestone["checkpoint"]).with_name("report.json"))
                inputs[milestone_report["_provenance"]["path"]] = milestone_report["_provenance"]["sha256"]
                _audit_report(milestone_report, protocol, audit, stage_ticks=target)
                cp = _audit_checkpoint(milestone["checkpoint"], milestone_report, protocol, audit, inputs, origins.get(key[0]),
                                       expected_sha=milestone["checkpoint_sha256"])
                audit.equal("CYCLE_CHECKPOINT_STATE", where, cp["semantic_hash"], milestone["state_hash"])
                changes = milestone_report["regime_changes"]
                audit.equal("CYCLE_REGIME_CHANGE_COUNT", where, len(changes), segment+2)
                for index, entry in enumerate(changes[1:]):
                    kind = spec["sequence"][index]
                    audit.equal("CYCLE_REGIME_BOUNDARY", f"{where}/{index}", entry["stage_tick"], spec["start_stage_ticks"]+index*spec["segment_ticks"])
                    audit.equal("CYCLE_REGIME_VALUES", f"{where}/{index}", [entry["regime"]["duration_range"], entry["regime"]["gap_range"]], spec["regimes"][kind])
                if previous_trace_hash is not None:
                    previous_length, digest = previous_trace_hash
                    audit.equal("CYCLE_TRACE_PREFIX_CONTINUITY", where, state_digest(milestone_report["trace"][:previous_length]), digest)
                previous_trace_hash = (len(milestone_report["trace"]), state_digest(milestone_report["trace"]))
                checkpoint_rows.append({"seed": key[0], "mode": key[1], "segment": segment, "offset": offset, **cp})
            pairs = {(record["condition"], record["intervention"]) for record in value["records"]}
            audit.equal("CYCLE_EVAL_ROSTER", where, sorted(pairs), sorted((condition, "none") for condition in spec["eval_conditions"]))
            audit.equal("CYCLE_EVAL_UNIQUENESS", where, len(value["records"]), len(pairs))
            fork_count += len(value["records"])
            for record in value["records"]:
                record, metrics = _record(record, protocol, audit, where, steps=spec["eval_steps"])
                if cp:
                    audit.equal("CYCLE_FORK_PARAMETERS", where, record["parameter_hash"], cp["hashes"]["model"])
                    audit.equal("CYCLE_FORK_WORLD", where, record["start_world_state_hash"], cp["hashes"]["world"])
                    audit.equal("CYCLE_FORK_HIDDEN", where, record["start_hidden_hash"], cp["hashes"]["hidden_fork"])
                    audit.equal("CYCLE_FORK_CONTINUATION", where, record["source_continuation_hash"], cp["hashes"]["source_continuation"])
                if metrics is not None:
                    for metric, number in metrics.items():
                        rows.append(dict(seed=key[0], mode=key[1], segment=segment, regime=name, offset=offset,
                                         condition=record["condition"], metric=metric, value=number))
        if completion.get("_provenance"):
            inputs[completion["_provenance"]["path"]] = completion["_provenance"]["sha256"]
    for seed in spec["seeds"]:
        for mode in spec["modes"]:
            audit.require("MISSING_CYCLE", f"{seed}/{mode}", (seed, mode) in found, missing=True)
    expected_forks = len(spec["seeds"])*len(spec["modes"])*len(spec["sequence"])*len(spec["milestones"])*len(spec["eval_conditions"])
    audit.require("CYCLE_FORK_COUNT", "cycles", fork_count == expected_forks,
                  {"actual":fork_count,"expected":expected_forks}, missing=fork_count < expected_forks)
    lookup = {(row["seed"], row["mode"], row["segment"], row["offset"], row["condition"], row["metric"]): row["value"] for row in rows}
    changes = []
    for row in rows:
        for kind, previous_segment, previous_offset in (
            ("within_segment_4096_minus_1024", row["segment"], spec["milestones"][0]),
            ("switch_minus_previous_segment_end", row["segment"]-1, spec["segment_ticks"]),
            ("same_regime_revisit", row["segment"]-2, row["offset"])):
            if (kind.startswith("within") and row["offset"] != spec["segment_ticks"]) or previous_segment < 0:
                continue
            key = (row["seed"], row["mode"], previous_segment, previous_offset, row["condition"], row["metric"])
            if key not in lookup:
                continue
            before, after = lookup[key], row["value"]
            changes.append({**row, "kind": kind, "previous_segment": previous_segment, "previous_offset": previous_offset,
                            "value": None if before is None or after is None else after-before})
    return {"status": "complete" if len(found) == len(spec["seeds"])*len(spec["modes"]) else "incomplete",
            "completed_runs": len(found), "per_seed": rows,
            "evaluation_device": "cpu" if measurement_revision is not None else "cuda",
            "measurement_revision": measurement_revision,
            "fork_count": fork_count, "expected_fork_count": expected_forks,
            "summary": _summary(rows, ("mode", "segment", "regime", "offset", "condition", "metric"), spec["seeds"]),
            "change_per_seed": changes,
            "paired_changes": _summary(changes, ("kind", "mode", "segment", "regime", "offset", "condition", "metric"), spec["seeds"]),
            "checkpoints": checkpoint_rows,
            "interpretation": ("Familiar-regime reacquisition/retention. All primary frozen measurements use CPU; original CUDA v1 prefixes remain auxiliary. Main and cycle future cases and horizons differ, so they are not matched cases."
                               if measurement_revision is not None else "Familiar-regime reacquisition/retention; CUDA within-cycle contrasts only; no clean CPU-main versus CUDA-cycle learning contrast.")}


def _compare_previous(checkpoints, audit, inputs):
    rows = []
    for (seed, mode), current in checkpoints.items():
        if mode not in {"frozen", "bp64"}:
            continue
        path = ROOT/f"data/integration/main-v1/seed_{seed:03d}/{mode}/checkpoint.pt"
        if not path.is_file():
            continue
        digest = file_sha256(path)
        state = torch.load(path, map_location="cpu", weights_only=False)["continuation"]
        equal = {name: state_digest(state[name]) == current["hashes"][name] for name in ("model", "optimizer", "hidden")}
        equal.update(action_hash=state["action_hash"] == current["base_action_hash"], data_hash=state["data_hash"] == current["base_data_hash"])
        audit.equal("PRIOR_CONTROL_BYTES_UNCHANGED", str(path), file_sha256(path), digest)
        inputs[str(path)] = digest
        rows.append(dict(seed=seed, mode=mode, previous_checkpoint=str(path), previous_checkpoint_sha256=digest,
                         equal=equal, exactly_equal=all(equal.values()),
                         interpretation="Training control reproduction if equal; no independent evidence and no evaluation-backend comparison."))
        del state
    return rows


def _csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)


def write_artifacts(result, output):
    output = Path(output)
    tables = ["per_seed", "main_summary", "contrast_per_seed", "paired_contrasts", "diagnostic_per_seed",
              "diagnostic_deltas", "state_per_seed", "state_summary"]
    if result["cycles"].get("status") != "not_requested":
        tables += ["cycles_"+key for key in ("per_seed", "summary", "change_per_seed", "paired_changes")]
    protected = {str(Path(path).resolve()) for path in result.get("inputs", {})}
    targets = [output/"summary.json", output/"audit.json"]+[output/(key+".csv") for key in tables]
    if any(str(path.resolve()) in protected for path in targets):
        raise ValueError("report output would overwrite an immutable input")
    output.mkdir(parents=True, exist_ok=True)
    def save(path, value):
        for attempt in range(8):
            try:
                atomic_json(path, value)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.05*2**attempt, .5))
    save(output/"summary.json", result)
    save(output/"audit.json", result["audit"])
    for key in ("per_seed", "main_summary", "contrast_per_seed", "paired_contrasts", "diagnostic_per_seed", "diagnostic_deltas", "state_per_seed", "state_summary"):
        _csv(output/f"{key}.csv", result[key])
    if result["cycles"].get("status") != "not_requested":
        for key in ("per_seed", "summary", "change_per_seed", "paired_changes"):
            _csv(output/f"cycles_{key}.csv", result["cycles"][key])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--cycles", nargs="+")
    parser.add_argument("--cycle-measurement-revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    protocol = load_protocol(args.protocol)
    result = aggregate(load_reports(args.root), load_evaluations(args.evaluations), protocol=protocol,
                       cycle_completions=load_cycles(args.cycles) if args.cycles else None,
                       allow_incomplete=args.allow_incomplete, verify_files=True, compare_previous=True,
                       cycle_measurement_revision=load_cycle_measurement_revision(args.cycle_measurement_revision)
                       if args.cycle_measurement_revision else None)
    result["reporter_source_hashes"] = {name: file_sha256(Path(__file__).with_name(name)) for name in ("report.py", "test_report.py")}
    write_artifacts(result, args.output)
    print(json.dumps({"counts": result["counts"], "audit": result["audit"]["status"],
                      "errors": result["audit"]["error_count"], "warnings": result["audit"]["warning_count"]}))
    if result["audit"]["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
