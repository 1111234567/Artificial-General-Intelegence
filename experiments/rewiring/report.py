"""Audit the complete finite recurrent birth/death study before aggregating it.

The public loaders run under preserved process state. No training or evaluation
is performed here, and no partial set is promoted to a complete report.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import Trainer, state_digest
from experiments.credit.run import CreditTrainer
from experiments.integration.report import _audit_record, _metrics
from experiments.loops.extension_report import (
    _adam, _exogenous, _expected_work, _input, _normalized_semantic_hash,
    _snapshot_hashes, _source_summary, _summaries, _world_without_regime_names,
    json_hash, normalize,
)
from experiments.loops.report import (
    AuditError, _Checks, _atomic_text, _close, _finite, _hash, _integer,
    _object, _read_json, _verify_files, file_sha256,
)
from experiments.loops.structure import analyze_weight
from experiments.rewiring.plasticity import tensor_hash
from experiments.rewiring.run import PARAMETER_NAMES, RewireTrainer


FORMAT_VERSION = "rewiring-study-summary-v1"
SEEDS = (0, 1, 2)
PHASES = ("B", "A2")
CONDITIONS = ("iid", "long_wait", "longer", "sparse", "long_sparse")
ARMS = ({"id": "dense", "arm": "dense", "shrink_step": 0.},) + tuple(
    {"id": f"{arm}_{dose}", "arm": arm, "shrink_step": step}
    for dose, step in (("low", 5e-5), ("high", 2e-4))
    for arm in ("retire", "learned", "random"))
CONDITION_IDS = tuple(row["id"] for row in ARMS)
CONTRASTS = tuple((left, right) for dose in ("low", "high") for left, right in (
    (f"learned_{dose}", f"random_{dose}"), (f"learned_{dose}", f"retire_{dose}"),
    (f"random_{dose}", f"retire_{dose}"), (f"learned_{dose}", "dense"),
    (f"retire_{dose}", "dense"), (f"random_{dose}", "dense")))
COHORTS = ("primary", "inherited", "carryover_contract")
SOURCE_GROUPS = {
    "rewiring": ("__init__.py", "plasticity.py", "run.py", "test_plasticity.py", "test_run.py", "study.py", "test_study.py"),
    "continuity": ("model.py", "world.py", "run.py", "evaluate.py"),
    "integration": ("run.py", "diagnose.py"),
    "credit": ("run.py", "local.py", "batch.py", "windows_runner.py", "evaluate_ready.py"),
}


def _plan(protocol, root, audit, inputs):
    plan = _read_json(protocol)
    for key, expected in (("format", "rewiring-study-protocol-v1"), ("seeds", list(SEEDS)),
            ("arms", list(ARMS)), ("phase_ticks", 65536), ("target_ticks", 131072),
            ("chunk_ticks", 64), ("window_updates", 8), ("training_device", "cuda"),
            ("evaluation_device", "cpu"), ("evaluation_steps", 2048), ("dt", .25),
            ("conditions", list(CONDITIONS)), ("phases", list(PHASES)),
            ("regimes", {"B": {"duration": [4., 8.], "gap": [4., 12.]},
                         "A2": {"duration": [1., 4.], "gap": [.5, 2.]}})):
        audit.equal("PROTOCOL", key, plan.get(key), expected)
    _verify_files(plan.get("sources"), audit, "FROZEN_SOURCE_SHA256")
    required = {f"experiments/{group}/{name}" for group, names in SOURCE_GROUPS.items() for name in names}
    actual = {"/".join(Path(path).parts[-3:]) for path in plan["sources"]}
    audit.require("FROZEN_RUNTIME_SOURCE_COVERAGE", str(protocol), required <= actual)
    inputs.update(plan["sources"])
    branches = plan.get("branches", [])
    audit.equal("BRANCH_COUNT", str(protocol), len(branches), 21)
    mapping, arms = {}, {row["id"]: row for row in ARMS}
    for row in branches:
        seed, condition, identifier = row.get("seed"), row.get("condition_id"), row.get("id")
        audit.require("BRANCH_ID", str(identifier), _integer(seed) and seed in SEEDS
            and condition in arms and identifier == f"seed_{seed:03d}_{condition}" and (seed, condition) not in mapping)
        audit.equal("BRANCH_ARM", identifier, {key: row.get(key) for key in ("arm", "shrink_step")},
                    {key: arms[condition][key] for key in ("arm", "shrink_step")})
        audit.equal("STRUCTURAL_SEED", identifier, row.get("structural_seed"), 91000 + seed)
        audit.equal("BRANCH_OUTPUT", identifier, Path(row["output"]).resolve(), root / "training" / f"seed_{seed:03d}" / condition)
        path, digest = _input(row["source"], inputs, audit, "BOUND_SOURCE")
        audit.equal("BOUND_SOURCE_SHA256", identifier, digest, row.get("source_sha256"))
        audit.equal("SOURCE_ABSOLUTE_PATH", identifier, str(path), row["source"])
        mapping[(seed, condition)] = row
    audit.equal("BRANCH_COVERAGE", str(protocol), set(mapping), {(seed, condition) for seed in SEEDS for condition in CONDITION_IDS})
    for seed in SEEDS:
        audit.equal("SHARED_SOURCE", str(seed), len({(mapping[(seed, condition)]["source"], mapping[(seed, condition)]["source_sha256"])
                                                   for condition in CONDITION_IDS}), 1)
    audit.equal("DISTINCT_TRAINED_SOURCES", str(protocol), len({mapping[(seed, "dense")]["source_sha256"] for seed in SEEDS}), 3)
    historical = plan.get("historical_bp64_checkpoints", [])
    audit.equal("HISTORICAL_COUNT", str(protocol), len(historical), 3)
    audit.equal("HISTORICAL_SEEDS", str(protocol), sorted(row["seed"] for row in historical), list(SEEDS))
    for row in historical:
        path, digest = _input(row["path"], inputs, audit, "HISTORICAL_CHECKPOINT")
        audit.equal("HISTORICAL_SHA256", str(path), digest, row.get("sha256"))
    return plan, mapping


def _source(path, device, audit, inputs):
    result = _source_summary(path, device, audit, inputs)
    with _preserve_process_state():
        trainer = Trainer.from_checkpoint(path, device="cpu", strict_sources=True)
        weight = trainer.model.recurrent.detach()
        audit.equal("FULL_INITIAL_SUPPORT", str(path), int(weight.ne(0).sum()), weight.numel())
        result.update(initial_mask_sha256=tensor_hash(weight.ne(0)),
                      replay_rng_hash=state_digest(trainer.replay.state_dict()["rng"]),
                      world_regime=normalize(trainer.world.state_dict()["regime"]),
                      initial_recurrent_weight_l2=float(weight.double().norm()))
    return result


def _jsonl(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    try:
        rows = [json.loads(line, object_pairs_hook=_object, parse_constant=reject)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSONL rows must be objects")
        return rows
    except (OSError, ValueError) as error:
        raise AuditError("READ_JSONL", str(path), str(error)) from error


def _row_counts(value, width, audit, where):
    audit.require("STRUCTURAL_ROW_COUNTS", where, isinstance(value, list) and len(value) == width
                  and all(_integer(count) and count <= width for count in value))
    return value


def _trace(base, credit, operator, origin, item, plan, ticks, audit, where):
    """Reconcile every complete optimizer boundary and independent edge event."""
    prefix, width = origin["trace_count"], origin["variant"]["width"]
    count = ticks // plan["chunk_ticks"]
    trace = base["trace"]
    audit.equal("TRACE_LENGTH", where, len(trace), prefix + count)
    audit.equal("TRACE_INHERITED_PREFIX", where, json_hash(trace[:prefix]), origin["trace_json_hash"])
    active, peak, minimum = [width] * width, [width] * width, [width] * width
    births, deaths, peak_total, min_total = [0] * width, [0] * width, width * width, width * width
    cumulative_births = cumulative_deaths = ever_born = 0
    survivors = width * width
    action, data, totals = origin["action_hash"], origin["data_hash"], {}
    events, windows = [], []
    trajectory = [{"seed": item["seed"], "condition_id": item["condition_id"], "arm": item["arm"], "learning_phase": "B",
        "stage_ticks": 0, "stage_update": 0, "window": False, "active": width * width, "births": 0, "deaths": 0,
        "cumulative_births": 0, "cumulative_deaths": 0, "peak_active": width * width, "min_active": width * width,
        "initial_survivors": width * width, "ever_born": 0, "proposal_candidates": 0, "eligible_dormant": 0,
        "recurrent_weight_l2": origin["initial_recurrent_weight_l2"]}]
    for index, row in enumerate(trace[prefix:], 1):
        label, stage_tick = f"{where}/{index}", index * plan["chunk_ticks"]
        for field, expected in (("stage", "credit"), ("stage_update", index), ("stage_ticks", stage_tick),
                ("update", origin["update"] + index), ("tick", origin["tick"] + stage_tick),
                ("world_tick", origin["world_tick"] + stage_tick), ("optimizer_steps_this_chunk", 1),
                ("gradient_updates", origin["gradient_updates"] + index), ("dt", plan["dt"]), ("old", None)):
            audit.equal("TRACE_BUDGET", f"{label}/{field}", row.get(field), expected)
        audit.require("TRACE_CLOCK", label, _close(row.get("now"), origin["world_now"] + stage_tick * plan["dt"]))
        audit.equal("TRACE_WORK", label, row.get("work"), _expected_work(stage_tick, plan["chunk_ticks"]))
        audit.require("TRACE_LOSS", label, _finite(row.get("loss"), 0))
        update, event = row["parameter_update"], row["parameter_update"]["structure"]
        for name in ("recurrent_delta", "encoder_delta", "head_delta", "all_delta", "gradient_norm", "clip_scale", "potential_recurrent_gradient_norm"):
            audit.require("UPDATE_FINITE", f"{label}/{name}", _finite(update.get(name), 0))
            if name != "potential_recurrent_gradient_norm":
                totals[name + "_sum"] = totals.get(name + "_sum", 0.) + update[name]
        audit.require("JOINT_CLIP", label, _close(update["clip_scale"], min(1., base["config"]["clip"] / (update["gradient_norm"] + 1e-6))))
        audit.equal("TRACE_UPDATE_NORM_KEYS", label, set(row["update_norms"]), set(totals))
        for name, expected in totals.items():
            audit.require("TRACE_UPDATE_NORMS", f"{label}/{name}", _close(row["update_norms"][name], expected, tolerance=1e-10))
        for name in ("action_chunk_hash", "data_chunk_hash", "action_hash", "data_hash", "event_hash", "live_hidden_hash"):
            audit.require("TRACE_HASH", f"{label}/{name}", _hash(row.get(name)))
        action = hashlib.sha256(bytes.fromhex(action) + bytes.fromhex(row["action_chunk_hash"])).hexdigest()
        data = hashlib.sha256(bytes.fromhex(data) + bytes.fromhex(row["data_chunk_hash"])).hexdigest()
        audit.equal("ACTION_CHAIN", label, row["action_hash"], action)
        audit.equal("DATA_CHAIN", label, row["data_hash"], data)
        events.append(row["event_hash"])
        window = index % plan["window_updates"] == 0
        for field, expected in (("arm", item["arm"]), ("shrink_step", item["shrink_step"]),
                ("updates", index), ("update", index), ("stage_update", index), ("stage_ticks", stage_tick),
                ("updates_in_window", index % plan["window_updates"]), ("window", window),
                ("initial_mask_sha256", origin["initial_mask_sha256"]), ("nonzero_hard_cuts", 0)):
            audit.equal("STRUCTURAL_EVENT", f"{label}/{field}", event.get(field), expected)
        for field in ("active", "births", "deaths", "cumulative_births", "cumulative_deaths", "initial_survivors", "ever_born",
                      "peak_active", "min_active", "proposal_candidates", "eligible_dormant", "shadow_lifetime_max", "proposal_observations"):
            audit.require("STRUCTURAL_COUNTS", f"{label}/{field}", _integer(event.get(field)))
        born = _row_counts(event["births_per_row"], width, audit, f"{label}/births")
        dead = _row_counts(event["deaths_per_row"], width, audit, f"{label}/deaths")
        audit.equal("BIRTH_EVENT_TOTAL", label, sum(born), event["births"])
        audit.equal("DEATH_EVENT_TOTAL", label, sum(dead), event["deaths"])
        audit.equal("ELIGIBLE_DORMANT", label, event["eligible_dormant"], width * width - sum(active))
        audit.require("EVENT_SUBSETS", label, all(b <= width-a and d <= a for a, b, d in zip(active, born, dead)))
        if item["arm"] == "retire" or (item["arm"] != "dense" and not window):
            audit.equal("NO_FORBIDDEN_BIRTHS", label, event["births"], 0)
        if item["arm"] in ("learned", "random"):
            audit.equal("PROPOSED_BIRTH_COUNT", label, event["births"], event["proposal_candidates"])
        audit.require("CANDIDATE_SUBSET", label, event["proposal_candidates"] <= event["eligible_dormant"])
        if not window:
            audit.equal("WINDOW_ONLY_CANDIDATES", label, event["proposal_candidates"], 0)
        before_retired = [a-d for a, d in zip(active, dead)]
        active = [a+b-d for a, b, d in zip(active, born, dead)]
        births = [a+b for a, b in zip(births, born)]
        deaths = [a+b for a, b in zip(deaths, dead)]
        peak = [max(a, b) for a, b in zip(peak, active)]
        observed_after_deaths = before_retired if item["arm"] != "dense" else active
        minimum = [min(a, b, c) for a, b, c in zip(minimum, observed_after_deaths, active)]
        peak_total, min_total = max(peak_total, sum(active)), min(min_total, sum(observed_after_deaths), sum(active))
        cumulative_births += event["births"]
        cumulative_deaths += event["deaths"]
        for field, expected in (("active", sum(active)), ("active_per_row", active), ("cumulative_births", cumulative_births),
                ("cumulative_deaths", cumulative_deaths), ("cumulative_births_per_row", births), ("cumulative_deaths_per_row", deaths),
                ("peak_active", peak_total), ("min_active", min_total), ("peak_per_row", peak), ("min_per_row", minimum)):
            audit.equal("CAPACITY_ACCOUNTING", f"{label}/{field}", event.get(field), expected)
        audit.require("SURVIVAL_ACCOUNTING", label, 0 <= event["initial_survivors"] <= min(survivors, sum(active))
            and survivors-event["initial_survivors"] <= event["deaths"]
            and ever_born <= event["ever_born"] <= min(width * width, cumulative_births)
            and event["ever_born"]-ever_born <= event["births"])
        survivors, ever_born = event["initial_survivors"], event["ever_born"]
        audit.require("STRUCTURAL_HASH", label, all(_hash(event.get(key)) for key in ("mask_sha256", "survival_sha256")))
        audit.require("RECURRENT_WEIGHT_NORM", label, _finite(event.get("recurrent_weight_l2"), 0))
        if window:
            windows.append(event)
        trajectory.append({"seed": item["seed"], "condition_id": item["condition_id"], "arm": item["arm"],
            "learning_phase": "B" if stage_tick <= plan["phase_ticks"] else "A2",
            **{field: event[field] for field in ("stage_ticks", "stage_update", "window", "active", "births", "deaths",
                "cumulative_births", "cumulative_deaths", "peak_active", "min_active", "initial_survivors", "ever_born",
                "proposal_candidates", "eligible_dormant", "recurrent_weight_l2")}})
    for field, expected in (("action_hash", action), ("data_hash", data), ("event_hash", trace[-1]["event_hash"])):
        audit.equal("FINAL_TRACE_CHAIN", f"{where}/{field}", base[field] if field != "event_hash" else base["world"][field], expected)
    for field, expected in totals.items():
        audit.require("UPDATE_NORM_SUM", f"{where}/{field}", _close(credit["update_norms"][field], expected, tolerance=1e-10))
    for field, expected in (("active", sum(active)), ("cumulative_births", cumulative_births), ("cumulative_deaths", cumulative_deaths),
                           ("initial_survivors", survivors), ("ever_born", ever_born), ("peak_active", peak_total), ("min_active", min_total)):
        audit.equal("FINAL_STRUCTURAL_TRACE", f"{where}/{field}", operator[field], expected)
    return {"count": count, "event_sequence_hash": json_hash(events), "trajectory": trajectory, "windows": windows}


def _phase_checkpoint(path, phase, item, origin, plan, binding, audit, inputs):
    path, digest = _input(path, inputs, audit, "PHASE_CHECKPOINT")
    where, ticks = f"{item['id']}/{phase}", plan["phase_ticks"] if phase == "B" else plan["target_ticks"]
    for field, expected in (("path", str(path)), ("sha256", digest), ("stage_ticks", ticks)):
        audit.equal("PHASE_BINDING", f"{where}/{field}", binding.get(field), expected)
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
        for namespace, texts, hashes in (("continuity", trainer.base.sources, trainer.base.source_hashes),
                ("credit", trainer.sources, trainer.source_hashes), ("credit-dependencies", trainer.dependencies, trainer.dependency_hashes),
                ("rewiring", trainer.rewire_sources, trainer.rewire_source_hashes),
                ("rewiring-dependencies", trainer.rewire_dependencies, trainer.rewire_dependency_hashes)):
            _snapshot_hashes(texts, hashes, audit, f"{where}/{namespace}")
        value = trainer.semantic_state()
        actual_hash = state_digest(value)
        normalized_hash = _normalized_semantic_hash(value, plan["training_device"])
        audit.equal("ACTUAL_CHECKPOINT_SEMANTIC", where, normalized_hash, binding.get("semantic_hash"))
        credit, base, operator = value["credit"], value["credit"]["continuation"], trainer.operator.summary()
        expected_config = {key: item[key] for key in ("arm", "shrink_step", "structural_seed")}
        expected_config["window_updates"] = plan["window_updates"]
        audit.equal("ACTUAL_REWIRE_CONFIG", where, value["rewire_config"], expected_config)
        audit.equal("PHASE_LENGTH", where, value["phase_ticks"], plan["phase_ticks"])
        audit.equal("PHASE_RETURNED", where, value["phase_returned"], phase == "A2")
        config = normalize(base["config"])
        config["device"] = plan["training_device"]
        audit.equal("INHERITED_TRAIN_CONFIG", where, config, origin["config"])
        audit.equal("ACTUAL_TRAINED_SEED", where, config["seed"], item["seed"])
        audit.equal("INHERITED_VARIANT", where, normalize(base["variant"]), origin["variant"])
        audit.equal("CREDIT_CONFIG", where, normalize(credit["credit_config"]), dict(mode="bp64", chunk_ticks=plan["chunk_ticks"],
            duration_range=plan["regimes"]["B"]["duration"], gap_range=plan["regimes"]["B"]["gap"], regime_name="rewiring-B"))
        updates = ticks // plan["chunk_ticks"]
        for field, expected in (("stage", "B"), ("adaptation_mode", "bp64"), ("hidden_resets", 0),
                ("ticks", origin["tick"] + ticks), ("updates", origin["update"] + updates),
                ("gradient_updates", origin["gradient_updates"] + updates), ("phase_start_tick", origin["tick"]),
                ("phase_start_update", origin["update"]), ("phase_start_gradient", origin["gradient_updates"]),
                ("initial_parameter_hash", origin["initial_parameter_hash"]), ("data_seed", origin["data_seed"])):
            audit.equal("TRAINING_BUDGET", f"{where}/{field}", base.get(field), expected)
        for inherited in (value["origin"], credit["origin"]):
            audit.equal("ORIGIN_PATH", where, Path(inherited.get("checkpoint", "")).resolve(), Path(item["source"]).resolve())
            audit.equal("ORIGIN_BYTES", where, inherited.get("checkpoint_sha256"), item["source_sha256"])
        audit.equal("INHERITED_SEMANTIC", where, value["origin"].get("inherited_semantic_hash"), origin["semantic_hash"])
        for field, expected in (("tick", origin["tick"]), ("update", origin["update"]), ("semantic_hash", origin["semantic_hash"]),
                ("inherited_source_hashes", origin["inherited_source_hashes"]),
                *((f"{key}_hash", origin["hashes"][key]) for key in ("model", "optimizer", "hidden", "world", "replay", "rng"))):
            audit.equal("ACTUAL_ORIGIN", f"{where}/{field}", credit["origin"].get(field), expected)
        audit.equal("WORK_BUDGET", where, credit["work"], _expected_work(ticks, plan["chunk_ticks"]))
        for name in ("replay", "rng"):
            audit.equal("INHERITED_STATE_UNCHANGED", f"{where}/{name}", state_digest(base[name]), origin["hashes"][name])
        audit.equal("SAMPLER_RNG_UNCHANGED", where, state_digest(credit["sampler_rng"]), origin["replay_rng_hash"])
        audit.equal("NAMED_ADAM_GROUPS", where, value["optimizer_parameter_names"], [PARAMETER_NAMES])
        adam = _adam(base["optimizer"], base["model"], [PARAMETER_NAMES], audit, where)
        for name, values in adam.items():
            audit.equal("ADAM_CONTINUOUS_STEP", f"{where}/{name}", values["step"], origin["adam"][name]["step"] + updates)
        audit.equal("INHERITED_ADAM_HYPERPARAMETERS", where, normalize(base["optimizer"]["param_groups"]), origin["optimizer_groups"])
        width, world = origin["variant"]["width"], base["world"]
        for name, tensor in {**base["model"], "hidden": base["hidden"]}.items():
            audit.require("FINITE_ACTUAL_TENSOR", f"{where}/{name}", bool(torch.isfinite(tensor).all()))
        audit.equal("MODEL_SHAPES", where, {name: list(tensor.shape) for name, tensor in base["model"].items()}, origin["model_shapes"])
        audit.equal("PERSISTENT_STATE", where, list(base["hidden"].shape), [config["batch_size"], width])
        audit.equal("WORLD_SPEC", where, normalize(world["spec"]), origin["world_spec"])
        audit.equal("WORLD_TICK", where, world["tick"], origin["world_tick"] + ticks)
        audit.require("WORLD_CLOCK", where, _close(world["now"], origin["world_now"] + ticks * plan["dt"]))
        audit.equal("WORLD_ACTION_BOUNDARY", where, world["acted"], True)
        regime = plan["regimes"][phase]
        expected_regime = {**origin["world_regime"], "duration_range": regime["duration"],
                           "gap_range": regime["gap"], "name": f"rewiring-{phase}"}
        audit.equal("LIVE_REGIME", where, normalize(world["regime"]), expected_regime)
        audit.equal("RETURN_REGIME", where, normalize(value["return_regime"]), dict(duration_range=plan["regimes"]["A2"]["duration"],
            gap_range=plan["regimes"]["A2"]["gap"], name="rewiring-A2"))
        changes = credit["regime_changes"]
        audit.equal("REGIME_CHANGE_COUNT", where, len(changes), 1 if phase == "B" else 2)
        for change, phase_name, at in zip(changes, PHASES, (0, plan["phase_ticks"])):
            for field, expected in (("stage_tick", at), ("tick", origin["tick"] + at), ("world_tick", origin["world_tick"] + at)):
                audit.equal("REGIME_BOUNDARY", f"{where}/{field}", change[field], expected)
            audit.equal("REGIME_BOUNDARY_SPEC", where, normalize(change["regime"]), {**origin["world_regime"],
                "duration_range": plan["regimes"][phase_name]["duration"], "gap_range": plan["regimes"][phase_name]["gap"], "name": f"rewiring-{phase_name}"})
        audit.equal("STRUCTURAL_UPDATES", where, operator["updates"], updates)
        audit.equal("ACTUAL_MASK_NONZEROS", where, operator["active"], int(base["model"]["recurrent"].ne(0).sum()))
        audit.equal("INITIAL_MASK", where, operator["initial_mask_sha256"], origin["initial_mask_sha256"])
        saved = value["rewiring"]
        audit.require("INITIAL_AND_BORN_PARTITION", where,
            not bool((saved["survival_mask"] & saved["ever_born"]).any())
            and not bool((saved["mask"] & ~(saved["survival_mask"] | saved["ever_born"])).any()))
        trace = _trace(base, credit, operator, origin, item, plan, ticks, audit, where)
        audit.equal("STRUCTURAL_WINDOW_TRACE", where, value["structural_windows"], trace["windows"])
        for field, expected in operator.items():
            audit.equal("FINAL_OPERATOR_TRACE", f"{where}/{field}", base["trace"][-1]["parameter_update"]["structure"].get(field), expected)
        audit.equal("FINAL_HIDDEN_TRACE", where, base["trace"][-1]["live_hidden_hash"], state_digest(base["hidden"]))
        audit.equal("LABEL_COUNT", where, int(base["stats"]["confusion"].sum()), ticks * config["batch_size"] * config["n_jobs"])
        hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world", "replay", "rng")}
        hashes.update(hidden_fork=state_digest({"hidden": base["hidden"]}), scheduler=state_digest(world["scheduler"]),
            world_without_regime_names=state_digest(_world_without_regime_names(world)),
            source_continuation=state_digest({"model": base["model"], "world": world, "hidden": base["hidden"]}))
        topology = analyze_weight(base["model"]["recurrent"].detach().cpu())
        graph = topology["graph"]["exact"]
        audit.equal("TOPOLOGY_EXACT_EDGES", where, graph["offdiagonal_edges"] + graph["self_loop_count"], operator["active"])
        structural_tensors = [tensor for tensor in value["rewiring"].values() if isinstance(tensor, torch.Tensor)]
        adam_tensors = [tensor for state in base["optimizer"]["state"].values() for tensor in state.values() if isinstance(tensor, torch.Tensor)]
        cost = {"model_parameters": sum(tensor.numel() for tensor in base["model"].values()),
            "recurrent_parameter_slots": width * width, "active_recurrent_edges": operator["active"],
            "hidden_scalars_per_actor": width, "hidden_scalars_all_actors": base["hidden"].numel(),
            "shadow_float_scalars": sum(value["rewiring"][name].numel() for name in ("shadow_m", "shadow_v", "proposal_sum")),
            "shadow_count_scalars": sum(value["rewiring"][name].numel() for name in ("shadow_count", "proposal_count")),
            "structural_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in structural_tensors),
            "live_adam_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in adam_tensors),
            "dense_recurrent_macs_per_field": width * width,
            "dense_forward_recurrent_macs": ticks * config["batch_size"] * 2 * width * width,
            "optimizer_steps": updates, "stage_wall_seconds": trainer.seconds - origin["seconds"],
            "checkpoint_bytes": path.stat().st_size}
        result = {"seed": item["seed"], "condition_id": item["condition_id"], "arm": item["arm"], "learning_phase": phase,
            "branch_id": item["id"], "checkpoint": str(path), "checkpoint_sha256": digest, "state_hash": actual_hash,
            "training_device_normalized_state_hash": normalized_hash, "stage_ticks": ticks, "hashes": hashes, "adam": adam,
            "structure": operator, "topology": topology, "cost": cost, "model_config": normalize(trainer.model.config()),
            "parameter_count": sum(tensor.numel() for tensor in base["model"].values()),
            "fork_tick": world["tick"], "fork_time": world["now"], "event_hash": world["event_hash"],
            "action_hash": base["action_hash"], "data_hash": base["data_hash"], "batch_size": config["batch_size"], "n_jobs": config["n_jobs"],
            "training_trace": {"count": trace["count"], "event_sequence_hash": trace["event_sequence_hash"]},
            "capacity_trajectory": trace["trajectory"], "structural_windows": deepcopy(value["structural_windows"]),
            "trace_json_hash": json_hash(base["trace"]), "runtime": deepcopy(trainer.runtime_history[-2]), "seconds": trainer.seconds}
        audit.equal("CHECKPOINT_LOAD_READ_ONLY", where, state_digest(trainer.semantic_state()), actual_hash)
    return result


def _audit_completed_branch(item, origin, plan, protocol_hash, audit, inputs):
    folder, where = Path(item["output"]).resolve(), item["id"]
    paths = {key: _input(path, inputs, audit)[0] for key, path in {
        "ready": folder / "checkpoint.pt", "report": folder / "report.json", "complete": folder / "launch-complete.json",
        "start": folder.with_name(folder.name + ".launch-start.json"), "trace": folder / "training.jsonl",
        "windows": folder / "structural_windows.jsonl"}.items()}
    marker, start, report = (_read_json(paths[key]) for key in ("complete", "start", "report"))
    audit.equal("START_BINDING", where, start, dict(format="rewiring-launch-start-v1", protocol_sha256=protocol_hash, branch=item))
    for field, expected in (("format", "rewiring-launch-complete-v1"), ("branch", item), ("protocol_sha256", protocol_hash), ("hidden_resets", 0)):
        audit.equal("COMPLETE_BINDING", f"{where}/{field}", marker.get(field), expected)
    audit.equal("PHASE_CHECKPOINT_SET", where, set(marker["checkpoints"]), set(PHASES))
    phases = {phase: _phase_checkpoint(folder / f"checkpoint_{phase}.pt", phase, item, origin, plan,
        marker["checkpoints"][phase], audit, inputs) for phase in PHASES}
    final = phases["A2"]
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(paths["ready"], device="cpu", strict_sources=True)
        semantic = trainer.semantic_state()
        audit.equal("READY_FINAL_CHECKPOINT", where, _normalized_semantic_hash(semantic, plan["training_device"]), final["training_device_normalized_state_hash"])
        actual = trainer.report()
        # Loading on CPU changes runtime metadata; saved training metadata is checked separately.
        for field in ("format", "variant", "model", "stage", "adaptation_mode", "updates", "stage_updates", "ticks", "stage_ticks",
                "gradient_updates", "stage_gradient_updates", "hidden_resets", "world_resets", "hidden_hash", "world_hash",
                "initial_parameter_hash", "final_parameter_hash", "action_hash", "data_hash", "event_hash", "work", "update_norms",
                "origin", "credit_config", "regime_changes", "arm", "rewire_config", "total_stage_ticks", "phase_ticks", "phase_returned",
                "return_regime", "rewire_origin", "structure", "structural_windows", "trace_points", "curve_points", "last_update",
                "rewiring_source_hashes", "rewiring_dependency_hashes", "credit_source_hashes", "dependency_hashes", "source_hashes"):
            audit.equal("CHECKPOINT_REPORT", f"{where}/{field}", report.get(field), normalize(actual.get(field)))
        audit.equal("REPORT_CONFIG", where, report.get("config"), origin["config"])
        audit.equal("RUNTIME_HISTORY", where, report.get("runtime_history"), normalize(trainer.runtime_history[:-1]))
        audit.require("RUNTIME_PRESENT", where, bool(report.get("runtime_history")))
        audit.equal("RUNTIME_LAST_TRAINING", where, report.get("runtime"), report["runtime_history"][-1])
        audit.equal("RUNTIME_DEVICE", where, report["runtime"].get("device"), plan["training_device"])
        audit.equal("WALL_SECONDS", where, report.get("seconds"), trainer.seconds)
        audit.require("WALL_SECONDS_FINITE", where, _finite(trainer.seconds, origin["seconds"]))
        suffix = semantic["credit"]["continuation"]["trace"][origin["trace_count"]:]
        audit.equal("TRAINING_JSONL", where, _jsonl(paths["trace"]), [{"event": "train", "arm": item["arm"], **row} for row in suffix])
        audit.equal("WINDOW_JSONL", where, _jsonl(paths["windows"]), semantic["structural_windows"])
        boundary = origin["trace_count"] + plan["phase_ticks"] // plan["chunk_ticks"]
        audit.equal("B_TRACE_PREFIX_OF_A2", where, json_hash(semantic["credit"]["continuation"]["trace"][:boundary]), phases["B"]["trace_json_hash"])
        for prefix, texts in (("rewiring", trainer.rewire_sources), ("credit", trainer.sources),
                ("continuity", trainer.base.sources), ("dependencies", trainer.rewire_dependencies)):
            for name, text in texts.items():
                path, _ = _input(folder / "sources" / prefix / name, inputs, audit, "SOURCE_SNAPSHOT")
                audit.equal("SOURCE_SNAPSHOT_TEXT", str(path), path.read_text(encoding="utf-8"), text)
    return phases


def _evaluation(path, item, checkpoints, plan, protocol_hash, audit, inputs):
    path, _ = _input(path, inputs, audit, "EVALUATION_FILE")
    value, records = _read_json(path), {}
    for field, expected in (("format", "rewiring-evaluation-v1"), ("branch", item),
                            ("protocol_sha256", protocol_hash), ("source_unchanged", True)):
        audit.equal("EVALUATION_PROVENANCE", f"{item['id']}/{field}", value.get(field), expected)
    audit.equal("EVALUATION_STATE_PHASES", item["id"], set(value["states"]), set(PHASES))
    for phase in PHASES:
        checkpoint = checkpoints[phase]
        expected = {field: checkpoint[field] for field in ("checkpoint", "checkpoint_sha256", "state_hash", "training_device_normalized_state_hash")}
        audit.equal("EVALUATION_STATE", f"{item['id']}/{phase}", value["states"][phase], expected)
    for record in value["records"]:
        phase, condition = record.get("learning_phase"), record.get("condition")
        key, where = (phase, condition), f"{item['id']}/{phase}/{condition}"
        audit.require("EXPECTED_FORK", where, phase in PHASES and condition in CONDITIONS)
        audit.require("DUPLICATE_FORK", where, key not in records)
        cp = checkpoints[phase]
        _audit_record(record, audit, where, dict(eval_steps=plan["evaluation_steps"], dt=plan["dt"], batch_size=cp["batch_size"], n_jobs=cp["n_jobs"]))
        for field in ("world_resets", "hidden_resets", "parameter_updates"):
            audit.require("FROZEN_FORK", f"{where}/{field}", _integer(record.get(field)) and record[field] == 0)
        audit.equal("OWN_ACTION_POLICY", where, record.get("intervention"), "none")
        audit.equal("OWN_ACTION_POLICY", where, record.get("action_policy"), "model")
        for field in ("model_action_changes", "observation_ablation_opportunities"):
            audit.equal("NO_ORACLE", f"{where}/{field}", record["opportunities"].get(field), 0)
        for field, expected in (("model", cp["model_config"]), ("parameter_hash", cp["hashes"]["model"]),
                ("source_continuation_hash", cp["hashes"]["source_continuation"]),
                ("start_world_state_hash", cp["hashes"]["world"]), ("start_hidden_hash", cp["hashes"]["hidden_fork"]),
                ("start_scheduler_hash", cp["hashes"]["scheduler"]), ("start_event_hash", cp["event_hash"]),
                ("fork_tick", cp["fork_tick"]), ("fork_time", cp["fork_time"])):
            audit.equal("ACTUAL_EVALUATION_BOUNDARY", f"{where}/{field}", record.get(field), expected)
        for field in ("action_hash", "model_action_hash", "observation_hash", "raw_observation_hash", "first_observation_hash",
                      "first_model_observation_hash", "end_world_state_hash", "end_hidden_hash"):
            audit.require("EVALUATION_HASH", f"{where}/{field}", _hash(record.get(field)))
        audit.equal("FIRST_OBSERVATION", where, record["first_observation_hash"], record["first_model_observation_hash"])
        total = record["opportunities"]
        audit.require("IDLE_SUBSET", where, total["inactive_actions"] <= total["inactive_opportunities"])
        audit.equal("DISABLED_OPPORTUNITIES", where, total["disabled_slot_opportunities"],
                    record["steps"] * record["batch_size"] * (record["n_jobs"] - record["active_jobs"]))
        for cohort in COHORTS:
            row, censored = record[cohort], record["censored"][cohort]
            audit.require("ACTOR_NONNEGATIVE", f"{where}/{cohort}", all(_integer(number) for field in ("matured", "completed", "clean_completed") for number in row["per_actor"][field]))
            audit.require("ACTOR_SUBSETS", f"{where}/{cohort}", all(0 <= clean <= complete <= mature for mature, complete, clean in
                zip(row["per_actor"]["matured"], row["per_actor"]["completed"], row["per_actor"]["clean_completed"])))
            audit.require("CENSORED_SUBSETS", f"{where}/{cohort}", all(_integer(censored.get(field)) for field in ("count", "completed", "clean_completed"))
                and 0 <= censored["clean_completed"] <= censored["completed"] <= censored["count"])
        records[key] = record
    audit.equal("FORK_COVERAGE", item["id"], set(records), {(phase, condition) for phase in PHASES for condition in CONDITIONS})
    return records


def _historical(plan, checkpoints, origins, audit, inputs):
    rows = []
    for reference in plan["historical_bp64_checkpoints"]:
        seed, path = reference["seed"], Path(reference["path"]).resolve()
        path, digest = _input(path, inputs, audit, "HISTORICAL_CHECKPOINT")
        audit.equal("HISTORICAL_SHA256", str(path), digest, reference["sha256"])
        current, origin = checkpoints[(seed, "dense", "B")], origins[seed]
        with _preserve_process_state():
            old = CreditTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
            for namespace, texts, hashes in (("continuity", old.base.sources, old.base.source_hashes),
                    ("credit", old.sources, old.source_hashes), ("dependencies", old.dependencies, old.dependency_hashes)):
                _snapshot_hashes(texts, hashes, audit, f"historical/{seed}/{namespace}")
            value, base = old.semantic_state(), old.semantic_state()["continuation"]
            for field, actual, expected in (("mode", value["credit_config"]["mode"], "bp64"),
                    ("seed", base["config"]["seed"], seed), ("ticks", old.stage_ticks, plan["phase_ticks"]),
                    ("source", value["origin"].get("checkpoint_sha256"), origin["sha256"]), ("hidden_resets", base["hidden_resets"], 0)):
                audit.equal("HISTORICAL_BINDING", f"{seed}/{field}", actual, expected)
            hashes = {name: state_digest(base[name]) for name in ("model", "optimizer", "hidden", "world")}
            hashes.update(action_hash=base["action_hash"], data_hash=base["data_hash"], world_without_regime_names=state_digest(_world_without_regime_names(base["world"])))
            new = {**{name: current["hashes"][name] for name in ("model", "optimizer", "hidden", "world", "world_without_regime_names")},
                   "action_hash": current["action_hash"], "data_hash": current["data_hash"]}
            equal = {name: hashes[name] == new[name] for name in hashes}
            required = ("model", "optimizer", "hidden", "action_hash", "data_hash", "world_without_regime_names")
            audit.require("DENSE_B_HISTORICAL_REPRODUCTION", str(seed), all(equal[name] for name in required), equal)
            rows.append({"seed": seed, "historical_checkpoint": str(path), "historical_sha256": digest,
                "new_checkpoint": current["checkpoint"], "historical_hashes": hashes, "new_hashes": new, "equal": equal,
                "core_reproduction_equal": all(equal[name] for name in required),
                "world_normalization": "Remove only regime.name and regime_history[*].name. Raw world equality is reported separately; semantic tensors and all other metadata must match."})
    return rows


def build_report(protocol, root):
    protocol, root = Path(protocol).resolve(), Path(root).resolve()
    audit, inputs = _Checks(), {}
    protocol, protocol_hash = _input(protocol, inputs, audit, "PROTOCOL_FILE")
    analysis_files = {str(path): file_sha256(path) for path in {
        Path(__file__).resolve(), Path(__file__).with_name("test_report.py").resolve(),
        Path(_audit_record.__code__.co_filename).resolve(), Path(_Checks.result.__code__.co_filename).resolve(),
        Path(_source_summary.__code__.co_filename).resolve(), Path(analyze_weight.__code__.co_filename).resolve()}}
    try:
        plan, branches = _plan(protocol, root, audit, inputs)
        expected_eval = {root / "evaluation" / f"{row['id']}.json" for row in branches.values()}
        audit.equal("EVALUATION_FILE_SET", str(root), set((root / "evaluation").rglob("*.json")), expected_eval)
        audit.equal("COMPLETION_FILE_SET", str(root), set((root / "training").rglob("launch-complete.json")),
                    {Path(row["output"]) / "launch-complete.json" for row in branches.values()})
        origins = {seed: _source(branches[(seed, "dense")]["source"], plan["training_device"], audit, inputs) for seed in SEEDS}
        for seed, origin in origins.items():
            audit.equal("ACTUAL_SOURCE_SEED", str(seed), origin["config"]["seed"], seed)
        checkpoints, records, per_seed, trajectory, windows, structural, costs = {}, {}, [], [], [], [], []
        training_pairing, evaluation_pairing = {}, {}
        for seed in SEEDS:
            for condition_id in CONDITION_IDS:
                item = branches[(seed, condition_id)]
                phases = _audit_completed_branch(item, origins[seed], plan, protocol_hash, audit, inputs)
                for phase, cp in phases.items():
                    checkpoints[(seed, condition_id, phase)] = cp
                    identity = {"scheduler": cp["hashes"]["scheduler"], "event_hash": cp["event_hash"],
                        "trace_event_sequence": cp["training_trace"]["event_sequence_hash"], "fork_tick": cp["fork_tick"], "fork_time": cp["fork_time"]}
                    key = seed, phase
                    if key in training_pairing:
                        audit.equal("TRAINING_EXOGENOUS_PAIRING", f"{item['id']}/{phase}", identity, training_pairing[key])
                    training_pairing[key] = identity
                    for metric in ("active", "cumulative_births", "cumulative_deaths", "peak_active", "min_active", "initial_survivors", "ever_born"):
                        structural.append(dict(seed=seed, condition_id=condition_id, learning_phase=phase, metric=metric, value=cp["structure"][metric]))
                    for metric in ("births", "deaths"):
                        previous = phases["B"]["structure"][f"cumulative_{metric}"] if phase == "A2" else 0
                        structural.append(dict(seed=seed, condition_id=condition_id, learning_phase=phase,
                            metric=f"phase_{metric}", value=cp["structure"][f"cumulative_{metric}"]-previous))
                    costs += [dict(seed=seed, condition_id=condition_id, learning_phase=phase, metric=metric, value=value)
                              for metric, value in cp["cost"].items()]
                trajectory += phases["A2"].pop("capacity_trajectory")
                windows += [{"seed": seed, "condition_id": condition_id, "learning_phase": "B" if row["stage_ticks"] <= plan["phase_ticks"] else "A2", **row}
                            for row in phases["A2"].pop("structural_windows")]
                phases["B"].pop("capacity_trajectory")
                phases["B"].pop("structural_windows")
                evaluated = _evaluation(root / "evaluation" / f"{item['id']}.json", item, phases, plan, protocol_hash, audit, inputs)
                for (phase, condition), row in evaluated.items():
                    records[(seed, condition_id, phase, condition)] = row
                    key, identity = (seed, phase, condition), _exogenous(row)
                    if key in evaluation_pairing:
                        audit.equal("EVALUATION_EXOGENOUS_PAIRING", f"{item['id']}/{phase}/{condition}", identity, evaluation_pairing[key])
                    evaluation_pairing[key] = identity
                    per_seed += [dict(seed=seed, condition_id=condition_id, learning_phase=phase, condition=condition, metric=metric, value=value)
                                 for metric, value in _metrics(row).items()]
        audit.equal("FORK_COUNT", str(root), len(records), 210)
        contrasts, phase_changes = [], []
        for seed in SEEDS:
            for condition in CONDITIONS:
                for phase in PHASES:
                    for left, right in CONTRASTS:
                        a, b = (_metrics(records[(seed, arm, phase, condition)]) for arm in (left, right))
                        contrasts += [dict(seed=seed, condition=condition, learning_phase=phase, left=left, right=right,
                            contrast=f"{left}-minus-{right}", metric=metric, value=a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None) for metric in a]
                for arm in CONDITION_IDS:
                    a, b = (_metrics(records[(seed, arm, phase, condition)]) for phase in ("A2", "B"))
                    phase_changes += [dict(seed=seed, condition_id=arm, condition=condition, contrast="A2-minus-B", metric=metric,
                        value=a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None) for metric in a]
        historical = _historical(plan, checkpoints, origins, audit, inputs)
        _verify_files(inputs, audit, "FINAL_INPUT_SHA256")
        _verify_files(analysis_files, audit, "ANALYSIS_SOURCE_SHA256")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise AuditError("MALFORMED_SCHEMA", str(root), str(error), checks=sum(audit.counts.values())) from error
    return {"format": FORMAT_VERSION, "complete": True, "audit": audit.result(),
        "counts": dict(branches=21, training_runs=21, phase_checkpoints=42, evaluation_files=21, forks=210),
        "protocol": str(protocol), "protocol_sha256": protocol_hash, "raw_root": str(root),
        "analysis_source_files": analysis_files, "input_files": inputs, "origins": list(origins.values()),
        "checkpoints": list(checkpoints.values()), "per_seed": per_seed,
        "main_summary": _summaries(per_seed, ("condition_id", "learning_phase", "condition", "metric")),
        "contrast_per_seed": contrasts, "paired_contrasts": _summaries(contrasts, ("learning_phase", "condition", "left", "right", "contrast", "metric")),
        "phase_change_per_seed": phase_changes, "phase_changes": _summaries(phase_changes, ("condition_id", "condition", "contrast", "metric")),
        "structural_per_seed": structural, "structural_summary": _summaries(structural, ("condition_id", "learning_phase", "metric")),
        "cost_per_seed": costs, "cost_summary": _summaries(costs, ("condition_id", "learning_phase", "metric")),
        "capacity_trajectory": trajectory, "structural_windows": windows, "historical_dense_B_comparison": historical,
        "statistics": {"replicate": "trained seed", "seeds": list(SEEDS), "sample_sd": "Sample SD across trained seeds, denominator n-1.",
            "paired": "Differences use the same trained seed, learning phase and evaluation condition; phase changes pair A2 with B within the same continuation.",
            "undefined": "Zero-denominator rates remain null; n_total is three and n_undefined is explicit.",
            "doses": "Both predeclared shrink doses are reported. No dose is selected using these outcomes."},
        "capacity_definitions": {"active": "Actual exact-nonzero recurrent entries W!=0, including diagonal entries.",
            "birth_death": "Independent counts; current = initial + cumulative births - cumulative deaths. Equal birth and death counts are not imposed.",
            "phase_births_and_phase_deaths": "B uses its endpoint cumulative counts. A2 uses final cumulative counts minus the B endpoint; A2 cumulative fields still cover the entire B+A2 continuation.",
            "ever_born": "Unique recurrent pair positions reactivated at least once. Every source starts fully connected, so these are previously retired positions, not novel allowed pairs or new neurons.",
            "cost": "All arms retain the dense 128x128 recurrent parameter, dense forward/backward arithmetic and inherited Adam storage. Shadow proposal state adds dense storage; active support is not a measured FLOP reduction.",
            "dense_forward_recurrent_macs": "Two Heun field evaluations per .25 live tick times dense W slots, batch and stage ticks; excludes encoder/head, backward, clipping, Adam, shrink and proposal arithmetic.",
            "storage": "Actual tensor bytes are counted for live Adam and all structural state tensors, including the private structural generator. Python/allocator overhead is excluded.",
            "stage_wall_seconds": "Saved phase seconds minus inherited A seconds, cumulative over B+A2 at A2. This includes retained completed optimizer boundaries; partial work discarded during controlled stop/replay is excluded. It is not end-to-end process wall time, and concurrent runs affect throughput.",
            "topology": "CPU float64 summaries use the complete learned W. Directed cycles and closed walks do not measure memory duration or establish task necessity."},
        "limits": deepcopy(plan.get("limits", []))}


def write_artifacts(report, output):
    output, raw = Path(output).resolve(), Path(report["raw_root"]).resolve()
    if output == raw or raw in output.parents:
        raise ValueError("output must be outside the raw-input directory")
    _atomic_text(output / "report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name in ("per_seed", "main_summary", "contrast_per_seed", "paired_contrasts", "phase_change_per_seed", "phase_changes",
                 "structural_per_seed", "structural_summary", "cost_per_seed", "cost_summary", "capacity_trajectory", "structural_windows"):
        rows, stream = report[name], io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
        _atomic_text(output / f"{name}.csv", stream.getvalue())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("path", "sha256"))
    writer.writerows(sorted(report["input_files"].items()))
    _atomic_text(output / "input_files.csv", stream.getvalue())


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
    except (AuditError, ValueError, OSError) as error:
        parser.exit(1, f"Rewiring audit rejected: {error}\n")
    print(f"Validated 21 continuations / 42 phase checkpoints / 210 frozen forks; saved {args.output.resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
