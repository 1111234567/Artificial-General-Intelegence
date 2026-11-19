"""Fail-closed audit of all 27 organization lifetimes and 648 frozen forks.

Checkpoint restoration always uses the organization trainer. Frozen lifelong
helpers are reused only for unchanged trajectory, reservoir and denominator
bookkeeping. Functional eligibility lifetimes are distinct from numerical zero
crossings and from claims about useful modular computation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong import report as legacy
from experiments.lifelong.replay import block_hash
from experiments.loops.report import AuditError, _Checks, _finite, _hash, _integer, _read_json, file_sha256
from experiments.organization import run as native
from experiments.organization import study


FORMAT_VERSION = "organization-study-summary-v1"
ROOT = Path(__file__).resolve().parents[2]
BRANCH_KEYS = ("field_mode", "structural_rule")
SEEDS = (0, 1, 2)
normalize, json_hash, _same = legacy.normalize, legacy.json_hash, legacy._same
_input, _fields, _exposures = legacy._input, legacy._fields, legacy._exposures
_difference, _rate = legacy._difference, legacy._rate
_METADATA = ("tick", "family", "condition", "steps")
CONTRASTS = (
    ("passive_parallel_minus_plain", ("parallel", "passive"), ("plain", "passive")),
    ("passive_coupled_minus_plain", ("coupled", "passive"), ("plain", "passive")),
    ("passive_coupled_minus_parallel", ("coupled", "passive"), ("parallel", "passive")),
    ("plain_task_minus_fixed", ("plain", "functional_task"), ("plain", "functional_fixed")),
    ("plain_magnitude_minus_fixed", ("plain", "functional_magnitude"), ("plain", "functional_fixed")),
    ("plain_random_minus_fixed", ("plain", "functional_random"), ("plain", "functional_fixed")),
    ("coupled_task_minus_fixed", ("coupled", "functional_task"), ("coupled", "functional_fixed")),
    ("functional_fixed_coupled_minus_plain", ("coupled", "functional_fixed"), ("plain", "functional_fixed")),
    ("functional_task_coupled_minus_plain", ("coupled", "functional_task"), ("plain", "functional_task")),
)


def expected_work(ticks, chunk, batch, jobs=4, rule="passive", window=8):
    work = legacy.expected_work(ticks, chunk, batch, jobs)
    calls = 4 * (max(0, ticks // chunk - 1) // window) if rule != "passive" else 0
    work.update(structural_probe_calls=calls, structural_probe_forward_ticks=calls * 3 * chunk,
        structural_probe_warmup_ticks=calls * chunk, structural_probe_loss_ticks=calls * 2 * chunk,
        structural_probe_forward_actor_ticks=calls * 3 * chunk * batch,
        structural_probe_loss_positions=calls * 2 * chunk * batch * jobs)
    return work


# The unchanged trace invariants are copied here with an organization-specific
# chain prefix, work vector and additional functional-event audit.
def _audit_trace(state, trainer, audit, where):
    config, trace = state["config"], state["trace"]
    chunk, batch, jobs, updates = config["chunk_ticks"], config["batch_size"], config["n_jobs"], state["updates"]
    audit.equal("TRACE_LENGTH", where, len(trace), updates)
    action = data = hashlib.sha256(f"{native.FORMAT_VERSION}/{state['birth']['data_seed']}".encode()).hexdigest()
    sources, source_counts, suffix_counts, ages = Counter(), Counter(), Counter(), []
    previous_exposures = {key: [0] * 4 for key in ("new", "replace", "commands")}
    for index, row in enumerate(trace, 1):
        tick, label = index * chunk, f"{where}/update_{index}"
        for key, expected in (("tick", tick), ("update", index), ("world_tick", tick - 1),
                ("gradient_updates", index), ("hidden_resets", 0),
                ("history_block_admitted", index % 2 == 0),
                ("phase", trainer.schedule.mode_for_tick(tick - chunk)["phase"]),
                ("reservoir_hash_kind", "compositional-full-content-v1")):
            audit.equal("TRACE_BUDGET", f"{label}/{key}", row.get(key), expected)
        for key in ("now", "hidden_time"):
            audit.require("TRACE_PHYSICAL_CLOCK", f"{label}/{key}", _same(row.get(key), (tick - 1) * config["dt"]))
        for key in ("action_chunk_hash", "data_chunk_hash", "action_hash", "data_hash", "event_hash",
                    "live_hidden_hash", "live_world_hash", "reservoir_hash", "operator_hash", "rng_hash"):
            audit.require("TRACE_HASH", f"{label}/{key}", _hash(row.get(key)))
        action = hashlib.sha256(bytes.fromhex(action) + bytes.fromhex(row["action_chunk_hash"])).hexdigest()
        data = hashlib.sha256(bytes.fromhex(data) + bytes.fromhex(row["data_chunk_hash"])).hexdigest()
        audit.equal("ACTION_HASH_CHAIN", label, row["action_hash"], action)
        audit.equal("DATA_HASH_CHAIN", label, row["data_hash"], data)
        for key in ("actual_action_counts", "target_action_counts"):
            counts = row[key]
            audit.require("ACTION_COUNT_BUDGET", f"{label}/{key}", isinstance(counts, list) and len(counts) == 11
                and all(_integer(number) for number in counts) and sum(counts) == chunk * batch * jobs)
        info = row["auxiliary"]
        source = ("startup" if index == 1 else "recent_repeat" if config["learning"] == "recent_repeat"
                  else "fallback_recent" if index == 2 else "history_replay")
        audit.equal("AUXILIARY_SOURCE", label, info["source"], source)
        sources[source] += 1
        for key, expected in (("forward_ticks", chunk if index == 1 else 2 * chunk),
                ("warmup_ticks", 0 if index == 1 else chunk), ("loss_ticks", chunk)):
            audit.equal("MATCHED_AUXILIARY_WORK", f"{label}/{key}", info[key], expected)
        audit.require("AUXILIARY_INTERVAL", label,
            _integer(info["start_tick"]) and _integer(info["end_tick"], 1)
            and info["end_tick"] - info["start_tick"] == info["forward_ticks"]
            and info["start_tick"] % chunk == 0 and info["end_tick"] % chunk == 0)
        if source == "history_replay":
            audit.require("NO_FUTURE_REPLAY", label, info["end_tick"] <= tick - chunk
                and info["start_tick"] % (2 * chunk) == 0)
        else:
            audit.equal("RECENT_INTERVAL", label, (info["start_tick"], info["end_tick"]),
                        (max(0, tick - 2 * chunk), tick))
        age = max(0, tick - chunk - info["end_tick"])
        audit.equal("HISTORY_AGE", label, info["age_ticks"], age)
        ages.append(age)
        audit.require("AUXILIARY_CONTENT_HASH", label, _hash(info["source_hash"]) and isinstance(info["source_id"], str))
        for key, aggregate in (("family_counts", source_counts), ("suffix_family_positions", suffix_counts)):
            audit.equal("AUXILIARY_FAMILY_KEYS", f"{label}/{key}", set(info[key]), set(range(4)))
            audit.require("AUXILIARY_FAMILY_COUNTS", f"{label}/{key}", all(_integer(count) for count in info[key].values()))
            aggregate.update(info[key])
        audit.require("SUFFIX_FAMILY_POPULATION", label, sum(info["suffix_family_positions"].values()) <= chunk * batch * jobs)
        audit.require("LOSS_FINITE", label, all(_finite(value, 0) for value in
            (row["live_loss"], info["loss"], row["combined_loss"])))
        audit.require("GRADIENT_OBJECTIVE_MIX", label,
            _same(row["combined_loss"], .5 * row["live_loss"] + .5 * info["loss"]))
        exposure = row["actual_family_exposures"]
        _exposures(exposure, audit, label)
        audit.require("EXPOSURE_MONOTONE", label, all(a >= b for key in exposure
            for a, b in zip(exposure[key], previous_exposures[key])))
        previous_exposures = exposure
        update = row["update_metrics"]
        audit.require("UPDATE_FINITE", label, all(_finite(update[key], 0) for key in
            ("gradient_norm", "clip_scale", "raw_recurrent_gradient_norm", "eligible_recurrent_gradient_norm", "recurrent_delta", "all_delta")))
        audit.require("JOINT_CLIP_SCALE", label,
            _same(update["clip_scale"], min(1., config["clip"] / (update["gradient_norm"] + 1e-6))))
    for key, expected in (("action_hash", action), ("data_hash", data)):
        audit.equal("FINAL_TRACE_CHAIN", f"{where}/{key}", state[key], expected)
    if trace:
        for key, expected in (("event_hash", trainer.world.event_hash),
                ("live_hidden_hash", state_digest(state["hidden"])), ("live_world_hash", state_digest(state["world"])),
                ("reservoir_hash", trainer.replay.retention_fingerprint()), ("operator_hash", state_digest(state["operator"]))):
            audit.equal("FINAL_TRACE_STATE", f"{where}/{key}", trace[-1][key], expected)
        audit.equal("FINAL_EXPOSURES", where, previous_exposures, trainer.world.family_exposures())
    auxiliary = state["auxiliary_stats"]
    for key in ("startup", "fallback_recent", "recent_repeat", "history_replay"):
        audit.equal("AUXILIARY_CALL_ACCOUNTING", f"{where}/{key}", auxiliary[key], sources[key])
    audit.equal("AUXILIARY_WINDOW_COMMANDS", where, auxiliary["source_family_counts"], dict.fromkeys(range(4), 0) | dict(source_counts))
    audit.equal("AUXILIARY_SUFFIX_POSITIONS", where, auxiliary["suffix_family_positions"], dict.fromkeys(range(4), 0) | dict(suffix_counts))
    audit.equal("AUXILIARY_AGE_SUM", where, auxiliary["source_age_ticks_sum"], sum(ages))
    audit.equal("AUXILIARY_AGE_MAX", where, auxiliary["source_age_ticks_max"], max(ages, default=0))
    audit.equal("REPLAY_SAMPLE_CALLS", where, trainer.replay.samples_drawn, sources["history_replay"])
    audit.equal("EXACT_COMPUTE_WORK", where, state["work"], expected_work(state["ticks"], chunk, batch, jobs, config["structural_rule"], config["structural_window_updates"]))
    _audit_schedule_and_statistics(state, trainer, audit, where)
    _audit_functional_trace(state, trainer, audit, where)
    return json_hash([row["event_hash"] for row in trace])


def _audit_schedule_and_statistics(state, trainer, audit, where):
    chunk = state["config"]["chunk_ticks"]
    history = [{"tick": 0, **trainer.schedule.mode_for_tick(0)}]
    # ScheduleConfig requires every possible boundary to align with chunk.
    for tick in range(chunk, state["ticks"], chunk):
        policy = trainer.schedule.mode_for_tick(tick)
        if any(policy[key] != history[-1][key] for key in policy):
            history.append({"tick": tick, **policy})
    audit.equal("COMPLETE_CONTINUOUS_COURSE_HISTORY", where, state["schedule_history"], history)
    predictions, targets = [0] * 11, [0] * 11
    last_auxiliary = [-1] * 4
    for row in state["trace"]:
        predictions = [a + b for a, b in zip(predictions, row["actual_action_counts"])]
        targets = [a + b for a, b in zip(targets, row["target_action_counts"])]
        for family in range(4):
            if row["auxiliary"]["suffix_family_positions"][family]:
                last_auxiliary[family] = row["tick"]
    confusion = state["stats"]["confusion"]
    audit.equal("TRACE_ACTION_STATISTICS_TARGETS", where, confusion.sum(1).tolist(), targets)
    audit.equal("TRACE_ACTION_STATISTICS_PREDICTIONS", where, confusion.sum(0).tolist(), predictions)
    audit.equal("TRACE_ACTUAL_AUXILIARY_RECENCY", where, state["last_aux_family_tick"], last_auxiliary)
    if state["trace"]:
        audit.equal("FINAL_GLOBAL_RNG_TRACE_BINDING", where, state["trace"][-1]["rng_hash"], state_digest(state["rng"]))
        audit.equal("FINAL_FAMILY_RECENCY_TRACE_BINDING", where,
                    state["trace"][-1]["family_recency"], trainer.family_recency())


def _proposal_fields():
    return {"generated_update", "execute_update", "adam_step", "proposal_lr", "birth_coordinates",
            "birth_delta", "birth_scores", "birth_weights_at_proposal", "exit_coordinates", "exit_scores",
            "exit_weights_at_proposal"}


def _coordinates(values, width, maximum, audit, where):
    audit.require("FUNCTIONAL_COORDINATE_SCHEMA", where, isinstance(values, list) and len(values) <= maximum
        and all(isinstance(pair, list) and len(pair) == 2 and all(_integer(x) and x < width for x in pair)
                for pair in values))
    audit.equal("FUNCTIONAL_COORDINATE_UNIQUENESS", where, len({tuple(pair) for pair in values}), len(values))
    return [tuple(pair) for pair in values]


def _audit_functional_trace(state, trainer, audit, where):
    config = state["config"]
    if config["structural_rule"] == "passive":
        audit.equal("PASSIVE_ZERO_PROBE_BUDGET", where, state["work"]["structural_probe_calls"], 0)
        return
    op, trace = state["operator"], state["trace"]
    initial = op["initial_mask"].clone()
    allowed = initial.clone()
    pending, events = None, []
    window, limit = config["structural_window_updates"], config["structural_candidates"]
    for index, row in enumerate(trace, 1):
        label = f"{where}/functional_{index}"
        structure = row["update_metrics"]["structure"]
        audit.equal("FULL_COORDINATE_CLIP_GRADIENT", label,
            row["update_metrics"]["raw_recurrent_gradient_norm"], row["update_metrics"]["eligible_recurrent_gradient_norm"])
        probe = structure.get("functional_probe")
        if pending is None:
            audit.equal("NO_UNSCHEDULED_FUNCTIONAL_PROBE", label, probe, None)
        else:
            audit.require("FUNCTIONAL_NEXT_BLOCK_PROBE", label, isinstance(probe, dict))
            audit.equal("FUNCTIONAL_PROPOSAL_EXECUTION_BINDING", label,
                {key: probe[key] for key in _proposal_fields()}, pending)
            audit.equal("FUNCTIONAL_PROBE_NEXT_UPDATE", label, probe["executed_update"], index)
            audit.equal("FUNCTIONAL_FOUR_CALL_BUDGET", label, probe["probe_calls"], 4)
            birth_coords = _coordinates(probe["birth_coordinates"], config["width"], limit, audit, label)
            exit_coords = _coordinates(probe["exit_coordinates"], config["width"], limit, audit, label)
            born = _coordinates(probe["actual_birth_coordinates"], config["width"], limit, audit, label)
            retired = _coordinates(probe["actual_retire_coordinates"], config["width"], limit, audit, label)
            audit.equal("FUNCTIONAL_ACCEPTED_BIRTH_COORDINATES", label,
                set(born), set(birth_coords) if probe["birth_accepted"] else set())
            exit_zero = {pair for pair, value in zip(exit_coords, probe["exit_trial_weights"]) if value == 0.}
            audit.equal("FUNCTIONAL_ACCEPTED_RETIRE_COORDINATES", label,
                set(retired), exit_zero if probe["exit_accepted"] else set())
            for kind, coordinates, expected in (("birth", birth_coords, False), ("exit", exit_coords, True)):
                audit.require("FUNCTIONAL_TRIAL_ELIGIBILITY", f"{label}/{kind}",
                    all(bool(allowed[pair]) == expected for pair in coordinates))
            for pair in born:
                allowed[pair] = True
            for pair in retired:
                allowed[pair] = False
            audit.equal("FUNCTIONAL_EVENT_COUNTS", label, (structure["births"], structure["deaths"]), (len(born), len(retired)))
            sources = structure["probe_sources"]
            info = row["auxiliary"]
            for key, expected in (("live_start_tick", row["tick"] - config["chunk_ticks"]),
                    ("live_end_tick", row["tick"]), ("history_source_hash", info["source_hash"]),
                    ("history_start_tick", info["start_tick"]), ("history_end_tick", info["end_tick"]),
                    ("history_warmup_ticks", info["warmup_ticks"])):
                audit.equal("FUNCTIONAL_REAL_BLOCK_SOURCE", f"{label}/{key}", sources[key], expected)
            audit.require("FUNCTIONAL_LIVE_SOURCE_HASH", label, _hash(sources["live_source_hash"]))
            if index == state["updates"]:
                audit.equal("FUNCTIONAL_LAST_LIVE_SOURCE_CONTENT", label, sources["live_source_hash"], block_hash(state["last_chunk"]))
            events.append(probe)
            pending = None
        audit.equal("FUNCTIONAL_TRACE_ALLOWED_COUNT", label, structure["allowed"], int(allowed.sum()))
        audit.equal("FUNCTIONAL_TRACE_ALLOWED_HASH", label, structure["allowed_mask_sha256"], _mask_hash(allowed))
        generated = structure.get("pending_generated")
        if index % window == 0:
            _fields(generated, _proposal_fields(), audit, label)
            for key, expected in (("generated_update", index), ("execute_update", index + 1), ("adam_step", index)):
                audit.equal("FUNCTIONAL_PENDING_CLOCK", f"{label}/{key}", generated[key], expected)
            for kind in ("birth", "exit"):
                coords = _coordinates(generated[f"{kind}_coordinates"], config["width"], limit, audit, label)
                for suffix in ("scores", "weights_at_proposal", *(("delta",) if kind == "birth" else ())):
                    values = generated[f"{kind}_{suffix}"]
                    audit.require("FUNCTIONAL_PENDING_VALUES", label, isinstance(values, list) and len(values) == len(coords)
                        and all(_finite(value) for value in values))
                audit.require("FUNCTIONAL_PROPOSAL_ELIGIBILITY", f"{label}/{kind}",
                    all(bool(allowed[pair]) == (kind == "exit") for pair in coords))
            audit.require("FUNCTIONAL_NONZERO_BIRTH_PROPOSALS", label,
                all(value != 0 for value in generated["birth_delta"]))
            audit.require("FUNCTIONAL_DORMANT_PROPOSAL_WEIGHTS_ZERO", label,
                all(value == 0 for value in generated["birth_weights_at_proposal"]))
            pending = generated
        else:
            audit.equal("FUNCTIONAL_NO_EARLY_PROPOSAL", label, generated, None)
    audit.equal("FUNCTIONAL_OPERATOR_EVENT_HISTORY", where, events, op["event_history"])
    audit.equal("FUNCTIONAL_FINAL_ELIGIBILITY", where, _mask_hash(allowed), _mask_hash(op["allowed_mask"]))
    audit.equal("FUNCTIONAL_FINAL_PENDING", where, pending,
        None if op["pending"] is None else {key: value.tolist() if isinstance(value, torch.Tensor) else value
                                           for key, value in op["pending"].items()})
    if op["pending"] is not None:
        _audit_pending_adam(state, audit, where)


def _mask_hash(value):
    from experiments.rewiring.plasticity import tensor_hash
    return tensor_hash(value)


def _audit_pending_adam(state, audit, where):
    """The end-window pending delta is checked against real saved Adam slots."""
    pending, weight = state["operator"]["pending"], state["model"]["recurrent"]
    index = state["optimizer_parameter_names"].index("recurrent")
    slots, group = state["optimizer"]["state"][index], state["optimizer"]["param_groups"][0]
    step, (beta1, beta2) = float(slots["step"]), group["betas"]
    corrected_m = slots["exp_avg"].double() / (1. - beta1 ** step)
    corrected_v = slots.get("max_exp_avg_sq", slots["exp_avg_sq"]).double() / (1. - beta2 ** step)
    delta = (-group["lr"] * corrected_m / (corrected_v.sqrt() + group["eps"])).to(weight.dtype)
    coords = pending["birth_coordinates"]
    actual = delta[coords[:, 0], coords[:, 1]]
    audit.require("FUNCTIONAL_PENDING_REAL_ADAM_DIRECTION", where, torch.equal(actual, pending["birth_delta"]))
    audit.equal("FUNCTIONAL_PENDING_REAL_ADAM_LR", where, pending["proposal_lr"], group["lr"])
    for kind in ("birth", "exit"):
        coords = pending[f"{kind}_coordinates"]
        audit.require("FUNCTIONAL_PENDING_REAL_PROPOSAL_WEIGHT", f"{where}/{kind}",
            torch.equal(weight[coords[:, 0], coords[:, 1]], pending[f"{kind}_weights_at_proposal"]))


def structural_history(state, audit=None, where="structure"):
    audit = _Checks() if audit is None else audit
    functional = state["config"]["structural_rule"] != "passive"
    op, weight = state["operator"], state["model"]["recurrent"]
    if functional:
        initial, support, ever = op["initial_mask"], op["_support_mask"], op["ever_born"]
        allowed, survival = op["allowed_mask"], op["survival_mask"]
        births, deaths = op["cumulative_births"], op["cumulative_deaths"]
    else:
        old = legacy.structural_history(state, audit, where)
        initial, support = op["inherited_operator"]["mask"], op["operator"]["mask"]
        ever, survival = op["factor_ever_born"], op["factor_survival_mask"]
        allowed = op["allowed_mask"]
        births, deaths = op["operator"]["cumulative_births"], op["operator"]["cumulative_deaths"]
    audit.require("STRUCTURE_EXACT_SUPPORT", where, torch.equal(support, weight.ne(0)))
    audit.equal("STRUCTURE_BIRTH_MASK_HASH", where, state_digest(initial), state["birth"]["initial_mask_hash"])
    half = weight.shape[0] // 2
    cross = torch.zeros_like(initial)
    cross[:half, half:] = cross[half:, :half] = True
    audit.require("STRUCTURE_NO_INITIAL_BRIDGES", where, not bool((initial & cross).any()))
    return dict(initial_edges=int(initial.sum()), active_edges=int(support.sum()), allowed_edges=int(allowed.sum()),
        active_edges_outside_initial=int((support & ~initial).sum()),
        unique_edges_ever_born_outside_initial=int((ever & ~initial).sum()),
        unique_cross_bank_coordinates_ever_born=int((ever & cross).sum()),
        active_cross_bank_bridges=int((support & cross).sum()), allowed_cross_bank_bridges=int((allowed & cross).sum()),
        initial_survivors=int(survival.sum()), unique_coordinates_with_recorded_birth=int(ever.sum()),
        cumulative_birth_events=births, cumulative_retire_events=deaths,
        reactivation_birth_events=births - int((ever & ~initial).sum()),
        event_semantics="eligibility allocation changes" if functional else "numerical support crossings",
        lifetime_semantics="complete eligibility spells; numerical zero crossings do not end a spell" if functional
                           else "complete coordinate lifetimes unavailable from passive count-only traces")


def edge_lifetimes(state):
    """Completed and right-censored eligibility spells; never pool them blindly."""
    if state["config"]["structural_rule"] == "passive":
        return []
    initial = state["operator"]["initial_mask"]
    starts = {tuple(pair): 0 for pair in initial.nonzero().tolist()}
    result = []
    chunk, half = state["config"]["chunk_ticks"], state["config"]["width"] // 2
    def spell(pair, start, end, censored):
        return dict(row=pair[0], column=pair[1], start_update=start, end_update=end,
                    duration_updates=end-start, duration_ticks=(end-start)*chunk,
                    right_censored=censored, initial_coordinate=bool(initial[pair]),
                    cross_bank=(pair[0] < half) != (pair[1] < half))
    for event in state["operator"]["event_history"]:
        end = event["executed_update"]
        for pair in event["actual_birth_coordinates"]:
            pair = tuple(pair)
            if pair in starts:
                raise ValueError("birth duplicates an active eligibility spell")
            starts[pair] = end
        for pair in event["actual_retire_coordinates"]:
            pair = tuple(pair)
            result.append(spell(pair, starts.pop(pair), end, False))
    result.extend(spell(pair, start, state["updates"], True) for pair, start in sorted(starts.items()))
    return result


def audit_checkpoint_payload(state, *, plan=None, branch=None, tick=None, audit=None, where="checkpoint"):
    audit = _Checks() if audit is None else audit
    try:
        with _preserve_process_state():
            trainer = native.NativeTrainer.from_state_dict(state, device="cpu", strict_sources=True)
            restored = trainer.semantic_state()
            restored["config"]["device"] = state["config"]["device"]
            audit.equal("FULL_ORGANIZATION_SEMANTIC", where, state_digest(restored),
                state_digest({key: state[key] for key in native._SEMANTIC_FIELDS}))
            if tick is not None:
                audit.equal("CHECKPOINT_TICK", where, trainer.ticks, tick)
            if branch is not None:
                audit.equal("EXACT_BRANCH_NATIVE_CONFIG", where, normalize(state["config"]), branch["native_config"])
            if plan is not None and branch is not None:
                study.validate(trainer, plan, branch, tick)
            audit.equal("NO_HIDDEN_RESETS", where, state["hidden_resets"], 0)
            audit.equal("PUBLIC_INPUT_WIDTH", where, trainer.model.input_size, 112)
            audit.equal("DECLARED_NONLINEAR_FIELD", where, trainer.model.field_mode, state["config"]["field_mode"])
            event_sequence_hash = _audit_trace(state, trainer, audit, where)
            legacy._audit_reservoir_policy(state, trainer, audit, where)
            blocks = list(trainer.replay.blocks)
            if trainer.replay.partial is not None:
                blocks.append(trainer.replay.partial)
            if state["last_chunk"] is not None:
                blocks.extend((state["last_chunk"], state["last_auxiliary"]["block"]))
                info, block = state["last_auxiliary"]["info"], state["last_auxiliary"]["block"]
                audit.equal("LAST_AUXILIARY_TRACE_BINDING", where, info, state["trace"][-1]["auxiliary"])
                audit.equal("LAST_AUXILIARY_CONTENT", where, block_hash(block), info["source_hash"])
                for key in ("start_tick", "end_tick", "family_counts", "source_id"):
                    audit.equal("LAST_AUXILIARY_METADATA", f"{where}/{key}", info[key], block[key])
                audit.equal("LAST_AUXILIARY_SUFFIX_LABELS", where, info["suffix_family_positions"],
                    {family: int(block["family_context"][info["warmup_ticks"]:].eq(family).sum()) for family in range(4)})
            for index, block in enumerate(blocks):
                legacy._audit_original_block(block, state, audit, f"{where}/history_{index}")
            if trainer.ticks == 0:
                fresh = native.NativeTrainer(replace(trainer.config, device="cpu"))
                expected = fresh.semantic_state()
                for key in ("model", "hidden", "optimizer", "world", "birth", "operator", "replay", "schedule", "rng"):
                    audit.equal("REGENERATED_ORGANIZATION_BIRTH", f"{where}/{key}", state_digest(state[key]), state_digest(expected[key]))
                del fresh
            report = trainer.report()
            return dict(tick=trainer.ticks, updates=trainer.updates, semantic_hash=state_digest(restored),
                cpu_semantic_hash=state_digest(trainer.semantic_state()), birth_hash=state_digest(state["birth"]),
                initial_parameter_hash=trainer.initial_parameter_hash, parameter_hash=state_digest(state["model"]),
                hidden_hash=state_digest({"hidden": state["hidden"]}), world_hash=state_digest(state["world"]),
                scheduler_hash=state_digest(state["world"]["scheduler"]), context_hash=state_digest(state["world"]["context"]),
                event_hash=trainer.world.event_hash, event_sequence_hash=event_sequence_hash,
                source_continuation_hash=state_digest({"model": state["model"], "world": state["world"], "hidden": state["hidden"]}),
                world_tick=trainer.world.tick, world_time=float(trainer.world.now), model_config=trainer.model.config(),
                family_recency=report["family_recency"], world_family_recency=trainer.world.family_recency(),
                actual_family_exposures=trainer.world.family_exposures(), work=deepcopy(state["work"]),
                structure={**report["structure"], **structural_history(state, audit, where)},
                replay=report["replay"], auxiliary_stats=report["auxiliary_stats"], report=report,
                birth_signature=study._birth_signature(trainer) if trainer.ticks == 0 else None,
                seconds=float(state["seconds"]), trace_json_hash=json_hash(state["trace"]),
                total_parameters=trainer.model.parameter_report()["total_parameters"], checks=audit.result())
    except AuditError:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as error:
        raise AuditError("ORGANIZATION_CHECKPOINT_REJECTED", where, str(error), checks=sum(audit.counts.values())) from error


def audit_native_record(record, checkpoint, *, plan, spec, branch, audit=None, where="record"):
    audit = _Checks() if audit is None else audit
    try:
        _fields(record, legacy.RECORD_FIELDS | {"organization"}, audit, where)
        audit.equal("ORGANIZATION_FROZEN_FIELD", where, record["organization"],
            dict(field_mode=branch["field_mode"], structural_rule=branch["structural_rule"], structural_updates_during_fork=0))
        audit.equal("ORGANIZATION_ACTUAL_MODEL_FIELD", where, record["model"]["field_mode"], branch["field_mode"])
        adapted = {key: value for key, value in record.items() if key != "organization"}
        view = dict(plan, batch_size=branch["native_config"]["batch_size"])
        metrics = legacy.audit_native_record(adapted, checkpoint, plan=view, spec=spec, audit=audit, where=where)
        detail = record["request_diagnosis"]["primary"]
        metrics["missed_finish_despite_legal"] = _rate(detail["uncompleted_despite_legal_opportunity"], detail["requests"])
        legal_opportunities = legal_finishes = 0
        for row in record["content_by_cohort_phase_age"]:
            for field in ("model_finish_attempts", "executed_finish_attempts", "first_legal_opportunities"):
                audit.require("FINISH_ACTION_OPPORTUNITY_COUNTS", f"{where}/{field}",
                    _integer(row[field]) and row[field] <= row["opportunities"])
            audit.equal("UNMODIFIED_FINISH_ACTION_COUNTS", where, row["model_finish_attempts"], row["executed_finish_attempts"])
            if row["cohort"] == "primary" and row["phase"] == "legal_finish":
                legal_opportunities += row["opportunities"]
                legal_finishes += row["executed_finish_attempts"]
        metrics["legal_finish_action_rate"] = _rate(legal_finishes, legal_opportunities)
        metrics["legal_finish_omission_rate"] = _rate(legal_opportunities - legal_finishes, legal_opportunities)
        return metrics
    except AuditError:
        raise
    except (ValueError, TypeError, KeyError, RuntimeError, IndexError, AttributeError) as error:
        raise AuditError("MALFORMED_ORGANIZATION_EVALUATION", where, str(error), checks=sum(audit.counts.values())) from error


def _protocol(path, root, audit, inputs):
    path, digest = _input(path, inputs, audit)
    value = _read_json(path)
    audit.require("COMPLETE_ORGANIZATION_PROTOCOL", str(path), isinstance(value.get("branches"), list) and len(value["branches"]) == 27)
    try:
        plan, _, checked_hash = study.read_plan(path, value["branches"][0]["id"])
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as error:
        raise AuditError("ORGANIZATION_PROTOCOL_REJECTED", str(path), str(error)) from error
    audit.equal("PROTOCOL_READ_STABLE", str(path), digest, checked_hash)
    audit.equal("EXACT_ORGANIZATION_COUNTS", str(path), plan["expected_counts"],
        dict(training_runs=27, full_checkpoints=108, forks_per_branch=24, frozen_forks=648))
    if root is not None:
        audit.equal("RAW_ROOT_BINDING", str(path), Path(plan["raw_root"]).resolve(), Path(root).resolve())
    for source, expected in plan["sources"].items():
        _, actual = _input(source, inputs, audit, "EXECUTION_SOURCE")
        audit.equal("FROZEN_EXECUTION_SOURCE", source, actual, expected)
    for kind in ("training", "evaluation", "baseline"):
        manifest, _ = _input(path.parent / f"organization-{kind}-commands-v1.json", inputs, audit)
        saved = _read_json(manifest)
        audit.equal("MANIFEST_CWD", str(manifest), saved["cwd"], str(ROOT))
        audit.equal("MANIFEST_FROZEN_FILES", str(manifest), saved["file_hashes"], {**plan["sources"], str(path): digest})
        expected_ids = set() if kind == "baseline" else {item["id"] for item in plan["branches"]}
        audit.equal("MANIFEST_JOB_INVENTORY", str(manifest), [row["id"] for row in saved["jobs"]],
                    [] if kind == "baseline" else [item["id"] for item in plan["branches"]])
        audit.equal("MANIFEST_UNIQUE_JOBS", str(manifest), len(saved["jobs"]), len(expected_ids))
        for job, item in zip(saved["jobs"], plan["branches"]):
            folder = Path(item["output"])
            tail = ["--protocol", str(path), "--branch-id", item["id"]]
            output = Path(plan["raw_root"]) / "evaluation" / f"{item['id']}.json"
            command_tail = ["-m", "experiments.organization.study", "train" if kind == "training" else "evaluate"] + tail
            if kind == "evaluation":
                command_tail += ["--checkpoint", str(folder / "latest.pt"), "--output", str(output)]
            audit.equal("MANIFEST_EXACT_COMMAND", item["id"], job["command"][1:], command_tail)
            audit.equal("MANIFEST_EXACT_OUTPUTS", item["id"], job["outputs"],
                [str(folder / "latest.pt"), str(folder / "launch-complete.json")] if kind == "training" else [str(output)])
    return plan, digest


def _branch(plan, item, protocol_hash, inputs, audit):
    folder, label = Path(item["output"]).resolve(), item["id"]
    marker_path, _ = _input(folder / "launch-complete.json", inputs, audit)
    marker = _read_json(marker_path)
    _fields(marker, {"format", "study_format", "branch", "protocol_sha256", "target", "stage_ticks", "checkpoint_sha256",
                     "checkpoints", "hidden_resets", "initial_parameter_hash", "birth_hash", "work"}, audit, label)
    for key, expected in (("format", "lifelong-launch-complete-v1"), ("study_format", "organization-launch-complete-v1"),
            ("branch", item), ("protocol_sha256", protocol_hash), ("target", plan["target_ticks"]),
            ("stage_ticks", plan["target_ticks"]), ("hidden_resets", 0)):
        audit.equal("COMPLETION_BINDING", f"{label}/{key}", marker[key], expected)
    start, _ = _input(folder.with_name(folder.name + ".launch-start.json"), inputs, audit)
    audit.equal("START_BINDING", label, _read_json(start),
        dict(format="organization-launch-start-v1", branch=item, protocol_sha256=protocol_hash))
    audit.equal("EXACT_CHECKPOINT_KEYS", label, set(marker["checkpoints"]), {str(tick) for tick in plan["checkpoint_ticks"]})
    expected_paths = {folder / "checkpoints" / f"tick_{tick:06d}.pt" for tick in plan["checkpoint_ticks"]}
    audit.equal("EXACT_CHECKPOINT_FILES", label, {path.resolve() for path in (folder / "checkpoints").glob("*.pt")}, expected_paths)
    checkpoints, previous, capacity, probes, lifetimes = {}, None, [], [], []
    identity = {key: item[key] for key in ("id", "seed", *BRANCH_KEYS)}
    for tick in plan["checkpoint_ticks"]:
        entry = marker["checkpoints"][str(tick)]
        _fields(entry, {"path", "sha256", "semantic_hash", "initial_parameter_hash", "birth_hash"}, audit, f"{label}/{tick}")
        path, digest = _input(folder / "checkpoints" / f"tick_{tick:06d}.pt", inputs, audit)
        audit.equal("CHECKPOINT_PATH_BINDING", label, entry["path"], str(path))
        audit.equal("CHECKPOINT_BYTE_BINDING", label, entry["sha256"], digest)
        state = torch.load(path, map_location="cpu", weights_only=False)
        cp = audit_checkpoint_payload(state, plan=plan, branch=item, tick=tick, audit=audit, where=str(path))
        for key in ("semantic_hash", "initial_parameter_hash", "birth_hash"):
            audit.equal("CHECKPOINT_FULL_BINDING", f"{label}/{tick}/{key}", cp[key], entry[key])
        if previous is not None:
            audit.equal("UNINTERRUPTED_ORGANIZATION_TRACE", label,
                json_hash(state["trace"][:previous["updates"]]), previous["trace_json_hash"])
            audit.equal("PERSISTENT_ORGANIZATION_BIRTH", label, cp["birth_hash"], previous["birth_hash"])
        cp.update(checkpoint=str(path), checkpoint_sha256=digest)
        if tick == plan["target_ticks"]:
            audit.equal("COMPLETE_FINAL_BIRTH", label, marker["birth_hash"], cp["birth_hash"])
            audit.equal("COMPLETE_FINAL_PARAMETER_ORIGIN", label, marker["initial_parameter_hash"], cp["initial_parameter_hash"])
            audit.equal("COMPLETE_FINAL_WORK", label, marker["work"], cp["work"])
            product, _ = _input(folder / "report.json", inputs, audit)
            expected_report = deepcopy(cp["report"])
            expected_report["config"]["device"] = state["config"]["device"]
            expected_report["runtime"] = state["runtime"]
            audit.equal("SAVED_ORGANIZATION_REPORT", label, json_hash(_read_json(product)), json_hash(expected_report))
            trace_path, _ = _input(folder / "trace.jsonl", inputs, audit)
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            audit.require("TRACE_JSONL_COMPLETE_LINES", label, all(line.strip() for line in lines))
            audit.equal("SAVED_ORGANIZATION_TRACE", label, json_hash([json.loads(line) for line in lines]), cp["trace_json_hash"])
            for namespace in ("sources", "dependencies"):
                for name, text in state[namespace].items():
                    source, _ = _input(folder / namespace / name, inputs, audit)
                    audit.equal("SAVED_RUNTIME_SOURCE_TEXT", str(source), source.read_text(encoding="utf-8"), text)
            for row in state["trace"]:
                metrics, structure = row["update_metrics"], row["update_metrics"]["structure"]
                capacity.append(identity | dict(tick=row["tick"], update=row["update"],
                    active=structure["active"], allowed=structure["allowed"], births=structure["births"], deaths=structure["deaths"],
                    cumulative_births=structure["cumulative_births"], cumulative_deaths=structure["cumulative_deaths"],
                    initial_survivors=structure["initial_survivors"], ever_born=structure["ever_born"],
                    probe_calls=structure.get("probe_calls", 0), gradient_norm=metrics["gradient_norm"],
                    actual_commands=row["actual_family_exposures"]["commands"]))
                if structure.get("functional_probe") is not None:
                    probes.append(identity | dict(tick=row["tick"], update=row["update"],
                        **deepcopy(structure["functional_probe"]), sources=deepcopy(structure["probe_sources"])))
            lifetimes = [identity | spell for spell in edge_lifetimes(state)]
        cp.pop("report")
        previous = cp
        checkpoints[tick] = cp
        del state
        gc.collect()
    latest, latest_hash = _input(folder / "latest.pt", inputs, audit)
    audit.equal("LATEST_COMPLETE_BYTES", label, latest_hash, marker["checkpoint_sha256"])
    state = torch.load(latest, map_location="cpu", weights_only=False)
    last = audit_checkpoint_payload(state, plan=plan, branch=item, tick=plan["target_ticks"], audit=audit, where=str(latest))
    audit.equal("LATEST_FINAL_SEMANTIC", label, last["semantic_hash"], checkpoints[plan["target_ticks"]]["semantic_hash"])
    del state, last
    gc.collect()
    return checkpoints, capacity, probes, lifetimes


def _branch_worker(job):
    """Return only pickle-safe summaries; AuditError itself is not transported."""
    plan, item, protocol_hash = job
    inputs, audit = {}, _Checks()
    try:
        torch.set_num_threads(1)
        checkpoints, capacity, probes, lifetimes = _branch(plan, item, protocol_hash, inputs, audit)
        return dict(branch=item, checkpoints=checkpoints, capacity=capacity, probes=probes,
                    lifetimes=lifetimes, inputs=inputs, audit=audit.result())
    except AuditError as error:
        return dict(branch=item, failure=error.audit, inputs=inputs, audit=audit.result())
    except Exception as error:
        return dict(branch=item, failure=dict(status="failed", errors=[dict(
            code="MALFORMED_ORGANIZATION_BRANCH", where=item["id"],
            details=f"{type(error).__name__}: {error}")]), inputs=inputs, audit=audit.result())


def _evaluation(value, item, checkpoints, plan, protocol_hash, audit, where):
    _fields(value, {"format", "branch", "protocol_sha256", "source_unchanged", "states", "records",
                    "birth_hash", "initial_parameter_hash"}, audit, where)
    for key, expected in (("format", "organization-evaluation-v1"), ("branch", item),
            ("protocol_sha256", protocol_hash), ("source_unchanged", True),
            ("birth_hash", checkpoints[0]["birth_hash"]), ("initial_parameter_hash", checkpoints[0]["initial_parameter_hash"])):
        audit.equal("EVALUATION_BINDING", f"{where}/{key}", value[key], expected)
    audit.equal("EVALUATED_CHECKPOINT_SET", where, set(value["states"]), {str(tick) for tick in plan["checkpoint_ticks"] if tick})
    for tick in plan["checkpoint_ticks"][1:]:
        saved, cp = value["states"][str(tick)], checkpoints[tick]
        expected = dict(checkpoint=cp["checkpoint"], checkpoint_sha256=cp["checkpoint_sha256"],
            state_hash=cp["cpu_semantic_hash"], training_device_normalized_state_hash=cp["semantic_hash"],
            family_recency=normalize(cp["family_recency"]), field_mode=item["field_mode"], structural_rule=item["structural_rule"])
        audit.equal("EVALUATION_FULL_CHECKPOINT_BINDING", f"{where}/{tick}", saved, expected)
    audit.equal("EXACT_BRANCH_FROZEN_FORK_COUNT", where, len(value["records"]), 24)
    rows, idle, content, streams = [], [], [], []
    for record, spec in zip(value["records"], plan["evaluation_specs"]):
        cp = checkpoints[spec["tick"]]
        label = f"{where}/{spec}"
        metrics = audit_native_record(record, cp, plan=plan, spec=spec, branch=item, audit=audit, where=label)
        identity = {key: item[key] for key in ("id", "seed", *BRANCH_KEYS)}
        meta = identity | spec
        rows.append(meta | dict(metrics=metrics, primary=record["primary"],
            primary_censored=record["censored"]["primary"]["count"],
            diagnosis=record["request_diagnosis"]["primary"], opportunities=record["opportunities"],
            actual_family_exposures=record["actual_family_exposures"], source_clock=legacy.family_clock_metrics(cp, spec["family"])))
        idle.extend(meta | row for row in record["inactive_history"])
        content.extend(meta | row for row in record["content_by_cohort_phase_age"])
        streams.append(((item["seed"], *[spec[key] for key in _METADATA]),
            {key: record[key] for key in ("start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash")}))
    return rows, idle, content, streams


def summarize_seed_values(rows):
    if {row["seed"] for row in rows} != set(SEEDS) or len(rows) != 3:
        raise ValueError("every statistical group must contain exactly trained seeds 0, 1 and 2")
    return legacy.summarize_seed_values(rows)


def summaries(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    return [dict(zip(fields, key)) | dict(metrics={metric: summarize_seed_values([
        dict(seed=row["seed"], value=row["metrics"][metric]) for row in selected])
        for metric in selected[0]["metrics"]}) for key, selected in sorted(groups.items())]


def family_averages(rows):
    fields = ("seed", *BRANCH_KEYS, "tick", "condition", "steps")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        if len(selected) != 4 or {row["family"] for row in selected} != set(range(4)):
            raise ValueError("family averages require exactly four families within each trained seed")
        metrics = {name: statistics.mean([row["metrics"][name] for row in selected])
            if all(row["metrics"][name] is not None for row in selected) else None for name in selected[0]["metrics"]}
        result.append(dict(zip(fields, key)) | dict(n_families=4, metrics=metrics))
    return result


def paired_contrasts(rows, *, family=True):
    fields = ("tick", *(("family",) if family else ()), "condition", "steps")
    groups = defaultdict(dict)
    for row in rows:
        key = (row["seed"], *[row[name] for name in fields])
        variant = row["field_mode"], row["structural_rule"]
        if variant in groups[key]:
            raise ValueError("duplicate variant in a paired seed group")
        groups[key][variant] = row
    results = []
    expected_variants = {tuple(item[key] for key in BRANCH_KEYS) for item in study.VARIANTS}
    for key, selected in sorted(groups.items()):
        if set(selected) != expected_variants:
            raise ValueError("paired contrasts require every predeclared organization variant")
        base = dict(seed=key[0], **dict(zip(fields, key[1:])))
        metric_names = next(iter(selected.values()))["metrics"]
        for name, left, right in CONTRASTS:
            results.append(base | dict(contrast=name, metrics={metric: _difference(
                selected[left]["metrics"][metric], selected[right]["metrics"][metric]) for metric in metric_names}))
        a, b, c, d = [selected[variant] for variant in (("coupled", "functional_task"),
            ("coupled", "functional_fixed"), ("plain", "functional_task"), ("plain", "functional_fixed"))]
        results.append(base | dict(contrast="task_fixed_coupled_minus_plain_difference_in_differences",
            metrics={metric: _difference(_difference(a["metrics"][metric], b["metrics"][metric]),
                                         _difference(c["metrics"][metric], d["metrics"][metric])) for metric in metric_names}))
    return results


def temporal_contrasts(rows, ticks):
    main, withdrawn, final = ticks[1:]
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["seed"], row["field_mode"], row["structural_rule"], row["family"], row["condition"], row["steps"])][row["tick"]] = row
    results = []
    for key, selected in sorted(groups.items()):
        if set(selected) != {main, withdrawn, final}:
            raise ValueError("temporal contrasts require all three matched-horizon stages")
        base = dict(zip(("seed", *BRANCH_KEYS, "family", "condition", "steps"), key))
        for name, before, after in (("withdrawal_change", main, withdrawn), ("return_change", withdrawn, final)):
            results.append(base | dict(contrast=name, before_tick=before, after_tick=after,
                metrics={metric: _difference(selected[after]["metrics"][metric], selected[before]["metrics"][metric])
                         for metric in selected[before]["metrics"]}))
    return results


def contrast_summaries(rows):
    if not rows:
        return []
    fields = tuple(key for key in rows[0] if key not in ("seed", "metrics"))
    return summaries(rows, fields)


def _probe_summary(probes):
    result = dict(trials=len(probes), accepted_birth_batches=sum(bool(row["birth_accepted"]) for row in probes),
        accepted_exit_batches=sum(bool(row["exit_accepted"]) for row in probes),
        actual_births=sum(row["births"] for row in probes), actual_retirements=sum(row["deaths"] for row in probes))
    for kind, damage in (("birth", "birth_damage"), ("exit", "half_exit_damage")):
        accepted = [row for row in probes if row[f"{kind}_accepted"]]
        for loss in ("live", "history", "mixed"):
            values = [row[damage][loss] for row in accepted]
            result[f"accepted_{kind}_{loss}_worsening_batches"] = sum(value > 0 for value in values)
            result[f"accepted_{kind}_{loss}_mean_damage"] = statistics.mean(values) if values else None
            result[f"accepted_{kind}_{loss}_max_damage"] = max(values) if values else None
    return result


def build_report(protocol, root=None, *, workers=6):
    """Audit independent branches in plan order; all 27 remain mandatory."""
    if not _integer(workers, 1) or workers > 6:
        raise ValueError("organization checkpoint workers must be between one and six")
    audit, inputs = _Checks(), {}
    analysis_files = {Path(__file__).resolve(), Path(__file__).with_name("test_report.py").resolve(),
        Path(legacy.__file__).resolve(), Path(legacy._legacy_audit_record.__code__.co_filename).resolve(),
        Path(_Checks.result.__code__.co_filename).resolve()}
    analysis_sources = {str(path): file_sha256(path) for path in analysis_files}
    plan, protocol_hash = _protocol(protocol, root, audit, inputs)
    raw = Path(plan["raw_root"]).resolve()
    evaluation_paths = {raw / "evaluation" / f"{item['id']}.json" for item in plan["branches"]}
    audit.equal("EXACT_EVALUATION_FILE_SET", str(raw), {path.resolve() for path in (raw / "evaluation").glob("*.json")}, evaluation_paths)
    training_root = raw / "training"
    seed_folders = {path.resolve() for path in training_root.iterdir() if path.is_dir()}
    audit.equal("EXACT_TRAINING_SEED_FOLDERS", str(raw), seed_folders,
                {training_root / f"seed_{seed:03d}" for seed in SEEDS})
    audit.equal("EXACT_TRAINING_BRANCH_FOLDERS", str(raw),
        {path.resolve() for folder in seed_folders for path in folder.iterdir() if path.is_dir()},
        {Path(item["output"]).resolve() for item in plan["branches"]})
    checkpoint_rows, capacity, probes, lifetimes, rows, idle, content, training = [], [], [], [], [], [], [], []
    training_streams, evaluation_streams = {}, {}
    jobs = [(plan, item, protocol_hash) for item in plan["branches"]]
    pool = None
    try:
        if workers == 1:
            results = map(_branch_worker, jobs)
        else:
            # Spawn avoids inheriting CUDA/process RNG state from a caller.
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            results = pool.map(_branch_worker, jobs, chunksize=1)
        for item, result in zip(plan["branches"], results, strict=True):
            audit.counts.update(result["audit"]["checks_by_code"])
            audit.equal("WORKER_EXACT_BRANCH_BINDING", item["id"], result["branch"], item)
            if "failure" in result:
                error = result["failure"]["errors"][0]
                raise AuditError(error["code"], error["where"], error.get("details"), checks=sum(audit.counts.values()))
            for path, digest in result["inputs"].items():
                if path in inputs:
                    audit.equal("WORKER_SHARED_INPUT_UNCHANGED", path, digest, inputs[path])
                inputs[path] = digest
            cps = result["checkpoints"]
            branch_capacity, branch_probes, branch_lifetimes = result["capacity"], result["probes"], result["lifetimes"]
            identity = {key: item[key] for key in ("id", "seed", *BRANCH_KEYS)}
            seed = str(item["seed"])
            audit.equal("ACTUAL_MATCHED_SEED_BIRTH", item["id"], cps[0]["birth_signature"], plan["birth_bindings"]["shared_by_seed"][seed])
            stream = [(tick, cp["event_sequence_hash"], cp["scheduler_hash"], cp["event_hash"]) for tick, cp in cps.items()]
            if seed in training_streams:
                audit.equal("MATCHED_ALL_VARIANT_EXOGENOUS_STREAM", item["id"], stream, training_streams[seed])
            training_streams[seed] = stream
            checkpoint_rows.extend(identity | {key: value for key, value in cp.items() if key != "checks"} for cp in cps.values())
            capacity.extend(branch_capacity)
            probes.extend(branch_probes)
            lifetimes.extend(branch_lifetimes)
            final = cps[plan["target_ticks"]]
            work = final["work"]
            base_forward = work["live_forward_ticks"] + work["aux_forward_ticks"]
            training.append(identity | dict(tick=final["tick"], updates=final["updates"], seconds=final["seconds"],
                work=work, structural_forward_fraction_of_base=work["structural_probe_forward_ticks"] / base_forward,
                actual_family_exposures=final["actual_family_exposures"], structure=final["structure"],
                probe_summary=_probe_summary(branch_probes), total_parameters=final["total_parameters"]))
            path, _ = _input(raw / "evaluation" / f"{item['id']}.json", inputs, audit)
            branch_rows, branch_idle, branch_content, streams = _evaluation(_read_json(path), item, cps, plan, protocol_hash, audit, str(path))
            rows.extend(branch_rows)
            idle.extend(branch_idle)
            content.extend(branch_content)
            for key, stream in streams:
                if key in evaluation_streams:
                    audit.equal("MATCHED_FROZEN_EXOGENOUS_STREAM", str(key), stream, evaluation_streams[key])
                evaluation_streams[key] = stream
    except AuditError:
        raise
    except Exception as error:
        raise AuditError("ORGANIZATION_BRANCH_WORKER_REJECTED", str(protocol),
                         f"{type(error).__name__}: {error}", checks=sum(audit.counts.values())) from error
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    audit.equal("ALL_ORGANIZATION_TRAINED_RUNS", str(raw), len(training), 27)
    audit.equal("ALL_ORGANIZATION_CHECKPOINTS", str(raw), len(checkpoint_rows), 108)
    audit.equal("ALL_ORGANIZATION_FROZEN_FORKS", str(raw), len(rows), 648)
    for path, expected in inputs.items():
        audit.equal("FINAL_RAW_INPUT_UNCHANGED", path, file_sha256(path), expected)
    for path, expected in analysis_sources.items():
        audit.equal("ANALYSIS_SOURCE_UNCHANGED", path, file_sha256(path), expected)
    family = family_averages(rows)
    pairs, family_pairs = paired_contrasts(rows), paired_contrasts(family, family=False)
    temporal = temporal_contrasts(rows, plan["checkpoint_ticks"])
    summary = summaries(rows, (*BRANCH_KEYS, *_METADATA))
    family_summary = summaries(family, (*BRANCH_KEYS, "tick", "condition", "steps"))
    counts = dict(training_runs=27, trained_seeds=3, full_checkpoints=108, evaluation_files=27, frozen_forks=648,
        functional_trial_batches=len(probes), eligibility_spells=len(lifetimes),
        structural_probe_calls=sum(row["work"]["structural_probe_calls"] for row in training),
        structural_probe_forward_ticks=sum(row["work"]["structural_probe_forward_ticks"] for row in training))
    header = dict(format=FORMAT_VERSION, complete=True, protocol_sha256=protocol_hash, counts=counts,
                  audit=audit.result(), main_summary=summary, family_average_summary=family_summary,
                  training_per_seed=training,
                  paired_contrasts=contrast_summaries(pairs), family_average_paired_contrasts=contrast_summaries(family_pairs))
    header = _compact_header(header)
    return dict(format=FORMAT_VERSION, complete=True, protocol=str(Path(protocol).resolve()),
        protocol_sha256=protocol_hash, protocol_plan=plan, counts=counts, audit=audit.result(), inputs=inputs,
        analysis_sources=analysis_sources, checkpoint_workers=workers, header=header,
        per_seed=rows, main_summary=summary, family_average_per_seed=family, family_average_summary=family_summary,
        paired_contrasts_per_seed=pairs, paired_contrasts=contrast_summaries(pairs),
        family_average_paired_contrasts_per_seed=family_pairs, family_average_paired_contrasts=contrast_summaries(family_pairs),
        temporal_contrasts_per_seed=temporal, temporal_contrasts=contrast_summaries(temporal),
        training_per_seed=training, checkpoints=checkpoint_rows, capacity_trajectory=capacity,
        functional_probes=probes, edge_lifetimes=lifetimes, idle_by_history_per_seed=idle,
        content_by_phase_age_per_seed=content,
        definitions=dict(statistical_unit="Exactly three independently trained seeds; families are averaged within seed.",
            primary="Assess main-end acquisition before interpreting withdrawal and return; family 0 is the withheld family.",
            finish_omission="legal_finish_omission_rate uses primary legal-finish action opportunities. missed_finish_despite_legal uses all matured primary requests and includes failed wrong FINISH attempts; it does not mean never choosing FINISH.",
            optimization="Functional policies share full-coordinate Adam moments and clipping. Passive uses its original eligibility convention; passive-vs-functional is not a pure structural-placement intervention.",
            acceptance="Only mixed loss strictly decreases. Live or history can individually worsen; accepted component damages are reported.",
            probes="Next actual training block and possibly reused replay; not independent held-out evaluation. Four extra loss calls per executed pending trial.",
            capacity="No hard total capacity or one-in/one-out constraint. Accepted capacity trajectories may diverge between functional policies.",
            topology="Two nonlinear banks and cross-bank weighted coordinates; graph metrics do not establish functional aggregation or modularity.",
            lifetimes="Functional eligibility spell durations include right-censored survivors separately. Passive numerical-support lifetimes cannot be reconstructed from count-only logs.",
            audit="Strict source-bound restoration and original-data hash chains; audits do not retrain every historical gradient or prove generalization."),
        limitations=plan["limits"])


def _compact_header(value):
    if isinstance(value, dict):
        return {key: _compact_header(item) for key, item in value.items() if key != "per_seed"}
    if isinstance(value, list):
        return [_compact_header(item) for item in value]
    return value


def _failure_json(value):
    """Audit details may contain sets of paths from exact-inventory checks."""
    if isinstance(value, dict):
        return {str(key): _failure_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_json(item) for item in value]
    if isinstance(value, set):
        return [_failure_json(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


def write_artifacts(report, output):
    tables = ("per_seed", "main_summary", "family_average_per_seed", "family_average_summary",
              "paired_contrasts_per_seed", "paired_contrasts", "family_average_paired_contrasts_per_seed",
              "family_average_paired_contrasts", "temporal_contrasts_per_seed", "temporal_contrasts",
              "training_per_seed", "checkpoints", "capacity_trajectory", "functional_probes", "edge_lifetimes",
              "idle_by_history_per_seed", "content_by_phase_age_per_seed")
    counts = report.get("counts", {})
    exact_counts = dict(training_runs=27, trained_seeds=3, full_checkpoints=108, evaluation_files=27, frozen_forks=648)
    if (report.get("complete") is not True or report.get("audit", {}).get("status") != "passed"
            or report.get("audit", {}).get("errors") != []
            or any(counts.get(key) != value for key, value in exact_counts.items())
            or any(not isinstance(report.get(name), list) for name in tables)
            or len(report.get("per_seed", [])) != 648 or len(report.get("checkpoints", [])) != 108
            or len(report.get("training_per_seed", [])) != 27
            or report.get("header", {}).get("complete") is not True
            or report.get("header", {}).get("counts") != counts):
        raise ValueError("only the complete passed 27-run audit may produce artifacts")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    legacy._atomic_json(output / "report.json", report)
    legacy._atomic_json(output / "header.json", report["header"])
    files = {name: file_sha256(output / name) for name in ("report.json", "header.json")}
    for name in tables:
        rows = []
        for row in report[name]:
            base = {key: value for key, value in row.items() if key != "metrics"}
            metrics = row.get("metrics", {})
            if metrics and isinstance(next(iter(metrics.values())), dict):
                rows.extend(base | dict(metric=metric, **value) for metric, value in metrics.items())
            else:
                rows.append(base | metrics)
        legacy._csv(output / f"{name}.csv", rows)
        files[f"{name}.csv"] = file_sha256(output / f"{name}.csv")
    legacy._atomic_json(output / "artifacts.json", files)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        result = build_report(args.protocol, args.root, workers=args.workers)
        write_artifacts(result, args.output)
        print(json.dumps(dict(complete=True, counts=result["counts"], audit=result["audit"]), ensure_ascii=False))
    except AuditError as error:
        legacy._atomic_json(args.output / "audit-failed.json", _failure_json(error.audit))
        raise


if __name__ == "__main__":
    main()
