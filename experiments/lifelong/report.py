"""Strict complete-state audit of the finite native lifelong study.

Seeds, rather than families, requests, ticks, or evaluation forks, are the
statistical replicates. Checkpoint workers inspect one branch at a time and
release each full checkpoint before loading its successor. Old experiments and
their auditors are imported read-only; native record adaptation is explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest, seed_for
from experiments.integration.report import _audit_record as _legacy_audit_record, _metrics
from experiments.loops.report import AuditError, _Checks, _hash, _integer, _finite, _read_json, file_sha256
from experiments.lifelong import evaluate as native_evaluate
from experiments.lifelong import run as native_run
from experiments.lifelong.replay import HistoryReservoir, block_hash, validate_block
from experiments.lifelong.schedule import CHECKPOINT_TICKS
from experiments.lifelong.world import FAMILY_MAPPINGS, INPUT_SIZE


ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = "lifelong-study-summary-v1"
SEEDS = (0, 1, 2)
BRANCH_KEYS = ("architecture", "schedule", "learning", "coverage")
PROTOCOL_FIELDS = set("""format created_utc sources branches variants seeds target_ticks main_ticks
    withdrawal_ticks return_ticks chunk_ticks batch_size width initial_fan_in dt training_device
    evaluation_device checkpoint_ticks evaluation_specs replay_capacity replay_block_ticks replay_warmup_ticks
    attachment family_mappings timing_coverage primary learning native limits""".split())
RECORD_FIELDS = set("""family family_mapping source_family_exposures source_family_recency end_family_recency
    actual_family_exposures family_cohorts source_context_hash end_context_hash context_policy source_hashes
    format condition intervention action_policy world_resets hidden_resets parameter_updates steps dt batch_size
    n_jobs active_jobs actor_ids case_ids fork_tick fork_time end_tick end_time evaluation_regime_id regime
    age_edges phase_codes primary_eligible_issued carryover_contract_eligible_issued initial_eligible_pending
    request_diagnosis censored content_by_cohort_phase_age opportunities inactive_actions_per_idle_opportunity
    inactive_actions_per_1000_idle_opportunities inactive_actions_per_1000_total_opportunities inactive_history
    observed_terminal_events terminal_request_first_error_patterns post_fork_counters post_fork_finish_precision
    start_world_state_hash end_world_state_hash start_scheduler_hash end_scheduler_hash start_hidden_hash
    end_hidden_hash start_event_hash end_event_hash first_observation_hash first_model_observation_hash
    raw_observation_hash observation_hash action_hash model_action_hash parameter_hash source_continuation_hash
    model definitions limitations primary inherited carryover_contract probe""".split())
EVALUATION_SOURCE_FILES = (
    "experiments/lifelong/evaluate.py", "experiments/lifelong/world.py",
    "experiments/integration/diagnose.py", "experiments/continuity/world.py", "experiments/continuity/evaluate.py",
)
MAIN_NATIVE_SOURCE_FILES = (
    "__init__.py", "evaluate.py", "queue.py", "replay.py", "run.py", "schedule.py", "study.py", "world.py",
    "test_queue.py", "test_replay.py", "test_run.py", "test_schedule.py", "test_study.py", "test_world.py",
)
SUPPLEMENT_FIELDS = set("""format created_utc parent_protocol parent_protocol_sha256 sources evaluation_specs
    raw_root training_changed reason interpretation""".split())


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def json_hash(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _difference(left, right):
    return None if left is None or right is None else left - right


def _same(left, right):
    return left is None and right is None or (_finite(left) and _finite(right)
        and math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-9))


def _input(path, inputs, audit, code="INPUT_FILE"):
    path = Path(path).resolve()
    audit.require(code, str(path), path.is_file())
    digest = file_sha256(path)
    key = str(path)
    if key in inputs:
        audit.equal("INPUT_UNCHANGED", key, digest, inputs[key])
    inputs[key] = digest
    return path, digest


def _fields(value, expected, audit, where):
    audit.require("EXACT_SCHEMA", where, isinstance(value, dict) and set(value) == set(expected),
                  {"expected": sorted(expected), "actual": sorted(value) if isinstance(value, dict) else type(value).__name__})


def _exposures(value, audit, where):
    _fields(value, {"new", "replace", "commands"}, audit, where)
    audit.require("FAMILY_EXPOSURES", where, all(isinstance(row, list) and len(row) == 4
        and all(_integer(number) for number in row) for row in value.values()))
    audit.equal("COMMAND_PARTITION", where, value["commands"],
                [a + b for a, b in zip(value["new"], value["replace"])])


def expected_work(ticks, chunk, batch, jobs=4):
    updates = ticks // chunk
    auxiliary = 2 * ticks - chunk if updates else 0
    warmup = ticks - chunk if updates else 0
    return dict(live_forward_ticks=ticks, aux_forward_ticks=auxiliary, aux_warmup_ticks=warmup,
        live_loss_ticks=ticks, aux_loss_ticks=ticks, live_forward_actor_ticks=ticks * batch,
        aux_forward_actor_ticks=auxiliary * batch, aux_warmup_actor_ticks=warmup * batch,
        live_loss_actor_ticks=ticks * batch, aux_loss_actor_ticks=ticks * batch,
        live_logit_positions=ticks * batch * jobs, aux_logit_positions=auxiliary * batch * jobs,
        live_loss_positions=ticks * batch * jobs, aux_loss_positions=ticks * batch * jobs,
        backward_calls=2 * updates, bp_graph_ticks=2 * ticks, bp_graph_actor_ticks=2 * ticks * batch,
        optimizer_steps=updates, real_ticks_admitted=ticks)


def _audit_trace(state, trainer, audit, where):
    config, trace = state["config"], state["trace"]
    chunk, batch, jobs, updates = config["chunk_ticks"], config["batch_size"], config["n_jobs"], state["updates"]
    audit.equal("TRACE_LENGTH", where, len(trace), updates)
    action = data = hashlib.sha256(f"{native_run.FORMAT_VERSION}/{state['birth']['data_seed']}".encode()).hexdigest()
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
    audit.equal("EXACT_COMPUTE_WORK", where, state["work"], expected_work(state["ticks"], chunk, batch, jobs))
    return json_hash([row["event_hash"] for row in trace])


def _audit_original_block(block, state, audit, where):
    """Link retained labels/actions back to the original immutable live hashes."""
    validate_block(block)
    config, trace = state["config"], state["trace"]
    chunk = config["chunk_ticks"]
    length, start = block["end_tick"] - block["start_tick"], block["start_tick"]
    batch, jobs, width = config["batch_size"], config["n_jobs"], config["width"]
    dtype = state["model"]["encoder.weight"].dtype
    for key, shape, expected_dtype in (("observations", (length, batch, INPUT_SIZE), dtype),
            ("targets", (length, batch, jobs), torch.long), ("actions", (length, batch, jobs), torch.long),
            ("family_context", (length, batch, jobs), torch.long), ("start_h", (batch, width), dtype)):
        audit.require("HISTORY_NATIVE_MODEL_SCHEMA", f"{where}/{key}", key in block
            and isinstance(block[key], torch.Tensor) and tuple(block[key].shape) == shape
            and block[key].dtype == expected_dtype)
    for key in ("targets", "actions"):
        audit.require("HISTORY_NATIVE_ACTION_CODES", f"{where}/{key}", bool(block[key].ge(0).all())
            and bool(block[key].lt(11).all()))
    audit.require("HISTORY_CAUSAL_ENDPOINT", where, block["end_tick"] <= state["ticks"]
        and start % chunk == 0 and length % chunk == 0)
    expected_h = (state_digest(torch.zeros(batch, width, dtype=dtype)) if start == 0
                  else trace[start // chunk - 1]["live_hidden_hash"])
    audit.equal("HISTORICAL_ACTUAL_H", where, state_digest(block["start_h"]), expected_h)
    for offset in range(0, length, chunk):
        begin, end = start + offset, start + offset + chunk
        original = trace[end // chunk - 1]
        causal_data = {key: block[key][offset:offset + chunk] for key in ("observations", "targets")}
        causal_data.update(elapsed=tuple(block["elapsed"][offset:offset + chunk]), start_tick=begin, end_tick=end)
        causal_data["family_context"] = block["family_context"][offset:offset + chunk]
        audit.equal("HISTORY_ORIGINAL_LABELLED_DATA", f"{where}/{begin}", state_digest(causal_data), original["data_chunk_hash"])
        audit.equal("HISTORY_ORIGINAL_ACTIONS", f"{where}/{begin}",
                    state_digest(block["actions"][offset:offset + chunk]), original["action_chunk_hash"])
    before = [0] * 4 if start == 0 else trace[start // chunk - 1]["actual_family_exposures"]["commands"]
    after = trace[block["end_tick"] // chunk - 1]["actual_family_exposures"]["commands"]
    audit.equal("HISTORY_ORIGINAL_FAMILY_COMMANDS", where, block["family_counts"],
                {family: after[family] - before[family] for family in range(4)})


def _audit_reservoir_policy(state, trainer, audit, where):
    """Replay the cheap index/RNG policy, never the model or historical world."""
    config = state["config"]
    size, length = config["replay_capacity"], 2 * config["chunk_ticks"]
    expected = HistoryReservoir(size, length, seed_for(config["seed"], "lifelong-history"))
    retained, seen = [], 0
    for index, point in enumerate(state["trace"], 1):
        if config["learning"] == "history_replay" and index >= 3:
            selected = retained[expected.sample_rng.randrange(len(retained))]
            source = point["auxiliary"]
            audit.equal("UNIFORM_PRIVATE_HISTORY_SAMPLE", f"{where}/{index}",
                        (source["start_tick"], source["end_tick"]), (selected * length, (selected + 1) * length))
        if index % 2 == 0:
            new_index, seen = seen, seen + 1
            if len(retained) < size:
                retained.append(new_index)
            else:
                slot = expected.admission_rng.randrange(seen)
                if slot < size:
                    retained[slot] = new_index
    audit.equal("UNIFORM_PRIVATE_HISTORY_ADMISSION", where,
                [[block["start_tick"], block["end_tick"]] for block in trainer.replay.blocks],
                [[index * length, (index + 1) * length] for index in retained])
    audit.equal("PRIVATE_ADMISSION_RNG_EXACT", where, state_digest(state["replay"]["admission_rng_state"]),
                state_digest(expected.admission_rng.getstate()))
    audit.equal("PRIVATE_SAMPLING_RNG_EXACT", where, state_digest(state["replay"]["sample_rng_state"]),
                state_digest(expected.sample_rng.getstate()))


def structural_history(state, audit=None, where="structure"):
    """Separate first activation of new coordinates from returns/repeated births."""
    audit = _Checks() if audit is None else audit
    operator = state["operator"]
    initial = operator["inherited_operator"]["mask"].bool()
    support = operator["operator"]["mask"].bool()
    ever = operator["factor_ever_born"].bool()
    survival = operator["factor_survival_mask"].bool()
    births = operator["operator"]["cumulative_births"]
    audit.require("STRUCTURE_NUMERICAL_SUPPORT", where, torch.equal(support, state["model"]["recurrent"].ne(0))
        and torch.equal(initial, operator["operator"]["initial_mask"].bool()))
    audit.equal("STRUCTURE_AUTHENTICATED_BIRTH_MASK", where, state_digest(initial), state["birth"]["initial_mask_hash"])
    outside_unique, inside_unique = int((ever & ~initial).sum()), int((ever & initial).sum())
    unique = outside_unique + inside_unique
    audit.require("STRUCTURE_UNIQUE_BIRTH_ACCOUNTING", where, births >= unique
        and not bool((support & ~initial & ~ever).any())
        and not bool((ever & initial & survival).any()))
    return dict(initial_support_coordinates=int(initial.sum()),
        active_edges_outside_initial=int((support & ~initial).sum()),
        active_initial_coordinates=int((support & initial).sum()),
        active_initial_coordinates_reactivated=int((support & initial & ever).sum()),
        unique_edges_ever_born_outside_initial=outside_unique,
        initial_edges_ever_reactivated=inside_unique,
        unique_coordinates_with_recorded_birth=unique,
        cumulative_birth_events=births,
        reactivation_birth_events=births - outside_unique,
        birth_events_minus_unique_coordinates=births - unique,
        birth_note="Birth counters exclude original initialization. Reactivation events include every initial-coordinate return and every later return of a new coordinate; births-minus-all-unique counts only second/later recorded births.")


def audit_checkpoint_payload(state, *, plan=None, branch=None, tick=None, audit=None, where="checkpoint"):
    """Audit one real native checkpoint; small fixtures may use smaller budgets.

    Full formal-matrix validation belongs to build_report, never to a weakened
    experimental protocol. This helper is also used by independent workers.
    """
    audit = _Checks() if audit is None else audit
    try:
        with _preserve_process_state():
            trainer = native_run.NativeTrainer.from_state_dict(state, device="cpu", strict_sources=True)
            raw_semantic = {key: state[key] for key in native_run._SEMANTIC_FIELDS}
            restored = trainer.semantic_state()
            restored["config"]["device"] = state["config"]["device"]
            audit.equal("FULL_NATIVE_SEMANTIC", where, state_digest(restored), state_digest(raw_semantic))
            if tick is not None:
                audit.equal("CHECKPOINT_TICK", where, trainer.ticks, tick)
            if branch is not None:
                for key in ("seed", *BRANCH_KEYS):
                    audit.equal("NATIVE_BRANCH_BINDING", f"{where}/{key}", state["config"][key], branch[key])
            if plan is not None:
                for key in ("width", "batch_size", "chunk_ticks", "initial_fan_in", "main_ticks", "withdrawal_ticks", "return_ticks", "dt", "replay_capacity"):
                    audit.equal("NATIVE_PROTOCOL_BUDGET", f"{where}/{key}", state["config"][key], plan[key])
                audit.equal("TRAINING_DEVICE", where, state["config"]["device"], plan["training_device"])
                if branch is not None:
                    values = {key: plan[key] for key in native_run.NativeConfig.__dataclass_fields__ if key in plan}
                    values.update({key: branch[key] for key in ("seed", *BRANCH_KEYS)})
                    values["device"] = plan["training_device"]
                    expected_config = native_run.NativeConfig(**values)
                    audit.equal("FULL_FROZEN_NATIVE_CONFIG", where, state["config"], asdict(expected_config))
            audit.equal("NATIVE_BIRTH_SOURCE", where, state["birth"]["source"], "from_scratch")
            audit.equal("NO_HIDDEN_RESETS", where, state["hidden_resets"], 0)
            audit.equal("MODEL_PUBLIC_INPUT", where, trainer.model.input_size, INPUT_SIZE)
            audit.equal("NO_IDLE_AUXILIARY_OBJECTIVE", where, trainer.operator.config.idle_lambda, 0.)
            audit.equal("WAIT_WEIGHT", where, state["config"]["wait_weight"], .1)
            event_sequence_hash = _audit_trace(state, trainer, audit, where)
            _audit_reservoir_policy(state, trainer, audit, where)
            for index, block in enumerate(trainer.replay.blocks):
                _audit_original_block(block, state, audit, f"{where}/retained_{index}")
            if trainer.replay.partial is not None:
                _audit_original_block(trainer.replay.partial, state, audit, f"{where}/partial")
            if state["last_chunk"] is not None:
                _audit_original_block(state["last_chunk"], state, audit, f"{where}/last_live")
                _audit_original_block(state["last_auxiliary"]["block"], state, audit, f"{where}/last_auxiliary")
                audit.equal("AUXILIARY_ORIGINAL_SOURCE_HASH", where, block_hash(state["last_auxiliary"]["block"]),
                            state["last_auxiliary"]["info"]["source_hash"])
                info, block = state["last_auxiliary"]["info"], state["last_auxiliary"]["block"]
                audit.equal("LAST_AUXILIARY_TRACE_BINDING", where, info, state["trace"][-1]["auxiliary"])
                for key in ("start_tick", "end_tick", "family_counts", "source_id"):
                    audit.equal("LAST_AUXILIARY_SOURCE_METADATA", f"{where}/{key}", info[key], block[key])
                audit.equal("LAST_AUXILIARY_SUFFIX_CONTEXT", where, info["suffix_family_positions"],
                    {family: int(block["family_context"][info["warmup_ticks"]:].eq(family).sum()) for family in range(4)})
            if trainer.ticks == 0:
                audit.equal("BIRTH_PARAMETER_IDENTITY", where, state_digest(state["model"]), state["birth"]["initial_parameter_hash"])
                audit.require("BIRTH_ONLY_ZERO_H", where, bool(state["hidden"].eq(0).all()))
                audit.require("BIRTH_EMPTY_ADAM_HISTORY", where, not state["optimizer"]["state"] and state["replay"]["ticks_seen"] == 0)
                audit.require("BIRTH_SPARSE_FANIN", where, bool(trainer.model.recurrent.ne(0).sum(-1).eq(trainer.config.effective_fan_in).all()))
                with _preserve_process_state():
                    regenerated = native_run.NativeTrainer(replace(trainer.config, device="cpu"))
                    expected_birth = regenerated.semantic_state()
                    for key in ("model", "hidden", "optimizer", "world", "birth", "operator", "replay", "schedule"):
                        audit.equal("REGENERATED_RANDOM_BIRTH", f"{where}/{key}", state_digest(state[key]), state_digest(expected_birth[key]))
                    del regenerated
            report = trainer.report()
            cpu_semantic_hash = state_digest(trainer.semantic_state())
            return dict(tick=trainer.ticks, updates=trainer.updates, semantic_hash=state_digest(restored),
                cpu_semantic_hash=cpu_semantic_hash, birth_hash=state_digest(state["birth"]),
                initial_parameter_hash=state["birth"]["initial_parameter_hash"],
                parameter_hash=state_digest(state["model"]), hidden_hash=state_digest({"hidden": state["hidden"]}),
                world_hash=state_digest(state["world"]), scheduler_hash=state_digest(state["world"]["scheduler"]),
                context_hash=state_digest(state["world"]["context"]), event_hash=trainer.world.event_hash,
                source_continuation_hash=state_digest({"model": state["model"], "world": state["world"], "hidden": state["hidden"]}),
                event_sequence_hash=event_sequence_hash, world_tick=trainer.world.tick, world_time=float(trainer.world.now),
                model_config=trainer.model.config(), family_recency=report["family_recency"],
                world_family_recency=trainer.world.family_recency(), actual_family_exposures=trainer.world.family_exposures(),
                work=state["work"], structure={**report["structure"], **structural_history(state, audit, where)}, replay=report["replay"],
                auxiliary_stats=report["auxiliary_stats"], report=report,
                seconds=float(state["seconds"]), total_parameters=trainer.model.parameter_report()["total_parameters"],
                trace_json_hash=json_hash(state["trace"]), checks=audit.result())
    except AuditError:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as error:
        raise AuditError("NATIVE_CHECKPOINT_REJECTED", where, str(error), checks=sum(audit.counts.values())) from error


def _audit_native_record(record, checkpoint, *, plan, spec, audit, where):
    """Adapt only the schema identity/condition lookup for the frozen old auditor."""
    record = normalize(record)
    _fields(record, RECORD_FIELDS, audit, where)
    audit.equal("NATIVE_EVALUATION_FORMAT", where, record["format"], native_evaluate.FORMAT_VERSION)
    audit.equal("NATIVE_AUTONOMOUS_INTERVENTION", where, record["intervention"], "none")
    audit.equal("NATIVE_AUTONOMOUS_POLICY", where, record["action_policy"], "model")
    audit.equal("PROBE_BINDING", where, record["probe"], spec)
    for key in ("family", "condition", "steps"):
        audit.equal("NATIVE_PROBE", f"{where}/{key}", record[key], spec[key])
    family = spec["family"]
    audit.equal("PAYLOAD_RULE_MAPPING", where, record["family_mapping"], list(FAMILY_MAPPINGS[family]))
    audit.equal("FUTURE_FAMILY_POLICY", where, record["context_policy"], [family])
    audit.equal("NATIVE_MODEL_CONFIG", where, record["model"], checkpoint["model_config"])
    audit.equal("NATIVE_OBSERVATION_WIDTH", where, record["model"]["input_size"], INPUT_SIZE)
    for key, expected in (("parameter_hash", checkpoint["parameter_hash"]),
            ("start_world_state_hash", checkpoint["world_hash"]), ("start_hidden_hash", checkpoint["hidden_hash"]),
            ("start_scheduler_hash", checkpoint["scheduler_hash"]), ("start_event_hash", checkpoint["event_hash"]),
            ("source_context_hash", checkpoint["context_hash"]),
            ("source_continuation_hash", checkpoint["source_continuation_hash"]),
            ("fork_tick", checkpoint["world_tick"]), ("fork_time", checkpoint["world_time"]),
            ("source_family_exposures", checkpoint["actual_family_exposures"]),
            ("source_family_recency", checkpoint["world_family_recency"])):
        audit.equal("FULL_NATIVE_FORK_ORIGIN", f"{where}/{key}", record[key], expected)
    for key, expected in native_evaluate.CONDITIONS[spec["condition"]].items():
        audit.equal("NATIVE_TIMING_CONDITION", f"{where}/{key}", record["regime"][key], list(expected))
    for key, expected in (("window_range", [1.5, 2.5]), ("cancel", .2), ("replace", .3)):
        audit.equal("NATIVE_PUBLIC_RULE", f"{where}/{key}", record["regime"][key], expected)
    expected_sources = {key: plan["sources"][str((ROOT / key).resolve())] for key in EVALUATION_SOURCE_FILES}
    audit.equal("NATIVE_EVALUATOR_SOURCE_BINDING", where, record["source_hashes"], expected_sources)
    # The old helper's v2 path checks matched cue/first-legal denominators. Its
    # legacy names overlap native long_wait with a different range, so prepend
    # native/ only inside this temporary view and check native ranges above.
    adapted = dict(record, format="integration-own-trace-v2", condition=f"native/{record['condition']}")
    _legacy_audit_record(adapted, audit, where, dict(evaluation_format="integration-own-trace-v2",
        eval_steps=spec["steps"], dt=plan["dt"], batch_size=plan["batch_size"], n_jobs=4))
    for name in ("source_family_exposures", "actual_family_exposures"):
        _exposures(record[name], audit, f"{where}/{name}")
    audit.require("NEW_ONLY_SELECTED_FAMILY", where,
        all(number == 0 for index, number in enumerate(record["actual_family_exposures"]["new"]) if index != family))
    audit.equal("FAMILY_COHORT_KEYS", where, set(record["family_cohorts"]), {str(index) for index in range(4)})
    for cohort in ("primary", "inherited", "carryover_contract"):
        for key in ("matured", "completed", "clean_completed"):
            values = [record["family_cohorts"][str(index)][cohort][key] for index in range(4)]
            audit.require("FAMILY_COHORT_COUNTS", where, all(_integer(value) for value in values))
            audit.equal("FAMILY_COHORT_PARTITION", f"{where}/{cohort}/{key}", sum(values), record[cohort][key])
            if cohort == "primary":
                audit.equal("PRIMARY_REQUESTED_RULE_ONLY", f"{where}/{key}", values[family], record[cohort][key])
    metrics = _metrics(record)
    detail = record["request_diagnosis"]["primary"]
    metrics.update(uncompleted_no_begin=_rate(detail["uncompleted_no_begin"], detail["requests"]),
        uncompleted_despite_legal=_rate(detail["uncompleted_despite_legal_opportunity"], detail["requests"]))
    return metrics


def audit_native_record(record, checkpoint, *, plan, spec, audit=None, where="record"):
    audit = _Checks() if audit is None else audit
    try:
        return _audit_native_record(record, checkpoint, plan=plan, spec=spec, audit=audit, where=where)
    except AuditError:
        raise
    except (ValueError, TypeError, KeyError, RuntimeError, IndexError, AttributeError) as error:
        raise AuditError("MALFORMED_NATIVE_EVALUATION", where, str(error), checks=sum(audit.counts.values())) from error


def _formal_protocol(path, root, audit, inputs):
    from experiments.lifelong.study import evaluation_specs
    path, digest = _input(path, inputs, audit, "PROTOCOL_FILE")
    plan = _read_json(path)
    _fields(plan, PROTOCOL_FIELDS, audit, str(path))
    audit.equal("PROTOCOL_FORMAT", str(path), plan["format"], "lifelong-study-protocol-v1")
    expected = dict(seeds=list(SEEDS), target_ticks=172032, main_ticks=131072, withdrawal_ticks=32768,
        return_ticks=8192, chunk_ticks=128, batch_size=32, width=128, initial_fan_in=32, dt=.25,
        training_device="cuda", evaluation_device="cpu", checkpoint_ticks=list(CHECKPOINT_TICKS),
        replay_capacity=32, replay_block_ticks=256, replay_warmup_ticks=128,
        family_mappings=[list(row) for row in FAMILY_MAPPINGS], evaluation_specs=evaluation_specs(),
        timing_coverage=dict(narrow=dict(duration=[1., 4.], gap=[.5, 2.]), wide=dict(duration=[1., 16.], gap=[.5, 32.])))
    for key, value in expected.items():
        audit.equal("FORMAL_PROTOCOL", f"{path}/{key}", plan[key], value)
    variants = [dict(architecture="fixed_shrink", schedule=schedule, learning=learning, coverage=coverage)
        for schedule in ("blocked", "mixed") for learning in ("recent_repeat", "history_replay") for coverage in ("narrow", "wide")]
    variants += [dict(architecture=architecture, schedule="mixed", learning="history_replay", coverage="wide")
                 for architecture in ("grow_shrink", "open_shrink")]
    audit.equal("FORMAL_VARIANTS", str(path), plan["variants"], variants)
    branches = []
    for seed in SEEDS:
        for variant in variants:
            label = "__".join(variant[key] for key in BRANCH_KEYS)
            branches.append(dict(id=f"seed_{seed:03d}__{label}", seed=seed, **variant,
                output=str(root / "training" / f"seed_{seed:03d}" / label)))
    audit.equal("FORMAL_30_BRANCHES", str(path), plan["branches"], branches)
    audit.require("PROTOCOL_SOURCE_SET", str(path), isinstance(plan["sources"], dict) and bool(plan["sources"]))
    # The amendment was added after this original protocol was frozen. New
    # analysis/measurement modules cannot silently enlarge its source identity.
    native_files = [ROOT / "experiments/lifelong" / name for name in MAIN_NATIVE_SOURCE_FILES]
    dependency_files = [ROOT / "experiments" / name for name in (
        "continuity/model.py", "continuity/world.py", "continuity/run.py", "continuity/evaluate.py",
        "integration/diagnose.py", "integration/run.py", "factors/structure.py", "rewiring/plasticity.py",
        "credit/run.py", "credit/local.py", "credit/batch.py", "credit/windows_runner.py",
        "loops/audit_run.py", "loops/structure.py", "loops/interventions.py")]
    required_sources = {str(file.resolve()) for file in native_files + dependency_files}
    optional_measurement_sources = {str((ROOT / "experiments/lifelong" / name).resolve())
                                    for name in ("supplement.py", "test_supplement.py")}
    audit.require("COMPLETE_PROTOCOL_SOURCE_SET", str(path), required_sources <= set(plan["sources"])
        and set(plan["sources"]) - required_sources <= optional_measurement_sources)
    for source, source_hash in plan["sources"].items():
        _, current = _input(source, inputs, audit, "PINNED_SOURCE_FILE")
        audit.equal("PINNED_SOURCE_SHA256", source, current, source_hash)
    _fields(plan["attachment"], {"path", "sha256"}, audit, "attachment")
    _, attachment_hash = _input(plan["attachment"]["path"], inputs, audit, "SOURCE_ATTACHMENT")
    audit.equal("SOURCE_ATTACHMENT_SHA256", str(path), attachment_hash, plan["attachment"]["sha256"])
    return plan, digest


def _supplement_protocol(path, parent_path, parent_hash, audit, inputs):
    from experiments.lifelong.supplement import specs
    path, digest = _input(path, inputs, audit, "SUPPLEMENT_PROTOCOL_FILE")
    plan = _read_json(path)
    _fields(plan, SUPPLEMENT_FIELDS, audit, str(path))
    audit.equal("SUPPLEMENT_FORMAT", str(path), plan["format"], "lifelong-coverage-supplement-v1")
    audit.equal("SUPPLEMENT_PARENT_PATH", str(path), str(Path(plan["parent_protocol"]).resolve()), str(Path(parent_path).resolve()))
    audit.equal("SUPPLEMENT_PARENT_SHA256", str(path), plan["parent_protocol_sha256"], parent_hash)
    audit.require("SUPPLEMENT_NO_TRAINING_CHANGE", str(path), plan["training_changed"] is False)
    audit.equal("SUPPLEMENT_EXACT_16_PROBES", str(path), plan["evaluation_specs"], specs())
    expected = {str((ROOT / "experiments/lifelong" / name).resolve()) for name in ("supplement.py", "test_supplement.py")}
    audit.equal("SUPPLEMENT_SOURCE_SET", str(path), set(plan["sources"]), expected)
    for source, expected_hash in plan["sources"].items():
        _, actual_hash = _input(source, inputs, audit, "SUPPLEMENT_SOURCE_FILE")
        audit.equal("SUPPLEMENT_SOURCE_SHA256", source, actual_hash, expected_hash)
    for name in ("reason", "interpretation", "created_utc"):
        audit.require("SUPPLEMENT_MEASUREMENT_RECORD", f"{path}/{name}", isinstance(plan[name], str) and bool(plan[name].strip()))
    audit.require("SUPPLEMENT_RAW_ROOT", str(path), isinstance(plan["raw_root"], str) and bool(plan["raw_root"]))
    return plan, digest


def _branch_worker(job):
    plan, item, protocol_hash = job
    audit, inputs = _Checks(), {}
    folder, label = Path(item["output"]).resolve(), item["id"]
    try:
        torch.set_num_threads(1)
        complete_path, _ = _input(folder / "launch-complete.json", inputs, audit)
        marker = _read_json(complete_path)
        _fields(marker, {"format", "branch", "protocol_sha256", "target", "stage_ticks", "checkpoint_sha256", "checkpoints", "hidden_resets"}, audit, str(complete_path))
        for key, expected in (("format", "lifelong-launch-complete-v1"), ("branch", item),
                ("protocol_sha256", protocol_hash), ("target", plan["target_ticks"]),
                ("stage_ticks", plan["target_ticks"]), ("hidden_resets", 0)):
            audit.equal("COMPLETE_BINDING", f"{label}/{key}", marker[key], expected)
        start_path, _ = _input(folder.with_name(folder.name + ".launch-start.json"), inputs, audit)
        audit.equal("START_BINDING", label, _read_json(start_path),
                    dict(format="lifelong-launch-start-v1", branch=item, protocol_sha256=protocol_hash))
        audit.equal("CHECKPOINT_SET", label, set(marker["checkpoints"]), {str(tick) for tick in plan["checkpoint_ticks"]})
        actual_paths = {path.resolve() for path in (folder / "checkpoints").glob("*.pt")}
        expected_paths = {folder / "checkpoints" / f"tick_{tick:06d}.pt" for tick in plan["checkpoint_ticks"]}
        audit.equal("CHECKPOINT_FILE_SET", label, actual_paths, expected_paths)
        checkpoints, previous = {}, None
        for tick in plan["checkpoint_ticks"]:
            entry = marker["checkpoints"][str(tick)]
            _fields(entry, {"path", "sha256", "semantic_hash"}, audit, f"{label}/{tick}")
            path, digest = _input(folder / "checkpoints" / f"tick_{tick:06d}.pt", inputs, audit)
            audit.equal("CHECKPOINT_PATH_BINDING", str(path), entry["path"], str(path))
            audit.equal("CHECKPOINT_BYTES_BINDING", str(path), entry["sha256"], digest)
            state = torch.load(path, map_location="cpu", weights_only=False)
            result = audit_checkpoint_payload(state, plan=plan, branch=item, tick=tick, audit=audit, where=str(path))
            audit.equal("CHECKPOINT_SEMANTIC_BINDING", str(path), result["semantic_hash"], entry["semantic_hash"])
            if previous is not None:
                audit.equal("UNCHANGED_BIRTH", str(path), result["birth_hash"], previous["birth_hash"])
                audit.equal("UNINTERRUPTED_TRACE_PREFIX", str(path),
                    json_hash(state["trace"][:previous["updates"]]), previous["trace_json_hash"])
            previous = result
            result.update(checkpoint=str(path), checkpoint_sha256=digest)
            # Reports are not retained thirteen times; the final one also
            # independently checks the saved JSON product and full trace.
            if tick == plan["target_ticks"]:
                latest, latest_hash = _input(folder / "latest.pt", inputs, audit)
                audit.equal("LATEST_COMPLETION_HASH", label, latest_hash, marker["checkpoint_sha256"])
                latest_state = torch.load(latest, map_location="cpu", weights_only=False)
                audit.equal("LATEST_FINAL_SEMANTIC", label, latest_state["state_hashes"]["semantic"], result["semantic_hash"])
                # Rehash the complete latest payload, not only its digest claim.
                audit_checkpoint_payload(latest_state, plan=plan, branch=item, tick=tick, audit=audit, where=str(latest))
                del latest_state
                report_path, _ = _input(folder / "report.json", inputs, audit)
                expected_report = deepcopy(result["report"])
                expected_report["config"]["device"] = state["config"]["device"]
                expected_report["runtime"] = state["runtime"]
                audit.equal("SAVED_NATIVE_REPORT", label, json_hash(_read_json(report_path)), json_hash(expected_report))
                trace_path, _ = _input(folder / "trace.jsonl", inputs, audit)
                lines = trace_path.read_text(encoding="utf-8").splitlines()
                audit.require("TRACE_JSONL_NONEMPTY", label, all(line.strip() for line in lines))
                audit.equal("SAVED_FULL_TRACE", label, json_hash([json.loads(line) for line in lines]), result["trace_json_hash"])
                for namespace, texts in (("sources", state["sources"]), ("dependencies", state["dependencies"])):
                    for name, text in texts.items():
                        snapshot, _ = _input(folder / namespace / name, inputs, audit, "SAVED_SOURCE_FILE")
                        audit.equal("SAVED_SOURCE_TEXT", str(snapshot), snapshot.read_text(encoding="utf-8"), text)
            result.pop("report")
            checkpoints[tick] = result
            del state
            gc.collect()
        return {"branch": item, "checkpoints": checkpoints, "inputs": inputs, "audit": audit.result()}
    except AuditError as error:
        return {"failure": error.audit, "branch": item, "audit": audit.result()}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError) as error:
        return {"failure": {"errors": [{"code": "MALFORMED_BRANCH", "where": label, "details": str(error)}]},
                "branch": item, "audit": audit.result()}


def summarize_seed_values(rows):
    rows = sorted(rows, key=lambda row: row["seed"])
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate seed in a statistical group")
    values = [row["value"] for row in rows if row["value"] is not None]
    if any(not _finite(value) for value in values):
        raise ValueError("nonfinite seed statistic")
    return dict(n=len(values), n_total=len(rows), n_undefined=len(rows) - len(values),
        mean=statistics.mean(values) if values else None,
        sample_sd=statistics.stdev(values) if len(values) > 1 else None, per_seed=rows)


def _summaries(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        metrics = {name: summarize_seed_values([dict(seed=row["seed"], value=row["metrics"][name]) for row in selected])
                   for name in selected[0]["metrics"]}
        result.append(dict(zip(fields, key)) | {"metrics": metrics})
    return result


def paired_contrasts(rows):
    result = []
    axes = (("learning", "history_replay", "recent_repeat", "replay_minus_recent"),
            ("schedule", "mixed", "blocked", "mixed_minus_blocked"),
            ("coverage", "wide", "narrow", "wide_minus_narrow"),
            ("architecture", "grow_shrink", "fixed_shrink", "grow_minus_fixed"),
            ("architecture", "open_shrink", "fixed_shrink", "open_minus_fixed"))
    base = (*BRANCH_KEYS, "tick", "purpose", "family", "condition", "steps")
    for axis, left, right, contrast in axes:
        fields = tuple(key for key in base if key != axis)
        grouped = defaultdict(dict)
        for row in rows:
            if row[axis] in (left, right):
                grouped[(*[row[field] for field in fields], row["seed"])][row[axis]] = row
        for key, pair in sorted(grouped.items()):
            if set(pair) != {left, right}:
                continue
            result.append(dict(zip(fields, key[:-1])) | dict(seed=key[-1], contrast=contrast,
                metrics={name: _difference(pair[left]["metrics"][name], pair[right]["metrics"][name])
                         for name in pair[left]["metrics"]}))
    return result


def _family_averages(rows):
    fields = ("seed", *BRANCH_KEYS, "tick", "purpose", "condition", "steps")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        families = [row["family"] for row in selected]
        if len(set(families)) != len(families):
            raise ValueError("duplicate family within a seed")
        metrics = {}
        for name in selected[0]["metrics"]:
            values = [row["metrics"][name] for row in selected if row["metrics"][name] is not None]
            metrics[name] = statistics.mean(values) if len(values) == len(selected) else None
        result.append(dict(zip(fields, key)) | dict(families=families, n_families=len(families), metrics=metrics))
    return result


def temporal_contrasts(rows):
    """Within-seed changes retain both fork horizons and actual checkpoint ages."""
    fields = ("seed", *BRANCH_KEYS, "family", "condition")
    groups = defaultdict(dict)
    for row in rows:
        if row["condition"] == "long_both":
            groups[tuple(row[field] for field in fields)][row["tick"]] = row
    comparisons = (("withdrawal_change", 131072, 163840),
                   ("return_after_1024", 163840, 164864),
                   ("return_after_4096", 163840, 167936),
                   ("return_after_8192", 163840, 172032))
    result = []
    for key, checkpoints in sorted(groups.items()):
        for name, before_tick, after_tick in comparisons:
            if before_tick not in checkpoints or after_tick not in checkpoints:
                continue
            if name.startswith("return_") and key[-2] != 0:
                continue
            before, after = checkpoints[before_tick], checkpoints[after_tick]
            result.append(dict(zip(fields, key)) | dict(contrast=name, before_tick=before_tick,
                after_tick=after_tick, before_steps=before["steps"], after_steps=after["steps"],
                live_ticks_between=after_tick - before_tick,
                before_family_recency=before["source_family_recency"],
                after_family_recency=after["source_family_recency"],
                horizon_matched=before["steps"] == after["steps"],
                interpretation="Original probes retained for provenance; unequal horizons and narrow long_both OOD prevent a within-support forgetting claim.",
                metrics={name: _difference(after["metrics"][name], before["metrics"][name])
                         for name in before["metrics"]}))
    return result


def within_training_support(coverage, condition):
    if coverage not in ("narrow", "wide") or condition not in native_evaluate.CONDITIONS:
        raise ValueError("unknown timing coverage or query condition")
    return condition in ("short", "narrow") or (coverage == "wide" and condition != "far")


def family_clock_metrics(checkpoint, family):
    """Keep world zero-based ticks distinct from post-update boundary stamps."""
    recency = checkpoint["family_recency"]
    world_tick, boundary = checkpoint["world_tick"], checkpoint["tick"]
    command = recency["last_command_tick"][family]
    command_time = recency["last_command_time"][family]
    live = recency["last_live_family_tick"][family]
    auxiliary = recency["last_aux_family_tick"][family]
    return dict(world_tick=world_tick, trainer_boundary_tick=boundary,
        last_real_command_tick=command, last_real_command_time=command_time,
        real_command_lag_world_ticks=None if command < 0 else world_tick - command,
        real_command_lag_physical_time=None if command < 0 else checkpoint["world_time"] - command_time,
        last_live_supervision_tick=live,
        live_supervision_lag_world_ticks=None if live < 0 else world_tick - live,
        last_auxiliary_supervision_boundary_tick=auxiliary,
        auxiliary_supervision_lag_boundary_ticks=None if auxiliary < 0 else boundary - auxiliary,
        live_supervised_positions=recency["live_supervised_family_positions"][family],
        auxiliary_supervised_positions=recency["aux_supervised_family_positions"][family])


def _evaluation_rows(value, item, checkpoints, specs, *, plan, protocol_hash,
                     origin, audit, where, parent_protocol_hash=None):
    fields = {"format", "branch", "protocol_sha256", "source_unchanged", "states", "records"}
    if parent_protocol_hash is not None:
        fields.add("parent_protocol_sha256")
    _fields(value, fields, audit, where)
    expected_format = "lifelong-evaluation-v1" if origin == "primary" else "lifelong-coverage-evaluation-v1"
    for key, expected in (("format", expected_format), ("branch", item),
            ("protocol_sha256", protocol_hash), ("source_unchanged", True)):
        audit.equal("EVALUATION_BINDING", f"{where}/{key}", value[key], expected)
    if parent_protocol_hash is not None:
        audit.equal("SUPPLEMENT_EVALUATION_PARENT", where, value["parent_protocol_sha256"], parent_protocol_hash)
    measured_ticks = {spec["tick"] for spec in specs}
    audit.equal("EVALUATED_CHECKPOINT_SET", where, set(value["states"]), {str(tick) for tick in measured_ticks})
    for tick in measured_ticks:
        state, cp = value["states"][str(tick)], checkpoints[tick]
        _fields(state, {"checkpoint", "checkpoint_sha256", "state_hash", "training_device_normalized_state_hash", "family_recency"}, audit, f"{where}/{tick}")
        for field, expected in (("checkpoint", cp["checkpoint"]), ("checkpoint_sha256", cp["checkpoint_sha256"]),
                ("state_hash", cp["cpu_semantic_hash"]), ("training_device_normalized_state_hash", cp["semantic_hash"]),
                ("family_recency", cp["family_recency"])):
            audit.equal("EVALUATED_NATIVE_STATE", f"{where}/{tick}/{field}", state[field], normalize(expected))
    audit.equal("EVALUATION_FORK_COUNT", where, len(value["records"]), len(specs))
    identity = {key: item[key] for key in ("id", "seed", *BRANCH_KEYS)}
    rows, idle_rows, content_rows, exogenous = [], [], [], []
    for record, spec in zip(value["records"], specs):
        cp = checkpoints[spec["tick"]]
        label = f"{where}/{spec['tick']}/{spec['family']}/{spec['condition']}/{spec['steps']}"
        metrics = audit_native_record(record, cp, plan=plan, spec=spec, audit=audit, where=label)
        meta = identity | spec | dict(probe_origin=origin,
            within_training_support=within_training_support(item["coverage"], spec["condition"]))
        rows.append(meta | dict(metrics=metrics, primary=record["primary"],
            actual_family_exposures=record["actual_family_exposures"], source_family_recency=cp["family_recency"],
            source_clock=family_clock_metrics(cp, spec["family"]),
            primary_censored=record["censored"]["primary"]["count"]))
        idle_rows.extend(meta | row for row in record["inactive_history"])
        content_rows.extend(meta | row for row in record["content_by_cohort_phase_age"])
        key = (item["seed"], item["schedule"], item["coverage"],
               spec["tick"], spec["family"], spec["condition"], spec["steps"])
        hashes = {field: record[field] for field in ("start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash")}
        exogenous.append((key, hashes, label))
    return rows, idle_rows, content_rows, exogenous


def matched_memory_rows(rows):
    ticks = (131072, 163840, 164864, 167936, 172032)
    selected, seen = [], set()
    for row in rows:
        key = (row["seed"], *[row[field] for field in BRANCH_KEYS], row["tick"], row["family"], row["condition"], row["steps"])
        if key in seen:
            raise ValueError("duplicate semantic frozen probe including its horizon")
        seen.add(key)
        if row["tick"] in ticks and row["condition"] in ("short", "long_both") and row["steps"] == 2048:
            selected.append(dict(row, query_role="withheld_family" if row["family"] == 0 else "other_family"))
    return selected


def matched_temporal_contrasts(rows):
    fields = ("seed", *BRANCH_KEYS, "family", "condition")
    groups = defaultdict(dict)
    for row in rows:
        if row["steps"] != 2048:
            raise ValueError("matched temporal analysis requires2048tick frozen probes")
        key = tuple(row[field] for field in fields)
        if row["tick"] in groups[key]:
            raise ValueError("duplicate matched frozen probe")
        groups[key][row["tick"]] = row
    result = []
    comparisons = (("withdrawal_change", 131072, 163840), ("return_after_1024", 163840, 164864),
                   ("return_after_4096", 163840, 167936), ("return_after_8192", 163840, 172032))
    for key, checkpoints in sorted(groups.items()):
        for name, before_tick, after_tick in comparisons:
            if before_tick not in checkpoints or after_tick not in checkpoints:
                continue
            before, after = checkpoints[before_tick], checkpoints[after_tick]
            result.append(dict(zip(fields, key)) | dict(contrast=name, before_tick=before_tick, after_tick=after_tick,
                before_steps=2048, after_steps=2048, live_ticks_between=after_tick - before_tick,
                query_role="withheld_family" if before["family"] == 0 else "other_family",
                within_training_support=within_training_support(before["coverage"], before["condition"]),
                before_primary_matured=before["primary"]["matured"], after_primary_matured=after["primary"]["matured"],
                before_primary_censored=before["primary_censored"], after_primary_censored=after["primary_censored"],
                before_query_family_exposures=before["actual_family_exposures"],
                after_query_family_exposures=after["actual_family_exposures"],
                before_source_clock=before["source_clock"], after_source_clock=after["source_clock"],
                metrics={metric: _difference(after["metrics"][metric], before["metrics"][metric]) for metric in before["metrics"]}))
    return result


def _matched_contrast_summaries(rows):
    # Populations, exposure counts and clocks belong to each seed, not to the
    # grouping identity. Retain them in the paired raw table alongside metrics.
    variable = {"before_primary_matured", "after_primary_matured", "before_primary_censored", "after_primary_censored",
                "before_query_family_exposures", "after_query_family_exposures", "before_source_clock", "after_source_clock"}
    return _contrast_summaries([{key: value for key, value in row.items() if key not in variable} for row in rows])


def build_report(protocol, root, *, supplement=None, workers=6):
    if not _integer(workers, 1) or workers > 6:
        raise ValueError("checkpoint workers must be between one and six")
    if supplement is None:
        raise ValueError("the frozen matched-horizon supplement protocol is required")
    root, audit, inputs = Path(root).resolve(), _Checks(), {}
    analysis_sources = {str(path.resolve()): file_sha256(path) for path in (
        Path(__file__), Path(_legacy_audit_record.__code__.co_filename), Path(_Checks.result.__code__.co_filename))}
    plan, protocol_hash = _formal_protocol(protocol, root, audit, inputs)
    supplement_plan, supplement_hash = _supplement_protocol(supplement, protocol, protocol_hash, audit, inputs)
    supplement_root = Path(supplement_plan["raw_root"]).resolve()
    for label, evaluation_root in (("PRIMARY", root / "evaluation"), ("SUPPLEMENT", supplement_root / "evaluation")):
        expected_files = {evaluation_root / f"{item['id']}.json" for item in plan["branches"]}
        audit.equal(f"{label}_EVALUATION_FILE_SET", str(evaluation_root),
                    {path.resolve() for path in evaluation_root.glob("*.json")}, expected_files)
    jobs = [(plan, item, protocol_hash) for item in plan["branches"]]
    if workers == 1:
        results = map(_branch_worker, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_branch_worker, jobs)
    branches = {}
    try:
        for result in results:
            audit.counts.update(result["audit"]["checks_by_code"])
            if "failure" in result:
                error = result["failure"]["errors"][0]
                raise AuditError(error["code"], error["where"], error.get("details"), checks=sum(audit.counts.values()))
            for path, digest in result["inputs"].items():
                if path in inputs:
                    audit.equal("WORKER_SHARED_INPUT", path, inputs[path], digest)
                inputs[path] = digest
            branches[result["branch"]["id"]] = result
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    seed_births, exogenous, evaluation_exogenous = {}, {}, {}
    per_seed, supplement_rows, checkpoint_rows, idle_rows, content_rows = [], [], [], [], []
    for item in plan["branches"]:
        branch = branches[item["id"]]
        cps, identity = branch["checkpoints"], {key: item[key] for key in ("id", "seed", *BRANCH_KEYS)}
        birth = cps[0]
        if item["seed"] in seed_births:
            audit.equal("COMMON_NATIVE_BIRTH_PARAMETERS", item["id"], birth["initial_parameter_hash"], seed_births[item["seed"]])
        seed_births[item["seed"]] = birth["initial_parameter_hash"]
        key = item["seed"], item["schedule"], item["coverage"]
        external = [(tick, cp["event_sequence_hash"], cp["scheduler_hash"], cp["event_hash"]) for tick, cp in cps.items()]
        if key in exogenous:
            audit.equal("MATCHED_TRAINING_EXOGENOUS_STREAM", item["id"], external, exogenous[key])
        exogenous[key] = external
        for tick, cp in cps.items():
            checkpoint_rows.append(identity | {key: value for key, value in cp.items() if key not in ("checks",)})
        for origin, evaluation_root, probe_specs, binding_hash, parent_hash, destination in (
                ("primary", root / "evaluation", plan["evaluation_specs"], protocol_hash, None, per_seed),
                ("supplement", supplement_root / "evaluation", supplement_plan["evaluation_specs"], supplement_hash, protocol_hash, supplement_rows)):
            path, _ = _input(evaluation_root / f"{item['id']}.json", inputs, audit)
            rows, idle, content, paired_streams = _evaluation_rows(_read_json(path), item, cps, probe_specs,
                plan=plan, protocol_hash=binding_hash, parent_protocol_hash=parent_hash,
                origin=origin, audit=audit, where=str(path))
            destination.extend(rows)
            idle_rows.extend(idle)
            content_rows.extend(content)
            for paired_key, external, label in paired_streams:
                if paired_key in evaluation_exogenous:
                    audit.equal("MATCHED_EVALUATION_EXOGENOUS_STREAM", label, external, evaluation_exogenous[paired_key])
                evaluation_exogenous[paired_key] = external
    audit.equal("TOTAL_NATIVE_FORKS", str(root), len(per_seed), 1380)
    audit.equal("TOTAL_SUPPLEMENT_FORKS", str(supplement_root), len(supplement_rows), 480)
    audit.equal("TOTAL_NATIVE_CHECKPOINTS", str(root), len(checkpoint_rows), 390)
    for path, expected in list(inputs.items()):
        audit.equal("FINAL_INPUT_SHA256", path, file_sha256(path), expected)
    for path, expected in analysis_sources.items():
        audit.equal("ANALYSIS_SOURCE_UNCHANGED", path, file_sha256(path), expected)
    contrasts = paired_contrasts(per_seed)
    family_average = _family_averages(per_seed)
    temporal = temporal_contrasts(per_seed)
    temporal_for_summary = [{key: value for key, value in row.items()
                            if key not in ("before_family_recency", "after_family_recency")} for row in temporal]
    training_rows = [row for row in checkpoint_rows if row["tick"] == plan["target_ticks"]]
    combined = per_seed + supplement_rows
    aligned = matched_memory_rows(combined)
    audit.equal("COMBINED_FROZEN_FORKS", str(root), len(combined), 1860)
    audit.equal("MATCHED_MEMORY_FROZEN_FORKS", str(root), len(aligned), 840)
    audit.equal("WITHHELD_MATCHED_MEMORY_FORKS", str(root), sum(row["family"] == 0 for row in aligned), 300)
    expected_aligned = {(tick, family, condition, 2048)
        for tick in (131072, 163840, 164864, 167936, 172032)
        for family in (range(4) if tick in (131072, 163840, 172032) else (0,))
        for condition in ("short", "long_both")}
    for item in plan["branches"]:
        audit.equal("EXACT_MATCHED_MEMORY_GRID", item["id"],
            {(row["tick"], row["family"], row["condition"], row["steps"]) for row in aligned if row["id"] == item["id"]}, expected_aligned)
    matched_changes = matched_temporal_contrasts(aligned)
    matched_pairs = paired_contrasts(aligned)
    group = (*BRANCH_KEYS, "tick", "purpose", "family", "condition", "steps")
    return dict(format=FORMAT_VERSION, complete=True, protocol=str(Path(protocol).resolve()), protocol_sha256=protocol_hash,
        supplement_protocol=str(Path(supplement).resolve()), supplement_protocol_sha256=supplement_hash,
        audit=audit.result(), checkpoint_workers=workers,
        counts=dict(training_runs=30, trained_seeds=3, full_checkpoints=390, primary_evaluation_files=30,
            supplemental_evaluation_files=30, primary_evaluation_forks=1380, supplemental_evaluation_forks=480,
            evaluation_forks=1860, matched_memory_forks=840, withheld_matched_memory_forks=300),
        inputs=inputs, protocol_plan=plan, checkpoints=checkpoint_rows, per_seed=per_seed,
        supplement_plan=supplement_plan, supplement_per_seed=supplement_rows, combined_per_seed=combined,
        main_summary=_summaries(per_seed, group),
        supplement_summary=_summaries(supplement_rows, group), combined_summary=_summaries(combined, group),
        family_average_per_seed=family_average,
        family_average_summary=_summaries(family_average, (*BRANCH_KEYS, "tick", "purpose", "condition", "steps")),
        paired_contrasts_per_seed=contrasts,
        paired_contrasts=_contrast_summaries(contrasts),
        temporal_contrasts_per_seed=temporal, temporal_contrasts=_contrast_summaries(temporal_for_summary),
        matched_memory_per_seed=aligned,
        matched_memory_summary=_summaries(aligned, (*BRANCH_KEYS, "tick", "family", "condition", "steps")),
        matched_temporal_contrasts_per_seed=matched_changes,
        matched_temporal_contrasts=_matched_contrast_summaries(matched_changes),
        matched_paired_contrasts_per_seed=matched_pairs, matched_paired_contrasts=_contrast_summaries(matched_pairs),
        training_per_seed=training_rows,
        idle_by_history_per_seed=idle_rows, content_by_phase_age_per_seed=content_rows,
        analysis_sources=analysis_sources,
        definitions=dict(statistical_unit="Three independent trained seeds; families/forks/requests are repeated measurements.",
            native_record_adaptation="Native format/model/context/ranges/source hashes audited first; temporary legacy-v2 view checks unchanged matured, cue and idle bookkeeping.",
            clock_axes="Request duration, actual terminal-to-idle age, real-family cue recency, live-supervision recency and auxiliary-supervision recency are distinct.",
            retention="Family0 complete-state frozen function queries at tick163840 precede return-phase live supervision. Queries consume new context/task cues and normal environment receipts without parameter learning; this does not isolate parameter-only memory.",
            return_learning="Matched2048tick queries at164864/167936/172032 measure function performance after fixed live training intervals. Actual family commands and supervised positions are retained; no cold-start learning-speed claim follows from tick counts alone.",
            measurement_amendment="The separately frozen supplement records a supported-domain/horizon measurement correction before formal-score inspection. Original1380forks and all30 training trajectories remain unchanged.",
            timing_support="Short is within both narrow/wide training support. Long_both is within wide support and outside narrow duration/gap support; a narrow long_both drop alone is not evidence of forgetting an acquired function.",
            matched_headline="Primary longitudinal comparisons use the same condition and2048tick horizon at131072/163840/164864/167936/172032. Family0 is primary; available other-family probes are controls.",
            legacy_temporal="Original temporal_contrasts remain separate and retain their factual1024/2048 horizons; prefer matched_temporal_contrasts for longitudinal conclusions.",
            compute="Both methods have equal live/auxiliary forward, detached warmup, differentiable suffix and Adam budgets, including identical startup lengths.",
            history_integrity="Retained observations, original labels and actions rehash to original live chunks; actual start_h links to the preceding live hidden hash.",
            structural_confound="Fixed/grow/open share initial sparse W and function. Their later capacity and optimizer/shadow behavior differ; contrasts test complete policies.",
            source_window_commands="Replay family command counts include detached warmup; suffix_family_positions separately counts supervised positions.",
            supervised_positions="Family position counts use the persistent pre-action context at every supervised slot, including idle WAIT; they are not counts of distinct NEW requests.",
            family_clock_basis="Cue and live-supervision recencies subtract zero-based world.tick; auxiliary-supervision ages subtract trainer.ticks post-update boundaries.",
            structure_history="Unique outside-initial births, initial-coordinate reactivations, repeat birth events, numerical support and eligibility are reported separately; all policies retain the same dense tensor allocation."),
        limitations=plan["limits"] + ["Even aligned frozen queries start from different inherited h/world/scheduler/receipt states and matured populations; they do not isolate parameter-only memory or pure spacing."])


def _contrast_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        fields = tuple(sorted(key for key in row if key not in ("seed", "metrics")))
        grouped[(fields, tuple(row[key] for key in fields))].append(row)
    return [dict(zip(fields, values)) | dict(metrics={metric: summarize_seed_values([
        dict(seed=row["seed"], value=row["metrics"][metric]) for row in selected]) for metric in selected[0]["metrics"]})
        for (fields, values), selected in sorted(grouped.items())]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalize(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows):
    rows = [normalize(row) for row in rows]
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_artifacts(report, output):
    if report.get("complete") is not True or report.get("audit", {}).get("status") != "passed":
        raise ValueError("only a complete passed audit can produce a result artifact")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "report.json", report)
    files = {"report.json": file_sha256(output / "report.json")}
    for name in ("per_seed", "main_summary", "family_average_per_seed", "family_average_summary",
                 "supplement_per_seed", "supplement_summary", "combined_per_seed", "combined_summary",
                 "paired_contrasts_per_seed", "paired_contrasts", "temporal_contrasts_per_seed", "temporal_contrasts",
                 "matched_memory_per_seed", "matched_memory_summary", "matched_temporal_contrasts_per_seed",
                 "matched_temporal_contrasts", "matched_paired_contrasts_per_seed", "matched_paired_contrasts",
                 "training_per_seed", "checkpoints",
                 "idle_by_history_per_seed", "content_by_phase_age_per_seed"):
        rows = []
        for row in report[name]:
            base = {key: value for key, value in row.items() if key != "metrics"}
            if "metrics" in row and row["metrics"] and isinstance(next(iter(row["metrics"].values())), dict):
                for metric, summary in row["metrics"].items():
                    rows.append(base | dict(metric=metric, **summary))
            else:
                rows.append(base | row.get("metrics", {}))
        path = output / f"{name}.csv"
        _csv(path, rows)
        files[path.name] = file_sha256(path)
    _atomic_json(output / "artifacts.json", files)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True,
        help="frozen coverage-probe-supplement-v1.json; all30 supplemental evaluation files are mandatory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.protocol, args.root, supplement=args.supplement, workers=args.workers)
        write_artifacts(report, args.output)
    except (AuditError, ValueError) as error:
        parser.exit(1, f"Native lifelong audit rejected: {error}\n")
    print(json.dumps({"audit": report["audit"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
