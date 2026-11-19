"""Independent, read-only accounting for every declared undirected branch.

Execution sources stay frozen.  This analysis restores actual complete states,
replays recorded world/teacher and read-only model primals without updates, and audits ledger
and evaluation provenance.  Optional independent JVP evidence is byte-bound
and described as two projection checks, never a proof of every S entry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.integration.report import _audit_record as legacy_ledger_audit, _metrics
from experiments.lifelong import evaluate as observer
from experiments.lifelong.world import ContextWorld, FAMILY_MAPPINGS, TRAINING_COVERAGE
from experiments.lifelong.replay import HistoryReservoir, block_hash, copied_block, combine_blocks
from experiments.undirected import ledger, run as native, study
from experiments.undirected.credit import stable_l2
from experiments.undirected.queue import atomic_json, file_sha256


FORMAT = "undirected-analysis-v1"
ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_SOURCES = ("experiments/lifelong/evaluate.py", "experiments/lifelong/world.py",
    "experiments/integration/diagnose.py", "experiments/continuity/world.py", "experiments/continuity/evaluate.py")
METRICS = ("strict_completion", "completion", "finish_precision", "own_state_content", "own_state_legal_content",
    "first_legal_content", "inactive_per_1000_idle", "inactive_per_1000_total", "first_error_fraction", "cue_content",
    "matched_cue_content", "matched_legal_content", "matched_loss_given_cue_correct", "uncompleted_no_begin",
    "uncompleted_despite_legal")
P = dict(interface="ports", mode="undirected", credit="rtrl", structure="fixed", course="warmup")
FIXED_PROBE_CHECK = "FIXED_PROBE_SAME_LEARNING_EXECUTION"
TRACE_FIELD_GROUPS = dict(
    learning_state=("parameter_hash", "optimizer_hash", "live_hidden_hash", "credit_gradient_hash", "physical_matrix"),
    behavior=("action_hash", "live_world_hash", "data_hash"),
    external_stream=("event_hash", "actual_family_exposures"))


def variant(**changes):
    return tuple(({**P, **changes})[key] for key in study.KEYS)


CONTRASTS = (
    ("core_shared_minus_untied_512", variant(interface="core", credit="recorded512"), variant(interface="core", mode="untied", credit="recorded512")),
    ("ports_shared_minus_untied_512", variant(credit="recorded512"), variant(mode="untied", credit="recorded512")),
    ("ports_minus_core_shared_512", variant(credit="recorded512"), variant(interface="core", credit="recorded512")),
    ("ports_minus_core_untied_512", variant(mode="untied", credit="recorded512"), variant(interface="core", mode="untied", credit="recorded512")),
    ("ports_512_minus_128", variant(credit="recorded512"), variant(credit="tbptt128")),
    ("ports_rtrl_minus_512", variant(), variant(credit="recorded512")),
    ("ports_rtrl_minus_reset", variant(), variant(credit="rtrl_reset")),
    ("ports_rtrl_minus_128", variant(), variant(credit="tbptt128")),
    ("rtrl_shared_minus_untied", variant(), variant(mode="untied")),
    ("both_minus_retire", variant(structure="both"), variant(structure="retire")),
    ("both_minus_grow", variant(structure="both"), variant(structure="grow")),
    ("both_minus_fixed_probe", variant(structure="both"), variant(structure="fixed_probe")),
    ("fixed_probe_minus_fixed", variant(structure="fixed_probe"), variant()),
    ("wide_minus_warmup_128", variant(credit="tbptt128", course="wide"), variant(credit="tbptt128")),
    ("wide_minus_warmup_rtrl", variant(course="wide"), variant()),
)


class AuditError(ValueError):
    pass


class Checks:
    def __init__(self):
        self.counts = Counter()

    def require(self, code, where, passed, details=None, **unused):
        self.counts[code] += 1
        if not passed:
            raise AuditError(f"{code} at {where}: {details!r}")

    def equal(self, code, where, actual, expected):
        self.require(code, where, actual == expected, dict(actual=actual, expected=expected))


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _input(path, inputs, expected=None):
    path = Path(path).resolve()
    digest = file_sha256(path)
    if (expected is not None and digest != expected) or (str(path) in inputs and inputs[str(path)] != digest):
        raise AuditError(f"input bytes changed: {path}")
    inputs[str(path)] = digest
    return path


def _plan(path, mechanical):
    raw = _json(path)
    if raw.get("mechanical_test") and not mechanical:
        raise AuditError("mechanical protocol requires explicit mechanical=True")
    if type(mechanical) is not bool or not raw.get("branches"):
        raise AuditError("explicit protocol and mechanical flag required")
    return study.read_plan(path, raw["branches"][0]["id"])[0]


def _key(item):
    return tuple(item[key] for key in study.KEYS)


def _graph(mask, pairs, width):
    neighbors = [set() for _ in range(width)]
    for i, j in pairs[:, mask].T.tolist():
        neighbors[i].add(j)
        neighbors[j].add(i)
    unseen, sizes = set(range(width)), []
    while unseen:
        pending, seen = [next(iter(unseen))], set()
        while pending:
            i = pending.pop()
            if i in seen:
                continue
            seen.add(i)
            pending.extend(neighbors[i] - seen)
        unseen -= seen
        sizes.append(len(seen))
    degrees = [len(row) for row in neighbors]
    return dict(edges=int(mask.sum()), components=len(sizes), largest_component=max(sizes),
                min_degree=min(degrees), max_degree=max(degrees), mean_degree=sum(degrees) / width)


def _matrix(parameters, buffers, mode, width):
    s, indices = parameters["edge_shared"], buffers["pair_index"]
    upper = lower = s * (1. / math.sqrt(2.))
    if mode == "untied":
        upper, lower = (s + parameters["edge_skew"]) * (1. / math.sqrt(2.)), (s - parameters["edge_skew"]) * (1. / math.sqrt(2.))
    result = torch.zeros(width, width, dtype=s.dtype)
    result[indices[0], indices[1]], result[indices[1], indices[0]] = upper, lower
    return result


def _matrix_checks(matrix, parameters, mask, mode, checks, where, expected=None):
    checks.require("PHYSICAL_MATRIX_FINITE_DIAGONAL", where, bool(torch.isfinite(matrix).all()) and not bool(matrix.diag().ne(0).any()))
    checks.require("PAIR_ATOMIC_SUPPORT", where, not bool(parameters["edge_shared"][~mask].ne(0).any())
                   and (mode != "untied" or not bool(parameters["edge_skew"][~mask].ne(0).any())))
    if mode == "undirected":
        checks.require("TRUE_SHARED_SYMMETRY", where, "edge_skew" not in parameters and torch.equal(matrix, matrix.T))
    support = parameters["edge_shared"].ne(0)
    if mode == "untied":
        support |= parameters["edge_skew"].ne(0)
    checks.require("ORTHONORMAL_PAIR_GEOMETRY", where, math.isclose(float(matrix.square().sum()),
        sum(float(parameters[name].square().sum()) for name in ("edge_shared", "edge_skew") if name in parameters), rel_tol=1e-12, abs_tol=1e-12))
    if mode == "undirected":
        checks.equal("UNORDERED_VS_ORIENTED", where, int(matrix.ne(0).sum()), 2 * int(support.sum()))
    if expected is not None:
        for name, actual in (("frobenius_norm", float(matrix.norm())),
            ("antisymmetric_frobenius_norm", float((matrix - matrix.T).norm())),
            ("oriented_nonzero_couplings", int(matrix.ne(0).sum()))):
            checks.require("ACTUAL_MATRIX_DIAGNOSTIC", f"{where}/{name}", math.isclose(actual, expected[name], rel_tol=1e-11, abs_tol=1e-12))
    return support


def _credit_checks(trainer, checks, where):
    c, age, engine = trainer.config, trainer.updates, trainer.credit
    checks.equal("CREDIT_AGE", where, engine.updates, age)
    checks.equal("FINITE_WINDOW_TAPE", where, len(engine.tape), min(age, engine.config.history_blocks))
    forward_ticks = 0
    for index, point in enumerate(trainer.trace, 1):
        m, w = point["credit"], point["credit"]["work"]
        path = index * c.chunk_ticks if c.credit == "rtrl" else min(index, engine.config.history_blocks) * c.chunk_ticks
        checks.equal("DECLARED_LIVE_CREDIT_PATH", where, m["credit_path_ticks"], path)
        checks.equal("AUXILIARY_LOCAL_SUFFIX", where, point["auxiliary"]["auxiliary_credit_hard_cap_ticks"], c.chunk_ticks)
        checks.equal("ONE_JOINT_OBJECTIVE", where, point["combined_loss"], .5 * (point["live_loss"] + point["auxiliary"]["loss"]))
        is_rtrl = c.credit.startswith("rtrl")
        forward_ticks += c.chunk_ticks if is_rtrl else min(index, engine.config.history_blocks) * c.chunk_ticks
        checks.equal("JACOBIAN_ROWS", where, w["jacobian_reverse_rows"], c.batch_size * c.width if is_rtrl else 0)
        checks.equal("FULL_J_S_COMPUTE", where, w["sensitivity_matmul_mac"], c.batch_size * c.width**2 * engine.dynamic_parameter_count if is_rtrl else 0)
        checks.equal("RESET_SAME_COMPUTE_CONTROL", where, w["reset_uses_full_jacobian_and_matmul"], c.credit == "rtrl_reset")
        diagnostics = m["gradient_diagnostics"]
        if c.credit == "rtrl_reset":
            checks.equal("RESET_CROSS_TERM_ZERO", where, diagnostics["cross_boundary_gradient_l2"], 0.)
        if c.credit == "rtrl":
            previous = 0. if index == 1 else trainer.trace[index - 2]["credit"]["gradient_diagnostics"]["next_sensitivity_l2"]
            checks.equal("UNRESET_RTRL_TRACE_NORM_CHAIN", where, diagnostics["previous_sensitivity_l2"], previous)
    if age and c.credit.startswith("rtrl"):
        s = engine.sensitivity
        checks.require("FULL_S_LAYOUT_FINITE", where, isinstance(s, torch.Tensor) and s.dtype == torch.float64
            and s.shape == (c.batch_size, c.width, engine.dynamic_parameter_count) and bool(torch.isfinite(s).all()))
    else:
        checks.equal("NO_UNDECLARED_S", where, engine.sensitivity, None)
    checks.equal("NO_HIDDEN_RESETS", where, trainer.hidden_resets, 0)
    is_rtrl, ticks, batch, chunk = c.credit.startswith("rtrl"), trainer.ticks, c.batch_size, c.chunk_ticks
    warmup, calls = max(0, ticks - chunk), 4 * len(trainer.operator.event_history)
    expected_work = dict(live_forward_ticks=ticks, live_forward_actor_ticks=ticks * batch, live_loss_positions=ticks * batch * c.n_jobs,
        aux_forward_ticks=ticks + warmup, aux_warmup_ticks=warmup, aux_loss_ticks=ticks,
        aux_forward_actor_ticks=(ticks + warmup) * batch, aux_loss_positions=ticks * batch * c.n_jobs,
        aux_reverse_sweeps=age, optimizer_steps=age, real_ticks_admitted=ticks,
        credit_loss_forward_calls=forward_ticks, credit_loss_forward_actor_ticks=forward_ticks * batch,
        credit_jacobian_forward_calls=ticks if is_rtrl else 0, credit_jacobian_forward_actor_ticks=ticks * batch if is_rtrl else 0,
        credit_loss_reverse_sweeps=age, credit_jacobian_reverse_rows=age * batch * c.width if is_rtrl else 0,
        credit_jacobian_reverse_chunks=age * math.ceil(c.width / c.jacrev_chunk_size) if is_rtrl else 0,
        credit_sensitivity_matmul_mac=age * batch * c.width**2 * engine.dynamic_parameter_count if is_rtrl else 0,
        structural_probe_calls=calls, structural_probe_forward_ticks=calls * 3 * chunk,
        structural_probe_forward_actor_ticks=calls * 3 * chunk * batch,
        structural_probe_loss_ticks=calls * 2 * chunk, structural_probe_warmup_ticks=calls * chunk)
    checks.equal("COMPLETE_SEPARATE_COMPUTE_ACCOUNTING", where, trainer.work, expected_work)
    checks.require("INTEGER_COMPUTE_ACCOUNTING", where, all(type(value) is int and value >= 0 for value in trainer.work.values()))
    expected_course = [dict(tick=0, coverage=c.coverage_at(0))]
    if c.course == "warmup" and trainer.ticks > c.warmup_ticks:
        expected_course.append(dict(tick=c.warmup_ticks, coverage="wide"))
    checks.equal("FUTURE_ONLY_COURSE_BOUNDARIES", where, trainer.course_history, expected_course)
    for point in trainer.trace:
        checks.equal("ORIGINAL_TRAINING_COVERAGE", where, point["coverage"], c.coverage_at(point["tick"] - c.chunk_ticks))
        checks.equal("ORIGINAL_TRAINING_PHASE", where, point["phase"], trainer.schedule.mode_for_tick(point["tick"] - c.chunk_ticks)["phase"])


def _checkpoint(path, plan, item, tick, checks, inputs, graph=False):
    path = _input(path, inputs)
    trainer = native.NativeTrainer.from_checkpoint(path, device="cpu")
    study.validate(trainer, plan, item, tick)
    checks.require("FORMAL_INTERFACE_DIMENSIONS", str(path), plan["mechanical_test"] or
        (trainer.config.n_jobs, trainer.model.input_size, trainer.config.width, trainer.config.batch_size) == (2, 56, 80, 8))
    _credit_checks(trainer, checks, str(path))
    parameters, buffers = dict(trainer.model.named_parameters()), dict(trainer.model.named_buffers())
    matrix = trainer.model.recurrent_matrix().detach().cpu()
    expected = trainer.trace[-1]["physical_matrix"] if tick else None
    support = _matrix_checks(matrix, parameters, trainer.operator.mask, item["mode"], checks, str(path), expected)
    physical = trainer._matrix_diagnostics()
    if expected is not None:
        for name in ("operator_norm", "spectral_radius"):
            checks.require("ACTUAL_PHYSICAL_SPECTRUM", str(path), math.isclose(physical[name], expected[name], rel_tol=1e-11, abs_tol=1e-12))
    count = sum(p.numel() for p in trainer.model.parameters())
    checks.equal("ACTUAL_PARAMETER_COUNT", str(path), count, trainer.model.parameter_report()["total_parameters"])
    checks.equal("DYNAMIC_LAYOUT_COUNT", str(path), trainer.credit.dynamic_parameter_count,
                 sum(parameters[name].numel() for name in trainer.model.dynamic_parameter_names()))
    if tick == 0:
        signature = study._birth_signature(trainer)
        checks.equal("ACTUAL_BIRTH_SIGNATURE", str(path), signature, plan["birth_bindings"]["by_branch"][item["id"]])
        birth_graph = _graph(trainer.operator.initial_mask, trainer.model.pair_index, trainer.config.width)
        checks.equal("CONNECTED_BIRTH", str(path), birth_graph["components"], 1)
        checks.require("REGULAR_BIRTH", str(path), birth_graph["min_degree"] == birth_graph["max_degree"] == trainer.config.initial_degree)
        checks.require("BIRTH_GAIN_IS_FROBENIUS_ONLY", str(path), math.isclose(float(matrix.norm()), trainer.config.birth_gain * math.sqrt(trainer.config.width), rel_tol=1e-12))
    row = dict(id=item["id"], seed=item["seed"], **{key: item[key] for key in study.KEYS}, tick=tick,
        checkpoint=str(path), checkpoint_sha256=inputs[str(path)], semantic_hash=state_digest(trainer.semantic_state()),
        parameter_hash=state_digest(trainer.model.state_dict()), hidden_hash=state_digest(trainer.h),
        wrapped_hidden_hash=state_digest({"hidden": trainer.h}), world_hash=state_digest(trainer.world.state_dict()),
        scheduler_hash=state_digest(trainer.world.state_dict()["scheduler"]), event_hash=trainer.world.event_hash,
        context_hash=state_digest(trainer.world.state_dict()["context"]), world_tick=trainer.world.tick, world_time=trainer.world.now,
        source_continuation_hash=state_digest(dict(model=trainer.model.state_dict(), world=trainer.world.state_dict(), hidden=trainer.h)),
        matrix_hash=state_digest(matrix), total_parameters=count, persistent_state_scalars=trainer.config.batch_size * trainer.config.width,
        state_width=trainer.config.width, dynamic_parameters=trainer.credit.dynamic_parameter_count,
        sensitivity_scalars=0 if trainer.credit.sensitivity is None else trainer.credit.sensitivity.numel(),
        sensitivity_bytes=0 if trainer.credit.sensitivity is None else trainer.credit.sensitivity.numel() * 8,
        sensitivity_l2=None if trainer.credit.sensitivity is None else stable_l2(trainer.credit.sensitivity),
        credit_tape_blocks=len(trainer.credit.tape), physical_matrix=physical,
        eligible_graph=_graph(trainer.operator.mask, trainer.model.pair_index, trainer.config.width),
        support_graph=_graph(support, trainer.model.pair_index, trainer.config.width), structure_summary=trainer.operator.summary(),
        work=deepcopy(trainer.work), actual_family_exposures=trainer.world.family_exposures(),
        world_family_recency=trainer.world.family_recency(), model_config=normalized(trainer.model.config()),
        replay_hash=state_digest(trainer.replay.state_dict()), per_actor_state_scalars=trainer.config.width,
        batched_state_scalars=trainer.config.width * trainer.config.batch_size,
        dense_field_mac_per_actor=trainer.config.width**2, heun_fields_per_substep=2,
        checkpoint_bytes=path.stat().st_size,
        credit_diagnostic_norm_elements=sum(point["credit"]["work"]["diagnostic_norm_elements"] for point in trainer.trace))
    row.update(matrix_operator_norm=physical["operator_norm"], matrix_spectral_radius=physical["spectral_radius"],
        matrix_frobenius_norm=physical["frobenius_norm"], matrix_difference_transpose_norm=physical["antisymmetric_frobenius_norm"],
        eligible_pairs=int(trainer.operator.mask.sum()), nonzero_pairs=int(support.sum()),
        eligible_components=row["eligible_graph"]["components"], support_components=row["support_graph"]["components"],
        branch_id=item["id"], checkpoint_tick=tick,
        snapshot_role=dict(zip(trainer.config.snapshot_ticks(), ("birth", "initial_coverage_budget_end", "middle_main", "main_end", "withdrawal_end", "return_end"))).get(tick, "noncanonical_last_valid"))
    if graph:
        pairs = trainer.model.pair_index.cpu()
        row["_graph_payload"] = dict(branch_id=item["id"], tick=tick, width=trainer.config.width,
            id=item["id"], seed=item["seed"], **{key: item[key] for key in study.KEYS}, state_width=trainer.config.width,
            checkpoint=str(path), checkpoint_sha256=inputs[str(path)], matrix_hash=row["matrix_hash"],
            eligible_edges=pairs[:, trainer.operator.mask].T.tolist(),
            support_edges=[[i, j, float(matrix[i, j]), float(matrix[j, i])] for i, j in pairs[:, support].T.tolist()],
            source_hashes=trainer.source_hashes, downsampled=False,
            input_ports=list(range(trainer.config.input_size)),
            output_ports=list(range(trainer.config.input_size, trainer.config.input_size + trainer.config.n_jobs * 11)),
            port_role_note="Ordinary evolving fixed external ports" if item["interface"] == "ports" else "Birth selector positions only; learned encoder/head need not retain these roles",
            scope=trainer.model.parameter_report()["graph_scope"])
    return trainer, row


def _world_block(world, model, payload, config, schedule, checks, point):
    r, live = payload["record"], payload["live"]
    hidden = r["start_h"].clone()
    for offset, observation in enumerate(r["observations"]):
        tick = r["start_tick"] + offset
        policy = schedule.mode_for_tick(tick)
        if tuple(world.family_allowed) != policy["future_family_allowed"]:
            world.set_family_policy(policy["future_family_allowed"])
        if config.course == "warmup" and tick == config.warmup_ticks:
            world.set_regime(**TRAINING_COVERAGE["wide"], name="undirected-wide")
        actual = world.observe().to(dtype=getattr(torch, config.dtype)).cpu()
        checks.require("ORIGINAL_OBSERVATION", str(tick), torch.equal(actual, observation))
        checks.require("CAUSAL_TEACHER_LABEL", str(tick), torch.equal(world.oracle("midpoint").cpu(), r["targets"][offset]))
        checks.require("ORIGINAL_CONTEXT", str(tick), torch.equal(world.current_family.cpu(), live["family_context"][offset]))
        checks.equal("ORIGINAL_ELAPSED", str(tick), float(world.elapsed), r["elapsed"][offset])
        logits, hidden = torch.func.functional_call(model, (r["parameters"], r["buffers"]),
            (actual, hidden, r["elapsed"][offset]), strict=True)
        checks.require("ACTUAL_NEURAL_PRIMAL_FINITE", str(tick), bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(logits).all()))
        checks.require("ACTUAL_MODEL_ARGMAX_ACTION", str(tick), torch.equal(logits.argmax(-1), live["actions"][offset]))
        world.act(live["actions"][offset])
    checks.require("ACTUAL_RECORDED_NEURAL_ENDPOINT", str(r["end_tick"]), torch.equal(hidden, r["end_h"]))
    checks.equal("REPLAYED_COMPLETE_WORLD", str(r["end_tick"]), state_digest(world.state_dict()), point["live_world_hash"])
    checks.equal("REPLAYED_EXTERNAL_EVENTS", str(r["end_tick"]), world.event_hash, point["event_hash"])
    checks.equal("REPLAYED_ACTUAL_EXPOSURE", str(r["end_tick"]), world.family_exposures(), point["actual_family_exposures"])


def _ledger_audit(trainer, newborn, plan, item, sha, checks, inputs, anchors):
    folder = Path(item["output"]) / "live_records"
    inventory = ledger.read_ledger(folder, expected_binding=study.ledger_binding(item, sha), verify_files=False)
    entries = inventory["entries"]
    checks.equal("COMPLETE_SUCCESSFUL_LIVE_LEDGER", item["id"], len(entries), trainer.updates)
    for key in ("binding_path", "manifest_path"):
        _input(inventory[key], inputs, inventory[key.replace("_path", "_sha256")])
    mask = trainer.operator.initial_mask.clone()
    world = ContextWorld.from_state_dict(newborn.world.state_dict(), device="cpu")
    reservoir = HistoryReservoir(trainer.config.replay_capacity, 2 * trainer.config.chunk_ticks,
                                native.seed_for(trainer.config.seed, "replay"))
    previous_chunk = None
    previous_h, rolling = newborn.h.detach().cpu(), hashlib.sha256(b"actual-physical-preparameter-matrices-v1").hexdigest()
    tape_hashes = {block["end_tick"]: ledger.content_hash(block) for block in trainer.credit.tape}
    buffers = {name: value.detach().cpu() for name, value in trainer.model.named_buffers()}
    for index, entry in enumerate(entries):
        _input(folder / entry["file"], inputs, entry["sha256"])
        payload = ledger.load_record(folder, entry, inventory["binding_hash"])
        record, point = payload["record"], trainer.trace[index]
        trainer.credit._validate_record(record)
        checks.equal("LIVE_SOURCE_TEXT_HASHES", item["id"], payload["source_hashes"], trainer.source_hashes)
        checks.equal("LIVE_DEPENDENCY_TEXT_HASHES", item["id"], payload["dependency_hashes"], trainer.dependency_hashes)
        checks.equal("LIVE_MODEL_INTERFACE", item["id"], payload["model_class"], type(trainer.model).__name__)
        checks.equal("LIVE_MODEL_CONFIGURATION", item["id"], payload["model_config"], normalized(trainer.model.config()))
        checks.equal("LIVE_DYNAMIC_LAYOUT", item["id"], payload["dynamic_names"], list(trainer.credit.dynamic_names))
        checks.equal("LIVE_LOCAL_LAYOUT", item["id"], payload["local_names"], list(trainer.credit.local_names))
        checks.equal("LIVE_CREDIT_METHOD", item["id"], payload["method"], item["credit"])
        checks.require("LIVE_H_NEVER_RESET", item["id"], torch.equal(record["start_h"], previous_h))
        for name, value in buffers.items():
            checks.require("FIXED_PUBLIC_BUFFER", name, torch.equal(record["buffers"][name], value))
        source_hash = newborn.birth["initial_parameter_hash"] if index == 0 else trainer.trace[index - 1]["parameter_hash"]
        checks.equal("PREPARAMETER_IS_PREVIOUS_REAL_BOUNDARY", item["id"], state_digest({**record["parameters"], **record["buffers"]}), source_hash)
        checks.equal("LIVE_CREDIT_RECORD", item["id"], ledger.content_hash(record), point["credit"]["record_hash"])
        checks.equal("LIVE_CREDIT_SNAPSHOT", item["id"], ledger.content_hash(record["parameters"]), point["credit"]["snapshot_hash"])
        checks.equal("LIVE_TRAJECTORY_CHAIN", item["id"], payload["trace_binding"]["trajectory_hash"], point["credit"]["trajectory_hash"])
        checks.equal("LIVE_ACTION_TRACE", item["id"], payload["trace_binding"]["action_chunk_hash"], point["action_chunk_hash"])
        checks.equal("LIVE_DATA_TRACE", item["id"], payload["trace_binding"]["data_chunk_hash"], point["data_chunk_hash"])
        if record["end_tick"] in tape_hashes:
            checks.equal("FINAL_RECORDED_TAPE", item["id"], ledger.content_hash(record), tape_hashes[record["end_tick"]])
        matrix = _matrix(record["parameters"], record["buffers"], item["mode"], trainer.config.width)
        expected = None if index == 0 else trainer.trace[index - 1]["physical_matrix"]
        _matrix_checks(matrix, record["parameters"], mask, item["mode"], checks, f"ledger/{index + 1}", expected)
        rolling = hashlib.sha256(bytes.fromhex(rolling) + bytes.fromhex(state_digest(matrix))).hexdigest()
        before_commands = world.family_exposures()["commands"]
        _world_block(world, newborn.model, payload, trainer.config, trainer.schedule, checks, point)
        after_commands = world.family_exposures()["commands"]
        checks.equal("ORIGINAL_BLOCK_COMMAND_COUNTS", item["id"], payload["live"]["family_counts"],
                     {family: after_commands[family] - before_commands[family] for family in range(4)})
        current = copied_block({**{key: record[key] for key in ("observations", "targets", "elapsed", "start_h", "end_h", "start_tick", "end_tick")},
                                **payload["live"]})
        if previous_chunk is None:
            auxiliary, source, warmup = copied_block(current), "startup", 0
        elif reservoir.eligible_count(current["start_tick"]):
            auxiliary, source, warmup = reservoir.sample(current["start_tick"]), "history_replay", trainer.config.chunk_ticks
        else:
            auxiliary, source, warmup = combine_blocks(previous_chunk, current), "fallback_recent", trainer.config.chunk_ticks
        checks.equal("EXACT_AUXILIARY_ORIGINAL_HISTORY", item["id"],
            (point["auxiliary"]["source"], point["auxiliary"]["source_hash"], point["auxiliary"]["warmup_ticks"]),
            (source, block_hash(auxiliary), warmup))
        reservoir.add_chunk(current)
        previous_chunk = current
        end = record["end_tick"]
        if end in anchors:
            checks.equal("LEDGER_CANONICAL_H", item["id"], state_digest(record["end_h"]), anchors[end]["hidden_hash"])
            checks.equal("LEDGER_CANONICAL_WORLD", item["id"], state_digest(world.state_dict()), anchors[end]["world_hash"])
            checks.equal("RECONSTRUCTED_CANONICAL_REPLAY_POLICY", item["id"], state_digest(reservoir.state_dict()), anchors[end]["replay_hash"])
        event = trainer.operator.update_history[index]
        mask[event["birth_indices"]] = True
        mask[event["retire_indices"]] = False
        previous_h = record["end_h"]
    checks.require("LAST_REAL_H", item["id"], torch.equal(previous_h, trainer.h.cpu()))
    checks.require("LAST_REAL_MASK", item["id"], torch.equal(mask, trainer.operator.mask))
    checks.equal("LAST_REAL_WORLD", item["id"], state_digest(world.state_dict()), state_digest(trainer.world.state_dict()))
    checks.equal("COMPLETE_RESERVOIR_PRIVATE_RNG_POLICY", item["id"], state_digest(reservoir.state_dict()), state_digest(trainer.replay.state_dict()))
    return dict(records=len(entries), matrix_hash_chain=rolling, world_teacher_replayed_ticks=trainer.ticks,
        binding_hash=inventory["binding_hash"], manifest_sha256=inventory["manifest_sha256"],
        scope="Every original observation/causal label/context/dt and complete world replayed; actual preparameters reconstruct neural state/argmax actions. No parameter update or gradient recomputed.")


def _evaluation_record(record, anchor, trainer, plan, item, spec, checks):
    where = f"{item['id']}/{spec}"
    checks.equal("ACTUAL_OBSERVER_FORMAT", where, record["format"], observer.FORMAT_VERSION)
    checks.equal("AUTONOMOUS_ARGMAX", where, (record["intervention"], record["action_policy"]), ("none", "model"))
    checks.equal("FROZEN_ACTUAL_MODEL", where, record["model"], anchor["model_config"])
    checks.equal("TRUE_OBSERVATION_WIDTH", where, record["model"]["input_size"], trainer.config.input_size)
    checks.equal("TRUE_JOB_COUNT", where, record["n_jobs"], trainer.config.n_jobs)
    checks.equal("ACTUAL_ACTOR_IDENTITY", where, record["actor_ids"], list(trainer.world.actor_ids))
    checks.equal("ACTUAL_CASE_IDENTITY", where, record["case_ids"], list(trainer.world.case_ids))
    checks.equal("PROBE_SPEC", where, record["probe"], spec)
    for key in ("family", "condition", "steps"):
        checks.equal("PROBE_FIELD", where, record[key], spec[key])
    checks.equal("LABEL_RULE", where, record["family_mapping"], list(FAMILY_MAPPINGS[spec["family"]]))
    checks.equal("FUTURE_FAMILY", where, record["context_policy"], [spec["family"]])
    for name, expected in (("parameter_hash", anchor["parameter_hash"]), ("start_hidden_hash", anchor["wrapped_hidden_hash"]),
        ("start_world_state_hash", anchor["world_hash"]), ("start_scheduler_hash", anchor["scheduler_hash"]),
        ("start_event_hash", anchor["event_hash"]), ("source_context_hash", anchor["context_hash"]),
        ("source_continuation_hash", anchor["source_continuation_hash"]), ("fork_tick", anchor["world_tick"]),
        ("fork_time", anchor["world_time"]), ("source_family_exposures", anchor["actual_family_exposures"]),
        ("source_family_recency", anchor["world_family_recency"])):
        checks.equal("FULL_FORK_SOURCE", f"{where}/{name}", record[name], expected)
    checks.equal("ACTUAL_EVALUATOR_SOURCES", where, record["source_hashes"],
        {name: plan["sources"][str((ROOT / name).resolve())] for name in EVALUATOR_SOURCES})
    for key, expected in observer.CONDITIONS[spec["condition"]].items():
        checks.equal("EXPLICIT_EVALUATION_REGIME", where, record["regime"][key], list(expected))
    for key, expected in (("window_range", list(trainer.config.window_range)), ("cancel", trainer.config.cancel), ("replace", trainer.config.replace)):
        checks.equal("PUBLIC_WORLD_RULE", where, record["regime"][key], expected)
    adapted = dict(record, format="integration-own-trace-v2", condition=f"undirected/{record['condition']}")
    legacy_ledger_audit(adapted, checks, where, dict(evaluation_format="integration-own-trace-v2", eval_steps=spec["steps"],
        dt=plan["dt"], batch_size=trainer.config.batch_size, n_jobs=trainer.config.n_jobs))
    for cohort in ("primary", "inherited", "carryover_contract"):
        for key in ("matured", "completed", "clean_completed"):
            checks.equal("FAMILY_COHORT_PARTITION", where,
                sum(record["family_cohorts"][str(family)][cohort][key] for family in range(4)), record[cohort][key])
            if cohort == "primary":
                checks.equal("PRIMARY_ONLY_REQUESTED_RULE", where,
                    record["family_cohorts"][str(spec["family"])][cohort][key], record[cohort][key])
    checks.require("ONLY_REQUESTED_FUTURE_NEW_FAMILY", where,
        all(value == 0 for family, value in enumerate(record["actual_family_exposures"]["new"]) if family != spec["family"]))
    metrics = _metrics(record)
    detail = record["request_diagnosis"]["primary"]
    metrics.update(uncompleted_no_begin=detail["uncompleted_no_begin"] / detail["requests"] if detail["requests"] else None,
        uncompleted_despite_legal=detail["uncompleted_despite_legal_opportunity"] / detail["requests"] if detail["requests"] else None)
    return metrics


def _sensitivity(path, terminal_path, terminal, checks, inputs):
    if path is None:
        if not terminal["ledger"]["records"]:
            return dict(status="not_applicable_no_successful_live_blocks", calibrated=False, scope="No successful live block or saved S exists.")
        if not terminal["branch"]["credit"].startswith("rtrl"):
            return dict(status="not_required_method_has_no_full_S", calibrated=False,
                        scope="Finite-window method has no full saved S; no projection calibration is claimed.")
        return dict(status="pending_independent_jvp", calibrated=False, scope="No independent numerical calibration supplied.")
    value = _json(_input(path, inputs))
    from experiments.undirected import sensitivity_audit as independent
    checks.equal("JVP_FORMAT", str(path), value.get("format"), independent.FORMAT_VERSION)
    checks.equal("JVP_TERMINAL_PATH", str(path), value.get("terminal_path"), str(Path(terminal_path).resolve()))
    checks.equal("JVP_TERMINAL_BYTES", str(path), value.get("terminal_sha256"), file_sha256(terminal_path))
    checks.equal("JVP_METHOD", str(path), value.get("method"), terminal["branch"]["credit"])
    checks.equal("JVP_MANIFEST", str(path), value.get("manifest_sha256"), terminal["ledger"]["manifest_sha256"])
    checks.equal("JVP_RECORD_INVENTORY", str(path), value.get("record_count"), terminal["ledger"]["records"])
    checks.equal("JVP_LEDGER_BINDING", str(path), value.get("binding_hash"), terminal["ledger"]["binding_hash"])
    checks.equal("JVP_FIXED_CALIBRATION", str(path), (value.get("directions"), value.get("seed"), value.get("atol"), value.get("rtol")),
                 (independent.DIRECTIONS, independent.AUDIT_SEED, independent.ATOL, independent.RTOL))
    checks.require("JVP_INTEGER_METADATA", str(path), type(value.get("directions")) is int and type(value.get("seed")) is int
                   and type(value.get("record_count")) is int)
    checks.require("JVP_EXPECTED_BINDING_CHECKED", str(path), value.get("expected_binding_verified") is True)
    checks.require("JVP_TERMINAL_OUTCOME", str(path), value.get("terminal_status") == terminal["status"]
                   and value.get("endpoint_comparable") is terminal["endpoint_comparable"])
    for source, digest in value.get("analysis_sources", {}).items():
        _input(source, inputs, digest)
    checks.require("JVP_ANALYSIS_CODE_BOUND", str(path), str((Path(__file__).parent / "sensitivity_audit.py").resolve()) in value.get("analysis_sources", {})
                   or value.get("status") == "not_applicable_no_successful_live_blocks")
    for source, digest in value.get("input_fingerprints", {}).items():
        _input(source, inputs, digest)
    required = {str(Path(terminal_path).resolve()): file_sha256(terminal_path),
        terminal["checkpoint_path"]: terminal["checkpoint_sha256"],
        terminal["ledger"]["binding_path"]: terminal["ledger"]["binding_sha256"],
        terminal["ledger"]["manifest_path"]: terminal["ledger"]["manifest_sha256"]}
    if terminal["ledger"]["records"]:
        required.update({row["path"]: row["sha256"] for row in terminal["checkpoints"].values()})
    ledger_folder = Path(terminal["ledger"]["manifest_path"]).parent
    inventory = ledger.read_ledger(ledger_folder, verify_files=False)
    has_saved_s = terminal["branch"]["credit"] in ("rtrl", "rtrl_reset")
    required.update({str((ledger_folder / row["file"]).resolve()): row["sha256"] for row in inventory["entries"]})
    expected = {int(tick) for tick in terminal["checkpoints"] if int(tick)}
    if terminal["last_tick"]:
        expected.add(terminal["last_tick"])
    def comparisons_ok(rows, count=independent.DIRECTIONS):
        return isinstance(rows, list) and len(rows) == count and all(
            type(row.get("max_tolerance_ratio")) in (int, float) and math.isfinite(row["max_tolerance_ratio"])
            and 0 <= row["max_tolerance_ratio"] <= 1 for row in rows)
    if inventory["entries"]:
        first = ledger.load_record(ledger_folder, inventory["entries"][0], inventory["binding_hash"])
        dynamic_names = first["dynamic_names"]
        dynamic_count = sum(first["record"]["parameters"][name].numel() for name in dynamic_names)
        generator = torch.Generator(device="cpu").manual_seed(independent.AUDIT_SEED)
        directions = (torch.randint(2, (independent.DIRECTIONS, dynamic_count), generator=generator, device="cpu") * 2 - 1).to(torch.float64)
        directions /= math.sqrt(dynamic_count)
        checks.equal("JVP_ACTUAL_DIRECTION_LAYOUT", str(path),
            (value.get("dynamic_names"), value.get("dynamic_parameters"), value.get("direction_hash")),
            (dynamic_names, dynamic_count, ledger.content_hash(directions)))
        analyses = {str((Path(__file__).parent / name).resolve()): file_sha256(Path(__file__).parent / name)
                    for name in ("sensitivity_audit.py", "ledger.py", "model.py")}
        checks.equal("JVP_COMPLETE_ANALYSIS_SOURCE_SET", str(path), value.get("analysis_sources"), analyses)
        required.update(analyses)
        for key, root in (("source_hashes", ROOT / "experiments/undirected"), ("dependency_hashes", ROOT / "experiments")):
            for name in first[key]:
                source = (root / name).resolve()
                checks.require("JVP_RUNTIME_SOURCE_PATH", str(path), source.is_relative_to(root.resolve()))
                required[str(source)] = file_sha256(source)
        checks.equal("JVP_CPU_F64", str(path), (value.get("dtype"), value.get("device"), value.get("cpu_threads")), ("float64", "cpu", 1))
        checks.require("JVP_CPU_THREAD_TYPE", str(path), type(value.get("cpu_threads")) is int)
        checks.equal("JVP_RECORD_ROWS", str(path), len(value.get("records", [])), len(inventory["entries"]))
        checks.require("JVP_INTEGER_WORK_COUNTS", str(path), isinstance(value.get("work"), dict)
                       and all(type(count) is int and count >= 0 for count in value["work"].values()))
        checks.equal("JVP_FULL_RECORDED_WORK", str(path), value.get("work"),
            dict(jvp_calls=independent.DIRECTIONS * len(inventory["entries"]) if has_saved_s else 0,
                 model_step_calls=(independent.DIRECTIONS if has_saved_s else 1) * terminal["last_tick"],
                 credit_engine_jacobian_calls=0, optimizer_steps=0, world_steps=0))
        for actual, entry in zip(value["records"], inventory["entries"]):
            for name in ("update_age", "start_tick", "end_tick", "content_hash", "sha256"):
                checks.equal("JVP_RECORD_SOURCE", str(path), actual[name], entry[name])
            checks.require("JVP_RESET_CONVENTION", str(path), actual.get("tangent_reset") is (terminal["branch"]["credit"] == "rtrl_reset"))
            checks.require("JVP_PRIMAL_NUMERICAL_COMPARISON", str(path),
                           comparisons_ok(actual.get("primal_comparisons"), independent.DIRECTIONS if has_saved_s else 1))
            checks.require("METHOD_SCOPED_AUDIT_TANGENTS", str(path), isinstance(actual.get("tangent_norms"), list)
                           and len(actual["tangent_norms"]) == (independent.DIRECTIONS if has_saved_s else 0))
        references = {int(tick): row for tick, row in terminal["checkpoints"].items()}
        references.setdefault(terminal["last_tick"], dict(path=terminal["checkpoint_path"], sha256=terminal["checkpoint_sha256"]))
        rows = value.get("checkpoints", [])
        ticks = [row.get("tick") for row in rows]
        checks.require("JVP_EXACT_CHECKPOINT_INVENTORY", str(path), type(value.get("checkpoint_count")) is int
            and value["checkpoint_count"] == len(rows) == len(references) and all(type(tick) is int for tick in ticks)
            and len(set(ticks)) == len(ticks) and set(ticks) == set(references))
        for row in rows:
            reference = references[row["tick"]]
            checks.equal("JVP_REFERENCE_CHECKPOINT_BYTES", str(path), (row.get("path"), row.get("sha256")), (reference["path"], reference["sha256"]))
            checks.require("JVP_CHECKPOINT_AGE_AND_SUPPLIED_SHA", str(path), type(row.get("update_age")) is int
                and row["update_age"] == row["tick"] // terminal["branch"]["native_config"]["chunk_ticks"]
                and row.get("supplied_sha256_verified") is True)
            if row["tick"] == 0:
                checks.equal("JVP_NEWBORN_NOT_CALIBRATION", str(path), (row.get("sensitivity_check"), row.get("projection_comparisons")), ("newborn_zero", []))
            elif terminal["branch"]["credit"].startswith("rtrl"):
                checks.require("JVP_ACTUAL_PROJECTION_COMPARISONS", str(path), row.get("sensitivity_check") == "projected_full_S"
                               and comparisons_ok(row.get("projection_comparisons")))
            else:
                checks.equal("JVP_UNSAVED_S_SKIPPED", str(path), (row.get("sensitivity_check"), row.get("projection_comparisons")),
                             ("skipped_method_has_no_full_S", []))
    checks.require("JVP_ALL_INPUT_BYTES_BOUND", str(path), all(value.get("input_fingerprints", {}).get(source) == digest for source, digest in required.items()))
    if expected and terminal["branch"]["credit"].startswith("rtrl"):
        checks.require("JVP_PROJECTED_S_SCOPE", str(path), value.get("status") == "passed" and value.get("sensitivity_calibrated") is True
                       and value.get("validation_scope") == "primal_history_and_projected_saved_S")
        checks.equal("JVP_ALL_S_REFERENCE_TICKS", str(path), set(value.get("sensitivity_checked_ticks", [])), expected)
        checks.equal("JVP_TWO_FIXED_DIRECTIONS", str(path), value.get("directions"), 2)
    elif not expected:
        checks.equal("NO_FALSE_EMPTY_HISTORY_CALIBRATION", str(path), (value.get("status"), value.get("sensitivity_calibrated")),
                     ("not_applicable_no_successful_live_blocks", False))
    elif not terminal["branch"]["credit"].startswith("rtrl"):
        checks.equal("NO_UNSAVED_S_CALIBRATION", str(path), (value.get("status"), value.get("sensitivity_calibrated"), value.get("sensitivity_checked_ticks")),
                     ("passed", False, []))
        checks.equal("JVP_PRIMAL_ONLY_SCOPE", str(path), value.get("validation_scope"), "primal_history_only_no_saved_S_comparison")
        checks.equal("JVP_PRIMAL_ONLY_SKIP_TICKS", str(path), set(value.get("sensitivity_skipped_ticks", [])), expected)
    scope = ("Two fixed random projections numerically calibrated; no all-entry/full-gradient proof." if value.get("sensitivity_calibrated") else
             "Independent recorded primal/history reconstruction only; zero JVP, no tangents propagated, and no full saved S compared." if inventory["entries"] else
             "No successful live history exists; no numerical calibration is claimed.")
    return dict(status=value["status"], calibrated=value.get("sensitivity_calibrated", False), path=str(Path(path).resolve()),
        sha256=inputs[str(Path(path).resolve())], scope=scope)


def audit_branch(protocol, identifier, *, mechanical=False, sensitivity=None):
    checks, inputs = Checks(), {}
    with _preserve_process_state(), torch.no_grad():
        plan = _plan(protocol, mechanical)
        plan, item, sha = study.read_plan(protocol, identifier)
        _input(protocol, inputs, sha)
        for path, digest in plan["sources"].items():
            _input(path, inputs, digest)
        out = Path(item["output"])
        terminal_path = _input(out / "terminal.json", inputs)
        # The frozen completion helper checks legitimate terminal status,
        # immutable index, full last-valid state and complete ledger inventory.
        terminal = study.completion(plan, item, sha)
        _input(out / "snapshots.json", inputs)
        final_path = _input(terminal["checkpoint_path"], inputs, terminal["checkpoint_sha256"])
        final, final_row = _checkpoint(final_path, plan, item, terminal["last_tick"], checks, inputs, graph=True)
        graphs = [dict(final_row.pop("_graph_payload"), role="last_valid")]
        checks.equal("TERMINAL_FULL_STATE", identifier, final_row["semantic_hash"], terminal["checkpoint_semantic_hash"])
        checks.equal("TERMINAL_WORK", identifier, terminal["work"], final.work)
        expected_ticks = {tick for tick in plan["checkpoint_ticks"] if tick <= terminal["last_tick"]}
        checks.equal("ALL_CANONICAL_SNAPSHOTS", identifier, {int(tick) for tick in terminal["checkpoints"]}, expected_ticks)
        rows, anchors, newborn = [], {}, None
        for tick in sorted(expected_ticks):
            entry = terminal["checkpoints"][str(tick)]
            _input(entry["path"], inputs, entry["sha256"])
            trainer, row = _checkpoint(entry["path"], plan, item, tick, checks, inputs, graph=tick == 0)
            checks.equal("CANONICAL_TRACE_IS_ORIGINAL_PREFIX", identifier, state_digest(trainer.trace),
                         state_digest(final.trace[:tick // final.config.chunk_ticks]))
            checks.equal("CANONICAL_STRUCTURE_IS_ORIGINAL_PREFIX", identifier, trainer.operator.update_history,
                         final.operator.update_history[:tick // final.config.chunk_ticks])
            if tick == 0:
                graphs.append(dict(row.pop("_graph_payload"), role="birth"))
            checks.equal("CANONICAL_SEMANTIC_HASH", identifier, row["semantic_hash"], entry["semantic_hash"])
            if tick == terminal["last_tick"]:
                checks.equal("LATEST_EQUALS_CANONICAL_SAME_TICK", identifier, final_row["semantic_hash"], row["semantic_hash"])
                # Plotters follow the immutable canonical anchor when it
                # exists; latest may serialize identical state to other bytes.
                graphs[0].update(checkpoint=row["checkpoint"], checkpoint_sha256=row["checkpoint_sha256"], matrix_hash=row["matrix_hash"])
            anchors[tick], rows = row, rows + [row]
            if tick == 0:
                newborn = trainer
        anchors.setdefault(terminal["last_tick"], final_row)
        if terminal["last_tick"] not in expected_ticks:
            rows.append(dict(final_row, noncanonical_last_valid=True))
        if terminal["status"] == "numerical_failure":
            failure = _json(_input(terminal["failure"]["path"], inputs, terminal["failure"]["sha256"]))
            checks.equal("SCIENTIFIC_FAILURE_KIND", identifier, (failure.get("kind"), failure.get("stage"), failure.get("exception_type")),
                         ("numerical_failure", "learning_update", "FloatingPointError"))
            checks.equal("FAILURE_LAST_VALID_BYTES", identifier, failure.get("last_complete_checkpoint_sha256"), inputs[str(final_path)])
            failed_tick, declared = failure.get("failed_tick"), failure.get("declared_target_tick")
            checks.require("NUMERICAL_FAILURE_CAUSAL_BUDGET", identifier, type(failed_tick) is int and type(declared) is int
                and terminal["last_tick"] <= failed_tick <= min(terminal["last_tick"] + final.config.chunk_ticks, declared)
                and failed_tick <= declared <= plan["target_ticks"] and declared % final.config.chunk_ticks == 0)
        ledger_result = _ledger_audit(final, newborn, plan, item, sha, checks, inputs, anchors)
        value_path = _input(Path(plan["raw_root"]) / "evaluation" / f"{identifier}.json", inputs)
        value = study._checked_evaluation(_json(value_path), plan, item, sha, terminal)
        evaluations, evaluation_trace = [], []
        for record, spec in zip(value["records"], plan["evaluation_specs"]):
            metrics = _evaluation_record(record, anchors[spec["tick"]], final, plan, item, spec, checks)
            evaluations.append(dict(id=identifier, branch_id=identifier, checkpoint_tick=spec["tick"], snapshot_role=anchors[spec["tick"]]["snapshot_role"],
                seed=item["seed"], **{key: item[key] for key in study.KEYS}, **spec,
                status="success", primary_matured=record["primary"]["matured"], primary_completed=record["primary"]["completed"],
                primary_clean_completed=record["primary"]["clean_completed"], **metrics))
            # These are the saved autonomous fork observations, distinct from
            # the training trace.  Do not infer fork equality from training
            # action hashes, or infer equal counters from equal fork actions.
            evaluation_trace.append(dict(**spec, action_hash=record["action_hash"],
                end_world_state_hash=record["end_world_state_hash"], end_hidden_hash=record["end_hidden_hash"],
                post_fork_counters=record["post_fork_counters"], opportunities=record["opportunities"],
                request_diagnosis=record["request_diagnosis"],
                content_by_cohort_phase_age=record["content_by_cohort_phase_age"], inactive_history=record["inactive_history"],
                primary={key: record["primary"][key] for key in ("matured", "completed", "clean_completed")}))
        jvp = _sensitivity(sensitivity, terminal_path, terminal, checks, inputs)
        trace = [dict(tick=p["tick"], parameter_hash=p["parameter_hash"], optimizer_hash=p["optimizer_hash"],
            live_hidden_hash=p["live_hidden_hash"], live_world_hash=p["live_world_hash"], event_hash=p["event_hash"],
            action_hash=p["action_hash"], data_hash=p["data_hash"], actual_family_exposures=p["actual_family_exposures"],
            credit_gradient_hash=p["credit"]["gradient_hash"], physical_matrix=p["physical_matrix"]) for p in final.trace]
        probes = []
        for p in final.trace:
            probe = p["update_metrics"]["structure"].get("probe")
            if probe is not None:
                probes.append(dict(id=identifier, seed=item["seed"], structure=item["structure"], tick=p["tick"],
                    birth_accepted=probe["birth_accepted"], fade_accepted=probe["fade_accepted"],
                    birth_history_damage=probe["birth_damage"]["history"], fade_history_damage=probe["half_fade_damage"]["history"],
                    full_ablation_history_damage=probe["full_ablation_damage"]["history"]))
        birth_accepted = [probe for probe in probes if probe["birth_accepted"]]
        fade_accepted = [probe for probe in probes if probe["fade_accepted"]]
        branch = dict(id=identifier, branch_id=identifier, seed=item["seed"], **{key: item[key] for key in study.KEYS}, status=terminal["status"],
            endpoint_comparable=terminal["endpoint_comparable"], last_tick=terminal["last_tick"], target_ticks=plan["target_ticks"],
            canonical_checkpoints=len(expected_ticks), evaluation_forks=len(evaluations), **ledger_result,
            total_parameters=final_row["total_parameters"], dynamic_parameters=final_row["dynamic_parameters"],
            persistent_state_scalars=final_row["persistent_state_scalars"], sensitivity_bytes=final_row["sensitivity_bytes"],
            work=final.work, structure_summary=final.operator.summary(), sensitivity_audit=jvp,
            functional_damage_summary=dict(probe_batches=len(probes), accepted_birth_batches=len(birth_accepted),
                accepted_fade_batches=len(fade_accepted),
                accepted_birth_history_worse=sum(p["birth_history_damage"] > 0 for p in birth_accepted),
                accepted_fade_history_worse=sum(p["fade_history_damage"] > 0 for p in fade_accepted),
                full_ablation_history_worse=sum(p["full_ablation_history_damage"] > 0 for p in probes),
                scope="Batchwise training live/history loss diagnostics, not independent task-retention guarantees"),
            last_valid_semantic_hash=final_row["semantic_hash"], terminal_sha256=inputs[str(terminal_path)])
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise AuditError(f"input changed during branch audit: {path}")
    return dict(branch=branch, checkpoints=rows, per_seed=evaluations, evaluation_trace=evaluation_trace,
                probes=probes, trace=trace, graphs=graphs,
                inputs=inputs, checks=dict(checks.counts))


def summarize_declared(rows):
    if (any(type(row["seed"]) is not int or row["status"] not in ("success", "numerical_failure") for row in rows)
            or sorted(row["seed"] for row in rows) != list(study.SEEDS)):
        raise AuditError("statistics require all three declared seeds exactly once")
    failed = sum(row["status"] != "success" for row in rows)
    defined = [row["value"] for row in rows if row["value"] is not None]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in defined):
        raise AuditError("nonfinite per-seed metric")
    complete = failed == 0 and len(defined) == 3
    return dict(n_total=3, n_success=3 - failed, failure_count=failed, n_defined=len(defined),
        mean=statistics.mean(defined) if complete else None, sample_sd=statistics.stdev(defined) if complete else None,
        per_seed=rows, summary_status="three_successful_seeds" if complete else "failed_seed_or_undefined_denominator")


def _tables(branches, values, plan):
    outcomes = {(_key(row), row["seed"]): row for row in branches}
    lookup = {(_key(row), row["seed"], row["tick"], row["family"], row["condition"]): row for row in values}
    summaries, contrasts = [], []
    for v in map(_key, study.VARIANTS):
        for spec in plan["evaluation_specs"]:
            for metric in METRICS:
                rows = []
                for seed in study.SEEDS:
                    outcome = outcomes[(v, seed)]
                    entry = lookup.get((v, seed, spec["tick"], spec["family"], spec["condition"]))
                    rows.append(dict(seed=seed, status=outcome["status"], value=None if entry is None else entry[metric]))
                summaries.append(dict(**dict(zip(study.KEYS, v)), **spec, metric=metric, **summarize_declared(rows)))
    for name, left, right in CONTRASTS:
        for spec in plan["evaluation_specs"]:
            for metric in METRICS:
                rows = []
                for seed in study.SEEDS:
                    a = lookup.get((left, seed, spec["tick"], spec["family"], spec["condition"]))
                    b = lookup.get((right, seed, spec["tick"], spec["family"], spec["condition"]))
                    okay = outcomes[(left, seed)]["status"] == outcomes[(right, seed)]["status"] == "success"
                    value = None if a is None or b is None or a[metric] is None or b[metric] is None else a[metric] - b[metric]
                    rows.append(dict(seed=seed, status="success" if okay else "numerical_failure", value=value,
                                     left_status=outcomes[(left, seed)]["status"], right_status=outcomes[(right, seed)]["status"]))
                contrasts.append(dict(contrast=name, left=list(left), right=list(right), **spec, metric=metric, **summarize_declared(rows)))
    return summaries, contrasts


def _family_tables(branches, values, plan):
    outcomes = {(_key(row), row["seed"]): row for row in branches}
    lookup = {(_key(row), row["seed"], row["tick"], row["family"], row["condition"]): row for row in values}
    specs = sorted({(spec["tick"], spec["condition"], spec["steps"]) for spec in plan["evaluation_specs"]})
    individuals, summaries, contrasts, averaged = [], [], [], {}
    for v in map(_key, study.VARIANTS):
        for seed in study.SEEDS:
            outcome = outcomes[(v, seed)]
            for tick, condition, steps in specs:
                family_rows = [lookup.get((v, seed, tick, family, condition)) for family in range(4)]
                metrics, defined = {}, {}
                for metric in METRICS:
                    numbers = [row[metric] for row in family_rows if row is not None and row[metric] is not None]
                    defined[metric] = len(numbers)
                    metrics[metric] = statistics.mean(numbers) if outcome["status"] == "success" and len(numbers) == 4 else None
                row = dict(id=outcome["id"], branch_id=outcome["id"], seed=seed, **dict(zip(study.KEYS, v)),
                    tick=tick, checkpoint_tick=tick, condition=condition, steps=steps, status=outcome["status"],
                    families_present=sum(row is not None for row in family_rows), defined_family_counts=defined,
                    primary_matured=sum(row["primary_matured"] for row in family_rows) if all(row is not None for row in family_rows) else None,
                    aggregation="Equal-weight mean of four family rates within one trained seed, never a pooled request rate", **metrics)
                individuals.append(row)
                averaged[(v, seed, tick, condition)] = row
        for tick, condition, steps in specs:
            for metric in METRICS:
                rows = [dict(seed=seed, status=outcomes[(v, seed)]["status"],
                    value=averaged[(v, seed, tick, condition)][metric]) for seed in study.SEEDS]
                summaries.append(dict(**dict(zip(study.KEYS, v)), tick=tick, checkpoint_tick=tick,
                    condition=condition, steps=steps, metric=metric, **summarize_declared(rows)))
    for name, left, right in CONTRASTS:
        for tick, condition, steps in specs:
            for metric in METRICS:
                rows = []
                for seed in study.SEEDS:
                    a, b = averaged[(left, seed, tick, condition)], averaged[(right, seed, tick, condition)]
                    rows.append(dict(seed=seed, status="success" if a["status"] == b["status"] == "success" else "numerical_failure",
                        value=None if a[metric] is None or b[metric] is None else a[metric] - b[metric],
                        left_status=a["status"], right_status=b["status"]))
                contrasts.append(dict(contrast=name, left=list(left), right=list(right), tick=tick,
                    checkpoint_tick=tick, condition=condition, steps=steps, metric=metric, **summarize_declared(rows)))
    return individuals, summaries, contrasts


def _numeric_difference_kinds(left, right):
    """Classify unequal JSON leaves; counts and floating diagnostics stay distinct."""
    missing = object()
    if isinstance(left, dict) and isinstance(right, dict):
        pairs = ((left.get(key, missing), right.get(key, missing)) for key in set(left) | set(right))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        pairs = ((left[i] if i < len(left) else missing, right[i] if i < len(right) else missing)
                 for i in range(max(len(left), len(right))))
    else:
        return {"integer" if type(value) is int else "float" for value in (left, right)
                if type(value) in (int, float)} if left != right else set()
    result = set()
    for a, b in pairs:
        result.update(_numeric_difference_kinds(a, b))
        if len(result) == 2:
            break
    return result


def _exact_record_comparison(left, right, identifiers, groups, *, numeric=False):
    """Retain exact whole-record equality as an outcome, without a tolerance.

    Each branch's provenance and internal consistency is audited separately.
    Differences between two legitimate policies are empirical observations,
    including different actions, worlds, record inventories, or missing fields.
    """
    def indexed(rows):
        values = {tuple(row[key] for key in identifiers): row for row in rows}
        if len(values) != len(rows):
            raise AuditError("duplicate between-policy comparison record identity")
        return values
    a, b = indexed(left), indexed(right)
    keys = sorted(set(a) | set(b))
    fields = sorted(set().union(*(set(row) for row in (*left, *right))) - set(identifiers)) if keys else []
    missing = object()
    def different(key, field=None):
        x, y = a.get(key, missing), b.get(key, missing)
        if field is not None:
            x = x.get(field, missing) if x is not missing else missing
            y = y.get(field, missing) if y is not missing else missing
        return x != y
    def identity(key):
        return dict(zip(identifiers, key))
    details = {}
    for field in fields:
        mismatched = [key for key in keys if different(key, field)]
        first = mismatched[0] if mismatched else None
        group = next((name for name, names in groups.items() if field in names), "other")
        details[field] = dict(group=group, mismatch_count=len(mismatched),
            first_mismatch=identity(first) if first is not None else None,
            first_mismatch_tick=first[0] if first is not None else None)
        if first is not None:
            difference = {name: row.get(field, {"missing": True})
                for name, row in (("fixed", a.get(first, {})), ("fixed_probe", b.get(first, {})))}
            details[field]["first_difference"] = difference if len(json.dumps(difference)) < 4096 else dict(
                value_hashes={name: state_digest(value) for name, value in difference.items()},
                scope="Full values remain in the original SHA-bound evaluation input; no approximate comparison.")
    mismatched = [key for key in keys if different(key)]
    first = mismatched[0] if mismatched else None
    group_results = {}
    for name in (*groups, "other"):
        names = [field for field in fields if details[field]["group"] == name]
        differences = [key for key in keys if any(different(key, field) for field in names)]
        group_results[name] = dict(status="different" if differences else "equal",
            fields=names, mismatch_count=len(differences),
            first_mismatch_tick=differences[0][0] if differences else None)
    result = dict(status="failed" if mismatched else "passed", compared_records=len(keys),
        fixed_records=len(left), fixed_probe_records=len(right), mismatch_count=len(mismatched),
        first_mismatch=identity(first) if first is not None else None,
        first_mismatch_tick=first[0] if first is not None else None,
        record_inventory_equal=set(a) == set(b), fields=details, groups=group_results)
    if numeric:
        kinds = {key: _numeric_difference_kinds(a.get(key, {}), b.get(key, {})) for key in keys}
        result["numeric_leaf_differences"] = {kind: dict(
            record_count=sum(kind in values for values in kinds.values()),
            first_mismatch_tick=next((key[0] for key, values in kinds.items() if kind in values), None))
            for kind in ("integer", "float")}
    return result


def _fixed_probe_comparison(results):
    by_variant = {(_key(r["branch"]), r["branch"]["seed"]): r for r in results}
    seeds = []
    for seed in study.SEEDS:
        a, b = by_variant[(variant(), seed)], by_variant[(variant(structure="fixed_probe"), seed)]
        training = _exact_record_comparison(a["trace"], b["trace"], ("tick",), TRACE_FIELD_GROUPS)
        evaluation = _exact_record_comparison(a["evaluation_trace"], b["evaluation_trace"],
            ("tick", "family", "condition", "steps"), dict(
                actions=("action_hash",), learning_state=("end_hidden_hash",),
                world_and_behavior_counts=("end_world_state_hash", "post_fork_counters", "opportunities", "primary"),
                request_and_content_diagnostics=("request_diagnosis", "content_by_cohort_phase_age", "inactive_history")),
            numeric=True)
        seeds.append(dict(seed=seed, fixed_branch=a["branch"]["id"], fixed_probe_branch=b["branch"]["id"],
            fixed_terminal_status=a["branch"]["status"], fixed_probe_terminal_status=b["branch"]["status"],
            training_trace=training, frozen_evaluation=evaluation))
    return dict(check_code=FIXED_PROBE_CHECK,
        status="failed" if any(row["training_trace"]["status"] == "failed" for row in seeds) else "passed",
        frozen_evaluation_status="different" if any(row["frozen_evaluation"]["status"] == "failed" for row in seeds) else "equal",
        exact_rule="Original whole training trace records compared exactly, without numerical tolerances; failed equality remains failed.",
        scope="Between-policy empirical comparison, separate from source, per-branch consistency, and count/rate audit certification. Empty histories prove no endpoint equivalence.",
        causal_attribution="Differences are observed; their cause is not established. Equal training actions do not imply equal frozen-fork actions or counters.",
        per_seed=seeds)


def _parent_checks(results, plan, checks):
    checks.equal("ALL_42_DECLARED_BRANCHES_ACCOUNTED", "study", {r["branch"]["id"] for r in results}, {b["id"] for b in plan["branches"]})
    checks.equal("ONE_RESULT_PER_BRANCH", "study", len(results), 42)
    external = {}
    for result in results:
        item = result["branch"]
        for point in result["trace"]:
            key = item["seed"], item["course"], point["tick"]
            actual = point["event_hash"], point["actual_family_exposures"]
            if key in external:
                checks.equal("MATCHED_REAL_EXTERNAL_COMMAND_STREAM", str(key), actual, external[key])
            external[key] = actual
    return _fixed_probe_comparison(results)


def _worker(job):
    return audit_branch(**job)


def _manifests(protocol, plan, checks, inputs):
    protocol = Path(protocol).resolve()
    sha = file_sha256(protocol)
    for kind in ("training", "evaluation", "baseline"):
        path = _input(protocol.parent / f"{kind}-commands-v1.json", inputs)
        value = _json(path)
        checks.equal("COMMAND_MANIFEST_IDENTITY", str(path),
            (value.get("format"), value.get("kind"), value.get("cwd"), value.get("protocol"), value.get("protocol_sha256")),
            ("undirected-command-plan-v1", kind, str(ROOT), str(protocol), sha))
        checks.equal("COMMAND_MANIFEST_SOURCE_PIN", str(path), value.get("file_hashes"), {**plan["sources"], str(protocol): sha})
        expected = [] if kind == "baseline" else plan["branches"]
        checks.equal("COMPLETE_COMMAND_INVENTORY", str(path), [job.get("branch") for job in value.get("jobs", [])], expected)
        for job, item in zip(value["jobs"], expected):
            tail = ["--protocol", str(protocol), "--branch-id", item["id"]]
            output = Path(item["output"])
            prefix = [str(study.PYTHON.resolve()), "-m", "experiments.undirected.study"]
            if kind == "training":
                command = prefix + ["train"] + tail
                outputs = [str(output / "checkpoint_latest.pt"), str(output / "terminal.json")]
            else:
                destination = Path(plan["raw_root"]) / "evaluation" / f"{item['id']}.json"
                command = prefix + ["evaluate"] + tail + ["--checkpoint", str(output / "checkpoint_latest.pt"), "--output", str(destination)]
                outputs = [str(destination)]
            checks.equal("EXACT_SOURCE_BOUND_COMMAND", str(path), job["command"], command)
            checks.equal("EXACT_BRANCH_OUTPUTS", str(path), job["outputs"], outputs)
            checks.equal("EXACT_BRANCH_TERMINAL", str(path),
                (job["id"], job["branch_id"], job["terminal"], job["protocol_sha256"], job["target_ticks"]),
                (item["id"], item["id"], str(output / "terminal.json"), sha, plan["target_ticks"]))


def build_report(protocol, *, workers=1, mechanical=False, sensitivity_results=None, cached_results=None):
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError("workers must be in 1..6")
    plan = _plan(protocol, mechanical)
    sources = {str(Path(__file__).resolve()): file_sha256(__file__),
               str((ROOT / "experiments/integration/report.py").resolve()): file_sha256(ROOT / "experiments/integration/report.py")}
    supplied = {} if sensitivity_results is None else dict(sensitivity_results)
    if set(supplied) - {item["id"] for item in plan["branches"]}:
        raise AuditError("unknown sensitivity result branch")
    checks, inputs = Checks(), {}
    _manifests(protocol, plan, checks, inputs)
    jobs = [dict(protocol=str(Path(protocol).resolve()), identifier=item["id"], mechanical=mechanical,
                 sensitivity=supplied.get(item["id"])) for item in plan["branches"]]
    if cached_results is not None:
        results = list(cached_results)
        checks.equal("CACHED_BRANCH_RESULT_DECLARED_ORDER", "study", [r["branch"]["id"] for r in results],
                     [item["id"] for item in plan["branches"]])
    elif workers == 1:
        results = [_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_worker, jobs))
    fixed_probe = _parent_checks(results, plan, checks)
    for result in results:
        checks.counts.update(result["checks"])
        for path, digest in result["inputs"].items():
            checks.require("PARENT_INPUT_IDENTITY", path, path not in inputs or inputs[path] == digest)
            inputs[path] = digest
    for path, digest in {**inputs, **sources}.items():
        checks.equal("PARENT_BYTES_UNCHANGED", path, file_sha256(path), digest)
    branches = [result["branch"] for result in results]
    values = [row for result in results for row in result["per_seed"]]
    summaries, contrasts = _tables(branches, values, plan)
    family_rows, family_summary, family_contrasts = _family_tables(branches, values, plan)
    success = sum(row["status"] == "success" for row in branches)
    pending = [row["id"] for row in branches if row["sensitivity_audit"]["status"] == "pending_independent_jvp"]
    prior_failure = None
    previous_log = ROOT / "data/undirected/completion-v2/report-001.log"
    if not mechanical and previous_log.is_file():
        text = previous_log.read_text(encoding="utf-8")
        if f"AuditError: {FIXED_PROBE_CHECK} at 0:" in text:
            _input(previous_log, inputs)
            prior_failure = dict(path=str(previous_log.resolve()), sha256=inputs[str(previous_log.resolve())],
                check_code=FIXED_PROBE_CHECK, status="failed", seed=0, first_mismatch_tick=145024)
    return dict(format=FORMAT, protocol=str(Path(protocol).resolve()), protocol_sha256=file_sha256(protocol),
        mechanical_test=plan["mechanical_test"], counts=dict(declared_branches=42, accounted_branches=len(branches),
            successful_branches=success, numerical_failure_branches=42 - success, declared_forks=1680, actual_forks=len(values)),
        audit=dict(status="passed", scope="provenance_and_accounting",
            branch_evidence="revalidated_branch_audit_cache" if cached_results is not None else "fresh_branch_audits",
            scopes=dict(source_and_branch_internal_consistency="passed", count_rate_accounting="passed",
                fixed_probe_exact_training_equality=fixed_probe["status"],
                fixed_probe_frozen_evaluation_comparison=fixed_probe["frozen_evaluation_status"]),
            note="Passed certifies the named provenance/internal-consistency/count scopes, not the success of an expected between-policy equality.",
            check_count=sum(checks.counts.values()), checks_by_code=dict(checks.counts)),
        scientific_status="completed_with_between_policy_differences" if fixed_probe["status"] == "failed" or fixed_probe["frozen_evaluation_status"] == "different" else "comparisons_completed",
        fixed_probe_comparison=fixed_probe,
        reporting_revision=dict(classification="post_hoc_analysis_only_revision_after_formal_equality_failure",
            change="Preserve the original strict equality outcome as a scientific finding instead of aborting provenance/count reporting; no runtime edits or tolerance relaxation.",
            prior_failed_report=prior_failure),
        primary_tick=plan["native_template"]["main_ticks"], checkpoint_ticks=plan["checkpoint_ticks"],
        native_template=plan["native_template"], variants=plan["variants"], declared_seeds=plan["seeds"],
        public_learning_contract=dict(action_policy="Eleven-way model argmax, without teacher-forced actions",
            teacher="Source-pinned causal midpoint oracle on each actual pre-action world state; supervision is not sparse reward-only learning",
            objective="Weighted cross entropy; equally combined live and auxiliary loss gradients, one Adam step",
            wait_weight=plan["native_template"]["wait_weight"], live_auxiliary_weights=[.5, .5],
            family_mappings=[list(values) for values in FAMILY_MAPPINGS], features_per_job=28,
            observation_normalization=dict(duration_divisor=16., window_divisor=2.5, birth_input_pulse_gain=.25),
            continuity="Course/withdrawal/return alter future NEW draws only; inherited contracts, receipts, h, optimizer and history persist"),
        independent_sensitivity=dict(status="pending_independent_jvp" if pending else "required_projection_checks_accounted",
            pending_branch_ids=pending, calibrated_branches=sum(row["sensitivity_audit"]["calibrated"] for row in branches),
            scope="Only two fixed-direction projection evidence; structural/provenance audit status does not certify full sensitivities or all gradients."),
        branches=branches, checkpoints=[row for result in results for row in result["checkpoints"]],
        per_seed=values, summaries=summaries, paired_contrasts=contrasts,
        family_average_per_seed=family_rows, family_average_summary=family_summary,
        family_average_paired_contrasts=family_contrasts, graphs=[graph for result in results for graph in result["graphs"]],
        functional_probes=[row for result in results for row in result["probes"]], inputs=inputs, analysis_sources=sources,
        definitions=dict(derivative="One common additive parameter offset over the actual recorded trajectory; no derivative through optimizer, structure decisions, argmax actions or world.",
            auxiliary="Equal-weight auxiliary objective; detached one-block warmup and only one 128-tick differentiable suffix in formal runs.",
            statistics="One declared training seed is one repetition. Means/sample SD only for all three successful seeds with defined rates at the same budget.",
            geometry="Edges are unordered pairs. ||W-W.T||F is twice the skew-part Frobenius norm. All potential coordinates consume optimizer/credit budget.",
            compute="Dense N*N field MACs; two fields per Heun substep. Sparse support does not imply sparse compute savings.",
            evaluation="Training primals/actions independently reconstructed. Frozen evaluation forks retain actual model/source bindings and audited count/rate accounting; their full behavioral trajectories are not rerun here.",
            jvp="Optional separately bound independent two-direction projection calibration; neither self-hashes nor these projections prove all full-S entries or all gradients."))


def _csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def write_artifacts(value, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ("branches", "checkpoints", "per_seed", "summaries", "paired_contrasts", "functional_probes",
             "family_average_per_seed", "family_average_summary", "family_average_paired_contrasts")
    for name in names:
        _csv(output / f"{name}.csv", value[name])
    _csv(output / "inputs.csv", [dict(path=path, sha256=digest) for path, digest in value["inputs"].items()])
    graph_path = output / "graphs.json"
    atomic_json(graph_path, dict(format="undirected-audited-graphs-v1", protocol_sha256=value["protocol_sha256"],
        analysis_sources=value["analysis_sources"], graphs=value["graphs"],
        edge_definition="support_edges are [u,v,W[u,v],W[v,u]]; W[i,j] receives from j; eligible_edges are unordered pairs; no downsampling"))
    header = {key: item for key, item in value.items() if key not in {*names, "inputs", "graphs"}}
    header["graph_artifact"] = dict(path=str(graph_path.resolve()), rows=len(value["graphs"]), sha256=file_sha256(graph_path))
    header["tables"] = {name: dict(path=str((output / f"{name}.csv").resolve()), rows=len(value[name]),
        sha256=file_sha256(output / f"{name}.csv")) for name in names}
    header["input_manifest"] = dict(path=str((output / "inputs.csv").resolve()), rows=len(value["inputs"]), sha256=file_sha256(output / "inputs.csv"))
    atomic_json(output / "report.json", header)
    return header


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mechanical", action="store_true")
    parser.add_argument("--sensitivity-manifest", help="JSON mapping declared branch IDs to independent audit JSON paths")
    args = parser.parse_args(argv)
    supplied = None if args.sensitivity_manifest is None else _json(args.sensitivity_manifest)
    result = build_report(args.protocol, workers=args.workers, mechanical=args.mechanical, sensitivity_results=supplied)
    write_artifacts(result, args.output)
    print(json.dumps(dict(counts=result["counts"], audit=result["audit"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
