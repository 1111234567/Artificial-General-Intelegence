"""Native autonomous evaluation of contextual, detached complete-state forks.

The ordinary branch always executes the model's eleven-way argmax. Scorer-only
access to request contents/readiness never enters its observations or hidden
state. The four appended context features are the only new public information.
Every observation preserves raw payload symbols; targets remain scorer-only.

No stream, scheduler, receipt, random generator or numerical hidden state is
reinitialized at the fork. The source continuity implementation stays frozen.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.evaluate import (
    _Cohort, _file_sha256, _new_since_fork, _pending_summary,
)
from experiments.continuity.run import state_digest
from experiments.lifelong.world import CANCEL, NEW, REPLACE, ContextWorld, FAMILY_MAPPINGS, _family

FORMAT_VERSION = "lifelong-own-trace-v1"
TRACE_HASH_VERSION = FORMAT_VERSION
CONDITIONS = {
    "short": {"duration_range": (1.0, 4.0), "gap_range": (0.5, 2.0)},
    "narrow": {"duration_range": (1.0, 4.0), "gap_range": (0.5, 2.0)},
    "long_wait": {"duration_range": (8.0, 16.0), "gap_range": (0.5, 2.0)},
    "long_idle": {"duration_range": (1.0, 4.0), "gap_range": (16.0, 32.0)},
    "long_both": {"duration_range": (8.0, 16.0), "gap_range": (16.0, 32.0)},
    "far": {"duration_range": (16.0, 32.0), "gap_range": (32.0, 64.0)},
}
DEFAULT_CONDITIONS = ("short", "long_wait", "long_idle", "long_both", "far")
DEFAULT_AGE_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
COHORTS = ("primary", "inherited", "carryover_contract", "ineligible")
PHASES = (
    "unstarted_feasible", "unstarted_missed_start", "processing", "legal_finish",
)
IDLE_CAUSES = ("no_request_yet", "inherited_inactive_unknown", "completion", "cancellation", "expiry")
FIRST_ERROR_PRIORITY = (
    "invalid_code", "stale_version", "duplicate_begin", "before_begin", "early", "late", "wrong_payload",
)
_EPS = 1e-6


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _hash_update(digest, value: torch.Tensor | np.ndarray) -> None:
    array = _array(value) if isinstance(value, torch.Tensor) else value
    digest.update(np.ascontiguousarray(array).tobytes())


def _age_edges(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(map(float, values))
    if not result or result[0] != 0 or any(not math.isfinite(x) for x in result) or any(
            right <= left for left, right in zip(result, result[1:])):
        raise ValueError("age_edges must start at zero and increase strictly with finite bounds")
    return result


def _age_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"[{left:g},{edges[index + 1]:g})" if index + 1 < len(edges) else f"[{left:g},inf)"
            for index, left in enumerate(edges)]


def _cohort_codes(world: ContextWorld, fork_tick: int, evaluation_regime_id: int) -> np.ndarray:
    issued_after = _array(_new_since_fork(world.request_id, world, fork_tick))
    regimes = _array(world.current_regime_id)
    eligible = _array(world.current_eligible)
    return np.where(~eligible, 3, np.where(~issued_after, 1,
                    np.where(regimes == evaluation_regime_id, 0, 2))).astype(np.int64)


def _phases(world: ContextWorld) -> np.ndarray:
    started = _array(world.started)
    feasible = world.now + _array(world.duration) <= _array(world.deadline) + _EPS
    ready = world.now >= _array(world.ready_time) - _EPS
    return np.where(~started, np.where(feasible, 0, 1), np.where(ready, 3, 2)).astype(np.int64)


class _ContentTable:
    """Every row is an own-state opportunity, never a teacher trajectory."""

    def __init__(self, edges: tuple[float, ...]):
        self.edges = edges
        self.rows: dict[tuple[int, int, int], dict[str, int | float]] = {}

    def add(self, *, cohort, phase, age, active, choice, target, model_action,
            executed_action, margin, first_legal) -> None:
        bins = np.searchsorted(self.edges, age, side="right") - 1
        codes = (cohort * len(PHASES) + phase) * len(self.edges) + bins
        for code in np.unique(codes[active]):
            mask = active & (codes == code)
            age_bin = int(code % len(self.edges))
            outer = int(code // len(self.edges))
            key = (outer // len(PHASES), outer % len(PHASES), age_bin)
            row = self.rows.setdefault(key, {name: 0 for name in (
                "opportunities", "finish_choice_correct", "version_choice_correct",
                "payload_choice_correct", "model_finish_attempts", "executed_finish_attempts",
                "first_legal_opportunities", "first_legal_choice_correct", "margin_sum")})
            row["opportunities"] += int(mask.sum())
            row["finish_choice_correct"] += int((mask & (choice == target)).sum())
            row["version_choice_correct"] += int((mask & (choice // 4 == target // 4)).sum())
            row["payload_choice_correct"] += int((mask & (choice % 4 == target % 4)).sum())
            row["model_finish_attempts"] += int((mask & (model_action >= 3)).sum())
            row["executed_finish_attempts"] += int((mask & (executed_action >= 3)).sum())
            row["first_legal_opportunities"] += int((mask & first_legal).sum())
            row["first_legal_choice_correct"] += int((mask & first_legal & (choice == target)).sum())
            row["margin_sum"] += float(margin[mask].sum(dtype=np.float64))

    def result(self) -> list[dict[str, Any]]:
        labels = _age_labels(self.edges)
        result = []
        for (cohort, phase, age_bin), values in sorted(self.rows.items()):
            row = dict(values)
            row.update(cohort=COHORTS[cohort], phase=PHASES[phase], age_bin=labels[age_bin],
                       age_lower=self.edges[age_bin],
                       age_upper=self.edges[age_bin + 1] if age_bin + 1 < len(self.edges) else None,
                       conditional_finish_accuracy=_rate(row["finish_choice_correct"], row["opportunities"]),
                       mean_correct_logit_margin=_rate(row.pop("margin_sum"), row["opportunities"]),
                       first_legal_conditional_accuracy=_rate(
                           row["first_legal_choice_correct"], row["first_legal_opportunities"]))
            result.append(row)
        return result


class _Requests:
    """A bounded per-slot observer; no information is supplied to the model."""

    def __init__(self, world: ContextWorld):
        shape = (world.batch_size, world.n_jobs)
        self.ids = _array(world.request_id)
        self.regimes = _array(world.current_regime_id)
        self.first_error = np.where(_array(world.had_error), "inherited_error_unknown", "").astype(object)
        self.first_error_cause = self.first_error.copy()
        self.first_error_age = np.full(shape, np.nan)
        self.began = _array(world.started) | (_array(world._score_completed) & (
            _array(world._score_request_id) == self.ids))
        self.legal_opportunities = np.zeros(shape, np.int64)
        self.first_legal_correct = np.full(shape, -1, np.int64)
        self.cue_correct = np.full(shape, -1, np.int64)
        self.finish_attempts = np.zeros(shape, np.int64)
        self.legal_finish_attempts = np.zeros(shape, np.int64)
        self.wrong_content_attempts = np.zeros(shape, np.int64)
        self.matured: dict[str, dict[str, Any]] = {}

    def reset(self, world: ContextWorld, command: np.ndarray) -> None:
        self.ids[command] = _array(world.request_id)[command]
        self.regimes[command] = _array(world.current_regime_id)[command]
        self.first_error[command] = ""
        self.first_error_cause[command] = ""
        self.first_error_age[command] = np.nan
        self.began[command] = False
        self.legal_opportunities[command] = 0
        self.first_legal_correct[command] = -1
        self.cue_correct[command] = -1
        self.finish_attempts[command] = 0
        self.legal_finish_attempts[command] = 0
        self.wrong_content_attempts[command] = 0

    def summary(self, mask: np.ndarray, completed: np.ndarray, clean: np.ndarray) -> dict[str, Any]:
        errors, error_counts = np.unique(self.first_error[mask], return_counts=True)
        error_patterns = {str(key): int(value) for key, value in zip(errors, error_counts) if key}
        causes, cause_counts = np.unique(self.first_error_cause[mask], return_counts=True)
        first_causes = {str(key): int(value) for key, value in zip(causes, cause_counts) if key}
        first_legal = mask & (self.first_legal_correct >= 0)
        cue = mask & (self.cue_correct >= 0)
        pair = cue & first_legal
        both_correct = pair & (self.cue_correct == 1) & (self.first_legal_correct == 1)
        lost = pair & (self.cue_correct == 1) & (self.first_legal_correct == 0)
        gained = pair & (self.cue_correct == 0) & (self.first_legal_correct == 1)
        both_wrong = pair & (self.cue_correct == 0) & (self.first_legal_correct == 0)
        failed = mask & ~completed
        no_begin = failed & ~self.began
        no_legal = failed & self.began & (self.legal_opportunities == 0)
        had_legal = failed & (self.legal_opportunities > 0)
        return {
            "requests": int(mask.sum()), "completed": int((mask & completed).sum()),
            "clean_completed": int((mask & clean).sum()),
            "with_recorded_or_inherited_error": int((mask & (self.first_error != "")).sum()),
            "first_error_patterns": error_patterns,
            "first_error_causes": first_causes,
            "observed_first_error_age_sum": float(np.nansum(self.first_error_age[mask])),
            "observed_first_error_age_count": int(np.isfinite(self.first_error_age[mask]).sum()),
            "began": int((mask & self.began).sum()),
            "with_observed_legal_opportunity": int((mask & (self.legal_opportunities > 0)).sum()),
            "first_observed_legal_count": int(first_legal.sum()),
            "first_observed_legal_content_correct": int((first_legal & (self.first_legal_correct == 1)).sum()),
            "cue_observed_count": int(cue.sum()),
            "cue_content_correct_count": int((cue & (self.cue_correct == 1)).sum()),
            "cue_legal_pair_count": int(pair.sum()),
            "cue_legal_both_correct": int(both_correct.sum()),
            "cue_legal_lost": int(lost.sum()),
            "cue_legal_gained": int(gained.sum()),
            "cue_legal_both_wrong": int(both_wrong.sum()),
            "cue_legal_cue_correct": int((pair & (self.cue_correct == 1)).sum()),
            "cue_legal_legal_correct": int((pair & (self.first_legal_correct == 1)).sum()),
            "cue_without_legal_count": int((cue & ~first_legal).sum()),
            "legal_without_cue_count": int((first_legal & ~cue).sum()),
            "neither_cue_nor_legal_count": int((mask & ~cue & ~first_legal).sum()),
            "active_finish_attempts": int(self.finish_attempts[mask].sum()),
            "legally_timed_finish_attempts": int(self.legal_finish_attempts[mask].sum()),
            "finish_attempts_with_wrong_content": int(self.wrong_content_attempts[mask].sum()),
            "uncompleted_no_begin": int(no_begin.sum()),
            "uncompleted_began_but_no_legal_opportunity": int(no_legal.sum()),
            "uncompleted_despite_legal_opportunity": int(had_legal.sum()),
        }

    def add_matured(self, world: ContextWorld, masks: dict[str, np.ndarray]) -> None:
        completed, clean = _array(world.matured_completed), _array(world.matured_clean_completed)
        for name, mask in masks.items():
            summary = self.summary(mask, completed, clean)
            total = self.matured.setdefault(name, {})
            for field, value in summary.items():
                if isinstance(value, dict):
                    target = total.setdefault(field, {})
                    for pattern, count in value.items():
                        target[pattern] = target.get(pattern, 0) + count
                else:
                    total[field] = total.get(field, 0) + value

    def before_action(self, *, active, legal, choice, executed_choice, target, finish) -> np.ndarray:
        first_legal = active & legal & (self.first_legal_correct < 0)
        self.first_legal_correct[first_legal] = (choice == target)[first_legal]
        self.legal_opportunities += active & legal
        self.finish_attempts += active & finish
        self.legal_finish_attempts += active & finish & legal
        self.wrong_content_attempts += active & finish & (executed_choice != target)
        return first_legal

    def capture_cue(self, *, command, choice, target) -> None:
        """Record once after consuming NEW/REPLACE, before any action/receipt."""
        if np.any(command & (self.cue_correct >= 0)):
            raise AssertionError("cue readout recorded twice for the same request")
        self.cue_correct[command] = (choice == target)[command]

    def after_action(self, *, active, rejected, accepted_begin, patterns, age) -> None:
        self.began |= accepted_begin
        first = active & rejected & (self.first_error == "")
        self.first_error[first] = patterns[first]
        self.first_error_age[first] = age[first]
        unassigned = first.copy()
        for cause in FIRST_ERROR_PRIORITY:
            matches = np.fromiter((cause in str(value).split("+") for value in patterns.flat),
                                  dtype=bool, count=patterns.size).reshape(patterns.shape)
            take = unassigned & matches
            self.first_error_cause[take] = cause
            unassigned &= ~take
        self.first_error_cause[unassigned] = "unclassified"


def _error_patterns(world: ContextWorld, actions: np.ndarray) -> np.ndarray:
    """Disjoint combinations preserve simultaneous timing and content errors."""
    active, started = _array(world.active), _array(world.started)
    version, payload = _array(world.version), _array(world.payload)
    begin, finish = (actions == 1) | (actions == 2), (actions >= 3) & (actions < world.n_actions)
    attempted = actions != 0
    action_version = np.where(begin, actions - 1, (actions - 3) // 4)
    same = action_version == version
    flags = {
        "inactive": attempted & ~active,
        "invalid_code": attempted & ~(begin | finish),
        "stale_version": attempted & (begin | finish) & ~same,
        "duplicate_begin": begin & active & same & started,
        "before_begin": finish & active & same & ~started,
        "early": finish & active & same & started & (world.now < _array(world.ready_time) - _EPS),
        "late": finish & same & (_array(world.request_id) >= 0) & (world.now > _array(world.deadline) + _EPS),
        "wrong_payload": finish & active & same & ((actions - 3) % 4 != payload),
    }
    result = np.full(actions.shape, "", dtype=object)
    for name, mask in flags.items():
        present = mask & (result != "")
        absent = mask & ~present
        result[present] = np.char.add(np.asarray(result[present], dtype=str), "+" + name)
        result[absent] = name
    return result


@torch.inference_mode()
def evaluate(model, world: ContextWorld, hidden: torch.Tensor, family: int = 0,
             condition: str = "short", steps: int = 2048, *, dt: float | None = None,
             age_edges: Iterable[float] = DEFAULT_AGE_EDGES, keep_trace: bool = False,
             device: str | torch.device | None = None) -> dict[str, Any]:
    """Run a detached native eleven-action continuation under future context.

    This observer is versioned from integration/diagnose.py. It preserves its
    matured/inherited/carryover denominator bookkeeping and own-action readouts.
    No diagnostic intervention is permitted: every action is the model argmax.
    """
    family = _family(family)
    dt = world.dt if dt is None else float(dt)
    intervention = "none"
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if condition not in CONDITIONS:
        raise ValueError("unknown evaluation condition")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    edges = _age_edges(age_edges)
    saved_world = world.state_dict()
    if not saved_world["acted"]:
        raise ValueError("diagnostic forks must follow a completed action")
    if tuple(hidden.shape) != (world.batch_size, model.width):
        raise ValueError("hidden state does not match world/model")
    if (model.n_jobs, model.input_size, model.n_actions) != (world.n_jobs, world.input_size, 11):
        raise ValueError("model and world observation/action shapes differ")
    source_hash = state_digest({"model": model.state_dict(), "world": saved_world, "hidden": hidden})
    target_device = torch.device(hidden.device if device is None else device)
    live_model = deepcopy(model).to(target_device).eval()
    state = hidden.detach().to(target_device).clone()
    live_world = ContextWorld.from_state_dict(saved_world, device=world.device)
    parameters_hash = state_digest(live_model.state_dict())
    start_hidden_hash = state_digest({"hidden": state})
    fork_tick, fork_time = live_world.tick, live_world.now
    initial_counts = {name: int(value.sum()) for name, value in live_world.counts.items()}
    initial_pending = int(live_world._score_pending.sum())
    initial_active = int(live_world.active.sum())
    initial_started = int((live_world.active & live_world.started).sum())
    live_world.set_family(family)
    source_family_exposures = live_world.family_exposures()
    family_cohorts = {str(index): {name: {"matured": 0, "completed": 0, "clean_completed": 0}
                                for name in COHORTS[:3]} for index in range(4)}
    live_world.set_regime(**CONDITIONS[condition], name=f"evaluation/{family}/{condition}", dt=dt)
    evaluation_regime_id = live_world.regime_id
    cohorts = {name: _Cohort(name, live_world) for name in COHORTS[:3]}
    requests = _Requests(live_world)
    content = _ContentTable(edges)
    enabled = np.broadcast_to(np.arange(world.n_jobs) < world.active_jobs,
                              (world.batch_size, world.n_jobs))
    idle_cause = np.where(_array(live_world.request_id) < 0, 0, 1).astype(np.int64)
    terminal_time = np.full(enabled.shape, np.nan)
    idle_rows: dict[tuple[int, int], dict[str, int | float]] = {}
    history_counts = {name: 0 for name in ("completion", "cancellation", "expiry", "replacement")}
    history_error_patterns = {name: {} for name in history_counts}
    totals = {name: 0 for name in (
        "opportunities", "active_opportunities", "inactive_opportunities", "inactive_actions",
        "inactive_begin_actions", "inactive_finish_actions", "inactive_finish_matches_retired_content",
        "disabled_slot_opportunities", "disabled_slot_actions", "model_action_changes",
        "observation_ablation_opportunities", "active_finish_attempts", "legally_timed_finish_attempts",
        "legally_timed_wrong_content_attempts", "illegally_timed_finish_attempts",
        "content_correct_but_illegally_timed_finish_attempts")}
    primary_issued = carryover_issued = 0
    raw_observation_digest = hashlib.sha256(f"{TRACE_HASH_VERSION}/observations".encode())
    observation_digest = hashlib.sha256(f"{TRACE_HASH_VERSION}/observations".encode())
    action_digest = hashlib.sha256(f"{TRACE_HASH_VERSION}/actions".encode())
    model_action_digest = hashlib.sha256(f"{TRACE_HASH_VERSION}/actions".encode())
    first_observation_hash = first_model_observation_hash = None
    frames = []

    def terminal_history(name: str, mask: np.ndarray) -> None:
        history_counts[name] += int((mask & enabled).sum())
        patterns, counts = np.unique(requests.first_error[mask & enabled], return_counts=True)
        for pattern, count in zip(patterns, counts):
            if pattern:
                target = history_error_patterns[name]
                target[str(pattern)] = target.get(str(pattern), 0) + int(count)

    for _ in range(steps):
        previous_active, previous_deadline = _array(live_world.active), _array(live_world.deadline)
        raw_observations = live_world.observe()
        mature = _array(live_world.matured_mask)
        if np.any(mature & (requests.ids != _array(live_world.matured_request_id))):
            raise AssertionError("matured request lost its own-state tracking identity")
        issued_after = _array(_new_since_fork(live_world.matured_request_id, live_world, fork_tick))
        primary = issued_after & (requests.regimes == evaluation_regime_id)
        masks = {"primary": mature & primary, "inherited": mature & ~issued_after,
                 "carryover_contract": mature & issued_after & ~primary}
        for name, mask in masks.items():
            cohorts[name].add(live_world, torch.as_tensor(mask, device=live_world.device))
        requests.add_matured(live_world, masks)
        for index in range(4):
            for name, mask in masks.items():
                selected = mask & (_array(live_world.matured_family) == index)
                row = family_cohorts[str(index)][name]
                row["matured"] += int(selected.sum())
                row["completed"] += int((selected & _array(live_world.matured_completed)).sum())
                row["clean_completed"] += int((selected & _array(live_world.matured_clean_completed)).sum())
        expired = previous_active & (live_world.now > previous_deadline + _EPS)
        terminal_history("expiry", expired)
        idle_cause[expired], terminal_time[expired] = 4, live_world.now
        kind = _array(live_world.event_kind)
        canceled, replaced = kind == CANCEL, kind == REPLACE
        terminal_history("cancellation", canceled)
        terminal_history("replacement", replaced)
        idle_cause[canceled], terminal_time[canceled] = 3, live_world.now
        command = (kind == NEW) | replaced
        eligible_command = command & _array(live_world.event_eligible)
        under_evaluation = _array(live_world.event_regime_id) == evaluation_regime_id
        primary_issued += int((eligible_command & under_evaluation).sum())
        carryover_issued += int((eligible_command & ~under_evaluation).sum())
        requests.reset(live_world, command)
        idle_cause[command], terminal_time[command] = 1, np.nan
        observations = raw_observations.clone()
        ablated = np.zeros((world.batch_size, world.n_jobs), dtype=bool)
        _hash_update(raw_observation_digest, raw_observations)
        _hash_update(observation_digest, observations)
        if first_observation_hash is None:
            first_observation_hash = state_digest({"observation": raw_observations})
            first_model_observation_hash = state_digest({"observation": observations})
        logits, state = live_model.step(observations.to(target_device), state, live_world.elapsed)
        if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite state/logits in diagnostic continuation")
        values = _array(logits)
        model_action = values.argmax(-1)
        actions = model_action.copy()
        choice = values[..., 3:].argmax(-1)
        target = 4 * _array(live_world.version) + _array(live_world.payload)
        requests.capture_cue(command=command & enabled, choice=choice, target=target)
        finish_logits = values[..., 3:]
        correct_logit = np.take_along_axis(finish_logits, target[..., None], -1)[..., 0]
        other_logits = finish_logits.copy()
        np.put_along_axis(other_logits, target[..., None], -np.inf, -1)
        margin = correct_logit - other_logits.max(-1)
        active = _array(live_world.active) & enabled
        phase = _phases(live_world)
        legal = active & (phase == 3)
        age = live_world.now - _array(live_world.issued_time)
        executed_choice = actions - 3
        finish = actions >= 3
        first_legal = requests.before_action(active=active, legal=legal, choice=choice,
                                             executed_choice=executed_choice, target=target, finish=finish)
        content.add(cohort=_cohort_codes(live_world, fork_tick, evaluation_regime_id),
                    phase=phase, age=age, active=active, choice=choice, target=target,
                    model_action=model_action, executed_action=actions, margin=margin,
                    first_legal=first_legal)
        inactive = enabled & ~active
        inactive_action = inactive & (actions != 0)
        total_increment = {
            "opportunities": int(enabled.sum()), "active_opportunities": int(active.sum()),
            "inactive_opportunities": int(inactive.sum()), "inactive_actions": int(inactive_action.sum()),
            "inactive_begin_actions": int((inactive & (actions > 0) & (actions < 3)).sum()),
            "inactive_finish_actions": int((inactive & finish).sum()),
            "inactive_finish_matches_retired_content": int((inactive & finish & (
                _array(live_world.request_id) >= 0) & (executed_choice == target)).sum()),
            "disabled_slot_opportunities": int((~enabled).sum()),
            "disabled_slot_actions": int((~enabled & (actions != 0)).sum()),
            "model_action_changes": int((enabled & (actions != model_action)).sum()),
            "observation_ablation_opportunities": int((enabled & ablated).sum()),
            "active_finish_attempts": int((active & finish).sum()),
            "legally_timed_finish_attempts": int((legal & finish).sum()),
            "legally_timed_wrong_content_attempts": int((legal & finish & (executed_choice != target)).sum()),
            "illegally_timed_finish_attempts": int((active & finish & ~legal).sum()),
            "content_correct_but_illegally_timed_finish_attempts": int((active & finish & ~legal & (
                executed_choice == target)).sum()),
        }
        for name, increment in total_increment.items():
            totals[name] += increment
        terminal_age = live_world.now - terminal_time
        terminal_bins = np.where(np.isfinite(terminal_age),
                                 np.searchsorted(edges, terminal_age, side="right") - 1, -1)
        for code in np.unique((idle_cause * (len(edges) + 1) + terminal_bins + 1)[inactive]):
            cause, age_bin = int(code // (len(edges) + 1)), int(code % (len(edges) + 1)) - 1
            mask = inactive & (idle_cause == cause) & (terminal_bins == age_bin)
            row = idle_rows.setdefault((cause, age_bin), {name: 0 for name in (
                "opportunities", "actions", "begin_actions", "finish_actions", "finish_matches_retired_content",
                "known_retired_content_opportunities", "conditional_finish_matches_retired_content",
                "model_wait_preferences", "model_finish_preferences", "finish_preferred_to_wait",
                "retired_exact_finish_preferred_to_wait", "retired_content_readout_with_wait",
                "finish_minus_wait_logit_sum")})
            row["opportunities"] += int(mask.sum())
            row["actions"] += int((mask & (actions != 0)).sum())
            row["begin_actions"] += int((mask & (actions > 0) & (actions < 3)).sum())
            row["finish_actions"] += int((mask & finish).sum())
            row["finish_matches_retired_content"] += int((mask & finish & (
                _array(live_world.request_id) >= 0) & (executed_choice == target)).sum())
            known_retired = mask & (_array(live_world.request_id) >= 0)
            row["known_retired_content_opportunities"] += int(known_retired.sum())
            row["conditional_finish_matches_retired_content"] += int((known_retired & (choice == target)).sum())
            row["model_wait_preferences"] += int((mask & (model_action == 0)).sum())
            row["model_finish_preferences"] += int((mask & (model_action >= 3)).sum())
            finish_wait_margin = finish_logits.max(-1) - values[..., 0]
            row["finish_preferred_to_wait"] += int((mask & (finish_wait_margin > 0)).sum())
            row["retired_exact_finish_preferred_to_wait"] += int((known_retired & (
                correct_logit > values[..., 0])).sum())
            row["retired_content_readout_with_wait"] += int((known_retired & (choice == target) & (
                model_action == 0)).sum())
            row["finish_minus_wait_logit_sum"] += float(finish_wait_margin[mask].sum(dtype=np.float64))
        patterns = _error_patterns(live_world, actions)
        _hash_update(action_digest, actions.astype(np.int64))
        _hash_update(model_action_digest, model_action.astype(np.int64))
        live_world.act(torch.as_tensor(actions, dtype=torch.long, device=live_world.device))
        accepted = _array(live_world.last_result) == 1
        accepted_begin = accepted & (actions > 0) & (actions < 3)
        accepted_finish = accepted & finish
        rejected = _array(live_world.last_result) == -1
        requests.after_action(active=active, rejected=rejected, accepted_begin=accepted_begin,
                              patterns=patterns, age=age)
        terminal_history("completion", accepted_finish)
        idle_cause[accepted_finish], terminal_time[accepted_finish] = 2, live_world.now
        if keep_trace:
            frames.append({
                "tick": live_world.tick, "time": live_world.now, "event_hash": live_world.event_hash,
                "request_id": requests.ids.tolist(), "event_kind": kind.tolist(),
                "event_family": _array(live_world.event_family).tolist(),
                "raw_event_payload": _array(live_world.raw_event_payload).tolist(),
                "active_before_action": active.tolist(), "phase": phase.tolist(), "task_age": age.tolist(),
                "conditional_finish_choice": choice.tolist(), "target_finish_choice": target.tolist(),
                "first_observed_legal": first_legal.tolist(), "model_action": model_action.tolist(),
                "executed_action": actions.tolist(), "result": _array(live_world.last_result).tolist(),
                "error_pattern": patterns.tolist(), "ablated": ablated.tolist(),
                "raw_observation_hash": state_digest({"observation": raw_observations}),
                "model_observation_hash": state_digest({"observation": observations}),
                "hidden_hash": state_digest({"hidden": state}),
            })

    cohort_results = {name: item.result() for name, item in cohorts.items()}
    cohort_results["inherited"].update(initial_eligible_pending=initial_pending,
                                         initial_active=initial_active, initial_started=initial_started)
    censored = _pending_summary(live_world, fork_tick, evaluation_regime_id)
    for name, issued in (("primary", primary_issued), ("carryover_contract", carryover_issued),
                         ("inherited", initial_pending)):
        if issued != cohort_results[name]["matured"] + censored[name]["count"]:
            raise AssertionError(f"{name} eligible requests were lost or counted twice")
    labels = _age_labels(edges)
    idle_result = []
    for (cause, age_bin), counts in sorted(idle_rows.items()):
        idle_result.append({
            "cause": IDLE_CAUSES[cause], "time_since_terminal_bin": labels[age_bin] if age_bin >= 0 else "unknown",
            **counts, "action_rate_per_idle_opportunity": _rate(counts["actions"], counts["opportunities"]),
            "actions_per_1000_total_opportunities": _rate(1000 * counts["actions"], totals["opportunities"]),
            "mean_finish_minus_wait_logit": _rate(counts["finish_minus_wait_logit_sum"], counts["opportunities"]),
        })
    counters = {name: int(value.sum()) - initial_counts[name] for name, value in live_world.counts.items()}
    request_results = deepcopy(requests.matured)
    for row in request_results.values():
        row.update(
            cue_content_accuracy=_rate(row["cue_content_correct_count"], row["cue_observed_count"]),
            cue_legal_cue_accuracy=_rate(row["cue_legal_cue_correct"], row["cue_legal_pair_count"]),
            cue_legal_legal_accuracy=_rate(row["cue_legal_legal_correct"], row["cue_legal_pair_count"]),
            cue_legal_accuracy_given_cue_correct=_rate(row["cue_legal_both_correct"], row["cue_legal_cue_correct"]),
        )
    if totals["inactive_actions"] != counters["false_action"] - totals["disabled_slot_actions"]:
        raise AssertionError("inactive-action denominator masks disagree with frozen world semantics")
    if parameters_hash != state_digest(live_model.state_dict()):
        raise AssertionError("frozen diagnosis changed model parameters")
    if source_hash != state_digest({"model": model.state_dict(), "world": world.state_dict(), "hidden": hidden}):
        raise AssertionError("diagnosis mutated the source continuation")
    actual_exposures = live_world.family_exposures()
    exposure_delta = {key: [a - b for a, b in zip(actual_exposures[key], source_family_exposures[key])]
                      for key in actual_exposures}
    result = {
        "family": family, "family_mapping": list(FAMILY_MAPPINGS[family]),
        "source_family_exposures": source_family_exposures,
        "source_family_recency": world.family_recency(),
        "end_family_recency": live_world.family_recency(),
        "actual_family_exposures": exposure_delta, "family_cohorts": family_cohorts,
        "source_context_hash": state_digest(saved_world["context"]),
        "end_context_hash": state_digest(live_world.state_dict()["context"]),
        "context_policy": list(live_world.family_allowed),
        "source_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): _file_sha256(path)
                          for path in (Path(__file__), Path(__file__).with_name("world.py"),
                                       ROOT / "experiments/integration/diagnose.py",
                                       ROOT / "experiments/continuity/world.py",
                                       ROOT / "experiments/continuity/evaluate.py")},
        "format": FORMAT_VERSION, "condition": condition, "intervention": intervention,
        "action_policy": "model",
        "world_resets": 0, "hidden_resets": 0, "parameter_updates": 0,
        "steps": steps, "dt": float(dt), "batch_size": world.batch_size,
        "n_jobs": world.n_jobs, "active_jobs": world.active_jobs,
        "actor_ids": list(world.actor_ids), "case_ids": list(world.case_ids),
        "fork_tick": fork_tick, "fork_time": fork_time, "end_tick": live_world.tick,
        "end_time": live_world.now, "evaluation_regime_id": evaluation_regime_id,
        "regime": asdict(live_world.regime), "age_edges": list(edges),
        "phase_codes": dict(enumerate(PHASES)), "primary_eligible_issued": primary_issued,
        "carryover_contract_eligible_issued": carryover_issued, "initial_eligible_pending": initial_pending,
        **cohort_results, "request_diagnosis": request_results, "censored": censored,
        "content_by_cohort_phase_age": content.result(), "opportunities": totals,
        "inactive_actions_per_idle_opportunity": _rate(totals["inactive_actions"], totals["inactive_opportunities"]),
        "inactive_actions_per_1000_idle_opportunities": _rate(1000 * totals["inactive_actions"], totals["inactive_opportunities"]),
        "inactive_actions_per_1000_total_opportunities": _rate(1000 * totals["inactive_actions"], totals["opportunities"]),
        "inactive_history": idle_result, "observed_terminal_events": history_counts,
        "terminal_request_first_error_patterns": history_error_patterns,
        "post_fork_counters": counters,
        "post_fork_finish_precision": _rate(counters["completed_all"], counters["finish_attempted"]),
        "start_world_state_hash": state_digest(saved_world), "end_world_state_hash": state_digest(live_world.state_dict()),
        "start_scheduler_hash": state_digest(saved_world["scheduler"]),
        "end_scheduler_hash": state_digest(live_world.state_dict()["scheduler"]),
        "start_hidden_hash": start_hidden_hash, "end_hidden_hash": state_digest({"hidden": state}),
        "start_event_hash": saved_world["event_hash"], "end_event_hash": live_world.event_hash,
        "first_observation_hash": first_observation_hash, "first_model_observation_hash": first_model_observation_hash,
        "raw_observation_hash": raw_observation_digest.hexdigest(), "observation_hash": observation_digest.hexdigest(),
        "action_hash": action_digest.hexdigest(), "model_action_hash": model_action_digest.hexdigest(),
        "parameter_hash": parameters_hash, "source_continuation_hash": source_hash,
        "model": live_model.config(),
        "definitions": {
            "context": "one-shot four-way family cue accompanying raw NEW/REPLACE payload; mapped targets never enter the model",
            "primary": "eligible post-fork requests under the evaluation NEW contract, with matured deadlines",
            "conditional_finish": "argmax of the existing eight FINISH logits on the branch's own actual hidden state",
            "legal_finish": "active, BEGIN accepted, ready_time <= now <= deadline; ignores chosen content",
            "first_legal": "first observed legal opportunity per request since this fork; inherited history may precede it",
            "cue": "existing eight-FINISH readout once after consuming NEW/REPLACE and before taking that tick's action",
            "cue_legal_pair": "same matured final-eligible request with observed cue and first observed legal readout; excludes never-legal cases",
            "cue_legal_transition": "both_correct/lost/gained/both_wrong partition the matched pair population; cue-correct denominator is both_correct+lost",
            "first_error": "first rejected active-request attempt; simultaneous causes retained as a disjoint pattern",
            "first_error_priority": list(FIRST_ERROR_PRIORITY),
            "idle": "enabled job slot with no active task before its action; separate from disabled slots",
            "terminal_time": "observed post-fork completion/cancellation/expiry; inherited inactive causes remain unknown",
            "retired_content": "a correct retained FINISH readout is not an error when the model appropriately chooses WAIT",
        },
        "limitations": [
            "Conditional readout combines encoding, retention and existing-head access; it does not prove information absence.",
            "Incorrect immediate cue readout means the native FINISH head did not read the content at that phase; it does not prove failed encoding.",
            "Cue-to-legal transitions condition on requests reaching a legal opportunity; no-cue/no-legal exclusions and matured denominators remain separate.",
            "Active opportunities and first legal times depend on the policy; compare their populations before interpreting accuracy differences.",
            "A complete checkpoint preserves inherited state but cannot reconstruct unlogged pre-fork first errors or terminal times.",
        ],
    }
    if keep_trace:
        result["trace"] = frames
    return result



# A familiar name for callers ported from continuity; semantics remain native only.
evaluate_fork = evaluate
