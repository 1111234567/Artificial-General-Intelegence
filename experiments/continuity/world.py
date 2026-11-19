"""An unbounded, action-independent request stream and persistent evaluator.

Each job exposes exactly the dynamics experiment's 24 public features: one-shot
NEW/REPLACE/CANCEL pulses, their public parameters, and the preceding action's
receipt. Readiness, pending requests, eligibility and scheduler state are never
observations. The default four-slot world can activate two or four slots while
keeping the observation and action shapes fixed.

There is no episode, finite schedule, implicit reset, end-of-chunk flush, or
global random generator. Save ``state_dict()`` together with the controller's
hidden state to resume at the same causal boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace as dataclass_replace
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F


NONE, NEW, REPLACE, CANCEL = 0, 1, 2, 3
N_JOBS, N_PAYLOADS, N_VERSIONS = 4, 4, 2
N_ACTIONS = 3 + N_VERSIONS * N_PAYLOADS
FEATURES_PER_JOB = 24
INPUT_SIZE = N_JOBS * FEATURES_PER_JOB
FEATURE_LAYOUT = {
    "kind": (0, 3), "version": (3, 5), "payload": (5, 9),
    "duration": (9, 10), "window": (10, 11),
    "previous_action": (11, 22), "previous_result": (22, 24),
}
FORMAT_VERSION = "continuity-world-v1"
_EPS = 1e-6
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _range(name: str, value: Iterable[float], *, positive: bool) -> tuple[float, float]:
    try:
        low, high = map(float, value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite bounds") from error
    if not all(map(math.isfinite, (low, high))) or high < low or (
            low <= 0 if positive else low < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} needs ordered finite {qualifier} bounds")
    return low, high


@dataclass(frozen=True)
class Regime:
    duration_range: tuple[float, float] = (1.0, 4.0)
    gap_range: tuple[float, float] = (0.5, 2.0)
    window_range: tuple[float, float] = (1.5, 2.5)
    cancel: float = 0.20
    replace: float = 0.30
    name: str = "initial"

    def __post_init__(self) -> None:
        for name, positive in (("duration_range", True), ("gap_range", False),
                               ("window_range", False)):
            object.__setattr__(self, name, _range(name, getattr(self, name), positive=positive))
        cancel, replacement = float(self.cancel), float(self.replace)
        if not all(map(math.isfinite, (cancel, replacement))) or min(cancel, replacement) < 0:
            raise ValueError("cancel and replace must be finite nonnegative probabilities")
        if cancel + replacement > 1:
            raise ValueError("cancel + replace must not exceed one")
        object.__setattr__(self, "cancel", cancel)
        object.__setattr__(self, "replace", replacement)
        if not isinstance(self.name, str):
            raise ValueError("regime name must be a string")


def _mix64(values: np.ndarray) -> np.ndarray:
    """SplitMix64's bijective mixer; overflow is the intended uint64 arithmetic."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class OnlineWorld:
    """Persistent batched worlds. Call observe(), oracle() if needed, then act().

    ``actor_ids`` identify independent lanes, and default to ``range(batch_size)``.
    A lane/job's stream is unchanged by other lanes, batch size, actions, or
    checkpoint/chunk placement. ``active_jobs`` activates the first such slots;
    it is fixed for the life of a world. ``set_regime`` affects requests whose
    NEW pulse has not happened yet, including neither pending replacements nor
    the current cycle's already sampled idle gap.
    """

    n_actions = N_ACTIONS
    features_per_job = FEATURES_PER_JOB

    _state_tensors = (
        "active", "started", "version", "payload", "duration", "deadline",
        "ready_time", "issued_time", "request_id", "current_eligible",
        "current_interrupted", "current_regime_id", "had_error",
        "_ack_action", "_ack_result", "_cancel_guard", "_cancel_violated",
        "_score_pending", "_score_completed", "_score_clean", "_score_interrupted",
        "_score_deadline", "_score_request_id", "finish_delay_sum", "last_result",
        "event_kind", "event_version", "event_payload", "event_duration",
        "event_window", "event_request_id", "event_eligible", "event_regime_id",
        "matured_mask", "matured_request_id", "matured_issued_at", "matured_duration",
        "matured_completed", "matured_clean_completed",
    )
    _scheduler_arrays = (
        "_rng_state", "_next_new_time", "_next_version", "_modify_time", "_cycle_gap",
        "_modify_kind", "_modify_version", "_modify_payload", "_modify_duration",
        "_modify_window", "_modify_request_id", "_modify_regime_id",
    )
    _count_names = (
        "commands", "eligible", "matured_eligible", "completed_matured",
        "clean_completed_matured", "completed_eligible", "clean_completed_eligible",
        "completed_all", "finish_attempted", "canceled", "canceled_correctly",
        "replaced", "expired", "attempted", "accepted", "rejected", "false_action",
        "early", "late", "wrong_payload", "stale", "before_begin", "duplicate_begin",
        "invalid_code", "interrupted_eligible", "interrupted_completed",
        "matured_interrupted", "completed_matured_interrupted",
    )

    def __init__(self, seed: int = 0, batch_size: int = 64, dt: float = 0.25,
                 duration_range: tuple[float, float] = (1.0, 4.0),
                 gap_range: tuple[float, float] = (0.5, 2.0),
                 cancel: float = 0.20, replace: float = 0.30,
                 device: str | torch.device = "cpu", *, n_jobs: int = N_JOBS,
                 active_jobs: int | None = None,
                 window_range: tuple[float, float] = (1.5, 2.5),
                 actor_ids: Iterable[int] | None = None, regime_name: str = "initial"):
        self.batch_size = _positive_integer("batch_size", batch_size)
        self.n_jobs = _positive_integer("n_jobs", n_jobs)
        self.active_jobs = self.n_jobs if active_jobs is None else _positive_integer(
            "active_jobs", active_jobs)
        if self.active_jobs > self.n_jobs:
            raise ValueError("active_jobs must not exceed n_jobs")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        identities = tuple(range(self.batch_size)) if actor_ids is None else tuple(actor_ids)
        if len(identities) != self.batch_size or any(
                isinstance(value, bool) or not isinstance(value, Integral) or
                not 0 <= value < 2**64 for value in identities) or len(set(identities)) != len(identities):
            raise ValueError("actor_ids must contain one distinct uint64 integer per actor")
        self.actor_ids = tuple(map(int, identities))
        self.device = torch.device(device)
        self.input_size = self.n_jobs * FEATURES_PER_JOB
        self.regime = Regime(duration_range, gap_range, window_range, cancel, replace, regime_name)
        self.regime_id = 0
        self.regime_history = [{"tick": -1, "regime_id": 0, **asdict(self.regime)}]
        self._step, self._acted, self.now, self._elapsed = -1, True, 0.0, 0.0
        self._initialize_scheduler()
        self._initialize_evaluator()
        initial_identity = {"format": FORMAT_VERSION, **self._spec(), "regime": asdict(self.regime)}
        self._event_hash = hashlib.sha256(json.dumps(
            initial_identity, sort_keys=True).encode()).hexdigest()

    @property
    def tick(self) -> int:
        return self._step

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(f"{FORMAT_VERSION}:{self.seed}:{actor}" for actor in self.actor_ids)

    @property
    def current_request_id(self) -> torch.Tensor:
        return self.request_id

    @property
    def current_started(self) -> torch.Tensor:
        return self.started

    @property
    def current_active(self) -> torch.Tensor:
        return self.active

    @property
    def issued_at(self) -> torch.Tensor:
        return self.issued_time

    @property
    def earliest_ready(self) -> torch.Tensor:
        return self.issued_time + self.duration

    @property
    def matured_earliest_ready(self) -> torch.Tensor:
        return self.matured_issued_at + self.matured_duration

    @property
    def event_hash(self) -> str:
        """Action-independent digest of the public event prefix observed so far."""
        return self._event_hash

    def _spec(self) -> dict[str, Any]:
        return {"seed": self.seed, "batch_size": self.batch_size, "dt": self.dt,
                "n_jobs": self.n_jobs, "active_jobs": self.active_jobs,
                "actor_ids": self.actor_ids}

    def _draw(self, mask: np.ndarray, number: int) -> np.ndarray:
        """Draw from independent lane/job generators, without per-actor loops."""
        old = self._rng_state[mask]
        with np.errstate(over="ignore"):
            offsets = np.arange(1, number + 1, dtype=np.uint64) * _GOLDEN
            words = _mix64(old[:, None] + offsets[None, :])
            self._rng_state[mask] = old + offsets[-1]
        return (words >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)

    def _initialize_scheduler(self) -> None:
        shape = (self.batch_size, self.n_jobs)
        actors = np.asarray(self.actor_ids, dtype=np.uint64)[:, None]
        jobs = np.arange(self.n_jobs, dtype=np.uint64)[None, :]
        with np.errstate(over="ignore"):
            self._rng_state = _mix64(np.uint64(self.seed % 2**64) ^
                                     _mix64(actors + np.uint64(0xD1B54A32D192ED03)) ^
                                     _mix64(jobs + np.uint64(0x8CB92BA72F3D8DD7)))
        initial = self._draw(np.ones(shape, dtype=bool), 2).reshape(*shape, 2)
        limit = max(2, int(math.ceil(1.5 / self.dt)))
        self._next_new_time = np.floor(initial[..., 0] * limit) * self.dt
        self._next_new_time[:, self.active_jobs:] = np.inf
        self._next_version = np.floor(initial[..., 1] * N_VERSIONS).astype(np.int64)
        self._modify_time = np.full(shape, np.inf, dtype=np.float64)
        self._cycle_gap = np.zeros(shape, dtype=np.float64)
        for name in ("_modify_kind", "_modify_version", "_modify_payload", "_modify_regime_id"):
            setattr(self, name, np.zeros(shape, dtype=np.int64))
        self._modify_request_id = np.full(shape, -1, dtype=np.int64)
        self._modify_duration = np.zeros(shape, dtype=np.float32)
        self._modify_window = np.zeros(shape, dtype=np.float32)

    def _initialize_evaluator(self) -> None:
        shape = (self.batch_size, self.n_jobs)
        def zeros(dtype=torch.float64):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        for name in ("active", "started", "current_eligible", "current_interrupted", "had_error",
                     "_cancel_guard", "_cancel_violated", "_score_pending", "_score_completed",
                     "_score_clean", "_score_interrupted", "event_eligible", "matured_mask",
                     "matured_completed", "matured_clean_completed"):
            setattr(self, name, zeros(torch.bool))
        for name in ("version", "payload", "current_regime_id", "_ack_action", "_ack_result",
                     "last_result", "event_kind", "event_version", "event_payload", "event_regime_id"):
            setattr(self, name, zeros(torch.long))
        for name in ("request_id", "_score_request_id", "event_request_id", "matured_request_id"):
            setattr(self, name, torch.full(shape, -1, dtype=torch.long, device=self.device))
        for name in ("duration", "deadline", "issued_time", "_score_deadline", "finish_delay_sum",
                     "matured_issued_at", "matured_duration"):
            setattr(self, name, zeros())
        self.ready_time = torch.full(shape, float("inf"), dtype=torch.float64, device=self.device)
        self.event_duration, self.event_window = zeros(torch.float32), zeros(torch.float32)
        self.counts = {name: zeros(torch.long) for name in self._count_names}

    def set_dt(self, dt: float) -> None:
        """Set the next observation interval, without changing the absolute clock."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = dt

    def set_regime(self, *, duration_range=None, gap_range=None, window_range=None,
                   cancel=None, replace=None, name=None, dt=None) -> None:
        """Change only future NEW draws; preserve every already scheduled consequence."""
        changes = {key: value for key, value in {
            "duration_range": duration_range, "gap_range": gap_range,
            "window_range": window_range, "cancel": cancel, "replace": replace,
            "name": name}.items() if value is not None}
        regime = dataclass_replace(self.regime, **changes)
        if dt is not None:
            self.set_dt(dt)
        if not changes:
            return
        self.regime = regime
        self.regime_id += 1
        self.regime_history.append({"tick": self.tick, "regime_id": self.regime_id,
                                    **asdict(self.regime)})

    def _events(self) -> dict[str, np.ndarray]:
        shape = (self.batch_size, self.n_jobs)
        event = {name: np.zeros(shape, dtype=np.int64) for name in
                 ("kind", "version", "payload", "regime_id")}
        event.update({"duration": np.zeros(shape, np.float32),
                      "window": np.zeros(shape, np.float32),
                      "eligible": np.zeros(shape, bool),
                      "request_id": np.full(shape, -1, np.int64)})
        new = self._next_new_time <= self.now + _EPS
        if new.any():
            draws = self._draw(new, 9)
            regime = self.regime
            def uniform(bounds, column):
                return bounds[0] + draws[:, column] * (bounds[1] - bounds[0])
            d1, w1 = uniform(regime.duration_range, 0).astype(np.float32), uniform(
                regime.window_range, 1).astype(np.float32)
            d2, w2 = uniform(regime.duration_range, 2).astype(np.float32), uniform(
                regime.window_range, 3).astype(np.float32)
            payload = np.floor(draws[:, 4] * N_PAYLOADS).astype(np.int64)
            revised = (payload + 1 + np.floor(draws[:, 5] * (N_PAYLOADS - 1)).astype(np.int64)) % N_PAYLOADS
            cancel = draws[:, 6] < regime.cancel
            replacement = (draws[:, 6] >= regime.cancel) & (draws[:, 6] < regime.cancel + regime.replace)
            modify = cancel | replacement
            # A physical time strictly before earliest possible completion.
            # A changed observation interval may deliver this pulse later, but
            # it still takes priority over any action at that observation.
            modified_at = self.now + (0.25 + draws[:, 7] * 0.45) * d1.astype(np.float64)
            version = self._next_version[new].copy()
            jobs = np.broadcast_to(np.arange(self.n_jobs), shape)[new]
            request_id = self.tick * self.n_jobs + jobs
            event["kind"][new], event["version"][new] = NEW, version
            event["payload"][new], event["duration"][new], event["window"][new] = payload, d1, w1
            event["eligible"][new], event["request_id"][new] = ~modify, request_id
            event["regime_id"][new] = self.regime_id
            self._modify_time[new] = np.where(modify, modified_at, np.inf)
            self._modify_kind[new] = np.where(cancel, CANCEL, REPLACE)
            self._modify_version[new] = np.where(replacement, 1 - version, version)
            self._modify_payload[new] = np.where(replacement, revised, payload)
            self._modify_duration[new] = np.where(replacement, d2, 0)
            self._modify_window[new] = np.where(replacement, w2, 0)
            self._modify_request_id[new] = request_id
            self._modify_regime_id[new] = self.regime_id
            gap = uniform(regime.gap_range, 8)
            self._cycle_gap[new] = gap
            final_deadline = self.now + d1.astype(np.float64) + w1.astype(np.float64)
            # Modified cycles await their actual public modification pulse
            # before scheduling another NEW. Never collapse two one-shot
            # events into the same job's observation after a clock change.
            self._next_new_time[new] = np.where(modify, np.inf,
                final_deadline + np.maximum(gap, 4 * _EPS))
            self._next_version[new] = np.where(replacement, version, 1 - version)
        modified = (self._modify_time <= self.now + _EPS) & ~new
        if modified.any():
            for name in ("kind", "version", "payload", "duration", "window", "request_id", "regime_id"):
                event[name][modified] = getattr(self, "_modify_" + name)[modified]
            event["eligible"][modified] = self._modify_kind[modified] == REPLACE
            jobs = np.broadcast_to(np.arange(self.n_jobs), shape)[modified]
            replacement = self._modify_kind[modified] == REPLACE
            event["request_id"][modified] = np.where(replacement,
                self.tick * self.n_jobs + jobs, self._modify_request_id[modified])
            final_deadline = self.now + self._modify_duration[modified].astype(
                np.float64) + self._modify_window[modified].astype(np.float64)
            self._next_new_time[modified] = final_deadline + np.maximum(self._cycle_gap[modified], 4 * _EPS)
            self._modify_time[modified] = np.inf
        if new.any() or modified.any():
            digest = hashlib.sha256(bytes.fromhex(self._event_hash))
            digest.update(np.asarray(self.tick, dtype=np.int64).tobytes())
            digest.update(np.asarray(self.now, dtype=np.float64).tobytes())
            for name in ("kind", "version", "payload", "duration", "window", "request_id"):
                digest.update(event[name].tobytes())
            self._event_hash = digest.hexdigest()
        return event

    def _mature(self) -> None:
        mature = self._score_pending & (self.now > self._score_deadline + _EPS)
        # Emit evaluator-only pulses before a NEW can overwrite retained fields.
        # An early FINISH clears active but leaves this outstanding score intact.
        self.matured_mask = mature.clone()
        self.matured_request_id = torch.where(mature, self._score_request_id, -1)
        self.matured_issued_at = torch.where(mature, self.issued_time, 0.0)
        self.matured_duration = torch.where(mature, self.duration, 0.0)
        self.matured_completed = mature & self._score_completed
        self.matured_clean_completed = mature & self._score_clean
        self.counts["matured_eligible"] += mature
        self.counts["completed_matured"] += mature & self._score_completed
        self.counts["clean_completed_matured"] += mature & self._score_clean
        self.counts["matured_interrupted"] += mature & self._score_interrupted
        self.counts["completed_matured_interrupted"] += mature & self._score_interrupted & self._score_completed
        self._score_pending &= ~mature

    def observe(self, tick: int | None = None) -> torch.Tensor:
        """Advance one physical tick, apply exogenous events, and consume receipts."""
        if not self._acted or (tick is not None and tick != self._step + 1):
            raise ValueError("observe must advance once, after the preceding act")
        self._elapsed = 0.0 if self._step < 0 else self.dt
        self.now += self._elapsed
        self._step += 1
        self._acted = False
        self._mature()
        expired = self.active & (self.now > self.deadline + _EPS)
        self.counts["expired"] += expired
        self.active &= ~expired
        for name, value in self._events().items():
            setattr(self, "event_" + name, torch.from_numpy(value).to(self.device))
        kind = self.event_kind
        command, cancel = (kind == NEW) | (kind == REPLACE), kind == CANCEL
        eligible_command = command & self.event_eligible
        self.counts["commands"] += command
        self.counts["eligible"] += eligible_command
        self.counts["replaced"] += kind == REPLACE
        self.counts["canceled"] += cancel
        self.counts["canceled_correctly"] += cancel
        self._cancel_guard = torch.where(command, False, self._cancel_guard) | cancel
        self._cancel_violated = torch.where(command | cancel, False, self._cancel_violated)
        matching_cancel = cancel & (self.event_version == self.version)
        self.active = torch.where(command, True, self.active & ~matching_cancel)
        self.started = torch.where(command | matching_cancel, False, self.started)
        for target, source in (("version", "event_version"), ("payload", "event_payload"),
                               ("duration", "event_duration"), ("request_id", "event_request_id"),
                               ("current_regime_id", "event_regime_id")):
            old = getattr(self, target)
            setattr(self, target, torch.where(command, getattr(self, source).to(old.dtype), old))
        self.deadline = torch.where(command, self.now + self.event_duration.double()
                                    + self.event_window.double(), self.deadline)
        self.issued_time = torch.where(command, self.now, self.issued_time)
        self.ready_time = torch.where(command | matching_cancel, float("inf"), self.ready_time)
        self.current_eligible = torch.where(command, self.event_eligible, self.current_eligible)
        self.current_interrupted = torch.where(command, False, self.current_interrupted)
        self.had_error = torch.where(command, False, self.had_error)
        self._score_pending |= eligible_command
        for name in ("_score_completed", "_score_clean", "_score_interrupted"):
            setattr(self, name, torch.where(eligible_command, False, getattr(self, name)))
        self._score_deadline = torch.where(eligible_command, self.deadline, self._score_deadline)
        self._score_request_id = torch.where(eligible_command, self.request_id, self._score_request_id)
        event = kind != NONE
        other_event = event.sum(1, keepdim=True) - event.long() > 0
        interruption = (other_event & self.current_eligible & ~self.current_interrupted
                        & (self.now > self.issued_time + _EPS)
                        & (self.now < self.issued_time + self.duration - _EPS))
        self.current_interrupted |= interruption
        self._score_interrupted |= interruption & self._score_pending
        self.counts["interrupted_eligible"] += interruption
        features = torch.cat((
            F.one_hot(kind, 4)[..., 1:].float(),
            F.one_hot(self.event_version, N_VERSIONS).float() * event[..., None],
            F.one_hot(self.event_payload, N_PAYLOADS).float() * command[..., None],
            (self.event_duration * command)[..., None],
            (self.event_window * command)[..., None],
            F.one_hot(self._ack_action.clamp(0, N_ACTIONS - 1), N_ACTIONS).float()
            * (self._ack_action != 0)[..., None],
            torch.stack((self._ack_result == 1, self._ack_result == -1), -1).float(),
        ), -1)
        self._ack_action.zero_()
        self._ack_result.zero_()
        return features.flatten(1)

    def oracle(self, policy: str = "midpoint") -> torch.Tensor:
        """Causal evaluator labels: inspect current jobs only, never future events."""
        if self._step < 0 or self._acted:
            raise ValueError("oracle labels are available between observe and act")
        if policy not in ("midpoint", "earliest"):
            raise ValueError("oracle policy must be 'midpoint' or 'earliest'")
        begin = self.active & ~self.started & (self.now + self.duration <= self.deadline + _EPS)
        finish = self.active & self.started & (self.now >= self.ready_time - _EPS) & (
            self.now <= self.deadline + _EPS)
        if policy == "midpoint":
            center = 0.5 * (self.ready_time + self.deadline)
            finish &= (self.now >= center - _EPS) | (self.now + self.dt > self.deadline + _EPS)
        return torch.where(begin, 1 + self.version,
                           torch.where(finish, 3 + N_PAYLOADS * self.version + self.payload, 0))

    def act(self, actions: torch.Tensor) -> None:
        """Apply one action per job and retain exactly one receipt for observe()."""
        if self._step < 0 or self._acted:
            raise ValueError("act must follow exactly one observe")
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if tuple(actions.shape) != (self.batch_size, self.n_jobs):
            raise ValueError("actions must have shape [batch,n_jobs]")
        attempted = actions != 0
        begin = (actions == 1) | (actions == 2)
        finish = (actions >= 3) & (actions < N_ACTIONS)
        valid = begin | finish | ~attempted
        action_version = torch.where(begin, actions - 1, torch.div(
            actions - 3, N_PAYLOADS, rounding_mode="floor"))
        action_payload = (actions - 3).remainder(N_PAYLOADS)
        same_version = action_version == self.version
        accepted_begin = begin & self.active & same_version & ~self.started
        accepted_finish = (finish & self.active & same_version & self.started
                           & (action_payload == self.payload)
                           & (self.now >= self.ready_time - _EPS)
                           & (self.now <= self.deadline + _EPS))
        accepted = accepted_begin | accepted_finish
        rejected = attempted & ~accepted
        clean_finish = accepted_finish & self.current_eligible & ~self.had_error
        categories = {
            "attempted": attempted, "accepted": accepted, "rejected": rejected,
            "false_action": attempted & ~self.active,
            "stale": attempted & valid & ~same_version,
            "invalid_code": attempted & ~valid,
            "early": finish & self.active & same_version & self.started & (self.now < self.ready_time - _EPS),
            "late": finish & same_version & (self.request_id >= 0) & (self.now > self.deadline + _EPS),
            "wrong_payload": finish & self.active & same_version & (action_payload != self.payload),
            "before_begin": finish & self.active & same_version & ~self.started,
            "duplicate_begin": begin & self.active & same_version & self.started,
            "completed_all": accepted_finish, "finish_attempted": finish,
            "completed_eligible": accepted_finish & self.current_eligible,
            "clean_completed_eligible": clean_finish,
            "interrupted_completed": accepted_finish & self.current_eligible & self.current_interrupted,
        }
        for name, mask in categories.items():
            self.counts[name] += mask
        scoring_finish = accepted_finish & self.current_eligible & self._score_pending & (
            self.request_id == self._score_request_id)
        self._score_completed |= scoring_finish
        self._score_clean |= scoring_finish & ~self.had_error
        self.had_error |= rejected & self.active
        cancellation_violation = self._cancel_guard & ~self._cancel_violated & attempted
        self.counts["canceled_correctly"] -= cancellation_violation.long()
        self._cancel_violated |= cancellation_violation
        self.finish_delay_sum += torch.where(accepted_finish, self.now - self.ready_time, 0.0)
        self.ready_time = torch.where(accepted_begin, self.now + self.duration, self.ready_time)
        self.started = (self.started | accepted_begin) & ~accepted_finish
        self.active &= ~accepted_finish
        self._ack_action = torch.where(attempted, actions, 0)
        self._ack_result = accepted.long() - rejected.long()
        self.last_result = self._ack_result.clone()
        self._acted = True

    def metrics(self, per_stream: bool = False) -> dict[str, Any]:
        """Score matured windows; report all outstanding windows without clipping."""
        result = {name: int(value.sum().item()) for name, value in self.counts.items()}
        denominator = result["matured_eligible"]
        result["completion_rate"] = result["completed_matured"] / denominator if denominator else 0.0
        result["clean_completion_rate"] = result["clean_completed_matured"] / denominator if denominator else 0.0
        result["finish_precision"] = result["completed_all"] / result["finish_attempted"] if result["finish_attempted"] else 0.0
        result["interrupted_completion_rate"] = (result["completed_matured_interrupted"] /
            result["matured_interrupted"] if result["matured_interrupted"] else 0.0)
        result["eligible_pending"] = int(self._score_pending.sum().item())
        result["eligible_pending_completed"] = int((self._score_pending & self._score_completed).sum().item())
        result["eligible_pending_active"] = int((self.active & self.current_eligible).sum().item())
        result["pending_active"] = int(self.active.sum().item())
        result["pending_started"] = int((self.active & self.started).sum().item())
        completed = self.counts["completed_all"].sum(0).cpu().tolist()
        result["per_job_finish_delay"] = [delay / count if count else None for delay, count in
            zip(self.finish_delay_sum.sum(0).cpu().tolist(), completed)]
        result.update(tick=self.tick, now=self.now, elapsed=self.elapsed, dt=self.dt,
                      observations=self.tick + 1,
                      batch_size=self.batch_size, n_jobs=self.n_jobs, active_jobs=self.active_jobs,
                      world_hash=self.event_hash, regime_id=self.regime_id, regime=asdict(self.regime))
        if per_stream:
            result["case_ids"] = list(self.case_ids)
            result["per_stream"] = {name: value.sum(1).cpu().tolist() for name, value in self.counts.items()}
            result["per_stream"]["eligible_pending"] = self._score_pending.sum(1).cpu().tolist()
            result["per_stream"]["pending_active"] = self.active.sum(1).cpu().tolist()
        return result

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Evaluator state for boundary tracking; none of these are extra inputs."""
        names = ("active", "started", "version", "payload", "duration", "deadline", "ready_time",
                 "issued_time", "request_id", "current_eligible", "current_interrupted",
                 "current_regime_id", "had_error", "event_kind", "event_request_id",
                 "matured_mask", "matured_request_id", "matured_issued_at", "matured_duration",
                 "matured_completed", "matured_clean_completed")
        result = {name: getattr(self, name).clone() for name in names}
        result.update(current_request_id=result["request_id"].clone(),
                      current_started=result["started"].clone(), current_active=result["active"].clone(),
                      issued_at=self.issued_at.clone(), earliest_ready=self.earliest_ready.clone())
        return result

    def state_dict(self) -> dict[str, Any]:
        """A detached full causal checkpoint, valid before or after an action."""
        scheduler = {name: torch.from_numpy((getattr(self, name).view(np.int64) if name ==
                     "_rng_state" else getattr(self, name)).copy()) for name in self._scheduler_arrays}
        return {"format": FORMAT_VERSION, "spec": self._spec(), "regime": asdict(self.regime),
                "regime_id": self.regime_id, "regime_history": deepcopy(self.regime_history),
                "tick": self.tick, "acted": self._acted, "now": self.now, "elapsed": self.elapsed,
                "event_hash": self.event_hash, "scheduler": scheduler,
                "tensors": {name: getattr(self, name).detach().cpu().clone() for name in self._state_tensors},
                "counts": {name: value.detach().cpu().clone() for name, value in self.counts.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported online world state format")
        spec = state["spec"]
        if (spec["batch_size"], spec["n_jobs"]) != (self.batch_size, self.n_jobs):
            raise ValueError("world checkpoint batch and job shapes must match")
        for name in ("seed", "dt", "active_jobs"):
            setattr(self, name, spec[name])
        self.actor_ids = tuple(spec["actor_ids"])
        self.regime = Regime(**state["regime"])
        self.regime_id = int(state["regime_id"])
        self.regime_history = deepcopy(state["regime_history"])
        self._step, self._acted, self.now = int(state["tick"]), bool(state["acted"]), float(state["now"])
        self._elapsed = float(state["elapsed"])
        self._event_hash = str(state["event_hash"])
        shape = (self.batch_size, self.n_jobs)
        for name in self._scheduler_arrays:
            value = state["scheduler"][name].detach().cpu().numpy().copy()
            if value.shape != shape:
                raise ValueError("malformed scheduler tensor shape")
            setattr(self, name, value.view(np.uint64) if name == "_rng_state" else value)
        for name in self._state_tensors:
            value = state["tensors"][name]
            if tuple(value.shape) != shape:
                raise ValueError("malformed world tensor shape")
            setattr(self, name, value.detach().to(self.device).clone())
        self.counts = {name: state["counts"][name].detach().to(self.device).clone() for name in self._count_names}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device: str | torch.device = "cpu") -> "OnlineWorld":
        world = cls(**state["spec"], device=device)
        world.load_state_dict(state)
        return world
