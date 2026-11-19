"""Two concurrent orders, one-shot commands, and physical processing time.

Only command pulses and the preceding action's accepted/rejected receipt are
observable. Pending orders, progress, deadlines and oracle labels are evaluator
state. In particular, processing completion does not generate a READY input.

Per job, actions are WAIT=0, BEGIN(v)=1+v and FINISH(v,p)=3+4*v+p.
The two public version values distinguish the immediately preceding request;
they cannot distinguish arbitrarily old requests after parity wraps (ABA).
This is a bounded persistent-control diagnostic, with a queue-and-timers oracle,
not a general intelligence or planning benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


NONE, NEW, REPLACE, CANCEL = 0, 1, 2, 3
N_JOBS, N_PAYLOADS, N_VERSIONS = 2, 4, 2
N_ACTIONS = 3 + N_VERSIONS * N_PAYLOADS
FEATURES_PER_JOB = 24
INPUT_SIZE = N_JOBS * FEATURES_PER_JOB
# Each block: event kind [0:3], version [3:5], payload [5:9],
# duration [9], finish window [10], last action [11:22], receipt [22:24].
FEATURE_LAYOUT = {
    "kind": (0, 3), "version": (3, 5), "payload": (5, 9),
    "duration": (9, 10), "window": (10, 11),
    "previous_action": (11, 22), "previous_result": (22, 24),
}


@dataclass(frozen=True)
class StreamConfig:
    seed: int = 0
    dt: float = 0.25
    condition: str = "train"
    n_jobs: int = N_JOBS
    payload_classes: int = N_PAYLOADS
    duration_min: float = 1.0
    duration_max: float = 4.0
    gap_min: float = 0.5
    gap_max: float = 2.0
    window_min: float = 1.5
    window_max: float = 2.5
    cancel_probability: float = 0.20
    replace_probability: float = 0.30

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (self.n_jobs, self.payload_classes) != (N_JOBS, N_PAYLOADS):
            raise ValueError("this experiment has exactly two jobs and four payloads")


@dataclass(frozen=True)
class StreamBatch:
    """Fixed exogenous schedule, shaped [steps,batch,job].

    ``eligible`` marks final, uncanceled requests with a complete deadline
    window inside the stream. ``interrupted`` marks a command followed by a
    different job's event before its earliest possible completion. Neither is
    supplied to the model. Case IDs and the hash are shared across policies.
    """

    config: StreamConfig
    kind: torch.Tensor
    version: torch.Tensor
    payload: torch.Tensor
    duration: torch.Tensor
    window: torch.Tensor
    eligible: torch.Tensor
    interrupted: torch.Tensor
    request_id: torch.Tensor
    times: torch.Tensor
    case_ids: tuple[str, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        shape = tuple(self.kind.shape)
        if len(shape) != 3 or shape[2] != N_JOBS or min(shape[:2]) <= 0:
            raise ValueError("schedule tensors must have shape [steps,batch,2]")
        for name in ("version", "payload", "duration", "window", "eligible",
                     "interrupted", "request_id"):
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} does not match schedule shape")
        if tuple(self.times.shape) != (shape[0],):
            raise ValueError("times must contain one absolute timestamp per step")
        times = self.times.detach().cpu().numpy()
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("timestamps must be finite and strictly increasing")
        kinds = self.kind.detach().cpu().numpy()
        if ((kinds < NONE) | (kinds > CANCEL)).any():
            raise ValueError("unknown command kind")
        command = (kinds == NEW) | (kinds == REPLACE)
        duration = self.duration.detach().cpu().numpy()
        window = self.window.detach().cpu().numpy()
        if not np.isfinite(duration).all() or not np.isfinite(window).all():
            raise ValueError("durations and windows must be finite")
        if (duration[command] <= 0).any() or (window[command] < 0).any():
            raise ValueError("commands need positive duration and nonnegative window")
        digest = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True).encode())
        for name in ("kind", "version", "payload", "duration", "window",
                     "eligible", "interrupted", "request_id", "times"):
            value = getattr(self, name).detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(value.tobytes())
        fingerprint = digest.hexdigest()
        object.__setattr__(self, "fingerprint", fingerprint)
        if self.case_ids and len(self.case_ids) != shape[1]:
            raise ValueError("case_ids must contain one ID per independent stream")
        if not self.case_ids:
            object.__setattr__(self, "case_ids", tuple(
                f"{self.config.seed}:{fingerprint[:16]}:{i}" for i in range(shape[1])))

    @property
    def steps(self) -> int:
        return self.kind.shape[0]

    @property
    def batch_size(self) -> int:
        return self.kind.shape[1]

    def to(self, device: str | torch.device) -> "StreamBatch":
        return StreamBatch(self.config, **{
            name: getattr(self, name).to(device) for name in (
                "kind", "version", "payload", "duration", "window", "eligible",
                "interrupted", "request_id", "times")}, case_ids=self.case_ids)


def generate_stream(seed: int, batch_size: int, steps: int, dt: float = 0.25,
                    condition: str = "train", device: str | torch.device = "cpu"
                    ) -> StreamBatch:
    """Generate action-independent renewal schedules, without per-world hot loops.

    ``overlap`` increases concurrent arrivals; ``conflict`` increases pre-ready
    replacements/cancellations. ``long_gap`` uses longer processing durations;
    ``sparse`` increases idle arrival gaps. ``long`` keeps the training
    distribution and is intended for a longer ``steps`` argument.
    """
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("steps must be a positive integer")
    batch_size, steps = int(batch_size), int(steps)
    settings: dict[str, dict[str, float]] = {
        "train": {}, "long": {},
        "overlap": {"gap_min": 0.25, "gap_max": 0.75},
        "conflict": {"cancel_probability": 0.30, "replace_probability": 0.55},
        "long_gap": {"duration_min": 4.0, "duration_max": 8.0},
        "sparse": {"gap_min": 4.0, "gap_max": 12.0},
    }
    if condition not in settings:
        raise ValueError(f"unknown condition {condition!r}; choose {tuple(settings)}")
    config = StreamConfig(seed=int(seed), dt=float(dt), condition=condition,
                          **settings[condition])
    rng = np.random.default_rng(seed)
    shape = (steps, batch_size, N_JOBS)
    kind = np.zeros(shape, np.int64)
    version = np.zeros(shape, np.int64)
    payload = np.zeros(shape, np.int64)
    duration = np.zeros(shape, np.float32)
    window = np.zeros(shape, np.float32)
    eligible = np.zeros(shape, bool)
    request_id = np.full(shape, -1, np.int64)
    times = np.arange(steps, dtype=np.float64) * dt
    last_time = float(times[-1])
    rows = np.arange(batch_size)

    def put(indices, valid, job, event_kind, vers, values, durations, windows, elig):
        ix, bs = indices[valid], rows[valid]
        kind[ix, bs, job] = event_kind
        version[ix, bs, job] = vers[valid]
        payload[ix, bs, job] = values[valid]
        duration[ix, bs, job] = durations[valid]
        window[ix, bs, job] = windows[valid]
        eligible[ix, bs, job] = elig[valid]
        request_id[ix, bs, job] = ix * N_JOBS + job

    for job in range(N_JOBS):
        arrival = rng.integers(0, max(2, int(math.ceil(1.5 / dt))), batch_size)
        vers = rng.integers(0, N_VERSIONS, batch_size)
        # Independent renewal clocks prevent fixed stage/calendar labels.
        while (arrival < steps).any():
            d1 = rng.uniform(config.duration_min, config.duration_max, batch_size)
            w1 = rng.uniform(config.window_min, config.window_max, batch_size)
            d2 = rng.uniform(config.duration_min, config.duration_max, batch_size)
            w2 = rng.uniform(config.window_min, config.window_max, batch_size)
            values = rng.integers(0, N_PAYLOADS, batch_size)
            revised = (values + rng.integers(1, N_PAYLOADS, batch_size)) % N_PAYLOADS
            draw = rng.random(batch_size)
            cancel = draw < config.cancel_probability
            replace = (draw >= config.cancel_probability) & (
                draw < config.cancel_probability + config.replace_probability)
            modify = cancel | replace
            # Modifications precede even the immediate-start completion, so
            # eligibility is independent of a policy's actions.
            max_delta = np.maximum(1, np.ceil(d1 / dt).astype(np.int64) - 1)
            delta = np.maximum(1, np.floor(rng.uniform(0.25, 0.70, batch_size)
                                           * d1 / dt).astype(np.int64))
            delta = np.minimum(delta, max_delta)
            modified_at = arrival + delta
            final_deadline = np.where(replace, modified_at * dt + d2 + w2,
                                      np.where(cancel, modified_at * dt,
                                               arrival * dt + d1 + w1))
            valid = (arrival < steps) & (final_deadline <= last_time)
            original_eligible = ~modify
            put(arrival, valid, job, NEW, vers, values, d1, w1, original_eligible)
            zeros = np.zeros(batch_size)
            put(modified_at, valid & cancel, job, CANCEL, vers, values,
                zeros, zeros, np.zeros(batch_size, bool))
            next_vers = 1 - vers
            put(modified_at, valid & replace, job, REPLACE, next_vers, revised,
                d2, w2, np.ones(batch_size, bool))
            gap = rng.uniform(config.gap_min, config.gap_max, batch_size)
            arrival = np.ceil((final_deadline + gap) / dt).astype(np.int64)
            # A canceled request also advances the generation; the next NEW
            # must differ from the most recently issued public version.
            vers = np.where(replace, 1 - next_vers, 1 - vers)
    command = (kind == NEW) | (kind == REPLACE)
    interrupted = np.zeros(shape, bool)
    bs = np.broadcast_to(rows[None, :], (steps, batch_size))
    ix = np.broadcast_to(np.arange(steps)[:, None], (steps, batch_size))
    for job in range(N_JOBS):
        prefix = np.cumsum(kind[:, :, 1 - job] != NONE, axis=0)
        end = np.ceil((times[:, None] + duration[:, :, job]) / dt).astype(np.int64) - 1
        end = np.clip(end, 0, steps - 1)
        interrupted[:, :, job] = command[:, :, job] & (
            prefix[end, bs] - prefix[ix, bs] > 0)
    arrays = {"kind": kind, "version": version, "payload": payload,
              "duration": duration, "window": window, "eligible": eligible,
              "interrupted": interrupted, "request_id": request_id, "times": times}
    return StreamBatch(config, **{k: torch.from_numpy(v).to(device)
                                  for k, v in arrays.items()})


class WorldBatch:
    """Vectorized evaluator; initialize once per independent batch of long streams.

    Call observe(t), obtain oracle labels for loss if needed, then act(actions).
    observe applies expiration and exogenous commands before an action at the
    same timestamp. It returns a command pulse and one preceding receipt, not
    a current-state description. Calls must advance chronologically once each.
    """

    input_size = INPUT_SIZE
    n_actions = N_ACTIONS

    def __init__(self, stream: StreamBatch, device: str | torch.device | None = None):
        self.stream = stream if device is None else stream.to(device)
        self.config = stream.config
        self.device = self.stream.kind.device
        self.batch_size = stream.batch_size
        self.reset()

    def reset(self) -> None:
        """Explicit independent-stream boundary; never called by observe/act."""
        shape = (self.batch_size, N_JOBS)
        def zeros(dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        self.active, self.started = zeros(torch.bool), zeros(torch.bool)
        self.version, self.payload = zeros(torch.long), zeros(torch.long)
        self.duration, self.deadline = zeros(), zeros()
        self.ready_time = torch.full(shape, float("inf"), device=self.device)
        self.current_eligible, self.current_interrupted = zeros(torch.bool), zeros(torch.bool)
        self.had_error = zeros(torch.bool)
        self.request_id = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self._ack_action, self._ack_result = zeros(torch.long), zeros(torch.long)
        self._cancel_guard, self._cancel_violated = zeros(torch.bool), zeros(torch.bool)
        self._step, self._acted, self.now = -1, True, 0.0
        self.counts = {name: zeros(torch.long) for name in (
            "commands", "eligible", "completed_eligible", "clean_completed_eligible",
            "completed_all", "finish_attempted", "canceled",
            "canceled_correctly", "replaced", "expired", "attempted", "accepted",
            "rejected", "false_action", "early", "late", "wrong_payload", "stale",
            "before_begin", "duplicate_begin", "invalid_code", "interrupted_eligible",
            "interrupted_completed")}
        self.finish_delay_sum = zeros()
        self.last_result = zeros(torch.long)

    def observe(self, t: int) -> torch.Tensor:
        if t != self._step + 1 or not self._acted or not 0 <= t < self.stream.steps:
            raise ValueError("observe must advance once, after the preceding act")
        self._step, self._acted = t, False
        self.now = float(self.stream.times[t].item())
        expired = self.active & (self.now > self.deadline + 1e-6)
        self.counts["expired"] += expired
        self.active &= ~expired
        kind = self.stream.kind[t]
        command = (kind == NEW) | (kind == REPLACE)
        cancel = kind == CANCEL
        self.counts["commands"] += command
        self.counts["replaced"] += kind == REPLACE
        self.counts["eligible"] += command & self.stream.eligible[t]
        self.counts["interrupted_eligible"] += (
            command & self.stream.eligible[t] & self.stream.interrupted[t])
        self.counts["canceled"] += cancel
        self.counts["canceled_correctly"] += cancel
        self._cancel_guard = torch.where(command, False, self._cancel_guard) | cancel
        self._cancel_violated = torch.where(command | cancel, False, self._cancel_violated)
        matching_cancel = cancel & (self.stream.version[t] == self.version)
        self.active = torch.where(command, True, self.active & ~matching_cancel)
        self.started = torch.where(command | matching_cancel, False, self.started)
        for name in ("version", "payload", "duration", "request_id"):
            setattr(self, name, torch.where(command, getattr(self.stream, name)[t],
                                           getattr(self, name)))
        self.deadline = torch.where(command, self.now + self.stream.duration[t]
                                    + self.stream.window[t], self.deadline)
        self.ready_time = torch.where(command | matching_cancel, float("inf"), self.ready_time)
        self.current_eligible = torch.where(command, self.stream.eligible[t],
                                             self.current_eligible)
        self.current_interrupted = torch.where(command, self.stream.interrupted[t],
                                                self.current_interrupted)
        self.had_error = torch.where(command, False, self.had_error)
        event = kind != NONE
        features = torch.cat((
            F.one_hot(kind, 4)[..., 1:].float(),
            F.one_hot(self.stream.version[t], 2).float() * event[..., None],
            F.one_hot(self.stream.payload[t], N_PAYLOADS).float() * command[..., None],
            (self.stream.duration[t] * command)[..., None],
            (self.stream.window[t] * command)[..., None],
            F.one_hot(self._ack_action.clamp(0, N_ACTIONS - 1), N_ACTIONS).float()
            * (self._ack_action != 0)[..., None],
            torch.stack((self._ack_result == 1, self._ack_result == -1), -1).float(),
        ), -1)
        self._ack_action.zero_()
        self._ack_result.zero_()
        return features.flatten(1)

    def oracle(self) -> torch.Tensor:
        """Evaluator labels only; no future events or schedule lookup."""
        if self._step < 0 or self._acted:
            raise ValueError("oracle labels are available between observe and act")
        feasible_begin = self.active & ~self.started & (
            self.now + self.duration <= self.deadline + 1e-6)
        finish = self.active & self.started & (self.now >= self.ready_time - 1e-6) & (
            self.now <= self.deadline + 1e-6)
        return torch.where(feasible_begin, 1 + self.version,
                           torch.where(finish, 3 + N_PAYLOADS * self.version + self.payload, 0))

    def act(self, actions: torch.Tensor) -> None:
        if self._step < 0 or self._acted:
            raise ValueError("act must follow exactly one observe")
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if tuple(actions.shape) != (self.batch_size, N_JOBS):
            raise ValueError("actions must have shape [batch,2]")
        attempted = actions != 0
        begin = (actions == 1) | (actions == 2)
        finish = (actions >= 3) & (actions < N_ACTIONS)
        valid = begin | finish | ~attempted
        action_version = torch.where(begin, actions - 1,
                                      torch.div(actions - 3, N_PAYLOADS, rounding_mode="floor"))
        action_payload = (actions - 3).remainder(N_PAYLOADS)
        same_version = action_version == self.version
        accepted_begin = begin & self.active & same_version & ~self.started
        accepted_finish = (finish & self.active & same_version & self.started
                           & (action_payload == self.payload)
                           & (self.now >= self.ready_time - 1e-6)
                           & (self.now <= self.deadline + 1e-6))
        accepted = accepted_begin | accepted_finish
        rejected = attempted & ~accepted
        categories = {
            "attempted": attempted, "accepted": accepted, "rejected": rejected,
            "false_action": attempted & ~self.active,
            "stale": attempted & valid & ~same_version,
            "invalid_code": attempted & ~valid,
            "early": finish & self.active & same_version & self.started
                     & (self.now < self.ready_time - 1e-6),
            "late": finish & same_version & (self.request_id >= 0)
                    & (self.now > self.deadline + 1e-6),
            "wrong_payload": finish & self.active & same_version
                             & (action_payload != self.payload),
            "before_begin": finish & self.active & same_version & ~self.started,
            "duplicate_begin": begin & self.active & same_version & self.started,
            "completed_all": accepted_finish,
            "finish_attempted": finish,
            "completed_eligible": accepted_finish & self.current_eligible,
            "clean_completed_eligible": accepted_finish & self.current_eligible & ~self.had_error,
            "interrupted_completed": accepted_finish & self.current_eligible
                                     & self.current_interrupted,
        }
        for name, mask in categories.items():
            self.counts[name] += mask
        self.had_error |= rejected & self.active
        cancellation_violation = self._cancel_guard & ~self._cancel_violated & attempted
        self.counts["canceled_correctly"] -= cancellation_violation.long()
        self._cancel_violated |= cancellation_violation
        self.finish_delay_sum += torch.where(accepted_finish,
                                             self.now - self.ready_time, 0.0)
        self.ready_time = torch.where(accepted_begin, self.now + self.duration,
                                      self.ready_time)
        self.started = (self.started | accepted_begin) & ~accepted_finish
        self.active &= ~accepted_finish
        self._ack_action = torch.where(attempted, actions, 0)
        self._ack_result = accepted.long() - rejected.long()
        self.last_result = self._ack_result.clone()
        self._acted = True

    def metrics(self, per_stream: bool = False) -> dict[str, Any]:
        """Aggregate once outside the hot path; diagnostic error counts overlap."""
        result: dict[str, Any] = {name: int(value.sum().item())
                                  for name, value in self.counts.items()}
        result["completion_rate"] = (result["completed_eligible"] / result["eligible"]
                                      if result["eligible"] else 0.0)
        result["clean_completion_rate"] = (
            result["clean_completed_eligible"] / result["eligible"]
            if result["eligible"] else 0.0)
        result["finish_precision"] = (result["completed_all"] / result["finish_attempted"]
                                       if result["finish_attempted"] else 0.0)
        result["interrupted_completion_rate"] = (
            result["interrupted_completed"] / result["interrupted_eligible"]
            if result["interrupted_eligible"] else 0.0)
        completed = self.counts["completed_all"].sum(0).cpu().tolist()
        delays = self.finish_delay_sum.sum(0).cpu().tolist()
        result["per_job_finish_delay"] = [value / n if n else None
                                           for value, n in zip(delays, completed)]
        result["world_hash"] = self.stream.fingerprint
        if per_stream:
            result["case_ids"] = list(self.stream.case_ids)
            result["per_stream"] = {name: value.sum(1).cpu().tolist()
                                     for name, value in self.counts.items()}
        return result

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Evaluator-only latent state for mechanism tests; never an observation."""
        return {name: getattr(self, name).clone() for name in (
            "active", "started", "version", "payload", "duration", "deadline",
            "ready_time", "request_id", "current_eligible", "current_interrupted", "had_error")}
