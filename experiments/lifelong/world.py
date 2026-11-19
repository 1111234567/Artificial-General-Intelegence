"""Contextual contracts on the frozen, action-independent continuity stream.

The payload pulse is always the original raw symbol.  A four-way context pulse
selects a real target permutation; only the evaluator's private target is
transformed.  Context is assigned once at NEW, survives a pending modification,
and is absent from idle observations.  Changing future policy never resets any
world, receipt, scheduler, clock, or random stream.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Integral
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from experiments.continuity.world import (
    CANCEL, NEW, NONE, REPLACE, N_ACTIONS, N_JOBS, N_PAYLOADS,
    FORMAT_VERSION as BASE_FORMAT_VERSION, OnlineWorld, _GOLDEN, _mix64,
)

FORMAT_VERSION = "lifelong-context-world-v1"
N_FAMILIES = 4
FEATURES_PER_JOB = 28
INPUT_SIZE = N_JOBS * FEATURES_PER_JOB
FAMILY_MAPPINGS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 2, 3, 0))
FEATURE_LAYOUT = {
    "kind": (0, 3), "version": (3, 5), "payload": (5, 9),
    "duration": (9, 10), "window": (10, 11),
    "previous_action": (11, 22), "previous_result": (22, 24),
    "context": (24, 28),
}
TRAINING_COVERAGE = {
    "narrow": {"duration_range": (1.0, 4.0), "gap_range": (0.5, 2.0)},
    "wide": {"duration_range": (1.0, 16.0), "gap_range": (0.5, 32.0)},
}


def _family(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < N_FAMILIES:
        raise ValueError("family must be an integer in 0..3")
    return int(value)


def _allowed(values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(_family(value) for value in values)
    except TypeError as error:
        raise ValueError("family policy must be a nonempty sequence of distinct families") from error
    if not result or len(set(result)) != len(result):
        raise ValueError("family policy must be a nonempty sequence of distinct families")
    return result


class ContextWorld(OnlineWorld):
    """OnlineWorld plus independently sampled, persistent per-contract context.

    ``future_family`` fixes future NEW contexts. ``family_allowed`` instead
    samples uniformly from that sequence.  A context RNG word is consumed for
    every NEW under both policies, so switches do not shift subsequent draws.
    The old scheduler and its raw pending payloads are left untouched.
    """

    features_per_job = FEATURES_PER_JOB

    def __init__(self, *args, future_family: int = 0,
                 family_allowed: Iterable[int] | None = None, **kwargs):
        policy = (_family(future_family),) if family_allowed is None else _allowed(family_allowed)
        super().__init__(*args, **kwargs)
        self.input_size = self.n_jobs * FEATURES_PER_JOB
        self.family_allowed = policy
        self.future_family = policy[0] if len(policy) == 1 else None
        self.context_history = [{"tick": -1, "allowed": list(policy)}]
        shape = (self.batch_size, self.n_jobs)
        actors = np.asarray(self.actor_ids, dtype=np.uint64)[:, None]
        jobs = np.arange(self.n_jobs, dtype=np.uint64)[None, :]
        with np.errstate(over="ignore"):
            self._context_rng_state = _mix64(np.uint64(self.seed % 2**64) ^
                _mix64(actors + np.uint64(0xA24BAED4963EE407)) ^
                _mix64(jobs + np.uint64(0x9FB21C651E98DF25)))
        self._assigned_context = np.full(shape, -1, dtype=np.int64)
        self.current_family = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.event_family = torch.full_like(self.current_family, -1)
        self.matured_family = torch.full_like(self.current_family, -1)
        self.raw_payload = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.raw_event_payload = torch.zeros_like(self.raw_payload)
        self.last_external_tick = torch.full_like(self.current_family, -1)
        self.last_external_time = torch.full(shape, -1.0, dtype=torch.float64, device=self.device)
        self.family_new_counts = torch.zeros((*shape, N_FAMILIES), dtype=torch.long, device=self.device)
        self.family_replace_counts = torch.zeros_like(self.family_new_counts)
        self.family_last_command_tick = torch.full((N_FAMILIES,), -1, dtype=torch.long, device=self.device)
        self.family_last_command_time = torch.full((N_FAMILIES,), -1.0, dtype=torch.float64, device=self.device)
        self._event_hash = hashlib.sha256(
            FORMAT_VERSION.encode() + bytes.fromhex(self._event_hash)).hexdigest()

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(f"{FORMAT_VERSION}:{self.seed}:{actor}" for actor in self.actor_ids)

    @property
    def current_context(self) -> torch.Tensor:
        return self.current_family

    @property
    def event_context(self) -> torch.Tensor:
        return self.event_family

    def set_family(self, family: int) -> None:
        self.set_family_policy((_family(family),))

    def set_family_policy(self, allowed: Iterable[int]) -> None:
        """Change only future NEW context; do not touch pending contracts."""
        if not self._acted:
            raise ValueError("family changes must follow a completed action")
        policy = _allowed(allowed)
        self.family_allowed = policy
        self.future_family = policy[0] if len(policy) == 1 else None
        self.context_history.append({"tick": self.tick, "allowed": list(policy)})

    def _events(self) -> dict[str, np.ndarray]:
        event = super()._events()
        new = event["kind"] == NEW
        if new.any():
            with np.errstate(over="ignore"):
                self._context_rng_state[new] += _GOLDEN
            draws = (_mix64(self._context_rng_state[new]) >> np.uint64(11)).astype(
                np.float64) * (1.0 / 2**53)
            choices = np.floor(draws * len(self.family_allowed)).astype(np.int64)
            self._assigned_context[new] = np.asarray(self.family_allowed, dtype=np.int64)[choices]
        external = event["kind"] != NONE
        command = new | (event["kind"] == REPLACE)
        contexts = np.where(external, self._assigned_context, -1)
        if np.any(external & (contexts < 0)):
            raise AssertionError("modification has no assigned NEW context")
        self.event_family = torch.from_numpy(contexts.copy()).to(self.device)
        self.raw_event_payload = torch.from_numpy(event["payload"].copy()).to(self.device)
        command_tensor = torch.from_numpy(command).to(self.device)
        self.raw_payload = torch.where(command_tensor, self.raw_event_payload, self.raw_payload)
        self.current_family = torch.where(command_tensor, self.event_family, self.current_family)
        if external.any():
            mask = torch.from_numpy(external).to(self.device)
            self.last_external_tick = torch.where(mask, self.tick, self.last_external_tick)
            self.last_external_time = torch.where(mask, self.now, self.last_external_time)
            # Include context and raw symbols, even where the selected mappings agree.
            digest = hashlib.sha256(bytes.fromhex(self._event_hash))
            digest.update(FORMAT_VERSION.encode())
            digest.update(contexts.tobytes())
            digest.update(event["payload"].tobytes())
            self._event_hash = digest.hexdigest()
        mapping = np.asarray(FAMILY_MAPPINGS, dtype=np.int64)
        event["payload"][external] = mapping[contexts[external], event["payload"][external]]
        onehot = F.one_hot(self.event_family.clamp_min(0), N_FAMILIES)
        self.family_new_counts += onehot * torch.from_numpy(new).to(self.device)[..., None]
        self.family_replace_counts += onehot * (
            torch.from_numpy(event["kind"] == REPLACE).to(self.device)[..., None])
        observed_family = (onehot * command_tensor[..., None]).sum((0, 1)) > 0
        self.family_last_command_tick = torch.where(
            observed_family, self.tick, self.family_last_command_tick)
        self.family_last_command_time = torch.where(
            observed_family, self.now, self.family_last_command_time)
        return event

    def observe(self, tick: int | None = None) -> torch.Tensor:
        base = super().observe(tick).reshape(self.batch_size, self.n_jobs, 24)
        command = (self.event_kind == NEW) | (self.event_kind == REPLACE)
        # Base evaluator consumes mapped targets; public observations recover raw symbols.
        base[..., 5:9] = F.one_hot(self.raw_event_payload, N_PAYLOADS).float() * command[..., None]
        context = F.one_hot(self.event_family.clamp_min(0), N_FAMILIES).float() * command[..., None]
        return torch.cat((base, context), -1).flatten(1)

    def _mature(self) -> None:
        # Capture before a simultaneous NEW can overwrite the retained context.
        super()._mature()
        self.matured_family = torch.where(self.matured_mask, self.current_family, -1)

    def family_exposures(self) -> dict[str, Any]:
        new = self.family_new_counts.sum((0, 1)).cpu().tolist()
        replacement = self.family_replace_counts.sum((0, 1)).cpu().tolist()
        return {"new": new, "replace": replacement,
                "commands": [a + b for a, b in zip(new, replacement)]}

    def family_recency(self) -> dict[str, Any]:
        """Last actual NEW/REPLACE across all lanes, retained after withdrawal."""
        return {"last_command_tick": self.family_last_command_tick.cpu().tolist(),
                "last_command_time": self.family_last_command_time.cpu().tolist()}

    def metrics(self, per_stream: bool = False) -> dict[str, Any]:
        result = super().metrics(per_stream=per_stream)
        result.update(family_allowed=list(self.family_allowed), future_family=self.future_family,
                      family_exposures=self.family_exposures(), family_recency=self.family_recency())
        return result

    def snapshot(self) -> dict[str, torch.Tensor]:
        result = super().snapshot()
        for name in ("current_family", "event_family", "matured_family", "raw_payload", "raw_event_payload",
                     "last_external_tick", "last_external_time"):
            result[name] = getattr(self, name).clone()
        return result

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["format"] = FORMAT_VERSION
        result["context"] = {
            "mappings": [list(row) for row in FAMILY_MAPPINGS],
            "family_allowed": list(self.family_allowed), "future_family": self.future_family,
            "history": deepcopy(self.context_history),
            "rng_state": torch.from_numpy(self._context_rng_state.view(np.int64).copy()),
            "assigned": torch.from_numpy(self._assigned_context.copy()),
            "tensors": {name: getattr(self, name).detach().cpu().clone() for name in (
                "current_family", "event_family", "matured_family", "raw_payload", "raw_event_payload",
                "last_external_tick", "last_external_time", "family_new_counts", "family_replace_counts",
                "family_last_command_tick", "family_last_command_time")},
        }
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Strictly restore every context field; reject incomplete/legacy checkpoints."""
        required = {"format", "spec", "regime", "regime_id", "regime_history", "tick", "acted",
                    "now", "elapsed", "event_hash", "scheduler", "tensors", "counts", "context"}
        if set(state) != required or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete context world state")
        spec = state["spec"]
        if set(spec) != {"seed", "batch_size", "dt", "n_jobs", "active_jobs", "actor_ids"}:
            raise ValueError("malformed context world specification")
        if (spec["batch_size"], spec["n_jobs"]) != (self.batch_size, self.n_jobs):
            raise ValueError("world checkpoint batch and job shapes must match")
        # Constructor validation is deterministic and uses only private RNGs.
        validation = type(self)(**spec, device=self.device)
        if (isinstance(state["tick"], bool) or not isinstance(state["tick"], Integral)
                or state["tick"] < -1 or type(state["acted"]) is not bool
                or not all(math.isfinite(float(state[key])) and state[key] >= 0
                           for key in ("now", "elapsed"))):
            raise ValueError("invalid world causal boundary")
        if not isinstance(state["event_hash"], str) or len(state["event_hash"]) != 64:
            raise ValueError("invalid event hash")
        try:
            bytes.fromhex(state["event_hash"])
        except ValueError as error:
            raise ValueError("invalid event hash") from error
        context = state["context"]
        context_keys = {"mappings", "family_allowed", "future_family", "history", "rng_state", "assigned", "tensors"}
        names = {"current_family", "event_family", "matured_family", "raw_payload", "raw_event_payload",
                 "last_external_tick", "last_external_time", "family_new_counts", "family_replace_counts",
                 "family_last_command_tick", "family_last_command_time"}
        if set(context) != context_keys or set(context["tensors"]) != names:
            raise ValueError("unsupported or incomplete context fields")
        if context["mappings"] != [list(row) for row in FAMILY_MAPPINGS]:
            raise ValueError("checkpoint context rule mappings differ")
        policy = _allowed(context["family_allowed"])
        if context["future_family"] != (policy[0] if len(policy) == 1 else None):
            raise ValueError("inconsistent future family policy")
        history = context["history"]
        if not isinstance(history, list) or not history:
            raise ValueError("missing context policy history")
        previous_tick = -1
        for item in history:
            if (not isinstance(item, dict) or set(item) != {"tick", "allowed"}
                    or isinstance(item["tick"], bool) or not isinstance(item["tick"], Integral)
                    or not previous_tick <= item["tick"] <= state["tick"]):
                raise ValueError("invalid context policy history")
            _allowed(item["allowed"])
            previous_tick = item["tick"]
        if history[0]["tick"] != -1 or tuple(history[-1]["allowed"]) != policy:
            raise ValueError("inconsistent context policy history")
        shape = (self.batch_size, self.n_jobs)
        for name in ("rng_state", "assigned"):
            value = context[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.int64:
                raise ValueError(f"malformed context {name}")
        if torch.any((context["assigned"] < -1) | (context["assigned"] >= N_FAMILIES)):
            raise ValueError("invalid assigned context")
        for name in names:
            value = context["tensors"][name]
            expected = ((N_FAMILIES,) if name.startswith("family_last_command_") else
                        (*shape, N_FAMILIES) if name.endswith("counts") else shape)
            dtype = torch.float64 if name in ("last_external_time", "family_last_command_time") else torch.int64
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or value.dtype != dtype:
                raise ValueError(f"malformed context tensor {name}")
            if name in ("current_family", "event_family", "matured_family") and torch.any((value < -1) | (value >= N_FAMILIES)):
                raise ValueError("invalid family tensor")
            if name in ("raw_payload", "raw_event_payload") and torch.any((value < 0) | (value >= N_PAYLOADS)):
                raise ValueError("invalid raw payload tensor")
            if name.endswith("counts") and torch.any(value < 0):
                raise ValueError("invalid family exposure counts")
        recency_ticks = context["tensors"]["family_last_command_tick"]
        recency_times = context["tensors"]["family_last_command_time"]
        if (torch.any((recency_ticks < -1) | (recency_ticks > state["tick"]))
                or not torch.isfinite(recency_times).all()
                or torch.any((recency_times < -1) | (recency_times > state["now"]))
                or not torch.equal(recency_ticks == -1, recency_times == -1)):
            raise ValueError("invalid per-family command recency")
        command_counts = (context["tensors"]["family_new_counts"] +
                          context["tensors"]["family_replace_counts"]).sum((0, 1))
        if not torch.equal(command_counts == 0, recency_ticks == -1):
            raise ValueError("command recency differs from actual family exposure")
        if set(state["scheduler"]) != set(self._scheduler_arrays) or set(state["tensors"]) != set(self._state_tensors) or set(state["counts"]) != set(self._count_names):
            raise ValueError("incomplete base world checkpoint")
        template = validation.state_dict()
        for section in ("scheduler", "tensors", "counts"):
            for name, value in state[section].items():
                expected = template[section][name]
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                        or value.dtype != expected.dtype):
                    raise ValueError(f"malformed {section} tensor {name}")
        assigned = context["assigned"]
        current = context["tensors"]["current_family"]
        events = context["tensors"]["event_family"]
        kinds = state["tensors"]["event_kind"]
        if (not torch.equal(assigned.cpu(), current.cpu())
                or torch.any((kinds == NONE) & (events != -1))
                or torch.any((kinds != NONE) & (events != current))):
            raise ValueError("inconsistent assigned/current/event context")
        raw = context["tensors"]["raw_payload"]
        mapped = torch.as_tensor(FAMILY_MAPPINGS, dtype=torch.long)[current.cpu().clamp_min(0), raw.cpu()]
        if torch.any((current.cpu() >= 0) & (state["tensors"]["payload"].cpu() != mapped)):
            raise ValueError("private target differs from the saved raw context rule")
        base = dict(state)
        base.pop("context")
        base["format"] = BASE_FORMAT_VERSION
        super().load_state_dict(base)
        self.family_allowed, self.future_family = policy, context["future_family"]
        self.context_history = deepcopy(context["history"])
        self._context_rng_state = context["rng_state"].detach().cpu().numpy().copy().view(np.uint64)
        self._assigned_context = context["assigned"].detach().cpu().numpy().copy()
        for name in names:
            setattr(self, name, context["tensors"][name].detach().to(self.device).clone())

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device: str | torch.device = "cpu") -> "ContextWorld":
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported context world state format")
        world = cls(**state["spec"], device=device)
        world.load_state_dict(state)
        return world

    def clone(self, *, device: str | torch.device | None = None) -> "ContextWorld":
        return self.from_state_dict(self.state_dict(), device=self.device if device is None else device)
