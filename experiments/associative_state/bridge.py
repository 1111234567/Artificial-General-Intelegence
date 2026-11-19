"""Public-receipt binding and a symbol adapter for the frozen GRU executor.

This is an explicitly hand-written binder, not a policy that learned binding.
The policy still receives the existing public observations.  Neither the binder
nor ``translate`` receives the world's private target or an oracle action.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from experiments.lifelong.world import (
    CANCEL, NEW, NONE, REPLACE, ContextWorld, FAMILY_MAPPINGS,
    FORMAT_VERSION as CONTEXT_FORMAT,
)

FEATURES_PER_JOB = 28
BINDER_FORMAT = "associative-public-receipt-binder-v1"
WORLD_FORMAT = "associative-random-mapping-world-v2"


def _contexts(values: Iterable[int]) -> tuple[int, int]:
    result = tuple(values)
    if (len(result) != 2 or len(set(result)) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            or not 0 <= value < 4 for value in result)):
        raise ValueError("contexts must contain two distinct family IDs in 0..3")
    return tuple(map(int, result))


def _permutations(values: Any, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if tuple(tensor.shape) != shape or tensor.dtype == torch.bool:
        raise ValueError(f"mappings must have shape {shape}")
    integer = tensor.to(dtype=torch.long, device="cpu")
    if not torch.equal(tensor.detach().cpu(), integer):
        raise ValueError("mapping entries must be integers")
    if not torch.equal(integer.sort(-1).values,
                       torch.arange(4).expand(shape)):
        raise ValueError("each mapping must be a permutation of 0..3")
    return integer.clone()


class PublicReceiptBinder:
    """Remember public cue identity until its successful FINISH receipt.

    ``consume`` must be called once per observation, before the current action.
    Receipts belong to the preceding action and are processed before the new
    cue, including when both occur in the same job's observation.  Four stored
    scalars per job (raw, family, version, valid) are extra controller state and
    must be counted in addition to the adapter's numeric fast-state budget.
    """

    def __init__(self, batch_size: int, n_jobs: int = 2,
                 contexts: Iterable[int] = (0, 3), device="cpu"):
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or value < 1 for value in (batch_size, n_jobs)):
            raise ValueError("batch_size and n_jobs must be positive integers")
        self.batch_size, self.n_jobs = int(batch_size), int(n_jobs)
        self.contexts = _contexts(contexts)
        self.device = torch.device(device)
        shape = (self.batch_size, self.n_jobs)
        self.raw = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.family = torch.zeros_like(self.raw)
        self.version = torch.zeros_like(self.raw)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def consume(self, obs: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if tuple(obs.shape) != (self.batch_size, self.n_jobs * FEATURES_PER_JOB):
            raise ValueError("observation shape differs from binder batch/jobs")
        public = obs.detach().to(self.device).reshape(
            self.batch_size, self.n_jobs, FEATURES_PER_JOB)
        action = public[..., 11:22].argmax(-1)
        finish = action >= 3
        # continuity.world.observe stacks accepted first, rejected second.
        success = (public[..., 22] > .5) & (public[..., 23] < .5)
        receipt_version = torch.div(action - 3, 4, rounding_mode="floor")
        matched = self.valid & finish & success & (receipt_version == self.version)
        family_index = (self.family == self.contexts[1]).long()
        keys = torch.where(matched, 4 * family_index + self.raw, 0)
        values = torch.where(matched, (action - 3).remainder(4), 0)
        writes = [(keys[:, job].clone(), values[:, job].clone(), matched[:, job].clone())
                  for job in range(self.n_jobs)]

        # Finish the previous request before a simultaneous NEW/REPLACE pulse.
        self.valid &= ~matched
        command = (public[..., NEW - 1] > .5) | (public[..., REPLACE - 1] > .5)
        cancel = public[..., CANCEL - 1] > .5
        cue_family = public[..., 24:28].argmax(-1)
        allowed = (cue_family == self.contexts[0]) | (cue_family == self.contexts[1])
        if torch.any(command & ~allowed):
            raise ValueError("command family is outside the binder contexts")
        self.valid = (self.valid & ~cancel) | command
        self.raw = torch.where(command, public[..., 5:9].argmax(-1), self.raw)
        self.family = torch.where(command, cue_family, self.family)
        self.version = torch.where(command, public[..., 3:5].argmax(-1), self.version)
        for name in ("raw", "family", "version"):
            setattr(self, name, torch.where(self.valid, getattr(self, name), 0))
        return writes

    def state_report(self) -> dict[str, Any]:
        return dict(kind="hand_written_public_receipt_binder", learned=False,
                    extra_state_scalars_per_actor=4 * self.n_jobs,
                    extra_state_bytes_per_actor=(3 * 8 + 1) * self.n_jobs,
                    fields=["raw:int64", "family:int64", "version:int64", "valid:bool"],
                    included_in_associative_matrix_budget=False)

    def state_dict(self) -> dict[str, Any]:
        return dict(format=BINDER_FORMAT, batch_size=self.batch_size,
                    n_jobs=self.n_jobs, contexts=list(self.contexts),
                    tensors={name: getattr(self, name).detach().cpu().clone()
                             for name in ("raw", "family", "version", "valid")})

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (set(state) != {"format", "batch_size", "n_jobs", "contexts", "tensors"}
                or state["format"] != BINDER_FORMAT
                or (state["batch_size"], state["n_jobs"]) != (self.batch_size, self.n_jobs)
                or _contexts(state["contexts"]) != self.contexts
                or set(state["tensors"]) != {"raw", "family", "version", "valid"}):
            raise ValueError("incompatible binder checkpoint")
        for name, value in state["tensors"].items():
            dtype = torch.bool if name == "valid" else torch.long
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                    or tuple(value.shape) != (self.batch_size, self.n_jobs)):
                raise ValueError(f"malformed binder tensor {name}")
        tensors = state["tensors"]
        family = tensors["family"]
        if (torch.any((tensors["raw"] < 0) | (tensors["raw"] >= 4))
                or torch.any((tensors["version"] < 0) | (tensors["version"] >= 2))
                or torch.any(tensors["valid"] & (family != self.contexts[0])
                             & (family != self.contexts[1]))):
            raise ValueError("invalid binder cue identity")
        for name, value in tensors.items():
            setattr(self, name, value.detach().to(self.device).clone())

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device="cpu") -> "PublicReceiptBinder":
        binder = cls(state["batch_size"], state["n_jobs"], state["contexts"], device)
        binder.load_state_dict(state)
        return binder


def translate(obs: torch.Tensor, pred_values: torch.Tensor,
              original_mappings=FAMILY_MAPPINGS) -> torch.Tensor:
    """Replace only command payloads with the executor's canonical raw symbol.

    The predicted values are desired FINISH payloads, not raw symbols.  The
    frozen executor still sees the original context, hence the inverse of its
    original mapping is necessary to avoid applying the context rule twice.
    """
    if obs.ndim != 2 or obs.shape[1] % FEATURES_PER_JOB:
        raise ValueError("obs must have shape [batch,n_jobs*28]")
    batch, jobs = obs.shape[0], obs.shape[1] // FEATURES_PER_JOB
    predictions = torch.as_tensor(pred_values, device=obs.device)
    if tuple(predictions.shape) != (batch, jobs):
        raise ValueError("pred_values must have shape [batch,n_jobs]")
    public = obs.reshape(batch, jobs, FEATURES_PER_JOB)
    command = (public[..., NEW - 1] > .5) | (public[..., REPLACE - 1] > .5)
    integer = predictions.long()
    if torch.any(command & ((predictions != integer) | (integer < 0) | (integer >= 4))):
        raise ValueError("command predictions must be payload integers in 0..3")
    inverse = _permutations(original_mappings, (4, 4)).argsort(-1).to(obs.device)
    family = public[..., 24:28].argmax(-1)
    canonical = inverse[family, integer.clamp(0, 3)]
    result = obs.clone().reshape(batch, jobs, FEATURES_PER_JOB)
    result[..., 5:9] = torch.where(command[..., None],
                                   F.one_hot(canonical, 4).to(obs.dtype),
                                   public[..., 5:9])
    return result.reshape_as(obs)


class RandomMappingWorld(ContextWorld):
    """ContextWorld with private per-actor permutations for two contexts.

    Only evaluator targets change.  Public raw/context pulses and the external
    schedule remain identical, and global FAMILY_MAPPINGS is never modified.
    ``mappings`` is environment configuration, never a controller input.
    """

    def __init__(self, *args, mappings=None, contexts: Iterable[int] = (0, 3), **kwargs):
        self.mapping_contexts = _contexts(contexts)
        kwargs.setdefault("n_jobs", 2)
        if "family_allowed" not in kwargs and "future_family" not in kwargs:
            kwargs["family_allowed"] = self.mapping_contexts
        super().__init__(*args, **kwargs)
        if any(family not in self.mapping_contexts for family in self.family_allowed):
            raise ValueError("family policy is outside random-mapping contexts")
        if mappings is None:
            mappings = torch.arange(4).expand(self.batch_size, 2, 4)
        self.private_mappings = _permutations(mappings, (self.batch_size, 2, 4))
        # A rule revision affects future commands.  Each pending request keeps
        # the complete rule row applied at its most recent NEW/REPLACE.
        self.applied_mappings = torch.arange(4).expand(self.batch_size, self.n_jobs, 4).clone()

    def set_mappings(self, mappings) -> None:
        """Change future NEW/REPLACE targets without rewriting pending work.

        This consumes no random draws and changes no public pulse, receipt,
        clock, pending payload, or already-issued command.  A later REPLACE is
        a new command and therefore uses the newly selected table.
        """
        self.private_mappings = _permutations(mappings, (self.batch_size, 2, 4))

    def set_family_policy(self, allowed: Iterable[int]) -> None:
        allowed = tuple(allowed)
        if any(family not in self.mapping_contexts for family in allowed):
            raise ValueError("family policy is outside random-mapping contexts")
        super().set_family_policy(allowed)

    def _events(self) -> dict[str, np.ndarray]:
        event = super()._events()
        external = event["kind"] != NONE
        if external.any():
            families = self.event_family.detach().cpu().numpy()
            indices = (families == self.mapping_contexts[1]).astype(np.int64)
            actors = np.broadcast_to(np.arange(self.batch_size)[:, None], external.shape)
            raw = self.raw_event_payload.detach().cpu().numpy()
            command = torch.from_numpy((event["kind"] == NEW) | (event["kind"] == REPLACE))
            new_rows = self.private_mappings[torch.from_numpy(actors.copy()), torch.from_numpy(indices)]
            self.applied_mappings = torch.where(command[..., None], new_rows, self.applied_mappings)
            selected = self.applied_mappings.gather(-1, torch.from_numpy(raw)[..., None])[..., 0]
            event["payload"][external] = selected.numpy()[external]
        return event

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["format"] = WORLD_FORMAT
        state["random_mapping"] = dict(contexts=list(self.mapping_contexts),
                                       mappings=self.private_mappings.clone(),
                                       applied_mappings=self.applied_mappings.clone())
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != WORLD_FORMAT or "random_mapping" not in state:
            raise ValueError("unsupported random-mapping world checkpoint")
        if (state["spec"]["batch_size"], state["spec"]["n_jobs"]) != (self.batch_size, self.n_jobs):
            raise ValueError("world checkpoint batch and job shapes must match")
        extra = state["random_mapping"]
        if set(extra) != {"contexts", "mappings", "applied_mappings"}:
            raise ValueError("malformed random-mapping specification")
        contexts = _contexts(extra["contexts"])
        mappings = _permutations(extra["mappings"], (self.batch_size, 2, 4))
        applied = _permutations(extra["applied_mappings"], (self.batch_size, self.n_jobs, 4))
        current = state["context"]["tensors"]["current_family"].cpu()
        raw = state["context"]["tensors"]["raw_payload"].cpu()
        assigned = current >= 0
        if (any(family not in contexts for family in state["context"]["family_allowed"])
                or torch.any(assigned & (current != contexts[0]) & (current != contexts[1]))):
            raise ValueError("saved family is outside random-mapping contexts")
        expected = applied.gather(-1, raw[..., None])[..., 0]
        saved_payload = state["tensors"]["payload"]
        if torch.any(assigned & (saved_payload.cpu() != expected)):
            raise ValueError("private target differs from saved random mapping")

        # Let the unchanged ContextWorld validate and restore the full causal
        # state.  Its one fixed-rule invariant is checked against a canonical
        # validation copy, then the already-validated dynamic target is restored.
        base = deepcopy(state)
        base.pop("random_mapping")
        base["format"] = CONTEXT_FORMAT
        original = torch.as_tensor(FAMILY_MAPPINGS)[current.clamp_min(0), raw]
        base["tensors"]["payload"] = torch.where(assigned, original, saved_payload.cpu())
        restored = ContextWorld.from_state_dict(base, device=self.device)
        self.__dict__.update(restored.__dict__)
        self.payload = saved_payload.detach().to(self.device).clone()
        self.mapping_contexts, self.private_mappings = contexts, mappings
        self.applied_mappings = applied

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device="cpu") -> "RandomMappingWorld":
        extra = state["random_mapping"]
        world = cls(**state["spec"], mappings=extra["mappings"],
                    contexts=extra["contexts"], family_allowed=state["context"]["family_allowed"],
                    device=device)
        world.load_state_dict(state)
        return world

    def clone(self, *, device=None) -> "RandomMappingWorld":
        return self.from_state_dict(self.state_dict(), device=self.device if device is None else device)
