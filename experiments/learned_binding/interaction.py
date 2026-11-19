"""Continuous own-action interaction and native request-ledger evaluation.

Rollouts use no gradients; PG callers replay their archived public observations
through the native model with gradients enabled.  No oracle is called unless a
caller explicitly requests the separate teacher-support diagnostic.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.bridge import PublicReceiptBinder, RandomMappingWorld, translate
from experiments.associative_state.shared import SharedAssociationMemory
from experiments.learned_binding import common, data
from experiments.learned_binding.model import soft_translate


KINDS = ("manual_hard", "manual_soft", "learned")
GROUPS = ("all", "A", "B", "A_conflict", "B_conflict", "A_shared", "B_shared")
ARRAY_FIELDS = ("obs", "actions", "prediction", "event_kind", "event_rid",
                "event_eligible", "event_family", "raw", "target", "mature_mask",
                "mature_rid", "mature_done", "mature_strict", "active", "gate",
                "effective_gate", "read_probs")


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return deepcopy(value)


def _ticks(ticks):
    if isinstance(ticks, bool) or not isinstance(ticks, Integral) or ticks < 1:
        raise ValueError("ticks must be a positive integer")
    return int(ticks)


def _public_masks(obs):
    public = obs.reshape(len(obs), 2, 28)
    action = public[..., 11:22].argmax(-1)
    successful_finish = ((action >= 3) & (public[..., 22] > .5)
                         & (public[..., 23] < .5))
    command = (public[..., 0] > .5) | (public[..., 1] > .5)
    return successful_finish, command


class Subject:
    """One complete controller, world, sampling generator and causal boundary."""

    def __init__(self, seed, kind, stream_id=100):
        if seed not in common.SETTINGS["seeds"] or isinstance(seed, bool):
            raise ValueError("seed must select source executor 0, 1, or 2")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.seed, self.kind, self.stream_id = int(seed), kind, int(stream_id)
        self.contexts = (common.SEEDS[seed]["history_cue"], common.SEEDS[seed]["B_cue"])
        self.original = torch.tensor(common.mapping_table(seed, "related"), dtype=torch.long)
        self.world = data.make_world(
            seed, batch=common.SETTINGS["actors"],
            duration_range=tuple(common.SETTINGS["support_duration"]),
            gap_range=tuple(common.SETTINGS["support_gap"]), stream_id=stream_id,
            random_seed=58300, split="eval")
        batch = self.world.batch_size
        if kind == "learned":
            self.model = common.load_native(seed).eval()
            self.executor = self.model.executor
            self.state = self.model.initial_state(batch)
            self.binder = None
        else:
            checkpoint = torch.load(common.manual_checkpoint(seed), map_location="cpu", weights_only=False)
            self.model = SharedAssociationMemory(**checkpoint["metadata"]["model_config"])
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval().requires_grad_(False)
            self.executor = common.executor(seed)
            self.state = dict(h=self.executor.initial_state(batch), memory=self.model.initial_state(batch))
            self.binder = PublicReceiptBinder(batch, n_jobs=2, contexts=self.contexts)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(31415000 + self.seed * 1009 + self.stream_id)
        self.pending_obs = None
        self.work = dict(ticks=0, model_steps=0, sampled_action_rows=0,
                         teacher_action_rows=0, public_success_finishes=0,
                         manual_public_examples=0)

    @property
    def memory(self):
        return self.state["memory"]

    @memory.setter
    def memory(self, value):
        self.state["memory"] = value

    @property
    def hidden(self):
        return self.state["h"]

    @hidden.setter
    def hidden(self, value):
        self.state["h"] = value

    def snapshot(self):
        return dict(
            seed=self.seed, kind=self.kind, stream_id=self.stream_id,
            model=_copy(self.model.state_dict()),
            executor=None if self.kind == "learned" else _copy(self.executor.state_dict()),
            state=_copy(self.state), binder=None if self.binder is None else self.binder.state_dict(),
            world=self.world.state_dict(), work=_copy(self.work), rng=self.rng.get_state().clone(),
            pending_obs=_copy(self.pending_obs),
            parameter_hash=common.parameter_hash(self.model), memory_hash=common.tensor_hash(self.memory))

    def restore(self, snapshot):
        """Restore without drawing new randomness or loading another checkpoint."""
        if (snapshot["seed"], snapshot["kind"]) != (self.seed, self.kind):
            raise ValueError("snapshot seed/kind must match this subject")
        self.model.load_state_dict(snapshot["model"])
        if self.kind != "learned":
            self.executor.load_state_dict(snapshot["executor"])
        self.stream_id = snapshot["stream_id"]
        self.state = _copy(snapshot["state"])
        self.world = RandomMappingWorld.from_state_dict(snapshot["world"])
        self.binder = (None if snapshot["binder"] is None
                       else PublicReceiptBinder.from_state_dict(snapshot["binder"]))
        self.work = _copy(snapshot["work"])
        self.rng.set_state(snapshot["rng"])
        self.pending_obs = _copy(snapshot["pending_obs"])
        self.model.eval()
        return self

    @torch.no_grad()
    def step_obs(self, obs, sample=False, write_enabled=True, read_enabled=True):
        """Consume public56 once and update only this controller's state.

        This method does not call world.observe/act or use any world field.
        All eleven actions remain in the sampled distribution.  Its detached
        logits are collection data; PG must replay model.forward separately.
        """
        if self.kind == "learned":
            logits, self.state, diagnostic = self.model.step(
                obs, self.state, write_enabled=write_enabled,
                read_enabled=read_enabled, return_diagnostics=True)
            read_probs = diagnostic["read_probs"]
            gate = diagnostic["gate"].squeeze(-1)
            effective_gate = gate if write_enabled else torch.zeros_like(gate)
        else:
            evidence = self.binder.consume(obs)
            gates = []
            for key, value, mask in evidence:
                self.work["manual_public_examples"] += int(mask.sum())
                if write_enabled:
                    self.memory = self.model.write(self.memory, key, value, mask)
                gates.append(mask.to(obs.dtype) * self.model.eta)
            gate = torch.stack(gates, 1)
            effective_gate = gate if write_enabled else torch.zeros_like(gate)
            public = obs.reshape(len(obs), 2, 28)
            family = public[..., 24:28].argmax(-1)
            raw = public[..., 5:9].argmax(-1)
            keys = 4 * (family == self.contexts[1]).long() + raw
            scores = torch.stack([self.model.read(self.memory, keys[:, job]) for job in range(2)], 1)
            if self.kind == "manual_soft":
                read_probs = scores.softmax(-1) if read_enabled else torch.full_like(scores, .25)
                mapped = soft_translate(obs, read_probs, self.original)
            else:
                prediction = scores.argmax(-1) if read_enabled else torch.zeros_like(keys)
                read_probs = F.one_hot(prediction, 4).to(obs.dtype)
                mapped = translate(obs, prediction, self.original)
            logits, self.hidden = self.executor.step(mapped, self.hidden)
        if sample:
            actions = torch.multinomial(logits.softmax(-1).reshape(-1, 11), 1,
                                        generator=self.rng).reshape(len(obs), 2)
            self.work["sampled_action_rows"] += actions.numel()
        else:
            actions = logits.argmax(-1)
        success, _ = _public_masks(obs)
        self.work["model_steps"] += 1
        self.work["public_success_finishes"] += int(success.sum())
        diagnostics = dict(read_probs=read_probs, prediction=read_probs.argmax(-1),
                           gate=gate, effective_gate=effective_gate)
        return actions, logits, diagnostics

    def _take_observation(self):
        if self.pending_obs is not None:
            obs, self.pending_obs = self.pending_obs, None
            return obs
        return self.world.observe()

    def _act(self, action, *, teacher=False):
        self.world.act(action)
        self.work["ticks"] += 1
        if teacher:
            self.work["teacher_action_rows"] += action.numel()

    def support(self, ticks, teacher=False, sample=True, write_enabled=True, read_enabled=True):
        """Execute support and finish after act; teacher mode is explicit only."""
        ticks = _ticks(ticks)
        start = self.snapshot()
        arrays = {key: [] for key in ("obs", "actions", "logits", "read_probs", "gate", "effective_gate")}
        for _ in range(ticks):
            obs = self._take_observation()
            action, logits, diagnostic = self.step_obs(
                obs, sample=sample and not teacher, write_enabled=write_enabled, read_enabled=read_enabled)
            if teacher:
                action = self.world.oracle("midpoint")
            for key, value in dict(obs=obs, actions=action, logits=logits,
                                   read_probs=diagnostic["read_probs"], gate=diagnostic["gate"],
                                   effective_gate=diagnostic["effective_gate"]).items():
                arrays[key].append(_copy(value))
            self._act(action, teacher=teacher)
        return dict(start=start, final=self.snapshot(), **{key: torch.stack(value) for key, value in arrays.items()})

    def rollout_block(self, ticks=512, sample=True, write_enabled=True, read_enabled=True):
        """Collect aligned outcomes, keeping the final observation unprocessed.

        The additional observe advances the world, consumes the last receipt,
        and may introduce a new command.  It does not advance controller state.
        The next block consumes exactly that pending observation, once.
        """
        result = self.support(ticks, teacher=False, sample=sample,
                              write_enabled=write_enabled, read_enabled=read_enabled)
        self.pending_obs = self.world.observe()
        next_observations = torch.cat((result["obs"][1:], self.pending_obs[None]), 0)
        result["rewards"] = data.reward_from_observation(next_observations)
        result["pending_obs"] = _copy(self.pending_obs)
        result["final"] = self.snapshot()
        return result

    def close_pending(self, sample=True, write_enabled=True, read_enabled=True):
        """Execute one explicitly counted action to return to after-act."""
        if self.pending_obs is None:
            return None
        obs = self._take_observation()
        action, logits, diagnostics = self.step_obs(
            obs, sample=sample, write_enabled=write_enabled, read_enabled=read_enabled)
        self._act(action)
        return dict(obs=_copy(obs), actions=_copy(action), logits=_copy(logits),
                    diagnostics=_copy(diagnostics), reward_for_previous_action=data.reward_from_observation(obs))


def _summarize(rows):
    result = {}
    for group in GROUPS:
        if group == "all":
            selected = rows
        else:
            family = group[0]
            selected = [row for row in rows if row["family"] == family]
            if group.endswith("conflict"):
                selected = [row for row in selected if row["conflict"]]
            elif group.endswith("shared"):
                selected = [row for row in selected if not row["conflict"]]
        result[group] = dict(n=len(selected), strict=sum(row["strict"] for row in selected),
                             ordinary=sum(row["ordinary"] for row in selected),
                             translation_correct=sum(row["translation_correct"] for row in selected))
    return result


def probe(subject, ticks=2048, write_enabled=True, read_enabled=True):
    """Native argmax behavior, with every issued eligible matured request counted.

    Private event/target/maturity fields are read only by this evaluator after
    the controller has chosen its action.  Never-BEGIN and never-FINISH requests
    remain in the same denominator.  Left inheritance and right censoring are
    archived separately; no parameter learning occurs during this probe.
    """
    ticks = _ticks(ticks)
    start = subject.snapshot()
    book, finished, left_boundary = {}, [], []
    first_seen = set()
    idle = idle_actions = translated = correct_translations = 0
    arrays = {key: [] for key in ARRAY_FIELDS}
    gates = {name: dict(sum=0., count=0) for name in ("all", "success_finish", "command", "other")}
    effective_gate_sum = 0.
    for _ in range(ticks):
        obs = subject._take_observation()
        action, _, diagnostic = subject.step_obs(
            obs, sample=False, write_enabled=write_enabled, read_enabled=read_enabled)
        world = subject.world
        prediction = diagnostic["prediction"]
        for lane, job in world.matured_mask.nonzero().tolist():
            key = lane, job, int(world.matured_request_id[lane, job])
            if key in book:
                row = book.pop(key)
                row.update(mature_tick=world.tick,
                           ordinary=bool(world.matured_completed[lane, job]),
                           strict=bool(world.matured_clean_completed[lane, job]))
                finished.append(row)
            else:
                left_boundary.append(dict(lane=lane, job=job, rid=key[2], mature_tick=world.tick,
                                          ordinary=bool(world.matured_completed[lane, job]),
                                          strict=bool(world.matured_clean_completed[lane, job])))
        command = (world.event_kind == 1) | (world.event_kind == 2)
        for lane, job in command.nonzero().tolist():
            family_id = int(world.event_family[lane, job])
            family_index = subject.contexts.index(family_id)
            raw, target = int(world.raw_event_payload[lane, job]), int(world.event_payload[lane, job])
            correct = int(prediction[lane, job]) == target
            translated += 1
            correct_translations += int(correct)
            if bool(world.event_eligible[lane, job]):
                rid = int(world.event_request_id[lane, job])
                first_key = lane, family_index, raw
                first = first_key not in first_seen
                first_seen.add(first_key)
                book[lane, job, rid] = dict(
                    lane=lane, actor_id=world.actor_ids[lane], job=job, rid=rid,
                    issued_tick=world.tick, family="AB"[family_index], family_id=family_id,
                    raw=raw, target=target,
                    conflict=bool(world.private_mappings[lane, 0, raw] != world.private_mappings[lane, 1, raw]),
                    translation_correct=bool(correct), first_for_actor_family_raw=first)
        idle_mask = ~world.active
        idle += int(idle_mask.sum())
        idle_actions += int(((action != 0) & idle_mask).sum())
        success, public_command = _public_masks(obs)
        gate = diagnostic["gate"]
        for name, mask in (("all", torch.ones_like(success)), ("success_finish", success),
                           ("command", public_command), ("other", ~success & ~public_command)):
            gates[name]["sum"] += float(gate[mask].sum())
            gates[name]["count"] += int(mask.sum())
        effective_gate_sum += float(diagnostic["effective_gate"].sum())
        values = dict(
            obs=obs, actions=action, prediction=prediction,
            event_kind=world.event_kind, event_rid=world.event_request_id,
            event_eligible=world.event_eligible, event_family=world.event_family,
            raw=world.raw_event_payload, target=world.event_payload,
            mature_mask=world.matured_mask, mature_rid=world.matured_request_id,
            mature_done=world.matured_completed, mature_strict=world.matured_clean_completed,
            active=world.active, gate=gate, effective_gate=diagnostic["effective_gate"],
            read_probs=diagnostic["read_probs"])
        for key, value in values.items():
            arrays[key].append(value.detach().cpu().numpy().copy())
        subject._act(action)
    final = subject.snapshot()
    if final["parameter_hash"] != start["parameter_hash"]:
        raise AssertionError("probe changed controller parameters")
    if not write_enabled and not torch.equal(start["state"]["memory"], final["state"]["memory"]):
        raise AssertionError("no-write probe changed M")
    for row in gates.values():
        row["mean"] = row["sum"] / row["count"] if row["count"] else None
    first_finished = [row for row in finished if row["first_for_actor_family_raw"]]
    report = dict(
        seed=subject.seed, kind=subject.kind, groups=_summarize(finished), requests=finished,
        right_censored=list(book.values()), left_boundary_matured=left_boundary,
        first_groups=_summarize(first_finished), first_requests=first_finished,
        first_right_censored=[row for row in book.values() if row["first_for_actor_family_raw"]],
        idle_opportunities=idle, idle_actions=idle_actions,
        command_translations=translated, correct_translations=correct_translations,
        start_tick=start["world"]["tick"], end_tick=subject.world.tick,
        start_had_pending_observation=start["pending_obs"] is not None,
        start_work=start["work"], final_work=final["work"],
        write_enabled=write_enabled, read_enabled=read_enabled,
        memory_readonly=not write_enabled, parameters_frozen_during_probe=True,
        parameters=subject.model.parameter_report(),
        memory_norm=float(subject.memory.norm(dim=-1).mean()),
        gate_statistics=gates, effective_gate_sum=effective_gate_sum,
        gate_category_note="success_finish and command may overlap; other excludes both")
    return dict(report=report, arrays={key: np.stack(value) for key, value in arrays.items()},
                start=start, final=final)
