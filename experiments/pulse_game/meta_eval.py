"""Local development audit of a frozen meta policy in the reference world."""

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_env import independent_context_map
from .meta_model import RolloutAgent, load_policy


class ReceiptManipulation:
    def __init__(self, agent: RolloutAgent, *, flip=False, withhold_after=None):
        self.agent = agent
        self.flip = flip
        self.withhold_after = None if withhold_after is None else withhold_after * world.TICKS

    def reset(self, n):
        self.agent.reset(n)
        self.tick = 0

    def act(self, obs):
        if self.flip or (self.withhold_after is not None and self.tick >= self.withhold_after):
            obs = obs.copy()
            if self.flip:
                obs[:, [world.REWARD_POS, world.REWARD_NEG]] = obs[:, [world.REWARD_NEG, world.REWARD_POS]]
            else:
                obs[:, [world.REWARD_POS, world.REWARD_NEG]] = 0
        self.tick += 1
        return self.agent.act(obs)


class CueVisibility:
    def __init__(self, agent, cue_ticks):
        self.agent = agent
        self.cue_ticks = cue_ticks

    def reset(self, n):
        self.agent.reset(n)
        self.tick = 0

    def act(self, obs):
        if self.tick % world.TICKS >= self.cue_ticks:
            obs = obs.copy()
            obs[:, :6] = 0
        self.tick += 1
        return self.agent.act(obs)


class FreezeWritesAfter(RolloutAgent):
    def __init__(self, policy, trial):
        self.cutoff_tick = trial * world.TICKS
        super().__init__(policy)

    def reset(self, n=1):
        super().reset(n)
        self.tick = 0

    def act(self, obs):
        self.write_enabled = self.tick < self.cutoff_tick
        self.tick += 1
        return super().act(obs)


class WrongAddress(RolloutAgent):
    """Move a receipt's fast update to a different cue after the model computes its address."""

    def reset(self, n=1):
        super().reset(n)
        self.current_cue = np.full(n, -1)

    @torch.no_grad()
    def act(self, obs):
        visible = obs[:, world.CUE].sum(axis=1) > 0
        self.current_cue[visible] = obs[visible, world.CUE].argmax(axis=1)
        receipt = (obs[:, 13:15].sum(axis=1) > 0) & (obs[:, world.CONTACT_ID].sum(axis=1) > 0)
        if not np.any(receipt):
            return super().act(obs)
        indices = np.arange(len(obs))
        variants = (indices[:, None] + np.arange(1, len(obs) + 1)) % len(obs)
        distinct = self.current_cue[variants] != self.current_cue[:, None]
        plausible = receipt & distinct.any(axis=1)
        if not np.any(plausible):
            return super().act(obs)
        donor_rows = variants[indices, distinct.argmax(axis=1)]
        donor_rows = np.where(plausible, donor_rows, indices)
        old_fast = self.state.fast
        pulses = super().act(obs)
        donor_traces = self.state.trace[torch.as_tensor(donor_rows, device=self.device)]
        wrong_key = torch.nn.functional.normalize(donor_traces, dim=1, eps=1e-6)
        association = self.state.association
        gain = (0.25 * association.gates[:, 1:2] + 0.5).clamp(0, 1)
        read = (wrong_key.unsqueeze(1) @ old_fast).squeeze(1)
        correction = gain * (association.value - read)
        alternate_fast = old_fast + wrong_key.unsqueeze(-1) * correction.unsqueeze(1)
        intervened = torch.as_tensor(plausible, device=self.device)[:, None, None]
        self.state = replace(self.state, fast=torch.where(intervened, alternate_fast, self.state.fast))
        return pulses


class GateRecord:
    def __init__(self, agent: RolloutAgent):
        self.agent = agent

    def reset(self, n):
        self.agent.reset(n)
        self.counts = np.zeros((2, 2, 3), dtype=float)
        width = self.agent.policy.config.keys
        self.opening_keys = np.zeros((n, width), dtype=float)
        self.trace_sums = np.zeros((2, 8, 2, width), dtype=float)
        self.receipt_alignment = np.zeros((8, 2), dtype=float)
        self.current_keys = np.zeros(n, dtype=int)
        self.current_layout = np.zeros(n, dtype=int)
        self.id_only = np.zeros(3, dtype=float)

    def act(self, obs):
        pulses = self.agent.act(obs)
        received = (obs[:, 13:15].sum(axis=1) > 0) & (obs[:, world.CONTACT_ID].sum(axis=1) > 0)
        expected = np.stack((obs[:, world.TRIAL_START].astype(bool), received), axis=1)
        gate = (0.25 * self.agent.state.association.gates + 0.5).clamp(0, 1).cpu().numpy()
        contact_without_signed_receipt = (obs[:, world.CONTACT_ID].sum(axis=1) > 0) & ~received
        self.id_only += (contact_without_signed_receipt.sum(), (contact_without_signed_receipt * gate[:, 1]).sum(),
                         (contact_without_signed_receipt & (gate[:, 1] > 0.5)).sum())
        for truth, mask in enumerate((~expected, expected)):
            self.counts[:, truth, 0] += mask.sum(axis=0)
            self.counts[:, truth, 1] += (mask * gate).sum(axis=0)
            self.counts[:, truth, 2] += (mask & (gate > 0.5)).sum(axis=0)
        opened = expected[:, 0]
        addresses = torch.nn.functional.normalize(self.agent.state.trace, dim=1, eps=1e-6).cpu().numpy()
        if np.any(opened):
            self.current_keys[opened] = obs[opened, world.CTX].argmax(1) * 4 + obs[opened, world.CUE].argmax(1)
            self.current_layout[opened] = obs[opened, world.LEFT_ID].argmax(1)
            self.opening_keys[opened] = addresses[opened]
            np.add.at(self.trace_sums[0], (self.current_keys[opened], self.current_layout[opened]), addresses[opened])
        if np.any(received):
            selected = self.current_keys[received]
            np.add.at(self.trace_sums[1], (selected, self.current_layout[received]), addresses[received])
            cosine = (addresses[received] * self.opening_keys[received]).sum(axis=1)
            np.add.at(self.receipt_alignment[:, 0], selected, cosine)
            np.add.at(self.receipt_alignment[:, 1], selected, 1)
        return pulses

    def summary(self):
        result = {}
        for name, index in (("address_capture", 0), ("fast_write", 1)):
            for timing, condition in (("other_ticks", 0), ("event_tick", 1)):
                n, total, open_count = self.counts[index, condition]
                result[f"{name}_{timing}"] = {"count": int(n), "mean_gain": total / n if n else None,
                                                "above_half": int(open_count)}
        by_layout = self.trace_sums / np.maximum(np.linalg.norm(self.trace_sums, axis=-1, keepdims=True), 1e-8)
        by_token = self.trace_sums.sum(axis=2)
        by_token /= np.maximum(np.linalg.norm(by_token, axis=-1, keepdims=True), 1e-8)
        align = self.receipt_alignment
        n, total, open_count = self.id_only
        result["write_on_contact_without_sign"] = {
            "count": int(n), "mean_gain": total / n if n else None, "above_half": int(open_count),
        }
        result["trace_geometry"] = {
            "mean_opening_to_receipt_cosine": float(align[:, 0].sum() / max(1, align[:, 1].sum())),
            "cue_order": [f"{ctx}{cue}" for ctx in ("A", "B") for cue in range(4)],
            "opening_cross_layout_cosine": (by_layout[0, :, 0] * by_layout[0, :, 1]).sum(axis=1).tolist(),
            "receipt_cross_layout_cosine": (by_layout[1, :, 0] * by_layout[1, :, 1]).sum(axis=1).tolist(),
            "opening_key_cosine": (by_token[0] @ by_token[0].T).tolist(),
            "receipt_key_cosine": (by_token[1] @ by_token[1].T).tolist(),
        }
        return result


def per_life_counts(tape, result):
    shape = tape.ctx.shape
    masks = {name: np.zeros(shape, dtype=bool) for name in (
        "first_cue", "once_other_context", "same_key_seen", "layout_swapped", "both_contexts_seen",
        "revisit_after_8_others", "settled_after_40",
    )}
    latest = np.full((tape.n, 2, 4), -1)
    last_layout = latest.copy()
    rows = np.arange(tape.n)
    contacted = result.contact_id >= 0
    for idx in range(tape.trials):
        ctx, cue, layout = tape.ctx[:, idx], tape.cue[:, idx], tape.left_id[:, idx]
        last_same = latest[rows, ctx, cue]
        last_other = latest[rows, 1 - ctx, cue]
        same_seen = last_same >= 0
        other_seen = last_other >= 0
        masks["first_cue"][:, idx] = ~same_seen & ~other_seen
        masks["once_other_context"][:, idx] = ~same_seen & other_seen
        masks["same_key_seen"][:, idx] = same_seen
        masks["layout_swapped"][:, idx] = same_seen & (last_layout[rows, ctx, cue] != layout)
        masks["both_contexts_seen"][:, idx] = same_seen & other_seen
        masks["revisit_after_8_others"][:, idx] = same_seen & (idx - last_same >= 9)
        masks["settled_after_40"][:, idx] = same_seen & (idx >= 40)
        hit_rows = contacted[:, idx]
        latest[rows[hit_rows], ctx[hit_rows], cue[hit_rows]] = idx
        last_layout[rows[hit_rows], ctx[hit_rows], cue[hit_rows]] = layout[hit_rows]
    masks["overall"] = np.ones(shape, dtype=bool)
    return {name: np.stack((mask.sum(axis=1), (result.correct & mask).sum(axis=1),
                            (contacted & mask).sum(axis=1)), axis=1) for name, mask in masks.items()}


def summarize(counts):
    output = {}
    for group, count in counts.items():
        trials, hits, contacts = map(int, count.sum(axis=0))
        output[group] = {"trials": trials, "hits": hits, "contacts": contacts,
                         "success": hits / trials if trials else None}
    return output


def paired_deltas(reference, other, samples):
    output = {}
    for name, base in reference.items():
        left = base[samples].sum(axis=1)
        right = other[name][samples].sum(axis=1)
        valid = (left[:, 0] > 0) & (right[:, 0] > 0)
        if np.any(valid):
            delta = left[valid, 1] / left[valid, 0] - right[valid, 1] / right[valid, 0]
            true_trials, true_hits = base.sum(axis=0)[:2]
            control_trials, control_hits = other[name].sum(axis=0)[:2]
            output[name] = {"delta": float(true_hits / true_trials - control_hits / control_trials),
                            "bootstrap_95": np.quantile(delta, [0.025, 0.975]).tolist()}
    return output


def audit(checkpoint, seeds, trials, *, distance_jitter=0.0, wind_scale=0.0, cue_ticks=world.TICKS,
          independent_contexts=False):
    torch.set_num_threads(2)
    checkpoint_bytes = Path(checkpoint).read_bytes()
    policy = load_policy(io.BytesIO(checkpoint_bytes))
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    tape = world.make_tape(seeds, trials, distance_jitter=distance_jitter, wind_scale=wind_scale)
    if independent_contexts:
        tape = independent_context_map(tape)
    temporal = policy.config.memory_kind == "temporal"
    base_agent = GateRecord(RolloutAgent(policy)) if temporal else RolloutAgent(policy)
    withheld_agent = GateRecord(RolloutAgent(policy)) if temporal else RolloutAgent(policy)
    conditions = {
        "true": (base_agent, {}),
        "no_receipt": (RolloutAgent(policy), {"feedback": "none"}),
        "disable_fast_write": (RolloutAgent(policy, write_enabled=False), {}),
        "disable_fast_read": (RolloutAgent(policy, read_enabled=False), {}),
        "disable_slow_recurrence": (RolloutAgent(policy, recurrence_enabled=False), {}),
        "invert_receipt": (ReceiptManipulation(RolloutAgent(policy), flip=True), {}),
        "withhold_after_16": (ReceiptManipulation(withheld_agent, withhold_after=16), {}),
        "withhold_and_freeze_after_16": (
            ReceiptManipulation(FreezeWritesAfter(policy, trial=16), withhold_after=16), {},
        ),
    }
    if temporal:
        conditions["wrong_address"] = (WrongAddress(policy), {})
    counts = {}
    for name, (agent, kwargs) in conditions.items():
        result = world.rollout(CueVisibility(agent, cue_ticks), tape, **kwargs)
        counts[name] = per_life_counts(tape, result)
        if name == "true":
            yoke = world.make_yoke(result, seed=63)
    yoked = world.rollout(CueVisibility(RolloutAgent(policy), cue_ticks), tape, feedback="yoked", yoke=yoke)
    counts["yoked_receipt"] = per_life_counts(tape, yoked)
    samples = np.random.default_rng(21196).integers(0, tape.n, (2000, tape.n))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": saved.get("step"),
        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "seeds": list(tape.seeds),
        "protocol": {"trials": trials, "distance_jitter": distance_jitter, "wind_scale": wind_scale,
                     "cue_visible_ticks": cue_ticks, "independent_contexts": independent_contexts,
                     "address_intervention": "redirect_fast_write_to_different_cue_after_projection" if temporal else None},
        "gates": base_agent.summary() if temporal else None,
        "gates_withheld_after_16": withheld_agent.summary() if temporal else None,
        "summary": {name: summarize(item) for name, item in counts.items()},
        "true_minus_intervention": {name: paired_deltas(counts["true"], item, samples)
                                    for name, item in counts.items() if name != "true"},
        "per_life_counts": {name: {group: item.tolist() for group, item in value.items()}
                            for name, value in counts.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1_119_400_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--distance-jitter", type=float, default=0.0)
    parser.add_argument("--wind-scale", type=float, default=0.0)
    parser.add_argument("--cue-ticks", type=int, choices=range(1, world.TICKS + 1), default=world.TICKS)
    parser.add_argument("--independent-contexts", action="store_true")
    args = parser.parse_args()
    report = audit(args.checkpoint, range(args.seed, args.seed + args.lives), args.trials,
                   distance_jitter=args.distance_jitter, wind_scale=args.wind_scale, cue_ticks=args.cue_ticks,
                   independent_contexts=args.independent_contexts)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for name, groups in report["summary"].items():
        print(json.dumps({"condition": name, "overall": groups["overall"], "revisit": groups["same_key_seen"]}))
