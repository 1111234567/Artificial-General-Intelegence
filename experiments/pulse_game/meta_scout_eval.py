"""Held-out-life diagnostics for a frozen raw Scout checkpoint."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_model import PULSES, ScoutRawAgent, load_scout_model
from .meta_scout_train import collect, make_batch, metrics, sequence, targets_from_public
from .scout_world import SCOUT_HIT, make_context_replay, rollout_scout


def signed_receipt(obs):
    return (obs[:, 13:15].sum(axis=1) > 0) & (obs[:, world.CONTACT_ID].sum(axis=1) > 0)


class ScoutWriteRecord(ScoutRawAgent):
    def reset(self, n):
        super().reset(n)
        self.record = np.zeros((4, 3), dtype=np.float64)

    @torch.no_grad()
    def act(self, obs):
        before = self.state.fast
        pulses = super().act(obs)
        delta = (self.state.fast - before).flatten(1).norm(dim=1).cpu().numpy()
        reward = signed_receipt(obs)
        context = obs[:, SCOUT_HIT] > 0
        active = obs[:, world.ACTIVE] > 0
        for index, mask in enumerate((reward, context & ~reward, active & ~reward & ~context, ~active & ~reward & ~context)):
            self.record[index] += (mask.sum(), delta[mask].sum(), (delta[mask] > 1e-4).sum())
        return pulses

    def summary(self):
        names = ("receipt", "sensor", "active_neutral", "parked_neutral")
        return {name: {"ticks": int(n), "mean_weight_delta_l2": float(total / max(1, n)),
                       "delta_above_1e_4": int(nonz)} for name, (n, total, nonz) in zip(names, self.record)}


class MaskScoutWrites(ScoutRawAgent):
    def __init__(self, model, *, only_receipt):
        super().__init__(model)
        self.only_receipt = only_receipt

    @torch.no_grad()
    def act(self, obs):
        receipt = signed_receipt(obs)
        old = self.state.fast
        pulses = super().act(obs)
        keep = receipt if self.only_receipt else ~receipt
        mask = torch.as_tensor(keep, device=self.device)[:, None, None]
        self.state = replace(self.state, fast=torch.where(mask, self.state.fast, old))
        return pulses


class OneSensorFork(ScoutRawAgent):
    def __init__(self, model, trials):
        super().__init__(model)
        self.trials = trials

    def reset(self, n):
        super().reset(n)
        self.tick = 0
        self.observed = np.full((n, self.trials), -1, dtype=np.int8)
        self.counterfactual = self.observed.copy()

    @torch.no_grad()
    def act(self, obs):
        trial = self.tick // world.TICKS
        self.tick += 1
        revealed = (obs[:, SCOUT_HIT] > 0) & (obs[:, world.CTX].sum(axis=1) == 1) & (obs[:, world.ACTIVE] > 0)
        if not np.any(revealed):
            return super().act(obs)
        before = self.state
        real = self.logits(obs).argmax(dim=1).cpu().numpy()
        fake_observation = obs.copy()
        fake_observation[revealed, 0], fake_observation[revealed, 1] = obs[revealed, 1], obs[revealed, 0]
        counterfactual = self.model(torch.as_tensor(fake_observation, device=self.device), before, self.weights)
        changed = counterfactual.logits.argmax(dim=1).cpu().numpy()
        self.observed[revealed, trial] = real[revealed]
        self.counterfactual[revealed, trial] = changed[revealed]
        return PULSES[real]


class ReversedReward(ScoutRawAgent):
    def act(self, obs):
        manipulated = obs.copy()
        manipulated[:, [world.REWARD_POS, world.REWARD_NEG]] = obs[:, [world.REWARD_NEG, world.REWARD_POS]]
        return super().act(manipulated)


class LateSwappedSensor(ScoutRawAgent):
    def __init__(self, model, first_trial):
        super().__init__(model)
        self.swap_after = first_trial * world.TICKS

    def reset(self, n):
        super().reset(n)
        self.tick = 0

    def act(self, obs):
        if self.tick >= self.swap_after:
            obs = obs.copy()
            obs[:, [0, 1]] = obs[:, [1, 0]]
        self.tick += 1
        return super().act(obs)


class WrongScoutAddress(ScoutRawAgent):
    @torch.no_grad()
    def act(self, obs):
        reward = signed_receipt(obs)
        if not np.any(reward):
            return super().act(obs)
        cue = obs[:, world.CUE].argmax(axis=1)
        items = np.arange(len(obs))
        choices = (items[:, None] + np.arange(1, len(obs) + 1)) % len(obs)
        differs = cue[choices] != cue[:, None]
        treated = reward & differs.any(axis=1)
        donor = np.where(treated, choices[items, differs.argmax(axis=1)], items)
        old_fast = self.state.fast
        normal = self.logits(obs)
        step = self.last_step
        wrong_key = step.key[torch.as_tensor(donor, device=self.device)]
        read = (wrong_key.unsqueeze(1) @ old_fast).squeeze(1)
        correction = step.write_gain * (step.payload - read)
        alternate_fast = old_fast + wrong_key.unsqueeze(-1) * correction.unsqueeze(1)
        mask = torch.as_tensor(treated, device=self.device)[:, None, None]
        self.state = replace(self.state, fast=torch.where(mask, alternate_fast, self.state.fast))
        return PULSES[normal.argmax(dim=1).cpu().numpy()]


def paired_outcome_delta(reference, intervention):
    two = reference.correct.mean(axis=1) - intervention.correct.mean(axis=1)
    rows = np.random.default_rng(13491).integers(0, len(two), (2000, len(two)))
    return {"delta": float(two.mean()), "ci95": np.quantile(two[rows].mean(axis=1), [0.025, 0.975]).tolist()}


def sensor_fork(model, tape, reference):
    fork = OneSensorFork(model, tape.trials)
    true_result = rollout_scout(fork, tape)
    np.testing.assert_array_equal(true_result.contact_id, reference.contact_id)
    late = np.arange(tape.trials)[None, :] >= tape.trials - min(16, tape.trials)
    destination = 1 + (tape.left_id != tape.good_id)
    revealed = fork.observed >= 0
    reverse = reference.sensitive_trial & ((destination - 1) != tape.scout_side)
    correct_pulse = fork.observed == destination
    subsets = {"mandatory_late": reverse & late & revealed,
               "insensitive_late": ~reference.sensitive_trial & late & revealed}
    output = {}
    for name, rows in subsets.items():
        before_right = correct_pulse & rows
        output[name] = {"paired_physical_probes": int(rows.sum()), "actual_good_direction": int(before_right.sum()),
                        "pulse_changes_total": int((rows & (fork.observed != fork.counterfactual)).sum()),
                        "good_to_opposite_direction": int((before_right & (fork.counterfactual == 3 - destination)).sum())}
    return output


@torch.no_grad()
def replay_diagnostics(model, examples):
    obs = torch.as_tensor(examples.obs)
    targets = targets_from_public(examples.obs, "cpu")
    predictions, memories, _, gates, _, activities = sequence(model, obs)
    preferred = model.preference_probe(memories)
    contextual = model.context_probe(activities).argmax(dim=-1)
    action_correct = predictions.argmax(dim=-1) == torch.as_tensor(examples.targets)
    known = targets.known[:, :, None, :]
    correct_memory = (preferred > 0) == targets.context_map[:, :, None, :]
    grow = (0.25 * gates[..., 1] + 0.5).clamp(0, 1)
    actual_context = targets.context
    active = obs[..., world.ACTIVE] > 0
    probe = obs[..., SCOUT_HIT] > 0
    start = obs[..., world.TRIAL_START] > 0
    known_for_context = known.expand_as(preferred).gather(-1, (actual_context - 1).clamp_min(0)[..., None]).squeeze(-1)

    def counts(ok, mask):
        total = int(mask.sum())
        return {"hits": int((ok & mask).sum()), "trials": total}

    return {
        "actions_start": counts(action_correct, start),
        "actions_at_scout": counts(action_correct, probe & active),
        "actions_at_scout_known": counts(action_correct, probe & active & known_for_context),
        "actions_after_scout": counts(action_correct, (actual_context > 0) & ~probe & active),
        "context_after_scout": counts(contextual == actual_context, (actual_context > 0) & ~probe & active),
        "memory_opening": counts(correct_memory[:, :, 0], known[:, :, 0]),
        "memory_scout": counts(correct_memory, known & (probe & active)[..., None]),
        "write_at_receipt": counts(grow > 0.5, targets.receipt),
        "write_without_receipt": counts(grow > 0.5, ~targets.receipt),
    }


def audit(checkpoint, roots, trials, diagnostics_lives, parity=None, sensitive_count=None):
    torch.set_num_threads(2)
    checkpoint = Path(checkpoint)
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = load_scout_model(checkpoint)
    tape = make_batch(roots.start, len(roots), trials, parity=parity, sensitive_count=sensitive_count)
    agent = ScoutWriteRecord(model)
    reference = rollout_scout(agent, tape)
    replay = make_context_replay(reference, seed=98939)
    yoke = world.make_yoke(reference, seed=83838)
    others = {
        "disable_fast_write": rollout_scout(ScoutRawAgent(model, write_enabled=False), tape),
        "disable_fast_read": rollout_scout(ScoutRawAgent(model, read_enabled=False), tape),
        "disable_recurrent_edges": rollout_scout(ScoutRawAgent(model, recurrence_enabled=False), tape),
        "wrong_write_address": rollout_scout(WrongScoutAddress(model), tape),
        "writes_only_on_receipt": rollout_scout(MaskScoutWrites(model, only_receipt=True), tape),
        "writes_only_off_receipt": rollout_scout(MaskScoutWrites(model, only_receipt=False), tape),
        "no_reward": rollout_scout(ScoutRawAgent(model), tape, feedback="none"),
        "inverted_reward": rollout_scout(ReversedReward(model), tape),
        "yoked_reward": rollout_scout(ScoutRawAgent(model), tape, feedback="yoked", yoke=yoke),
        "no_context": rollout_scout(ScoutRawAgent(model), tape, sensor="none"),
        "yoked_context": rollout_scout(ScoutRawAgent(model), tape, sensor="replay", replay=replay),
        "swap_late_context": rollout_scout(LateSwappedSensor(model, trials - min(16, trials)), tape),
    }
    output = {"step": saved["step"], "checkpoint": str(checkpoint), "sha256": checksum, "life_seeds": list(roots),
              "address_intervention": "redirect_fast_write_to_different_cue_after_projection", "map_parity": parity,
              "sensitive_count_override": sensitive_count,
              "actual_fast_weight_deltas": agent.summary(),
              "single_sensor_pulse_fork": sensor_fork(model, tape, reference),
              "report": {"true": metrics(reference, tape), **{key: metrics(value, tape) for key, value in others.items()}},
              "reference_effect": {key: paired_outcome_delta(reference, value) for key, value in others.items()}}
    if diagnostics_lives:
        diagnostic = make_batch(roots.start, min(len(roots), diagnostics_lives), trials, parity=parity,
                                sensitive_count=sensitive_count)
        teacher = collect(diagnostic, seed=11384)
        student = collect(diagnostic, seed=11384, student=ScoutRawAgent(model))
        output["replay_diagnostics"] = {
            "teacher_history": replay_diagnostics(model, teacher),
            "student_history_teacher_labels": replay_diagnostics(model, student),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4_260_000_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--diagnostic-lives", type=int, default=32)
    parser.add_argument("--map-parity", type=int, choices=(0, 1))
    parser.add_argument("--sensitive-count", type=int, choices=range(5))
    args = parser.parse_args()
    report = audit(args.checkpoint, range(args.seed, args.seed + args.lives), args.trials, args.diagnostic_lives,
                   args.map_parity, args.sensitive_count)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for condition, result in report["report"].items():
        print(json.dumps({"condition": condition, **result}))
