"""Scout transfer tests with unchanged public mechanics and revised private rules."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_env import varied_sensitive_count
from .meta_scout_model import PULSES, ScoutRawAgent, load_scout_model
from .meta_scout_train import make_batch, metrics
from .scout_world import ScoutHistoryPolicy, rollout_scout


def surprise_switch(tape, switch, *, newly_sensitive=True):
    target = np.empty(tape.n, dtype=np.int8)
    for row, seed in enumerate(tape.seeds):
        choices = np.flatnonzero(tape.sensitive[row] != newly_sensitive)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x8AF4628D]))
        target[row] = rng.choice(choices)
    changed = tape.cue == target[:, None]
    after_switch = np.arange(tape.trials)[None, :] >= switch
    good = tape.good_id ^ (after_switch & changed & (tape.ctx == 1))
    return replace(tape, good_id=good.astype(tape.good_id.dtype)), target


def aggregate(outcome, mask):
    count = int(mask.sum())
    hits = int((outcome & mask).sum())
    return {"hits": hits, "trials": count, "rate": hits / count if count else None}


class ChangeOneFastWrite(ScoutRawAgent):
    def __init__(self, model, absolute_tick, mode):
        super().__init__(model)
        self.absolute_tick = absolute_tick
        self.mode = mode

    def reset(self, n):
        super().reset(n)
        self.tick = 0

    @torch.no_grad()
    def act(self, obs):
        timed = self.absolute_tick == self.tick
        self.tick += 1
        if not np.any(timed):
            return super().act(obs)
        if not np.all(obs[timed, world.REWARD_NEG] > 0):
            raise RuntimeError("The replay reached a different observation before the targeted negative receipt")
        before = self.state
        logits = self.logits(obs)
        mask = torch.as_tensor(timed, device=self.device)[:, None, None]
        if self.mode == "erase":
            value = before.fast
        else:
            alternate = obs.copy()
            alternate[timed, world.REWARD_POS], alternate[timed, world.REWARD_NEG] = (
                obs[timed, world.REWARD_NEG], obs[timed, world.REWARD_POS],
            )
            hypothetical = self.model(torch.as_tensor(alternate, device=self.device), before, self.weights)
            value = hypothetical.state.fast
        self.state = replace(self.state, fast=torch.where(mask, value, self.state.fast))
        return PULSES[logits.argmax(dim=1).cpu().numpy()]


def acquisition(factory, tape, switch, newly_sensitive, model=None):
    changed_tape, token = surprise_switch(tape, switch, newly_sensitive=newly_sensitive)
    same = rollout_scout(factory(), tape, record=True)
    changed = rollout_scout(factory(), changed_tape, record=True)
    np.testing.assert_array_equal(same.pulses[:, :switch], changed.pulses[:, :switch])
    np.testing.assert_array_equal(same.contact_id[:, :switch], changed.contact_id[:, :switch])
    trial = np.arange(tape.trials)[None, :]
    changed_cue = tape.cue == token[:, None]
    after_switch = trial >= switch
    prior_side = tape.left_id != tape.good_id
    old_side_away_from_beacon = tape.scout_side != prior_side
    observed_wrong = (changed.contact_id == tape.good_id) & (tape.good_id != changed_tape.good_id)
    evidence = after_switch & changed_cue & observed_wrong
    if newly_sensitive:
        evidence &= ~changed.probed
    first = np.where(evidence, trial, tape.trials).min(axis=1)
    after_evidence = trial > first[:, None]
    eligible = after_evidence & changed_cue & old_side_away_from_beacon
    occurrence = np.cumsum(eligible, axis=1)
    changed_first_pulse = np.take_along_axis(changed.pulses[:, :, 0, :], tape.scout_side[..., None], axis=2).squeeze(2)
    same_first_pulse = np.take_along_axis(same.pulses[:, :, 0, :], tape.scout_side[..., None], axis=2).squeeze(2)

    groups = {
        "first_probe_opportunity": eligible & (occurrence == 1),
        "next_two_opportunities": eligible & (occurrence >= 2) & (occurrence <= 3),
        "later_probe_opportunities": eligible & (occurrence > 3),
        "late_changed_cue": changed_cue & (trial >= tape.trials - 16),
        "late_other_cues": ~changed_cue & (trial >= tape.trials - 16),
    }
    changed_observation = np.any(same.obs != changed.obs, axis=(2, 3))
    first_observable_difference = np.where(changed_observation, trial, tape.trials).min(axis=1)
    strict_receipt = (first == first_observable_difference) & (first < tape.trials)
    next_encounter = np.where(after_evidence & changed_cue, trial, tape.trials).min(axis=1)
    strict = strict_receipt[:, None] & (trial == next_encounter[:, None]) & old_side_away_from_beacon
    groups["strict_next_same_cue"] = strict
    report = {"switch": switch, "newly_sensitive": newly_sensitive,
              "lives_with_diagnostic_receipt": int((first < tape.trials).sum()), "n": tape.n}
    for name, mask in groups.items():
        report[name] = {
            "probe_changed": aggregate(changed.probed, mask),
            "probe_sham": aggregate(same.probed, mask),
            "move_toward_beacon_changed": aggregate(changed_first_pulse, mask),
            "move_toward_beacon_sham": aggregate(same_first_pulse, mask),
            "correct_changed": aggregate(changed.correct, mask),
            "correct_sham": aggregate(same.correct, mask),
        }
    pulse_delta = (changed_first_pulse.astype(float) - same_first_pulse)[strict]
    if pulse_delta.size:
        draws = np.random.default_rng(3900).integers(0, pulse_delta.size, (2500, pulse_delta.size))
        report["strict_next_same_cue"]["first_pulse_delta_ci95"] = np.quantile(
            pulse_delta[draws].mean(axis=1), [0.025, 0.975],
        ).tolist()
    if model is not None and newly_sensitive and np.any(strict):
        rows = np.flatnonzero(strict.any(axis=1))
        alter_tick = np.full(tape.n, -1)
        alter_tick[rows] = first[rows] * world.TICKS + changed.contact_tick[rows, first[rows]] + 1
        for mode in ("erase", "positive_in_fast_only"):
            result = rollout_scout(ChangeOneFastWrite(model, alter_tick, mode), changed_tape, record=True)
            first_choice = np.take_along_axis(result.pulses[:, :, 0, :], tape.scout_side[..., None], axis=2).squeeze(2)
            report["strict_next_same_cue"][mode] = {
                "probe": aggregate(result.probed, strict), "first_pulse": aggregate(first_choice, strict),
                "correct": aggregate(result.correct, strict),
            }
    return report


def transfer(checkpoint, root, lives, trials):
    torch.set_num_threads(2)
    model = load_scout_model(checkpoint)
    tape = make_batch(root, lives, trials)
    counts = {}
    for amount in range(5):
        variant = varied_sensitive_count(tape, amount)
        result = rollout_scout(ScoutRawAgent(model), variant)
        counts[str(amount)] = metrics(result, variant)
    return {
        "checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "life_seeds": list(tape.seeds), "trials": trials, "range_of_sensitive_counts": counts,
        "restart_probe": acquisition(lambda: ScoutRawAgent(model), tape, trials // 2, newly_sensitive=True, model=model),
        "retire_probe": acquisition(lambda: ScoutRawAgent(model), tape, trials // 2, newly_sensitive=False),
        "teacher_restart_probe": acquisition(lambda: ScoutHistoryPolicy(77819), tape, trials // 2, newly_sensitive=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_290_000_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--trials", type=int, default=96)
    args = parser.parse_args()
    report = transfer(args.checkpoint, args.seed, args.lives, args.trials)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for count, result in report["range_of_sensitive_counts"].items():
        print(json.dumps({"sensitive_count": count, **result}))
    print(json.dumps({key: report[key] for key in ("restart_probe", "retire_probe", "teacher_restart_probe")}))
