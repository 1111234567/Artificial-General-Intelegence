"""Keep a Scout alive across several private rule regimes without resetting it."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_model import ScoutRawAgent, load_scout_model
from .meta_scout_train import make_batch
from .scout_world import rollout_scout


class WithholdAfter(ScoutRawAgent):
    def __init__(self, model, first_hidden, *, freeze=False, restore_each_trial=False):
        super().__init__(model)
        self.first_hidden = first_hidden * world.TICKS
        self.freeze = freeze
        self.restore_each_trial = restore_each_trial

    def reset(self, n):
        super().reset(n)
        self.tick = 0
        self.unsigned_count = 0
        self.write_count = 0
        self.gain_sum = 0.0
        self.snapshot = None

    def act(self, obs):
        hidden = self.tick >= self.first_hidden
        if hidden:
            if self.snapshot is None:
                self.snapshot = self.state.fast
            if self.restore_each_trial and obs[0, world.TRIAL_START]:
                self.state = replace(self.state, fast=self.snapshot)
            obs = obs.copy()
            obs[:, [world.REWARD_POS, world.REWARD_NEG]] = 0
        self.write_enabled = not (hidden and self.freeze)
        self.tick += 1
        pulse = super().act(obs)
        if hidden:
            unsigned = obs[:, world.CONTACT_ID].sum(axis=1) > 0
            if np.any(unsigned):
                max_gain = self.last_step.write_gain.amax(dim=1).cpu().numpy()[unsigned]
                self.unsigned_count += len(max_gain)
                self.write_count += int((max_gain > 0.5).sum())
                self.gain_sum += float(max_gain.sum())
        return pulse

    def summary(self):
        return {"contacts_without_sign": self.unsigned_count, "peak_gain_above_half": self.write_count,
                "mean_peak_gain": self.gain_sum / max(1, self.unsigned_count)}


def from_regimes(original, length):
    maps = [original.good_map, original.good_map.copy(), original.good_map, 1 - original.good_map]
    maps[1][:, 1] ^= 1
    outcomes, sensitive = [], []
    for index, current in enumerate(maps):
        span = slice(index * length, (index + 1) * length)
        cue = original.cue[:, span]
        ctx = original.ctx[:, span]
        life = np.arange(original.n)[:, None]
        outcomes.append(current[life, ctx, cue])
        sensitive.append((current[:, 0] != current[:, 1])[life, cue])
    return replace(original, good_id=np.concatenate(outcomes, axis=1)), np.concatenate(sensitive, axis=1)


def summarize(tape, sensitive, result, length):
    good_side = tape.left_id != tape.good_id
    change_direction = tape.scout_side != good_side
    requires_probe = sensitive & change_direction
    can_go_direct = ~sensitive & change_direction
    rows = []
    for regime in range(4):
        start = regime * length
        for name, span in (("first_8", slice(start, start + 8)), ("first_16", slice(start, start + 16)),
                           ("last_16", slice(start + length - 16, start + length))):
            mandatory = requires_probe[:, span]
            optional = can_go_direct[:, span]
            correct = result.correct[:, span]
            rows.append({
                "regime": regime, "window": name, "correct": int(correct.sum()), "opportunities": int(correct.size),
                "scouted_and_reversed": int((result.reversed[:, span] & correct & mandatory).sum()),
                "reverse_opportunities": int(mandatory.sum()),
                "correct_direct": int((~result.probed[:, span] & correct & optional).sum()),
                "correct_direct_opportunities": int(optional.sum()),
                "mean_ticks": float(np.where(result.timed_out[:, span], world.MOVEMENT_TICKS,
                                              result.latency_ticks[:, span]).mean()),
            })
    return rows


def audit(checkpoint, base, lives, length):
    torch.set_num_threads(2)
    model = load_scout_model(checkpoint)
    tape = make_batch(base, lives, 4 * length)
    changed, sensitive = from_regimes(tape, length)
    static_sensitive = tape.sensitive[np.arange(tape.n)[:, None], tape.cue]
    normal = rollout_scout(ScoutRawAgent(model), changed)
    stationary = rollout_scout(ScoutRawAgent(model), tape)
    receive_without_sign = WithholdAfter(model, length)
    withheld = rollout_scout(receive_without_sign, tape)
    frozen = rollout_scout(WithholdAfter(model, length, freeze=True), tape)
    held_at_each_start = rollout_scout(WithholdAfter(model, length, restore_each_trial=True), tape)
    np.testing.assert_array_equal(normal.correct[:, :length], stationary.correct[:, :length])
    np.testing.assert_array_equal(withheld.correct[:, :length], stationary.correct[:, :length])
    return {
        "path": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "life_seeds": list(tape.seeds), "regime_length": length,
        "rule_schedule": ["original", "invert all B preferences", "restore original", "invert all A and B preferences"],
        "changing_rules": summarize(changed, sensitive, normal, length),
        "stationary": summarize(tape, static_sensitive, stationary, length),
        "stationary_withheld_after_first_regime": summarize(tape, static_sensitive, withheld, length),
        "stationary_withheld_and_frozen_after_first_regime": summarize(tape, static_sensitive, frozen, length),
        "stationary_withheld_cross_trial_fast_state_fixed": summarize(tape, static_sensitive, held_at_each_start, length),
        "withheld_write_gains": receive_without_sign.summary(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_310_000_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--regime-length", type=int, default=64)
    args = parser.parse_args()
    if args.regime_length < 16:
        raise ValueError("Regimes need at least 16 trials")
    report = audit(args.checkpoint, args.seed, args.lives, args.regime_length)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for conditions in ("changing_rules", "stationary", "stationary_withheld_after_first_regime",
                       "stationary_withheld_and_frozen_after_first_regime", "stationary_withheld_cross_trial_fast_state_fixed"):
        records = report[conditions]
        for regime in range(4):
            summary = [round(row["correct"] / row["opportunities"], 5) for row in records if row["regime"] == regime]
            print(conditions, regime, summary)
