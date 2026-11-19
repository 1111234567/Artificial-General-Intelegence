"""Diagnostic relay from the learned value read to the public motor layout."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_model import PULSES, ScoutRawAgent, load_scout_model
from .meta_scout_train import make_batch, metrics
from .scout_world import SCOUT_HIT, rollout_scout


class ReadoutDiagnostic(ScoutRawAgent):
    """Use learned values, and replace only post-scout motor selection by a fixed route."""

    def reset(self, n):
        super().reset(n)
        self.placed = np.full(n, -1, dtype=np.int8)

    @torch.no_grad()
    def act(self, obs):
        pulse = super().act(obs)
        begin = obs[:, world.TRIAL_START] > 0
        sensor = (obs[:, SCOUT_HIT] > 0) & (obs[:, world.CTX].sum(axis=1) > 0)
        self.placed[begin] = -1
        target_id = self.last_step.read.argmax(dim=1).cpu().numpy()
        self.placed[sensor] = target_id[sensor]
        visible_left = obs[:, world.LEFT_ID].argmax(axis=1)
        assist = (self.placed >= 0) & (obs[:, world.ACTIVE] > 0)
        pulse[assist] = PULSES[1 + (self.placed[assist] != visible_left[assist])]
        return pulse


class WrongReadAgent(ScoutRawAgent):
    def __init__(self, model, *, mode, trials, only_probe):
        super().__init__(model)
        self.mode = mode
        self.trials = trials
        self.only_probe = only_probe

    def reset(self, n):
        super().reset(n)
        self.tick = 0
        self.sensed = np.zeros(n, dtype=bool)
        self.donor = torch.as_tensor(np.roll(np.arange(n), 1), device=self.device)
        self.true_at_probe = np.full((n, self.trials), -1, dtype=np.int8)
        self.fake_at_probe = self.true_at_probe.copy()

    @torch.no_grad()
    def act(self, obs):
        trial = self.tick // world.TICKS
        self.tick += 1
        begin = obs[:, world.TRIAL_START] > 0
        probe = (obs[:, SCOUT_HIT] > 0) & (obs[:, world.CTX].sum(axis=1) == 1) & (obs[:, world.ACTIVE] > 0)
        self.sensed[begin] = False
        self.sensed |= probe
        before = self.state.activity
        real = self.logits(obs)
        active = probe if self.only_probe else self.sensed & (obs[:, world.ACTIVE] > 0)
        if not np.any(active):
            return PULSES[real.argmax(dim=1).cpu().numpy()]
        read = self.last_step.read
        bad = read[:, [1, 0]] if self.mode == "swap" else read[self.donor]
        alternative, activity = self.model.response_with_read(torch.as_tensor(obs, device=self.device), before, self.weights, bad)
        self.true_at_probe[probe, trial] = real.argmax(dim=1).cpu().numpy()[probe]
        self.fake_at_probe[probe, trial] = alternative.argmax(dim=1).cpu().numpy()[probe]
        if self.only_probe:
            return PULSES[real.argmax(dim=1).cpu().numpy()]
        rows = torch.as_tensor(active, device=self.device)[:, None]
        self.state = replace(self.state, activity=torch.where(rows, activity, self.state.activity))
        return PULSES[torch.where(rows, alternative, real).argmax(dim=1).cpu().numpy()]


def evaluate(checkpoint, seed, lives):
    torch.set_num_threads(2)
    model = load_scout_model(checkpoint)
    if not model.config.indexed_outcomes:
        raise ValueError("The relay requires values already indexed by actual endpoint identity")
    studies = []
    for sensitive in (4, 2):
        tape = make_batch(seed, lives, 64, sensitive_count=sensitive)
        policy = rollout_scout(ScoutRawAgent(model), tape)
        relay = rollout_scout(ReadoutDiagnostic(model), tape)
        removed = rollout_scout(ReadoutDiagnostic(model, write_enabled=False), tape)
        studies.append({"sensitive_count": sensitive, "policy": metrics(policy, tape),
                        "fixed_post_scout_motor_relay": metrics(relay, tape),
                        "fixed_relay_disabled_writes": metrics(removed, tape)})
    return {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "seeds": list(range(seed, seed + lives)), "studies": studies}


def evaluate_wrong_reads(checkpoint, seed, lives, *, sensitive_count):
    torch.set_num_threads(2)
    if lives < 2:
        raise ValueError("A donor read requires at least two distinct lives")
    model = load_scout_model(checkpoint)
    if not model.config.indexed_outcomes:
        raise ValueError("Wrong endpoint reads require identity-indexed values")
    tape = make_batch(seed, lives, 64, sensitive_count=sensitive_count)
    normal = rollout_scout(ScoutRawAgent(model), tape)
    late = np.arange(tape.trials)[None, :] >= tape.trials - 16
    desired_side = tape.left_id != tape.good_id
    mandatory = normal.sensitive_trial & (tape.scout_side != desired_side) & late
    samples = np.random.default_rng(994401).integers(0, tape.n, size=(2500, tape.n))
    report = {"true": metrics(normal, tape), "same_state": {}, "closed_loop": {}}
    for name in ("swap", "yoke"):
        matched = WrongReadAgent(model, mode=name, trials=tape.trials, only_probe=True)
        kept = rollout_scout(matched, tape)
        np.testing.assert_array_equal(kept.contact_id, normal.contact_id)
        known = (matched.true_at_probe >= 0) & mandatory
        wanted = 1 + desired_side
        correct = known & (matched.true_at_probe == wanted)
        report["same_state"][name] = {
            "physical_probe_rows": int(known.sum()), "true_direction_correct": int(correct.sum()),
            "changed_action": int((known & (matched.true_at_probe != matched.fake_at_probe)).sum()),
            "correct_turned_opposite": int((correct & (matched.fake_at_probe == 3 - wanted)).sum()),
        }
        result = rollout_scout(WrongReadAgent(model, mode=name, trials=tape.trials, only_probe=False), tape)
        gap = normal.correct.mean(axis=1) - result.correct.mean(axis=1)
        report["closed_loop"][name] = {"metrics": metrics(result, tape), "true_minus": float(gap.mean()),
                                       "ci95": np.quantile(gap[samples].mean(axis=1), [0.025, 0.975]).tolist()}
    return {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "seeds": list(range(seed, seed + lives)), "sensitive_count": sensitive_count, "results": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_359_500_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--wrong-reads", action="store_true")
    parser.add_argument("--sensitive-count", type=int, choices=range(5), default=4)
    args = parser.parse_args()
    if args.wrong_reads:
        report = evaluate_wrong_reads(args.checkpoint, args.seed, args.lives, sensitive_count=args.sensitive_count)
    else:
        report = evaluate(args.checkpoint, args.seed, args.lives)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.wrong_reads:
        print(json.dumps(report["results"]))
    else:
        for study in report["studies"]:
            print(json.dumps({"sensitive_count": study["sensitive_count"],
                              "policy": study["policy"]["accuracy"],
                              "relay": study["fixed_post_scout_motor_relay"]["accuracy"],
                              "no_write": study["fixed_relay_disabled_writes"]["accuracy"]}))
