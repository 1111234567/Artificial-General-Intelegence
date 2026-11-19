"""Independent world-level analysis of agents acting in the native pulse game."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from . import world
from experiments.seed_evolution.large_runner import _write


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "pulse-game" / "research-2026-11-19"
SPLITS_SHA256 = "bde8c35f51c13d45a1c5c538da7914bf24b6973fadd5b6be47dd65a9abb16e25"


def split(name):
    raw = (OUT / "world-splits.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLITS_SHA256:
        raise ValueError("World split commitment differs")
    return json.loads(raw)[name]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimates(values, *, clusters=1):
    """Equal-weight lives; adjacent lives form one cluster in a paired reward yoke."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) % clusters or not np.isfinite(x).all():
        raise ValueError("Expected finite complete life clusters")
    means = x.reshape(-1, clusters).mean(1)
    rng = np.random.default_rng(9247)
    draws = rng.integers(len(means), size=(4096, len(means)))
    ci = np.quantile(means[draws].mean(1), [.025, .975]).tolist() if len(means) > 1 else None
    return {"mean": float(x.mean()), "ci95": ci, "lives": len(x), "clusters": len(means)}


def pair_counts(ctx, cue, correct):
    """Count one-to-one A/B revisits, after both contexts for that cue have appeared."""
    n = len(ctx)
    num = np.zeros(n, dtype=np.int32)
    den = np.zeros(n, dtype=np.int32)
    for actor in range(n):
        for symbol in np.unique(cue[actor]):
            visits = [np.flatnonzero((ctx[actor] == context) & (cue[actor] == symbol)) for context in (0, 1)]
            if not all(len(v) for v in visits):
                continue
            after = max(v[0] for v in visits)
            a, b = [v[v > after] for v in visits]
            count = min(len(a), len(b))
            num[actor] += np.count_nonzero(correct[actor, a[:count]] & correct[actor, b[:count]])
            den[actor] += count
    return num, den


def life_rows(tape, result):
    contacted = result.contact_id >= 0
    correct = np.asarray(result.correct, dtype=bool)
    if not np.array_equal(correct, contacted & (result.contact_id == tape.good_id)):
        raise ValueError("Contact and correctness fields disagree")
    num, den = pair_counts(tape.ctx, tape.cue, correct)
    half = tape.trials // 2
    first = min(8, half)
    out = {"all_accuracy": correct.mean(1), "early_accuracy": correct[:, :first].mean(1),
           "late_accuracy": correct[:, half:].mean(1), "contact_rate": contacted.mean(1),
           "late_contact": contacted[:, half:].mean(1), "pair_successes": num, "pairs": den}
    # Coverage of the pairing diagnostic is exogenous and identical in every arm.
    out["pair_rate"] = np.divide(num, den, out=np.full(tape.n, np.nan), where=den > 0)
    out["learning_gain"] = out["late_accuracy"] - out["early_accuracy"]
    for t in range(0, tape.trials, 8):
        out[f"trials_{t+1}_{min(t+8,tape.trials)}"] = correct[:, t:t+8].mean(1)
    return {k: v.tolist() for k, v in out.items()}


class ContactMatched:
    """Present an unrelated donor's outcome sign at the recipient's actual contact."""

    def __init__(self, policy, donor_sign):
        self.policy = policy
        self.sign = np.asarray(donor_sign)

    def reset(self, n):
        if len(self.sign) != n:
            raise ValueError("Donor and recipient count differs")
        self.t = -1
        self.policy.reset(n)

    def act(self, obs):
        if np.all(obs[:, world.TRIAL_START]):
            self.t += 1
        frame = obs.copy()
        contact = obs[:, world.CONTACT_ID].any(1)
        frame[:, world.REWARD_POS] = contact & (self.sign[:, self.t] > 0)
        frame[:, world.REWARD_NEG] = contact & (self.sign[:, self.t] < 0)
        return self.policy.act(frame)


def iid_outcomes(tape):
    """Break all cue/context-to-reward predictability without changing public cues or layout."""
    identities = np.stack([np.random.default_rng(np.random.SeedSequence([s, 5318391])).integers(
        2, size=tape.trials, dtype=np.int8) for s in tape.seeds])
    return replace(tape, good_id=identities)


def normal_and_controls(factory, tape):
    if tape.n % 2:
        raise ValueError("Pair-exchanged schedules need an even number of lives")
    timed = world.rollout(factory(), tape)
    rows = np.arange(tape.n) ^ 1
    yoke = world.YokeSchedule(timed.observed_reward[rows].copy(), rows, tuple(tape.seeds[i] for i in rows))
    results = {
        "true": life_rows(tape, timed),
        "no_outcome": life_rows(tape, world.rollout(factory(), tape, feedback="none")),
        "yoked": life_rows(tape, world.rollout(factory(), tape, feedback="yoked", yoke=yoke)),
        "contact_matched_yoke": life_rows(tape, world.rollout(ContactMatched(factory(), timed.reward[rows]), tape)),
    }
    scrambled = iid_outcomes(tape)
    results["unpredictable"] = life_rows(scrambled, world.rollout(factory(), scrambled))
    return results


def overview(conditions):
    cols = ("all_accuracy", "early_accuracy", "late_accuracy", "learning_gain", "contact_rate", "pair_rate")
    summary = {}
    for name, rows in conditions.items():
        summary[name] = {k: estimates(np.array(rows[k]), clusters=2) for k in cols}
    contrasts = {name: {k: estimates(np.subtract(conditions["true"][k], rows[k]), clusters=2)
                        for k in ("late_accuracy", "contact_rate", "pair_rate")}
                 for name, rows in conditions.items() if name not in ("true", "unpredictable")}
    return summary, contrasts


def write_audit(name, metadata, tape, conditions):
    summary, contrasts = overview(conditions)
    path = OUT / f"{name}.json"
    _write(path, {"schema": 1, **metadata, "world_seeds": list(tape.seeds), "trials": tape.trials,
                  "map_codes": tape.map_codes.tolist(), "summary": summary, "true_minus_control": contrasts,
                  "per_world": conditions, "analysis_sha256": fingerprint(__file__),
                  "world_sha256": fingerprint(world.__file__)})
    return path
