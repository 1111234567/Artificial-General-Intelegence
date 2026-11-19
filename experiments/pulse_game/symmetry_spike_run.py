"""Matched native-body experiments for the compact synaptic skill model."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from . import mixture_flash, symmetry_world, world
from .spike_analysis import estimates
from .symmetry_spike import Skill, SpikingSkill


class Receipts:
    """Evaluator masks unsupervised queries without sending a query signal."""
    def __init__(self, brain, protocol, mode):
        self.brain, self.protocol, self.mode = brain, protocol, mode

    def reset(self, n):
        self.brain.reset(n)
        self.trial = -1

    def act(self, obs):
        onset = obs[0, world.TRIAL_START] > 0
        self.trial += int(onset)
        pulse = obs.copy()
        probe = self.protocol.query[:, self.trial]
        if self.mode == "hide-a" and self.trial < self.protocol.exposed:
            probe = np.ones_like(probe)
        pulse[probe, world.REWARD_POS:world.REWARD_NEG + 1] = 0.
        if self.mode == "flip-a" and self.trial < self.protocol.exposed:
            pulse[:, world.REWARD_POS:world.REWARD_NEG + 1] = pulse[:, world.REWARD_POS:world.REWARD_NEG + 1][:, ::-1]
        if self.mode == "invert-gain" and onset and self.trial == self.protocol.exposed + 1:
            self.brain.reverse_direction()
        return self.brain.act(pulse)


def paired(tests, skill, birth, *, mode="actual", actor=SpikingSkill):
    result = []
    brain_states = []
    sides = []
    early_observed = []
    probe_index = tests[0].exposed + 1
    for p in tests:
        act = [np.random.SeedSequence((s, 0x5B983C)) for s in p.menu.body.seeds]
        brain = actor(skill, birth=birth, action_seeds=act)
        played = mixture_flash.rollout(Receipts(brain, p, mode), p.menu)
        metrics = symmetry_world.metrics(p, played)
        np.testing.assert_array_equal(metrics["first"].astype(bool), played.reward[:, probe_index] > 0)
        result.append(metrics)
        brain_states.append({"direction_at_first_probe": brain.direction_history[probe_index].tolist(),
                             "recall_at_first_probe": brain.recall_energy[probe_index].tolist(),
                             "receipts": brain.receipts.tolist(), "spikes": brain.total_spikes.tolist()})
        sides.append(played.contact_side[:, probe_index])
        early_observed.append(played.observed_reward[:, :p.exposed])
    same = (early_observed[0] == early_observed[1]).all(axis=(1, 2))
    actual_bites = (sides[0] >= 0) & (sides[1] >= 0)
    first = .5 * (result[0]["first"] + result[1]["first"])
    paired_both = result[0]["first"] * result[1]["first"]
    changed = (sides[0] != sides[1]) & actual_bites
    return {"skill": asdict(skill), "birth": birth, "policy_receipts": mode,
            "all_correct": first.tolist(), "both_correct": paired_both.tolist(),
            "switched_side": changed.tolist(), "both_contact": actual_bites.tolist(),
            "same_a_observed_waveform": same.tolist(),
            "family_rows": [{k: v.tolist() for k, v in row.items()} for row in result], "state": brain_states}


def make_pair(seeds, *, probes=8, permuted=False, jitter=.08, noise=.04):
    args = dict(pairs=12, probes=probes, interleave=4, noise=noise, jitter=jitter,
                permute_signs=permuted, balanced_stream=True)
    tables = tuple(symmetry_world.make(seeds, family=f, **args) for f in (1, -1))
    k = tables[0].exposed
    np.testing.assert_array_equal(tables[0].menu.realized[:, :k], tables[1].menu.realized[:, :k])
    np.testing.assert_array_equal(tables[0].menu.chemicals[:, k:], tables[1].menu.chemicals[:, k:])
    return tables


def brief(rows):
    stats = ["all_correct", "both_correct", "switched_side", "both_contact", "same_a_observed_waveform"]
    groups = {}
    for name in dict.fromkeys(r["name"] for r in rows):
        group = [r for r in rows if r["name"] == name]
        value = {k: estimates(np.mean([r[k] for r in group], axis=0)) for k in stats}
        value["family_correct"] = [estimates(np.mean([r["family_rows"][i]["first"] for r in group], axis=0)) for i in (0, 1)]
        value["births"] = {str(r["birth"]): float(np.mean(r["all_correct"])) for r in group}
        value["common_world_contrast"] = {}
        if any(x["name"] == "native" for x in rows):
            baseline = [r for r in rows if r["name"] == "native"]
            for k in ("all_correct", "both_correct", "switched_side"):
                value["common_world_contrast"][k] = estimates(np.mean([r[k] for r in group], axis=0) -
                                                               np.mean([r[k] for r in baseline], axis=0))
        groups[name] = value
    return groups


def write_row(name, tests, config, seed, *, mode="actual", actor=SpikingSkill):
    row = paired(tests, config, seed, mode=mode, actor=actor)
    row["name"] = name
    print(f"{seed:4d} {name:27s} first={np.mean(row['all_correct']):.3f} both={np.mean(row['both_correct']):.3f} "
          f"switch={np.mean(row['switched_side']):.3f} contact={np.mean(row['both_contact']):.3f} "
          f"sameA={np.mean(row['same_a_observed_waveform']):.3f}", flush=True)
    return row


def presets():
    base = Skill()
    cells = [(name, replace(base, encoding=name)) for name in ("cubic", "linear", "even", "opponent", "random", "centered", "voltage")]
    for e in ("opponent", "random", "centered", "voltage"):
        for key, value in (("memory_hold", .72), ("memory_hold", .92), ("signal", 2.1),
                           ("signal", .88), ("receptor", .68), ("feature_trace", .82),
                           ("recurrence", .95), ("update_gain", "correlation"), ("address_power", 2),
                           ("meta_fast_hold", .48)):
            cells.append((f"{e}-{key}-{value}", replace(base, encoding=e, **{key: value})))
        cells.append((f"{e}-two-rates", replace(base, encoding=e, memory_hold=.92, meta_fast_hold=.48)))
        cells.append((f"{e}-quadratic-two-rates", replace(base, encoding=e, address_power=2,
                                                        memory_hold=.92, meta_fast_hold=.48)))
    return cells


def confirm(seeds, config, birth_seeds):
    truth, scramble = make_pair(seeds), make_pair(seeds, permuted=True)
    arms = (
        ("native", truth, config, "actual"),
        ("fixed_positive", truth, replace(config, fixed_gain=1.), "actual"),
        ("fixed_negative", truth, replace(config, fixed_gain=-1.), "actual"),
        ("fixed_zero", truth, replace(config, fixed_gain=0.), "actual"),
        ("mute_a", truth, config, "hide-a"),
        ("invert_a", truth, config, "flip-a"),
        ("invert_gain_at_first_b", truth, config, "invert-gain"),
        ("permute_signs_a", scramble, config, "actual"),
        ("cut_recurrence", truth, replace(config, recurrence=0.), "actual"),
        ("shared_habitat_weights", truth, replace(config, share_habitats=True), "actual"),
        ("single_spike_trace_readout", truth, replace(config, encoding="centered"), "actual"),
        ("generic_linear_reference", truth, replace(config, encoding="linear"), "actual"),
        ("generic_cubic_reference", truth, replace(config, encoding="cubic"), "actual"),
    )
    return [write_row(name, tables, spec, seed, mode=mode) for seed in birth_seeds for name, tables, spec, mode in arms]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("discover", "confirm"))
    p.add_argument("artifact", type=Path)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--births", type=int, nargs="+", required=True)
    p.add_argument("--frozen", type=Path)
    args = p.parse_args()
    seeds = tuple(range(args.start, args.start + args.n))
    if args.stage == "discover":
        tested = make_pair(seeds, probes=1)
        rows = [write_row(name, tested, spec, seed) for seed in args.births for name, spec in presets()]
    else:
        spec = Skill(**json.loads(args.frozen.read_text(encoding="utf-8"))["skill"])
        rows = confirm(seeds, spec, args.births)
    output = {"stage": args.stage, "seeds": seeds, "births": args.births, "pairs": 12, "interleave": 4,
              "balanced_stream": True, "rows": rows, "summary": brief(rows)}
    args.artifact.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
