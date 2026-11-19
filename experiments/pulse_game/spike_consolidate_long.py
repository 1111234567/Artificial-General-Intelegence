"""Evaluate repeated switch retention without feeding diagnostic probes back."""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import Consolidation
from .spike_consolidate_run import ACQUIRE, ASSESS, behavior, curriculum
from .spike_policy import SpikeConfig
from .world import PulseWorld, make_tape


def readout(policy, tape):
    probe = copy.deepcopy(policy)
    world = PulseWorld(tape, feedback="none")
    obs = world.observe()
    while obs is not None:
        obs = world.step(probe.act(obs))
    result = world.result()
    old = tape.cue < 2
    return {
        "old": ((result.correct & old).sum(axis=1) / old.sum(axis=1)).tolist(),
        "new": ((result.correct & ~old).sum(axis=1) / (~old).sum(axis=1)).tolist(),
    }


def life(policy, tape, probe_tape, checks):
    policy.reset(tape.n)
    world = PulseWorld(tape)
    obs = world.observe()
    diagnostics = []
    while obs is not None:
        obs = world.step(policy.act(obs))
        tick = world.trial_index
        if world.tick_index != 0 or tick not in checks:
            continue
        report = readout(policy, probe_tape)
        report["training_trials"] = tick
        if policy.mechanism.kind == "decorrelated":
            report["median_min_lateral_eigenvalue"] = float(np.median([
                np.linalg.eigvalsh(agent.lateral)[0] for agent in policy.agents
            ]))
        diagnostics.append(report)
        print(f"trial {tick:3d}, old {np.mean(report['old']):.3f}, new {np.mean(report['new']):.3f}", flush=True)
    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    candidate = Consolidation(**frozen["mechanism"])
    cfg = SpikeConfig(**frozen["config"])
    phases = 10
    checks = (ACQUIRE, ACQUIRE * 2, ACQUIRE * 3, ACQUIRE * 4, ACQUIRE * 6, ACQUIRE * 9, ACQUIRE * 10)
    codes = (3, 12)
    tape = curriculum(list(range(101401, 101417)), codes=codes, phases=phases)
    cutoff = ACQUIRE * phases
    arrays = {name: getattr(tape, name)[:, :cutoff] for name in ("ctx", "cue", "left_id", "good_id", "distances", "wind")}
    tape = replace(tape, **arrays)
    aside = make_tape(list(range(101501, 101517)), trials=ASSESS, map_codes=codes)
    labels = tape.good_map[np.arange(tape.n)[:, None], aside.ctx, aside.cue]
    aside = replace(aside, good_map=tape.good_map.copy(), good_id=labels)
    rows = []
    for seed in frozen["birth_seeds"]:
        for name, rule in (("ordinary", Consolidation()), ("candidate", candidate)):
            print(f"birth {seed}: {name}", flush=True)
            policy = behavior(tape, rule, cfg, seed)
            rows.append({"network_seed": seed, "method": name, "checkpoints": life(policy, tape, aside, checks)})
    summary = {}
    for trial in checks:
        scores = {name: {key: np.mean([pt[key] for row in rows for pt in row["checkpoints"]
                                      if row["method"] == name and pt["training_trials"] == trial], axis=0)
                         for key in ("old", "new")} for name in ("ordinary", "candidate")}
        summary[trial] = {name: {key: estimates(v) for key, v in values.items()} for name, values in scores.items()}
        summary[trial]["difference"] = {key: estimates(scores["candidate"][key] - scores["ordinary"][key])
                                       for key in ("old", "new")}
    args.artifact.write_text(json.dumps({"training_seeds": tape.seeds, "probe_seeds": aside.seeds,
                                         "summary": summary, "results": rows}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
