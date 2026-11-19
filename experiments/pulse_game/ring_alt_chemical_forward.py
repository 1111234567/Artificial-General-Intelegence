"""Test whether age-dependent sensory competition limits old/new interference."""

import argparse
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec, ChemicalForager
from .ring_alt_chemical_age import DatedReceptors
from .ring_alt_chemical_shift import PaletteShift, audit_tape
from .ring_alt_run import preference_code, world_panel
from .ring_world import rollout
from .spike_analysis import estimates


class AgeCompetingReceptors(DatedReceptors):
    def __init__(self, mode, capacity, spec, rng, credit, horizon):
        super().__init__(mode, capacity, spec, rng)
        self.order_salt = rng.random(capacity).astype(np.float32)
        self.credit, self.horizon = credit, horizon

    def match(self, trace):
        response = super().match(trace)
        if self.credit == "ordinary":
            return response
        leader = response.argmax(axis=1)
        columns = np.arange(self.capacity)
        if self.credit == "winner":
            gate = columns == leader[:, None]
        else:
            ages = self.born_at
            if self.credit == "sham":
                live = self.alive
                physical_order = np.argsort(~live, axis=1, kind="stable")
                random_order = np.argsort(np.where(live, self.order_salt, 2 + columns), axis=1)
                ages = ages.copy()
                np.put_along_axis(ages, physical_order, np.take_along_axis(ages, random_order, axis=1), axis=1)
            gap = np.abs(ages - ages[np.arange(len(leader)), leader, None]).astype(np.float32)
            gate = np.exp(-gap / self.horizon)
        response *= gate
        return response / np.maximum(response.sum(axis=1, keepdims=True), 1e-8)


class CohortForager(ChemicalForager):
    def __init__(self, spec, *, birth_seed, action_seeds, rule, horizon):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds)
        rng = np.random.default_rng(np.random.SeedSequence((birth_seed, 563278)))
        changed = AgeCompetingReceptors(spec.fruit_bank, spec.capacity, spec, rng, rule, horizon)
        changed.seed_weights = self.fruit.seed_weights.copy()
        self.fruit = changed


def run(tape, probe, cfg, birth_seeds, protocol):
    modes = ("ordinary", "age", "sham", "winner")
    results = []
    for seed in birth_seeds:
        for mode in modes:
            action = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
            brain = CohortForager(cfg, birth_seed=seed, action_seeds=action, rule=mode, horizon=protocol["horizon"])
            scope = PaletteShift(brain, tape.seeds, "novel", cut=protocol["cut"], assay=probe)
            trajectory = rollout(scope, tape)
            initial, final = scope.snapshots
            results.append({"birth": seed, "name": mode, "old_before": initial["old"],
                            "old_after": final["old"], "new_after": final["new"],
                            "network": brain.stats(), "first_correct": trajectory.first_correct.tolist()})
            print(f"{seed:2d} {mode:10s} old {np.mean(initial['old']):.3f}→{np.mean(final['old']):.3f}; "
                  f"new {np.mean(final['new']):.3f}", flush=True)
    keys = ("old_before", "old_after", "new_after")
    all_scores = {mode: {key: np.mean([r[key] for r in results if r["name"] == mode], axis=0) for key in keys}
                  for mode in modes}
    summary = {mode: {key: estimates(x) for key, x in group.items()} for mode, group in all_scores.items()}
    for mode, group in all_scores.items():
        summary[mode]["minus_ordinary"] = {k: estimates(x - all_scores["ordinary"][k]) for k, x in group.items()}
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    cfg = ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    setup = {"start": 263501, "cut": 96, "n": 28, "horizon": 32, "births": [2, 5], "panel": "development"}
    if args.protocol:
        setup = json.loads(args.protocol.read_text(encoding="utf-8"))
    tape = world_panel(setup["start"], setup["n"], 2 * setup["cut"] + 1, panel=setup["panel"])
    probe = audit_tape(tape, setup["start"] + 4000, setup["n"], 64, setup["panel"])
    rows, summary = run(tape, probe, cfg, setup["births"], setup)
    args.artifact.write_text(json.dumps({"protocol": setup, "train_seeds": tape.seeds, "probe_seeds": probe.seeds,
                                         "map_codes": preference_code(tape.preferences).tolist(), "summary": summary,
                                         "rows": rows}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
