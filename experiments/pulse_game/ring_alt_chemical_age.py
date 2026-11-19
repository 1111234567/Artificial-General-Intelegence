"""Protect prior associations using the physical ages of recruited receptors."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec, ChemicalForager, ReceptorPopulation
from .ring_alt_chemical_shift import PaletteShift, audit_tape
from .ring_alt_run import preference_code, world_panel
from .ring_world import rollout
from .spike_analysis import estimates


class DatedReceptors(ReceptorPopulation):
    def reset(self, n):
        super().reset(n)
        self.born_at = np.full((n, self.capacity), -1, dtype=np.int32)

    def observe(self, trace, valid, trial, *, adapt=True):
        rows, branches = super().observe(trace, valid, trial, adapt=adapt)
        self.born_at[rows, branches] = trial
        return rows, branches


class AgeForager(ChemicalForager):
    def __init__(self, spec, *, birth_seed, action_seeds, credit, horizon):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds)
        rng = np.random.default_rng(np.random.SeedSequence((birth_seed, 563278)))
        self.reorder = rng.random(spec.capacity).astype(np.float32)
        branch = DatedReceptors(spec.fruit_bank, spec.capacity, spec, rng)
        branch.seed_weights = self.fruit.seed_weights.copy()
        self.fruit = branch
        self.credit = credit
        self.horizon = horizon

    def _post_credit(self, post, fire):
        original = super()._post_credit(post, fire)
        if self.credit == "ordinary" or not np.any(fire):
            return original
        leader = post[fire].argmax(axis=1)
        columns = np.arange(original.shape[1])
        if self.credit == "winner":
            gate = columns == leader[:, None]
        else:
            ages = self.fruit.born_at[fire]
            if self.credit == "sham":
                live = self.fruit.alive[fire]
                physical_order = np.argsort(~live, axis=1, kind="stable")
                shuffled_order = np.argsort(np.where(live, self.reorder, 2 + columns), axis=1)
                ages = ages.copy()
                np.put_along_axis(ages, physical_order, np.take_along_axis(ages, shuffled_order, axis=1), axis=1)
            age_leader = ages[np.arange(len(leader)), leader]
            gap = np.abs(ages - age_leader[:, None]).astype(np.float32)
            gate = np.exp(-gap / self.horizon)
        updated = original * gate
        size = np.linalg.norm(updated, axis=1)
        factor = np.divide(np.linalg.norm(original, axis=1), size,
                           out=np.zeros_like(size), where=size > 1e-8)
        return np.where(size[:, None] > 1e-8, factor[:, None] * updated, original)


def run(tape, probe, cfg, birth_seeds, style, modes, cut):
    records = []
    for seed in birth_seeds:
        for name, credit, horizon in modes:
            action = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
            brain = AgeForager(cfg, birth_seed=seed, action_seeds=action, credit=credit, horizon=horizon)
            interface = PaletteShift(brain, tape.seeds, style, cut=cut, assay=probe)
            results = rollout(interface, tape)
            first, final = interface.snapshots
            records.append({"name": name, "birth_seed": seed, "credit": credit, "horizon": horizon,
                            "old_before": first["old"], "old_after": final["old"], "new_after": final["new"],
                            "first_correct": results.first_correct.tolist(), "network": brain.stats()})
            print(f"{seed:2d} {name:20s} old {np.mean(first['old']):.3f} → {np.mean(final['old']):.3f}; new {np.mean(final['new']):.3f}", flush=True)
    cond = {name: {k: np.mean([row[k] for row in records if row["name"] == name], axis=0)
                   for k in ("old_before", "old_after", "new_after")} for name, _, _ in modes}
    summary = {name: {key: estimates(v) for key, v in agg.items()} for name, agg in cond.items()}
    for name, agg in cond.items():
        summary[name]["minus_ordinary"] = {key: estimates(v - cond["ordinary"][key]) for key, v in agg.items()}
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    cfg = ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    if args.selection is None:
        panel, base, n, cut, births = "development", 262501, 24, 96, (2, 5)
        modes = (("ordinary", "ordinary", 0), ("age-16", "age", 16), ("age-32", "age", 32),
                 ("age-64", "age", 64), ("sham-32", "sham", 32), ("winner-credit", "winner", 0))
    else:
        protocol = json.loads(args.selection.read_text(encoding="utf-8"))
        panel, base, n, cut, births = "evaluation", protocol["start"], protocol["worlds"], protocol["cut"], protocol["births"]
        chosen = protocol["horizon"]
        modes = (("ordinary", "ordinary", 0), ("age-gated", "age", chosen),
                 ("shuffled-age", "sham", chosen), ("winner-credit", "winner", 0))
    tape = world_panel(base, n, 2 * cut + 1, panel=panel)
    probe = audit_tape(tape, base + 4000, n, 64, panel)
    rows, summary = run(tape, probe, cfg, births, "novel", modes, cut)
    args.artifact.write_text(json.dumps({"spec": asdict(cfg), "life_seeds": tape.seeds, "probe_seeds": probe.seeds,
                                         "map_codes": preference_code(tape.preferences).tolist(),
                                         "summary": summary, "rows": rows}, separators=(",", ":")) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
