"""Replay the same physical bites with matched realized plastic write norms."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec, ChemicalForager, ReceptorPopulation, unit
from .ring_alt_chemical_run import actor
from .ring_alt_chemical_shift import PaletteShift, audit_tape
from .ring_alt_run import preference_code, world_panel
from .ring_world import NEGATIVE, NEW, POSITIVE, TASTED, TICKS, rollout
from .spike_analysis import estimates


class MatchedPopulation(ReceptorPopulation):
    """Competitive centers use cosine readout and a replayed Euclidean step."""

    def similarity(self, incoming, trace):
        return super().similarity(unit(incoming), trace)

    def reset(self, n):
        super().reset(n)
        self.budget = np.zeros(n, dtype=np.float32)

    def observe(self, trace, valid, trial, *, adapt=True):
        observed = np.flatnonzero(valid)
        self.presentations[observed] += 1
        rows = observed[self.budget[observed] > 0] if adapt else observed[:0]
        if rows.size:
            targets = trace[rows]
            scores = self.similarity(self.prototype[rows], targets) - .045 * self.usage[rows]
            favored = scores.argmax(axis=1)
            old = self.prototype[rows, favored]
            dz = targets - old
            close = np.linalg.norm(dz, axis=1) <= 1e-8
            dz[close] = self.seed_weights[(favored[close] + 1) % self.capacity] - old[close]
            # Exact weight-space budgets can overshoot a nearby target. Cosine
            # normalization occurs at readout, leaving the actual write norm auditable.
            new = old + self.budget[rows, None] * unit(dz)
            self.prototype[rows, favored] = new
            self.centroid_l2[rows] += np.linalg.norm(new - old, axis=1)
            self.centroid_writes[rows] += 1
            self.last[rows, favored] = trial
            self.usage[rows] *= .97
            self.usage[rows, favored] += .03
        return observed[:0], observed[:0]


class MatchActor(ChemicalForager):
    def __init__(self, spec, *, birth_seed, action_seeds):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds)
        normal = self.fruit
        self.fruit = MatchedPopulation("competitive", spec.capacity, spec, np.random.default_rng(birth_seed))
        self.fruit.seed_weights = normal.seed_weights.copy()

    def reset(self, n):
        super().reset(n)
        self.association_budget = np.zeros(n, dtype=np.float32)

    def _receipt(self, obs):
        seen = (obs[:, TASTED] > 0).any(axis=1)
        sign = obs[:, POSITIVE] - obs[:, NEGATIVE]
        indicated = (sign < 0) if self.spec.outcomes == "negative" else (sign != 0)
        ids = np.flatnonzero(seen & indicated) if self.adapt_association else np.empty(0, np.intp)
        if not ids.size:
            return super()._receipt(obs)
        before = self.weights[ids].copy()
        total = self.weight_l2_sum[ids].copy()
        super()._receipt(obs)
        dw = self.weights[ids] - before
        norm = np.linalg.norm(dw.reshape(len(ids), -1), axis=1)
        goal = self.association_budget[ids]
        scale = np.divide(goal, norm, out=np.zeros_like(goal), where=norm > 1e-9)
        moved = np.clip(before + dw * scale[:, None, None], -.85, .85)
        error = goal - np.linalg.norm((moved - before).reshape(len(ids), -1), axis=1)
        for i in np.flatnonzero((error > 1e-6) & (norm > 1e-9)):
            # Find the scalar on the same synapse-local direction after clipping.
            a, b = float(scale[i]), 1024. * max(float(scale[i]), 1.)
            for _ in range(22):
                mid = (a + b) / 2
                if np.linalg.norm(np.clip(before[i] + mid * dw[i], -.85, .85) - before[i]) < goal[i]:
                    a = mid
                else:
                    b = mid
            moved[i] = np.clip(before[i] + b * dw[i], -.85, .85)
        self.weights[ids] = moved
        self.weight_l2_sum[ids] = total + np.linalg.norm((moved - before).reshape(len(ids), -1), axis=1)


class BudgetTrack(PaletteShift):
    def __init__(self, child, seeds, style, *, cut, assay, budgets=None):
        super().__init__(child, seeds, style, cut=cut, assay=assay, quiet=2 * cut)
        self.budgets = budgets

    def reset(self, n):
        super().reset(n)
        self.child.force_phase = True
        self.steps = np.zeros((2, n, 2 * self.cut + 1, TICKS), dtype=np.float32)

    def act(self, obs):
        start = bool(obs[0, NEW])
        trial = self.trial + int(start)
        tick = 0 if start else self.child.clock + 1
        if self.budgets is not None:
            self.child.fruit.budget = self.budgets[0, :, trial, tick]
            self.child.association_budget = self.budgets[1, :, trial, tick]
        a = self.child.fruit.centroid_l2.copy()
        b = self.child.weight_l2_sum.copy()
        pulses = super().act(obs)
        self.steps[0, :, trial, tick] = self.child.fruit.centroid_l2 - a
        self.steps[1, :, trial, tick] = self.child.weight_l2_sum - b
        return pulses


def run(tape, probe, cfg, seeds, style, *, cut):
    rows = []
    choices = ("growing", "competitive", "competitive-matched", "fixed-bank-reassignment")
    for seed in seeds:
        source = None
        reference = None
        for name in choices:
            spec = cfg if name == "growing" else replace(
                cfg, fruit_bank="reassign-fixed" if name == "fixed-bank-reassignment" else "competitive")
            if name == "competitive-matched":
                action = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
                model = MatchActor(spec, birth_seed=seed, action_seeds=action)
            else:
                model = actor(tape, spec, seed)
            schedule = BudgetTrack(model, tape.seeds, style, cut=cut, assay=probe,
                                   budgets=source if name == "competitive-matched" else None)
            results = rollout(schedule, tape)
            if source is None:
                source, reference = schedule.steps.copy(), results
            else:
                for a, b in ((results.first, reference.first), (results.first_tick, reference.first_tick),
                             (results.motion, reference.motion), (results.waveform, reference.waveform)):
                    np.testing.assert_array_equal(a, b)
            edits = schedule.steps[:, :, :2 * cut]
            target = source[:, :, :2 * cut]
            total = edits.sum(axis=(1, 2, 3), dtype=np.float64)
            ref_total = target.sum(axis=(1, 2, 3), dtype=np.float64)
            mismatch = np.abs(edits - target)
            row = {"name": name, "birth": seed, "snapshots": schedule.snapshots, "numerical_budget": total.tolist(),
                   "absolute_mismatch": mismatch.sum(axis=(1, 2, 3), dtype=np.float64).tolist(),
                   "ratio_absolute_mismatch": (mismatch.sum(axis=(1, 2, 3), dtype=np.float64) / np.maximum(ref_total, 1e-9)).tolist(),
                   "effective_nonzero_edits": (edits > 1e-7).sum(axis=(1, 2, 3)).tolist()}
            rows.append(row)
            probe_last = schedule.snapshots[-1]
            print(f"{seed:2d} {name:24s} quiet old/new {np.mean(probe_last['old']):.3f}/{np.mean(probe_last['new']):.3f}; "
                  f"prototype/association budget error {row['ratio_absolute_mismatch']}", flush=True)
    vals = {name: {key: np.mean([row["snapshots"][-1][key] for row in rows if row["name"] == name], axis=0)
                   for key in ("old", "new")} for name in choices}
    summary = {name: {key: estimates(x) for key, x in values.items()} for name, values in vals.items()}
    summary["growth_minus_matched"] = {key: estimates(vals["growing"][key] - vals["competitive-matched"][key])
                                       for key in ("old", "new")}
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--cut", type=int, default=128)
    parser.add_argument("--style", choices=("novel", "noisy"), default="novel")
    args = parser.parse_args()
    cfg = replace(ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"]), capacity=8)
    tape = world_panel(args.start, args.n, 2 * args.cut + 1, panel="evaluation")
    probe = audit_tape(tape, args.start + 4000, args.n, 64, "evaluation")
    rows, summary = run(tape, probe, cfg, args.births, args.style, cut=args.cut)
    args.artifact.write_text(json.dumps({"life_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(),
                                         "style": args.style, "phases": args.cut, "rows": rows, "summary": summary},
                                        separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
