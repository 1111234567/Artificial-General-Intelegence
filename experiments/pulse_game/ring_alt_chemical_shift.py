"""Within-life odor replacement and independent quiet recall of both palettes."""

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec
from .ring_alt_chemical_run import actor
from .ring_alt_run import preference_code, world_panel
from .ring_sensors import ChemicalInterface, codebooks
from .ring_world import NEGATIVE, NEW, POSITIVE, rollout
from .spike_analysis import estimates


class Continue:
    def __init__(self, model):
        self.model = model

    def reset(self, n):
        if len(self.model.v) != n:
            raise ValueError("probe requires a clone of the same physical lives")

    def act(self, obs):
        return self.model.act(obs)


class PaletteShift(ChemicalInterface):
    """Evaluator schedules odors; the agent receives neither phase nor codes."""

    def __init__(self, child, seeds, style, *, cut, change_cue=False, quiet=None, assay=None, frozen="none"):
        super().__init__(child, seeds, style)
        shifted = [int(s) + 0x1000000 for s in seeds]
        self.old_cues, self.old_fruits = self.cues, self.fruits
        self.new_cues = codebooks(shifted, 1, style) if change_cue else self.old_cues
        self.new_fruits = codebooks(shifted, 2, style)
        self.cut = int(cut)
        self.quiet = quiet
        self.assay = assay
        self.frozen = frozen
        self.distance = np.linalg.norm(self.old_fruits[:, :, None] - self.new_fruits[:, None, :], axis=3)

    def reset(self, n):
        super().reset(n)
        self.cues, self.fruits = self.old_cues, self.old_fruits
        self.trial = -1
        self.snapshots = []

    def _measure(self):
        stats = self.child.stats()
        result = {"before_trial": self.trial, "active_fruit": stats["fruit"]["active"],
                  "births": stats["fruit"]["births"], "retirements": stats["fruit"]["retirements"],
                  "cumulative_erased_weight_l2": stats["retired_weight_l2"],
                  "prototype_writes": stats["fruit"]["prototype_writes"],
                  "prototype_write_l2": stats["fruit"]["prototype_l2"]}
        for label, cues, fruits in (("old", self.old_cues, self.old_fruits), ("new", self.new_cues, self.new_fruits)):
            fork = copy.deepcopy(self.child)
            fork.adapt_receptors = fork.adapt_association = False
            fork.force_phase = False
            quiet = ChemicalInterface(Continue(fork), self.assay.seeds, self.style)
            quiet.cues, quiet.fruits = cues, fruits
            record = rollout(quiet, self.assay, feedback="none")
            result[label] = record.first_correct.mean(axis=1).tolist()
            result[label + "_contact"] = (record.bites > 0).mean(axis=1).tolist()
        self.snapshots.append(result)

    def act(self, obs):
        if obs[0, NEW]:
            self.trial += 1
            if self.assay is not None and self.trial in (self.cut, 2 * self.cut):
                self._measure()
            if self.trial == self.cut:
                self.cues, self.fruits = self.new_cues, self.new_fruits
                if self.frozen in {"receptors", "all"}:
                    self.child.adapt_receptors = False
                if self.frozen in {"association", "all"}:
                    self.child.adapt_association = False
        if self.quiet is not None and self.trial >= self.quiet:
            obs = obs.copy()
            obs[:, POSITIVE:NEGATIVE + 1] = 0.
        return super().act(obs)


def audit_tape(train, start, n, trials, panel):
    tape = world_panel(start, n, trials, panel=panel)
    good = train.preferences[np.arange(n)[:, None], tape.ctx, tape.cue]
    return replace(tape, preferences=train.preferences.copy(), goal=good)


def run(tape, assay, spec, births, style, *, cut, change_cue):
    modes = [(f"growing-{capacity}", replace(spec, capacity=capacity), "none") for capacity in (2, 4, 8, 12)]
    modes += [("competitive-8", replace(spec, fruit_bank="competitive"), "none"),
              ("same-rate-growing-8", replace(spec, fruit_bank="growth-fixed-writes"), "none"),
              ("fixed-bank-reassignment-8", replace(spec, fruit_bank="reassign-fixed"), "none"),
              ("receptor-freeze-8", spec, "receptors"), ("association-freeze-8", spec, "association"),
              ("freeze-all-8", spec, "all")]
    records = []
    for seed in births:
        for name, config, frozen in modes:
            model = actor(tape, config, seed)
            interface = PaletteShift(model, tape.seeds, style, cut=cut, change_cue=change_cue,
                                     assay=assay, frozen=frozen)
            trace = rollout(interface, tape)
            records.append({"name": name, "birth": seed, "spec": asdict(config), "frozen": frozen,
                            "phases": interface.snapshots, "first_correct": trace.first_correct.tolist(),
                            "scores_early_shift": trace.first_correct[:, cut:cut + 32].mean(axis=1).tolist(),
                            "scores_late_shift": trace.first_correct[:, 2 * cut - 32:2 * cut].mean(axis=1).tolist()})
            first, last = interface.snapshots
            print(f"{seed} {name:25s} first old/new {np.mean(first['old']):.3f}/{np.mean(first['new']):.3f} "
                  f"last {np.mean(last['old']):.3f}/{np.mean(last['new']):.3f}; "
                  f"retired {np.mean(last['retirements']):.2f}, active {np.mean(last['active_fruit']):.2f}", flush=True)
    summary = {}
    names = tuple(dict.fromkeys(row["name"] for row in records))
    values = {name: {f"phase_{phase}_{key}": np.mean([row["phases"][phase][key] for row in records if row["name"] == name], axis=0)
                     for phase in (0, 1) for key in ("old", "new", "retirements", "active_fruit", "prototype_writes",
                                                     "prototype_write_l2", "cumulative_erased_weight_l2")}
              for name in names}
    for name, group in values.items():
        summary[name] = {k: estimates(v) for k, v in group.items()}
        summary[name]["old_late_minus_old_early"] = estimates(group["phase_1_old"] - group["phase_0_old"])
        summary[name]["new_late_minus_new_early"] = estimates(group["phase_1_new"] - group["phase_0_new"])
        summary[name]["late_old_minus_growing8"] = estimates(group["phase_1_old"] - values["growing-8"]["phase_1_old"])
        summary[name]["late_new_minus_growing8"] = estimates(group["phase_1_new"] - values["growing-8"]["phase_1_new"])
    conflicting = interface.distance[:, ~np.eye(interface.distance.shape[1], dtype=bool)]
    return records, summary, {"all": interface.distance.min(axis=(1, 2)).tolist(),
                              "different_target_labels": conflicting.min(axis=1).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--cut", type=int, default=128)
    parser.add_argument("--style", choices=("novel", "noisy"), default="novel")
    parser.add_argument("--panel", choices=("evaluation", "development"), default="evaluation")
    parser.add_argument("--also-cue", action="store_true")
    args = parser.parse_args()
    config = replace(ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"]), capacity=8)
    tape = world_panel(args.start, args.n, 2 * args.cut + 1, panel=args.panel)
    assay = audit_tape(tape, args.start + 4000, args.n, 64, args.panel)
    rows, summary, cross = run(tape, assay, config, args.births, args.style, cut=args.cut, change_cue=args.also_cue)
    out = {"train_seeds": tape.seeds, "audit_seeds": assay.seeds, "map_codes": preference_code(tape.preferences).tolist(),
           "style": args.style, "cue_changes": args.also_cue, "min_old_new_odor_distances": cross,
           "trials_before_odor_shift": args.cut, "summary": summary, "rows": rows}
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
