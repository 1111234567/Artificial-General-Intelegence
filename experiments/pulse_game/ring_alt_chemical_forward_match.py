"""Replay age-gated and ordinary sensory codes on an identical physical track."""

import argparse
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec
from .ring_alt_chemical_forward import AgeCompetingReceptors, CohortForager
from .ring_alt_chemical_match import BudgetTrack, MatchActor
from .ring_alt_chemical_shift import audit_tape
from .ring_alt_run import preference_code, world_panel
from .ring_world import rollout
from .spike_analysis import estimates


class MatchedCohort(MatchActor):
    def __init__(self, spec, *, birth_seed, action_seeds, horizon):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds)
        original = self.fruit.seed_weights.copy()
        rng = np.random.default_rng(np.random.SeedSequence((birth_seed, 563278)))
        self.fruit = AgeCompetingReceptors(spec.fruit_bank, spec.capacity, spec, rng, "age", horizon)
        self.fruit.seed_weights = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    cfg = ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    registration = json.loads(args.protocol.read_text(encoding="utf-8"))
    core = registration["matched"]
    start, n, cut = core["start"], core["n"], core["cut"]
    tape = world_panel(start, n, 2 * cut + 1, panel="evaluation")
    probe = audit_tape(tape, start + 4000, n, 64, "evaluation")
    horizon = registration["horizon"]
    arms = ("ordinary", "age", "age-matched")
    observed = []
    for birth in core["births"]:
        reference = None
        budget_source = None
        original_receivers = None
        for name in arms:
            seeds = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
            settings = {"birth_seed": birth, "action_seeds": seeds, "horizon": horizon}
            model = (MatchedCohort(cfg, **settings) if name == "age-matched"
                     else CohortForager(cfg, rule=name, **settings))
            scope = BudgetTrack(model, tape.seeds, "novel", cut=cut, assay=probe,
                                budgets=budget_source if name == "age-matched" else None)
            outcomes = rollout(scope, tape)
            if reference is None:
                reference = outcomes
                budget_source = scope.steps.copy()
                original_receivers = model.fruit.prototype.copy()
            else:
                np.testing.assert_array_equal(reference.first, outcomes.first)
                np.testing.assert_array_equal(reference.first_tick, outcomes.first_tick)
                np.testing.assert_array_equal(reference.motion, outcomes.motion)
                np.testing.assert_array_equal(reference.waveform, outcomes.waveform)
                np.testing.assert_array_equal(original_receivers, model.fruit.prototype)
            mismatch = np.abs(scope.steps - budget_source).sum(axis=(1, 2, 3), dtype=np.float64)
            total = budget_source.sum(axis=(1, 2, 3), dtype=np.float64)
            errors = mismatch / np.maximum(total, 1e-8)
            captured = scope.snapshots[-1]
            observed.append({"name": name, "birth_seed": birth, "only_probe": captured,
                             "relative_prototype_and_association_errors": errors.tolist()})
            print(f"{birth:2d} {name:16s} old/new {np.mean(captured['old']):.3f}/{np.mean(captured['new']):.3f}, "
                  f"budget residual {errors.tolist()}", flush=True)
    cond = {name: {key: np.mean([r["only_probe"][key] for r in observed if r["name"] == name], axis=0)
                   for key in ("old", "new")} for name in arms}
    result = {name: {key: estimates(values) for key, values in pair.items()} for name, pair in cond.items()}
    result["matched_minus_ordinary"] = {key: estimates(cond["age-matched"][key] - cond["ordinary"][key])
                                        for key in ("old", "new")}
    out = {"world_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(), "protocol": core,
           "rows": observed, "summary": result}
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
