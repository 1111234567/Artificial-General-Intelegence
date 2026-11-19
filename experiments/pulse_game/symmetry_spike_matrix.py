"""A locally updated meta-synapse matrix with no stored episodes or rule labels."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .symmetry_spike import Skill, SpikingSkill
from .symmetry_spike_run import brief, make_pair, write_row


@dataclass(frozen=True)
class MatrixSkill(Skill):
    diagonal: bool = False


class MatrixLearner(SpikingSkill):
    def reset(self, n):
        super().reset(n)
        if self.dims > 96:
            raise ValueError("Matrix meta-synapses use the bounded unexpanded address")
        self.operator = np.zeros((n, self.dims, self.dims), dtype=np.float32)
        self.transmit = np.zeros((n, self.dims), dtype=np.float32)

    def value(self, field):
        memory = self.weights[np.arange(len(field)), field]
        self.transmit = memory.copy()
        if self.fast_weights is not None:
            self.transmit -= .88 * self.fast_weights[np.arange(len(field)), field]
        if self.skill.fixed_gain is None:
            guess = np.einsum("nij,nj->ni", self.operator, memory, optimize=False)
            witness = np.einsum("nij,nj->ni", self.operator, self.transmit, optimize=False)
        else:
            guess = self.skill.fixed_gain * memory
            witness = self.skill.fixed_gain * self.transmit
        self.meta_recall = np.einsum("nbi,ni->nb", self.key, witness, optimize=False)
        self.recall = np.einsum("nbi,ni->nb", self.key, guess, optimize=False)
        return self.recall

    def adapt_direction(self, rows, side, score, earlier):
        if self.skill.fixed_gain is not None:
            return
        cue = self.key[rows, side]
        past = self.transmit[rows]
        error = score - np.tanh(earlier) if self.skill.update_gain == "delta" else score
        factor = self.skill.gain_rate * error / (.35 + np.square(past).sum(axis=-1))
        evidence = factor[:, None, None] * cue[:, :, None] * past[:, None, :]
        if self.skill.diagonal:
            evidence *= np.eye(self.dims, dtype=np.float32)
        self.operator[rows] = self.skill.gain_hold * self.operator[rows] + evidence
        self.direction[rows] = np.trace(self.operator[rows], axis1=1, axis2=2) / self.dims

    def reverse_direction(self):
        super().reverse_direction()
        self.operator *= -1.


def alternate(tests, mode):
    """Evaluator changes odor geometry, leaving the native truth/sign streams intact."""
    if mode == "opposite":
        return tests
    if mode not in ("near-identical", "quarter-turn"):
        raise ValueError(f"Unknown transform {mode}")
    maps = []
    for seeds in tests[0].menu.body.seeds:
        if mode == "near-identical":
            maps.append(np.eye(4))
        else:
            rng = np.random.default_rng(np.random.SeedSequence((seeds, 4056192)))
            q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
            rotation = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., -1.], [0., 0., 1., 0.]])
            maps.append(q @ rotation @ q.T)
    fixed = np.asarray(maps, dtype=np.float32)
    output = []
    for p in tests:
        chem = p.menu.chemicals.copy()
        altered = np.zeros_like(p.menu.realized, dtype=bool)
        changed_a = (p.pairmates >= 0) & (np.arange(p.menu.body.trials)[None] < p.exposed)
        altered[changed_a] = True
        trials = np.nonzero(p.query)
        altered[*trials, p.related_side[trials]] = True
        x = (chem * 2. - 1.)
        x = -np.einsum("ntsi,nji->ntsj", x, fixed, optimize=False)
        chem[altered] = np.clip((x[altered] + 1.) * .5, 0., 1.)
        output.append(replace(p, menu=replace(p.menu, chemicals=chem)))
    return tuple(output)


def developmental(tests, births):
    result = []
    for shift in ("opposite", "near-identical", "quarter-turn"):
        geometry = alternate(tests, shift)
        for seed in births:
            for name, arm, spec in (
                ("scalar_linear", SpikingSkill, Skill(encoding="linear", memory_hold=.90, meta_fast_hold=.48, update_gain="correlation")),
                ("scalar_voltage", SpikingSkill, Skill(encoding="voltage", memory_hold=.90, meta_fast_hold=.48, update_gain="correlation")),
                ("matrix_linear", MatrixLearner, MatrixSkill(encoding="linear", memory_hold=.90, meta_fast_hold=.48,
                                                            gain_rate=.82, update_gain="correlation")),
                ("matrix_linear_delta", MatrixLearner, MatrixSkill(encoding="linear", memory_hold=.90, meta_fast_hold=.48,
                                                                  gain_rate=.82, update_gain="delta")),
                ("matrix_voltage", MatrixLearner, MatrixSkill(encoding="voltage", memory_hold=.90, meta_fast_hold=.48,
                                                             gain_rate=.82, update_gain="correlation")),
                ("matrix_voltage_sparse", MatrixLearner, MatrixSkill(encoding="centered", memory_hold=.90, meta_fast_hold=.48,
                                                                    gain_rate=.82, update_gain="correlation")),
                ("diagonal_linear", MatrixLearner, MatrixSkill(encoding="linear", memory_hold=.90, meta_fast_hold=.48,
                                                              gain_rate=.82, update_gain="correlation", diagonal=True)),
                ("diagonal_voltage", MatrixLearner, MatrixSkill(encoding="voltage", memory_hold=.90, meta_fast_hold=.48,
                                                               gain_rate=.82, update_gain="correlation", diagonal=True)),
                ("matrix_opponent", MatrixLearner, MatrixSkill(encoding="opponent", memory_hold=.90, meta_fast_hold=.48,
                                                              gain_rate=.82, update_gain="correlation")),
            ):
                row = write_row(f"{shift}-{name}", geometry, spec, seed, actor=arm)
                row["geometry"] = shift
                result.append(row)
    return result


def verify(tests, scrambled, freeze):
    result = []
    for shift in ("opposite", "near-identical", "quarter-turn"):
        host = alternate(tests, shift)
        false = alternate(scrambled, shift)
        for seed in freeze["births"]:
            for name, spec, arm, scene, mode in (
                ("matrix_native", MatrixSkill(**freeze["skill"]), MatrixLearner, host, "actual"),
                ("matrix_diagonal", replace(MatrixSkill(**freeze["skill"]), diagonal=True), MatrixLearner, host, "actual"),
                ("matrix_frozen", replace(MatrixSkill(**freeze["skill"]), fixed_gain=0.), MatrixLearner, host, "actual"),
                ("matrix_scrambled", MatrixSkill(**freeze["skill"]), MatrixLearner, false, "actual"),
                ("matrix_reversed", MatrixSkill(**freeze["skill"]), MatrixLearner, host, "invert-gain"),
                ("scalar_same_encoding", Skill(**{k: v for k, v in freeze["skill"].items() if k != "diagonal"}), SpikingSkill, host, "actual"),
            ):
                row = write_row(f"{shift}-{name}", scene, spec, seed, mode=mode, actor=arm)
                row["geometry"] = shift
                result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discover", "confirm"))
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--births", type=int, nargs="+")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(args.start, args.start + args.n))
    tests = make_pair(seeds, probes=1)
    if args.stage == "discover":
        rows = developmental(tests, args.births)
    else:
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        rows = verify(tests, make_pair(seeds, probes=1, permuted=True), freeze)
    retained = {"stage": args.stage, "world_seeds": seeds, "rows": rows, "summary": brief(rows)}
    if args.stage == "confirm":
        from .spike_analysis import estimates
        contrasts = {}
        for mode in ("opposite", "near-identical", "quarter-turn"):
            truth = [row for row in rows if row["name"] == f"{mode}-matrix_native"]
            current = {}
            for arm in ("matrix_diagonal", "matrix_frozen", "matrix_scrambled", "matrix_reversed", "scalar_same_encoding"):
                foil = [row for row in rows if row["name"] == f"{mode}-{arm}"]
                current[arm] = {key: estimates(np.mean([row[key] for row in truth], axis=0)
                                               - np.mean([row[key] for row in foil], axis=0))
                                for key in ("all_correct", "both_correct", "switched_side")}
            contrasts[mode] = current
        retained["matrix_matches"] = contrasts
    args.artifact.write_text(json.dumps(retained,
                                        separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
