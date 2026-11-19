"""Paired evidence-dose and unseen-rotation checks with frozen learning rules."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import mixture_flash, symmetry_world, world
from .spike_analysis import estimates
from .symmetry_spike import Skill, SpikingSkill
from .symmetry_spike_matrix import MatrixLearner, MatrixSkill, alternate
from .symmetry_spike_run import Receipts


class Dose(Receipts):
    def __init__(self, child, protocol, shown_pairs):
        super().__init__(child, protocol, "actual")
        self.begin_feedback = protocol.exposed - 2 * shown_pairs

    def act(self, observation):
        frame = observation.copy()
        tick = self.trial + int(frame[0, world.TRIAL_START] > 0)
        if tick < self.begin_feedback:
            frame[:, world.REWARD_POS:world.REWARD_NEG + 1] = 0.
        return super().act(frame)


def random_rotations(tapes):
    """Apply a fresh proper orthogonal transform per life, outside the actor."""
    rules = []
    for seed in tapes[0].menu.body.seeds:
        rng = np.random.default_rng(np.random.SeedSequence((seed, 752119)))
        q, upper = np.linalg.qr(rng.normal(size=(4,4)))
        q *= np.sign(np.diag(upper))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rules.append(q)
    q = np.asarray(rules, dtype=np.float32)
    result = []
    for p in tapes:
        chem = p.menu.chemicals.copy()
        filtered = np.zeros(chem.shape[:-1], bool)
        echoes = (p.pairmates >= 0) & (np.arange(p.menu.body.trials) < p.exposed)
        filtered[echoes] = True
        probes = np.nonzero(p.query)
        filtered[*probes, p.related_side[probes]] = True
        original = chem * 2. - 1.
        transformed = -np.einsum("nhsi,nji->nhsj", original, q, optimize=False)
        chem[filtered] = np.clip((transformed[filtered] + 1.) * .5, 0., 1.)
        result.append(replace(p, menu=replace(p.menu, chemicals=chem)))
    return tuple(result)


def sample(tapes, config, actor, birth, n_pair):
    index = tapes[0].exposed + 1
    grade, chosen, earlier = [], [], []
    for tape in tapes:
        seeds = [np.random.SeedSequence((s, 0x5B983C)) for s in tape.menu.body.seeds]
        brain = actor(config, birth=birth, action_seeds=seeds)
        result = mixture_flash.rollout(Dose(brain, tape, n_pair), tape.menu)
        observed = symmetry_world.metrics(tape, result)
        np.testing.assert_array_equal(observed["first"].astype(bool), result.reward[:, index] > 0)
        np.testing.assert_array_equal(brain.receipts, np.full(len(seeds), 2 * n_pair + 1))
        grade.append(observed["first"])
        chosen.append(result.contact_side[:, index])
        earlier.append(result.observed_reward[:, :tape.exposed])
    np.testing.assert_array_equal(*earlier)
    return {"mean_pair": (.5 * (grade[0] + grade[1])).tolist(), "both": (grade[0] * grade[1]).tolist(),
            "contact_both": ((chosen[0] >= 0) & (chosen[1] >= 0)).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix_frozen", type=Path)
    p.add_argument("protocol", type=Path)
    p.add_argument("artifact", type=Path)
    args = p.parse_args()
    model = json.loads(args.matrix_frozen.read_text(encoding="utf-8"))["skill"]
    design = json.loads(args.protocol.read_text(encoding="utf-8"))
    ds = tuple(range(design["seed_start"], design["seed_start"] + design["worlds"]))
    tests = tuple(symmetry_world.make(ds, pairs=design["full_pairs"], probes=1, interleave=4,
                                     family=rule, balanced_stream=True) for rule in (1, -1))
    cfg = MatrixSkill(**model)
    scalar = Skill(**{key: value for key, value in model.items() if key != "diagonal"})
    rows = []
    for label, tape in (("quarter-turn", alternate(tests, "quarter-turn")), ("haar-SO4", random_rotations(tests))):
        arms = [(f"matrix-{dose}", MatrixLearner, cfg, dose) for dose in design["visible_pairs"]]
        arms.append(("scalar-full", SpikingSkill, scalar, design["full_pairs"]))
        for seed in design["births"]:
            for name, component, skill, shown in arms:
                picked = sample(tape, skill, component, seed, shown)
                rows.append({"name": name, "orientation": label, "visible_pairs": shown,
                             "birth": seed, **picked})
                print(f"{seed:3d} {label:12s} {name:14s} score {np.mean(picked['mean_pair']):.3f}", flush=True)
    scores = {}
    for scene in dict.fromkeys(r["orientation"] for r in rows):
        agents = {name: {key: np.mean([r[key] for r in rows if r["orientation"] == scene and r["name"] == name], axis=0)
                         for key in ("mean_pair", "both", "contact_both")}
                  for name in dict.fromkeys(r["name"] for r in rows)}
        full = agents[f"matrix-{design['full_pairs']}"]
        scores[scene] = {name: {**{key: estimates(x) for key, x in values.items()},
                                "minus_full": estimates(values["mean_pair"] - full["mean_pair"])}
                         for name, values in agents.items()}
    args.artifact.write_text(json.dumps({"world_seeds": ds, "protocol": design, "rows": rows, "summary": scores},
                                        separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
