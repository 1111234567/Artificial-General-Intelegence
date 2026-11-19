"""Evaluate within-life receptor competition on newly mixed smells."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .ring_alt_chemical import ChemSpec, ChemicalForager
from .ring_alt_policy import ForagingSpec, ReceptorForager
from .ring_alt_run import ContactMatched, contact_yoke, evidence, preference_code, summarize_result, world_panel
from .ring_sensors import ChemicalInterface, codebooks
from .ring_world import CUE, FRUIT, TASTED, rollout


def actor(tape, spec, birth):
    seeds = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
    return ChemicalForager(spec, birth_seed=birth, action_seeds=seeds)


def population_audit(population, odors):
    """Read trained sensory geometry offline, without giving identity labels to the agent."""
    response = np.stack([population.match(odors[:, i]) for i in range(odors.shape[1])], axis=1)
    live = response.sum(axis=2) > 0.
    strongest = response.argmax(axis=2)
    distinct = [len(np.unique(row[seen])) for row, seen in zip(strongest, live, strict=True)]
    size = np.square(response).sum(axis=2)
    participation = np.divide(1., size, out=np.zeros_like(size), where=size > 1e-6)
    normalized = response / np.maximum(np.linalg.norm(response, axis=2, keepdims=True), 1e-8)
    similarity = normalized @ normalized.transpose(0, 2, 1)
    a, b = np.triu_indices(odors.shape[1], k=1)
    return {"different_winners": distinct, "mean_active_support": participation.mean(axis=1).tolist(),
            "between_odor_cosine": similarity[:, a, b].mean(axis=1).tolist()}


class DominantReceptor:
    """A deliberately label-fixed comparator for mixtures of several channels."""

    def __init__(self, child):
        self.child = child

    def reset(self, n):
        self.child.reset(n)

    def act(self, obs):
        frame = obs.copy()
        families = (CUE, FRUIT, TASTED)
        for family in families:
            raw = obs[:, family]
            frame[:, family] = np.eye(4, dtype=obs.dtype)[raw.argmax(axis=1)] * raw.any(axis=1, keepdims=True)
        return self.child.act(frame)


def evaluate(name, tape, spec, birth, style, *, feedback="true", yoke=None):
    if name.startswith("original-"):
        action = [np.random.SeedSequence((s, 0x4348454D)) for s in tape.seeds]
        model = ReceptorForager(ForagingSpec(outcomes="negative", scoring="linear", cov_rate=.16),
                               birth_seed=birth, action_seeds=action)
        child = DominantReceptor(model) if name == "original-dominant" else model
    else:
        child = model = actor(tape, spec, birth)
    if yoke is not None:
        child = ContactMatched(child, yoke)
    interface = ChemicalInterface(child, tape.seeds, style)
    result = rollout(interface, tape, feedback=feedback)
    score = summarize_result(tape, result)
    info = {"name": name, "birth_seed": birth, "spec": asdict(spec), "style": style,
            "scores": score, "network": model.stats(), "first_correct": result.first_correct.tolist(),
            "contact": (result.bites > 0).tolist(), "feedback": feedback if yoke is None else "bite-yoked"}
    if yoke is not None:
        info["yoke_nonzero_bites"] = child.aligned.tolist()
        info["yoke_exposed_bites"] = child.exposed.tolist()
    if isinstance(model, ChemicalForager):
        info["organizing_fruit"] = population_audit(model.fruit, interface.fruits)
        info["organizing_cue"] = population_audit(model.cue, interface.cues)
    print(f"{birth:2d} {style:11s} {name:25s} {np.mean(score['early_first']):.3f} → {np.mean(score['late_first']):.3f}; "
          f"contact {np.mean(score['late_contact']):.3f}", flush=True)
    return info, result


def discovery(tape, seeds, style):
    base = ChemSpec()
    settings = (
        ("raw-only", {"fruit_bank": "raw"}),
        ("fixed-both", {"fruit_bank": "fixed", "cue_bank": "fixed"}),
        ("fixed-fruit", {"fruit_bank": "fixed"}),
        ("competition-fruit", {"fruit_bank": "competitive"}),
        ("competition-both", {"fruit_bank": "competitive", "cue_bank": "competitive"}),
        ("grow-fruit", {}),
        ("grow-both", {"cue_bank": "growing"}),
        ("grow-both-wide", {"cue_bank": "growing", "width": .33}),
        ("grow-both-narrow", {"cue_bank": "growing", "width": .14}),
        ("grow-both-hard", {"cue_bank": "growing", "width": 0.}),
        ("grow-fruit-signed", {"outcomes": "signed"}),
        ("grow-both-signed", {"cue_bank": "growing", "outcomes": "signed"}),
        ("grow-both-lower-threshold", {"cue_bank": "growing", "novelty": .94}),
        ("reassign", {"fruit_bank": "reassign-fixed", "cue_bank": "growing"}),
        ("original-direct", {}),
        ("original-dominant", {}),
    )
    return [evaluate(name, tape, replace(base, **options), seed, style)[0]
            for seed in seeds for name, options in settings]


def confirmation(tape, seeds, style, spec):
    output = []
    for seed in seeds:
        row, donor = evaluate("negative-live", tape, spec, seed, style)
        output.append(row)
        output.append(evaluate("negative-yoked", tape, spec, seed, style, feedback="none", yoke=contact_yoke(donor))[0])
        output.append(evaluate("negative-frozen", tape, spec, seed, style, feedback="none")[0])
        output.append(evaluate("negative-inverted", tape, spec, seed, style, feedback="invert")[0])
        for name, variant in (
            ("random-receptors", replace(spec, fruit_bank="fixed", cue_bank="fixed")),
            ("raw-continuous", replace(spec, fruit_bank="raw", cue_bank="raw")),
            ("raw-cue", replace(spec, cue_bank="raw")),
            ("raw-fruit", replace(spec, fruit_bank="raw")),
            ("competition", replace(spec, fruit_bank="competitive", cue_bank="competitive")),
            ("same-rate-growth", replace(spec, fruit_bank="growth-fixed-writes")),
            ("fixed-population-reassignment", replace(spec, fruit_bank="reassign-fixed")),
            ("winner-take-all", replace(spec, width=0., cue_width=0., fruit_cov_rate=0.)),
            ("no-fruit-decorrelation", replace(spec, fruit_cov_rate=0.)),
            ("original-direct", spec), ("original-dominant", spec),
        ):
            output.append(evaluate(name, tape, variant, seed, style)[0])
    return output


def population_widths(tape, seeds, style):
    base = ChemSpec(cue_bank="growing", cue_width=.14)
    definitions = (("narrow", {"width": .14}), ("medium", {"width": .22}), ("wide", {"width": .30}),
                   ("medium-orthogonal", {"width": .22, "fruit_cov_rate": .025}),
                   ("wide-orthogonal", {"width": .30, "fruit_cov_rate": .025}),
                   ("wide-orthogonal-strong", {"width": .30, "fruit_cov_rate": .08}),
                   ("wide-cues-orthogonal", {"width": .30, "cue_width": .30, "fruit_cov_rate": .025}),
                   ("grow-winner", {"width": 0., "cue_width": 0.}),
                   ("fixed-competition-medium", {"fruit_bank": "competitive", "cue_bank": "competitive", "width": .22}),
                   ("raw-with-orthogonal", {"fruit_bank": "raw", "cue_bank": "raw", "fruit_cov_rate": .025}))
    return [evaluate(name, tape, replace(base, **options), seed, style)[0]
            for seed in seeds for name, options in definitions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("discover", "widths", "confirm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trials", type=int, default=224)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--style", choices=("permutation", "anchored", "novel", "noisy"), required=True)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    panel = "evaluation" if args.action == "confirm" else "development"
    tape = world_panel(args.start, args.n, args.trials, panel=panel)
    if args.action == "discover":
        rows = discovery(tape, args.births, args.style)
    elif args.action == "widths":
        rows = population_widths(tape, args.births, args.style)
    else:
        spec = ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
        rows = confirmation(tape, args.births, args.style, spec)
    labels = {role: (codebooks(tape.seeds, bank, args.style)).argmax(axis=2).tolist() for role, bank in (("cues", 1), ("fruits", 2))}
    data = {"phase": args.action, "style": args.style, "source_panel": panel, "world_seeds": tape.seeds,
            "privates": preference_code(tape.preferences).tolist(), "dominant_receptor_by_label": labels,
            "trials": tape.trials, "rows": rows, "summary": evidence(rows)}
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
