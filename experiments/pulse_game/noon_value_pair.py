"""Isolate A/B angular interference in the sealed two-object review replay."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import noon_value as base, noon_value_report as report
from .interleave_interference import ROOT


ANGLES = (15, 45, 75)
WORLDS = 128
PLAN = base.EVIDENCE / "value-pair-plan.md"


def pair(tape, reverse):
    slots = torch.tensor([1, 0] if reverse else [0, 1])
    support = tape.support[:, slots]
    preview = tape.preview_order.numpy()
    preview = np.take_along_axis(preview, np.argsort(preview >= 2, axis=1, kind="stable"), axis=1)[:, :2]
    scramble = tape.scramble.numpy()
    scramble = np.take_along_axis(scramble, np.argsort(scramble >= 2, axis=1, kind="stable"), axis=1)[:, :2]
    return replace(tape, axis=tape.axis[:, :2], cue=tape.cue[:, :2], law=tape.law[:, :, :2],
                   support=support, support_order=tape.support_order[:, slots], support_sign=tape.support_sign[:, slots],
                   support_draw=tape.support_draw[:, slots], support_wait=tape.support_wait[:, slots],
                   preview_order=torch.from_numpy(preview), preview_sign=tape.preview_sign[:, :2],
                   priority=tape.priority[:, :2], scramble=torch.from_numpy(scramble),
                   terminal_sign=tape.terminal_sign[:, :2], terminal_draw=tape.terminal_draw[:, :2])


@torch.no_grad()
def evaluate(graph, rule, tape):
    worlds, maps, count = tape.law.shape
    public = torch.arange(worlds).repeat_interleave(maps)
    law = tape.law.reshape(-1, count)
    coeff = rule.expand(len(public))
    initial = base.begin(graph, coeff, tape, public, law, joint=False)
    snapshot, logits, _ = base.preview(coeff, initial, tape, public)["all"]
    raw = base.evaluate_panel(rule, coeff, snapshot, tape, public, law, joint=False,
                              samples_per_fold=base.SAMPLES_PER_FOLD)
    return {key: value.reshape(worlds, maps, *value.shape[1:]).numpy() for key, value in {**raw, "logit": logits}.items()}


def measures(fold, actual, priority, score, law):
    treatment = (fold[:, :, 0] - fold[:, :, 1]).mean(2)
    own, other, _ = report.other_effect(treatment[..., 2:])
    values = {"own_probability_change": own.mean(-1), "other_probability_change": other.mean(-1),
              "joint_probability_change": treatment[..., 0].mean(-1),
              "own_probability_change_review_a": own[..., 0], "own_probability_change_review_b": own[..., 1],
              "other_probability_change_review_a": other[..., 0], "other_probability_change_review_b": other[..., 1],
              "joint_probability_change_review_a": treatment[..., 0, 0], "joint_probability_change_review_b": treatment[..., 1, 0]}
    priority = report.broadcast(priority, score)
    allow = np.ones_like(score, dtype=bool)
    for label, choice_score in (("uniform", priority), ("weak", -np.abs(score)), ("strong", np.abs(score))):
        choice = report.pick(choice_score, allow, priority)
        values[f"{label}_actual_joint"] = report.take_scalar(actual[:, :, 0, :, 0], choice)
    values["weak_minus_uniform_actual_joint"] = values["weak_actual_joint"] - values["uniform_actual_joint"]
    values["weak_minus_strong_actual_joint"] = values["weak_actual_joint"] - values["strong_actual_joint"]
    agree = law[..., 0] == law[..., 1]
    return {f"{kind}/{name}": np.where(condition, value, 0).sum(-1) / condition.sum(-1)
            for kind, condition in (("same_law", agree), ("opposite_laws", ~agree))
            for name, value in values.items()}


def execute(cohort, body, data, plan_signature):
    first, seed = base.COHORTS[cohort]
    seed += 9127
    births = list(range(first, first + 6))
    source, arm, step = base.source(cohort, body)
    intervals = report.PairedCrossed(len(births), WORLDS, seed + 3191)
    tables = {}
    files = []
    for angle in ANGLES:
        tape = base.make_tape(f"angle{angle}", seed, WORLDS)
        orders = [pair(tape, reverse) for reverse in (False, True)]
        results = []
        for birth in births:
            path = data / f"{angle}-{birth}.npz"
            checkpoint = base.fitting.checkpoint_path(source, "odor", arm, birth, step)
            identity = base.digest(checkpoint)
            if not path.exists():
                graph, rule = base.fitting.load_checkpoint(source, "odor", arm, birth, step)
                ways = []
                for order in orders:
                    chunks = [evaluate(graph, rule, order[i:i + 32]) for i in range(0, WORLDS, 32)]
                    ways.append({name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]})
                raw = {f"{order}/{key}": value for order, bank in zip(("ab", "ba"), ways, strict=True)
                       for key, value in bank.items()}
                raw.update(law=orders[0].law.numpy(), priority=orders[0].priority.numpy(),
                           signature=plan_signature, checkpoint_sha256=identity)
                base.io._write_raw(path, **raw)
                print(cohort, body, angle, birth, flush=True)
            with np.load(path, allow_pickle=False) as raw:
                if str(raw["signature"]) != plan_signature or str(raw["checkpoint_sha256"]) != identity:
                    raise ValueError(f"Wrong frozen replay {path}")
                directions = {label: measures(raw[f"{label}/fold"], raw[f"{label}/actual"], raw["priority"],
                                              raw[f"{label}/logit"], raw["law"]) for label in ("ab", "ba")}
                mean = {key: (directions["ab"][key] + directions["ba"][key]) / 2 for key in directions["ab"]}
                results.append({f"{order}/{key}": value for order, block in {**directions, "averaged": mean}.items()
                                for key, value in block.items()})
            files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=base.digest(path),
                              checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=identity))
        tables[angle] = {key: np.stack([result[key] for result in results]) for key in results[0]}
    diffs = {key: tables[15][key] - tables[75][key] for key in tables[15]}
    interaction = {key.replace("opposite_laws/", ""): value - diffs[key.replace("opposite_laws/", "same_law/")]
                   for key, value in diffs.items() if "opposite_laws/" in key}
    outcomes = {angle: intervals.summary(rows) for angle, rows in tables.items()}
    outcomes.update(angle15_minus_75=intervals.summary(diffs), angle_by_conflict=intervals.summary(interaction))
    result = dict(cohort=cohort, body=body, births=births, worlds=WORLDS, maps_per_geometry=4,
                  study_orders=["ab", "ba"], seed=seed, plan_sha256=base.digest(PLAN), signature=plan_signature,
                  review_samples_per_fold=base.SAMPLES_PER_FOLD, bootstrap_draws=4000, outcomes=outcomes, files=files)
    base.io._write_json(base.EVIDENCE / f"value-pair-{cohort}-{body}.json", result)


def run(cohort, bodies):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = (Path(__file__), Path(base.__file__), Path(report.__file__), PLAN)
    signature = "\n".join(f"{path.name} {base.digest(path)}" for path in inputs)
    for body in bodies:
        target = base.DATA / "sealed_pair" / cohort / body
        target.mkdir(parents=True, exist_ok=True)
        execute(cohort, body, target, signature)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=base.COHORTS, required=True)
    parser.add_argument("--body", choices=("uniform", "mixed_repair"), nargs="+", default=("uniform", "mixed_repair"))
    cli = parser.parse_args()
    run(cli.cohort, cli.body)
