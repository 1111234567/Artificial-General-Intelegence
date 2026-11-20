"""Test whether a review menu inherits the graph's ordinary signed collection response."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_review_world as world


SEEDS = {"original": 76202421, "independent": 76202422}


@torch.no_grad()
def evaluate(graph, rule, tape):
    state = native.start(graph, tape.lanes)
    view = rule.expand(tape.lanes)
    snapshot, _ = world.introduce(view, state, tape)
    left, right = world.candidate_menu(tape)
    natural = world.select(view, snapshot, left, right, tape.decision_draw)
    menus = {"unchanged": natural, **{key: world.select(view, snapshot, a, b, tape.decision_draw)
                                    for key, a, b in (("reverse_left", -left, right), ("reverse_right", left, -right),
                                                      ("reverse_both", -left, -right))}}

    options = []
    for side in (-1, 1):
        reviewed, _ = world.review(view, natural.state, tape, torch.full_like(natural.side, side))
        final = [world.final_probe(view, reviewed, tape, item) for item in range(tape.count)]
        options.append(torch.stack([action.side == truth for action, truth in final], dim=1))

    sign_proxy = -tape.law.gather(1, tape.candidates) * tape.menu_orientation
    return dict(nominal_sign=sign_proxy.numpy(), sides=np.stack([action.side.numpy() for action in menus.values()]),
                logit=np.stack([action.motor_logit.numpy() for action in menus.values()]),
                menu_names=tuple(menus), actual_options=torch.stack(options, dim=1).numpy())


def readings(raw):
    signed = raw["nominal_sign"]
    sides = raw["sides"]
    choices = (sides > 0).astype(int)
    rows = np.arange(len(signed))
    opposed = signed[:, 0] != signed[:, 1]
    chosen_taste = signed[rows, choices[0]]
    scores = raw["actual_options"].mean(2)
    result = {"nominal_positive_given_opposed": (chosen_taste[opposed] > 0).mean(),
              "world_fraction_nominally_opposed": opposed.mean()}
    for index, name in enumerate(raw["menu_names"][1:], 1):
        swaps = sides[0] != sides[index]
        result[f"{name}_switch_rate"] = swaps.mean()
        result[f"{name}_switch_rate_nominally_opposed"] = swaps[opposed].mean()
        result[f"{name}_loss_when_side_donated"] = (scores[rows, choices[0]] - scores[rows, choices[index]]).mean()
    return result


def cohort(path, cohort_name, stages, worlds, evidence):
    torch.set_num_threads(1)
    seed = SEEDS[cohort_name]
    births = world.COHORTS[cohort_name]
    tape = world.make_tape(seed, worlds, 4)
    intervals = prior.BirthIntervals(list(births), 20_000, seed)
    results, hashes = {}, {}
    for step in stages:
        data = []
        for birth in births:
            graph, rule = world.load_frozen(cohort_name, "native", birth, step, path=path)
            raw = evaluate(graph, rule, tape)
            data.append(readings(raw))
            dest = path / "evaluation" / "orientation" / f"{step:06d}" / cohort_name / f"{birth}-n{worlds}.npz"
            archive._write_raw(dest, **raw)
            hashes[dest.relative_to(path).as_posix()] = prior.digest(dest)
        results[str(step)] = {key: intervals.report([item[key] for item in data]) for key in data[0]}
        print(cohort_name, step, {k: results[str(step)][k]["mean"] for k in (
            "nominal_positive_given_opposed", "reverse_both_switch_rate_nominally_opposed",
            "reverse_both_loss_when_side_donated")}, flush=True)
    output = dict(cohort=cohort_name, births=births, worlds=worlds, seed=seed, results=results, raw_sha256=hashes,
                  source_sha256=prior.digest(Path(__file__)),
                  meaning="The two odors retain physical identity in all menus. x,-x are the known display orientations "
                          "of one object. Ordinary nominal value of a shown vector is -law times presentation sign. No "
                          "nominal or actual value is provided during this menu. In orientation branches only the motor "
                          "choice is brought to the original live post-menu state, using its true objects and independent "
                          "revisit and probe orientations, to measure any effect through active choice.")
    archive._write_json(evidence / f"noon-review-orientation-{cohort_name}-n{worlds}-{'-'.join(map(str, stages))}.json", output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=world.DATA / "core")
    parser.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    parser.add_argument("--stages", nargs="+", choices=world.CHECKPOINTS, type=int, required=True)
    parser.add_argument("--worlds", type=int, default=384)
    parser.add_argument("--cohorts", nargs="+", choices=world.COHORTS, default=list(world.COHORTS))
    a = parser.parse_args()
    with ProcessPoolExecutor(max_workers=len(a.cohorts)) as pool:
        jobs = [pool.submit(cohort, a.path, name, a.stages, a.worlds, a.evidence) for name in a.cohorts]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
