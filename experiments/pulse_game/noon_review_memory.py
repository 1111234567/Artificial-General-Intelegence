"""Test whether past taste changes only the emitted review decision on a held-out world."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_review_train as trained
from . import noon_review_world as world


SEEDS = {"original": 76202411, "independent": 76202412}
DONORS = ("left_candidate", "right_candidate", "noncandidate")
COMPONENTS = {"whole": ("graph", "hidden", "trace"), "graph": ("graph",), "activity": ("hidden", "trace")}
READINGS = ("changed_decision", "mean_abs_probability_change", "native_minus_donor_actual_accuracy",
            "native_minus_donor_expected_decision_accuracy", "donor_actual_accuracy")


def replay_taste(view, initial, tape, contacts, target):
    state = initial
    for item, original in enumerate(contacts):
        left, right = world.sensory_for(tape, item, tape.study_orientation[:, item])
        motor = native.act(view, state, left, right, side_override=original.motor.side)
        taste = torch.where(target == item, -original.earned, original.earned)
        experience = native.observe(view, motor, left, right, taste)
        end = native.commit(view, experience, tape.study_draw[:, item, 1:]).state
        state = native.wait(view, end, world.GAP)
    return state


@torch.no_grad()
def evaluate(graph, rule, tape):
    count = tape.count
    coeff = rule.expand(tape.lanes)
    initial = native.start(graph, tape.lanes)
    resident, contacts = world.introduce(coeff, initial, tape)
    left, right = world.candidate_menu(tape)
    draw = tape.decision_draw
    live = world.select(coeff, resident, left, right, draw)
    original_p = live.motor_logit.sigmoid()

    replays = replay_taste(coeff, initial, tape, contacts, torch.full_like(tape.final_item, -1))
    for field in ("hidden", "sender_trace", "receiver_trace"):
        if not torch.equal(getattr(replays, field), getattr(resident, field)):
            raise AssertionError(f"Replay changed {field} without a taste intervention")
    if not all(torch.equal(getattr(replays.graph, field), getattr(resident.graph, field))
               for field in ("weights", "bias", "present")):
        raise AssertionError("Replay changed the graph without a taste intervention")

    options = []
    for side in (-1, 1):
        snapshot, _ = world.review(coeff, live.state, tape, torch.full_like(live.side, side))
        result = [world.final_probe(coeff, snapshot, tape, item) for item in range(count)]
        options.append(torch.stack([action.side == truth for action, truth in result], dim=1))
    option_score = torch.stack(options, dim=1).float().mean(2)
    actual_left, actual_right = option_score.unbind(dim=1)
    resident_score = torch.where(live.side > 0, actual_right, actual_left)
    value_gap = actual_right - actual_left

    ids = torch.arange(count)[None, :].expand(tape.lanes, -1)
    unused = ids[(ids[..., None] != tape.candidates[:, None, :]).all(2)].reshape(tape.lanes, count - 2)
    unused_id = (tape.external_draw * (count - 2)).long()
    noncandidate = unused[torch.arange(tape.lanes), unused_id]
    targets = (tape.candidates[:, 0], tape.candidates[:, 1], noncandidate)
    out = {"native_side": live.side, "native_p_right": original_p, "actual_option_correct": torch.stack(options, dim=1),
           "actual_native_correct": resident_score, "candidate": tape.candidates,
           "negative_or_positive_taste": torch.stack([contact.earned for contact in contacts], dim=1)}
    for name, target in zip(DONORS, targets, strict=True):
        donor = replay_taste(coeff, initial, tape, contacts, target)
        out[name + "/flipped_object"] = target
        for component, fields in COMPONENTS.items():
            transplanted = native.swap(resident, donor, fields)
            action = world.select(coeff, transplanted, left, right, draw)
            donor_p = action.motor_logit.sigmoid()
            donor_score = torch.where(action.side > 0, actual_right, actual_left)
            score = torch.stack(((live.side != action.side).float(), (original_p - donor_p).abs(),
                                 resident_score - donor_score, (original_p - donor_p) * value_gap, donor_score), dim=1)
            out[f"{name}/{component}/side"] = action.side
            out[f"{name}/{component}/score"] = score
    return {key: value.numpy() for key, value in out.items()}


def cohort(path, cohort_name, stages, worlds, evidence):
    torch.set_num_threads(1)
    seed = SEEDS[cohort_name]
    births = world.COHORTS[cohort_name]
    tape = world.make_tape(seed, worlds, 4)
    intervals = prior.BirthIntervals(list(births), 20_000, seed)
    computed, files = {}, {}
    for stage in stages:
        values = {donor: {c: [] for c in COMPONENTS} for donor in DONORS}
        totals = []
        for birth in births:
            g, r = world.load_frozen(cohort_name, "native", birth, stage, path=path)
            raw = evaluate(g, r, tape)
            dest = path / "evaluation" / "memory" / f"{stage:06d}" / cohort_name / f"{birth}-n{worlds}.npz"
            archive._write_raw(dest, **raw)
            files[dest.relative_to(path).as_posix()] = prior.digest(dest)
            source = trained.checkpoint_path(path, cohort_name, "native", birth, stage)
            files[source.relative_to(path).as_posix()] = prior.digest(source)
            totals.append(raw["actual_native_correct"].mean())
            for donor, by_component in values.items():
                for component, rows in by_component.items():
                    rows.append(raw[f"{donor}/{component}/score"].mean(0))
        computed[str(stage)] = dict(actual_baseline=intervals.report(totals), donors={
            donor: {c: {name: intervals.report(column) for name, column in
                        zip(READINGS, np.array(rows).T, strict=True)} for c, rows in values_for_donor.items()}
            for donor, values_for_donor in values.items()})
        print(cohort_name, stage, {d: computed[str(stage)]["donors"][d]["whole"]
                                  ["native_minus_donor_actual_accuracy"]["mean"] for d in DONORS}, flush=True)
    report = dict(cohort=cohort_name, births=births, seed=seed, worlds=worlds, results=computed, sha256=files,
                  code_sha256=prior.digest(Path(__file__)))
    name = f"noon-review-memory-{cohort_name}-n{worlds}-{'-'.join(map(str, stages))}.json"
    archive._write_json(evidence / name, report)


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
