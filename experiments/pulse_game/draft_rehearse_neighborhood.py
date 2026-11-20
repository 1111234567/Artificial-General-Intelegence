"""Diagnose the finite edge-revision neighborhood of the frozen native draft."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import draft_rehearse as draft
from . import edge_site_eval_stats as stats
from . import edge_site_replay as replay
from . import interleave_interference as environment
from . import interleave_revisit_train as storage
from . import noon_ticket as parents


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/draft-rehearsal-20261120"
OUTPUT = ROOT / "data/draft-rehearsal-20261120/neighborhood"
COHORTS = ("original", "independent")
SEED = 80721127
COUNT = 96
ANGLES = (15, 30)
MODES = ("full", "b_only", "random", "native")
WEIGHTS = (0, 0.5, 1, 2)
PREDICTED_B_ALLOWANCE = 0.01


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def conflict_world(count, angle):
    world = environment.worlds(SEED, 0, count, angle)
    conflict = world.law_a != world.law_b
    return replace(world, **{field.name: getattr(world, field.name)[conflict] for field in fields(world)})


def receipts(prefix):
    return tuple((recall.right, recall.side * recall.taste)
                 for recall in (prefix.study_a.archived, prefix.study_b.archived))


def expected(prefix, choice, archive):
    return replay.expected_scores(draft.apply(prefix, choice), archive)


def select(prefix, reference):
    archive = receipts(prefix)
    base = expected(prefix, reference, archive)
    sweeps = [reference.clone() for _ in WEIGHTS]
    lanes = torch.arange(len(reference))
    floor = 1e-12
    queried = 1
    for group in range(reference.shape[1]):
        if not prefix.candidates.valid[:, group].any():
            continue
        changed = reference.clone()
        scores = []
        for address in range(draft.site.GROUP_SIZE):
            changed[:, group] = address
            scores.append(expected(prefix, changed, archive))
        queried += len(scores)
        predictions = torch.stack(scores, 1).clamp_min(floor).log()
        log_a, log_b = predictions.unbind(-1)
        for weight, option in zip(WEIGHTS, sweeps, strict=True):
            option[:, group] = (log_b + weight * log_a).argmax(1)
    expected_complete = torch.stack([base, *(expected(prefix, choice, archive) for choice in sweeps)], dim=1)
    queried += len(sweeps)
    options = torch.stack([reference, *sweeps], dim=1)
    a, b = expected_complete.unbind(-1)
    predicted_joint = (a * b).masked_fill(b < b[:, :1] - PREDICTED_B_ALLOWANCE, -torch.inf)
    joint_pick = predicted_joint.argmax(1)
    b_pick = b[:, :2].argmax(1)
    return {"full": options[lanes, joint_pick], "b_only": options[lanes, b_pick], "random": reference}, queried


def final_world(world, count, angle, first):
    tape = np.random.default_rng([SEED, 146289, angle, first])
    orders, draws = world.orders.clone(), world.draws.clone()
    orders[:, 3:5] = torch.from_numpy(np.repeat(2 * tape.integers(0, 2, (count, 2)) - 1, 2, axis=0))
    draws[:, 3:5, 0] = torch.from_numpy(np.repeat(tape.random((count, 2), dtype=np.float32), 2, axis=0))
    return replace(world, orders=orders, draws=draws)


@torch.no_grad()
def measure(birth, angle, count):
    world = conflict_world(count, angle)
    g, rule = draft.load_parent(birth)
    prefix = draft.prepare(g, rule, world, seed=[SEED, 16928, angle], lanes_per_geometry=2)
    shape = (count, prefix.candidates.valid.shape[1])
    address = np.random.default_rng([SEED, 274899, angle])
    reference = torch.from_numpy(np.repeat(address.integers(draft.site.GROUP_SIZE, size=shape), 2, axis=0))
    picks, query_count = select(prefix, reference)
    downstream = [final_world(world, count, angle, first) for first in (0, 1)]
    laws = torch.stack((world.law_a, world.law_b), 1)
    target = torch.stack([laws * tape.orders[:, 3:5] for tape in downstream], dim=1).numpy().astype(np.int8)
    actions, budgets, addresses = [], [], []
    for mode in MODES:
        choice = picks.get(mode)
        live = draft.apply(prefix, choice, mode="native" if mode == "native" else "selected")
        dose, edit = draft.physical_dose(prefix, live, choice, mode)
        budgets.append(dose)
        addresses.append(edit)
        side = [draft.probe(prefix.rule, live, tape, order=first).side for first, tape in enumerate(downstream)]
        actions.append(torch.stack(side, dim=1).numpy().astype(np.int8))
    signed_budget = np.stack(budgets, axis=0)
    np.testing.assert_array_equal(signed_budget[:3, :, 6], np.broadcast_to(signed_budget[0, :, 6], (3, 2 * count)))
    max_error = float(np.abs(signed_budget[:3, :, 7:9] - signed_budget[:3, :, 10:12]).max())
    if max_error > 3e-6:
        raise AssertionError(f"Unequal signed scratch revision doses: {max_error}")
    study = [contact.action.side.to(torch.int8).numpy() for contact in (prefix.study_c, prefix.study_a, prefix.study_b)]
    taste = [contact.experience.taste.to(torch.int8).numpy() for contact in (prefix.study_c, prefix.study_a, prefix.study_b)]
    return dict(actions=np.stack(actions, axis=-2), target=target, signed_budget=signed_budget,
                address=np.stack(addresses, axis=0), study=np.stack(study, axis=-1), taste=np.stack(taste, axis=-1),
                law=laws.numpy().astype(np.int8), queries=query_count, max_dose_error=max_error)


def birth_file(birth, count):
    return OUTPUT / ("measured" if count == COUNT else f"smoke-{count}") / f"birth-{birth}.npz"


def run(cohort, count):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = digest(EVIDENCE / "neighborhood-plan.md")
    sources = [digest(p) for p in (__file__, draft.__file__, draft.site.__file__, draft.net.__file__,
                                   draft.net.native.__file__, environment.__file__, replay.__file__)]
    for birth in parents.births(cohort):
        output = birth_file(birth, count)
        body = draft.site.storage.checkpoint_path(parents.source(cohort), "odor", "uniform", birth, 8000)
        parent_sha = digest(body)
        if output.exists():
            with np.load(output) as file:
                if (str(file["plan"]) == plan and file["sources"].tolist() == sources
                        and int(file["seed"]) == SEED and str(file["body"]) == parent_sha):
                    continue
        archive = {}
        for angle in ANGLES:
            row = measure(birth, angle, count)
            archive.update((f"{angle}/{key}", value) for key, value in row.items())
        storage._write_raw(output, **archive, modes=MODES, angles=ANGLES, count=count, birth=birth,
                           body=parent_sha, seed=SEED, plan=plan, sources=sources)
        print(cohort, birth, count, flush=True)


def publish():
    plan = digest(EVIDENCE / "neighborhood-plan.md")
    result = {"geometry_seed": SEED, "roots_per_birth": COUNT, "angles_degrees": ANGLES, "plan_sha256": plan,
              "scratch_query": "One complete candidate uses four projected motor reads: each tasted menu in two orientations.",
              "reference_key": "full is the joint scratch-search strategy", "cohorts": {}}
    for cohort in COHORTS:
        scored, queries, files, dose_errors = [], [], {}, []
        for birth in parents.births(cohort):
            path = birth_file(birth, COUNT)
            files[path.relative_to(ROOT).as_posix()] = digest(path)
            with np.load(path) as data:
                assert int(data["birth"]) == birth and int(data["seed"]) == SEED and str(data["plan"]) == plan
                parts = []
                for angle in ANGLES:
                    side, target = (data[f"{angle}/{key}"] for key in ("actions", "target"))
                    assert (side.shape, target.shape) == ((2 * COUNT, 2, len(MODES), 2), (2 * COUNT, 2, 2))
                    parts.append(stats.score(side == target[:, :, None, :]).reshape(COUNT, 4, len(MODES), 3))
                    queries.append(int(data[f"{angle}/queries"]))
                    dose_errors.append(float(data[f"{angle}/max_dose_error"]))
                scored.append(np.concatenate(parts, axis=1))
        scores = np.stack(scored)
        bootstrap = stats.CrossedBootstrap(len(scored), COUNT)
        measures = stats.summarize(scores, MODES, bootstrap)
        measures["by_angle"] = {str(angle): stats.summarize(scores[:, :, i * 4:(i + 1) * 4], MODES, bootstrap)
                                for i, angle in enumerate(ANGLES)}
        result["cohorts"][cohort] = dict(births=parents.births(cohort), measures=measures, data_sha256=files,
                                          candidate_queries_range=[min(queries), max(queries)],
                                          maximum_signed_dose_error=max(dose_errors))
    storage._write_json(EVIDENCE / "neighborhood-results.json", result)
    for cohort, row in result["cohorts"].items():
        numbers = row["measures"]["contrasts"]
        print(cohort, {name: [round(numbers[name][metric]["mean"] * 100, 2) for metric in stats.METRICS]
                       for name in numbers}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*COHORTS, "publish", "smoke"))
    settings = parser.parse_args()
    if settings.phase == "publish":
        publish()
    elif settings.phase == "smoke":
        run("original", 4)
    else:
        run(settings.phase, COUNT)
