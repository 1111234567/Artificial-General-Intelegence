"""Held-out action selection in worlds where two pairs of odors must share similar directions."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_review_eval as replay
from . import noon_review_world as world


SEEDS = {"original": 76202471, "independent": 76202472}
BRANCHES = ("ordinary", "native", "random", "uncertain", "native_sham", "flip_action")


def cluster(tape, seed):
    worlds = tape.lanes // 2
    rng = np.random.default_rng([world.EVAL_DOMAIN, seed, 401])
    rotation, triangular = np.linalg.qr(rng.normal(size=(worlds, 3, 3)))
    diagonal = np.diagonal(triangular, axis1=1, axis2=2)
    rotation *= np.where(diagonal >= 0, 1, -1)[:, None, :]
    rotation[:, :, 2] *= np.linalg.det(rotation)[:, None]
    angle = np.deg2rad(15)
    cos, sin = np.cos(angle), np.sin(angle)
    frame = np.array(((1, 0, 0), (cos, 0, sin), (0, 1, 0), (0, cos, -sin)))
    dots = (1.35 * (frame @ rotation)).astype(np.float32)
    permute = rng.permuted(np.broadcast_to(np.arange(4), dots.shape[:2]), axis=1)
    clustered = np.take_along_axis(dots, permute[..., None], axis=1)
    labels = np.repeat(permute // 2, 2, axis=0)
    candidates = tape.candidates.numpy()
    cluster_pair = np.take_along_axis(labels, candidates, axis=1)
    return replace(tape, odor=torch.from_numpy(np.repeat(clustered, 2, axis=0))), cluster_pair[:, 0] == cluster_pair[:, 1]


def rates(raw, select):
    accuracy = {key: (raw[f"{key}/final_side"] == raw["truth"])[select] for key in BRANCHES}
    by_policy = {key: np.asarray((hits.mean(), hits.all(1).mean())) for key, hits in accuracy.items()}
    by_policy.update({"native_minus_" + key: by_policy["native"] - by_policy[key]
                      for key in BRANCHES if key != "native"})
    return by_policy


def cohort(path, name, stage, worlds, evidence):
    torch.set_num_threads(1)
    seed = SEEDS[name]
    births = world.COHORTS[name]
    base = world.make_tape(seed, worlds, 4)
    crowded, nearby = cluster(base, seed)
    intervals = prior.BirthIntervals(list(births), 20_000, seed)
    conditions = (("independent_sphere", base, {"all": slice(None)}),
                  ("two_clusters", crowded, {"all": slice(None), "same_cluster": nearby, "other_cluster": ~nearby}))
    results, raw_sha = {}, {}
    for scenario, tape, strata in conditions:
        samples = {tag: [] for tag in strata}
        for birth in births:
            graph, rule = world.load_frozen(name, "native", birth, stage, path=path)
            old_graph, old_rule = world.load_parent(name, birth)
            view = old_rule.expand(tape.lanes)
            state, _ = world.introduce(view, native.start(old_graph, tape.lanes), tape)
            uncertain, confidence = world.parent_uncertainty(view, state, tape)
            measured = replay.evaluate(graph, rule, tape, uncertain, confidence)
            for tag, criterion in strata.items():
                samples[tag].append(rates(measured, criterion))
            file = path / "evaluation" / "geometry-shift" / f"{stage:06d}" / name / f"{scenario}-{birth}-n{worlds}.npz"
            archive._write_raw(file, geometry=tape.odor.numpy(), candidate_same_cluster=nearby, **measured)
            raw_sha[file.relative_to(path).as_posix()] = prior.digest(file)
        results[scenario] = {tag: {key: {metric: intervals.report(values) for metric, values in
                                        zip(("accuracy", "all_correct"), np.array([item[key] for item in stats]).T,
                                            strict=True)} for key in stats[0]} for tag, stats in samples.items()}
        print(name, stage, scenario, results[scenario]["all"]["native_minus_random"]["accuracy"]["mean"], flush=True)
    result = dict(stage=stage, births=births, cohort=name, seed=seed, worlds=worlds, results=results,
                  raw_sha256=raw_sha, source_sha256=prior.digest(Path(__file__)))
    archive._write_json(evidence / f"noon-review-geometry-shift-{stage:06d}-{name}-n{worlds}.json", result)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=world.DATA / "core")
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--stage", type=int, choices=world.CHECKPOINTS, required=True)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--cohorts", nargs="+", choices=world.COHORTS, default=list(world.COHORTS))
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=len(a.cohorts)) as pool:
        jobs = [pool.submit(cohort, a.path, name, a.stage, a.worlds, a.evidence) for name in a.cohorts]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
