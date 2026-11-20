"""Check geometry isolation and summarize actual-action changes between saved heads."""

import argparse
import json

import numpy as np

from . import edge_site_eval as eval_run
from . import edge_site_eval_stats as stats
from . import edge_site_train_fit as fit


def vector_keys(vectors):
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    return array.view(np.dtype((np.void, array.shape[-1] * array.dtype.itemsize))).reshape(-1)


def geometry_isolation():
    heldout = []
    for angle in eval_run.ANGLES:
        world = eval_run.worlds(eval_run.SEED, 0, eval_run.ROOTS, angle)
        for value in (world.a, world.b):
            vectors = value.numpy()[::4]
            heldout.extend((vector_keys(vectors), vector_keys(-vectors)))
    protected = np.unique(np.concatenate(heldout))
    exact_hits = routes = 0
    for replica_seed in fit.SEEDS:
        for arm in fit.ARMS:
            for update in range(1, fit.STEPS + 1):
                world, _ = fit.training_worlds(replica_seed, update, arm, fit.WORLDS)
                routes += len(world.a)
                tested = vector_keys(np.concatenate((world.a.numpy(), world.b.numpy())))
                index = np.minimum(protected.searchsorted(tested), len(protected) - 1)
                exact_hits += int((protected[index] == tested).sum())
    return dict(roots=eval_run.ROOTS, angles=eval_run.ANGLES, training_angles=fit.ANGLES,
                training_births=fit.BIRTHS, heldout_births=eval_run.BIRTHS,
                checked_training_route_geometries=routes, checked_training_endpoint_axes=routes * 2,
                heldout_endpoint_vectors_including_antipodes=len(protected), exact_heldout_axis_hits=exact_hits,
                boundary="Exact float32 endpoint comparison in the four new meta-fits, including antipodes. "
                         "Does not measure near neighbors or re-enumerate inherited body training.")


def primary_rows(label, step):
    folder = eval_run.DATA / f"evaluation-{label}-{step:04d}"
    report = eval_run.EVIDENCE / f"evaluation-{label}-{step:04d}.json"
    source = json.loads(report.read_text(encoding="utf8"))
    angles = (15, 30)
    reads = tuple(tuple(read) for read in source["read_keys"])
    rows = {}
    for angle in angles:
        rows[angle] = []
        for birth in source["births"]:
            file = folder / f"{birth}-{angle}.npz"
            if eval_run.digest(file) != source["raw_sha256"][file.relative_to(eval_run.ROOT).as_posix()]:
                raise ValueError(f"Trajectory checksum mismatch: {file}")
            with np.load(file, allow_pickle=False) as raw:
                rows[angle].append({name: raw[name] for name in
                                    ("correct", "law_a", "law_b", "a", "b", "orders", "draws")})
    scores = eval_run.regime(rows, angles, eval_run.PRIMARY_READ, reads, False)
    return scores, rows, source, eval_run.digest(report)


def trends(label):
    steps = fit.CHECKPOINTS
    initial, initial_rows, spec, initial_hash = primary_rows(label, steps[0])
    paired = stats.CrossedBootstrap(*initial.shape[:2])
    initial = initial.mean(axis=2, dtype=np.float32)
    compared = ("full", "no_old", "native", "random")
    indices = tuple(eval_run.MODES.index(mode) for mode in compared)
    results = {}
    for step in steps[1:]:
        active, rows, source, report_hash = primary_rows(label, step)
        for key in ("births", "geometry_roots", "modes", "training_replica_seeds"):
            if spec[key] != source[key]:
                raise ValueError(f"Checkpoint comparison changed {key}")
        for angle in rows:
            for fresh, start in zip(rows[angle], initial_rows[angle], strict=True):
                for key in ("law_a", "law_b", "a", "b", "orders", "draws"):
                    if not np.array_equal(fresh[key], start[key]):
                        raise ValueError(f"Checkpoint comparison changed {key}")
        diff = active.mean(axis=2, dtype=np.float32) - initial
        diff = diff[:, :, indices, :]
        bounds = paired.intervals(diff)
        per_birth = diff.mean(axis=1, dtype=np.float64)
        results[step] = dict(summary_sha256=report_hash, vs_step_zero={
            mode: {
                metric: dict(mean=float(per_birth[:, i, j].mean()), ci95=bounds[:, i, j].tolist(),
                             per_birth=per_birth[:, i, j].tolist())
                for j, metric in enumerate(stats.METRICS)
            }
            for i, mode in enumerate(compared)
        })
        if diff[:, :, 2:].any():
            raise AssertionError("Training step changed a checkpoint-independent control")
    return dict(label=label, baseline_step=steps[0], baseline_summary_sha256=initial_hash, comparisons=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--label", default="heldout")
    args = parser.parse_args()
    if args.geometry:
        name, output = "geometry-isolation", geometry_isolation()
    else:
        name, output = f"{args.label}-trends", trends(args.label)
    path = eval_run.EVIDENCE / f"evaluation-{name}.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(path)


if __name__ == "__main__":
    main()
