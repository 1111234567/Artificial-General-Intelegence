"""Recompute published claims and replay a complete sealed evaluation cell."""

from argparse import ArgumentParser
import json

import numpy as np
import torch

from . import draft_rehearse_eval as evaluation


def source(path_info):
    path = evaluation.ROOT / path_info["path"]
    if evaluation.fingerprint(path) != path_info:
        raise ValueError(f"Published input changed: {path}")
    return path


def frozen_heads(manifest):
    heads, metadata = evaluation.read_heads(source(manifest["heads"]))
    evaluation.check_fit_lineage(metadata)
    summary = json.loads((evaluation.EVIDENCE / "fit-summary.json").read_text(encoding="utf-8"))
    if summary["heads_sha256"] != manifest["heads"]["sha256"]:
        raise ValueError("Frozen evaluation heads disagree with the training seal")
    checkpoints = []
    for replica, seed in enumerate(metadata["replica_seeds"]):
        path = evaluation.DATA / f"fit-replica-{replica}" / f"step-{metadata['final_step']:04d}.npz"
        provenance = evaluation.fingerprint(path)
        if summary["replicas"][replica]["files"][provenance["path"]] != provenance["sha256"]:
            raise ValueError("Training checkpoint seal disagrees with the frozen bundle")
        with np.load(path, allow_pickle=False) as final:
            actual = (int(final["step"]), int(final["seed"]), int(final["replica"]), str(final["protocol"]))
            expected = (metadata["final_step"], seed, replica, metadata["fit_protocol_sha256"])
            if actual != expected:
                raise ValueError("Head was sourced from the wrong training stage")
            for arm, rows in heads.items():
                np.testing.assert_array_equal(final[arm], rows[replica])
        checkpoints.append(provenance)
    return heads, checkpoints


def counts(row, read, arms, roots):
    selected = row["side"][row["reads"].tolist().index(read)]
    laws = row["laws"]
    target = row["orders"][:, 3:5] * laws
    scores = {}
    for index, name in enumerate(arms):
        a, b = np.moveaxis(selected[:, index] == target, -1, 0)
        by_metric = np.stack((a, b, a & b), axis=-1)
        paired = by_metric.reshape(len(a), a.shape[1], roots, 4, 3)[:, :, :, 1:3]
        scores[name] = paired.sum(axis=(0, 1, 3), dtype=np.int64)
    return scores


def recompute(report, manifest, raw):
    roots = manifest["geometry_roots"]
    births = manifest["births"]
    geometry_panel_size = len(manifest["replica_seeds"]) * len(manifest["orders"]) * 2 * 2
    labels = tuple(report["results"]["live_4/primary"]["rates"]["post"])
    rng = np.random.default_rng(report["bootstrap"]["seed"])

    def weights(size):
        draws = rng.multinomial(size, np.repeat(1 / size, size), size=report["bootstrap"]["draws"])
        return draws.astype(np.float32) / size

    weight_births, weight_roots = weights(len(births)), weights(roots)
    maximum_ci_error = 0.0
    counts_checked = 0
    contrasts_checked = 0
    for read in ("live_4", "common_4", "live_32", "b_three_4"):
        key = f"{read}/primary"
        if key not in report["results"]:
            continue
        cells = {arm: [] for arm in manifest["modes"]}
        for birth in births:
            per_angle = []
            for angle in (15, 30):
                with np.load(raw[f"{birth}-{angle}.npz"], allow_pickle=False) as row:
                    per_angle.append(counts(row, read, manifest["modes"], roots))
            for arm in cells:
                cells[arm].append(per_angle[0][arm] + per_angle[1][arm])
        cells = {arm: np.stack(values) for arm, values in cells.items()}
        for arm, values in cells.items():
            for metric, label in enumerate(labels):
                rate = report["results"][key]["rates"][arm][label]
                expected = values[:, :, metric].sum(1)
                np.testing.assert_array_equal(expected, rate["successes_per_birth"])
                if int(expected.sum()) != rate["successes"]:
                    raise AssertionError("Published action tally does not match the raw lives")
                counts_checked += 1
        for contrast, terms in evaluation.stats.CONTRASTS:
            change = sum(cells[arm] * coefficient for arm, coefficient in terms.items()) / geometry_panel_size
            for index, metric in enumerate(labels):
                values = change[:, :, index]
                recorded = report["results"][key]["contrasts"][contrast][metric]
                np.testing.assert_allclose(values.mean(1), recorded["per_birth"], rtol=0, atol=1e-10)
                draws = np.einsum("ib,br,ir->i", weight_births, values, weight_roots, optimize=True, dtype=np.float64)
                ci = np.quantile(draws, [0.025, 0.975])
                maximum_ci_error = max(maximum_ci_error, float(np.max(np.abs(ci - recorded["ci95"]))))
                contrasts_checked += 1
    if maximum_ci_error > 5e-7:
        raise AssertionError(f"Published confidence intervals failed raw-tally reconstruction: {maximum_ci_error}")
    return dict(counts_checked=counts_checked, contrast_ci_checked=contrasts_checked, maximum_ci_error=maximum_ci_error)


def verify(label):
    report = json.loads((evaluation.EVIDENCE / f"eval-{label}.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    for info in (*manifest["inputs"].values(), *manifest["parent_sha256"].values()):
        source(info)
    heads, checkpoints = frozen_heads(manifest)
    raw = {filename: source(info) for filename, info in manifest["files"].items()}
    audit = recompute(report, manifest, raw)
    birth, angle = manifest["births"][0], 15
    graph, rule = evaluation.draft.load_parent(birth)
    world = evaluation.worlds(manifest["seed"], 0, manifest["geometry_roots"], angle)
    rerun = evaluation.evaluate(graph, rule, world, heads, manifest["repeat_b"])
    with np.load(raw[f"{birth}-{angle}.npz"], allow_pickle=False) as archived:
        for key, values in rerun.items():
            np.testing.assert_array_equal(values, archived[key])
    result = dict(**audit, raw_cells_verified=len(raw), rerun_birth=birth, rerun_angle=angle,
                  rerun_arrays=sorted(rerun), final_checkpoints=checkpoints,
                  coefficient_l2={arm: np.linalg.norm(head, axis=1).tolist() for arm, head in heads.items()})
    result_path = evaluation.EVIDENCE / f"eval-{label}-audit.json"
    evaluation.write_json(result_path, result)
    print(result_path, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="heldout")
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(args.label)
