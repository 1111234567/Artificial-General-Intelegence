"""Fit fixed linear readouts on new calibration geometries and score disjoint roots."""

import numpy as np

from . import edge_site_eval_stats as stats
from . import native_rehearsal_prospect as prospect


RIDGE = 0.03
METRIC_NAMES = ("a", "b")
RISK_NAMES = ("calibration_mean", "activity_only", "baseline", "augmented", "wrong_history",
              "opposite_calibration_mean", "opposite_activity_only", "opposite_baseline", "opposite_augmented",
              "swap_calibration_mean", "swap_activity_only", "swap_baseline", "swap_augmented")
EFFECT_NAMES = ("native_minus_frozen", "matched_plus_minus_minus", "wrong_feedback_minus_native",
                "wrong_feedback_changes_the_native_minus_frozen_contrast")


class Ridge:
    def __init__(self, features, target):
        features = features.astype(np.float64)
        self.center = features.mean(axis=0)
        spread = features.std(axis=0)
        self.scale = np.where(spread > 1e-10, spread, 1.)
        self.intercept = target.astype(np.float64).mean(axis=0)
        x = (features - self.center) / self.scale
        gram = x.T @ x / len(x)
        gram.flat[::len(gram) + 1] += RIDGE
        self.coef = np.linalg.solve(gram, x.T @ (target - self.intercept) / len(x))

    def predict(self, features):
        return (features - self.center) / self.scale @ self.coef + self.intercept


def augment(base, change):
    return np.concatenate((base, change), axis=1)


def change_modes(values):
    native, frozen, plus, minus, swap_native, swap_frozen = values.transpose(1, 0, 2)
    return (native - frozen, plus - minus, swap_native - swap_frozen)


def audit_split(data, kind, birth, split, digest, source):
    seed, roots = prospect.SPLITS[split]
    if (str(data["kind"]), int(data["birth"]), str(data["split"]), int(data["seed"]), int(data["roots"])) != (
            kind, birth, split, seed, roots):
        raise ValueError("Readout inputs have an incorrect geometry identity")
    if (str(data["plan"]), str(data["source"]), str(data["checkpoint"])) != (
            digest(prospect.PLAN), source, digest(prospect.checkpoint(kind, birth))):
        raise ValueError("Readout inputs have a mismatched model or source signature")
    scene = prospect.graph.scenes(seed, roots, evaluation=True)
    np.testing.assert_array_equal(data["first_query"], scene.first_query.numpy())
    np.testing.assert_array_equal(data["angle"], scene.angle.numpy())
    np.testing.assert_array_equal(data["modes"], prospect.MODES[:data["actions"].shape[1]])
    future_orientation = np.take_along_axis(scene.orientation[:, 3:].numpy(),
                                          np.stack((scene.first_query, 1 - scene.first_query), axis=1), axis=1)
    expected_payout = data["actions"] * future_orientation[:, None, :] * scene.laws.numpy()[:, None, ::2]
    np.testing.assert_array_equal(data["payouts"], expected_payout)
    assert np.isin(data["actions"], (-1, 1)).all()
    assert np.isin(data["payouts"], (-1, 1)).all()
    assert np.isfinite(data["future_prob"]).all() and ((data["future_prob"] > 0) & (data["future_prob"] < 1)).all()
    hard = np.isin(data["angle"], (15, 30)) & (scene.laws[:, 0] != scene.laws[:, 2]).numpy()
    return hard.reshape(roots, prospect.REPEATS)


def score_head(calibration, heldout):
    goal = calibration["future_prob"][:, 0] - calibration["future_prob"][:, 1]
    readout_mean = goal.mean(axis=0)
    activity_only = Ridge(calibration["activity_delta"], goal)
    baseline = Ridge(calibration["base"], goal)
    augmented = Ridge(augment(calibration["base"], calibration["activity_delta"]), goal)
    target = change_modes(heldout["future_prob"])
    base = heldout["base"]
    native = heldout["activity_delta"]
    pairs = ((base, native), (heldout["base_sign"], heldout["activity_sign"]),
             (heldout["base_swap"], heldout["activity_swap"]))
    errors = []
    for index, ((x_base, activity), true) in enumerate(zip(pairs, target, strict=True)):
        predictions = [readout_mean, activity_only.predict(activity), baseline.predict(x_base),
                       augmented.predict(augment(x_base, activity))]
        if index == 0:
            predictions.append(augmented.predict(augment(x_base, heldout["activity_donor"])))
        errors.extend((prediction - true) ** 2 for prediction in predictions)
    heads = {"activity_only": activity_only, "baseline": baseline, "augmented": augmented}
    parameters = {f"{name}_{field}": getattr(head, field) for name, head in heads.items()
                  for field in ("center", "scale", "intercept", "coef")}
    return np.stack(errors, axis=1), parameters


def effects(values):
    native, sign, swap = change_modes(values)
    return np.stack((native, sign, values[:, 4] - values[:, 0], swap - native), axis=1)


def summarize(metrics, names, bootstrap, hard):
    births, _, cols, objects = metrics.shape
    roots = hard.shape[0]
    grouped = metrics.reshape(births, roots, prospect.REPEATS, cols, objects)
    if not hard.any(axis=1).all():
        # Hard-angle subsets remove entire roots while preserving all paired rows of each remaining root.
        valid_roots = hard.any(axis=1)
        hard = hard[valid_roots]
        grouped = grouped[:, valid_roots]
        bootstrap = stats.CrossedBootstrap(births, len(hard))
    weighted = np.einsum("brnmo,rn->brmo", grouped, hard) / hard.sum(axis=1)[None, :, None, None]
    limits = bootstrap.intervals(weighted)
    point = weighted.mean(axis=1)
    return {name: {obj: {"mean": float(point.mean(0)[i, j]), "ci95": limits[:, i, j].tolist(),
                        "by_birth": point[:, i, j].tolist()}
                   for j, obj in enumerate(METRIC_NAMES)} for i, name in enumerate(names)}


def publish():
    digest = prospect.fit.digest
    source = digest(prospect.__file__)
    result = dict(plan_sha256=digest(prospect.PLAN),
                  analysis_plan_sha256=digest(prospect.EVIDENCE / "prospect-analysis-plan.md"),
                  source_sha256=source, report_source_sha256=digest(__file__),
                  ridge=RIDGE, roots_and_seeds=prospect.SPLITS, births=prospect.BIRTHS,
                  quiet_activity_ticks=prospect.SAMPLE_TICKS,
                  error_unit="squared percentage points", true_effect_unit="percentage points", arms={})
    for kind in prospect.KINDS:
        rows, true, sampled, paths, write_error = [], [], [], {}, []
        for birth in prospect.BIRTHS:
            splits = {name: prospect.artifact(kind, birth, name) for name in prospect.SPLITS}
            statuses = {}
            for split, path in splits.items():
                with np.load(path) as data:
                    statuses[split] = audit_split(data, kind, birth, split, digest, source)
                paths[path.relative_to(prospect.fit.ROOT).as_posix()] = digest(path)
            with np.load(splits["calibration"]) as calibration, np.load(splits["heldout"]) as heldout:
                error, parameters = score_head(calibration, heldout)
                rows.append(error)
                true.append(effects(heldout["future_prob"]))
                sampled.append(effects((heldout["payouts"] > 0).astype(np.float32)))
                write_error.append(float(heldout["dose_error"].max()))
            readout = prospect.DATA / kind / f"readout-seed-{birth}.npz"
            prospect.storage._write_raw(readout, **parameters, ridge=RIDGE, objects=METRIC_NAMES,
                                        calibration_sha256=digest(splits["calibration"]),
                                        report_source_sha256=digest(__file__))
            paths[readout.relative_to(prospect.fit.ROOT).as_posix()] = digest(readout)
        risk = np.stack(rows) * 1e4
        true = np.stack(true) * 100
        sampled = np.stack(sampled) * 100
        all_roots = np.ones_like(statuses["heldout"])
        bootstrap = stats.CrossedBootstrap(len(prospect.BIRTHS), len(all_roots))
        comparisons = ("activity_only_vs_calibration_mean", "increment_over_baseline", "matched_vs_wrong_history")
        gain = np.stack((risk[:, :, 0] - risk[:, :, 1], risk[:, :, 2] - risk[:, :, 3],
                         risk[:, :, 4] - risk[:, :, 3]), axis=2)
        measurements = {name: dict(risk=summarize(risk, RISK_NAMES, bootstrap, mask),
                                   gain=summarize(gain, comparisons, bootstrap, mask),
                                   true_future=summarize(true, EFFECT_NAMES, bootstrap, mask),
                                   sampled_future=summarize(sampled, EFFECT_NAMES, bootstrap, mask),
                                   roots=int(mask.any(axis=1).sum()), repeats=int(mask.sum()))
                        for name, mask in (("all", all_roots), ("hard_conflict", statuses["heldout"]))}
        result["arms"][kind] = dict(**measurements, raw_files_sha256=paths, max_opposite_write_error=max(write_error))
        print(kind, {name: {obj: round(value["mean"], 4) for obj, value in entry.items()}
                     for name, entry in measurements["all"]["gain"].items()}, flush=True)
    prospect.storage._write_json(prospect.EVIDENCE / "prospect-results.json", result)


if __name__ == "__main__":
    publish()
