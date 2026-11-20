"""Verify the raw edge receipts and summarize paired behavioral contrasts."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import edge_site_diagnostic as run
from . import interleave_revisit_train as io
from . import noon_value_report as common


WORLDS = run.WORLDS
BASELINES = ("no_write", "all_edges", "full_native")
METRICS = ("actual_a", "actual_b", "actual_joint", "prob_a", "prob_b", "prob_joint")
METHOD_NAMES = tuple(f"{k}/{method}" for k in run.BUDGETS for method in run.METHODS)
BOOTSTRAP_DRAWS = 4000


def outcomes(archive, prefix):
    actual = archive[f"{prefix}/correct"]
    p = archive[f"{prefix}/probability"].astype(np.float64)
    return np.stack((actual[..., 0], actual[..., 1], actual.all(-1),
                     p[..., 0], p[..., 1], p.prod(-1)), axis=-1)


def subgroup(laws, values):
    same = (laws[:, 0] == laws[:, 1]).reshape(-1, 4)
    v = values.reshape(-1, 4, *values.shape[1:])
    if not (same.sum(axis=1) == 2).all():
        raise AssertionError("Each geometry must pair all four assignments")
    return {"same": v[same].reshape(-1, 2, *v.shape[2:]).mean(1),
            "opposite": v[~same].reshape(-1, 2, *v.shape[2:]).mean(1)}


def dose_check(archive):
    result = {}
    for k in run.BUDGETS:
        native_delta = archive[f"{k}/native/delta"]
        enabled = archive[f"{k}/native/enabled"]
        original = archive[f"{k}/native/address"]
        actual_delta = archive[f"{k}/random/delta"]
        address = archive[f"{k}/random/address"]
        fallback = archive[f"{k}/random/fallback"]
        receipt = archive["receipt/delta"]
        expected_original = receipt[np.arange(len(receipt))[:, None, None], original]
        if not np.array_equal(np.where(enabled[..., None], expected_original, 0), native_delta):
            raise ValueError("Original option does not match the native B receipt")
        same_sign = np.sign(actual_delta) == np.sign(native_delta)[:, :, None, :]
        moved = address != original[:, :, None, :]
        duplicate = (np.diff(np.sort(address, axis=3), axis=3) == 0).any(-1)
        if (duplicate & enabled[:, :, None]).any():
            raise ValueError("A multi-edge relocation reused a destination")
        random_direction = receipt[np.arange(len(receipt))[:, None, None, None], address]
        if not np.array_equal(np.sign(random_direction[fallback == 0]), np.sign(actual_delta[fallback == 0])):
            raise ValueError("Unmarked native-direction mismatch at the destination")
        if (moved[fallback == 2]).any() or not moved[fallback == 0].all():
            raise ValueError("Unmarked location fallback")
        a = archive[f"{k}/random/side"]
        b = archive[f"{k}/native/side"]
        if not np.array_equal(a[~enabled], np.broadcast_to(b[:, :, None, :], a.shape)[~enabled]):
            raise ValueError("Random control changed an omitted write")
        if not same_sign.all():
            raise ValueError("Real signed dose differs between arms")
        on = fallback >= 0
        gap = np.abs(actual_delta - native_delta[:, :, None, :])
        group_gap = np.abs(np.abs(actual_delta).sum(-1) - np.abs(native_delta).sum(-1)[:, :, None])
        result[k] = dict(payloads=int(on.sum()), same_direction=int((fallback == 0).sum()),
                         moved=int((moved & on).sum()), relaxed_direction=int((fallback == 1).sum()),
                         original_destination=int((fallback == 2).sum()), native_l1=float(np.abs(native_delta).sum()),
                         near_exact_step=int(((gap <= 1e-7) & on).sum()), absolute_step_sum=float(gap.sum()),
                         max_step=float(gap.max()), max_group=float(group_gap.max()),
                         max_score_error=float(archive[f"{k}/score_replay_max_error"].max()))
    return result


def read_archive(path, cohort, birth, angle, seed, memo):
    with np.load(path, allow_pickle=False) as archive:
        if (int(archive["worlds"]), str(archive["cohort"]), int(archive["birth"]),
                int(archive["angle"]), int(archive["seed"])) != (WORLDS, cohort, birth, angle, seed):
            raise ValueError(f"Unexpected replay provenance in {path}")
        sources = {"plan": run.PLAN, "runner": Path(run.__file__), "checkpoint": run.ROOT / str(archive["checkpoint"])}
        for file in sources.values():
            if file not in memo:
                memo[file] = run.prior.digest(file)
        signatures = {key: memo[file] for key, file in sources.items()}
        if any(signatures[key] != str(archive[f"sha256/{key}"]) for key in sources):
            raise ValueError(f"Stale inputs for {path}")
        world = run.pack_worlds(seed, WORLDS, angle)
        laws = np.column_stack((world.law_a.numpy(), world.law_b.numpy()))
        for saved, target in (("a", world.a.numpy()), ("b", world.b.numpy()), ("laws", laws),
                              ("orders", world.orders.numpy())):
            if not np.array_equal(archive[saved], target):
                raise ValueError(f"Shared world mismatch in {path} for {saved}")
        taste_goal = laws[:, 1] * archive["orders"][:, 1]
        if not np.array_equal(archive["receipt/side"] * archive["receipt/taste"], taste_goal):
            raise ValueError(f"B taste differs from the side contacted in {path}")
        base = outcomes(archive, "baseline")
        absolute = {name: base[:, i] for i, name in enumerate(BASELINES)}
        comparison = {"all_edges-no_write": base[:, 1] - base[:, 0],
                      "full_native-no_write": base[:, 2] - base[:, 0],
                      "full_native-all_edges": base[:, 2] - base[:, 1]}
        counts = {"edge_count": (archive["receipt/delta"] != 0).sum(-1),
                  "receipt_l1": np.abs(archive["receipt/delta"]).sum(-1),
                  "born_fraction": archive["receipt/born"] >= 0,
                  "retired_fraction": archive["receipt/retired"] >= 0,
                  "bias_l1": archive["receipt/bias_l1"]}
        for k in run.BUDGETS:
            native = outcomes(archive, f"{k}/native")
            random = outcomes(archive, f"{k}/random").mean(2)
            counts[f"{k}/both_conditional_gain_1pp"] = archive[f"{k}/feasible_both_gain"]
            for j, method in enumerate(run.METHODS):
                key = f"{k}/{method}"
                absolute[key] = native[:, j]
                absolute[f"{key}/relocated"] = random[:, j]
                comparison[f"{key}-no_write"] = native[:, j] - base[:, 0]
                comparison[f"{key}-all_edges"] = native[:, j] - base[:, 1]
                comparison[f"{key}-relocated"] = native[:, j] - random[:, j]
                counts[f"{key}/write_fraction"] = archive[f"{k}/native/enabled"][:, j]
                counts[f"{key}/l1"] = np.abs(archive[f"{k}/native/delta"][:, j]).sum(-1)
            comparison[f"{k}/joint-dose"] = native[:, 1] - native[:, 0]
        metadata = {"path": path.relative_to(run.ROOT).as_posix(), "sha256": run.prior.digest(path),
                    "checkpoint": str(archive["checkpoint"]), "checkpoint_sha256": signatures["checkpoint"]}
        return ([{key: subgroup(laws, value) for key, value in group.items()}
                 for group in (absolute, comparison, counts)], dose_check(archive), metadata)


def block(rows, index, relation):
    return {key: np.stack([row[index][key][relation] for row in rows])
            for key in rows[0][index]}


def stat_metrics(interval, values):
    flat = {f"{key}/{metric}": row[..., i]
            for key, row in values.items() for i, metric in enumerate(METRICS)}
    summary = interval.summary(flat)
    return {key: {m: summary[f"{key}/{m}"] for m in METRICS} for key in values}


def combine_dose(rows):
    summed = ("payloads", "same_direction", "moved", "relaxed_direction", "original_destination",
              "native_l1", "near_exact_step", "absolute_step_sum")
    maxima = ("max_step", "max_group", "max_score_error")
    return {k: {**{key: sum(row[k][key] for row in rows) for key in summed},
                **{key: max(row[k][key] for row in rows) for key in maxima}}
            for k in run.BUDGETS}


def report(cohort, data, output):
    first, seed = run.COHORTS[cohort]
    births = list(range(first, first + 6))
    interval = common.PairedCrossed(len(births), WORLDS, seed + 1407, draws=BOOTSTRAP_DRAWS)
    rotation, relocated, files, memo = {}, [], [], {}
    for angle in run.ANGLES:
        raw = []
        for birth in births:
            rows, dose, metadata = read_archive(data / cohort / f"{birth}-{angle}.npz",
                                               cohort, birth, angle, seed, memo)
            raw.append(rows)
            relocated.append(dose)
            files.append(metadata)
        rotation[angle] = {relation: tuple(block(raw, i, relation) for i in range(3))
                           for relation in ("same", "opposite")}
    estimates = {}
    for angle, relations in rotation.items():
        estimates[angle] = {}
        for relation, (absolute, contrasts, counts) in relations.items():
            base = {**contrasts, **{f"4-1/{mode}": absolute[f"4/{mode}"] - absolute[f"1/{mode}"]
                                  for mode in run.METHODS}}
            estimates[angle][relation] = {"absolute": stat_metrics(interval, absolute),
                                          "contrasts": stat_metrics(interval, base),
                                          "counts": interval.summary(counts)}
    focus = ("all_edges-no_write", "1/joint-relocated", "2/joint-relocated", "4/joint-relocated",
             "1/protect_a-relocated", "4/protect_a-relocated")
    aa, bb = run.ANGLES[0], run.ANGLES[-1]
    angle_by_law = {key: (rotation[aa]["opposite"][1][key] - rotation[bb]["opposite"][1][key])
                    - (rotation[aa]["same"][1][key] - rotation[bb]["same"][1][key]) for key in focus}
    summary = dict(cohort=cohort, births=births, seed=seed, worlds=WORLDS, maps_per_world=4,
                   matched_draws=run.CONTROLS, bootstrap_draws=BOOTSTRAP_DRAWS,
                   evidence={"plan_sha256": memo[run.PLAN], "runner_sha256": memo[Path(run.__file__)],
                             "reporter_sha256": run.prior.digest(Path(__file__)), "files": files},
                   outcomes=estimates, angle_by_law_15_90=stat_metrics(interval, angle_by_law),
                   relocation=combine_dose(relocated))
    io._write_json(output / f"diagnostic-{cohort}.json", summary)
    print(f"{cohort} verified {len(files)} raw archives", flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("cohort", nargs="+", choices=run.COHORTS)
    p.add_argument("--data", type=Path, default=run.DATA)
    p.add_argument("--output", type=Path, default=run.EVIDENCE)
    a = p.parse_args()
    for cohort in a.cohort:
        report(cohort, a.data, a.output)
