"""Audit physical dose and estimate the registered paired endpoint differences."""

import numpy as np

from . import edge_site_eval_stats as paired


CONTRASTS = (
    ("post_minus_native", {"post": 1, "native": -1}),
    ("post_minus_random", {"post": 1, "random": -1}),
    ("post_minus_post_wrong", {"post": 1, "post_wrong": -1}),
    ("post_minus_wrong", {"post": 1, "wrong": -1}),
    ("post_minus_pre", {"post": 1, "pre": -1}),
    ("pre_minus_native", {"pre": 1, "native": -1}),
    ("pre_minus_random", {"pre": 1, "random": -1}),
    ("pre_minus_pre_wrong", {"pre": 1, "pre_wrong": -1}),
    ("wrong_minus_random", {"wrong": 1, "random": -1}),
    ("post_wrong_minus_random", {"post_wrong": 1, "random": -1}),
    ("rehearsal_content_interaction", {"post": 1, "post_wrong": -1, "pre": -1, "pre_wrong": 1}),
)
DOSE_TOLERANCE = 2e-6
LAW_PAIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PRECISION = ("written_count", "positive_count", "negative_count", "deferred_count", "edge_births",
             "edge_retirements", "selected_count")
SIGNED_DOSE = ("positive_l1", "negative_l1", "l1", "deferred_positive_l1", "deferred_negative_l1", "deferred_l1",
               "plan_positive_l1", "plan_negative_l1", "positive_bias_l1", "negative_bias_l1")


def validate(row, manifest):
    """Reconstruct truth, exact random picks and pairable scores from archived arrays."""
    count = manifest["geometry_roots"]
    modes = tuple(manifest["modes"])
    native, random = (modes.index(name) for name in ("native", "random"))
    reads = row["reads"].tolist()
    replicas = len(manifest["replica_seeds"])
    orders = len(manifest["orders"])
    lanes = count * 4
    expected = (len(reads), replicas, len(modes), orders, lanes, 2)
    sides = row["side"]
    if sides.shape != expected or not np.isin(sides, [-1, 1]).all():
        raise ValueError("Invalid recorded real actions")
    laws = row["laws"]
    np.testing.assert_array_equal(laws, np.tile(LAW_PAIRS, (count, 1)))
    orientation = row["orders"][:, 3:5]
    truth = orientation * laws
    np.testing.assert_array_equal(row["truth"], truth)
    real_sides = row["receipt_side"]
    real_tastes = row["receipt_taste"]
    real_menus = row["receipt_menus"]
    np.testing.assert_array_equal(real_menus[:, 1:, :], row["geometry"] * row["orders"][:, :2, None])
    np.testing.assert_array_equal(real_sides[:, 1:] * real_tastes[:, 1:], row["orders"][:, :2] * laws)
    if not np.isin(real_sides, (-1, 1)).all() or not np.isin(real_tastes, (-1, 1)).all():
        raise AssertionError("Expected genuine signed tastes and emitted sides for each study contact")
    np.testing.assert_allclose(np.linalg.norm(real_menus[:, 0, :], axis=1),
                               np.linalg.norm(row["unrelated_c"], axis=1), rtol=0, atol=1e-6)
    c_alignment = np.einsum("ni,nki->nk", row["unrelated_c"], row["geometry"])
    if np.max(np.abs(c_alignment)) > 2e-6:
        raise AssertionError("Unrelated C menu overlaps the target geometries")
    unrelated_sign = np.einsum("ni,ni->n", real_menus[:, 0, :], row["unrelated_c"]) > 0
    c_law = real_sides[:, 0] * real_tastes[:, 0] * (2 * unrelated_sign - 1)
    if not np.array_equal(c_law.reshape(count, 4), np.broadcast_to(c_law[::4, None], (count, 4))):
        raise AssertionError("Unrelated C varies with the A/B assignment within a geometry root")

    same_start = row["transient_error"]
    live = sides[reads.index("live_4")]
    common = sides[reads.index("common_4")]
    differences = int(np.count_nonzero(live != common))
    if differences or np.any(same_start != 0):
        raise AssertionError("Common-start graph readout changed the verified common transients")

    dose = row["dose"]
    columns = manifest["dose_columns"]
    exact = [columns.index(key) for key in PRECISION]
    magnitudes = [columns.index(key) for key in SIGNED_DOSE]
    physical = {key: dose[..., columns.index(key)] for key in columns}
    address = row["address"]
    option = row["candidate"]
    mask = row["plan"] != 0
    if not np.all((option >= 0) == mask[..., None]):
        raise AssertionError("Candidate validity does not match the prespecified dose")
    if np.any(address[:, native] != -1) or np.any(physical["max_unselected_change"] != 0):
        raise AssertionError("A native or unselected address changed contrary to the intervention")
    deferred = np.stack((np.maximum(row["plan"], 0).sum(1, dtype=np.float64),
                         -np.minimum(row["plan"], 0).sum(1, dtype=np.float64)), axis=1)
    sign_columns = [columns.index(key) for key in ("deferred_positive_l1", "deferred_negative_l1")]
    drawn_index = (row["selector_draw"][0].repeat(4, axis=1) * option.shape[-1]).astype(int)
    expected_random = np.take_along_axis(option[None], drawn_index[..., None], axis=-1)[..., 0]
    np.testing.assert_array_equal(address[:, random], expected_random)

    observed_dose_error = 0.0
    measured_vs_plan = 0.0
    for mode in range(len(modes)):
        if mode == native:
            continue
        np.testing.assert_array_equal(address[:, mode] >= 0, np.broadcast_to(mask, address[:, mode].shape))
        if not np.all((option[None] == address[:, mode, :, :, None]).any(axis=-1)):
            raise AssertionError("An irreversible action addressed an edge outside the native candidate group")
        np.testing.assert_array_equal(dose[:, mode, :, exact], dose[:, random, :, exact])
        error = np.max(np.abs(dose[:, mode, :, magnitudes] - dose[:, random, :, magnitudes]))
        observed_dose_error = max(observed_dose_error, float(error))
        error = np.max(np.abs(dose[:, mode][..., sign_columns] - deferred))
        measured_vs_plan = max(measured_vs_plan, float(error))
    if max(observed_dose_error, measured_vs_plan) > DOSE_TOLERANCE:
        raise AssertionError(f"The main address-only comparison drifted: {observed_dose_error}, {measured_vs_plan}")
    if np.any(dose[:, native, :, sign_columns] != 0):
        raise AssertionError("The full native comparison withheld numerical dose")

    good = sides == truth
    scores = paired.score(good)
    return {name: scores[index] for index, name in enumerate(reads)}, dict(
        live_vs_common_action_mismatches=differences, max_transient_error=float(same_start.max()),
        max_matched_dose_error=observed_dose_error, max_realized_minus_planned_deferred_error=measured_vs_plan,
        max_unrelated_c_geometric_dot=float(np.max(np.abs(c_alignment))),
        eligible_groups=int(mask.sum()), min_groups_per_lane=int(mask.sum(1).min()),
        max_groups_per_lane=int(mask.sum(1).max()), native_weight_l1=float(physical["l1"][:, native].mean()),
        selected_weight_l1=float(physical["l1"][:, random].mean()),
        signed_deferred_l1=float(physical["deferred_l1"][:, random].mean()))


def summarize(by_birth, modes, bootstrap, count):
    """Fold deterministic law, read-order and head repeats into each crossed cell."""
    means = by_birth.mean(2, dtype=np.float32)
    contrasts = np.zeros((*means.shape[:2], len(CONTRASTS), means.shape[-1]), dtype=np.float32)
    for column, (_, terms) in enumerate(CONTRASTS):
        for arm, weight in terms.items():
            contrasts[:, :, column] += means[:, :, modes.index(arm)] * weight
    both = np.concatenate((means, contrasts), axis=2)
    intervals = bootstrap.intervals(both)
    per_birth = both.mean(axis=1, dtype=np.float64)
    successes = by_birth.sum(axis=(1, 2), dtype=np.int64)
    denominator = count * by_birth.shape[2]

    def rows(index, is_rate):
        metrics = {}
        for k, metric in enumerate(paired.METRICS):
            values = per_birth[:, index, k]
            record = dict(mean=float(values.mean()), ci95=intervals[:, index, k].tolist(),
                          per_birth=values.tolist(), birth_range=[float(values.min()), float(values.max())])
            if is_rate:
                observed = successes[:, index, k]
                record.update(successes=int(observed.sum()), denominator=int(denominator * len(by_birth)),
                              successes_per_birth=observed.tolist(), denominator_per_birth=int(denominator))
            metrics[metric] = record
        return metrics

    return dict(rates={name: rows(i, True) for i, name in enumerate(modes)},
                contrasts={name: rows(len(modes) + i, False) for i, (name, _) in enumerate(CONTRASTS)})


def panel(panels, births, angles, read, agree, count, modes, bootstrap):
    laws = [i for i, law in enumerate(LAW_PAIRS) if (law[0] == law[1]) == agree]
    by_birth = []
    for birth in births:
        sets = []
        for angle in angles:
            # Shape before reordering: training head, arm, answer order, root, law, metric.
            raw = panels[birth, angle][read].reshape(-1, len(modes), 2, count, 4, len(paired.METRICS))
            crossed = np.take(raw, laws, axis=4).transpose(3, 0, 2, 4, 1, 5)
            sets.append(crossed.reshape(count, -1, len(modes), len(paired.METRICS)))
        by_birth.append(np.concatenate(sets, axis=1))
    return summarize(np.stack(by_birth), modes, bootstrap, count)


def threshold(value, lower):
    return dict(ci95=value["ci95"], lower_bound_required=lower, passed=value["ci95"][0] > lower)


def primary_gates(result):
    comparisons = result["contrasts"]
    gates = {}
    for arm in ("post", "pre"):
        native = comparisons[f"{arm}_minus_native"]
        random = comparisons[f"{arm}_minus_random"]
        wrong = comparisons[f"{arm}_minus_{arm}_wrong"]
        passes = dict(a_over_native=threshold(native["a"], 0), both_over_native=threshold(native["both"], 0),
                      b_noninferior_to_native=threshold(native["b"], paired.B_MARGIN),
                      both_over_random=threshold(random["both"], 0), both_over_wrong=threshold(wrong["both"], 0))
        gates[arm] = dict(advance=all(item["passed"] for item in passes.values()), **passes,
                          a_over_wrong=threshold(wrong["a"], 0),
                          b_noninferior_to_wrong=threshold(wrong["b"], paired.B_MARGIN))
    timing = comparisons["post_minus_pre"]
    interaction = comparisons["rehearsal_content_interaction"]["both"]
    gates["post_over_pre"] = dict(a=threshold(timing["a"], 0), b_noninferior=threshold(timing["b"], paired.B_MARGIN),
                                  both=threshold(timing["both"], 0))
    gates["content_interaction"] = dict(positive=threshold(interaction, 0), negative=interaction["ci95"][1] < 0)
    return gates


def analyze(panels, manifest):
    births, count, modes = (manifest[name] for name in ("births", "geometry_roots", "modes"))
    bootstrap = paired.CrossedBootstrap(len(births), count)
    results = {}
    for read in next(iter(panels.values())):
        results[f"{read}/primary"] = panel(panels, births, (15, 30), read, False, count, modes, bootstrap)
        if read == "live_4":
            for angle in manifest["angles"]:
                for agree in (False, True):
                    name = f"{read}/{angle}/{'agree' if agree else 'conflict'}"
                    results[name] = panel(panels, births, (angle,), read, agree, count, modes, bootstrap)
    gates = {read: primary_gates(results[f"{read}/primary"]) for read in ("live_4", "common_4")}
    advance = {arm: all(check[arm]["advance"] for check in gates.values()) for arm in ("post", "pre")}
    return dict(bootstrap=dict(draws=paired.N_BOOT, seed=paired.BOOT_SEED, B_margin=paired.B_MARGIN),
                gates=gates, advance_to_three_objects=advance, results=results)
