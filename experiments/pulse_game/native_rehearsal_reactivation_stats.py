"""Rebuild actual motor scores and paired measurements of the quiet current."""

import numpy as np

from . import edge_site_eval_stats as paired


LANES = 16
WINDOWS = {"b_taste": slice(5, 7), "quiet_early": slice(7, 11), "quiet_late": slice(15, 19)}
BEHAVIOR = (("live_minus_cut", "live", "cut"), ("live_minus_own", "live", "own"),
            ("own_minus_swap_a", "own", "swap_a"), ("own_minus_swap_c", "own", "swap_c"),
            ("own_minus_rotated", "own", "rotated"), ("swap_c_minus_swap_a", "swap_c", "swap_a"),
            ("rotated_minus_swap_a", "rotated", "swap_a"))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def dots(signal, target):
    return np.einsum("tni,ni->tn", signal, target)


def measure(row, roots):
    count = LANES * roots
    answer = row["sides"]
    target = row["laws"][:, (0, 2)]
    direction = row["orientation"][:, 3:5]
    truth = target[None] * np.stack((direction, direction[:, ::-1]))
    pay = answer * truth[None]
    np.testing.assert_array_equal(pay, row["pay"])
    if not np.isin(answer, (-1, 1)).all():
        raise ValueError("Actual actions must be signed choices")
    obj = row["study_obj"]
    given = row["orientation"][:, :3] * np.take_along_axis(row["laws"], obj, axis=1)
    np.testing.assert_array_equal(row["study_side"] * row["study_taste"], given)
    receipt_directions = row["study_right"] * (row["study_side"] * row["study_taste"])[..., None]
    training_direction = np.take_along_axis(row["favorable"], obj[..., None], axis=1)
    np.testing.assert_array_equal(receipt_directions, training_direction)
    good = paired.score(pay > 0)
    modes = row["modes"].tolist()
    outcomes = {name: results.reshape(2, roots, LANES, 3).transpose(1, 0, 2, 3).reshape(roots, -1, 3)
                for name, results in zip(modes, good, strict=True)}
    margins = {name: results.reshape(2, roots, LANES, 2).mean((0, 2), dtype=np.float64)
               for name, results in zip(modes, row["margins"], strict=True)}
    masks = row["angle"][::LANES]
    if "currents" not in row:
        return dict(angle=masks, behavior=outcomes, margins=margins, current=None, audit={})

    incoming = np.concatenate((row["b_contact_current"], row["currents"]), axis=0).astype(np.float64)
    if incoming.shape != (19, count, 20) or not np.isfinite(incoming).all():
        raise ValueError("Expected the physical primary population's incoming currents")
    cue = (incoming[..., 3:6] - incoming[..., :3]) / 2
    old_a, old_c, b = np.moveaxis(unit(row["favorable"].astype(np.float64)), 1, 0)
    a_tangent = unit(old_a - (old_a * b).sum(-1, keepdims=True) * b)
    lanes = np.arange(count)
    a_diff = (cue - cue[:, lanes ^ 4]) / 2
    c_diff = (cue - cue[:, lanes ^ 2]) / 2
    same_side = row["study_side"] == row["study_side"][lanes ^ 4]
    same_taste = row["study_taste"] == row["study_taste"][lanes ^ 4]
    same_receipts = np.all((same_side & same_taste) | (obj == 0), axis=1)
    stable = (same_receipts & same_receipts[lanes ^ 8]).reshape(roots, LANES)
    # The mirror-invariant contribution cancels in the mean of the main metric.
    # Retain the explicit mirror-difference measurement as an independent check.
    a_mirror = (a_diff - a_diff[:, lanes ^ 8]) / 2
    current = dict(a_law_tangent=dots(a_diff, a_tangent), c_law_aligned=dots(c_diff, old_c),
                   a_law_mirror_tangent=dots(a_mirror, a_tangent), a_law_whole=dots(a_diff, old_a),
                   raw_a_cosine=dots(unit(cue), old_a), raw_c_cosine=dots(unit(cue), old_c),
                   raw_b_cosine=dots(unit(cue), b), sensory_cue_norm=np.linalg.norm(cue, axis=-1))
    panels = {name: values.reshape(19, roots, LANES).mean(-1).T for name, values in current.items()}
    np.testing.assert_allclose(panels["a_law_tangent"], panels["a_law_mirror_tangent"], atol=2e-7, rtol=0)
    audit = dict(own_state_error=row["own_state_error"].tolist(), rotation_max_error=row["rotation_max_error"].tolist(),
                 live_own_changed_actions=int(np.count_nonzero(answer[0] != answer[1])),
                 dose_rms_per_arm=row["dose_rms"].mean(axis=(1, 2), dtype=np.float64).tolist())
    stable_motor = {name: (values.reshape(roots, 2, LANES, 3) * stable[:, None, :, None]).sum(axis=(1, 2)) / 2
                    for name, values in outcomes.items()}
    stable_current = {name: (values.reshape(19, roots, LANES) * stable[None]).sum(-1).T
                      for name, values in current.items()}
    reliable = dict(raw_matches=same_receipts.reshape(roots, LANES).sum(-1), matches=stable.sum(-1),
                    current=stable_current, motor=stable_motor)
    return dict(angle=masks, behavior=outcomes, margins=margins, current=panels, audit=audit, matched=reliable)


def estimate(cells, bootstrap):
    """Return intervals for columns of already matched birth-by-root cells."""
    keys = tuple(cells)
    values = np.stack([cells[k] for k in keys], axis=-1)
    bounds = bootstrap.intervals(values)
    per_birth = values.mean(axis=1, dtype=np.float64)
    return {key: dict(mean=float(per_birth[:, i].mean()), ci95=bounds[:, i].tolist(),
                      per_birth=per_birth[:, i].tolist()) for i, key in enumerate(keys)}


def current_panel(rows, mask, bootstrap):
    if rows[0]["current"] is None:
        return None
    window_cells, time_cells = {}, {}
    for name in rows[0]["current"]:
        view = np.stack([row["current"][name][mask] for row in rows])
        for t in range(view.shape[-1]):
            time_cells[f"{name}/{t}"] = view[..., t]
        window = {key: view[..., section].mean(-1) for key, section in WINDOWS.items()}
        for key, values in window.items():
            window_cells[f"{name}/{key}"] = values
        window_cells[f"{name}/late_minus_early"] = window["quiet_late"] - window["quiet_early"]
        window_cells[f"{name}/late_minus_b_taste"] = window["quiet_late"] - window["b_taste"]
    timeline = estimate(time_cells, bootstrap)
    return dict(windows=estimate(window_cells, bootstrap), timeline={name: [timeline[f"{name}/{t}"] for t in range(19)]
                for name in rows[0]["current"]})


def motor_panel(rows, mask, bootstrap):
    rates = {}
    valid = rows[0]["behavior"]
    means = {}
    for arm in valid:
        raw = np.stack([row["behavior"][arm][mask] for row in rows])
        by_cell = raw.mean(axis=2, dtype=np.float64)
        means[arm] = by_cell
        rates[arm] = estimate({k: by_cell[..., i] for i, k in enumerate(paired.METRICS)}, bootstrap)
        for i, k in enumerate(paired.METRICS):
            rates[arm][k].update(successes=int(raw[..., i].sum()), denominator=int(np.prod(raw.shape[:-1])))
    contrasts = {}
    if len(means) > 1:
        effects = {name: means[high] - means[low] for name, high, low in BEHAVIOR}
        flat = {f"{name}/{key}": differences[..., i] for name, differences in effects.items()
                for i, key in enumerate(paired.METRICS)}
        summary = estimate(flat, bootstrap)
        contrasts = {name: {key: summary[f"{name}/{key}"] for key in paired.METRICS} for name in effects}
    margins = {arm: np.stack([row["margins"][arm][mask] for row in rows]).mean((0, 1)).tolist() for arm in valid}
    return dict(rates=rates, contrasts=contrasts, mean_signed_motor_logit=margins)


def gate(value, lower=0):
    return dict(passed=value["ci95"][0] > lower, minimum_lower_bound=lower, ci95=value["ci95"])


def decisions(panel):
    if panel["current"] is None:
        return None
    amps = panel["current"]["windows"]
    effects = panel["behavior"]["contrasts"]
    old_a = effects["own_minus_swap_a"]
    old_c = effects["swap_c_minus_swap_a"]
    equal = effects["rotated_minus_swap_a"]
    information = dict(late_a=gate(amps["a_law_tangent/quiet_late"]),
                       a_late_minus_early=gate(amps["a_law_tangent/late_minus_early"]),
                       a_late_minus_b_taste=gate(amps["a_law_tangent/late_minus_b_taste"]),
                       late_c=gate(amps["c_law_aligned/quiet_late"]))
    motor = dict(a=gate(old_a["a"]), b_noninferior=gate(old_a["b"], paired.B_MARGIN),
                 both=gate(old_a["both"]), a_after_c_disruption=gate(old_c["a"]),
                 a_after_equal_dose_disruption=gate(equal["a"]))
    behavior = all(item["passed"] for item in motor.values())
    strengthens = all(information[k]["passed"] for k in ("late_a", "a_late_minus_early", "a_late_minus_b_taste"))
    return dict(info=information, motor=motor, behavior_pass=behavior, a_strengthens=strengthens,
                late_a_and_behavior=information["late_a"]["passed"] and behavior)


def fixed_receipts(rows, mask, bootstrap):
    if "matched" not in rows[0]:
        return None
    data = [row["matched"] for row in rows]
    eligible = np.stack([row["matches"][mask] for row in data])
    all_lanes = int(np.prod(eligible.shape) * LANES)
    variants = {}
    motor = {name: np.stack([row["motor"][name][mask] for row in data]) for name in data[0]["motor"]}
    for name, first, second in BEHAVIOR:
        differences = motor[first] - motor[second]
        for i, metric in enumerate(paired.METRICS):
            variants[f"{name}/{metric}"] = differences[..., i]
    for name in ("a_law_tangent", "a_law_whole", "c_law_aligned"):
        current = np.stack([row["current"][name][mask] for row in data])
        windows = {stage: current[..., selection].mean(-1) for stage, selection in WINDOWS.items()}
        for stage, totals in windows.items():
            variants[f"{name}/{stage}"] = totals
        variants[f"{name}/late_minus_early"] = windows["quiet_late"] - windows["quiet_early"]
        variants[f"{name}/late_minus_b_taste"] = windows["quiet_late"] - windows["b_taste"]
    matched = int(eligible.sum())
    simple = int(sum(row["raw_matches"][mask].sum() for row in data))
    if not matched:
        return dict(matched=0, total=all_lanes, matched_before_mirror=simple, ratio=None)
    birth = bootstrap.birth
    root = bootstrap.root
    weights = np.einsum("ib,br,ir->i", birth, eligible, root, optimize=True, dtype=np.float64)
    evaluations = {}
    for name, values in variants.items():
        sampled_sums = np.einsum("ib,br,ir->i", birth, values, root, optimize=True, dtype=np.float64)
        intervals = np.quantile(sampled_sums[weights > 0] / weights[weights > 0], [0.025, 0.975]).tolist()
        evaluations[name] = dict(mean=float(values.sum() / matched), ci95=intervals)
    return dict(matched=matched, total=all_lanes, matched_before_mirror=simple,
                matched_fraction=matched / all_lanes, valid_bootstrap=int(np.count_nonzero(weights > 0)),
                ratio=evaluations)


def report(rows, manifest):
    output = {}
    roots = manifest["roots"]
    for kind in manifest["kinds"]:
        by_birth = [rows[kind, birth] for birth in manifest["births"]]
        angle = by_birth[0]["angle"]
        for row in by_birth:
            np.testing.assert_array_equal(row["angle"], angle)
        masks = {"all": np.ones(roots, dtype=bool), "15_30": np.isin(angle, (15, 30))}
        panels = {}
        for name, mask in masks.items():
            bootstrap = paired.CrossedBootstrap(len(by_birth), int(mask.sum()))
            panels[name] = dict(geometry_roots=int(mask.sum()), current=current_panel(by_birth, mask, bootstrap),
                                behavior=motor_panel(by_birth, mask, bootstrap),
                                same_b_c_receipts=fixed_receipts(by_birth, mask, bootstrap))
        output[kind] = dict(panels=panels, decisions=decisions(panels["all"]),
                            audit={str(b): row["audit"] for b, row in zip(manifest["births"], by_birth, strict=True)})
    return dict(results=output, timeline=[f"b{i}" for i in range(1, 8)] + [f"quiet{i}" for i in range(1, 13)],
                bootstrap=dict(draws=paired.N_BOOT, seed=paired.BOOT_SEED, b_margin=paired.B_MARGIN),
                angles={str(degree): int(np.sum(angle == degree)) for degree in np.unique(angle)})
