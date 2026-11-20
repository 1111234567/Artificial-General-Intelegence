"""Audit private frozen rollouts and quantify paired prediction and behavior effects."""

from itertools import product

import numpy as np
import torch

from . import fork_lesson_eval as ev
from . import edge_site_eval_stats as inference


SUBSETS = ("all", "opposite", "close_opposite", "close_private_and_payoff_opposite")
METRICS = ("paid_positive", "greedy_paid_positive", "mean_payout", "forecast_order", "unseen_squared_error",
           "seen_squared_error", "unseen_sign", "seen_sign", "signed_forecast_margin", "signed_motor_logit",
           "both_mirrors_paid", "both_histories_paid", "action_flip_from_natural")
BETWEEN_METRICS = ("paid_positive", "greedy_paid_positive", "forecast_order", "unseen_squared_error", "unseen_sign")
CAUSAL_MODES = ("cut_out", "cut_back", "donor_out", "donor_back", "shuffle_out", "shuffle_back", "forecast_swap")


def with_half_ties(value):
    return np.where(value == 0, 0.5, value > 0)


def audit(raw, scene, geometry, angles, donor, mirror, kind, birth, split, checkpoint):
    for key, value in ev.signature(kind, birth, split, checkpoint).items():
        if raw[key].item() != value:
            raise ValueError(f"Input {key} has changed in {kind}/{birth}/{split}")
    for name, expected in (("angle", angles), ("geometry", geometry), ("donor", donor), ("mirror", mirror),
                           ("private_values", scene.values), ("vectors", scene.vectors[::ev.REPEATS]),
                           ("draws", scene.draws), ("forced_payoff", ev.world.oracle(scene))):
        np.testing.assert_array_equal(raw[name], expected.numpy())
    np.testing.assert_array_equal(raw["modes"], ev.active_modes(kind))
    if (raw["immediate"] != 0).any() or raw["max_energy_error"] > 1e-12:
        raise AssertionError("Immediate payoff or equal-current invariant failed")
    scores = raw["score"]
    motor = raw["motor_logit"]
    if not np.isfinite(scores).all() or not np.isfinite(motor).all():
        raise AssertionError("A forecast or motor logit is non-finite")
    probabilities = torch.from_numpy(motor).sigmoid().numpy()
    np.testing.assert_array_equal(raw["side"], raw["draws"][:, None] < probabilities)
    np.testing.assert_array_equal(raw["greedy_side"], motor > 0)
    for side, payoff in (("side", "payoff"), ("greedy_side", "greedy_payoff")):
        matched = np.take_along_axis(raw["forced_payoff"], raw[side], axis=1)
        np.testing.assert_array_equal(raw[payoff], matched)
    if kind in ev.INTERVENTION_KINDS:
        for name in ("side", "payoff", "greedy_side", "greedy_payoff", "score", "motor_logit"):
            np.testing.assert_array_equal(raw[name][:, 0], raw[name][:, 1])
        swapped = ev.MODES.index("forecast_swap")
        np.testing.assert_array_equal(scores[:, 0, ::-1], scores[:, swapped])
        for name, partners in (("no_history", donor), ("no_menu", mirror)):
            index = ev.MODES.index(name)
            np.testing.assert_array_equal(raw["side"][:, index], raw["side"][partners.numpy(), index])
            for observed in (scores, motor):
                # Identical inputs in different rows can land in different floating point kernel tiles.
                np.testing.assert_allclose(observed[:, index], observed[partners.numpy(), index], atol=1e-4, rtol=0)


def scores_and_masks(raw):
    side = raw["side"]
    score = raw["score"].astype(np.float64)
    actual = raw["forced_payoff"]
    good = raw["payoff"] > 0
    better_right = (actual[:, 1] - actual[:, 0]) / 2
    payout_margin = (score[:, :, 1] - score[:, :, 0]) * better_right[:, None]
    motor_margin = raw["motor_logit"] * better_right[:, None]
    natural_side = side[:, 0, None, None]
    target = actual[:, None, :]
    square = (score - target) ** 2
    sign = with_half_ties(score * target)

    def chosen(values, branch):
        return np.take_along_axis(values, branch, axis=2).squeeze(2)

    quantities = np.stack((good, raw["greedy_payoff"] > 0, raw["payoff"], with_half_ties(payout_margin),
                           chosen(square, 1 - natural_side), chosen(square, natural_side),
                           chosen(sign, 1 - natural_side), chosen(sign, natural_side), payout_margin, motor_margin,
                           good & good[raw["mirror"]], good & good[raw["donor"]], side != side[:, :1]), axis=2)
    opposite = better_right != 0
    close = np.repeat(raw["angle"] < ev.ANGLE_BOUNDS[1], ev.REPEATS)
    laws = raw["private_values"]
    private_opposite = laws[:, 0] != laws[:, 1]
    masks = (np.ones(len(actual), dtype=bool), opposite, opposite & close, opposite & close & private_opposite)
    return quantities, dict(zip(SUBSETS, masks, strict=True))


def group_by_root(values, mask):
    roots = len(mask) // ev.REPEATS
    grouped = values.reshape(len(values), roots, ev.REPEATS, *values.shape[2:])
    selected = mask.reshape(roots, ev.REPEATS)
    valid = selected.any(axis=1)
    local = selected[valid]
    means = np.einsum("brn...,rn->br...", grouped[:, valid], local)
    denominator_shape = (1, len(local)) + (1,) * (means.ndim - 2)
    return means / local.sum(1).reshape(denominator_shape)


def estimates(matrix):
    point = matrix.mean(1)
    bounds = inference.CrossedBootstrap(*matrix.shape[:2]).intervals(matrix.astype(np.float32))
    return point, bounds


def summary(point, bounds):
    return dict(mean=float(point.mean()), ci95=bounds.tolist(), by_birth=point.tolist(),
                birth_range=[float(point.min()), float(point.max())])


def summarize_arm(values, mask, kind):
    modes = ev.active_modes(kind)
    means = group_by_root(values, mask)
    contrasts = means[:, :, :1] - means[:, :, 1:]
    point, bounds = estimates(np.concatenate((means, contrasts), axis=2))
    actual_counts = values[:, mask].sum(1)
    rates = {}
    count_columns = {"paid_positive", "greedy_paid_positive", "seen_sign", "unseen_sign", "forecast_order"}
    for i, mode in enumerate(modes):
        rates[mode] = {}
        for j, metric in enumerate(METRICS):
            record = summary(point[:, i, j], bounds[:, i, j])
            if metric in count_columns:
                record.update(points=float(actual_counts[:, i, j].sum()), denominator=int(mask.sum() * len(values)),
                              points_by_birth=actual_counts[:, i, j].tolist(), denominator_by_birth=int(mask.sum()))
            rates[mode][metric] = record
    deltas = {mode: {metric: summary(point[:, len(modes) + i, j], bounds[:, len(modes) + i, j])
                     for j, metric in enumerate(METRICS)} for i, mode in enumerate(modes[1:])}
    return dict(roots=means.shape[1], rows_per_birth=int(mask.sum()), rates=rates, natural_minus=deltas)


def summarize_between(arms, mask):
    keys = tuple(("whole", kind) for kind in ev.fit.KINDS if kind != "whole")
    keys += (("bridge", "sham"), ("bridge", "cut_out"), ("bridge", "cut_back"))
    ids = [METRICS.index(name) for name in BETWEEN_METRICS]
    natural = {kind: group_by_root(values[:, :, 0, ids], mask) for kind, values in arms.items()}
    bundles = np.stack([natural[a] - natural[b] for a, b in keys], axis=2)
    point, bounds = estimates(bundles)
    return {f"{a}_minus_{b}": {metric: summary(point[:, i, j], bounds[:, i, j])
                                 for j, metric in enumerate(BETWEEN_METRICS)} for i, (a, b) in enumerate(keys)}


def passes(result, positive=True, reference=0.):
    if positive:
        return result["ci95"][0] > reference and min(result["by_birth"]) > reference
    return result["ci95"][1] < reference and max(result["by_birth"]) < reference


def declarations(arms, paired):
    calls = {}
    for kind in ev.INTERVENTION_KINDS:
        all_rows = arms[kind]["all"]["rates"]["natural"]
        hard = arms[kind]["close_opposite"]
        contrast = hard["natural_minus"]
        hard_between = paired["close_opposite"]
        flags = dict(unseen_mse_better_than_zero=passes(all_rows["unseen_squared_error"], positive=False, reference=1.),
                     unseen_mse_better_than_sham=passes(paired["all"][f"{kind}_minus_sham"]["unseen_squared_error"],
                                                      positive=False),
                     close_forecast_order_above_chance=passes(hard["rates"]["natural"]["forecast_order"], reference=0.5),
                     own_identical=True)
        flags.update({f"forecast_rank_above_training_{side}":
                      passes(hard_between[f"{kind}_minus_{side}"]["forecast_order"]) for side in ("cut_out", "cut_back")})
        for mode in CAUSAL_MODES:
            flags[f"actual_choice_benefits_vs_{mode}"] = passes(contrast[mode]["paid_positive"])
            if mode.startswith(("cut", "donor")):
                flags[f"forecast_rank_benefits_vs_{mode}"] = passes(contrast[mode]["forecast_order"])
            if mode.startswith("donor"):
                flags[f"forecast_margin_shifts_to_{mode}"] = passes(contrast[mode]["signed_forecast_margin"])
        flags["native_forecast_swap_changes_motor_in_target_direction"] = passes(contrast["forecast_swap"]["signed_motor_logit"])
        flags["learned_forecast"] = flags["unseen_mse_better_than_zero"] and flags["unseen_mse_better_than_sham"]
        flags["both_directions_for_prediction"] = all(value for name, value in flags.items()
                                                     if name.startswith(("forecast_rank_", "forecast_margin_")))
        flags["both_directions_for_real_choice"] = all(flags[f"actual_choice_benefits_vs_{mode}"] for mode in CAUSAL_MODES
                                                       if mode != "forecast_swap")
        calls[kind] = flags
    return calls


def report_world(split):
    scene, angles, donor, mirror, geometry = ev.heldout(ev.WORLDS[split], ev.ROOTS_PER_ANGLE)
    raw_paths = {}
    models = {}
    all_masks = None
    for kind in ev.fit.KINDS:
        collected = []
        for birth in ev.fit.BIRTHS:
            path = ev.artifact(kind, birth, split)
            checkpoint = ev.fit.digest(ev.fit.destination(kind, birth))
            with np.load(path) as raw:
                audit(raw, scene, geometry, angles, donor, mirror, kind, birth, split, checkpoint)
                scores, masks = scores_and_masks(raw)
                collected.append(scores)
            raw_paths[path.relative_to(ev.ROOT).as_posix()] = ev.digest(path)
            if all_masks is None:
                all_masks = masks
            else:
                for name in SUBSETS:
                    np.testing.assert_array_equal(masks[name], all_masks[name])
        models[kind] = np.stack(collected)
    arms = {kind: {name: summarize_arm(values, mask, kind) for name, mask in all_masks.items()}
            for kind, values in models.items()}
    paired = {name: summarize_between(models, mask) for name, mask in all_masks.items()}
    return dict(arms=arms, paired_training=paired, gates=declarations(arms, paired), raw_sha256=raw_paths)


def publish():
    torch.set_num_threads(1)
    geometries = np.concatenate([ev.heldout(seed, ev.ROOTS_PER_ANGLE)[0].vectors[::ev.REPEATS].numpy()
                                 for seed in ev.WORLDS.values()]).reshape(-1, 6)
    if len(np.unique(geometries, axis=0)) != len(geometries):
        raise AssertionError("Distinct held-out geometry roots repeat")
    reports = {split: report_world(split) for split in ev.WORLDS}
    double_pass = {kind: {name: all(report["gates"][kind][name] for report in reports.values())
                          for name in reports["private_a"]["gates"][kind]} for kind in ev.INTERVENTION_KINDS}
    results = dict(**ev.sources(), report_source=ev.digest(__file__), seeds=ev.WORLDS, births=ev.fit.BIRTHS,
                   roots_per_angle=ev.ROOTS_PER_ANGLE, repeats=ev.REPEATS, crossed_bootstrap_draws=inference.N_BOOT,
                   checkpoint_sha256={f"{kind}-{birth}": ev.digest(ev.fit.destination(kind, birth))
                                      for kind, birth in product(ev.fit.KINDS, ev.fit.BIRTHS)},
                   gates_passed_in_both_worlds=double_pass, worlds=reports)
    ev.storage._write_json(ev.EVIDENCE / "eval-results.json", results)
    print(double_pass, flush=True)


if __name__ == "__main__":
    publish()
