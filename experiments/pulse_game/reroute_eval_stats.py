"""Paired statistics for frozen reroute action and forced-branch predictions."""

import numpy as np


BOOTSTRAP_DRAWS = 8192
RETENTION_MARGIN = -0.05


def measurement(values, mask, roots):
    """Average within root, keeping the eligible case count and total."""
    eligible = np.bincount(roots[mask])
    if len(eligible) != len(np.unique(roots)) or np.any(eligible != eligible[0]):
        raise ValueError("Each root must include the same nonzero denominator")
    selected = values[:, mask]
    shape = (len(values), len(eligible), eligible[0])
    means = selected.reshape(shape).mean(axis=2, dtype=np.float64)
    return means, selected.sum(axis=1), int(mask.sum())


def statistics(data, differences, seed):
    """Cross the same birth and root bootstrap draws in all arms and endpoints."""
    samples = {}
    pairs = {}
    for name, metrics in data.items():
        for key, (means, _, _) in metrics.items():
            samples[name, key] = means
    for one, two in differences:
        name = f"{one} minus {two}"
        pairs[name] = sorted(data[one].keys() & data[two].keys())
        for key in pairs[name]:
            a, _, n = data[one][key]
            b, _, m = data[two][key]
            if n != m:
                raise ValueError(f"Different case count for paired endpoint {name}/{key}")
            samples[name, key] = a - b

    births, roots = next(iter(samples.values())).shape
    if any(values.shape != (births, roots) for values in samples.values()):
        raise ValueError("All metrics must preserve every birth and world root")
    rng = np.random.default_rng([seed, 78491])
    birth = rng.multinomial(births, np.full(births, 1 / births), BOOTSTRAP_DRAWS) / births
    root = rng.multinomial(roots, np.full(roots, 1 / roots), BOOTSTRAP_DRAWS) / roots
    # A separate root draw is shared across births within each replication.
    weights = (birth[:, :, None] * root[:, None, :]).reshape(BOOTSTRAP_DRAWS, -1)
    rates = {name: {} for name in data}
    contrasts = {name: {} for name in pairs}
    keys = list(samples)
    for offset in range(0, len(keys), 32):
        chunk = keys[offset : offset + 32]
        values = np.stack([samples[key] for key in chunk], axis=-1)
        draws = weights @ values.reshape(births * roots, -1)
        intervals = np.quantile(draws, [0.025, 0.975], axis=0).T
        per_birth = values.mean(axis=1).T
        for (name, key), by_birth, ci in zip(chunk, per_birth, intervals):
            report = dict(mean=float(by_birth.mean()), ci95=ci.tolist(), by_birth=by_birth.tolist())
            if name in data:
                _, total, count = data[name][key]
                report.update(total=float(total.sum()), denominator=count * births,
                              total_by_birth=total.tolist(), denominator_by_birth=count)
                rates[name][key] = report
            else:
                contrasts[name][key] = report
    return dict(rates=rates, contrasts=contrasts)


def evaluate_gates(worlds, kinds):
    """Apply the predeclared direction and retention rules separately in both worlds."""
    def contrast(one, two, endpoint, margin):
        return {seed: world["contrasts"][f"{one} minus {two}"][endpoint]["ci95"][0] > margin
                for seed, world in worlds.items()}

    teaching_gain = contrast("next_room/natural", "terminal/natural", "flip_optimal", 0)
    teaching_retention = contrast("next_room/natural", "terminal/natural", "stay_optimal", RETENTION_MARGIN)
    teaching = dict(flip_gain=teaching_gain, stay_noninferior=teaching_retention,
                    pass_both=all(teaching_gain.values()) and all(teaching_retention.values()))
    trial = {}
    prediction = {}
    for kind in kinds:
        native = f"{kind}/natural"
        blank = f"{kind}/blank"
        gain = contrast(native, blank, "flip_optimal", 0)
        retention = contrast(native, blank, "stay_optimal", RETENTION_MARGIN)
        trial[kind] = dict(flip_gain=gain, stay_noninferior=retention,
                           pass_both=all(gain.values()) and all(retention.values()))
        chance = {seed: world["rates"][native]["forecast_rank"]["ci95"][0] > 0.5
                  for seed, world in worlds.items()}
        blank_gain = contrast(native, blank, "forecast_rank", 0)
        prediction[kind] = dict(rank_above_chance=chance, rank_above_blank=blank_gain,
                                pass_both=all(chance.values()) and all(blank_gain.values()))
    return dict(retention_margin=RETENTION_MARGIN, teaching_action=teaching, actual_trial_gain=trial,
                conditional_prediction=prediction)
