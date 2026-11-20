"""Summarize sealed review replays with paired birth/geometry intervals."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as io, noon_value as protocol
from .interleave_interference import ROOT


@dataclass(frozen=True)
class Observation:
    strength: np.ndarray
    visible: np.ndarray
    priority: np.ndarray
    permutation: np.ndarray
    recency: np.ndarray


def broadcast(public, private):
    return np.broadcast_to(public[:, None], private.shape)


def pick(score, allowed, priority):
    maximum = np.max(np.where(allowed, score, -np.inf), axis=-1, keepdims=True)
    eligible = allowed & (score == maximum)
    return np.argmax(np.where(eligible, priority, -np.inf), axis=-1)


def take(candidates, indices):
    return np.take_along_axis(candidates, indices[..., None, None], axis=-2)[..., 0, :]


def take_scalar(candidates, indices):
    return np.take_along_axis(candidates, indices[..., None], axis=-1)[..., 0]


def scrambled(score, visible, permutation):
    # Restrict permutation to visible items, keeping the original mask and all scores.
    slots = np.argsort(~visible, axis=-1, kind="stable")
    order = np.argsort(~np.take_along_axis(visible, permutation, -1), axis=-1, kind="stable")
    sources = np.take_along_axis(permutation, order, -1)
    result = score.copy()
    np.put_along_axis(result, slots, np.take_along_axis(score, sources, -1), -1)
    return result


def methods(observation, schedule, visible_only=False, bilateral_only=False):
    x, seen, priority, permutation, age = (observation.strength, observation.visible, observation.priority,
                                          observation.permutation, observation.recency)
    everyone = np.ones_like(seen)
    allowed = seen.copy() if visible_only else everyone.copy()
    if bilateral_only:
        allowed[..., -1] = False
    choices = {"uniform": pick(priority, allowed, priority), "oldest": pick(-age, allowed, priority),
               "newest": pick(age, allowed, priority)}
    if schedule == "blind":
        choices.update(external_weak=pick(-x, allowed, priority), external_strong=pick(x, allowed, priority),
                       external_scrambled=pick(-scrambled(x, everyone, permutation), allowed, priority))
    else:
        score = np.where(seen, x, np.nanmedian(x, axis=-1, keepdims=True))
        permuted = scrambled(score, seen & allowed, permutation)
        choices.update(weak=pick(-score, allowed, priority), strong=pick(score, allowed, priority),
                       scrambled=pick(-permuted, allowed, priority))
        if schedule == "half" and not visible_only:
            # Balance visibility of the chosen item exactly against the uniform ticket.
            uniform = choices["uniform"]
            can_use_signal = take_scalar(seen, uniform)
            for name, signal in (("weak", -score), ("strong", score), ("scrambled", -permuted)):
                choices[f"{name}_balanced"] = np.where(can_use_signal, pick(signal, seen, priority), uniform)
    return choices, allowed


def tau(left, right, eligible):
    i, j = np.triu_indices(left.shape[-1], 1)
    keep = eligible[..., i] & eligible[..., j]
    ordering = np.sign(left[..., i] - left[..., j]) * np.sign(right[..., i] - right[..., j])
    return np.where(keep, ordering, 0).sum(-1) / keep.sum(-1)


def other_effect(score):
    items = score.shape[-1]
    diag = np.diagonal(score, axis1=-2, axis2=-1)
    other_mean = (score.sum(-1) - diag) / (items - 1)
    other_min = np.where(np.eye(items, dtype=bool), np.inf, score).min(-1)
    return diag, other_mean, other_min


def limits(metrics, active, sham, actual, actual_sham, allowed, priority):
    threshold = protocol.THRESHOLD
    for statistic, column in (("joint", 0), ("mean", 1)):
        for objective, value in (("end", active[..., column]), ("write", (active - sham)[..., column])):
            uniform = np.where(allowed[:, :, None], value, 0).sum(-1) / allowed.sum(-1)[..., None]
            lo = pick(value[:, :, 0], allowed, priority)
            hi = pick(value[:, :, 1], allowed, priority)
            a = value[:, :, 0]
            b = value[:, :, 1]
            record = f"hindsight/{statistic}/{objective}"
            sample = actual[..., column] if objective == "end" else (actual - actual_sham)[..., column]
            sample_uniform = np.where(allowed, sample, 0).sum(-1) / allowed.sum(-1)
            sample_winner = pick(sample, allowed, priority)
            sample_fold_other = take_scalar(b, sample_winner)
            centered = value - uniform[..., None]
            same = (take_scalar(a, lo) + take_scalar(b, hi)) / 2 - uniform.mean(2)
            cross = (take_scalar(b, lo) + take_scalar(a, hi)) / 2 - uniform.mean(2)
            metrics.update({
                f"{record}/same_fold_advantage": same,
                f"{record}/cross_fold_advantage": cross,
                f"{record}/same_minus_cross": same - cross,
                f"{record}/single_realization_same_advantage": take_scalar(sample, sample_winner) - sample_uniform,
                f"{record}/single_realization_other_fold_advantage": sample_fold_other - uniform[:, :, 1],
                f"{record}/winner_agreement": lo == hi,
                f"{record}/rank_agreement": tau(a, b, allowed),
                f"{record}/candidate_beats_uniform_both_folds_1pp":
                    ((centered > threshold).all(2) & allowed).any(-1),
                f"{record}/candidate_choice_spread_both_folds_1pp": stable_spread(value, allowed),
            })
            if objective == "write":
                metrics[f"{record}/any_candidate_positive_both_folds_1pp"] = \
                    ((value > threshold).all(2) & allowed).any(-1)
                metrics[f"{record}/all_candidates_negative_both_folds_1pp"] = \
                    ((value < -threshold).all(2) | ~allowed).all(-1)


def stable_spread(value, allowed):
    proposed = value[..., :, None] - value[..., None, :]
    above = (proposed > protocol.THRESHOLD).all(2)
    return (above & allowed[..., :, None] & allowed[..., None, :]).any((-1, -2))


def visibility_metrics(raw, schedule, lower, joint, bilateral_only=False):
    prefix = "half_visible" if lower else "all_bilateral_only" if bilateral_only else schedule
    x = np.abs(raw[f"{schedule}/logit"])
    seen = raw[f"{schedule}/visible"]
    visits = np.zeros((len(x), x.shape[-1]), dtype=int)
    for slot, item in enumerate(raw["support"].T):
        visits[np.arange(len(x)), item] = slot
    observation = Observation(x, seen, broadcast(raw["priority"], x), broadcast(raw["scramble"], x),
                              broadcast(visits, x))
    choice, allowed = methods(observation, schedule, lower, bilateral_only)
    folded = raw[f"{schedule}/fold"]
    active, sham = folded[:, :, 0], folded[:, :, 1]
    expected, matched = active.mean(2), sham.mean(2)
    gain = expected - matched
    actual = raw[f"{schedule}/actual"][:, :, 0]
    sham_actual = raw[f"{schedule}/actual"][:, :, 1]
    idle_exp = raw[f"{schedule}/idle_expected"]
    idle_act = raw[f"{schedule}/idle_actual"]
    paid = raw[f"{schedule}/paid"]
    prior = raw[f"{schedule}/prior_expected"]
    summary = {"idle/expected_joint": idle_exp[..., 0], "idle/actual_joint": idle_act[..., 0],
               "idle/expected_mean": idle_exp[..., 1], "idle/actual_mean": idle_act[..., 1],
               "prior/expected_joint": prior[..., 0], "prior/expected_mean": prior[..., 1]}
    if joint:
        summary["idle/paid_both"] = raw[f"{schedule}/idle_paid"][..., 0]
    score = -x
    observed = (np.ones_like(seen) if schedule == "blind" else seen) & allowed
    actual_gain = actual - sham_actual
    own, other_mean, other_min = other_effect(gain[..., 2:])
    summary.update({
        "rank/weak_vs_prior_error": tau(score, 1 - prior[..., 2:], observed),
        "rank/weak_vs_mean_endpoint": tau(score, expected[..., 1], observed),
        "rank/weak_vs_joint_endpoint": tau(score, expected[..., 0], observed),
        "rank/weak_vs_own_write_gain": tau(score, own, observed),
        "rank/weak_vs_other_mean_write_gain": tau(score, other_mean, observed),
        "rank/weak_vs_mean_write_gain": tau(score, gain[..., 1], observed),
        "rank/weak_vs_joint_write_gain": tau(score, gain[..., 0], observed),
        "rank/own_vs_joint_write_gain": tau(own, gain[..., 0], allowed),
    })
    threshold = protocol.THRESHOLD
    own_fold, other_mean_fold, other_min_fold = other_effect((active - sham)[..., 2:])
    present = lambda value: np.where(allowed, value, 0).sum(-1) / allowed.sum(-1)
    repair = (own_fold > threshold).all(2)
    at_least_one_harm = (other_min_fold < -threshold).all(2)
    actual_own, _, actual_other = other_effect(actual_gain[..., 2:])
    summary.update({
        "interference/own_mean_gain": present(own),
        "interference/other_mean_gain": present(other_mean),
        "interference/worst_other_mean_gain": present(other_min),
        "interference/expected_own_repair_both_folds": present(repair),
        "interference/expected_own_repair_with_other_harm_both_folds": present(repair & at_least_one_harm),
        "interference/expected_own_repair_with_negative_others_mean_both_folds":
            present(repair & (other_mean_fold < -threshold).all(2)),
        "interference/actual_own_fix": present(actual_own > 0),
        "interference/actual_own_fix_with_other_damage": present((actual_own > 0) & (actual_other < 0)),
        "interference/actual_own_fix_any_candidate_with_other_damage":
            ((actual_own > 0) & (actual_other < 0) & allowed).any(-1),
        "interference/candidate_review_1_effect_on_0": gain[..., 1, 2],
        "interference/candidate_review_0_effect_on_1": gain[..., 0, 3],
        "interference/candidate_review_0_own": gain[..., 0, 2],
        "interference/candidate_review_1_own": gain[..., 1, 3],
        "interference/effective_topology_edit_frequency": present(raw[f"{schedule}/topology_rate"].mean(2)),
    })
    if joint:
        summary.update({"interference/context_a_review_effect_on_b": gain[..., 3, 6],
                        "interference/context_b_review_effect_on_a": gain[..., 4, 5],
                        "interference/context_a_own": gain[..., 3, 5],
                        "interference/context_b_own": gain[..., 4, 6]})
        if not bilateral_only:
            summary["interference/mark_review_own"] = gain[..., 5, 7]
    limits(summary, active, sham, actual, sham_actual, allowed, observation.priority)
    own_choice_a = pick(own_fold[:, :, 0], allowed, observation.priority)
    own_choice_b = pick(own_fold[:, :, 1], allowed, observation.priority)
    for name, value in (("own", own_fold), ("joint", active[..., 0] - sham[..., 0]),
                        ("mean", active[..., 1] - sham[..., 1])):
        chosen = (take_scalar(value[:, :, 0], own_choice_b) + take_scalar(value[:, :, 1], own_choice_a)) / 2
        summary[f"hindsight/own_only/cross_fold_{name}_write_advantage"] = chosen - present(value.mean(2))
    for label, chosen in choice.items():
        read = take(expected, chosen)
        control = take(matched, chosen)
        act = take(actual, chosen)
        actual_control = take(sham_actual, chosen)
        positive_review = take_scalar(raw[f"{schedule}/first_contact"][..., 1], chosen) > 0
        key = f"policy/{label}"
        summary.update({
            f"{key}/expected_joint": read[..., 0], f"{key}/actual_joint": act[..., 0],
            f"{key}/expected_mean": read[..., 1], f"{key}/actual_mean": act[..., 1],
            f"{key}/expected_joint_minus_sham": read[..., 0] - control[..., 0],
            f"{key}/expected_joint_minus_idle": read[..., 0] - idle_exp[..., 0],
            f"{key}/expected_mean_minus_sham": read[..., 1] - control[..., 1],
            f"{key}/expected_mean_minus_idle": read[..., 1] - idle_exp[..., 1],
            f"{key}/actual_joint_minus_sham": act[..., 0] - actual_control[..., 0],
            f"{key}/actual_joint_minus_idle": act[..., 0] - idle_act[..., 0],
            f"{key}/actual_mean_minus_sham": act[..., 1] - actual_control[..., 1],
            f"{key}/actual_mean_minus_idle": act[..., 1] - idle_act[..., 1],
            f"{key}/expected_own_write_gain": take_scalar(own, chosen),
            f"{key}/expected_other_mean_write_gain": take_scalar(other_mean, chosen),
            f"{key}/selected_unseen": ~take_scalar(seen, chosen),
            f"{key}/review_received_positive_taste": positive_review,
            f"{key}/review_plus_one_uniform_read_success_count": positive_review + read[..., 1],
            f"{key}/review_plus_all_independent_reads_success_count": positive_review + read[..., 2:].sum(-1),
        })
        for i in range(x.shape[-1]):
            summary[f"{key}/selected_{i}"] = chosen == i
        if joint:
            paid_act, paid_sham = take(paid[:, :, 0], chosen), take(paid[:, :, 1], chosen)
            summary[f"{key}/review_plus_two_actual_paid_success_count"] = positive_review + paid_act[..., 1:].sum(-1)
            for metric, i in (("both", 0), ("triangle", 1), ("context_or_mark", 2)):
                summary[f"{key}/paid_{metric}"] = paid_act[..., i]
                summary[f"{key}/paid_{metric}_minus_sham"] = paid_act[..., i].astype(float) - paid_sham[..., i]
                summary[f"{key}/paid_{metric}_minus_idle"] = paid_act[..., i].astype(float) - raw[f"{schedule}/idle_paid"][..., i]
    signals = ("external_weak", "external_strong", "external_scrambled") if schedule == "blind" else ("weak", "strong", "scrambled")
    weak, strong, shuffle = signals
    endpoints = ("actual_joint", "actual_mean", "expected_joint", "expected_mean", "expected_joint_minus_sham",
                 "expected_mean_minus_sham", "expected_own_write_gain", "expected_other_mean_write_gain",
                 "review_received_positive_taste", "review_plus_one_uniform_read_success_count",
                 "review_plus_all_independent_reads_success_count")
    if joint:
        endpoints += ("paid_both", "paid_triangle", "paid_context_or_mark", "review_plus_two_actual_paid_success_count")
    comparisons = [(weak, "uniform"), (strong, "uniform"), (weak, strong), (weak, shuffle),
                   (weak, "oldest"), (weak, "newest"), ("oldest", "uniform"), ("newest", "uniform")]
    if "weak_balanced" in choice:
        comparisons += [("weak_balanced", "uniform"), ("strong_balanced", "uniform"),
                        ("weak_balanced", "strong_balanced"), ("weak_balanced", "scrambled_balanced")]
    for lhs, rhs in comparisons:
        for endpoint in endpoints:
            a = summary[f"policy/{lhs}/{endpoint}"].astype(float)
            b = summary[f"policy/{rhs}/{endpoint}"]
            summary[f"contrast/{lhs}_minus_{rhs}/{endpoint}"] = a - b
    return prefix, summary


def natural_pairs(raw, joint):
    if joint:
        return {}
    axis = raw["axis"]
    unit = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    dots = np.einsum("wki,wji->wkj", unit, unit)
    near = np.abs(dots) * (1 - np.eye(unit.shape[1]))
    neighbor = near.argmax(-1)
    dot = np.take_along_axis(dots, neighbor[..., None], -1)[..., 0]
    angle = np.rad2deg(np.arccos(np.clip(np.abs(dot), 0, 1)))
    law = raw["law"]
    aligned_neighbor = np.take_along_axis(law, broadcast(neighbor, law), -1) * np.sign(dot[:, None, :])
    conflict = law != aligned_neighbor
    data = raw["all/fold"]
    gain = (data[:, :, 0] - data[:, :, 1]).mean(2)
    own, _, _ = other_effect(gain[..., 2:])
    near_gain = np.take_along_axis(gain[..., 2:], broadcast(neighbor, law)[..., None], -1)[..., 0]
    result = {}
    for start in (0, 30, 60):
        in_bin = ((angle >= start) & (angle < start + 30))[:, None, :]
        for label, flag in (("conflicting", conflict), ("aligned", ~conflict)):
            mask = in_bin & flag
            denominator = mask.mean((1, 2))
            for metric, scores in (("own", own), ("neighbor", near_gain), ("joint", gain[..., 0]),
                                   ("mean", gain[..., 1])):
                value = np.where(mask, scores, 0).mean((1, 2))
                result[f"natural_nearest/{start}_{start + 30}/{label}/{metric}"] = (value, denominator)
    return result


def collect(raw, assay):
    joint = assay == "joint6"
    law = raw["law"]
    masks = {"all_maps": np.full(law.shape[:2], True)}
    if joint:
        conflict = law[..., 3] != law[..., 4]
        hard = (law[..., :3] == law[..., :1]).all(-1)
        masks.update(context_conflict=conflict, context_agree=~conflict, hard_and_context_conflict=hard & conflict)
    elif assay.startswith("angle"):
        agree = law[..., 0] == law[..., 1]
        masks.update(agree=agree, conflict=~agree)
    summary = dict(visibility_metrics(raw, schedule, visible, joint)
                   for schedule, visible in (("all", False), ("blind", False), ("half", False), ("half", True)))
    if joint:
        label, bilateral_only = visibility_metrics(raw, "all", False, joint, bilateral_only=True)
        summary[label] = bilateral_only
    endpoints = ("actual_joint", "actual_mean", "expected_joint", "expected_mean")
    if joint:
        endpoints += ("paid_both", "paid_triangle", "paid_context_or_mark")
    preview = {}
    for mode in ("all", "half"):
        for field in endpoints:
            preview[f"{mode}_minus_blind/uniform/{field}"] = \
                summary[mode][f"policy/uniform/{field}"].astype(float) - summary["blind"][f"policy/uniform/{field}"]
            preview[f"{mode}_minus_blind/weak_vs_uniform/{field}"] = \
                summary[mode][f"contrast/weak_minus_uniform/{field}"] - summary["blind"][f"contrast/external_weak_minus_uniform/{field}"]
    for field in endpoints:
        preview[f"half_visible_minus_half/uniform/{field}"] = \
            summary["half_visible"][f"policy/uniform/{field}"].astype(float) - summary["half"][f"policy/uniform/{field}"]
    summary["schedule_contrasts"] = preview
    rows = {}
    for mask_name, mask in masks.items():
        denominator = mask.sum(-1)
        if (denominator == 0).any():
            raise ValueError(f"Missing map class {mask_name} within a geometry")
        for visibility, entries in summary.items():
            for name, value in entries.items():
                rows[f"{mask_name}/{visibility}/{name}"] = np.where(mask, value, 0).sum(-1) / denominator
    return rows, {name: int(mask.sum(-1)[0]) for name, mask in masks.items()}, natural_pairs(raw, joint)


class PairedCrossed:
    def __init__(self, births, worlds, seed, draws=4000):
        random = np.random.default_rng(seed)
        birth = random.multinomial(births, np.full(births, 1 / births), draws) / births
        world = random.multinomial(worlds, np.full(worlds, 1 / worlds), draws) / worlds
        self.weights = (birth[:, :, None] * world[:, None, :]).reshape(draws, births * worlds)
        self.shape = (births, worlds)

    def summary(self, values, denominator=None):
        names = list(values)
        matrix = np.stack([values[key] for key in names], -1)
        if matrix.shape[:2] != self.shape:
            raise ValueError("Wrong number of births or geometry clusters")
        flat = matrix.reshape(-1, len(names))
        estimates = self.weights @ flat
        counts = np.ones_like(matrix)
        if denominator is not None:
            counts = np.stack([denominator[key] for key in names], -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates /= self.weights @ counts.reshape(-1, len(names))
        with np.errstate(divide="ignore", invalid="ignore"):
            average = matrix.mean((0, 1)) / counts.mean((0, 1))
            by_birth = matrix.mean(1) / counts.mean(1)
        lower, upper = np.nanquantile(estimates, (.025, .975), axis=0)
        result = {}
        for i, name in enumerate(names):
            stats = {"mean": finite(average[i]), "ci95": [finite(lower[i]), finite(upper[i])],
                     "by_birth": [finite(x) for x in by_birth[:, i]]}
            if denominator is not None:
                stats["candidate_weight_fraction"] = float(counts[..., i].mean())
            result[name] = stats
        return result


def finite(value):
    return float(value) if np.isfinite(value) else None


def rows_for(split, cohort, body, assay, requested_births):
    first, seed = protocol.COHORTS[cohort]
    births = requested_births or list(range(first, first + 6))
    directory = protocol.DATA / split / cohort / body
    rows, pairs, files = [], [], []
    tape_hash = None
    implementation = None
    for birth in births:
        file = directory / f"{assay}-{birth}.npz"
        with np.load(file, allow_pickle=False) as raw:
            if str(raw["plan_sha256"]) != protocol.digest(protocol.PLAN):
                raise ValueError(f"Unsealed data from {file}")
            checkpoint, checkpoint_hash = str(raw["checkpoint"]), str(raw["source_sha256"])
            if protocol.digest(ROOT / checkpoint) != checkpoint_hash:
                raise ValueError(f"Checkpoint changed after replay {file}")
            implementation_id = str(raw["implementation"])
            self_hash = implementation_id.splitlines()[0].split()[1]
            if self_hash != protocol.digest(Path(protocol.__file__)):
                raise ValueError(f"Replay code changed after replay {file}")
            common = sha256()
            for field in protocol.Tape.__dataclass_fields__:
                common.update(raw[field].tobytes())
            if tape_hash is None:
                tape_hash = common.hexdigest()
                implementation = implementation_id
            if tape_hash != common.hexdigest() or implementation != implementation_id or int(raw["seed"]) != seed:
                raise ValueError(f"Replay missing its cohort-matched tape or implementation {file}")
            outcomes, mask_counts, nearby = collect(raw, assay)
            files.append({"path": file.relative_to(ROOT).as_posix(), "sha256": protocol.digest(file),
                          "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_hash})
        rows.append(outcomes)
        pairs.append(nearby)
    values = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    nearby = {name: tuple(np.stack([row[name][i] for row in pairs]) for i in (0, 1)) for name in pairs[0]}
    worlds = next(iter(values.values())).shape[1]
    return births, seed, worlds, mask_counts, values, nearby, files, tape_hash, implementation


def run(cohort, body, assays, smoke=False, births=None):
    split = "smoke" if smoke else "sealed"
    folder = protocol.DATA if smoke else protocol.EVIDENCE
    bank = {}
    source_hash = protocol.digest(Path(__file__))
    for assay in assays:
        population, seed, worlds, masks, rows, nearby, files, tape, implementation = rows_for(split, cohort, body, assay, births)
        crossed = PairedCrossed(len(population), worlds, seed + 3191)
        measures = crossed.summary(rows)
        if nearby:
            measures.update(crossed.summary({name: v[0] for name, v in nearby.items()},
                                            {name: v[1] for name, v in nearby.items()}))
        bank[assay] = rows
        result = dict(cohort=cohort, body=body, assay=assay, births=population, seed=seed, geometries_per_birth=worlds,
                      maps_per_geometry=masks, review_samples_per_fold=1 if smoke else protocol.SAMPLES_PER_FOLD,
                      bootstrap_draws=4000, files=files, plan_sha256=protocol.digest(protocol.PLAN),
                      report_sha256=source_hash, matched_tape_sha256=tape, evaluator_sources=implementation.splitlines(),
                      measures=measures)
        io._write_json(folder / f"value-{cohort}-{body}-{assay}.json", result)
        print(cohort, body, assay, len(measures), flush=True)
    if all(assay in bank for assay in ("angle15", "angle45", "angle75")):
        differences = {}
        paired = bank["angle15"]
        crossed = PairedCrossed(len(population), next(iter(paired.values())).shape[1], seed + 3191)
        for angle in ("angle45", "angle75"):
            differences[angle + "_minus_angle15"] = crossed.summary(
                {name: bank[angle][name] - value for name, value in paired.items()
                 if ("/all/" in name and ("/interference/" in name or "/rank/" in name or "/contrast/weak_minus_uniform/" in name))})
        io._write_json(folder / f"value-{cohort}-{body}-paired-angles.json",
                       dict(cohort=cohort, body=body, births=population, seed=seed,
                            plan_sha256=protocol.digest(protocol.PLAN), report_sha256=source_hash, differences=differences))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=protocol.COHORTS)
    parser.add_argument("--body", required=True, choices=("uniform", "mixed_repair"))
    parser.add_argument("--assay", choices=protocol.ASSAYS, nargs="+", required=True)
    parser.add_argument("--births", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.cohort, args.body, args.assay, args.smoke, args.births)
