"""Verify and summarize the signed joint-birth experiment on physical requests."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import coupled_birth as model, coupled_birth_study as study
from . import ring_logic_study as io
from .joint_live_model import BIRTH_FIELDS


def replay_graphs(values, arm):
    initial = values["initial_mask"]
    edits = values[f"{arm}/edits"]
    final = values[f"{arm}/final_mask"]
    raw = values[f"{arm}/final_raw"]
    if np.any(raw[~final] != 0):
        raise RuntimeError("Inactive weights must be zero")
    receipts = values[f"{arm}/physical"][..., 2] != 0
    if np.any((edits >= 0) & ~receipts[..., None]) or np.any(values[f"{arm}/weight_step"][~receipts] != 0):
        raise RuntimeError("Found a write without a signed receipt")
    groups, rules, roots = final.shape[:3]
    minimum, maximum = model.WIDTH ** 2, 0
    for row in range(groups * rules):
        for root in range(roots):
            group, rule = divmod(row, rules)
            index = group, rule, root
            support = initial[row].copy()
            count = int(support.sum())
            minimum, maximum = min(minimum, count), max(maximum, count)
            for add, remove in edits[index]:
                if add >= 0:
                    target, source = divmod(int(add), model.WIDTH)
                    if target == source or support[target, source]:
                        raise RuntimeError("Invalid dormant position")
                    support[target, source] = True
                    count += 1
                    maximum = max(maximum, count)
                if remove >= 0:
                    if not support.flat[remove]:
                        raise RuntimeError("Retired an absent edge")
                    support.flat[remove] = False
                    count -= 1
                    minimum = min(minimum, count)
                    if not model.strongly_connected(model.WIDTH, zip(*np.nonzero(support.T), strict=True)):
                        raise RuntimeError("A retirement broke strong connectivity")
            np.testing.assert_array_equal(support, final[index])
    return [minimum, maximum]


def contrast(difference, bg, br):
    pooled = difference.mean(axis=1)
    distribution = ((bg @ pooled) * br).sum(axis=1)
    return dict(delta=float(difference.mean()), simultaneous_95pct=np.quantile(distribution, (.0125, .9875)).tolist(),
                delta_by_group=pooled.mean(axis=1).tolist())


def observe_births(values, arm):
    grown, retired = np.moveaxis(values[f"{arm}/edits"] >= 0, -1, 0)
    fields = {name: i for i, name in enumerate(BIRTH_FIELDS)}
    events = values[f"{arm}/event_records"]
    removed_records = events[..., 1, :][retired]
    exited_newborn = removed_records[removed_records[:, fields["trial"]] >= 0]
    terminal = values[f"{arm}/final_newborn_records"]
    terminal = terminal[terminal[..., fields["trial"]] >= 0]
    history = np.concatenate((terminal, exited_newborn))
    if len(history) != grown.sum() or (exited_newborn[:, fields["later_receipts"]] < 2).any():
        raise RuntimeError("Newborn history is incomplete or an edit occurred before age eligibility")
    can_change = history[:, fields["later_receipts"]] > 0
    went_twice = history[:, fields["later_receipts"]] >= 2
    changed = history[:, fields["changed_receipts"]]
    audit = {name: i for i, name in enumerate(model.AUDIT_FIELDS)}
    checks = values[f"{arm}/birth_checks"]
    np.testing.assert_array_equal(np.isfinite(checks).all(axis=-1), grown)
    checks = checks[grown]
    np.testing.assert_array_equal(checks[:, audit["birth_weight"]], events[..., 0, fields["initial_raw"]][grown])
    exact = checks[:, audit["exact_loss_reduction"]]
    predicted = checks[:, audit["predicted_loss_reduction"]]
    if arm == "signed" and not (predicted > 0).all():
        raise RuntimeError("The signed birth did not follow the receipt gradient")
    counts = values[f"{arm}/final_mask"].sum(axis=(-1, -2))
    return dict(
        grows=int(grown.sum()), retires=int(retired.sum()), retired_birth_arcs=int(len(removed_records) - len(exited_newborn)),
        retired_newborns=int(len(exited_newborn)), surviving_newborns=int(len(terminal)),
        final_edges=[int(counts.min()), float(counts.mean()), int(counts.max())],
        median_abs_birth_weight=float(np.median(np.abs(history[:, fields["initial_raw"]]))),
        median_abs_end_weight=float(np.median(np.abs(history[:, fields["end_raw"]]))),
        median_total_weight_change_after_birth=float(np.median(history[:, fields["path_raw"]])),
        newborns_with_followup=int(can_change.sum()), fraction_changed=float(np.mean(changed[can_change] > 0)),
        newborns_with_two_followups=int(went_twice.sum()), fraction_changed_twice=float(np.mean(changed[went_twice] >= 2)),
        observed_loss_decline=float(np.mean(exact > 0)), observed_loss_increase=float(np.mean(exact < 0)),
        mean_actual_surrogate_loss_reduction=float(exact.mean()), mean_predicted_surrogate_loss_reduction=float(predicted.mean()))


def first_birth(values, bg, br):
    signed, control = study.ARMS
    physical, grows, audits = ({arm: values[f"{arm}/{key}"] for arm in study.ARMS}
                               for key in ("physical", "edits", "birth_checks"))
    first = {arm: (grows[arm][..., 0] >= 0).argmax(axis=-1) for arm in study.ARMS}
    trial = first[signed]
    np.testing.assert_array_equal(trial, first[control])
    if np.any((trial <= 0) | (trial + 1 >= study.shared.TRIALS)):
        raise RuntimeError("A paired life has no usable first birth")
    pre_birth = np.arange(study.shared.TRIALS) <= trial[..., None]
    np.testing.assert_array_equal(physical[signed][pre_birth], physical[control][pre_birth])
    np.testing.assert_array_equal(grows[signed][pre_birth], grows[control][pre_birth])
    np.testing.assert_array_equal(values[f"{signed}/weight_step"][pre_birth], values[f"{control}/weight_step"][pre_birth])
    at = (*np.indices(trial.shape), trial)
    np.testing.assert_array_equal(grows[signed][at], grows[control][at])
    birth = {arm: audits[arm][at] for arm in study.ARMS}
    i = model.AUDIT_FIELDS.index("birth_weight")
    exact = model.AUDIT_FIELDS.index("exact_loss_reduction")
    np.testing.assert_array_equal(birth[signed][..., 0], birth[control][..., 0])
    np.testing.assert_array_equal(np.abs(birth[signed][..., i]), np.abs(birth[control][..., i]))
    disagreed = np.signbit(birth[signed][..., i]) != np.signbit(birth[control][..., i])
    after = (*np.indices(trial.shape), trial + 1)
    following = {arm: physical[arm][after] for arm in study.ARMS}
    np.testing.assert_array_equal(following[signed][~disagreed], following[control][~disagreed])
    changed = following[signed][..., 0] != following[control][..., 0]
    diff = (following[signed][..., 2] > 0).astype(float) - (following[control][..., 2] > 0)
    return dict(
        counted_lives=int(trial.size), first_trial_index_range=[int(trial.min()), int(trial.max())],
        number_of_opposite_birth_signs=int(disagreed.sum()), next_request_contact_side_changes=int(changed.sum()),
        next_request_contact_side_changes_in_opposite_sign_lives=int((changed & disagreed).sum()),
        next_request_accuracy={arm: float(np.mean(following[arm][..., 2] > 0)) for arm in study.ARMS},
        next_request_contrast=contrast(diff, bg, br),
        same_receipt_mean_exact_loss_reduction={arm: float(birth[arm][..., exact].mean()) for arm in study.ARMS})


def verify_pre_change(stable, reversal):
    for key in ("pairs", "flip_trial", "initial_mask"):
        np.testing.assert_array_equal(stable[key], reversal[key])
    for index, flip in enumerate(stable["flip_trial"]):
        for arm in study.ARMS:
            for key in ("physical", "edits", "event_records", "birth_checks", "weight_step"):
                name = f"{arm}/{key}"
                np.testing.assert_array_equal(stable[name][:, :, index, :flip], reversal[name][:, :, index, :flip])


def report(out):
    recipe = study.sealed_recipe(out)
    heads = io.read(out / "confirmed-heads.json")
    if heads["recipe_sha256"] != io.sha(out / "recipe.json"):
        raise RuntimeError("Confirmation differs from sealed recipe")
    old_heads = {"dev": io.read(out / "development.json")["checkpoints"], "heldout": heads["checkpoints"]}
    verified = study.prior.verify_global_heads(old_heads)
    registration = io.read(out / "registered.json")
    rng = np.random.default_rng(registration["bootstrap_seed"])
    groups = len(study.shared.TEST_GROUPS)
    roots = len(study.TEST_ROOTS)
    bg = rng.multinomial(groups, np.full(groups, 1. / groups), size=20_000) / groups
    br = rng.multinomial(roots, np.full(roots, 1. / roots), size=20_000) / roots
    results = {}
    signed, control = study.ARMS
    with np.load(out / "stable.npz", allow_pickle=False) as stable, np.load(out / "reversal.npz", allow_pickle=False) as reversal:
        verify_pre_change(stable, reversal)
        for case, data in (("stable", stable), ("reversal", reversal)):
            accuracy = {arm: study.prior.windows(data[f"{arm}/physical"], data["flip_trial"]) for arm in study.ARMS}
            arms = {}
            for arm, windows in accuracy.items():
                arms[arm] = dict(accuracy={name: float(value.mean()) for name, value in windows.items()},
                                 timeout=float((data[f"{arm}/physical"][..., 0] < 0).mean()),
                                 lifetime_edge_range=replay_graphs(data, arm), **observe_births(data, arm))
            results[case] = dict(arms=arms, last20_contrast=contrast(accuracy[signed]["last20"] - accuracy[control]["last20"], bg, br),
                                 first_birth=first_birth(data, bg, br))
    io.write(out / "summary.json", dict(results=results, config={name: recipe[name] for name in ("rate", "birth_scale")},
                                       validation=dict(global_birth_manifest_verified=verified, pre_change_histories_equal=True,
                                                       structural_replay=True, paired_first_birth_matches=True),
                                       scores_sha256={case: io.sha(out / f"{case}.npz") for case in study.CONDITIONS}))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    report(parser.parse_args().out)


if __name__ == "__main__":
    main()
