"""Verify archived actions, writes and support, then score registered paired contrasts."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import coupled_birth_analysis as prior, joint_live_model as joint
from . import prospective_edge_study as study, prospective_edges as model, ring_logic_study as io


COMPARATORS = ("instant", "shuffled")
RECEIPT_FIELDS = ("physical", "edits", "event_records", "weight_step", "input_step",
                  "edge_count", "recurrent_write", "birth_checks")


def paired(diff, boot_groups, boot_roots, comparisons):
    by_group_root = diff.mean(axis=1)
    samples = ((boot_groups @ by_group_root) * boot_roots).sum(axis=1)
    tail = .025 / comparisons
    return dict(delta=float(diff.mean()), interval_95=np.quantile(samples, (tail, 1 - tail)).tolist(),
                delta_by_rule=diff.mean(axis=(0, 2)).tolist())


def check_support_and_writes(data, arm):
    edge_count = data[f"{arm}/edge_count"]
    growth, retirement = np.moveaxis(data[f"{arm}/edits"] >= 0, -1, 0)
    observed = data[f"{arm}/physical"][..., 2] != 0
    events = data[f"{arm}/event_records"]
    serial = np.cumsum(observed, axis=-1)
    np.testing.assert_array_equal(edge_count, 64 + np.cumsum(growth.astype(int) - retirement, axis=-1))
    assert not np.any(growth & (serial % 9 != 0))
    assert not np.any(retirement & (serial % 6 != 0))
    # Independently verify reachability and inactive coefficients after each edit.
    edge_range = prior.replay_graphs(data, arm)
    writes = data[f"{arm}/recurrent_write"]
    np.testing.assert_array_equal(writes[~observed], 0)
    edges_before = np.concatenate((np.full_like(edge_count[..., :1], 64), edge_count[..., :-1]), axis=-1)
    np.testing.assert_array_equal(writes[..., 0].sum(axis=-1), (edges_before + growth) * observed)
    np.testing.assert_array_equal(writes[..., 2, 0], growth)
    assert np.all(writes[..., 1] <= writes[..., 0])
    result = prior.observe_births(data, arm)
    result.update(
        graph_edge_range_including_edit_intermediate=edge_range,
        exit_opportunities=int((observed & (serial % 6 == 0)).sum()),
        first_weight_step_changed_counts={key: int(writes[..., i, 1].sum()) for i, key in enumerate(model.WRITE_GROUPS)},
        weight_update_opportunities={key: int(writes[..., i, 0].sum()) for i, key in enumerate(model.WRITE_GROUPS)},
        weight_total_absolute_step={key: float(writes[..., i, 2].sum(dtype=np.float64))
                                    for i, key in enumerate(model.WRITE_GROUPS)},
        other_weights_receipts_with_update={key: int((data[f"{arm}/weight_step"][..., i] > 0).sum())
                                            for i, key in enumerate(joint.WEIGHT_NAMES) if key != "raw"},
    )
    ages = events[..., 1, joint.BIRTH_FIELDS.index("later_receipts")][retirement & (events[..., 1, 0] >= 0)]
    result["retired_newborn_age_quantiles"] = np.quantile(ages, (0, .5, 1)).tolist() if len(ages) else []
    return result


def centered_correlation(groups, score, target):
    count = np.bincount(groups)
    left = score - (np.bincount(groups, weights=score) / np.maximum(count, 1))[groups]
    right = target - (np.bincount(groups, weights=target) / np.maximum(count, 1))[groups]
    scale = np.sqrt(np.bincount(groups, weights=left * left) * np.bincount(groups, weights=right * right))
    usable = (count >= 3) & (scale > 0)
    products = np.bincount(groups, weights=left * right)
    return dict(slots=int(usable.sum()), correlation=float(np.mean(products[usable] / scale[usable])))


def audit(data, arm):
    lanes, trials, arcs, births = data[f"{arm}/audit_index"].T
    vals = data[f"{arm}/audit_value"]
    samples = data[f"{arm}/audit_later_samples"]
    physical = data[f"{arm}/physical"].reshape(-1, study.shared.TRIALS, 3)
    observed = physical[..., 2] != 0
    receipts = observed.cumsum(axis=1)
    clock = receipts[lanes, trials]
    assert np.all(observed[lanes, trials] & (clock % 6 == 0))
    assert np.all(samples >= 2)
    assert np.all(clock - births >= samples + 1)
    assert np.isfinite(vals).all()
    columns = {key: i for i, key in enumerate(model.AUDIT_VALUE_FIELDS)}
    later = vals[:, columns["later_score"]]
    policy = vals[:, columns["policy_score"]]
    gradient = vals[:, columns["past_instant_score"]]
    exact = vals[:, columns["heldout_effect"]]
    exited = data[f"{arm}/edits"].reshape(-1, study.shared.TRIALS, 2)[lanes, trials, 1] == arcs
    assert int(exited.sum()) == int((data[f"{arm}/edits"][..., 1] >= 0).sum())
    points = {}
    for cohort, mask in (("all", np.ones(len(births), dtype=bool)), ("initial", births == 0), ("grown", births > 0)):
        score_group = lanes[mask].astype(np.int32) * study.shared.TRIALS + trials[mask]
        points[cohort] = {
            key: centered_correlation(score_group, score[mask], exact[mask])
            for key, score in (("later", later), ("policy", policy), ("instant_prior", gradient))
        }
        selected = exited & mask
        points[cohort].update(
            candidates=int(mask.sum()), selected=int(selected.sum()),
            selection_help_rate=float((exact[selected] > 0).mean()) if selected.any() else None,
            selection_mean_heldout_effect=float(exact[selected].mean()) if selected.any() else None,
        )
    return points


def first_divergence(data, compare, bootstrap):
    policy = "later"
    a, b = (data[f"{arm}/edits"] for arm in (policy, compare))
    disagreement = a[..., 1] != b[..., 1]
    any_difference = disagreement.any(axis=-1)
    first = np.where(any_difference, disagreement.argmax(axis=-1), study.shared.TRIALS - 1)
    before = np.arange(study.shared.TRIALS) < first[..., None]
    for key in RECEIPT_FIELDS:
        np.testing.assert_array_equal(data[f"{policy}/{key}"][before], data[f"{compare}/{key}"][before])
    at = (*np.indices(first.shape), first)
    common_growth = a[at][..., 0]
    np.testing.assert_array_equal(common_growth, b[at][..., 0])
    for key in ("physical", "recurrent_write", "weight_step", "birth_checks"):
        np.testing.assert_array_equal(data[f"{policy}/{key}"][at], data[f"{compare}/{key}"][at])
    usable = any_difference & (first + 1 < study.shared.TRIALS)
    following = (*np.indices(first.shape), np.minimum(first + 1, study.shared.TRIALS - 1))
    observed_a, observed_b = (data[f"{arm}/physical"][following] for arm in (policy, compare))
    impact = np.where(usable, (observed_a[..., 2] > 0).astype(float) - (observed_b[..., 2] > 0), 0.)
    changed = usable & (observed_a[..., 0] != observed_b[..., 0])
    return dict(
        lives=int(first.size), diverged=int(any_difference.sum()), usable=int(usable.sum()),
        first_trial_range=[int(first[usable].min()), int(first[usable].max())] if usable.any() else [],
        action_side_changed=int(changed.sum()), later_correct_only=int((impact > 0).sum()),
        comparator_correct_only=int((impact < 0).sum()), paired_next_request=paired(impact, *bootstrap, 6),
    )


def verify_common_history(a, b, cutoff):
    for key in ("pairs", "flip_trial", "initial_mask"):
        np.testing.assert_array_equal(a[key], b[key])
    for root, limit in enumerate(cutoff):
        for arm in study.ARMS:
            for field in RECEIPT_FIELDS:
                key = f"{arm}/{field}"
                np.testing.assert_array_equal(a[key][:, :, root, :limit], b[key][:, :, root, :limit])


def report(out):
    registration = study.require(out)
    rng = np.random.default_rng(registration["bootstrap_seed"])
    group_count = len(registration["groups"])
    root_count = len(registration["roots"])
    bootstrap = tuple(
        rng.multinomial(n, np.full(n, 1. / n), size=20_000) / n for n in (group_count, root_count)
    )
    results = {}
    with (
        np.load(out / "stable.npz", allow_pickle=False) as stable,
        np.load(out / "reversal.npz", allow_pickle=False) as reversal,
        np.load(out / "return.npz", allow_pickle=False) as returns,
    ):
        verify_common_history(stable, reversal, reversal["flip_trial"])
        verify_common_history(reversal, returns, returns["flip_trial"] + 40)
        for case, archive in zip(study.CONDITIONS, (stable, reversal, returns), strict=True):
            arms = {}
            physical = {}
            for arm in study.ARMS:
                actions = archive[f"{arm}/physical"]
                physical[arm] = study.shared.windows(actions, archive["flip_trial"])
                arms[arm] = dict(
                    accuracy={key: float(value.mean()) for key, value in physical[arm].items()},
                    last20_accuracy_by_rule=physical[arm]["last20"].mean(axis=(0, 2)).tolist(),
                    full_life_accuracy=float((actions[..., 2] > 0).mean()),
                    structure_and_weights=check_support_and_writes(archive, arm),
                    heldout_receipts=audit(archive, arm),
                )
            results[case] = dict(
                arms=arms,
                last20_contrasts={
                    compare: paired(physical["later"]["last20"] - physical[compare]["last20"], *bootstrap, 6)
                    for compare in COMPARATORS
                },
                first_divergence={compare: first_divergence(archive, compare, bootstrap) for compare in COMPARATORS},
            )
            print("verified", case, flush=True)
    io.write(out / "summary.json", dict(
        registration_sha256=io.sha(out / "registered.json"),
        raw_sha256={case: io.sha(out / f"{case}.npz") for case in study.CONDITIONS},
        shared_physical_prefix_verified=True, all_graph_edits_replayed=True, results=results,
    ))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    args = parser.parse_args()
    report(args.out)


if __name__ == "__main__":
    main()
