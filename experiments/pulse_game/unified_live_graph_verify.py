"""Independently recompute key physical score and intervention checks from saved lives."""

import argparse
from pathlib import Path

import numpy as np

from .unified_live_graph_study import ARMS, MODES, TAIL, audit_edits, bootstrap, digest, read, write


def fetch(folder, mode, arm, hashes):
    path = folder / f"{mode}-{arm}.npz"
    assert digest(path) == hashes[mode][arm]
    with np.load(path) as arrays:
        return dict(arrays)


def check_primary(folder):
    summary = read(folder / "summary.json")
    hashes = summary["npz_sha256"]
    rows = {mode: {arm: fetch(folder, mode, arm, hashes) for arm in ARMS} for mode in MODES}
    pattern = rows["stable"]["joint"]
    trial = np.arange(pattern["correct"].shape[1])
    before_switch = trial[None, :] < pattern["change"][:, None]
    before_restore = trial[None, :] < pattern["recovery"][:, None]
    comparisons = {}
    totals = dict(lives=0, contacts=0, growth=0, retirement=0, newborn_written=0)
    for mode, arms in rows.items():
        for arm, raw in arms.items():
            for field in ("birth", "world", "initial_active", "initial_weight", "change", "recovery"):
                np.testing.assert_array_equal(raw[field], pattern[field])
            np.testing.assert_array_equal(raw["correct"], raw["contact_side"] == raw["good_side"])
            np.testing.assert_array_equal(raw["contact_side"] >= 0, raw["contact_tick"] >= 0)
            np.testing.assert_array_equal(raw["contact_side"] >= 0, raw["reward"] != 0)
            assert not raw["shown"][:, -TAIL:].any()
            assert not raw["writes"][raw["shown"] == 0].any()
            assert not (raw["edits"][raw["shown"] == 0] >= 0).any()
            if arm == "no_receipt":
                assert not raw["shown"].any()
            elif arm == "independent":
                np.testing.assert_array_equal(raw["shown"][:, :-TAIL] != 0, raw["reward"][:, :-TAIL] != 0)
                assert (raw["shown"][:, :-TAIL] != raw["reward"][:, :-TAIL]).any()
            else:
                np.testing.assert_array_equal(raw["shown"][:, :-TAIL], raw["reward"][:, :-TAIL])
            audit_edits(raw["initial_active"], raw["initial_weight"], raw["edits"], raw["final_active"], raw["final_weight"])
            if arm == "joint":
                totals["lives"] += len(raw["birth"])
                totals["contacts"] += int((raw["contact_side"] >= 0).sum())
                totals["growth"] += int((raw["edits"][:, :, 0] >= 0).sum())
                totals["retirement"] += int((raw["edits"][:, :, 1] >= 0).sum())
                writes = (raw["final_generation_writes"] > 0).sum() + (raw["writes"][:, :, 2] > 0).sum()
                totals["newborn_written"] += int(writes)
            for field in ("contact_side", "contact_tick", "edits", "writes"):
                np.testing.assert_array_equal(raw[field][before_switch], rows["stable"][arm][field][before_switch])
                if mode == "restore":
                    np.testing.assert_array_equal(raw[field][before_restore], rows["reverse"][arm][field][before_restore])
            if arm in ("no_receipt", "independent"):
                for field in ("contact_side", "contact_tick", "edits", "writes", "final_weight"):
                    np.testing.assert_array_equal(raw[field], rows["stable"][arm][field])
            if arm == "no_receipt":
                np.testing.assert_array_equal(raw["initial_weight"], raw["final_weight"])
        if mode in MODES[:2]:
            joint = arms["joint"]["correct"][:, -TAIL:].mean(axis=1)
            for arm in ("frozen", "independent", "weights"):
                pair = joint - arms[arm]["correct"][:, -TAIL:].mean(axis=1)
                name = f"{mode}_joint_minus_{arm}"
                interval = bootstrap(pair, alpha=.05 / 4)
                if arm != "weights":
                    np.testing.assert_array_equal(interval, summary["primary"][name])
                else:
                    comparisons[name] = interval
    for replay in summary["source_cuts"].values():
        for result, active_nodes in replay:
            assert active_nodes == pattern["initial_active"].shape[-1]
            assert result["motor"]["max_other"] > 0
            assert result["taste"]["max_receipt_one_hop"] > 0
            assert result["taste"]["max_receipt_one_hop_cut"] == 0
    return dict(joint_totals=totals, exploratory_weights_comparisons=comparisons,
                masked_and_independent_controls_trajectory_invariant=True, worlds_identical_before_each_switch=True)


def check_ablation(folder, initial_attempt):
    hashes = read(folder / "summary.json")["npz_sha256"]
    rows = {}
    for mode, arms in hashes.items():
        output = {arm: fetch(folder, mode, arm, hashes) for arm in arms}
        donor = output["joint"]
        random = output["matched_random"]
        if initial_attempt is not None:
            for name in arms:
                filename = f"{mode}-{name}.npz"
                assert digest(initial_attempt / filename) == digest(folder / filename)
        np.testing.assert_array_equal(random["edits"] >= 0, donor["edits"] >= 0)
        for name in ("cut_motor", "cut_taste"):
            cut = output[name]
            for field in ("contact_side", "contact_tick", "reward"):
                np.testing.assert_array_equal(donor[field][:, :-TAIL], cut[field][:, :-TAIL])
            np.testing.assert_array_equal(donor["edits"], cut["edits"])
            np.testing.assert_array_equal(donor["final_weight"], cut["final_weight"])
        audit_edits(donor["initial_active"], donor["initial_weight"], random["edits"], random["final_active"], random["final_weight"])
        counts = (donor["edits"] >= 0).sum(axis=(0, 1))
        rows[mode] = dict(matched_growth=int(counts[0]), matched_retirement=int(counts[1]),
                          raw_sha256_identical_after_summary_fix=True if initial_attempt is not None else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--first-attempt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = dict(primary=check_primary(args.primary), ablation=check_ablation(args.ablation, args.first_attempt))
    write(args.out, results, exclusive=True)
    print(results)


if __name__ == "__main__":
    main()
