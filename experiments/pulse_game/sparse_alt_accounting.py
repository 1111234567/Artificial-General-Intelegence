"""Independently audit saved corroborating-graph trajectories and pair work."""

import argparse
from pathlib import Path

import numpy as np

from .sparse_alt_study import ARMS, MODES, TAIL, TRIALS, audit, legal_count
from .unified_live_graph_study import digest, read, write


SHARED_FIELDS = ("birth", "world", "change", "initial_mask", "initial_weight", "truth")


def archive_arm(raw, arm):
    prefix = f"{arm}/"
    return {field: raw[field] for field in SHARED_FIELDS} | {
        field.removeprefix(prefix): raw[field] for field in raw.files if field.startswith(prefix)
    }


def accounting(folder):
    summary = read(folder / "summary.json")
    assert summary["development_sha256"] == digest(folder / "development.json")
    assert read(folder / "development.json")["registered_sha256"] == digest(folder / "registered.json")
    rows = {}
    for mode in MODES:
        path = folder / f"{mode}.npz"
        assert digest(path) == summary["results"][mode]["raw_sha256"]
        with np.load(path, allow_pickle=False) as raw:
            for arm in ARMS:
                data = archive_arm(raw, arm)
                audit(data, arm)
                np.testing.assert_array_equal(data["events"][:, -1, 1], legal_count(data["final_mask"]))
                expected = summary["results"][mode]["means"][arm]["tail"]["mean"]
                assert np.isclose(data["correct"][:, -TAIL:].mean(), expected)
            for arm in ("intact", "cut"):
                contact, tick, truth = (raw[f"probe/{arm}/{field}"] for field in ("contact", "tick", "truth"))
                np.testing.assert_array_equal(contact >= 0, tick >= 0)
                np.testing.assert_array_equal(raw[f"probe/{arm}/correct"], contact == truth)
            np.testing.assert_array_equal(raw["probe/intact/truth"], raw["probe/cut/truth"])
            data = archive_arm(raw, "corroborating")
            receipt = data["shown"] != 0
            live, legal, probes, probe_gradients = data["events"][:, :, :4].transpose(2, 0, 1)
            live_before = np.column_stack((np.zeros(len(live), dtype=np.int16), live[:, :-1]))
            legal_before = np.column_stack((legal_count(data["initial_mask"]), legal[:, :-1]))
            live_gradients = (live_before * receipt).sum(axis=1)
            legal_gradients = (legal_before * receipt).sum(axis=1)
            probe_gradients = probe_gradients.sum(axis=1)
            eval_share = (live_gradients + probe_gradients) / legal_gradients
            rows[mode] = dict(
                lives=len(live), trials=TRIALS, real_receipts_mean=float(receipt.sum(axis=1).mean()),
                live_gradient_evaluations_mean=float(live_gradients.mean()),
                probe_gradient_evaluations_mean=float(probe_gradients.mean()),
                legal_pair_receipt_opportunities_mean=float(legal_gradients.mean()),
                gradient_evaluation_share_mean=float(eval_share.mean()),
                gradient_evaluation_share_max=float(eval_share.max()),
                live_allocated_fraction_max=float((live / legal).max()),
                probe_and_live_fraction_max=float(((live + probes) / legal).max()),
                probe_and_live_final_fraction_mean=float(((live[:, -1] + probes[:, -1]) / legal[:, -1]).mean()),
            )
    return dict(summary_sha256=digest(folder / "summary.json"), modes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = accounting(args.out)
    write(args.out / "accounting.json", result, exclusive=True)
    print(result)


if __name__ == "__main__":
    main()
