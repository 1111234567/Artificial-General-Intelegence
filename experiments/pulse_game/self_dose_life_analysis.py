"""Analyze paired lifetime outcomes, keeper roots and committed edit replay."""

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from . import self_dose_life as life


class CrossedBootstrap:
    """Sample each root factor independently and reuse the draws for every arm."""

    def __init__(self, shape, config):
        self.shape = shape
        self.quantiles = config["interval"]
        rng = np.random.default_rng(config["root"])
        birth, donor, receiver = [
            np.eye(size, dtype=np.float32)[rng.integers(size, size=(config["draws"], size))].sum(axis=1)
            for size in shape
        ]
        weights = birth[:, :, None, None] * donor[:, None, :, None] * receiver[:, None, None, :]
        self.weights = weights.reshape(config["draws"], -1) / np.prod(shape, dtype=np.float32)

    def interval(self, cells):
        if cells.shape[:3] != self.shape:
            raise ValueError("Each endpoint must retain its crossed birth, donor and receiver axes")
        tail = cells.shape[3:]
        simulated = self.weights @ np.ascontiguousarray(cells.reshape(np.prod(self.shape), -1), dtype=np.float32)
        ci = np.quantile(simulated, self.quantiles, axis=0)
        return {
            "mean": cells.mean(axis=(0, 1, 2), dtype=np.float64).tolist(),
            "ci95": ci.reshape(2, *tail).tolist(),
        }


def action_cells(archive, prefix, shape, ecology_count, windows):
    """Cell axes are birth, donor root, receiver root, ecology and action time."""
    correct = archive[f"{prefix}_correct"][windows]
    reshaped = correct.reshape(len(windows), *shape, ecology_count, 2)
    result = np.moveaxis(reshaped.mean(axis=-1), 0, -1)
    np.testing.assert_array_equal(result[..., 0], 0.5)
    return result


def average_acquisition(curve):
    return curve[..., 1:].mean(axis=-1) - curve[..., 0]


def count_summary(values):
    return {
        "mean_l2_or_hamming": values.mean(axis=0).tolist(),
        "changed_lives": (values > 1e-7).sum(axis=0).tolist(),
        "max_l2_or_hamming": values.max(axis=0).tolist(),
    }


def verify_snapshot(archive, dose):
    source = f"d{dose}_donor_active"
    reference = "d0_donor_active"
    graft = f"d{dose}_initial_writer"
    null = f"d{dose}_initial_birth_null"
    for field in life.FIELDS:
        archived = archive[f"{source}_{field}"]
        birth = archive[f"{reference}_{field}"]
        protected = archive[f"d{dose}_donor_protected_{field}"]
        received = archive[f"{graft}_{field}"]
        np.testing.assert_array_equal(archived[:, life.core.WRITER], received[:, life.core.WRITER])
        np.testing.assert_array_equal(birth[:, life.REGIONS[0]], received[:, life.REGIONS[0]])
        np.testing.assert_array_equal(birth[:, life.core.WRITER], protected[:, life.core.WRITER])
        np.testing.assert_array_equal(birth, archive[f"{null}_{field}"])
    np.testing.assert_array_equal(archive[f"{source}_connected"], True)
    actual = [life.core.connected(row) for row in archive[f"{graft}_mask"]]
    np.testing.assert_array_equal(actual, archive[f"{graft}_connected"])


def verify_action_log(archive, prefix, tape_name):
    sides = archive[f"{prefix}_side"]
    tastes = archive[f"{prefix}_taste"]
    truth = archive[f"{tape_name}_good"] * archive[f"{tape_name}_orientation"]
    copies = (1, sides.shape[1] // truth.shape[1])
    truth = np.tile(truth, copies)
    flipped = np.tile(archive[f"{tape_name}_flipped"], copies)
    np.testing.assert_array_equal(sides == truth, archive[f"{prefix}_correct"])
    expected = np.where(sides == truth, 1, -1) * np.where(flipped, -1, 1)
    np.testing.assert_array_equal(tastes, expected[:len(tastes)])
    if tape_name == "receiver":
        np.testing.assert_array_equal(sides, np.where(archive[f"{prefix}_motor"] > 0, 1, -1))


def action_summary(interval, curve):
    gain = average_acquisition(curve)
    return {
        "by_ecology_and_receipts": interval(curve),
        "pooled_by_receipts": interval(curve.mean(axis=-2)),
        "mean_post_receipt_gain": interval(gain.mean(axis=-1)),
        "mean_post_receipt_gain_by_ecology": interval(gain),
    }


def connectivity_description(delta, connected):
    """Disconnection is an outcome. Stratification is descriptive and never filters the estimate."""
    data = {}
    pooled = average_acquisition(delta).mean(axis=(2, 3))
    for name, mask in (("connected", connected), ("disconnected", ~connected)):
        group = pooled[mask.reshape(pooled.shape)]
        data[name] = {"donor_lives": group.size, "mean_gain_difference": float(group.mean()) if group.size else None}
    return data


def dose_report(archive, dose, plan, bootstrap):
    shape = bootstrap.shape
    windows = plan["receiver_receipts_before_action"]
    curves = {}
    group = {"actions": {}, "contrasts": {}, "replay": {}, "graft": {}, "donors": {}}
    verify_snapshot(archive, dose)
    for arm in life.ARMS:
        prefix = f"d{dose}_{arm}"
        verify_action_log(archive, prefix, "receiver")
        curves[arm] = action_cells(archive, prefix, shape, len(plan["ecology"]), windows)
        group["actions"][arm] = action_summary(bootstrap.interval, curves[arm])
        initial = f"d{dose}_initial_{arm}"
        group["graft"][arm] = {
            "parameters": count_summary(archive[f"{initial}_distance_from_birth"]),
            "disconnected_lives": int((~archive[f"{initial}_connected"]).sum()),
            "zero_receipt_action_disagreement_with_birth": float(np.mean(
                archive[f"{prefix}_side"][0] != archive[f"d{dose}_birth_null_side"][0]
            )),
        }
    group["graft"]["scramble_vs_writer"] = count_summary(archive[f"d{dose}_scramble_distance_from_graft"])
    contrasts = {}
    for control in life.ARMS[1:]:
        field = f"writer_vs_{control}"
        contrast = curves["writer"] - curves[control]
        contrasts[field] = contrast
        group["contrasts"][field] = action_summary(bootstrap.interval, contrast)
        group["contrasts"][field]["connectivity_sensitivity"] = connectivity_description(
            contrast, archive[f"d{dose}_initial_writer_connected"],
        )
        replay = archive[f"d{dose}_{field}_replay_difference"]
        replay_cells = replay.reshape(*shape, len(plan["ecology"]), 2, 2, 3).mean(axis=(3, 4))
        group["replay"][field] = {
            **count_summary(replay),
            "crossed_mean": bootstrap.interval(replay_cells),
        }
    through = dose * plan["contacts_per_donor_problem"]
    for arm in ("active", "protected"):
        prefix = f"d{dose}_donor_{arm}"
        accuracy = archive[f"donor_{arm}_correct"][:through]
        counts = archive[f"donor_{arm}_edits"][:through].sum(axis=(0, 1), dtype=np.int64)
        status = archive[f"donor_{arm}_status"][:through]
        group["donors"][arm] = {
            "accuracy": accuracy.mean() if through else None,
            "accepted_total_counts_by_region": counts.tolist(),
            "structural_status_counts_by_birth_and_retirement": [
                np.bincount(status[:, :, i].flatten(), minlength=len(life.core.STATUS)).tolist()
                for i in range(2)
            ],
            "weights_at_limit": archive[f"{prefix}_weights_at_limit"].sum(axis=0).tolist(),
            "biases_at_limit": archive[f"{prefix}_biases_at_limit"].sum(axis=0).tolist(),
        }
    return group, curves, contrasts


def report(path, plan, evidence, provenance):
    sizes = (
        len(plan["birth_roots"]), len(plan["donor_world_roots"]), len(plan["receiver_world_roots"]),
    )
    bootstrap = CrossedBootstrap(sizes, plan["bootstrap"])
    results, action_cell_archive, gains = {}, {}, {}
    with np.load(path, allow_pickle=False) as raw:
        for arm in ("active", "protected"):
            verify_action_log(raw, f"donor_{arm}", "donor")
        np.testing.assert_array_equal(raw["donor_protected_edits"][:, :, 1], 0)
        for dose in plan["doses"]:
            item, curves, contrasts = dose_report(raw, dose, plan, bootstrap)
            results[str(dose)] = item
            action_cell_archive.update({f"d{dose}_{arm}_accuracy": curve for arm, curve in curves.items()})
            gains[dose] = {name: average_acquisition(curve).mean(axis=-1) for name, curve in contrasts.items()}
    null = "writer_vs_birth_null"
    scramble = "writer_vs_numeric_scramble"
    primary = {
        "late_vs_null": bootstrap.interval(gains[512][null]),
        "late_minus_early_vs_null": bootstrap.interval(gains[512][null] - gains[8][null]),
        "late_vs_scramble": bootstrap.interval(gains[512][scramble]),
    }
    summary = {
        **provenance,
        "python_version": platform.python_version(),
        "raw": {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": life.fingerprint(path)},
        "crossed_root_shape": list(sizes),
        "recipient_lives_per_arm_and_dose": int(np.prod(sizes) * len(plan["ecology"]) * 2),
        "axes": {
            "receipts": plan["receiver_receipts_before_action"],
            "ecology": list(plan["ecology"]),
            "regions": ["body", "writer"],
            "replay_and_parameter_metrics": ["weight_l2", "bias_l2", "support_hamming"],
            "donor_edit_metrics": list(life.EDIT_FIELDS),
            "structural_status": life.core.STATUS,
            "crossed_cell": ["birth_root", "donor_world_root", "receiver_world_root", "ecology", "action_time"],
        },
        "primary": primary,
        "positive_rule_met": all(values["ci95"][0] > 0 for values in primary.values()),
        "doses": results,
    }
    evidence.mkdir(parents=True, exist_ok=True)
    cells = evidence / "paired-action-cells.npz"
    np.savez_compressed(cells, **action_cell_archive)
    summary["cell_evidence"] = {"path": cells.as_posix(), "sha256": life.fingerprint(cells)}
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(primary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=life.PLAN)
    parser.add_argument("--raw-file", type=Path, default=life.RAW / "raw.npz")
    parser.add_argument("--evidence", type=Path, default=life.EVIDENCE)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    _, source_manifest = life.load_births(plan)
    source_files = (life.__file__, life.core.__file__, life.transfer.__file__, __file__)
    report(args.raw_file, plan, args.evidence, {
        "preregistered_plan_sha256": life.fingerprint(args.plan),
        "checkpoint_sha256": source_manifest,
        "source_sha256": {Path(p).name: life.fingerprint(Path(p)) for p in source_files},
        "numpy_version": np.__version__,
    })


if __name__ == "__main__":
    main()
