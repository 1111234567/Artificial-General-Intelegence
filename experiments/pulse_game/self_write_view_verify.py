"""Replay the archived self-readback writers and check recorded interventions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import self_write_view as env
from . import self_write_view_study as study


def load_writers(weights, seed):
    writers = {}
    for mode in env.MODES:
        prefix = f"{seed}__{mode}__"
        writer = study.Writer()
        writer.load_state_dict({key[len(prefix):]: torch.from_numpy(weights[key])
                                for key in weights.files if key.startswith(prefix)}, strict=True)
        writers[mode] = writer.eval()
    return writers


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    boot = json.loads((directory / "boot.json").read_text(encoding="utf-8"))
    seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    for name, recorded in summary["sha256"].items():
        if study.sha256(directory / name) != recorded:
            raise ValueError(f"Evidence digest mismatch for {name}")
    for path in (Path(study.__file__), Path(env.__file__)):
        if study.sha256(path) != boot["source_sha256"][path.name]:
            raise ValueError(f"Source digest mismatch for {path.name}")
    if summary["sha256"]["holdout.npz"] != seal["holdout_sha256"]:
        raise ValueError("Seal digest mismatch")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    replayed = 0
    with (np.load(directory / "holdout.npz") as holdout, np.load(directory / "weights.npz") as weights,
          np.load(directory / "rounds.npz") as rounds):
        saved = {key: rounds[key] for key in rounds.files}
        writer_bank = [load_writers(weights, seed) for seed in seal["actor_seeds"]]
        for block, world_seed in enumerate(seal["block_seeds"]):
            regenerated = study.trial_arrays(env.random_trial(world_seed, seal["trials_per_block"]))
            for key, value in regenerated.items():
                np.testing.assert_array_equal(holdout[key][block], value)
            trial = study.trial_from_arrays(regenerated)
            for index, writers in enumerate(writer_bank):
                panel = study.evaluate(writers, trial, world_seed + 7000000)
                for mode, record in panel.items():
                    for key, value in record.items():
                        np.testing.assert_array_equal(saved[f"{mode}__{key}"][index, block], value)
                    replayed += len(trial.drive)

        damaged = holdout["disturbance"] != env.NOOP
        in_birth_palette = ((~holdout["damaged_exists"]) | np.isin(holdout["damaged_values"], env.DELTAS)).all(axis=2)
        subsets = {"all_values_in_birth_palette": damaged & in_birth_palette,
                   "any_value_outside_birth_palette": damaged & ~in_birth_palette}
        palette_audit = {label: {"trial_count": int(mask.sum()),
                                 "exact_recovery": {mode: float(saved[f"{mode}__restored"][:, mask].mean())
                                                    for mode in env.MODES},
                                 "current_only_minus_blind": study.paired_interval(
                                     saved["current_only__restored"], saved["blind__restored"], mask)}
                         for label, mask in subsets.items()}
    return {"replayed_policy_and_control_rows": replayed, "verified_evidence_sha256": summary["sha256"],
            "verified_source_sha256": boot["source_sha256"], "palette_subset_audit_exploratory": palette_audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=study.EVIDENCE)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(verify(arguments.directory), indent=2) + "\n"
    if arguments.report is None:
        print(report, end="")
    else:
        with arguments.report.open("x", encoding="utf-8") as output:
            output.write(report)
