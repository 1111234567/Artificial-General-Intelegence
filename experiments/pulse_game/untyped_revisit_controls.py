"""Ablate each selected feature at frozen weights on the same physical outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import untyped_revisit_study as study, untyped_senses as legacy


def run(folder):
    complete = json.loads((folder / "complete.json").read_text())
    chosen = json.loads((folder / "selected.json").read_text())
    birth_ids = complete["births"]
    registration = json.loads((study.BIRTHS / "registered.json").read_text())
    alpha = registration["reopening_rule"]
    weights = np.asarray(chosen["selected"][1], np.float32)
    scenarios = ("standard", "prolonged", "side_heavy", "matched_jitter")
    output = {}
    for birth in birth_ids:
        phenotype = study.v1.organism(study.BIRTHS / f"birth-{birth}")
        output[birth] = {}
        for case in scenarios:
            source = json.loads((folder / f"external-{birth}-{case}.json").read_text())
            ep, pk = study.episode(source["seeds"], case, source["nonce"])
            n = len(source["seeds"])
            _, raw_v1, _ = legacy.prepare(ep, pk, [1, 0, 0])
            _, new_v1, _, _ = study.model.prepare(ep, pk, [1, 0, 0, 0])
            entry = {"baseline_role_mismatches": int((raw_v1 != new_v1).sum()), "rows": {}}
            for i in np.flatnonzero(weights):
                ablated = weights.copy()
                ablated[i] = 0
                real = study.play(ep, pk, phenotype, alpha, ablated)
                entry["rows"][f"without_{study.FEATURES[i]}"] = study.role_rows(ep, pk, real, n)
            output[birth][case] = entry
        print("ablated", birth, flush=True)
        study.save(folder / "fixed_feature_ablations.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.HOME)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.out)


if __name__ == "__main__":
    main()
