"""Continue the same optimizer trajectory and test mark meanings on new random tapes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from . import revisit_choice_value as v


STAGES = (0, 500, 2000, 4000, 8000, 12000)
ARMS = ("fixed", "variable")
ROOTS = list(range(731, 737))


def graph(path):
    with np.load(path) as z:
        return v.fw.Graph(*(torch.from_numpy(z[k]) for k in ("w", "b", "m")))


@torch.no_grad()
def verify_and_score(job):
    root, arm, seed = job
    new = root / "longitudinal"
    torch.set_num_threads(1)
    for step in STAGES[:4]:
        prev = graph(root / arm / f"{seed}-{step}.npz")
        redo = graph(new / arm / f"{seed}-{step}.npz")
        for key in ("weights", "bias", "present"):
            if not torch.equal(getattr(prev, key), getattr(redo, key)):
                raise AssertionError(f"Optimizer history changed: {arm}/{seed}/{step}/{key}")
    scene = v.scenes([4119751, seed], 96, balanced=True)
    cells = {}
    for step in STAGES[3:]:
        g = graph(new / arm / f"{seed}-{step}.npz")
        records = {"book": scene.book.numpy(), "order": scene.first_mark.numpy()}
        for treatment in ("own", "blank", "yoked", "disable", "restore_graph", "zero_h"):
            a, r, _ = v.interact(g, scene, len(scene.book), treatment=treatment)
            good = (r > 0).numpy()
            records[treatment] = good
            records[treatment + "_side"] = a.side.numpy().astype(np.int8)
            if treatment in ("blank", "yoked") and good.mean() != .5:
                raise AssertionError("Inaccessible polarity altered behavior")
        for history in ("stable", "switched"):
            a, r, _ = v.interact(g, scene, len(scene.book), history=history)
            records[history] = (r > 0).numpy()
        result = {k: float(arr.mean()) for k, arr in records.items() if arr.dtype == np.bool_}
        for law in (-1, 1):
            result[f"law{law}"] = float(records["own"][records["book"] == law].mean())
        result["worse_law"] = min(result["law1"], result["law-1"])
        cells[step] = result
        v.write_arrays(new / "validation" / arm / f"{seed}-{step}.npz", **records)
    return seed, arm, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/revisit-choice-20261119/value"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root, new = args.root, args.root / "longitudinal"
    new.mkdir(parents=True, exist_ok=True)
    plan = dict(births=ROOTS, stages=STAGES, batch=24, arms=ARMS, holdout_domain=4119751,
                tapes_per_birth=96, purpose="After seeing one-polarity specialization at 4000, extend both unchanged fits to 12000. Require original checkpoints through 4000 to replay bit-identically; examine poorer polarity per birth on a separate evaluation tape.")
    (new / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf8")
    jobs = [(new, ROOTS[i:i+3], arm, max(STAGES), 24, STAGES) for arm in ARMS for i in (0, 3)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(v.fit_group, jobs))
        cells = list(pool.map(verify_and_score, [(root, a, s) for a in ARMS for s in ROOTS]))
    results = {arm: {seed: data for seed, a, data in cells if a == arm} for arm in ARMS}
    (new / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
