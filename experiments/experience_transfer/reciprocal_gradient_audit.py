"""Reproduce the first eight Bprime Adam updates and actual carried h exactly."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.experience_transfer import reciprocal_train as r
from experiments.experience_transfer import train_audit


def main():
    torch.set_num_threads(1)
    r.checked()
    for seed in range(3):
        began = time.perf_counter()
        folder = r.OUT / f"seed_{seed:03d}"
        boundary = torch.load(folder / "B_boundary.pt", map_location="cpu", weights_only=False)
        checks = {}
        for arm in r.ARMS:
            trajectory = torch.load(folder / arm / "trajectory_B_000.pt", map_location="cpu", weights_only=False)
            checks[arm] = train_audit.reproduce_early_updates(folder / arm, trajectory, boundary[arm])
        r.t.write(r.OUT / f"gradient_audit_seed_{seed:03d}.json", dict(seed=seed, passed=True,
            checks=checks, source_sha256=r.t.sha(__file__),
            generic_independent_gradient_auditor_sha256=r.t.sha(train_audit.__file__),
            elapsed_seconds=time.perf_counter()-began,
            coverage="Both history branches: exact first2/8 checkpoint model, Adam, and actual public-history hidden state. Full2048-update rerun not claimed."))
        print(f"reciprocal exact-gradient audit seed={seed} passed seconds={time.perf_counter()-began:.1f}", flush=True)


if __name__ == "__main__":
    main()
