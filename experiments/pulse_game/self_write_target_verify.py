"""Check recorded target-value trajectories against replay from frozen birth graphs."""
import hashlib
import json

import numpy as np

from . import self_write_core_lifetime as life
from . import self_write_target as target
from . import self_write_target_lifetime as study


def verify_npz(path, expected):
    with np.load(path) as stored:
        if set(stored.files) != set(expected):
            raise AssertionError(f"Receipt fields differ at {path}")
        for key, values in expected.items():
            np.testing.assert_array_equal(stored[key], values, err_msg=f"{path}: {key}")


def main():
    summary = json.loads((target.EVIDENCE / "lifetime-summary.json").read_text(encoding="utf-8"))
    for path, expected in summary["files_sha256"].items():
        with open(path, "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise AssertionError(f"Digest mismatch at {path}")
    total_rounds = 0
    for root_label in summary["roots"]:
        root = int(root_label)
        folder = study.DATA / root_label
        world = life.worlds(root, seed=study.WORLD_SEED)
        for label, (training, view, arm) in study.ARMS.items():
            path = target.EVIDENCE / f"bootstrap-{training}-{root}.npz"
            birth = life.load_graph(path)
            verify_npz(folder / f"{label}-simple.npz", study.run_simple(birth, root, view, arm))
            total_rounds += study.PAIR_CONTACTS * study.PAIR_WORLDS
            for regime in life.REGIMES:
                endpoint, h, audit, edits, snapshots = life.run_world(
                    birth, world, regime, arm, view=view, step_fn=target.step,
                )
                prefix = folder / f"{label}-{regime}"
                verify_npz(f"{prefix}-receipts.npz", {**audit, **edits})
                fresh = study.endpoint_pair(birth, endpoint, root, view, arm)
                verify_npz(f"{prefix}-snapshots.npz", {
                    "w": endpoint.w, "b": endpoint.b, "mask": endpoint.mask, "h": h,
                    **snapshots, **{f"fresh_pair_{key}": arr for key, arr in fresh.items()},
                })
                total_rounds += audit["correct"].size
    print(f"Exact replay of {total_rounds} contact rounds", flush=True)


if __name__ == "__main__":
    main()
