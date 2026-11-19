"""Reconstruct order training pools and replay frozen physical outcomes."""

from argparse import ArgumentParser
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import order_model as model, order_study as study, order_world as env, world


def array_digest(value):
    return {"sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            "dtype": str(value.dtype), "shape": value.shape}


def provenance():
    output = {}
    for profile in env.TIMING:
        tape = env.make(study.TRAIN, study.TRAIN_REQUESTS, profile)
        observations, side, taste, _ = model.pool(tape)
        output[profile] = {
            "roots_sha256": array_digest(np.asarray(tape.body.seeds, dtype="<u8")),
            "public_observation_life_request_tick_sense": array_digest(observations),
            "reached_side_life_request": array_digest(side),
            "delivered_taste_life_request": array_digest(taste),
            "endpoint_distances": array_digest(tape.body.distances),
            "winds": array_digest(tape.body.wind),
            "receipts": {"opportunities": taste.size, "delivered": int((taste != 0).sum()),
                         "positive": int((taste > 0).sum()), "negative": int((taste < 0).sum())},
        }
    return output


def verify_cohort(result, pools):
    reg = study.read(result / "registered.json")
    manifest = study.read(result / "frozen.json")
    for birth in reg["confirm_births"]:
        for arm in study.TOPOLOGIES:
            name = f"births/{birth}/{arm}.pt"
            log = study.read(result / "births" / str(birth) / f"{arm}-training.json")
            assert log["example_order_seed"] == study.sampler_seed(birth)
            assert log["checkpoint_sha256"] == manifest["checkpoints"][name] == study.digest(result / name)
            assert log["receipt_counts"] == {key: value["receipts"] for key, value in pools.items()}

    birth = reg["confirm_births"][0]
    verified = []
    for profile in env.TIMING:
        tape = env.make(reg["holdout_roots"], reg["holdout_requests"], profile)
        with np.load(result / "births" / str(birth) / f"{profile}.npz") as raw:
            for arm in study.TOPOLOGIES:
                net = study.load_net(result, birth, arm, manifest)
                for name, change in (("intact", 0), ("drop_candidate_edge", 1), ("erase_history", 2), ("no_recurrence", 3)):
                    np.testing.assert_array_equal(study.outcomes(model.run(net, tape, change)), raw[f"{arm}_{name}"])
                    verified.append(f"{profile}/{arm}/{name}")
                for name in study.FORKS:
                    fork = study.outcomes(model.run(net, env.fork(tape, name)), last=True)
                    np.testing.assert_array_equal(fork, raw[f"{arm}_{name}"])
                    verified.append(f"{profile}/{arm}/{name}")
    return {"replay_birth": birth, "exact_rollout_cases": verified,
            "training_log_and_checkpoint_pairs_checked": len(reg["confirm_births"]),
            "frozen_sha256": study.digest(result / "frozen.json")}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", default=[study.DEFAULT])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pools = provenance()
    git_version = subprocess.check_output(["git", "ls-tree", "HEAD", "experiments/pulse_game/world.py"],
                                          cwd=study.ROOT, text=True).split()[2]
    study.write(args.output, {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Post-run reconstruction from archived origins and fixed versions. Checkpoint hashes and actual explorer-receipt counts agree with every training log.",
        "serialization": "C-contiguous raw bytes; dtype and shape recorded; root IDs uint64 little-endian.",
        "body_world_py_sha256": study.digest(Path(world.__file__)),
        "body_world_py_tracked_git_blob": git_version,
        "source_sha256": study.source_hashes(),
        "training_pools": pools,
        "verify_frozen_cohorts": {path.name: verify_cohort(path, pools) for path in args.results},
    })


if __name__ == "__main__":
    main()
