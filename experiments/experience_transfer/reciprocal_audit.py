"""Independent restoration, budget, demonstration and heldout checks for Bprime."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.continuity.run import state_digest
from experiments.experience_transfer import reciprocal_train as r, reciprocal_eval as q
from experiments.experience_transfer import state_audit as independent
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import evaluate


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def a(value):
    return value.detach().cpu().numpy().copy()


def training(seed):
    folder = r.OUT / f"seed_{seed:03d}"
    spec = r.t.SEEDS[seed]
    bindings = q.e.read(folder / "branch_bindings.json")
    reports = []
    for arm in r.ARMS:
        old = load(r.origin(seed, arm))
        zero = load(folder / arm / "B000.pt")
        binding = next(x for x in bindings if x["arm"] == arm)
        assert r.t.sha(r.origin(seed, arm)) == binding["source_sha256"]
        for key in ("model", "optimizer", "hidden", "numpy_rng", "torch_rng", "cuda_rng", "work"):
            assert state_digest(old[key]) == state_digest(zero[key]), (seed, arm, key)
        expected = deepcopy(old["world"])
        expected["context"]["mappings"] = r.table(seed, arm)
        assert state_digest(expected) == state_digest(zero["world"])
        assert zero["transfer"]["spec"]["B"] == r.bprime(seed)
        for snapshot in r.SNAPSHOTS:
            cp = load(folder / arm / f"{snapshot}.pt")
            round_number = int(snapshot[1:])
            assert cp["work"]["gradient_updates"] - zero["work"]["gradient_updates"] == round_number * 256
            assert cp["work"]["environment_ticks"] - zero["work"]["environment_ticks"] == round_number * 1024 + bool(round_number)
            assert cp["work"]["old_history_replay_positions"] == 0
            if round_number:
                source = load(r.t.OUT / f"seed_{seed:03d}" / arm / f"{snapshot}.pt")
                assert state_digest(cp["world"]["scheduler"]) == state_digest(source["world"]["scheduler"])
                assert torch.equal(cp["world"]["context"]["tensors"]["family_new_counts"], source["world"]["context"]["tensors"]["family_new_counts"])
        total_mask = total_nonwait = 0
        for index in range(8):
            current = load(folder / arm / f"trajectory_B_{index:03d}.pt")
            old_data = load(r.t.OUT / f"seed_{seed:03d}" / arm / f"trajectory_B_{index:03d}.pt")
            peer_arm = r.ARMS[1] if arm == r.ARMS[0] else r.ARMS[0]
            peer = load(folder / peer_arm / f"trajectory_B_{index:03d}.pt")
            for key in ("labels", "actions", "mask"):
                assert torch.equal(current[key], peer[key]), (seed, index, key)
            assert torch.equal(current["mask"], old_data["mask"])
            assert current["obs"].shape == (1024, 64, 56)
            assert current["labels"].shape == current["mask"].shape == (1024, 64, 2)
            assert torch.equal(current["actions"], current["labels"])
            total_mask += int(current["mask"].sum())
            total_nonwait += int((current["actions"] != 0).sum())
            if index in (0, 7):
                # Reconstruct real world/teacher demonstrations, not just array
                # agreement. The oracle remains scorer-only in this TRAIN check.
                with r.t.mapping_scope(r.table(seed, arm)):
                    world = ContextWorld.from_state_dict(current["initial_world"], device="cpu")
                    for tick in range(1024):
                        obs = world.observe()
                        assert torch.equal(obs, current["obs"][tick])
                        mask = world.current_family == spec["B_cue"]
                        labels = torch.where(mask, world.oracle("midpoint"), torch.zeros((64, 2), dtype=torch.long))
                        assert torch.equal(mask, current["mask"][tick])
                        assert torch.equal(labels, current["labels"][tick])
                        world.act(current["actions"][tick])
                    endpoint = load(folder / arm / ("B001.pt" if index == 0 else "B008.pt"))
                    assert state_digest(world.state_dict()) == state_digest(endpoint["world"])
        end = load(folder / arm / "B008.pt")
        assert total_mask == end["work"]["B_unique_label_positions"]
        assert total_nonwait == end["work"]["B_unique_nonwait_labels"]
        assert end["work"]["B_gradient_label_positions"] == 32 * total_mask
        assert state_digest(old["model"]) == binding["model_hash"]
        reports.append(dict(arm=arm, exact_B0_restore=True, paired_rounds=8, replayed_real_rounds=[0, 7],
            B_unique_label_positions=total_mask, B_gradient_label_positions=32*total_mask,
            B_supervised_NEW_work=end["work"]["B_unique_new_commands"],
            actual_B_NEW_total=int(end["world"]["context"]["tensors"]["family_new_counts"][..., spec["B_cue"]].sum())))
    return reports


def semantic_receipts(frames, record):
    active, started = frames["active"], frames["started"]
    actions, version = frames["actions"], frames["version"]
    begin = (actions == 1) | (actions == 2)
    finish = (actions >= 3) & (actions <= 10)
    begin_ok = begin & active & ~started & (actions-1 == version)
    finish_ok = finish & active & started & ((actions-3)//4 == version) & ((actions-3)%4 == frames["payload"]) & (frames["now"][:, None, None] >= frames["ready_time"]-1e-6) & (frames["now"][:, None, None] <= frames["deadline"]+1e-6)
    accepted = begin_ok | finish_ok
    rejected = (actions != 0) & ~accepted
    assert np.array_equal(frames["result"], accepted.astype(int)-rejected.astype(int))
    primary = (frames["request_id"]//2 > record["fork_tick"]) & (frames["current_regime_id"] == record["evaluation_regime_id"]) & frames["current_eligible"]
    feasible = begin_ok & (frames["now"][:, None, None] + frames["duration"] <= frames["deadline"]+1e-6)
    tests = dict(begin_attempts=begin, valid_begin=begin_ok, feasible_begin=feasible, finish_attempts=finish, legal_finish=finish_ok)
    for key, mask in tests.items():
        assert int((primary & mask).sum()) == record["counts"].get(key, 0), key
    return dict(position_count=int(actions.size), accepted=int(accepted.sum()), rejected=int(rejected.sum()))


@torch.inference_mode()
def evaluations(seed):
    records = q.e.read(q.OUT / f"seed_{seed:03d}.json")["records"]
    assert len(records) == 10
    checks = []
    original_oracle = ContextWorld.oracle
    ContextWorld.oracle = independent.no_oracle
    try:
        for record in records:
            arm, snapshot = record["arm"], record["snapshot"]
            stem = q.OUT / f"seed_{seed:03d}" / f"{arm}_{snapshot}_Bprime"
            for path, expected in record["artifacts"].items():
                assert q.e.sha(path) == expected
            assert q.e.sha(record["checkpoint_path"]) == record["checkpoint_sha256"]
            model, _, _, cp = r.t.load_checkpoint(record["checkpoint_path"], device="cpu")
            model.eval()
            states = load(stem.with_suffix(".pt"))
            requests = q.e.read(stem.with_suffix(".requests.json"))
            with np.load(stem.with_suffix(".npz")) as frames:
                reduced = deepcopy(record)
                reduced["primary_requests"] = requests["matured"]
                reduced["counts"] = {key: record["counts"].get(key, 0) for key in ("matured", "completed", "clean_completed", "idle_actions", "idle_opportunities", "primary_issued", "primary_censored")}
                independent.independent_counts(frames, states, reduced)
                semantic = semantic_receipts(frames, record)
                h = states["initial_h"].clone()
                for index, obs in enumerate(frames["obs"]):
                    logits, h = model(torch.from_numpy(obs)[None], h)
                    assert np.array_equal(a(logits[0]), frames["logits"][index])
                    assert np.array_equal(a(logits[0].argmax(-1)), frames["actions"][index])
                assert torch.equal(h, states["end_h"])
                # Every family/raw subgroup's matured outcomes are independently
                # counted directly from the saved request rows.
                bf = str(record["B_family"])
                for group, members in r.checked()["spec"][str(seed)]["frozen_raw_groups"].items():
                    selected = [v for v in requests["matured"] if v["family"] == record["B_family"] and v["raw"] in members]
                    expected = record["strata"][bf][group]["counts"]
                    assert len(selected) == expected.get("matured", 0)
                    assert sum(v["completed"] for v in selected) == expected.get("completed", 0)
                    assert sum(v["clean_completed"] for v in selected) == expected.get("clean_completed", 0)
            native_replayed = snapshot in ("B000", "B008")
            if native_replayed:
                with r.t.mapping_scope(r.table(seed, arm)):
                    world = ContextWorld.from_state_dict(states["initial_world"], device="cpu")
                    native = evaluate(model, world, states["initial_h"], family=record["B_family"], condition="long_both", steps=2048, device="cpu")
                for key in ("matured", "completed", "clean_completed"):
                    assert native["primary"][key] == record["counts"][key]
                assert native["opportunities"]["inactive_actions"] == record["counts"]["idle_actions"]
                assert native["opportunities"]["inactive_opportunities"] == record["counts"]["idle_opportunities"]
                assert native["end_world_state_hash"] == record["end_world_hash"]
                assert native["end_hidden_hash"] == state_digest({"hidden": states["end_h"]})
            checks.append(dict(arm=arm, snapshot=snapshot, native_replayed=native_replayed, semantic=semantic,
                checkpoint_sha256=record["checkpoint_sha256"], primary_identity_hash=record["primary_identity_hash"]))
    finally:
        ContextWorld.oracle = original_oracle
    return checks


def verify(seed):
    torch.set_num_threads(1)
    r.checked(); q.checked()
    began = time.perf_counter()
    trained = training(seed)
    evaluated = evaluations(seed)
    q.e.write(r.OUT / f"audit_seed_{seed:03d}.json", dict(seed=seed, passed=True,
        source_sha256=q.e.sha(__file__), independent_array_auditor_sha256=q.e.sha(independent.__file__),
        training=trained, evaluations=evaluated, elapsed_seconds=time.perf_counter()-began,
        coverage=dict(complete_B0_restorations=2, paired_training_rounds=8, real_training_trajectories_replayed=4,
            evaluated_arrays=10, exact_model_recurrence=10, native_complete_world_evaluations=4),
        limitations="Gradient history is inherited and code/Adam-bound; no independent rerun of all2048 B optimizer steps. All saved budgets/demonstrations and complete endpoints are checked."))
    print(f"reciprocal audit seed={seed} complete seconds={time.perf_counter()-began:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(3), required=True)
    verify(parser.parse_args().seed)
