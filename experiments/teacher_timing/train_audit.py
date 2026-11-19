"""Reconstruct all 24 first CUDA updates without experiment training calls.

Only frozen independent plain-PyTorch audit primitives are reused. The new
wrapper checks all teacher/objective paths, 36 snapshots, paired budgets and
18 complete earliest-history checkpoint regressions against the prior run.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.continuity.run import state_digest
from experiments.remaining_time.train_audit import (
    IndependentPolicy, exact, independent_identity, independent_rows, load,
    read, reconstruct_update, sha,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/teacher-timing/train-v1"
OUT = ROOT / "data/teacher-timing/train-audit-v1"
PLAN = OUT / "audit_plan.json"
TEACHERS = ("earliest", "midpoint")
OBJECTIVES = ("action", "joint")
SEEDS, SNAPSHOTS = (0, 1, 2), (0, 64, 256)
FIELDS = ("obs", "actions", "time_target", "time_mask")
REGRESSION_FIELDS = ("model", "optimizer", "hidden", "stream_cursor", "step", "work")


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def folder(seed, teacher, objective):
    return TRAIN / teacher / f"seed_{seed:03d}" / objective


def checked_training():
    plan, freeze = read(TRAIN / "protocol.json"), read(TRAIN / "globalfreeze.json")
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert state_digest({k: v for k, v in freeze.items() if k != "freeze_hash"}) == freeze["freeze_hash"]
    assert plan["teachers"] == list(TEACHERS) and plan["objectives"] == list(OBJECTIVES)
    assert plan["seeds"] == list(SEEDS) and plan["snapshots"] == list(SNAPSHOTS)
    assert freeze["protocol_hash"] == plan["protocol_hash"]
    assert freeze["endpoint_count"] == 12 and freeze["snapshot_count"] == 36
    assert freeze["heldout_scoring_started"] is False
    assert sha(freeze["earliest_regression_path"]) == freeze["earliest_regression_sha256"]
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, path
    for group in plan["input_sha256"].values():
        for items in group.values():
            for path, expected in items.items():
                assert sha(path) == expected, path
    for name in ("data_protocol", "previous_training_protocol", "previous_freeze"):
        assert sha(plan[f"{name}_path"]) == plan[f"{name}_sha256"]
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    return plan, freeze


def declare():
    training, freeze = checked_training()
    paths = [TRAIN / name for name in ("protocol.json", "globalfreeze.json", "earliest_regression.json")]
    for teacher in TEACHERS:
        for seed in SEEDS:
            paths.extend(map(Path, training["input_sha256"][teacher][str(seed)]))
            for objective in OBJECTIVES:
                directory = folder(seed, teacher, objective)
                paths.extend(directory / name for name in ("updates.jsonl", "completed.json"))
                for step in SNAPSHOTS:
                    paths.extend(directory / f"step{step:03d}{suffix}" for suffix in (".pt", ".json"))
                for step in (1, 2):
                    for name in ("before", "after", "gradients"):
                        paths.append(directory / f"step{step:03d}_{name}.pt")
    sources = [Path(__file__), ROOT / "experiments/remaining_time/train_audit.py",
               ROOT / "experiments/continuity/run.py"]
    write_new(PLAN, dict(format="teacher-timing-independent-training-audit-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        training_protocol_hash=training["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        inputs={str(path): sha(path) for path in dict.fromkeys(paths)},
        sources={str(path): sha(path) for path in sources},
        reconstruction="Frozen independent nn.GRU(56,192), native linear192->22, time linear192->2, direct WAIT-weighted CE plus selected-row MSE, action-only time-feature detach, separately clipped policy/time groups and Adam; no learner/model/loss call",
        scope="All12 warm starts, all3072 block identities/counts/hidden digest chains, all36 named snapshot state and binding checks, all24 actual first/second optimizer steps, within-teacher zero-head first-step equality and all18 earliest historical model/Adam/h/cursor/step/work comparisons"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for path, expected in plan[table].items():
            assert sha(path) == expected, path
    training, freeze = checked_training()
    assert plan["training_protocol_hash"] == training["protocol_hash"]
    assert plan["freeze_hash"] == freeze["freeze_hash"]
    return plan, training, freeze


def independent_stream(seed, teacher, training):
    if teacher == "earliest":
        raw = ROOT / f"data/action-split/streams-v1/seed_{seed:03d}/stream.npz"
        targets = ROOT / f"data/remaining-time/targets-v1/seed_{seed:03d}/targets.npz"
        with np.load(raw, allow_pickle=False) as archive:
            stream = {key: archive[key] for key in ("obs", "actions")}
        with np.load(targets, allow_pickle=False) as archive:
            stream.update({key: archive[key] for key in ("time_target", "time_mask")})
        paths = (raw, targets)
    else:
        raw = ROOT / f"data/teacher-timing/streams-v1/seed_{seed:03d}/midpoint/stream.npz"
        with np.load(raw, allow_pickle=False) as archive:
            stream = {key: archive[key] for key in FIELDS}
        paths = (raw,)
    bound = training["input_sha256"][teacher][str(seed)]
    assert all(str(path) in bound for path in paths)
    assert stream["obs"].shape == (33280, 32, 56) and stream["obs"].dtype == np.float32
    assert np.isfinite(stream["obs"]).all()
    for key in FIELDS[1:]:
        assert stream[key].shape == (33280, 32, 2)
    assert stream["actions"].dtype == np.int64
    assert ((stream["actions"] >= 0) & (stream["actions"] <= 10)).all()
    assert stream["time_mask"].dtype == np.bool_ and stream["time_target"].dtype == np.float32
    assert np.isfinite(stream["time_target"]).all()
    assert (stream["time_target"][~stream["time_mask"]] == 0).all()
    assert independent_identity(stream, 0, 33280) == training["full_stream_identity"][teacher][str(seed)]
    return stream


def expected_work(stream, step):
    counts = independent_rows(stream, 512, 512 + 128 * step)
    return dict(warm_ticks=512, warm_lane_ticks=512 * 32, warm_loss_rows=0,
        training_updates=step, training_ticks=128 * step, training_lane_ticks=128 * step * 32,
        **counts, action_gradient_rows=counts["action_rows"], time_gradient_rows=counts["time_rows"])


def audit_case(seed, teacher, training, device):
    stream = independent_stream(seed, teacher, training)
    expected = independent_rows(stream, 512, 33280)
    exact(expected, training["expected_training_rows"][teacher][str(seed)], "declared available rows")
    original_path = ROOT / f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
    original = load(original_path)
    assert str(original_path) in training["input_sha256"][teacher][str(seed)]
    initial_states, first_states, result = {}, {}, {}
    historical = []
    for objective in OBJECTIVES:
        directory = folder(seed, teacher, objective)
        initial = load(directory / "step000.pt")
        assert initial["arm"] == objective and initial["seed"] == seed
        initial_states[objective] = initial
        assert initial["step"] == 0 and initial["stream_cursor"] == 512
        assert initial["warm_completed"] and not initial["optimizer"]["state"]
        assert initial["settings"] == dict(warm=512, block=128, updates=256)
        for key, value in original["model"].items():
            exact(value, initial["model"][key], "copied original native policy")
        for key in ("time_head.weight", "time_head.bias"):
            assert not bool(initial["model"][key].any())
        model = IndependentPolicy(initial, device)
        assert sum(p.numel() for p in model.parameters()) == 148632
        assert all(p.requires_grad for p in model.parameters())
        model.eval()
        with torch.no_grad():
            _, _, warm_h = model(torch.from_numpy(stream["obs"][:512]).to(device),
                                  torch.zeros(32, 192, device=device))
        exact(warm_h, initial["hidden"], "independent original512tick warm")
        del model
        updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
        assert len(updates) == 256
        sums = {key: 0 for key in expected}
        previous_h = state_digest(initial["hidden"])
        for step, row in enumerate(updates, 1):
            start, stop = 512 + (step - 1) * 128, 512 + step * 128
            assert row["arm"] == objective and row["seed"] == seed
            assert (row["step"], row["tick_start"], row["tick_stop"]) == (step, start, stop)
            assert row["input_sha256"] == independent_identity(stream, start, stop)
            assert row["time_features_detached"] == (objective == "action")
            assert row["numerical_state_carried"] and row["hidden_before_sha256"] == previous_h
            previous_h = row["hidden_after_sha256"]
            for key, value in independent_rows(stream, start, stop).items():
                assert row[key] == value
                sums[key] += value
        exact(sums, expected, "all256 actual update rows")
        actual = [reconstruct_update(directory, stream, step, device) for step in (1, 2)]
        for item in actual:
            exact(item["loss"], updates[item["step"] - 1]["loss"], "actual loss log")
            exact(item["preclip_grad_norm"], updates[item["step"] - 1]["preclip_grad_norm"], "actual norm log")
        first_states[objective] = load(directory / "step001_after.pt")
        snapshots = []
        for step in SNAPSHOTS:
            path = directory / f"step{step:03d}.pt"
            cp, metadata = load(path), read(path.with_suffix(".json"))
            assert sha(path) == metadata["sha256"]
            assert cp["step"] == step and cp["stream_cursor"] == 512 + 128 * step
            assert cp["arm"] == objective and cp["seed"] == seed and cp["warm_completed"]
            assert cp["protocol_hash"] == metadata["protocol_hash"] == training["protocol_hash"]
            expected_h = state_digest(initial["hidden"]) if step == 0 else updates[step - 1]["hidden_after_sha256"]
            assert state_digest(cp["hidden"]) == metadata["hidden_sha256"] == expected_h
            assert cp["hidden"].dtype == torch.float32 and tuple(cp["hidden"].shape) == (32, 192)
            assert bool(torch.isfinite(cp["hidden"]).all())
            exact(cp["bindings"], training["input_sha256"][teacher][str(seed)], "snapshot input bindings")
            exact(cp["source_hashes"], training["source_sha256"], "snapshot source bindings")
            exact(cp["work"], expected_work(stream, step), "snapshot actual work")
            exact(metadata["work"], cp["work"], "snapshot metadata work")
            for value in cp["model"].values():
                assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            assert len(cp["optimizer"]["param_groups"]) == 1
            group = cp["optimizer"]["param_groups"][0]
            assert len(group["params"]) == 8 and group["lr"] == .001
            assert group["betas"] == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0.
            assert len(cp["optimizer"]["state"]) == (8 if step else 0)
            assert all(int(value["step"]) == step for value in cp["optimizer"]["state"].values())
            if teacher == "earliest":
                old_path = ROOT / f"data/remaining-time/train-v1/seed_{seed:03d}/{objective}/step{step:03d}.pt"
                assert str(old_path) in training["input_sha256"][teacher][str(seed)]
                old = load(old_path)
                for key in REGRESSION_FIELDS:
                    exact(cp[key], old[key], f"historical/{seed}/{objective}/{step}/{key}")
                historical.append(dict(seed=seed, objective=objective, step=step, passed=True,
                    reference_sha256=sha(old_path), new_sha256=sha(path), fields=list(REGRESSION_FIELDS)))
            snapshots.append(dict(step=step, passed=True, checkpoint_sha256=sha(path),
                hidden_sha256=expected_h, work=cp["work"], binding_and_float32_checks=True))
        final = load(directory / "step256.pt")
        assert previous_h == state_digest(final["hidden"])
        for prefix in ("core.", "head."):
            assert any(not torch.equal(initial["model"][key], final["model"][key])
                       for key in initial["model"] if key.startswith(prefix))
        result[objective] = dict(passed=True, updates=256, rows=expected,
            actual_update_reconstruction=actual, snapshots=snapshots,
            final_sha256=sha(directory / "step256.pt"))
        print(json.dumps(dict(seed=seed, teacher=teacher, objective=objective, passed=True)), flush=True)
    for key in ("model", "hidden", "optimizer", "work"):
        exact(initial_states["action"][key], initial_states["joint"][key], "paired initialization/warm")
    for key in ("model", "optimizer", "hidden"):
        exact(first_states["action"][key], first_states["joint"][key], "zero-head paired first actual update")
    return dict(seed=seed, teacher=teacher, passed=True, objectives=result,
        paired_initial_and_first_update_bitwise_exact=True, historical_regression=historical)


def run(device="cuda"):
    audit_plan, training, freeze = checked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases = [audit_case(seed, teacher, training, device) for seed in SEEDS for teacher in TEACHERS]
    budgets = []
    for seed in SEEDS:
        rows = {case["teacher"]: case for case in cases if case["seed"] == seed}
        for objective in OBJECTIVES:
            left, right = [rows[teacher]["objectives"][objective]["snapshots"][-1]["work"]
                           for teacher in TEACHERS]
            for key in ("warm_ticks", "warm_lane_ticks", "warm_loss_rows", "training_updates",
                        "training_ticks", "training_lane_ticks", "action_rows", "action_gradient_rows"):
                exact(left[key], right[key], "paired teacher budget")
        budgets.append(dict(seed=seed, passed=True,
            counts={teacher: rows[teacher]["objectives"]["action"]["rows"] for teacher in TEACHERS},
            auxiliary_rows_equalized=False))
    assert sum(len(case["historical_regression"]) for case in cases) == 18
    write_new(OUT / "audit.json", dict(format="teacher-timing-independent-training-audit-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), device=device,
        training_protocol_hash=training["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        audit_plan_sha256=sha(PLAN), auditor_sha256=sha(__file__),
        scope=audit_plan["scope"], cases=cases, paired_teacher_budgets=budgets,
        actual_updates_reconstructed=24, snapshots_checked=36, historical_snapshots_exact=18,
        implementation_reused="Immutable independent audit primitives only; no experiment model, targets, learner, loss or update routine called"))
    print(json.dumps(dict(passed=True, path=str(OUT / "audit.json"), sha256=sha(OUT / "audit.json"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.device)
