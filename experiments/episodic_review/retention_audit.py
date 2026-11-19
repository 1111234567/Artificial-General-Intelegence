"""Independent audit of the prospective fixed64 no-replay B suffix.

Only auditor utilities are shared with train_audit. No suffix learner, model,
loss, data collector, teacher or target construction routine is imported.
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
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.continuity.run import state_digest
from experiments.episodic_review.train_audit import (
    IndependentPolicy, exact, load, read, sha, write_new, identity, rows, loss,
    optimizer_steps, check_path_hashes, collect_path_hashes,
)

TRAIN = ROOT / "data/episodic-review/retention-train-v1"
DATA = ROOT / "data/episodic-review/retention-streams-v1"
PRIMARY = ROOT / "data/episodic-review/train-v1"
OUT = ROOT / "data/episodic-review/retention-audit-v1"
PLAN = OUT / "audit_plan.json"
ARMS, SEEDS, SNAPSHOTS = ("none", "replay_A", "replay_B"), (0, 1, 2), (0, 64)
FIELDS = ("obs", "actions", "time_target", "time_mask")


def folder(seed, arm):
    return TRAIN / f"seed_{seed:03d}" / arm


def source_checkpoint(seed, arm):
    return PRIMARY / f"seed_{seed:03d}" / arm / "step256.pt"


def raw_path(seed):
    return DATA / f"seed_{seed:03d}" / "stream.npz"


def states_path(seed):
    return raw_path(seed).with_name("world_states.pt")


def reconstruct_update(directory, stream, step, device="cuda"):
    before = load(directory / f"step{step:03d}_before.pt")
    after = load(directory / f"step{step:03d}_after.pt")
    saved = load(directory / f"step{step:03d}_gradients.pt")
    assert before["step"] == step-1 and after["step"] == step
    assert before["study_step"] == 255+step and after["study_step"] == 256+step
    start, stop = 128*(step-1), 128*step
    assert before["stream_cursor"] == start and after["stream_cursor"] == stop
    optimizer_steps(before["optimizer"], 512+step-1)
    model = IndependentPolicy(before, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    batch = {key: torch.from_numpy(stream[key][start:stop]).to(device) for key in FIELDS}
    model.train()
    logits, prediction, hidden = model(batch["obs"], before["hidden"].to(device))
    total, action, timing = loss(logits, prediction, batch)
    total.backward()
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    exact(gradients, saved["preclip"], "suffix independent preclip gradients")
    norms = dict(policy=float(torch.nn.utils.clip_grad_norm_(
        [parameter for module in (model.core, model.head) for parameter in module.parameters()], 1.)),
        time_head=float(torch.nn.utils.clip_grad_norm_(model.time_head.parameters(), 1.)))
    clipped = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
               for name, parameter in model.named_parameters()}
    exact(clipped, saved["clipped"], "suffix independently clipped gradients")
    exact(norms, saved["preclip_grad_norm"], "suffix independent split gradient norms")
    exact(hidden, saved["next_hidden"], "suffix live numerical h artifact")
    exact(hidden, after["hidden"], "suffix carry pre-update live h unchanged")
    optimizer.step()
    exact(model.state_dict(), after["model"], "suffix actual parameters after Adam")
    exact(optimizer.state_dict(), after["optimizer"], "suffix actual carried Adam moments")
    optimizer_steps(after["optimizer"], 512+step)
    return dict(step=step, study_step=256+step, passed=True, tensor_comparison="bitwise exact",
        optimizer_step_before=512+step-1, optimizer_step_after=512+step,
        loss={key: float(value.detach()) for key, value in (("total", total), ("action", action), ("time", timing))},
        preclip_grad_norm=norms,
        artifacts={name: sha(directory / f"step{step:03d}_{name}.pt") for name in ("before", "after", "gradients")})


def checked_training():
    training, freeze = read(TRAIN / "protocol.json"), read(TRAIN / "globalfreeze.json")
    primary_plan, primary_freeze = read(PRIMARY / "protocol.json"), read(PRIMARY / "globalfreeze.json")
    for value, key in ((training, "protocol_hash"), (freeze, "freeze_hash"),
                       (primary_plan, "protocol_hash"), (primary_freeze, "freeze_hash")):
        assert state_digest({name: item for name, item in value.items() if name != key}) == value[key]
    assert training["seeds"] == list(SEEDS) and training["arms"] == list(ARMS)
    assert training["snapshots"] == list(SNAPSHOTS)
    assert training["source_stream_cursor"] == 66048 and training["source_optimizer_step"] == 512
    assert training["primary_protocol_hash"] == primary_plan["protocol_hash"]
    assert training["primary_freeze_hash"] == primary_freeze["freeze_hash"]
    assert sha(PRIMARY / "protocol.json") == training["primary_protocol_sha256"]
    assert sha(PRIMARY / "globalfreeze.json") == training["primary_freeze_sha256"]
    assert sha(DATA / "protocol.json") == training["data_protocol_sha256"]
    assert freeze["protocol_hash"] == training["protocol_hash"]
    assert freeze["primary_freeze_hash"] == primary_freeze["freeze_hash"]
    assert freeze["endpoint_count"] == 9 and freeze["snapshot_count"] == 18 and freeze["study_step"] == 320
    assert freeze["heldout_scoring_started"] is False
    check_path_hashes(training["source_sha256"])
    check_path_hashes(training["input_sha256"])
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    return training, freeze


def declare():
    training, freeze = checked_training()
    paths = [TRAIN / "protocol.json", TRAIN / "globalfreeze.json", DATA / "protocol.json"]
    paths.extend(collect_path_hashes(training["input_sha256"]))
    paths.extend(collect_path_hashes(training["source_sha256"]))
    for seed in SEEDS:
        paths.extend((raw_path(seed), states_path(seed),
            ROOT / f"data/coverage-retention/streams-v1/seed_{seed:03d}/withdraw/world_states.pt"))
        for arm in ARMS:
            directory = folder(seed, arm)
            paths.extend((source_checkpoint(seed, arm), source_checkpoint(seed, arm).parent / "updates.jsonl",
                          directory / "updates.jsonl", directory / "completed.json"))
            for step in SNAPSHOTS:
                paths.extend(directory / f"step{step:03d}{ext}" for ext in (".pt", ".json"))
            for step in (1, 2):
                paths.extend(directory / f"step{step:03d}_{name}.pt" for name in ("before", "after", "gradients"))
    sources = (Path(__file__), Path(__file__).with_name("train_audit.py"),
        ROOT / "experiments/remaining_time/train_audit.py", ROOT / "experiments/continuity/run.py")
    write_new(PLAN, dict(format="episodic-review-prospective-retention-audit-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), training_protocol_hash=training["protocol_hash"],
        freeze_hash=freeze["freeze_hash"], inputs={str(path): sha(path) for path in dict.fromkeys(paths)},
        sources={str(path): sha(path) for path in sources},
        scope="All9 exact primary256 model/Adam512/h/RNG copies; all576 full B block identities/counts/h chains; all18 snapshots including final Adam576; all18 actual suffix001/002 CUDA updates reconstructed with plain GRU; all3 complete native source-world boundaries; original primary artifacts rehashed after suffix audit"))


def checked():
    plan = read(PLAN)
    for name in ("inputs", "sources"):
        check_path_hashes(plan[name])
    training, freeze = checked_training()
    assert plan["training_protocol_hash"] == training["protocol_hash"]
    assert plan["freeze_hash"] == freeze["freeze_hash"]
    return plan, training, freeze


def source_world_continuity(seed, stream):
    source = load(ROOT / f"data/coverage-retention/streams-v1/seed_{seed:03d}/withdraw/world_states.pt")["final"]
    states = load(states_path(seed))
    exact(states["initial"], source, "suffix complete source world copied without changes")
    assert states["initial"]["tick"] == 66047 and states["initial"]["acted"]
    assert states["final"]["tick"] == 74239 and states["final"]["acted"]
    with np.load(raw_path(seed), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["tick"], np.arange(8192))
        np.testing.assert_array_equal(archive["world_tick"], np.arange(66048, 74240))
        np.testing.assert_array_equal(archive["now"], np.arange(66048, 74240)*.25)
    first = stream["obs"][0].reshape(32, 2, 28)
    action = source["tensors"]["_ack_action"].numpy()
    result = source["tensors"]["_ack_result"].numpy()
    np.testing.assert_array_equal(first[..., 11:22], np.eye(11, dtype=np.float32)[action]*(action != 0)[..., None])
    np.testing.assert_array_equal(first[..., 22], result == 1)
    np.testing.assert_array_equal(first[..., 23], result == -1)
    return dict(passed=True, complete_native_world_boundary_exact=True, no_boundary_setters=True,
        first_new_tick=66048, final_new_tick=74239, previous_teacher_action_receipt_carried=True)


def expected_work(stream, step):
    counts = rows({name: value[:128*step] for name, value in stream.items()})
    return dict(warm_ticks=0, warm_lane_ticks=0, warm_loss_rows=0,
        training_updates=step, training_ticks=128*step, training_lane_ticks=128*step*32,
        **counts, action_gradient_rows=counts["action_rows"], time_gradient_rows=counts["time_rows"])


def audit_case(seed, arm, stream, training, device):
    directory = folder(seed, arm)
    source, initial = load(source_checkpoint(seed, arm)), load(directory / "step000.pt")
    for name in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng"):
        exact(initial[name], source[name], "suffix exact primary source "+name)
    assert state_digest(initial["numpy_rng"]) == state_digest(source["numpy_rng"])
    optimizer_steps(source["optimizer"], 512)
    primary_updates = [json.loads(line) for line in (source_checkpoint(seed, arm).parent / "updates.jsonl").read_text().splitlines()]
    assert primary_updates[-1]["hidden_after_sha256"] == state_digest(source["hidden"])
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 64
    previous_h = state_digest(source["hidden"])
    for step, record in enumerate(updates, 1):
        start, stop = 128*(step-1), 128*step
        assert record["seed"] == seed and record["arm"] == "action"
        assert (record["step"], record["tick_start"], record["tick_stop"]) == (step, start, stop)
        assert record["input_sha256"] == identity(stream, start, stop)
        assert record["time_features_detached"] and record["numerical_state_carried"]
        assert record["hidden_before_sha256"] == previous_h
        previous_h = record["hidden_after_sha256"]
        for name, value in rows({key: stream[key][start:stop] for key in FIELDS}).items():
            assert record[name] == value, "suffix exact full native labels/"+name
    reconstructed = [reconstruct_update(directory, stream, step, device) for step in (1, 2)]
    for actual in reconstructed:
        exact(actual["loss"], updates[actual["step"]-1]["loss"], "suffix actual independent loss log")
        exact(actual["preclip_grad_norm"], updates[actual["step"]-1]["preclip_grad_norm"], "suffix actual independent norm log")
    snapshots = []
    for step in SNAPSHOTS:
        path = directory / f"step{step:03d}.pt"
        cp, metadata = load(path), read(path.with_suffix(".json"))
        assert cp["primary_arm"] == cp["experiment_arm"] == arm and cp["seed"] == seed
        assert cp["arm"] == cp["objective"] == "action" and cp["actual_world_arm"] == "withdraw"
        assert cp["replay"] is False and cp["primary_step"] == 256 and cp["study_step"] == 256+step
        assert cp["step"] == cp["local_optimizer_updates"] == step
        assert cp["stream_cursor"] == 128*step and cp["warm_completed"]
        assert cp["settings"] == dict(warm=0, block=128, updates=64)
        assert cp["source_optimizer_step"] == 512 and cp["cumulative_optimizer_step"] == 512+step
        assert cp["source_stream_cursor"] == 66048 and cp["cumulative_stream_cursor"] == 66048+128*step
        assert cp["source_checkpoint"] == str(source_checkpoint(seed, arm))
        assert cp["protocol_hash"] == metadata["protocol_hash"] == training["protocol_hash"]
        assert sha(path) == metadata["sha256"]
        assert cp["native_world_path"] == str(states_path(seed))
        assert cp["native_world_stage"] == ("initial" if step == 0 else "final")
        assert cp["native_world_global_cursor"] == 66048+128*step
        exact(cp["source_hashes"], training["source_sha256"], "suffix source bindings")
        exact(cp["bindings"], training["input_sha256"][str(seed)][arm], "suffix input bindings")
        exact(cp["work"], expected_work(stream, step), "suffix independent exact work")
        exact(cp["source_work"], source["cumulative_work"], "full preceding512-update work preserved")
        exact(cp["primary_work"], source["work"], "primary local work preserved separately")
        exact(cp["source_replay_work"], source["replay_work"], "primary replay work preserved separately")
        exact(cp["cumulative_work"], {name: value+cp["work"][name] for name, value in source["cumulative_work"].items()}, "suffix cumulative work")
        expected_h = state_digest(source["hidden"]) if step == 0 else updates[step-1]["hidden_after_sha256"]
        assert state_digest(cp["hidden"]) == metadata["hidden_sha256"] == expected_h
        optimizer_steps(cp["optimizer"], 512+step)
        for value in list(cp["model"].values())+[cp["hidden"]]:
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        snapshots.append(dict(step=step, study_step=256+step, passed=True, checkpoint_sha256=sha(path),
                              hidden_sha256=expected_h, optimizer_step=512+step, work=cp["work"]))
    assert previous_h == snapshots[-1]["hidden_sha256"]
    print(json.dumps(dict(seed=seed, arm=arm, independent_suffix_training_audit_passed=True)), flush=True)
    return dict(seed=seed, arm=arm, passed=True, source_model_Adam512_h_RNG_exact=True,
                updates=64, replay=False, actual_update_reconstruction=reconstructed, snapshots=snapshots)


def run(device="cuda"):
    plan, training, freeze = checked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases, worlds = [], []
    for seed in SEEDS:
        with np.load(raw_path(seed), allow_pickle=False) as archive:
            stream = {key: archive[key] for key in FIELDS}
        assert stream["obs"].shape == (8192, 32, 56) and stream["obs"].dtype == np.float32
        for name in FIELDS[1:]:
            assert stream[name].shape == (8192, 32, 2)
        assert stream["actions"].dtype == np.int64 and stream["time_mask"].dtype == np.bool_
        assert stream["time_target"].dtype == np.float32 and np.isfinite(stream["time_target"]).all()
        exact(rows(stream), training["expected_live_rows"][str(seed)], "suffix declared full native rows")
        worlds.append(dict(seed=seed, **source_world_continuity(seed, stream)))
        cases.extend(audit_case(seed, arm, stream, training, device) for arm in ARMS)
    check_path_hashes(plan["inputs"])
    check_path_hashes(plan["sources"])
    write_new(OUT / "audit.json", dict(format="episodic-review-prospective-retention-audit-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), device=device,
        training_protocol_hash=training["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        primary_freeze_hash=training["primary_freeze_hash"],
        audit_plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), scope=plan["scope"],
        actual_updates_reconstructed=18, snapshots_checked=18,
        original_primary_artifacts_unchanged=True, cases=cases, native_world_continuity=worlds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.device)
