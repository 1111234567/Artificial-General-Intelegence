"""Independent continuation, plain-GRU gradient and Adam reconstruction.

The source learner is never called. The trained time readout remains detached
from the policy, and the eight Adam slots continue at global update 256.
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
from torch import nn
from torch.nn import functional as F

from experiments.continuity.run import state_digest
from experiments.remaining_time.train_audit import (
    exact, independent_identity, independent_rows, load, read, sha,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/coverage-retention/train-v1"
DATA = ROOT / "data/coverage-retention/streams-v1"
OUT = ROOT / "data/coverage-retention/train-audit-v1"
PLAN = OUT / "audit_plan.json"
ARMS, SEEDS, SNAPSHOTS = ("control", "wide", "withdraw"), (0, 1, 2), (0, 64, 256)
FIELDS = ("obs", "actions", "time_target", "time_mask")


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class IndependentPolicy(nn.Module):
    """Original tensor layout; no experiment policy, chooser or loss helpers."""

    def __init__(self, cp, device="cuda"):
        super().__init__()
        assert cp["model_config"] == dict(width=192, input_size=56, n_jobs=2, n_actions=11)
        self.core = nn.GRU(56, 192, num_layers=1)
        self.head = nn.Linear(192, 22)
        self.time_head = nn.Linear(192, 2)
        self.load_state_dict(cp["model"], strict=True)
        self.to(device)

    def forward(self, obs, anchor):
        values, hidden = self.core(obs, anchor.unsqueeze(0))
        logits = F.linear(values, self.head.weight, self.head.bias).reshape(
            obs.shape[0], obs.shape[1], 2, 11)
        prediction = F.linear(values.detach(), self.time_head.weight, self.time_head.bias)
        return logits, prediction, hidden.squeeze(0)


def optimizer_steps(optimizer, expected):
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert len(group["params"]) == len(optimizer["state"]) == 8
    assert group["lr"] == .001 and group["betas"] == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0.
    assert all(int(value["step"]) == expected for value in optimizer["state"].values())


def reconstruct_update(folder, stream, step, device="cuda"):
    """Reconstruct all gradients, split clipping, Adam moments and live h."""
    before = load(folder / f"step{step:03d}_before.pt")
    after = load(folder / f"step{step:03d}_after.pt")
    saved = load(folder / f"step{step:03d}_gradients.pt")
    assert before["step"] == step - 1 and after["step"] == step
    start, stop = before["stream_cursor"], before["stream_cursor"] + 128
    assert (start, stop) == (128 * (step - 1), 128 * step)
    optimizer_steps(before["optimizer"], 256 + step - 1)
    model = IndependentPolicy(before, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    batch = {key: torch.from_numpy(stream[key][start:stop]).to(device) for key in FIELDS}
    logits, prediction, hidden = model(batch["obs"], before["hidden"].to(device))
    weights = logits.new_ones(11)
    weights[0] = .25
    action = F.cross_entropy(logits.reshape(-1, 11), batch["actions"].reshape(-1), weight=weights)
    mask = batch["time_mask"]
    timing = (F.mse_loss(prediction[mask], batch["time_target"][mask])
              if bool(mask.any()) else prediction.reshape(-1)[:0].sum())
    total = action + timing
    total.backward()
    gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                 for name, p in model.named_parameters()}
    exact(gradients, saved["preclip"], "continuation actual preclip gradients")
    norms = dict(policy=float(torch.nn.utils.clip_grad_norm_(
        [p for module in (model.core, model.head) for p in module.parameters()], 1.)),
        time_head=float(torch.nn.utils.clip_grad_norm_(model.time_head.parameters(), 1.)))
    clipped = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
               for name, p in model.named_parameters()}
    exact(clipped, saved["clipped"], "continuation separately clipped gradients")
    exact(norms, saved["preclip_grad_norm"], "continuation actual split gradient norms")
    exact(hidden, saved["next_hidden"], "continuation gradient artifact h")
    exact(hidden, after["hidden"], "carry pre-update numerical h")
    optimizer.step()
    exact(model.state_dict(), after["model"], "continuation actual postupdate parameters")
    exact(optimizer.state_dict(), after["optimizer"], "continuation actual postupdate Adam")
    optimizer_steps(after["optimizer"], 256 + step)
    assert after["stream_cursor"] == stop
    # A second direct action-only backward proves a trained, nonzero time head
    # still contributes no policy gradient under the detached-feature objective.
    policy_gradients = {key: value for key, value in gradients.items() if not key.startswith("time_head.")}
    model.zero_grad(set_to_none=True)
    model.load_state_dict(before["model"], strict=True)
    direct_logits, _, _ = model(batch["obs"], before["hidden"].to(device))
    F.cross_entropy(direct_logits.reshape(-1, 11), batch["actions"].reshape(-1), weight=weights).backward()
    direct = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
              for name, p in model.named_parameters() if not name.startswith("time_head.")}
    exact(direct, policy_gradients, "trained time readout has zero policy-gradient influence")
    return dict(step=step, passed=True, tensor_comparison="bitwise exact", start=start, stop=stop,
        optimizer_step_before=256+step-1, optimizer_step_after=256+step,
        loss={key: float(value.detach()) for key, value in
              (("total", total), ("action", action), ("time", timing))},
        preclip_grad_norm=norms, time_head_policy_gradient_independence=True,
        before_sha256=sha(folder / f"step{step:03d}_before.pt"),
        after_sha256=sha(folder / f"step{step:03d}_after.pt"),
        gradients_sha256=sha(folder / f"step{step:03d}_gradients.pt"))


def folder(seed, arm):
    return TRAIN / f"seed_{seed:03d}" / arm


def source_checkpoint(seed):
    return ROOT / f"data/teacher-timing/train-v1/midpoint/seed_{seed:03d}/action/step256.pt"


def source_world(seed):
    return ROOT / f"data/teacher-timing/streams-v1/seed_{seed:03d}/midpoint/world_states.pt"


def checked_training():
    training, freeze = read(TRAIN / "protocol.json"), read(TRAIN / "globalfreeze.json")
    for value, key in ((training, "protocol_hash"), (freeze, "freeze_hash")):
        assert state_digest({name: item for name, item in value.items() if name != key}) == value[key]
    assert training["seeds"] == list(SEEDS) and training["arms"] == list(ARMS)
    assert training["snapshots"] == list(SNAPSHOTS) and training["objective"] == "action"
    assert training["stream"]["warm_ticks"] == 0 and training["stream"]["source_stream_cursor"] == 33280
    assert freeze["protocol_hash"] == training["protocol_hash"]
    assert freeze["endpoint_count"] == 9 and freeze["snapshot_count"] == 27
    assert freeze["heldout_scoring_started"] is False
    for name, expected in training["source_sha256"].items():
        assert sha(name) == expected, name
    for arms in training["input_sha256"].values():
        for paths in arms.values():
            for name, expected in paths.items():
                assert sha(name) == expected, name
    for key in ("data_protocol", "source_training_protocol", "source_freeze"):
        assert sha(training[key+"_path"]) == training[key+"_sha256"]
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    return training, freeze


def declare():
    training, freeze = checked_training()
    paths = [TRAIN / name for name in ("protocol.json", "globalfreeze.json")]
    for seed in SEEDS:
        paths.extend((source_checkpoint(seed), source_world(seed),
            source_checkpoint(seed).parent / "updates.jsonl"))
        for arm in ARMS:
            paths.extend(map(Path, training["input_sha256"][str(seed)][arm]))
            directory = folder(seed, arm)
            paths.extend(directory / name for name in ("updates.jsonl", "completed.json"))
            for step in SNAPSHOTS:
                paths.extend(directory / f"step{step:03d}{ext}" for ext in (".pt", ".json"))
            for step in (1, 2):
                paths.extend(directory / f"step{step:03d}_{name}.pt" for name in ("before", "after", "gradients"))
    sources = (Path(__file__), ROOT / "experiments/remaining_time/train_audit.py",
               ROOT / "experiments/continuity/run.py")
    write_new(PLAN, dict(format="coverage-retention-independent-training-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), training_protocol_hash=training["protocol_hash"],
        freeze_hash=freeze["freeze_hash"], inputs={str(path): sha(path) for path in dict.fromkeys(paths)},
        sources={str(path): sha(path) for path in sources},
        scope="All9 exact source model/Adam/h/RNG copies and native-world boundaries; all2304 block identities/counts/h chains; all27 snapshots; all18 first/second actual CUDA updates including independent action-only gradient proof and Adam steps256+local; source/local/cumulative work"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for path, expected in plan[table].items():
            assert sha(path) == expected, path
    training, freeze = checked_training()
    assert training["protocol_hash"] == plan["training_protocol_hash"]
    assert freeze["freeze_hash"] == plan["freeze_hash"]
    return plan, training, freeze


def source_continuity(seed, arm, stream):
    source = load(source_world(seed))["final"]
    states = load(DATA / f"seed_{seed:03d}" / arm / "world_states.pt")
    before, after = states["before_change"], states["after_change"]
    exact(before, source, "full source native world preserved before intervention")
    assert before["tick"] == 33279 and before["acted"] and before["now"] == 8319.75
    # Check all undeclared leaves independently of the collector boundary helper.
    for name in set(before) - {"regime", "regime_id", "regime_history", "context"}:
        exact(before[name], after[name], f"boundary preserves {name}")
    for name in set(before["context"]) - {"family_allowed", "future_family", "history"}:
        exact(before["context"][name], after["context"][name], f"boundary preserves context/{name}")
    expected_duration = (8., 32.) if arm == "wide" else (8., 16.)
    assert after["regime"]["duration_range"] == expected_duration
    assert after["regime"]["gap_range"] == (16., 32.)
    assert after["regime_id"] == before["regime_id"] + 1
    assert after["regime_history"][:-1] == before["regime_history"]
    assert after["context"]["history"][:-1] == before["context"]["history"]
    assert after["context"]["family_allowed"] == (
        [before["context"]["family_allowed"][1]] if arm == "withdraw" else before["context"]["family_allowed"])
    first = stream["obs"][0].reshape(32, 2, 28)
    action, result = before["tensors"]["_ack_action"].numpy(), before["tensors"]["_ack_result"].numpy()
    np.testing.assert_array_equal(first[..., 11:22], np.eye(11, dtype=np.float32)[action]*(action != 0)[..., None])
    np.testing.assert_array_equal(first[..., 22], result == 1)
    np.testing.assert_array_equal(first[..., 23], result == -1)
    with np.load(DATA / f"seed_{seed:03d}" / arm / "stream.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["world_tick"], np.arange(33280, 66048))
        np.testing.assert_array_equal(archive["now"], np.arange(33280, 66048)*.25)
    assert states["final"]["tick"] == 66047 and states["final"]["acted"]
    return dict(passed=True, before_change_source_exact=True, source_tick=33279,
        first_new_tick=33280, final_new_tick=66047, previous_action_receipt_carried=True,
        unchanged_native_state_digest=state_digest({key: value for key, value in before.items()
            if key not in ("regime", "regime_id", "regime_history", "context")}),
        source_world_sha256=sha(source_world(seed)))


def expected_work(stream, step):
    counts = independent_rows(stream, 0, 128*step)
    return dict(warm_ticks=0, warm_lane_ticks=0, warm_loss_rows=0,
        training_updates=step, training_ticks=128*step, training_lane_ticks=128*step*32,
        **counts, action_gradient_rows=counts["action_rows"], time_gradient_rows=counts["time_rows"])


def audit_case(seed, arm, training, device):
    with np.load(DATA / f"seed_{seed:03d}" / arm / "stream.npz", allow_pickle=False) as archive:
        stream = {key: archive[key] for key in FIELDS}
    assert stream["obs"].shape == (32768, 32, 56) and stream["obs"].dtype == np.float32
    assert np.isfinite(stream["obs"]).all() and np.isfinite(stream["time_target"]).all()
    for key in FIELDS[1:]:
        assert stream[key].shape == (32768, 32, 2)
    assert stream["actions"].dtype == np.int64 and stream["time_mask"].dtype == np.bool_
    assert stream["time_target"].dtype == np.float32 and (stream["time_target"][~stream["time_mask"]] == 0).all()
    assert ((stream["actions"] >= 0) & (stream["actions"] <= 10)).all()
    assert independent_identity(stream, 0, 32768) == training["full_stream_identity"][str(seed)][arm]
    counts = independent_rows(stream, 0, 32768)
    exact(counts, training["expected_training_rows"][str(seed)][arm], "declared new rows")
    directory = folder(seed, arm)
    source, initial = load(source_checkpoint(seed)), load(directory / "step000.pt")
    for key in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng"):
        exact(initial[key], source[key], f"full source {key} copied exactly")
    assert state_digest(initial["numpy_rng"]) == state_digest(source["numpy_rng"])
    optimizer_steps(source["optimizer"], 256)
    assert source["step"] == 256 and source["stream_cursor"] == 33280
    old_updates = [json.loads(line) for line in (source_checkpoint(seed).parent / "updates.jsonl").read_text().splitlines()]
    assert old_updates[-1]["hidden_after_sha256"] == state_digest(source["hidden"])
    assert old_updates[-1]["tick_stop"] == 33280
    native = source_continuity(seed, arm, stream)
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 256
    previous_h = state_digest(source["hidden"])
    total = {key: 0 for key in counts}
    for step, row in enumerate(updates, 1):
        start, stop = 128*(step-1), 128*step
        assert row["arm"] == "action" and row["seed"] == seed
        assert (row["step"], row["tick_start"], row["tick_stop"]) == (step, start, stop)
        assert row["input_sha256"] == independent_identity(stream, start, stop)
        assert row["time_features_detached"] and row["numerical_state_carried"]
        assert row["hidden_before_sha256"] == previous_h
        previous_h = row["hidden_after_sha256"]
        for key, value in independent_rows(stream, start, stop).items():
            assert row[key] == value
            total[key] += value
    exact(total, counts, "all256 actual row counts")
    reconstructed = [reconstruct_update(directory, stream, step, device) for step in (1, 2)]
    for actual in reconstructed:
        exact(actual["loss"], updates[actual["step"]-1]["loss"], "actual continuation loss log")
        exact(actual["preclip_grad_norm"], updates[actual["step"]-1]["preclip_grad_norm"], "actual continuation norm log")
    snapshots = []
    for step in SNAPSHOTS:
        path = directory / f"step{step:03d}.pt"
        cp, metadata = load(path), read(path.with_suffix(".json"))
        assert cp["experiment_arm"] == arm and cp["objective"] == cp["arm"] == "action" and cp["seed"] == seed
        assert cp["step"] == cp["local_optimizer_updates"] == step
        assert cp["stream_cursor"] == 128*step and cp["warm_completed"]
        assert cp["settings"] == dict(warm=0, block=128, updates=256)
        assert cp["source_optimizer_step"] == 256 and cp["cumulative_optimizer_step"] == 256+step
        assert cp["source_stream_cursor"] == 33280 and cp["cumulative_stream_cursor"] == 33280+128*step
        assert cp["source_checkpoint"] == str(source_checkpoint(seed))
        assert sha(path) == metadata["sha256"]
        assert cp["protocol_hash"] == metadata["protocol_hash"] == training["protocol_hash"]
        exact(cp["bindings"], training["input_sha256"][str(seed)][arm], "snapshot input bindings")
        exact(cp["source_hashes"], training["source_sha256"], "snapshot source bindings")
        exact(cp["work"], expected_work(stream, step), "new work counters")
        exact(cp["source_work"], source["work"], "untouched source work counters")
        exact(cp["cumulative_work"], {key: value+cp["work"][key] for key, value in source["work"].items()}, "cumulative work counters")
        expected_h = state_digest(source["hidden"]) if step == 0 else updates[step-1]["hidden_after_sha256"]
        assert state_digest(cp["hidden"]) == metadata["hidden_sha256"] == expected_h
        assert cp["hidden"].dtype == torch.float32 and tuple(cp["hidden"].shape) == (32, 192)
        assert bool(torch.isfinite(cp["hidden"]).all())
        for value in cp["model"].values():
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        optimizer_steps(cp["optimizer"], 256+step)
        snapshots.append(dict(step=step, passed=True, checkpoint_sha256=sha(path),
            hidden_sha256=expected_h, optimizer_step=256+step, work=cp["work"]))
    assert previous_h == snapshots[-1]["hidden_sha256"]
    result = dict(seed=seed, arm=arm, passed=True, source_world_continuity=native,
        source_model_Adam_h_RNG_bitwise_exact=True, rows=counts,
        updates=256, actual_update_reconstruction=reconstructed, snapshots=snapshots)
    print(json.dumps(dict(seed=seed, arm=arm, training_audit_passed=True)), flush=True)
    return result


def run(device="cuda"):
    plan, training, freeze = checked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases = [audit_case(seed, arm, training, device) for seed in SEEDS for arm in ARMS]
    paired = []
    for seed in SEEDS:
        initial = [load(folder(seed, arm) / "step000.pt") for arm in ARMS]
        for key in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "work", "source_work"):
            for other in initial[1:]:
                exact(initial[0][key], other[key], f"three paired start states/{key}")
        rows = [case for case in cases if case["seed"] == seed]
        assert len({row["rows"]["action_rows"] for row in rows}) == 1
        paired.append(dict(seed=seed, passed=True, start_states="bitwise exact", new_action_rows=2097152,
            new_updates=256, valid_auxiliary_rows_and_request_counts_equalized=False))
    write_new(OUT / "audit.json", dict(format="coverage-retention-independent-training-audit-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), device=device,
        training_protocol_hash=training["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        audit_plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), scope=plan["scope"],
        actual_updates_reconstructed=18, snapshots_checked=27, cases=cases, paired_budgets=paired))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.device)
