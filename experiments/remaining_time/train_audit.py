"""Independently reconstruct actual native-policy remaining-time updates.

No remaining_time model, target, loss, learner, or training routine is called.
Plain PyTorch modules rebuild the archived first two CUDA steps in each arm;
all 256 block identities, counts, checkpoints and frozen policies are checked.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.continuity.run import state_digest

TRAIN = ROOT / "data/remaining-time/train-v1"
OUT = ROOT / "data/remaining-time/train-audit-v1"
ARMS = ("probe", "time", "action", "joint")
FIELDS = ("obs", "actions", "time_target", "time_mask")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def exact(a, b, name):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, name
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), name
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), name
        for key in a:
            exact(a[key], b[key], f"{name}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), name
        for index, (left, right) in enumerate(zip(a, b)):
            exact(left, right, f"{name}[{index}]")
    else:
        assert a == b, name


class IndependentPolicy(nn.Module):
    def __init__(self, cp, device="cuda"):
        super().__init__()
        assert cp["model_config"] == dict(width=192, input_size=56, n_jobs=2, n_actions=11)
        self.arm = cp["arm"]
        assert self.arm in ARMS
        self.core = nn.GRU(56, 192, num_layers=1)
        self.head = nn.Linear(192, 22)
        self.time_head = nn.Linear(192, 2)
        self.load_state_dict(cp["model"], strict=True)
        self.core.requires_grad_(self.arm != "probe")
        self.head.requires_grad_(self.arm in ("action", "joint"))
        self.to(device)

    def forward(self, obs, anchor):
        values, hidden = self.core(obs, anchor.unsqueeze(0))
        logits = F.linear(values, self.head.weight, self.head.bias).reshape(
            obs.shape[0], obs.shape[1], 2, 11)
        features = values.detach() if self.arm == "action" else values
        timing = F.linear(features, self.time_head.weight, self.time_head.bias)
        return logits, timing, hidden.squeeze(0)


def independent_rows(stream, start, stop):
    action = stream["actions"][start:stop]
    wait = int(np.count_nonzero(action == 0))
    return dict(action_rows=int(action.size), action_WAIT_rows=wait,
        action_BEGIN_rows=int(np.count_nonzero((action >= 1) & (action <= 2))),
        action_FINISH_rows=int(np.count_nonzero(action >= 3)),
        action_weight_sum=.25 * wait + int(action.size) - wait,
        time_rows=int(np.count_nonzero(stream["time_mask"][start:stop])))


def independent_identity(stream, start, stop):
    result = hashlib.sha256()
    for key in FIELDS:
        value = np.ascontiguousarray(stream[key][start:stop])
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        result.update(value.tobytes())
    return result.hexdigest()


def reconstruct_update(folder, stream, step, device):
    before = load(folder / f"step{step:03d}_before.pt")
    after = load(folder / f"step{step:03d}_after.pt")
    saved = load(folder / f"step{step:03d}_gradients.pt")
    model = IndependentPolicy(before, device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    start, stop = before["stream_cursor"], before["stream_cursor"] + 128
    assert start == 512 + 128 * (step - 1)
    batch = {name: torch.from_numpy(stream[name][start:stop]).to(device) for name in FIELDS}
    logits, prediction, hidden = model(batch["obs"], before["hidden"].to(device))
    weights = logits.new_ones(11)
    weights[0] = .25
    action = F.cross_entropy(logits.reshape(-1, 11), batch["actions"].reshape(-1), weight=weights)
    mask = batch["time_mask"]
    timing = (F.mse_loss(prediction[mask], batch["time_target"][mask])
              if bool(mask.any()) else prediction.reshape(-1)[:0].sum())
    total = action + timing if before["arm"] in ("action", "joint") else timing
    total.backward()
    gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                 for name, p in model.named_parameters()}
    exact(gradients, saved["preclip"], "preclip gradients")
    policy = [p for module in (model.core, model.head) for p in module.parameters() if p.requires_grad]
    norms = dict(policy=float(torch.nn.utils.clip_grad_norm_(policy, 1.)),
                 time_head=float(torch.nn.utils.clip_grad_norm_(model.time_head.parameters(), 1.)))
    clipped = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
               for name, p in model.named_parameters()}
    exact(clipped, saved["clipped"], "separately clipped gradients")
    exact(norms, saved["preclip_grad_norm"], "separate gradient norms")
    exact(hidden, saved["next_hidden"], "gradient artifact h")
    exact(hidden, after["hidden"], "carried preupdate h")
    optimizer.step()
    exact(model.state_dict(), after["model"], "actual postupdate parameters")
    exact(optimizer.state_dict(), after["optimizer"], "actual postupdate Adam")
    assert after["stream_cursor"] == stop and after["step"] == step
    assert all(g is None for key, g in gradients.items() if key.startswith("head.")) == (
        before["arm"] in ("probe", "time"))
    if before["arm"] == "probe":
        assert all(g is None for key, g in gradients.items() if key.startswith("core."))
    if before["arm"] == "time":
        core = [g for key, g in gradients.items() if key.startswith("core.")]
        assert all(g is not None for g in core)
        if step == 1:
            assert all(not bool(g.any()) for g in core)
        else:
            assert any(bool(g.any()) for g in core)
    return dict(step=step, passed=True, tensor_comparison="bitwise exact",
        loss={key: float(value.detach()) for key, value in
              (("total", total), ("action", action), ("time", timing))},
        preclip_grad_norm=norms, start=start, stop=stop,
        before_sha256=sha(folder / f"step{step:03d}_before.pt"),
        after_sha256=sha(folder / f"step{step:03d}_after.pt"),
        gradients_sha256=sha(folder / f"step{step:03d}_gradients.pt"))


def audit_seed(seed, plan, device):
    bound = list(map(Path, plan["input_sha256"][str(seed)]))
    raw_path = next(path for path in bound if path.name == "stream.npz")
    target_path = next(path for path in bound if path.name == "targets.npz")
    original_path = next(path for path in bound if path.name == "B008.pt")
    original = load(original_path)
    with np.load(raw_path, allow_pickle=False) as archive:
        stream = {name: archive[name] for name in ("obs", "actions")}
    with np.load(target_path, allow_pickle=False) as archive:
        stream.update({name: archive[name] for name in ("time_target", "time_mask")})
    assert stream["obs"].shape == (33280, 32, 56)
    assert stream["time_mask"].dtype == np.bool_
    assert stream["time_target"].dtype == np.float32
    expected = independent_rows(stream, 512, 33280)
    exact(expected, plan["expected_training_rows"][str(seed)], "declared available rows")
    initial_states, first_states, result = {}, {}, {}
    for arm in ARMS:
        folder = TRAIN / f"seed_{seed:03d}" / arm
        initial = load(folder / "step000.pt")
        initial_states[arm] = initial
        assert initial["step"] == 0 and initial["stream_cursor"] == 512
        assert initial["warm_completed"] and not initial["optimizer"]["state"]
        assert initial["protocol_hash"] == plan["protocol_hash"]
        exact(initial["bindings"], plan["input_sha256"][str(seed)], "checkpoint input bindings")
        exact(initial["source_hashes"], plan["source_sha256"], "checkpoint source bindings")
        for key, value in original["model"].items():
            exact(value, initial["model"][key], "copied original native policy")
        assert not bool(initial["model"]["time_head.weight"].any())
        assert not bool(initial["model"]["time_head.bias"].any())
        model = IndependentPolicy(initial, device)
        model.eval()
        with torch.no_grad():
            _, _, warm_h = model(torch.from_numpy(stream["obs"][:512]).to(device),
                                  torch.zeros(32, 192, device=device))
        exact(warm_h, initial["hidden"], "independent original512tick warm")
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        assert len(updates) == 256
        sums = {key: 0 for key in expected}
        previous_h = state_digest(initial["hidden"])
        for step, row in enumerate(updates, start=1):
            start, stop = 512 + (step - 1) * 128, 512 + step * 128
            assert (row["step"], row["tick_start"], row["tick_stop"]) == (step, start, stop)
            assert row["input_sha256"] == independent_identity(stream, start, stop)
            assert row["time_features_detached"] == (arm == "action")
            assert row["numerical_state_carried"]
            assert row["hidden_before_sha256"] == previous_h
            previous_h = row["hidden_after_sha256"]
            counts = independent_rows(stream, start, stop)
            for key, value in counts.items():
                assert row[key] == value
                sums[key] += value
        exact(sums, expected, "all256 actual update row counts")
        actual = [reconstruct_update(folder, stream, step, device) for step in (1, 2)]
        for item in actual:
            exact(item["loss"], updates[item["step"] - 1]["loss"], "actual loss log")
            exact(item["preclip_grad_norm"], updates[item["step"] - 1]["preclip_grad_norm"], "actual norm log")
        first_states[arm] = load(folder / "step001_after.pt")
        final = load(folder / "step256.pt")
        assert final["step"] == 256 and final["stream_cursor"] == 33280
        assert previous_h == state_digest(final["hidden"])
        for key, value in expected.items():
            assert final["work"][key] == value
        assert final["work"]["warm_ticks"] == 512 and final["work"]["warm_loss_rows"] == 0
        assert final["work"]["warm_lane_ticks"] == 512 * 32
        assert final["work"]["training_updates"] == 256
        assert final["work"]["training_ticks"] == 32768
        assert final["work"]["training_lane_ticks"] == 32768 * 32
        assert final["work"]["action_gradient_rows"] == (expected["action_rows"] if arm in ("action", "joint") else 0)
        assert final["work"]["time_gradient_rows"] == expected["time_rows"]
        equal = {}
        for prefix in ("core.", "head."):
            equal[prefix] = all(torch.equal(initial["model"][key], final["model"][key])
                                for key in initial["model"] if key.startswith(prefix))
        assert equal["core."] == (arm == "probe")
        assert equal["head."] == (arm in ("probe", "time"))
        assert all(int(value["step"]) == 256 for value in final["optimizer"]["state"].values())
        for step in (0, 64, 256):
            checkpoint = folder / f"step{step:03d}.pt"
            meta = read(checkpoint.with_suffix(".json"))
            cp = load(checkpoint)
            assert sha(checkpoint) == meta["sha256"]
            assert cp["step"] == step and cp["stream_cursor"] == 512 + 128 * step
            assert cp["protocol_hash"] == meta["protocol_hash"] == plan["protocol_hash"]
            assert meta["hidden_sha256"] == state_digest(cp["hidden"])
            exact(cp["bindings"], plan["input_sha256"][str(seed)], "snapshot bindings")
            exact(cp["source_hashes"], plan["source_sha256"], "snapshot source bindings")
        result[arm] = dict(passed=True, updates=256, rows=expected,
            actual_update_reconstruction=actual, frozen_core_equal=equal["core."],
            frozen_native_head_equal=equal["head."], final_sha256=sha(folder / "step256.pt"))
    for arm in ARMS[1:]:
        exact(initial_states["probe"]["model"], initial_states[arm]["model"], "paired initial function")
        exact(initial_states["probe"]["hidden"], initial_states[arm]["hidden"], "paired warm h")
        exact(first_states["probe"]["hidden"], first_states[arm]["hidden"], "paired first actual h")
        for key in ("time_head.weight", "time_head.bias"):
            exact(first_states["probe"]["model"][key], first_states[arm]["model"][key], "paired first timehead update")
    exact(first_states["probe"]["model"], first_states["time"]["model"], "zero-head first time core gradient")
    exact(first_states["action"]["model"], first_states["joint"]["model"], "paired first native update")
    exact(first_states["action"]["optimizer"], first_states["joint"]["optimizer"], "paired first native Adam")
    return dict(seed=seed, passed=True, arms=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    plan = read(TRAIN / "protocol.json")
    assert state_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, path
    for items in plan["input_sha256"].values():
        for path, expected in items.items():
            assert sha(path) == expected, path
    freeze = read(TRAIN / "globalfreeze.json")
    assert state_digest({key: value for key, value in freeze.items() if key != "freeze_hash"}) == freeze["freeze_hash"]
    assert freeze["protocol_hash"] == plan["protocol_hash"]
    assert freeze["endpoint_count"] == 12 and freeze["snapshot_count"] == 36
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    result = dict(format="remaining-time-independent-training-audit-v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), device=args.device,
        training_protocol_hash=plan["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        auditor_sha256=sha(__file__),
        scope="All12 initial/warm/endpoints and256 block identities/counts; independent first2 actual optimizer steps perarm; separate clipping; no experiment model/target/loss/update call",
        seeds=[audit_seed(seed, plan, args.device) for seed in (0, 1, 2)])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "audit.json").open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(passed=True, path=str(OUT / "audit.json"), sha256=sha(OUT / "audit.json"))))


if __name__ == "__main__":
    main()
