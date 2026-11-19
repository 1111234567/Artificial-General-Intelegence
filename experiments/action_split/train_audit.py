"""Independently reconstruct actual archived split-policy training updates.

This read-only auditor does not call the experiment model, loss, learner, or
update routines. It rebuilds a plain torch GRU and Linear from saved tensors,
then checks actual gradients, Adam, carried h, stream identities and budgets.
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

TRAIN = ROOT / "data/action-split/train-v1"
STREAMS = ROOT / "data/action-split/streams-v1"
OUT = ROOT / "data/action-split/train-audit-v1"
FIELDS = ("obs", "mode_target", "begin_target", "finish_target", "active")


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
        config = cp["model_config"]
        assert config == dict(width=192, input_size=56, n_jobs=2, n_actions=13)
        self.core = nn.GRU(56, 192, num_layers=1)
        self.head = nn.Linear(192, 26)
        self.load_state_dict(cp["model"], strict=True)
        self.core.requires_grad_(cp["arm"] == "full")
        self.to(device)

    def forward(self, obs, anchor):
        values, hidden = self.core(obs, anchor.unsqueeze(0))
        scores = F.linear(values, self.head.weight, self.head.bias)
        return scores.reshape(obs.shape[0], obs.shape[1], 2, 13), hidden.squeeze(0)


def independent_rows(stream, start, stop):
    modes = stream["mode_target"][start:stop]
    wait, begin, finish = [int(np.count_nonzero(modes == value)) for value in (0, 1, 2)]
    return dict(mode_rows=int(modes.size), mode_WAIT_rows=wait, mode_BEGIN_rows=begin,
                mode_FINISH_rows=finish, mode_weight_sum=.25 * wait + begin + finish,
                begin_rows=begin, finish_rows=int(np.count_nonzero(stream["active"][start:stop])))


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
    saved_grad = load(folder / f"step{step:03d}_gradients.pt")
    model = IndependentPolicy(before, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    start, stop = before["stream_cursor"], before["stream_cursor"] + 128
    assert start == 512 + 128 * (step - 1)
    values = {name: torch.from_numpy(stream[name][start:stop]).to(device) for name in FIELDS}
    model.train()
    scores, hidden = model(values["obs"], before["hidden"].to(device))
    flat = scores.reshape(-1, 13)
    modes = values["mode_target"].reshape(-1)
    begin_labels = values["begin_target"].reshape(-1)
    finish_labels = values["finish_target"].reshape(-1)
    begin_mask, active = modes.eq(1), values["active"].reshape(-1)
    # Direct PyTorch CE calls reproduce the declared objective independently;
    # no model.py loss code or train.py learner routine is called.
    mode = F.cross_entropy(flat[:, :3], modes, weight=flat.new_tensor([.25, 1., 1.]))
    begin = (F.cross_entropy(flat[begin_mask, 3:5], begin_labels[begin_mask])
             if bool(begin_mask.any()) else flat[:, 3:5].sum() * 0.)
    finish = (F.cross_entropy(flat[active, 5:], finish_labels[active])
              if bool(active.any()) else flat[:, 5:].sum() * 0.)
    total = mode + begin + finish
    total.backward()
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    exact(gradients, saved_grad["preclip"], "preclip gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    clipped = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
               for name, parameter in model.named_parameters()}
    exact(clipped, saved_grad["clipped"], "clipped gradients")
    assert float(norm) == saved_grad["preclip_grad_norm"]
    exact(hidden, saved_grad["next_hidden"], "gradient artifact h")
    exact(hidden, after["hidden"], "carried preupdate h")
    optimizer.step()
    exact(model.state_dict(), after["model"], "actual postupdate parameters")
    exact(optimizer.state_dict(), after["optimizer"], "actual postupdate Adam")
    assert after["stream_cursor"] == stop and after["step"] == step
    if step == 1:
        core_grads = [g for name, g in gradients.items() if name.startswith("core.")]
        if before["arm"] == "full":
            assert all(g is not None and not bool(g.any()) for g in core_grads)
        else:
            assert all(g is None for g in core_grads)
    if step == 2 and before["arm"] == "full":
        assert any(g is not None and bool(g.any()) for name, g in gradients.items()
                   if name.startswith("core."))
    return dict(step=step, passed=True, tensor_comparison="bitwise exact",
                loss={key: float(value.detach()) for key, value in
                      (("total", total), ("mode", mode), ("begin", begin), ("finish", finish))},
                preclip_grad_norm=float(norm), start=start, stop=stop,
                before_sha256=sha(folder / f"step{step:03d}_before.pt"),
                after_sha256=sha(folder / f"step{step:03d}_after.pt"),
                gradients_sha256=sha(folder / f"step{step:03d}_gradients.pt"))


def audit_seed(seed, plan, device):
    # The stream path is taken from the frozen training bindings, not guessed.
    stream_path = next(Path(path) for path in plan["input_sha256"][str(seed)] if path.endswith(".npz"))
    source_path = next(Path(path) for path in plan["input_sha256"][str(seed)] if path.endswith(".pt"))
    source = load(source_path)
    with np.load(stream_path, allow_pickle=False) as archive:
        stream = {name: archive[name] for name in FIELDS}
    assert stream["obs"].shape == (33280, 32, 56)
    expected = independent_rows(stream, 512, 33280)
    exact(expected, plan["expected_training_rows"][str(seed)], "declared row counts")
    results = {}
    initials, firsts = {}, {}
    for arm in ("frozen", "full"):
        folder = TRAIN / f"seed_{seed:03d}" / arm
        initial = load(folder / "step000.pt")
        initials[arm] = initial
        assert initial["step"] == 0 and initial["stream_cursor"] == 512
        assert not initial["optimizer"]["state"]
        assert not bool(initial["model"]["head.weight"].any())
        assert not bool(initial["model"]["head.bias"].any())
        for key, value in source["model"].items():
            if key.startswith("core."):
                exact(value, initial["model"][key], "copied pretrained core")
        warm_model = IndependentPolicy(initial, device)
        warm_model.eval()
        with torch.no_grad():
            _, warm_h = warm_model(torch.from_numpy(stream["obs"][:512]).to(device),
                                    torch.zeros(32, 192, device=device))
        exact(warm_h, initial["hidden"], "independent full warm history")
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        assert len(updates) == 256
        sums = {key: 0 for key in expected}
        for step, row in enumerate(updates, start=1):
            start, stop = 512 + (step - 1) * 128, 512 + step * 128
            assert row["step"] == step and row["tick_start"] == start and row["tick_stop"] == stop
            assert row["input_sha256"] == independent_identity(stream, start, stop)
            counts = independent_rows(stream, start, stop)
            for key, value in counts.items():
                assert row[key] == value
                sums[key] += value
        exact(sums, expected, "all256 actual update counts")
        actual = [reconstruct_update(folder, stream, step, device) for step in (1, 2)]
        for item in actual:
            exact(item["loss"], updates[item["step"] - 1]["loss"], "actual loss log")
            assert item["preclip_grad_norm"] == updates[item["step"] - 1]["preclip_grad_norm"]
        firsts[arm] = load(folder / "step001_after.pt")
        final = load(folder / "step256.pt")
        assert final["step"] == 256 and final["stream_cursor"] == 33280
        for key, value in expected.items():
            assert final["work"][key] == value
        assert final["work"]["warm_loss_rows"] == 0
        assert final["work"]["training_updates"] == 256
        core_equal = all(torch.equal(initial["model"][key], final["model"][key])
                         for key in initial["model"] if key.startswith("core."))
        assert core_equal == (arm == "frozen")
        assert all(int(state["step"]) == 256 for state in final["optimizer"]["state"].values())
        for step in (0, 64, 256):
            checkpoint = folder / f"step{step:03d}.pt"
            assert sha(checkpoint) == read(checkpoint.with_suffix(".json"))["sha256"]
        results[arm] = dict(passed=True, updates=256, rows=expected,
                            warm_h="bitwise exact independent512tick reconstruction",
                            actual_update_reconstruction=actual, frozen_core_equal=core_equal,
                            final_sha256=sha(folder / "step256.pt"))
    exact(initials["frozen"]["model"], initials["full"]["model"], "paired initial function")
    exact(initials["frozen"]["hidden"], initials["full"]["hidden"], "paired initial h")
    exact(firsts["frozen"]["model"], firsts["full"]["model"], "paired first actual update")
    exact(firsts["frozen"]["hidden"], firsts["full"]["hidden"], "paired first actual h")
    return dict(seed=seed, passed=True, arms=results)


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
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    result = dict(format="action-split-independent-training-audit-v1", passed=True,
                  created_utc=datetime.now(timezone.utc).isoformat(), device=args.device,
                  training_protocol_hash=plan["protocol_hash"], freeze_hash=freeze["freeze_hash"],
                  auditor_sha256=sha(__file__),
                  scope="All6 initial/warm/endpoints and256 update data identities/counts; independent first2 actual optimizer steps perarm; no experiment model/loss/update call",
                  seeds=[audit_seed(seed, plan, args.device) for seed in (0, 1, 2)])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "audit.json").open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(passed=True, path=str(OUT / "audit.json"), sha256=sha(OUT / "audit.json"))))


if __name__ == "__main__":
    main()
