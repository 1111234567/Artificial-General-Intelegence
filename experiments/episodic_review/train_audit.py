"""Independent plain-GRU audit of episodic rehearsal continuations.

No episodic-review learner, model, bank loader, loss, or prefix-state helper is
called. Actual CUDA updates are rebuilt from raw files with plain nn.GRU,
masked native CE, a detached diagnostic head and the carried Adam moments.
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
from experiments.remaining_time.train_audit import exact, load, read, sha

TRAIN = ROOT / "data/episodic-review/train-v1"
BANK = ROOT / "data/episodic-review/bank-v1"
OUT = ROOT / "data/episodic-review/train-audit-v1"
PLAN = OUT / "audit_plan.json"
OLD_TRAIN = ROOT / "data/coverage-retention/train-v1"
OLD_DATA = ROOT / "data/coverage-retention/streams-v1"
ARMS, SEEDS, SNAPSHOTS = ("none", "replay_A", "replay_B"), (0, 1, 2), (0, 64, 256)
FIELDS = ("obs", "actions", "time_target", "time_mask")
BANK_FIELDS = FIELDS + ("action_mask",)
VECTORS = ("anchor_start", "actor_lane", "target_slot", "rid", "raw", "family",
           "version", "cue_kind", "cue_tick", "finish_tick", "supervised_length",
           "first_legal_offset", "pair_id")


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def folder(seed, arm):
    return TRAIN / f"seed_{seed:03d}" / arm


def source_checkpoint(seed):
    return ROOT / f"data/teacher-timing/train-v1/midpoint/seed_{seed:03d}/action/step256.pt"


def history_path(seed):
    return ROOT / f"data/teacher-timing/streams-v1/seed_{seed:03d}/midpoint/stream.npz"


def bank_path(seed):
    return BANK / f"seed_{seed:03d}" / "bank.npz"


def live_path(seed):
    return OLD_DATA / f"seed_{seed:03d}" / "withdraw/stream.npz"


def array_identity(values, fields):
    digest = hashlib.sha256()
    for name in fields:
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def identity(stream, start, stop):
    return array_identity({key: stream[key][start:stop] for key in FIELDS}, FIELDS)


def rows(values, masked=False):
    action = values["actions"]
    if masked:
        action = action[values["action_mask"]]
    wait = int(np.count_nonzero(action == 0))
    return dict(action_rows=int(action.size), action_WAIT_rows=wait,
        action_BEGIN_rows=int(np.count_nonzero((action == 1) | (action == 2))),
        action_FINISH_rows=int(np.count_nonzero(action >= 3)),
        action_weight_sum=.25 * wait + int(action.size) - wait,
        time_rows=int(np.count_nonzero(values["time_mask"])))


class IndependentPolicy(nn.Module):
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
        time = F.linear(values.detach(), self.time_head.weight, self.time_head.bias)
        return logits, time, hidden.squeeze(0)


def optimizer_steps(optimizer, expected):
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert len(group["params"]) == len(optimizer["state"]) == 8
    assert group["lr"] == .001 and group["betas"] == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0.
    assert all(int(value["step"]) == expected for value in optimizer["state"].values())


def rebuild_anchors(model, history, lanes, starts, device):
    """Current weights from h0; no archive h and no cross-update cache.

    Fixed batch/chunk boundaries match the declared numerical protocol, while
    this implementation uses only a fresh plain nn.GRU and the original obs.
    """
    lanes, starts = np.asarray(lanes, np.int64), np.asarray(starts, np.int64)
    unique = np.unique(lanes)
    maximum = int(starts.max())
    boundaries = sorted({0, maximum, *map(int, starts), *range(512, maximum, 512)})
    state = torch.zeros(1, len(unique), 192, device=device)
    anchors = torch.empty(len(lanes), 192, device=device)
    lookup = {int(lane): index for index, lane in enumerate(unique)}
    model.eval()
    with torch.no_grad():
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            inputs = torch.from_numpy(np.ascontiguousarray(history[low:high, unique])).to(device)
            _, state = model.core(inputs, state)
            state = state.detach()
            for index in np.flatnonzero(starts == high):
                anchors[int(index)] = state[0, lookup[int(lanes[index])]]
        for index in np.flatnonzero(starts == 0):
            anchors[int(index)].zero_()
    return anchors.detach(), dict(prefix_lane_ticks=maximum * len(unique),
        prefix_chunks=len(boundaries)-1, unique_lanes=len(unique), maximum_anchor=maximum)


def selected_episode(bank, arm, step):
    selected = bank["schedule"][(step // 8)-1]
    group = bank[arm[-1]]
    result = {name: np.ascontiguousarray(group[name][:, selected]) for name in BANK_FIELDS}
    result.update({name: group[name][selected].copy() for name in VECTORS})
    return selected, result


def loss(logits, prediction, batch):
    weights = logits.new_ones(11)
    weights[0] = .25
    if "action_mask" in batch:
        action_mask = batch["action_mask"]
        action = F.cross_entropy(logits[action_mask], batch["actions"][action_mask], weight=weights)
    else:
        action = F.cross_entropy(logits.reshape(-1, 11), batch["actions"].reshape(-1), weight=weights)
    mask = batch["time_mask"]
    time = F.mse_loss(prediction[mask], batch["time_target"][mask]) if bool(mask.any()) else prediction.reshape(-1)[:0].sum()
    return action + time, action, time


def reconstruct_update(directory, stream, history, bank, arm, step, device):
    before = load(directory / f"step{step:03d}_before.pt")
    after = load(directory / f"step{step:03d}_after.pt")
    saved = load(directory / f"step{step:03d}_gradients.pt")
    assert before["step"] == step-1 and after["step"] == step
    start, stop = 128*(step-1), 128*step
    assert before["stream_cursor"] == start and after["stream_cursor"] == stop
    optimizer_steps(before["optimizer"], 256+step-1)
    model = IndependentPolicy(before, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    live_batch = {key: torch.from_numpy(stream[key][start:stop]).to(device) for key in FIELDS}
    replay = arm != "none" and step % 8 == 0
    reconstruction = None
    if replay:
        model.train()
        with torch.no_grad():
            _, _, live_hidden = model(live_batch["obs"], before["hidden"].to(device))
        selected, episode = selected_episode(bank, arm, step)
        np.testing.assert_array_equal(selected, np.asarray(saved["selected_pair_indices"]))
        anchor, reconstruction = rebuild_anchors(model, history, episode["actor_lane"], episode["anchor_start"], device)
        exact(anchor, saved["replay_anchor_hidden"], "current model full-history anchor reconstruction")
        np.testing.assert_array_equal(episode["action_mask"], saved["replay_action_mask"])
        np.testing.assert_array_equal(episode["time_mask"], saved["replay_time_mask"])
        exact(rows(episode, True), saved["masked_rows"], "independent actual masked gradient rows")
        assert saved["prefix"]["prefix_lane_ticks"] == reconstruction["prefix_lane_ticks"]
        assert saved["prefix"]["prefix_chunks"] == reconstruction["prefix_chunks"]
        batch = {key: torch.from_numpy(episode[key]).to(device) for key in BANK_FIELDS}
        model.train()
        logits, prediction, replay_hidden = model(batch["obs"], anchor)
        exact(replay_hidden, saved["replay_next_hidden"], "independent replay final h")
    else:
        batch = live_batch
        model.train()
        logits, prediction, live_hidden = model(batch["obs"], before["hidden"].to(device))
    total, action, timing = loss(logits, prediction, batch)
    total.backward()
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    exact(gradients, saved["preclip"], "actual preclip gradients")
    norms = dict(policy=float(torch.nn.utils.clip_grad_norm_(
        [parameter for module in (model.core, model.head) for parameter in module.parameters()], 1.)),
        time_head=float(torch.nn.utils.clip_grad_norm_(model.time_head.parameters(), 1.)))
    clipped = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
               for name, parameter in model.named_parameters()}
    exact(clipped, saved["clipped"], "separately clipped gradients")
    exact(norms, saved["preclip_grad_norm"], "independent split gradient norms")
    exact(live_hidden, saved["next_hidden"], "live B forward state")
    exact(live_hidden, after["hidden"], "only live B state carried across optimizer")
    optimizer.step()
    exact(model.state_dict(), after["model"], "actual parameters after Adam")
    exact(optimizer.state_dict(), after["optimizer"], "actual carried Adam moments")
    optimizer_steps(after["optimizer"], 256+step)
    # A second CE-only backward proves that the nonzero diagnostic head cannot
    # influence the native policy gradient, including in the masked replay arm.
    model.zero_grad(set_to_none=True)
    model.load_state_dict(before["model"], strict=True)
    direct_anchor = anchor if replay else before["hidden"].to(device)
    direct_logits, _, _ = model(batch["obs"], direct_anchor)
    weights = direct_logits.new_ones(11)
    weights[0] = .25
    if replay:
        mask = batch["action_mask"]
        F.cross_entropy(direct_logits[mask], batch["actions"][mask], weight=weights).backward()
    else:
        F.cross_entropy(direct_logits.reshape(-1, 11), batch["actions"].reshape(-1), weight=weights).backward()
    direct = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
              for name, parameter in model.named_parameters() if not name.startswith("time_head.")}
    exact(direct, {name: gradient for name, gradient in gradients.items() if not name.startswith("time_head.")},
          "detached time objective has no native-policy gradient influence")
    return dict(step=step, gradient_source=arm[-1] if replay else "live_B", passed=True,
        tensor_comparison="bitwise exact", optimizer_step_before=256+step-1, optimizer_step_after=256+step,
        loss={key: float(value.detach()) for key, value in (("total", total), ("action", action), ("time", timing))},
        preclip_grad_norm=norms, current_history_reconstruction=reconstruction,
        live_hidden_carried=True, time_head_policy_gradient_independence=True,
        artifacts={name: sha(directory / f"step{step:03d}_{name}.pt") for name in ("before", "after", "gradients")})


def load_arrays(seed):
    bank_plan = read(BANK / "protocol.json")
    specs = bank_plan["family_specs"][str(seed)]
    with np.load(live_path(seed), allow_pickle=False) as archive:
        stream = {key: archive[key] for key in FIELDS}
    with np.load(history_path(seed), allow_pickle=False) as archive:
        history = archive["obs"]
        historical = {key: archive[key] for key in
            ("actions", "time_mask", "time_target", "active", "started", "event_kind", "receipts",
             "rid", "raw", "version", "family", "event_duration", "now")}
    with np.load(bank_path(seed), allow_pickle=False) as archive:
        bank = {group: {name: archive[f"{group}__{name}"] for name in BANK_FIELDS+VECTORS}
                for group in ("A", "B")}
        bank["schedule"] = archive["schedule"]
    assert stream["obs"].shape == (32768, 32, 56) and history.shape == (33280, 32, 56)
    assert stream["obs"].dtype == history.dtype == np.float32
    assert bank["schedule"].shape == (32, 32) and bank["schedule"].dtype == np.int64
    for order in bank["schedule"].reshape(4, 256):
        np.testing.assert_array_equal(np.sort(order), np.arange(256))
    for group in ("A", "B"):
        assert bank[group]["obs"].shape == (128, 256, 56)
        for name in BANK_FIELDS[1:]:
            assert bank[group][name].shape == (128, 256, 2)
        assert bank[group]["action_mask"].dtype == bank[group]["time_mask"].dtype == np.bool_
        assert (bank[group]["time_mask"] <= bank[group]["action_mask"]).all()
        np.testing.assert_array_equal(np.bincount(bank[group]["raw"], minlength=4), [64]*4)
        assert np.all(bank[group]["family"] == specs["history_cue" if group == "A" else "B_cue"])
        for index in range(256):
            target = int(bank[group]["target_slot"][index])
            mask = bank[group]["action_mask"][:, index]
            expected = np.zeros((128, 2), np.bool_)
            length = int(bank[group]["supervised_length"][index])
            assert 1 < length <= 96
            expected[32:, target] = True
            np.testing.assert_array_equal(mask, expected)
            lane = int(bank[group]["actor_lane"][index])
            anchor = int(bank[group]["anchor_start"][index])
            cue = int(bank[group]["cue_tick"][index])
            finish = int(bank[group]["finish_tick"][index])
            assert cue >= 512 and cue-anchor == 32 and finish-cue+1 == length
            assert anchor >= 0 and anchor+128 <= 33280
            np.testing.assert_array_equal(bank[group]["obs"][:, index], history[anchor:anchor+128, lane])
            np.testing.assert_array_equal(bank[group]["actions"][:, index], historical["actions"][anchor:anchor+128, lane])
            native_mask = historical["time_mask"][anchor:anchor+128, lane] & expected
            np.testing.assert_array_equal(bank[group]["time_mask"][:, index], native_mask)
            np.testing.assert_array_equal(bank[group]["time_target"][:, index],
                np.where(native_mask, historical["time_target"][anchor:anchor+128, lane], np.float32(0)))
            for name in ("rid", "raw", "version", "family"):
                assert np.all(historical[name][cue:finish+1, lane, target] == bank[group][name][index])
            assert historical["event_kind"][cue, lane, target] == bank[group]["cue_kind"][index] in (1, 2)
            assert historical["actions"][cue, lane, target] == 1+bank[group]["version"][index]
            payload = specs[group][int(bank[group]["raw"][index])]
            assert historical["actions"][finish, lane, target] == 3+4*bank[group]["version"][index]+payload
            assert historical["receipts"][cue, lane, target] == historical["receipts"][finish, lane, target] == 1
            assert not np.any(historical["actions"][cue+1:finish, lane, target])
            assert np.all(historical["active"][cue:finish+1, lane, target])
            assert not historical["started"][cue, lane, target]
            assert np.all(historical["started"][cue+1:finish+1, lane, target])
            duration = float(historical["event_duration"][cue, lane, target])
            assert bank[group]["first_legal_offset"][index] == int(np.ceil(duration/.25-1e-6))
            derived_time = np.zeros((128, 2), np.float32)
            derived_time[33:32+length, target] = ((historical["now"][cue]+duration -
                historical["now"][cue+1:finish+1])/16.).astype(np.float32)
            np.testing.assert_array_equal(bank[group]["time_target"][:, index], derived_time)
            for name in ("active", "started", "event_kind", "actions"):
                assert not np.any(historical[name][finish+1:anchor+128, lane, target]), "native inactive WAIT tail/"+name
    for name in ("action_mask", "time_mask", "raw", "version", "cue_kind", "supervised_length", "first_legal_offset", "target_slot", "pair_id"):
        np.testing.assert_array_equal(bank["A"][name], bank["B"][name], err_msg="paired "+name)
    skeleton = []
    for group in ("A", "B"):
        code = bank[group]["actions"]
        skeleton.append(np.where(code == 0, 0, np.where(code <= 2, 1, 2))[bank[group]["action_mask"]])
    np.testing.assert_array_equal(*skeleton)
    np.testing.assert_array_equal(np.sign(bank["A"]["time_target"]), np.sign(bank["B"]["time_target"]))
    return stream, history, bank


def check_path_hashes(tree):
    """Walk only declared path-to-SHA tables, without importing the runner."""
    for key, value in tree.items():
        if isinstance(value, dict):
            check_path_hashes(value)
        else:
            path = Path(key)
            if not path.is_absolute():
                path = ROOT / path
            assert isinstance(value, str) and len(value) == 64, key
            assert sha(path) == value, f"declared source/input changed: {path}"


def collect_path_hashes(tree):
    result = []
    for key, value in tree.items():
        if isinstance(value, dict):
            result.extend(collect_path_hashes(value))
        else:
            path = Path(key)
            result.append(path if path.is_absolute() else ROOT / path)
    return result


def checked_training():
    training = read(TRAIN / "protocol.json")
    freeze = read(TRAIN / "globalfreeze.json")
    for value, key in ((training, "protocol_hash"), (freeze, "freeze_hash")):
        assert state_digest({name: item for name, item in value.items() if name != key}) == value[key]
    assert training["seeds"] == list(SEEDS) and training["arms"] == list(ARMS)
    assert training["snapshots"] == list(SNAPSHOTS)
    assert freeze["protocol_hash"] == training["protocol_hash"]
    assert freeze["endpoint_count"] == 9 and freeze["snapshot_count"] == 27
    assert freeze["heldout_scoring_started"] is False
    check_path_hashes(training["source_sha256"])
    check_path_hashes(training["input_sha256"])
    for key, value in training.items():
        if key.endswith("_path") and key[:-5]+"_sha256" in training:
            assert sha(value) == training[key[:-5]+"_sha256"], key
    for endpoint in freeze["endpoints"].values():
        assert sha(endpoint["path"]) == endpoint["sha256"]
        for snapshot in endpoint["snapshots"].values():
            assert sha(snapshot["path"]) == snapshot["sha256"]
            assert sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    return training, freeze


def declare():
    training, freeze = checked_training()
    paths = [TRAIN / "protocol.json", TRAIN / "globalfreeze.json", BANK / "protocol.json"]
    paths.extend(collect_path_hashes(training["input_sha256"]))
    paths.extend(collect_path_hashes(training["source_sha256"]))
    for seed in SEEDS:
        paths.extend((source_checkpoint(seed), source_checkpoint(seed).parent / "updates.jsonl",
            history_path(seed), live_path(seed), bank_path(seed),
            ROOT / f"data/teacher-timing/streams-v1/seed_{seed:03d}/midpoint/world_states.pt",
            OLD_DATA / f"seed_{seed:03d}/withdraw/world_states.pt"))
        for step in SNAPSHOTS:
            paths.append(OLD_TRAIN / f"seed_{seed:03d}/withdraw/step{step:03d}.pt")
        for arm in ARMS:
            directory = folder(seed, arm)
            paths.extend(directory / name for name in ("updates.jsonl", "accounting.jsonl", "completed.json"))
            for step in SNAPSHOTS:
                paths.extend(directory / f"step{step:03d}{ext}" for ext in (".pt", ".json"))
            for step in (1, 8):
                paths.extend(directory / f"step{step:03d}_{name}.pt" for name in ("before", "after", "gradients"))
    sources = (Path(__file__), ROOT / "experiments/remaining_time/train_audit.py",
               ROOT / "experiments/continuity/run.py")
    write_new(PLAN, dict(format="episodic-review-independent-training-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), training_protocol_hash=training["protocol_hash"],
        freeze_hash=freeze["freeze_hash"], inputs={str(path): sha(path) for path in dict.fromkeys(paths)},
        sources={str(path): sha(path) for path in sources},
        scope="All9 exact source model/Adam/h/RNG copies; all2304 live B block identities/counts/h chains; all192 paired replay batches and current-prefix work counts; all27 snapshots; all18 step001/008 actual CUDA updates reconstructed with plain GRU and detached-time policy-gradient proof; all9 none/withdraw snapshot regressions; immutable inputs rehashed after reconstruction"))


def checked():
    plan = read(PLAN)
    for name in ("inputs", "sources"):
        check_path_hashes(plan[name])
    training, freeze = checked_training()
    assert training["protocol_hash"] == plan["training_protocol_hash"]
    assert freeze["freeze_hash"] == plan["freeze_hash"]
    return plan, training, freeze


def source_world_continuity(seed):
    source = load(ROOT / f"data/teacher-timing/streams-v1/seed_{seed:03d}/midpoint/world_states.pt")["final"]
    states = load(OLD_DATA / f"seed_{seed:03d}/withdraw/world_states.pt")
    exact(states["before_change"], source, "complete source native world unchanged")
    before, after = states["before_change"], states["after_change"]
    for name in set(before)-{"regime", "regime_id", "regime_history", "context"}:
        exact(before[name], after[name], "withdraw boundary preserves "+name)
    for name in set(before["context"])-{"family_allowed", "future_family", "history"}:
        exact(before["context"][name], after["context"][name], "withdraw boundary context/"+name)
    assert before["tick"] == 33279 and before["acted"] and before["now"] == 8319.75
    assert after["regime"]["duration_range"] == (8., 16.)
    assert after["regime"]["gap_range"] == (16., 32.)
    assert after["context"]["family_allowed"] == [before["context"]["family_allowed"][1]]
    assert states["final"]["tick"] == 66047 and states["final"]["acted"]
    return dict(passed=True, source_world_complete_exact=True, source_tick=33279,
                first_new_tick=33280, final_new_tick=66047, same_withdraw_world_for_all_arms=True)


def expected_work(stream, bank, arm, step):
    native = {name: array[:128*step] for name, array in stream.items()}
    counts = rows(native)
    action_gradient, time_gradient = 0, 0
    for update in range(1, step+1):
        if arm != "none" and update % 8 == 0:
            _, episode = selected_episode(bank, arm, update)
            gradient = rows(episode, masked=True)
        else:
            gradient = rows({key: stream[key][128*(update-1):128*update] for key in FIELDS})
        action_gradient += gradient["action_rows"]
        time_gradient += gradient["time_rows"]
    return dict(warm_ticks=0, warm_lane_ticks=0, warm_loss_rows=0,
        training_updates=step, training_ticks=128*step, training_lane_ticks=128*step*32,
        **counts, action_gradient_rows=action_gradient, time_gradient_rows=time_gradient)


def empty_replay_counts():
    return dict(live_forward_updates=0, live_forward_ticks=0, live_forward_lane_ticks=0,
        live_gradient_updates=0, live_action_gradient_rows=0, live_time_gradient_rows=0,
        historical_gradient_updates=0, historical_window_ticks=0,
        historical_window_lane_ticks=0, historical_presentations=0,
        historical_action_rows=0, historical_action_WAIT_rows=0,
        historical_action_BEGIN_rows=0, historical_action_FINISH_rows=0,
        historical_action_weight_sum=0., historical_time_rows=0,
        prefix_ticks=0, prefix_lane_ticks=0, prefix_chunks=0)


TIME_FIELDS = ("wall_seconds", "live_forward_seconds", "prefix_seconds", "historical_gradient_seconds")


def check_replay_work(actual, expected, previous_times=None):
    assert actual.keys() == expected.keys() | set(TIME_FIELDS)
    for name, value in expected.items():
        exact(actual[name], value, "independent replay accounting/"+name)
    for name in TIME_FIELDS:
        assert isinstance(actual[name], (int, float)) and np.isfinite(actual[name]) and actual[name] >= 0
        if previous_times is not None:
            assert actual[name] >= previous_times[name], "monotonic measured time/"+name


def exact_none_regression(seed, step, cp):
    reference = OLD_TRAIN / f"seed_{seed:03d}/withdraw/step{step:03d}.pt"
    old = load(reference)
    fields = ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "step", "stream_cursor", "work", "source_work", "cumulative_work")
    for key in fields:
        exact(cp[key], old[key], f"none exact historical withdraw/{key}")
    assert state_digest(cp["numpy_rng"]) == state_digest(old["numpy_rng"])
    return dict(step=step, passed=True, reference_path=str(reference), reference_sha256=sha(reference),
                fields=list(fields)+["numpy_rng"])


def audit_case(seed, arm, stream, history, bank, training, device):
    directory = folder(seed, arm)
    source, initial = load(source_checkpoint(seed)), load(directory / "step000.pt")
    for name in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng"):
        exact(initial[name], source[name], "untouched learned source "+name)
    assert state_digest(initial["numpy_rng"]) == state_digest(source["numpy_rng"])
    optimizer_steps(source["optimizer"], 256)
    assert source["step"] == 256 and source["stream_cursor"] == 33280
    source_updates = [json.loads(line) for line in (source_checkpoint(seed).parent / "updates.jsonl").read_text().splitlines()]
    assert source_updates[-1]["hidden_after_sha256"] == state_digest(source["hidden"])
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    accounting = [json.loads(line) for line in (directory / "accounting.jsonl").read_text().splitlines()]
    assert len(updates) == 256
    assert len(accounting) == 256
    previous_h = state_digest(source["hidden"])
    all_replay = []
    expected_account = empty_replay_counts()
    accounting_snapshots = {0: expected_account.copy()}
    previous_times = {name: 0. for name in TIME_FIELDS}
    for update, record in enumerate(updates, 1):
        start, stop = 128*(update-1), 128*update
        assert record["seed"] == seed and record["arm"] == "action"
        assert (record["step"], record["tick_start"], record["tick_stop"]) == (update, start, stop)
        assert record["input_sha256"] == identity(stream, start, stop)
        assert record["time_features_detached"] and record["numerical_state_carried"]
        assert record["hidden_before_sha256"] == previous_h
        previous_h = record["hidden_after_sha256"]
        native_counts = rows({key: stream[key][start:stop] for key in FIELDS})
        for name, value in native_counts.items():
            assert record[name] == value, f"live exposure/{name}"
        expected_account["live_forward_updates"] += 1
        expected_account["live_forward_ticks"] += 128
        expected_account["live_forward_lane_ticks"] += 128*32
        account = accounting[update-1]
        assert account["step"] == update
        if arm != "none" and update % 8 == 0:
            selected, episode = selected_episode(bank, arm, update)
            gradient_counts = rows(episode, masked=True)
            assert record["gradient_source"] == account["gradient_source"] == "historical_"+arm[-1]
            np.testing.assert_array_equal(record["selected_pair_indices"], selected)
            assert record["replay_input_sha256"] == identity(episode, 0, 128)
            assert record["historical_hidden_committed"] is False
            assert record["execution_modes"] == dict(live="train/no_grad", prefix="eval/no_grad", historical="train/grad")
            for name, value in gradient_counts.items():
                assert record["replay_"+name] == value
                expected_account["historical_"+name] += value
            maximum = int(episode["anchor_start"].max())
            unique = np.unique(episode["actor_lane"])
            boundaries = sorted({0, maximum, *map(int, episode["anchor_start"]), *range(512, maximum, 512)})
            expected_prefix = dict(prefix_ticks=maximum, prefix_lane_ticks=maximum*len(unique),
                unique_lanes=unique.tolist(), boundaries=boundaries, prefix_chunks=len(boundaries)-1)
            exact(record["prefix"], expected_prefix, "independent complete-prefix schedule")
            assert record["prefix_lane_ticks"] == expected_prefix["prefix_lane_ticks"]
            assert record["prefix_chunks"] == expected_prefix["prefix_chunks"]
            expected_account["historical_gradient_updates"] += 1
            expected_account["historical_window_ticks"] += 128
            expected_account["historical_window_lane_ticks"] += 128*32
            expected_account["historical_presentations"] += 32
            for name in ("prefix_ticks", "prefix_lane_ticks", "prefix_chunks"):
                expected_account[name] += expected_prefix[name]
            all_replay.append(dict(step=update, selected_pair_indices=selected.tolist(),
                masked_rows=gradient_counts, input_sha256=identity(episode, 0, 128),
                prefix_lane_ticks=expected_prefix["prefix_lane_ticks"]))
        else:
            assert account["gradient_source"] == "live_B"
            expected_account["live_gradient_updates"] += 1
            expected_account["live_action_gradient_rows"] += native_counts["action_rows"]
            expected_account["live_time_gradient_rows"] += native_counts["time_rows"]
        check_replay_work(account["replay_work"], expected_account, previous_times)
        previous_times = {name: account["replay_work"][name] for name in TIME_FIELDS}
        if update in SNAPSHOTS:
            accounting_snapshots[update] = expected_account.copy()
    reconstructed = [reconstruct_update(directory, stream, history, bank, arm, step, device) for step in (1, 8)]
    for actual in reconstructed:
        exact(actual["loss"], updates[actual["step"]-1]["loss"], "actual independent loss log")
        exact(actual["preclip_grad_norm"], updates[actual["step"]-1]["preclip_grad_norm"], "actual independent norm log")
    snapshots, regression = [], []
    for step in SNAPSHOTS:
        path = directory / f"step{step:03d}.pt"
        cp, metadata = load(path), read(path.with_suffix(".json"))
        assert cp["replay_arm"] == arm and cp["seed"] == seed
        assert cp["arm"] == cp["objective"] == "action"
        assert cp["step"] == cp["local_optimizer_updates"] == step
        assert cp["stream_cursor"] == 128*step and cp["warm_completed"]
        assert cp["settings"] == dict(warm=0, block=128, updates=256)
        assert cp["source_optimizer_step"] == 256 and cp["cumulative_optimizer_step"] == 256+step
        assert cp["source_stream_cursor"] == 33280 and cp["cumulative_stream_cursor"] == 33280+128*step
        assert cp["source_checkpoint"] == str(source_checkpoint(seed))
        assert cp["protocol_hash"] == metadata["protocol_hash"] == training["protocol_hash"]
        assert sha(path) == metadata["sha256"]
        exact(cp["source_hashes"], training["source_sha256"], "frozen source bindings")
        exact(cp["bindings"], training["input_sha256"][str(seed)], "frozen input bindings")
        exact(cp["work"], expected_work(stream, bank, arm, step), "native exposures and actual gradient rows")
        exact(cp["source_work"], source["work"], "preserved source work")
        exact(cp["cumulative_work"], {name: value+cp["work"][name] for name, value in source["work"].items()}, "exact cumulative work")
        check_replay_work(cp["replay_work"], accounting_snapshots[step])
        if step:
            exact(cp["replay_work"], accounting[step-1]["replay_work"], "snapshot cumulative measured accounting")
        else:
            assert all(cp["replay_work"][name] == 0 for name in TIME_FIELDS)
        expected_h = state_digest(source["hidden"]) if step == 0 else updates[step-1]["hidden_after_sha256"]
        assert state_digest(cp["hidden"]) == metadata["hidden_sha256"] == expected_h
        optimizer_steps(cp["optimizer"], 256+step)
        for value in list(cp["model"].values())+[cp["hidden"]]:
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        if arm == "none":
            regression.append(exact_none_regression(seed, step, cp))
        snapshots.append(dict(step=step, passed=True, checkpoint_sha256=sha(path), hidden_sha256=expected_h,
                              optimizer_step=256+step, work=cp["work"]))
    assert previous_h == snapshots[-1]["hidden_sha256"]
    print(json.dumps(dict(seed=seed, arm=arm, independent_training_audit_passed=True)), flush=True)
    return dict(seed=seed, arm=arm, passed=True, source_model_Adam_h_RNG_exact=True,
        updates=256, replay_updates=len(all_replay), replay_batches=all_replay,
        actual_update_reconstruction=reconstructed, snapshots=snapshots, exact_none_regression=regression)


def run(device="cuda"):
    plan, training, freeze = checked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases, paired = [], []
    for seed in SEEDS:
        stream, history, bank = load_arrays(seed)
        native = source_world_continuity(seed)
        cases.extend(audit_case(seed, arm, stream, history, bank, training, device) for arm in ARMS)
        initial = [load(folder(seed, arm) / "step000.pt") for arm in ARMS]
        for name in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "work", "source_work"):
            for other in initial[1:]:
                exact(initial[0][name], other[name], "three exact paired initial states/"+name)
        for step in range(8, 257, 8):
            _, a = selected_episode(bank, "replay_A", step)
            _, b = selected_episode(bank, "replay_B", step)
            exact(rows(a, True), rows(b, True), "paired gradient labels and CE denominators")
        paired.append(dict(seed=seed, passed=True, native_world_continuity=native,
            paired_replay_updates=32, presentations_per_replay_arm=1024,
            A_B_masks_counts_and_schedule_exact=True,
            none_exposure_not_claimed_equal_gradient_labels=True,
            incidental_other_slot_A_observations_in_B_control_allowed=True))
        del stream, history, bank
    # Rehash original checkpoints, source code, bank, and raw streams after the
    # numerical reconstruction, proving that the audit has not mutated them.
    check_path_hashes(plan["inputs"])
    check_path_hashes(plan["sources"])
    write_new(OUT / "audit.json", dict(format="episodic-review-independent-training-audit-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), device=device,
        training_protocol_hash=training["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        audit_plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), scope=plan["scope"],
        actual_updates_reconstructed=18, snapshots_checked=27, historical_none_regressions=9,
        immutable_inputs_unchanged_after_audit=True, cases=cases, paired_budgets=paired))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.device)
