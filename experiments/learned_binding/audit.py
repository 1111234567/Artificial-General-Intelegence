"""Independent artifact, world, ledger, and public-outcome gradient audit.

No formal training or evaluation is started here.  An ordinary invocation
requires the complete saved experiment; ``--self-test`` uses temporary fixtures.
All world replays execute archived actions with the oracle disabled.  Policy
replays reconstruct models from the saved parameters, without opening another
trained checkpoint.  PG recomputation deliberately does not import its loss or
return calculation from run.py.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.bridge import PublicReceiptBinder, RandomMappingWorld
from experiments.associative_state.shared import SharedAssociationMemory
from experiments.learn_remember.gru_train import GRUAgent
from experiments.learned_binding import common
from experiments.learned_binding.interaction import Subject, probe
from experiments.learned_binding.model import NativeMemoryPolicy

GROUPS = ("all", "A", "B", "A_conflict", "B_conflict", "A_shared", "B_shared")
BASE_PROBES = ("own_immediate", "own_after_B_frozen", "teacher_probe")
EXTRA_PROBES = ("own_after_B_outcome", "teacher_clear_memory", "teacher_clear_encoder", "teacher_no_read")
WORLD_ARRAYS = dict(event_kind="event_kind", event_rid="event_request_id",
    event_eligible="event_eligible", event_family="event_family", raw="raw_event_payload",
    target="event_payload", mature_mask="matured_mask", mature_rid="matured_request_id",
    mature_done="matured_completed", mature_strict="matured_clean_completed", active="active")


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def array_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def same_tree(left, right, label="value"):
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and
                array_equal(left.detach().cpu().numpy(), right.detach().cpu().numpy()), label)
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray) and array_equal(left, right), label)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), label + ": keys")
        for key in left:
            same_tree(left[key], right[key], label + "." + str(key))
    elif isinstance(left, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right), label + ": sequence")
        for index, (a, b) in enumerate(zip(left, right)):
            same_tree(a, b, label + f"[{index}]")
    else:
        require(type(left) is type(right) and left == right, label + f": {left!r} != {right!r}")


def read_json(path):
    require(Path(path).is_file(), f"missing artifact: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pt(path):
    require(Path(path).is_file(), f"missing artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_bundle(prefix):
    prefix = Path(prefix)
    report = read_json(prefix.with_suffix(".json"))
    for suffix, key in ((".npz", "arrays_sha256"), (".pt", "states_sha256")):
        require(prefix.with_suffix(suffix).is_file(), f"missing artifact: {prefix}{suffix}")
        require(file_hash(prefix.with_suffix(suffix)) == report[key], f"{prefix}: {key}")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    states = load_pt(prefix.with_suffix(".pt"))
    return report, arrays, states


def check_snapshot(snapshot, label="snapshot"):
    require(parameter_hash(snapshot["model"]) == snapshot["parameter_hash"], label + ": parameter hash")
    require(tensor_hash(snapshot["state"]["memory"]) == snapshot["memory_hash"], label + ": memory hash")
    expected = {"h", "z", "memory"} if snapshot["kind"] == "learned" else {"h", "memory"}
    require(set(snapshot["state"]) == expected, label + ": state names")
    for name, value in list(snapshot["state"].items()) + list(snapshot["model"].items()):
        require(bool(torch.isfinite(value).all()), label + ": nonfinite " + name)
    world = RandomMappingWorld.from_state_dict(snapshot["world"])
    pending = snapshot["pending_obs"]
    require((pending is None) == snapshot["world"]["acted"], label + ": pending/acted boundary")
    if pending is not None:
        require(tuple(pending.shape) == (world.batch_size, 56), label + ": pending shape")
    require((snapshot["binder"] is None) == (snapshot["kind"] == "learned"), label + ": binder kind")
    if snapshot["binder"] is not None:
        PublicReceiptBinder.from_state_dict(snapshot["binder"])


def independent_rewards(observations):
    """Read only public preceding-action/result features, independently of data.py."""
    public = np.asarray(observations).reshape(*np.asarray(observations).shape[:-1], 2, 28)
    action_features = public[..., 11:22]
    action = action_features.argmax(-1)
    attempted = (action_features.sum(-1) > .5) & (action > 0)
    rejected = attempted & (public[..., 23] > .5)
    finished = (action >= 3) & attempted & (public[..., 22] > .5) & ~rejected
    return finished.astype(np.float32) - np.float32(.1) * rejected.astype(np.float32)


def _forbid_oracle(*args, **kwargs):
    raise AuditError("oracle called in an own-action audit path")


def _teacher_label(world):
    """Audit the midpoint action from current contracts; never create policy input."""
    can_begin = world.active & ~world.started & (world.now + world.duration <= world.deadline + 1e-6)
    can_finish = world.active & world.started & (world.now >= world.ready_time - 1e-6) & (world.now <= world.deadline + 1e-6)
    midpoint = .5 * (world.ready_time + world.deadline)
    can_finish &= (world.now >= midpoint - 1e-6) | (world.now + world.dt > world.deadline + 1e-6)
    return torch.where(can_begin, 1 + world.version,
                       torch.where(can_finish, 3 + 4 * world.version + world.payload, 0))


def replay_world(arrays, states, *, teacher=False):
    """Replay every actual action and native public/private observation exactly."""
    start, final = states["start"], states["final"]
    world = RandomMappingWorld.from_state_dict(start["world"])
    world.oracle = _forbid_oracle
    pending = start["pending_obs"]
    for index, action in enumerate(arrays["actions"]):
        if pending is not None:
            obs, pending = pending, None
        else:
            obs = world.observe()
        require(array_equal(obs.numpy(), arrays["obs"][index]), f"world public observation {index}")
        for key, attribute in WORLD_ARRAYS.items():
            if key in arrays:
                require(array_equal(getattr(world, attribute).numpy(), arrays[key][index]), f"world {key} {index}")
        if teacher:
            require(array_equal(_teacher_label(world).numpy(), action), f"actual midpoint teacher action {index}")
        world.act(torch.from_numpy(action))
    if "pending_obs" in arrays:
        pending = world.observe()
        require(array_equal(pending.numpy(), arrays["pending_obs"]), "last action's pending observation")
    same_tree(world.state_dict(), final["world"], "world replay final")
    same_tree(pending, final["pending_obs"], "world replay pending")
    return dict(ticks=len(arrays["actions"]), world_and_public_bitwise=True, oracle_disabled=True)


def check_block(report, arrays, states, *, teacher=False):
    start, final = states["start"], states["final"]
    check_snapshot(start, "block.start")
    check_snapshot(final, "block.final")
    obs, actions = arrays["obs"], arrays["actions"]
    expected_arrays = {"obs", "actions", "logits", "read_probs", "gate", "effective_gate"}
    if "rewards" in arrays:
        expected_arrays |= {"rewards", "pending_obs"}
    require(set(arrays) == expected_arrays, "unexpected/missing collection array fields")
    require(obs.ndim == 3 and obs.shape[-1] == 56 and obs.dtype == np.float32, "block public56 FP32")
    ticks, batch = obs.shape[:2]
    require(actions.shape == (ticks, batch, 2) and actions.dtype == np.int64, "block action shape/dtype")
    require(bool(np.all((actions >= 0) & (actions < 11))), "block action codes")
    require(arrays["logits"].shape == (ticks, batch, 2, 11), "block logit shape")
    for name in ("obs", "logits", "gate", "effective_gate", "read_probs"):
        require(bool(np.isfinite(arrays[name]).all()), "nonfinite block " + name)
    for name in ("gate", "effective_gate", "read_probs"):
        require(bool(np.all((arrays[name] >= 0) & (arrays[name] <= 1))), "block unit-interval " + name)
    require(np.allclose(arrays["read_probs"].sum(-1), 1., rtol=1e-6, atol=1e-6), "block normalized read probabilities")
    same_tree(start["model"], final["model"], "collection changed parameters")
    same_tree(start["executor"], final["executor"], "collection changed executor")
    require(final["work"]["ticks"] - start["work"]["ticks"] == ticks, "block action count")
    require(final["work"]["model_steps"] - start["work"]["model_steps"] == ticks, "block controller count")
    if "rewards" in arrays:
        require(not teacher and "pending_obs" in arrays, "outcome block needs own actions and final public receipt")
        following = np.concatenate((obs[1:], arrays["pending_obs"][None]), axis=0)
        require(array_equal(independent_rewards(following), arrays["rewards"]), "public outcome reward alignment")
        require(final["pending_obs"] is not None, "outcome block must retain pending observation")
    else:
        require(final["pending_obs"] is None, "support must end after act")
    expected_samples = 0 if teacher else ticks * batch * 2
    require(final["work"]["sampled_action_rows"] - start["work"]["sampled_action_rows"] == expected_samples,
            "block all-action sampling count")
    expected_teacher = ticks * batch * 2 if teacher else 0
    require(final["work"]["teacher_action_rows"] - start["work"]["teacher_action_rows"] == expected_teacher,
            "block teacher count")
    return replay_world(arrays, states, teacher=teacher)


def _groups(rows):
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if group == "all" or
                    (row["family"] == group[0] and ("_" not in group or
                     row["conflict"] == group.endswith("conflict")))]
        groups[group] = dict(n=len(selected), strict=sum(int(row["strict"]) for row in selected),
            ordinary=sum(int(row["ordinary"]) for row in selected),
            translation_correct=sum(int(row["translation_correct"]) for row in selected))
    return groups


def recount_probe(report, arrays, states, *, world_replay=True):
    """Rebuild the ledger from cue deadlines and maturity arrays, not saved rows."""
    start, final = states["start"], states["final"]
    check_snapshot(start, "probe.start")
    check_snapshot(final, "probe.final")
    obs = arrays["obs"]
    require(obs.ndim == 3 and obs.shape[-1] == 56, "probe public56 shape")
    ticks, batch = obs.shape[:2]
    require(set(arrays) == set(WORLD_ARRAYS) | {"obs", "actions", "prediction", "gate", "effective_gate", "read_probs"},
            "unexpected/missing probe array fields")
    for key in set(WORLD_ARRAYS) | {"actions", "prediction", "gate", "effective_gate"}:
        require(arrays[key].shape == (ticks, batch, 2), "probe shape: " + key)
    require(arrays["read_probs"].shape == (ticks, batch, 2, 4), "probe read-probability shape")
    require(np.isfinite(obs).all() and np.isfinite(arrays["read_probs"]).all(), "finite probe")
    require(np.all((arrays["actions"] >= 0) & (arrays["actions"] < 11)), "probe action codes")
    require(np.array_equal(arrays["read_probs"].argmax(-1), arrays["prediction"]), "native prediction from read probabilities")
    require(np.allclose(arrays["read_probs"].sum(-1), 1., atol=1e-6), "probe read normalization")
    public = obs.reshape(ticks, batch, 2, 28)
    kind = np.where(public[..., :3].sum(-1) > .5, public[..., :3].argmax(-1) + 1, 0)
    require(np.array_equal(kind, arrays["event_kind"]), "probe public event kind")
    commands = (kind == 1) | (kind == 2)
    require(np.all(~arrays["event_eligible"] | commands), "eligible event is a command")
    require(np.all(~arrays["mature_strict"] | arrays["mature_done"]), "strict implies ordinary")
    require(np.all(~arrays["mature_done"] | arrays["mature_mask"]), "ordinary implies maturity")
    world = start["world"]
    contexts = tuple(world["random_mapping"]["contexts"])
    table = world["random_mapping"]["mappings"].numpy()
    offset = 0 if start["pending_obs"] is not None else 1
    absolute_ticks = world["tick"] + offset + np.arange(ticks)
    times = absolute_ticks * world["spec"]["dt"]
    maturities = {}
    for index, lane, job in np.argwhere(arrays["mature_mask"]):
        key = int(lane), int(job), int(arrays["mature_rid"][index, lane, job])
        require(key not in maturities, "duplicate maturity")
        maturities[key] = int(index)
    finished, censored, eligible_keys, seen = [], [], set(), set()
    correct_translations = 0
    for index, lane, job in np.argwhere(commands):
        family = int(arrays["event_family"][index, lane, job])
        require(family in contexts, "unknown command context")
        family_index = contexts.index(family)
        raw, target = int(arrays["raw"][index, lane, job]), int(arrays["target"][index, lane, job])
        require(raw == int(public[index, lane, job, 5:9].argmax()), "raw symbol differs from public cue")
        require(family == int(public[index, lane, job, 24:28].argmax()), "context differs from public cue")
        require(target == int(table[lane, family_index, raw]), "private mapping target")
        correct = int(arrays["prediction"][index, lane, job]) == target
        correct_translations += int(correct)
        rid = int(arrays["event_rid"][index, lane, job])
        require(rid == int(absolute_ticks[index]) * 2 + int(job), "request ID is not issue tick/job")
        if not arrays["event_eligible"][index, lane, job]:
            continue
        key = int(lane), int(job), rid
        require(key not in eligible_keys, "duplicate eligible command")
        eligible_keys.add(key)
        identity = int(lane), family_index, raw
        first = identity not in seen
        seen.add(identity)
        row = dict(lane=int(lane), actor_id=world["spec"]["actor_ids"][lane], job=int(job), rid=rid,
            issued_tick=int(absolute_ticks[index]), family="AB"[family_index], family_id=family,
            raw=raw, target=target, conflict=bool(table[lane, 0, raw] != table[lane, 1, raw]),
            translation_correct=bool(correct), first_for_actor_family_raw=first)
        deadline = times[index] + float(public[index, lane, job, 9]) + float(public[index, lane, job, 10])
        mature_index = int(np.searchsorted(times, deadline + 1e-6, side="right"))
        if mature_index < ticks:
            require(key in maturities and maturities[key] == mature_index, "missing/mistimed maturity " + str(key))
            row.update(mature_tick=int(absolute_ticks[mature_index]),
                ordinary=bool(arrays["mature_done"][mature_index, lane, job]),
                strict=bool(arrays["mature_strict"][mature_index, lane, job]))
            finished.append(row)
        else:
            require(key not in maturities, "censored request has a maturity")
            censored.append(row)
    finished.sort(key=lambda row: (row["mature_tick"], row["lane"], row["job"]))
    left = []
    for key, index in sorted(maturities.items(), key=lambda item: (item[1], item[0][0], item[0][1])):
        if key in eligible_keys:
            continue
        lane, job, rid = key
        require(bool(world["tensors"]["_score_pending"][lane, job]) and
                int(world["tensors"]["_score_request_id"][lane, job]) == rid, "unknown inherited maturity")
        left.append(dict(lane=lane, job=job, rid=rid, mature_tick=int(absolute_ticks[index]),
            ordinary=bool(arrays["mature_done"][index, lane, job]),
            strict=bool(arrays["mature_strict"][index, lane, job])))
    require(finished == report["requests"], "finished request ledger")
    require(censored == report["right_censored"], "right-censored request ledger")
    require(left == report["left_boundary_matured"], "left-boundary ledger")
    require(_groups(finished) == report["groups"], "group denominator/counts")
    first = [row for row in finished if row["first_for_actor_family_raw"]]
    require(first == report["first_requests"] and _groups(first) == report["first_groups"], "first-request results")
    require([row for row in censored if row["first_for_actor_family_raw"]] == report["first_right_censored"],
            "first-request censoring")
    idle = ~arrays["active"]
    require(report["idle_opportunities"] == int(idle.sum()), "idle opportunity denominator")
    require(report["idle_actions"] == int((idle & (arrays["actions"] != 0)).sum()), "idle action count")
    require(report["command_translations"] == int(commands.sum()) and
            report["correct_translations"] == correct_translations, "translation counts")
    receipt_action = public[..., 11:22].argmax(-1)
    success = (receipt_action >= 3) & (public[..., 22] > .5) & (public[..., 23] < .5)
    for name, mask in (("all", np.ones_like(success)), ("success_finish", success),
                       ("command", commands), ("other", ~success & ~commands)):
        row = report["gate_statistics"][name]
        count, total = int(mask.sum()), float(arrays["gate"][mask].sum(dtype=np.float64))
        require(row["count"] == count and np.isclose(row["sum"], total, rtol=1e-6, atol=1e-5), "gate total " + name)
        require((row["mean"] is None and not count) or
                (count and np.isclose(row["mean"], total / count, rtol=1e-6, atol=1e-7)), "gate mean " + name)
    require(np.isclose(report["effective_gate_sum"], arrays["effective_gate"].sum(dtype=np.float64),
                       rtol=1e-6, atol=1e-5), "effective gate total")
    same_tree(start["model"], final["model"], "probe altered parameters")
    same_tree(start["executor"], final["executor"], "probe altered executor")
    require(report["start_tick"] == world["tick"] and report["end_tick"] == int(absolute_ticks[-1]), "probe clock bounds")
    require(report["start_work"] == start["work"] and report["final_work"] == final["work"], "probe work records")
    require(final["work"]["ticks"] - start["work"]["ticks"] == ticks, "probe executed tick count")
    require(report["memory_norm"] == float(final["state"]["memory"].norm(dim=-1).mean()), "probe memory norm")
    if not report["write_enabled"]:
        same_tree(start["state"]["memory"], final["state"]["memory"], "no-write memory")
    if world_replay:
        replay_world(arrays, states)
    return dict(seed=report["seed"], kind=report["kind"], name=report.get("name"), ticks=ticks,
        eligible=len(eligible_keys), matured=len(finished), right_censored=len(censored),
        inherited=len(left), groups=_groups(finished), world_replayed=world_replay)


def model_from_snapshot(snapshot):
    if snapshot["kind"] == "learned":
        model = NativeMemoryPolicy(original_mappings=snapshot["model"]["original_mappings"])
    else:
        codes = snapshot["model"]["key_embedding.codes"]
        model = SharedAssociationMemory(kind="delta", key_dim=codes.shape[1], n_keys=codes.shape[0], n_values=4)
        model.requires_grad_(False)
    model.load_state_dict(snapshot["model"])
    return model.eval()


def subject_from_snapshot(snapshot):
    subject = Subject.__new__(Subject)
    subject.seed, subject.kind = snapshot["seed"], snapshot["kind"]
    subject.stream_id = snapshot["stream_id"]
    subject.contexts = tuple(snapshot["world"]["random_mapping"]["contexts"])
    subject.original = torch.tensor(common.mapping_table(subject.seed, "related"), dtype=torch.long)
    subject.model = model_from_snapshot(snapshot)
    if subject.kind == "learned":
        subject.executor = subject.model.executor
    else:
        subject.executor = GRUAgent().eval().requires_grad_(False)
    subject.rng = torch.Generator(device="cpu")
    subject.restore(snapshot)
    same_tree(subject.snapshot(), snapshot, "snapshot reconstruction")
    subject.world.oracle = _forbid_oracle
    return subject


def native_probability_check(arrays, states, tolerance=2e-4):
    model = model_from_snapshot(states["start"])
    with torch.no_grad():
        logits, end, diagnostic = model(torch.from_numpy(arrays["obs"]), states["start"]["state"],
                                       return_diagnostics=True)
    error = float((logits - torch.from_numpy(arrays["logits"])).abs().max())
    require(error <= tolerance, f"native sequence/step logits differ: {error}")
    diagnostic_errors = {}
    for name, value in (("gate", diagnostic["gate"].squeeze(-1)),
                        ("effective_gate", diagnostic["gate"].squeeze(-1)),
                        ("read_probs", diagnostic["read_probs"])):
        difference = float((value - torch.from_numpy(arrays[name])).abs().max())
        require(difference <= 2e-5, "native sequence/step diagnostic " + name)
        diagnostic_errors[name] = difference
    state_error = {}
    for name, value in end.items():
        difference = float((value - states["final"]["state"][name]).abs().max())
        require(difference <= tolerance, "native sequence/step state " + name)
        state_error[name] = difference
    return dict(logit_maxabs=error, state_maxabs=state_error, diagnostic_maxabs=diagnostic_errors,
                tolerance=tolerance)


def independent_returns_advantages(rewards, gamma):
    """One joint actor objective: both jobs can affect their shared memory."""
    actor_rewards = rewards.sum(dim=-1)
    require(actor_rewards.shape[1] > 1, "LOAO baseline requires at least two actors")
    result = torch.empty_like(actor_rewards)
    accumulator = torch.zeros_like(actor_rewards[0])
    for index in reversed(range(actor_rewards.shape[0])):
        accumulator = actor_rewards[index] + gamma * accumulator
        result[index] = accumulator
    other_actor_mean = (result.sum(dim=1, keepdim=True) - result) / (result.shape[1] - 1)
    return result, result - other_actor_mean


def recompute_pg(report, arrays, states, settings, previous_optimizer=None):
    """Recompute public rewards, joint PG, clipped gradients and Adam exactly."""
    require(states["start"]["kind"] == "learned", "PG requires native learned policy")
    following = np.concatenate((arrays["obs"][1:], arrays["pending_obs"][None]), axis=0)
    rewards = torch.from_numpy(independent_rewards(following))
    require(array_equal(rewards.numpy(), arrays["rewards"]), "PG public rewards")
    model = model_from_snapshot(states["start"]).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings["pg_lr"])
    if previous_optimizer is not None:
        optimizer.load_state_dict(deepcopy(previous_optimizer))
    logits, _ = model(torch.from_numpy(arrays["obs"]), states["start"]["state"])
    error = float((logits.detach() - torch.from_numpy(arrays["logits"])).abs().max())
    require(error <= 2e-4, "PG replay probabilities")
    returns, advantages = independent_returns_advantages(rewards, settings["gamma"])
    actions = torch.from_numpy(arrays["actions"])
    joint_logprob = logits.log_softmax(dim=-1).gather(-1, actions[..., None]).squeeze(-1).sum(-1)
    loss = -(joint_logprob * advantages.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.))
    require(all(parameter.grad is None for parameter in model.executor.parameters()), "executor received PG gradients")
    optimizer.step()
    same_tree(model.state_dict(), states["after_update"]["model"], "independent PG/Adam parameters")
    same_tree(optimizer.state_dict(), states["optimizer_after"], "independent PG/Adam optimizer")
    for key, value in dict(loss=float(loss.detach()), grad_norm=norm, step_sequence_logit_maxabs=error,
                            reward_sum=float(rewards.sum()), positive_finishes=int((rewards > 0).sum()),
                            rejected=int((rewards < 0).sum()), return_mean=float(returns.mean()),
                            advantage_mean=float(advantages.mean())).items():
        require(np.isclose(value, report["update"][key], rtol=1e-6, atol=1e-7), "PG metric " + key)
    return dict(index=report["index"], loss=float(loss.detach()), grad_norm=norm,
                model_and_adam_bitwise_equal=True, public_rewards_recomputed=True,
                joint_actor_objective=True, logit_maxabs=error)


def _executor_state(snapshot):
    if snapshot["kind"] != "learned":
        return snapshot["executor"]
    return {key[len("executor."):]: value for key, value in snapshot["model"].items() if key.startswith("executor.")}


def _after_policy(world_state, allowed):
    world = RandomMappingWorld.from_state_dict(world_state)
    world.set_family_policy(tuple(allowed))
    return world.state_dict()


def _check_probe_branch(base, start, *, clear=None):
    for key in ("seed", "kind", "stream_id", "model", "executor", "binder", "work", "rng", "pending_obs"):
        same_tree(base[key], start[key], "probe branch " + key)
    state = deepcopy(base["state"])
    if clear is not None:
        state[clear].zero_()
    same_tree(state, start["state"], "probe branch dynamic state")
    contexts = base["world"]["random_mapping"]["contexts"]
    same_tree(_after_policy(base["world"], contexts), start["world"], "probe branch world")


def _exogenous_hash(observations):
    public = observations.reshape(*observations.shape[:2], 2, 28)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(public[..., :11]).tobytes())
    digest.update(np.ascontiguousarray(public[..., 24:28]).tobytes())
    return digest.hexdigest()


def audit_phase(folder, support_final, settings, *, learn):
    folder = Path(folder)
    require(len(list(folder.glob("block_*.json"))) == settings["pg_blocks"], "unexpected/missing B block artifacts")
    boundary = load_pt(folder / "boundary.pt")
    initial, phase_final = boundary["initial"], boundary["final"]
    for key in ("state", "model", "executor", "binder", "work", "rng", "pending_obs"):
        same_tree(support_final[key], initial[key], "B fork " + key)
    contexts = initial["world"]["random_mapping"]["contexts"]
    same_tree(_after_policy(support_final["world"], (contexts[1],)), initial["world"], "B fork world policy")
    expected = initial
    previous_optimizer = None
    pg_checks, probability_checks, external, updates = [], [], [], []
    expected_executor = _executor_state(initial)
    for index in range(settings["pg_blocks"]):
        report, arrays, states = load_bundle(folder / f"block_{index:03d}")
        require(report["index"] == index and report["learn"] == learn, "B block index/learning flag")
        require(len(arrays["obs"]) == settings["pg_ticks"], "B block length")
        same_tree(expected, states["start"], "B block state chain")
        check_block(report, arrays, states)
        rewards = arrays["rewards"]
        for key, value in dict(reward_sum=float(torch.from_numpy(rewards).sum()),
                              positive_finishes=int((rewards > 0).sum()),
                              rejected=int((rewards < 0).sum())).items():
            require(report["update"][key] == value, "block public reward summary " + key)
        updates.append(report["update"])
        updated, before = states["after_update"], states["final"]
        check_snapshot(updated, "after_update")
        for key in ("state", "executor", "binder", "world", "rng", "pending_obs"):
            same_tree(before[key], updated[key], "optimizer transplanted/mutated live " + key)
        expected_work = dict(before["work"])
        if learn:
            expected_work["parameter_updates"] = expected_work.get("parameter_updates", 0) + 1
        same_tree(expected_work, updated["work"], "optimizer work count")
        same_tree(expected_executor, _executor_state(updated), "B executor changed")
        if not learn:
            same_tree(before["model"], updated["model"], "frozen B parameters changed")
        else:
            require("optimizer_after" in states, "absent Adam checkpoint")
            if index in (0, settings["pg_blocks"] - 1):
                pg_checks.append(recompute_pg(report, arrays, states, settings, previous_optimizer))
            previous_optimizer = states["optimizer_after"]
        if initial["kind"] == "learned":
            probability_checks.append(native_probability_check(arrays, states))
        external.append(_exogenous_hash(arrays["obs"]))
        expected = updated
    # Closure consumes the one already observed pending tick, and executes it.
    subject = subject_from_snapshot(expected)
    closure = subject.close_pending(sample=True)
    same_tree(closure, boundary["closure"], "B pending closure")
    same_tree(subject.snapshot(), phase_final, "B closure full final state")
    completion = read_json(folder / "completion.json")
    require(completion["blocks"] == settings["pg_blocks"] and completion["learn"] == learn, "B completion")
    require(completion["records"] == updates, "B completion records differ from blocks")
    require(completion["counted_extra_closure_ticks"] == 1 and
            completion["phase_ticks"] == settings["pg_blocks"] * settings["pg_ticks"] + 1, "B closure budget")
    require(phase_final["work"]["ticks"] - initial["work"]["ticks"] == completion["phase_ticks"], "B total actions")
    require(phase_final["world"]["tick"] - initial["world"]["tick"] == completion["phase_ticks"], "B total physical ticks")
    return dict(initial=initial, final=phase_final, exogenous=external, blocks=settings["pg_blocks"],
                pg_checks=pg_checks, probability_checks=probability_checks,
                clocks_pending_and_live_state_chain_verified=True)


def replay_probe(report, arrays, states):
    subject = subject_from_snapshot(states["start"])
    replay = probe(subject, len(arrays["obs"]), write_enabled=report["write_enabled"],
                   read_enabled=report["read_enabled"])
    for name, expected in arrays.items():
        require(array_equal(replay["arrays"][name], expected), "native policy replay " + name)
    same_tree(replay["final"], states["final"], "native policy replay final")
    for name, value in replay["report"].items():
        require(value == report[name], "native policy replay report " + name)
    return dict(seed=report["seed"], kind=report["kind"], name=report.get("name"),
                ticks=len(arrays["obs"]), arrays_bitwise=True, full_final_state_bitwise=True,
                oracle_disabled=True)


def _check_sources(root):
    protocol = read_json(root / "protocol.json")
    require(protocol["settings"] == common.SETTINGS, "declared settings changed")
    for path, expected in protocol["sources"].items():
        require(file_hash(ROOT / path) == expected, "source changed: " + path)
    for origin in protocol["origins"].values():
        require(file_hash(ROOT / origin["path"]) == origin["sha256"], "origin changed: " + origin["path"])
    freeze = read_json(root / "training-freeze.json")
    require(freeze["all_complete_before_scoring"] is True, "training barrier absent")
    require(len(freeze["checkpoints"]) == len(common.SETTINGS["seeds"]), "incomplete native training freeze")
    for path, expected in freeze["checkpoints"].items():
        require(file_hash(ROOT / path) == expected, "native checkpoint changed: " + path)
    return dict(source_files=len(protocol["sources"]), origins=len(protocol["origins"]),
                trained_checkpoints=len(freeze["checkpoints"]))


def check_no_oracle_paths():
    """Fail if an oracle call is added outside the explicit teacher branch."""
    source = ast.parse(Path(__file__).with_name("interaction.py").read_text(encoding="utf-8"))
    calls = []

    def visit(node, guarded=False):
        if isinstance(node, ast.If):
            teacher_guard = isinstance(node.test, ast.Name) and node.test.id == "teacher"
            visit(node.test, guarded)
            for child in node.body:
                visit(child, guarded or teacher_guard)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "oracle":
            calls.append(node.lineno)
            require(guarded, f"unguarded oracle call in interaction.py:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(source)
    require(len(calls) == 1, "expected exactly one explicit teacher oracle call")
    run_source = ast.parse(Path(__file__).with_name("run.py").read_text(encoding="utf-8"))
    for node in ast.walk(run_source):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            require(node.func.attr != "oracle", "oracle call in own-action/PG driver")
    return dict(only_oracle_call_is_explicit_teacher_support=True, own_driver_has_no_oracle_call=True)


def audit_all(root=None, *, settings=None, strict_sources=True, native_replays=True):
    """Audit the complete saved 39-probe / 384-block experiment, fail closed."""
    torch.set_num_threads(1)
    root = Path(common.OUT if root is None else root)
    settings = common.SETTINGS if settings is None else settings
    source_check = _check_sources(root) if strict_sources else dict(temporary_fixture=True)
    oracle_check = check_no_oracle_paths()
    counted, phases, replays, pairing = [], [], [], []
    for seed in settings["seeds"]:
        teacher_reference = own_external = world_reference = executor_reference = None
        if strict_sources:
            protocol = read_json(root / "protocol.json")
            source = load_pt(ROOT / protocol["origins"][f"{seed}/executor"]["path"])
            executor_reference = {key: value for key, value in source["model"].items()
                                  if key.startswith(("core.", "head."))}
            pretraining = load_pt(root / f"seed_{seed:03d}" / "pretrain" / "final.pt")
            require(pretraining["executor_hash"] == parameter_hash(executor_reference), "CE executor origin hash")
            same_tree(executor_reference, _executor_state(dict(kind="learned", model=pretraining["model"])),
                      "CE executor weights differ from origin")
        for kind in common.KINDS:
            folder = root / f"seed_{seed:03d}" / "behavior" / kind
            manifest = read_json(folder / "completion.json")
            expected_reports = set(BASE_PROBES) | (set(EXTRA_PROBES) if kind == "learned" else set())
            expected_names = {name + ".json" for name in expected_reports | {"own_support", "teacher_support"}}
            require(set(manifest["files"]) == expected_names, "incomplete behavior manifest")
            require(manifest["seed"] == seed and manifest["kind"] == kind, "behavior manifest identity")
            for name, expected in manifest["files"].items():
                require(file_hash(folder / name) == expected, "behavior manifest changed: " + name)
            own_report, own_arrays, own_states = load_bundle(folder / "own_support")
            teacher_report, teacher_arrays, teacher_states = load_bundle(folder / "teacher_support")
            require(own_report["teacher"] is False and teacher_report["teacher"] is True, "support source flags")
            check_block(own_report, own_arrays, own_states)
            check_block(teacher_report, teacher_arrays, teacher_states, teacher=True)
            require(_exogenous_hash(own_arrays["obs"]) == _exogenous_hash(teacher_arrays["obs"]),
                    "own/teacher exogenous requests differ")
            if strict_sources:
                if kind == "learned":
                    same_tree(pretraining["model"], own_states["start"]["model"], "native acquisition CE origin")
                else:
                    manual = load_pt(ROOT / protocol["origins"][f"{seed}/manual"]["path"])
                    same_tree(manual["model"], own_states["start"]["model"], "manual acquisition origin")
            same_tree(own_states["start"]["world"], teacher_states["start"]["world"], "own/teacher initial environment")
            if world_reference is None:
                world_reference = own_states["start"]["world"]
                if executor_reference is None:
                    executor_reference = _executor_state(own_states["start"])
                teacher_reference = teacher_arrays
                own_external = _exogenous_hash(own_arrays["obs"])
            else:
                same_tree(world_reference, own_states["start"]["world"], "three-kind initial environment")
                require(_exogenous_hash(own_arrays["obs"]) == own_external, "own-action exogenous stream differs")
                for name in ("obs", "actions"):
                    require(array_equal(teacher_reference[name], teacher_arrays[name]), "three-kind teacher " + name)
            for stage in (own_states["start"], own_states["final"], teacher_states["start"], teacher_states["final"]):
                same_tree(executor_reference, _executor_state(stage), "support source executor changed")
            frozen = audit_phase(folder / "frozen_B", own_states["final"], settings, learn=False)
            phase_records = [("frozen_B", frozen)]
            outcome = None
            if kind == "learned":
                outcome = audit_phase(folder / "outcome_B", own_states["final"], settings, learn=True)
                same_tree(frozen["initial"], outcome["initial"], "native frozen/PG common initial state")
                require(frozen["exogenous"] == outcome["exogenous"], "native frozen/PG exogenous cues differ")
                phase_records.append(("outcome_B", outcome))
            for name, phase in phase_records:
                phases.append(dict(seed=seed, kind=kind, phase=name, blocks=phase["blocks"],
                    pg_checks=phase["pg_checks"], probability_checks=phase["probability_checks"],
                    clocks_pending_and_live_state_chain_verified=True))
            branches = {"own_immediate": own_states["final"], "own_after_B_frozen": frozen["final"],
                        "teacher_probe": teacher_states["final"]}
            if outcome is not None:
                branches.update(own_after_B_outcome=outcome["final"], teacher_clear_memory=teacher_states["final"],
                    teacher_clear_encoder=teacher_states["final"], teacher_no_read=teacher_states["final"])
            for name, base in branches.items():
                report, arrays, states = load_bundle(folder / name)
                require(len(arrays["obs"]) == settings["probe_ticks"], "probe length")
                clear = "memory" if name == "teacher_clear_memory" else ("z" if name == "teacher_clear_encoder" else None)
                _check_probe_branch(base, states["start"], clear=clear)
                require(report["read_enabled"] == (name != "teacher_no_read") and report["write_enabled"] is True,
                        "prospective probe read/write flags")
                counted.append(recount_probe(report, arrays, states))
                if native_replays and ((kind == "learned" and name in ("own_after_B_outcome", "teacher_probe")) or
                                       (kind == "manual_hard" and name == "teacher_probe")):
                    replays.append(replay_probe(report, arrays, states))
            pairing.append(dict(seed=seed, kind=kind, teacher_actual_stream_equal=True,
                own_teacher_initial_world_equal=True, own_exogenous_stream_equal=True,
                frozen_outcome_common_start_and_exogenous=True if outcome is not None else None))
            print(json.dumps(dict(event="audit_kind", seed=seed, kind=kind, probes=len(branches))), flush=True)
    require(len(counted) == 13 * len(settings["seeds"]), "incomplete score audit")
    require(sum(phase["blocks"] for phase in phases) == 4 * len(settings["seeds"]) * settings["pg_blocks"],
            "incomplete B block audit")
    return dict(format="learned-binding-independent-audit-v1", sources=source_check, oracle_paths=oracle_check,
        probes=counted, probe_count=len(counted), B_phases=phases,
        B_block_count=sum(phase["blocks"] for phase in phases), pairing=pairing,
        native_replays=replays, native_replay_count=len(replays),
        independently_recomputed_pg_updates=sum(len(phase["pg_checks"]) for phase in phases),
        all_world_actions_and_public_rewards_replayed=True,
        all_controller_live_states_preserved_across_optimizer_updates=True,
        all_source_executors_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=common.OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("experiments.learned_binding.audit_test")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    report = audit_all(args.root)
    common.write_json(args.root / "audit.json", report)
    print(json.dumps(dict(event="audit_complete", probes=report["probe_count"], blocks=report["B_block_count"],
        native_replays=report["native_replay_count"], pg_updates=report["independently_recomputed_pg_updates"])))


if __name__ == "__main__":
    main()
