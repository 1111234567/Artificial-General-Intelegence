"""Fixed episodic-gradient replacements on the immutable B-only continuation.

Live observations and numerical recurrent state always follow the full B
stream. Historical windows replace 32 of 256 gradients; their anchors are
reconstructed from the complete public history under the current weights.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.action_split.train import deterministic, runtime, sha, write_new, append
from experiments.continuity.run import cpu_tree, state_digest
from experiments.coverage_retention import train as frozen, data as live_data
from experiments.remaining_time.model import (RemainingTimeGRUAgent,
    native_action_loss, time_loss, clip_gradients)

OUT = ROOT / "data/episodic-review/train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "episodic-review-continuous-live-gradient-replacement-v1"
ARMS, SEEDS = ("none", "replay_A", "replay_B"), (0, 1, 2)
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 0, 32768, 128, 32, 256
SNAPSHOTS, AUDIT_STEPS = (0, 64, 256), (1, 8)
REPLAY_EVERY, PREFIX_CHUNK = 8, 512
HEAD_FORMAT, OPTIMIZER, STREAM_FIELDS = frozen.HEAD_FORMAT, frozen.OPTIMIZER, frozen.STREAM_FIELDS
assert_exact, optimizer_steps = frozen.assert_exact, frozen.optimizer_steps
summarize_rows, stream_identity, validate_stream = frozen.summarize_rows, frozen.stream_identity, frozen.validate_stream
source_checkpoint, load_source = frozen.source_checkpoint, frozen.load_source
original_checkpoint = source_checkpoint


def bank_module():
    from experiments.episodic_review import bank
    return bank


def checkpoint_path(seed, arm, step=256):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


def states_path(seed):
    """All arms consume exactly the same actual withdraw teacher world."""
    return live_data.states_path(seed, "withdraw")


def load_stream(seed, arm=None):
    assert arm is None or arm in ARMS
    return live_data.load_stream(seed, "withdraw")


def initialize(seed, arm="none", device="cuda"):
    assert arm in ARMS
    return frozen.initialize(seed, "withdraw", device=device)


def replay_loss(scores, prediction, actions, target, time_mask, action_mask):
    """CE and detached-feature MSE on the designated episode slot only."""
    assert action_mask.dtype == time_mask.dtype == torch.bool
    assert action_mask.shape == time_mask.shape == actions.shape
    assert bool(action_mask.any())
    action = native_action_loss(scores[action_mask], actions[action_mask], .25)
    timing_mask = time_mask & action_mask
    timing = time_loss(prediction, target, timing_mask)
    return dict(total=action + timing["mse"], action=action, time=timing["mse"],
        action_rows=int(action_mask.sum()), time_rows=int(timing_mask.sum()))


def reconstruct_anchors(model, history, lanes, starts, chunk=PREFIX_CHUNK):
    """Recompute from h0, never reuse a numerical state from an older weight.

    Boundaries include all requested starts and multiples of 512. Every unique
    selected lane is run to max(start); capture is before the window's first
    observation. Only one prefix chunk is transferred to the device at a time.
    """
    lanes, starts = np.asarray(lanes, np.int64), np.asarray(starts, np.int64)
    assert lanes.ndim == starts.ndim == 1 and lanes.shape == starts.shape and len(lanes)
    assert (starts >= 0).all() and int(starts.max()) <= len(history)
    unique = np.unique(lanes)
    assert (unique >= 0).all() and (unique < history.shape[1]).all()
    maximum = int(starts.max())
    boundaries = sorted({0, maximum, *starts.tolist(), *range(chunk, maximum, chunk)})
    device = next(model.parameters()).device
    anchors = torch.empty((len(starts), model.width), device=device)
    hidden = model.initial_state(len(unique), device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index, stop in enumerate(boundaries):
            if index:
                start = boundaries[index - 1]
                obs = torch.from_numpy(np.ascontiguousarray(history[start:stop, unique])).to(device)
                _, hidden = model(obs, hidden)
                hidden = hidden.detach()
            chosen = np.flatnonzero(starts == stop)
            if len(chosen):
                positions = np.searchsorted(unique, lanes[chosen])
                anchors[torch.as_tensor(chosen, device=device)] = hidden[torch.as_tensor(positions, device=device)]
    model.train(was_training)
    return anchors.detach(), dict(prefix_ticks=maximum,
        prefix_lane_ticks=maximum * len(unique), unique_lanes=unique.tolist(),
        boundaries=boundaries, prefix_chunks=max(0, len(boundaries) - 1))


def masked_rows(batch):
    mask = batch["action_mask"]
    action = batch["actions"][mask]
    wait = int((action == 0).sum())
    return dict(action_rows=int(action.size), action_WAIT_rows=wait,
        action_BEGIN_rows=int(((action >= 1) & (action <= 2)).sum()),
        action_FINISH_rows=int((action >= 3).sum()),
        action_weight_sum=float(action.size - .75 * wait),
        time_rows=int((batch["time_mask"] & mask).sum()))


class Learner(frozen.Learner):
    def __init__(self, model, stream, experiment_arm, *, source, bank=None,
                 history=None, replay_every=REPLAY_EVERY, prefix_chunk=PREFIX_CHUNK, **kwargs):
        assert experiment_arm in ARMS
        # Keep the source learner's actual-world arm valid. replay_arm is the
        # new experimental intervention; the actual native world is withdraw.
        super().__init__(model, stream, "withdraw", source=source, **kwargs)
        self.replay_arm, self.bank, self.history = experiment_arm, bank, history
        self.replay_every, self.prefix_chunk = int(replay_every), int(prefix_chunk)
        self._defer_after, self._pending_after = False, None
        if experiment_arm != "none":
            assert bank is not None and history is not None
        self.replay_work = dict(live_forward_updates=0, live_forward_ticks=0,
            live_forward_lane_ticks=0, live_gradient_updates=0,
            live_action_gradient_rows=0, live_time_gradient_rows=0,
            historical_gradient_updates=0, historical_window_ticks=0,
            historical_window_lane_ticks=0, historical_presentations=0,
            historical_action_rows=0, historical_action_WAIT_rows=0,
            historical_action_BEGIN_rows=0, historical_action_FINISH_rows=0,
            historical_action_weight_sum=0., historical_time_rows=0,
            prefix_ticks=0, prefix_lane_ticks=0, prefix_chunks=0,
            wall_seconds=0., live_forward_seconds=0., prefix_seconds=0.,
            historical_gradient_seconds=0.)

    def payload(self):
        cp = super().payload()
        cp.update(format=FORMAT, experiment_arm=self.replay_arm,
            replay_arm=self.replay_arm, actual_world_arm="withdraw",
            replay_work=deepcopy(self.replay_work),
            replay_settings=dict(every=self.replay_every, prefix_chunk=self.prefix_chunk),
            initialization="complete source model+Adam256+h+RNG restored; no warm or reinitialization",
            carried_hidden_semantics="full B stream numerical pre-update h; historical h never committed")
        return cp

    def save(self, name):
        if self._defer_after and name.endswith("_after"):
            self._pending_after = name
            return
        assert self.folder is not None
        path = self.folder / (name + ".pt")
        if path.exists():
            raise FileExistsError(path)
        cp = self.payload()
        torch.save(cp, path)
        write_new(path.with_suffix(".json"), dict(path=str(path), sha256=sha(path),
            step=self.step, stream_cursor=self.cursor, seed=self.seed,
            arm="action", objective="action", experiment_arm=self.replay_arm,
            actual_world_arm="withdraw", work=self.work, replay_work=self.replay_work,
            source_work=self.source_work, cumulative_work=self.cumulative_work(),
            optimizer_state_steps=optimizer_steps(self.optimizer),
            source_optimizer_step=self.source_optimizer_step,
            cumulative_optimizer_step=self.source_optimizer_step + self.step,
            hidden_sha256=state_digest(cp["hidden"]),
            initial_hidden_sha256=self.source_state_hashes["hidden"],
            core_sha256=state_digest(cpu_tree(self.model.core.state_dict())),
            native_head_sha256=state_digest(cpu_tree(self.model.head.state_dict())),
            native_world_path=str(states_path(self.seed)),
            native_world_stage=cp["native_world_stage"],
            native_world_global_cursor=cp["native_world_global_cursor"],
            protocol_hash=self.protocol_hash))

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self, *, save_audit=False):
        self.synchronize()
        began = time.perf_counter()
        next_step = self.step + 1
        is_replay = self.replay_arm != "none" and next_step % self.replay_every == 0
        start = self.cursor
        self._defer_after = True
        if is_replay:
            row = self.replay_update(save_audit=save_audit)
        else:
            # Numerical operations are the frozen coverage-retention update.
            row = super().update(save_audit=save_audit)
            counts = summarize_rows(self.stream, start, self.cursor)
            self.replay_work["live_gradient_updates"] += 1
            self.replay_work["live_action_gradient_rows"] += counts["action_rows"]
            self.replay_work["live_time_gradient_rows"] += counts["time_rows"]
        self.replay_work["live_forward_updates"] += 1
        self.replay_work["live_forward_ticks"] += self.block
        self.replay_work["live_forward_lane_ticks"] += self.block * self.stream["obs"].shape[1]
        self.synchronize()
        self.replay_work["wall_seconds"] += time.perf_counter() - began
        self._defer_after = False
        if self._pending_after is not None:
            name, self._pending_after = self._pending_after, None
            self.save(name)
        if self.folder is not None:
            append(self.folder / "accounting.jsonl", dict(step=self.step,
                gradient_source="historical_" + self.replay_arm[-1] if is_replay else "live_B",
                replay_work=deepcopy(self.replay_work)))
        return row

    def replay_update(self, *, save_audit=False):
        assert self.warm_completed and self.step < self.updates
        assert self.cursor == self.block * self.step
        start, stop, next_step = self.cursor, self.cursor + self.block, self.step + 1
        assert next_step % self.replay_every == 0
        if save_audit:
            self.save(f"step{next_step:03d}_before")
        before_h = state_digest(self.hidden.detach().cpu())
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        began = time.perf_counter()
        with torch.no_grad():
            live_obs = torch.from_numpy(self.stream["obs"][start:stop]).to(self.device)
            _, next_live_h = self.model(live_obs, self.hidden)
            next_live_h = next_live_h.detach()
        self.synchronize()
        self.replay_work["live_forward_seconds"] += time.perf_counter() - began
        replay_index = next_step // self.replay_every - 1
        selected = self.bank["schedule"][replay_index]
        group = self.bank[self.replay_arm[-1]]
        batch_np = {key: group[key][:, selected] for key in
            (*STREAM_FIELDS, "action_mask")}
        counts = masked_rows(batch_np)
        lanes, starts = group["actor_lane"][selected], group["anchor_start"][selected]
        began = time.perf_counter()
        anchors, prefix = reconstruct_anchors(self.model, self.history, lanes, starts, self.prefix_chunk)
        self.synchronize()
        self.replay_work["prefix_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        batch = {key: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            for key, value in batch_np.items()}
        self.model.train()
        scores, prediction, replay_h = self.model.forward_with_time(batch["obs"], anchors)
        losses = replay_loss(scores, prediction, batch["actions"], batch["time_target"],
            batch["time_mask"], batch["action_mask"])
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        preclip = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
            for name, p in self.model.named_parameters()} if save_audit else None
        norms = clip_gradients(self.model, 1.)
        assert all(torch.isfinite(norm) for norm in norms.values())
        if save_audit:
            torch.save(dict(step=next_step, preclip=preclip,
                clipped={name: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name, p in self.model.named_parameters()},
                preclip_grad_norm={key: float(value) for key, value in norms.items()},
                clip_groups=dict(policy="trainable core+native head", time_head="time_head only"),
                next_hidden=next_live_h.cpu().clone(),
                replay_anchor_hidden=anchors.cpu().clone(), replay_next_hidden=replay_h.detach().cpu().clone(),
                selected_pair_indices=selected.copy(), replay_group=self.replay_arm[-1],
                replay_action_mask=batch_np["action_mask"].copy(),
                replay_time_mask=(batch_np["time_mask"] & batch_np["action_mask"]).copy(),
                prefix=prefix, masked_rows=counts), self.folder / f"step{next_step:03d}_gradients.pt")
        self.optimizer.step()
        self.hidden = next_live_h  # Commit only the actual B state, never replay_h.
        self.cursor, self.step = stop, next_step
        live_counts = summarize_rows(self.stream, start, stop)
        self.work["training_updates"] += 1
        self.work["training_ticks"] += self.block
        self.work["training_lane_ticks"] += self.block * self.stream["obs"].shape[1]
        for name, value in live_counts.items():
            self.work[name] += value
        self.work["action_gradient_rows"] += counts["action_rows"]
        self.work["time_gradient_rows"] += counts["time_rows"]
        self.replay_work["historical_gradient_updates"] += 1
        self.replay_work["historical_window_ticks"] += self.block
        self.replay_work["historical_window_lane_ticks"] += self.block * len(selected)
        self.replay_work["historical_presentations"] += len(selected)
        for name, value in counts.items():
            self.replay_work["historical_" + name] += value
        for name in ("prefix_ticks", "prefix_lane_ticks", "prefix_chunks"):
            self.replay_work[name] += prefix[name]
        self.synchronize()
        self.replay_work["historical_gradient_seconds"] += time.perf_counter() - began
        row = dict(event="update", seed=self.seed, arm="action", experiment_arm=self.replay_arm,
            step=self.step, tick_start=start, tick_stop=stop,
            input_sha256=stream_identity(self.stream, start, stop),
            gradient_source="historical_" + self.replay_arm[-1],
            selected_pair_indices=selected.tolist(),
            action_mask="target slot, all96 postcue positions including real post-FINISH idle WAIT",
            time_mask="action mask intersect active & started & finite ready",
            time_features_detached=True, **live_counts,
            **{"replay_" + key: value for key,value in counts.items()},
            replay_input_sha256=stream_identity(batch_np, 0, self.block),
            loss={key: float(losses[key].detach()) for key in ("total", "action", "time")},
            preclip_grad_norm={key: float(value) for key, value in norms.items()},
            prefix=prefix, prefix_lane_ticks=prefix["prefix_lane_ticks"],
            prefix_chunks=prefix["prefix_chunks"],
            execution_modes=dict(live="train/no_grad",prefix="eval/no_grad",historical="train/grad"),
            hidden_before_sha256=before_h,
            hidden_after_sha256=state_digest(self.hidden.cpu()),
            numerical_state_carried=True, historical_hidden_committed=False)
        if self.folder is not None:
            append(self.folder / "updates.jsonl", row)
        if save_audit:
            self.save(f"step{next_step:03d}_after")
        return row

    @classmethod
    def restore(cls, cp, stream, *, bank=None, history=None, device="cpu", folder=None):
        if isinstance(cp, (str, Path)):
            cp = torch.load(cp, map_location="cpu", weights_only=False)
        assert cp["format"] == FORMAT and cp["arm"] == cp["objective"] == "action"
        model = RemainingTimeGRUAgent(**cp["model_config"])
        model.load_state_dict(cp["model"])
        source = deepcopy(cp)
        source.update(work=deepcopy(cp["source_work"]), stream_cursor=cp["source_stream_cursor"],
            protocol_hash=cp["source_protocol_hash"])
        obj = cls(model.to(device), stream, cp["replay_arm"], source=source,
            bank=bank, history=history, seed=cp["seed"], block=cp["settings"]["block"],
            updates=cp["settings"]["updates"], replay_every=cp["replay_settings"]["every"],
            prefix_chunk=cp["replay_settings"]["prefix_chunk"], folder=folder,
            protocol_hash=cp["protocol_hash"], bindings=cp["bindings"], source_hashes=cp["source_hashes"])
        obj.source_optimizer_step = cp["source_optimizer_step"]
        obj.source_state_hashes = deepcopy(cp["source_state_hashes"])
        obj.cursor, obj.step = cp["stream_cursor"], cp["step"]
        obj.initial_core_hash, obj.initial_head_hash = cp["initial_core_hash"], cp["initial_head_hash"]
        obj.work, obj.replay_work = deepcopy(cp["work"]), deepcopy(cp["replay_work"])
        obj.restore_rng(cp)
        assert obj.cursor == obj.step * obj.block
        assert optimizer_steps(obj.optimizer) == [obj.source_optimizer_step + obj.step]
        return obj


def learner_from_source(seed, arm, stream, *, device="cuda", bank=None, history=None, **kwargs):
    return Learner(initialize(seed, arm, device), stream, arm, source=load_source(seed),
        seed=seed, bank=bank, history=history, **kwargs)


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), Path(__file__).with_name("train_test.py"),
        Path(__file__).with_name("train_audit.py"),
        *frozen.sources(), *bank_module().sources()]))


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    bank = bank_module()
    bank_plan, parent_plan = bank.checked(), frozen.checked()
    assert runtime() == parent_plan["runtime"]
    parent_paths = [frozen.PLAN, frozen.OUT / "globalfreeze.json",
        ROOT / "data/coverage-retention/train-audit-v1/audit_plan.json",
        ROOT / "data/coverage-retention/train-audit-v1/audit.json", bank.PLAN]
    parent_audit = json.loads(parent_paths[3].read_text(encoding="utf-8"))
    assert parent_audit["passed"] and parent_audit["training_protocol_hash"] == parent_plan["protocol_hash"]
    assert parent_audit["audit_plan_sha256"] == sha(parent_paths[2])
    bindings, expected, identities, initial = {}, {}, {}, {}
    for seed in SEEDS:
        stream = load_stream(seed)
        validate_stream(stream, WARM, BLOCK, UPDATES)
        paths = [*parent_paths, *map(Path, bank.input_paths(seed)),
            *map(Path, live_data.input_paths(seed, "withdraw")), source_checkpoint(seed),
            source_checkpoint(seed).with_suffix(".json"), live_data.source_world_path(seed)]
        # Bind every historical source byte and the archived regression target.
        paths.extend(frozen.checkpoint_path(seed, "withdraw", step) for step in SNAPSHOTS)
        bindings[str(seed)] = {str(path): sha(path) for path in dict.fromkeys(paths)}
        expected[str(seed)] = summarize_rows(stream, 0, TRAIN_STEPS)
        identities[str(seed)] = stream_identity(stream, 0, TRAIN_STEPS)
        cp = load_source(seed)
        initial[str(seed)] = {key: state_digest(cp[key]) for key in
            ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng")}
    paths = list(dict.fromkeys([*sources(), *map(Path, extra_sources)]))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), snapshots=list(SNAPSHOTS), audit_steps=list(AUDIT_STEPS),
        runtime=runtime(), source_sha256={str(path): sha(path) for path in paths},
        input_sha256=bindings, bank_protocol_path=str(bank.PLAN), bank_protocol_sha256=sha(bank.PLAN),
        bank_protocol_hash=bank_plan.get("protocol_hash"), parent_protocol_path=str(frozen.PLAN),
        parent_protocol_sha256=sha(frozen.PLAN), parent_audit_sha256=sha(parent_paths[3]),
        initial_states=initial, source_policy="same seed midpoint/action learned step256 model+Adam256+live h+RNG; no warm, reset, zeroing, or reinitialization",
        actual_world="one immutable coverage-retention withdraw stream and actual world per seed, shared by every arm",
        optimizer=dict(kind="Adam", **OPTIMIZER, reset=False, source_step=256, final_step=512,
            clipping="separate policy(core+native head) norm1 and detached-feature time-head norm1"),
        replay=dict(every=REPLAY_EVERY, steps=list(range(8,257,8)), replacements=32,
            batch=BATCH, window_ticks=BLOCK, precue_ticks=32, matched_pairs=256,
            presentations=1024, presentations_per_pair=4, schedule="four fixed seeded shuffle passes, same pair indices A/B",
            live_forward="model.train no_grad full current B128; preserve next_live_h; no B gradient on replacement",
            anchors="current fixed weights model.eval/no_grad whole old public-history prefix from h0; sorted unique lanes, boundaries at all selected starts and multiples512; no cached/stale h",
            prefix_chunk=PREFIX_CHUNK,
            historical_gradient="model.train old128 windows from freshly reconstructed anchors; target slot all96 postcue positions including real post-FINISH idle WAIT; CE WAIT .25 weighted mean + MSE intersect active-started-finite rows; time features detached",
            commit="same Adam once then commit exact pre-update live B h; never historical h",
            labels="historical prefix labels never inputs; fewer historical supervised rows than full B gradient"),
        non_replay="frozen coverage_retention.Learner.update verbatim; first7 updates identical in all arms",
        stream=dict(warm_ticks=0, training_ticks=TRAIN_STEPS, block=BLOCK, batch=BATCH,
            updates=UPDATES, recurrent_state="all B ticks always processed; full numerical live state continuous"),
        expected_live_rows=expected, live_stream_identity=identities,
        work_accounting="work row counts are live B exposures, gradient_rows actual label rows; replay_work separates live forward, live gradient, masked historical categories, historical windows, extra prefix lane ticks, and wall times; source_work separate; no equal FLOPs claim",
        checkpoint="000064256 full model+Adam+h+RNG+counters; actual001/008 before/after+preclip/clipped gradients and live/replay h",
        regression="none exact model+Adam+h+RNG+local cursor+work against old withdraw 000064256",
        no_adaptation="fixed three arms x three seeds; freeze all9 and27 snapshots before any new heldout evaluation; no adaptive budgets, selected snapshot or changed precision")
    plan["optimizer"]["betas"] = list(OPTIMIZER["betas"])
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k:v for k,v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(bank_module().PLAN) == plan["bank_protocol_sha256"]
    assert sha(frozen.PLAN) == plan["parent_protocol_sha256"]
    for path, digest in plan["source_sha256"].items():
        assert sha(path) == digest, "frozen source changed: " + path
    for paths in plan["input_sha256"].values():
        for path, digest in paths.items():
            assert sha(path) == digest, "frozen input changed: " + path
    return plan


def compare_none(seed, cp):
    old = torch.load(frozen.checkpoint_path(seed, "withdraw", cp["step"]), map_location="cpu", weights_only=False)
    fields = ("model", "optimizer", "hidden", "stream_cursor", "step", "work", "source_work",
        "cumulative_work", "torch_rng", "cuda_rng", "numpy_rng")
    for field in fields:
        assert_exact(cp[field], old[field], f"none_regression.{seed}.{cp['step']}.{field}")
    return dict(seed=seed, step=cp["step"], passed=True, fields=list(fields),
        old_checkpoint=str(frozen.checkpoint_path(seed,"withdraw",cp["step"])),
        old_checkpoint_sha256=sha(frozen.checkpoint_path(seed,"withdraw",cp["step"])))


def run_arm(seed, arm, plan, stream=None, bank=None, history=None):
    folder = checkpoint_path(seed, arm).parent
    if folder.exists():
        raise FileExistsError(folder)
    stream = load_stream(seed) if stream is None else stream
    learner = learner_from_source(seed, arm, stream, bank=bank, history=history,
        folder=folder, protocol_hash=plan["protocol_hash"],
        bindings=plan["input_sha256"][str(seed)], source_hashes=plan["source_sha256"])
    learner.save("step000")
    regressions = [compare_none(seed, learner.payload())] if arm == "none" else []
    while learner.step < UPDATES:
        row = learner.update(save_audit=learner.step + 1 in AUDIT_STEPS)
        assert optimizer_steps(learner.optimizer) == [256 + learner.step]
        if learner.step in SNAPSHOTS:
            learner.save(f"step{learner.step:03d}")
            if arm == "none":
                regressions.append(compare_none(seed, learner.payload()))
        if learner.step % 32 == 0:
            print(json.dumps(dict(event="progress", seed=seed, experiment_arm=arm,
                step=learner.step, optimizer_step=256 + learner.step,
                loss=row["loss"], cursor=learner.cursor,
                prefix_lane_ticks=learner.replay_work["prefix_lane_ticks"])), flush=True)
    assert learner.cursor == TRAIN_STEPS
    assert all(learner.work[key] == value for key,value in plan["expected_live_rows"][str(seed)].items())
    assert learner.replay_work["historical_gradient_updates"] == (0 if arm == "none" else 32)
    path = checkpoint_path(seed, arm)
    report = dict(seed=seed, arm=arm, passed=True, final_checkpoint=str(path),
        final_checkpoint_sha256=sha(path), updates=learner.step,
        optimizer_step=256+learner.step, work=learner.work, replay_work=learner.replay_work,
        source_work=learner.source_work, cumulative_work=learner.cumulative_work(),
        none_regression=regressions, protocol_hash=plan["protocol_hash"])
    write_new(folder / "completed.json", report)
    del learner
    torch.cuda.empty_cache()
    return report


def run(seed):
    plan = checked()
    deterministic(820000 + seed)
    assert runtime() == plan["runtime"]
    bank = bank_module().load_bank(seed)
    history, stream = bank_module().load_history(seed), load_stream(seed)
    results = [run_arm(seed, arm, plan, stream, bank, history) for arm in ARMS]
    # Historical A/B gradient labels are matched, although extra prefix work can differ.
    a,b = (row["replay_work"] for row in results[1:])
    for key in ("historical_presentations", "historical_action_rows", "historical_time_rows"):
        assert a[key] == b[key], key
    for key in ("action_rows", "time_rows", "training_updates", "training_lane_ticks"):
        assert len({row["work"][key] for row in results}) == 1
    shared_fields = ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng",
        "stream_cursor", "step", "work", "source_work", "cumulative_work")
    for name in ("step000", "step001_after", "step008_before"):
        cps = [torch.load(checkpoint_path(seed,arm).parent / (name+".pt"),
            map_location="cpu",weights_only=False) for arm in ARMS]
        for key in shared_fields:
            for cp in cps[1:]:
                assert_exact(cps[0][key],cp[key],f"three_arms.{seed}.{name}.{key}")
    write_new(OUT / f"seed_{seed:03d}_completed.json", dict(seed=seed,runs=results,
        same_actual_B_stream=True, paired_historical_label_rows_matched=True,
        initial_and_first_seven_updates_exact_all_three=True))


def freeze_endpoints():
    plan = checked()
    endpoints = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = checkpoint_path(seed,arm)
            report = json.loads((path.parent / "completed.json").read_text(encoding="utf-8"))
            assert report["passed"] and sha(path) == report["final_checkpoint_sha256"]
            snapshots = {}
            for step in SNAPSHOTS:
                snap = checkpoint_path(seed,arm,step)
                meta = snap.with_suffix(".json")
                info = json.loads(meta.read_text(encoding="utf-8"))
                assert info["sha256"] == sha(snap) and info["protocol_hash"] == plan["protocol_hash"]
                snapshots[f"step{step:03d}"] = dict(path=str(snap),sha256=sha(snap),metadata_path=str(meta),metadata_sha256=sha(meta))
            endpoints[f"{seed}:{arm}"] = dict(path=str(path),sha256=sha(path),
                seed=seed,arm=arm,snapshots=snapshots,work=report["work"],replay_work=report["replay_work"])
    result = dict(format="episodic-review-all9-frozen-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"],endpoints=endpoints,endpoint_count=9,snapshot_count=27,
        heldout_scoring_started=False,
        statement="All9 fixed-budget endpoints and27 snapshots frozen before new heldout evaluation")
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json",result)
    print(json.dumps(dict(event="globally_frozen",path=str(OUT / "globalfreeze.json"),freeze_hash=result["freeze_hash"])),flush=True)
    return result


def all_seeds():
    checked()
    pending, active, results = list(SEEDS), [], []
    # A fixed queue, no more than two GPU processes. Each seed's arms are serial.
    while pending or active:
        while pending and len(active) < 2:
            seed = pending.pop(0)
            log = (OUT / f"seed_{seed:03d}.log").open("x",encoding="utf-8")
            proc = subprocess.Popen([sys.executable,"-u","-m","experiments.episodic_review.train","run","--seed",str(seed)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active.append((seed,proc,log))
        for seed,proc,log in list(active):
            status = proc.poll()
            if status is not None:
                log.close()
                active.remove((seed,proc,log))
                results.append(dict(seed=seed,exit_code=status))
        if active:
            time.sleep(.5)
    write_new(OUT / "exit_codes.json",results)
    if any(row["exit_code"] for row in results):
        raise RuntimeError(results)
    return freeze_endpoints()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","run","all","freeze"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--extra-source",action="append",default=[])
    args = parser.parse_args()
    if args.command == "declare":
        declare(args.extra_source)
    elif args.command == "all":
        all_seeds()
    elif args.command == "freeze":
        freeze_endpoints()
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run(args.seed)


if __name__ == "__main__":
    main()
