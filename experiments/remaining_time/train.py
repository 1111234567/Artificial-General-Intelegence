"""Four fixed native-policy continuations on one archived stream per seed.

The original core and native action head are copied exactly.  A zero diagnostic
time head never controls actions.  Numerical h crosses every optimizer boundary
without being reset or recomputed; its graph is detached at those boundaries.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.action_split import data as raw_data
from experiments.action_split.train import (sha, write_new, append, deterministic,
    runtime)
from experiments.continuity.run import cpu_tree, state_digest
from experiments.learn_remember.gru_train import GRUAgent
from experiments.remaining_time import targets
from experiments.remaining_time.model import RemainingTimeGRUAgent, experiment_loss, clip_gradients

OUT = ROOT / "data/remaining-time/train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "remaining-time-native-policy-training-v1"
HEAD_FORMAT = "original-native11-plus-zero-linear-time2-v1"
ARMS = ("probe", "time", "action", "joint")
SEEDS = (0, 1, 2)
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 512, 32768, 128, 32, 256
SNAPSHOTS = (0, 64, 256)
STREAM_FIELDS = ("obs", "actions", "time_target", "time_mask")
OPTIMIZER = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def checkpoint_path(seed, arm, step=256):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


def original_checkpoint(seed):
    assert int(seed) in SEEDS
    return ROOT / f"data/finish-position/train-v1/seed_{int(seed):03d}/midpoint/B008.pt"


def load_stream(seed):
    stream = raw_data.load_stream(seed, keys=("obs", "actions"))
    stream.update(targets.load_targets(seed, keys=("time_target", "time_mask")))
    return stream


def sources():
    return [Path(__file__).resolve(), Path(__file__).with_name("train_test.py"),
        Path(__file__).with_name("model.py"), Path(__file__).with_name("model_test.py"),
        Path(__file__).with_name("targets.py"), Path(__file__).with_name("targets_test.py"),
        ROOT / "experiments/action_split/train.py", ROOT / "experiments/action_split/data.py",
        ROOT / "experiments/learn_remember/gru_train.py", ROOT / "experiments/continuity/run.py"]


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    raw_data.checked()
    target_plan = targets.checked()
    bindings, expected_counts = {}, {}
    for seed in SEEDS:
        raw_path, aux_path = raw_data.stream_path(seed), targets.target_path(seed)
        paths = [raw_path, raw_path.with_suffix(".json"), raw_path.parent / "world_states.pt",
            raw_path.parent / "raw_audit.json", aux_path, aux_path.with_suffix(".json"),
            original_checkpoint(seed)]
        bindings[str(seed)] = {str(path): sha(path) for path in paths}
        stream = load_stream(seed)
        validate_stream(stream, WARM, BLOCK, UPDATES)
        expected_counts[str(seed)] = summarize_rows(stream, WARM, WARM + TRAIN_STEPS)
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), snapshots=list(SNAPSHOTS), runtime=runtime(),
        source_sha256={str(path): sha(path) for path in [*sources(), *map(Path, extra_sources)]},
        input_sha256=bindings, raw_protocol_sha256=sha(raw_data.PLAN),
        target_protocol_sha256=sha(targets.PLAN), target_protocol_hash=target_plan.get("protocol_hash"),
        source_policy="Seed-matched original midpoint B008; both native core and 11-action head copied bit-exact; original artifacts untouched",
        initial_model=dict(width=192, input_size=56, n_jobs=2, n_actions=11,
            total_parameters=148632, original_total_parameters=148246, increase=386,
            time_head="new linear192->2 weights and biases exactly zero; diagnostic only",
            preserved_native_function="forward and step inherit original GRU operations; native argmax unchanged in all4 arms"),
        optimizer=dict(kind="Adam", lr=.001, betas=[.9, .999], eps=1e-8,
            weight_decay=0., reset_for_all_arms=True, single_optimizer_per_arm=True,
            clipping="separate clip_grad_norm_(trainable core+native head,1) and clip_grad_norm_(time head,1); time-head gradients never rescale native-policy group"),
        loss=dict(action="native11 CE every tick and slot; WAIT weight .25, all other classes1; weighted mean divides by sum of target class weights",
            time="MSE prediction vs signed (ready-now)/16 in physical time units; mean over active & started & finite ready rows; exactly zero otherwise; no eligibility or future-outcome selection",
            coefficients=dict(action=1., time=1.), time_prediction_never_controls_action=True,
            no_oracle_in_inputs=True),
        arms_detail=dict(probe="freeze core+native action head; train only time head by MSE",
            time="train core+time head by MSE only; native action head frozen",
            action="train core+native action head by native weighted CE; concurrently train time head by MSE on detached h; independent gradient clipping",
            joint="train core+native action head+time head by CE+MSE"),
        first_update="zero time head makes auxiliary core gradient exactly zero; action/joint core can move immediately through preserved native CE",
        stream=dict(warm_ticks=WARM, training_ticks=TRAIN_STEPS, batch=BATCH, slots=2,
            block=BLOCK, updates=UPDATES, replay_epochs=0, same_raw_obs_actions_all4=True,
            inputs="original public56 observation only; constant dt accepted by native API but not appended",
            recurrent_state="zero only at independent stream origin; first512 warm with no loss/grad; carry numerical h across every update; detach graph at128; never recompute h after update"),
        expected_training_rows=expected_counts,
        checkpoint="step000 after warm, step064, step256: full model, Adam, live h, cursor, work, RNG, protocol/source/input hashes; first two actual updates before/after+preclip/clipped gradients",
        no_adaptation="Exactly four arms x three seeds, fixed256 updates; no sweep, repeated epochs, best-snapshot selection or heldout-directed change; freeze all12 endpoints and36 snapshots before any heldout scoring")
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(raw_data.PLAN) == plan["raw_protocol_sha256"]
    assert sha(targets.PLAN) == plan["target_protocol_sha256"]
    raw_data.checked(); targets.checked()
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, "frozen source changed: " + path
    for items in plan["input_sha256"].values():
        for path, expected in items.items():
            assert sha(path) == expected, "frozen input changed: " + path
    return plan


def validate_stream(stream, warm, block, updates):
    steps, batch, inputs = stream["obs"].shape
    assert steps == warm + block * updates and inputs == 56
    assert stream["obs"].dtype == np.float32
    for name in STREAM_FIELDS[1:]:
        assert stream[name].shape == (steps, batch, 2), name
    assert stream["actions"].dtype == np.int64
    assert ((stream["actions"] >= 0) & (stream["actions"] <= 10)).all()
    assert stream["time_target"].dtype == np.float32 and np.isfinite(stream["time_target"]).all()
    assert stream["time_mask"].dtype == np.bool_
    assert (stream["time_target"][~stream["time_mask"]] == 0).all()


def summarize_rows(stream, start, stop):
    action = stream["actions"][start:stop]
    wait = int((action == 0).sum())
    return dict(action_rows=int(action.size), action_WAIT_rows=wait,
        action_BEGIN_rows=int(((action >= 1) & (action <= 2)).sum()),
        action_FINISH_rows=int((action >= 3).sum()), action_weight_sum=float(action.size - .75 * wait),
        time_rows=int(stream["time_mask"][start:stop].sum()))


def stream_identity(stream, start, stop):
    digest = hashlib.sha256()
    for name in STREAM_FIELDS:
        array = np.ascontiguousarray(stream[name][start:stop])
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(array.tobytes())
    return digest.hexdigest()


def initialize(seed, arm, device="cuda"):
    assert arm in ARMS
    cp = torch.load(original_checkpoint(seed), map_location="cpu", weights_only=False)
    original = GRUAgent(**cp["model_config"])
    original.load_state_dict(cp["model"])
    model = RemainingTimeGRUAgent.from_original(original, arm).to(device)
    assert sum(p.numel() for p in model.parameters()) == 148632
    assert state_digest(model.core.state_dict()) == state_digest(original.core.state_dict())
    assert state_digest(model.head.state_dict()) == state_digest(original.head.state_dict())
    assert all(p.count_nonzero().item() == 0 for p in model.time_head.parameters())
    return model


class Learner:
    """Stateful single-pass runner used by training and real resume tests."""
    def __init__(self, model, stream, arm, *, seed=0, warm=WARM, block=BLOCK,
                 updates=UPDATES, folder=None, protocol_hash="test", bindings=None,
                 source_hashes=None):
        assert arm in ARMS
        validate_stream(stream, warm, block, updates)
        self.model, self.stream, self.arm, self.seed = model, stream, arm, int(seed)
        self.warm, self.block, self.updates = int(warm), int(block), int(updates)
        self.device = next(model.parameters()).device
        self.model.set_arm(arm)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], **OPTIMIZER)
        self.hidden = self.model.initial_state(stream["obs"].shape[1], device=self.device)
        self.cursor, self.step, self.warm_completed = 0, 0, False
        self.folder = Path(folder) if folder is not None else None
        self.protocol_hash, self.bindings = protocol_hash, deepcopy(bindings or {})
        self.source_hashes = deepcopy(source_hashes or {})
        self.initial_core_hash = state_digest(cpu_tree(self.model.core.state_dict()))
        self.initial_head_hash = state_digest(cpu_tree(self.model.head.state_dict()))
        self.work = dict(warm_ticks=0, warm_lane_ticks=0, warm_loss_rows=0,
            training_updates=0, training_ticks=0, training_lane_ticks=0,
            action_rows=0, action_WAIT_rows=0, action_BEGIN_rows=0,
            action_FINISH_rows=0, action_weight_sum=0., time_rows=0,
            action_gradient_rows=0, time_gradient_rows=0)
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=False)

    def warm_start(self):
        assert self.cursor == self.step == 0 and not self.warm_completed
        self.model.eval()
        with torch.no_grad():
            _, self.hidden = self.model(torch.from_numpy(self.stream["obs"][:self.warm]).to(self.device), self.hidden)
        self.hidden = self.hidden.detach()
        self.cursor, self.warm_completed = self.warm, True
        self.work["warm_ticks"] = self.warm
        self.work["warm_lane_ticks"] = self.warm * self.stream["obs"].shape[1]
        assert not self.optimizer.state

    def payload(self):
        return dict(format=FORMAT, model=cpu_tree(self.model.state_dict()), model_config=self.model.config(),
            head_format=HEAD_FORMAT, optimizer=cpu_tree(self.optimizer.state_dict()),
            hidden=self.hidden.detach().cpu().clone(), stream_cursor=self.cursor, step=self.step,
            arm=self.arm, seed=self.seed, protocol_hash=self.protocol_hash, bindings=deepcopy(self.bindings),
            source_hashes=deepcopy(self.source_hashes), initial_core_hash=self.initial_core_hash,
            initial_head_hash=self.initial_head_hash, warm_completed=self.warm_completed,
            settings=dict(warm=self.warm, block=self.block, updates=self.updates), work=deepcopy(self.work),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            numpy_rng=np.random.get_state())

    def save(self, name):
        assert self.folder is not None
        path = self.folder / (name + ".pt")
        if path.exists():
            raise FileExistsError(path)
        torch.save(self.payload(), path)
        write_new(path.with_suffix(".json"), dict(path=str(path), sha256=sha(path), step=self.step,
            stream_cursor=self.cursor, arm=self.arm, seed=self.seed, work=self.work,
            hidden_sha256=state_digest(self.hidden.detach().cpu()),
            core_sha256=state_digest(cpu_tree(self.model.core.state_dict())),
            native_head_sha256=state_digest(cpu_tree(self.model.head.state_dict())), protocol_hash=self.protocol_hash))

    @classmethod
    def restore(cls, cp, stream, *, device="cpu", folder=None):
        if isinstance(cp, (str, Path)):
            cp = torch.load(cp, map_location="cpu", weights_only=False)
        assert cp["format"] == FORMAT and cp["head_format"] == HEAD_FORMAT
        model = RemainingTimeGRUAgent(**cp["model_config"]).to(device)
        model.load_state_dict(cp["model"])
        obj = cls(model, stream, cp["arm"], seed=cp["seed"], folder=folder,
            protocol_hash=cp["protocol_hash"], bindings=cp["bindings"],
            source_hashes=cp["source_hashes"], **cp["settings"])
        obj.optimizer.load_state_dict(cp["optimizer"])
        obj.hidden = cp["hidden"].to(device).detach().clone()
        obj.cursor, obj.step, obj.warm_completed = cp["stream_cursor"], cp["step"], cp["warm_completed"]
        obj.initial_core_hash, obj.initial_head_hash = cp["initial_core_hash"], cp["initial_head_hash"]
        obj.work = deepcopy(cp["work"])
        assert obj.cursor == obj.warm + obj.step * obj.block and obj.warm_completed
        torch.set_rng_state(cp["torch_rng"])
        np.random.set_state(cp["numpy_rng"])
        if obj.device.type == "cuda":
            torch.cuda.set_rng_state_all(cp["cuda_rng"])
        return obj

    def update(self, *, save_audit=False):
        assert self.warm_completed and self.step < self.updates
        assert self.cursor == self.warm + self.block * self.step
        start, stop, next_step = self.cursor, self.cursor + self.block, self.step + 1
        if save_audit:
            self.save(f"step{next_step:03d}_before")
        before_h = state_digest(self.hidden.detach().cpu())
        batch = {name: torch.from_numpy(self.stream[name][start:stop]).to(self.device) for name in STREAM_FIELDS}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        scores, time_prediction, next_hidden = self.model.forward_with_time(batch["obs"], self.hidden)
        losses = experiment_loss(scores, time_prediction, batch["actions"], batch["time_target"], batch["time_mask"], self.arm)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        gradients_before = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
            for name, p in self.model.named_parameters()} if save_audit else None
        norms = clip_gradients(self.model, 1.)
        assert all(torch.isfinite(norm) for norm in norms.values())
        if save_audit:
            torch.save(dict(step=next_step, preclip=gradients_before,
                clipped={name: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name, p in self.model.named_parameters()},
                preclip_grad_norm={key: float(value) for key, value in norms.items()},
                clip_groups=dict(policy="trainable core+native head", time_head="time_head only"),
                next_hidden=next_hidden.detach().cpu().clone()), self.folder / f"step{next_step:03d}_gradients.pt")
        self.optimizer.step()
        self.hidden = next_hidden.detach()  # Carry the pre-update state, never replay.
        self.cursor, self.step = stop, next_step
        counts = summarize_rows(self.stream, start, stop)
        self.work["training_updates"] += 1
        self.work["training_ticks"] += self.block
        self.work["training_lane_ticks"] += self.block * self.stream["obs"].shape[1]
        for name, value in counts.items():
            self.work[name] += value
        self.work["action_gradient_rows"] += counts["action_rows"] if self.arm in ("action", "joint") else 0
        self.work["time_gradient_rows"] += counts["time_rows"]
        if self.arm == "probe":
            assert state_digest(cpu_tree(self.model.core.state_dict())) == self.initial_core_hash
        if self.arm in ("probe", "time"):
            assert state_digest(cpu_tree(self.model.head.state_dict())) == self.initial_head_hash
        record = dict(event="update", seed=self.seed, arm=self.arm, step=self.step,
            tick_start=start, tick_stop=stop, input_sha256=stream_identity(self.stream, start, stop),
            action_mask="every row", time_mask="active & started & finite ready; no eligibility filter",
            time_features_detached=self.arm == "action", **counts,
            loss={key: float(losses[key].detach()) for key in ("total", "action", "time")},
            preclip_grad_norm={key: float(value) for key, value in norms.items()},
            hidden_before_sha256=before_h, hidden_after_sha256=state_digest(self.hidden.cpu()),
            numerical_state_carried=True)
        if self.folder is not None:
            append(self.folder / "updates.jsonl", record)
        if save_audit:
            self.save(f"step{next_step:03d}_after")
        return record


def run_arm(seed, arm, plan):
    deterministic(820000 + seed)
    folder = OUT / f"seed_{seed:03d}" / arm
    if folder.exists():
        raise FileExistsError(folder)
    stream = load_stream(seed)
    learner = Learner(initialize(seed, arm), stream, arm, seed=seed, folder=folder,
        protocol_hash=plan["protocol_hash"], bindings=plan["input_sha256"][str(seed)],
        source_hashes=plan["source_sha256"])
    learner.warm_start(); learner.save("step000")
    while learner.step < UPDATES:
        row = learner.update(save_audit=learner.step < 2)
        if learner.step in SNAPSHOTS:
            learner.save(f"step{learner.step:03d}")
        if learner.step % 32 == 0:
            print(json.dumps(dict(event="progress", seed=seed, arm=arm, step=learner.step,
                loss=row["loss"], cursor=learner.cursor)), flush=True)
    expected = plan["expected_training_rows"][str(seed)]
    assert all(learner.work[key] == value for key, value in expected.items())
    assert learner.cursor == WARM + TRAIN_STEPS
    final_core = state_digest(cpu_tree(learner.model.core.state_dict()))
    final_head = state_digest(cpu_tree(learner.model.head.state_dict()))
    assert (final_core == learner.initial_core_hash) == (arm == "probe")
    assert (final_head == learner.initial_head_hash) == (arm in ("probe", "time"))
    report = dict(seed=seed, arm=arm, passed=True, updates=learner.step, work=learner.work,
        initial_core_sha256=learner.initial_core_hash, final_core_sha256=final_core,
        initial_native_head_sha256=learner.initial_head_hash, final_native_head_sha256=final_head,
        core_bit_exact=arm == "probe", native_head_bit_exact=arm in ("probe", "time"),
        final_checkpoint=str(folder / "step256.pt"), final_checkpoint_sha256=sha(folder / "step256.pt"))
    write_new(folder / "completed.json", report)
    del learner, stream
    torch.cuda.empty_cache()
    return report


def run(seed):
    plan = checked()
    deterministic(820000 + seed)
    assert runtime() == plan["runtime"], "worker runtime drift"
    results = [run_arm(seed, arm, plan) for arm in ARMS]
    shared = ("warm_ticks", "warm_lane_ticks", "warm_loss_rows", "training_updates",
        "training_ticks", "training_lane_ticks", "action_rows", "action_WAIT_rows",
        "action_BEGIN_rows", "action_FINISH_rows", "action_weight_sum", "time_rows", "time_gradient_rows")
    assert all(all(row["work"][key] == results[0]["work"][key] for key in shared) for row in results)
    write_new(OUT / f"seed_{seed:03d}" / "completed.json", dict(seed=seed, arms=results,
        matched_stream_and_available_rows=True, action_loss_arms_only=["action", "joint"]))


def freeze_endpoints():
    plan = checked()
    endpoints = {}
    for seed in SEEDS:
        for arm in ARMS:
            folder = OUT / f"seed_{seed:03d}" / arm
            report = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
            path = folder / "step256.pt"
            assert report["passed"] and sha(path) == report["final_checkpoint_sha256"]
            snapshots = {}
            for step in SNAPSHOTS:
                snap = checkpoint_path(seed, arm, step)
                meta = snap.with_suffix(".json")
                snap_meta = json.loads(meta.read_text(encoding="utf-8"))
                assert snap_meta["sha256"] == sha(snap)
                snapshots[f"step{step:03d}"] = dict(path=str(snap), sha256=sha(snap),
                    metadata_path=str(meta), metadata_sha256=sha(meta))
            endpoints[f"{seed}:{arm}"] = dict(path=str(path), sha256=sha(path),
                snapshots=snapshots, work=report["work"])
    result = dict(format="remaining-time-all12-frozen-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], endpoints=endpoints, endpoint_count=12, snapshot_count=36,
        heldout_scoring_started=False, statement="All12 fixed-budget endpoints and36 snapshots frozen before heldout scoring")
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json", result)
    return result


def all_seeds():
    checked()
    workers = []
    for seed in SEEDS:
        log = (OUT / f"seed_{seed:03d}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.remaining_time.train", "run", "--seed", str(seed)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        workers.append((seed, process, log))
    results = []
    for seed, process, log in workers:
        results.append(dict(seed=seed, exit_code=process.wait()))
        log.close()
    write_new(OUT / "exit_codes.json", results)
    if any(result["exit_code"] for result in results):
        raise RuntimeError(results)
    freeze_endpoints()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "all", "freeze"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--extra-source", action="append", default=[])
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
