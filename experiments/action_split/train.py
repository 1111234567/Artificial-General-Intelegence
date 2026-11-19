"""Two fixed, paired single-pass continuations with an explicit split action head.

The archived native teacher stream is consumed once. Numerical recurrent state
is carried across every optimizer boundary; only its graph is detached there.
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

from experiments.action_split import data
from experiments.action_split.model import SplitGRUAgent, split_loss
from experiments.continuity.run import cpu_tree, state_digest
from experiments.learn_remember.gru_train import GRUAgent
from experiments.timing_boundary.data import checkpoint as source_checkpoint

OUT = ROOT / "data/action-split/train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "action-split-training-v1"
HEAD_FORMAT = "per-slot-mode3-begin2-finish8-v1"
ARMS = ("frozen", "full")
SEEDS = (0, 1, 2)
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 512, 32768, 128, 32, 256
SNAPSHOTS = (0, 64, 256)
STREAM_FIELDS = ("obs", "mode_target", "begin_target", "finish_target", "active")
OPTIMIZER = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def checkpoint_path(seed, arm, step=256):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def deterministic(seed):
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def runtime():
    return dict(executable=sys.executable, torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        GPU_name=torch.cuda.get_device_name(0),
        GPU_total_memory=torch.cuda.get_device_properties(0).total_memory,
        CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=True, cudnn_deterministic=True,
        cudnn_benchmark=False, matmul_TF32=False, cudnn_TF32=False,
        dtype="float32", AMP=False)


def sources():
    return [Path(__file__).resolve(), Path(__file__).with_name("train_test.py"),
        ROOT / "experiments/action_split/model.py", ROOT / "experiments/action_split/model_test.py",
        ROOT / "experiments/action_split/data.py", ROOT / "experiments/action_split/data_test.py",
        ROOT / "experiments/learn_remember/gru_train.py", ROOT / "experiments/continuity/run.py",
        ROOT / "experiments/timing_boundary/data.py"]


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    data_plan = data.checked()
    bindings = {}
    expected_counts = {}
    for seed in SEEDS:
        path = data.stream_path(seed)
        original = source_checkpoint(seed, "midpoint")
        bindings[str(seed)] = {str(path): sha(path), str(original): sha(original)}
        stream = data.load_stream(seed, keys=STREAM_FIELDS)
        validate_stream(stream, WARM, BLOCK, UPDATES)
        expected_counts[str(seed)] = summarize_rows(stream, WARM, WARM + TRAIN_STEPS)
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), snapshots=list(SNAPSHOTS), runtime=runtime(),
        source_sha256={str(path): sha(path) for path in [*sources(), *map(Path, extra_sources)]},
        input_sha256=bindings, data_protocol_sha256=sha(data.PLAN),
        data_protocol_hash=data_plan.get("protocol_hash"), source_policy="original midpoint B008 is untouched reference",
        initial_model=dict(width=192, input_size=56, n_jobs=2, n_actions=13,
            total_parameters=149018, original_total_parameters=148246, increase=772,
            recurrent_core="copied only from seed-matched original midpoint B008",
            head="all new mode/begin/finish weights and biases exactly zero; all-WAIT start by argmax ties",
            note="This is a new all-WAIT policy, not a function-preserving replacement of original head."),
        optimizer=dict(kind="Adam", lr=.001, betas=[.9, .999], eps=1e-8,
            weight_decay=0., reset_for_both_arms=True, global_gradient_norm_clip=1.),
        loss=dict(mode="CE every tick/actor/slot; target WAIT weight .25; divide by sum of weights",
            begin="CE correct version only where teacher mode is BEGIN; divide by BEGIN rows",
            finish="8-way CE correct current version and payload on ALL active rows; divide by active rows",
            coefficients=dict(mode=1., begin=1., finish=1.), eligible_filter=False),
        stream=dict(warm_ticks=WARM, training_ticks=TRAIN_STEPS, batch=BATCH,
            slots=2, block=BLOCK, updates=UPDATES, replay_epochs=0,
            common_physical_observations_and_targets_for_both_arms=True,
            recurrent_state="zero only at independent stream origin; teacher warm no gradients/loss; carry numerical h across ALL blocks, detach graph at128; never recompute h after optimizer updates"),
        expected_training_rows=expected_counts,
        arms_detail=dict(frozen="core requires_grad False, core bit-exact before/after; train new head",
            full="same initial parameters and stream, train core and head; head zero makes first core gradient zero; save second actual update audit"),
        checkpoint="step000 after warm, step064, step256; model/config/head format, full Adam, live h, stream cursor, protocol/source hashes, RNG, work; first two real updates before/after and gradients",
        no_adaptation="Exactly two arms, three seeds, fixed256 updates; no LR/epoch/result-dependent selection; freeze all6 endpoints before heldout scoring.")
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(data.PLAN) == plan["data_protocol_sha256"]
    data.checked()
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
    assert stream["active"].dtype == np.bool_
    assert ((stream["mode_target"] >= 0) & (stream["mode_target"] <= 2)).all()
    begin = stream["mode_target"] == 1
    active = stream["active"]
    assert ((stream["begin_target"][begin] >= 0) & (stream["begin_target"][begin] < 2)).all()
    assert ((stream["finish_target"][active] >= 0) & (stream["finish_target"][active] < 8)).all()


def summarize_rows(stream, start, stop):
    mode = stream["mode_target"][start:stop]
    wait = int((mode == 0).sum())
    return dict(mode_rows=int(mode.size), mode_WAIT_rows=wait,
        mode_BEGIN_rows=int((mode == 1).sum()), mode_FINISH_rows=int((mode == 2).sum()),
        mode_weight_sum=float(mode.size - .75 * wait), begin_rows=int((mode == 1).sum()),
        finish_rows=int(stream["active"][start:stop].sum()))


def stream_identity(stream, start, stop):
    digest = hashlib.sha256()
    for name in STREAM_FIELDS:
        array = np.ascontiguousarray(stream[name][start:stop])
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(array.tobytes())
    return digest.hexdigest()


def initialize(seed, arm, device="cuda"):
    assert arm in ARMS
    original_cp = torch.load(source_checkpoint(seed, "midpoint"), map_location="cpu", weights_only=False)
    original = GRUAgent(**original_cp["model_config"])
    original.load_state_dict(original_cp["model"])
    model = SplitGRUAgent.from_original(original, frozen_core=arm == "frozen").to(device)
    assert sum(p.numel() for p in model.parameters()) == 149018
    assert all(p.count_nonzero().item() == 0 for p in model.head.parameters())
    return model


class Learner:
    """Small stateful runner also used by the exact resume/carry tests."""
    def __init__(self, model, stream, arm, *, seed=0, warm=WARM, block=BLOCK,
                 updates=UPDATES, folder=None, protocol_hash="test", bindings=None):
        assert arm in ARMS
        validate_stream(stream, warm, block, updates)
        self.model, self.stream, self.arm, self.seed = model, stream, arm, int(seed)
        self.warm, self.block, self.updates = int(warm), int(block), int(updates)
        self.device = next(model.parameters()).device
        self.model.core.requires_grad_(arm == "full")
        self.model.head.requires_grad_(True)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], **OPTIMIZER)
        self.hidden = self.model.initial_state(stream["obs"].shape[1], device=self.device)
        self.cursor, self.step, self.warm_completed = 0, 0, False
        self.folder = Path(folder) if folder is not None else None
        self.protocol_hash, self.bindings = protocol_hash, deepcopy(bindings or {})
        self.initial_core_hash = state_digest(cpu_tree(self.model.core.state_dict()))
        self.work = dict(warm_ticks=0, warm_lane_ticks=0, warm_loss_rows=0,
            training_updates=0, training_ticks=0, training_lane_ticks=0,
            mode_rows=0, mode_WAIT_rows=0, mode_BEGIN_rows=0, mode_FINISH_rows=0,
            mode_weight_sum=0., begin_rows=0, finish_rows=0)
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
            initial_core_hash=self.initial_core_hash, warm_completed=self.warm_completed,
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
            core_sha256=state_digest(cpu_tree(self.model.core.state_dict())), protocol_hash=self.protocol_hash))

    @classmethod
    def restore(cls, cp, stream, *, device="cpu", folder=None):
        if isinstance(cp, (str, Path)):
            cp = torch.load(cp, map_location="cpu", weights_only=False)
        assert cp["format"] == FORMAT and cp["head_format"] == HEAD_FORMAT
        model = SplitGRUAgent(**cp["model_config"]).to(device)
        model.load_state_dict(cp["model"])
        obj = cls(model, stream, cp["arm"], seed=cp["seed"], folder=folder,
            protocol_hash=cp["protocol_hash"], bindings=cp["bindings"], **cp["settings"])
        obj.optimizer.load_state_dict(cp["optimizer"])
        obj.hidden = cp["hidden"].to(device).detach().clone()
        obj.cursor, obj.step, obj.warm_completed = cp["stream_cursor"], cp["step"], cp["warm_completed"]
        obj.initial_core_hash, obj.work = cp["initial_core_hash"], deepcopy(cp["work"])
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
        scores, next_hidden = self.model(batch["obs"], self.hidden)
        losses = split_loss(scores, batch["mode_target"], batch["begin_target"], batch["finish_target"], batch["active"])
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        gradients_before = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
            for name, p in self.model.named_parameters()} if save_audit else None
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.)
        assert torch.isfinite(norm)
        if save_audit:
            torch.save(dict(step=next_step, preclip=gradients_before,
                clipped={name: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name, p in self.model.named_parameters()}, preclip_grad_norm=float(norm),
                next_hidden=next_hidden.detach().cpu().clone()), self.folder / f"step{next_step:03d}_gradients.pt")
        self.optimizer.step()
        # Carry the state computed before this optimizer step. No replay/recompute.
        self.hidden = next_hidden.detach()
        self.cursor, self.step = stop, next_step
        counts = summarize_rows(self.stream, start, stop)
        self.work["training_updates"] += 1
        self.work["training_ticks"] += self.block
        self.work["training_lane_ticks"] += self.block * self.stream["obs"].shape[1]
        for name, value in counts.items():
            self.work[name] += value
        if self.arm == "frozen":
            assert state_digest(cpu_tree(self.model.core.state_dict())) == self.initial_core_hash
        record = dict(event="update", seed=self.seed, arm=self.arm, step=self.step,
            tick_start=start, tick_stop=stop, input_sha256=stream_identity(self.stream, start, stop),
            mode_mask="every row", begin_mask="mode_target==1", finish_mask="active; no eligibility filter",
            **counts, loss={key: float(losses[key].detach()) for key in ("total", "mode", "begin", "finish")},
            preclip_grad_norm=float(norm), hidden_before_sha256=before_h,
            hidden_after_sha256=state_digest(self.hidden.cpu()), numerical_state_carried=True)
        if self.folder is not None:
            append(self.folder / "updates.jsonl", record)
        if save_audit:
            self.save(f"step{next_step:03d}_after")
        return record


def run_arm(seed, arm, plan):
    deterministic(810000 + seed)
    folder = OUT / f"seed_{seed:03d}" / arm
    if folder.exists():
        raise FileExistsError(folder)
    stream = data.load_stream(seed, keys=STREAM_FIELDS)
    learner = Learner(initialize(seed, arm), stream, arm, seed=seed, folder=folder,
        protocol_hash=plan["protocol_hash"], bindings=plan["input_sha256"][str(seed)])
    learner.warm_start()
    learner.save("step000")
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
    final_core_hash = state_digest(cpu_tree(learner.model.core.state_dict()))
    assert (final_core_hash == learner.initial_core_hash) == (arm == "frozen")
    report = dict(seed=seed, arm=arm, passed=True, updates=learner.step, work=learner.work,
        initial_core_sha256=learner.initial_core_hash, final_core_sha256=final_core_hash,
        frozen_core_bit_exact=arm == "frozen", full_core_changed=arm == "full",
        final_checkpoint=str(folder / "step256.pt"), final_checkpoint_sha256=sha(folder / "step256.pt"))
    write_new(folder / "completed.json", report)
    del learner, stream
    torch.cuda.empty_cache()
    return report


def run(seed):
    plan = checked()
    deterministic(810000 + seed)
    assert runtime() == plan["runtime"], "worker runtime drift"
    results = [run_arm(seed, arm, plan) for arm in ARMS]
    assert results[0]["work"] == results[1]["work"]
    write_new(OUT / f"seed_{seed:03d}" / "completed.json", dict(seed=seed, arms=results, matched_work=True))


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
    result = dict(format="action-split-all6-frozen-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], endpoints=endpoints,
        heldout_scoring_started=False, statement="All6 fixed-budget endpoints frozen before heldout scoring")
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json", result)
    return result


def all_seeds():
    checked()
    workers = []
    for seed in SEEDS:
        log = (OUT / f"seed_{seed:03d}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.action_split.train", "run", "--seed", str(seed)],
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
