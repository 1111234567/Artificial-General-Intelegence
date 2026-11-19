"""Nine actual continuations of the learned midpoint/action endpoint.

Only the future native-world stream changes. Model, Adam, numerical recurrent
state and RNG are restored from the same seed-matched learned endpoint. The
frozen remaining_time Learner supplies every numerical optimizer operation.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.action_split.train import deterministic, runtime, sha, write_new
from experiments.continuity.run import cpu_tree, state_digest
from experiments.remaining_time import train as previous
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.teacher_timing import train as source_train
from experiments.coverage_retention import data

OUT = ROOT / "data/coverage-retention/train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "coverage-retention-learned-endpoint-continuation-v1"
ARMS = ("control", "wide", "withdraw")
SEEDS = (0, 1, 2)
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 0, 32768, 128, 32, 256
SNAPSHOTS = (0, 64, 256)
HEAD_FORMAT = previous.HEAD_FORMAT
STREAM_FIELDS = previous.STREAM_FIELDS
OPTIMIZER = previous.OPTIMIZER
validate_stream = previous.validate_stream
summarize_rows = previous.summarize_rows
stream_identity = previous.stream_identity


def checkpoint_path(seed, arm, step=256):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


def source_checkpoint(seed):
    return source_train.checkpoint_path(seed, "midpoint", "action", 256)


original_checkpoint = source_checkpoint


def load_stream(seed, arm):
    assert int(seed) in SEEDS and arm in ARMS
    return data.load_stream(seed, arm)


def optimizer_steps(optimizer):
    states = optimizer["state"] if isinstance(optimizer, dict) else optimizer.state_dict()["state"]
    return sorted({int(value["step"]) for value in states.values()})


def assert_exact(left, right, name="value"):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape, name
        assert np.array_equal(left, right), name
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), name
        for key in left:
            assert_exact(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), name
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{name}[{index}]")
    else:
        source_train.assert_exact(left, right, name)


def load_source(seed):
    cp = torch.load(source_checkpoint(seed), map_location="cpu", weights_only=False)
    assert cp["arm"] == "action" and cp["step"] == 256
    assert cp["stream_cursor"] == 512 + 32768 and cp["warm_completed"]
    assert cp["hidden"].shape == (BATCH, 192)
    assert cp["work"]["training_updates"] == 256
    assert optimizer_steps(cp["optimizer"]) == [256]
    assert len(cp["optimizer"]["state"]) == 8
    assert sum(float(state["exp_avg"].abs().sum()) for state in cp["optimizer"]["state"].values()) > 0
    assert any(bool(value.any()) for key, value in cp["model"].items() if key.startswith("time_head."))
    return cp


def initialize(seed, arm="control", device="cuda"):
    """Load all learned modules on CPU, then move the existing policy to device."""
    assert arm in ARMS
    cp = load_source(seed)
    model = RemainingTimeGRUAgent(**cp["model_config"])
    model.load_state_dict(cp["model"], strict=True)
    model.set_arm("action")
    assert_exact(cpu_tree(model.state_dict()), cp["model"], "source_model")
    return model.to(device)


class Learner(previous.Learner):
    """Exact source-state copy with explicit local counters and no warm pass.

    `arm` remains action because it selects the frozen loss implementation;
    `experiment_arm` identifies the changed future environment. The inherited
    update is used verbatim, including detach, separate clipping and Adam.
    """
    def __init__(self, model, stream, experiment_arm, *, source, seed=0,
                 block=BLOCK, updates=UPDATES, folder=None, protocol_hash="test",
                 bindings=None, source_hashes=None):
        assert experiment_arm in ARMS and source["arm"] == "action"
        super().__init__(model, stream, "action", seed=seed, warm=0, block=block,
            updates=updates, folder=folder, protocol_hash=protocol_hash,
            bindings=bindings, source_hashes=source_hashes)
        self.experiment_arm = experiment_arm
        self.source_checkpoint = str(source_checkpoint(seed))
        self.source_optimizer_step = optimizer_steps(source["optimizer"])[0]
        self.source_stream_cursor = int(source["stream_cursor"])
        self.source_work = deepcopy(source["work"])
        self.source_protocol_hash = source["protocol_hash"]
        self.source_state_hashes = {key: state_digest(source[key]) for key in ("model", "optimizer", "hidden")}
        self.optimizer.load_state_dict(deepcopy(source["optimizer"]))
        self.hidden = source["hidden"].to(self.device).detach().clone()
        self.cursor, self.step, self.warm_completed = 0, 0, True
        # Model construction consumes RNG; copying the archived RNG is the final
        # initialization operation, before the first new observation is read.
        self.restore_rng(source)
        assert_exact(cpu_tree(self.model.state_dict()), source["model"], "model_copy")
        assert_exact(cpu_tree(self.optimizer.state_dict()), source["optimizer"], "Adam_copy")
        assert_exact(self.hidden.cpu(), source["hidden"], "h_copy")
        assert all(p.grad is None for p in self.model.parameters())

    def restore_rng(self, cp):
        torch.set_rng_state(cp["torch_rng"])
        np.random.set_state(cp["numpy_rng"])
        if self.device.type == "cuda" and cp.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(cp["cuda_rng"])

    def warm_start(self):
        raise RuntimeError("Learned-state continuation has no warm operation")

    def cumulative_work(self):
        assert self.source_work.keys() == self.work.keys()
        return {key: self.source_work[key] + self.work[key] for key in self.work}

    def payload(self):
        cp = super().payload()
        cp.update(format=FORMAT, experiment_arm=self.experiment_arm, objective="action",
            source_checkpoint=self.source_checkpoint, source_protocol_hash=self.source_protocol_hash,
            source_optimizer_step=self.source_optimizer_step,
            cumulative_optimizer_step=self.source_optimizer_step + self.step,
            optimizer_state_steps=optimizer_steps(self.optimizer), local_optimizer_updates=self.step,
            source_stream_cursor=self.source_stream_cursor,
            cumulative_stream_cursor=self.source_stream_cursor + self.cursor,
            source_work=deepcopy(self.source_work), cumulative_work=self.cumulative_work(),
            source_state_hashes=deepcopy(self.source_state_hashes),
            source_world_path=str(data.source_world_path(self.seed)), source_world_stage="final",
            native_world_path=str(data.states_path(self.seed, self.experiment_arm)),
            native_world_stage="after_change" if self.step == 0 else ("final" if self.step == self.updates else None),
            native_world_global_cursor=self.source_stream_cursor + self.cursor,
            native_world_tick=self.source_stream_cursor + self.cursor - 1,
            native_world_now=(self.source_stream_cursor + self.cursor - 1) * .25,
            initial_hidden_sha256=self.source_state_hashes["hidden"],
            initialization="complete learned model+Adam+h+RNG restored; no warm; no h recomputation",
            carried_hidden_semantics="source and continued h were produced before the immediately preceding optimizer update; carried numerically unchanged")
        return cp

    def save(self, name):
        assert self.folder is not None
        path = self.folder / (name + ".pt")
        if path.exists():
            raise FileExistsError(path)
        cp = self.payload()
        torch.save(cp, path)
        write_new(path.with_suffix(".json"), dict(path=str(path), sha256=sha(path),
            step=self.step, stream_cursor=self.cursor, arm="action", objective="action",
            experiment_arm=self.experiment_arm, seed=self.seed, work=self.work,
            source_work=self.source_work, cumulative_work=self.cumulative_work(),
            source_optimizer_step=self.source_optimizer_step,
            cumulative_optimizer_step=self.source_optimizer_step + self.step,
            optimizer_state_steps=optimizer_steps(self.optimizer),
            hidden_sha256=state_digest(self.hidden.detach().cpu()),
            initial_hidden_sha256=self.source_state_hashes["hidden"],
            core_sha256=state_digest(cpu_tree(self.model.core.state_dict())),
            native_head_sha256=state_digest(cpu_tree(self.model.head.state_dict())),
            source_world_path=cp["source_world_path"], source_world_stage="final",
            native_world_path=cp["native_world_path"], native_world_stage=cp["native_world_stage"],
            native_world_global_cursor=cp["native_world_global_cursor"],
            native_world_tick=cp["native_world_tick"], native_world_now=cp["native_world_now"],
            protocol_hash=self.protocol_hash))

    @classmethod
    def restore(cls, cp, stream, *, device="cpu", folder=None):
        if isinstance(cp, (str, Path)):
            cp = torch.load(cp, map_location="cpu", weights_only=False)
        assert cp["format"] == FORMAT and cp["objective"] == cp["arm"] == "action"
        assert cp["settings"]["warm"] == 0 and cp["warm_completed"]
        model = RemainingTimeGRUAgent(**cp["model_config"])
        model.load_state_dict(cp["model"])
        model = model.to(device)
        # Create the numerical runner, without consuming source observations or
        # resetting moments; restore the disk checkpoint directly afterwards.
        source = deepcopy(cp)
        source.update(work=deepcopy(cp["source_work"]),
            stream_cursor=cp["source_stream_cursor"], protocol_hash=cp["source_protocol_hash"])
        obj = cls(model, stream, cp["experiment_arm"], source=source, seed=cp["seed"],
            block=cp["settings"]["block"], updates=cp["settings"]["updates"], folder=folder,
            protocol_hash=cp["protocol_hash"], bindings=cp["bindings"], source_hashes=cp["source_hashes"])
        obj.source_optimizer_step = cp["source_optimizer_step"]
        obj.source_state_hashes = deepcopy(cp["source_state_hashes"])
        obj.cursor, obj.step = cp["stream_cursor"], cp["step"]
        obj.initial_core_hash, obj.initial_head_hash = cp["initial_core_hash"], cp["initial_head_hash"]
        obj.work = deepcopy(cp["work"])
        obj.restore_rng(cp)
        assert obj.cursor == obj.step * obj.block
        assert optimizer_steps(obj.optimizer) == [obj.source_optimizer_step + obj.step]
        assert_exact(obj.cumulative_work(), cp["cumulative_work"], "resume_work")
        return obj


def learner_from_source(seed, arm, stream, *, device="cuda", **kwargs):
    source = load_source(seed)
    model = initialize(seed, arm, device=device)
    return Learner(model, stream, arm, source=source, seed=seed, **kwargs)


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), Path(__file__).with_name("train_test.py"),
        Path(__file__).with_name("train_audit.py"),
        ROOT / "experiments/teacher_timing/train.py", *previous.sources(), *data.sources()]))


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    data_plan = data.checked()
    old_plan = source_train.checked()
    assert runtime() == old_plan["runtime"]
    completion_paths = [data.OUT / "completion.json", data.OUT / "audit_completion.json"]
    completion, audit_completion = [json.loads(path.read_text(encoding="utf-8")) for path in completion_paths]
    for report in (completion, audit_completion):
        assert report["complete"] and report["seeds"] == list(SEEDS) and report["arms"] == list(ARMS)
        assert report["protocol_sha256"] == sha(data.PLAN)
    bindings, counts, identities, initial_states = {}, {}, {}, {}
    for seed in SEEDS:
        cp = load_source(seed)
        initial_states[str(seed)] = dict(source_checkpoint=str(source_checkpoint(seed)),
            source_checkpoint_sha256=sha(source_checkpoint(seed)),
            source_world_path=str(data.source_world_path(seed)), source_world_stage="final",
            state_sha256={key: state_digest(cp[key]) for key in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng")},
            source_optimizer_step=256, source_stream_cursor=33280)
        bindings[str(seed)], counts[str(seed)], identities[str(seed)] = {}, {}, {}
        for arm in ARMS:
            assert completion["metadata"][str(seed)][arm] == sha(data.metadata_path(seed, arm))
            assert audit_completion["audits"][str(seed)][arm] == sha(data.audit_path(seed, arm))
            audit = json.loads(data.audit_path(seed, arm).read_text(encoding="utf-8"))
            assert audit["complete"] and audit["protocol_sha256"] == sha(data.PLAN)
            assert audit["stream_sha256"] == sha(data.raw_path(seed, arm))
            paths = [source_checkpoint(seed), source_checkpoint(seed).with_suffix(".json"), data.source_world_path(seed),
                *completion_paths, *map(Path, data.input_paths(seed, arm))]
            bindings[str(seed)][arm] = {str(path): sha(path) for path in dict.fromkeys(paths)}
            stream = load_stream(seed, arm)
            validate_stream(stream, WARM, BLOCK, UPDATES)
            assert stream["obs"].shape[1] == BATCH
            counts[str(seed)][arm] = summarize_rows(stream, 0, TRAIN_STEPS)
            identities[str(seed)][arm] = stream_identity(stream, 0, TRAIN_STEPS)
            del stream
    frozen_sources = list(dict.fromkeys([*sources(), *map(Path, extra_sources)]))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), snapshots=list(SNAPSHOTS), runtime=runtime(),
        source_sha256={str(path): sha(path) for path in frozen_sources}, input_sha256=bindings,
        data_protocol_path=str(data.PLAN), data_protocol_sha256=sha(data.PLAN),
        data_protocol_hash=data_plan.get("protocol_hash", sha(data.PLAN)),
        source_training_protocol_path=str(source_train.PLAN), source_training_protocol_sha256=sha(source_train.PLAN),
        source_freeze_path=str(source_train.OUT / "globalfreeze.json"), source_freeze_sha256=sha(source_train.OUT / "globalfreeze.json"),
        source_policy="seed-matched learned midpoint/action step256; core+native11+trained time head, existing Adam step256 moments, live numerical h and RNG copied exactly to all3 arms",
        initial_model=dict(width=192, input_size=56, n_jobs=2, n_actions=11, total_parameters=148632,
            core="exact learned source GRU", native_head="exact learned source11-way native head",
            time_head="exact learned source linear192->2; already trained, never reset or zeroed",
            requires_grad="all source action parameters trainable; diagnostic time features detached",
            new_parameters=0), initial_states=initial_states,
        optimizer=dict(kind="Adam", **OPTIMIZER, reset=False, source_step=256, final_step=512,
            clipping="separate policy(core+native head) norm1 and time-head norm1; one existing Adam"),
        loss=old_plan["loss"], objective="action", objective_detail=old_plan["objectives_detail"]["action"],
        implementation="frozen remaining_time.Learner.update verbatim; constructor warm=0, no warm_start; only full source state and local counters overridden",
        stream=dict(warm_ticks=0, training_ticks=TRAIN_STEPS, batch=BATCH, slots=2,
            block=BLOCK, updates=UPDATES, replay_epochs=0, new_stream_cursor_at_source=0,
            source_stream_cursor=33280, inputs="public56 only; no oracle or time prediction in control",
            recurrent_state="source h copied exactly at local000; never reset/recomputed; numerical pre-update h carried; only graph detached at128"),
        expected_training_rows=counts, full_stream_identity=identities,
        work_accounting="work is new local work only; source_work copied independently; cumulative_work is their exact sum, including preexisting warm work; optimizer step is source256+local updates, independent of new label counts",
        checkpoint="local000 before first new observation,064,256 full model+Adam+h+RNG+local/source/cumulative work; actual first2 updates before/after and preclip/clipped gradients",
        no_adaptation="exactly control/wide/withdraw x3 seeds,256 new updates; no new head, no mechanism, no sweep, replay, warm, repeated epochs or heldout-directed selection; globally freeze all9 endpoints and27 snapshots before heldout scoring")
    plan["optimizer"]["betas"] = list(OPTIMIZER["betas"])
    assert state_digest(plan) == state_digest(json.loads(json.dumps(plan)))
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(data.PLAN) == plan["data_protocol_sha256"]
    assert sha(source_train.PLAN) == plan["source_training_protocol_sha256"]
    assert sha(source_train.OUT / "globalfreeze.json") == plan["source_freeze_sha256"]
    data.checked()
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, "frozen source changed: " + path
    for by_arm in plan["input_sha256"].values():
        for paths in by_arm.values():
            for path, expected in paths.items():
                assert sha(path) == expected, "frozen input changed: " + path
    return plan


def run_arm(seed, arm, plan):
    folder = checkpoint_path(seed, arm).parent
    if folder.exists():
        raise FileExistsError(folder)
    stream = load_stream(seed, arm)
    learner = learner_from_source(seed, arm, stream, folder=folder,
        protocol_hash=plan["protocol_hash"], bindings=plan["input_sha256"][str(seed)][arm],
        source_hashes=plan["source_sha256"])
    learner.save("step000")
    while learner.step < UPDATES:
        row = learner.update(save_audit=learner.step < 2)
        assert optimizer_steps(learner.optimizer) == [256 + learner.step]
        if learner.step in SNAPSHOTS:
            learner.save(f"step{learner.step:03d}")
        if learner.step % 32 == 0:
            print(json.dumps(dict(event="progress", seed=seed, experiment_arm=arm, objective="action",
                step=learner.step, optimizer_step=256 + learner.step, loss=row["loss"], cursor=learner.cursor)), flush=True)
    expected = plan["expected_training_rows"][str(seed)][arm]
    assert all(learner.work[key] == value for key, value in expected.items())
    assert learner.cursor == TRAIN_STEPS and learner.work["warm_ticks"] == 0
    final_core = state_digest(cpu_tree(learner.model.core.state_dict()))
    final_head = state_digest(cpu_tree(learner.model.head.state_dict()))
    assert final_core != learner.initial_core_hash and final_head != learner.initial_head_hash
    path = checkpoint_path(seed, arm)
    report = dict(seed=int(seed), arm=arm, objective="action", passed=True,
        updates=learner.step, optimizer_step=256 + learner.step, work=learner.work,
        source_work=learner.source_work, cumulative_work=learner.cumulative_work(),
        initial_core_sha256=learner.initial_core_hash, final_core_sha256=final_core,
        initial_native_head_sha256=learner.initial_head_hash, final_native_head_sha256=final_head,
        final_checkpoint=str(path), final_checkpoint_sha256=sha(path), protocol_hash=plan["protocol_hash"])
    write_new(folder / "completed.json", report)
    del learner, stream
    torch.cuda.empty_cache()
    return report


def run(seed):
    assert int(seed) in SEEDS
    plan = checked()
    deterministic(820000 + seed)
    assert runtime() == plan["runtime"]
    results = [run_arm(seed, arm, plan) for arm in ARMS]
    for key in ("training_updates", "training_ticks", "training_lane_ticks", "action_rows", "action_gradient_rows"):
        assert len({row["work"][key] for row in results}) == 1
    cps = [torch.load(checkpoint_path(seed, arm, 0), map_location="cpu", weights_only=False) for arm in ARMS]
    for key in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng"):
        for cp in cps[1:]:
            assert_exact(cps[0][key], cp[key], f"initial_three_arms.{key}")
    write_new(OUT / f"seed_{seed:03d}_completed.json", dict(seed=seed, runs=results,
        all_three_start_states_exact=True, action_rows_and_new_updates_matched=True))


def freeze_endpoints():
    plan = checked()
    endpoints = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = checkpoint_path(seed, arm)
            report = json.loads((path.parent / "completed.json").read_text(encoding="utf-8"))
            assert report["passed"] and sha(path) == report["final_checkpoint_sha256"]
            snapshots = {}
            for step in SNAPSHOTS:
                snap = checkpoint_path(seed, arm, step)
                meta = snap.with_suffix(".json")
                metadata = json.loads(meta.read_text(encoding="utf-8"))
                assert metadata["sha256"] == sha(snap) and metadata["protocol_hash"] == plan["protocol_hash"]
                snapshots[f"step{step:03d}"] = dict(path=str(snap), sha256=sha(snap), metadata_path=str(meta), metadata_sha256=sha(meta))
            endpoints[f"{seed}:{arm}"] = dict(path=str(path), sha256=sha(path), snapshots=snapshots,
                work=report["work"], cumulative_work=report["cumulative_work"], seed=seed, arm=arm)
    result = dict(format="coverage-retention-all9-frozen-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], endpoints=endpoints, endpoint_count=9, snapshot_count=27,
        heldout_scoring_started=False,
        statement="All9 fixed-budget learned-state continuations and27 named snapshots frozen before heldout scoring")
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json", result)
    print(json.dumps(dict(event="globally_frozen", path=str(OUT / "globalfreeze.json"), freeze_hash=result["freeze_hash"])), flush=True)
    return result


def all_seeds():
    checked()
    workers = []
    for seed in SEEDS:
        log = (OUT / f"seed_{seed:03d}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.coverage_retention.train", "run", "--seed", str(seed)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        workers.append((seed, process, log))
    results = []
    for seed, process, log in workers:
        results.append(dict(seed=seed, exit_code=process.wait()))
        log.close()
    write_new(OUT / "exit_codes.json", results)
    if any(row["exit_code"] for row in results):
        raise RuntimeError(results)
    return freeze_endpoints()


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
