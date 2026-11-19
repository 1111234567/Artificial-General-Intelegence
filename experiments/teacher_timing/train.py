"""Two actual native teachers, two fixed objectives, three paired seeds.

Every numerical training operation is the frozen remaining_time Learner.  The
only intervention is which teacher-generated physical stream it consumes.
Earliest is rerun and must reproduce the prior controls exactly at all three
named checkpoints before the twelve endpoints can be globally frozen.
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
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.action_split.train import deterministic, runtime, sha, write_new
from experiments.continuity.run import cpu_tree, state_digest
from experiments.remaining_time import train as previous
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.teacher_timing import data

OUT = ROOT / "data/teacher-timing/train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "teacher-timing-native-training-v1"
TEACHERS = ("earliest", "midpoint")
OBJECTIVES = ARMS = ("action", "joint")
SEEDS = previous.SEEDS
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = (
    previous.WARM, previous.TRAIN_STEPS, previous.BLOCK, previous.BATCH, previous.UPDATES)
SNAPSHOTS = previous.SNAPSHOTS
HEAD_FORMAT = previous.HEAD_FORMAT
STREAM_FIELDS = previous.STREAM_FIELDS
OPTIMIZER = previous.OPTIMIZER

# Reusing this exact class, rather than copying its loop, preserves the prior
# Adam ordering, loss arithmetic, clipping, TBPTT and numerical-state transfer.
Learner = previous.Learner
initialize = previous.initialize
original_checkpoint = previous.original_checkpoint
validate_stream = previous.validate_stream
summarize_rows = previous.summarize_rows
stream_identity = previous.stream_identity


def checkpoint_path(seed, teacher, objective, step=256):
    assert int(seed) in SEEDS and teacher in TEACHERS and objective in OBJECTIVES
    assert int(step) in SNAPSHOTS
    return OUT / teacher / f"seed_{int(seed):03d}" / objective / f"step{int(step):03d}.pt"


def load_stream(seed, teacher):
    assert int(seed) in SEEDS and teacher in TEACHERS
    return data.load_stream(seed, teacher)


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), Path(__file__).with_name("train_test.py"),
        Path(__file__).with_name("train_audit.py"),
        ROOT / "experiments/remaining_time/train_audit.py", *previous.sources(),
        *data.sources(), ROOT / "experiments/lifelong/world.py"]))


def data_inputs(seed, teacher):
    """Pin every teacher artifact, including targets and actual-world ledgers."""
    if hasattr(data, "input_paths"):
        return list(map(Path, data.input_paths(seed, teacher)))
    metadata = data.metadata_path(seed, teacher)
    return [data.raw_path(seed, teacher), data.states_path(seed, teacher), metadata,
        data.audit_path(seed, teacher), metadata.parent / "finish_latency.npz",
        metadata.parent.parent / "pair_comparison.json"]


def assert_exact(left, right, name="value"):
    """Compare contents, never torch.save container bytes or floating tolerance."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape, name
        assert torch.equal(left.detach().cpu(), right.detach().cpu()), name
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), name
        for key in left:
            assert_exact(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), name
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{name}[{index}]")
    else:
        assert left == right, name


REGRESSION_FIELDS = ("model", "optimizer", "hidden", "stream_cursor", "step", "work")


def compare_historical(seed, objective, step):
    fresh_path = checkpoint_path(seed, "earliest", objective, step)
    old_path = previous.checkpoint_path(seed, objective, step)
    fresh = torch.load(fresh_path, map_location="cpu", weights_only=False)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    for key in REGRESSION_FIELDS:
        assert_exact(fresh[key], old[key], f"seed{seed}.{objective}.step{step}.{key}")
    return dict(seed=int(seed), objective=objective, step=int(step), passed=True,
        new_path=str(fresh_path), new_sha256=sha(fresh_path),
        reference_path=str(old_path), reference_sha256=sha(old_path),
        fields={key: state_digest(fresh[key]) for key in REGRESSION_FIELDS})


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    old_plan, data_plan = previous.checked(), data.checked()
    assert runtime() == old_plan["runtime"], "historical CUDA runtime drift"
    inputs, counts, identities = {}, {}, {}
    for teacher in TEACHERS:
        inputs[teacher], counts[teacher], identities[teacher] = {}, {}, {}
        for seed in SEEDS:
            # Retain all the prior raw/target/model bindings, even for midpoint.
            paths = list(dict.fromkeys([*map(Path, old_plan["input_sha256"][str(seed)]),
                *data_inputs(seed, teacher),
                *(path for objective in OBJECTIVES for step in SNAPSHOTS
                    for path in (previous.checkpoint_path(seed, objective, step),
                        previous.checkpoint_path(seed, objective, step).with_suffix(".json")))]))
            inputs[teacher][str(seed)] = {str(path): sha(path) for path in paths}
            stream = load_stream(seed, teacher)
            validate_stream(stream, WARM, BLOCK, UPDATES)
            assert stream["obs"].shape[1] == BATCH
            counts[teacher][str(seed)] = summarize_rows(stream, WARM, WARM + TRAIN_STEPS)
            identities[teacher][str(seed)] = stream_identity(stream, 0, WARM + TRAIN_STEPS)
            if teacher == "earliest":
                historical = previous.load_stream(seed)
                assert identities[teacher][str(seed)] == stream_identity(historical, 0, WARM + TRAIN_STEPS)
                del historical
            del stream
    action_rows = TRAIN_STEPS * BATCH * 2
    assert all(row["action_rows"] == action_rows for group in counts.values() for row in group.values())
    frozen_sources = list(dict.fromkeys([*sources(), *map(Path, extra_sources)]))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        teachers=list(TEACHERS), objectives=list(OBJECTIVES), seeds=list(SEEDS), snapshots=list(SNAPSHOTS),
        runtime=runtime(), source_sha256={str(path): sha(path) for path in frozen_sources},
        input_sha256=inputs, data_protocol_path=str(data.PLAN), data_protocol_sha256=sha(data.PLAN),
        data_protocol_hash=data_plan.get("protocol_hash"),
        previous_training_protocol_path=str(previous.PLAN), previous_training_protocol_sha256=sha(previous.PLAN),
        previous_training_protocol_hash=old_plan["protocol_hash"],
        previous_freeze_path=str(previous.OUT / "globalfreeze.json"),
        previous_freeze_sha256=sha(previous.OUT / "globalfreeze.json"),
        source_policy=old_plan["source_policy"], initial_model=old_plan["initial_model"],
        optimizer=old_plan["optimizer"], loss=old_plan["loss"],
        objectives_detail={key: old_plan["arms_detail"][key] for key in OBJECTIVES},
        first_update=old_plan["first_update"],
        teacher_intervention="earliest executes native earliest oracle; midpoint executes native fixed midpoint oracle with actual accepted actions and receipts; no relabel-only stream",
        teacher_history="same physical external stream per seed; observation, receipt and request lifetime can differ because teacher actions differ",
        primary_comparison="midpoint versus earliest with action-only core update; detached auxiliary h and separate clipping prevent auxiliary target-gradient confounding",
        secondary_comparison="joint objective measures interaction with CE+MSE core updates",
        stream=dict(warm_ticks=WARM, training_ticks=TRAIN_STEPS, batch=BATCH, slots=2,
            block=BLOCK, updates=UPDATES, replay_epochs=0, same_raw_obs_actions_within_teacher=True,
            inputs="original public56 observation only; time prediction never inputs, gates or controls action",
            recurrent_state=old_plan["stream"]["recurrent_state"]),
        matched_budget="same action rows and updates across all twelve; class composition and valid auxiliary rows can differ across actual teachers and are reported rather than equalized",
        expected_training_rows=counts, full_stream_identity=identities,
        checkpoint="every step000/064/256: full model, Adam, live h, cursor, work, RNG, protocol/source/input hashes; first two actual updates before/after plus preclip/clipped gradients for all12",
        historical_regression=dict(teacher="earliest", objectives=list(OBJECTIVES), snapshots=list(SNAPSHOTS),
            reference=str(previous.OUT), fields=list(REGRESSION_FIELDS), comparison="recursive torch.equal with dtype/shape and exact scalar equality; metadata may differ"),
        no_adaptation="Exactly two actual teachers x two objectives x three seeds, fixed256 updates; no LR or time-weight sweep, no teacher sweep, repeated epochs, best selection or heldout-directed change; freeze all12 endpoints and36 named snapshots before heldout scoring")
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(data.PLAN) == plan["data_protocol_sha256"]
    assert sha(previous.PLAN) == plan["previous_training_protocol_sha256"]
    assert sha(previous.OUT / "globalfreeze.json") == plan["previous_freeze_sha256"]
    previous.checked(); data.checked()
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, "frozen source changed: " + path
    for teacher in plan["input_sha256"].values():
        for items in teacher.values():
            for path, expected in items.items():
                assert sha(path) == expected, "frozen input changed: " + path
    return plan


def run_objective(seed, teacher, objective, plan):
    deterministic(820000 + seed)  # Identical original initialization RNG, all runs.
    folder = checkpoint_path(seed, teacher, objective).parent
    if folder.exists():
        raise FileExistsError(folder)
    stream = load_stream(seed, teacher)
    learner = Learner(initialize(seed, objective), stream, objective, seed=seed, folder=folder,
        protocol_hash=plan["protocol_hash"], bindings=plan["input_sha256"][teacher][str(seed)],
        source_hashes=plan["source_sha256"])
    learner.warm_start(); learner.save("step000")
    regression = [compare_historical(seed, objective, 0)] if teacher == "earliest" else []
    while learner.step < UPDATES:
        row = learner.update(save_audit=learner.step < 2)
        if learner.step in SNAPSHOTS:
            learner.save(f"step{learner.step:03d}")
            if teacher == "earliest":
                regression.append(compare_historical(seed, objective, learner.step))
        if learner.step % 32 == 0:
            print(json.dumps(dict(event="progress", seed=seed, teacher=teacher, objective=objective,
                step=learner.step, loss=row["loss"], cursor=learner.cursor)), flush=True)
    expected = plan["expected_training_rows"][teacher][str(seed)]
    assert all(learner.work[key] == value for key, value in expected.items())
    assert learner.cursor == WARM + TRAIN_STEPS
    final_core = state_digest(cpu_tree(learner.model.core.state_dict()))
    final_head = state_digest(cpu_tree(learner.model.head.state_dict()))
    assert final_core != learner.initial_core_hash and final_head != learner.initial_head_hash
    path = checkpoint_path(seed, teacher, objective)
    report = dict(seed=int(seed), teacher=teacher, objective=objective, arm=objective, passed=True,
        updates=learner.step, work=learner.work, initial_core_sha256=learner.initial_core_hash,
        final_core_sha256=final_core, initial_native_head_sha256=learner.initial_head_hash,
        final_native_head_sha256=final_head, final_checkpoint=str(path), final_checkpoint_sha256=sha(path),
        historical_regression=regression, protocol_hash=plan["protocol_hash"])
    write_new(folder / "completed.json", report)
    del learner, stream
    torch.cuda.empty_cache()
    return report


def run(seed):
    assert int(seed) in SEEDS
    plan = checked()
    deterministic(820000 + seed)
    assert runtime() == plan["runtime"], "worker runtime drift"
    results = [run_objective(seed, teacher, objective, plan)
        for teacher in TEACHERS for objective in OBJECTIVES]
    for teacher in TEACHERS:
        paired = [row for row in results if row["teacher"] == teacher]
        assert_exact(paired[0]["work"], paired[1]["work"], f"{teacher}.matched_work")
    shared = ("warm_ticks", "warm_lane_ticks", "warm_loss_rows", "training_updates",
        "training_ticks", "training_lane_ticks", "action_rows", "action_gradient_rows")
    assert all(all(row["work"][key] == results[0]["work"][key] for key in shared) for row in results)
    write_new(OUT / f"seed_{seed:03d}_completed.json", dict(seed=seed, runs=results,
        within_teacher_stream_and_available_rows_matched=True, all_action_rows_and_updates_matched=True,
        auxiliary_rows_across_teachers_intentionally_not_equalized=True))


def freeze_endpoints():
    plan = checked()
    endpoints, regression = {}, []
    for teacher in TEACHERS:
        for seed in SEEDS:
            for objective in OBJECTIVES:
                path = checkpoint_path(seed, teacher, objective)
                report = json.loads((path.parent / "completed.json").read_text(encoding="utf-8"))
                assert report["passed"] and sha(path) == report["final_checkpoint_sha256"]
                snapshots = {}
                for step in SNAPSHOTS:
                    snap = checkpoint_path(seed, teacher, objective, step)
                    meta = snap.with_suffix(".json")
                    snap_meta = json.loads(meta.read_text(encoding="utf-8"))
                    assert snap_meta["sha256"] == sha(snap)
                    assert snap_meta["protocol_hash"] == plan["protocol_hash"]
                    snapshots[f"step{step:03d}"] = dict(path=str(snap), sha256=sha(snap),
                        metadata_path=str(meta), metadata_sha256=sha(meta))
                    if teacher == "earliest":
                        regression.append(compare_historical(seed, objective, step))
                endpoints[f"{teacher}:{seed}:{objective}"] = dict(path=str(path), sha256=sha(path),
                    snapshots=snapshots, work=report["work"], teacher=teacher, seed=seed, objective=objective)
    regression_report = dict(format="teacher-timing-earliest-exact-regression-v1", passed=True,
        checkpoint_count=len(regression), fields=list(REGRESSION_FIELDS), comparisons=regression,
        protocol_hash=plan["protocol_hash"])
    write_new(OUT / "earliest_regression.json", regression_report)
    result = dict(format="teacher-timing-all12-frozen-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], endpoints=endpoints, endpoint_count=12, snapshot_count=36,
        earliest_regression_path=str(OUT / "earliest_regression.json"),
        earliest_regression_sha256=sha(OUT / "earliest_regression.json"),
        heldout_scoring_started=False,
        statement="All12 fixed-budget endpoints and36 named snapshots frozen after18 exact earliest regressions and before heldout scoring")
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json", result)
    print(json.dumps(dict(event="globally_frozen", path=str(OUT / "globalfreeze.json"),
        freeze_hash=result["freeze_hash"])), flush=True)
    return result


def all_seeds():
    checked()
    workers = []
    for seed in SEEDS:
        log = (OUT / f"seed_{seed:03d}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.teacher_timing.train", "run", "--seed", str(seed)],
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
