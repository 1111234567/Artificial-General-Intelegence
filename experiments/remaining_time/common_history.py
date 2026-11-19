"""Read-only native policies on identical actual earliest-teacher histories.

All six histories are archived and hash-frozen before the first policy replay.
Only the native teacher acts in these worlds.  Auxiliary predictions and native
logits are measured without modifying observations, state, or action decoding.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from experiments.action_split import evaluate as base
from experiments.action_split.data import sha, read, write_new, array_hash
from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.remaining_time import evaluate, train
from experiments.remaining_time.model import RemainingTimeGRUAgent

ROOT = base.ROOT
OUT = ROOT / "data/remaining-time/common-v1"
PLAN = OUT / "protocol.json"
FREEZE = OUT / "histories.freeze.json"
SEEDS = (0, 1, 2)
ARMS = ("probe", "time", "action", "joint")
CONDITIONS = ("long", "longer")
WARM, STEPS, BATCH, DT = 512, 4096, 32, .25
FORMAT = "remaining-time-common-native-teacher-v1"


def raw_path(seed, condition):
    assert seed in SEEDS and condition in CONDITIONS
    return OUT / f"seed_{seed:03d}" / f"teacher_{condition}.npz"


def sidecar_path(seed, arm, condition):
    assert seed in SEEDS and arm in ARMS and condition in CONDITIONS
    return OUT / f"seed_{seed:03d}" / f"{arm}_{condition}.npz"


def source_paths():
    return [Path(__file__), Path(__file__).with_name("common_history_test.py"),
        Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("model.py"),
        ROOT / "experiments/action_split/evaluate.py", ROOT / "experiments/action_split/data.py",
        ROOT / "experiments/learn_remember/gru_train.py", ROOT / "experiments/timing_boundary/data.py",
        ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
        ROOT / "experiments/continuity/run.py", ROOT / "experiments/experience_transfer/train_transfer.py"]


def declare():
    assert not PLAN.exists()
    training = train.checked()
    globalfreeze = train.OUT / "globalfreeze.json"
    assert globalfreeze.exists()
    freeze = read(globalfreeze)
    assert len(freeze["endpoints"]) == 12 and freeze["snapshot_count"] == 36
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), conditions=list(CONDITIONS),
        warm=WARM, test=STEPS, batch=BATCH, dt=DT,
        world_configs={f"{seed}:{condition}": evaluate.config(seed, condition)
            for seed in SEEDS for condition in CONDITIONS},
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_paths()},
        checkpoints={str(train.checkpoint_path(seed, arm, 256).relative_to(ROOT)):
            sha(train.checkpoint_path(seed, arm, 256)) for seed in SEEDS for arm in ARMS},
        training_protocol_hash=training["protocol_hash"], training_freeze_sha256=sha(globalfreeze),
        collection="Six independent ContextWorld origins; exact same seed853000+seed, actors98000..98031, duration conditions and gap16..32 as autonomous eval. Every tick observe, native oracle('earliest'), archive current facts, act(actual teacher labels), archive actual receipts. No tested policy acts in these worlds.",
        history_freeze="Archive all six actual histories, native receipt/request audit and initial/final world states; hash-freeze all files before first test-model replay.",
        inputs="Only original public56 observation; no clock, private readiness, targets or oracle labels enter tested models.",
        replay="Load all4 fixedstep256 models with original native GRU core+11-action head; .eval().requires_grad_(False), CPU float32 torch.inference_mode; native step once per archived observation; h=0 only at origin and continuous across all4608 steps. Record normalized time_head(h) and unchanged native_logits. Model actions never feed teacher worlds.",
        metrics="Reuse frozen evaluate.time_statistics: physical prediction=time_prediction*16, signed target ready-now; postwarm active&started&finite ready rows only; absolute/squared/signed sums and MAE/RMSE/bias for all_valid,processing,legal,remaining>4,1<remaining<=4,0<remaining<=1,abs(remaining)<=.25,overdue<-.25. Fixed predicted-ready sign<=0 with no calibration.",
        boundary="For every postwarm eligible NEW/REPLACE matured request in shared teacher report, pair consecutive last-illegal/first-legal valid rows when present; retain full matured denominator and pair counts; both signs, direction, early/missed first, 8-way conditional-content accuracy at first legal and all legal rows.",
        interpretation="Diagnostic availability on common observations, not autonomous completion or deployment. Teacher summary reports only teacher request coverage. No optimizer, adaptation, threshold fitting, step064 selection or extra model sweep.")
    # World configurations contain tuples; hash their actual JSON form so the
    # declaration survives a read/write round trip without changing types.
    plan = json.loads(json.dumps(plan))
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    return plan


def checked():
    plan = read(PLAN)
    assert state_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    assert sha(train.OUT / "globalfreeze.json") == plan["training_freeze_sha256"]
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in plan["checkpoints"].items():
        assert sha(ROOT / name) == digest, name
    return plan


@torch.inference_mode()
def collect_native(seed, condition, *, total=WARM + STEPS, world_config=None):
    """Actual native teacher collection, also usable by a short causal test."""
    torch.set_num_threads(1)
    cfg = evaluate.config(seed, condition) if world_config is None else world_config
    streams = {key: [] for key in (*base.FACT_FIELDS, "obs", "actions", "receipts", "now")}
    with mapping_scope(mapping_table(seed, "related")):
        world = ContextWorld(**cfg)
        initial = world.state_dict()
        for _ in range(total):
            obs = world.observe()
            actions = world.oracle("earliest")
            streams["obs"].append(obs.numpy().copy())
            streams["actions"].append(actions.numpy().copy())
            streams["now"].append(world.now)
            for key, attr in base.FACT_FIELDS.items():
                streams[key].append(getattr(world, attr).numpy().copy())
            world.act(actions)
            streams["receipts"].append(world.last_result.numpy().copy())
        external_hash = world.event_hash
        final = world.state_dict()
    facts = {key: np.stack(value) for key, value in streams.items()}
    facts["actor_ids"] = np.asarray(cfg["actor_ids"])
    return facts, dict(initial=initial, final=final), external_hash


def verify_teacher(facts):
    now = facts["now"][:, None, None]
    begin = facts["active"] & ~facts["started"] & (now + facts["duration"] <= facts["deadline"] + base.EPS)
    finish = facts["active"] & facts["started"] & (now >= facts["ready"] - base.EPS) & (now <= facts["deadline"] + base.EPS)
    expected = np.where(begin, 1 + facts["version"], np.where(finish,
        3 + 4 * facts["version"] + facts["payload"], 0))
    np.testing.assert_array_equal(facts["actions"], expected)
    np.testing.assert_array_equal(facts["receipts"], (expected != 0).astype(np.int64))
    np.testing.assert_array_equal(facts["now"], np.arange(len(expected)) * DT)
    obs = facts["obs"].reshape(len(expected), expected.shape[1], 2, 28)
    np.testing.assert_array_equal(obs[1:, ..., 11:22],
        np.eye(11, dtype=np.float32)[expected[:-1]] * (expected[:-1] != 0)[..., None])
    np.testing.assert_array_equal(obs[1:, ..., 22], facts["receipts"][:-1] == 1)
    np.testing.assert_array_equal(obs[1:, ..., 23], facts["receipts"][:-1] == -1)
    assert not obs[0, ..., 11:24].any()
    base.accepted_actions(facts)
    return dict(native_earliest_actions_exact=True, native_actual_receipts_exact=True,
        public_previous_action_receipts_exact=True, continuous_native_clock_exact=True,
        ticks=len(expected), opportunities=int(expected.size))


def collect(seed, condition):
    checked()
    path = raw_path(seed, condition)
    if path.exists():
        raise FileExistsError(path)
    facts, worlds, external_hash = collect_native(seed, condition)
    audit = verify_teacher(facts)
    report = base.summarize_facts(facts, seed, warm=WARM)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **facts)
    worlds_path = path.with_name(path.stem + "_worlds.pt")
    torch.save(worlds, worlds_path)
    report.update(seed=seed, condition=condition, execution="actual native earliest teacher only",
        teacher_request_report_only=True, autonomous_test_model_score=False,
        world_config=evaluate.config(seed, condition), external_hash=external_hash,
        raw_facts_sha256=sha(path), worlds_sha256=sha(worlds_path),
        protocol_hash=checked()["protocol_hash"], audit=audit,
        obs_actions_sha256=array_hash(facts, ("obs", "actions")))
    write_new(path.with_suffix(".json"), report)
    return report


def freeze_histories():
    plan = checked()
    files = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            path = raw_path(seed, condition)
            metadata = read(path.with_suffix(".json"))
            assert metadata["raw_facts_sha256"] == sha(path)
            for item in (path, path.with_suffix(".json"), path.with_name(path.stem + "_worlds.pt")):
                files[str(item.relative_to(ROOT))] = sha(item)
    result = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], protocol_sha256=sha(PLAN), histories=6,
        files=files, model_replay_started=False,
        statement="All6 actual native earliest-teacher histories frozen before first model replay")
    result["freeze_hash"] = state_digest(result)
    write_new(FREEZE, result)
    return result


def checked_history(seed, condition):
    plan, freeze = checked(), read(FREEZE)
    assert freeze["protocol_hash"] == plan["protocol_hash"] and sha(PLAN) == freeze["protocol_sha256"]
    assert state_digest({key: value for key, value in freeze.items() if key != "freeze_hash"}) == freeze["freeze_hash"]
    path = raw_path(seed, condition)
    for item in (path, path.with_suffix(".json")):
        assert sha(item) == freeze["files"][str(item.relative_to(ROOT))]
    return path


@torch.inference_mode()
def replay(model, obs):
    """Native per-tick inference; no request reset, world, labels or clock."""
    model.eval().requires_grad_(False)
    hidden = model.initial_state(obs.shape[1])
    predictions, logits = [], []
    for row in obs:
        native, hidden = model.step(torch.from_numpy(row), hidden, DT)
        predictions.append(model.time_head(hidden).numpy().copy())
        logits.append(native.numpy().copy())
    return np.stack(predictions), np.stack(logits), hidden.numpy().copy()


def score(seed, condition):
    torch.set_num_threads(1)
    path = checked_history(seed, condition)
    with np.load(path, allow_pickle=False) as archive:
        facts = dict(archive)
    request_report = read(path.with_suffix(".json"))
    results = {}
    for arm in ARMS:
        cp_path = train.checkpoint_path(seed, arm, 256)
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        model = RemainingTimeGRUAgent(**cp["model_config"])
        model.load_state_dict(cp["model"])
        before = state_digest(model.state_dict())
        prediction, logits, final_h = replay(model, facts["obs"])
        assert state_digest(model.state_dict()) == before
        assert prediction.dtype == logits.dtype == np.float32
        summary = evaluate.time_statistics(facts, prediction, logits, request_report, warm=WARM)
        side = sidecar_path(seed, arm, condition)
        with side.open("xb") as handle:
            np.savez_compressed(handle, time_prediction=prediction, native_logits=logits, final_h=final_h)
        summary.update(seed=seed, arm=arm, condition=condition,
            common_history=True, diagnostic_only=True, no_test_model_actions_executed=True,
            facts_path=str(path.relative_to(ROOT)), facts_sha256=sha(path),
            teacher_request_report_sha256=sha(path.with_suffix(".json")),
            checkpoint_path=str(cp_path.relative_to(ROOT)), checkpoint_sha256=sha(cp_path),
            raw_predictions_path=str(side.relative_to(ROOT)), raw_predictions_sha256=sha(side),
            protocol_sha256=sha(PLAN), history_freeze_sha256=sha(FREEZE),
            model_before_sha256=before, model_after_sha256=state_digest(model.state_dict()))
        write_new(side.with_suffix(".json"), summary)
        results[arm] = summary
    assert len({row["facts_sha256"] for row in results.values()}) == 1
    assert len({row["mature_request_denominator"] for row in results.values()}) == 1
    assert len({row["paired_requests"] for row in results.values()}) == 1
    return results


def run_parallel(command, workers):
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(key):
        seed, condition = key
        with (logs / f"{command}_{seed}_{condition}.log").open("x", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", "-m", "experiments.remaining_time.common_history", command,
                "--seed", str(seed), "--condition", condition], cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return dict(seed=seed, condition=condition, exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(job, [(seed, condition) for seed in SEEDS for condition in CONDITIONS]))
    write_new(OUT / f"{command}_workers.json", dict(runs=runs))
    assert all(row["exit_code"] == 0 for row in runs), runs
    return runs


def complete():
    checked()
    files, cases = {}, {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            checked_history(seed, condition)
            rows = {}
            for arm in ARMS:
                side = sidecar_path(seed, arm, condition)
                report = read(side.with_suffix(".json"))
                assert report["raw_predictions_sha256"] == sha(side)
                for item in (side, side.with_suffix(".json")):
                    files[str(item.relative_to(ROOT))] = sha(item)
                rows[arm] = {key: report[key] for key in ("groups", "boundary", "mature_request_denominator", "paired_requests")}
            cases[f"{seed}:{condition}"] = rows
    result = dict(complete=True, format=FORMAT, histories=6, model_replays=24,
        protocol_sha256=sha(PLAN), history_freeze_sha256=sha(FREEZE),
        common_raw_history_per_case=True, all_models_unchanged=True,
        no_test_model_actions_executed=True, diagnostic_only=True, files=files, cases=cases)
    write_new(OUT / "completion.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "collect", "freeze", "score", "all"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "freeze":
        freeze_histories()
    elif args.command == "collect":
        collect(args.seed, args.condition)
    elif args.command == "score":
        score(args.seed, args.condition)
    else:
        checked(); run_parallel("collect", args.workers); freeze_histories()
        run_parallel("score", args.workers); complete()


if __name__ == "__main__":
    main()
