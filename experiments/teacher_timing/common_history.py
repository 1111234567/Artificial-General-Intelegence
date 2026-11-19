"""Four frozen policies on both actual teachers' same fresh long histories.

All six teacher histories are collected and hash-frozen before any test-model
replay. Native teacher actions alone cause the archived observations/receipts.
The frozen models only read those observations and cannot change the worlds.
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
from experiments.action_split.data import array_hash, read, sha, write_new
from experiments.continuity.run import state_digest
from experiments.remaining_time import common_history as previous
from experiments.remaining_time.evaluate import time_statistics
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.teacher_timing import data, evaluate, train

ROOT = train.ROOT
OUT = ROOT / "data/teacher-timing/common-v1"
PLAN = OUT / "protocol.json"
FREEZE = OUT / "histories.freeze.json"
SEEDS, TEACHERS = train.SEEDS, train.TEACHERS
POLICIES = ("earliest_action", "earliest_joint", "midpoint_action", "midpoint_joint")
WARM, STEPS, BATCH, DT = 512, 4096, 32, .25
FORMAT = "teacher-timing-crossed-common-history-v1"
replay = previous.replay


def raw_path(seed, teacher):
    assert seed in SEEDS and teacher in TEACHERS
    return OUT / f"seed_{seed:03d}" / f"teacher_{teacher}_long.npz"


def sidecar_path(seed, teacher, policy):
    assert seed in SEEDS and teacher in TEACHERS and policy in POLICIES
    return OUT / f"seed_{seed:03d}" / f"history_{teacher}__{policy}.npz"


def checkpoint(seed, policy):
    assert policy in POLICIES
    training_teacher, objective = policy.split("_")
    return train.checkpoint_path(seed, training_teacher, objective, 256)


def config(seed):
    return evaluate.config(seed, "long")


def source_paths():
    return list(dict.fromkeys([Path(__file__), Path(__file__).with_name("common_history_test.py"),
        Path(evaluate.__file__), *previous.source_paths(), *data.sources(),
        Path(train.__file__), Path(train.__file__).with_name("train_test.py")]))


def declare():
    assert not PLAN.exists()
    training = train.checked()
    globalfreeze = train.OUT / "globalfreeze.json"
    frozen = read(globalfreeze)
    assert frozen["endpoint_count"] == 12 and frozen["snapshot_count"] == 36
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), test_teachers=list(TEACHERS), policies=list(POLICIES), conditions=["long"],
        warm=WARM, test=STEPS, batch=BATCH, dt=DT,
        world_configs={str(seed): config(seed) for seed in SEEDS},
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_paths()},
        checkpoints={str(checkpoint(seed, policy).relative_to(ROOT)): sha(checkpoint(seed, policy))
            for seed in SEEDS for policy in POLICIES},
        training_protocol_hash=training["protocol_hash"], training_freeze_sha256=sha(globalfreeze),
        collection="Six fresh native ContextWorld origins: three seeds crossed with actual earliest/midpoint oracle. Main long8..16 only, seed854000+seed actors99000..99031 gap16..32 exactly autonomous main evaluation config. Actual native teacher alone acts; actual observations and receipts are archived. No tested policy acts in these worlds.",
        history_freeze="Freeze all six raw histories, request reports, initial/final native world states and paired exogenous checks before first tested-model replay.",
        pairing="Same physical exogenous event hashes, request IDs, external fields, BEGIN ticks, scheduler and context RNG across both test teachers; teacher finish times and subsequent receipt/state history differ. Compare same event identities with autonomous main eval when available.",
        inputs="Only original public56 observations; no clock, oracle action, private readiness or target enters tested policies.",
        replay="Each of four fixedstep256 endpoints reads each seed-matched actual test-teacher history once:24 total CPU float32 inference_mode replays, h zero only at origin then continuous4608 ticks. Archive normalized time_prediction, native_logits and final_h. No optimizer or action feedback.",
        metrics="Unchanged remaining_time.evaluate.time_statistics; physical time prediction*16, postwarm active&started&finite ready all_valid and phase/distance bins. This row denominator is separate from full matured eligible new-request report and consecutive last-illegal/first-legal pair subset.",
        boundary="Fixed sign threshold zero, no calibration. Report both signs, direction, early/missed first, conditional8way content at first legal and all legal. Full matured denominator retained when boundary pair unavailable.",
        interpretation="Cross training teacher with test teacher to avoid preferring one chosen common history. Diagnostics measure readout on common observed history, not autonomous tested-model completion. Teacher100% completion belongs only to teacher report. No longer-duration common-history branch or threshold/weight/model sweep.")
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


def collect_native(seed, teacher, *, total=WARM + STEPS, world_config=None):
    facts, context = data.collect_native(seed, teacher, total=total,
        world_config=config(seed) if world_config is None else world_config)
    return facts, dict(initial=context["initial_world"], final=context["final_world"]), context["events_sha256"]


def verify_teacher(facts, teacher):
    result = data.verify_facts(facts, teacher)
    base.accepted_actions(facts)
    result.update(native_actual_receipts_exact=True, public_previous_action_receipts_exact=True,
        actual_teacher=teacher, autonomous_test_model_score=False)
    return result


def collect(seed, teacher):
    plan = checked()
    path = raw_path(seed, teacher)
    if path.exists():
        raise FileExistsError(path)
    facts, worlds, external_hash = collect_native(seed, teacher)
    audit = verify_teacher(facts, teacher)
    report = base.summarize_facts(facts, seed, warm=WARM)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **facts)
    worlds_path = path.with_name(path.stem + "_worlds.pt")
    with worlds_path.open("xb") as handle:
        torch.save(worlds, handle)
    report.update(seed=seed, teacher=teacher, condition="long", execution="actual native teacher only",
        teacher_request_report_only=True, autonomous_test_model_score=False, world_config=config(seed),
        external_hash=external_hash, raw_facts_sha256=sha(path), worlds_sha256=sha(worlds_path),
        protocol_hash=plan["protocol_hash"], audit=audit, obs_actions_sha256=array_hash(facts, ("obs", "actions")))
    write_new(path.with_suffix(".json"), report)
    return report


def freeze_histories():
    plan, files = checked(), {}
    for seed in SEEDS:
        pair_facts, metadata, worlds = {}, {}, {}
        for teacher in TEACHERS:
            path = raw_path(seed, teacher)
            metadata[teacher] = read(path.with_suffix(".json"))
            assert metadata[teacher]["raw_facts_sha256"] == sha(path)
            with np.load(path, allow_pickle=False) as archive:
                pair_facts[teacher] = dict(archive)
            world_path = path.with_name(path.stem + "_worlds.pt")
            worlds[teacher] = torch.load(world_path, map_location="cpu", weights_only=False)
            for item in (path, path.with_suffix(".json"), world_path):
                files[str(item.relative_to(ROOT))] = sha(item)
        paired = data.compare_pair(pair_facts["earliest"], pair_facts["midpoint"])
        assert metadata["earliest"]["external_hash"] == metadata["midpoint"]["external_hash"]
        ids = [{(r["actor"], r["job"], r["rid"]) for r in metadata[t]["requests"]} for t in TEACHERS]
        assert ids[0] == ids[1]
        for stage in ("initial", "final"):
            assert state_digest(worlds["earliest"][stage]["scheduler"]) == state_digest(worlds["midpoint"][stage]["scheduler"])
            for field in ("rng_state", "assigned"):
                assert torch.equal(worlds["earliest"][stage]["context"][field], worlds["midpoint"][stage]["context"][field])
        paired.update(seed=seed, condition="long", same_external_hash=True, same_matured_request_ids=True,
            initial_final_scheduler_context_rng_bit_exact=True, protocol_hash=plan["protocol_hash"])
        pair_path = OUT / f"seed_{seed:03d}" / "teacher_pair.json"
        write_new(pair_path, paired)
        files[str(pair_path.relative_to(ROOT))] = sha(pair_path)
    result = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"], protocol_sha256=sha(PLAN), histories=6,
        files=files, model_replay_started=False,
        statement="All6 actual earliest/midpoint native teacher histories frozen before first tested-model replay")
    result["freeze_hash"] = state_digest(result)
    write_new(FREEZE, result)
    return result


def checked_history(seed, teacher):
    plan, frozen = checked(), read(FREEZE)
    assert frozen["protocol_hash"] == plan["protocol_hash"] and sha(PLAN) == frozen["protocol_sha256"]
    assert state_digest({key: value for key, value in frozen.items() if key != "freeze_hash"}) == frozen["freeze_hash"]
    path = raw_path(seed, teacher)
    for item in (path, path.with_suffix(".json"), path.with_name(path.stem + "_worlds.pt")):
        assert sha(item) == frozen["files"][str(item.relative_to(ROOT))]
    return path


def score(seed, teacher):
    torch.set_num_threads(1)
    path = checked_history(seed, teacher)
    with np.load(path, allow_pickle=False) as archive:
        facts = dict(archive)
    request_report = read(path.with_suffix(".json"))
    results = {}
    for policy in POLICIES:
        cp_path = checkpoint(seed, policy)
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        model = RemainingTimeGRUAgent(**cp["model_config"])
        model.load_state_dict(cp["model"])
        before = state_digest(model.state_dict())
        prediction, logits, final_h = replay(model, facts["obs"])
        assert state_digest(model.state_dict()) == before
        assert prediction.dtype == logits.dtype == np.float32
        summary = time_statistics(facts, prediction, logits, request_report, warm=WARM)
        side = sidecar_path(seed, teacher, policy)
        with side.open("xb") as handle:
            np.savez_compressed(handle, time_prediction=prediction, native_logits=logits, final_h=final_h)
        summary.update(seed=seed, policy=policy, test_teacher=teacher, condition="long",
            common_history=True, diagnostic_only=True, no_test_model_actions_executed=True,
            facts_path=str(path.relative_to(ROOT)), facts_sha256=sha(path),
            teacher_request_report_sha256=sha(path.with_suffix(".json")),
            checkpoint_path=str(cp_path.relative_to(ROOT)), checkpoint_sha256=sha(cp_path),
            raw_predictions_path=str(side.relative_to(ROOT)), raw_predictions_sha256=sha(side),
            protocol_sha256=sha(PLAN), history_freeze_sha256=sha(FREEZE),
            model_before_sha256=before, model_after_sha256=state_digest(model.state_dict()))
        write_new(side.with_suffix(".json"), summary)
        results[policy] = summary
    for key in ("facts_sha256", "mature_request_denominator", "paired_requests"):
        assert len({row[key] for row in results.values()}) == 1
    return results


def run_parallel(command, workers):
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(key):
        seed, teacher = key
        with (logs / f"{command}_{seed}_{teacher}.log").open("x", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", "-m", "experiments.teacher_timing.common_history", command,
                "--seed", str(seed), "--teacher", teacher], cwd=ROOT, stdout=handle,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return dict(seed=seed, teacher=teacher, exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(job, [(seed, teacher) for seed in SEEDS for teacher in TEACHERS]))
    write_new(OUT / f"{command}_workers.json", dict(runs=runs))
    assert all(row["exit_code"] == 0 for row in runs), runs
    return runs


def complete():
    checked()
    files, cases = {}, {}
    for seed in SEEDS:
        for teacher in TEACHERS:
            checked_history(seed, teacher)
            rows = {}
            for policy in POLICIES:
                side = sidecar_path(seed, teacher, policy)
                report = read(side.with_suffix(".json"))
                assert report["raw_predictions_sha256"] == sha(side)
                for item in (side, side.with_suffix(".json")):
                    files[str(item.relative_to(ROOT))] = sha(item)
                rows[policy] = {key: report[key] for key in ("groups", "boundary", "mature_request_denominator", "paired_requests")}
            cases[f"{seed}:{teacher}"] = rows
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
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "freeze":
        freeze_histories()
    elif args.command == "collect":
        collect(args.seed, args.teacher)
    elif args.command == "score":
        score(args.seed, args.teacher)
    else:
        checked(); run_parallel("collect", args.workers); freeze_histories()
        run_parallel("score", args.workers); complete()


if __name__ == "__main__":
    main()
