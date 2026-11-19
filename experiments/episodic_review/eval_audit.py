"""Independent native/plain-GRU audits of all fresh and continuous returns.

The evaluator, native-policy model, rollout routine, time-statistics producer,
and rule classifier are never called. Frozen independent state-machine and
decoding arithmetic are reused without changing their original sources.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import json

import numpy as np
import torch
from torch import nn

from experiments.action_split import eval_audit as native
from experiments.continuity.run import state_digest
from experiments.coverage_retention.eval_audit import score_live, EXTERNAL_FIELDS
from experiments.coverage_retention.error_signature_audit import independent_records, independent_groups
from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.remaining_time.eval_audit import exact, independent_time_statistics, write_new
from experiments.timing_boundary.data import FACT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
EVALS = {"fresh": ROOT / "data/episodic-review/eval-v1", "live": ROOT / "data/episodic-review/live-v1"}
OUT = ROOT / "data/episodic-review/eval-audit-v1"
PLAN = OUT / "audit_plan.json"
FRESH_POLICIES = ("baseline", "none256", "replay_A256", "replay_B256", "none320", "replay_A320", "replay_B320")
LIVE_POLICIES = FRESH_POLICIES[1:]
CONDITIONS = {"long": [8., 16.], "longer": [16., 32.]}
RULE_SUFFIX = "_rule"
sha, read, digest = native.sha, native.read, native.digest


def paths_for(run):
    stem = ROOT / run["stem"]
    return stem, stem.with_name(stem.name+"_time"), stem.with_name(stem.name+RULE_SUFFIX)


def stage(policy):
    return None if policy == "baseline" else int(policy[-3:])


def expected_checkpoint(seed, policy):
    if policy == "baseline":
        return ROOT / f"data/teacher-timing/train-v1/midpoint/seed_{seed:03d}/action/step256.pt"
    selected = stage(policy)
    root = "train-v1" if selected == 256 else "retention-train-v1"
    step = 256 if selected == 256 else 64
    return ROOT / f"data/episodic-review/{root}/seed_{seed:03d}/{policy[:-3]}/step{step:03d}.pt"


def manifest_files(completion):
    for field in ("files", "mainfiles", "time_files", "rule_files"):
        for name, value in completion.get(field, {}).items():
            yield name, value


def declare():
    inputs, pinned, runs = {}, {}, []
    for mode, directory in EVALS.items():
        protocol, frozen, completion = [read(directory/name) for name in ("protocol.json", "freeze.json", "completion.json")]
        policies = list(FRESH_POLICIES if mode == "fresh" else LIVE_POLICIES)
        expected = 42 if mode == "fresh" else 18
        assert protocol["seeds"] == [0, 1, 2] and protocol["policies"] == policies
        assert protocol["warm"] == 512 and protocol["test"] == 4096 and protocol["dt"] == .25
        assert protocol["conditions"] == (CONDITIONS if mode == "fresh" else {"long": [8., 16.]})
        assert completion["complete"] and completion["runs"] == completion["sidecars"] == expected
        assert completion["rules"] == expected
        assert frozen["protocol_sha256"] == sha(directory/"protocol.json")
        for name in ("protocol.json", "freeze.json", "completion.json"):
            path = directory/name
            inputs[str(path.relative_to(ROOT))] = sha(path)
        frozen_paths = frozen.get("files", frozen.get("checkpoints", {}))
        assert frozen_paths
        for name, expected_hash in (*frozen_paths.items(), *manifest_files(completion)):
            assert sha(ROOT/name) == expected_hash, name
            inputs[name] = expected_hash
        for name, expected_hash in protocol["sources"].items():
            assert sha(ROOT/name) == expected_hash, name
            pinned[name] = expected_hash
        for seed in protocol["seeds"]:
            for policy in policies:
                for condition in protocol["conditions"]:
                    run = dict(mode=mode, seed=seed, policy=policy, condition=condition,
                        study_step=stage(policy), stem=str((directory/f"seed_{seed:03d}"/f"{policy}_{condition}").relative_to(ROOT)))
                    stem, side, rule = paths_for(run)
                    report = read(stem.with_suffix(".json"))
                    assert report["facts_sha256"] == sha(stem.with_suffix(".npz"))
                    assert report["protocol_sha256"] == sha(directory/"protocol.json")
                    assert report["freeze_sha256"] == sha(directory/"freeze.json")
                    checkpoint = ROOT/report["checkpoint_path"]
                    assert checkpoint == expected_checkpoint(seed, policy)
                    assert report["checkpoint_sha256"] == sha(checkpoint)
                    assert frozen_paths[report["checkpoint_path"]] == report["checkpoint_sha256"]
                    saved_time, saved_rule = read(side.with_suffix(".json")), read(rule.with_suffix(".json"))
                    assert saved_time["facts_sha256"] == saved_rule["facts_sha256"] == report["facts_sha256"]
                    assert saved_time["sidecar_sha256"] == sha(side.with_suffix(".npz"))
                    assert saved_rule["posthoc"] is False
                    assert saved_rule["report_sha256"] == sha(stem.with_suffix(".json"))
                    assert saved_rule["seed"] == seed and saved_rule["policy"] == policy
                    assert saved_rule["mode"] == mode and saved_rule["condition"] == condition
                    assert saved_rule["protocol_sha256"] == sha(directory/"protocol.json")
                    parent = ROOT/"data/coverage-retention/error-signature-v1"
                    for key, path in (("protocol_sha256", parent/"protocol.json"),
                        ("parser_amendment_sha256", parent/"parser_amendment_001.json")):
                        assert saved_rule["parent_implementation_provenance"][key] == sha(path)
                        inputs[str(path.relative_to(ROOT))] = sha(path)
                    for path in (stem.with_suffix(".json"), stem.with_suffix(".npz"), side.with_suffix(".json"),
                                 side.with_suffix(".npz"), rule.with_suffix(".json"), checkpoint):
                        inputs[str(path.relative_to(ROOT))] = sha(path)
                    if mode == "live":
                        for field in ("initial_state", "source_native_world"):
                            path = ROOT/report[field+"_path"]
                            assert sha(path) == report[field+"_sha256"]
                            inputs[str(path.relative_to(ROOT))] = sha(path)
                    runs.append(run)
    assert len(runs) == 60
    sources = (Path(__file__), ROOT/"experiments/action_split/eval_audit.py",
        ROOT/"experiments/coverage_retention/eval_audit.py", ROOT/"experiments/coverage_retention/error_signature_audit.py",
        ROOT/"experiments/remaining_time/eval_audit.py", ROOT/"experiments/experience_transfer/train_transfer.py",
        ROOT/"experiments/timing_boundary/data.py", ROOT/"experiments/lifelong/world.py", ROOT/"experiments/continuity/world.py")
    pinned.update({str(path.relative_to(ROOT)): sha(path) for path in sources})
    write_new(PLAN, dict(format="episodic-review-all60-independent-evaluation-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), runs=runs, inputs=inputs, sources=pinned,
        warm=512, test=4096, expected_runs=60, expected_fresh=42, expected_live=18,
        rule_classification_preplanned=True,
        scope="All60 native pre-action histories, request-state metrics and complete denominators; all60 time/logit sidecars; all60 independently decoded preplanned rule signatures; native world/plainGRU/action/time/finalh bitwise replay. Fresh cases paired across7 policies; continuous returns paired only within primary256 or suffix320 stage, never across stages."))


def checked(all_inputs=True):
    plan = read(PLAN)
    manifests = {str((directory/name).relative_to(ROOT)) for directory in EVALS.values()
                 for name in ("protocol.json", "freeze.json", "completion.json")}
    for table in ("inputs", "sources"):
        for name, expected in plan[table].items():
            if table == "inputs" and not all_inputs and name not in manifests:
                continue
            assert sha(ROOT/name) == expected, name
    return plan


def metrics():
    plan = checked()
    behavior, auxiliary, classified = [], [], []
    for run in plan["runs"]:
        stem, side, rule = paths_for(run)
        report, timing, signature = (read(path.with_suffix(".json")) for path in (stem, side, rule))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
        inherited = None
        if run["mode"] == "live":
            states = torch.load(ROOT/report["initial_state_path"], map_location="cpu", weights_only=False)
            actual, inherited = score_live(facts, run["seed"], plan["warm"], states["initial_world"])
        else:
            actual = native.score(facts, run["seed"], plan["warm"])
        for key in ("coverage", "groups", "requests", "idle_opportunities", "idle_actions"):
            exact(actual[key], report[key], f"{run['stem']}/{key}")
        ident = {key: run[key] for key in ("mode", "seed", "policy", "condition", "study_step")}
        behavior.append(dict(**ident, passed=True, coverage=actual["coverage"], groups=actual["groups"],
            request_count=len(actual["requests"]), requests_digest=digest(actual["requests"]),
            idle_opportunities=actual["idle_opportunities"], idle_actions=actual["idle_actions"],
            inherited_request_audit=inherited, facts_sha256=sha(stem.with_suffix(".npz"))))
        with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
            prediction, logits = archive["time_prediction"], archive["native_logits"]
        np.testing.assert_array_equal(logits.argmax(-1), facts["actions"])
        for key in ("facts_sha256", "checkpoint_sha256", "protocol_sha256", "freeze_sha256"):
            assert timing[key] == report[key]
        result = independent_time_statistics(facts, prediction, logits, actual["requests"], plan["warm"])
        for key, value in result.items():
            exact(value, timing[key], f"{run['stem']}/time/{key}")
        auxiliary.append(dict(**ident, passed=True, groups=result["groups"], boundary=result["boundary"],
            mature_request_denominator=result["mature_request_denominator"], paired_requests=result["paired_requests"],
            boundary_rows_digest=digest(result["boundary_rows"])))
        records = independent_records(facts, run["seed"], plan["warm"])
        groups = independent_groups(records)
        exact(records, signature["requests"], f"{run['stem']}/independent rule records")
        exact(groups, signature["groups"], f"{run['stem']}/independent rule groups")
        assert signature["posthoc"] is False
        assert signature["checkpoint_sha256"] == report["checkpoint_sha256"]
        keys = ("actor", "lane", "job", "rid", "issued", "mature_tick")
        assert [tuple(row[key] for key in keys) for row in records] == [tuple(row[key] for key in keys) for row in actual["requests"]]
        for row, original in zip(records, actual["requests"]):
            for key in ("family", "raw", "conflict", "ordinary", "strict", "no_begin", "started_no_window", "legal_no_finish", "any_early", "first_error_early"):
                exact(row[key], original[key], f"{run['stem']}/rule vs request/{key}")
        classified.append(dict(**ident, passed=True, requests=len(records), records_digest=digest(records), groups=groups,
            rule_sha256=sha(rule.with_suffix(".json"))))
        print(json.dumps(dict(**ident, independent_metrics_time_rules_passed=True)), flush=True)
    pairing = paired_cases(plan)
    write_new(OUT/"metrics.json", dict(format="episodic-review-all60-independent-metrics-v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__), plan_sha256=sha(PLAN),
        runs=behavior, auxiliary=auxiliary, rules=classified, paired_external_cases=pairing,
        rule_classification_preplanned=True,
        interpretation="Every mature test-issued eligible request remains in the outcome and rule-signature denominators, including no-BEGIN/no-window/no-FINISH. Time-row and adjacent-pair subsets are diagnostics only. Continuous returns are paired within stage only."))


def paired_cases(plan):
    result = []
    for mode in ("fresh", "live"):
        for seed in (0, 1, 2):
            for condition in (CONDITIONS if mode == "fresh" else ("long",)):
                for selected_stage in ((None,) if mode == "fresh" else (256, 320)):
                    runs = [run for run in plan["runs"] if run["mode"] == mode and run["seed"] == seed
                        and run["condition"] == condition and (mode == "fresh" or run["study_step"] == selected_stage)]
                    assert len(runs) == (7 if mode == "fresh" else 3)
                    anchor = ROOT/runs[0]["stem"]
                    report = read(anchor.with_suffix(".json"))
                    identities = [(row["actor"], row["job"], row["rid"]) for row in report["requests"]]
                    with np.load(anchor.with_suffix(".npz"), allow_pickle=False) as original:
                        for run in runs[1:]:
                            stem = ROOT/run["stem"]
                            other_report = read(stem.with_suffix(".json"))
                            assert [(row["actor"], row["job"], row["rid"]) for row in other_report["requests"]] == identities
                            if mode == "fresh":
                                assert other_report["external_hash"] == report["external_hash"]
                            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as other:
                                for key in EXTERNAL_FIELDS:
                                    take = slice(plan["warm"], None) if mode == "live" and key != "actor_ids" else slice(None)
                                    np.testing.assert_array_equal(original[key][take], other[key][take],
                                        err_msg=f"paired {mode}/{seed}/{condition}/{selected_stage}/{key}")
                    result.append(dict(mode=mode, seed=seed, condition=condition, study_step=selected_stage,
                        passed=True, policies=len(runs), requests=len(identities), request_ids_digest=digest(identities)))
    return result


@torch.inference_mode()
def replay(index):
    torch.set_num_threads(1)
    plan = checked(all_inputs=False)
    run = plan["runs"][int(index)]
    stem, side, rule = paths_for(run)
    report = read(stem.with_suffix(".json"))
    paths = [stem.with_suffix(ext) for ext in (".json", ".npz")]
    paths.extend(side.with_suffix(ext) for ext in (".json", ".npz"))
    paths.extend((rule.with_suffix(".json"), ROOT/report["checkpoint_path"]))
    if run["mode"] == "live":
        paths.extend(ROOT/report[key] for key in ("initial_state_path", "source_native_world_path"))
    for path in paths:
        assert sha(path) == plan["inputs"][str(path.relative_to(ROOT))], path
    cp = torch.load(ROOT/report["checkpoint_path"], map_location="cpu", weights_only=False)
    assert cp["model_config"] == dict(width=192, input_size=56, n_jobs=2, n_actions=11)
    core, head, time_head = nn.GRU(56, 192), nn.Linear(192, 22), nn.Linear(192, 2)
    for prefix, module in (("core.", core), ("head.", head), ("time_head.", time_head)):
        module.load_state_dict({key.removeprefix(prefix): value for key, value in cp["model"].items() if key.startswith(prefix)})
        module.eval().requires_grad_(False)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        facts = {key: archive[key] for key in archive.files}
    with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
        prediction, logits = archive["time_prediction"], archive["native_logits"]
    total, batch = 4608, 32
    assert facts["obs"].shape == (total, batch, 56) and facts["obs"].dtype == np.float32
    assert facts["actions"].shape == (total, batch, 2) and facts["actions"].dtype == np.int64
    assert prediction.shape == (total, batch, 2) and logits.shape == (total, batch, 2, 11)
    assert prediction.dtype == logits.dtype == np.float32
    assert np.isfinite(prediction).all() and np.isfinite(logits).all()
    hashes = {name: hashlib.sha256() for name in FACT_FIELDS}
    expected = {name: hashlib.sha256(np.ascontiguousarray(facts[name])).hexdigest() for name in FACT_FIELDS}
    config, states = report["world_config"], None
    h = torch.zeros(1, batch, 192)
    if run["mode"] == "live":
        states = torch.load(ROOT/report["initial_state_path"], map_location="cpu", weights_only=False)
        assert set(states) == {"initial_world", "evaluation_initial_world", "initial_h", "final_world", "final_h"}
        assert report["source_native_world_stage"] == "final"
        source = torch.load(ROOT/report["source_native_world_path"], map_location="cpu", weights_only=False)["final"]
        expected_tick = 66047 if run["study_step"] == 256 else 74239
        assert states["initial_world"]["tick"] == expected_tick and states["initial_world"]["acted"]
        assert report["global_start_tick"] == expected_tick+1
        assert state_digest(states["initial_world"]) == state_digest(source)
        assert torch.equal(states["initial_h"], cp["hidden"])
        changed = deepcopy(states["initial_world"])
        allowed = [native.SEEDS[run["seed"]]["history_cue"], native.SEEDS[run["seed"]]["B_cue"]]
        changed["context"]["family_allowed"], changed["context"]["future_family"] = allowed, None
        changed["context"]["history"].append(dict(tick=expected_tick, allowed=allowed))
        assert state_digest(changed) == state_digest(states["evaluation_initial_world"])
        h = states["initial_h"].unsqueeze(0).clone()
        np.testing.assert_array_equal(facts["world_tick"], np.arange(expected_tick+1, expected_tick+1+total))
    else:
        assert config["seed"] == 856000+run["seed"]
        assert config["actor_ids"] == list(range(101000, 101032))
        assert config["duration_range"] == CONDITIONS[run["condition"]]
        assert config["gap_range"] == [16., 32.]
        assert config["dt"] == .25
    with mapping_scope(mapping_table(run["seed"], "related")):
        world = ContextWorld.from_state_dict(states["evaluation_initial_world"], device="cpu") if states is not None else ContextWorld(**config)
        for tick in range(total):
            obs = world.observe()
            np.testing.assert_array_equal(obs.numpy(), facts["obs"][tick])
            for name, attr in FACT_FIELDS.items():
                hashes[name].update(getattr(world, attr).numpy())
            global_tick = report["global_start_tick"]+tick if states is not None else tick
            assert world.now == facts["now"][tick] == global_tick*.25
            values, h = core(obs.unsqueeze(0), h)
            scores = head(values[0]).reshape(batch, 2, 11)
            np.testing.assert_array_equal(scores.numpy(), logits[tick])
            np.testing.assert_array_equal(scores.argmax(-1).numpy(), facts["actions"][tick])
            np.testing.assert_array_equal(time_head(values[0]).numpy(), prediction[tick])
            world.act(torch.from_numpy(facts["actions"][tick]))
            np.testing.assert_array_equal(world.last_result.numpy(), facts["receipts"][tick])
        assert world.event_hash == report["external_hash"]
        if states is not None:
            assert state_digest(world.state_dict()) == state_digest(states["final_world"])
            assert torch.equal(h[0], states["final_h"])
    for name, value in hashes.items():
        assert value.hexdigest() == expected[name], (index, name)
    np.testing.assert_array_equal(h[0].numpy(), facts["final_h"])
    write_new(OUT/"replay_runs"/f"run_{int(index):03d}.json", dict(index=int(index),
        **{key: run[key] for key in ("mode", "seed", "policy", "condition", "study_step")},
        passed=True, ticks=total, opportunities=total*batch*2,
        native_observations_private_facts_actions_receipts="bitwise exact every tick",
        time_logits="bitwise exact every tick", final_h="bitwise exact",
        live_final_world_exact=states is not None, plan_sha256=sha(PLAN), auditor_sha256=sha(__file__)))
    print(json.dumps(dict(index=int(index), replay_passed=True)), flush=True)


def replay_all(workers=6):
    plan = checked()
    assert 1 <= workers <= 6
    logs = OUT/"replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(index):
        path = logs/f"run_{index:03d}.log"
        with path.open("x", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", "-m", "experiments.episodic_review.eval_audit",
                "replay", "--index", str(index)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        row = dict(index=index, exit_code=result.returncode, log=str(path.relative_to(ROOT)))
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finished = list(pool.map(job, range(len(plan["runs"]))))
    assert all(row["exit_code"] == 0 for row in finished), finished
    results = [read(OUT/"replay_runs"/f"run_{index:03d}.json") for index in range(len(plan["runs"]))]
    write_new(OUT/"replay.json", dict(format="episodic-review-all60-independent-native-policy-replay-v1",
        passed=True, runs=len(results), ticks_compared=sum(row["ticks"] for row in results),
        opportunities_compared=sum(row["opportunities"] for row in results), sidecars_checked=len(results),
        native_observations_private_facts_actions_receipts_time_logits_finalh="all bitwise exact",
        plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), workers=finished, results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "metrics", "replay", "replay-all", "all"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "metrics":
        metrics()
    elif args.command == "replay":
        assert args.index is not None
        replay(args.index)
    elif args.command == "replay-all":
        replay_all(args.workers)
    else:
        metrics()
        replay_all(args.workers)


if __name__ == "__main__":
    main()
