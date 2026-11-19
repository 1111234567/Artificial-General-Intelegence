"""Independent raw request metrics and native/plain-GRU evaluation replay.

No experiment evaluator, model, target, action chooser, scorer or teacher is
called. Existing frozen independent arithmetic is reused for the diagnostics.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from experiments.action_split import eval_audit as native
from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.remaining_time.eval_audit import exact, independent_time_statistics, write_new
from experiments.timing_boundary.data import FACT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data/coverage-retention/eval-v1"
OUT = ROOT / "data/coverage-retention/eval-audit-v1"
PLAN = OUT / "audit_plan.json"
LIVE = False
sha, read, digest = native.sha, native.read, native.digest
EXTERNAL_FIELDS = ("actor_ids", "now", "event_kind", "event_id", "event_eligible",
    "event_family", "event_raw", "event_version", "event_duration", "event_window",
    "mature", "mature_id", "mature_issued", "mature_duration", "mature_family")


def declare():
    protocol, freeze, completion = [read(EVAL / name) for name in
                                      ("protocol.json", "freeze.json", "completion.json")]
    assert protocol["seeds"] == [0, 1, 2]
    assert protocol["policies"] == (["control", "withdraw"] if LIVE else
                                    ["baseline", "control256", "wide256", "withdraw256"])
    assert (protocol["warm"], protocol["dt"]) == (512, .25)
    assert list(protocol["conditions"].values()) == ([[8., 16.]] if LIVE else
                                                   [[8., 16.], [16., 32.], [32., 64.]])
    assert protocol["test"] == 4096
    assert completion["complete"]
    assert freeze["protocol_sha256"] == sha(EVAL / "protocol.json")
    expected = len(protocol["seeds"]) * len(protocol["policies"]) * len(protocol["conditions"])
    assert completion["runs"] == completion["sidecars"] == expected == (6 if LIVE else 36)
    inputs = {str((EVAL / name).relative_to(ROOT)): sha(EVAL / name) for name in
              ("protocol.json", "freeze.json", "completion.json")}
    frozen_inputs = freeze["files"] if LIVE else freeze["checkpoints"]
    for name, expected_hash in frozen_inputs.items():
        assert sha(ROOT / name) == expected_hash, name
        inputs[name] = expected_hash
    for table in (completion["files"], completion.get("time_files", {})):
        for name, expected_hash in table.items():
            assert sha(ROOT / name) == expected_hash, name
            inputs[name] = expected_hash
    runs = []
    for seed in protocol["seeds"]:
        for policy in protocol["policies"]:
            for condition in protocol["conditions"]:
                stem = EVAL / f"seed_{seed:03d}" / f"{policy}_{condition}"
                report = read(stem.with_suffix(".json"))
                raw = stem.with_suffix(".npz")
                assert report["facts_sha256"] == sha(raw)
                assert report["protocol_sha256"] == sha(EVAL / "protocol.json")
                assert report["freeze_sha256"] == sha(EVAL / "freeze.json")
                assert frozen_inputs[report["checkpoint_path"]] == report["checkpoint_sha256"]
                assert str(stem.with_suffix(".json").relative_to(ROOT)) in completion["files"]
                inputs[str(raw.relative_to(ROOT))] = sha(raw)
                if LIVE:
                    state_path = ROOT / report["initial_state_path"]
                    assert sha(state_path) == report["initial_state_sha256"]
                    inputs[str(state_path.relative_to(ROOT))] = sha(state_path)
                    source_path = ROOT / report["source_native_world_path"]
                    assert sha(source_path) == report["source_native_world_sha256"]
                    inputs[str(source_path.relative_to(ROOT))] = sha(source_path)
                side = stem.with_name(stem.name + "_time")
                has_time = side.with_suffix(".json").exists()
                assert has_time, f"every learned policy including baseline has a time sidecar: {side}"
                if has_time:
                    saved = read(side.with_suffix(".json"))
                    assert saved["facts_sha256"] == sha(raw)
                    assert saved["sidecar_sha256"] == sha(side.with_suffix(".npz"))
                    for ext in (".json", ".npz"):
                        path = side.with_suffix(ext)
                        inputs[str(path.relative_to(ROOT))] = sha(path)
                runs.append(dict(seed=seed, policy=policy, condition=condition,
                                 stem=str(stem.relative_to(ROOT)), has_time=has_time))
    sources = (Path(__file__), ROOT / "experiments/action_split/eval_audit.py",
        ROOT / "experiments/remaining_time/eval_audit.py",
        ROOT / "experiments/experience_transfer/train_transfer.py",
        ROOT / "experiments/timing_boundary/data.py",
        ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py")
    pinned = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    for name, expected_hash in protocol["sources"].items():
        assert sha(ROOT / name) == expected_hash, name
        pinned[name] = expected_hash
    write_new(PLAN, dict(format="coverage-retention-independent-evaluation-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), runs=runs, inputs=inputs, sources=pinned,
        scope="All raw pre-action facts, actions, receipts, request ledgers and complete mature-request denominators; independent native-world and plain-GRU replay; every archived time/logits sidecar; exact external pairing across every frozen policy",
        expected_runs=len(runs), warm=protocol["warm"], test=protocol["test"], live=LIVE))


def checked(all_inputs=True):
    plan = read(PLAN)
    assert plan["live"] == LIVE
    for table in ("inputs", "sources"):
        for name, expected_hash in plan[table].items():
            if table == "inputs" and not all_inputs and name not in {
                str((EVAL / value).relative_to(ROOT)) for value in ("protocol.json", "freeze.json", "completion.json")}:
                continue
            assert sha(ROOT / name) == expected_hash, name
    return plan


def metrics():
    plan = checked()
    protocol = read(EVAL / "protocol.json")
    behavior, auxiliary = [], []
    for run in plan["runs"]:
        stem = ROOT / run["stem"]
        report = read(stem.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
        inherited = None
        if LIVE:
            states = torch.load(ROOT / report["initial_state_path"], map_location="cpu", weights_only=False)
            actual, inherited = score_live(facts, run["seed"], plan["warm"], states["initial_world"])
        else:
            actual = native.score(facts, run["seed"], plan["warm"])
        for key in ("coverage", "groups", "requests", "idle_opportunities", "idle_actions"):
            exact(actual[key], report[key], f"{run['stem']}/{key}")
        identity = {key: run[key] for key in ("seed", "policy", "condition")}
        behavior.append(dict(**identity, passed=True, coverage=actual["coverage"], groups=actual["groups"],
            requests_digest=digest(actual["requests"]), request_count=len(actual["requests"]),
            idle_opportunities=actual["idle_opportunities"], idle_actions=actual["idle_actions"],
            facts_sha256=sha(stem.with_suffix(".npz")), report_sha256=sha(stem.with_suffix(".json")),
            inherited_request_audit=inherited))
        if run["has_time"]:
            side = stem.with_name(stem.name + "_time")
            saved = read(side.with_suffix(".json"))
            with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
                prediction, logits = archive["time_prediction"], archive["native_logits"]
            np.testing.assert_array_equal(logits.argmax(-1), facts["actions"])
            for key in ("checkpoint_sha256", "protocol_sha256", "freeze_sha256"):
                assert saved[key] == report[key]
            result = independent_time_statistics(facts, prediction, logits, actual["requests"], plan["warm"])
            for key, value in result.items():
                exact(value, saved[key], f"{run['stem']}/time/{key}")
            auxiliary.append(dict(**identity, passed=True, groups=result["groups"], boundary=result["boundary"],
                mature_request_denominator=result["mature_request_denominator"],
                paired_requests=result["paired_requests"], boundary_rows_digest=digest(result["boundary_rows"]),
                sidecar_sha256=sha(side.with_suffix(".npz"))))
        print(json.dumps(dict(**identity, metrics_passed=True)), flush=True)
    pairing = []
    for seed in protocol["seeds"]:
        for condition in protocol["conditions"]:
            directory = EVAL / f"seed_{seed:03d}"
            anchor_policy = protocol["policies"][0]
            anchor_stem = directory / f"{anchor_policy}_{condition}"
            anchor_report = read(anchor_stem.with_suffix(".json"))
            ids = [(r["actor"], r["job"], r["rid"]) for r in anchor_report["requests"]]
            with np.load(anchor_stem.with_suffix(".npz"), allow_pickle=False) as anchor:
                for policy in protocol["policies"][1:]:
                    other_stem = directory / f"{policy}_{condition}"
                    report = read(other_stem.with_suffix(".json"))
                    if not LIVE:
                        assert report["external_hash"] == anchor_report["external_hash"]
                    assert [(r["actor"], r["job"], r["rid"]) for r in report["requests"]] == ids
                    with np.load(other_stem.with_suffix(".npz"), allow_pickle=False) as other:
                        for key in EXTERNAL_FIELDS:
                            select = slice(plan["warm"], None) if LIVE and key != "actor_ids" else slice(None)
                            np.testing.assert_array_equal(anchor[key][select], other[key][select],
                                err_msg=f"{seed}/{condition}/{policy}/external/{key}")
            pairing.append(dict(seed=seed, condition=condition, passed=True,
                policies=len(protocol["policies"]), mature_requests=len(ids),
                mature_request_ids_digest=digest(ids),
                external_hash=anchor_report["external_hash"] if not LIVE else "history prefixes differ; all postwarm raw external fields exact"))
    write_new(OUT / "metrics.json", dict(format="coverage-retention-independent-metrics-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        plan_sha256=sha(PLAN), runs=behavior, auxiliary=auxiliary, paired_external_cases=pairing,
        interpretation="All mature test-issued eligible requests remain denominators, including no-BEGIN and no-window failures; time-valid rows and adjacent pairs are diagnostics only."))


def score_live(facts, seed, warm, initial_world):
    """Retain every actual future command and independently check old receipts.

    The frozen scorer begins with an empty command ledger. Only pre-existing
    requests are removed from that ledger's view, after checking every real
    receipt independently. All inherited activity must end before scoring.
    """
    action = facts["actions"]
    begin = np.isin(action, (1, 2))
    finish = (action >= 3) & (action <= 10)
    version = np.where(begin, action-1, (action-3)//4)
    now = facts["now"][:, None, None]
    accepted_begin = begin & facts["active"] & ~facts["started"] & (version == facts["version"])
    accepted_finish = finish & facts["active"] & facts["started"] & (version == facts["version"]) & (
        (action-3)%4 == facts["payload"]) & (now >= facts["ready"]-1e-6) & (now <= facts["deadline"]+1e-6)
    receipts = np.where(action == 0, 0, np.where(accepted_begin | accepted_finish, 1, -1))
    np.testing.assert_array_equal(receipts, facts["receipts"])
    first_tick = initial_world["tick"] + 1
    inherited = facts["rid"] < 2*first_tick
    inherited_mature = facts["mature"] & (facts["mature_id"] < 2*first_tick)
    for tick, lane, job in zip(*np.nonzero(facts["active"] & inherited)):
        assert int(facts["rid"][tick, lane, job]) == int(initial_world["tensors"]["request_id"][lane, job])
    for tick, lane, job in zip(*np.nonzero(inherited_mature)):
        assert int(facts["mature_id"][tick, lane, job]) == int(initial_world["tensors"]["_score_request_id"][lane, job])
    assert not (facts["active"][warm:] & inherited[warm:]).any()
    assert not inherited_mature[warm:].any()
    # Every actual new/replacement request has a new global issue identity.
    command = np.isin(facts["event_kind"], (1, 2))
    assert (facts["event_id"][command] >= 2*first_tick).all()
    view = dict(facts)
    view["active"] = facts["active"] & ~inherited
    view["mature"] = facts["mature"] & ~inherited_mature
    view["actions"] = facts["actions"].copy()
    view["receipts"] = facts["receipts"].copy()
    old_active = facts["active"] & inherited
    view["actions"][old_active] = 0
    view["receipts"][old_active] = 0
    result = native.score(view, seed, warm)
    idle = ~facts["active"][warm:]
    assert result["idle_opportunities"] == int(idle.sum())
    assert result["idle_actions"] == int((idle & (action[warm:] != 0)).sum())
    return result, dict(all_real_receipts_exact=True, inherited_active_rows=int(old_active.sum()),
        inherited_maturity_pulses=int(inherited_mature.sum()), inherited_gone_before_scoring=True,
        initial_world_tick=initial_world["tick"], first_new_world_tick=first_tick)


@torch.inference_mode()
def replay(index):
    torch.set_num_threads(1)
    plan = checked(all_inputs=False)
    run = plan["runs"][index]
    stem = ROOT / run["stem"]
    report = read(stem.with_suffix(".json"))
    paths = [stem.with_suffix(ext) for ext in (".json", ".npz")]
    paths.append(ROOT / report["checkpoint_path"])
    if run["has_time"]:
        paths.extend(stem.with_name(stem.name + "_time").with_suffix(ext) for ext in (".json", ".npz"))
    if LIVE:
        paths.extend(ROOT / report[key] for key in ("initial_state_path", "source_native_world_path"))
    for path in paths:
        assert sha(path) == plan["inputs"][str(path.relative_to(ROOT))], path
    cp = torch.load(ROOT / report["checkpoint_path"], map_location="cpu", weights_only=False)
    assert cp["model_config"] == dict(width=192, input_size=56, n_jobs=2, n_actions=11)
    core, head = nn.GRU(56, 192), nn.Linear(192, 22)
    for prefix, module in (("core.", core), ("head.", head)):
        module.load_state_dict({name.removeprefix(prefix): value for name, value in cp["model"].items()
                                if name.startswith(prefix)})
        module.eval().requires_grad_(False)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        facts = {key: archive[key] for key in archive.files}
    total, batch = plan["warm"] + plan["test"], 32
    assert facts["obs"].shape == (total, batch, 56) and facts["obs"].dtype == np.float32
    assert facts["actions"].shape == (total, batch, 2) and facts["actions"].dtype == np.int64
    time_head = None
    if run["has_time"]:
        side = stem.with_name(stem.name + "_time")
        with np.load(side.with_suffix(".npz"), allow_pickle=False) as archive:
            predictions, logits = archive["time_prediction"], archive["native_logits"]
        time_head = nn.Linear(192, 2).eval().requires_grad_(False)
        time_head.load_state_dict({name.removeprefix("time_head."): value for name, value in cp["model"].items()
                                   if name.startswith("time_head.")})
        assert predictions.dtype == logits.dtype == np.float32
        assert predictions.shape == (total, batch, 2) and logits.shape == (total, batch, 2, 11)
    expected = {name: hashlib.sha256(np.ascontiguousarray(facts[name])).hexdigest() for name in FACT_FIELDS}
    hashes = {name: hashlib.sha256() for name in FACT_FIELDS}
    h = torch.zeros(1, batch, 192)
    protocol = read(EVAL / "protocol.json")
    config = report["world_config"]
    assert config["dt"] == .25
    states = None
    if LIVE:
        states = torch.load(ROOT / report["initial_state_path"], map_location="cpu", weights_only=False)
        assert set(states) == {"initial_world", "evaluation_initial_world", "initial_h", "final_world", "final_h"}
        source = torch.load(ROOT / report["source_native_world_path"], map_location="cpu", weights_only=False)["final"]
        assert report["source_native_world_stage"] == "final"
        assert state_digest(states["initial_world"]) == state_digest(source)
        assert torch.equal(states["initial_h"], cp["hidden"])
        assert states["initial_world"]["tick"] == 66047 and states["initial_world"]["acted"]
        assert report["global_start_tick"] == states["initial_world"]["tick"] + 1
        expected_initial = deepcopy(states["initial_world"])
        allowed = [native.SEEDS[run["seed"]]["history_cue"], native.SEEDS[run["seed"]]["B_cue"]]
        expected_initial["context"]["family_allowed"] = allowed
        expected_initial["context"]["future_family"] = None
        expected_initial["context"]["history"].append(dict(tick=expected_initial["tick"], allowed=allowed))
        assert state_digest(expected_initial) == state_digest(states["evaluation_initial_world"])
        h = states["initial_h"].unsqueeze(0).clone()
        np.testing.assert_array_equal(facts["world_tick"], np.arange(report["global_start_tick"], report["global_start_tick"]+total))
    else:
        assert config["gap_range"] == [16., 32.]
        assert config["duration_range"] == protocol["conditions"][run["condition"]]
        assert config["actor_ids"] == protocol["actor_ids"] == list(range(100000, 100032))
        assert config["seed"] == 855000 + run["seed"]
    with mapping_scope(mapping_table(run["seed"], "related")):
        world = (ContextWorld.from_state_dict(states["evaluation_initial_world"], device="cpu")
                 if LIVE else ContextWorld(**config))
        for tick in range(total):
            obs = world.observe()
            np.testing.assert_array_equal(obs.numpy(), facts["obs"][tick])
            for name, attr in FACT_FIELDS.items():
                hashes[name].update(getattr(world, attr).numpy())
            assert world.now == facts["now"][tick] == ((report["global_start_tick"]+tick) if LIVE else tick) * .25
            values, h = core(obs.unsqueeze(0), h)
            scores = head(values[0]).reshape(batch, 2, 11)
            np.testing.assert_array_equal(scores.argmax(-1).numpy(), facts["actions"][tick])
            if time_head is not None:
                np.testing.assert_array_equal(scores.numpy(), logits[tick])
                np.testing.assert_array_equal(time_head(values[0]).numpy(), predictions[tick])
            world.act(torch.from_numpy(facts["actions"][tick]))
            np.testing.assert_array_equal(world.last_result.numpy(), facts["receipts"][tick])
        assert world.event_hash == report["external_hash"]
        if LIVE:
            assert state_digest(world.state_dict()) == state_digest(states["final_world"])
    for name, value in hashes.items():
        assert value.hexdigest() == expected[name], (index, name)
    np.testing.assert_array_equal(h[0].numpy(), facts["final_h"])
    if LIVE:
        assert torch.equal(h[0], states["final_h"])
    write_new(OUT / "replay_runs" / f"run_{index:03d}.json", dict(index=index,
        **{key: run[key] for key in ("seed", "policy", "condition")}, passed=True, ticks=total,
        opportunities=total*batch*2, native_facts="all bitwise exact", actions="all bitwise exact",
        final_h="bitwise exact", sidecar=run["has_time"], auditor_sha256=sha(__file__), plan_sha256=sha(PLAN)))
    print(json.dumps(dict(index=index, replay_passed=True)), flush=True)


def replay_all(workers=6):
    plan = checked()
    assert 1 <= workers <= 6
    logs = OUT / "replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(index):
        path = logs / f"run_{index:03d}.log"
        with path.open("x", encoding="utf-8") as handle:
            done = subprocess.run([sys.executable, "-m", "experiments.coverage_retention.eval_audit",
                "replay", "--index", str(index), *(["--live"] if LIVE else [])], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = dict(index=index, exit_code=done.returncode, log=str(path.relative_to(ROOT)))
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finished = list(pool.map(job, range(len(plan["runs"]))))
    assert all(row["exit_code"] == 0 for row in finished), finished
    results = [read(OUT / "replay_runs" / f"run_{index:03d}.json") for index in range(len(plan["runs"]))]
    write_new(OUT / "replay.json", dict(format="coverage-retention-independent-native-policy-replay-v1",
        passed=True, runs=len(results), ticks_compared=sum(row["ticks"] for row in results),
        opportunities_compared=sum(row["opportunities"] for row in results),
        native_observations_private_facts_receipts_actions_finalh="all bitwise exact",
        sidecars_checked=sum(int(row["sidecar"]) for row in results),
        plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), workers=finished, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "metrics", "replay", "replay-all", "all"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.live:
        LIVE = True
        EVAL = ROOT / "data/coverage-retention/live-v1"
        OUT = ROOT / "data/coverage-retention/live-audit-v1"
        PLAN = OUT / "audit_plan.json"
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
