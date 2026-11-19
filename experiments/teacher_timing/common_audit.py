"""Independently audit both crossed teacher histories and all 24 sidecars.

Only raw archives, frozen native request-state-machine scoring and independent
time arithmetic are used. No teacher, evaluator or tested policy is executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.action_split import eval_audit as native
from experiments.continuity.run import state_digest
from experiments.remaining_time.eval_audit import exact, independent_time_statistics, write_new
from experiments.teacher_timing.eval_audit import EXTERNAL_FIELDS

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "data/teacher-timing/common-v1"
EVAL = ROOT / "data/teacher-timing/eval-v1"
OUT = ROOT / "data/teacher-timing/common-audit-v1"
PLAN = OUT / "metrics_plan.json"
TEACHERS = ("earliest", "midpoint")
POLICIES = ("earliest_action", "earliest_joint", "midpoint_action", "midpoint_joint")
sha, read, digest = native.sha, native.read, native.digest
EPS, WARM = 1e-6, 512


def raw_path(seed, teacher):
    return COMMON / f"seed_{seed:03d}" / f"teacher_{teacher}_long.npz"


def side_path(seed, teacher, policy):
    return COMMON / f"seed_{seed:03d}" / f"history_{teacher}__{policy}.npz"


def declare():
    protocol, freeze, completion = [read(COMMON / name) for name in
        ("protocol.json", "histories.freeze.json", "completion.json")]
    assert state_digest({k: v for k, v in protocol.items() if k != "protocol_hash"}) == protocol["protocol_hash"]
    assert state_digest({k: v for k, v in freeze.items() if k != "freeze_hash"}) == freeze["freeze_hash"]
    assert protocol["seeds"] == [0, 1, 2] and protocol["test_teachers"] == list(TEACHERS)
    assert protocol["policies"] == list(POLICIES) and protocol["conditions"] == ["long"]
    assert completion["complete"] and completion["histories"] == 6 and completion["model_replays"] == 24
    assert freeze["model_replay_started"] is False and freeze["histories"] == 6
    assert freeze["protocol_sha256"] == completion["protocol_sha256"] == sha(COMMON / "protocol.json")
    assert completion["history_freeze_sha256"] == sha(COMMON / "histories.freeze.json")
    inputs = {str((COMMON / name).relative_to(ROOT)): sha(COMMON / name)
              for name in ("protocol.json", "histories.freeze.json", "completion.json")}
    for table in (freeze["files"], completion["files"], protocol["checkpoints"]):
        for name, expected in table.items():
            assert sha(ROOT / name) == expected, name
            inputs[name] = expected
    for seed in (0, 1, 2):
        for suffix in (".npz", ".json"):
            path = EVAL / f"seed_{seed:03d}" / f"original_long{suffix}"
            inputs[str(path.relative_to(ROOT))] = sha(path)
    sources = [Path(__file__), ROOT / "experiments/action_split/eval_audit.py",
               ROOT / "experiments/remaining_time/eval_audit.py",
               ROOT / "experiments/teacher_timing/eval_audit.py",
               ROOT / "experiments/continuity/run.py"]
    write_new(PLAN, dict(format="teacher-timing-independent-crossed-common-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        sources={**protocol["source_sha256"], **{str(p.relative_to(ROOT)): sha(p) for p in sources}},
        scope="Six teacher request state machines, actual oracle timing and public feedback, independently accepted-BEGIN-derived ready times and labels, every24 sidecar time metric and shared history denominators; paired external raw arrays/world RNG and autonomous case IDs"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for name, expected in plan[table].items():
            assert sha(ROOT / name) == expected, name
    return plan


def independent_teacher(facts, teacher):
    actions = facts["actions"]
    assert actions.shape == (4608, 32, 2) and actions.dtype == np.int64
    assert facts["obs"].shape == (4608, 32, 56) and facts["obs"].dtype == np.float32
    np.testing.assert_array_equal(facts["now"], np.arange(4608) * .25)
    np.testing.assert_array_equal(facts["actor_ids"], np.arange(99000, 99032))
    current = facts["now"][:, None, None]
    can_start = facts["active"] & ~facts["started"] & (current + facts["duration"] <= facts["deadline"] + EPS)
    can_finish = facts["active"] & facts["started"] & (current >= facts["ready"] - EPS) & (current <= facts["deadline"] + EPS)
    if teacher == "midpoint":
        can_finish &= ((current >= (facts["ready"] + facts["deadline"]) / 2. - EPS)
                       | (current + .25 > facts["deadline"] + EPS))
    expected = np.zeros_like(actions)
    expected[can_start] = 1 + facts["version"][can_start]
    expected[can_finish] = 3 + 4 * facts["version"][can_finish] + facts["payload"][can_finish]
    np.testing.assert_array_equal(expected, actions)
    obs = facts["obs"].reshape(4608, 32, 2, 28)
    onehot = np.eye(11, dtype=np.float32)[actions[:-1]]
    onehot[actions[:-1] == 0] = 0
    np.testing.assert_array_equal(obs[1:, ..., 11:22], onehot)
    np.testing.assert_array_equal(obs[1:, ..., 22], facts["receipts"][:-1] == 1)
    np.testing.assert_array_equal(obs[1:, ..., 23], facts["receipts"][:-1] == -1)
    # Rebuild ready times from accepted earlier BEGINs and command durations.
    ledger, normalized, mask = {}, np.zeros(actions.shape, np.float32), np.zeros(actions.shape, bool)
    for tick in range(len(actions)):
        for lane, job in np.argwhere(facts["active"][tick] & facts["started"][tick]):
            key = int(lane), int(job), int(facts["rid"][tick, lane, job])
            assert key in ledger
            ready = ledger[key]
            assert ready == float(facts["ready"][tick, lane, job])
            mask[tick, lane, job] = True
            normalized[tick, lane, job] = (ready - float(facts["now"][tick])) / 16.
        for lane, job in np.argwhere(np.isin(actions[tick], (1, 2)) & (facts["receipts"][tick] == 1)):
            key = int(lane), int(job), int(facts["rid"][tick, lane, job])
            assert key not in ledger
            ledger[key] = float(facts["now"][tick]) + float(facts["duration"][tick, lane, job])
    np.testing.assert_array_equal(mask, facts["time_mask"])
    np.testing.assert_array_equal(normalized, facts["time_target"])
    return dict(passed=True, teacher=teacher, oracle_actions_exact=True,
        public_previous_action_receipts_exact=True, independent_begin_ledger_labels_exact=True,
        accepted_begin_requests=len(ledger), valid_time_rows=int(mask.sum()))


def run():
    plan, completion = checked(), read(COMMON / "completion.json")
    outputs, pairing = [], []
    for seed in (0, 1, 2):
        pair_facts, pair_reports, pair_worlds = {}, {}, {}
        anchor_path = EVAL / f"seed_{seed:03d}" / "original_long.npz"
        anchor_report = read(anchor_path.with_suffix(".json"))
        for teacher in TEACHERS:
            path = raw_path(seed, teacher)
            teacher_report = read(path.with_suffix(".json"))
            assert teacher_report["teacher_request_report_only"] and not teacher_report["autonomous_test_model_score"]
            assert teacher_report["raw_facts_sha256"] == sha(path)
            with np.load(path, allow_pickle=False) as archive:
                facts = dict(archive)
            causal = independent_teacher(facts, teacher)
            state_machine = native.score(facts, seed, WARM)
            for key in ("coverage", "groups", "requests", "idle_opportunities", "idle_actions"):
                exact(state_machine[key], teacher_report[key], f"teacher/{seed}/{teacher}/{key}")
            assert anchor_report["external_hash"] == teacher_report["external_hash"]
            ids = [(r["actor"], r["job"], r["rid"]) for r in state_machine["requests"]]
            assert ids == [(r["actor"], r["job"], r["rid"]) for r in anchor_report["requests"]]
            with np.load(anchor_path, allow_pickle=False) as anchor:
                for key in EXTERNAL_FIELDS:
                    np.testing.assert_array_equal(anchor[key], facts[key],
                        err_msg=f"common/autonomous/{seed}/{teacher}/{key}")
            policies = {}
            for policy in POLICIES:
                side = side_path(seed, teacher, policy)
                report = read(side.with_suffix(".json"))
                assert report["diagnostic_only"] and report["no_test_model_actions_executed"]
                assert report["facts_sha256"] == sha(path)
                assert report["teacher_request_report_sha256"] == sha(path.with_suffix(".json"))
                assert report["raw_predictions_sha256"] == sha(side)
                assert report["model_before_sha256"] == report["model_after_sha256"]
                assert report["protocol_sha256"] == completion["protocol_sha256"]
                assert report["history_freeze_sha256"] == completion["history_freeze_sha256"]
                assert sha(ROOT / report["checkpoint_path"]) == report["checkpoint_sha256"]
                with np.load(side, allow_pickle=False) as archive:
                    prediction, logits, final_h = archive["time_prediction"], archive["native_logits"], archive["final_h"]
                assert final_h.shape == (32, 192) and final_h.dtype == np.float32 and np.isfinite(final_h).all()
                actual = independent_time_statistics(facts, prediction, logits, state_machine["requests"])
                for key, value in actual.items():
                    exact(value, report[key], f"common/{seed}/{teacher}/{policy}/{key}")
                for key in ("groups", "boundary", "mature_request_denominator", "paired_requests"):
                    exact(actual[key], completion["cases"][f"{seed}:{teacher}"][policy][key], "completion summary")
                policies[policy] = dict(passed=True, groups=actual["groups"], boundary=actual["boundary"],
                    mature_request_denominator=actual["mature_request_denominator"],
                    paired_requests=actual["paired_requests"], boundary_rows_digest=digest(actual["boundary_rows"]),
                    sidecar_sha256=sha(side), report_sha256=sha(side.with_suffix(".json")))
            for key in ("mature_request_denominator", "paired_requests"):
                assert len({row[key] for row in policies.values()}) == 1
            assert len({row["groups"]["all_valid"]["rows"] for row in policies.values()}) == 1
            outputs.append(dict(seed=seed, test_teacher=teacher, condition="long", passed=True,
                teacher_causal_audit=causal, teacher_facts_sha256=sha(path),
                teacher_request_report_sha256=sha(path.with_suffix(".json")),
                teacher_requests=len(ids), teacher_request_digest=digest(state_machine["requests"]),
                matched_autonomous_external_raw_arrays_and_request_ids=True, policies=policies))
            pair_facts[teacher], pair_reports[teacher] = facts, teacher_report
            worlds_path = path.with_name(path.stem + "_worlds.pt")
            assert sha(worlds_path) == teacher_report["worlds_sha256"]
            pair_worlds[teacher] = torch.load(worlds_path, map_location="cpu", weights_only=False)
        for key in EXTERNAL_FIELDS:
            np.testing.assert_array_equal(pair_facts["earliest"][key], pair_facts["midpoint"][key])
        for stage in ("initial", "final"):
            left, right = [pair_worlds[teacher][stage] for teacher in TEACHERS]
            assert state_digest(left["scheduler"]) == state_digest(right["scheduler"])
            for field in ("rng_state", "assigned"):
                assert torch.equal(left["context"][field], right["context"][field])
        pairing.append(dict(seed=seed, passed=True, paired_teacher_exogenous_raw_arrays="bitwise exact",
            initial_final_scheduler_context_rng="bitwise exact",
            external_fields=list(EXTERNAL_FIELDS), external_hash=pair_reports["earliest"]["external_hash"]))
    write_new(OUT / "metrics.json", dict(format="teacher-timing-independent-crossed-common-metrics-v1",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        plan_sha256=sha(PLAN), scope=plan["scope"], histories=6, auxiliary_replays=24,
        cases=outputs, paired_test_teachers=pairing,
        interpretation="All models are compared on identical observations within each actual test-teacher history. Teacher outcomes validate requests and denominators; they are not tested-policy autonomous performance."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    args = parser.parse_args()
    declare() if args.command == "declare" else run()
