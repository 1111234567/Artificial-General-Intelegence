"""Independent fixed-phase posthoc count reconstruction from archived logits.

No phase_readout routine, evaluator, teacher or model inference is called.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.action_split import eval_audit as native
from experiments.remaining_time.eval_audit import exact, write_new

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "data/teacher-timing/phase-readout-v1"
COMMON = ROOT / "data/teacher-timing/common-v1"
OUT = ROOT / "data/teacher-timing/phase-audit-v1"
PLAN = OUT / "audit_plan.json"
POLICIES = ("earliest_action", "earliest_joint", "midpoint_action", "midpoint_joint")
PHASES = ("first_legal", "teacher_FINISH")
sha, read, digest = native.sha, native.read, native.digest


def declare():
    protocol, result = read(PHASE / "protocol.json"), read(PHASE / "results.json")
    assert protocol["posthoc"] and result["posthoc"]
    assert protocol["test_teacher"] == "midpoint" and protocol["condition"] == "long"
    assert protocol["seeds"] == [0, 1, 2] and protocol["policies"] == list(POLICIES)
    assert protocol["phases"] == list(PHASES) and protocol["expected_requests_total"] == 4057
    assert digest({k: v for k, v in protocol.items() if k != "protocol_hash"}) == protocol["protocol_hash"]
    assert result["protocol_sha256"] == sha(PHASE / "protocol.json")
    assert result["protocol_hash"] == protocol["protocol_hash"]
    paths = [PHASE / name for name in ("protocol.json", "results.json", "per_seed.csv")]
    paths.extend(map(Path, result["evidence_files"]))
    paths.extend(map(Path, protocol["input_sha256"]))
    for path, expected in result["evidence_files"].items():
        assert sha(path) == expected
    sources = {str(p): sha(p) for p in (Path(__file__),
        ROOT / "experiments/action_split/eval_audit.py", ROOT / "experiments/remaining_time/eval_audit.py")}
    for table in ("source_sha256", "preserved_producer_source_sha256", "input_sha256"):
        for path, expected in protocol[table].items():
            assert sha(path) == expected, path
            if table != "input_sha256":
                sources[path] = expected
    write_new(PLAN, dict(format="teacher-timing-independent-fixed-phase-posthoc-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), posthoc=True,
        inputs={str(p): sha(p) for p in dict.fromkeys(paths)}, sources=sources,
        scope="Independently recover4057 mature requests by raw state machine; one streaming pass finds first legal and unique accepted teacher FINISH; all request identities/ticks/targets and32456 fixed policy-logit rows, twelve seed-policy summaries, four pooled summaries and CSV counts exact",
        interpretation="Posthoc diagnostic after main outcomes, on actual midpoint teacher histories only; no new model inference or autonomous tested-policy completion claims"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for path, expected in plan[table].items():
            assert sha(path) == expected, path
    return plan


def stream_positions(facts, seed):
    scored = native.score(facts, seed, 512)
    requests = scored["requests"]
    counts = (1338, 1363, 1356)
    assert len(requests) == counts[seed]
    keys = [(r["lane"], r["job"], r["rid"]) for r in requests]
    assert len(set(keys)) == len(keys)
    wanted, first, finishes = set(keys), {}, {}
    for tick, now in enumerate(facts["now"]):
        for lane, job in np.argwhere(facts["active"][tick] & facts["started"][tick]):
            lane, job = int(lane), int(job)
            key = lane, job, int(facts["rid"][tick, lane, job])
            if key not in wanted:
                continue
            legal = (np.isfinite(facts["ready"][tick, lane, job])
                     and float(now) >= float(facts["ready"][tick, lane, job]) - 1e-6
                     and float(now) <= float(facts["deadline"][tick, lane, job]) + 1e-6)
            if legal and key not in first:
                first[key] = tick
            if facts["actions"][tick, lane, job] >= 3 and facts["receipts"][tick, lane, job] == 1:
                assert legal and key not in finishes
                finishes[key] = tick
    assert set(first) == set(finishes) == wanted
    identities, ticks, versions, payloads, seconds = [], [], [], [], []
    for request, key in zip(requests, keys):
        lane, job, rid = key
        start, finish = first[key], finishes[key]
        issued, mature = request["issued"], request["mature_tick"]
        actor = int(facts["actor_ids"][lane])
        assert request["actor"] == actor and issued >= 512 and start <= finish
        assert facts["event_kind"][issued, lane, job] in (1, 2)
        assert facts["event_eligible"][issued, lane, job]
        assert facts["event_id"][issued, lane, job] == rid
        assert facts["mature"][mature, lane, job] and facts["mature_id"][mature, lane, job] == rid
        version = int(facts["event_version"][issued, lane, job])
        payload = int(facts["event_payload"][issued, lane, job])
        for tick in range(start, finish + 1):
            assert facts["active"][tick, lane, job] and facts["started"][tick, lane, job]
            assert facts["rid"][tick, lane, job] == rid
            assert facts["version"][tick, lane, job] == version
            assert facts["payload"][tick, lane, job] == payload
        assert facts["actions"][finish, lane, job] == 3 + 4 * version + payload
        identities.append((actor, lane, job, rid, issued, mature))
        ticks.append((start, finish)); versions.append(version); payloads.append(payload)
        seconds.append([float(facts["now"][t]) - float(facts["ready"][t, lane, job])
                        for t in (start, finish)])
    result = dict(identity=np.asarray(identities, np.int64), ticks=np.asarray(ticks, np.int64),
        version=np.asarray(versions, np.int64), payload=np.asarray(payloads, np.int64),
        phase_seconds_after_ready=np.asarray(seconds, np.float64))
    result["target_content"] = result["version"] * 4 + result["payload"]
    return result, scored


def independent_counts(logits, selection):
    n = len(selection["identity"])
    native_choice, content_choice = np.empty((n, 2), np.int64), np.empty((n, 2), np.int64)
    content_ok, finish_ok = np.empty((n, 2), bool), np.empty((n, 2), bool)
    phase_counts = [Counter() for _ in PHASES]
    transitions = [Counter(), Counter()]
    for index, identity in enumerate(selection["identity"]):
        _, lane, job, _, _, _ = map(int, identity)
        target = int(selection["target_content"][index])
        for phase, tick in enumerate(selection["ticks"][index]):
            row = logits[int(tick), lane, job]
            assert row.shape == (11,) and row.dtype == np.float32 and np.isfinite(row).all()
            action = int(np.argmax(row))
            content = int(np.argmax(row[3:11]))
            native_choice[index, phase], content_choice[index, phase] = action, content
            content_ok[index, phase], finish_ok[index, phase] = content == target, action == target + 3
            group = phase_counts[phase]
            group["requests"] += 1
            group["conditional8_content_correct"] += int(content == target)
            group["native11_correct_FINISH"] += int(action == target + 3)
            group["native11_WAIT"] += int(action == 0)
            group["native11_wrong_FINISH"] += int(action >= 3 and action != target + 3)
            group["native11_BEGIN"] += int(action in (1, 2))
        for correct, group in zip((content_ok, finish_ok), transitions):
            before, after = map(bool, correct[index])
            name = ("bothcorrect" if before and after else "firstcorrect_laterwrong" if before
                    else "firstwrong_latercorrect" if after else "neither")
            group[name] += 1
    paired_keys = ("firstwrong_latercorrect", "firstcorrect_laterwrong", "bothcorrect", "neither")
    summary = dict(requests=n, phases={phase: dict(count) for phase, count in zip(PHASES, phase_counts)},
        paired_conditional8_content={k: transitions[0][k] for k in paired_keys},
        paired_native11_correct_FINISH={k: transitions[1][k] for k in paired_keys})
    details = dict(native11_argmax=native_choice, conditional8_argmax=content_choice,
        conditional8_correct=content_ok, native11_correct_FINISH=finish_ok)
    return summary, details


def run():
    plan, saved = checked(), read(PHASE / "results.json")
    summaries, audited = [], []
    assert len(saved["rows"]) == 12 and set(saved["pooled_by_policy"]) == set(POLICIES)
    saved_rows = {(r["seed"], r["policy"]): r for r in saved["rows"]}
    assert len(saved_rows) == 12
    for seed in (0, 1, 2):
        raw = COMMON / f"seed_{seed:03d}" / "teacher_midpoint_long.npz"
        with np.load(raw, allow_pickle=False) as archive:
            facts = dict(archive)
        selection, scored = stream_positions(facts, seed)
        report = read(raw.with_suffix(".json"))
        exact(scored["requests"], report["requests"], "independent full mature request records")
        evidence_path = PHASE / f"seed_{seed:03d}_paired_requests.npz"
        with np.load(evidence_path, allow_pickle=False) as evidence:
            expected_keys = set(selection)
            for key, value in selection.items():
                assert value.dtype == evidence[key].dtype and value.shape == evidence[key].shape
                np.testing.assert_array_equal(value, evidence[key], err_msg=f"seed{seed}/{key}")
            for policy in POLICIES:
                predictions = COMMON / f"seed_{seed:03d}" / f"history_midpoint__{policy}.npz"
                with np.load(predictions, allow_pickle=False) as archive:
                    logits = archive["native_logits"]
                assert logits.shape == (4608, 32, 2, 11)
                summary, details = independent_counts(logits, selection)
                summary.update(seed=seed, policy=policy)
                exact(summary, saved_rows[(seed, policy)], "independent per-seed policy counts")
                for key, value in details.items():
                    name = f"{policy}__{key}"
                    expected_keys.add(name)
                    assert value.dtype == evidence[name].dtype and value.shape == evidence[name].shape
                    np.testing.assert_array_equal(value, evidence[name], err_msg=f"seed{seed}/{name}")
                summaries.append(summary)
                audited.append(dict(seed=seed, policy=policy, passed=True, requests=len(selection["identity"]),
                    native_logits_sha256=sha(predictions), evidence_sha256=sha(evidence_path)))
            assert set(evidence.files) == expected_keys
    pooled = {}
    for policy in POLICIES:
        rows = [r for r in summaries if r["policy"] == policy]
        total = dict(requests=sum(r["requests"] for r in rows), phases={},
            paired_conditional8_content={}, paired_native11_correct_FINISH={})
        for phase in PHASES:
            total["phases"][phase] = {key: sum(r["phases"][phase][key] for r in rows)
                                      for key in rows[0]["phases"][phase]}
        for key in ("paired_conditional8_content", "paired_native11_correct_FINISH"):
            counter = Counter()
            for row in rows:
                counter.update(row[key])
            total[key] = {k: counter[k] for k in rows[0][key]}
        exact(total, saved["pooled_by_policy"][policy], "independent pooled counts")
        assert total["requests"] == 4057
        pooled[policy] = total
    with (PHASE / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 12
    for row in csv_rows:
        expected = saved_rows[(int(row["seed"]), row["policy"])]
        for key, value in row.items():
            if key == "policy":
                assert value == expected[key]
            elif key in ("seed", "requests"):
                assert int(value) == expected[key]
            elif key.startswith("paired_content__"):
                assert int(value) == expected["paired_conditional8_content"][key.split("__")[1]]
            else:
                phase, field = key.split("__")
                assert int(value) == expected["phases"][phase][field]
    checked()
    write_new(OUT / "audit.json", dict(format="teacher-timing-independent-fixed-phase-posthoc-audit-v1",
        posthoc=True, passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), audit_plan_sha256=sha(PLAN), scope=plan["scope"],
        fixed_requests=4057, policy_phase_rows_checked=32456, rows=audited, pooled_by_policy=pooled,
        result_sha256=sha(PHASE / "results.json"), interpretation=plan["interpretation"]))
    print(digest(dict(passed=True, audit_sha256=sha(OUT / "audit.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    args = parser.parse_args()
    declare() if args.command == "declare" else run()
