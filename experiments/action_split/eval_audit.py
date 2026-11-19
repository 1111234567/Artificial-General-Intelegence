"""Independent streaming state-machine reconstruction of all native eval metrics.

This scorer consumes archived pre-action facts, actions and receipts directly.
It never calls evaluate.summarize_facts or any model or evaluation routine.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.experience_transfer.train_transfer import SEEDS

EVAL = ROOT / "data/action-split/eval-v1"
OUT = ROOT / "data/action-split/eval-audit-v1"
EPS = 1e-6


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def score(facts, seed, warm):
    actions = facts["actions"]
    assert bool(((actions >= 0) & (actions <= 10)).all())
    begin, finish = np.isin(actions, (1, 2)), actions >= 3
    version = np.where(begin, actions - 1, (actions - 3) // 4)
    now = facts["now"][:, None, None]
    legal = facts["active"] & facts["started"] & (now >= facts["ready"] - EPS) & (now <= facts["deadline"] + EPS)
    processing = facts["active"] & facts["started"] & (now < facts["ready"] - EPS)
    accepted_begin = begin & facts["active"] & ~facts["started"] & (version == facts["version"])
    accepted_finish = finish & legal & (version == facts["version"]) & ((actions - 3) % 4 == facts["payload"])
    expected_receipts = np.where(actions == 0, 0, np.where(accepted_begin | accepted_finish, 1, -1))
    np.testing.assert_array_equal(expected_receipts, facts["receipts"])
    records = {}
    # Records retain insertion order, which is external issue tick/lane/job
    # order. Completed and replaced records stay available for native maturity.
    for tick in range(actions.shape[0]):
        for lane, job in np.argwhere(np.isin(facts["event_kind"][tick], (1, 2))):
            lane, job = int(lane), int(job)
            rid = int(facts["event_id"][tick, lane, job])
            key = (lane, job, rid)
            assert key not in records
            cue = int(facts["family"][tick, lane, job])
            assert cue in (SEEDS[seed]["history_cue"], SEEDS[seed]["B_cue"])
            raw = int(facts["raw"][tick, lane, job])
            records[key] = dict(actor=int(facts["actor_ids"][lane]), lane=lane, job=job,
                rid=rid, issued=tick, family="A" if cue == SEEDS[seed]["history_cue"] else "B",
                raw=raw, conflict=raw in SEEDS[seed]["conflict_raw"],
                eligible=bool(facts["event_eligible"][tick, lane, job]),
                mature_tick=None, ordinary=False, strict=False, started=False,
                legal_window=False, first_legal_attempt="none", first_legal_delay=None,
                any_early=False, first_error_early=False, rejected=False)
        for lane, job in np.argwhere(facts["active"][tick]):
            lane, job = int(lane), int(job)
            key = (lane, job, int(facts["rid"][tick, lane, job]))
            request = records[key]
            if accepted_begin[tick, lane, job]:
                request["started"] = True
            if legal[tick, lane, job]:
                request["legal_window"] = True
                if finish[tick, lane, job] and request["first_legal_attempt"] == "none":
                    request["first_legal_attempt"] = "correct" if accepted_finish[tick, lane, job] else "wrong"
                    request["first_legal_delay"] = float(facts["now"][tick] - facts["ready"][tick, lane, job])
            early = bool(processing[tick, lane, job] and finish[tick, lane, job])
            request["any_early"] |= early
            if expected_receipts[tick, lane, job] == -1:
                if not request["rejected"]:
                    request["first_error_early"] = early
                request["rejected"] = True
            if accepted_finish[tick, lane, job]:
                assert not request["ordinary"]
                request["ordinary"] = True
                request["strict"] = not request["rejected"]
        for lane, job in np.argwhere(facts["mature"][tick]):
            lane, job = int(lane), int(job)
            key = (lane, job, int(facts["mature_id"][tick, lane, job]))
            request = records[key]
            assert request["eligible"] and request["mature_tick"] is None
            request["mature_tick"] = tick
            assert request["ordinary"] == bool(facts["mature_done"][tick, lane, job])
            assert request["strict"] == bool(facts["mature_strict"][tick, lane, job])
    requests, coverage = [], Counter()
    for request in records.values():
        if request["issued"] < warm:
            coverage["prewarm_commands"] += 1
            continue
        coverage["postwarm_commands"] += 1
        if not request["eligible"]:
            coverage["ineligible"] += 1
            continue
        coverage["eligible_issued"] += 1
        if request["mature_tick"] is None:
            coverage["eligible_censored"] += 1
            continue
        coverage["matured"] += 1
        report = {key: value for key, value in request.items()
                  if key not in ("eligible", "rejected")}
        report["no_begin"] = not request["started"]
        report["started_no_window"] = request["started"] and not request["legal_window"]
        report["legal_no_finish"] = request["legal_window"] and request["first_legal_attempt"] == "none"
        requests.append(report)
    groups = {}
    selected = {"all": requests,
                "family_A": [r for r in requests if r["family"] == "A"],
                "family_A_conflict": [r for r in requests if r["family"] == "A" and r["conflict"]],
                "family_B": [r for r in requests if r["family"] == "B"],
                "family_B_conflict": [r for r in requests if r["family"] == "B" and r["conflict"]],
                "shared": [r for r in requests if not r["conflict"]]}
    flags = ("ordinary", "strict", "started", "legal_window", "no_begin", "started_no_window",
             "legal_no_finish", "any_early", "first_error_early")
    for name, rows in selected.items():
        group = {"matured": len(rows)}
        for flag in flags:
            group[flag] = sum(int(row[flag]) for row in rows)
        for answer in ("correct", "wrong", "none"):
            group["first_legal_" + answer] = sum(row["first_legal_attempt"] == answer for row in rows)
        delays = [row["first_legal_delay"] for row in rows if row["first_legal_delay"] is not None]
        group["first_legal_delay_sum"], group["first_legal_delay_n"] = sum(delays), len(delays)
        assert group["no_begin"] + group["started_no_window"] + group["legal_window"] == len(rows)
        assert group["legal_no_finish"] + group["first_legal_correct"] + group["first_legal_wrong"] == group["legal_window"]
        groups[name] = group
    idle = ~facts["active"][warm:]
    return dict(coverage=dict(coverage), groups=groups, requests=requests,
                idle_opportunities=int(np.count_nonzero(idle)),
                idle_actions=int(np.count_nonzero(idle & (actions[warm:] != 0))))


def main():
    protocol, frozen, completion = [read(EVAL / name) for name in ("protocol.json", "freeze.json", "completion.json")]
    protocol_hash, freeze_hash = sha(EVAL / "protocol.json"), sha(EVAL / "freeze.json")
    assert protocol_hash == frozen["protocol_sha256"]
    for path, expected in protocol["sources"].items():
        assert sha(ROOT / path) == expected, path
    assert completion["complete"] and completion["runs"] == 30
    results = []
    for seed in protocol["seeds"]:
        for policy in protocol["policies"]:
            for condition in protocol["conditions"]:
                base = EVAL / f"seed_{seed:03d}" / f"{policy}_{condition}"
                report_path, facts_path = base.with_suffix(".json"), base.with_suffix(".npz")
                report = read(report_path)
                assert sha(report_path) == completion["files"][str(report_path.relative_to(ROOT))]
                assert sha(facts_path) == report["facts_sha256"]
                assert report["protocol_sha256"] == protocol_hash and report["freeze_sha256"] == freeze_hash
                assert sha(ROOT / report["checkpoint_path"]) == report["checkpoint_sha256"]
                assert frozen["checkpoints"][report["checkpoint_path"]] == report["checkpoint_sha256"]
                with np.load(facts_path, allow_pickle=False) as archive:
                    facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
                actual = score(facts, seed, protocol["warm"])
                for key in ("coverage", "groups", "requests", "idle_opportunities", "idle_actions"):
                    assert actual[key] == report[key], f"{seed}/{policy}/{condition}/{key}"
                results.append(dict(seed=seed, policy=policy, condition=condition, passed=True,
                    facts_sha256=sha(facts_path), report_sha256=sha(report_path),
                    requests_digest=digest(actual["requests"]), request_count=len(actual["requests"]),
                    coverage=actual["coverage"], groups=actual["groups"],
                    idle_opportunities=actual["idle_opportunities"], idle_actions=actual["idle_actions"]))
                print(json.dumps(dict(seed=seed, policy=policy, condition=condition, passed=True)), flush=True)
    result = dict(format="action-split-independent-evaluation-metrics-v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        evaluation_protocol_sha256=protocol_hash, evaluation_freeze_sha256=freeze_hash,
        completion_sha256=sha(EVAL / "completion.json"),
        scope="All30 raw native action receipts; tick-wise independent request state machines; all request records/groups/idle and native maturity ordinary/strict equality. Does not call evaluate.summarize_facts.",
        runs=results)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.json").open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(passed=True, path=str(OUT / "metrics.json"), sha256=sha(OUT / "metrics.json"))))


if __name__ == "__main__":
    main()
