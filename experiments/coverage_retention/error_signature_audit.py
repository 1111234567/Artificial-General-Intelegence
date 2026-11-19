"""Independent decoder of the declared, existing-trajectory error signatures.

All mature test-issued requests are reconstructed from external command and
maturity arrays before any classified output rows are read. No inference,
training, model, evaluator or signature-classifier routine is called.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.action_split.eval_audit import sha, read, digest
from experiments.experience_transfer.train_transfer import SEEDS
from experiments.remaining_time.eval_audit import exact, write_new

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/coverage-retention/error-signature-v1"
OUT = ROOT / "data/coverage-retention/error-signature-audit-v1"
PLAN = OUT / "audit_plan.json"
CATEGORIES = ("correct_current_rule", "other_rule_substitution",
              "correct_version_other_payload", "wrong_version", "no_attempt")
GROUPS = ("A_conflict", "B_conflict", "shared")


def independent_records(facts, seed, warm=512):
    """Command-indexed scan with direct action decoding and literal rule maps."""
    actions = facts["actions"]
    assert actions.dtype == np.int64 and ((actions >= 0) & (actions <= 10)).all()
    now = facts["now"][:, None, None]
    world_ticks = np.rint(facts["now"]/.25).astype(np.int64)
    np.testing.assert_array_equal(facts["now"], world_ticks*.25)
    if "world_tick" in facts:
        np.testing.assert_array_equal(facts["world_tick"], world_ticks)
    start = (actions == 1) | (actions == 2)
    finish = actions >= 3
    decoded_version = np.where(start, actions-1, (actions-3)//4)
    legal = (facts["active"] & facts["started"] & (now >= facts["ready"]-1e-6)
             & (now <= facts["deadline"]+1e-6))
    accepted_start = start & facts["active"] & ~facts["started"] & (decoded_version == facts["version"])
    accepted_finish = (finish & legal & (decoded_version == facts["version"])
                       & ((actions-3)%4 == facts["payload"]))
    receipt = np.where(actions == 0, 0, np.where(accepted_start | accepted_finish, 1, -1))
    np.testing.assert_array_equal(receipt, facts["receipts"])
    matured = {}
    for t, lane, job in np.argwhere(facts["mature"]):
        key = int(lane), int(job), int(facts["mature_id"][t, lane, job])
        assert key not in matured
        matured[key] = int(t)
    spec = SEEDS[seed]
    records = []
    commands = np.argwhere((facts["event_kind"] == 1) | (facts["event_kind"] == 2))
    seen = set()
    for issued, lane, job in commands:
        issued, lane, job = int(issued), int(lane), int(job)
        rid = int(facts["event_id"][issued, lane, job])
        key = lane, job, rid
        assert key not in seen
        seen.add(key)
        if issued < warm or not facts["event_eligible"][issued, lane, job] or key not in matured:
            continue
        mature_tick = matured[key]
        assert rid == 2*int(world_ticks[issued])+job
        cue, raw = int(facts["event_family"][issued, lane, job]), int(facts["event_raw"][issued, lane, job])
        assert cue in (spec["history_cue"], spec["B_cue"])
        family = "A" if cue == spec["history_cue"] else "B"
        a, b = int(spec["A"][raw]), int(spec["B"][raw])
        target, other = (a, b) if family == "A" else (b, a)
        conflict = a != b
        assert conflict == (raw in spec["conflict_raw"])
        group = family+"_conflict" if conflict else "shared"
        version = int(facts["event_version"][issued, lane, job])
        assert bool(facts["eligible"][issued, lane, job])
        assert int(facts["mature_family"][mature_tick, lane, job]) == cue
        assert facts["mature_duration"][mature_tick, lane, job] == facts["duration"][issued, lane, job]
        assert facts["mature_issued"][mature_tick, lane, job] == facts["issued_time"][issued, lane, job] == facts["now"][issued]
        same = facts["rid"][:, lane, job] == rid
        rows = np.flatnonzero(same)
        assert len(rows) and rows[0] == issued
        np.testing.assert_array_equal(facts["raw"][rows, lane, job], raw)
        np.testing.assert_array_equal(facts["family"][rows, lane, job], cue)
        np.testing.assert_array_equal(facts["version"][rows, lane, job], version)
        np.testing.assert_array_equal(facts["payload"][rows, lane, job], target)
        attempted = np.flatnonzero(same & legal[:, lane, job] & finish[:, lane, job])
        begins = np.flatnonzero(same & accepted_start[:, lane, job])
        windows = np.flatnonzero(same & legal[:, lane, job])
        active = np.flatnonzero(same & facts["active"][:, lane, job])
        rejected = active[receipt[active, lane, job] == -1]
        successful = np.flatnonzero(same & accepted_finish[:, lane, job])
        processing = active[facts["started"][active, lane, job] &
                            (facts["now"][active] < facts["ready"][active, lane, job]-1e-6)]
        early = processing[finish[processing, lane, job]]
        ordinary = bool(len(successful))
        strict = ordinary and not bool(np.any(rejected < successful[0]))
        assert ordinary == bool(facts["mature_done"][mature_tick, lane, job])
        assert strict == bool(facts["mature_strict"][mature_tick, lane, job])
        first = int(attempted[0]) if len(attempted) else None
        chosen = int(actions[first, lane, job]) if first is not None else None
        cv = (chosen-3)//4 if chosen is not None else None
        cp = (chosen-3)%4 if chosen is not None else None
        if first is None:
            category = "no_attempt"
        elif cv != version:
            category = "wrong_version"
        elif cp == target:
            category = "correct_current_rule"
        elif conflict and cp == other:
            category = "other_rule_substitution"
        else:
            category = "correct_version_other_payload"
        if first is not None:
            assert int(receipt[first, lane, job]) == (1 if category == "correct_current_rule" else -1)
        assert not strict or category == "correct_current_rule"
        assert category != "correct_current_rule" or ordinary
        assert category != "no_attempt" or not ordinary
        records.append(dict(actor=int(facts["actor_ids"][lane]), lane=lane, job=job, rid=rid,
            issued=issued, mature_tick=mature_tick, family=family, cue=cue, raw=raw, conflict=conflict,
            group=group, version=version, target_payload=target, other_rule_payload=other,
            A_target_payload=a, B_target_payload=b, target_native_action=3+4*version+target,
            other_rule_native_action=3+4*version+other,
            paired_request_id=f"{seed}:{int(facts['actor_ids'][lane])}:{job}:{rid}",
            other_rule_applicability="conflict" if conflict else "other_rule_notapplicable",
            issued_world_tick=int(world_ticks[issued]), issued_native_time=float(facts["now"][issued]),
            mature_world_tick=int(world_ticks[mature_tick]), mature_native_time=float(facts["now"][mature_tick]),
            legal_ticks_count=len(windows), first_legal_opportunity_tick=int(windows[0]) if len(windows) else None,
            first_legal_finish_tick=first,
            first_legal_finish_world_tick=int(world_ticks[first]) if first is not None else None,
            first_legal_finish_native_time=float(facts["now"][first]) if first is not None else None,
            first_legal_finish_action=chosen, first_legal_finish_version=cv, first_legal_finish_payload=cp,
            first_legal_finish_receipt=int(receipt[first, lane, job]) if first is not None else None,
            category=category, no_attempt_reason=(None if first is not None else "no_begin" if not len(begins)
                else "started_no_window" if not len(windows) else "legal_no_finish"), no_begin=not bool(len(begins)),
            started_no_window=bool(len(begins)) and not bool(len(windows)),
            legal_no_finish=bool(len(windows)) and first is None, ordinary=ordinary, strict=strict,
            any_early=bool(len(early)), first_error_early=bool(len(rejected) and rejected[0] in early)))
    return records


def independent_groups(records):
    result = {}
    flags = ("ordinary", "strict", "no_begin", "started_no_window", "legal_no_finish", "any_early", "first_error_early")
    for group in GROUPS:
        rows = [record for record in records if record["group"] == group]
        categories = Counter(record["category"] for record in rows)
        n = len(rows)
        values = dict(matured=n, categories={key: categories[key] for key in CATEGORIES},
            category_rates={key: categories[key]/n if n else None for key in CATEGORIES},
            other_rule_applicability="other_rule_notapplicable" if group == "shared" else "conflict")
        values.update({key: sum(int(row[key]) for row in rows) for key in flags})
        values["strict_failure_categories"] = {key: sum(row["category"] == key and not row["strict"] for row in rows) for key in CATEGORIES}
        values["ordinary_failure_categories"] = {key: sum(row["category"] == key and not row["ordinary"] for row in rows) for key in CATEGORIES}
        assert sum(values["categories"].values()) == n
        assert sum(values[key] for key in ("no_begin", "started_no_window", "legal_no_finish")) == categories["no_attempt"]
        assert sum(values["strict_failure_categories"].values()) == n-values["strict"]
        assert group != "shared" or categories["other_rule_substitution"] == 0
        result[group] = values
    assert sum(value["matured"] for value in result.values()) == len(records)
    return result


def checked_amendment(protocol):
    """Verify the preserved failed parser and unchanged classification AST."""
    amendment_path = INPUT / "parser_amendment_001.json"
    amendment = read(amendment_path)
    failure = read(INPUT / "parser_failure_before_outputs.json")
    assert amendment["original_protocol_sha256"] == sha(INPUT / "protocol.json")
    assert amendment["categories_cohorts_and_estimands_changed"] is False
    assert amendment["classification_outputs_before_amendment"] == failure["classification_outputs"] == 0
    assert failure["parser_amendment_sha256"] == sha(amendment_path)
    archived = ROOT / failure["original_source_archive_path"]
    assert sha(archived) == failure["original_source_sha256"]
    source_name, new_hash = next(iter(amendment["sources"].items()))
    assert len(amendment["sources"]) == len(amendment["previous_sources"]) == 1
    assert amendment["previous_sources"][source_name] == protocol["sources"][source_name] == sha(archived)
    assert sha(ROOT / source_name) == new_hash
    old_tree, new_tree = ast.parse(archived.read_text(encoding="utf-8")), ast.parse((ROOT/source_name).read_text(encoding="utf-8"))
    def named(tree, name):
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    for name in ("run_specs", "category", "self_test", "summarize", "aggregate"):
        assert ast.dump(named(old_tree, name)) == ast.dump(named(new_tree, name)), f"amendment changed {name}"
    for name in ("CATEGORIES", "GROUPS", "POLICIES", "EPS"):
        def assignment(tree):
            return next(node for node in tree.body if isinstance(node, ast.Assign) and
                        any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.dump(assignment(old_tree)) == ast.dump(assignment(new_tree)), f"amendment changed {name}"
    # Normalize exactly the removed missing-column assertion and the added
    # amendment provenance field; all remaining classification syntax must match.
    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [item for item in node.keywords if item.arg != "parser_amendment_sha256"]
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            selected = [(op, value) for op, value in zip(node.ops, node.comparators)
                        if "event_payload" not in ast.dump(value)]
            node.ops, node.comparators = [item[0] for item in selected], [item[1] for item in selected]
            return node
    assert ast.dump(Normalize().visit(named(old_tree, "classify_run"))) == ast.dump(Normalize().visit(named(new_tree, "classify_run")))
    for name, expected in protocol["sources"].items():
        assert sha(ROOT/name) == amendment["sources"].get(name, expected), name
    for name, expected in protocol["inputs"].items():
        assert sha(ROOT/name) == expected, name
    return dict(passed=True, original_protocol_sha256=sha(INPUT / "protocol.json"),
        amendment_sha256=sha(amendment_path), original_source_sha256=sha(archived), amended_source_sha256=new_hash,
        unchanged_category_cohort_summary_functions_AST=True, unchanged_classifier_except_missing_column_and_provenance=True,
        failed_before_any_classification_output=True)


def declare():
    protocol = read(INPUT / "protocol.json")
    completion = read(INPUT / "completion.json")
    assert protocol["posthoc"] and completion["posthoc"] and completion["complete"]
    assert protocol["expected_runs"] == completion["runs"] == 15
    assert protocol["categories"] == list(CATEGORIES) and protocol["groups"] == list(GROUPS)
    assert protocol["policies"] == {"fresh": ["baseline", "control256", "withdraw256"], "live": ["control", "withdraw"]}
    assert protocol["conditions"] == ["long"] and protocol["warm"] == 512
    amendment = checked_amendment(protocol)
    assert completion["protocol_sha256"] == sha(INPUT / "protocol.json")
    assert completion["parser_amendment_sha256"] == amendment["amendment_sha256"]
    assert completion["summary_sha256"] == sha(INPUT / "summary.json")
    assert completion["paired_requests_sha256"] == sha(INPUT / "paired_requests.json")
    paths = [INPUT/name for name in ("protocol.json", "completion.json", "summary.json", "paired_requests.json",
        "parser_amendment_001.json", "parser_failure_before_outputs.json", "error_signature.pre_parser_amendment.py.txt")]
    for name, expected in completion["run_outputs"].items():
        assert sha(ROOT/name) == expected, name
        paths.append(ROOT/name)
    paths.extend(ROOT/name for name in protocol["inputs"])
    sources = [Path(__file__), ROOT/"experiments/experience_transfer/train_transfer.py",
               *(ROOT/name for name in protocol["sources"])]
    write_new(PLAN, dict(format="coverage-retention-independent-posthoc-error-signature-audit-plan-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), posthoc=True, runs=protocol["runs"],
        inputs={str(path.relative_to(ROOT)): sha(path) for path in dict.fromkeys(paths)},
        sources={str(path.relative_to(ROOT)): sha(path) for path in dict.fromkeys(sources)}, amendment_audit=amendment,
        scope="Independent complete NEW/REPLACE-to-maturity ledger on all15 archived runs, every native receipt, literal A/B map and first actual legal FINISH decoding; all full-denominator record/category/strict/ordinary cross-tabs, all seed/pooled/equal-seed summaries and paired request entries; explicit original parser failure/amendment and unchanged classification AST"))


def checked():
    plan = read(PLAN)
    for table in ("inputs", "sources"):
        for name, expected in plan[table].items():
            assert sha(ROOT/name) == expected, name
    checked_amendment(read(INPUT / "protocol.json"))
    return plan


def run():
    plan = checked()
    records_by_case, groups_by_case, reports = {}, {}, []
    for spec in plan["runs"]:
        stem = ROOT/spec["stem"]
        source = read(stem.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            facts = {key: archive[key] for key in archive.files if key not in ("obs", "final_h")}
        actual = independent_records(facts, spec["seed"])
        saved = read(ROOT/spec["output"])
        exact(actual, saved["requests"], f"{spec['output']}/all independent request records")
        groups = independent_groups(actual)
        exact(groups, saved["groups"], f"{spec['output']}/all groups")
        assert saved["facts_sha256"] == sha(stem.with_suffix(".npz")) == source["facts_sha256"]
        assert saved["report_sha256"] == sha(stem.with_suffix(".json"))
        assert saved["checkpoint_sha256"] == source["checkpoint_sha256"]
        assert saved["protocol_sha256"] == sha(INPUT/"protocol.json")
        assert saved["parser_amendment_sha256"] == plan["amendment_audit"]["amendment_sha256"]
        identity = lambda row: tuple(row[key] for key in ("actor", "lane", "job", "rid", "issued", "mature_tick"))
        assert list(map(identity, actual)) == list(map(identity, source["requests"]))
        for row, prior in zip(actual, source["requests"]):
            for key in ("family", "raw", "conflict", "ordinary", "strict", "no_begin", "started_no_window", "legal_no_finish", "any_early", "first_error_early"):
                exact(row[key], prior[key], f"original request ledger/{key}")
        key = spec["mode"], spec["seed"], spec["policy"]
        records_by_case[key], groups_by_case[key] = actual, groups
        reports.append(dict(mode=spec["mode"], seed=spec["seed"], policy=spec["policy"], passed=True,
            requests=len(actual), records_digest=digest(actual), groups=groups, output_sha256=sha(ROOT/spec["output"])))
        print(dict(mode=spec["mode"], seed=spec["seed"], policy=spec["policy"], error_signature_audit_passed=True), flush=True)
    summaries = {}
    policies = {"fresh": ("baseline", "control256", "withdraw256"), "live": ("control", "withdraw")}
    paired = []
    for mode, names in policies.items():
        summaries[mode] = {}
        for policy in names:
            groups = independent_groups([row for seed in range(3) for row in records_by_case[(mode, seed, policy)]])
            for group in GROUPS:
                groups[group]["equal_seed_category_rates"] = {category: float(np.mean([
                    groups_by_case[(mode, seed, policy)][group]["category_rates"][category] for seed in range(3)])) for category in CATEGORIES}
                for flag in ("strict", "ordinary"):
                    groups[group][f"equal_seed_{flag}_rate"] = float(np.mean([
                        groups_by_case[(mode, seed, policy)][group][flag]/groups_by_case[(mode, seed, policy)][group]["matured"] for seed in range(3)]))
            summaries[mode][policy] = dict(seeds={str(seed): groups_by_case[(mode, seed, policy)] for seed in range(3)}, pooled=groups)
        for seed in range(3):
            tables = {policy: {row["paired_request_id"]: row for row in records_by_case[(mode, seed, policy)]} for policy in names}
            ids = list(tables[names[0]])
            assert all(list(table) == ids for table in tables.values())
            for request_id in ids:
                anchor = tables[names[0]][request_id]
                for table in tables.values():
                    for key in ("actor", "lane", "job", "rid", "issued", "mature_tick", "family", "raw", "version", "group", "target_payload", "other_rule_payload"):
                        assert table[request_id][key] == anchor[key]
                paired.append(dict(mode=mode, seed=seed, paired_request_id=request_id, group=anchor["group"], family=anchor["family"], raw=anchor["raw"],
                    policies={policy: {key: table[request_id][key] for key in ("category", "ordinary", "strict", "first_legal_finish_tick", "first_legal_finish_action", "no_attempt_reason")}
                              for policy, table in tables.items()}))
    exact(summaries, read(INPUT/"summary.json")["groups"], "all seed/pooled/equal-seed category summaries")
    exact(paired, read(INPUT/"paired_requests.json")["requests"], "all paired request entries")
    checked()
    write_new(OUT/"audit.json", dict(format="coverage-retention-independent-posthoc-error-signature-audit-v1",
        passed=True, posthoc=True, created_utc=datetime.now(timezone.utc).isoformat(), runs=15,
        requests_checked=sum(row["requests"] for row in reports), paired_requests_checked=len(paired),
        audit_plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), amendment_audit=plan["amendment_audit"],
        scope=plan["scope"], cases=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    args = parser.parse_args()
    declare() if args.command == "declare" else run()
