"""Small metadata fixtures for expanded search audits, without training."""
from copy import deepcopy
import json

import pytest

from experiments.learned_binding.audit import AuditError
from experiments.seed_evolution import audit, large_audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def install_candidate(root, descriptor, fitness):
    folder = audit.candidate_directory(root, descriptor["rep"], descriptor["id"])
    record = dict(descriptor, fitness=fitness, genome_hash=descriptor["id"])
    write(folder / "candidate.json", record)
    return record


def commit(root, plan, history):
    generation = plan["generation"]
    plan_path = root / "plans" / f"generation_{generation:02d}.json"
    write(plan_path, plan)
    write(root / f"generation_{generation}.json", dict(
        generation=generation, history=history,
        candidate_ids=[{key: candidate[key] for key in ("rep", "id")} for candidate in plan["candidates"]],
        plan_sha256=audit.common.sha(plan_path),
    ))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    config = dict(source_seeds=[0], population=3, parents=2, generations=2, sigma=.15)
    protocol = dict(unique_candidates=9, budget_per_arm_per_repeat=6)
    executors = {0: object()}
    ledger = large_audit.SearchLedger(tmp_path, config, executors)
    plan = ledger.expected_plan(0)
    ids = plan["entries"][0]["new_ids"]
    first_scores = (0.1, 0.3, 0.3)
    records = {candidate["id"]: install_candidate(tmp_path, candidate, score)
               for candidate, score in zip(plan["candidates"], first_scores)}
    history = [dict(rep=0, arm=arm, generation=0, top8_before=[], new_ids=ids,
                    evaluated_ids=ids, best_id=ids[1], best_fitness=0.3)
               for arm in large_audit.ARMS]
    commit(tmp_path, plan, history)
    ledger.records = {"0": records}
    ledger.cumulative = {(0, arm): ids.copy() for arm in large_audit.ARMS}

    def audit_artifact(_root, record, _config, _executor, genomes):
        genomes[record["id"]] = object()
        return ["paired world"]

    monkeypatch.setattr(audit, "audit_candidate", audit_artifact)
    monkeypatch.setattr(large_audit, "verify_attempt", lambda *args: None)
    return tmp_path, config, protocol, executors, ledger, history


def test_metadata_checks_partial_epoch_without_certifying_full_run(staged):
    root, config, protocol, executors, ledger, _ = staged
    active = ledger.expected_plan(1)
    plan_path = root / "plans" / "generation_01.json"
    write(plan_path, active)
    install_candidate(root, active["candidates"][2], 0.4)
    incomplete = active["candidates"][3]
    folder = audit.candidate_directory(root, incomplete["rep"], incomplete["id"])
    folder.mkdir(parents=True)
    (folder / "initial.pt").write_bytes(b"uncommitted")
    status = large_audit.audit_search(root, protocol, config, executors, metadata_only=True)
    assert status["sealed_generations"] == 1
    assert status["active_generation_candidates"] == 1
    assert status["candidates_provenance_checked"] == 4
    assert status["committed_budget_by_repeat"] == {"0": {"evolution": 3, "random": 3}}
    assert not status["search_complete"]
    assert status["winners"] is None
    with pytest.raises(AuditError, match="expanded search is incomplete"):
        large_audit.audit_search(root, protocol, config, executors, metadata_only=False)

    altered = deepcopy(active)
    altered["candidates"][2]["parent_id"] = "random_g01_i01"
    write(plan_path, altered)
    with pytest.raises(AuditError, match="independent generation plan"):
        large_audit.audit_search(root, protocol, config, executors, metadata_only=True)


def test_frozen_search_checks_history_budget_and_winner(staged):
    root, config, protocol, executors, ledger, first_history = staged
    plan = ledger.expected_plan(1)
    records = ledger.records["0"]
    for candidate, fitness in zip(plan["candidates"], (0.5, 0.4, 0.3, 0.2, 0.4, 0.6)):
        records[candidate["id"]] = install_candidate(root, candidate, fitness)
    history = []
    winners = []
    for entry, winning_index, best in zip(plan["entries"], (0, 2), (0.5, 0.6)):
        arm = entry["arm"]
        winner = entry["new_ids"][winning_index]
        history.append(dict(entry, evaluated_ids=ledger.cumulative[0, arm] + entry["new_ids"],
                            best_id=winner, best_fitness=best))
        winners.append(dict(rep=0, arm=arm, id=winner))
    commit(root, plan, history)
    summary = dict(history=first_history + history, records={"0": records}, winners=winners,
                   unique_candidates=9)
    write(root / "search.json", summary)
    status = large_audit.audit_search(root, protocol, config, executors, metadata_only=False)
    assert status["search_complete"] and status["winners"] == winners
    assert status["candidates_provenance_checked"] == 9
    assert status["committed_budget_by_repeat"] == {"0": {"evolution": 6, "random": 6}}

    summary["winners"][0] = dict(summary["winners"][0], id=plan["entries"][0]["new_ids"][1])
    write(root / "search.json", summary)
    with pytest.raises(AuditError, match="expanded search freeze"):
        large_audit.audit_search(root, protocol, config, executors, metadata_only=False)


def test_rejects_noncontiguous_sealed_generations(tmp_path):
    write(tmp_path / "generation_1.json", {})
    with pytest.raises(AuditError, match="noncontiguous generation commits"):
        large_audit.committed_generations(tmp_path, generations=32)


def test_progress_report_is_separate_from_full_audit(tmp_path, monkeypatch):
    protocol = dict(unique_candidates=12096, budget_per_arm_per_repeat=2048,
                    source_hashes={"experiments/seed_evolution/" + name: "" for name in
                                   ("core.py", "runner.py", "large_runner.py", "large-plan.md")})
    write(tmp_path / "protocol.json", protocol)
    config = dict(source_seeds=[0, 1, 2], generations=32)
    monkeypatch.setattr(large_audit, "validate_protocol", lambda _: config)
    monkeypatch.setattr(audit, "verify_sources", lambda _: {})
    monkeypatch.setattr(audit.common, "executor", lambda rep: rep)
    monkeypatch.setattr(large_audit.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(large_audit, "audit_search", lambda *args: {
        "candidates_provenance_checked": 192, "sealed_generations": 1, "search_complete": False})

    status = large_audit.audit_all(tmp_path, metadata_only=True)
    assert status["mode"] == "metadata_only" and not status["full_audit_passed"]
    assert status["candidates_provenance_checked"] == 192
    assert status["target_unique_candidates"] == 12096
    assert not (tmp_path / "audit.json").exists()
    with pytest.raises(AuditError, match="final-freeze.json"):
        large_audit.audit_all(tmp_path)


def test_recovery_manifest_must_match_committed_artifact_hashes(tmp_path):
    write(tmp_path / "attempt.json", dict(attempt=2))
    write(tmp_path / "training.json", dict(train_attempt=1, train_seconds=5, checkpoint_sha256="trained"))
    write(tmp_path / "evaluation-time.json", dict(eval_attempt=2, eval_seconds=2, score_sha256="scored"))
    record = dict(attempt=2, elapsed_seconds=3, train_attempt=1, train_seconds=5,
                  eval_attempt=2, eval_seconds=2, reused_artifacts=["initial.pt", "inner64.pt"],
                  artifacts={"inner64.pt": "trained", "score.json": "scored"})
    large_audit.verify_attempt(tmp_path, record)
    with pytest.raises(AuditError, match="candidate stage recovery differs"):
        large_audit.verify_attempt(tmp_path, dict(record, reused_artifacts=["initial.pt"]))
    with pytest.raises(AuditError, match="candidate stage hash differs"):
        large_audit.verify_attempt(tmp_path, dict(record, artifacts={"inner64.pt": "other", "score.json": "scored"}))
