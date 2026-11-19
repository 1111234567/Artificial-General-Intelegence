"""Mocked stage and reporting checks. No formal search or training is run."""
from contextlib import nullcontext

import pytest

from experiments.seed_evolution import large_results, pipeline


@pytest.fixture
def mocked_pipeline(tmp_path, monkeypatch):
    runner = pipeline.runner
    calls = []
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "_activate", lambda: None)
    monkeypatch.setattr(runner, "_lock", lambda _: nullcontext())
    for name in ("search", "finals", "tests"):
        monkeypatch.setattr(runner, name, lambda workers, stage=name: calls.append((stage, workers)))

    def audit(root, workers, metadata_only):
        assert root == tmp_path
        calls.append(("audit", workers, metadata_only))
        return dict(full_audit_passed=True, candidates_provenance_checked=12096,
                    heldout_reports=36, heldout_ledgers_recounted=72, heldout_policy_replays=9)

    def results(root):
        assert root == tmp_path
        assert runner.base.read(root / "audit.json")["full_audit_passed"] is True
        calls.append(("results",))
        return {"output": "fixture"}

    monkeypatch.setattr(pipeline.large_audit, "audit_all", audit)
    monkeypatch.setattr(pipeline.large_results, "aggregate", results)
    return tmp_path, calls


def test_pipeline_waits_for_search_final_test_and_audit(mocked_pipeline):
    root, calls = mocked_pipeline
    pipeline.run(workers=2)
    assert calls == [("search", 2), ("finals", 6), ("tests", 6), ("audit", 9, False), ("results",)]
    completed = pipeline.runner.base.read(root / "pipeline-completion.json")
    assert completed["full_audit_passed"] is True
    assert completed["result"] == {"output": "fixture"}


def test_failed_test_stage_never_starts_audit_or_publication(mocked_pipeline, monkeypatch):
    root, calls = mocked_pipeline

    def fail_test(workers):
        calls.append(("tests", workers))
        raise ValueError("test worker failed")

    monkeypatch.setattr(pipeline.runner, "tests", fail_test)
    with pytest.raises(ValueError, match="test worker failed"):
        pipeline.run(workers=2)
    assert calls == [("search", 2), ("finals", 6), ("tests", 6)]
    assert pipeline.runner.base.read(root / "pipeline-failure.json")["stage"] == "test"
    assert not (root / "pipeline-completion.json").exists()
    assert not (root / "audit.json").exists()


def test_partial_audit_never_becomes_a_pipeline_completion(mocked_pipeline, monkeypatch):
    root, calls = mocked_pipeline
    monkeypatch.setattr(pipeline.large_audit, "audit_all",
                        lambda *args, **kwargs: {"full_audit_passed": False})
    with pytest.raises(ValueError, match="audit"):
        pipeline.run(workers=2)
    assert calls == [("search", 2), ("finals", 6), ("tests", 6)]
    assert pipeline.runner.base.read(root / "pipeline-failure.json")["stage"] == "audit"
    assert not (root / "pipeline-completion.json").exists()
    assert not (root / "audit.json").exists()


def test_results_refuse_to_publish_from_progress_audit(tmp_path):
    large_results.write_json(tmp_path / "audit.json", {"full_audit_passed": False})
    dest = tmp_path / "public"
    with pytest.raises(ValueError, match="complete independent audit"):
        large_results.aggregate(tmp_path, dest)
    assert not dest.exists()


@pytest.fixture
def audited_scores(tmp_path, monkeypatch):
    root, dest = tmp_path / "run", tmp_path / "public"
    monkeypatch.setattr(large_results, "ROOT", tmp_path)
    arms = ("evolution", "random")
    choices = {0: (((1, 1), (0, 9)), ((0, 1), (0, 9))),
               1: (((1, 5), (1, 5)), ((3, 5), (3, 5))),
               2: (((9, 10), (9, 10)), ((1, 10), (1, 10)))}
    config = dict(source_seeds=[0, 1, 2], test_streams=[2100, 2101])
    large_results.write_json(root / "protocol.json", dict(config=config))
    winners, completed, history, records = [], [], [], {}
    for rep, pair in choices.items():
        records[str(rep)] = {}
        for arm, counts in zip(arms, pair):
            cid = f"{arm}_g31_i00"
            winner = dict(rep=rep, arm=arm, id=cid)
            winners.append(winner)
            streams = []
            for stream_id, (success, n) in zip(config["test_streams"], counts):
                group = dict(n=n, strict=success, translation_correct=success)
                streams.append(dict(stream_id=stream_id,
                    groups=dict(all=group, A_conflict=group, B_conflict=group),
                    probe=dict(idle_actions=0, idle_opportunities=1)))
            fitness = sum(success / n for success, n in counts) / 2
            records[str(rep)][cid] = dict(fitness=fitness)
            history.append(dict(rep=rep, arm=arm, generation=31, new_ids=[cid],
                                evaluated_ids=[cid], best_id=cid, best_fitness=fitness))
            reports = []
            for step, condition in large_results.CONDITIONS:
                name = f"{step:03d}_{condition}"
                path = root / "final" / f"rep_{rep}" / arm / "evaluations" / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                archive = path.with_suffix(".pt")
                archive.write_bytes(f"fixture {rep} {arm} {name}".encode())
                large_results.write_json(path, dict(winner=winner, step=step, condition=condition,
                    streams=streams, fitness=fitness, source_seed=rep,
                    archives_sha256=large_results.sha(archive)))
                reports.append(dict(name=name, fitness=fitness, sha256=large_results.sha(path)))
            row = dict(winner=winner, conditions=reports)
            completed.append(row)
            large_results.write_json(path.parent / "completion.json", row)
    large_results.write_json(root / "search.json", dict(records=records, history=history, winners=winners,
                                                        unique_candidates=12096))
    large_results.write_json(root / "final-freeze.json", dict(all_final_training_complete_before_test=True,
                                                             models=[dict(winner=w) for w in winners]))
    large_results.write_json(root / "test-completion.json", dict(models=completed))
    audit = dict(full_audit_passed=True, candidates_provenance_checked=12096, heldout_reports=36,
                 heldout_ledgers_recounted=72, heldout_policy_replays=9)
    for key, name in (("protocol", "protocol"), ("search", "search"),
                      ("final_freeze", "final-freeze"), ("test_completion", "test-completion")):
        audit[f"{key}_sha256"] = large_results.sha(root / f"{name}.json")
    large_results.write_json(root / "audit.json", audit)
    return root, dest


def test_results_average_stream_rates_and_three_repeats(audited_scores):
    root, dest = audited_scores
    result = large_results.aggregate(root, dest)
    first = result["comparisons"][0]
    assert first["evolution_mean"] == pytest.approx((0.5 + 0.2 + 0.9) / 3)
    assert first["random_mean"] == pytest.approx((0 + 0.6 + 0.1) / 3)
    assert first["paired_differences"] == pytest.approx([0.5, -0.4, 0.8])
    evidence = large_results.read(dest / "results.json")
    assert evidence["tests"][0]["all_strict"] == pytest.approx(0.5)
    assert evidence["tests"][0]["A_conflict"] == pytest.approx(0.5)


@pytest.mark.parametrize("relative, message", [
    ("final/rep_0/evolution/evaluations/000_own.json", "audited test score changed"),
    ("final/rep_0/evolution/evaluations/000_own.pt", "audited trajectory changed"),
    ("search.json", "audited source changed"),
    ("test-completion.json", "audited source changed"),
])
def test_results_refuse_an_artifact_changed_after_audit(audited_scores, relative, message):
    root, dest = audited_scores
    path = root / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        large_results.aggregate(root, dest)
    assert not dest.exists()
