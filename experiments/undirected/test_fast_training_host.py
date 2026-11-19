import json

import torch

from . import local_final_comparison
from . import local_rule_evaluator
from . import local_rule_run as run
from .fast_training_host import _ProgressSummary, installed


def _metrics(score):
    return {"selection_score": score, "strict": 0.25,
            "first_return_worst_context": 0.125, "first_return_mean": 0.1875}


def test_summary_preserves_ties_exact_means_and_optional_fields():
    def row(i, island, value, **extra):
        metrics = {**_metrics(0.25), "cancellation": value, "bool_is_not_numeric": True, **extra}
        return {"id": i, "island": island, **metrics, "metrics": metrics}

    rows = [row(0, 0, 1e16, separate=2), row(1, 0, 1, separate=4), row(2, 1, -1e16, separate=False),
            row(3, 1, 0, sparsity_bonus=0.1), row(4, 0, 0, separate=6)]
    replay = _ProgressSummary(run.rank)
    live = []
    for item in rows:
        live.append(item)
        assert replay(live, 3) == run._summary(live, 3)
    summary = replay(live, 3)
    assert summary["champion"]["id"] == 3
    assert summary["islands"]["0"]["champion"]["id"] == 0
    assert summary["mean"]["cancellation"] == 0.2
    assert "separate" not in summary["mean"]
    assert "bool_is_not_numeric" not in summary["mean"]
    assert summary["islands"]["0"]["mean"]["separate"] == 4.0
    assert replay(rows[:2], 3) == run._summary(rows[:2], 3)


def test_fast_reporting_preserves_sealed_artifacts_after_resume(tmp_path, monkeypatch):
    def plan(config, generation, previous, warm):
        return [{"id": f"g{generation}-{i}", "island": i, "seed": i} for i in range(config["population"])]

    def select(config, generation, rows, previous, warm):
        return {"champions": [{"row": row, "genome": row["seed"]} for row in rows],
                "islands": [[row] for row in rows], "genomes": {row["id"]: row["seed"] for row in rows},
                "perturbation_summary": {"paired": []}}

    class Evaluate:
        def __init__(self, device, kind):
            assert device == "cpu"

        def __call__(self, genomes, **settings):
            return [_metrics(genome / 10) for genome in genomes]

    monkeypatch.setattr(run.genetics, "plan", plan)
    monkeypatch.setattr(run.genetics, "candidate", lambda row, *_: row["seed"])
    monkeypatch.setattr(run.genetics, "select", select)
    monkeypatch.setattr(run, "_audit", lambda *_: None)
    monkeypatch.setattr(local_rule_evaluator, "Evaluation", Evaluate)
    monkeypatch.setattr(local_final_comparison, "compare_final", lambda *_: None)

    config = {"population": 4, "generations": 2, "islands": 4, "device": "cpu", "seed": 3,
              "development_seed_offset": 4, "batch_size": 2, "eval_batch_size": 1,
              "actors": 8, "visits": 16, "final_audit_seeds": 1}
    warm = {"descriptors": [], "genomes": {}}
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    run.train(reference, config, warm)
    original_summary, original_write = run._summary, run._write
    with installed():
        run.train(resumed, config, warm, batch_limit=3)
    assert run._summary is original_summary
    assert run._write is original_write
    progress = json.loads((resumed / "progress.json").read_bytes())
    assert (progress["state"], progress["current_arm"], progress["generation_completed"]) == ("paused", "local", 2)
    assert (resumed / "arms/control/generations/0001/summary.json").is_file()
    with installed():
        run.train(resumed, config, warm)
    for original in reference.rglob("*"):
        if original.is_file() and original.name not in ("process.json", "progress.json"):
            candidate = resumed / original.relative_to(reference)
            if original.suffix == ".pt":
                assert torch.load(candidate, weights_only=True) == torch.load(original, weights_only=True)
            else:
                assert candidate.read_bytes() == original.read_bytes()
