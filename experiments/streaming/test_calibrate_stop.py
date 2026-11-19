"""Small CPU checks for the separate, halt-only calibration stage."""
import json
import math
from pathlib import Path
import sys

import pytest
import torch

from experiments.streaming import calibrate_stop as calibration
from experiments.streaming.cache import CacheAgent
from experiments.streaming.run_stream import hashes


@pytest.fixture(autouse=True)
def cpu_rng():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(73)
        yield


def checkpoint(path, **overrides):
    config = dict(contexts=3, vocab=8, slots=8, hidden=8, max_reads=4,
                  width=12, read_dim=16, nodes=4, batch=2, eval_batch=1,
                  queries=1, cost=.02, device="cpu", halt_loss="mixture")
    model = CacheAgent(**{key: config[key] for key in
                         ("contexts", "vocab", "slots", "hidden", "max_reads", "width", "read_dim")})
    with torch.no_grad():
        model.key.weight.zero_()
        model.query.weight.zero_()
        width = config["contexts"] + config["vocab"]
        model.key.weight[:width] = torch.eye(width)
        model.query.weight[:width] = torch.eye(width)
        model.log_scale.fill_(math.log(100))
        model.halt.weight.zero_()
        model.halt.bias.zero_()
    payload = dict(model=model.state_dict(), config=config, seed=0, policy="mru", source=hashes())
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def arguments(monkeypatch, input_dir, output_dir):
    monkeypatch.setattr(sys, "argv", ["calibrate_stop.py", "--input", str(input_dir),
                        "--output", str(output_dir), "--seeds", "0", "--steps", "1",
                        "--lr", ".02", "--device", "cpu"])


@pytest.mark.parametrize("name", ["cache.py", "stream.py"])
def test_checkpoint_rejects_different_reader_or_task_source(tmp_path, name):
    path = tmp_path / "mru-0.pt"
    source = hashes()
    source[name] = "different implementation"
    checkpoint(path, source=source)
    with pytest.raises(ValueError, match="source mismatch"):
        calibration.model_from_checkpoint(path, "cpu")


@pytest.mark.parametrize("field,value", [("policy", "fifo"), ("seed", 1)])
def test_main_rejects_wrong_checkpoint_identity(tmp_path, monkeypatch, field, value):
    input_dir = tmp_path / "input"
    checkpoint(input_dir / "mru-0.pt", **{field: value})
    arguments(monkeypatch, input_dir, tmp_path / "output")
    with pytest.raises(ValueError, match="identity mismatch"):
        calibration.main()


def test_frozen_hash_excludes_only_halt_parameters():
    model = CacheAgent(hidden=8, width=12)
    initial = calibration.weights_hash(model, exclude_halt=True)
    full = calibration.weights_hash(model)
    with torch.no_grad():
        model.halt.bias.add_(1)
    assert calibration.weights_hash(model, exclude_halt=True) == initial
    assert calibration.weights_hash(model) != full
    with torch.no_grad():
        model.query.weight[0, 0].add_(1)
    assert calibration.weights_hash(model, exclude_halt=True) != initial


def test_actual_calibration_changes_only_halt_and_ignores_depth_phase(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    original = checkpoint(input_dir / "mru-0.pt")
    created_models, evaluation_seeds = [], []
    model_type = calibration.CacheAgent
    evaluate = calibration.evaluate

    def capture_model(**kwargs):
        model = model_type(**kwargs)
        created_models.append(model)
        return model

    def small_evaluation(model, config, policy, seed, **kwargs):
        evaluation_seeds.append(seed)
        # Exercise real hard evaluation with a tiny CPU workload.
        kwargs.update(cycles=1, noise=0)
        return evaluate(model, config, policy, seed, **kwargs)

    monkeypatch.setattr(calibration, "CacheAgent", capture_model)
    monkeypatch.setattr(calibration, "evaluate", small_evaluation)
    first = tmp_path / "original_labels"
    arguments(monkeypatch, input_dir, first)
    calibration.main()

    make_cycle = calibration.cycle

    def poisoned_labels(*args, **kwargs):
        events = make_cycle(*args, **kwargs)
        for event in events:
            if event.kind == "query":
                event.depth = event.depth + 137
                event.phase = "forbidden_label_poison"
        return events

    monkeypatch.setattr(calibration, "cycle", poisoned_labels)
    second = tmp_path / "poisoned_depth_phase"
    arguments(monkeypatch, input_dir, second)
    calibration.main()

    a = torch.load(first / "mru-0.pt", weights_only=False)
    b = torch.load(second / "mru-0.pt", weights_only=False)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    changed = [name for name in a["model"] if not torch.equal(original["model"][name], a["model"][name])]
    assert changed and all(name.startswith("halt.") for name in changed)
    assert a["config"]["halt_loss"] == b["config"]["halt_loss"] == "expected"
    for model in created_models:
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3
        for name, parameter in model.named_parameters():
            if name.startswith("halt."):
                assert parameter.requires_grad and parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            else:
                assert not parameter.requires_grad and parameter.grad is None
    assert set(evaluation_seeds) == {2700000, 3700000}
    for output in (first, second):
        result = json.loads((output / "mru-0.json").read_text(encoding="utf-8"))
        assert result["non_halt_hash_before"] == result["non_halt_hash_after"]
        assert result["trainable_parameters"] == 3
        assert result["calibration"]["steps"] == 1
        assert "before_calibration" in result["heldout"]


def test_calibration_refuses_existing_output(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    checkpoint(input_dir / "mru-0.pt")
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    arguments(monkeypatch, input_dir, output_dir)
    with pytest.raises(ValueError, match="nonempty"):
        calibration.main()
