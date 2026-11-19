"""Persistent-state boundaries, causal rollout, and reproducible evidence tests."""
from dataclasses import replace
import json

import pytest
import torch

from . import run
from .model import DynamicsModel
from .world import WorldBatch, generate_stream


class RecordingModel(torch.nn.Module):
    """A differentiable counter whose state makes accidental resets visible."""

    n_actions, width, input_size, kind, max_step = 11, 2, 48, "learned", .25

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(11))
        self.calls, self.initial_calls = [], 0

    def initial_state(self, batch_size, device=None):
        self.initial_calls += 1
        return torch.zeros(batch_size, self.width, device=device)

    def step(self, obs, state, dt, max_step=None):
        self.calls.append({"state": state.detach().clone(), "requires_grad": state.requires_grad,
                           "dt": dt, "max_step": max_step, "observation": obs.detach().clone()})
        state = state + 1 + self.bias[0] * 0
        logits = self.bias.view(1, 1, -1).expand(len(obs), 2, -1)
        return logits, state


class ScriptedOracle(RecordingModel):
    """Ceiling policy supplied a fixed script only inside this test fixture."""

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def initial_state(self, batch_size, device=None):
        self.tick = 0
        return super().initial_state(batch_size, device)

    def step(self, obs, state, dt, max_step=None):
        _, state = super().step(obs, state, dt, max_step)
        logits = obs.new_full((len(obs), 2, 11), -20)
        logits.scatter_(-1, self.targets[self.tick].to(obs.device)[..., None], 20)
        self.tick += 1
        return logits, state


def tiny_args(*extra):
    return run.parse_args(["--device", "cpu", "--steps", "4", "--updates", "5",
                           "--batch", "2", "--tbptt", "2", "--width", "4",
                           "--eval-steps", "8", "--eval-batch", "2",
                           "--log-every", "10", *extra])


def test_split_seeds_and_worlds_are_reproducible_and_disjoint():
    seeds = {split: [run.seed_for(17, split, i) for i in range(8)]
             for split in ("train", "validation", "test")}
    assert seeds == {split: [run.seed_for(17, split, i) for i in range(8)] for split in seeds}
    assert len(set(sum(seeds.values(), []))) == 24
    first = generate_stream(seeds["train"][0], 3, 64)
    second = generate_stream(seeds["train"][0], 3, 64)
    assert first.fingerprint == second.fingerprint
    assert first.case_ids == second.case_ids
    assert first.fingerprint != generate_stream(seeds["validation"][0], 3, 64).fingerprint


def test_teacher_cache_is_exact_causal_oracle_experience():
    stream = generate_stream(12, 4, 128)
    trajectory = run.teacher_trajectory(stream)
    world = WorldBatch(stream)
    for tick in range(stream.steps):
        assert torch.equal(trajectory.observations[tick], world.observe(tick))
        target = run.teacher_action(world)
        assert torch.equal(trajectory.targets[tick], target)
        world.act(target)
    assert trajectory.oracle_metrics == world.metrics()
    assert trajectory.oracle_metrics["clean_completion_rate"] == 1.0
    assert trajectory.oracle_metrics["rejected"] == 0
    assert trajectory.fingerprint == run.teacher_trajectory(stream).fingerprint
    assert trajectory.observations.shape == (128, 4, 48)


@pytest.mark.parametrize("dt", [.125, .25, .375, .5])
def test_teacher_leaves_timing_margin_and_remains_causal_and_legal(dt):
    for condition in ("train", "conflict", "long_gap"):
        stream = generate_stream(46, 4, 160, dt=dt, condition=condition)
        world = WorldBatch(stream)
        finished = 0
        for tick in range(stream.steps):
            world.observe(tick)
            action = run.teacher_action(world)
            finish = action >= 3
            if finish.any():
                midpoint = (world.ready_time + world.deadline) * .5
                assert (world.now >= midpoint[finish] - 1e-6).all()
                assert (world.now < midpoint[finish] + dt + 1e-5).all()
                finished += int(finish.sum())
            world.act(action)
        assert finished > 0
        assert world.metrics()["clean_completion_rate"] == 1
        assert world.metrics()["rejected"] == 0


def test_initial_time_is_zero_and_subsequent_elapsed_uses_real_timestamps():
    stream = generate_stream(0, 1, 4)
    stream = replace(stream, times=torch.tensor([2.0, 2.125, 2.5, 3.25], dtype=torch.float64))
    assert run.elapsed_times(stream) == (0, .125, .375, .75)
    trajectory = run.teacher_trajectory(stream)
    model = RecordingModel()
    state = model.initial_state(1)
    _, state = run.forward_chunk(model, trajectory, state, 0, 2)
    _, state = run.forward_chunk(model, trajectory, state.detach(), 2, 4)
    assert [call["dt"] for call in model.calls] == [0, .125, .375, .75]
    assert torch.equal(state, torch.full((1, 2), 4.0))


def test_training_preserves_state_at_tbptt_and_resets_only_new_streams(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(run, "make_model", lambda kind, args: model)
    _, report, _ = run.train_one("learned", tiny_args())
    assert model.initial_calls == 3
    assert [float(call["state"][0, 0]) for call in model.calls] == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert [call["requires_grad"] for call in model.calls[:4]] == [False, True, False, True]
    assert [call["dt"] for call in model.calls] == [0, .125, .125, .125, 0, .25, .25, .25, 0, .5]
    assert [item["consumed_steps"] for item in report["streams"]] == [4, 4, 2]
    assert report["action_labels"] == 10 * 2 * 2
    assert len({item["world_hash"] for item in report["streams"]}) == 3


def test_training_data_is_paired_across_model_kinds(monkeypatch):
    monkeypatch.setattr(run, "make_model", lambda kind, args: RecordingModel())
    args = tiny_args("--updates", "3")
    reports = [run.train_one(kind, args)[1] for kind in ("reservoir", "learned", "gru")]
    assert reports[0]["streams"] == reports[1]["streams"] == reports[2]["streams"]


def test_scripted_oracle_has_perfect_closed_loop_and_midpoint_completion():
    stream = generate_stream(145, 8, 192)
    teacher = run.teacher_trajectory(stream)
    model = ScriptedOracle(teacher.targets)
    result = run.closed_loop_rollout(model, stream)
    assert result["world"]["clean_completion_rate"] == 1.0
    assert result["world"]["finish_precision"] == 1.0
    assert result["world"]["rejected"] == 0
    assert result["action_metrics"]["accuracy"] == 1.0
    assert result["midpoint_pending"] > 0
    assert result["midpoint_completion_rate"] == 1.0
    assert result["midpoint_clean_completion_rate"] == 1.0
    reset = run.closed_loop_rollout(model, stream, control="reset")
    assert reset["midpoint_target_hash"] == result["midpoint_target_hash"]
    assert reset["midpoint_completion_rate"] == 1.0


def test_closed_loop_experience_changes_with_actions():
    stream = generate_stream(145, 4, 64)
    teacher = run.teacher_trajectory(stream)
    model = RecordingModel()  # All tied logits choose WAIT.
    result = run.closed_loop_rollout(model, stream)
    observed = torch.stack([call["observation"] for call in model.calls])
    assert not torch.equal(observed, teacher.observations)
    assert result["world"]["completed_eligible"] == 0
    assert result["world"]["eligible"] > 0
    assert result["world_hash"] == stream.fingerprint


def test_reset_and_noise_are_single_midpoint_interventions_and_repeatable():
    stream = generate_stream(3, 3, 32)
    baseline_model, reset_model, noise_model = RecordingModel(), RecordingModel(), RecordingModel()
    baseline = run.closed_loop_rollout(baseline_model, stream)
    reset = run.closed_loop_rollout(reset_model, stream, control="reset")
    noise = run.closed_loop_rollout(noise_model, stream, control="small_noise", noise_seed=42)
    repeated = run.closed_loop_rollout(RecordingModel(), stream, control="small_noise", noise_seed=42)
    assert reset["intervention_count"] == noise["intervention_count"] == 1
    assert baseline["intervention_count"] == 0
    assert reset["intervention"]["tick"] == 16
    assert float(reset_model.calls[16]["state"].abs().sum()) == 0
    assert float(reset_model.calls[17]["state"].mean()) == 1
    assert all(torch.equal(baseline_model.calls[i]["state"], noise_model.calls[i]["state"])
               for i in range(16))
    assert not torch.equal(baseline_model.calls[16]["state"], noise_model.calls[16]["state"])
    assert noise["final_state_hash"] == repeated["final_state_hash"]
    assert noise["intervention"] == repeated["intervention"]


def test_clock_and_solver_controls_keep_observations_and_action_opportunities():
    stream = generate_stream(5, 2, 16, dt=.375)
    models = {control: RecordingModel() for control in ("baseline", "half_step", "clock_half", "clock_double")}
    records = {control: run.closed_loop_rollout(model, stream, control=control)
               for control, model in models.items()}
    for control, model in models.items():
        assert len(model.calls) == 16
        assert records[control]["prediction_hash"] == records["baseline"]["prediction_hash"]
        assert records[control]["world_hash"] == records["baseline"]["world_hash"]
        assert all(torch.equal(a["observation"], b["observation"])
                   for a, b in zip(models["baseline"].calls, model.calls))
    assert all(call["max_step"] == .125 for call in models["half_step"].calls)
    assert [call["dt"] for call in models["half_step"].calls] == [0] + [.375] * 15
    assert [call["dt"] for call in models["clock_half"].calls] == [0] + [.1875] * 15
    assert [call["dt"] for call in models["clock_double"].calls] == [0] + [.75] * 15


def test_wait_dominance_is_exposed_by_nonwait_diagnostics():
    stats = run.ActionStats(11, "cpu")
    target = torch.tensor([[0, 0], [0, 3], [1, 0]])
    logits = torch.zeros(3, 2, 11)
    logits[..., 0] = 5
    stats.add(logits, target)
    result = stats.result()
    assert result["accuracy"] == 4 / 6
    assert result["wait"] == {"count": 4, "accuracy": 1.0}
    assert result["nonwait"] == {"count": 2, "accuracy": 0.0}
    assert result["weighted_ce"] > result["nll"]


def test_evaluation_namespaces_and_clocks_are_explicit():
    args = tiny_args()
    validation = run.evaluation_specs(args, "validation")
    test = run.evaluation_specs(args, "test")
    by_name = {row["name"]: row for row in test}
    assert validation[0]["seed"] != by_name["iid"]["seed"]
    assert by_name["clock_0375"]["dt"] == .375
    assert by_name["clock_0375"]["condition"] == "train"
    assert by_name["long"]["steps"] == 4 * args.eval_steps
    assert by_name["iid"]["dt"] == .25


def test_saved_checkpoint_can_evaluate_without_training(tmp_path, monkeypatch):
    model = DynamicsModel(width=4, kind="learned")
    model_config = {"width": 4, "kind": "learned", "max_step": .25}
    path = tmp_path / "saved.pt"
    torch.save({"model_config": model_config, "state_dict": model.state_dict(),
                "source_hashes": run.source_hashes(), "config": {"seed": 0}}, path)
    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint evaluation must never train")
    monkeypatch.setattr(run, "train_one", forbidden)
    output = tmp_path / "evaluation"
    summary = run.main(["--checkpoint", str(path), "--output", str(output), "--device", "cpu",
                        "--eval-steps", "8", "--eval-batch", "2", "--skip-test"])
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["test_skipped"] is True and "test_streams" not in saved
    assert summary["models"]["learned"]["parameter_hash"] == run.tensor_digest(model.state_dict())
    assert set(saved["source_hashes"]) == {"run.py", "model.py", "world.py", "test_run.py"}
    assert (output / "source" / "run.py").is_file()
    with pytest.raises(FileExistsError):
        run.main(["--output", str(output), "--device", "cpu"])


def test_checkpoint_world_or_model_changes_cannot_silently_change_evaluation():
    hashes = run.source_hashes()
    saved = {"source_hashes": dict(hashes)}
    run.check_checkpoint_sources(saved, hashes)
    saved["source_hashes"]["test_run.py"] = "test-only change"
    run.check_checkpoint_sources(saved, hashes)
    saved["source_hashes"]["world.py"] = "different world"
    with pytest.raises(ValueError, match="world.py"):
        run.check_checkpoint_sources(saved, hashes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_teacher_cache_transfers_once_to_cuda_and_rollout_runs_on_cuda():
    stream = generate_stream(2, 2, 16)
    teacher = run.teacher_trajectory(stream, "cuda")
    assert teacher.observations.device.type == teacher.targets.device.type == "cuda"
    assert teacher.fingerprint == run.teacher_trajectory(stream).fingerprint
    model = DynamicsModel(width=4).cuda()
    result = run.closed_loop_rollout(model, stream, device="cuda", control="reset")
    assert result["action_metrics"]["count"] == 16 * 2 * 2
    assert result["intervention_count"] == 1


@pytest.mark.parametrize("flag,value", [("--tbptt", "0"), ("--dt", "nan"),
                                        ("--train-dts", "-1"), ("--wait-weight", "0")])
def test_invalid_budgets_and_time_values_fail(flag, value):
    with pytest.raises(SystemExit):
        run.parse_args([flag, value])
