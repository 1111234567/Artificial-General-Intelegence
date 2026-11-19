"""Boundary, budget, and reproducibility tests for the experiment runner."""
import json
from pathlib import Path
import shutil

import pytest
import torch

from . import run, world


class FixedScores(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, features, legal, rounds=None):
        return torch.tensor(self.scores, dtype=torch.float32,
                            device=features.device).expand(len(features), -1)


def chain_problem():
    return world.Problem(
        initial=(1, 0, 0, 0), goal=(0, 0, 0, 1),
        consume=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
        produce=((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        metadata={"solver_plan": [0, 1, 2], "secret": 99},
    )


def fork_problem():
    # Actions 0 and 3 are equivalent. Either independent lane can go first.
    return world.Problem(
        initial=(1, 1, 0, 0, 0), goal=(0, 0, 0, 0, 1),
        consume=((1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                 (0, 0, 1, 1, 0), (1, 0, 0, 0, 0)),
        produce=((0, 0, 1, 0, 0), (0, 0, 0, 1, 0),
                 (0, 0, 0, 0, 1), (0, 0, 1, 0, 0)),
    )


def test_labels_include_noncanonical_states_and_all_optimal_actions():
    problem = fork_problem()
    labels = run.labelled_states(problem, seed=12, max_states=8)
    solution = world.solve(problem)
    assert labels[0]["inventory"] == problem.initial
    assert labels[0]["target_mask"] == [True, True, False, True]
    # Canonical shortest path starts with 0; the state after 1 is off that path.
    assert world.step(problem, problem.initial, 1) in [entry["inventory"] for entry in labels]
    for entry in labels:
        expected = world.optimal_actions(problem, entry["inventory"], solution)
        actual = tuple(action for action, selected in enumerate(entry["target_mask"]) if selected)
        assert actual == expected
    assert len(labels) <= 8


def test_search_limit_never_silently_becomes_a_label(monkeypatch):
    def fail(*args, **kwargs):
        raise world.SearchLimitExceeded("cap")
    monkeypatch.setattr(world, "solve", fail)
    with pytest.raises(world.SearchLimitExceeded):
        run.labelled_states(chain_problem(), seed=0)


def test_split_namespaces_are_disjoint_and_groups_are_explicit():
    specs = {split: run.split_specs(split, 30 if split == "train" else 2, 7)
             for split in ("train", "val", "test")}
    sets = {split: {row["seed"] for row in rows} for split, rows in specs.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert all(len(values) == len(specs[split]) for split, values in sets.items())
    assert {(row["family"], row["depth"]) for row in specs["train"]} == set(run.TRAIN_GROUPS)
    assert len(specs["test"]) == 2 * len(run.TEST_GROUPS)
    assert run.distribution("cross", 4) == "structural_ood"
    assert run.distribution("cross", 8) == "structural_length_ood"
    assert specs == {split: run.split_specs(split, 30 if split == "train" else 2, 7)
                     for split in specs}


def test_sampling_is_paired_reproducible_and_balances_initial_states():
    records = [{"observation": fork_problem().observation(),
                "states": run.labelled_states(fork_problem(), seed=seed)} for seed in (1, 2)]
    pool = run.build_pool(records, "cpu")
    first = run.sample_batch_indices(pool, 20000, torch.Generator().manual_seed(123))
    second = run.sample_batch_indices(pool, 20000, torch.Generator().manual_seed(123))
    assert torch.equal(first, second)
    is_initial = (first[:, None] == pool.initial_indices).any(-1)
    assert 0.48 < float(is_initial.float().mean()) < 0.52
    assert not (pool.targets & ~pool.legal).any()


def test_greedy_core_budget_counts_rounds_and_stops_without_partial_call():
    result = run.greedy_rollouts(FixedScores([9, 8, 7]), [chain_problem()],
                                 rounds=2, core_budget=5, max_steps=24)[0]
    assert result["core_calls"] == 4
    assert result["transition_calls"] == 2
    assert result["plan"] == (0, 1)
    assert result["valid_plan"] and not result["success"]
    assert result["stop_reason"] == "core_budget"
    no_call = run.greedy_rollouts(FixedScores([9, 8, 7]), [chain_problem()],
                                  rounds=8, core_budget=7)[0]
    assert no_call["core_calls"] == no_call["transition_calls"] == 0


def test_greedy_masks_current_illegality_and_obeys_hard_step_limit():
    result = run.greedy_rollouts(FixedScores([0, 10, 20]), [chain_problem()],
                                 rounds=1, max_steps=2)[0]
    assert result["plan"] == (0, 1)
    assert result["core_calls"] == 2
    assert result["stop_reason"] == "max_steps"
    assert not world.validate_plan(chain_problem(), [2])


def test_independent_execution_can_reject_a_simulator_claim(monkeypatch):
    problem = world.Problem((1, 0), (0, 1), ((1, 0),), ((0, 0),))
    monkeypatch.setattr(world, "step", lambda p, state, action: p.goal)
    result = run.greedy_rollouts(FixedScores([1]), [problem], rounds=1)[0]
    assert result["simulator_goal_reached"] is True
    assert result["independent_validation"] is False
    assert result["success"] is False


def test_policy_rollouts_never_receive_metadata_or_call_solver(monkeypatch):
    original_tensorize = run.tensorize
    def checked_tensorize(problems, *args, **kwargs):
        assert all(problem.metadata == {} for problem in problems)
        return original_tensorize(problems, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("the exact solver entered policy evaluation")
    monkeypatch.setattr(run, "tensorize", checked_tensorize)
    monkeypatch.setattr(world, "solve", forbidden)
    model = FixedScores([3, 2, 1])
    assert run.greedy_rollouts(model, [chain_problem()], rounds=1)[0]["success"]
    assert run.neural_best_first(model, chain_problem(), core_budget=3)["success"]


def test_external_search_recovers_from_trap_and_groups_duplicate_recipes():
    problem = world.Problem(
        (1, 0, 0, 0), (0, 0, 0, 1),
        ((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)),
        ((0, 0, 1, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
    )
    model = FixedScores([4, 3, 2, 1])
    assert not run.greedy_rollouts(model, [problem], rounds=1)[0]["success"]
    result = run.neural_best_first(model, problem, core_budget=2, branch=2)
    assert result["success"] and world.validate_plan(problem, result["plan"])
    assert result["core_calls"] == result["queried_distinct_states"] == 2
    # The duplicate trap recipe is grouped before simulation: 2 first steps + 1 second step.
    assert result["transition_calls"] == 3
    assert result["transition_calls"] <= result["transition_budget"]
    exhausted = run.neural_best_first(model, problem, core_budget=1, branch=2)
    assert not exhausted["success"]
    assert exhausted["core_calls"] == 1
    assert exhausted["valid_plan"]


def test_external_search_sums_equivalent_action_probability():
    # Three individually weaker correct action IDs jointly outweigh one trap.
    problem = world.Problem(
        (1, 0, 0), (0, 0, 1),
        ((1, 0, 0),) * 4 + ((0, 1, 0),),
        ((0, 0, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1)),
    )
    model = FixedScores([1.0, 0.5, 0.5, 0.5, 0.0])
    result = run.neural_best_first(model, problem, core_budget=2, branch=1)
    assert result["success"]
    assert result["plan"][0] in (1, 2, 3)


def test_search_subset_is_paired_and_balanced_across_groups():
    records = run.split_specs("test", 5, 0)
    selected = run.paired_subset(records, 2 * len(run.TEST_GROUPS))
    counts = {(family, depth): sum(row["family"] == family and row["depth"] == depth
                                  for row in selected) for family, depth in run.TEST_GROUPS}
    assert set(counts.values()) == {2}
    assert len({row["case_id"] for row in selected}) == len(selected)
    assert run.paired_subset(records, 0) == []


def test_nonempty_output_is_rejected_before_any_generation(tmp_path):
    (tmp_path / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        run.freeze_experiment(tmp_path, {}, "cpu")


def test_frozen_dataset_and_source_hashes_are_enforced(tmp_path):
    run.write_json(tmp_path / "config.json", {"seed": 0})
    run.write_json(tmp_path / "environment.json", {"torch": str(torch.__version__)})
    data = tmp_path / "test.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    sources = {}
    (tmp_path / "source").mkdir()
    for name in run.SOURCE_FILES:
        source = Path(run.__file__).with_name(name)
        if source.exists():
            shutil.copyfile(source, tmp_path / "source" / name)
            sources[name] = run.file_sha256(source)
    run.write_json(tmp_path / "manifest.json", {
        "schema_version": run.SCHEMA_VERSION,
        "config_sha256": run.file_sha256(tmp_path / "config.json"),
        "environment_sha256": run.file_sha256(tmp_path / "environment.json"),
        "sources": sources,
        "datasets": {"test": {"file": "test.jsonl", "sha256": run.file_sha256(data)}},
    })
    assert run.verify_experiment(tmp_path)[0] == {"seed": 0}
    data.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset was modified"):
        run.verify_experiment(tmp_path)


def test_evaluation_is_explicit_and_cannot_be_accidentally_repeated(tmp_path, monkeypatch):
    config = {"seed": 0, "threads": 1, "beam_limit": 0}
    manifest = {"config_sha256": "fixed", "datasets": {"test": {"file": "test.jsonl"}}}
    (tmp_path / "test.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(run, "verify_experiment", lambda path: (config, manifest))
    monkeypatch.setattr(run, "load_models", lambda *args: {})
    monkeypatch.setattr(run, "evaluate_models", lambda *args, **kwargs: [])
    run.evaluate(tmp_path, split="test", device="cpu")
    assert (tmp_path / "summary_test.json").exists()
    with pytest.raises(FileExistsError):
        run.evaluate(tmp_path, split="test", device="cpu")
    with pytest.raises(ValueError, match="frozen"):
        run.evaluate(tmp_path, split="test", device="cpu", beam_limit=9)


def test_eval_json_keeps_offline_metadata_outside_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(world, "generate_problem", lambda *args, **kwargs: chain_problem())
    path = tmp_path / "cases.jsonl"
    run.generate_dataset(path, [run.split_specs("test", 1, 0)[0]],
                         {"n_items": 40, "n_rules": 32, "solver_max_states": 50_000},
                         training=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["optimal_length"] == 3
    assert set(record["observation"]) == {"initial", "goal", "consume", "produce"}
    assert record["generator_metadata"]["secret"] == 99
    assert "states" not in record and "plan" not in record
    assert run.problem_from_record(record).metadata == {}


def test_tiny_end_to_end_training_checkpoints_and_one_shot_eval(tmp_path, monkeypatch):
    # Hand-written padded task isolates the runner from generator revisions.
    original = chain_problem()
    pad = lambda row: row + (0,) * 36
    problem = world.Problem(
        pad(original.initial), pad(original.goal),
        tuple(pad(original.consume[action % 3]) for action in range(32)),
        tuple(pad(original.produce[action % 3]) for action in range(32)),
        metadata={"test_fixture": True},
    )
    monkeypatch.setattr(world, "generate_problem", lambda *args, **kwargs: problem)
    out = tmp_path / "experiment"
    run.main(["train", "--out", str(out), "--device", "cpu", "--seed", "3",
              "--train-problems", "4", "--updates", "2", "--batch-size", "4",
              "--width", "8", "--rounds", "2", "--include-detach",
              "--val-per-group", "1", "--test-per-group", "1", "--val-every", "0",
              "--beam-limit", "2", "--beam-budget", "2", "--threads", "1"])
    assert not (out / "evaluation_test.started.json").exists()
    summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["optimizer_steps"] == {"shallow": 4, "recurrent": 2, "untied": 2, "detach": 2}
    assert summary["core_layer_examples_each"] == 16
    curves = run.load_records(out / "curves.jsonl")
    for update in (1, 2):
        hashes = {row["paired_batch_sha256_so_far"] for row in curves
                  if row["external_update"] == update}
        assert len(hashes) == 1
    early = torch.load(out / "shallow_equal_updates.pt", weights_only=True)
    assert early["optimizer_steps"] == 2 and early["external_update"] == 1
    config, manifest = run.verify_experiment(out)
    models = run.load_models(out, config, manifest, "cpu")
    assert set(models) == {"shallow", "recurrent", "untied", "detach"}
    def forbidden(*args, **kwargs):
        raise AssertionError("solver used after training")
    monkeypatch.setattr(world, "solve", forbidden)
    run.evaluate(out, split="test", device="cpu")
    rows = run.load_records(out / "results_test.jsonl")
    assert rows and all(row["valid_plan"] for row in rows)
    assert all(row["core_calls"] <= 96 for row in rows)
    assert {row["method"] for row in rows} >= {"recurrent_r4", "shallow_bestfirst_c2", "random"}
    with pytest.raises(FileExistsError):
        run.evaluate(out, split="test", device="cpu")
