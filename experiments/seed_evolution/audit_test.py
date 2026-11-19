"""Small fixtures for the independent audit. No search artifacts are created."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.learned_binding.audit import AuditError
from experiments.learned_binding.frozen_trace import FrozenTracePolicy
from experiments.seed_evolution import audit, core


def test_fitness_weights_worlds_and_families_equally():
    first = dict(stream_id=1100, fitness=0.5, groups={
        "A_conflict": dict(n=1, strict=1), "B_conflict": dict(n=100, strict=0)})
    second = dict(stream_id=1101, fitness=0.25, groups={
        "A_conflict": dict(n=4, strict=1), "B_conflict": dict(n=8, strict=2)})
    assert audit.conflict_fitness([first, second]) == 0.375
    with pytest.raises(AuditError, match="repeated scored world"):
        audit.conflict_fitness([first, first])
    with pytest.raises(AuditError, match="world ledger"):
        audit.conflict_fitness([dict(first, fitness=0.01)])
    invalid = deepcopy(first)
    invalid["groups"]["A_conflict"]["n"] = 0
    with pytest.raises(AuditError, match="invalid A_conflict"):
        audit.conflict_fitness([invalid])


def test_generation_requires_cumulative_budget_parent_and_fitness_winner():
    candidates = {
        "a": dict(fitness=0.4, parent_id=None), "b": dict(fitness=0.4, parent_id=None),
        "c": dict(fitness=0.2, parent_id=None), "d": dict(fitness=0.5, parent_id="b"),
    }
    row = dict(top2_before=["a", "b"], evaluated_ids=["b", "a", "c", "d"],
               new_ids=["d"], best_id="d", best_fitness=0.5)
    assert audit.audit_generation(row, ["b", "a", "c"], ["d"], candidates, "evolution") == (
        ["b", "a", "c", "d"], "d")
    with pytest.raises(AuditError, match="parent pool"):
        audit.audit_generation(dict(row, top2_before=["b", "a"]), ["b", "a", "c"],
                               ["d"], candidates, "evolution")
    with pytest.raises(AuditError, match="ineligible parent"):
        audit.audit_generation(row, ["b", "a", "c"], ["d"],
                               dict(candidates, d=dict(fitness=0.5, parent_id="c")), "evolution")
    with pytest.raises(AuditError, match="cumulative evaluation budget"):
        audit.audit_generation(dict(row, evaluated_ids=["b", "a", "d"]), ["b", "a", "c"],
                               ["d"], candidates, "evolution")
    with pytest.raises(AuditError, match="cumulative winner"):
        audit.audit_generation(dict(row, best_id="b"), ["b", "a", "c"],
                               ["d"], candidates, "evolution")


def test_official_training_stream_schedule():
    assert [audit.historical_stream_at(step) for step in (0, 7, 8, 63, 127, 128, 143, 144, 511)] == [
        0, 7, 0, 7, 7, 0, 15, 0, 15]


def test_replay_ce_and_adam_rejects_changed_weight_and_optimizer():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(548932)
            model = FrozenTracePolicy().train()
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
            stream = dict(obs=torch.randn(4, 2, 56), actions=torch.randint(0, 11, (4, 2, 2)))
            checkpoints = []
            for step in range(3):
                checkpoints.append(dict(model=deepcopy(model.state_dict()), model_config=model.config(),
                                        optimizer=deepcopy(optimizer.state_dict()), updates=step,
                                        source_seed=0))
                if step == 2:
                    break
                logits, _ = model(stream["obs"], model.initial_state(2))
                weights = torch.tensor([0.25] + [1.0] * 10)
                loss = F.cross_entropy(logits.flatten(0, 2), stream["actions"].flatten(), weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            before, middle, after = checkpoints
            assert audit.replay_checkpoint_update(before, middle, stream)["frozen_gradients_absent"]
            assert audit.replay_checkpoint_update(middle, after, stream)["step"] == 2
            changed = deepcopy(after)
            changed["model"]["gate.bias"].add_(1.0)
            with pytest.raises(AuditError, match="independent CE/Adam parameters"):
                audit.replay_checkpoint_update(middle, changed, stream)
            changed = deepcopy(middle)
            next(iter(changed["optimizer"]["state"].values()))["step"].add_(1.0)
            with pytest.raises(AuditError, match="wrong Adam update counter"):
                audit.replay_checkpoint_update(changed, after, stream)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("condition", ["own", "teacher", "no_read", "Bdelay"])
def test_heldout_archive_recount_and_replay(condition):
    old_threads = torch.get_num_threads()
    try:
        model = core.model_from_genome(0, core.new_genome(0, 710000))
        config = dict(test_streams=[901], support_ticks=24, probe_ticks=480,
                      actors=4, random_seed=78111, b_ticks=40)
        score = core.score_model(0, model.state_dict(), **audit.scoring_options(config, "test_streams"),
                                 archive=True, teacher_support=condition == "teacher",
                                 read_enabled=condition != "no_read",
                                 b_ticks=config["b_ticks"] if condition == "Bdelay" else 0)
        archives = score.pop("archives")
        score["condition"] = condition
        audit.verify_scoring(score, 0, [901], score["parameter_hash"], config, condition)
        assert audit.audit_evaluation_archive(score, archives, config, [901]) == 1
        changed = deepcopy(score)
        changed["streams"][0]["groups"]["A_conflict"]["strict"] += 1
        with pytest.raises(AuditError, match="held-out compact conflict ledger"):
            audit.audit_evaluation_archive(changed, archives, config, [])
    finally:
        torch.set_num_threads(old_threads)


def test_regenerates_initial_genome_and_child_without_training():
    executor = audit.common.executor(0)
    genome = audit.regenerate_genome(0, 710000, executor)
    audit.same_tree(genome, core.new_genome(0, 710000), "initial genome")
    child = audit.mutate_initial_genome(genome, 810100, .15)
    audit.same_tree(child, core.mutate_genome(genome, 810100, .15), "child genome")
