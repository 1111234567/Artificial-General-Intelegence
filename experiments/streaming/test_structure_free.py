"""Signal, continuity, counter, probe, and capacity-control semantics."""
import pytest
import torch

from experiments.streaming.local_learning import make_task
from experiments.streaming.structure_free import (Population, PRIMARY, compare_probe, evaluate,
    initial_arrays, initialize_population, peak_capacity_mask, teacher_residual, tensor_hash)


def single(name, weight):
    tensor = torch.tensor(weight, dtype=torch.float32).reshape(1, 1, -1)
    return Population((name,), tensor, tensor.ne(0), int(tensor.ne(0).sum()))


def test_gradual_edges_fade_to_zero_instead_of_being_cut_for_new_births():
    model = single("gradual_free", [.1, 0., 0.])
    x = torch.tensor([[0., 1., 0.]])
    feedback = torch.tensor([[[.5]]])
    model.local_update(x, feedback, .1, .1)
    assert model.weight[0, 0, 0].item() == pytest.approx(.09)
    event = model.structural_window(.1, .1)
    assert model.pool[0, 0, :2].tolist() == [True, True]
    assert model.weight[0, 0, 0].item() == pytest.approx(.09)
    assert model.weight[0, 0, 1].item() == pytest.approx(.04)
    assert sum(event["pool_removed"][0]) == 0
    assert sum(event["nonzero_removed_at_window"][0]) == 0


def test_gradual_retirement_happens_only_at_zero_and_activity_can_reinforce_old_edge():
    model = single("gradual_free", [.03, 0.])
    for _ in range(2):
        model.local_update(torch.zeros((1, 2)), torch.zeros((1, 1, 1)), .1, .1)
        assert model.pool[0, 0, 0]
    assert model.weight[0, 0, 0].item() == pytest.approx(.01, abs=1e-7)
    model.local_update(torch.tensor([[1., 0.]]), torch.tensor([[[.5]]]), .1, .1)
    assert model.weight[0, 0, 0].item() == pytest.approx(.05, abs=1e-7)
    for _ in range(7):
        model.local_update(torch.zeros((1, 2)), torch.zeros((1, 1, 1)), .1, .1)
    assert model.weight[0, 0, 0].item() == 0.
    assert not model.pool[0, 0, 0]
    assert model.counts()["gradual_free"]["zero_exits"] == 1


def test_immediate_budget_can_cut_a_nonzero_old_edge_that_free_preserves():
    immediate = single("immediate_budget", [.03, 0., 0.])
    gradual = single("gradual_free", [.03, 0., 0.])
    for model in (immediate, gradual):
        model.local_update(torch.tensor([[0., 1., 0.]]), torch.tensor([[[1.]]]), .1, .01)
        model.structural_window(.1, .01)
    assert immediate.weight[0, 0, 0] == 0 and immediate.weight.ne(0).sum() == 1
    assert immediate.counts()["immediate_budget"]["nonzero_budget_cuts"] == 1
    assert gradual.weight[0, 0, 0] > 0 and gradual.weight.ne(0).sum() == 2
    assert gradual.counts()["gradual_free"]["nonzero_budget_cuts"] == 0


def test_only_inactive_activity_accumulates_and_zero_exit_starts_fresh_trace():
    model = single("gradual_free", [.01, 0.])
    model.local_update(torch.tensor([[1., 2.]]), torch.tensor([[[-.1]]]), .1, .1)
    assert not model.pool[0, 0, 0]
    assert model.candidate[0, 0, 0] == 0
    assert model.candidate[0, 0, 1].item() == pytest.approx(-.2)


def test_local_update_matches_direct_soft_threshold_without_dormant_training():
    full, mask = initial_arrays(10, 3, 4, 8, .01)
    model = initialize_population(full, mask, "cpu")
    x = torch.arange(20).float().reshape(2, 10) / 10
    p = model.probabilities(x)
    residual = teacher_residual(p, torch.tensor([0, 2]))
    initial = model.weight.clone()
    gradient = residual.permute(0, 2, 1) @ x / 2
    proposed = initial + .2 * gradient * model.pool
    expected = proposed.sign() * (proposed.abs() - .2 * .001).clamp_min(0) * model.pool
    model.local_update(x, residual, .2, .001)
    assert torch.allclose(model.weight, expected, atol=1e-7)
    assert not model.weight[~model.pool].any()


def test_initial_sparse_comparisons_are_identical_and_dense_budget_is_reported():
    full, mask = initial_arrays(66, 3, 16, 33, .01)
    model = initialize_population(full, mask, "cpu")
    assert model.names == PRIMARY
    assert torch.equal(model.weight[0], model.weight[1]) and torch.equal(model.weight[1], model.weight[2])
    assert model.pool[:3].sum((1, 2)).tolist() == [48] * 3
    assert model.pool[3].sum() == 198


def test_peak_capacity_is_train_trajectory_only_and_contains_initial_edges():
    _, initial = initial_arrays(10, 3, 3, 42, .01)
    peak = [5, 7, 4]
    matched = peak_capacity_mask(initial, peak, 42)
    assert matched.sum(1).tolist() == peak
    assert not (initial & ~matched).any()
    assert torch.equal(matched, peak_capacity_mask(initial, peak, 42))


def test_fixed_probe_is_read_only_and_detects_actual_prediction_shock():
    task = make_task(15, input_dim=4, classes=3, n_train=16, n_val=13, n_test=17)
    full, mask = initial_arrays(10, 3, 4, 18, .01)
    model = initialize_population(full, mask, "cpu")
    before_hash = (tensor_hash(model.weight), tensor_hash(model.pool), tensor_hash(model.candidate))
    before_metrics, before_p = evaluate(model, task, "val", torch.arange(7), True)
    _, same_p = evaluate(model, task, "val", torch.arange(7), True)
    change = compare_probe(model, before_p, same_p)
    assert (tensor_hash(model.weight), tensor_hash(model.pool), tensor_hash(model.candidate)) == before_hash
    assert all(change[name][ctx]["prediction_flip_rate"] == 0 for name in PRIMARY for ctx in ("A", "B"))
    assert all(before_metrics[name]["A"]["count"] == 7 for name in PRIMARY)
    model.weight[0].neg_()
    _, after_p = evaluate(model, task, "val", torch.arange(7), True)
    change = compare_probe(model, before_p, after_p)
    assert change["fixed_sparse"]["A"]["probability_total_variation"] > 0


def test_continuous_survival_cannot_be_restored_by_regrowing_same_edge():
    model = single("gradual_free", [.01, 0.])
    model.remember_survival("old")
    model.local_update(torch.zeros((1, 2)), torch.zeros((1, 1, 1)), .1, .2)
    assert model.survival["old"].sum() == 0
    model.local_update(torch.tensor([[1., 0.]]), torch.tensor([[[1.]]]), .1, .01)
    model.structural_window(.1, .01)
    assert model.weight[0, 0, 0] > 0
    assert model.survival["old"].sum() == 0


def test_pool_and_actual_change_counters_equal_independent_mask_differences():
    full, mask = initial_arrays(18, 3, 6, 71, .01)
    model = initialize_population(full, mask, "cpu")
    born = torch.zeros((4, 3), dtype=torch.long)
    retired = torch.zeros_like(born)
    rng = torch.Generator().manual_seed(72)
    for step in range(12):
        before = model.weight.ne(0).clone()
        x = torch.randn((11, 18), generator=rng)
        y = torch.randint(0, 3, (11,), generator=rng)
        model.local_update(x, teacher_residual(model.probabilities(x), y), .2, .01)
        after = model.weight.ne(0)
        born += (~before & after).sum(-1)
        retired += (before & ~after).sum(-1)
        if step % 3 == 2:
            before = after.clone()
            model.structural_window(.2, .01)
            after = model.weight.ne(0)
            born += (~before & after).sum(-1)
            retired += (before & ~after).sum(-1)
        assert not model.weight[~model.pool].any()
    assert torch.equal(model.actual_born, born)
    assert torch.equal(model.actual_retired, retired)
    assert model.pool[1].sum(1).tolist() == [6] * 3
    assert model.budget_nonzero_cuts[2].sum() == 0


def test_fused_probabilities_match_independent_readouts():
    full, mask = initial_arrays(10, 3, 4, 90, .01)
    model = initialize_population(full, mask, "cpu")
    x = torch.randn((17, 10), generator=torch.Generator().manual_seed(91))
    fused = model.probabilities(x)
    for index in range(4):
        assert torch.allclose(fused[index], (x @ model.weight[index].T).softmax(-1), atol=1e-7)


def test_cut_only_counterfactual_does_not_mutate_or_add_new_weights():
    task = make_task(81, input_dim=4, classes=3, n_train=16, n_val=11, n_test=13)
    full, mask = initial_arrays(10, 3, 4, 82, .01)
    model = initialize_population(full, mask, "cpu")
    old = model.weight.clone()
    x, y = task.batch("train", context=0)
    model.local_update(x, teacher_residual(model.probabilities(x), y), .2, .001)
    old.copy_(model.weight)
    model.structural_window(.2, .001)
    actual_hash = tensor_hash(model.weight)
    cut_only = old * model.pool
    assert not cut_only[old.eq(0)].any()
    _, original_probability = evaluate(model, task, "val", return_probabilities=True, weight_override=old)
    _, cut_probability = evaluate(model, task, "val", return_probabilities=True, weight_override=cut_only)
    assert tensor_hash(model.weight) == actual_hash
    for label in ("A", "B"):
        # Free window only adds edges, so removal-only forward is identical.
        assert torch.equal(original_probability[label][2], cut_probability[label][2])
