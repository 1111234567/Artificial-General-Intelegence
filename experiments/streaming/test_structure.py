"""Check genuine, budget-matched structural changes and local signal boundaries."""

import pytest
import torch

from experiments.streaming.structure import (SparseReadout, evaluate, generator,
    initialize_models, teacher_residual, tensor_hash)
from experiments.streaming.local_learning import derived_seed, make_task, sample_batch, stream_indices


def test_sparse_arms_start_identically_and_dense_is_separately_larger():
    models = initialize_models(66, 3, 16, 19, .01)
    for arm in ("learned", "random"):
        assert torch.equal(models["fixed"].weight, models[arm].weight)
        assert torch.equal(models["fixed"].mask, models[arm].mask)
        assert models["fixed"].weight.data_ptr() != models[arm].weight.data_ptr()
    for arm in ("fixed", "learned", "random"):
        counts = models[arm].connection_counts()
        assert counts["connection_slots"] == counts["actual_nonzero"] == 48
        assert counts["per_output_slots"] == [16, 16, 16]
        assert counts["dormant_nonzero"] == 0
    assert models["dense"].connection_counts()["actual_nonzero"] == 198
    assert models["dense"].correlation.numel() == models["fixed"].correlation.numel()


def test_local_weight_update_matches_independent_per_example_calculation():
    model = initialize_models(6, 3, 2, 5, .01)["fixed"]
    features = torch.tensor([[1., 0., -2., 3., .5, -1.], [1., 2., 0., -1., 1., 2.]])
    residual = torch.tensor([[.2, -.3, .1], [-.1, .4, -.3]])
    original = model.weight.clone()
    model.update(features, residual, .2)
    for output in range(3):
        for feature in range(6):
            correlation = sum(float(features[row, feature]) * float(residual[row, output]) for row in range(2)) / 2
            expected = float(original[output, feature]) + .2 * correlation if model.mask[output, feature] else 0.
            assert float(model.weight[output, feature]) == pytest.approx(expected, abs=1e-7)
            assert float(model.correlation[output, feature]) == pytest.approx(correlation, abs=1e-7)
    assert model.window_batches == 1
    assert not model.weight[~model.mask].any()


@pytest.mark.parametrize("strategy", ["fixed", "learned", "random"])
def test_rewiring_counts_real_mask_changes_and_preserves_each_output_budget(strategy):
    model = initialize_models(66, 3, 16, 15, .01)["fixed"]
    original = model.mask.clone()
    # A real activity/feedback observation supplies candidate correlations.
    features = torch.randn((17, 66), generator=generator(23))
    targets = torch.randint(0, 3, (17,), generator=generator(24))
    model.update(features, teacher_residual(model.probabilities(features), targets), .2)
    event = model.rewire(strategy, 2, generator(57))
    expected = 0 if strategy == "fixed" else 6
    assert event["removed"] == event["added"] == expected
    assert event["changed"] == int(original.ne(model.mask).sum()) == 2 * expected
    assert set(map(tuple, event["removed_edges"])).isdisjoint(map(tuple, event["added_edges"]))
    assert model.mask.sum(1).tolist() == [16] * 3
    assert model.connection_counts()["actual_nonzero"] <= 48
    assert model.connection_counts()["actual_nonzero"] >= 42
    assert not model.weight[~model.mask].any()
    assert not model.correlation.any() and model.window_batches == 0


def test_learned_growth_uses_only_observed_activity_times_task_residual():
    model = SparseReadout(torch.tensor([[.1, .9, 0., 0., 0.]]),
                          torch.tensor([[True, True, False, False, False]]))
    features = torch.tensor([[1., 0., 2., 0., 8.]])
    model.update(features, torch.tensor([[.5]]), .01)
    event = model.rewire("learned", 1, generator(12))
    assert event["removed_edges"] == [[0, 0]]
    assert event["added_edges"] == [[0, 4]]
    assert event["born_signed_window_correlation"] == [4.0]


def test_random_topology_is_independent_of_teacher_feedback_and_matches_rng_consumption():
    models = initialize_models(66, 3, 16, 4, .01)
    other = models["random"].clone()
    features = torch.randn((13, 66), generator=generator(18))
    models["random"].update(features, torch.ones((13, 3)), .2)
    other.update(features, -torch.ones((13, 3)), .2)
    models["random"].rewire("random", 2, generator(38))
    other.rewire("random", 2, generator(38))
    assert torch.equal(models["random"].mask, other.mask)
    random_rng, learned_rng = generator(40), generator(40)
    models["fixed"].rewire("random", 2, random_rng)
    models["learned"].rewire("learned", 2, learned_rng)
    assert torch.equal(random_rng.get_state(), learned_rng.get_state())


def test_task_and_feedback_are_shared_without_phase_specific_model_inputs():
    task = make_task(30, input_dim=32, n_train=32, n_val=17, n_test=19, device="cpu")
    indices = torch.tensor([0, 4, 9, 15])
    features_a, targets_a = task.batch("train", indices=indices, context=0)
    features_b, targets_b = task.batch("train", indices=indices, context=1)
    assert features_a.shape == features_b.shape == (4, 66)
    assert torch.equal(features_a[:, 1:33], features_b[:, 1:33])
    assert torch.equal(targets_b, (targets_a + 1) % 3)
    model = initialize_models(66, 3, 16, 37, .01)["fixed"]
    probabilities = model.probabilities(features_a)
    residual = teacher_residual(probabilities, targets_a)
    assert torch.allclose(residual.sum(1), torch.zeros(4), atol=1e-6)
    assert torch.equal(residual.argmax(1), targets_a)


def test_stream_exactly_matches_the_shared_local_learning_branch():
    from argparse import Namespace

    task = make_task(42, input_dim=4, n_train=41, n_val=7, n_test=9, device="cpu")
    configuration = Namespace(n_train=41, batch_size=7)
    for phase, context in (("A1", 0), ("B", 1), ("A2", 0)):
        indices = stream_indices(42, phase, 5, configuration)
        rng = generator(derived_seed(42, f"stream/{phase}"))
        for index in indices:
            observed = sample_batch(task, "train", context, 7, rng)
            expected = task.batch("train", indices=index, context=context)
            assert all(torch.equal(a, b) for a, b in zip(observed, expected))


def test_evaluation_cannot_mutate_weights_masks_statistics_or_rng():
    task = make_task(35, input_dim=4, n_train=16, n_val=23, n_test=29, device="cpu")
    model = initialize_models(10, 3, 4, 36, .01)["learned"]
    features, targets = sample_batch(task, "train", 0, 7, generator(44))
    model.update(features, teacher_residual(model.probabilities(features), targets), .2)
    state = (tensor_hash(model.weight), tensor_hash(model.mask), tensor_hash(model.correlation), model.window_batches)
    global_rng = torch.random.get_rng_state().clone()
    values = evaluate(model, task, "val")
    assert set(values) == {"A", "B"}
    assert all(value["count"] == 23 for value in values.values())
    assert state == (tensor_hash(model.weight), tensor_hash(model.mask), tensor_hash(model.correlation), model.window_batches)
    assert torch.equal(global_rng, torch.random.get_rng_state())


def test_many_rewirings_never_train_dormant_weights_or_change_parameter_count():
    model = initialize_models(66, 3, 16, 13, .01)["learned"]
    rng = generator(111)
    for step in range(30):
        features = torch.randn((16, 66), generator=rng)
        targets = torch.randint(0, 3, (16,), generator=rng)
        model.update(features, teacher_residual(model.probabilities(features), targets), .2)
        if step % 3 == 2:
            event = model.rewire("learned", 2, rng)
            assert event["removed"] == event["added"] == 6
        assert model.mask.sum(1).tolist() == [16] * 3
        assert model.weight.ne(0).sum() <= 48
        assert not model.weight[~model.mask].any()


def test_bad_dormant_weights_or_impossible_budget_are_rejected():
    with pytest.raises(ValueError, match="Dormant"):
        SparseReadout(torch.ones((1, 3)), torch.tensor([[True, False, False]]))
    model = initialize_models(6, 3, 5, 2, .01)["fixed"]
    with pytest.raises(ValueError, match="Replacement"):
        model.rewire("learned", 2, generator(12))
