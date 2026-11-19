"""Fixed-world, observation-boundary, and held-out conjunction checks.

Run from the repository root:
    python -m pytest experiments/consolidation/test_world.py -q
"""

import pytest
import torch

from experiments.consolidation.world import CONTEXTS, World


def _small_world(seed: int = 21, **kwargs) -> World:
    options = dict(input_dim=8, n_train=257, n_val=79, n_test=101)
    options.update(kwargs)
    return World(seed, **options)


def _reference_a(world: World, x: torch.Tensor) -> torch.Tensor:
    # Compute per-class dot products independently of targets' batched matmul.
    return torch.stack([(x * row).sum(dim=-1) for row in world.teacher_w], dim=-1).argmax(-1)


def test_seeded_teachers_and_splits_do_not_use_or_depend_on_global_rng():
    saved = torch.random.get_rng_state().clone()
    first = _small_world()
    assert torch.equal(saved, torch.random.get_rng_state())
    try:
        torch.manual_seed(999_821)
        torch.randn(157)
        advanced = torch.random.get_rng_state().clone()
        second = _small_world()
        assert torch.equal(advanced, torch.random.get_rng_state())
    finally:
        torch.random.set_rng_state(saved)
    assert torch.equal(first.teacher_w, second.teacher_w)
    assert torch.equal(first.teacher_delta, second.teacher_delta)
    assert not torch.equal(first.teacher_w, first.teacher_delta)
    for name in first.splits:
        assert torch.equal(first.splits[name], second.splits[name])
    other = _small_world(seed=22)
    assert not torch.equal(first.teacher_w, other.teacher_w)
    assert not torch.equal(first.splits["train"], other.splits["train"])
    for teacher in (first.teacher_w, first.teacher_delta):
        torch.testing.assert_close(teacher.norm(dim=-1), torch.ones(first.classes))


def test_splits_are_raw_disjoint_inputs_with_correct_joint_region():
    world = _small_world(n_train=2048)
    expected_counts = dict(train=2048, val=79, test=101, transfer_val=79, transfer_test=101)
    assert set(world.splits) == set(expected_counts)
    for name, x in world.splits.items():
        assert x.shape == (expected_counts[name], world.input_dim)
        assert x.dtype == torch.float32
        assert x.device == world.device
        joint_positive = (x[:, 0] > 0) & (x[:, 1] > 0)
        assert bool(joint_positive.all()) if name.startswith("transfer_") else not bool(joint_positive.any())
    combined = torch.cat(list(world.splits.values()))
    assert torch.unique(combined, dim=0).shape[0] == combined.shape[0]
    train = world.splits["train"]
    # Both factors and all three allowed sign combinations are represented.
    for dim in (0, 1):
        assert bool((train[:, dim] > 0).any())
        assert bool((train[:, dim] < 0).any())
    for first_positive, second_positive in ((False, False), (True, False), (False, True)):
        assert bool((((train[:, 0] > 0) == first_positive) & ((train[:, 1] > 0) == second_positive)).any())


@pytest.mark.parametrize("classes", [3, 4, 7])
def test_teacher_rules_conflict_and_only_a_revises(classes: int):
    world = _small_world(classes=classes)
    x = torch.cat(list(world.splits.values()))
    a0 = _reference_a(world, x)
    b_scores = torch.stack(
        [(x * (w + 0.5 * d)).sum(-1) for w, d in zip(world.teacher_w, world.teacher_delta)],
        dim=-1,
    )
    expected = {"A": a0, "B": b_scores.argmax(-1), "C": (a0 + 1) % classes}
    for context in CONTEXTS:
        assert torch.equal(world.targets(x, context, 0), expected[context])
    a1 = world.targets(x, "A", 1)
    c0 = world.targets(x, "C", 0)
    assert torch.equal(a1, (a0 + 2) % classes)
    assert bool((a1 != a0).all())
    assert bool((c0 != a0).all())
    assert bool((a1 != c0).all())
    for context in ("B", "C"):
        assert torch.equal(world.targets(x, context, 1), world.targets(x, context, 0))


def test_b_has_real_agreement_and_conflict_and_diagnostics_stay_outside_features():
    world = _small_world(input_dim=32, n_train=4096)
    x = world.splits["train"]
    agree = world.targets(x, "A", 0) == world.targets(x, "B", 0)
    assert agree.dtype == torch.bool
    # Broad bounds detect accidental identical or entirely unrelated teachers.
    assert 0.5 < agree.float().mean().item() < 0.98
    assert torch.equal(world.diagnostics(x)["ab_agree"], agree)
    assert world.features(x, "B", 0).shape[-1] == world.input_dim + 5


def test_features_only_encode_x_context_and_observed_version(monkeypatch):
    world = _small_world()
    x = world.splits["train"][:6]
    contexts = torch.tensor([0, 1, 2, 2, 0, 1])
    versions = torch.tensor([0, 0, 0, 1, 1, 1])

    def forbidden(*args, **kwargs):
        raise AssertionError("feature construction must not request labels or diagnostics")

    monkeypatch.setattr(world, "targets", forbidden)
    monkeypatch.setattr(world, "diagnostics", forbidden)
    observed = world.features(x, contexts, versions)
    assert observed.shape == (6, world.feature_dim)
    assert torch.equal(observed[:, : world.input_dim], x)
    assert observed[:, world.input_dim : world.input_dim + 3].tolist() == [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ]
    assert observed[:, -2:].tolist() == [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3
    world.teacher_w.zero_()
    world.teacher_delta.fill_(100)
    assert torch.equal(world.features(x, contexts, versions), observed)


def test_labels_and_train_inputs_do_not_depend_on_future_data_or_model_activity():
    world = _small_world()
    changed_future = _small_world(n_val=183, n_test=29)
    assert torch.equal(world.teacher_w, changed_future.teacher_w)
    assert torch.equal(world.teacher_delta, changed_future.teacher_delta)
    assert torch.equal(world.splits["train"], changed_future.splits["train"])
    x = world.splits["train"][:31].clone()
    before = {(c, v): world.targets(x, c, v) for c in CONTEXTS for v in (0, 1)}
    rng_state = torch.random.get_rng_state().clone()
    try:
        model = torch.nn.Linear(world.feature_dim, world.classes)
        model(world.features(x, "A", 0)).square().sum().backward()
        torch.randn(89)
        # Evaluation data is never used to choose or update a teacher.
        world.splits["transfer_test"].fill_(999)
        world.splits["test"].zero_()
        for (context, version), labels in before.items():
            assert torch.equal(world.targets(x, context, version), labels)
            assert torch.equal(changed_future.targets(x, context, version), labels)
        # Returned feedback is freshly computed, rather than a mutable label table.
        world.targets(x, "A", 0).fill_(-1)
        assert torch.equal(world.targets(x, "A", 0), before[("A", 0)])
    finally:
        torch.random.set_rng_state(rng_state)


def test_independent_splits_and_float32_definition_survive_changed_global_defaults():
    world = _small_world()
    saved_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        resized_train = _small_world(n_train=514)
    finally:
        torch.set_default_dtype(saved_dtype)
    assert torch.equal(world.teacher_w, resized_train.teacher_w)
    assert torch.equal(world.teacher_delta, resized_train.teacher_delta)
    assert torch.equal(world.splits["train"], resized_train.splits["train"][:257])
    for name in ("val", "test", "transfer_val", "transfer_test"):
        assert torch.equal(world.splits[name], resized_train.splits[name])
        assert resized_train.splits[name].dtype == torch.float32


def test_tensor_indices_and_per_example_metadata_preserve_order_and_shapes():
    world = _small_world()
    indices = torch.tensor([[7, 2, 19], [7, 0, 3]])
    contexts = torch.tensor([[0, 1, 2], [2, 0, 1]])
    versions = torch.tensor([[0, 1, 0], [1, 1, 0]])
    observations, labels = world.batch("train", indices, contexts, versions)
    assert observations.shape == (2, 3, world.feature_dim)
    assert labels.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            x = world.splits["train"][indices[i, j]]
            assert torch.equal(observations[i, j], world.features(x, contexts[i, j], versions[i, j]))
            assert torch.equal(labels[i, j], world.targets(x, contexts[i, j], versions[i, j]))
    for context, index in zip(CONTEXTS, range(3)):
        assert torch.equal(world.features(world.splits["val"], context, 0),
                           world.features(world.splits["val"], index, torch.tensor(0)))
    mask = torch.arange(len(world.splits["val"])) % 3 == 0
    observed, target = world.batch("val", mask, "B", 0)
    assert observed.shape[0] == int(mask.sum())
    assert torch.equal(target, world.targets(world.splits["val"][mask], "B", 0))


def test_queries_do_not_consume_rng_and_empty_splits_work():
    world = _small_world(n_val=0, n_test=0)
    saved = torch.random.get_rng_state().clone()
    empty_features, empty_targets = world.batch("val", torch.empty(0, dtype=torch.long), "A", 0)
    assert empty_features.shape == (0, world.feature_dim)
    assert empty_targets.shape == (0,)
    world.features(world.splits["train"], "C", 1)
    world.targets(world.splits["train"], "C", 1)
    world.diagnostics(world.splits["train"])
    assert torch.equal(saved, torch.random.get_rng_state())


@pytest.mark.parametrize("kwargs", [{"classes": 2}, {"input_dim": 1}, {"n_train": -1}, {"seed": True}])
def test_invalid_world_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _small_world(**kwargs)


@pytest.mark.parametrize("context,version", [("D", 0), (-1, 0), (3, 0), ("A", 2), ("A", 0.5)])
def test_invalid_observations_are_rejected(context, version):
    world = _small_world()
    with pytest.raises(ValueError):
        world.features(world.splits["train"][:4], context, version)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_indices_work_on_cuda_world_with_reproducible_raw_data():
    cpu = _small_world()
    cuda = _small_world(device="cuda")
    assert torch.equal(cpu.teacher_w, cuda.teacher_w.cpu())
    for name in cpu.splits:
        assert torch.equal(cpu.splits[name], cuda.splits[name].cpu())
    indices = torch.tensor([8, 3, 8])
    features, targets = cuda.batch("train", indices, torch.tensor([0, 1, 2]), 0)
    assert features.device.type == targets.device.type == "cuda"
    assert torch.equal(features.cpu(), cpu.features(cpu.splits["train"][indices], torch.tensor([0, 1, 2]), 0))
