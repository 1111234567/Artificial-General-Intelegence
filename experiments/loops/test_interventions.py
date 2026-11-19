"""Synthetic graph controls and cloned observer tests; no formal runs/training."""
import json

import pytest
import torch

from experiments.continuity.model import AgentModel
from experiments.continuity.run import rng_state, state_digest
from experiments.continuity.world import OnlineWorld
from experiments.integration.diagnose import diagnose_fork
from experiments.loops.interventions import (
    INTERVENTIONS, build_interventions, clone_with_recurrent,
)


def _dense(width=8, dtype=torch.float64):
    values = torch.arange(1, width * width + 1, dtype=dtype).reshape(width, width) / width**2
    return values * torch.where(torch.arange(width)[:, None] % 2 == 0, 1, -1)


def _offdiagonal(value):
    value = value.clone()
    value.fill_diagonal_(0)
    return value


def _acyclic(weight):
    # Independent Kahn check with the model's source -> target convention.
    graph = _offdiagonal(weight).ne(0)
    incoming = graph.sum(1)
    ready = [index for index in range(len(weight)) if incoming[index] == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in graph[:, source].nonzero(as_tuple=True)[0].tolist():
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    return visited == len(weight)


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_dag_is_acyclic_and_preserves_self_recurrence(seed):
    original = _dense()
    weights, metadata = build_interventions(original, seed)
    assert tuple(weights) == INTERVENTIONS
    assert sorted(metadata["node_order"]) == list(range(len(original)))
    for name, value in weights.items():
        torch.testing.assert_close(value.diagonal(), original.diagonal(), rtol=0, atol=0)
        assert not value.requires_grad
        assert value.dtype == original.dtype and value.device == original.device
        assert value.data_ptr() != original.data_ptr()
    for name in ("diagonal_only", "dag_cut", "dag_norm_matched"):
        assert _acyclic(weights[name])
    assert not _acyclic(weights["none"])
    assert torch.equal(weights["none"], original)
    assert int(_offdiagonal(weights["dag_cut"]).ne(0).sum()) == len(original) * (len(original)-1) // 2
    reordered = weights["dag_cut"][metadata["node_order"]][:, metadata["node_order"]]
    assert torch.equal(reordered.triu(1), torch.zeros_like(reordered))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sparse", [False, True])
def test_raw_row_counts_and_normalized_row_norms_are_matched(dtype, sparse):
    original = _dense(dtype=dtype)
    if sparse:
        original[1::2, ::3] = 0
    weights, metadata = build_interventions(original, 29)
    source_support = _offdiagonal(original).ne(0)
    for left, right in (("dag_cut", "random_cut"), ("dag_norm_matched", "random_norm_matched")):
        left_value, right_value = _offdiagonal(weights[left]), _offdiagonal(weights[right])
        assert torch.equal(left_value.ne(0).sum(1), right_value.ne(0).sum(1))
        assert not bool((left_value.ne(0) & ~source_support).any())
        assert not bool((right_value.ne(0) & ~source_support).any())
    dag_norm = torch.linalg.vector_norm(_offdiagonal(weights["dag_norm_matched"]), dim=1)
    random_norm = torch.linalg.vector_norm(_offdiagonal(weights["random_norm_matched"]), dim=1)
    torch.testing.assert_close(dag_norm, random_norm)
    torch.testing.assert_close(dag_norm.double(), torch.tensor(
        metadata["common_target_offdiagonal_row_l2"], dtype=torch.float64),
        rtol=2e-7 if dtype == torch.float32 else 1e-14, atol=1e-14)
    for normalized, raw in (("dag_norm_matched", "dag_cut"), ("random_norm_matched", "random_cut")):
        assert bool((weights[normalized].abs() <= weights[raw].abs()).all())
        assert all(0 <= scale <= 1 for scale in metadata["interventions"][normalized]["row_scales"])
    for raw in ("dag_cut", "random_cut"):
        keep = weights[raw].ne(0)
        assert torch.equal(weights[raw][keep], original[keep])


def test_private_rng_determinism_detachment_json_and_no_input_aliases():
    original = _dense().requires_grad_()
    before = original.detach().clone()
    process_rng = rng_state()
    weights, metadata = build_interventions(original, 11)
    assert state_digest(process_rng) == state_digest(rng_state())
    again, repeated = build_interventions(original, 11)
    assert metadata == repeated
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert len({value.data_ptr() for value in weights.values()}) == len(weights)
    for name, value in weights.items():
        assert torch.equal(value, again[name])
    weights["none"][0, 0] = 99
    assert torch.equal(original, before)
    assert weights["dag_cut"][0, 0] == before[0, 0]
    _, another = build_interventions(original, 29)
    assert metadata["node_order"] != another["node_order"]
    assert "outgoing degree per source" in metadata["unmatched_properties"]
    assert "spectrum and spectral radius" in metadata["unmatched_properties"]


@pytest.mark.parametrize("width", [1, 4])
def test_zero_offdiagonal_rows_have_finite_unit_scale(width):
    original = torch.diag(torch.arange(width, dtype=torch.float64) - .5)
    weights, metadata = build_interventions(original, 47)
    for name, value in weights.items():
        assert torch.equal(value, original)
        assert metadata["interventions"][name]["row_scales"] == (None if name.endswith("quarter") else [1.] * width)
        assert metadata["interventions"][name]["offdiagonal_row_l2"] == [0.] * width
    json.dumps(metadata, allow_nan=False)


def test_quarter_variants_are_partial_attenuations_with_unmatched_interpolated_norms():
    original = _dense()
    weights, metadata = build_interventions(original, 11)
    for name, endpoint in (("dag_quarter", "dag_norm_matched"),
                           ("random_quarter", "random_norm_matched")):
        torch.testing.assert_close(weights[name], .75 * original + .25 * weights[endpoint], rtol=0, atol=0)
        assert torch.equal(weights[name].ne(0), original.ne(0))
        assert not _acyclic(weights[name])
        assert bool((weights[name].abs() <= original.abs()).all())
        record = metadata["interventions"][name]
        assert record["interpolation_alpha"] == .25
        assert record["interpolation_endpoint"] == endpoint
        assert record["row_scales"] is None
        assert not record["distinct_node_acyclic_by_construction"]
        assert not record["interpolated_row_l2_matching_claimed"]
    assert not torch.allclose(torch.linalg.vector_norm(_offdiagonal(weights["dag_quarter"]), dim=1),
                              torch.linalg.vector_norm(_offdiagonal(weights["random_quarter"]), dim=1))


@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.zeros(0, 0), torch.zeros(3),
                                 torch.ones(2, 2, dtype=torch.int64),
                                 torch.full((2, 2), torch.nan), torch.full((2, 2), torch.inf)])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        build_interventions(bad, 11)


@pytest.mark.parametrize("seed", [True, -1, 1.5, 2**63])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError):
        build_interventions(_dense(), seed)


def test_clone_and_frozen_observer_preserve_full_source_state_and_history(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(513)
        world = OnlineWorld(seed=71, batch_size=2, n_jobs=2, dt=.25)
        model = AgentModel(input_size=world.input_size, n_jobs=world.n_jobs, width=8)
        optimizer = torch.optim.Adam(model.parameters())
        hidden = model.initial_state(world.batch_size)
        for _ in range(4):
            with torch.no_grad():
                logits, hidden = model.step(world.observe(), hidden, world.elapsed)
            world.act(logits.argmax(-1))
        model.train()
        # Synthetic retained optimization/replay history, with no training run.
        for parameter in model.parameters():
            optimizer.state[parameter] = dict(
                step=torch.tensor(7.), exp_avg=torch.full_like(parameter, .1),
                exp_avg_sq=torch.full_like(parameter, .2))
        model.recurrent.grad = torch.full_like(model.recurrent, .3)
        saved_gradient = model.recurrent.grad.clone()
        replay = [dict(hidden=hidden.clone(), world=world.state_dict(), tag="saved-live-boundary")]
        source = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), hidden=hidden,
                      world=world.state_dict(), replay=replay, rng=rng_state())
        checkpoint = tmp_path / "synthetic-source.pt"
        torch.save(source, checkpoint)
        before_file = checkpoint.read_bytes()
        before_source = state_digest(source)
        before_rng = state_digest(rng_state())
        weights, _ = build_interventions(model.recurrent, 11)
        for name in ("none", "dag_norm_matched", "random_norm_matched"):
            clone = clone_with_recurrent(model, weights[name])
            assert clone is not model and not clone.training
            assert all(not parameter.requires_grad for parameter in clone.parameters())
            assert all(parameter.grad is None for parameter in clone.parameters())
            assert clone.config() == model.config()
            for key, value in model.state_dict().items():
                assert clone.state_dict()[key].data_ptr() != value.data_ptr()
                expected = weights[name] if key == "recurrent" else value
                assert torch.equal(clone.state_dict()[key], expected)
            observed = diagnose_fork(clone, world, hidden, steps=12, condition="iid")
            assert observed["start_hidden_hash"] == state_digest({"hidden": hidden})
            assert observed["start_world_state_hash"] == state_digest(world.state_dict())
            assert observed["action_hash"] == observed["model_action_hash"]
            assert observed["hidden_resets"] == observed["world_resets"] == observed["parameter_updates"] == 0
            assert observed["opportunities"]["model_action_changes"] == 0
            assert model.training and model.recurrent.requires_grad
            assert torch.equal(model.recurrent.grad, saved_gradient)
            assert before_source == state_digest(source)
            assert before_file == checkpoint.read_bytes()
            assert before_rng == state_digest(rng_state())


def test_clone_rejects_missing_shape_or_dtype_mismatched_recurrence():
    with torch.random.fork_rng(devices=[]):
        dense = AgentModel(width=8)
        gru = AgentModel(width=8, kind="gru")
    for model, weight in ((gru, _dense(dtype=torch.float32)), (dense, _dense(4, torch.float32)),
                          (dense, _dense(dtype=torch.float64))):
        with pytest.raises(ValueError):
            clone_with_recurrent(model, weight)
