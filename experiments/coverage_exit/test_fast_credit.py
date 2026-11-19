"""Scoped FP64 numerical and state-preservation checks for fast block RTRL."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.coverage_exit.fast_credit import (
    FastTemporalCreditEngine, analytic_block_jacobians, install_fast_credit,
)
from experiments.undirected.credit import CreditConfig, TemporalCreditEngine, state_digest
from experiments.undirected.model import PortGraphModel, UndirectedModel
from experiments.undirected.test_credit import block
from experiments.undirected.run import NativeConfig, NativeTrainer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_for(interface, mode):
    cls = PortGraphModel if interface == "ports" else UndirectedModel
    model = cls(input_size=3, n_jobs=1, n_actions=2, width=6, mode=mode,
                input_scale=(1., .25, 2.)).double()
    generator = torch.Generator().manual_seed(321)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * .12)
    return model


def tensor_difference(actual, expected, atol=3e-12, rtol=3e-11):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("interface", ["ports", "core"])
@pytest.mark.parametrize("mode", ["undirected", "untied"])
@pytest.mark.parametrize("scale", [1., 1e150])
def test_full_j_k_multi_substep_zero_dt_and_large_finite(interface, mode, scale):
    model = model_for(interface, mode)
    h = torch.linspace(-.5, .5, 12, dtype=torch.float64).reshape(2, 6) * scale
    chunk, snapshot = block(model, h, 0, 4, elapsed=(0., .125, .625, .25))
    native = TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=4))
    record = native._record(chunk, snapshot)
    before = state_digest((model.state_dict(), record))
    expected = native._block_jacobians(record)
    actual = analytic_block_jacobians(native, record)
    for first, second in zip(actual, expected):
        tensor_difference(first, second)
        assert bool(torch.isfinite(first).all())
    assert state_digest((model.state_dict(), record)) == before


@pytest.mark.parametrize("mode", ["undirected", "untied"])
def test_input_is_a_post_flow_pulse_and_all_potential_edges_differentiate(mode):
    model = model_for("ports", mode)
    with torch.no_grad():
        model.edge_shared.zero_()
        if mode == "untied":
            model.edge_skew.zero_()
    h = model.initial_state(1)
    # First pulse follows dt=.25. It cannot affect that tick's edge derivative.
    observations = torch.ones(1, 1, 3, dtype=torch.float64)
    chunk, snapshot = block(model, h, 0, 1, observations=observations, elapsed=(.25,))
    native = TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=1))
    _, first_k = analytic_block_jacobians(native, native._record(chunk, snapshot))
    assert torch.equal(first_k, torch.zeros_like(first_k))
    second, snapshot = block(model, chunk["end_h"], 0, 1,
                             observations=torch.zeros_like(observations), elapsed=(.25,))
    _, second_k = analytic_block_jacobians(native, native._record(second, snapshot))
    assert torch.count_nonzero(second_k) > 0  # Dormant potential edges included.
    tensor_difference(second_k, native._block_jacobians(native._record(second, snapshot))[1])


@pytest.mark.parametrize("interface", ["ports", "core"])
@pytest.mark.parametrize("mode", ["undirected", "untied"])
@pytest.mark.parametrize("method", ["rtrl", "rtrl_reset"])
def test_changed_parameters_next_s_and_live_gradients(interface, mode, method):
    model = model_for(interface, mode)
    config = CreditConfig(method, update_ticks=4)
    native, fast = TemporalCreditEngine(model, config), FastTemporalCreditEngine(model, config)
    h = model.initial_state(2)
    for step in range(3):
        chunk, snapshot = block(model, h, step * 4, 4)
        actual, metrics = fast.gradients(chunk, snapshot)
        expected, _ = native.gradients(chunk, snapshot)
        assert metrics["work"]["jacobian_reverse_chunks"] == 0
        assert metrics["work"]["analytic_block_reverse_substeps"] > 0
        assert fast.last_metrics == metrics
        for name in expected:
            tensor_difference(actual[name], expected[name])
        tensor_difference(fast.sensitivity, native.sensitivity)
        assert state_digest(fast.tape) == state_digest(native.tape)
        h = chunk["end_h"]
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.001 * (step + 1))


@pytest.mark.parametrize("scale", [1e150, 1e-150])
def test_full_large_or_tiny_previous_s_keeps_native_gradient_handling(scale):
    model = model_for("ports", "untied")
    config = CreditConfig("rtrl", update_ticks=3)
    native, fast = TemporalCreditEngine(model, config), FastTemporalCreditEngine(model, config)
    chunk, snapshot = block(model, model.initial_state(2), 0, 3)
    full_s = torch.linspace(-1., 1., 2 * 6 * native.dynamic_parameter_count,
                            dtype=torch.float64).reshape(2, 6, -1) * scale
    native.sensitivity, fast.sensitivity = full_s.clone(), full_s.clone()
    expected, _ = native.gradients(chunk, snapshot)
    actual, _ = fast.gradients(chunk, snapshot)
    for name in expected:
        tensor_difference(actual[name], expected[name], atol=1e-155, rtol=3e-11)
    tensor_difference(fast.sensitivity, native.sensitivity, atol=3e-12, rtol=3e-11)


def tiny_config(interface, mode):
    return NativeConfig(seed=17, interface=interface, mode=mode, width=40,
                        n_jobs=1, batch_size=2, chunk_ticks=4, initial_degree=4,
                        main_ticks=32, withdrawal_ticks=8, return_ticks=8,
                        warmup_ticks=8, macrocycle_ticks=16, block_ticks=4,
                        replay_capacity=2, clip=.01)


@pytest.mark.parametrize("interface", ["ports", "core"])
@pytest.mark.parametrize("mode", ["undirected", "untied"])
def test_trainer_adapter_aux_combination_clipping_adam_actions_and_resume(interface, mode):
    starter = NativeTrainer(tiny_config(interface, mode))
    starter.train_update()  # Nonzero S/Adam state for calibrated matched input.
    state = starter.state_dict()
    native = NativeTrainer.from_state_dict(deepcopy(state))
    fast = install_fast_credit(NativeTrainer.from_state_dict(deepcopy(state)))
    old_s = fast.credit.sensitivity.clone()
    assert torch.equal(old_s, starter.credit.sensitivity)
    captured = {"native": {}, "fast": {}}
    for label, trainer in (("native", native), ("fast", fast)):
        original_credit = trainer.credit.gradients
        original_aux = trainer._auxiliary_gradients
        original_apply = trainer._apply_gradients
        def credit(*args, original=original_credit, label=label, **kwargs):
            result = original(*args, **kwargs)
            captured[label]["live"] = result[0]
            return result
        def auxiliary(*args, original=original_aux, label=label, **kwargs):
            result = original(*args, **kwargs)
            captured[label]["aux"] = result[0]
            return result
        def apply(gradients, *args, original=original_apply, label=label, **kwargs):
            captured[label]["combined"] = deepcopy(gradients)
            return original(gradients, *args, **kwargs)
        trainer.credit.gradients = credit
        trainer._auxiliary_gradients = auxiliary
        trainer._apply_gradients = apply
        trainer.train_update()
    for kind in ("live", "aux", "combined"):
        for name in captured["native"][kind]:
            tensor_difference(captured["fast"][kind][name], captured["native"][kind][name])
    for first, second in zip(fast.model.parameters(), native.model.parameters()):
        tensor_difference(first, second)
    for key, expected in native.optimizer.state_dict()["state"].items():
        for name, value in expected.items():
            tensor_difference(fast.optimizer.state_dict()["state"][key][name], value)
    assert fast.action_hash == native.action_hash
    assert fast.data_hash == native.data_hash
    assert state_digest(fast.world.state_dict()) == state_digest(native.world.state_dict())
    assert torch.equal(fast.h, native.h)  # Live forward path is untouched.
    assert state_digest(fast.last_auxiliary) == state_digest(native.last_auxiliary)
    assert fast.trace[-1]["update_metrics"]["clip_scale"] < 1.
    assert fast.trace[-1]["update_metrics"]["clip_scale"] == pytest.approx(
        native.trace[-1]["update_metrics"]["clip_scale"], abs=1e-14)
    resumed = install_fast_credit(NativeTrainer.from_state_dict(fast.state_dict()))
    assert torch.equal(resumed.credit.sensitivity, fast.credit.sensitivity)


def test_unrecognized_float32_uses_native_fallback_and_opt_in_refuses_it():
    model = model_for("ports", "undirected").float()
    fast = FastTemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=1))
    chunk, snapshot = block(model, model.initial_state(1), 0, 1)
    record = fast._record(chunk, snapshot)
    expected = TemporalCreditEngine(model, fast.config)._block_jacobians(record)
    for first, second in zip(fast._block_jacobians(record), expected):
        assert torch.equal(first, second)
    with pytest.raises(ValueError, match="FP64"):
        install_fast_credit(SimpleNamespace(model=model, credit=TemporalCreditEngine(model, fast.config)))
