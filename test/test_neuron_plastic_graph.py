"""Exercise the existing plastic graph backends from their shared entry point."""
import sys

import pytest
import torch

from experiments.undirected.evolved_plasticity import RULE_FEATURES
from src.neurons.plastic_graph import (
    PLASTIC_FIELDS, CreditConfig, FastLocalLife, FastLocalTrainingEvaluator,
    LocalRuleEvaluator, LocalRulePolicy, PortGraphModel, ResidentLocalRuleEvaluator,
    TemporalCreditEngine, default_plastic, make_local_policy, snapshot_parameters, stack_local_birth,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _genome():
    nodes = 96
    arcs = nodes * (nodes - 1)
    weights = torch.zeros(arcs)
    weight_mask = torch.zeros(arcs, dtype=torch.bool)
    weights[0] = .1
    weight_mask[0] = True
    plastic = default_plastic()
    plastic[PLASTIC_FIELDS.index("weight_pre")] = 1.
    return dict(birth_w=weights, birth_mask=weight_mask, node_bias=torch.zeros(nodes),
                rule=torch.zeros(2, len(RULE_FEATURES)), directed=torch.tensor(True),
                reward_rate=torch.tensor(0.), plastic=plastic)


@pytest.mark.parametrize("arc_kernel", ("tiled", "compact"))
def test_tick_updates_on_cpu_with_optional_arc_kernel(arc_kernel):
    birth = stack_local_birth([_genome()], device="cpu", arm="local")
    expected = LocalRulePolicy(birth, 1, "cpu", "local", backend="torch")
    actual = make_local_policy(birth, 1, "cpu", "local", arc_kernel=arc_kernel)
    pulse = torch.zeros(actual.input_shape)
    pulse[..., 1] = 1.
    start = actual.weight_state.clone()
    for obs in (pulse, torch.zeros_like(pulse), torch.zeros_like(pulse)):
        torch.testing.assert_close(actual.act(obs), expected.act(obs), rtol=0, atol=0)
    torch.testing.assert_close(actual.state_tensors(), expected.state_tensors(), rtol=0, atol=0)
    assert actual.clock.item() == 3
    assert actual.h.dtype == actual.w.dtype == torch.float16
    assert actual.weight_state.dtype == torch.float32
    assert not torch.equal(actual.weight_state, start)


def test_training_evaluators_fall_back_to_cpu():
    genomes = [_genome()]
    settings = dict(arm="modulated", seed=107, actors=4, visits=4)
    reference = LocalRuleEvaluator("cpu", "training")(genomes, **settings)
    resident = ResidentLocalRuleEvaluator("cpu", "training", requests_per_launch=2)
    fast = FastLocalTrainingEvaluator("cpu", "training", backend="life", requests=2)
    assert resident(genomes, **settings) == fast(genomes, **settings) == reference
    assert "selection_score" in reference[0]


@pytest.mark.skipif(sys.platform != "win32", reason="bundled NVRTC requires Windows CUDA")
def test_resident_and_compact_evaluators_replay_sealed_cuda_state():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires CUDA Tensor Cores")
    genomes = [_genome()]
    settings = dict(arm="modulated", seed=108, actors=4, visits=4)
    reference = LocalRuleEvaluator("cuda", "training")
    resident = ResidentLocalRuleEvaluator("cuda", "training", requests_per_launch=2)
    compact = FastLocalTrainingEvaluator("cuda", "training", backend="compact", requests=2)
    expected = reference(genomes, **settings)
    assert resident(genomes, **settings) == compact(genomes, **settings) == expected
    resident_replay = next(iter(resident.replays.values()))
    assert isinstance(resident_replay, FastLocalLife)
    expected_replay = next(iter(reference.replays.values()))
    for replay in (resident_replay, next(iter(compact.replays.values()))):
        torch.testing.assert_close(replay.policy.state_tensors(), expected_replay.policy.state_tensors(),
                                   rtol=0, atol=0)


def test_recorded_window_and_rtrl_match_across_one_optimizer_change():
    model = PortGraphModel(input_size=2, n_jobs=1, n_actions=2, width=5,
                           input_scale=(1., 1.)).double()
    model.initialize_birth(degree=2, gain=.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    rtrl = TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=2))
    recorded = TemporalCreditEngine(model, CreditConfig("recorded512", update_ticks=2))
    generator = torch.Generator().manual_seed(17)
    state = model.initial_state(1)
    for _ in range(2):
        observations = torch.randn(2, 1, 2, dtype=state.dtype, generator=generator)
        targets = torch.ones(2, 1, 1, dtype=torch.long)
        elapsed = (.25, .25)
        start_state = state
        parameters = snapshot_parameters(model)
        with torch.no_grad():
            for obs, interval in zip(observations, elapsed, strict=True):
                _, state = model(obs, state, interval)
        chunk = dict(observations=observations, targets=targets, elapsed=elapsed,
                     start_h=start_state, end_h=state)
        gradients, diagnostics = rtrl.gradients(chunk, parameters)
        reference, _ = recorded.gradients(chunk, parameters)
        torch.testing.assert_close(gradients, reference, rtol=1e-10, atol=1e-11)
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        optimizer.step()
    assert rtrl.updates == recorded.updates == 2
    assert diagnostics["gradient_diagnostics"]["cross_boundary_gradient_l2"] > 0.
