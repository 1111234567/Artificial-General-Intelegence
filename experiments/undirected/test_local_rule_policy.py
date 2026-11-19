import sys

import pytest
import torch

from .evolved_plasticity import INPUTS, RULE_FEATURES
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, READY
from .local_rule_policy import PLASTIC_FIELDS, LocalRulePolicy, default_plastic
from .uncapped_directed_reward import _stack_directed_genomes
from .uncapped_reward_precision import _ACK, _REJ


@pytest.fixture(autouse=True)
def _single_thread_reference():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(count)


def _genome(nodes=78, **coefficients):
    arcs = nodes * (nodes - 1)
    genome = {
        "birth_w": torch.zeros(arcs),
        "birth_mask": torch.zeros(arcs, dtype=torch.bool),
        "node_bias": torch.zeros(nodes),
        "rule": torch.zeros((2, len(RULE_FEATURES))),
        "directed": torch.tensor(True),
        "reward_rate": torch.tensor(0.),
        "plastic": torch.zeros_like(default_plastic()),
    }
    for name, coeff in coefficients.items():
        genome["plastic"][PLASTIC_FIELDS.index(name)] = coeff
    genome["birth_mask"][::arcs // 2] = True
    genome["birth_w"][::arcs // 2] = .1
    return genome


def _stack(*genomes, device="cpu"):
    _, birth = _stack_directed_genomes(genomes)
    birth["plastic"] = torch.stack([g["plastic"] for g in genomes])
    return {name: value.to(device) for name, value in birth.items()}


def test_local_pre_trace_is_directional_and_receiver_gain_can_reverse_updates():
    genome = _genome(weight_pre=1., mod_activity=1.)
    genome["node_bias"][0] = -.8
    birth = _stack(genome)
    local = LocalRulePolicy(birth, 1, "cpu", "local")
    modulated = LocalRulePolicy(birth, 1, "cpu", "modulated")
    pulse = torch.zeros(1, 1, INPUTS)
    pulse[..., 1] = 1.
    for policy in (local, modulated):
        policy.act(pulse)
        policy.act(torch.zeros_like(pulse))

    anchor = local.initial_w[0, 0, 0, 0]
    assert local.weight_state[0, 0, 0, 0] > anchor
    assert local.weight_state[0, 0, 1, 0] < anchor
    assert modulated.weight_state[0, 0, 0, 0] < anchor
    assert modulated.weight_state[0, 0, 1, 0] < anchor


def test_public_receipt_learning_changes_weights_bias_and_gate_without_pending():
    genome = _genome(weight_ack=1., weight_rej=-1., bias_ack=1., bias_rej=-1., gate_ack=1., gate_rej=-1.)
    policy = LocalRulePolicy(_stack(genome), 4, "cpu", "local")
    frame = torch.zeros(1, 4, INPUTS)
    frame[..., :2] = 1.
    policy.act(frame)
    assert not policy.pending.any()
    reply = torch.zeros_like(frame)
    reply[0, 0, _ACK] = 1.
    reply[0, 1, _REJ] = 1.
    reply[0, 2, FEATURES_PER_SLOT + _REJ] = 1.
    policy.act(reply)

    for state in (policy.weight_state[0, :, 0, 0], policy.bias[0, :, 0], policy.g[0, :, 0, 0]):
        assert state[1] < state[2] < state[3] < state[0]
    summary = policy.summary()
    assert summary["reward_events"].item() == 0
    assert summary["public_ack_events"].item() == 1
    assert summary["public_rej_events"].item() == 2


def test_support_evidence_accumulates_and_effective_weight_ramps_both_ways():
    growing = _genome(gate_bias=.1)
    growing["rule"][1, 0] = 1.5
    growing["rule"][0, 0] = .5
    birth = _stack(growing, _genome(gate_bias=-1.))
    policy = LocalRulePolicy(birth, 1, "cpu", "local")
    frame = torch.zeros(2, 1, INPUTS)
    for _ in range(35):
        policy.act(frame)
    assert not policy.births.any() and not policy.deaths.any()
    assert 0. < policy.g[0, 0, 0, 1] < .45
    while policy.clock < 65:
        policy.act(frame)
    assert policy.mask[0, 0, 0, 1] and policy.mask[1, 0, 0, 0]
    new_strength = policy.strength[0, 0, 0, 1].item()
    fading_strength = policy.strength[1, 0, 0, 0].item()
    assert 0. < new_strength < 1. and 0. < fading_strength < 1.
    assert policy.weight_state[0, 0, 0, 1].item() == (.08 * torch.tensor(.5).tanh() + .01).half().item()
    assert policy.w[0, 0, 0, 1] < policy.weight_state[0, 0, 0, 1]
    assert policy.w[1, 0, 0, 0] < policy.weight_state[1, 0, 0, 0]
    policy.act(frame)
    assert policy.strength[0, 0, 0, 1].item() == new_strength + 1. / 32.
    assert policy.strength[1, 0, 0, 0].item() == fading_strength - 1. / 32.
    while policy.clock < 100:
        policy.act(frame)
    assert policy.strength[0, 0, 0, 1].item() == 1.
    assert not policy.mask[1, 0, 0, 0]
    assert policy.w[1, 0, 0, 0].item() == 0.
    assert policy.deaths[1, 0].item() == 2


@pytest.mark.skipif(not torch.cuda.is_available() or sys.platform != "win32", reason="bundled NVRTC requires Windows CUDA")
def test_cuda_tensor_core_matches_reference_and_one_request_graph_replays_state():
    genomes = [_genome(96, gate_bias=.9, weight_pre=.1, weight_ack=.6, bias_trace=.1, mod_activity=.2),
               _genome(96, gate_bias=-.9, weight_post=-.1, bias_rej=-.6, mod_trace=.2)]
    for g in genomes:
        g["node_bias"][INPUTS + FIRST_FINISH] = .3
        g["reward_rate"].fill_(.02)
        g["rule"][0, 0] = -.3
        g["rule"][0, 6] = .3
        g["rule"][1, 0] = .01
        g["rule"][1, 6] = .05
    birth = _stack(*genomes, device="cuda")
    reference = LocalRulePolicy(_stack(*genomes), 2, "cpu", "modulated")
    actual = LocalRulePolicy(birth, 2, "cuda", "modulated", backend="cuda")
    frames = torch.zeros(18, 2, 2, INPUTS, dtype=torch.float16)
    frames[0, ..., :2] = 1.
    frames[0, ..., READY] = 1.
    frames[1, ..., _REJ] = 1.
    frames[2, ..., _ACK] = 1.
    frames[2, ..., 14] = 1.
    frames[3, ..., FEATURES_PER_SLOT + _REJ] = 1.
    device_frames = frames.cuda()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for frame in device_frames:
            actual.act(frame)
        actual.reset(birth)
        initial = {name: state.clone() for name, state in actual.state_tensors().items()}
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actions = torch.stack([actual.act(frame) for frame in device_frames])
    for _ in range(6):
        graph.replay()
        assert torch.equal(actions.cpu(), torch.stack([reference.act(frame) for frame in frames]))
    assert actual.clock.item() == reference.clock.item() == 108
    for name, expected in reference.state_tensors().items():
        torch.testing.assert_close(actual.state_tensors()[name].cpu(), expected, atol=2.e-6, rtol=1.e-5, msg=name)
    for name, expected in reference.summary().items():
        torch.testing.assert_close(actual.summary()[name].cpu(), expected, atol=2.e-6, rtol=1.e-5, msg=name)

    first_replay = None
    for _ in range(2):
        for name, state in actual.state_tensors().items():
            state.copy_(initial[name])
        graph.replay()
        snapshot = {name: value.clone() for name, value in actual.state_tensors().items()}
        if first_replay is not None:
            for name, expected in first_replay.items():
                assert torch.equal(snapshot[name], expected), name
        first_replay = snapshot
