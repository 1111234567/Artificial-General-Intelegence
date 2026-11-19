"""Compare the resident kernel with the independently launched sealed CUDA path."""

import sys

import pytest
import torch

from . import local_rule_evaluator as sealed
from . import uncapped_private_symbol_world as private
from .evolved_plasticity_world import FIRST_FINISH, _receipt_table
from .fast_local_life import FastLocalLife
from .local_rule_policy import INPUTS, JOBS
from .test_local_rule_policy import _genome


def _assert_bits(actual, expected, name):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                       expected.contiguous().reshape(-1).view(torch.uint8)), name


@torch.no_grad()
def _sealed_run(policy, birth, tape, pairs):
    requests, ticks, actors = tape.stage.shape
    pop, batch = policy.population, policy._batch
    opts = {"device": policy.device}
    frame, stage, target = (value[0].clone() for value in (tape.public, tape.stage, tape.target))
    receipts = torch.zeros((pop, actors, 2), dtype=torch.long, **opts)
    flags = torch.zeros((6, pop, actors), dtype=torch.bool, **opts)
    counts = torch.zeros((8, pop, actors), dtype=torch.int32, **opts)
    outcome = torch.empty((6, 1, pop, actors), dtype=torch.bool, **opts)
    outcomes = torch.empty((6, requests, pop, actors), dtype=torch.bool, **opts)
    actions = torch.empty((ticks, pop, actors, JOBS), dtype=torch.long, **opts)
    all_actions = torch.empty((requests, *actions.shape), dtype=torch.long, **opts)
    obs = torch.empty((pop, actors, INPUTS), dtype=torch.float16, **opts)
    table = _receipt_table(policy.device).half()
    kernels = private._kernels(policy.device)

    def request():
        for tick in range(ticks):
            kernels["directed_world_observation"](
                ((batch * INPUTS + 127) // 128,), (128,), frame[tick], table,
                receipts, flags, obs, actors, batch, int(tick == 0))
            action = policy.act(obs)
            actions[tick].copy_(action)
            kernels["directed_world_action"](
                ((batch + 127) // 128,), (128,), action, stage[tick], target, counts, receipts,
                flags, outcome, batch, actors, 1, 0, int(tick == 0), int(tick == ticks - 1), *action.stride())

    request()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        request()
    policy.reset(birth)
    for state in (counts, receipts, flags):
        state.zero_()
    for request_index in range(requests):
        frame.copy_(tape.public[request_index])
        stage.copy_(tape.stage[request_index])
        target.copy_(tape.target[request_index])
        graph.replay()
        all_actions[request_index].copy_(actions)
        outcomes[:, request_index].copy_(outcome[:, 0])

    sums = counts.sum(-1)
    metrics = {**policy.summary(), "idle": sums[0] / sums[1].clamp_min(1),
               **sealed.trace_statistics(all_actions[..., 0], tape, pairs)}
    return metrics, all_actions, outcomes, counts, flags, receipts


@torch.no_grad()
def verify(genomes, arm, *, actors=8, visits=16, seed=20261119, training=True, spans=(8,)):
    """Check raw dual-port actions, exact scores and the entire final CUDA state."""
    device = torch.device("cuda", torch.cuda.current_device())
    cpu = sealed.world.tape_on_cpu(seed, actors, visits, training)
    tape = sealed._convert(cpu, device)
    pairs = sealed.world.partners(cpu.raw, cpu.context).to(device)
    birth = sealed.stack_birth(genomes, device, arm)
    expected = sealed.make_policy(genomes, actors, device, arm)
    metrics, actions, outcomes, counts, flags, receipts = _sealed_run(expected, birth, tape, pairs)
    actual = sealed.make_policy(genomes, actors, device, arm)
    for span in spans:
        replay = FastLocalLife(actual, span, capture_world=True)
        actual_metrics = replay.run(birth, tape, pairs)
        _assert_bits(replay.trace, actions[..., 0], "primary trace")
        _assert_bits(replay.spare_trace, actions[..., 1], "spare trace")
        _assert_bits(replay.outcomes, outcomes, "outcomes")
        _assert_bits(replay.world_counts, counts, "world counts")
        _assert_bits(replay.world_flags, flags, "world flags")
        _assert_bits(replay.world_receipts, receipts, "world receipts")
        for name, reference in expected.state_tensors().items():
            _assert_bits(actual.state_tensors()[name], reference, name)
        for name in ("_local_activity", "_gain", "_receipts"):
            _assert_bits(getattr(actual, name), getattr(expected, name), name)
        assert actual_metrics.keys() == metrics.keys()
        for name, reference in metrics.items():
            _assert_bits(actual_metrics[name], reference, name)
        for idx in range(len(genomes)):
            before = {key: val[idx].item() for key, val in metrics.items()}
            after = {key: val[idx].item() for key, val in actual_metrics.items()}
            assert sealed.score(after, cpu.raw.numel()) == sealed.score(before, cpu.raw.numel())
    return {"requests": tape.stage.size(0), "ticks": expected.clock.item(), "raw_actions": actions.numel(),
            "births": int(metrics["births"].sum()), "deaths": int(metrics["deaths"].sum()),
            "public_ack_events": int(metrics["public_ack_events"].sum()),
            "public_rej_events": int(metrics["public_rej_events"].sum()),
            "metric_tensors": len(metrics), "state_tensors": len(expected.state_tensors()) + 3}


@pytest.mark.skipif(sys.platform != "win32" or not torch.cuda.is_available()
                    or torch.cuda.get_device_capability()[0] < 8, reason="requires Windows CUDA Tensor Cores")
@pytest.mark.parametrize("arm", ("local", "modulated"))
def test_resident_life_matches_sealed_cuda_bit_for_bit(arm):
    genomes = [_genome(96, gate_bias=.82, weight_pre=.17, weight_ack=.63, bias_trace=.11, mod_activity=.25),
               _genome(96, gate_bias=-.9, weight_post=-.13, bias_rej=-.55, mod_trace=.2)]
    random = torch.Generator(device="cpu").manual_seed(3183)
    for genome in genomes:
        genome["birth_mask"].copy_(torch.rand(genome["birth_mask"].shape, generator=random).lt(.08))
        genome["birth_w"].copy_(.04 * torch.randn(genome["birth_w"].shape, generator=random))
        genome["node_bias"].copy_(.03 * torch.randn(96, generator=random))
        genome["node_bias"][INPUTS + FIRST_FINISH] = .5
        genome["node_bias"][INPUTS + 11 + 1] = .3
        genome["reward_rate"].fill_(.02)
        genome["rule"][0, 0] = -.3
        genome["rule"][0, 6] = .3
        genome["rule"][1, 0] = .01
        genome["rule"][1, 6] = .05
    result = verify(genomes, arm, actors=4, visits=4, spans=(3, 8, 32))
    assert all(result[key] > 0 for key in ("births", "deaths", "public_ack_events", "public_rej_events"))
