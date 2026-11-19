import sys

import pytest
import torch

from . import local_rule_evaluator as sealed, local_rule_world as world
from .fast_local_arcs_compact import install
from .fast_training_evaluator import RequestChunks
from .test_fast_local_life import _assert_bits, _sealed_run
from .test_local_rule_policy import _genome


@pytest.mark.skipif(sys.platform != "win32" or not torch.cuda.is_available(), reason="Windows CUDA backend")
@pytest.mark.parametrize("arm", ["local", "modulated"])
@torch.no_grad()
def test_compact_both_ports_states_and_chunk_replay(arm):
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("requires Tensor Core arc specialization")
    genomes = [_genome(96, gate_bias=sign, weight_coactivity=.6, bias_ack=.1, mod_activity=.4)
               for sign in (1., -1.)]
    for genome in genomes:
        genome["birth_mask"][::7] = True
        genome["birth_w"][::7] = .13
    actors = 4
    birth = sealed.stack_birth(genomes, device, arm)
    before = sealed.make_policy(genomes, actors, device, arm)
    after = install(sealed.make_policy(genomes, actors, device, arm))
    replay = None
    for seed in (107, 124):
        tape = sealed._convert(world.tape_on_cpu(seed, actors, 4, True), device)
        pairs = world.partners(tape.raw, tape.context)
        expected, actions, _, counts, flags, receipts = _sealed_run(before, birth, tape, pairs)
        observed, trial_actions, *_ = _sealed_run(after, birth, tape, pairs)
        _assert_bits(trial_actions, actions, "both action ports")
        if replay is None:
            replay = RequestChunks(after, tape, 4)
        result = replay.run(birth, tape, pairs)
        for key, value in expected.items():
            _assert_bits(observed[key], value, key)
            _assert_bits(result[key], value, key)
        _assert_bits(replay.trace, actions[..., 0], "chunk primary actions")
        for name, value in (("counts", counts), ("flags", flags), ("receipts", receipts)):
            _assert_bits(getattr(replay, name), value, name)
        for key, value in before.state_tensors().items():
            _assert_bits(after.state_tensors()[key], value, key)
        assert result["births"][0] > 0 and result["deaths"][1] > 0
