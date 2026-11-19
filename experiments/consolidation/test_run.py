import tempfile
from collections import Counter
from pathlib import Path

import torch

from experiments.consolidation.model import Predictor
from experiments.consolidation.run import Config, CONDITIONS, evaluate, run_one, schedule, tensor_digest
from experiments.consolidation.world import World


def test_schedules_have_identical_exposure_counts():
    blocked, recurrent = schedule('blocked'), schedule('recurrent')
    assert Counter(blocked) == Counter(recurrent) == {'A0': 6, 'B0': 9, 'C0': 9, 'A1': 3}
    assert 'A1' not in blocked[:18] and 'A1' not in recurrent[:18]
    assert set(CONDITIONS) == {'A0', 'A1', 'B0', 'C0'}


def test_evaluation_cannot_train_or_change_parameters():
    world = World(8, input_dim=4, n_train=40, n_val=12, n_test=13)
    model = Predictor(9, 4, 8, 4)
    before = tensor_digest(model.state_dict())
    metrics = evaluate(model, world, 'val', chunk=5)
    assert before == tensor_digest(model.state_dict())
    assert set(metrics) == {'val', 'transfer_val'}
    assert all(p.grad is None for p in model.parameters())


def test_paired_external_stream_compute_and_memory_accounting():
    config = Config(input_dim=4, width=8, n_train=40, n_val=12, n_test=13,
                    batch=4, capacity=12, steps_per_block=2, rounds=2,
                    revision_rounds=1, device='cpu', threads=1)
    world = World(8, input_dim=4, n_train=40, n_val=12, n_test=13)
    with tempfile.TemporaryDirectory() as tmp:
        results = {}
        for order in ('blocked', 'recurrent'):
            for method in ('online', 'repeat', 'replay'):
                results[order, method] = run_one(config, world, 8, method, order, Path(tmp) / (order + method))
        for result in results.values():
            assert result['new_examples'] == result['buffer_seen'] == 72
            assert result['buffer_size'] == 12
            assert result['external_stream_hashes'] == results['blocked', 'online']['external_stream_hashes']
            assert result['initial_hash'] == results['blocked', 'online']['initial_hash']
            assert result['optimizer_steps'] == (18 if result['method'] == 'online' else 36)
            assert result['buffer_hash'] == results[result['order'], 'online']['buffer_hash']
        assert results['blocked', 'replay']['replayed_examples'] == 68
        assert results['blocked', 'replay']['trajectory'][-1]['after_age_external_steps']['C0'] == 0


def test_initialization_does_not_change_rng():
    state = torch.random.get_rng_state()
    Predictor(9, 4, 8, 99)
    assert torch.equal(state, torch.random.get_rng_state())
