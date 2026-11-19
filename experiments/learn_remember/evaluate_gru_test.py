"""Check the new evidence scorer against the frozen native evaluator."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.continuity.run import state_digest
from experiments.learn_remember.gru_train import GRUAgent
from experiments.learn_remember.evaluate_gru import rollout, score_arrays
from experiments.lifelong.evaluate import evaluate
from experiments.lifelong.world import ContextWorld


@pytest.mark.parametrize('condition', ['short', 'long_both'])
def test_matches_frozen_native_evaluator_without_source_mutation(condition):
    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = GRUAgent(width=16).eval()
    world = ContextWorld(seed=100053, actor_ids=[10000, 10001], batch_size=2, n_jobs=2)
    hidden = model.initial_state(2)
    with torch.inference_mode():
        for _ in range(32):
            _, hidden = model.step(world.observe(), hidden, world.elapsed)
            world.act(torch.zeros((2, 2), dtype=torch.long))
    before = state_digest(dict(model=model.state_dict(), world=world.state_dict(), hidden=hidden))
    scored, frames, states = rollout(model, world, hidden, family=3, condition=condition, steps=256)
    native = evaluate(model, world, hidden, family=3, condition=condition, steps=256, device='cpu')
    for own, old in [('matured', 'matured'), ('completed', 'completed'), ('clean_completed', 'clean_completed')]:
        assert scored['counts'][own] == native['primary'][old]
    assert scored['counts']['idle_opportunities'] == native['opportunities']['inactive_opportunities']
    assert scored['counts']['idle_actions'] == native['opportunities']['inactive_actions']
    assert scored['end_world_hash'] == native['end_world_state_hash']
    assert scored['end_scheduler_hash'] == native['end_scheduler_hash']
    assert scored['end_event_hash'] == native['end_event_hash']
    assert before == state_digest(dict(model=model.state_dict(), world=world.state_dict(), hidden=hidden))
    assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
    rebuilt = score_arrays(frames, states['initial_world'], fork_tick=world.tick,
                           evaluation_regime_id=scored['evaluation_regime_id'])
    assert rebuilt['primary_identity_hash'] == scored['primary_identity_hash']
    assert rebuilt['primary_requests'] == scored['primary_requests']


def test_opposite_policy_keeps_identical_future_cases_and_raw_counts():
    torch.set_num_threads(1)
    models = [GRUAgent(width=8).eval(), GRUAgent(width=8).eval()]
    with torch.no_grad():
        for model, action in zip(models, [0, 10]):
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias.reshape(2, 11)[:, action] = 100.
    world = ContextWorld(seed=100010, actor_ids=[10000, 10001], batch_size=2, n_jobs=2)
    world.observe()
    world.act(torch.zeros((2, 2), dtype=torch.long))
    results = [rollout(model, world, model.initial_state(2), family=3, condition='short', steps=256)[0]
               for model in models]
    for field in ('end_scheduler_hash', 'end_context_hash', 'end_event_hash', 'primary_identity_hash'):
        assert results[0][field] == results[1][field]
    assert results[0]['counts']['idle_actions'] == 0
    assert results[1]['counts']['idle_actions'] == results[1]['counts']['idle_opportunities']
    for result in results:
        assert sum(row['matured'] for row in result['by_raw'].values()) == result['counts']['matured']
        assert result['unique_raw_1_3']['matured'] == sum(result['by_raw'][str(raw)]['matured'] for raw in (1, 3))
