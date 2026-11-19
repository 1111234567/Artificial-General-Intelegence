"""Meaningful scorer and case-matching checks before protocol freezing."""
import numpy as np
import pytest
import torch

from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.world import ContextWorld, FAMILY_MAPPINGS
from experiments.lifelong.evaluate import evaluate
from experiments.experience_transfer import eval_transfer as e


def start():
    world = ContextWorld(seed=730053, actor_ids=[730000, 730001], batch_size=2, n_jobs=2)
    for _ in range(32):
        world.observe()
        world.act(torch.zeros((2, 2), dtype=torch.long))
    return world


def test_random_policy_matches_frozen_native_evaluator():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    model = GRUAgent(width=8).eval()
    world = start()
    h = model.initial_state(2)
    row, frames, states = e.rollout(model, world, h, families=(1,), steps=512,
        table=FAMILY_MAPPINGS, history_family=0, b_family=1)
    native = evaluate(model, world, h, family=1, condition='long_both', steps=512, device='cpu')
    for field in ('matured', 'completed', 'clean_completed'):
        assert row['counts'][field] == native['primary'][field]
    assert row['counts']['idle_actions'] == native['opportunities']['inactive_actions']
    assert row['counts']['idle_opportunities'] == native['opportunities']['inactive_opportunities']
    assert row['end_world_hash'] == native['end_world_state_hash']
    assert row['end_event_hash'] == native['end_event_hash']
    rebuilt = e.score(frames, states['initial_world'], table=FAMILY_MAPPINGS,
        history_family=0, b_family=1, fork_tick=row['fork_tick'], regime_id=row['evaluation_regime_id'])
    assert rebuilt['counts'] == {k:v for k,v in row['counts'].items() if k!='primary_censored'}


class PublicTimer(GRUAgent):
    """Public-input-only scripted controller exercises successful score paths.

    This is a counter verification fixture, never an experimental checkpoint,
    and never calls the private oracle or receives the current private phase.
    """
    def __init__(self):
        super().__init__(width=10)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, 10)

    def step(self, obs, h, dt=None, max_step=None):
        p = obs.reshape(len(obs), 2, 28)
        state = h.reshape(len(obs), 2, 5).clone()
        state[..., 4] += float(dt or 0.)
        command = (p[..., 0] + p[..., 1]) > 0
        cancel = p[..., 2] > 0
        raw = p[..., 5:9].argmax(-1)
        family = p[..., 24:28].argmax(-1)
        target = torch.tensor(FAMILY_MAPPINGS)[family, raw]
        state[..., 0] = torch.where(command, 1., state[..., 0])
        state[..., 1] = torch.where(command, p[..., 3:5].argmax(-1), state[..., 1])
        state[..., 2] = torch.where(command, target, state[..., 2])
        state[..., 3] = torch.where(command, p[..., 9] + .5 * p[..., 10], state[..., 3])
        state[..., 4] = torch.where(command, 0., state[..., 4])
        state[..., 0] = torch.where(cancel, 0., state[..., 0])
        finish = (state[..., 0] == 1) & (state[..., 4] >= state[..., 3]) & ~command
        actions = torch.where(command, 1 + state[..., 1].long(),
            torch.where(finish, 3 + 4 * state[..., 1].long() + state[..., 2].long(), 0))
        state[..., 0] = torch.where(finish, 0., state[..., 0])
        logits = torch.full((len(obs), 2, 11), -100.)
        logits.scatter_(-1, actions[..., None], 100.)
        return logits, state.flatten(1)


def test_successful_public_timer_and_receipt_tamper_rejection():
    torch.set_num_threads(1)
    model = PublicTimer().eval()
    world = start()
    row, frames, states = e.rollout(model, world, model.initial_state(2), families=(0,1), steps=512,
        table=FAMILY_MAPPINGS, history_family=0, b_family=1)
    assert row['counts']['legal_finish'] > 5
    assert row['counts']['clean_completed'] > 5
    assert row['counts']['matured_with_legal_finish'] == row['counts']['completed']
    assert row['counts']['matured_with_feasible_begin'] >= row['counts']['completed']
    assert row['actor_ticks_with_two_different_active_contexts'] > 0
    altered = dict(frames, result=frames['result'].copy())
    indices = np.argwhere(frames['result'] == 1)[0]
    altered['result'][tuple(indices)] = -1
    with pytest.raises(AssertionError):
        e.score(altered, states['initial_world'], table=FAMILY_MAPPINGS,
            history_family=0, b_family=1, fork_tick=row['fork_tick'], regime_id=row['evaluation_regime_id'])


def test_opposite_policies_keep_same_external_cases_and_raw_denominators():
    torch.set_num_threads(1)
    world = start()
    results = []
    for action in (0, 10):
        model = GRUAgent(width=4).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.head.bias.reshape(2, 11)[:, action] = 1.
        row, _, _ = e.rollout(model, world, model.initial_state(2), families=(0,1), steps=512,
            table=FAMILY_MAPPINGS, history_family=0, b_family=1)
        results.append(row)
    for key in ('external_digest', 'primary_identity_hash', 'end_scheduler_hash', 'end_event_hash'):
        assert results[0][key] == results[1][key]
    assert results[0]['counts']['idle_actions'] == 0
    assert results[1]['counts']['idle_actions'] == results[1]['counts']['idle_opportunities']
    for row in results:
        assert sum(v['counts'].get('matured',0) for v in row['by_family_raw'].values()) == row['counts']['matured']


def test_mapping_scope_preserves_public_stream_with_different_targets():
    from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table, SEEDS
    worlds, observations = [], []
    for arm in ('related','disjoint'):
        table = mapping_table(0, arm)
        with mapping_scope(table):
            world = ContextWorld(seed=730000, actor_ids=[730000,730001], batch_size=2,n_jobs=2,future_family=0)
            obs = []
            for _ in range(128):
                obs.append(world.observe().clone())
                world.act(torch.zeros((2,2),dtype=torch.long))
            worlds.append(world.state_dict())
            observations.append(torch.stack(obs))
    assert torch.equal(*observations)
    assert e.state_digest(worlds[0]['scheduler']) == e.state_digest(worlds[1]['scheduler'])
    assert worlds[0]['event_hash'] == worlds[1]['event_hash']
    assert worlds[0]['context']['mappings'] != worlds[1]['context']['mappings']
    for seed in range(3):
        ss=SEEDS[seed]
        groups=e.groups_for(mapping_table(seed,'related'),ss['history_cue'],ss['B_cue'])
        assert groups['shared']==ss['shared_raw'] and groups['conflict']==ss['conflict_raw']
