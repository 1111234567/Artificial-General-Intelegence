"""Scorer checks using real worlds, including nonteacher11-way continuations."""
from copy import deepcopy
import torch
from torch import nn
from experiments.wait_content import eval as e
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import evaluate


class ConstantModel(nn.Module):
    width, input_size, n_jobs, n_actions = 80, 56, 2, 11
    def __init__(self, action=0):
        super().__init__()
        self.action = action
        self.register_buffer('dummy', torch.zeros(1, dtype=torch.float64))
    def step(self, obs, h, dt, max_step=None):
        out = torch.zeros((len(obs), 2, 11), dtype=torch.float64)
        out[..., self.action] = 1
        return out, h+obs.sum(-1, keepdim=True)*.0001
    def config(self):
        return dict(kind='constant-unit-test-only', action=self.action)


def source():
    world = ContextWorld(seed=71, batch_size=8, n_jobs=2, dt=.25)
    world.observe()
    world.act(torch.zeros((8, 2), dtype=torch.long))
    return world, torch.ones((8, 80), dtype=torch.float64)*.1


def test_observer_exact_frozen_evaluator_and_idle_partitions():
    torch.set_num_threads(1)
    world, h = source()
    model = ConstantModel(3)
    before = e.state_digest({'world': world.state_dict(), 'h': h, 'model': model.state_dict()})
    original = evaluate(model, world, h, family=2, condition='short', steps=128)
    value, observer = e.evaluate_observed(model, world, h, family=2, condition='short', steps=128)
    assert e.normalized(original) == e.normalized(value['record'])
    assert before == e.state_digest({'world': world.state_dict(), 'h': h, 'model': model.state_dict()})
    held = value['shadow']['heldout']['counts']
    assert held['idle_actions'] == held['idle_begin']+held['idle_finish']
    assert held['idle_finish'] > 0 and held['idle_begin'] == 0
    assert held['before_begin_finish'] > 0 and held['processing_early_finish'] == 0
    assert held.get('primary_completed', 0) == 0
    assert len(observer.frames) == 128


def test_processing_is_only_started_before_ready_and_request_maturity():
    world, h = source()
    observer = e.Observer(world)
    world.set_family(1)
    world.set_regime(duration_range=(1., 4.), gap_range=(.5, 2.), name='test')
    expected_early = 0
    for _ in range(128):
        obs = world.observe()
        actions = world.oracle('midpoint')
        processing = world.active & world.started & (world.now < world.ready_time-e.EPS)
        # Deliberately early exact content; BEGIN/legally timed FINISH still come
        # from the fixture oracle to exercise true receipts and maturation.
        actions[processing] = (3+4*world.version+world.payload)[processing]
        expected_early += int(processing[4:].sum())
        logits = torch.zeros((8, 2, 11), dtype=torch.float64)
        logits.scatter_(-1, actions.unsqueeze(-1), 1.)
        observer.last_logits = logits
        observer.last_obs_hash = e.state_digest({'observation': obs})
        observer.act(world, actions)
    counts = observer.result()['heldout']['counts']
    assert counts['processing_early_finish'] == expected_early > 0
    assert counts['processing_early_finish'] == counts['processing_early_correct_content']
    matured = [r for r in observer.rows.values() if r['lane'] >= 4 and r['matured']]
    assert len(matured) == counts['primary_matured'] > 0
    assert sum(r['completed'] for r in matured) == counts['primary_completed']
    assert counts['primary_clean_completed'] < counts['primary_completed']
    assert counts['legal_finish_correct']+counts['legal_finish_wrong'] == counts['legal_finish_attempts']


def test_warmup_preserves_nonzero_initial_state_and_anchor_exactness():
    world, h = source()
    model = ConstantModel()
    observations = []
    elapsed = []
    end = h.clone()
    for _ in range(5):
        obs = world.observe().double()
        observations.append(obs)
        elapsed.append(world.elapsed)
        _, end = model.step(obs, end, world.elapsed)
        world.act(torch.zeros((8, 2), dtype=torch.long))
    actual, info = e.warmup(model, h, observations, elapsed, expected_h=end,
                           expected_actions=torch.zeros((5, 8, 2), dtype=torch.long))
    assert torch.equal(actual, end) and not torch.equal(actual, torch.zeros_like(actual))
    assert info['steps'] == 5 and info['hidden_resets'] == 0
    assert info['anchor_end_h_exact'] and info['anchor_all_actions_exact']
