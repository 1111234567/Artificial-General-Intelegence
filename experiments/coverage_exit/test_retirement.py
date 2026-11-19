"""New retirement semantics, retained lifetime, and complete-state continuation tests."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from experiments.undirected.test_plasticity import fixture, step, scripted
from experiments.coverage_exit.retirement import RetirementOperator, RetirementConfig, adopt_fixed, _digest
from experiments.coverage_exit import retirement_run as run
from experiments.undirected.run import NativeTrainer as OriginalTrainer
from experiments.undirected.test_run import toy
from experiments.continuity.run import state_digest
from experiments.coverage_exit.fast_window import install_window


def trial(full, values):
    model, old, optimizer = fixture('fixed', tiny=False, candidates=2)
    operator = adopt_fixed(model, old.state_dict(), full_eligible=full)
    step(model, operator, optimizer)
    proposal = deepcopy(operator.pending)
    probe, count = scripted(values, model)
    result = step(model, operator, optimizer, probe)
    return model, operator, optimizer, proposal, result['probe'], count[0]


def test_full_improvement_commits_real_deletion_even_when_half_fails():
    model, op, opt, proposal, event, calls = trial(True, [10., 9., 8., 11.])
    assert calls == 4 and event['full_accepted'] and not event['fade_improves']
    assert event['deaths'] == 2 and not event['birth_accepted']
    assert not bool(op.mask[proposal['fade_indices']].any())
    assert not bool(model.edge_shared[proposal['fade_indices']].ne(0).any())
    RetirementOperator.from_state(model, op.state_dict())
    assert float(opt.state[model.edge_shared]['step']) == op.updates
    assert bool(opt.state[model.edge_shared]['exp_avg'][~op.mask].ne(0).any())


def test_control_uses_old_half_rule_and_rankings_and_probe_budget_match():
    a = trial(False, [10., 9., 8., 11.])
    b = trial(True, [10., 9., 8., 11.])
    assert _digest(a[3]) == _digest(b[3])
    assert a[-1] == b[-1] == 4
    assert a[-2]['deaths'] == 0 and not a[-2]['fade_accepted']
    RetirementOperator.from_state(a[0], a[1].state_dict())


@pytest.mark.parametrize('values,accepted,deaths', [([10.,9.,11.,8.],True,0), ([10.,9.,10.,10.],False,0)])
def test_full_fallback_is_exact_old_gradual_rule(values, accepted, deaths):
    model, op, _, _, event, _ = trial(True, values)
    assert not event['full_accepted'] and event['fade_accepted'] == accepted
    assert event['deaths'] == deaths
    RetirementOperator.from_state(model, op.state_dict())


@pytest.mark.parametrize('field,value', [('full_accepted',False),('full_improves',False),('deaths',0)])
def test_new_event_semantics_are_validated(field,value):
    model, op, *_ = trial(True,[10.,9.,8.,11.])
    state=op.state_dict()
    state['event_history'][0]['event'][field]=value
    state['state_hash']=_digest({k:v for k,v in state.items() if k!='state_hash'})
    with pytest.raises(ValueError):
        RetirementOperator.from_state(model,state)


def test_adopt_fixed_preserves_lifetime_ledger_and_full_moments(tmp_path):
    torch.set_num_threads(1)
    original=OriginalTrainer(toy(interface='core',credit='recorded512',structural_window_updates=2))
    original.run_updates(2)
    source=tmp_path/'source.pt'; original.save_checkpoint(source)
    before=original.semantic_state()
    adopted=run.from_original(source,full_eligible=True)
    install_window(adopted)
    after=adopted.semantic_state()
    for name in before.keys()-{'format','config','operator'}:
        assert state_digest(before[name]) == state_digest(after[name]), name
    assert adopted.operator.updates == 2 and adopted.credit.updates == 2
    assert state_digest(adopted.operator.update_history)==state_digest(before['operator']['update_history'])
    assert adopted.operator.window_count == 0 and adopted.operator.pending is None
    restored=run.NativeTrainer.from_state_dict(adopted.state_dict())
    install_window(restored)
    assert state_digest(restored.semantic_state())==state_digest(adopted.semantic_state())
    for _ in range(3): adopted.train_update()
    restored=run.NativeTrainer.from_state_dict(adopted.state_dict())
    install_window(restored)
    assert len(restored.operator.event_history)==1
    # Test actual restored execution, not merely parsing the new envelope.
    expected=adopted.train_update()
    actual=restored.train_update()
    assert state_digest(expected)==state_digest(actual)
    assert state_digest(restored.semantic_state())==state_digest(adopted.semantic_state())
