"""Magnitude proposal order, real deletion, inherited validation and recovery."""
from copy import deepcopy
import pytest
import torch

from experiments.coverage_exit import retirement_magnitude as m
from experiments.coverage_exit.retirement import adopt_fixed, _digest
from experiments.coverage_exit.fast_window import install_window
from experiments.undirected.test_plasticity import fixture, step, scripted
from experiments.undirected.test_run import toy
from experiments.undirected.run import NativeTrainer as OriginalTrainer
from experiments.continuity.run import state_digest


def magnitude_fixture():
    model,old,optimizer=fixture('fixed',tiny=False,candidates=2)
    ids=old.mask.nonzero().flatten()
    with torch.no_grad():
        model.edge_shared[ids]=1.
        model.edge_shared[ids[-2]]=0.01
        model.edge_shared[ids[-1]]=-0.01
    op=adopt_fixed(model,old.state_dict(),full_eligible=True)
    op.__class__=m.MagnitudeOperator
    return model,op,optimizer,ids


def test_magnitude_order_stable_ties_and_real_retirement_keep_adam():
    model,op,opt,ids=magnitude_fixture()
    step(model,op,opt)
    assert op.pending['fade_indices'].tolist()==ids[-2:].tolist()
    probe,calls=scripted([10.,9.,8.,11.],model)
    output=step(model,op,opt,probe)
    assert calls[0]==4 and output['deaths']==2
    assert not bool(op.mask[ids[-2:]].any())
    assert not bool(model.edge_shared[ids[-2:]].ne(0).any())
    assert float(opt.state[model.edge_shared]['step'])==op.updates
    assert bool(opt.state[model.edge_shared]['exp_avg'][~op.mask].ne(0).any())
    # Executed proposal used its historical weights; current ranking excludes
    # retired indices and generally differs. Both ledgers must still validate.
    restored=m.MagnitudeOperator.from_state(model,op.state_dict())
    assert _digest(restored.state_dict())==_digest(op.state_dict())


def test_rehashed_wrong_ranking_is_refused():
    model,op,opt,ids=magnitude_fixture();step(model,op,opt)
    state=op.state_dict();state['pending']['fade_indices']=ids[:2]
    state['state_hash']=_digest({k:v for k,v in state.items() if k!='state_hash'})
    with pytest.raises(ValueError):m.MagnitudeOperator.from_state(model,state)


def test_full_state_adoption_and_next_update_after_restore_are_exact(tmp_path):
    torch.set_num_threads(1)
    original=OriginalTrainer(toy(interface='core',credit='recorded512',structural_window_updates=2))
    original.run_updates(2)
    source=tmp_path/'original.pt';original.save_checkpoint(source)
    before=original.semantic_state();trainer=m.from_original(source);install_window(trainer)
    after=trainer.semantic_state()
    for key in before.keys()-{'format','config','operator'}:
        assert state_digest(before[key])==state_digest(after[key]),key
    trainer.run_updates(3)
    restored=m.NativeTrainer.from_state_dict(trainer.state_dict());install_window(restored)
    assert state_digest(trainer.semantic_state())==state_digest(restored.semantic_state())
    assert state_digest(trainer.train_update())==state_digest(restored.train_update())
    assert state_digest(trainer.semantic_state())==state_digest(restored.semantic_state())
