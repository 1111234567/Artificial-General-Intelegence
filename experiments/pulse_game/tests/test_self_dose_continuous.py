"""Tests for the carried expanded writer and its exact-gradient diagnostic."""
import numpy as np
import torch

from experiments.pulse_game import self_dose_continuous as study
from experiments.pulse_game import self_dose_expected_credit as audit
from experiments.pulse_game import self_dose_carried_fork as fork


def test_carried_hidden_changes_next_calculation_and_preserves_private_complements():
    torch.set_num_threads(1)
    tape,truth=study.make(1536,8,8,channels=2,noise=.2,twins=True)
    np.testing.assert_array_equal(tape.odor[::2],tape.odor[1::2])
    np.testing.assert_array_equal(truth[::2],-truth[1::2])
    g=study.fw.spawn(91,8)
    draws=torch.from_numpy(np.random.default_rng(312).random((8,8,3),dtype=np.float32))
    values,final,h=study.interact(g,tape,draws,stochastic=False,disabled=True)
    assert not torch.equal(h,torch.zeros_like(h))
    assert torch.equal(final.weights,g.weights.expand_as(final.weights))
    assert torch.equal(final.present,g.present.expand_as(final.present))
    np.testing.assert_array_equal(values[3].numpy(),values[4].numpy()*tape.outcomes)
    np.testing.assert_array_equal(values[4][::2,0].numpy(),values[4][1::2,0].numpy())


def test_no_dummy_value_at_uninstalled_addresses_in_both_credit_paths():
    torch.set_num_threads(1)
    graph=study.fw.spawn(315,4)
    graph=study.fw.Graph(graph.weights.double().requires_grad_(),graph.bias.double().requires_grad_(),graph.present)
    loss,score,_,mass,total=audit.objective(graph)
    g1,w1=torch.autograd.grad(loss,(graph.weights,graph.bias),retain_graph=True)
    g2,w2=torch.autograd.grad(score,(graph.weights,graph.bias))
    torch.testing.assert_close(mass.detach(),mass.new_tensor(1.))
    torch.testing.assert_close(g1,g2,rtol=1e-12,atol=1e-12);torch.testing.assert_close(w1,w2,rtol=1e-12,atol=1e-12)
    assert torch.count_nonzero(g1[~graph.present])==0 and total==4224


def test_same_edits_with_common_future_have_identical_outcomes():
    env=fork.worlds([51,72])
    initial=study.fw.spawn(132,8).repeat(4)
    state=torch.zeros_like(initial.bias)
    a,ha,sa,la=fork.step(initial,state,env,0)
    b,hb,sb,lb=fork.step(initial,state,env,0)
    assert torch.equal(a.next_graph.weights,b.next_graph.weights) and torch.equal(ha,hb)
    assert torch.equal(sa,sb) and torch.equal(la,lb)
