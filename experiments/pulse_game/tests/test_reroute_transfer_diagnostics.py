import numpy as np
import torch

from experiments.pulse_game import reroute_midpoint as mid
from experiments.pulse_game import reroute_novel as novel
from experiments.pulse_game import reroute_update_diagnostic as update
from experiments.pulse_game import reroute_world as world


def test_two_prior_map_partitions_are_disjoint_but_have_the_same_observable_options():
    episodes,idx,cases,held=novel.panel()
    assert len(episodes)==864
    assert set(idx[held]).isdisjoint(idx[~held])
    for part in (held,~held):
        np.testing.assert_array_equal(np.bincount(cases[part],minlength=4),[48,96,144,144])
        assert set(episodes.inspected_room[part].tolist())=={0,1}
    a,b=update.pairs(episodes,cases)
    assert torch.equal(episodes.goals[a],episodes.goals[b])
    assert torch.equal(episodes.prior_endpoints[a],episodes.prior_endpoints[b])
    assert torch.equal(episodes.door_rooms[a],episodes.door_rooms[b])
    assert torch.all(episodes.observed_endpoint[a]!=episodes.observed_endpoint[b])


def test_shared_action_noise_distinguishes_correct_switches_from_simple_repetition():
    q0=np.array([.8,.15]);q1=np.array([.2,.7])
    old=np.array([[0.,1.],[.25,.25]])
    new=np.array([[0.,1.],[.75,.75]])
    # First pair always follows the query; the second uses different real-door probabilities only.
    correct,wrong,change=update.common_draw_outcomes(q0,q1,old,new,np.array([1,0]),np.array([0,1]))
    np.testing.assert_allclose(correct,[.6,.5])
    np.testing.assert_allclose(wrong,[0.,0.])
    np.testing.assert_allclose(change,[.6,.5])


def test_room_transplant_donors_are_equal_on_every_endpoint_and_forecasted_reward():
    _,one,two,q,_,signs,_=mid.events(mid.SEEDS[0])
    a,b=world.oracle(one),world.oracle(two)
    assert torch.equal(a.rooms,1-b.rooms)
    assert torch.equal(a.endpoints,b.endpoints) and torch.equal(a.prior_payoffs,b.prior_payoffs)
    assert torch.equal(a.payoffs,b.payoffs) and a.payoffs.count_nonzero()==0
    assert torch.all(signs.sum(1)==0) and set(q.tolist())=={0,1}
