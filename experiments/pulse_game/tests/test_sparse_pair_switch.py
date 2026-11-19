import numpy as np

from experiments.pulse_game import sparse_pair_switch as check
from experiments.pulse_game.sparse_pairs_model import SparseDendriteGraph, PairPolicy


def test_disjoint_hidden_rule_preserves_public_tape_and_reverses_taste():
    roots=[17,18,17,18]
    plus,noise=check.meal(roots,1,probe=True)
    minus,other=check.meal(roots,1,probe=True,negative=True)
    np.testing.assert_array_equal(plus.chemicals,minus.chemicals)
    np.testing.assert_array_equal(plus.body.wind,minus.body.wind)
    np.testing.assert_array_equal(noise,other)
    np.testing.assert_array_equal(plus.realized,-minus.realized)
    assert np.array_equal(plus.probability.argmax(-1),1-minus.probability.argmax(-1))
    assert not set(map(tuple,check.task.CROSS)) & {(0,1),(2,3)}
    a,_=check.meal(roots,0)
    returned,_=check.meal(roots,2)
    np.testing.assert_array_equal(a.weights,returned.weights)
    assert not np.array_equal(a.chemicals,returned.chemicals)


def test_silent_probe_preserves_whole_live_state_with_no_installed_pairs():
    roots=[31,91]
    brain=SparseDendriteGraph(roots,check.Recipe(),world_seeds=roots,pair_rate=.65,pair_decay=.993,policy=PairPolicy())
    menu,noise=check.meal(roots,0,probe=True)
    prior=brain.state.copy()
    physical,events,counts=check.play(brain,check.portion(menu,slice(0,3)),noise[:,:3],0,silent=True)
    assert physical.shape==(2,3,5)
    assert (events<0).all() and not counts.any()
    np.testing.assert_array_equal(brain.state,prior)
    assert not brain.receipts.any()
