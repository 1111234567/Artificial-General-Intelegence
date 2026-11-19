from dataclasses import replace

import numpy as np

from .. import ring_world as ring


def test_identity_geometry_and_public_information():
    t=ring.make_tape(range(12),trials=20)
    assert np.all(t.preferences[:,0]!=t.preferences[:,1])
    assert np.all(np.sort(t.layout,axis=2)==np.arange(4))
    for multi in (False,True):
        a=ring.rollout(ring.PublicForager("oracle",t.goal),t,retry=multi,cue_ticks=1)
        assert a.solved.all() and a.first_correct.all()
        assert np.all(a.bites==1) and (a.negative==0).all()
        for i,j in np.ndindex(t.goal.shape):
            assert a.waveform[i,j,a.first_tick[i,j]+1]==1
        silent=ring.rollout(ring.PublicForager("silent"),t,retry=multi)
        assert not silent.bites.any()
    # A policy that samples the first physically visible fruit gets exactly
    # the same first choices under an unseen reward remapping.
    alt=replace(t,goal=(t.goal+1)%4)
    x=ring.rollout(ring.PublicForager("first"),t)
    y=ring.rollout(ring.PublicForager("first"),alt)
    assert np.array_equal(x.first,y.first)
    assert np.array_equal(x.first_tick,y.first_tick)
    assert not np.array_equal(x.waveform,y.waveform)


def test_retry_can_finish_without_learning_and_cannot_reconsume_a_bad_fruit():
    t=ring.make_tape(range(8),trials=24)
    raw=ring.rollout(ring.PublicForager("first"),t,retry=False)
    again=ring.rollout(ring.PublicForager("first"),t,retry=True)
    assert np.all(raw.bites==1)
    assert np.array_equal(raw.first,again.first)
    assert again.solved.all() and np.all(again.bites<=4)
    assert np.count_nonzero(again.negative)>0
