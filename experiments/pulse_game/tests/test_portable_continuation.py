"""State continuation, unideal timing and direct public observation regressions."""
from dataclasses import replace
import numpy as np
import json
import pytest
from .. import delayed_receipts as feedback, lifelong_agent as deployed, habitat_world,unified_world,world


def tiny_model(path):
    rng=np.random.default_rng(83371);n=6;hidden=9;slots=3
    f=dict(matrix=(rng.normal(size=(n,n))*.12).astype("f"),sensors=rng.normal(size=(6,n)).astype("f"),
           ground=(rng.normal(size=(n,4)).astype("f"),np.zeros(4,"f")),transform=-np.eye(4,dtype="f"),
           scalars=[14.,.75,1.,.96,1.5],capacity=9,age_cost=.025)
    r=dict(routing="learned",updater="rls",slots=slots,eta=.8,amp=2.,ridge=.35,retention=1.,
           first=(rng.normal(size=(n,hidden)).astype("f"),np.zeros(hidden,"f")),
           second=(rng.normal(size=(hidden,slots)).astype("f"),np.zeros(slots,"f")),
           release=[2,.04,0.,0.,0.])
    motion=np.array([.55,2.,.7,.13,1.1,1.3,0.],np.float32)
    deployed.save(path,f,r,motion,dict(test=True))


@pytest.mark.parametrize("address",["distributed","local","local-selective"])
def test_snapshot_crosses_noise_chunk_without_a_new_birth(tmp_path,address):
    path=tmp_path/"agent.npz";tiny_model(path)
    (tmp_path/"readout-options.json").write_text(json.dumps({"params":{"evidence":7.,"prediction":1.,"global_rate":.7,"prior_limit":2.,"hold":.95,"read":True,"protect":True}}))
    a=deployed.new_lives([67,89],path,address=address);a.reset(2)
    rng=np.random.default_rng(195)
    frames=np.zeros((185,2,20),np.float32)
    frames[:,:,2:10]=.5;frames[::16,:,17]=1;frames[:,:,12]=1
    frames[1::16,0,0]=1.;frames[1::16,1,1]=1.
    frames[2::16,:,2:10]=rng.random(frames[2::16,:,2:10].shape,dtype=np.float32)
    frames[7::16,:,15]=1.;frames[7::32,:,13]=1.;frames[23::32,:,14]=1.
    for obs in frames[:75]:a.act(obs)
    assert np.any(a.count>0)
    snap=tmp_path/"life.npz";deployed.save_life(snap,a)
    b=deployed.restore_life(snap,path)
    for obs in frames[75:]:
        np.testing.assert_array_equal(a.act(obs),b.act(obs))
    for k in deployed.STATE_ARRAYS:np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
    if address=="local-selective":
        for k in deployed.READER_ARRAYS:np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
    if address in ("local","local-selective"):
        for k in ("prototypes","current","counts","overflows"):
            np.testing.assert_array_equal(getattr(a.places,k),getattr(b.places,k))


def test_zero_added_delay_reproduces_original_contacts(tmp_path):
    path=tmp_path/"agent.npz";tiny_model(path)
    seeds=[97,109,113,127];e=unified_world.make(seeds,initial=6,depths=(0,2,4),span=2)
    local=deployed.new_lives(seeds,path,address="local")
    ref=habitat_world.rollout(local,e.arena,silent_probes=False)
    delayed=deployed.new_lives(seeds,path,address="local")
    test,diag=feedback.run(e,delayed,seeds,feedback.Channel(lag=0,wait=0),"fifo")
    np.testing.assert_array_equal(ref.contact_tick,test.contact_tick);np.testing.assert_array_equal(ref.reward,test.reward)
    np.testing.assert_array_equal(diag["correct_binding"],np.ones(len(seeds)))


def test_ordered_feedback_can_rebind_after_several_physical_ticks(tmp_path):
    path=tmp_path/"agent.npz";tiny_model(path)
    seeds=[59,97,113,229];e=unified_world.make(seeds,initial=6,depths=(0,2,4),span=2)
    actor=deployed.new_lives(seeds,path,address="local")
    result,diag=feedback.run(e,actor,seeds,feedback.Channel(lag=42,ordered=True,wait=60),"fifo")
    assert np.all(diag["peak_pending"]>=2)
    assert not np.any(diag["overflow"])
    np.testing.assert_array_equal(diag["correct_binding"],np.ones(len(seeds)))
    anchors=np.flatnonzero(e.arena.probe[0])-1
    # An untrained body may time out; only an actual anchor contact can deliver a receipt.
    np.testing.assert_array_equal(diag["anchor_available"],(result.contact_side[:,anchors]>=0).mean(1))


