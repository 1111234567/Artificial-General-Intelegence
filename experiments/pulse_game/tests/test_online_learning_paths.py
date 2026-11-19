"""Regression checks for public feedback, causal writes and fixed-budget physical scores."""
import numpy as np
import pytest
import torch

from .. import habitat_meta, longrun_world, recurring_world, symmetry_audit, unified_world, unified_train


def test_identical_teaching_signs_change_only_their_binding():
    roots=[8123,8131,8167,8191]
    pair=symmetry_audit.joint(roots,pairs=12,probes=2,span=4,noise=.08,jitter=.15,matched=True)
    n,cut=len(roots),pair.exposed
    np.testing.assert_array_equal(pair.menu.realized[:n,:cut],pair.menu.realized[n:,:cut])
    # In each matched group both branches encounter the same set of echoes.
    # Sorting whole byte strings preserves the four-dimensional identity.
    for i in range(n):
        for j in range(4,cut,8):
            a=sorted(v.tobytes() for v in pair.menu.chemicals[i,j:j+4,0])
            b=sorted(v.tobytes() for v in pair.menu.chemicals[n+i,j:j+4,0])
            assert a==b
    np.testing.assert_array_equal(pair.menu.chemicals[:n,cut:],pair.menu.chemicals[n:,cut:])
    np.testing.assert_array_equal(pair.menu.probability[:n,cut+1::2],1-pair.menu.probability[n:,cut+1::2])


def test_repeated_tastes_do_not_reveal_retrospective_target_or_private_rule():
    seeds=[271,479,647]
    stream,_=recurring_world.make(seeds,gap=32,distinct=6,query="random",nonce=42396)
    n=len(seeds);first=np.flatnonzero(stream.query[0])[0]
    assert first==stream.exposed+33
    np.testing.assert_array_equal(stream.menu.realized[:n,:first],stream.menu.realized[n:,:first])
    np.testing.assert_array_equal(stream.menu.chemicals[:n,stream.exposed:],stream.menu.chemicals[n:,stream.exposed:])
    assert np.all(stream.menu.probability[:n,first]==1-stream.menu.probability[n:,first])


def test_first_B_sensation_does_not_contain_B_relation():
    seeds=[227,237,257,277]*2
    fam=np.repeat([[1,1],[1,-1]],4,axis=0)
    e=unified_world.make(seeds,force=fam)
    k=e.checks["B_after_0"]
    # Includes one legitimately tasted new B anchor.
    assert np.all((e.arena.menu.body.ctx[:,:k]==1).sum(1)==1)
    np.testing.assert_array_equal(e.arena.menu.chemicals[:4,:k+1],e.arena.menu.chemicals[4:,:k+1])
    np.testing.assert_array_equal(e.arena.menu.realized[:4,:k],e.arena.menu.realized[4:,:k])
    np.testing.assert_array_equal(e.arena.menu.probability[:4,k],1-e.arena.menu.probability[4:,k])


@pytest.mark.parametrize("updater",["delta","rls"])
@pytest.mark.parametrize("route",["learned","shared_public"])
def test_parallel_learning_matches_online_local_updates(updater,route):
    torch.manual_seed(2891)
    model=habitat_meta.MetaSynapses(habitat_meta.Config(updater=updater,routing=route,nodes=5,width=8,slots=3)).double()
    n,t=3,17
    state=torch.randn(n,t,5,dtype=torch.float64)
    query=torch.randn_like(state)
    context=torch.rand(n,t,2,dtype=torch.float64)
    y=torch.sign(torch.randn(n,t,dtype=torch.float64))
    signed_at=torch.rand(n,t,dtype=torch.float64)
    signed_at[:,4::5]=0.
    now=torch.rand_like(y)
    pred=model(state,context,query,signed_at,now,y)
    x=model.code(state,context);read=model.code(query,context)
    slots=x.shape[-1]
    w=torch.zeros(n,slots,dtype=torch.float64)
    eta=.025+1.5*model.rate.sigmoid()
    if updater=="rls":
        inv=torch.eye(slots,dtype=torch.float64).expand(n,-1,-1).clone()/(.015+model.log_ridge.exp())
    amp=.4+4*model.scale.sigmoid()
    expected=[]
    for j in range(t):
        expected.append((amp*(w*read[:,j]).sum(1)).tanh()*now[:,j])
        k=x[:,j]*signed_at[:,j,None]
        error=y[:,j]-(w*k).sum(1)
        if updater=="rls":
            z=(inv@k.unsqueeze(-1)).squeeze(-1);gain=z/(1.+(z*k).sum(1,keepdims=True))
            inv=inv-gain[:,:,None]*z[:,None,:]
        else:gain=eta*k
        w=w+gain*error[:,None]
    torch.testing.assert_close(pred,torch.stack(expected,1),rtol=1e-9,atol=1e-10)
    hidden=y.clone();hidden[:,4::5]*=-1
    torch.testing.assert_close(model(state,context,query,signed_at,now,hidden),pred,rtol=1e-10,atol=1e-11)


def test_hidden_probes_never_enter_finite_event_retrieval():
    from .. import symmetry_memory
    torch.manual_seed(10491)
    net=symmetry_memory.Memory(symmetry_memory.Config(capacity=3))
    x=torch.nn.functional.normalize(torch.randn(4,12,16),dim=-1);y=torch.ones(4,12)
    y[:,2:9]=0
    actual=unified_train.retrieve(net,x,x,y)
    other=x.clone();other[:,2:9]*=-1.
    changed=unified_train.retrieve(net,x,other,y)
    torch.testing.assert_close(actual,changed)
    # Retaining up to three shown signs counts actual receipts, not silent bouts.
    assert torch.all(actual[:,9]>0)


def test_long_life_has_no_state_change_marker_in_the_outcome_stream():
    roots=[13,37,59,71]
    a=longrun_world.make(roots,blocks=36,mode="stable")
    b=longrun_world.make(roots,blocks=36,mode="reverse_return")
    np.testing.assert_array_equal(a.block_context,b.block_context)
    np.testing.assert_array_equal(a.arena.appearance,b.arena.appearance)
    off=~a.arena.probe
    np.testing.assert_array_equal(a.arena.menu.realized[off],b.arena.menu.realized[off])
    j=np.arange(36)[None,:]
    active=(a.block_context==1)&(j>=b.boundary[:,:1])&(j<b.boundary[:,1:])
    assert np.any(active)
    xor=a.arena.truths[:,a.query_at[:,1]]!=b.arena.truths[:,b.query_at[:,1]]
    np.testing.assert_array_equal(xor,active)


def test_rotated_trajectories_do_not_change_body_goals_or_feedback():
    from .. import replay_life
    p=symmetry_audit.joint([17,51,77],pairs=8,probes=1,span=4,noise=.07,jitter=.08,matched=True)
    q=replay_life.geometry(p,"rotation")
    np.testing.assert_array_equal(p.menu.realized,q.menu.realized)
    np.testing.assert_array_equal(p.menu.body.distances,q.menu.body.distances)
    assert not np.array_equal(p.menu.chemicals,q.menu.chemicals)
