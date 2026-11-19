import numpy as np
import torch

from experiments.pulse_game import free_graph_model as model, free_hub_audit as audit


def test_high_degree_cut_and_scattered_cuts_match_each_target():
    arcs = model.graph_for(7714,"fully_free")
    hub, cuts = audit.matched_masks(arcs,7714)
    out = np.bincount(arcs[:,0], minlength=model.WIDTH)
    assert out[hub]==out.max()
    assert cuts[0].sum()==0 and np.all(cuts[1:].sum(axis=(1,2))==out[hub])
    assert (cuts[1: ].sum(axis=2)==cuts[1].sum(axis=1)).all()
    assert not cuts[1,:,np.arange(model.WIDTH)!=hub].any()


def test_prefix_is_intact_before_a_surgical_query():
    seed=8111
    _,cuts=audit.matched_masks(model.graph_for(seed,"fully_free"),seed)
    bank=audit.QueryOnlyBank(seed,cuts)
    base=model.Bank([seed],["fully_free"])
    rng=torch.Generator().manual_seed(510)
    x=torch.randn(1,3,15,5,generator=rng)
    with torch.no_grad():
        _,state=bank(x,bank.start(3))
        _,ref=base(x,base.start(3))
        torch.testing.assert_close(state,ref.expand_as(state))
        out,_=bank(x,state,retain=3)
        exact,_=base(x,ref,retain=3)
    torch.testing.assert_close(out[:1],exact)
    assert torch.max(torch.abs(out[1]-out[0]))>.0001
