import numpy as np
import torch

from experiments.pulse_game import noon_ticket as ticket, noon_ticket_eval as tickets
from experiments.pulse_game import noon_coverage as coverage, noon_coverage_eval as coverage_eval
from experiments.pulse_game import noon_paired_lesson as paired, noon_paired_eval as pairs
from experiments.pulse_game import noon_online_readout as online


def body():
    torch.set_num_threads(1)
    return ticket.net.spawn(731),ticket.net.Rule([731])


def test_public_only_gate_and_equal_cost_shuffle():
    g,r=body();t=ticket.tape(111,7,3,5,price=.25,alternative="unseen")
    v,s,_,history=ticket.prefix(g,r,t)
    valid,seen=tickets.public_evidence(t,history)
    assert torch.equal(valid,t["evidenced"]) and torch.equal(seen,t["evidenced"])
    natural=ticket.finish(v,s,t,"native")
    future={**t,"final_value":-t["final_value"],"review_value":-t["review_value"]}
    opposite=ticket.finish(v,s,future,"native")
    assert torch.equal(natural["choice"],opposite["choice"])
    assert torch.equal(natural["taste"],natural["buy"]*natural["review_side"]*t["review_value"])
    twin=torch.arange(len(valid))^4
    assert torch.equal(t["offer_left"],t["offer_left"][twin])
    assert (natural["buy"]*t["price"]).sum()==(natural["buy"][twin]*t["price"]).sum()


def test_six_values_are_streaming_physical_coactivation():
    g,r=body()
    t=coverage_eval.scene(3003,4,.25,"unseen",6,"after",None)
    _,_,_,trace,history=coverage.physical_prefix(g,r,t,"evidence")
    selected=torch.nn.functional.normalize(t["study"]*history[:,:,0,None],dim=2)
    dense=torch.einsum("nti,ntj,nt->nij",selected,selected,history[:,:,1].abs())
    x=torch.nn.functional.normalize(t["preview"],dim=1)
    expected=torch.einsum("ni,nij,nj->n",x,dense,x)
    torch.testing.assert_close(coverage.projection(trace,x),expected,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(coverage.cue(trace,x,"dose"),dense.diagonal(dim1=1,dim2=2).sum(1)/3)
    assert trace.shape==(32,6)


def test_no_coverage_and_birth_circuit_are_identical():
    g,r=body();t=ticket.tape(931,0,0,3,alternative="omitted")
    view,state,_,record=ticket.prefix(g,r,t)
    a=ticket.finish(view,state,t,"native")
    for arm in coverage.ARMS:
        v,s,_,tr,h=coverage.physical_prefix(g,r,t,arm)
        assert torch.equal(h,record)
        u=coverage_eval.offer(t,torch.tensor(0.),tr,arm)
        z=ticket.finish(v,s,u,"native")
        for key in ("choice","logit","review_side","taste","final","utility"):assert torch.equal(a[key],z[key])


def test_omission_yokes_locations_and_uniform_route_pay():
    g,r=body();t,missing=paired.prepare(732,(),0,4,4,omissions=True,heldout=True)
    raw,world=pairs.measure(g,r,t,missing)
    ix=np.arange(t.lanes)
    sides=raw["selected"]
    active=raw["test"]
    gain=(active[ix,sides].all(1).astype(float)-active.all(2).mean(1)).reshape(4,4).mean(1)
    np.testing.assert_array_equal(gain,world["active/native_minus_uniform/all"])
    assert np.all(raw["study"][ix,raw["missing"],1]==0)
    twin=ix^1
    np.testing.assert_array_equal(raw["value"],raw["value"][twin])
    assert np.all(raw["missing"]!=raw["missing"][twin])


def test_one_actual_online_route_matches_its_enumerated_continuation():
    g,r=body()
    t,first,_=online.bank.tape(4813,4,4)
    a,_=online.observe(g,r,t,first)
    side=torch.where(t.external_draw>.5,1.,-1.)[:,None]
    single=online.execute(g,r,t,a,side)
    paired_pay=online.execute(g,r,t,a,torch.cat((side,-side),dim=1))
    torch.testing.assert_close(single[:,0],paired_pay[:,0],rtol=0,atol=0)


def test_a_hidden_reversal_cannot_be_seen_at_the_ticket():
    from experiments.pulse_game import noon_agreement_eval as evaluation

    g,r=body()
    scene=evaluation.scene(4611,3,.25,2)
    raw,_=evaluation.measure(g,r,torch.tensor([1.1]),torch.tensor([.8]),scene,"agreement","bilateral")
    # Switch the private law and both receipt slips together.
    sibling=np.arange(len(raw["choice"]))^11
    np.testing.assert_array_equal(raw["actual_history"],raw["actual_history"][sibling])
    np.testing.assert_array_equal(raw["trace"],raw["trace"][sibling])
    np.testing.assert_array_equal(raw["choice"],raw["choice"][sibling])
    np.testing.assert_array_equal(raw["target"],-raw["target"][sibling])


def test_commit_uses_taste_only_after_the_action_and_matches_real_endings():
    from experiments.pulse_game import noon_commit, noon_commit_eval

    g,r=body()
    scene=noon_commit.tape(349,0,0,4,theta=15,delay=4)
    gate=torch.tensor([[0.,3.]])
    raw,rates=noon_commit_eval.measure(g,r,gate,scene,"confirmation")
    side,taste,logit=raw["study"][:,-1].T
    sibling=np.arange(len(side))^1
    np.testing.assert_array_equal(side[sibling],side)
    np.testing.assert_array_equal(taste[sibling],-taste)
    expected=1/(1+np.exp(-3*side*taste*np.tanh(logit/2)))
    np.testing.assert_allclose(raw["gate_probability"],expected,rtol=1e-6)
    chosen=raw["serial"][np.arange(len(side)),raw["gate"].astype(int)]
    truth=raw["target"][:,3:5]
    reconstructed=((chosen==truth).all(axis=1)*raw["weight"]).reshape(-1,8).sum(axis=1)
    np.testing.assert_allclose(rates["all/native/serial"],reconstructed)


def test_forecast_complement_changes_meaning_but_preserves_route_probability():
    from experiments.pulse_game.noon_self_assess import Assessor

    head=Assessor()
    with torch.no_grad():
        head.forecast.copy_(torch.tensor([.1,.7,-.4]))
        head.choice.copy_(torch.tensor([1.8,-.5]))
        z=torch.linspace(-4,5,48,dtype=torch.double).reshape(12,2,2)
        attempted=torch.where(z[:,:,0]>0,1.,-1.)
        menu=torch.linspace(-1,2,12,dtype=torch.double)
        logits,estimated,route=head(z,attempted,menu)
        head.forecast.neg_()
        head.choice[0].neg_()
        flipped,other,new_route=head(z,attempted,menu)
    torch.testing.assert_close(flipped,-logits)
    torch.testing.assert_close(other,1-estimated)
    torch.testing.assert_close(new_route,route,rtol=0,atol=1e-12)


def test_optional_diagnostic_is_not_included_in_a_paired_contrast():
    from experiments.pulse_game import noon_market_stats

    v=np.array([[1.,2.,3.],[0.,2.,1.]])
    out=noon_market_stats.summarize({"a":{"shared":v,"extra":-v},"b":{"shared":2*v}},
                                    3,[731,732],91,[("a","b")])
    assert set(out["a minus b"])=={"shared"}
    np.testing.assert_allclose(out["a minus b"]["shared"]["mean"],-v.mean())
