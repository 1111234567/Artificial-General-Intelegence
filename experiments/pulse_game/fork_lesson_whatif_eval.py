"""Test whether the same primary organism gives the right counterfactual answer after different private motor trials."""

import argparse
from dataclasses import fields, replace

import numpy as np
import torch

from . import fork_lesson_whatif as sim
from . import fork_lesson_whatif_fit as fit
from . import fork_lesson_fit as base
from . import fork_lesson_world as world
from . import noon_market_stats as stats


ROOTS=144
SEEDS=(8619551,8619649)
KINDS=("binding","misbound")
MODES=("own","echo","wrong_history","cut","blank","drive_flip")
DATA=base.DATA.parent/"whatif-eval"
PLAN=base.EVIDENCE/"whatif-eval-plan.md"


def scenes(seed):
    first=world.scenes(seed,ROOTS)
    bits=torch.from_numpy(np.tile(np.indices((2,)*6).reshape(6,-1).T,(ROOTS,1)))
    v0,v1,order,f0,f1,old=bits.T
    data={f.name:getattr(first,f.name).repeat_interleave(64,0) for f in fields(first)}
    data.update(values=torch.stack((v0,v1),1).float()*2-1,first=old,
                door_order=torch.stack((order,1-order),1),flips=torch.stack((f0,f1),1).float()*2-1)
    return world.Episode(**data)


def matched_direction(delta,alternate):
    # Preserve the total returned contrast norm independently for every life and physical tick.
    size=delta.double().norm(dim=-1,keepdim=True)
    vector=alternate.double()
    target=vector.norm(dim=-1,keepdim=True)
    # A silent donor carries no direction. Use a fixed public axis independent of both
    # histories instead of accidentally letting the control remove the whole pulse.
    unit=torch.zeros_like(vector)
    unit[...,0]=1.
    unit=torch.where(target>0,vector/target.clamp_min(1e-40),unit)
    return (unit*size).to(delta.dtype)


@torch.no_grad()
def batch(model,e,query_draw,action_draw):
    state=sim.start_trial(model,e)
    n=len(e.first)
    left=torch.zeros(n,dtype=torch.long);right=1-left
    native=[sim.perform_trial(model,state,q) for q in (left,right)]
    midpoint=(native[1].feedback+native[0].feedback)/2
    delta=(native[1].feedback-native[0].feedback)/2
    ticks=delta.shape[0]
    same_geometry=delta.reshape(ticks,-1,4,16,20)
    generic=same_geometry.mean(dim=2,keepdim=True).expand_as(same_geometry).reshape_as(delta)
    twin=torch.arange(n)^48  # Reverse both private values, preserve all visible door features.
    assert torch.equal(e.values[twin],-e.values) and torch.equal(world.door_cues(e)[twin],world.door_cues(e))
    echo=matched_direction(delta,generic);wrong=matched_direction(delta,delta[:,twin])
    nd=max(float((v.norm(dim=-1)-delta.norm(dim=-1)).abs().max()) for v in (echo,wrong))
    if nd>2e-5:raise ValueError(f"Unmatched trial cue echo {nd}")
    alternatives={"own":native}
    for name,controller in (("echo",echo),("wrong_history",wrong)):
        alternatives[name]=[sim.perform_trial(model,state,q,feedback_override=midpoint+(2*side-1)*controller)
                            for side,q in enumerate((left,right))]
    for name,options in (("cut",dict(cut_return=True)),("blank",dict(blank=True))):
        alternatives[name]=[sim.perform_trial(model,state,q,**options) for q in (left,right)]
    alternatives["drive_flip"]=[native[1],native[0]]
    opposite=sim.perform_trial(model,state,right,driven=left)
    for candidate in (opposite,sim.perform_trial(model,state,left,feedback_override=native[0].feedback)):
        for attrib in ("h","fast","trace","candidate"):
            torch.testing.assert_close(getattr(candidate.final_state,attrib),getattr(native[0].final_state,attrib),atol=0,rtol=0)
        torch.testing.assert_close(candidate.forecast,native[0].forecast,atol=0,rtol=0)
    torch.testing.assert_close(alternatives["blank"][0].forecast,alternatives["blank"][1].forecast,rtol=0,atol=0)
    q_prob=model.query_logit(state).sigmoid()
    q=(query_draw<q_prob).long()
    row=torch.arange(n)
    replacement_undefined=((generic.norm(dim=-1)==0)|(delta[:,twin].norm(dim=-1)==0))&(delta.norm(dim=-1)>0)
    out={"self_q":q,"q_p":q_prob,"random_q":(query_draw<.5).long(),"public_fallback_ticks":replacement_undefined.sum(0),
         "norm_error":torch.full((n,),nd),"delta_current":delta.square().mean((0,2)).sqrt()}
    truth=None
    for name,tries in alternatives.items():
        estimates=torch.stack([item.forecast for item in tries],1)
        prob=torch.stack([model.motor_logit(item.final_state).sigmoid() for item in tries],1)
        acts=(action_draw[:,None]<prob).long()
        paid=[]
        for j,item in enumerate(tries):
            z=model.motor_logit(item.final_state)
            choice=world.Choice(z,acts[:,j],torch.zeros_like(z),torch.zeros_like(z))
            _,res=world.commit(model.body,item.final_state,e,choice)
            paid.append(res.payoff)
            if name=="own":
                physical=[]
                for door in (left,right):
                    _,got=world.commit(model.body,item.final_state,e,world.Choice(z,door,torch.zeros_like(z),torch.zeros_like(z)))
                    physical.append(got.payoff)
                outcomes=torch.stack(physical,1)
                if truth is not None:torch.testing.assert_close(outcomes,truth,rtol=0,atol=0)
                truth=outcomes
        actual=torch.stack(paid,1)
        torch.testing.assert_close(actual,truth.gather(1,acts),rtol=0,atol=0)
        out.update({name+"/forecast":estimates,name+"/motor_p":prob,name+"/action":acts,name+"/actual_pay":actual})
    out["door_returns"]=truth
    return {key:val.numpy() for key,val in out.items()}


def run(kind,birth):
    torch.set_num_threads(1);model,_,identity=fit.load_final(kind,birth)
    for seed in SEEDS:
        scene=scenes(seed)
        rng=np.random.default_rng([seed,99])
        draws=torch.tensor(np.repeat(rng.random((ROOTS,2),dtype=np.float32),64,axis=0))
        chunks=[]
        for i in range(0,len(scene.first),256):
            e=world.Episode(**{f.name:getattr(scene,f.name)[i:i+256] for f in fields(scene)})
            choices=draws[i:i+256]
            chunks.append(batch(model,e,*choices.T))
        raw={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        raw.update(source=base.digest(__file__),plan=base.digest(PLAN),body=identity,seed=seed)
        base.storage._write_raw(DATA/str(seed)/f"{kind}-{birth}.npz",**raw)
        print(kind,birth,seed,flush=True)


def summarize():
    report={}
    for seed in SEEDS:
        banks={};sha={};deviation=0.;fallback=0
        for kind in KINDS:
            for birth in base.BIRTHS:
                p=DATA/str(seed)/f"{kind}-{birth}.npz";sha[p.name]=base.digest(p)
                with np.load(p,allow_pickle=False) as z:
                    if str(z["source"])!=base.digest(__file__) or str(z["plan"])!=base.digest(PLAN):
                        raise ValueError("Changed evaluation archive")
                    truth=z["door_returns"];strict=truth[:,0]!=truth[:,1]
                    direction=(truth[:,1]-truth[:,0])/2;row=np.arange(len(truth));q=z["self_q"];fair=z["random_q"]
                    deviation=max(deviation,float(z["norm_error"].max()))
                    fallback+=int(z["public_fallback_ticks"].sum())
                    # Shift one whole geometry, and independently permute balanced law
                    # configurations within it. This keeps the query histogram exact.
                    remapped=np.random.default_rng([seed,39389]).permuted(q.reshape(ROOTS,64),axis=1)
                    unrelated=np.roll(remapped,1,axis=0).ravel()
                    for name in MODES:
                        forecast=z[name+"/forecast"];pay=z[name+"/actual_pay"];side=z[name+"/action"]
                        np.testing.assert_array_equal(pay,truth[row[:,None],side])
                        counter=forecast[row,1-q]
                        diff=(forecast[:,1]-forecast[:,0])*direction
                        compare=lambda a,mask: a[mask].reshape(ROOTS,-1).mean(1)
                        correct=pay[row,q]>0
                        actual=side[row,q]
                        values={
                            "minus_mse":-((forecast-truth)**2).mean(1).reshape(ROOTS,64).mean(1),
                            "unasked_sign": ((counter*truth[row,1-q])>0).reshape(ROOTS,64).mean(1),
                            "unwalked_sign":(forecast[row,1-actual]*truth[row,1-actual]>0).reshape(ROOTS,64).mean(1),
                            "queried_sign":(forecast[row,q]*truth[row,q]>0).reshape(ROOTS,64).mean(1),
                            "independent_drive_requested_sign":(forecast[row,unrelated]*truth[row,q]>0).reshape(ROOTS,64).mean(1),
                            "strict_rank":compare((diff>0)+.5*(diff==0),strict),
                            "strict_signed_gap":compare(diff,strict),
                            "strict_preferred":compare(correct,strict),
                            "strict_own_minus_fair":compare(correct.astype(float)-(pay[row,fair]>0),strict),
                            "strict_own_minus_opposite":compare(correct.astype(float)-(pay[row,1-q]>0),strict),
                            "ties_sign":compare((forecast*truth>0).mean(1),~strict),
                        }
                        group=banks.setdefault(kind+"/"+name,{})
                        for key,v in values.items():group.setdefault(key,[]).append(v)
        matrices={k:{m:np.stack(v) for m,v in d.items()} for k,d in banks.items()}
        comparisons=[("binding/own","misbound/own")]+[
            (kind+"/own",kind+"/"+control) for kind in KINDS for control in ("cut","echo","wrong_history","blank","drive_flip")]
        report[str(seed)]=dict(scores=stats.summarize(matrices,ROOTS,base.BIRTHS,seed+89,comparisons),
                               raw_sha256=sha,max_echo_norm_error=deviation,public_fallback_ticks=fallback)
    base.storage._write_json(base.EVIDENCE/"whatif-eval-results.json",dict(reports=report,plan=base.digest(PLAN),
        source=base.digest(__file__),geometry_roots=ROOTS,model_seeds=base.BIRTHS))
    for seed,v in report.items():
        for k in ("binding/own","misbound/own","binding/echo","binding/wrong_history","binding/cut","binding/own minus binding/echo"):
            scores=v["scores"][k]
            print(seed,k,{n:(round(scores[n]["mean"],4),np.round(scores[n]["ci95"],4).tolist()) for n in
                          ("strict_rank","unasked_sign","strict_preferred","strict_own_minus_fair")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kind",choices=KINDS);p.add_argument("--birth",type=int)
    a=p.parse_args()
    if a.kind:run(a.kind,a.birth)
    else:summarize()
