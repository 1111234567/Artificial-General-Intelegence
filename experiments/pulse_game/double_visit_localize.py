"""Ask for a room directly and intervene on native predictions before real choice."""

import argparse

import numpy as np
import torch

from . import double_visit_world as world, double_visit_fit as fit
from . import interleave_revisit_train as storage, noon_market_stats as statistics


ROOT=fit.FIT_ROOT
EVIDENCE=fit.PROTOCOL.parent
PLAN=EVIDENCE/"localization-plan.md"
KINDS=("baseline","identity")
SEEDS=(7160803,7160831)
SOURCES=("door","room_cue","door_cue","endpoint_cue","blank")


def external_probe(model,prepared,e,side,tag):
    """Diagnostic copies see only the single indicated cue for two ticks."""
    row=torch.arange(len(e))
    room=e.door_rooms[row,side]
    if tag=="room_cue":pulse=world.senses(prepared,room=room)
    elif tag=="door_cue":pulse=world.senses(prepared,door=side)
    elif tag=="endpoint_cue":pulse=world.senses(prepared,endpoint=world.current_endpoints(e)[row,room])
    else:raise ValueError(tag)
    state=prepared;zero=torch.zeros_like(pulse)
    for i in range(world.FORECAST_TICK):
        state,_=model.tick(state,pulse if i<2 else zero)
    return state.h[:,:world.REGION]


def present_after_readout(model,trial,*,given=None,units=None):
    """Change only the declared primary cell(s), then let all recurrent paths run."""
    state=trial.states[world.FORECAST_TICK-1]
    if units is not None:
        from dataclasses import replace
        hidden=state.h.clone();hidden[:,units]=given
        state=replace(state,h=hidden)
    for _ in range(world.TRIAL_TICKS-world.FORECAST_TICK):
        state,_=model.tick(state,world.senses(state))
    return state


@torch.no_grad()
def paired_action(model,e):
    pairs=world.first_observation_pairs(e)
    pairs=pairs[world.heldout_map(e)[pairs[:,0]]]
    raw={}
    scenes=e[pairs.flatten()]
    q=(scenes.door_rooms[:,0]!=scenes.inspected_rooms[:,0]).long()
    for offset in range(0,len(scenes),96):
        sl=slice(offset,offset+96);sub=scenes[sl];query=q[sl];n=len(sub)
        swap=torch.arange(n)^1
        tr=world.try_side(model,world.prepare(model,sub),query)
        read=tr.states[world.FORECAST_TICK-1].h
        for name,ix in (("natural",None),("value",slice(14,15)),("identity",world.ENDPOINT_READOUT),
                        ("value_and_identity",[14,16,17,18])):
            alt=present_after_readout(model,tr,units=ix,given=read[swap,ix] if ix is not None and isinstance(ix,slice)
                                      else (read[swap][:,ix] if ix is not None else None))
            if name=="natural":
                for field in ("h","fast","trace","candidate"):
                    torch.testing.assert_close(getattr(alt,field),getattr(tr.final,field),rtol=0,atol=0)
            raw.setdefault(name+"/p_queried",[]).append((model.logit(alt)*(2*query-1)).sigmoid().numpy())
            for seed in SEEDS:
                u=np.random.default_rng(seed).random(len(pairs),dtype=np.float32).repeat(2)[sl]
                decision=world.select(model,alt,torch.from_numpy(u.copy()),0.)
                pay=world.settle(model,alt,sub,decision).paid
                raw.setdefault(f"{name}/{seed}/action",[]).append(decision.side.numpy())
                raw.setdefault(f"{name}/{seed}/pay",[]).append(pay.numpy())
        raw.setdefault("reported_return",[]).append(tr.forecast.numpy())
        raw.setdefault("reported_endpoint",[]).append(tr.endpoint_logits.argmax(1).numpy())
    data={"pair/"+k:np.concatenate(v) for k,v in raw.items()}
    return dict(**data,pairs=pairs.numpy(),forced_question=q.numpy())


@torch.no_grad()
def collect(kind,birth):
    torch.set_num_threads(1)
    model,_,sha=fit.load_final(birth,kind)
    e,_,maps=world.panel()
    collected={}
    for offset in range(0,len(e),192):
        part=e[slice(offset,offset+192)]
        s=world.prepare(model,part)
        for tag in SOURCES:
            samples=[]
            for i in (0,1):
                q=torch.full((len(part),),i,dtype=torch.long)
                if tag in ("blank","door"):
                    tr=world.try_side(model,s,q,mode="blank" if tag=="blank" else "natural")
                    h=tr.states[world.FORECAST_TICK-1].h[:,:world.REGION]
                    if tag=="door":
                        a=world.select(model,tr.final,torch.zeros(len(part)),0.,forced=q)
                        physical=world.settle(model,tr.final,part,a)
                        assert torch.equal(physical.endpoint,world.oracle(part).endpoints[:,i])
                else:h=external_probe(model,s,part,q,tag)
                samples.append(h.numpy())
            collected.setdefault(tag,[]).append(np.stack(samples,1))
    data={k:np.concatenate(v) for k,v in collected.items()}
    truth=world.oracle(e)
    data.update(paired_action(model,e))
    data.update(endpoint=truth.endpoints.numpy(),pay=truth.payoffs.numpy(),room=truth.rooms.numpy(),
                maps=maps.numpy(),kind=kind,birth=birth,model=sha,plan=fit.digest(PLAN),source=fit.digest(__file__))
    storage._write_raw(ROOT/"localization"/f"{kind}-{birth}.npz",**data)


def readout(cal,targets,test):
    x=cal.reshape(-1,cal.shape[-1]).astype(float);y=np.eye(3)[targets.ravel()]
    xm=x.mean(0);ym=y.mean(0);x-=xm
    rms=np.sqrt(np.sum(x**2)/len(x)).clip(1e-8)
    x/=rms
    weights=np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@(y-ym))
    return ((test-xm)/rms)@weights+ym


def summarize():
    e,_,m=world.panel();held=world.heldout_map(e).numpy();map_ids=m.numpy()
    unique_ids=np.unique(map_ids[held]);goal=e.goals.numpy()
    pool={};hashes={}
    def add(group,key,v,ids):
        pool.setdefault(group,{}).setdefault(key,[]).append([float(v[ids==i].mean()) for i in unique_ids])
    for kind in KINDS:
        for birth in fit.BIRTHS:
            path=ROOT/"localization"/f"{kind}-{birth}.npz";hashes[path.name]=fit.digest(path)
            with np.load(path,allow_pickle=False) as z:
                assert str(z["source"])==fit.digest(__file__) and str(z["model"])==fit.digest(fit.destination(birth,kind))
                assert str(z["plan"])==fit.digest(PLAN)
                target=z["endpoint"];pairs=z["pairs"];maps=map_ids[pairs[:,0]]
                early=(e.door_rooms[pairs[:,0],0]!=e.inspected_rooms[pairs[:,0],0]).numpy().astype(int)
                for tag in SOURCES:
                    x=z[tag];group=kind+"/"+tag
                    outputs={"native":x[:,:,world.ENDPOINT_READOUT],
                             "calibrated":readout(x[~held],target[~held],x)}
                    for name,values in outputs.items():
                        label=values.argmax(2)
                        add(group,name+"/all",((label==target).mean(1))[held],map_ids[held])
                        for which,q in (("early",early),("late",1-early)):
                            y=label[pairs,q[:,None]]==target[pairs,q[:,None]]
                            add(group,name+"/"+which,y.mean(1),maps)
                            add(group,name+"/"+which+"_both",y.all(1),maps)
                    # Compare the actual payoff report and the value implied by its endpoint label;
                    # these analytical scores do not choose for the native agent.
                    value=x[:,:,world.READOUT]
                    add(group,"native/value_mse",((value-z["pay"])**2).mean(1)[held],map_ids[held])
                    implied=goal[np.arange(len(e))[:,None],outputs["native"].argmax(2)]
                    v=implied[pairs,early[:,None]]
                    add(group,"native/endpoint_makes_correct_choices",(v[:,0]>0)&(v[:,1]<0),maps)
                for tag in ("natural","value","identity","value_and_identity"):
                    group=kind+"/act_"+tag
                    prob=z["pair/"+tag+"/p_queried"].reshape(-1,2)
                    both=np.maximum(prob[:,0]-prob[:,1],0)
                    # Common action uniform thresholds, same queried door within each twin pair.
                    add(group,"both_expected",both,maps)
                    add(group,"correct_margin",prob[:,0]-prob[:,1],maps)
                    for seed in SEEDS:
                        action=z[f"pair/{tag}/{seed}/action"].reshape(-1,2)
                        ok=action==np.column_stack((early,1-early))
                        add(group,f"both_{seed}",ok.all(1),maps)
                    score=z["pair/reported_return"].reshape(-1,2)
                    add(group,"both_forecasts_right",(score[:,0]>0)&(score[:,1]<0),maps)
    matrix={g:{k:np.array(v) for k,v in d.items()} for g,d in pool.items()}
    contrasts=[(f"{kind}/room_cue",f"{kind}/door") for kind in KINDS]+[
        (f"{kind}/act_natural",f"{kind}/act_{c}") for kind in KINDS for c in ("value","identity","value_and_identity")]
    for tag in SOURCES:contrasts.append(("identity/"+tag,"baseline/"+tag))
    out=statistics.summarize(matrix,3,fit.BIRTHS,411503,contrasts)
    storage._write_json(EVIDENCE/"localization-results.json",dict(
        results=out,raw_sha256=hashes,source=fit.digest(__file__),plan=fit.digest(PLAN),
        caution="The direct room cue is a diagnostic change of query format. The calibrated decoders are "
                "fit only on old even layouts, never used by the body. Correct terminal labels during the "
                "explicit endpoint cue are supplied for a positive control. The native door predicts without it."))
    for k in ("baseline/door","identity/door","identity/room_cue","identity/door_cue","identity/endpoint_cue","identity/blank"):
        print(k,{m:round(out[k][m]["mean"],3) for m in ("native/early","native/late","calibrated/early","calibrated/late")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=KINDS);parser.add_argument("--birth",type=int)
    args=parser.parse_args()
    if args.kind:collect(args.kind,args.birth)
    else:summarize()
