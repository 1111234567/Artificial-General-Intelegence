"""Swap the replica's intermediate room activity while crossing downstream consequences."""

import argparse
from dataclasses import replace
from itertools import product

import numpy as np
import torch

from . import reroute_fit as fit, reroute_world as world
from . import noon_market_stats as statistics
from .reroute_novel import PERMS


KINDS=("terminal","next_room")
SEEDS=(93156371,93156413)
PLAN=fit.EVIDENCE/"midpoint-plan.md"
DATA=fit.DATA/"midpoint"


def events(seed):
    cells=np.array(list(product(range(6),range(2),range(2),range(2),range(2),range(8))))
    mp,door,inspection,first,query,_=cells.T
    roles=np.array(PERMS)[mp]
    room_map=np.stack((door,1-door),1)
    rr=np.arange(len(mp))
    other=roles[rr,1-inspection]
    old=roles[rr,inspection]
    fixed=np.ones((len(mp),3),dtype=np.float32)
    fixed[rr,other]=0
    fixed[rr,old]=-1
    base=world.Episode(torch.tensor(room_map),torch.tensor(roles[:,:2]),torch.tensor(inspection),
                       torch.tensor(roles[:,2]),torch.tensor(np.stack((first,1-first),1)),torch.tensor(fixed))
    donor=replace(base,observed_endpoint=torch.tensor(other))
    mirror=world.swap_room_labels(donor)
    rng=np.random.default_rng(seed)
    draw=torch.from_numpy(rng.random(len(base),dtype=np.float32))
    # Matched ten-of-twenty coordinate exchange, constant across recipient goals and downstream layouts.
    order=rng.random((len(base),20)).argsort(1)
    signs=torch.from_numpy(np.where(order<10,-1.,1.).astype(np.float32))
    return base,donor,mirror,torch.tensor(query),draw,signs,mp


def outcomes(model,state,e,query,draw):
    act=world.select(model,state,draw,0.)
    settled=world.settle(model,state,e,act)
    assert torch.equal(world.oracle(e).payoffs.gather(1,act.side[:,None])[:,0],settled.paid)
    direction=2*query-1
    return dict(motor=(act.logit*direction).sigmoid(),chose_q=(act.side==query).float(),pay=settled.paid)


def continuation(model,natural,e,query,draw,shadow,main_room=None):
    st=natural.states[world.MID_TICK-1]
    pri=st.h[:,:world.REGION]
    if main_room is not None:
        pri=pri.clone();pri[:,world.MID_READOUT]=main_room
    st=replace(st,h=torch.cat((pri,shadow),1))
    zero=st.h.new_zeros((len(e),world.PORTS))
    forecast=None
    for j in range(world.MID_TICK,world.TRIAL_TICKS):
        st,_=model.tick(st,zero)
        if j==world.FORECAST_TICK-1:forecast=st.h[:,world.READOUT]
    return st,dict(forecast=forecast,**outcomes(model,st,e,query,draw))


@torch.no_grad()
def measure(kind,birth):
    torch.set_num_threads(1)
    model,_,identity=fit.load_final(kind,birth)
    for seed in SEEDS:
        base,donor,opp,q,draw,signs,mp=events(seed)
        donor0=world.oracle(donor);donor1=world.oracle(opp)
        assert torch.equal(donor0.payoffs,donor1.payoffs) and torch.all(donor0.payoffs==0)
        assert torch.equal(donor0.endpoints,donor1.endpoints) and torch.equal(donor0.prior_payoffs,donor1.prior_payoffs)
        assert torch.equal(donor.prior_endpoints.gather(1,donor.door_rooms),opp.prior_endpoints.gather(1,opp.door_rooms))
        assert torch.equal(donor0.rooms,1-donor1.rooms)
        packs=[]
        for lo in range(0,len(base),128):
            sl=slice(lo,lo+128)
            d1,d2=donor[sl],opp[sl];receiver=base[sl];qs=q[sl];u=draw[sl]
            a=world.try_side(model,world.prepare(model,d1),qs)
            b=world.try_side(model,world.prepare(model,d2),qs)
            room=d1.door_rooms.gather(1,qs[:,None])[:,0]
            zero_src=torch.where(room[:,None]==0,a.mid_activity,b.mid_activity)
            one_src=torch.where(room[:,None]==1,a.mid_activity,b.mid_activity)
            midpoint=(zero_src+one_src)*.5;contrast=(zero_src-one_src)*.5
            shuffled=(midpoint+signs[sl]*contrast,midpoint-signs[sl]*contrast)
            torch.testing.assert_close((shuffled[0]-shuffled[1]).norm(dim=1),(zero_src-one_src).norm(dim=1),rtol=1e-6,atol=5e-7)
            result=dict(donor_distance=(zero_src-one_src).norm(dim=1))
            room0,room1=world.oracle(receiver).endpoints.gather(1,receiver.door_rooms.argsort(1)).unbind(1)
            row=torch.arange(len(receiver))
            natural_prob={}
            for tg in (-1,1):
                goals=receiver.goals.new_zeros((len(row),3))
                goals[row,room0]=tg;goals[row,room1]=-tg
                for label,seen in (("separate",receiver),("joined",d1)):
                    e=replace(seen,goals=goals);prepared=world.prepare(model,e)
                    natural=world.try_side(model,prepared,qs)
                    unchanged,_=continuation(model,natural,e,qs,u,natural.mid_activity)
                    for k in ("h","fast","trace","candidate"):
                        torch.testing.assert_close(getattr(unchanged,k),getattr(natural.final,k),rtol=0,atol=0)
                    for strategy,shadows in (("room",(zero_src,one_src)),("scramble",shuffled),
                                             ("primary_report",(natural.mid_activity,natural.mid_activity))):
                        for ix,shadow in enumerate(shadows):
                            # The forced native room readout is a separate observability probe.
                            marker=torch.full_like(natural.mid,float(2*ix-1)) if strategy=="primary_report" else None
                            _,values=continuation(model,natural,e,qs,u,shadow,marker)
                            for k,v in values.items():
                                result[f"{label}/{strategy}/{tg}/{ix}/{k}"]=v
                    natural_prob[(label,tg)]=natural.mid
            result["natural_room_accuracy"]=torch.stack(
                [(v.sign()==(2*room-1)).float() for v in natural_prob.values()]).mean(0)
            packs.append({k:v.numpy() for k,v in result.items()})
        raw={k:np.concatenate([p[k] for p in packs]) for k in packs[0]}
        raw.update(map_id=mp,query=q.numpy(),draw=draw.numpy(),plan=fit.digest(PLAN),
                   source=fit.digest(__file__),sha=identity,birth=birth,seed=seed)
        fit.storage._write_raw(DATA/str(seed)/f"{kind}-{birth}.npz",**raw)


def summarize():
    out={};hashes={}
    for seed in SEEDS:
        groups={}
        scene,_,_,queries,_,_,_=events(seed)
        correct_room=scene.door_rooms.gather(1,queries[:,None])[:,0].numpy()
        for kind in KINDS:
            for b in fit.BIRTHS:
                p=DATA/str(seed)/f"{kind}-{b}.npz";hashes[f"{seed}/{p.name}"]=fit.digest(p)
                with np.load(p,allow_pickle=False) as z:
                    assert str(z["plan"])==fit.digest(PLAN) and str(z["source"])==fit.digest(__file__)
                    assert str(z["sha"])==fit.digest(fit.destination(kind,b))
                    maps=z["map_id"]
                    for layout in ("separate","joined"):
                        for policy in ("room","scramble","primary_report"):
                            tag="/".join((kind,layout,policy))
                            series=groups.setdefault(tag,{})
                            for key in ("forecast","motor","chose_q","pay"):
                                contrasts=[]
                                for sign in (-1,1):
                                    one,two=(z[f"{layout}/{policy}/{sign}/{room}/{key}"] for room in (0,1))
                                    contrasts.append(sign*(one-two))
                                value=np.mean(contrasts,axis=0)
                                series.setdefault(key,[]).append([float(value[maps==i].mean()) for i in range(6)])
                            # True payout difference for observations with donor matching native room vs other.
                            aligned=np.mean([np.where(correct_room==0,1,-1)*
                                    (z[f"{layout}/{policy}/{s}/0/pay"]-z[f"{layout}/{policy}/{s}/1/pay"]) for s in (-1,1)],axis=0)
                            series.setdefault("aligned_donor_pay",[]).append([aligned[maps==i].mean() for i in range(6)])
        arrays={t:{m:np.array(v) for m,v in d.items()} for t,d in groups.items()}
        pairs=[(f"{k}/separate/room",f"{k}/{b}") for k in KINDS for b in ("joined/room","separate/scramble")]
        pairs.append(("next_room/separate/room","terminal/separate/room"))
        out[str(seed)]=statistics.summarize(arrays,6,fit.BIRTHS,seed+21,pairs)
    fit.storage._write_json(fit.EVIDENCE/"midpoint-results.json",dict(
        results=out,raw_sha256=hashes,seeds=SEEDS,source=fit.digest(__file__),plan=fit.digest(PLAN),
        scale="Half difference of the donor-room-0 versus room-1 differences under two opposing receiver goals. "
              "Motor and chose_q are probabilities on [0,1]. aligned_donor_pay is a separate actual raw-payoff contrast."))
    for seed,data in out.items():
        for name in ("next_room/separate/room","next_room/joined/room","next_room/separate/room minus next_room/separate/scramble"):
            print(seed,name,{k:round(v["mean"],4) for k,v in data[name].items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--kind",choices=KINDS)
    parser.add_argument("--birth",type=int)
    args=parser.parse_args()
    if args.kind:measure(args.kind,args.birth)
    else:summarize()
