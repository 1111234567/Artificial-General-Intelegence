"""Replay paired action draws and infer the endpoint implied by the six goal-conditioned forecasts."""

from collections import defaultdict

import numpy as np

from . import reroute_novel as novel, reroute_world as world
from . import noon_market_stats as stats


def pairs(e,cases):
    keys=defaultdict(dict)
    for i in range(len(e)):
        key=(*e.prior_endpoints[i].tolist(),int(e.door_rooms[i,0]),int(e.inspected_room[i]),
             int(e.visit_order[i,0]),*e.goals[i].tolist())
        keys[key][int(cases[i])]=i
    joined=[]
    for triple in keys.values():
        changed=triple.get(world.FLIP,triple.get(world.STAY))
        if changed is None or len(triple)!=3:
            raise ValueError("Missing matched change")
        joined.append((triple[world.UNCHANGED],changed))
    return np.asarray(joined).T


def common_draw_outcomes(qold,qnew,actold,actnew,oldright,newright):
    """Integrate the shared query and action uniforms exactly over their four possible cells."""
    z=np.zeros(len(qold));correct=z.copy();reverse=z.copy();change=z.copy()
    for old_q in (0,1):
        for new_q in (0,1):
            lo=np.maximum(0 if old_q else qold,0 if new_q else qnew)
            hi=np.minimum(qold if old_q else 1,qnew if new_q else 1)
            pq=np.maximum(hi-lo,0)
            a,b=actold[:,old_q],actnew[:,new_q]
            for a0,a1,prob in ((1,1,np.minimum(a,b)),(0,0,1-np.maximum(a,b)),
                               (1,0,np.maximum(a-b,0)),(0,1,np.maximum(b-a,0))):
                correct+=pq*prob*((oldright==a0)&(newright==a1))
                reverse+=pq*prob*((oldright!=a0)&(newright!=a1))
                if a0!=a1:change+=pq*prob
    return correct,reverse,change


def endpoint_groups(e,case):
    grouped=defaultdict(list)
    for row in np.flatnonzero(np.isin(case,(world.FLIP,world.STAY))):
        key=(*e.prior_endpoints[row].tolist(),int(e.door_rooms[row,0]),
             int(e.inspected_room[row]),int(e.visit_order[row,0]))
        grouped[key].append(row)
    rows=np.array(list(grouped.values()))
    assert rows.shape[1]==6
    return rows


def calculate(z,e,ids,case,requested,goalsets,mode,draws):
    a0,a1=requested
    qr=z["query_probability"];ar=z[mode+"/probability"];true=z["payoffs"]
    oldside=true[a0].argmax(1);newside=true[a1].argmax(1)
    cc,ww,switch=common_draw_outcomes(qr[a0],qr[a1],ar[a0],ar[a1],oldside,newside)
    assert np.allclose((cc+ww)[oldside!=newside],switch[oldside!=newside],atol=1e-6)
    out={}
    def store(key,vals,mapids):
        out[key]=np.array([vals[mapids==i].mean() for i in range(6)])
    for ca,tag in ((world.FLIP,"flip"),(world.STAY,"stay")):
        eligible=case[a1]==ca
        for k,v in (("two_correct",cc),("two_wrong",ww),("switch",switch)):
            store(tag+"/"+k,v[eligible],ids[a1][eligible])
    for seed,tape in draws.items():
        r=np.arange(len(a0))[:,None];qs,acts=tape[:len(a0),:,0],tape[:len(a0),:,1]
        def choose(ix):
            side=(qs<qr[ix,None]).astype(np.int64)
            return acts<ar[ix[r],side]
        first,second=choose(a0),choose(a1)
        for ca,tag in ((world.FLIP,"flip"),(world.STAY,"stay")):
            ok=case[a1]==ca
            store(f"{tag}/{seed}/two_correct",((first==oldside[:,None])&(second==newside[:,None])).mean(1)[ok],ids[a1][ok])
    start=goalsets[:,0];inspected=e.inspected_room.numpy()[start]
    selected=(e.door_rooms.numpy()[start,0]!=inspected).astype(int)
    forecast=z[mode+"/estimate"][goalsets,selected[:,None]]
    loss=((forecast[:,:,None]-e.goals.numpy()[goalsets])**2).mean(1)
    minima=np.isclose(loss,loss.min(1,keepdims=True),atol=1e-6,rtol=0)
    choice=minima/minima.sum(1,keepdims=True)
    current=e.observed_endpoint.numpy()[start]
    old=e.prior_endpoints.numpy()[start,inspected]
    other=3-current-old
    r=np.arange(len(start))
    for name,index in (("updated",current),("previous",old),("other",other)):
        store("implied/"+name,choice[r,index],ids[start])
    mid=z[mode+"/mid"][goalsets,selected[:,None]]
    store("implied/correct_room",((mid>0)==inspected[:,None]).mean(1),ids[start])
    return out


def main():
    e,ids,case,hold=novel.panel()
    requested=pairs(e,case);goalsets=endpoint_groups(e,case)
    draws={s:novel.random_tape(s+959,len(requested[0])) for s in novel.SEEDS}
    groups={}
    for kind in novel.KINDS:
        for b in novel.fit.BIRTHS:
            with np.load(novel.DATA/f"{kind}-{b}.npz") as z:
                for mode in ("natural","blank"):
                    result=calculate(z,e,ids,case,requested,goalsets,mode,draws)
                    d=groups.setdefault(kind+"/"+mode,{})
                    for k,v in result.items():d.setdefault(k,[]).append(v)
    archive={}
    for tag,selection in (("unseen",hold),("seen",~hold)):
        subset=np.unique(ids[selection])
        matrices={name:{k:np.stack(r)[:,subset] for k,r in data.items()} for name,data in groups.items()}
        comparisons=[("next_room/natural","terminal/natural")]+[(k+"/natural",k+"/blank") for k in novel.KINDS]
        archive[tag]=stats.summarize(matrices,3,novel.fit.BIRTHS,267593,comparisons)
    novel.fit.storage._write_json(novel.fit.EVIDENCE/"update-results.json",dict(scores=archive,
        source=novel.fit.digest(__file__),plan=novel.fit.digest(novel.fit.EVIDENCE/"update-diagnostic-plan.md"),
        prior_evaluation=novel.fit.digest(novel.fit.EVIDENCE/"novel-map-results.json")))
    print("Paired changed outcomes and independent goal decoding verified.")


if __name__=="__main__":
    main()
