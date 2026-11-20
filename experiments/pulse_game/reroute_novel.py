"""Exhaustive held-out prior room maps and sampled first choices after a local route change."""

import argparse
from itertools import permutations, product

import numpy as np
import torch

from . import reroute_fit as fit, reroute_fit_transfer as transfer, reroute_world as world
from . import noon_market_stats as stats


PERMS=tuple(permutations(range(3)))
KINDS=("terminal","next_room")
MODES=("natural","blank","before")
SEEDS=(5469301,5469361)
DRAWS=32
PLAN=fit.EVIDENCE/"novel-map-plan.md"
DATA=fit.DATA/"novel-map"


def panel():
    prior=[];doors=[];room=[];observation=[];visit=[];goals=[];idx=[];cases=[]
    for i,(x,y,z) in enumerate(PERMS):
        for dm,look,order,update,goal in product(range(2),range(2),range(2),range(3),world.GOAL_ORDERS):
            mapping=(dm,1-dm)
            old=(x,y)
            observed=(old[look],old[1-look],z)[update]
            after=list(old);after[look]=observed
            old_paid=[goal[old[r]] for r in mapping];new_paid=[goal[after[r]] for r in mapping]
            case=(world.TIE if new_paid[0]==new_paid[1] else world.UNCHANGED if update==0 else
                  world.FLIP if np.argmax(old_paid)!=np.argmax(new_paid) else world.STAY)
            prior.append(old);doors.append(mapping);room.append(look);observation.append(observed)
            visit.append((order,1-order));goals.append(goal);idx.append(i);cases.append(case)
    pack=world.Episode(*(torch.tensor(k,dtype=torch.long) for k in (doors,prior,room,observation,visit)),
                       torch.tensor(goals,dtype=torch.float32))
    mask=transfer.heldout_map(pack).numpy()
    expected={i for i,p in enumerate(PERMS) if sum(p[a]>p[b] for a in range(3) for b in range(a+1,3))%2}
    assert set(np.array(idx)[mask])==expected and mask.sum()==len(idx)//2
    return pack,np.array(idx),np.array(cases),mask


def random_tape(seed,rows):
    return np.random.default_rng(seed).random((rows,DRAWS,2),dtype=np.float32)


@torch.no_grad()
def collect(kind,birth):
    torch.set_num_threads(1)
    model,_,digest=transfer.load_final(kind,birth)
    e,idx,case,ood=panel()
    cache={}
    for offset in range(0,len(e),144):
        sub=e[slice(offset,offset+144)]
        s=world.prepare(model,sub)
        n=len(sub)
        p_q=model.logit(s).sigmoid()
        data={"query_probability":p_q.numpy()}
        left=torch.zeros(n,dtype=torch.long)
        for mode in MODES:
            action_p=[];truth=[];est=[];rooms=[]
            for j,q in enumerate((left,left+1)):
                tr=world.try_side(model,s,q,mode="blank" if mode=="blank" else "natural")
                # The 'before' comparator uses the physical action sampled at the initial state.
                decision_state=s if mode=="before" else tr.final
                action_p.append(model.logit(decision_state).sigmoid())
                est.append(tr.forecast);rooms.append(tr.mid)
                payout=[]
                for door in (left,left+1):
                    decision=world.select(model,decision_state,torch.zeros(n),0,forced=door)
                    # Preserve full time even when the chosen side was fixed before the trial.
                    observed=world.settle(model,tr.final,sub,decision)
                    payout.append(observed.paid)
                truth.append(torch.stack(payout,1))
            torch.testing.assert_close(truth[0],truth[1],atol=0,rtol=0)
            torch.testing.assert_close(truth[0],world.oracle(sub).payoffs,atol=0,rtol=0)
            data["payoffs"]=truth[0].numpy()
            data[mode+"/probability"]=torch.stack(action_p,1).numpy()
            data[mode+"/estimate"]=torch.stack(est,1).numpy()
            data[mode+"/mid"]=torch.stack(rooms,1).numpy()
        for name,val in data.items():cache.setdefault(name,[]).append(val)
    archive={k:np.concatenate(v) for k,v in cache.items()}
    archive.update(map_id=idx,case=case,ood=ood,room_of_door=e.door_rooms.numpy(),prior=e.prior_endpoints.numpy())
    for seed in SEEDS:
        draws=random_tape(seed,len(e));r=np.arange(len(e))[:,None]
        query=(draws[:,:,0]<archive["query_probability"][:,None]).astype(np.int8)
        archive[f"{seed}/query"]=query
        for mode in MODES:
            action=(draws[:,:,1]<archive[mode+"/probability"][r,query]).astype(np.int8)
            paid=archive["payoffs"][r,action]
            archive[f"{seed}/{mode}/action"]=action
            archive[f"{seed}/{mode}/paid"]=paid.astype(np.int8)
    fit.storage._write_raw(DATA/f"{kind}-{birth}.npz",**archive,source=fit.digest(__file__),plan=fit.digest(PLAN),
                           parent=digest,fit=fit.digest(transfer.__file__))


def summarize():
    e,idx,case,ood=panel()
    result={};hashes={};separation={}
    for split,mask in (("unseen",ood),("seen",~ood)):
        ids=np.unique(idx[mask])
        separation[split]=[list(PERMS[i]) for i in ids]
        series={}
        for kind in KINDS:
            for b in fit.BIRTHS:
                path=DATA/f"{kind}-{b}.npz"
                hashes[path.name]=fit.digest(path)
                with np.load(path,allow_pickle=False) as z:
                    assert str(z["source"])==fit.digest(__file__) and str(z["plan"])==fit.digest(PLAN)
                    assert str(z["parent"])==fit.digest(transfer.destination(kind,b))
                    truth=z["payoffs"];old=world.oracle(e).prior_payoffs.numpy()
                    wanted=truth.argmax(1);q_p=z["query_probability"]
                    same=truth[:,0]==truth[:,1]
                    for mode in MODES:
                        action_p=z[mode+"/probability"]
                        right=(1-q_p)*action_p[:,0]+q_p*action_p[:,1]
                        expected=np.where(same,1.,np.where(wanted==1,right,1-right))
                        forecast=z[mode+"/estimate"];gap=(forecast[:,1]-forecast[:,0])*np.sign(truth[:,1]-truth[:,0])
                        metrics={"expected":expected,"forecast_rank":(gap>0)+.5*(gap==0),
                                 "room_accuracy":((z[mode+"/mid"]>0)==z["room_of_door"]).mean(1),
                                 "old_best":np.where(same,1.,old.argmax(1)==wanted)}
                        for seed in SEEDS:
                            draws=random_tape(seed,len(idx));q=(draws[:,:,0]<q_p[:,None]).astype(np.int8)
                            a=draws[:,:,1]<action_p[np.arange(len(idx))[:,None],q]
                            np.testing.assert_array_equal(a,z[f"{seed}/{mode}/action"])
                            paid=truth[np.arange(len(idx))[:,None],a.astype(int)]
                            np.testing.assert_array_equal(paid,z[f"{seed}/{mode}/paid"])
                            metrics[f"{seed}/choice"]=(paid==truth.max(1)[:,None]).mean(1)
                            metrics[f"{seed}/paid"]=paid.mean(1)
                        group=series.setdefault(kind+"/"+mode,{})
                        for ci,label in enumerate(world.CASE_NAMES):
                            for k,v in metrics.items():
                                if ci==world.TIE and k=="forecast_rank":continue
                                group.setdefault(label+"/"+k,[]).append(
                                    np.array([v[(idx==i)&(case==ci)].mean() for i in ids]))
        arrays={g:{k:np.stack(v) for k,v in data.items()} for g,data in series.items()}
        contrasts=[("next_room/natural","terminal/natural")]
        contrasts.extend((g+"/natural",g+"/"+c) for g in KINDS for c in ("blank","before"))
        result[split]=stats.summarize(arrays,len(ids),fit.BIRTHS,1639821,contrasts)
    fit.storage._write_json(fit.EVIDENCE/"novel-map-results.json",dict(
        results=result,raw_sha256=hashes,plan=fit.digest(PLAN),support=separation,draw_seeds=SEEDS,
        draws_per_exact_cell=DRAWS,cells_per_split=len(e)//2,
        note="Three distinct prior room-to-endpoint pairs in each partition, enumerated exhaustively. "
             "Intervals cross trained births and prior maps, not nominal repeated random draws."))
    for group in ("terminal/natural","next_room/natural","next_room/natural minus terminal/natural",
                  "next_room/natural minus next_room/blank"):
        vals=result["unseen"][group]
        print(group,{q:np.round(100*np.array([vals[q+"/expected"]["mean"],*vals[q+"/expected"]["ci95"]]),2).tolist()
                     for q in ("flip","stay","unchanged")},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kind",choices=KINDS);p.add_argument("--birth",type=int)
    a=p.parse_args()
    if a.kind:collect(a.kind,a.birth)
    else:summarize()
