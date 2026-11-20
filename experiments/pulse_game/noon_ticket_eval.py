"""Paired analysis of voluntary information purchase, its price, and its causal return."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_ticket as task
from .interleave_capacity_report import Crossed


STATES=("known","unseen","unfed")
PRICES=(.05,.15,.25,.40,.55)
TRAIN_PRICES=(0,2,4)


def public_evidence(t,history):
    # Both signs of a scent identify the same physical object; the two menus
    # always display an axis and its opposite. Only previously delivered tastes count.
    v=t["study"];query=t["preview"][:,None,:]
    same=((v-query).abs().amax(2)<1e-6)|((v+query).abs().amax(2)<1e-6)
    return (same&(history[:,:,1]!=0)).any(1),same.any(1)


@torch.no_grad()
def measure(g,r,t,extra=True):
    view,state,_,hist=task.prefix(g,r,t)
    natural=task.finish(view,state,t,"native")
    all_buy=torch.ones_like(natural["buy"])
    bought=task.finish(view,state,t,"native",decision=all_buy)
    skipped=task.finish(view,state,t,"native",decision=~all_buy)
    twin=torch.arange(len(all_buy))^4
    which=natural["buy"]
    for field in ("utility","gross","final","taste","review_side"):
        torch.testing.assert_close(natural[field],torch.where(which,bought[field],skipped[field]),rtol=0,atol=0)
    observed,seen=public_evidence(t,hist)
    if not torch.equal(observed,t["evidenced"]) or not torch.equal(seen,t["evidenced"]|t["omitted"]):
        raise ValueError("Public evidence did not reconstruct the paired histories")
    results={"known":observed,"seen":seen,"buy":which,"choice":natural["choice"],"logit":natural["logit"],
             "utility":natural["utility"],"gross":natural["gross"],"forced_buy":bought["utility"],
             "forced_skip":skipped["utility"],"buy_correct":bought["gross"],"skip_correct":skipped["gross"],
             "target":t["final_value"],"route_side":t["offer_side"],"final":natural["final"],
             "study_side":hist[:,:,0],"study_taste":hist[:,:,1]}
    swapped=torch.where(which[twin],bought["utility"],skipped["utility"])
    results["matched_random"]=.5*(natural["utility"]+swapped)
    if extra:
        bad=task.finish(view,state,t,"native",deliver="zero")
        results.update(no_information=bad["utility"],no_information_correct=bad["gross"])
        if not torch.equal(natural["choice"],bad["choice"]):raise ValueError("Action anticipated its own withheld taste")
        for k in ("graph","hidden","trace","all"):
            components=("hidden","trace","graph") if k=="all" else k
            graft=task.net.swap(state,state.subset(twin),components)
            l,r_menu=t["offer_left"],t["offer_right"]
            a=task.net.act(view,graft,l,r_menu,t["draws"][:,2,0])
            results["graft_"+k]=a.side==t["offer_side"]
        if not torch.equal(results["graft_all"],which[twin]):raise ValueError("Complete state swap failed to reproduce donor route")
    return {k:v.detach().cpu().numpy() for k,v in results.items()}


def batch_eval(g,r,seed,worlds,extra):
    cells={}
    for pi,price in enumerate(PRICES):
        alternatives=[]
        for label in ("unseen","omitted"):
            t=task.tape(seed,0,0,worlds,domain=572911,price=price,alternative=label)
            chunks=[]
            for first in range(0,worlds,32):
                z={k:v[8*first:8*(first+32)] for k,v in t.items()}
                chunks.append(measure(g,r,z,extra))
            alternatives.append({k:np.concatenate([z[k] for z in chunks]).reshape(worlds,2,4,*chunks[0][k].shape[1:])
                                 for k in chunks[0]})
            # Eight lanes are four law assignments, paired within each evidence condition.
        for k,left in alternatives[0].items():
            right=alternatives[1][k]
            if k.startswith("graft_") or k=="matched_random":
                # The known recipient has two different possible donors. Average
                # its two yoked outcomes; this preserves exactly the purchase budget.
                known=.5*(left[:,0].astype(float)+right[:,0].astype(float))
            else:
                np.testing.assert_array_equal(left[:,0],right[:,0])
                known=left[:,0]
            cells.setdefault(k,[]).append(np.stack((known,left[:,1],right[:,1]),axis=1))
    return {k:np.stack(v,axis=1) for k,v in cells.items()}


def average_worlds(v):
    # Known is half of the training population; unseen and unrewarded make one quarter each.
    w=np.asarray([.5,.25,.25])
    return (v.mean(-1)*w).sum(-1)


def metrics(cal,z):
    # Arrays are world × tariff × evidence status × paired private assignments.
    out={};indices=list(TRAIN_PRICES)
    for key in ("utility","gross","buy","forced_buy","forced_skip","matched_random","no_information"):
        out[key]=average_worlds(z[key])[:,indices].mean(1)
    out["selection_gain"]=out["utility"]-out["matched_random"]
    out["taste_gain"]=out["utility"]-out["no_information"]
    cal_change=(cal["forced_buy"]-cal["forced_skip"]).mean((0,3))
    unconditional=(cal_change*np.array([.5,.25,.25])).sum(1)
    out["price_only"]=average_worlds(np.where(unconditional[None,:,None,None]>0,z["forced_buy"],z["forced_skip"]))[:,indices].mean(1)
    out["public_tracker"]=average_worlds(np.where(cal_change[None,:,:,None]>0,z["forced_buy"],z["forced_skip"]))[:,indices].mean(1)
    out["over_price_only"]=out["utility"]-out["price_only"]
    out["gap_to_public_tracker"]=out["public_tracker"]-out["utility"]
    for i,p in enumerate(PRICES):
        for j,s in enumerate(STATES):
            for key in ("buy","gross","utility","forced_buy","forced_skip","graft_graph","graft_hidden","graft_trace","graft_all"):
                out[f"{p}/{s}/{key}"]=z[key][:,i,j].mean(-1)
        out[f"{p}/selection_gain"]=average_worlds(z["utility"]-z["matched_random"])[:,i]
        out[f"{p}/unseen_minus_known_buy"]=(z["buy"][:,i,1].astype(float)-z["buy"][:,i,0]).mean(-1)
        out[f"{p}/unfed_minus_known_buy"]=(z["buy"][:,i,2].astype(float)-z["buy"][:,i,0]).mean(-1)
    out["unseen_minus_known_buy"]=np.mean([out[f"{PRICES[i]}/unseen_minus_known_buy"] for i in indices],axis=0)
    out["unfed_minus_known_buy"]=np.mean([out[f"{PRICES[i]}/unfed_minus_known_buy"] for i in indices],axis=0)
    return out


def run(cohort,step,worlds,model_arms=("native","coin"),label=None,seed_offset=0):
    torch.set_num_threads(1)
    ids=task.births(cohort);base=task.DATA/"ticket-eval"/(label or cohort)/str(step)
    base.mkdir(parents=True,exist_ok=True)
    seed=(5207791 if cohort=="original" else 6297749)+seed_offset
    sha=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    evaluator=sha(__file__);archive={};hashes={}
    for arm in model_arms:
        rows=[]
        for b in ids:
            f=task.old.checkpoint_path(task.DATA/"tickets"/cohort,"odor",arm,b,step)
            g,r=task.old.load_checkpoint(task.DATA/"tickets"/cohort,"odor",arm,b,step)
            checkpoint=sha(f);rawfile=base/f"{arm}-{b}.npz"
            if rawfile.exists():
                with np.load(rawfile) as z:
                    if str(z["checkpoint"])!=checkpoint or str(z["evaluator"])!=evaluator or int(z["seed"])!=seed:
                        raise ValueError("Cached choice assay has different inputs")
                    rows.append({k[7:]:z[k] for k in z.files if k.startswith("metric/")})
            else:
                cal=batch_eval(g,r,5207719+seed_offset,max(48,worlds//2),False)
                raw=batch_eval(g,r,seed,worlds,True)
                vals=metrics(cal,raw);rows.append(vals)
                task.storage._write_raw(rawfile,**raw,**{"metric/"+k:v for k,v in vals.items()},
                                        checkpoint=checkpoint,evaluator=evaluator,seed=seed)
            hashes[rawfile.relative_to(task.DATA).as_posix()]=sha(rawfile)
            hashes[f.relative_to(task.DATA).as_posix()]=checkpoint
            print(cohort,step,arm,b,flush=True)
        archive[arm]={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    if "native" in archive and "coin" in archive:
        archive["native minus coin"]={k:v-archive["coin"][k] for k,v in archive["native"].items()}
    bounds=Crossed(len(ids),worlds,seed+97,3000)
    report=dict(cohort=cohort,step=step,worlds=worlds,births=ids,prices=PRICES,seed=seed,hashes=hashes,
                summary={a:{k:bounds(v) for k,v in values.items()} for a,values in archive.items()},
                notes="Within each world, tariff and four private X/B law combinations, exchange purchase indicators "
                      "between true-target-experience and the matched alternative. The matched-random policy averages "
                      "the natural and exchanged allocations, paying exactly the same total cost. The calibrated "
                      "tariff-only and public-history tables are fixed on distinct geometries. All branches are "
                      "evaluated in the same physically contingent test market.")
    name=label or cohort
    task.storage._write_json(task.EVIDENCE/f"ticket-{name}-{step}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--step",type=int,required=True);p.add_argument("--worlds",type=int,default=192)
    p.add_argument("--arms",nargs="+",default=["native","coin"]);p.add_argument("--label");p.add_argument("--seed-offset",type=int,default=0)
    a=p.parse_args();run(a.cohort,a.step,a.worlds,a.arms,a.label,a.seed_offset)
