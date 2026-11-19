"""Select replay from actually tasted outcomes, then test it inside untouched physical lives."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from itertools import product
import json
import time
import numpy as np

from . import signed_replay as infer, replay_life as embodied, symmetry_grounded as sensory
from . import symmetry_audit as design, mixture_flash as flash, mixture_metric as control
from . import symmetry_world as game, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def make(ss,pairs,geom,span=4,noise=.04,jitter=.08):
    v=design.joint(ss,pairs=pairs,probes=1,span=span,noise=noise,jitter=jitter,matched=True)
    return embodied.geometry(v,geom)


def sampled(ss,pairs,geom):
    p=make(ss,pairs,geom)
    n=len(ss)*2;body=p.menu.body;T=body.trials
    r=flash.rollout(control.Explorer(body.seeds,T),p.menu)
    if (r.contact_side<0).any():raise ValueError("Missing sampled estimate")
    ids=np.arange(n)[:,None];j=np.arange(T)
    # These externally present odors are sampled by the actual contacting body;
    # no unsampled taste or hidden relation is part of selection.
    seen=infer.norm(2.*p.menu.chemicals[ids,j,r.contact_side]-1.)
    public_sign=r.observed_reward.sum(2)
    if not np.array_equal(public_sign,r.reward):raise ValueError("Reward sensor mismatch")
    cut=p.exposed
    return (seen[:,:cut]*public_sign[:,:cut,None],body.ctx[:,:cut],np.full(n,cut,int),
            seen[:,cut]*public_sign[:,cut,None],seen[:,cut+1],public_sign[:,cut+1])


def selection_score(cfg,bank):
    out=[]
    for x,ctx,n,anchor,test,y in bank:
        state=infer.initialize(x,ctx,n,cfg)
        row=[]
        for k in range(5):
            if k in (0,1,4):
                q=(test*((anchor[:,None,:]@state).squeeze(1))).sum(1)
                row.append(float(np.mean((y-q)**2)))
            if k<4:state=infer.revise(x,ctx,n,state,cfg)
        out.append(row)
    return np.mean(out,axis=0)


def select(folder):
    base=np.random.default_rng(26514381).integers(3_000_000,475_000_000,112).tolist()
    bank=[sampled(base,n,"rotation") for n in (8,16,24)]
    proposals=[infer.Settings(lookback=i,temperature=t,threshold=h,polar=p,balance=b,mix=.7)
               for i,t,h,p,b in product((4,8,12),(4.,9.,17.),(.43,.72),(False,True),(False,True))]
    results=[selection_score(c,bank) for c in proposals]
    val=np.random.default_rng(9183751).integers(3_000_000,475_000_000,128).tolist()
    confirm=[sampled(val,n,"rotation") for n in (8,16,24)]
    ranked=np.argsort(np.mean(results,axis=1))[:8].tolist()
    second=[selection_score(proposals[k],confirm).tolist() for k in ranked]
    pick=ranked[int(np.argmin(np.mean(second,axis=1)))]
    _write(folder/"frozen.json",dict(chosen=asdict(proposals[pick]),discovery=base,validation=val,
          all_proposals=[asdict(c) for c in proposals],discovery_brier=np.array(results).tolist(),
          shortlist=ranked,selected=pick,validation_brier=second,criterion="actually consumed signed probe"))
    print("selected",proposals[pick],flush=True)


def evaluate(job):
    folder,dest,ss,birth,pairs,geom,variants=job
    path=dest/f"{birth}-{geom}-{pairs}.json"
    if path.exists():return
    cfg=infer.Settings(**json.loads((folder/"frozen.json").read_text())["chosen"])
    front=sensory.load(OUT/f"symmetry-grounded-v1/public-delta-s{birth}-before.pt")
    p=make(ss,pairs,geom);n=len(ss);records={}
    for label in variants:
        passes,adaptive,at_query,control_name={
          "one_pass":(0,False,True,"true"),"one_revision":(1,False,True,"true"),"four_revisions":(4,False,True,"true"),
          "eight_revisions":(8,False,True,"true"),"validate_replay":(5,True,True,"true"),
          "idle_four":(4,False,False,"true"),"signed_history_erased":(4,False,True,"unsigned_history"),
          "inverse_operator":(4,False,True,"invert"),"diagonal":(4,False,True,"cut_cross_dimensions")}[label]
        brain=embodied.Actor(front,cfg,ss+ss,passes=passes,adaptive=adaptive,control=control_name,at_query=at_query)
        res=flash.rollout(brain,p.menu,hidden_after=p.exposed+1)
        d=game.metrics(p,res);a,b=np.split(d["first"],2)
        y=res.observed_reward[:,:p.exposed]
        records[label]=dict(first=((a+b)*.5).tolist(),both=(a*b).tolist(),
               coverage=((d["contact"][:n]+d["contact"][n:])*.5).tolist(),
               teaching_sign_waveforms=float(np.mean((y[:n]==y[n:]).all(2))),
               mean_total_revision_steps=float(brain.work.mean()),mean_accepted_steps=float(brain.accepts.mean()))
    _write(path,dict(birth=birth,geometry=geom,pairs=pairs,seeds=ss,config=asdict(cfg),rows=records))
    print(birth,geom,pairs,{k:round(float(np.mean(v["first"])),3) for k,v in records.items()},flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--worlds",type=int,default=192);args=ap.parse_args()
    folder=OUT/"continuous-replay-v1";folder.mkdir(exist_ok=True);dest=audit.OUT/"continuous-replay-sealed";dest.mkdir(exist_ok=True)
    if not (folder/"frozen.json").exists():select(folder)
    ss=np.random.default_rng(123859041).integers(3_000_000,475_000_000,args.worlds).tolist()
    jobs=[]
    for birth in (7243,4187,9679):
        for pairs,geom in ((4,"rotation"),(12,"rotation"),(24,"rotation"),(48,"rotation"),(24,"quarter"),(24,"antipodal"),(24,"stretched")):
            labels=["one_pass","four_revisions","validate_replay"]
            if birth==7243 and pairs in (12,24) and geom=="rotation":
                labels+=["one_revision","eight_revisions","idle_four","signed_history_erased","inverse_operator","diagonal"]
            jobs.append((folder,dest,ss,birth,pairs,geom,labels))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(evaluate,j) for j in jobs]):f.result()
    _write(dest/"complete.json",dict(worlds=len(ss),graphs=3,cases=len(jobs)))


if __name__=="__main__":main()
