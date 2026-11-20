"""Test neutral object cues under switched private receipts and donated route choices."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import noon_identity as game, noon_paired_lesson as paired
from .interleave_capacity_report import Crossed


@torch.no_grad()
def all_tests(state,t,tags):
    right=[];raw=[]
    for j in range(t.count):
        item=torch.full((t.lanes,),j,dtype=torch.long);a,value=game.probe(state,t,tags,item)
        right.append(a.side==value);raw.append(a.side)
    return torch.stack(right,1),torch.stack(raw,1)


@torch.no_grad()
def replay(g,r,t,missing,tag,unbound,arm):
    view,s,_,hist=game.prefix(g,r,t,missing,tag)
    a=game.choose(view,s,t,tag,unbound,arm)
    swap=game.choose(view,s,t,tag,unbound,arm,swapped=True)
    flip=game.choose(view,s,t,tag,unbound,arm,flipped=True)
    no_tag=game.choose(view,s,t,torch.zeros_like(tag),torch.zeros_like(unbound),arm)
    ix=torch.arange(t.lanes);tw=ix^1
    true_side=(a.side>0).float()
    policies=dict(native=true_side,random=torch.full_like(true_side,.5),other_history=true_side[tw],
                  swapped=(swap.side>0).float(),reverse_polarity=(flip.side>0).float(),blank_tags=(no_tag.side>0).float())
    if (missing>=0).all():policies["public_missing"]=(t.candidates[:,1]==missing).float()
    raw=dict(choice=a.side.numpy().astype(np.int8),route_logit=a.motor_logit.numpy(),candidates=t.candidates.numpy(),
             missing=missing.numpy(),support=hist.numpy().astype(np.int8),final_truth=(t.law*t.final_orientation).numpy())
    values={"right":true_side,"swap_changes_side":(a.side!=swap.side).float(),"omission_switch":(a.side!=a.side[tw]).float()}
    if "public_missing" in policies:
        values["choose_missing"]=(true_side==policies["public_missing"]).float()
        values["choose_missing_probability"]=torch.where(policies["public_missing"]>0,a.motor_logit.sigmoid(),1-a.motor_logit.sigmoid())
    ordinary,_=all_tests(s,t,tag);values["pre_all"]=ordinary.all(1).float();values["pre_mean"]=ordinary.float().mean(1)
    for mode,allow in (("active",True),("sham",False)):
        good=[]
        for sg in (-1.,1.):
            side=torch.full_like(a.side,sg)
            ev=game.recollect(view,a.state,t,tag,side,write=allow)
            correct,action=all_tests(ev.state,t,tag);good.append(correct)
            raw[f"{mode}/{sg}/raw"]=action.numpy().astype(np.int8)
            raw[f"{mode}/{sg}/received"]=torch.stack((ev.motor.side,ev.earned),1).numpy().astype(np.int8)
            if not allow:
                for field in game.families.capacity.GRAPH_FIELDS:
                    if not torch.equal(getattr(s.graph,field),getattr(ev.state.graph,field)):raise ValueError("Sham modified the graph")
        z=torch.stack(good,1);offers=torch.gather(z,2,t.candidates[:,None,:].expand(t.lanes,2,2))
        result=dict(all=z.all(2).float(),mean=z.float().mean(2),both=offers.all(2).float(),offered_mean=offers.float().mean(2),
                    paid=z[ix[:,None],torch.arange(2)[None,:],t.final_item[:,None]].float())
        for key,v in result.items():
            for name,ind in policies.items():
                values[f"{mode}/{name}/{key}"]=v[:,0]+ind*(v[:,1]-v[:,0])
            for name in ("random","other_history","swapped"):
                values[f"{mode}/native_minus_{name}/{key}"]=(true_side-policies[name])*(v[:,1]-v[:,0])
    return raw,{k:v.reshape(-1,4).mean(1).numpy() for k,v in values.items()}


def run(cohort,stage,worlds,offset=0,label=None):
    torch.set_num_threads(1);ids=game.families.births(cohort)
    seed=(3719451 if cohort=="original" else 3719467)+offset
    dest=game.world.DATA/"identity-eval"/(label or cohort)/str(stage);dest.mkdir(parents=True,exist_ok=True)
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    code=digest(__file__);hashes={};series={}
    for arm in game.ARMS:
        by_root={}
        for b in ids:
            cp=game.old.checkpoint_path(game.OUT/cohort,"odor",arm,b,stage);identity=digest(cp);file=dest/f"{arm}-{b}.npz"
            if not file.exists():
                g,r=game.old.load_checkpoint(game.OUT/cohort,"odor",arm,b,stage);raw={}
                for count in (2,4,8):
                    for condition,skip in (("one_missing",True),("all_seen",False)):
                        t,missing=paired.prepare(seed,(),0,worlds,count,omissions=skip,heldout=True)
                        tags,other=game.context(seed,worlds,count)
                        parts=[];stats=[]
                        for j in range(0,worlds,32):
                            idx=torch.arange(4*j,min(t.lanes,4*(j+32)))
                            z,v=replay(g,r,paired.expand(t,idx),missing[idx],tags[idx],other[idx],arm)
                            parts.append(z);stats.append(v)
                        raw.update({f"{count}/{condition}/{k}":np.concatenate([d[k] for d in parts]) for k in parts[0]})
                        raw.update({f"metric/{count}/{condition}/{k}":np.concatenate([d[k] for d in stats]) for k in stats[0]})
                    # Ordinary, fully observed untagged memories from a separate new life.
                    t,missing=paired.prepare(seed,(),0,worlds,count,omissions=False,heldout=True)
                    held=[]
                    for j in range(0,worlds,32):
                        idx=torch.arange(4*j,min(t.lanes,4*(j+32)));q=paired.expand(t,idx)
                        ztag=torch.zeros_like(q.odor);v,s,_,_=game.prefix(g,r,q,missing[idx],ztag)
                        trial,_=all_tests(s,q,ztag);held.append(trial.numpy())
                    correct=np.concatenate(held);raw[f"{count}/untagged"]=correct
                    raw[f"metric/{count}/untagged/all"]=correct.all(1).reshape(worlds,4).mean(1)
                    raw[f"metric/{count}/untagged/mean"]=correct.mean(1).reshape(worlds,4).mean(1)
                game.lesson.storage._write_raw(file,**raw,source=code,checkpoint=identity,worlds=worlds,seed=seed)
            with np.load(file) as z:
                if (str(z["source"]),str(z["checkpoint"]),int(z["worlds"]),int(z["seed"]))!=(code,identity,worlds,seed):raise ValueError("Identity archive mismatch")
                for k in z:
                    if k.startswith("metric/"):by_root.setdefault(k[7:],[]).append(z[k])
            hashes[cp.relative_to(game.world.DATA).as_posix()]=identity;hashes[file.relative_to(game.world.DATA).as_posix()]=digest(file)
            print(cohort,stage,arm,b,flush=True)
        series[arm]={k:np.stack(v) for k,v in by_root.items()}
    for k in ("flavor","unbound"):
        series["tag minus "+k]={m:v-series[k][m] for m,v in series["tag"].items()}
    ci=Crossed(len(ids),worlds,seed+7,3000)
    report=dict(births=ids,stage=stage,seed=seed,worlds=worlds,hashes=hashes,
                results={a:{k:ci(v) for k,v in row.items()} for a,row in series.items()},
                note="Flipping received evidence yokes which candidate had a zero taste at study. The physical "
                     "body is held fixed at the executing recipient when a swapped menu or other history donates "
                     "a route action. Test three roles on matched fresh scenes, including an ordinary untagged "
                     "retention life with all true receipts.")
    game.lesson.storage._write_json(game.world.EVIDENCE/f"identity-{label or cohort}-{stage}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=game.world.COHORTS,required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--worlds",type=int,default=192)
    p.add_argument("--offset",type=int,default=0);p.add_argument("--label");run(**vars(p.parse_args()))
