"""Counterfactual choice, private evidence and earned action in the omission curriculum."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import noon_paired_lesson as study, noon_review_world as scene
from . import local_trace_graph as net,local_trace_train as old,revisit_choice_graph as motor
from .interleave_capacity_report import Crossed


@torch.no_grad()
def measure(g,r,t,missing):
    choice,_,physical=study.prefix(g,r,t,missing)
    u=t.decision_draw
    sides=torch.stack((-torch.ones_like(u),torch.ones_like(u)),1)
    z=study.branches(g,r,t,choice,sides)
    no_write=study.branches(g,r,t,choice,sides,write=False)
    row=torch.arange(t.lanes);target=(missing==t.candidates[:,1]).long()
    index=(choice.side>0).long()
    twin=torch.arange(t.lanes)^1
    # The same observed candidate menu and future draws, but opposite missing value.
    if not torch.equal(t.odor,t.odor[twin]) or not torch.equal(t.law,t.law[twin]):
        raise ValueError("Yoked histories have different cues or future private laws")
    scores=[]
    for ref in (z,no_write):
        after=ref["state"];tw=ref["tape"];cnt=tw.lanes
        actions=[]
        for i in range(t.count):
            orient=tw.final_orientation[:,i];left=tw.odor[:,i]*orient[:,None]
            q=motor.act(after,left,-left,tw.final_draw[:,i])
            actions.append(q.side==tw.law[:,i]*orient)
        scores.append(torch.stack(actions,1).reshape(t.lanes,2,t.count))
    results=dict(selected=index.numpy().astype(np.int8),candidates=t.candidates.numpy(),
                 missing=missing.numpy(),study=physical.numpy().astype(np.int8),route_logit=choice.motor_logit.numpy(),
                 outcome=z["reward"].numpy().astype(np.int8),test= scores[0].numpy(),no_write=scores[1].numpy(),
                 value=t.law.numpy().astype(np.int8),final_query=t.final_item.numpy())
    metrics={}
    def offered(ok):
        return torch.gather(ok,2,t.candidates[:,None,:].expand(t.lanes,2,2))
    for mode,ok in (("active",scores[0]),("sham",scores[1])):
        pair=offered(ok)
        rates=dict(mean=ok.float().mean(2),all=ok.all(2).float(),pair_both=pair.all(2).float(),
                   pair_mean=pair.float().mean(2),paid=ok[row[:,None],torch.arange(2)[None],t.final_item[:,None]].float())
        if mode=="active" and not torch.equal(rates["paid"]>0,z["reward"]>0):
            raise ValueError("Final single earned action differed from sibling probe")
        if (missing>=0).all():
            other=t.candidates[row,1-target]
            rates.update(missing_correct=ok[row[:,None],torch.arange(2)[None],missing[:,None]].float(),
                         known_correct=ok[row[:,None],torch.arange(2)[None],other[:,None]].float())
        for k,v in rates.items():
            for branch,reading in (("native",v[row,index]),("uniform",v.mean(1)),("opposite",v[row,1-index]),
                                   ("other_history_action",v[row,index[twin]])):
                metrics[f"{mode}/{branch}/{k}"]=reading
            if (missing>=0).all():metrics[f"{mode}/public_omission/{k}"]=v[row,target]
            metrics[f"{mode}/native_minus_uniform/{k}"]=v[row,index]-v.mean(1)
            metrics[f"{mode}/own_minus_other_history/{k}"]=v[row,index]-v[row,index[twin]]
    if (missing>=0).all():
        metrics["select_missing"]=(index==target).float()
        metrics["select_missing_expected"]=torch.where(target>0,choice.motor_logit.sigmoid(),1-choice.motor_logit.sigmoid())
        metrics["changed_choice_across_omissions"]=(index!=index[twin]).float()
    metrics["choose_right"]=index.float()
    return results,{k:v.reshape(-1,4).mean(1).numpy() for k,v in metrics.items()}


def run(cohort,step,worlds=192,*,label=None,seed_offset=0):
    torch.set_num_threads(1)
    ids=study.families.births(cohort);seed=(8447617 if cohort=="original" else 8447629)+seed_offset
    directory=study.families.DATA/"lesson-eval"/(label or cohort)/str(step);directory.mkdir(parents=True,exist_ok=True)
    models=("paired",) if step==0 else ("sampled","paired")
    scores={};checks={};source_hash=sha256(Path(__file__).read_bytes()).hexdigest()
    for arm in models:
        cells={}
        for b in ids:
            path=old.checkpoint_path(study.ROOT/cohort,"odor",arm,b,step);identity=sha256(path.read_bytes()).hexdigest()
            g,r=old.load_checkpoint(study.ROOT/cohort,"odor",arm,b,step)
            f=directory/f"{arm}-{b}.npz"
            if not f.exists():
                raw={}
                for count in (2,4,8):
                    for mode,omitted in (("one_missing",True),("all_observed",False)):
                        t,missing=study.prepare(seed,(),0,worlds,count,omissions=omitted,heldout=True)
                        strata=[];readings=[]
                        for j in range(0,worlds,32):
                            sub=torch.arange(4*j,min(4*(j+32),t.lanes))
                            a,m=measure(g,r,study.expand(t,sub),missing[sub])
                            strata.append(a);readings.append(m)
                        raw.update({f"{count}/{mode}/"+k:np.concatenate([c[k] for c in strata]) for k in strata[0]})
                        raw.update({f"metric/{count}/{mode}/"+k:np.concatenate([c[k] for c in readings]) for k in readings[0]})
                study.storage._write_raw(f,**raw,checkpoint=identity,evaluator=source_hash,seed=seed,worlds=worlds)
            with np.load(f) as raw:
                if str(raw["checkpoint"])!=identity or str(raw["evaluator"])!=source_hash or int(raw["worlds"])!=worlds:raise ValueError("Mismatched lesson cache")
                for k in raw.files:
                    if k.startswith("metric/"):cells.setdefault(k[7:],[]).append(raw[k])
            checks[path.relative_to(study.families.DATA).as_posix()]=identity
            checks[f.relative_to(study.families.DATA).as_posix()]=sha256(f.read_bytes()).hexdigest()
            print(cohort,step,arm,b,flush=True)
        scores[arm]={k:np.stack(v) for k,v in cells.items()}
    if step:scores["paired minus sampled"]={k:v-scores["sampled"][k] for k,v in scores["paired"].items()}
    ci=Crossed(len(ids),worlds,seed+97,2500)
    result=dict(births=ids,seed=seed,step=step,worlds=worlds,hashes=checks,
                outcomes={a:{k:ci(v) for k,v in rec.items()} for a,rec in scores.items()},
                definition="Each world crosses two complementary private laws and which offered object returned no "
                           "value evidence. Each terminal action is tested before feedback. Random review is the "
                           "mean of the two actually executed continuations on the same graph. The yoked policy "
                           "takes the native route chosen in the opposite omission history while executing the "
                           "recipient's true review/taste. Blank study is a public event; no missing-ID input to "
                           "the policy. Worlds are shared across births; pair before crossed resampling.")
    study.storage._write_json(study.families.EVIDENCE/f"lesson-{label or cohort}-{step}.json",result)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True);p.add_argument("--step",type=int,required=True)
    p.add_argument("--worlds",type=int,default=192);p.add_argument("--seed-offset",type=int,default=0);p.add_argument("--label")
    a=p.parse_args();run(**vars(a))
