"""Attribute information acquisition to its sensory evidence trace, including longer lifetimes."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_coverage as cov
from .interleave_capacity_report import Crossed


def offer(t,gain,trace,arm,query=None):
    return cov.instrument(t,cov.cue(trace,t["preview"] if query is None else query,arm),gain)


def scene(seed,worlds,price,alternative,extra,where,angle):
    t=cov.ticket.tape(seed,0,0,worlds,domain=436221,price=price,alternative=alternative)
    rng=np.random.default_rng([882123,seed,worlds])
    if angle is not None and alternative=="unseen":
        # Slot identity is reconstructed from which prior public menu differs between twins.
        n=len(t["price"]);twin=torch.arange(n)^4
        diff=(t["study"]-t["study"][twin]).abs().amax(2)>1e-6
        rows=torch.arange(n)[~t["evidenced"]]
        if not torch.all(diff[rows].sum(1)==1):raise ValueError("No unique public distractor")
        at=diff[rows].to(torch.int64).argmax(1)
        x=t["study"][twin[rows],at]/1.35
        tangent=torch.from_numpy(np.repeat(rng.normal(size=(worlds,3)).astype(np.float32),4,axis=0))
        tangent-=x*(tangent*x).sum(1,keepdim=True)
        tangent=torch.nn.functional.normalize(tangent,dim=1)
        direction=np.deg2rad(angle)
        t["study"][rows,at]=1.35*(np.cos(direction)*x+np.sin(direction)*tangent)
    if extra:
        vec=rng.normal(size=(worlds,extra,3))
        vec=(1.35*vec/np.linalg.norm(vec,axis=2,keepdims=True)).astype(np.float32)
        def copy(v):return torch.from_numpy(np.repeat(v,8,axis=0))
        values=rng.choice(np.array([-1.,1.],np.float32),(worlds,extra))
        for key,v,base in (("study",copy(vec),t["study"]),("study_value",copy(values),t["study_value"]),
                           ("study_draws",copy(rng.random((worlds,extra,5),dtype=np.float32)),t["draws"][:,:2]),
                           ("study_gaps",copy(rng.integers(0,7,size=(worlds,extra))),t["gaps"][:,:2])):
            t[key]=torch.cat((v,base) if where=="before" else (base,v),dim=1)
    return t


@torch.no_grad()
def measure(g,r,gain,t,arm,full):
    view,state,_,trace,hist=cov.physical_prefix(g,r,t,arm)
    check=cov.record(hist,t["study"],every_exposure=arm=="exposure")
    torch.testing.assert_close(trace,check,rtol=2e-6,atol=2e-6)
    n=len(trace);twin=torch.arange(n)^4
    base=offer(t,gain,trace,arm)
    v=cov.ticket.finish(view,state,base,"native")
    ones=torch.ones_like(v["buy"])
    bought=cov.ticket.finish(view,state,base,"native",decision=ones)
    skip=cov.ticket.finish(view,state,base,"native",decision=~ones)
    def outcome(side):return torch.where(side,bought["utility"],skip["utility"])
    torch.testing.assert_close(v["utility"],outcome(v["buy"]),rtol=0,atol=0)
    flip=offer(t,gain,trace[twin],arm)
    donation=cov.ticket.finish(view,state,flip,"native")
    measured=cov.cue(trace,t["preview"],arm)
    raw=dict(trace=trace.numpy(),support=hist.numpy().astype(np.int8),action=v["choice"].numpy().astype(np.int8),
             terminal=v["final"].numpy().astype(np.int8),terminal_value=t["final_value"].numpy().astype(np.int8),
             donor_action=donation["choice"].numpy().astype(np.int8),purchased=v["buy"].numpy(),
             review_side=v["review_side"].numpy().astype(np.int8),review_taste=v["taste"].numpy().astype(np.int8))
    values=dict(buy=v["buy"].float(),gross=v["gross"],utility=v["utility"],forced_buy=bought["utility"],forced_skip=skip["utility"],
                selection_gain=.5*(v["utility"]-outcome(v["buy"][twin])),signal=measured,
                own_minus_swapped_signal_action=v["utility"]-outcome(donation["buy"]),
                own_minus_swapped_signal=v["utility"]-donation["utility"],forced_value_gain=bought["gross"]-skip["gross"])
    if full:
        for label,modified in (("ignore",t),("wrong_axis",offer(t,gain,trace,arm,t["preview"].roll(1,1))),
                               ("dose",offer(t,gain,trace,"dose") if arm!="plain" else t)):
            z=cov.ticket.finish(view,state,modified,"native")
            values[label+"_action_loss"]=v["utility"]-outcome(z["buy"])
            values[label+"_utility_loss"]=v["utility"]-z["utility"]
        erased=cov.ticket.finish(view,state,base,"native",deliver="zero")
        if not torch.equal(erased["buy"],v["buy"]):raise ValueError("A future taste changed initial choice")
        values["taste_value"]=v["utility"]-erased["utility"]
    return raw,{k:x.reshape(-1,2,4).mean(2).numpy() for k,x in values.items()}


def evaluate(g,r,gain,seed,worlds,arm,scenario):
    extra,order,angle,prices,full=scenario
    all_data={};measurements={}
    for price in prices:
        alts=[]
        for alt in ("unseen","omitted"):
            t=scene(seed,worlds,price,alt,extra,order,angle);parts=[];readings=[]
            for j in range(0,worlds,32):
                cut={k:v[8*j:8*(j+32)] for k,v in t.items()}
                data,vals=measure(g,r,gain,cut,arm,full)
                parts.append(data);readings.append(vals)
            for k in parts[0]:all_data[f"{price}/{alt}/{k}"]=np.concatenate([v[k] for v in parts])
            alts.append({k:np.concatenate([v[k] for v in readings]) for k in readings[0]})
        for k in alts[0]:
            a,b=alts[0][k],alts[1][k]
            v=np.stack((.5*(a[:,0]+b[:,0]),a[:,1],b[:,1]),1)
            for i,label in enumerate(("known","unseen","unfed")):
                measurements[f"{price}/{label}/{k}"]=v[:,i]
            measurements[f"{price}/overall/{k}"]=v@np.array([.5,.25,.25])
            if k=="buy":
                measurements[f"{price}/unseen_minus_known"]=v[:,1]-v[:,0]
                measurements[f"{price}/unfed_minus_known"]=v[:,2]-v[:,0]
    basis=prices if len(prices)==1 else cov.ticket.PRICES
    for k in list(measurements):
        if k.startswith(str(basis[0])+"/"):
            metric=k.split("/",1)[1]
            measurements["pooled/"+metric]=np.mean([measurements[f"{price}/{metric}"] for price in basis],axis=0)
    return all_data,measurements


def scenarios(stress):
    if not stress:return {"basic":(0,"none",None,(.05,.15,.25,.40,.55),True)}
    out={"control":(0,"none",None,(.25,),False)}
    for k in (2,6,14):
        for pos in ("before","after"):out[f"{pos}{k}"]=(k,pos,None,(.25,),False)
    for a in (5,15,45,90):out[f"angle{a}"]=(0,"none",a,(.25,),False)
    return out


def run(cohort,stage,arms,worlds,*,stress=False,name=None,offset=0):
    torch.set_num_threads(1);ids=cov.ticket.births(cohort)
    seed=(8359031 if cohort=="original" else 8359049)+(333571 if stress else 0)+offset
    label=name or (("stress-" if stress else "")+cohort)
    folder=cov.ticket.DATA/"coverage-eval"/label/str(stage);folder.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    identity=sha(__file__);bank={};hashes={}
    for arm in arms:
        by_root={}
        hp=cov.OUT/cohort/arm/f"gain-{stage}.npz";hh=sha(hp)
        hashes[hp.relative_to(cov.ticket.DATA).as_posix()]=hh
        with np.load(hp) as h:gains=torch.from_numpy(h["gain"]).flatten()
        for ix,b in enumerate(ids):
            cp=cov.ticket.old.checkpoint_path(cov.OUT/cohort,"odor",arm,b,stage);checksum=sha(cp)
            hashes[cp.relative_to(cov.ticket.DATA).as_posix()]=checksum
            g,r=cov.ticket.old.load_checkpoint(cov.OUT/cohort,"odor",arm,b,stage)
            for condition,settings in scenarios(stress).items():
                f=folder/f"{arm}-{b}-{condition}.npz"
                if not f.exists():
                    raw,vals=evaluate(g,r,gains[ix],seed,worlds,arm,settings)
                    cov.ticket.storage._write_raw(f,**raw,**{"metric/"+k:v for k,v in vals.items()},checkpoint=checksum,
                                                  gain=hh,evaluator=identity,seed=seed,worlds=worlds)
                with np.load(f) as q:
                    if (str(q["checkpoint"]),str(q["gain"]),str(q["evaluator"]),int(q["seed"]),int(q["worlds"]))!=(checksum,hh,identity,seed,worlds):
                        raise ValueError("Coverage cache does not match")
                    for k in q:
                        if k.startswith("metric/"):by_root.setdefault(condition+"/"+k[7:],[]).append(q[k])
                hashes[f.relative_to(cov.ticket.DATA).as_posix()]=sha(f)
            print(cohort,stage,arm,b,label,flush=True)
        bank[arm]={k:np.stack(v) for k,v in by_root.items()}
    for a,b in (("evidence","dose"),("evidence","exposure"),("evidence","plain")):
        if a in bank and b in bank:bank[a+" minus "+b]={k:v-bank[b][k] for k,v in bank[a].items()}
    ci=Crossed(len(ids),worlds,seed+7,3000)
    bootstrap=(ci.br[:,:,None]*ci.bw[:,None,:]).reshape(len(ci.br),-1)
    def summarize(item):
        out={};keys=list(item)
        for j in range(0,len(keys),32):
            block=keys[j:j+32]
            raw=np.stack([item[k] for k in block],axis=-1)
            draws=bootstrap@raw.reshape(len(ids)*worlds,-1)
            lower,upper=np.quantile(draws,[.025,.975],axis=0)
            for i,k in enumerate(block):
                out[k]=dict(mean=float(raw[:,:,i].mean()),ci95=[float(lower[i]),float(upper[i])],
                            by_birth=raw[:,:,i].mean(1).tolist())
        return out
    gain_values={}
    for a in arms:
        with np.load(cov.OUT/cohort/a/f"gain-{stage}.npz") as data:gain_values[a]=data["gain"].flatten().tolist()
    report=dict(cohort=cohort,stage=stage,worlds=worlds,seed=seed,births=ids,hashes=hashes,
                gains=gain_values,
                results={a:summarize(v) for a,v in bank.items()},note="All fixed branches preserve actual sensory "
                "contacts and native updates. When only a swapped-trace action is evaluated it is executed in the "
                "original recipient. Missing history and novel incoming value were crossed with the known twin in "
                "each public world, so the matched-random comparator pays the same total price.")
    cov.ticket.storage._write_json(cov.ticket.EVIDENCE/f"coverage-{label}-{stage}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--arms",nargs="+",choices=cov.ARMS,required=True)
    p.add_argument("--worlds",type=int,default=192);p.add_argument("--stress",action="store_true")
    p.add_argument("--name");p.add_argument("--offset",type=int,default=0)
    a=p.parse_args();run(**vars(a))
