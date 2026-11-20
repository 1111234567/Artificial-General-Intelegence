"""Separate forecasting real own action outcomes from using the forecasts to choose a revisit."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_self_assess as model,noon_online_eval as intervals
from .interleave_capacity_report import Crossed


def seed_for(cohort):return 2721577 if cohort=="original" else 2721581


@torch.no_grad()
def collect(cohort,family,count,worlds=384):
    torch.set_num_threads(1);seed=seed_for(cohort);ids=model.w.COHORTS[cohort]
    t,first,_=model.bank.tape(seed,worlds,count)
    draws,_,scored=model.random_inputs(seed,count,worlds)
    draws=draws.repeat_interleave(8,0);scored=scored.repeat_interleave(8)
    folder=model.OUT/"tests"/cohort/family/str(count);folder.mkdir(parents=True,exist_ok=True)
    hashes={};sha=lambda f:sha256(Path(f).read_bytes()).hexdigest();code=sha(__file__)
    for b in ids:
        if family=="uniform":path,arm,stage=model.w.parent_path(cohort),"uniform",8000
        else:path,arm,stage=model.w.PARENT_DATA/"one-life-repair"/cohort,"mixed",2000
        f=model.bank.old.checkpoint_path(path,"odor",arm,b,stage);hashes[str(b)+"/body"]=sha(f)
        g,r=model.bank.old.load_checkpoint(path,"odor",arm,b,stage)
        dest=folder/f"{b}.npz"
        if not dest.exists():
            parts=[]
            for j in range(0,worlds,32):
                ix=torch.arange(8*j,min(t.lanes,8*(j+32)));q=model.bank.lesson.expand(t,ix)
                a,z,side,pre=model.observe(g,r,q,first[ix],draws[ix])
                one=torch.ones_like(a.side)
                res=model.bank.lesson.branches(g,r,q,a,torch.stack((-one,one),1))
                tq=res["tape"];rows=model.w._rows(tq);correct=[]
                for index in range(tq.count):
                    item=torch.full((tq.lanes,),index,dtype=torch.long);orient=tq.final_orientation[:,index]
                    left,right=model.w.sensory_for(tq,item,orient)
                    out=model.motor.act(res["state"],left,right,tq.final_draw[:,index])
                    correct.append(out.side==tq.law[:,index]*orient)
                good=torch.stack(correct,1).reshape(q.lanes,2,tq.count)
                actual=good[torch.arange(q.lanes)[:,None],torch.arange(2)[None,:],q.final_item[:,None]]
                if not torch.equal(actual,res["reward"]>0):raise ValueError("Actual paid action did not match")
                parts.append(dict(z=z.numpy(),side=side.numpy(),early=pre.numpy(),scored=scored[ix].numpy(),u=q.decision_draw.numpy(),
                                  native=a.motor_logit.numpy(),correct=good.numpy(),final=actual.numpy(),
                                  support_side=res["side"].numpy().astype(np.int8),support_taste=res["taste"].numpy().astype(np.int8)))
            model.bank.lesson.storage._write_raw(dest,**{k:np.concatenate([d[k] for d in parts]) for k in parts[0]},
                                                  body=hashes[str(b)+"/body"],seed=seed,worlds=worlds,source=code)
        with np.load(dest) as z:
            if (str(z["body"]),int(z["seed"]),int(z["worlds"]),str(z["source"]))!=(hashes[str(b)+"/body"],seed,worlds,code):
                raise ValueError("Wrong frozen self-assessment archive")
        hashes[str(b)+"/raw"]=sha(dest)
    model.bank.lesson.storage._write_json(folder/"index.json",dict(seed=seed,worlds=worlds,births=list(ids),hashes=hashes))
    print(cohort,family,count,flush=True)


def measure(d,learner):
    z=torch.from_numpy(d["z"]).double();side=torch.from_numpy(d["side"]).double();menu=torch.from_numpy(d["native"]).double()
    with torch.no_grad():
        pred,expected,gate=learner(z,side,menu)
        q=pred.sigmoid().numpy();c=expected.numpy();p=gate.sigmoid().numpy()
        other=[]
        for lag in range(8):
            ix=(np.arange(len(z))//8)*8+(np.arange(len(z))+lag)%8
            moved=learner(z[ix],side[ix],menu)[2].sigmoid().numpy()
            other.append((d["u"]<moved).astype(float))
    idx=np.arange(len(z));k=d["scored"];target=d["early"][idx,k];prediction=q[idx,k]
    native_conf=(side*z[:,:,1]).sigmoid().numpy()[idx,k]
    selected=(d["u"]<p).astype(int)
    future=d["correct"]
    res=dict(brier=(prediction-target)**2,native_confidence_brier=(native_conf-target)**2,
             calibrated_gain=(native_conf-target)**2-(prediction-target)**2,
             early_success=target,forecast=prediction,absolute_forecast_error=np.abs(prediction-target),
             choose_lower=np.where(selected>0,c[:,1]<=c[:,0],c[:,0]<=c[:,1]).astype(float))
    base=target.mean();res["better_than_fixed_brier"]=(base-target)**2-(prediction-target)**2
    donor=np.stack(other).mean(0)
    for lab,truth in (("all",future.all(2)),("mean",future.mean(2)),("paid",d["final"])):
        point=truth[idx,selected].astype(float);dy=truth[:,1].astype(float)-truth[:,0].astype(float)
        res[lab]=point;res[lab+"_minus_random"]=(selected-.5)*dy
        res[lab+"_expected_minus_random"]=(p-.5)*dy
        res[lab+"_own_minus_donor"]=(selected-donor)*dy
    return {k:v.reshape(-1,8).mean(1) for k,v in res.items()}


def summarize(cohort,family,count):
    torch.set_num_threads(1);path=model.OUT/"tests"/cohort/family/str(count)
    meta=json.loads((path/"index.json").read_text())
    data=[]
    for b in meta["births"]:
        f=path/f"{b}.npz"
        if sha256(f.read_bytes()).hexdigest()!=meta["hashes"][str(b)+"/raw"]:raise ValueError("Self-assessment file changed")
        with np.load(f) as z:data.append({k:z[k] for k in z.files if k not in ("source","body","worlds","seed")})
    series={};ids={};coeff={}
    for arm in model.ARMS:
        groups={};coeff[arm]={}
        for seed in model.SEEDS:
            f=model.OUT/str(seed)/arm/"step-4000.pt";obj=torch.load(f,weights_only=True,map_location="cpu")
            if obj["step"]!=4000:raise ValueError("Early forecast stage")
            ids[f.relative_to(model.w.DATA).as_posix()]=sha256(f.read_bytes()).hexdigest()
            m=model.Assessor();m.load_state_dict(obj["model"]);coeff[arm][seed]={k:v.tolist() for k,v in obj["model"].items()}
            evaluated=[measure(d,m) for d in data]
            for k in evaluated[0]:groups.setdefault(k,[]).append(np.stack([v[k] for v in evaluated]))
        series[arm]={k:np.stack(v) for k,v in groups.items()}
    for other in ("uncredited","shuffled"):
        series["forecast minus "+other]={k:v-series[other][k] for k,v in series["forecast"].items()}
    ci=Crossed(len(data),meta["worlds"],meta["seed"]+3*count,3500)
    result=dict(cohort=cohort,family=family,count=count,worlds=meta["worlds"],seed=meta["seed"],births=meta["births"],
                raw_sha256=meta["hashes"],fitted_sha256=ids,coefficients=coeff,
                results={a:{k:intervals.interval(v,ci) for k,v in scores.items()} for a,scores in series.items()},
                note="One randomly designated actual tentative action per life calibrates Brier. The corresponding "
                     "early payoff is withheld from the organism until after the review. Phase-specific route "
                     "output adds five learned parameters and two bound confidence records. All learned policies "
                     "were fitted on original bodies only, then deployed unaltered across cohorts.")
    model.bank.lesson.storage._write_json(model.w.EVIDENCE/f"self-assess-{cohort}-{family}-{count}.json",result)
    print(cohort,family,count,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=model.w.COHORTS,required=True)
    p.add_argument("--mode",choices=("collect","score"),required=True);a=p.parse_args()
    for n in (4,2,8):
        for family in ("uniform","repair"):
            if a.mode=="collect":collect(a.cohort,family,n)
            else:summarize(a.cohort,family,n)
