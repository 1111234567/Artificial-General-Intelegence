"""Decompose the learned observation-to-action rule and include the real cost of the review itself."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_online_readout as train,noon_online_eval as analysis,noon_readback as bank
from .interleave_capacity_report import Crossed


@torch.no_grad()
def material(g,r,t,ordering):
    a,z=train.observe(g,r,t,ordering);n=t.lanes;view=r.expand(n);data={}
    for mode,write,contact in (("active",True,True),("sham",False,True),("quiet",False,False)):
        end=[];earned=[];acts=[]
        for side in (-1.,1.):
            state,c=bank.world.review(view,a.state,t,torch.full_like(a.side,side),write=write,contact=contact)
            rows=bank.world._rows(t);choices=[]
            for i in range(t.count):
                l,rv=bank.world.sensory_for(t,torch.full((n,),i,dtype=torch.long),t.final_orientation[:,i])
                choices.append(train.motor.act(state,l,rv,t.final_draw[:,i]).side)
            ans=torch.stack(choices,1);ok=ans==t.law*t.final_orientation
            end.append(ok);acts.append(ans);earned.append(c.earned if c is not None else torch.zeros_like(a.side))
        data[mode+"/correct"]=torch.stack(end,1).numpy()
        data[mode+"/actions"]=torch.stack(acts,1).numpy().astype(np.int8)
        data[mode+"/taste"]=torch.stack(earned,1).numpy().astype(np.int8)
    np.testing.assert_array_equal(data["active/taste"],data["sham/taste"])
    return {**data,"z":z.numpy(),"menu":a.motor_logit.numpy(),"u":t.decision_draw.numpy(),"query":t.final_item.numpy(),
            "offered":t.candidates.numpy(),"orient":t.menu_orientation.numpy(),"laws":t.law.numpy()}


def readings(d,learner):
    z=torch.tensor(d["z"],dtype=torch.double);menu=torch.tensor(d["menu"],dtype=torch.double)
    with torch.no_grad():
        f=learner.scores(z);op=learner.scores(-z);even=(f+op)*.5;odd=(f-op)*.5
        base=learner.gain*menu
        def diff(v):return v[:,1]-v[:,0]
        all_=base+diff(f)
        torch.testing.assert_close(all_,base+diff(even)+diff(odd),atol=1e-12,rtol=1e-12)
        policies={k:v.sigmoid().numpy() for k,v in dict(full=all_,sign_invariant=base+diff(even),
                 sign_sensitive=base+diff(odd),only_menu=base,only_readings=diff(f),unassisted=menu).items()}
        weak=(z[:,:,0]-z[:,:,1]).abs().numpy()
        policies["weak"]=np.where(weak[:,1]==weak[:,0],.5,weak[:,1]<weak[:,0])
    row=np.arange(len(menu));u=d["u"]
    data={}
    valence=-np.take_along_axis(d["laws"],d["offered"],axis=1)*d["orient"]
    for name,p in policies.items():
        i=(u<p).astype(int)
        data[name+"/right"]=i
        data[name+"/positive_when_opposed"]=((valence[row,i]>0)&(valence[:,0]!=valence[:,1])).astype(float)*2
        data[name+"/select_weaker"]=np.where(i>0,weak[:,1]<=weak[:,0],weak[:,0]<=weak[:,1])
        for mode in ("active","sham","quiet"):
            z=d[mode+"/correct"];q=z[row[:,None],np.arange(2)[None,:],d["query"][:,None]].astype(float)
            both=z.all(2).astype(float);mean=z.mean(2);immediate=(d[mode+"/taste"]>0).astype(float)
            parts=dict(all=both,mean=mean,paid=q)
            if mode!="quiet":parts.update(immediate=immediate,resource=immediate+q)
            for k,v in parts.items():
                choice=v[row,i]
                data[f"{name}/{mode}/{k}"]=choice
                data[f"{name}/{mode}/minus_random/{k}"]=choice-v.mean(1)
                if name=="full" and mode=="active":
                    pbar=np.repeat(i.reshape(-1,8).mean(1),8)
                    data[f"full/active/own_minus_private_yoke/{k}"]=choice-(v[:,0]+pbar*(v[:,1]-v[:,0]))
        for k in ("mean","all","paid"):
            data[f"{name}/true_minus_sham/{k}"]=data[f"{name}/active/{k}"]-data[f"{name}/sham/{k}"]
            data[f"{name}/true_minus_quiet/{k}"]=data[f"{name}/active/{k}"]-data[f"{name}/quiet/{k}"]
    for other in ("sign_invariant","sign_sensitive","only_menu","only_readings","unassisted","weak"):
        for m in ("mean","all","paid","immediate","resource"):
            data[f"full minus {other}/{m}"]=data["full/active/"+m]-data[other+"/active/"+m]
    return {k:np.asarray(v).reshape(-1,8).mean(1) for k,v in data.items()}


def run(cohort,family):
    torch.set_num_threads(1);ss=train.SEEDS;ids=bank.world.COHORTS[cohort]
    seed=9651577 if cohort=="original" else 9651581
    worlds=384;count=4;t,first,_=bank.tape(seed,worlds,count)
    dest=bank.world.DATA/"readout-cause"/cohort/family;dest.mkdir(exist_ok=True,parents=True)
    model_bank,student_hashes=analysis.policies(4000)
    models=model_bank["live"];all_stats=[{} for _ in ss];sha=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    checks={};code=sha(__file__)
    for b in ids:
        src=bank.world.parent_path(cohort) if family=="uniform" else bank.world.PARENT_DATA/"one-life-repair"/cohort
        arm="uniform" if family=="uniform" else "mixed";step=8000 if family=="uniform" else 2000
        g,r=bank.old.load_checkpoint(src,"odor",arm,b,step);cp=bank.old.checkpoint_path(src,"odor",arm,b,step)
        archive=train.OUT/"heldout"/cohort/family/"4"/f"{b}.npz"
        with np.load(archive) as before:
            f=dest/f"{b}.npz"
            if not f.is_file():
                parts=[]
                for j in range(0,worlds,32):
                    ix=torch.arange(8*j,8*(j+32));parts.append(material(g,r,bank.lesson.expand(t,ix),first[ix]))
                values={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
                for k1,k2 in (("active/correct","correct"),("active/taste","review_taste"),("z","z"),("menu","menu")):
                    np.testing.assert_array_equal(values[k1],before[k2])
                bank.lesson.storage._write_raw(f,**values,body=sha(cp),original=sha(archive),source=code)
            with np.load(f) as raw:
                if (str(raw["source"]),str(raw["body"]),str(raw["original"]))!=(code,sha(cp),sha(archive)):raise ValueError("Readout decomposition cache mismatch")
                d={k:raw[k] for k in raw.files if k not in ("body","source","original")}
        for i,m in enumerate(models):
            for k,v in readings(d,m).items():all_stats[i].setdefault(k,[]).append(v)
        checks[f.relative_to(bank.world.DATA).as_posix()]=sha(f)
        print(cohort,family,b,flush=True)
    ci=Crossed(len(ids),worlds,seed+4,3500)
    result=dict(cohort=cohort,family=family,seed=seed,worlds=worlds,births=list(ids),training_seeds=list(ss),hashes=checks,
                readout_hashes=student_hashes,results={k:analysis.interval(np.stack([np.stack(x[k]) for x in all_stats]),ci) for k in all_stats[0]},
                note="The sign-even and sign-odd learned signals decompose the readout's f(z1,z2) under simultaneous "
                     "sign reversal; they are explicit diagnostic interventions on the learned readout, not hidden "
                     "layers discovered in the native body. Immediate resource pays once for the actual review "
                     "choice if its taste is positive; the final paid resource is one actual uniform query.")
    bank.lesson.storage._write_json(bank.world.EVIDENCE/f"readout-cause-{cohort}-{family}.json",result)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=bank.world.COHORTS,required=True)
    a=p.parse_args()
    for family in ("uniform","repair"):run(a.cohort,family)
