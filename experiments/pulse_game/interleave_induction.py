"""Transfer earned experience to never-seen odors, then learn one true exception."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


CUTS=(0,1,2,4,8,16,32,64)
MODES=("coherent","false_history","unrelated_history","sham_history")
FIELDS=("novel_X","novel_Y","X_after_feedback","Y_after_X","both_after_X","old_after_X",
        "public_sum_X","nearest_X","public_sum_Y","sampled_taste_X")


def signs_of(x):
    return np.where(x>=0,1.,-1.).astype(np.float32)


def world(seed,birth,count,geom,corrupt,max_k):
    rng=np.random.default_rng([619187,seed,birth])
    xyz=age.directions(rng,count,max_k+2)
    rotation,_=np.linalg.qr(rng.normal(size=(count,3,3)))
    projection=xyz@rotation
    if geom=="linear":base=signs_of(projection[:,:,0])
    elif geom=="cubic":base=signs_of(projection.prod(2))
    else:raise ValueError(geom)
    other=age.signs(rng,(count,max_k))
    exceptions=np.where(rng.random((count,max_k))<corrupt,-1.,1.).astype(np.float32)
    def shared(v):return torch.from_numpy(np.repeat(v,4,axis=0))
    orientation=age.signs(rng,(count,max_k+5))
    rule_sign=np.tile([-1.,-1.,1.,1.],count).astype(np.float32)
    x_exception=np.tile([-1.,1.,-1.,1.],count).astype(np.float32)
    law=np.repeat(base,4,axis=0)*rule_sign[:,None]
    law[:,:max_k]*=np.repeat(exceptions,4,axis=0)
    law[:,max_k]*=x_exception
    return dict(x=shared(xyz),law=torch.from_numpy(law),donor=shared(other),orient=shared(orientation),
                u=shared(rng.random((count,max_k+5,6),dtype=np.float32)),x_sign=torch.from_numpy(x_exception),
                prefix_exception=shared(exceptions),margin=shared(np.abs(projection[:,max_k,0])),
                rule_sign=torch.from_numpy(rule_sign),max_k=max_k)


def read(rule,state,t,obj,event):
    v=t["x"][:,obj]*t["orient"][:,event,None]
    return stream.action(rule,state,-v,v,t["u"][:,event,0])


@torch.no_grad()
def evaluate(g,r,t,cuts=CUTS):
    n=len(t["law"])
    coeff=None if r is None else r.expand(n)
    end=t["max_k"];x=end;y=end+1
    outputs={}
    for mode in MODES:
        state=stream.start(g,r,n)
        empirical=state.hidden.new_zeros((n,3))
        keys=[];values=[]
        for k in range(end+1):
            if k in cuts:
                ux=t["orient"][:,end];uy=t["orient"][:,end+1]
                ax=read(coeff,state,t,x,end)
                ay=read(coeff,state,t,y,end+1)
                true_x=t["law"][:,x]*ux;true_y=t["law"][:,y]*uy
                rv=t["x"][:,x]*ux[:,None]
                x_state,actual_side,earned=stream.contact(coeff,state,-rv,rv,true_x,
                        torch.zeros_like(ux),t["u"][:,end],"active")
                if not torch.equal(actual_side,ax.side):raise AssertionError("First X altered by its future receipt")
                x_state=stream.wait(coeff,x_state,4)
                after=read(coeff,x_state,t,x,end+2)
                yafter=read(coeff,x_state,t,y,end+1)
                old=read(coeff,x_state,t,0,end+3)
                good_x=after.side==t["law"][:,x]*t["orient"][:,end+2]
                good_y=yafter.side==true_y
                def prototype(obj,event):
                    margin=(empirical*t["x"][:,obj]).sum(1)*t["orient"][:,event]
                    return torch.where(margin.abs()>1e-7,margin.sign(),
                                       torch.where(t["u"][:,event,0]<.5,1.,-1.))
                if keys:
                    similarity=(torch.stack(keys,1)*rv[:,None]).sum(2)
                    near=similarity.abs().argmax(1);rows=torch.arange(n)
                    near_side=similarity[rows,near].sign()*torch.stack(values,1)[rows,near]
                else:
                    near_side=prototype(x,end)
                data=(ax.side==true_x,ay.side==true_y,good_x,good_y,good_x&good_y,
                      old.side==t["law"][:,0]*t["orient"][:,end+3],
                      prototype(x,end)==true_x,near_side==true_x,prototype(y,end+1)==true_y,earned>0)
                outputs[f"{mode}/{k}"]=torch.stack(data,1).numpy()
                outputs[f"{mode}/{k}/X_side"]=ax.side.numpy().astype(np.int8)
            if k==end:break
            o=t["orient"][:,k]
            inp=t["x"][:,k]*o[:,None]
            value=t["donor"][:,k] if mode=="unrelated_history" else t["law"][:,k]
            if mode=="false_history":value=-value
            state,side,received=stream.contact(coeff,state,-inp,inp,value*o,torch.zeros_like(o),t["u"][:,k],
                                               "sham" if mode=="sham_history" else "active")
            selected=inp*side[:,None]
            normalized=torch.nn.functional.normalize(selected,dim=1)
            empirical+=received[:,None]*normalized
            keys.append(normalized);values.append(received)
            state=stream.wait(coeff,state,4)
    # An independent past and common random tapes make first-X exactly 50% when both rule signs are counted.
    normal=t["x_sign"].numpy()>0
    for k in cuts:
        first=outputs[f"unrelated_history/{k}"][normal,0]
        if first.mean()!=.5:raise AssertionError("Independent history leaked the target rule")
        s=outputs[f"coherent/{k}/X_side"].reshape(-1,4)
        if not (np.array_equal(s[:,0],s[:,1]) and np.array_equal(s[:,2],s[:,3])):
            raise AssertionError("Exception status visible before the first X action")
    return outputs


def run(source,label,births,worlds,seed,conditions=(("linear",0.),("linear",.2),("cubic",0.)),cuts=CUTS):
    torch.set_num_threads(1)
    dest=DATA/"induction"/label;dest.mkdir(parents=True,exist_ok=True)
    out=dict(births=births,seed=seed,worlds_per_birth=worlds,cuts=cuts,fields=FIELDS,
             design="Each life has a new randomly rotated reward relation. A first-ever X and Y are sampled without "
                    "showing X/Y feedback first. The exception on X and sign of the world rule cross independently. "
                    "Learning X and then probing both X and Y forks the same state. Past-object tastes and actions are "
                    "physical; unrelated-history uses an independent set of prior values, not an extra reference label.",
             roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        rows={};packed={}
        for geom,noise in conditions:
            tape=world(seed,birth,worlds,geom,noise,max(cuts))
            raw=evaluate(g,r,tape,cuts)
            usual=tape["x_sign"].numpy()>0
            for key,v in raw.items():
                tag=f"{geom}/{noise}/{key}"
                packed[tag]=v
                if key.endswith("side"):continue
                rows[tag]={name:dict(zip(FIELDS,v[mask].mean(0).tolist(),strict=True))
                           for name,mask in (("typical",usual),("exception",~usual))}
            packed[f"{geom}/{noise}/first_law"]=tape["law"][:,0].numpy()
            packed[f"{geom}/{noise}/new_law"]=tape["law"][:,-2].numpy()
        out["roots"][birth]=rows
        path=dest/f"{birth}.npz";np.savez_compressed(path,**packed)
        out["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"induction-{label}.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=96);p.add_argument("--seed",type=int,default=311285)
    args=p.parse_args()
    run(stream.Source(args.family,args.run,args.arm,args.step),args.label,args.births,args.worlds,args.seed)
