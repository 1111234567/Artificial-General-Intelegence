"""Blind task tests and same-prefix causal probes of the evolved recurrent wiring."""

import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import attractor_search as engine,research_audit as audit,ring_world


@lru_cache(maxsize=1)
def intervention_program():
    file=Path(engine.__file__).with_name("attractor_gpu.cu")
    source=file.read_text(encoding="utf-8")
    first="if(!cut){"
    if source.count(first)!=1:raise ValueError("Unexpected encoder revision")
    source=source.replace(first,"""const int after_cue = t - issued[ix];
            const bool disabled = cut == 1 || (cut == 2 && after_cue >= 1 && after_cue <= 4)
                || (cut == 3 && t > query[ix] && t <= query[ix]+4)
                || (cut == 4 && t > issued[ix] && t < query[ix]);
            if(!disabled){""")
    s="feature[cell]=(probe_kind==1?spike:trace)-mean;"
    if source.count(s)!=1:raise ValueError("Unexpected spike readout")
    source=source.replace(s,"""feature[cell]=probe_kind==2 ? spike : probe_kind==3 ? membrane :
                          probe_kind==4 ? syn : (probe_kind==1 ? spike : trace)-mean;""")
    return compile_kernels(f"#define NODES {engine.N}\n#define DEGREE {engine.D}\n#define KEYS {engine.K}\n#define NPARAM 6\n"+source,
                           ("encode_lives","score_lives"),device=torch.device("cuda:0"))


class Decoder:
    def __init__(self,v,birth):
        a=engine.genome(v,birth);s=engine.genome(v,birth,shuffle=True)
        tensor=lambda x:torch.as_tensor(np.ascontiguousarray(x),device="cuda")
        self.original=[tensor(x) for x in a];self.shuffled=[tensor(x) for x in s]
        self.code=intervention_program();self.rate=self.original[-1].reshape(1)

    def features(self,t,cut=0,kind=0,shuffle=False):
        n,trials=t.goal.shape
        args=[torch.as_tensor(np.ascontiguousarray(x),device="cuda") for x in
              (t.cue,t.ctx,t.issue,t.query,t.length,t.raw_at,t.distractor,t.at)]
        z=torch.empty((n,trials,engine.K),device="cuda",dtype=torch.float32)
        count=torch.empty((n,engine.N),device="cuda",dtype=torch.int32)
        a=self.shuffled if shuffle else self.original
        self.code["encode_lives"]((n,1),(engine.N,),*a[:6],*args,z,count,n,trials,cut,kind)
        return z,float(count.sum().item()/(engine.physical_time(t)*engine.N))

    def answers(self,t,z,*,feedback=True,nlms=False):
        n,trials=t.goal.shape
        goal=torch.as_tensor(t.goal,device="cuda")
        tie=torch.as_tensor(t.tie,device="cuda")
        out=torch.empty((n,trials),device="cuda",dtype=torch.uint8)
        self.code["score_lives"]((n,),(128,),z,goal,tie,self.rate,out,n,trials,int(feedback),int(not nlms))
        return out.cpu().numpy()


def stats(t,ans):
    hit=ans==t.goal
    n,d=audit.pair_counts(t.ctx,t.cue,hit)
    return {"early":hit[:,:8].mean(1).tolist(),"late":hit[:,32:].mean(1).tolist(),
            "all":hit.mean(1).tolist(),"both":(n/d).tolist()}


def decode_probe(t,z):
    # Classify the earlier externally presented cue using a world-held-out
    # linear decoder; the classifier is never used to choose an agent action.
    z=np.asarray(z,dtype=np.float64)
    folds=np.arange(len(t.seeds))%4
    prediction={k:np.empty_like(t.goal) for k in ("context","cue","joint")}
    labels={"context":t.ctx,"cue":t.cue,"joint":4*t.ctx+t.cue}
    for f in range(4):
        train=z[folds!=f].reshape(-1,z.shape[-1])
        mu=train.mean(0);scale=np.maximum(train.std(0),1e-5)
        design=(train-mu)/scale
        test=(z[folds==f]-mu)/scale
        gram=design.T@design+.02*len(design)*np.eye(z.shape[-1])
        for key,y in labels.items():
            freq=np.eye(int(y.max())+1)[y[folds!=f].ravel()]
            b=np.linalg.solve(gram,design.T@(freq-freq.mean(0)))
            prediction[key][folds==f]=np.argmax(test@b+freq.mean(0),axis=-1)
    return {k:(prediction[k]==y).mean(1).tolist() for k,y in labels.items()}


def recontext(t,ctx,goal):
    c=t.ctx.copy();c[:,-1]=ctx
    g=t.goal.copy();g[:,-1]=goal
    return replace(t,ctx=c,goal=g)


def fork(dec,t,cut):
    # Only the context at the final request changes. Targets at the two
    # endpoints are private, and the scoring kernel consumes them after action.
    mp=ring_world.make_tape(t.seeds,trials=t.goal.shape[1]).preferences
    i=np.arange(len(t.seeds));cue=t.cue[:,-1]
    qs=[recontext(t,c,mp[i,c,cue]) for c in (0,1)]
    answers=[]
    for q in qs:
        z,_=dec.features(q,cut);answers.append(dec.answers(q,z))
    assert np.array_equal(answers[0][:,:-1],answers[1][:,:-1])
    left,right=(a[:,-1] for a in answers)
    lgoal,rgoal=(q.goal[:,-1] for q in qs)
    aligned=(left==lgoal).astype(np.int8)+(right==rgoal);swapped=(left==rgoal).astype(np.int8)+(right==lgoal)
    return {"switch":(left!=right).astype(float).tolist(),"both":((left==lgoal)&(right==rgoal)).astype(float).tolist(),
            "alignment":((aligned-swapped)/2).tolist()}


def trial(v,graph,design,t,*,rich):
    dec=Decoder(v,graph);traincut=design=="cut"
    z,spikes=dec.features(t,int(traincut))
    ans=dec.answers(t,z)
    rows={"native":stats(t,ans),"no_outcome":stats(t,dec.answers(t,z,feedback=False))}
    if rich:
        for name,cut,kind,shuf in (("cut",1,0,False),("gap_four",2,0,False),("post_query_four",3,0,False),
                                   ("all_gap",4,0,False),("rewired",0,0,True),
                                   ("spike_only",int(traincut),2,False),("membrane",int(traincut),3,False)):
            if traincut and name in ("gap_four","post_query_four","all_gap","rewired"):continue
            alt,_=dec.features(t,cut,kind,shuf)
            rows[name]=stats(t,dec.answers(t,alt))
        for bits in (128,32,8):
            rows[f"step_1_over_{bits}"]=stats(t,dec.answers(t,(z*bits).round()/bits))
        rows["no_covariance"]=stats(t,dec.answers(t,z,nlms=True))
    return {"rows":rows,"activity":spikes,"linear_probe":decode_probe(t,z.cpu().numpy()) if rich else None,
            "last_context_fork":fork(dec,t,int(traincut)) if rich else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--tag",default="overnight")
    ap.add_argument("--worlds",type=int,default=128);ap.add_argument("--births",nargs="+",type=int,default=[11,43,73,97])
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args();torch.set_num_threads(1)
    folder=audit.OUT/f"attractor-{args.tag}-external";folder.mkdir(exist_ok=True)
    # Independent of all validation sets used by evolutionary selection.
    seeds=[int(np.random.SeedSequence([s,738921]).generate_state(1)[0]) for s in audit.split("replication")[:args.worlds]]
    archive=engine.OUT/f"attractor-{args.tag}-status.json"
    sweep=json.loads(archive.read_text());rules=[]
    for item in sweep["finished"]:
        p=engine.OUT/item["name"]/"search.json"
        d=json.loads(p.read_text());rules.append((p.parent.name,d["best"]["vector"],d["protocol"]))
    for name,vec,protocol in rules:
        for condition in ("short","interfere","long","serial","serial_noise","extrapolate"):
            for graph in args.births:
                path=folder/f"{name}-g{graph}-{condition}-{'quick' if args.quick else 'full'}.json"
                if path.is_file():continue
                t=engine.task(seeds,condition)
                o=trial(vec,graph,"cut" if protocol["cut"] else "full",t,rich=not args.quick)
                _write(path,{"training":name,"condition":condition,"birth":graph,"seeds":seeds,"details":o,
                             "settings":engine.decode(vec),"audit_sha256":audit.fingerprint(__file__)})
        print("audited",name,flush=True)


if __name__=="__main__":
    main()
