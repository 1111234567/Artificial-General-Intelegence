"""Cross inherited graphs and shared writing rules, then let each recipient learn normally."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as capacity, interleave_induction as induction
from . import local_trace_graph as net, local_trace_train as previous
from .interleave_interference import DATA,EVIDENCE,ROOT


SOURCES=("one_item","recent","uniform")
CUTS=(0,1,4,16)


def blend(base,donor,keys,birth):
    rule=net.Rule(birth)
    rule.load_state_dict({k:(donor if k in keys else base).state_dict()[k] for k in base.state_dict()})
    return rule.eval().requires_grad_(False)


def mechanisms(g,r,warm,r_warm,birth):
    vals={"one_item":warm,**g};rules={"one_item":r_warm,**r}
    for graph in vals.values():
        if not torch.equal(graph.present,warm.present):raise AssertionError("Birth masks differ within matched root")
    for name in SOURCES:
        for writer in SOURCES:
            yield name+"+"+writer,vals[name],rules[writer]
    blank_g,blank_r=net.spawn(birth),net.Rule(birth).eval()
    for writer in ("one_item","recent","uniform"):
        yield "untrained+"+writer,blank_g,rules[writer]
    yield "uniform+untrained",vals["uniform"],blank_r
    for part,keys in (("trace",("trace_input","trace_keep")),("numeric",("modulation","scales")),
                      ("structure",("gates","ranking"))):
        yield "recent+uniform_"+part,vals["recent"],blend(r["recent"],r["uniform"],keys,birth)
    yield "uniform_weights+recent",net.Graph(g["uniform"].weights,g["recent"].bias,g["recent"].present),r["recent"]
    yield "uniform_bias+recent",net.Graph(g["recent"].weights,g["uniform"].bias,g["recent"].present),r["recent"]


@torch.no_grad()
def run(root,warm,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"transplant"/label;dest.mkdir(parents=True,exist_ok=True)
    tape=capacity.make_tape("odor",seed,worlds,max(CUTS))
    out=dict(births=births,worlds=worlds,seed=seed,cuts=CUTS,roots={},raw_sha256={},checkpoint_sha256={},
             interpretation="Body means all inherited weights, biases and present bits; present bits are paired and "
                            "identical before autonomous experience. Rule is the full inherited shared update mapping. "
                            "Every graft starts with zero activity/traces and may still change every graph parameter "
                            "throughout the recipient life. Partial rule grafts have the same geometry and scale as "
                            "their donors, which can interact with the recipient body.")
    for b in births:
        base,rb=previous.load_checkpoint(warm,"odor","active",b,8000)
        g={};r={}
        for a in ("recent","uniform"):
            g[a],r[a]=previous.load_checkpoint(root,"odor",a,b,8000)
            path=previous.checkpoint_path(root,"odor",a,b,8000)
            out["checkpoint_sha256"][f"{b}/{a}"]=sha256(path.read_bytes()).hexdigest()
        out["checkpoint_sha256"][f"{b}/one_item"]=sha256(previous.checkpoint_path(warm,"odor","active",b,8000).read_bytes()).hexdigest()
        cells={};raw={}
        for key,body,rule in mechanisms(g,r,base,rb,b):
            initial=net.start(body,tape.lanes)
            coeff=rule.expand(tape.lanes)
            state=initial
            results={}
            for k in range(max(CUTS)+1):
                state,_=capacity._study(coeff,state,tape,k,write=True)
                if k not in CUTS:continue
                snap=net.wait(coeff,state,4)
                side=np.stack([capacity._probe(coeff,snap,tape,i).side.detach().numpy() for i in range(k+1)])
                truth=tape.right_value[:,:k+1,-1].numpy().T
                good=side==truth
                m=dict(A=float(good[0].mean()),all=float(good.all(0).mean()))
                if k:
                    m.update(B_latest=float(good[-1].mean()),both=float((good[0]&good[-1]).mean()),
                             B_mean=float(good[1:].mean()))
                results[k]=m
                raw[f"{key}/{k}"]=side.astype(np.int8)
            cells[key]=results
        p=dest/f"{b}.npz";np.savez_compressed(p,**raw,truth=tape.right_value[:,:,-1].numpy().T)
        out["raw_sha256"][p.name]=sha256(p.read_bytes()).hexdigest();out["roots"][b]=cells
        print(label,b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"transplant-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


@torch.no_grad()
def transfer_rule(warm,root,label,births,worlds,seed):
    """Check rule/body contributions specifically for the first genuinely novel action."""
    torch.set_num_threads(1)
    dest=DATA/"induction-transplants"/label;dest.mkdir(parents=True,exist_ok=True)
    report=dict(seed=seed,births=births,worlds=worlds,roots={},raw_sha256={})
    for birth in births:
        bodies={};rules={}
        bodies["naive"],rules["naive"]=net.spawn(birth),net.Rule(birth).eval().requires_grad_(False)
        bodies["prior"],rules["prior"]=previous.load_checkpoint(warm,"odor","active",birth,8000)
        bodies["uniform"],rules["uniform"]=previous.load_checkpoint(root,"odor","uniform",birth,8000)
        tape=induction.world(seed,birth,worlds,"linear",0.,16)
        mask=tape["x_sign"].numpy()>0
        result={};packed={}
        for body in bodies:
            for rule in rules:
                key=body+"+"+rule
                raw=induction.evaluate(bodies[body],rules[rule],tape,(0,4,8,16))
                cells={}
                for k in (0,4,8,16):
                    own=raw[f"coherent/{k}"]
                    donor=raw[f"unrelated_history/{k}"]
                    cells[k]=dict(novel=float(own[mask,0].mean()),unrelated=float(donor[mask,0].mean()),
                                  exception=float(own[~mask,2].mean()),old_and_new=float(own[~mask,4].mean()))
                    packed[f"{key}/{k}"]=own
                result[key]=cells
        path=dest/f"{birth}.npz";np.savez_compressed(path,**packed)
        report["raw_sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        report["roots"][birth]=result;print(label,"transfer",birth,flush=True)
    (EVIDENCE/f"induction-transplants-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--run",type=Path,default=DATA/"retention-objective")
    p.add_argument("--warm",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=96);p.add_argument("--seed",type=int,default=871831)
    p.add_argument("--novel-only",action="store_true")
    a=p.parse_args()
    if a.novel_only:
        transfer_rule(a.warm,a.run,a.label,a.births,a.worlds,a.seed)
    else:
        run(a.run,a.warm,a.label,a.births,a.worlds,a.seed)
