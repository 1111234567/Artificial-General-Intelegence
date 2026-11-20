"""Measure derivative magnitude and direction on exactly the same unbroken histories."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import interleave_age_train as age, interleave_credit_train as credit, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


def cosine(x,y):
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/denom) if denom else None


def run(src,label,births,seed,samples,blocks):
    torch.set_num_threads(1)
    dest=DATA/"credit-audit"/label;dest.mkdir(parents=True,exist_ok=True)
    cfg=age.Config(pairs=4,seed=seed)
    result=dict(births=births,seed=seed,independent_batches=samples,blocks=blocks,paired_lives_per_batch=4,
                modes=credit.MODES,
                note="Same frozen parameters, physical world and draws for all gradient modes, no optimizer applied. "
                     "All pathwise term cuts leave complete forward values and realized rewards invariant. Norms are "
                     "pre-clipping and coordinates include all birth weights, biases and shared local rule coefficients.",
                roots={},sha256={})
    for birth in births:
        base,rule=src.load(birth)
        g=age.Graph(nn.Parameter(base.weights.clone()),nn.Parameter(base.bias.clone()),base.present)
        rule=deepcopy(rule).requires_grad_(True)
        params=(g.weights,g.bias,*rule.parameters())
        packed={};cells={}
        for length in blocks:
            gradients=[];scores=[]
            for k in range(samples):
                t=age.make_tape(cfg,[birth],k+1,length,domain=384127)
                modes=[];score=[]
                for mode in credit.MODES:
                    loss,val=credit.loss(g,rule,t,mode,cfg)
                    total=torch.autograd.grad(loss,params,allow_unused=True)
                    vec=torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1).double()
                                   for p,x in zip(params,total,strict=True)]).numpy()
                    modes.append(vec);score.append(val.numpy()[0])
                if not all(np.array_equal(score[0],x) for x in score):
                    raise AssertionError("Forward history changed under a derivative intervention")
                gradients.append(np.stack(modes));scores.append(score[0])
            v=np.stack(gradients);norm=np.linalg.norm(v,axis=2)
            packed[f"grads_{length}"]=v;packed[f"scores_{length}"]=np.stack(scores)
            cells[length]={}
            for i,mode in enumerate(credit.MODES):
                fine=[cosine(x[0],x[i]) for x in v]
                batchwise=[cosine(v[j,i],v[j+1,i]) for j in range(0,len(v)-1,2)]
                finite=[x for x in fine if x is not None]
                independent=[x for x in batchwise if x is not None]
                cells[length][mode]=dict(median_norm=float(np.median(norm[:,i])),
                       q90_norm=float(np.quantile(norm[:,i],.9)),max_norm=float(norm[:,i].max()),
                       fraction_clipped=float((norm[:,i]>2).mean()),
                       mean_cosine_to_full=float(np.mean(finite)) if finite else None,
                       mean_cosine_independent_batches=float(np.mean(independent)) if independent else None)
        p=dest/f"{birth}.npz";np.savez_compressed(p,**packed)
        result["sha256"][p.name]=sha256(p.read_bytes()).hexdigest();result["roots"][birth]=cells
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"credit-audit-{label}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--samples",type=int,default=8);p.add_argument("--blocks",nargs="+",type=int,default=(1,2,4,8))
    p.add_argument("--seed",type=int,default=2138833)
    a=p.parse_args();run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.samples,a.blocks)
