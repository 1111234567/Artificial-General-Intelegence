"""Common paired birth/world score summaries for the morning ticket games."""

import numpy as np

from .interleave_capacity_report import Crossed


def summarize(groups,worlds,births,seed,contrasts=()):
    paired=dict(groups)
    for one,two in contrasts:
        if one in groups and two in groups:
            # Optional diagnostic interventions may exist for only one policy
            # (e.g. a partially faded current). Contrast like-for-like readings.
            paired[one+" minus "+two]={k:v-groups[two][k] for k,v in groups[one].items() if k in groups[two]}
    ci=Crossed(len(births),worlds,seed,3500)
    weights=(ci.br[:,:,None]*ci.bw[:,None,:]).reshape(len(ci.br),-1)
    output={}
    for a,data in paired.items():
        keys=list(data);result={}
        for j in range(0,len(keys),32):
            subset=keys[j:j+32];v=np.stack([data[k] for k in subset],-1)
            if v.shape[:2]!=(len(births),worlds):raise ValueError("Unpaired market statistics")
            sampled=weights@v.reshape(len(births)*worlds,-1)
            lo,hi=np.quantile(sampled,[.025,.975],axis=0)
            for i,k in enumerate(subset):
                result[k]=dict(mean=float(v[:,:,i].mean()),ci95=[float(lo[i]),float(hi[i])],by_birth=v[:,:,i].mean(1).tolist())
        output[a]=result
    return output


def fit_seed_intervals(values,ci,seed=182883):
    """Retain fit seeds, independent bodies and held-out worlds as distinct axes."""
    keys=list(values);sample=next(iter(values.values()))
    n,b,w=sample.shape;rng=np.random.default_rng(seed)
    init=rng.multinomial(n,np.full(n,1/n),len(ci.bw))/n
    weights=(init[:,:,None,None]*ci.br[:,None,:,None]*ci.bw[:,None,None,:]).reshape(len(ci.bw),n*b*w)
    out={}
    for j in range(0,len(keys),32):
        subset=keys[j:j+32];v=np.stack([values[k] for k in subset],axis=-1)
        draws=weights@v.reshape(n*b*w,-1);lo,hi=np.quantile(draws,[.025,.975],axis=0)
        for i,k in enumerate(subset):
            x=v[:,:,:,i];out[k]=dict(mean=float(x.mean()),ci95=[float(lo[i]),float(hi[i])],
                                      by_training_seed=x.mean((1,2)).tolist(),by_body=x.mean((0,2)).tolist())
    return out
