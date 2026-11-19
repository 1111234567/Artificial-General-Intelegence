"""Study whether time and direction of repeated local innovation matter at the same storage size."""
import json
import time
import numpy as np
import torch

from . import coherent_plastic as update, longrun_study as primary, longrun_world as game, research_audit as audit
from . import symmetry_physics as gpu
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def discovery(parent,folder):
    f,r,m,body,_,fit=primary.origin(parent);groups=update.proposals()
    names=list(groups);opts=np.concatenate(list(groups.values()))
    plans=[(40,.04,"stable"),(40,.04,"reverse_return"),(80,.13,"stable"),(80,.13,"reverse_return"),
           (104,.22,"stable"),(104,.22,"reverse_return")]
    scores=[]
    for j,(blocks,noise,mode) in enumerate(plans):
        seeds=np.random.default_rng(68141597+j//2).integers(7_000_000,475_000_000,48).tolist()
        w=game.make(seeds,blocks=blocks,noise=noise,mode=mode,nonce=953531)
        co=update.Cohort(w,f,r,m)
        scores.append(co.gates(body,opts,batch=64)[:,4])
    values=np.asarray(scores);short={}
    for i,name in enumerate(names):
        width=len(groups[name]);region=values[:,i*width:(i+1)*width]
        short[name]=np.argsort(-region.mean(0),kind="stable")[:6].tolist()
    chosen={}
    ids=np.concatenate([groups[k][short[k]] for k in names])
    rerank=[]
    for j,(noise,mode) in enumerate(((.04,"stable"),(.04,"reverse_return"),(.19,"stable"),(.19,"reverse_return"))):
        seeds=np.random.default_rng(19317139+j//2).integers(7_000_000,475_000_000,72).tolist()
        w=game.make(seeds,blocks=120,noise=noise,mode=mode,nonce=709133)
        rerank.append(update.Cohort(w,f,r,m).gates(body,ids)[:,4])
    rerank=np.asarray(rerank)
    for i,name in enumerate(names):
        block=rerank[:,i*6:(i+1)*6]
        j=int(np.argmax(block.mean(0)))
        chosen[name]=dict(params=groups[name][short[name][j]].tolist(),validation=block[:,j].tolist(),
                          proposal=short[name][j])
    data=dict(parent=parent.name,source=audit.fingerprint(parent/"frozen.json"),choices=chosen,short=short,
              train=values.tolist(),validation=rerank.tolist(),kinds=names,budget_per_method=len(opts)//3)
    _write(folder/(parent.name+".json"),data)
    print("frozen",parent.name,{k:round(np.mean(v["validation"]),3) for k,v in chosen.items()},flush=True)
    return data


def confirm(parent,freeze,dest):
    f,r,m,body,_,fit=primary.origin(parent);pars={k:v["params"] for k,v in freeze["choices"].items()}
    ss=np.random.default_rng(391837511).integers(8_000_000,470_000_000,160).tolist()
    for name,noise,mode,blocks in (("clean_stable",.05,"stable",192),("clean_change",.05,"reverse_return",192),
                 ("noisy_stable",.21,"stable",192),("noisy_change",.21,"reverse_return",192),
                 ("long_change",.09,"reverse_return",768)):
        filename=dest/(parent.name+"-"+name+".json")
        if filename.exists():continue
        seeds=ss[:64] if blocks==768 else ss
        rows={};trace={};prefix_correction=0
        for offset in range(0,len(seeds),16):
            ids=seeds[offset:offset+16]
            w=game.make(ids,blocks=blocks,noise=noise,mode=mode,nonce=2141183)
            co=update.Cohort(w,f,r,m);rigid=np.zeros(10,np.float32)
            base,_=co.gates(body,rigid,detail=True)[0]
            def keep(label,out):
                data=game.count_regions(w,out)
                destrow=rows.setdefault(label,{})
                for reg,fields in data.items():
                    for metric,v in fields.items():destrow.setdefault(reg,{}).setdefault(metric,[]).extend(v)
            keep("rigid",base)
            for method,setting in pars.items():
                out,amount=co.gates(body,setting,detail=True,track=True)[0]
                keep(method,out)
                trace.setdefault(method,[]).extend(amount.mean(1).tolist())
                if method=="coherent" and mode!="stable":
                    # Same starting brain. Activate the candidate rule only once
                    # the environment has changed (an evaluator timing ablation).
                    begin=w.boundary[:,0].astype(np.int32)*(4+2*4)
                    post,_=co.gates(body,setting,detail=True,begin=begin)[0]
                    mask=np.arange(co.tr)[None,:]<begin[:,None]
                    if not np.array_equal(base.contact_tick[mask],post.contact_tick[mask]) or not np.array_equal(base.contact_side[mask],post.contact_side[mask]):
                        raise RuntimeError("The pre-change physical history moved")
                    prefix_correction+=int(mask.sum());keep("enabled_from_identical_prefix",post)
                    # Swap gain amplitudes between neighboring independent lives.
                    # Match by actual informational-event ordinal within each life,
                    # not by tick. Choices and signed outcomes remain the recipient's.
                    visible=(~w.arena.probe)&(out.contact_side>=0)
                    donor=np.arange(len(ids))^1;given=np.zeros_like(amount)
                    for life,d in enumerate(donor):
                        slots=np.flatnonzero(visible[life]);dd=amount[d,visible[d]]
                        if len(dd):given[life,slots]=dd[np.minimum(np.arange(len(slots)),len(dd)-1)]
                    yoked,_=co.gates(body,setting,detail=True,schedule=given)[0]
                    keep("other_life_gate_amplitude",yoked)
                    # Only paired teaching receipts in B are hidden here. Both
                    # independent novel anchor tastes remain available at each visit.
                    block_idx=np.arange(blocks)[None,:]
                    affected=(block_idx>=w.boundary[:,0,None])&(w.block_context==1)
                    pattern=np.zeros(12,bool);pattern[2:10]=True
                    missing=(affected[:,:,None]&pattern[None,None,:]).reshape(co.n,co.tr)
                    old=co.inputs[3]
                    co.inputs[3]=gpu.tensor((w.arena.probe|missing).astype(np.uint8))
                    cut,_=co.gates(body,setting,detail=True)[0]
                    keep("B_anchors_visible_examples_hidden",cut)
                    co.inputs[3]=old
        _write(filename,dict(parent=parent.name,seed=seeds,mode=mode,noise=noise,blocks=blocks,requests=blocks*12,
             choices=freeze["choices"],trace=trace,rows=rows,identical_prefix_opportunities=prefix_correction))
        print("examined",parent.name,name,flush=True)


def main():
    torch.set_num_threads(1);folder=OUT/"signed-innovation-v1";folder.mkdir(exist_ok=True)
    dest=audit.OUT/"signed-innovation-sealed";dest.mkdir(exist_ok=True)
    parents=sorted(p.parent for p in (OUT/"unified-physical-evolution-v1").glob("*--full-*/frozen.json"))
    for parent in parents:
        p=folder/(parent.name+".json");cfg=json.loads(p.read_text()) if p.exists() else discovery(parent,folder)
        confirm(parent,cfg,dest)
    _write(dest/"complete.json",dict(parents=[x.name for x in parents]))


if __name__=="__main__":main()
