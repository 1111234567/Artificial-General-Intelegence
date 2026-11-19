"""Test the minimal change of separating previous-knowledge direction and available step magnitude."""
import json
import time
import numpy as np
import torch
from . import plastic_floor as floor,longrun_study as previous,longrun_plastic as reopen,longrun_world as game
from . import habitat_world as bodyworld,habitat_deploy as native,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


NAMES=["keep_covariance_direction","raw_direction_same_length","raw_only_below_floor","fixed_length_in_covariance_direction"]


def select(parent,folder):
    f,r,m,body,_,fit=previous.origin(parent)
    grid=np.r_[0.,np.geomspace(.0007,1.6,52)]
    trials=np.array([[i,z] for i in range(4) for z in grid])
    tab=[]
    for j,(blocks,mode,noise) in enumerate(((56,"stable",.04),(56,"reverse_return",.04),(88,"stable",.14),
                                      (88,"reverse_return",.14),(112,"stable",.22),(112,"reverse_return",.22))):
        seeds=np.random.default_rng(96583171+j//2).integers(7_000_000,475_000_000,48).tolist()
        w=game.make(seeds,blocks=blocks,noise=noise,mode=mode,nonce=547193)
        co=floor.Cohort(w,f,r,m)
        y=co.rates(body,trials)[:,4];tab.append(y)
        baseline=co.run(fit["best"]["genome"],fit["settings"]["mode"])[0,4]
        if y[0]!=baseline:raise ValueError((y[0],baseline))
    scores=np.asarray(tab);winners={}
    for i,name in enumerate(NAMES):
        js=slice(i*len(grid),(i+1)*len(grid));idx=int(np.argmax(scores[:,js].mean(0)))
        winners[name]=[i,float(grid[idx])]
    out=dict(parent=parent.name,source=audit.fingerprint(parent/"frozen.json"),options=trials.tolist(),
             scores=scores.tolist(),winners=winners,criterion="actual positive free-choice outcomes, six development cohorts")
    _write(folder/(parent.name+".json"),out);return out


def confirm(parent,fit,dest):
    f,r,m,body,material,base=previous.origin(parent)
    ps=json.loads((OUT/"long-life-plasticity-v1"/(parent.name+".json")).read_text())["settings"]
    opts=fit["winners"]
    roots=np.random.default_rng(311792681).integers(7_000_000,475_000_000,160).tolist()
    for name,blocks,noise,mode,appearance in (("stable",192,.05,"stable","smooth"),("flip",192,.05,"reverse_return","smooth"),
              ("noisy_stable",192,.20,"stable","smooth"),("noisy_flip",192,.20,"reverse_return","smooth"),
              ("near_context",192,.08,"reverse_return","nearby"),("long",768,.07,"reverse_return","smooth")):
        path=dest/(parent.name+"-"+name+".json")
        if path.exists():continue
        seeds=roots[:64] if blocks==768 else roots
        rows={};matches=0
        for i in range(0,len(seeds),16):
            ss=seeds[i:i+16];w=game.make(ss,blocks=blocks,noise=noise,mode=mode,appearance=appearance,nonce=18564127)
            co=floor.Cohort(w,f,r,m)
            def save(key,res):
                d=game.count_regions(w,res);target=rows.setdefault(key,{})
                for region,metrics in d.items():
                    for label,v in metrics.items():target.setdefault(region,{}).setdefault(label,[]).extend(v)
            rigid=co.rates(body,[0.,0.],detail=True)[0];save("rigid",rigid)
            out=co.rates(body,np.asarray(list(opts.values())),detail=True)
            for arm,res in zip(opts,out,strict=True):save(arm,res)
            co.long_code=reopen.code(f["matrix"].shape[0],r["first"][1].size,r["slots"],f["capacity"])
            re=reopen.Cohort.adjustments(co,body,ps["uncertainty_direction"]["parameters"],detail=True)[0]
            save("ongoing_covariance_reopening",re)
            if mode!="stable":
                at=w.boundary[:,0]*12
                after=co.rates(body,opts[NAMES[0]],begin=at,detail=True)[0]
                mask=np.arange(co.tr)[None,:]<at[:,None]
                if not (np.array_equal(after.contact_side[mask],rigid.contact_side[mask]) and
                        np.array_equal(after.contact_tick[mask],rigid.contact_tick[mask])):raise ValueError("Unmatched factual prefix")
                matches+=int(mask.sum());save("floor_enabled_only_from_identical_prefix",after)
        _write(path,dict(parent=parent.name,condition=name,seeds=seeds,blocks=blocks,requests=blocks*12,
                         winners=opts,rows=rows,verified_prefix_opportunities=matches))
        print(parent.name,name,flush=True)
    ss=roots[:16];w=game.make(ss,blocks=32);co=floor.Cohort(w,f,r,m)
    diag={}
    for k,pv in opts.items():
        ff,rr,mm=material
        a=native.Forager(ff,{**rr,"magnitude_floor":pv},ss,start=0,motor=mm)
        x=bodyworld.rollout(a,w.arena,silent_probes=True);ref=co.rates(body,pv,detail=True)[0]
        diag[k]=dict(choice=float(np.mean(x.contact_side==ref.contact_side)),tick=float(np.mean(x.contact_tick==ref.contact_tick)),
                     signed=float(np.mean(x.reward==ref.reward)))
    _write(dest/(parent.name+"-parity.json"),diag)


def main():
    torch.set_num_threads(1);folder=OUT/"update-floor-v1";folder.mkdir(exist_ok=True)
    dest=audit.OUT/"update-floor-sealed";dest.mkdir(exist_ok=True)
    parents=sorted(f.parent for f in (OUT/"unified-physical-evolution-v1").glob("*--full-*/frozen.json"))
    for par in parents:
        p=folder/(par.name+".json");f=json.loads(p.read_text()) if p.exists() else select(par,folder)
        confirm(par,f,dest)
    _write(dest/"complete.json",dict(parents=[p.name for p in parents]))


if __name__=="__main__":main()
