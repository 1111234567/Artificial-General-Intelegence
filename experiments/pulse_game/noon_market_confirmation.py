"""Reproduce ticket/coverage effects with new worlds, verified shared-prefix physical execution."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import noon_ticket_speed as speed,noon_ticket as ticket,noon_coverage as cov,noon_option_binding as attach
from . import noon_market_stats as intervals


def resolve(cohort,arm,b):
    if arm in ("native","coin"):path=ticket.DATA/"tickets"/cohort;mode=None;role="bilateral"
    elif arm in cov.ARMS:path=cov.OUT/cohort;mode=arm;role="bilateral"
    else:path=attach.OUT/cohort;mode="evidence";role=arm
    g,r=ticket.old.load_checkpoint(path,"odor",arm,b,8000)
    cp=ticket.old.checkpoint_path(path,"odor",arm,b,8000)
    hp=path/arm/"gain-8000.npz"
    if mode is not None:
        with np.load(hp) as z:k=torch.from_numpy(z["gain"][z["births"].tolist().index(b)]).reshape(-1)[0]
        def sensor(t,hist):
            tr=cov.record(hist,t["study"],every_exposure=mode=="exposure")
            value=cov.cue(tr,t["preview"],mode)
            return attach.instrument(t,value,k,role)
    else:sensor=None
    return g,r,sensor,cp,hp if mode is not None else None


@torch.no_grad()
def play(g,r,sensor,seed,n):
    records=[]
    for alt in ("unseen","omitted"):
        bank=[ticket.tape(seed,0,0,n,domain=814139,price=p,alternative=alt) for p in ticket.PRICES]
        out=[]
        for j in range(0,n,32):
            tapes=[{k:v[8*j:8*(j+32)] for k,v in t.items()} for t in bank]
            fast=speed.cached(g,r,tapes,sensor)
            if j==0:
                direct=[]
                for t in tapes:
                    view,s,_,hist=ticket.prefix(g,r,t)
                    aug=t if sensor is None else sensor(t,hist)
                    yes=torch.ones_like(t["coin"])
                    paths=[ticket.finish(view,s,aug,"native",decision=d) for d in (None,yes,~yes)]
                    paths.append(ticket.finish(view,s,aug,"native",deliver="zero"))
                    direct.append(torch.stack([speed.recorded(q) for q in paths],1))
                torch.testing.assert_close(fast,torch.stack(direct,1),rtol=0,atol=0)
            out.append(fast.numpy())
        records.append(np.concatenate(out).reshape(n,2,4,len(ticket.PRICES),4,8))
    return np.stack(records,1) # world, alternative, known/not, law lane, tariff, replay, record


def metrics(cal,raw):
    # The record fields are (choice, buy, logit, review side, taste, final, gross, net).
    n=raw.shape[0];values={}
    learn={}
    for key,data in (("cal",cal),("test",raw)):
        buy=data[...,0,1];nat=data[...,0,-1];bought=data[...,1,-1];skip=data[...,2,-1]
        if not np.array_equal(nat,np.where(buy.astype(bool),bought,skip)):raise ValueError("Incorrect route reconstruction")
        shuffled=np.flip(buy,axis=2)
        yoke=.5*(nat+np.where(shuffled.astype(bool),bought,skip))
        z=dict(net=nat,gross=data[...,0,-2],buy=buy,always_buy=bought,skip=skip,matched=nat-yoke,
               true_info=nat-data[...,3,-1])
        curves={}
        for m,v in z.items():
            means=v.mean(axis=3)
            curves[m]=np.stack((means[:,:,0].mean(1),means[:,0,1],means[:,1,1]),axis=2) # n,tariff,status
        learn[key]=curves
    calibration=learn["cal"];test=learn["test"];w=np.array([.5,.25,.25])
    buy_price=((calibration["always_buy"]-calibration["skip"])@w).mean(0)>0
    buy_public=(calibration["always_buy"]-calibration["skip"]).mean(0)>0
    test["price_only"]=np.where(buy_price[None,:,None],test["always_buy"],test["skip"])
    test["public_evidence"]=np.where(buy_public[None],test["always_buy"],test["skip"])
    for j,p in enumerate(ticket.PRICES):
        for k,v in test.items():
            values[f"{p}/overall/{k}"]=v[:,j]@w
            for i,group in enumerate(("known","unseen","unfed")):values[f"{p}/{group}/{k}"]=v[:,j,i]
        values[f"{p}/unseen_minus_known"]=test["buy"][:,j,1]-test["buy"][:,j,0]
        values[f"{p}/unfed_minus_known"]=test["buy"][:,j,2]-test["buy"][:,j,0]
        values[f"{p}/overall/over_price_only"]=(test["net"][:,j]-test["price_only"][:,j])@w
    for k in list(values):
        if k.startswith(".05/") or k.startswith("0.05/"):
            suffix=k.split("/",1)[1];values["pooled/"+suffix]=np.mean([values[f"{p}/{suffix}"] for p in ticket.PRICES],0)
    return values


def run(cohort):
    torch.set_num_threads(1);ids=ticket.births(cohort)
    seed=4761951 if cohort=="original" else 4761953
    out=ticket.DATA/"market-confirmation"/cohort;out.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest();code=sha(__file__);groups={};hashes={}
    for arm in ("native","coin","evidence","dose","plain","exposure","local","opposite","detached"):
        collected={}
        for b in ids:
            g,r,sensor,cp,hp=resolve(cohort,arm,b);identity=sha(cp)
            f=out/f"{arm}-{b}.npz"
            binding="none" if hp is None else sha(hp)
            if not f.exists():
                cal=play(g,r,sensor,seed+73711,96);raw=play(g,r,sensor,seed,320)
                values=metrics(cal,raw)
                ticket.storage._write_raw(f,cal=cal,test=raw,**{"metric/"+k:v for k,v in values.items()},
                                          checkpoint=identity,additional=binding,source=code)
            with np.load(f) as z:
                if (str(z["checkpoint"]),str(z["additional"]),str(z["source"]))!=(identity,binding,code):raise ValueError("Market confirmation cache different")
                for k in z:
                    if k.startswith("metric/"):collected.setdefault(k[7:],[]).append(z[k])
            hashes[cp.relative_to(ticket.DATA).as_posix()]=identity;hashes[f.relative_to(ticket.DATA).as_posix()]=sha(f)
            if hp is not None:hashes[hp.relative_to(ticket.DATA).as_posix()]=binding
        groups[arm]={k:np.stack(v) for k,v in collected.items()}
        print(cohort,arm,flush=True)
    pairs=[("native","coin"),("evidence","plain"),("evidence","dose"),("evidence","exposure"),("local","evidence"),
           ("local","opposite"),("local","detached")]
    ticket.storage._write_json(ticket.EVIDENCE/f"market-confirmation-{cohort}.json",
        dict(seed=seed,worlds=320,calibration_worlds=96,cohort=cohort,births=ids,hashes=hashes,
             scores=intervals.summarize(groups,320,ids,seed+1,pairs),note="Same public prices, paired histories and "
                    "physical acquisition; known half, new and unrewarded each one quarter. The fast current replay "
                    "was compared against all four ordinary physical branches on the first 32 world roots for "
                    "each model/alternative. All model families and both-world comparisons fixed before evaluation."))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    run(p.parse_args().cohort)
