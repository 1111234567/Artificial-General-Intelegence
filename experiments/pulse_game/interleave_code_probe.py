"""Distinguish linear superposition of signed odors from binding separate item values."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_binding as binding, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


SCENES=("orthogonal","random","triangle","shared_context","non_antipodal")


def make_scene(seed,birth,worlds,name):
    tape=binding.geometry_tape(seed,birth,worlds,"random" if name=="triangle" else name)
    if name=="triangle":
        rng=np.random.default_rng([seed,birth,687359])
        basis,_=np.linalg.qr(rng.normal(size=(worlds,3,3)))
        ref=np.array(((1.,0,0),(-.5,np.sqrt(3)/2,0),(-.5,-np.sqrt(3)/2,0)))
        tri=torch.from_numpy((1.35*np.einsum("ic,wcd->wid",ref,basis)).astype(np.float32)).repeat_interleave(8,0)
        if tri.sum(1).abs().max()>2e-6:raise AssertionError("Triangle is not zero sum")
        tape={**tape,"right":tri,"left":-tri}
    return tape


@torch.no_grad()
def trials(g,r,tape,mode):
    n=len(tape["law"]);idx=torch.arange(n)
    coeff=None if r is None else r.expand(n)
    state=stream.start(g,r,n)
    observed=torch.empty_like(tape["law"])
    for t in range(3):
        item=tape["order"][:,t];flip=tape["study_flip"][:,t]
        l,rv=binding.present(tape,item,flip)
        state,side,reward=stream.contact(coeff,state,l,rv,tape["law"][idx,item]*flip,torch.zeros_like(flip),
                                         tape["study_u"][:,t],mode)
        # Express the signed empirical evidence in the canonical coordinates
        # of the publicly presented pair. Correct labels are used only to score below.
        observed[idx,item]=side*reward*flip
        state=stream.wait(coeff,state,4)
    if not torch.equal(observed,tape["law"]):
        raise AssertionError("Public reconstruction of the sampled taste failed")
    logit,choice=[],[]
    for i in range(3):
        signs=[];sides=[]
        for orient in (1,-1):
            goal=torch.full((n,),float(orient))
            l,rv=binding.present(tape,torch.full((n,),i),goal)
            # Mirrored motor uniforms preserve exact left/right symmetry if the policy has it.
            u=tape["probe_u"][:,i]
            a=stream.action(coeff,state,l,rv,u if orient==1 else 1-u)
            signs.append(a.motor_logit)
            sides.append(a.side*orient)
        logit.append(torch.stack(signs,1))
        choice.append(torch.stack(sides,1).to(torch.int8))
    return dict(logit=torch.stack(logit,1).numpy(),selected=torch.stack(choice,1).numpy(),
                observed=observed.numpy())


def fit_folds(x,target):
    """Predict label-induced logit changes. Whole geometries stay in one fold."""
    residuals=[]
    for which in (0,1):
        test=np.arange(len(x))%2==which
        design=x[~test].reshape(-1,x.shape[-1])
        y=target[~test].reshape(-1)
        coef=np.linalg.lstsq(design,y,rcond=1e-9)[0]
        predicted=np.einsum("wmif,f->wmi",x[test],coef)
        residuals.append(np.square(predicted-target[test]).sum())
    var=float(np.square(target).sum())
    return float(1-sum(residuals)/var) if var>1e-12 else None


def analyze(tape,raw,name):
    raw_logit=raw["logit"].reshape(-1,8,3,2).astype(float)
    values=(raw_logit[...,0]-raw_logit[...,1])/2
    observed=raw["observed"].reshape(-1,8,3)
    laws=observed[0]
    if not np.array_equal(observed,np.broadcast_to(laws,observed.shape)):
        raise AssertionError("Each geometry must cross all eight law histories")
    centered=values-values.mean(1,keepdims=True)
    single=np.einsum("wmi,mj->wij",values,laws)/8
    diagonal=single[:,np.arange(3),np.arange(3)]
    total_energy=float(np.square(centered).sum()/8)
    own=float(np.square(diagonal).sum())
    first=float(np.square(single).sum())
    key=(tape["right"]-tape["left"]).numpy().reshape(-1,8,3,3)[:,0].astype(float)
    key/=np.linalg.norm(key,axis=-1,keepdims=True)
    cos=key@key.transpose(0,2,1)
    order=tape["order"].numpy().reshape(-1,8,3)[:,0]
    features=cos[:,None,:,:]*observed[:,:,None,:]
    indices=np.broadcast_to(order[:,None,None,:],features.shape)
    rank=np.take_along_axis(features,indices,axis=-1)
    kernel=np.concatenate((rank,rank**3),axis=-1)
    signed=values*observed
    actual=raw["selected"].reshape(-1,8,3,2)==observed[...,None]
    # The equal-sign maps +++ and --- on zero-sum triangle directions are both
    # impossible for any strictly correct odd linear readout of a three-vector.
    uniform_maps=[0,7]
    others=[1,2,3,4,5,6]
    margins_good=signed>.05
    pair_pass=margins_good[:,uniform_maps].all((1,2))
    return dict(correct_sampled=float(actual.mean()),all_sampled=float(actual.all(2).mean()),
                correct_greedy_symmetrized=float((signed>0).mean()),
                all_greedy_symmetrized=float((signed>0).all(2).mean()),
                uniform_all_greedy=float(margins_good[:,uniform_maps].all(2).mean()),
                other_all_greedy=float(margins_good[:,others].all(2).mean()),
                both_uniform_histories_pass=int(pair_pass.sum()),geometries=len(pair_pass),
                own_first_order_energy=(own/total_energy if total_energy>1e-12 else None),
                other_item_first_order_energy=((first-own)/total_energy if total_energy>1e-12 else None),
                higher_order_energy=((total_energy-first)/total_energy if total_energy>1e-12 else None),
                logit_rms=float(np.sqrt(np.square(values).mean())),
                law_dependent_rms=float(np.sqrt(np.square(centered).mean())),
                r2_sum=fit_folds(features.sum(-1,keepdims=True),centered),
                r2_lagged_linear=fit_folds(rank,centered),
                r2_lagged_cubic=fit_folds(kernel,centered),
                sensitivity=single.mean(0).tolist(),
                public_dictionary_optimal=float((observed==tape["law"].numpy().reshape(-1,8,3)).all(2).mean()))


def run(source,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"code-probe"/label
    dest.mkdir(parents=True,exist_ok=True)
    result=dict(seed=seed,births=births,independent_geometries=worlds,labels_per_geometry=8,
                margin_threshold=.05,roots={},sha256={},
                estimand="Same final snapshot; evaluate each odor in both motor orientations and take half the logit "
                         "difference. Fully crossing eight histories gives exact per-geometry expansion in the three "
                         "received item signs. R2 is for within-geometry law-dependent logit changes, tested on disjoint "
                         "geometries, and does not identify the internal algorithm.")
    for birth in births:
        g,r=source.load(birth)
        packed={}
        result["roots"][birth]={}
        for geom in SCENES:
            tape=make_scene(seed,birth,worlds,geom)
            result["roots"][birth][geom]={}
            for mode in ("active","sham"):
                raw=trials(g,r,tape,mode)
                result["roots"][birth][geom][mode]=analyze(tape,raw,geom)
                packed.update({f"{geom}/{mode}/{k}":v for k,v in raw.items()})
            packed.update({f"{geom}/{k}":v.numpy() for k,v in tape.items() if k!="study_u"})
        path=dest/f"{birth}.npz"
        np.savez_compressed(path,**packed)
        result["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"code-probe-{label}.json").write_text(json.dumps(result,indent=1,allow_nan=False)+"\n",encoding="utf8")
    print(label,"completed",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--seed",type=int,default=3421185)
    p.add_argument("--births",type=int,nargs="+",default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
