"""Flip exactly one object's history while holding other tastes, displays and draws fixed."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_train as common, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


N=3
GEOMETRIES=("orthogonal","random","non_antipodal","shared_context","mixed")


def geometry_tape(seed,birth,worlds,geom):
    rng=np.random.default_rng([685074,seed,birth])
    geom_rng=np.random.default_rng([685100,seed,birth])
    vectors=age.directions(geom_rng,worlds,N)
    if geom=="orthogonal":
        basis,_=np.linalg.qr(geom_rng.normal(size=(worlds,3,3)))
        vectors=(1.35*basis).astype(np.float32)
    left=-vectors
    right=vectors
    cue=np.zeros_like(vectors)
    if geom=="non_antipodal":
        left=age.directions(geom_rng,worlds,N)
        close=(left*right).sum(-1)/(1.35**2)>.6
        while close.any():
            left[close]=age.directions(geom_rng,int(close.sum()))
            close=(left*right).sum(-1)/(1.35**2)>.6
    if geom=="shared_context":
        right=np.repeat(vectors[:,:1],N,axis=1)
        left=-right
        order=geom_rng.permuted(np.tile(np.arange(3),(worlds,1)),axis=1)
        cue=common.CUE_BANK[order]
    if geom=="mixed":
        # Two odor objects and one feeder relation. All three use one physical menu
        # for each study here; the feeder demonstrations are tested separately below.
        right[:,2]=common.MARK_AXIS
        left[:,2]=-common.MARK_AXIS
    labels=np.array(list(product((-1.,1.),repeat=N)),dtype=np.float32)
    masks=len(labels)
    def public(value):return torch.as_tensor(np.repeat(value,masks,axis=0))
    return dict(left=public(left+cue),right=public(right+cue),
                law=torch.from_numpy(np.tile(labels,(worlds,1))),
                order=public(rng.permuted(np.tile(np.arange(N),(worlds,1)),axis=1)),
                study_flip=public(age.signs(rng,(worlds,N))),probe_flip=public(age.signs(rng,(worlds,N))),
                study_u=public(rng.random((worlds,N,6),dtype=np.float32)),
                probe_u=public(rng.random((worlds,N),dtype=np.float32)))


def present(tape,item,flip):
    rows=torch.arange(len(item))
    l,r=tape["left"][rows,item],tape["right"][rows,item]
    normal=flip[:,None]>0
    return torch.where(normal,l,r),torch.where(normal,r,l)


@torch.no_grad()
def physical(g,r,tape,mode):
    n=len(tape["law"])
    state=stream.start(g,r,n)
    coeff=r.expand(n) if r is not None else None
    rows=torch.arange(n)
    history,answers,prob=[],[],[]
    for t in range(N):
        item=tape["order"][:,t]
        flip=tape["study_flip"][:,t]
        left,right=present(tape,item,flip)
        state,side,reward=stream.contact(coeff,state,left,right,tape["law"][rows,item]*flip,
                                         torch.zeros_like(flip),tape["study_u"][:,t],mode)
        state=stream.wait(coeff,state,4)
        history.append(torch.stack((side,reward),1))
    for item in range(N):
        flip=tape["probe_flip"][:,item]
        left,right=present(tape,torch.full((n,),item),flip)
        a=stream.action(coeff,state,left,right,tape["probe_u"][:,item])
        # Express the action relative to each item's canonical geometry, not motor side.
        answers.append((a.side*flip).to(torch.int8))
        prob.append((a.motor_logit*flip).sigmoid())
    return dict(answer=torch.stack(answers,1).numpy(),p_plus=torch.stack(prob,1).numpy(),
                contact=torch.stack(history,1).numpy().astype(np.int8))


def summarize(raw,law):
    choices=raw["answer"].reshape(-1,2**N,N)
    p=raw["p_plus"].reshape(choices.shape)
    pref=law.reshape(choices.shape)
    correct=choices==pref
    output_ids=((choices>0)*(2**np.arange(N)[::-1])).sum(-1)
    distinct=np.array([len(np.unique(c)) for c in output_ids])
    transition=np.zeros((N,N))
    signed=np.zeros_like(transition)
    collateral=np.zeros_like(transition)
    for j in range(N):
        negative=np.array([k for k in range(2**N) if not k & (1<<(N-j-1))])
        positive=negative^(1<<(N-j-1))
        a,b=choices[:,negative],choices[:,positive]
        transition[j]=(a!=b).mean((0,1))
        signed[j]=((b.astype(float)-a)/2).mean((0,1))
        collateral[j]=np.abs(p[:,positive]-p[:,negative]).mean((0,1))
    return dict(correct=float(correct.mean()),all=float(correct.all(2).mean()),
                by_object=correct.mean((0,1)).tolist(),by_private_map=correct.all(2).mean(0).tolist(),
                distinct_output_patterns=float(distinct.mean()),
                change_one_input_flip_rate=transition.tolist(),
                direction_of_response_to_positive=signed.tolist(),
                probability_shift_abs=collateral.tolist(),
                targeted_minus_collateral=float(np.diag(transition).mean()-transition[~np.eye(N,dtype=bool)].mean()))


def run(source,label,births,seed,worlds):
    torch.set_num_threads(1)
    dest=DATA/"binding"/label
    dest.mkdir(parents=True,exist_ok=True)
    report=dict(seed=seed,births=births,worlds=worlds,labels_per_world=2**N,results={},sha256={},
                meaning="For each fixed display history and uniform tape, independently cross all eight stable "
                        "assignments to three objects. Each was experienced once. Three terminal motor queries fork "
                        "from the same unmodified final state. Flip probabilities compare matched assignments differing "
                        "at a single input bit. Distinct output count is descriptive, not an information-theoretic capacity.")
    for birth in births:
        g,r=source.load(birth)
        zall,summary={},{}
        for name in GEOMETRIES:
            tape=geometry_tape(seed,birth,worlds,name)
            summary[name]={}
            for mode in ("active","sham"):
                z=physical(g,r,tape,mode)
                summary[name][mode]=summarize(z,tape["law"].numpy())
                zall.update({f"{name}/{mode}/{k}":v for k,v in z.items()})
            zall.update({f"{name}/{k}":v.numpy() for k,v in tape.items() if k!="study_u"})
        report["results"][birth]=summary
        path=dest/f"{birth}.npz"
        np.savez_compressed(path,**zall)
        report["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"binding-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")
    print(label,"complete",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--seed",type=int,default=771905)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.worlds)
