"""Does a slow summary of the expressed graph preserve a rehearsed memory and fresh learning?"""

from argparse import ArgumentParser
from dataclasses import dataclass,replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_homeostasis as home, local_trace_graph as net, local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


AGES=(0,4,16,64,128)
GRID=(0.,.005,.02,.08,.25)
FIELDS=("old_before","old_after","B","new_X","all_three","old_and_X","slow_read_A")
PLAN=EVIDENCE/"slow-reference-plan.json"


@dataclass(frozen=True)
class Memory:
    live:net.State
    slow_w:torch.Tensor
    slow_b:torch.Tensor
    ancestry:torch.Tensor

    def subset(self,ids):
        return Memory(self.live.subset(ids),self.slow_w[ids],self.slow_b[ids],self.ancestry[ids])


def contact(coeff,m,right,factor,u,gamma,beta):
    a=net.act(coeff,m.live,-right,right,u[:,0])
    taste=a.side*factor
    e=net.observe(coeff,a,-right,right,taste)
    written=net.commit(coeff,e,u[:,1:])
    template=net.Graph(m.slow_w,m.slow_b,m.ancestry)
    fast,alive=home.interpolate(coeff,e,written,template,m.ancestry,"birth",gamma)
    if beta:
        slow_w=torch.where(alive,m.slow_w.lerp(fast.graph.weights,beta),m.slow_w)
        slow_b=m.slow_b.lerp(fast.graph.bias,beta)
    else:
        slow_w,slow_b=m.slow_w,m.slow_b
    return Memory(net.wait(coeff,fast,home.WAIT),slow_w,slow_b,alive)


def alter_slow(m,twin):
    donor=m.subset(twin)
    both=m.ancestry&donor.ancestry
    return replace(m,slow_w=torch.where(both,donor.slow_w,m.slow_w),slow_b=donor.slow_b)


def slow_read(m,rule,right,draw):
    g=m.live.graph
    reflected=replace(g,weights=torch.where(m.ancestry,net._nonzero(m.slow_w),g.weights),bias=m.slow_b)
    return net.act(rule,replace(m.live,graph=reflected),-right,right,draw).side


def public_tapes(n,seed,domain):
    h=home.make_tape(n,seed,domain,max(AGES))
    rng=np.random.default_rng([seed,domain,513833])
    first=rng.integers(2,size=(3,n))
    extra=np.stack((first,1-first),1).reshape(6,n)
    return h,torch.from_numpy(extra),torch.from_numpy(home._sign(rng,(6,n))),torch.from_numpy(
        rng.random((6,n,5),dtype=np.float32))


@torch.no_grad()
def live(g,rule,tapes,ecology,exposures,gamma,beta,*,swap_slow=False):
    tape,extra,extra_orient,extra_draw=tapes
    n=tape.pairs;row=torch.arange(n);lanes=4*n
    s=net.start(g,lanes)
    m=Memory(s,s.graph.weights,s.graph.bias,s.graph.present)
    view=rule.expand(lanes);view_x=rule.expand(2*lanes)
    asign=torch.tensor([-1.,-1.,1.,1.]).repeat(n)
    bsign=torch.tensor([-1.,1.,-1.,1.]).repeat(n)
    def event(ids,ori,draw,mem):
        right=(tape.objects[row,ids]*ori[:,None]).repeat_interleave(4,dim=0)
        sign=torch.where(ids.repeat_interleave(4)==0,asign,bsign)
        factor=(tape.laws[row,ids]*ori).repeat_interleave(4)*sign
        return contact(view,mem,right,factor,draw.repeat_interleave(4,dim=0),gamma,beta)
    visits=tape.visits[home.ECOLOGIES.index(ecology)]
    for t in range(2):
        m=event(visits[t],tape.orientation[t],tape.draws[t],m)
    if exposures==4:
        for t in range(6):m=event(extra[t],extra_orient[t],extra_draw[t],m)
    elif exposures!=1:raise ValueError("Unsupported pre-exposure count")
    if swap_slow:
        m=alter_slow(m,torch.arange(lanes)^2)
    two=torch.arange(lanes).repeat_interleave(2)
    x_factor=tape.x_orientation.repeat_interleave(8)*torch.tensor([-1.,1.]).repeat(lanes)
    x_right=(tape.objects[:,2]*tape.x_orientation[:,None]).repeat_interleave(8,dim=0)
    x_draw=tape.x_draws.repeat_interleave(8,dim=0)
    menus=(tape.objects[:,:3]*tape.probe_orientation[:,:,None]).repeat_interleave(8,dim=0)
    target=tape.laws[:,:2].repeat_interleave(4,dim=0)*torch.stack((asign,bsign),1)
    target=torch.cat((target.repeat_interleave(2,dim=0),torch.tensor([-1.,1.]).repeat(lanes)[:,None]),1)
    target*=tape.probe_orientation.repeat_interleave(8,dim=0)
    draws=tape.probe_draws.repeat_interleave(8,dim=0)
    results=[];distance=[]
    for k in range(max(AGES)+1):
        if k in AGES:
            snap=m.subset(two)
            a_right=menus[:,0]
            before=net.act(view_x,snap.live,-a_right,a_right,draws[:,0]).side
            slow=slow_read(snap,view_x,a_right,draws[:,0])
            acquired=contact(view_x,snap,x_right,x_factor,x_draw,gamma,beta)
            after=torch.stack([net.act(view_x,acquired.live,-menus[:,i],menus[:,i],draws[:,i]).side for i in range(3)],1)
            good=after==target
            values=torch.stack((before==target[:,0],*good.unbind(1),good.all(1),good[:,0]&good[:,2],slow==target[:,0]),1)
            results.append(values.reshape(n,4,2,len(FIELDS)).numpy())
            distance.append(torch.stack(((m.slow_w-s.graph.weights).abs().flatten(1).sum(1),
                                         (m.live.graph.weights-m.slow_w).abs().flatten(1).sum(1),
                                         m.ancestry.flatten(1).sum(1)),1).reshape(n,4,3).numpy())
        if k<max(AGES):
            t=k+2;m=event(visits[t],tape.orientation[t],tape.draws[t],m)
    return dict(correct=np.stack(results),distance=np.stack(distance))


def split(out,source,births,worlds,seed,domain,variants):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    tapes=public_tapes(worlds,seed,domain)
    results={};hashes={}
    for b in births:
        g,r=old.load_checkpoint(source,"odor","uniform",b,8000)
        per_root={}
        for ecology in home.ECOLOGIES:
            for exposure in (1,4):
                for name,gamma,beta,swap in variants:
                    file=out/f"{b}-{ecology}-{exposure}-{name}.npz"
                    if file.is_file():
                        with np.load(file) as z:
                            raw={"correct":z["correct"],"distance":z["distance"]}
                        if raw["correct"].shape!=(len(AGES),worlds,4,2,len(FIELDS)):
                            raise ValueError("Saved slow-state trace has incorrect dimensions")
                    else:
                        raw=live(g,r,tapes,ecology,exposure,gamma,beta,swap_slow=swap)
                        home._save_raw(file,raw)
                    hashes[file.name]=sha256(file.read_bytes()).hexdigest()
                    per_root[f"{ecology}/{exposure}/{name}"]=raw["correct"].mean(axis=(1,2,3)).tolist()
        results[b]=per_root;print(out.name,b,flush=True)
    return results,hashes


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"slow-reference")
    p.add_argument("--cal-worlds",type=int,default=48);p.add_argument("--test-worlds",type=int,default=96)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cp=DATA/"retention-objective"; vp=DATA/"retention-objective-confirmation"
    seeds=(6341857,6341879)
    manifest=dict(plan_sha256=sha256(PLAN.read_bytes()).hexdigest(),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  gamma=.12,grid=GRID,ages=AGES,fields=FIELDS,cal_worlds=a.cal_worlds,test_worlds=a.test_worlds,
                  seeds=seeds,cost="A live per-lane dense fp32 slow template of 20x20 weights and 20 biases "
                  "(1680 bytes) plus a 20x20 bool ancestry mask (400 bytes) in this implementation. No oracle "
                  "value, task ID or reward polarity enters the slow update. Only the shared beta is externally selected.")
    home._lock(a.out/"run.json",manifest)
    calibration=[(f"slow-{v:g}",.12,v,False) for v in GRID]
    cal,hashes=split(a.out/"calibration",cp,home.CAL_BIRTHS,a.cal_worlds,seeds[0],732189,calibration)
    idx=[AGES.index(k) for k in (16,64,128)]
    scores={v:float(np.mean([np.array(c[f"absent_a/4/slow-{v:g}"])[idx,4].mean() for c in cal.values()])) for v in GRID}
    best=max(GRID,key=lambda v:scores[v])
    signature=home._lock(a.out/"selection.json",dict(beta=best,cal_scores=scores,hashes=hashes))
    variants=[("native",0.,0.,False),("static",.12,0.,False),("selected",.12,best,False),
              ("opposite_A_slow",.12,best,True)]
    held,hh=split(a.out/"confirmation",vp,home.CONFIRM_BIRTHS,a.test_worlds,seeds[1],732193,variants)
    out=dict(manifest=manifest,selected_beta=best,cal_scores=scores,selection_sha256=signature,
             confirmation_births=home.CONFIRM_BIRTHS,results=held,raw_sha256=hh)
    home._lock(a.out/"summary.json",out)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/"slow-reference.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    print("frozen beta",best,"calibration",scores,flush=True)


if __name__=="__main__":main()
