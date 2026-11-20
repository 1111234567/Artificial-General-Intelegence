"""Can a two-rate local slow summary retain old content without overconsolidating every change?"""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_slow_reference as slow, interleave_homeostasis as home
from . import local_trace_graph as net, local_trace_train as previous
from .interleave_interference import DATA,EVIDENCE


GRID=(0.,.005,.02,.08)
METRICS=("a_before_x","a","b","x","all_three","slow_a")
AGES=(4,16,64,128)


def permute_eligible(values,mask,priorities):
    shape=values.shape;n=len(values)
    v=values.flatten(1);m=mask.reshape_as(v)
    order=(priorities.reshape_as(v)+2*(~m)).argsort(1)
    count=m.sum(1).clamp_min(1)
    src=(torch.arange(v.shape[1])[None]+1)%count[:,None]
    rotated=v.gather(1,order).gather(1,src)
    return v.scatter(1,order,rotated).reshape(shape)


def advance(rule,m,right,factor,u,rates,permutation,mode):
    a=net.act(rule,m.live,-right,right,u[:,0])
    obs=net.observe(rule,a,-right,right,a.side*factor)
    result=net.commit(rule,obs,u[:,1:])
    reference=net.Graph(m.slow_w,m.slow_b,m.ancestry)
    fast,alive=home.interpolate(rule,obs,result,reference,m.ancestry,"birth",.12)
    def gain(delta,separation,shape):
        compatible=(delta*separation).sign()
        mid=(rates[:,0]+rates[:,1]).reshape(shape)/2
        contrast=(rates[:,0]-rates[:,1]).reshape(shape)/2
        return mid+contrast*compatible
    wgain=gain(result.receipt.weight_delta,m.live.graph.weights-m.slow_w,(-1,1,1))
    bgain=gain(result.receipt.bias_delta,m.live.graph.bias-m.slow_b,(-1,1))
    if mode=="permuted":
        wgain=permute_eligible(wgain,alive,permutation[:,:400])
        bgain=permute_eligible(bgain,torch.ones_like(bgain,dtype=torch.bool),permutation[:,400:])
    elif mode=="mean":
        wgain=((wgain*alive).sum((1,2))/alive.sum((1,2)).clamp_min(1))[:,None,None]
        bgain=bgain.mean(1,keepdim=True)
    next_w=torch.where(alive,torch.lerp(m.slow_w,fast.graph.weights,wgain),m.slow_w)
    next_b=torch.lerp(m.slow_b,fast.graph.bias,bgain)
    return slow.Memory(net.wait(rule,fast,home.WAIT),next_w,next_b,alive)


@torch.no_grad()
def run_bank(g,r,tapes,ecology,exposures,variants,priority_seed,*,mode="local",opposite=False):
    tape,extra,extra_orientation,extra_draw=tapes
    n=tape.pairs;v=len(variants);lanes=n*4*v
    def copies(x):return x.repeat((v,)+(1,)*(x.ndim-1))
    rates=torch.as_tensor(variants,dtype=g.bias.dtype).repeat_interleave(n*4,dim=0)
    rng=np.random.default_rng([priority_seed,19237])
    perm=copies(torch.from_numpy(rng.random((n,420),dtype=np.float32)).repeat_interleave(4,0))
    state=net.start(g,lanes)
    m=slow.Memory(state,state.graph.weights,state.graph.bias,state.graph.present)
    view=r.expand(lanes);twice=r.expand(lanes*2);rows=torch.arange(n)
    asign=torch.tensor([-1.,-1.,1.,1.]).repeat(n);bsign=torch.tensor([-1.,1.,-1.,1.]).repeat(n)
    def event(mem,ids,o,u):
        right=(tape.objects[rows,ids]*o[:,None]).repeat_interleave(4,0)
        sign=torch.where(ids.repeat_interleave(4)==0,asign,bsign)
        factor=(tape.laws[rows,ids]*o).repeat_interleave(4)*sign
        return advance(view,mem,copies(right),copies(factor),copies(u.repeat_interleave(4,0)),rates,perm,mode)
    seq=tape.visits[home.ECOLOGIES.index(ecology)]
    for t in range(2):m=event(m,seq[t],tape.orientation[t],tape.draws[t])
    if exposures==4:
        for t in range(6):m=event(m,extra[t],extra_orientation[t],extra_draw[t])
    if opposite:m=slow.alter_slow(m,torch.arange(lanes)^2)
    duplicate=torch.arange(lanes).repeat_interleave(2)
    xs=copies((tape.objects[:,2]*tape.x_orientation[:,None]).repeat_interleave(8,0))
    xgoal=copies(tape.x_orientation.repeat_interleave(8)*torch.tensor([-1.,1.]).repeat(4*n))
    xu=copies(tape.x_draws.repeat_interleave(8,0))
    probes=copies((tape.objects[:,:3]*tape.probe_orientation[:,:,None]).repeat_interleave(8,0))
    ab=(tape.laws[:,:2].repeat_interleave(4,0)*torch.stack((asign,bsign),1)).repeat_interleave(2,0)
    goals=copies(torch.cat((ab,torch.tensor([-1.,1.]).repeat(4*n)[:,None]),1)*
                 tape.probe_orientation.repeat_interleave(8,0))
    u=copies(tape.probe_draws.repeat_interleave(8,0));records=[]
    for k in range(max(AGES)+1):
        if k in AGES:
            snap=m.subset(duplicate);obj=probes[:,0]
            before=net.act(twice,snap.live,-obj,obj,u[:,0]).side==goals[:,0]
            slow_a=slow.slow_read(snap,twice,obj,u[:,0])==goals[:,0]
            after=advance(twice,snap,xs,xgoal,xu,rates[duplicate],perm[duplicate],mode)
            obs=torch.stack([net.act(twice,after.live,-probes[:,i],probes[:,i],u[:,i]).side for i in range(3)],1)==goals
            good=torch.stack((before,*obs.unbind(1),obs.all(1),slow_a),1)
            records.append(good.reshape(v,n,4,2,len(METRICS)).numpy())
        if k<max(AGES):
            t=k+2;m=event(m,seq[t],tape.orientation[t],tape.draws[t])
    return np.stack(records,axis=1)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"local-consolidation")
    p.add_argument("--cal",type=int,default=32);p.add_argument("--test",type=int,default=96)
    a=p.parse_args();torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(cal_births=list(home.CAL_BIRTHS),test_births=list(home.CONFIRM_BIRTHS),
              cal_worlds=a.cal,test_worlds=a.test,cal_seed=761841,test_seed=4319773,
              grid=GRID,ages=AGES,metrics=METRICS,gamma=.12,code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol="Same budget and live slow template as the earlier static/global study. Compare real update "
                       "direction to pre-write fast-minus-slow direction at each continuously surviving weight and bias. "
                       "Concordant updates use rate a, discordant updates rate b, zero product uses their arithmetic mean. "
                       "No semantic variable is seen by the local operation. Fix one pair on original births by "
                       "maximizing true joint A/B/X after 4 initial reviews and an A absence across ages 16/64/128. "
                       "Select scalar comparator on the diagonal by the same rule. Confirm on disjoint six births/worlds, "
                       "including mean-dose, shuffled-rate and opposite-A slow-bank interventions.")
    home._lock(a.out/"run.json",plan)
    options=[(v,w) for v in GRID for w in GRID]
    ct=slow.public_tapes(a.cal,plan["cal_seed"],661937)
    cal=[];fingerprints={}
    for birth in home.CAL_BIRTHS:
        g,r=previous.load_checkpoint(DATA/"retention-objective","odor","uniform",birth,8000)
        chunks=[]
        for i in range(0,len(options),4):
            chunks.append(run_bank(g,r,ct,"absent_a",4,options[i:i+4],plan["cal_seed"]))
        cells=np.concatenate(chunks,axis=0)
        path=a.out/f"cal-{birth}.npz";home._save_raw(path,dict(correct=cells))
        fingerprints[path.name]=sha256(path.read_bytes()).hexdigest()
        cal.append(cells[:,1:,:,:,:,4].mean((1,2,3,4)))
        print("calibration",birth,flush=True)
    value=np.stack(cal).mean(0)
    best=options[int(value.argmax())]
    diagonal=[i for i,(x,y) in enumerate(options) if x==y]
    matched=options[max(diagonal,key=lambda i:value[i])]
    selection=dict(local=best,global_rate=matched,grid=options,means=value.tolist(),hashes=fingerprints)
    home._lock(a.out/"selection.json",selection)
    print("selected",best,matched,flush=True)
    tt=slow.public_tapes(a.test,plan["test_seed"],661943)
    results={};hashes={}
    for b in home.CONFIRM_BIRTHS:
        g,r=previous.load_checkpoint(DATA/"retention-objective-confirmation","odor","uniform",b,8000)
        cells={}
        for eco in home.ECOLOGIES:
            for times in (1,4):
                for key,variant,mode,swap in (("birth",(0.,0.),"local",False),("global",matched,"local",False),
                                              ("local",best,"local",False),("permuted",best,"permuted",False),
                                              ("mean",best,"mean",False),("opposite_A",best,"local",True)):
                    raw=run_bank(g,r,tt,eco,times,[variant],plan["test_seed"],mode=mode,opposite=swap)[0]
                    path=a.out/f"test-{b}-{eco}-{times}-{key}.npz";home._save_raw(path,dict(correct=raw))
                    hashes[path.name]=sha256(path.read_bytes()).hexdigest()
                    cells[f"{eco}/{times}/{key}"]=raw.mean((1,2,3)).tolist()
        results[b]=cells;print("confirmation",b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    report=dict(plan=plan,selection=selection,confirmation=results,confirmation_raw_sha256=hashes)
    home._lock(a.out/"summary.json",report)
    (EVIDENCE/"local-consolidation.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":main()
