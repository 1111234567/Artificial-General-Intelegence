"""Batched CUDA forward-only physics, independently checked against native NumPy bodies."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
import torch

from experiments.undirected._cuda_nvrtc import compile_kernels
from . import mixture_flash as flash, symmetry_memory_deploy as native


CHANGES=("true","transplant_rule","clear_rule","transplant_events","no_learning","no_anchor",
         "invert_rule","same_rule","opposite_rule")


@dataclass
class Outcome:
    contact_side:np.ndarray
    contact_tick:np.ndarray
    reward:np.ndarray


@lru_cache(maxsize=8)
def program(nodes,hidden,k,capacity):
    if capacity>32 or hidden>64 or nodes>64 or k>32:raise ValueError("Unexpected CUDA tile")
    source=Path(__file__).with_name("symmetry_physics_gpu.cu").read_text()
    return compile_kernels(f"#define N {nodes}\n#define WIDTH {hidden}\n#define K {k}\n#define CAP {capacity}\n"+source,
                           ("smell","act_and_taste"),device=torch.device("cuda:0"),options=("--fmad=false",))


def tensor(x,dtype=None):
    a=np.ascontiguousarray(x)
    return torch.as_tensor(a,device="cuda",dtype=dtype)


class Physical:
    def __init__(self,protocol):
        menu=protocol.menu;baseline=menu.body
        self.n,self.trials=baseline.ctx.shape
        if self.n%2:raise ValueError("Need paired physical worlds")
        self.boundary=protocol.exposed
        ot,ct=flash.timing(baseline.seeds,self.trials)
        public=[menu.chemicals*2.-1.,baseline.ctx.astype(np.uint8),ot.astype(np.uint8),ct.astype(np.uint8)]
        self.senses=[tensor(v) for v in public]
        chunks=(self.trials*16+127)//128
        noise=np.stack([np.random.default_rng(np.random.SeedSequence([s,745930])).standard_normal(
                         (128*chunks,2),dtype=np.float32)[:self.trials*16] for s in baseline.seeds])
        self.body=[tensor(baseline.distances),tensor(baseline.wind),tensor(menu.realized),tensor(noise)]

    def run(self,p,*,changes=("true",),tonic=.39,gain=1.6):
        g="ground" in p
        hidden=p["a"][1].size if not g else 4
        k=p["latent"];code=program(p["matrix"].shape[0],hidden,k,p["capacity"])
        dummy=np.zeros(1,np.float32)
        w={"a":p["ground"] if g else p["a"],**{key:p.get(key,(dummy,dummy)) for key in ("b","read","write")}}
        sensory=[p["matrix"],p["sensors"],*w["a"],*w["b"],*w["read"],*w["write"],p.get("transform",dummy)]
        enc=[tensor(v) for v in sensory]
        features=torch.empty((self.n,self.trials,16,2,2,k),device="cuda")
        code["smell"]((self.n,2),(64,),*self.senses,*enc,features,self.trials,int(g))
        rules=[u for pair in (p["rule"] or [(dummy,dummy),(dummy,dummy)]) for u in pair]
        weights=[tensor(np.asarray(p["scalars"],np.float32)),*[tensor(v) for v in rules]]
        kind={"delta":0,"learned":1,"fixed":2}[p["change"]]
        ids=[CHANGES.index(c) for c in changes]
        def kernel(sub,mode):
            shape=(len(sub),self.n,self.trials)
            outs=[torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)]
            master=torch.empty(shape,device="cuda",dtype=torch.float32)
            code["act_and_taste"]((self.n//2,len(sub)),(64,),features,*self.body,*weights,*outs,master,
                                   self.n//2,self.trials,self.boundary,kind,mode,float(tonic),float(gain))
            a,b,c=(v.cpu().numpy() for v in outs)
            v=master.cpu().numpy()
            return {CHANGES[j]:(Outcome(a[i],b[i],c[i]),v[i]) for i,j in enumerate(sub)}
        if ids==list(range(min(ids),max(ids)+1)):return kernel(ids,-1-min(ids))
        results={}
        for i in ids:results.update(kernel([i],i))
        return results


def verify(p,protocol,seeds,changes):
    now=time.perf_counter()
    sim=Physical(protocol);data=sim.run(p,changes=changes)
    first_gpu=time.perf_counter()-now
    now=time.perf_counter();sim.run(p,changes=changes);warm=time.perf_counter()-now
    measures={}
    for c in changes:
        agent=native.Forager(p,seeds+seeds,intervention=c,boundary=protocol.exposed)
        clock=time.perf_counter();expected=flash.rollout(agent,protocol.menu)
        seconds=time.perf_counter()-clock
        result,gain=data[c]
        a=result.contact_side==expected.contact_side
        b=result.contact_tick==expected.contact_tick
        measure=dict(choice_agreement=float(a.mean()),tick_agreement=float(b.mean()),
                     signed_agreement=float((result.reward==expected.reward).mean()),
                     master_max=float(np.max(np.abs(gain-np.stack(agent.record,axis=1)))),
                     cpu_seconds=seconds)
        measures[c]=measure
    return dict(cases=measures,first_gpu_seconds=first_gpu,warm_gpu_seconds=warm,worlds=len(seeds),
                bouts_per_world=protocol.menu.body.trials,branches=2,changes=changes)
