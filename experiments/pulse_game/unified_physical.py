"""Public graph ticks are shared across phenotypes; each real-action life has its own plastic state."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from experiments.undirected._cuda_nvrtc import compile_kernels
from . import mixture_flash, symmetry_physics as device, unified_audit as baseline


MOTION_STEP=np.array([.17,.65,.09,.05,.19,.18,.18],np.float32)
REST_STEP=np.array([2.5,.06,.045,.26,.55,.045,.18,.18,.12,.30,.014,.22],np.float32)
STEP=np.r_[MOTION_STEP,REST_STEP].astype(np.float32)
BOUNDS=np.array([[-.4,1.25],[.2,12.],[.12,.98],[.012,.7],[.02,2.4],[.04,2.9],[-.65,1.8],
                 [3.,45.],[.12,.998],[0,.70],[.02,2.],[.25,8.],[.006,2.0],
                 [-1.8,2.5],[-2.3,.7],[-.8,.8],[-1.2,1.2],[.90,1.],[.08,2.0]],np.float32)


def params(front,route,motion):
    value=front["scalars"]
    return np.array([*motion,value[0],value[1],0.,route["eta"],route["amp"],route["ridge"] or .1,
                     1.,-1.,0.,0.,1.,1.],np.float32)


def phenotype(native,genes,mode):
    x=np.atleast_2d(genes)
    full=np.zeros((len(x),len(native)),np.float32)
    if mode=="motor":full[:,:7]=x
    elif mode=="full":full[:]=x
    elif mode=="read":full[:,:10]=x
    else:raise ValueError(mode)
    return np.clip(native+STEP*full,BOUNDS[:,0],BOUNDS[:,1])


def materialize(front,route,motion,x,mode):
    v=phenotype(params(front,route,motion),x,mode)[0]
    front={**front,"scalars":[float(v[7]),float(v[8]),*front["scalars"][2:]],"age_cost":float(v[9])}
    route={**route,"eta":float(v[10]),"amp":float(v[11]),"ridge":float(v[12]),
           "reward_factor":float(v[13]),"prediction_factor":float(v[14]),"offset":float(v[15]),
           "familiarity_factor":float(v[16]),"retention":float(v[17]),"rls_step":float(v[18])}
    return front,route,v[:7]


@lru_cache(maxsize=10)
def kernels(n,h,s,cap):
    body=Path(__file__).with_name("unified_gpu.cu").read_text()
    code=f"#define N {n}\n#define H {h}\n#define S {s}\n#define CAP {cap}\n#define G {len(STEP)}\n"+body
    return compile_kernels(code,("prepare","play"),device=torch.device("cuda:0"),options=("--fmad=false",))


class Cohort:
    def __init__(self,episode,front,route,motion):
        arena=episode.arena;menu=arena.menu;seeds=menu.body.seeds
        self.n,self.tr=menu.body.ctx.shape;self.episode=episode
        self.initial=params(front,route,motion)
        self.code=kernels(front["matrix"].shape[0],route["first"][1].size,route["slots"],front["capacity"])
        ot,ct=(episode.sensory_timing if hasattr(episode,"sensory_timing") else mixture_flash.timing(seeds,self.tr))
        self.keys=torch.empty((self.n,self.tr,16,2,8),device="cuda")
        self.address=torch.empty((self.n,self.tr,16,route["slots"]),device="cuda")
        arrays=[2*menu.chemicals-1,arena.appearance,ot.astype(np.uint8),ct.astype(np.uint8),front["matrix"],
                front["sensors"],*front["ground"],front["transform"],*route["first"],*route["second"]]
        kind={"learned":0,"random":0,"single":1,"public":2,"shared_public":3}[route["routing"]]
        self.code["prepare"]((self.n,),(64,),*[device.tensor(a) for a in arrays],self.keys,self.address,
            self.tr,kind,float(route.get("shared",0.)))
        num=(self.tr*16+127)//128
        noise=np.stack([np.random.default_rng(np.random.SeedSequence([s,745930])).standard_normal((num*128,2),dtype=np.float32)[:self.tr*16] for s in seeds])
        self.inputs=[device.tensor(v) for v in (menu.body.distances,menu.body.wind,menu.realized,arena.probe.astype(np.uint8),noise)]
        self.kind=int(route["updater"]=="rls")
        self.checks=(episode.headline_probes if hasattr(episode,"headline_probes") else
                     (episode.checks["B_after_0"],episode.checks[f"B_after_{episode.steps[-1]}"],episode.checks["return_A"]))
        self.normalizer=torch.tensor([1/self.n]*4+[1/(self.n*len(episode.checks))]*2,device="cuda")
        self.dummy8=torch.empty(1,device="cuda",dtype=torch.int8);self.dummy32=torch.empty(1,device="cuda")

    def run(self,genes,mode,*,detail=False,variant=0,batch=1024):
        pop=phenotype(self.initial,genes,mode);out=[]
        for j in range(0,len(pop),batch):
            v=pop[j:j+batch]
            counts=torch.empty((len(v),self.n,6),device="cuda",dtype=torch.int32)
            shape=(len(v),self.n,self.tr)
            if detail:
                actions=[torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)]
                log=torch.empty(shape,device="cuda")
            else:actions=[self.dummy8]*3;log=self.dummy32
            self.code["play"]((self.n,len(v)),(32,),self.keys,self.address,*self.inputs,device.tensor(v),*actions,log,
                        counts,self.n,self.tr,*self.checks,self.kind,int(detail),int(variant))
            if detail:
                side,tick,sign=[x.cpu().numpy() for x in actions]
                gain=log.cpu().numpy();out.extend((device.Outcome(a,b,c),g) for a,b,c,g in zip(side,tick,sign,gain,strict=True))
            else:out.append((counts.sum(1).float()*self.normalizer).cpu().numpy())
        return out if detail else np.concatenate(out)


