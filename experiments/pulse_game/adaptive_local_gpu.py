"""Exactly replay public prototype formation with no rule identity in the policy interface."""
from functools import lru_cache
import numpy as np
import torch
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import longrun_plastic as previous,unified_physical as ordinary,mixture_flash as signals
from . import symmetry_physics as gpu,adaptive_local as address


@lru_cache(maxsize=8)
def program(n,h,slots,cap):
    src=previous.source(n,h,slots,cap)
    edits={
     "int* summary,int lives":"int* summary,const unsigned char* born,const unsigned char* clock,int inherit,int lives",
     "master[i]=p[17]*master[i]+(rule==1?p[18]:p[10])*saved_err*dir;":
       "if(codes[(e-1)*S+i]!=0.f)master[i]=p[17]*master[i]+(rule==1?p[18]:p[10])*saved_err*dir;",
     "if(i==0){if(has)++count;pending=0;":
       "if(i==0){if(has)++count;pending=0;if(inherit && t==clock[ix] && born[ix]!=255)master[born[ix]]=master[0];"
    }
    for a,b in edits.items():
        if src.count(a)!=1:raise ValueError("Frozen update source changed: "+a)
        src=src.replace(a,b)
    return compile_kernels(src,("play",),device=torch.device("cuda:0"),options=("--fmad=false",))["play"]


class Cohort(ordinary.Cohort):
    def __init__(self,ep,front,route,motion,slots=6,radius=.09,inherit=True):
        stub={**route,"slots":slots,"routing":"single"}
        super().__init__(ep,front,stub,motion)
        _,ct=(ep.sensory_timing if hasattr(ep,"sensory_timing") else signals.timing(ep.arena.menu.body.seeds,self.tr))
        self.ct=gpu.tensor(ct.astype(np.uint8))
        ids,born,self.grid=address.public_addresses(ep.arena.appearance,slots,radius,ct)
        # The extra all-zero row stands for "no context yet observed".
        lut=np.vstack((np.eye(slots,dtype=np.float32),np.zeros((1,slots),np.float32)))
        self.address=gpu.tensor(lut[np.where(ids==255,slots,ids)])
        self.born=gpu.tensor(born);self.inherit=inherit
        self.fn=program(front["matrix"].shape[0],route["first"][1].size,slots,front["capacity"])
    def play(self,body,alpha):
        v=np.r_[body,np.asarray(alpha,np.float32)].astype(np.float32)[None]
        actions=[torch.empty((self.n,self.tr),device="cuda",dtype=torch.int8) for _ in range(3)]
        g=torch.empty((self.n,self.tr),device="cuda");stats=torch.empty((1,self.n,6),device="cuda",dtype=torch.int32)
        self.fn((self.n,1),(32,),self.keys,self.address,*self.inputs,gpu.tensor(v),*actions,g,stats,
            self.born,self.ct,int(self.inherit),self.n,self.tr,*self.checks,self.kind,1,0)
        return gpu.Outcome(*(a.cpu().numpy() for a in actions))

