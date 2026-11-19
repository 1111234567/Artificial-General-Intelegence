"""Use long-history covariance for direction while keeping a lower bound on the available update magnitude."""
from functools import lru_cache
import numpy as np
import torch
from . import unified_physical as ordinary,symmetry_physics as gpu
from experiments.undirected._cuda_nvrtc import compile_kernels
from pathlib import Path


@lru_cache(maxsize=8)
def compiled(n,h,s,cap):
    raw=Path(ordinary.__file__).with_name("unified_gpu.cu").read_text()
    edits={
       "int* summary,int lives":"int* summary,const int* begin,int lives",
       "float dir=gain[i];":"""
     float ng=0.f,nk=0.f;
     for(int t=0;t<S;++t){ng+=gain[t]*gain[t];nk+=feature[t]*feature[t];}
     ng=sqrtf(ng);nk=sqrtf(nk);
     const float floor=j>=begin[life]?p[20]:0.f;
     const float required=floor*nk;
     float dir=gain[i];
     if((int)p[19]==0){
       if(ng<required)dir=gain[i]*required/fmaxf(ng,1.e-10f);
     }else if((int)p[19]==1){
       dir=feature[i]*fmaxf(ng,required)/fmaxf(nk,1.e-10f);
     }else if((int)p[19]==2){
       // Only the added part points along raw activity. Historical direction
       // is intact in all updates already larger than the floor.
       if(ng<required)dir=feature[i]*required/fmaxf(nk,1.e-10f);
     }else if((int)p[19]==3){
       dir=gain[i]*required/fmaxf(ng,1.e-10f);
     }
     """,
    }
    for old,new in edits.items():
        if raw.count(old)!=1:raise ValueError("The sealed organism changed: "+old)
        raw=raw.replace(old,new)
    src=f"#define N {n}\n#define H {h}\n#define S {s}\n#define CAP {cap}\n#define G 21\n"+raw
    return compile_kernels(src,("play",),device=torch.device("cuda:0"),options=("--fmad=false",))["play"]


class Cohort(ordinary.Cohort):
    def __init__(self,stream,front,route,motion):
        super().__init__(stream,front,route,motion)
        self.code_floor=compiled(front["matrix"].shape[0],route["first"][1].size,route["slots"],front["capacity"])
    def rates(self,body,settings,*,begin=None,detail=False):
        f=np.atleast_2d(settings)
        pop=np.column_stack((np.broadcast_to(body,(len(f),len(body))),f)).astype(np.float32)
        if begin is None:begin=np.zeros(self.n,np.int32)
        begin=gpu.tensor(np.asarray(begin,np.int32))
        outs=[]
        for j in range(0,len(pop),64):
            v=pop[j:j+64];sh=(len(v),self.n,self.tr)
            rows=torch.empty((len(v),self.n,6),device="cuda",dtype=torch.int32)
            actions=[torch.empty(sh,device="cuda",dtype=torch.int8) for _ in range(3)] if detail else [self.dummy8]*3
            g=torch.empty(sh,device="cuda",dtype=torch.float32) if detail else self.dummy32
            self.code_floor((self.n,len(v)),(32,),self.keys,self.address,*self.inputs,gpu.tensor(v),*actions,g,rows,begin,
                           self.n,self.tr,*self.checks,self.kind,int(detail),0)
            if detail:
                z=[a.cpu().numpy() for a in actions]
                outs.extend(gpu.Outcome(a,b,c) for a,b,c in zip(*z,strict=True))
            else:outs.append((rows.sum(1).float()*self.normalizer).cpu().numpy())
        return outs if detail else np.concatenate(outs)
