"""Keep plasticity available over thousands of requests, globally or along a current addressed direction."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from . import unified_physical as earlier, symmetry_physics as gpu
from experiments.undirected._cuda_nvrtc import compile_kernels


@lru_cache(maxsize=8)
def source(n,h,s,cap):
    body=Path(earlier.__file__).with_name("unified_gpu.cu").read_text()
    edits={
       "__shared__ float saved_r,saved_err,denom,":"__shared__ float unit[S],release,unit_dot; __shared__ float saved_r,saved_err,denom,",
       "if(i<S)gain[i]=z[i]/denom;":"""
    if(i==0){
      float sq=0.f,uncert=0.f;
      for(int k=0;k<S;++k){sq+=feature[k]*feature[k];uncert+=z[k]*feature[k];}
      const int mode=(int)p[19];
      float amount=p[20];
      if(mode==3 || mode==4 || mode==5){
        float proj=0.f;for(int k=0;k<S;++k)proj+=master[k]*feature[k];
        const float y=(float)taste[ix*2+choice];
        amount=amount/(1.f+expf(-(p[21]+p[22]*fabsf(y-proj)+p[23]*fabsf(saved_r))));
      }
      release=(sq>1.e-12f?amount:0.f);
      float norm=0.f;for(int k=0;k<S;++k){float a=(mode==2 || mode==4?z[k]:feature[k]);unit[k]=a;norm+=a*a;}
      norm=fmaxf(sqrtf(norm),1.e-8f);unit_dot=0.f;
      for(int k=0;k<S;++k){unit[k]/=norm;unit_dot+=unit[k]*feature[k];}
    }
    __syncwarp();
    if(i<S){bool global=((int)p[19]==0 || (int)p[19]==5);
      z[i]+=release*(global?feature[i]/S:unit[i]*unit_dot);}
    __syncwarp();
    if(i==0){float size=0.f;for(int k=0;k<S;++k)size+=z[k]*feature[k];denom=1.f+size;}
    __syncwarp();
    if(i<S)gain[i]=z[i]/denom;""",
       "if(rule==1)for(int k=0;k<S;++k)cov[i][k]-=gain[i]*z[k];":"""
     if(rule==1){bool global=((int)p[19]==0 || (int)p[19]==5);
      for(int k=0;k<S;++k){float re=global?(i==k?1.f/S:0.f):unit[i]*unit[k];
       cov[i][k]+=release*re-gain[i]*z[k];}}
     """,
    }
    for old,new in edits.items():
        if body.count(old)!=1:raise ValueError("Sealed long-life physics changed")
        body=body.replace(old,new)
    return f"#define N {n}\n#define H {h}\n#define S {s}\n#define CAP {cap}\n#define G 24\n"+body


@lru_cache(maxsize=8)
def code(n,h,s,cap):
    return compile_kernels(source(n,h,s,cap),("play",),device=torch.device("cuda:0"),options=("--fmad=false",))["play"]


METHOD={"global":0,"local":1,"uncertainty_direction":2,"surprise_local":3,"surprise_uncertainty":4,"surprise_global":5}


class Cohort(earlier.Cohort):
    def __init__(self,episode,front,route,motion):
        super().__init__(episode,front,route,motion)
        self.long_code=code(front["matrix"].shape[0],route["first"][1].size,route["slots"],front["capacity"])

    def adjustments(self,body,parameters,*,detail=False,batch=64):
        all_settings=np.atleast_2d(parameters)
        heights=np.column_stack((np.broadcast_to(body,(len(all_settings),19)),all_settings)).astype(np.float32)
        result=[]
        for i in range(0,len(heights),batch):
            v=heights[i:i+batch];shape=(len(v),self.n,self.tr)
            stats=torch.empty((len(v),self.n,6),device="cuda",dtype=torch.int32)
            actions=[torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)] if detail else [self.dummy8]*3
            gain=torch.empty(shape,device="cuda") if detail else self.dummy32
            self.long_code((self.n,len(v)),(32,),self.keys,self.address,*self.inputs,gpu.tensor(v),*actions,gain,stats,
                           self.n,self.tr,*self.checks,self.kind,int(detail),0)
            if detail:
                a,b,c=(x.cpu().numpy() for x in actions)
                result.extend(gpu.Outcome(s,t,r) for s,t,r in zip(a,b,c,strict=True))
            else:result.append((stats.sum(1).float()*self.normalizer).cpu().numpy())
        return result if detail else np.concatenate(result)


def choices():
    vals=[]
    for kind,id_ in METHOD.items():
        if kind.startswith("surprise"):
            for scale in (.04,.16,.64,2.0):
                for bias,werror,wrecall in ((-2.2,1.,0.),(-3.5,2.3,0.),(-3.,1.6,2.5),(-1.,-.85,2.5),(-2.,0.,3.8)):
                    vals.append([id_,scale,bias,werror,wrecall])
        else:
            for amplitude in (0.,.001,.004,.016,.064,.256,1.,2.):
                vals.append([id_,amplitude,0.,0.,0.])
    return np.asarray(vals,np.float32)


