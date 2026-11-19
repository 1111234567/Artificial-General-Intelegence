"""Let repeated, directionally consistent local errors regulate future plasticity."""
from functools import lru_cache
import numpy as np
import torch
from . import longrun_plastic as previous,unified_physical as simulation,symmetry_physics as gpu
from experiments.undirected._cuda_nvrtc import compile_kernels


@lru_cache(maxsize=8)
def kernel(n,h,s,cap):
    ref=previous.source(n,h,s,cap)
    edits={
      "#define G 24":"#define G 29",
      "int* summary,int lives":"int* summary,const int* begins,const float* schedule,float* log_release,int force_schedule,int track,int lives",
      "__shared__ float unit[S],release,unit_dot;":"__shared__ float recent[S],unit[S],release,unit_dot;",
      "if(i<S){master[i]=0.f;":"if(i<S){recent[i]=0.f;master[i]=0.f;",
      "if(i==0){pos=0.f;vel=0.f;":"if(i==0){if(track)log_release[out]=0.f;pos=0.f;vel=0.f;",
      "release=(sq>1.e-12f?amount:0.f);":"""
      float prediction=0.f,error,align=0.f,rsq=0.f,prior_projection=0.f;
      for(int k=0;k<S;++k){
       prediction+=master[k]*feature[k];align+=recent[k]*feature[k];
       rsq+=recent[k]*recent[k];prior_projection+=recent[k]*codes[(e-1)*S+k];
      }
      error=(float)taste[ix*2+choice]-prediction;
      if(mode>=6){
       float agreement=error*align/sqrtf(.04f+sq);
       float activation=p[21]+p[22]*fabsf(error)+p[23]*fabsf(saved_r)+p[25]*agreement+p[26]*sqrtf(rsq);
       amount=p[27]+p[20]/(1.f+expf(-activation));
      }
      if(force_schedule)amount=schedule[ix];
      release=(sq>1.e-12f && j>=begins[life]?amount:0.f);
      if(track)log_release[out]=release;
      float hold=p[24];
      for(int k=0;k<S;++k){
       float grad=fminf(2.f,fmaxf(-2.f,error))*feature[k];
       if(p[28]>.5f)recent[k]+=(1.f-hold)*(grad-prior_projection*codes[(e-1)*S+k]);
       else recent[k]=hold*recent[k]+(1.f-hold)*grad;
      }""",
      "mode==2 || mode==4?z[k]:feature[k]":"mode==2 || mode==4 || mode==7?z[k]:feature[k]",
    }
    for old,new in edits.items():
        if ref.count(old)!=1:raise ValueError("Checked plastic program changed: "+old[:80])
        ref=ref.replace(old,new)
    ref=ref.replace("(int)p[19]==5)","(int)p[19]==5 || (int)p[19]==8)")
    return compile_kernels(ref,("play",),device=torch.device("cuda:0"),options=("--fmad=false",))["play"]


class Cohort(simulation.Cohort):
    def __init__(self,ep,f,r,m):
        super().__init__(ep,f,r,m);self.fn=kernel(f["matrix"].shape[0],r["first"][1].size,r["slots"],f["capacity"])
    def gates(self,body,pars,*,detail=False,track=False,begin=None,schedule=None,batch=48):
        p=np.atleast_2d(pars).astype(np.float32);allv=np.column_stack((np.broadcast_to(body,(len(p),19)),p)).astype(np.float32)
        if begin is None:begin=np.zeros(self.n,np.int32)
        begin=gpu.tensor(np.asarray(begin,np.int32))
        provided=schedule is not None
        feeder=gpu.tensor(np.asarray(schedule,np.float32)) if provided else self.dummy32
        result=[]
        for j in range(0,len(allv),batch):
            v=allv[j:j+batch];shape=(len(v),self.n,self.tr)
            summ=torch.empty((len(v),self.n,6),device="cuda",dtype=torch.int32)
            obs=[torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)] if detail else [self.dummy8]*3
            g=torch.empty(shape,device="cuda") if detail else self.dummy32
            releases=torch.empty(shape,device="cuda") if track else self.dummy32
            self.fn((self.n,len(v)),(32,),self.keys,self.address,*self.inputs,gpu.tensor(v),*obs,g,summ,begin,feeder,releases,
                    int(provided),int(track),self.n,self.tr,*self.checks,self.kind,int(detail),0)
            if detail:
                side,tick,sign=[x.cpu().numpy() for x in obs]
                amounts=releases.cpu().numpy() if track else [None]*len(v)
                result.extend((gpu.Outcome(a,b,c),d) for a,b,c,d in zip(side,tick,sign,amounts,strict=True))
            else:result.append((summ.sum(1).float()*self.normalizer).cpu().numpy())
        return result if detail else np.concatenate(result)


def proposals(seed=619413,n=96):
    rng=np.random.default_rng(seed);out={}
    for kind,mode in (("constant",2),("instant",4),("coherent",7)):
        options=[]
        for i in range(n):
            gain=0. if i==0 else float(np.exp(rng.uniform(np.log(.0007),np.log(2.8))))
            bias=float(rng.uniform(-5.5,1.5))
            err=float(rng.uniform(-1.,5.))
            recall=float(rng.uniform(-2.,4.))
            hold=float(rng.uniform(.55,.99));align=float(rng.uniform(-8.,95.));strength=float(rng.uniform(-5.,30.))
            floor=float(rng.choice([0.,0.,.0003,.001,.004,.012]))
            local=float(rng.integers(2))
            options.append([mode,gain,bias,err,recall,hold,align,strength,floor,local])
        if kind!="constant":
            # Include literal constant-gate fits inside the larger model class.
            for i,a in enumerate((0.,.002,.008,.032,.128,.512,1.6)):
                options[i]=[mode,2*a,0.,0.,0.,.82,0.,0.,0.,0.]
        out[kind]=np.asarray(options,np.float32)
    return out
