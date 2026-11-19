"""Consolidate similar actual tastes at an existing address under a fixed storage budget."""
from functools import lru_cache
import numpy as np
import torch
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import symmetry_policy as base,symmetry_memory_deploy as native


# New coefficients are shared across slots. A positive merge prediction directs
# the actual receipt to an existing location. It receives only local public signals.
DEFAULT=np.array([.92,0.,0.,.45,.25,0.,0.],np.float32)
STEP=np.array([.07,.07,.04,.20,.16,.45,.45],np.float32)
LOW=np.array([.57,-.45,-.3,.04,.0,-3.,-3.],np.float32)
HIGH=np.array([1.02,.45,.3,1.,1.,3.,3.],np.float32)


def traits(coords):
    return np.asarray(np.clip(DEFAULT+np.atleast_2d(coords)*STEP,LOW,HIGH),np.float32)


@lru_cache(maxsize=8)
def kernel(size,cap,rule):
    offset=size-7
    s=base.policy_source(4,cap,size)
    old="""if(has){
    float y=float(tastes[ix*2+dest]);int slot=used[branch]%CAP;
    if(lane<K)bank[branch][slot][lane]=keys[(((event-1)*2+dest)*2+1)*K+lane];
    if(lane==0){"""
    new="""
   if(has && lane==0){
    float y=float(tastes[ix*2+dest]);int valid=min(used[branch],CAP);
    const float* wr=keys+(((event-1)*2+dest)*2+1)*K;
    const float* coeff=phenotype+WRITE_AT;
    int best=-1;float highest=-2.f;
    for(int m=0;m<valid;++m){
     float sim=0.f;for(int z=0;z<K;++z)sim+=wr[MODE==6?(z+1)%K:z]*bank[branch][m][z];
     if(sim>highest){best=m;highest=sim;}
    }
    float threshold=coeff[0];
    if(best>=0)threshold+=coeff[1]*(1.f-y*sign[branch][best]) + coeff[2]*fminf(1.f,support[branch][best]/4.f);
    const bool similar=(MODE>0 && best>=0 && highest>threshold);
    const bool merge=similar && MODE!=3;
    const float cached=(MODE==3 && similar)?sign[branch][best]+coeff[3]*(y-sign[branch][best]):y;
    int slot=merge?best:used[branch]%CAP;
    if(!merge && MODE>=2 && valid==CAP){
     float vmin=1.e25f;
     for(int m=0;m<CAP;++m){
      float age=float(ordinal[branch]-touched[branch][m]);
      float priority=coeff[5]*log1pf(float(support[branch][m]))-coeff[6]*log1pf(age);
      if(priority<vmin){vmin=priority;slot=m;}
     }
    }
    if(merge){
     float v[K],norm=0.f;
     for(int d=0;d<K;++d){v[d]=bank[branch][slot][d]+coeff[4]*(wr[d]-bank[branch][slot][d]);norm+=v[d]*v[d];}
     for(int d=0;d<K;++d)bank[branch][slot][d]=v[d]/fmaxf(sqrtf(norm),1.e-12f);
     sign[branch][slot]+=(MODE==4?1.f:(MODE==5?0.f:coeff[3]))*(y-sign[branch][slot]);++support[branch][slot];
    }else{
     for(int d=0;d<K;++d)bank[branch][slot][d]=wr[d];
     sign[branch][slot]=cached;support[branch][slot]=1;++used[branch];
    }
    touched[branch][slot]=ordinal[branch]++;
    {"""
    edits={old:new,
           "sign[branch][slot]=y;\n    }":"\n    }",
           "if(lane==0){if(has)++used[branch];pending[branch]=0;}":"if(lane==0)pending[branch]=0;",
           "__shared__ int used[2],last_side":"__shared__ int support[2][CAP],touched[2][CAP],ordinal[2]; __shared__ int used[2],last_side",
           "if(lane==0){used[branch]=0;":"if(lane==0){ordinal[branch]=0;used[branch]=0;",
           "if(lane<CAP)sign[branch][lane]=0.f;":"if(lane<CAP){sign[branch][lane]=0.f;support[branch][lane]=0;touched[branch][lane]=0;}",
           "memory_age*((used[branch]-1-lane+CAP)%CAP)":"memory_age*(ordinal[branch]-1-touched[branch][lane])"}
    for old,new in edits.items():
        if s.count(old)!=1:raise ValueError("Sealed policy changed: "+old[:90])
        s=s.replace(old,new)
    s=f"#define MODE {rule}\n#define WRITE_AT {offset}\n"+s
    return compile_kernels(s,("policy_search",),device=torch.device("cuda:0"),options=("--fmad=false",))["policy_search"]


class Cohort(base.Cohort):
    def replacements(self,body,genes,*,mode=2,individual=False):
        props=traits(np.atleast_2d(genes))
        pop=np.column_stack((np.broadcast_to(body,(len(props),len(body))),props)).astype(np.float32)
        code=kernel(pop.shape[1],self.params["capacity"],mode)
        return self.run_decoded(pop,code,individual=individual)


class Forager(native.Forager):
    def __init__(self,p,genes,seeds,mode):
        self.mod=mode;self.traits=traits(genes)[0]
        super().__init__(p,seeds)
    def reset(self,n):
        super().reset(n)
        self.num_seen=np.zeros(n,int);self.support=np.zeros((n,self.model["capacity"]),int)
        self.recency=np.zeros_like(self.support)
        self.merges=np.zeros(n,int)
        self.last_slot=np.full(n,-1,int);self.last_merged=np.zeros(n,bool)
    def store(self,ids,side,sign):
        cap=self.model["capacity"];coeff=self.traits
        for i,sd,y in zip(ids,side,sign,strict=True):
            fill=min(self.count[i],cap);x=self.previous_write[i,sd];bank=self.addresses[i]
            probe=np.roll(x,-1) if self.mod==6 else x
            best=int((bank[:fill]@probe).argmax()) if fill else -1
            threshold=coeff[0]
            if best>=0:threshold+=coeff[1]*(1.-y*self.signs[i,best])+coeff[2]*min(1.,self.support[i,best]/4.)
            similar=self.mod>0 and best>=0 and bank[best]@probe>threshold
            merge=similar and self.mod!=3
            cached=self.signs[i,best]+coeff[3]*(y-self.signs[i,best]) if similar and self.mod==3 else y
            if merge:
                k=best;v=bank[k]+coeff[4]*(x-bank[k]);bank[k]=v/max(np.linalg.norm(v),1e-12)
                alpha=1. if self.mod==4 else 0. if self.mod==5 else coeff[3]
                self.signs[i,k]+=alpha*(y-self.signs[i,k]);self.support[i,k]+=1;self.merges[i]+=1
            else:
                k=self.count[i]%cap
                if self.mod>=2 and fill==cap:
                    pr=coeff[5]*np.log1p(self.support[i])-coeff[6]*np.log1p(self.num_seen[i]-self.recency[i])
                    k=int(pr.argmin())
                bank[k]=x;self.signs[i,k]=cached;self.support[i,k]=1;self.count[i]+=1
            self.recency[i,k]=self.num_seen[i];self.num_seen[i]+=1
            self.last_slot[i]=k;self.last_merged[i]=merge
    def event_age(self):
        return self.num_seen[:,None]-1-self.recency


