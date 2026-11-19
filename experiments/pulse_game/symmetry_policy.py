"""Use actual sampled game rewards to refine motor dynamics and the neural plasticity rule."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import symmetry_physics as physics


# Normalized mutation coordinates map to action dynamics and existing plastic parameters.
# Age bias is shared by both motor reading and synaptic error evaluation.
DIMS=("motor_tonic","motor_gain","motor_leak","motor_noise","motor_self","motor_opponent",
      "contact_confidence","age_cost","temperature","null","rate","retention","output_gain")
BASE=np.array([.39,1.6,.78,.13,1.08,1.28,0.,0.],np.float32)
STEPS=np.array([.19,.65,.10,.055,.20,.22,.22,.055,2.6,.07,.24,.035,.36],np.float32)
BOUNDS=np.array([[-.28,1.25],[.2,10.],[.20,.96],[.016,.65],[.05,2.2],[.1,2.6],[-.60,1.65],
                [0.,1.8],[2.,40.],[.08,.997],[.05,1.80],[.45,.9999],[.25,7.]],np.float32)


def decoded(params,coords,mode):
    coords=np.atleast_2d(np.asarray(coords,np.float32))
    scalars=BASE.tolist()+params["scalars"]
    if mode=="motor":
        raw=np.zeros((len(coords),len(DIMS)),np.float32)
        # The original actuator and a single confidence-dependent tonic term.
        raw[:,:7]=coords[:,:7]
    else:raw=coords[:,:len(DIMS)]
    out=np.clip(np.asarray(scalars)[None,:]+raw*STEPS,BOUNDS[:,0],BOUNDS[:,1]).astype(np.float32)
    old=[a.ravel() for pair in (params["rule"] or []) for a in pair]
    inherited=np.concatenate(old) if old else np.zeros(0,np.float32)
    if len(inherited):
        changes=coords[:,len(DIMS):] if mode=="full" else np.zeros((len(coords),len(inherited)),np.float32)
        inherited=inherited[None,:]+.11*changes
    else:inherited=np.empty((len(coords),0),np.float32)
    return np.ascontiguousarray(np.column_stack((out,inherited)),np.float32)


def dimensions(params,mode):
    count=sum(a.size for pair in (params["rule"] or []) for a in pair)
    return 7 if mode=="motor" else len(DIMS)+(count if mode=="full" else 0)


@lru_cache(maxsize=8)
def policy_source(k,cap,size):
    src=Path(physics.__file__).with_name("symmetry_physics_gpu.cu").read_text()
    act=src[src.index('extern "C" __global__ void act_and_taste'):]
    edits={
        "void act_and_taste(":"void policy_search(",
        "int pairs,int trials,int boundary,int rule,int change,float tonic,float motor_gain)":
          "int pairs,int trials,int boundary,int rule,int change,const float* population)",
        "const int mod=change<0?-change-1+variant:change;": """
 const int mod=change;
 const float* phenotype=population+variant*GENES;
 const float tonic=phenotype[0],motor_gain=phenotype[1],memory_age=phenotype[7];
 if(rule==1){r0=phenotype+13;b0=r0+48;r1=b0+12;b1=r1+24;}
 """,
        "const float temp=scalars[0],null=scalars[1],rate=scalars[2],hold=scalars[3],scale=scalars[4];":
          "const float temp=phenotype[8],null=phenotype[9],rate=phenotype[10],hold=phenotype[11],scale=phenotype[12];",
        "lane<items?score*temp:-1.e4f;": "lane<items?score*temp-memory_age*((used[branch]-1-lane+CAP)%CAP):-1.e4f;",
        "(active?tonic:-.85f)+motor_gain":"(active?tonic+phenotype[6]*fabsf(q0-q1):-.85f)+motor_gain",
        ".13f*jitter[event*2+s]":"phenotype[3]*jitter[event*2+s]",
        "+1.08f*(s==0?pulse0:pulse1)-1.28f*(s==0?pulse1:pulse0)":
          "+phenotype[4]*(s==0?pulse0:pulse1)-phenotype[5]*(s==0?pulse1:pulse0)",
        "fmaxf(-1.f,.78f*mv[branch][s]+drive)":"fmaxf(-1.f,phenotype[2]*mv[branch][s]+drive)",
    }
    for old,new in edits.items():
        if act.count(old)!=1:raise ValueError("Reference simulator changed: "+old[:60])
        act=act.replace(old,new)
    return f"#define CAP {cap}\n#define K {k}\n#define GENES {size}\n"+act


@lru_cache(maxsize=8)
def program(k,cap,size):
    return compile_kernels(policy_source(k,cap,size),("policy_search",),device=torch.device("cuda:0"),options=("--fmad=false",))


class Cohort:
    def __init__(self,params,w):
        self.world=w
        self.params=params
        n,tr=w.menu.body.ctx.shape
        self.shape=(n,tr)
        self.ports=physics.Physical(w)
        if "ground" not in params:raise ValueError("Use the compact grounded encoder")
        self.query=np.flatnonzero(w.query[0])
        k=params["latent"];hidden=4
        device=torch.device("cuda")
        enc=physics.program(params["matrix"].shape[0],hidden,k,params["capacity"])
        self.features=torch.empty((n,tr,16,2,2,k),device=device)
        empty=physics.tensor(np.zeros(1,np.float32));self.empty=empty
        theta=[physics.tensor(v) for v in (params["matrix"],params["sensors"],*params["ground"])]
        enc["smell"]((n,2),(64,),*self.ports.senses,*theta,*([empty]*6),physics.tensor(params["transform"]),
                      self.features,tr,1)
        self.rtype={"delta":0,"learned":1,"fixed":2}[params["change"]]
        self.qdevice=torch.as_tensor(self.query,device=device)
        self.input_sha=None

    def run(self,vector,mode,*,individual=False,intervention=0,batch=64):
        params=self.params;pop=decoded(params,vector,mode)
        if not individual and intervention==0:
            from . import physical_counts
            source=policy_source(params["latent"],params["capacity"],pop.shape[1])
            return physical_counts.score(self,pop,physical_counts.warp_normalize(source))
        kern=program(params["latent"],params["capacity"],pop.shape[1])["policy_search"]
        return self.run_decoded(pop,kern,individual=individual,intervention=intervention,batch=batch)

    def run_decoded(self,pop,kern,*,individual=False,intervention=0,batch=64):
        n,tr=self.shape
        rows=[]
        for j in range(0,len(pop),batch):
            current=pop[j:j+batch]
            shape=(len(current),n,tr)
            side,tick,sign=[torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)]
            gain=torch.empty(shape,device="cuda",dtype=torch.float32)
            kern((n//2,len(current)),(64,),self.features,*self.ports.body,
                 *([self.empty]*5),side,tick,sign,gain,n//2,tr,self.world.exposed,self.rtype,
                 int(intervention),physics.tensor(current))
            if individual:
                rows.extend((physics.Outcome(x,y,z),g) for x,y,z,g in zip(side.cpu().numpy(),tick.cpu().numpy(),
                             sign.cpu().numpy(),gain.cpu().numpy(),strict=True));continue
            # Positive taste requires physical contact. The optimizer only observes actually
            # experienced signs, including during each free-choice probe.
            chosen=sign.index_select(2,self.qdevice)
            first=chosen[:,:,0]>0
            both=first[:,:n//2]&first[:,n//2:]
            all_good=chosen>0
            contact=chosen!=0
            rows.append(torch.stack((first.float().mean(1),both.float().mean(1),
                        all_good.float().mean((1,2)),contact.float().mean((1,2))),1).cpu().numpy())
        return rows if individual else np.concatenate(rows)

def materialize(params,vector,mode):
    """Standalone NumPy phenotype for an independent native physics reproduction."""
    p=dict(params)
    v=decoded(p,np.atleast_2d(vector),mode)[0]
    p["motion"]=v[:7].copy();p["age_cost"]=float(v[7])
    p["scalars"]=list(map(float,v[8:13]))
    if p["change"]=="learned":
        p["rule"]=[(v[13:61].reshape(4,12).copy(),v[61:73].copy()),(v[73:97].reshape(12,2).copy(),v[97:99].copy())]
    return p


