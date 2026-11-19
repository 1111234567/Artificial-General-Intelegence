"""Fuse terminal physical reward counts while retaining the detailed independent forward simulator."""
from functools import lru_cache
import torch
import numpy as np
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import symmetry_physics as gpu


@lru_cache(maxsize=12)
def compiled(original):
    edits={
      "void policy_search(":"void policy_counts(",
      "signed char* side_out,signed char* tick_out,signed char* signed_out,float* gains,":
        "int* __restrict__ results,const unsigned char* __restrict__ test_mask,",
      "__shared__ float mv[2][2]":"__shared__ int hits[2],visited[2],first_right[2],nprobes[2]; __shared__ float mv[2][2]",
      "if(lane==0){position[branch]=velocity[branch]=0.f;":
        "if(lane==0){if(j==0){hits[branch]=0;visited[branch]=0;first_right[branch]=0;nprobes[branch]=0;} position[branch]=velocity[branch]=0.f;",
      "if(lane==0)gains[oi]=gain[branch];":"",
      """if(lane==0){side_out[oi]=last_side[branch];tick_out[oi]=touch[branch];
   signed_out[oi]=last_side[branch]>=0?tastes[ix*2+last_side[branch]]:0;}
  __syncthreads();
 }
}""":"""if(lane==0 && test_mask[j]){
    const int dest=last_side[branch];const bool reached=dest>=0;
    const bool good=reached && tastes[ix*2+dest]>0;
    if(nprobes[branch]==0)first_right[branch]=int(good);
    ++nprobes[branch];hits[branch]+=int(good);visited[branch]+=int(reached);
  }
  __syncthreads();
 }
 if(branch==0 && lane==0){
   int* out=results+4*(variant*pairs+pair);
   out[0]=first_right[0]+first_right[1];out[1]=first_right[0]*first_right[1];
   out[2]=hits[0]+hits[1];out[3]=visited[0]+visited[1];
 }
}"""
    }
    for old,new in edits.items():
        if original.count(old)!=1:raise ValueError("Physical reference source changed: "+old[:100])
        original=original.replace(old,new)
    return compile_kernels(original,("policy_counts",),device=torch.device("cuda:0"),options=("--fmad=false",))["policy_counts"]


def score(cohort,phenotypes,source,*,block_candidates=2048):
    n,tr=cohort.shape
    checks=np.asarray(cohort.world.query[0],np.uint8)
    if not np.array_equal(cohort.world.query,np.broadcast_to(checks,cohort.world.query.shape)):
        raise ValueError("Free-choice times differ across life rows")
    mask=gpu.tensor(checks);p=n//2
    normalizer=torch.tensor([1/n,1/p,1/(n*int(checks.sum())),1/(n*int(checks.sum()))],device="cuda",dtype=torch.float32)
    run=compiled(source)
    out=[]
    for i in range(0,len(phenotypes),block_candidates):
        gene=phenotypes[i:i+block_candidates]
        counts=torch.empty((len(gene),p,4),device="cuda",dtype=torch.int32)
        run((p,len(gene)),(64,),cohort.features,*cohort.ports.body,*([cohort.empty]*5),counts,mask,
            p,tr,cohort.world.exposed,cohort.rtype,0,gpu.tensor(gene))
        out.append((counts.sum(1).float()*normalizer).cpu().numpy())
    return np.concatenate(out)


def warp_normalize(source):
    """The two counterfactual lives occupy disjoint warps throughout a physical tick."""
    old="""if(lane<CAP){
    for(int s=0;s<2;++s){float score=0.f;const float* query=keys+((event*2+s)*2)*K;
     for(int d=0;d<K;++d)score+=query[d]*bank[branch][lane][d];
     similarity[branch][s][lane]=lane<items?score*temp-memory_age*((used[branch]-1-lane+CAP)%CAP):-1.e4f;
    }
   }
   __syncthreads();
   if(lane<2){
    float max_score=temp*null;
    for(int m=0;m<items;++m)max_score=fmaxf(max_score,similarity[branch][lane][m]);
    float absent=expf(temp*null-max_score),total=absent,num=0.f;
    for(int m=0;m<items;++m){float w=expf(similarity[branch][lane][m]-max_score);total+=w;num+=w*sign[branch][m];}
    recalled[branch][lane]=num/total;certainty[branch][lane]=1.f-absent/total;
   }"""
    new="""float absences[2];
   for(int s=0;s<2;++s){
    float raw=-1.e4f;
    if(lane<items){
     const float* query=keys+((event*2+s)*2)*K;
     float dot=0.f;for(int d=0;d<K;++d)dot+=query[d]*bank[branch][lane][d];
     raw=dot*temp-memory_age*((used[branch]-1-lane+CAP)%CAP);
    }
    float mx=fmaxf(raw,temp*null);
    #pragma unroll
    for(int stride=16;stride>=1;stride/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,stride));
    mx=__shfl_sync(0xffffffff,mx,0);
    absences[s]=expf(temp*null-mx);
    if(lane<items)similarity[branch][s][lane]=expf(raw-mx);
   }
   __syncwarp();
   if(lane<2){
    float total=absences[lane],num=0.f;
    // Keep the original ascending-slot order of all floating-point sums.
    for(int m=0;m<items;++m){float w=similarity[branch][lane][m];total+=w;num+=w*sign[branch][m];}
    recalled[branch][lane]=num/total;certainty[branch][lane]=1.f-absences[lane]/total;
   }"""
    if source.count(old)!=1:raise ValueError("Reference normalization no longer matches")
    source=source.replace(old,new)
    # A cross-warp intervention can take place only at the boundary before
    # a trial. Leave that barrier and the terminal pair count barrier intact.
    first=source.index("for(int t=0;t<16;++t)")
    end=source.index("if(lane==0){side_out",first)
    return source[:first]+source[first:end].replace("__syncthreads();","__syncwarp();")+source[end:]
