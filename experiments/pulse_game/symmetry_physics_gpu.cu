// Independent forward simulation for physically sampled odor-rule learning.
// The sensory program never receives goal, taste, action or future-dependent tensors.
extern "C" __global__ void smell(
 const float* odor,const unsigned char* ctx,const unsigned char* ot,const unsigned char* ct,
 const float* graph,const float* inputs,const float* a,const float* ab,const float* b,const float* bb,
 const float* rq,const float* rb,const float* wk,const float* wb,const float* transform,
 float* keys,int trials,int grounded){
 const int life=blockIdx.x, side=blockIdx.y, i=threadIdx.x;
 __shared__ float h[N], state[N], hidden[WIDTH], read[K], write[K];
 if(i<N)h[i]=0.f;
 __syncthreads();
 for(int j=0;j<trials;++j){
  const int ix=life*trials+j;
  const float* x=odor+(ix*2+side)*4;
  for(int t=0;t<16;++t){
   if(i<N){
    float step=0.f, perception=0.f;
    for(int d=0;d<N;++d)step+=h[d]*graph[d*N+i];
    if(t==ot[ix])for(int d=0;d<4;++d)perception+=x[d]*inputs[d*N+i];
    if(t==ct[ix])perception+=inputs[(4+ctx[ix])*N+i];
    state[i]=step+perception;
   }
   __syncthreads();
   if(grounded){
    if(i<K){float v=0.f;for(int d=0;d<N;++d)v+=state[d]*a[d*K+i];read[i]=v+ab[i];}
    __syncthreads();
    if(i<K){float v=0.f;for(int d=0;d<K;++d)v+=read[d]*transform[d*K+i];write[i]=v;}
   }else{
    if(i<WIDTH){float va=0.f,vb=0.f;
     for(int d=0;d<N;++d){va+=state[d]*a[d*WIDTH+i];vb+=state[d]*b[d*WIDTH+i];}
     hidden[i]=tanhf((va+ab[i])*(1.f+tanhf(vb+bb[i])));
    }
    __syncthreads();
    if(i<K){float va=0.f,vb=0.f;for(int d=0;d<WIDTH;++d){va+=hidden[d]*rq[d*K+i];vb+=hidden[d]*wk[d*K+i];}
     read[i]=va+rb[i];write[i]=vb+wb[i];
    }
   }
   __syncthreads();
   if(i<K){
    float r=0.f,w=0.f;for(int d=0;d<K;++d){r+=read[d]*read[d];w+=write[d]*write[d];}
    float* dst=keys+(((ix*16+t)*2+side)*2)*K;
    dst[i]=read[i]/fmaxf(sqrtf(r),1.e-12f);dst[K+i]=write[i]/fmaxf(sqrtf(w),1.e-12f);
   }
   if(i<N)h[i]=state[i];
   __syncthreads();
  }
 }
}

// Each block holds both independent lives in a counterfactual pair. The only
// use of private taste is the one sensor receipt after an actual threshold crossing.
extern "C" __global__ void act_and_taste(
 const float* keys,const float* distances,const float* wind,const signed char* tastes,const float* jitter,
 const float* scalars,const float* r0,const float* b0,const float* r1,const float* b1,
 signed char* side_out,signed char* tick_out,signed char* signed_out,float* gains,
 int pairs,int trials,int boundary,int rule,int change,float tonic,float motor_gain){
 const int pair=blockIdx.x, variant=blockIdx.y, branch=threadIdx.x/32,lane=threadIdx.x%32;
 const int life=pair+branch*pairs, actors=2*pairs;
 // Multiple intervention variants share the same observation features.
 const int mod=change<0?-change-1+variant:change;
 __shared__ float bank[2][CAP][K], sign[2][CAP], gain[2],tmpgain[2];
 __shared__ float similarity[2][2][CAP], recalled[2][2], certainty[2][2];
 __shared__ float mv[2][2],motor[2][2],previous_recall[2],previous_conf[2],position[2],velocity[2];
 __shared__ int used[2],last_side[2],touch[2],pending[2];
 const float temp=scalars[0],null=scalars[1],rate=scalars[2],hold=scalars[3],scale=scalars[4];
 if(lane==0){used[branch]=0;gain[branch]=0.f;mv[branch][0]=mv[branch][1]=0.f;motor[branch][0]=motor[branch][1]=0.f;}
 for(int i=lane;i<CAP*K;i+=32)((float*)bank[branch])[i]=0.f;
 if(lane<CAP)sign[branch][lane]=0.f;
 __syncthreads();
 for(int j=0;j<trials;++j){
  const int ix=life*trials+j, oi=variant*actors*trials+ix;
  if(lane==0){position[branch]=velocity[branch]=0.f;last_side[branch]=-1;touch[branch]=-1;pending[branch]=0;
   tmpgain[branch]=gain[branch];}
  __syncthreads();
  if(j==boundary){
   if(mod==1 && lane==0)gain[branch]=tmpgain[1-branch];
   if(mod==2 && lane==0)gain[branch]=0.f;
   if(mod==3){ // Swap all stored events, preserving the rest of each body.
    if(branch==0){
     for(int n=lane;n<CAP*K;n+=32){float* p=(float*)bank[0];float* q=(float*)bank[1];float z=p[n];p[n]=q[n];q[n]=z;}
     if(lane<CAP){float z=sign[0][lane];sign[0][lane]=sign[1][lane];sign[1][lane]=z;}
     if(lane==0){int z=used[0];used[0]=used[1];used[1]=z;}
    }
   }
  }
  __syncthreads();
  if(lane==0)gains[oi]=gain[branch];
  for(int t=0;t<16;++t){
   const int event=(ix*16+t);
   const int dest=last_side[branch];
   const int has=pending[branch] && mod!=4 && !(mod==5 && j==boundary);
   if(has){
    float y=float(tastes[ix*2+dest]);int slot=used[branch]%CAP;
    if(lane<K)bank[branch][slot][lane]=keys[(((event-1)*2+dest)*2+1)*K+lane];
    if(lane==0){
     float rr=previous_recall[branch];
     if(rule==0)gain[branch]=hold*gain[branch]+rate*rr*(y-gain[branch]*rr);
     else if(rule==1){
      float out0=b1[0],out1=b1[1];
      for(int m=0;m<12;++m){float z=tanhf(y*r0[m]+rr*r0[12+m]+y*rr*r0[24+m]+previous_conf[branch]*r0[36+m]+b0[m]);out0+=z*r1[2*m];out1+=z*r1[2*m+1];}
      gain[branch]=(.55f+.449f/(1.f+expf(-out0)))*gain[branch]+.9f*tanhf(out1);
     }
     sign[branch][slot]=y;
    }
   }
   __syncthreads();
   if(lane==0){if(has)++used[branch];pending[branch]=0;}
   __syncthreads();
   const int items=min(used[branch],CAP);
   if(lane<CAP){
    for(int s=0;s<2;++s){float score=0.f;const float* query=keys+((event*2+s)*2)*K;
     for(int d=0;d<K;++d)score+=query[d]*bank[branch][lane][d];
     similarity[branch][s][lane]=lane<items?score*temp:-1.e4f;
    }
   }
   __syncthreads();
   if(lane<2){
    float max_score=temp*null;
    for(int m=0;m<items;++m)max_score=fmaxf(max_score,similarity[branch][lane][m]);
    float absent=expf(temp*null-max_score),total=absent,num=0.f;
    for(int m=0;m<items;++m){float w=expf(similarity[branch][lane][m]-max_score);total+=w;num+=w*sign[branch][m];}
    recalled[branch][lane]=num/total;certainty[branch][lane]=1.f-absent/total;
   }
   __syncthreads();
   if(lane==0){
    const bool active=touch[branch]<0 && t<15;
    float direction=rule==2?1.f:tanhf(scale*gain[branch]);
    if(mod==6)direction=-direction;
    if(mod==7)direction=1.f;
    if(mod==8)direction=-1.f;
    float q0=recalled[branch][0]*direction,q1=recalled[branch][1]*direction;
    const float pulse0=motor[branch][0],pulse1=motor[branch][1];
    for(int s=0;s<2;++s){
     float drive=(active?tonic:-.85f)+motor_gain*tanhf(s==0?q0-q1:q1-q0)+.13f*jitter[event*2+s]
                 +1.08f*(s==0?pulse0:pulse1)-1.28f*(s==0?pulse1:pulse0);
     float v=fmaxf(-1.f,.78f*mv[branch][s]+drive);
     motor[branch][s]=(v>=1.f?1.f:0.f);mv[branch][s]=v*(1.f-motor[branch][s]);
    }
    if(active){
     float v=.65f*velocity[branch]+.15f*(motor[branch][1]-motor[branch][0])+wind[ix*15+t];
     float pos=position[branch]+v;
     int pick=pos<=-distances[ix*2]?0:(pos>=distances[ix*2+1]?1:-1);
     position[branch]=fminf(fmaxf(pos,-distances[ix*2]),distances[ix*2+1]);velocity[branch]=v;
     if(pick>=0){
      velocity[branch]=0.f;last_side[branch]=pick;touch[branch]=t;pending[branch]=1;
      previous_recall[branch]=recalled[branch][pick];previous_conf[branch]=certainty[branch][pick];
     }
    }
   }
   __syncthreads();
  }
  if(lane==0){side_out[oi]=last_side[branch];tick_out[oi]=touch[branch];
   signed_out[oi]=last_side[branch]>=0?tastes[ix*2+last_side[branch]]:0;}
  __syncthreads();
 }
}
