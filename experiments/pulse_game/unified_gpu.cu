// Public pulses only. Both physical hemispheres retain their state through every trial.
extern "C" __global__ void prepare(const float* odor,const float* habitat,
 const unsigned char* ot,const unsigned char* ct,const float* A,const float* B,
 const float* d,const float* db,const float* transform,const float* cw0,const float* cb0,
 const float* cw1,const float* cb1,float* keys,float* codes,int trials,int route,float shared){
 const int i=threadIdx.x,life=blockIdx.x,side=i/N,unit=i%N;
 __shared__ float old[2][N],h[2][N],read[2][4],write[2][4],mid[H],loc[S],last[2];
 if(i<2*N)old[side][unit]=0.f;if(i<2)last[i]=0.f;
 __syncthreads();
 for(int j=0;j<trials;++j){const int ix=life*trials+j;
  for(int t=0;t<16;++t){const int ev=ix*16+t;
   if(i<2*N){
    float v=0.f,s=0.f,c=0.f;
    for(int m=0;m<N;++m)v+=old[side][m]*A[m*N+unit];
    if(t==ot[ix])for(int m=0;m<4;++m)s+=odor[(ix*2+side)*4+m]*B[m*N+unit];
    if(t==ct[ix])for(int m=0;m<2;++m)c+=habitat[2*ix+m]*B[(m+4)*N+unit];
    h[side][unit]=v+s+c;
   }
   if(i<2 && t==ct[ix])last[i]=habitat[2*ix+i];
   __syncthreads();
   if(i<8){const int s=i/4,k=i%4;float v=0.f;
    for(int m=0;m<N;++m)v+=h[s][m]*d[m*4+k];read[s][k]=v+db[k];}
   if(i<H && route==0){float v=0.f;
    for(int m=0;m<N;++m)v+=(.5f*(h[0][m]+h[1][m]))*cw0[m*H+i];mid[i]=tanhf(v+cb0[i]);}
   __syncthreads();
   if(i<8){const int s=i/4,k=i%4;float v=0.f;for(int m=0;m<4;++m)v+=read[s][m]*transform[m*4+k];write[s][k]=v;}
   if(i<S){
    if(route==0){float v=0.f;for(int m=0;m<H;++m)v+=mid[m]*cw1[m*S+i];loc[i]=v+cb1[i];}
    if(route==1)loc[i]=1.f;
    if(route==2 || route==3)loc[i]=i<2?last[i]:shared;
   }
   __syncthreads();
   if(i<8){const int s=i/4,k=i%4;float r=0.f,w=0.f;
    for(int m=0;m<4;++m){r+=read[s][m]*read[s][m];w+=write[s][m]*write[s][m];}
    float* out=keys+(ev*2+s)*8;
    out[k]=read[s][k]/fmaxf(sqrtf(r),1.e-12f);out[4+k]=write[s][k]/fmaxf(sqrtf(w),1.e-12f);
   }
   if(i<S){float norm=0.f;for(int m=0;m<S;++m)norm+=loc[m]*loc[m];codes[ev*S+i]=loc[i]/fmaxf(sqrtf(norm),1.e-7f);}
   if(i<2*N)old[side][unit]=h[side][unit];
   __syncthreads();
  }
 }
}

// One life per warp. All private tastes are read strictly at the food reached
// by the body. Silent test rewards enter external scoring only.
extern "C" __global__ void play(const float* keys,const float* codes,const float* dist,
 const float* wind,const signed char* taste,const unsigned char* tests,const float* jitter,
 const float* genes,signed char* sides,signed char* ticks,signed char* reward,float* gainlog,
 int* summary,int lives,int trials,int first_probe,int last_b,int ret_a,int rule,int detail,int variant){
 const int life=blockIdx.x,pop=blockIdx.y,i=threadIdx.x,base=life*trials;
 const float* p=genes+pop*G;
 __shared__ float bank[CAP][4],signs[CAP],master[S],cov[S][S],feature[S],z[S],gain[S];
 __shared__ float saved_r,saved_err,denom,similarity[2][CAP],recalled[2],uncertain[2],mv[2],motor[2],previous_q;
 __shared__ float pos,vel,mod;
 __shared__ int count,choice,time_at,pending,first_good,late_good,return_good,all_good,contacts;
 if(i<CAP){signs[i]=0.f;for(int k=0;k<4;++k)bank[i][k]=0.f;}
 if(i<S){master[i]=0.f;for(int k=0;k<S;++k)cov[i][k]=i==k?1.f/p[12]:0.f;}
 if(i<2){mv[i]=0.f;motor[i]=0.f;}
 if(i==0){count=0;first_good=0;late_good=0;return_good=0;all_good=0;contacts=0;}
 __syncwarp();
 for(int j=0;j<trials;++j){
  const int ix=base+j,out=(pop*lives+life)*trials+j;
  const bool hide=tests[ix]!=0;
  if(i==0){pos=0.f;vel=0.f;choice=-1;time_at=-1;pending=0;}
  __syncwarp();
  for(int t=0;t<16;++t){
   const int e=ix*16+t;
   const bool has=pending && !hide && variant!=1;
   if(has){
    if(i<S)feature[i]=codes[(e-1)*S+i]*saved_r;
    __syncwarp();
    if(i<S){
     if(rule==1){float u=0.f;for(int k=0;k<S;++k)u+=cov[i][k]*feature[k];z[i]=u;}
     else z[i]=feature[i];
    }
    __syncwarp();
    if(i==0){
     float proj=0.f,sz=0.f;for(int k=0;k<S;++k){proj+=master[k]*feature[k];sz+=z[k]*feature[k];}
     denom=rule==1?1.f+sz:1.f;float r=float(taste[ix*2+choice]);
     saved_err=(p[13]*r+p[14]*proj+p[15])*(1.f+p[16]*(fabsf(saved_r)-.5f));
    }
    __syncwarp();
    if(i<S)gain[i]=z[i]/denom;
    __syncwarp();
    if(i<S){
     if(rule==1)for(int k=0;k<S;++k)cov[i][k]-=gain[i]*z[k];
     float dir=gain[i];
     if(variant==2)dir=gain[(i+S-1)%S];
     if(variant==3){float a=0.f,b=0.f;for(int k=0;k<S;++k){a+=gain[k]*gain[k];b+=feature[k]*feature[k];}
      dir=feature[i]*sqrtf(a)/fmaxf(1.e-8f,sqrtf(b));}
     master[i]=p[17]*master[i]+(rule==1?p[18]:p[10])*saved_err*dir;
    }
    const int slot=count%CAP;
    if(i<4)bank[slot][i]=keys[((e-1)*2+choice)*8+4+i];
    if(i==0)signs[slot]=float(taste[ix*2+choice]);
   }
   __syncwarp();
   if(i==0){if(has)++count;pending=0;
    float u=0.f;for(int k=0;k<S;++k)u+=master[k]*codes[e*S+k];mod=tanhf(p[11]*u);
   }
   __syncwarp();
   const int items=min(count,CAP);float absence[2];
   for(int side=0;side<2;++side){
    float raw=-1.e4f;if(i<items){float score=0.f;
     for(int k=0;k<4;++k)score+=keys[(e*2+side)*8+k]*bank[i][k];
     raw=score*p[7]-p[9]*((count-1-i+CAP)%CAP);
    }
    float mx=fmaxf(raw,p[7]*p[8]);
    #pragma unroll
    for(int stride=16;stride>=1;stride/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,stride));
    mx=__shfl_sync(0xffffffff,mx,0);absence[side]=expf(p[7]*p[8]-mx);
    if(i<items)similarity[side][i]=expf(raw-mx);
   }
   __syncwarp();
   if(i<2){float total=absence[i],sum=0.f;for(int m=0;m<items;++m){total+=similarity[i][m];sum+=similarity[i][m]*signs[m];}
    recalled[i]=sum/total;uncertain[i]=1.f-absence[i]/total;}
   __syncwarp();
   if(i==0){
    bool active=time_at<0 && t<15;float q[2]={mod*recalled[0],mod*recalled[1]};
    float old[2]={motor[0],motor[1]};
    for(int m=0;m<2;++m){
     float delta=q[m]-q[1-m];
     float drv=(active?p[0]+p[6]*fabsf(delta):-.85f)+p[1]*tanhf(delta)
       +p[3]*jitter[e*2+m]+p[4]*old[m]-p[5]*old[1-m];
     float v=fmaxf(-1.f,p[2]*mv[m]+drv);
     motor[m]=v>=1.f?1.f:0.f;mv[m]=v*(1.f-motor[m]);
    }
    if(active){vel=.65f*vel+.15f*(motor[1]-motor[0])+wind[ix*15+t];
     float x=pos+vel;int side=x<=-dist[ix*2]?0:(x>=dist[ix*2+1]?1:-1);
     pos=fminf(fmaxf(x,-dist[ix*2]),dist[ix*2+1]);
     if(side>=0){vel=0.f;choice=side;time_at=t;pending=1;saved_r=recalled[side];}
    }
    if(detail && t==5)gainlog[out]=mod;
   }
   __syncwarp();
  }
  if(i==0){
   signed char y=choice>=0?taste[ix*2+choice]:0;
   if(detail){sides[out]=choice;ticks[out]=time_at;reward[out]=y;}
   if(hide){all_good+=int(y>0);contacts+=int(choice>=0);}
   if(j==first_probe)first_good=int(y>0);if(j==last_b)late_good=int(y>0);if(j==ret_a)return_good=int(y>0);
  }
  __syncwarp();
 }
 if(i==0){int* a=summary+(pop*lives+life)*6;
  a[0]=first_good;a[1]=late_good;a[2]=return_good;a[3]=late_good*return_good;a[4]=all_good;a[5]=contacts;}
}
