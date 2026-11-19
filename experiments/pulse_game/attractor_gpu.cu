// A block owns one continuous life. This kernel receives public cues and
// timing only: private fruit/answer preferences are absent from the ABI.
extern "C" __global__ void encode_lives(
    const float* __restrict__ inputs, const float* __restrict__ offsets,
    const int* __restrict__ sources,const float* __restrict__ edges,
    const float* __restrict__ projection,const float* __restrict__ parameters,
    const unsigned char* __restrict__ cue,const unsigned char* __restrict__ context,
    const unsigned char* __restrict__ issued,const unsigned char* __restrict__ query,
    const unsigned char* __restrict__ length,const unsigned char* __restrict__ raw_at,const unsigned char* __restrict__ intruder,
    const unsigned char* __restrict__ intrusion_tick,float* __restrict__ features,
    int* __restrict__ pulse_counts,int actors,int trials,int cut,int probe_kind) {
    const int life=blockIdx.x, genome=blockIdx.y, cell=threadIdx.x;
    const int unit=genome*NODES+cell;
    __shared__ float transmitted[NODES], feature[NODES];
    transmitted[cell]=0.f;
    feature[cell]=0.f;
    const float* g=parameters+genome*NPARAM;
    const float leak=g[0], syn_decay=g[1], read_decay=g[2], adapt_decay=g[3], adapt=g[4];
    const float bias=offsets[2*unit], threshold=offsets[2*unit+1];
    float membrane=0.f, nuisance=0.f, trace=0.f, syn=0.f, mean=0.12f;
    int neighbors[DEGREE];float strengths[DEGREE];
    #pragma unroll
    for(int d=0;d<DEGREE;++d){
        neighbors[d]=sources[unit*DEGREE+d];
        strengths[d]=edges[unit*DEGREE+d];
    }
    int pulses=0;
    __syncthreads();
    for(int trial=0;trial<trials;++trial){
        const int ix=life*trials+trial;
        const unsigned char c=cue[ix],ctx=context[ix],noise=intruder[ix];
        const float* sensory=inputs+unit*7;
        const float stimulus=sensory[ctx]+(raw_at[ix]==255?sensory[2+c]:0.f)+sensory[6];
        const float distractor=sensory[noise/4]+sensory[2+noise%4];
        for(int t=0;t<length[ix];++t){
            float recurrent=0.f;
            if(!cut){
                #pragma unroll
                for(int d=0;d<DEGREE;++d)
                    recurrent+=strengths[d]*transmitted[neighbors[d]];
            }
            float drive=bias+recurrent;
            if(t==issued[ix])drive+=stimulus;
            if(t==raw_at[ix])drive+=sensory[2+c];
            if(t==intrusion_tick[ix])drive+=distractor;
            membrane=fmaxf(-3.f,leak*membrane+drive);
            const float spike=(membrane>=threshold+adapt*nuisance)?1.f:0.f;
            if(spike>0.f)membrane=0.f;
            nuisance=adapt_decay*nuisance+(1.f-adapt_decay)*spike;
            trace=read_decay*trace+(1.f-read_decay)*spike;
            mean+=g[5]*(trace-mean);
            syn=syn_decay*syn+(1.f-syn_decay)*spike;
            pulses+=int(spike);
            __syncthreads();
            transmitted[cell]=syn;
            feature[cell]=(probe_kind==1?spike:trace)-mean;
            __syncthreads();
            if(t==query[ix] && cell<KEYS){
                float sum=0.f;
                const float* weights=projection+(genome*KEYS+cell)*NODES;
                #pragma unroll 4
                for(int k=0;k<NODES;++k)sum+=feature[k]*weights[k];
                features[((genome*actors+life)*trials+trial)*KEYS+cell]=sum;
            }
            // A second barrier is needed before another warp can overwrite
            // feature[] after the query. Non-query ticks already synchronize
            // before writing features for the next step.
        }
    }
    pulse_counts[(genome*actors+life)*NODES+cell]=pulses;
}

// Predict, take exactly one action, receive only its sign, update that
// action's synapses. Per-action sensory covariance is kept in shared memory.
extern "C" __global__ void score_lives(
    const float* __restrict__ raw,const unsigned char* __restrict__ goal,
    const float* __restrict__ tie,const float* __restrict__ lr,
    unsigned char* __restrict__ answers,int actors,int trials,int feedback,int geometry){
    const int b=blockIdx.x,life=b%actors,genome=b/actors,tid=threadIdx.x;
    __shared__ float cov[4][KEYS][KEYS],w[4][KEYS],x[KEYS],z[KEYS],value[4];
    __shared__ float invnorm, scale; __shared__ int choice;
    for(int i=tid;i<4*KEYS*KEYS;i+=blockDim.x)
        ((float*)cov)[i]=float((i%KEYS)==((i/KEYS)%KEYS))*4.f;
    for(int i=tid;i<4*KEYS;i+=blockDim.x)((float*)w)[i]=0.f;
    __syncthreads();
    for(int j=0;j<trials;++j){
        const float* key=raw+(b*trials+j)*KEYS;
        if(tid==0){
            float sq=.01f;
            for(int k=0;k<KEYS;++k)sq+=key[k]*key[k];
            invnorm=1.f/sqrtf(sq);
        }
        __syncthreads();
        if(tid<KEYS)x[tid]=key[tid]*invnorm;
        __syncthreads();
        if(tid<4){float v=0.f;for(int k=0;k<KEYS;++k)v+=w[tid][k]*x[k];value[tid]=v;}
        __syncthreads();
        if(tid==0){
            const float* jitter=tie+(life*trials+j)*4;
            int best=0;
            for(int a=1;a<4;++a)
                if(value[a]+1.e-5f*jitter[a]>value[best]+1.e-5f*jitter[best])best=a;
            choice=best;answers[b*trials+j]=best;
        }
        __syncthreads();
        if(tid<KEYS){float a=0.f;
            if(geometry){for(int k=0;k<KEYS;++k)a+=cov[choice][tid][k]*x[k];}
            else a=x[tid];
            z[tid]=a;
        }
        __syncthreads();
        if(tid==0){float n=geometry?1.f:.10f;for(int k=0;k<KEYS;++k)n+=x[k]*z[k];scale=1.f/n;}
        __syncthreads();
        if(feedback){
            if(geometry){
                for(int i=tid;i<KEYS*KEYS;i+=blockDim.x)
                    cov[choice][i/KEYS][i%KEYS]-=scale*z[i/KEYS]*z[i%KEYS];
            }
            if(tid<KEYS){
                const float r=(goal[life*trials+j]==choice)?1.f:-1.f;
                w[choice][tid]+=lr[genome]*(r-value[choice])*scale*z[tid];
            }
        }
        __syncthreads();
    }
}
