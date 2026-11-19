"""A continuous embodied circuit with locally addressed higher-order plastic synapses."""
import numpy as np
from . import mixture_world as food, world


class Forager:
    def __init__(self,perception,route,seeds,*,start,intervention="true",appearance=None,probes=None,motor=None):
        self.perception=perception;self.route=route;self.start=start
        self.change=intervention;self.appearance=appearance;self.probes=probes
        self.motion=np.asarray([.39,1.6,.78,.13,1.08,1.28,0.] if motor is None else motor)
        self.labels=None
        self.seeds=tuple(map(int,seeds))

    def reset(self,n):
        if n!=len(self.seeds):raise ValueError("One seeded actuator per life is required")
        p=self.perception;r=self.route
        self.n=n;self.round=-1;self.it=0
        self.rng=[np.random.default_rng(np.random.SeedSequence([s,745930])) for s in self.seeds]
        self.noise=np.empty((0,n,2),np.float32)
        self.motor=np.zeros((n,2),np.float32);self.mv=np.zeros_like(self.motor)
        self.h=np.zeros((n,2,p["matrix"].shape[0]),np.float32)
        self.g=np.zeros((n,r["slots"]),np.float32)
        cap=p["capacity"]
        self.mem=np.zeros((n,cap,4),np.float32);self.signed=np.zeros((n,cap),np.float32);self.count=np.zeros(n,np.int32)
        self.context=np.zeros((n,2),np.float32);self.code=np.zeros_like(self.g)
        self.rlast=np.zeros((n,2),np.float32);self.wlast=np.zeros((n,2,4),np.float32)
        self.csum=np.zeros((n,2,r["slots"]),np.float32);self.ccount=np.zeros((n,2),np.int32)
        self.at_test=None;self.gain_query=[];self.row=np.arange(n)
        self.cov=None
        if r.get("updater")=="rls":
            self.cov=np.broadcast_to(np.eye(r["slots"],dtype=np.float32)/r["ridge"],(n,r["slots"],r["slots"])).copy()

    def _noise(self):
        slot=self.it%128
        if slot==0:self.noise=np.stack([r.standard_normal((128,2),dtype=np.float32) for r in self.rng],1)
        self.it+=1
        return self.noise[slot]

    def address(self):
        r=self.route
        if r["routing"]=="single":return np.ones((self.n,1),np.float32)
        if r["routing"]=="public":v=self.context
        elif r["routing"]=="shared_public":v=np.column_stack((self.context,np.full(self.n,r["shared"],np.float32)))
        else:
            w,b=r["first"];z=np.tanh(self.h.mean(1)@w+b);w,b=r["second"];v=z@w+b
        return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-7)

    def act(self,obs):
        p=self.perception;r=self.route;n=self.n
        if np.all(obs[:,world.TRIAL_START]):
            self.round+=1
            if self.round==self.start:
                cod=self.csum/np.maximum(self.ccount[:,:,None],1)
                self.at_test=dict(g=self.g.copy(),codes=cod.copy())
                if self.change=="erase_rule":self.g.fill(0.)
                if self.change=="erase_events":self.signed.fill(0.);self.mem.fill(0.);self.count.fill(0)
                if self.change in ("erase_A_direction","erase_B_direction"):
                    k=0 if self.change=="erase_A_direction" else 1
                    u=cod[:,k];proj=(self.g*u).sum(1)/np.maximum((u*u).sum(1),1e-8)
                    self.g-=proj[:,None]*u
        y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        ids=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        if len(ids) and self.change!="no_write":
            side=obs[ids,world.CONTACT_ID].argmax(1)
            c=self.code[ids]
            if self.round<self.start and self.labels is not None:
                tag=self.labels[ids,self.round]
                self.csum[ids,tag]+=c;self.ccount[ids,tag]+=1
            recalled=self.rlast[ids,side]
            k=c*recalled[:,None]
            prediction=(self.g[ids]*k).sum(1)
            error=(r.get("reward_factor",1.)*y[ids]+r.get("prediction_factor",-1.)*prediction+r.get("offset",0.))
            error*=1.+r.get("familiarity_factor",0.)*(np.abs(recalled)-.5)
            if self.cov is not None:
                pmat=self.cov[ids]
                z=(pmat@k[:,:,None]).squeeze(2)
                if "release" in r:
                    mode,maximum,bias,werr,wrec=r["release"]
                    amount=np.full(len(ids),maximum,np.float32) if mode<3 else maximum/(1.+np.exp(-(bias+werr*np.abs(y[ids]-prediction)+wrec*np.abs(recalled))))
                    amount=amount*((k*k).sum(1)>1e-12)
                    if mode in (0,5):
                        proj=np.broadcast_to(np.eye(k.shape[1],dtype=np.float32)/k.shape[1],pmat.shape)
                    else:
                        u=z if mode in (2,4) else k
                        u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8)
                        proj=u[:,:,None]*u[:,None,:]
                    pmat=pmat+amount[:,None,None]*proj
                    z=(pmat@k[:,:,None]).squeeze(2)
                direction=z/(1.+(z*k).sum(1,keepdims=True))
                self.cov[ids]=pmat-direction[:,:,None]*z[:,None,:]
                if self.change=="same_step_length":
                    direction=k*(np.linalg.norm(direction,axis=1,keepdims=True)/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-8))
                if self.change=="diagonal_geometry":
                    diag=np.diagonal(pmat,axis1=1,axis2=2)*k
                    direction=diag/(1.+(diag*k).sum(1,keepdims=True))
                if "magnitude_floor" in r:
                    floor_mode,step_floor=r["magnitude_floor"]
                    gnorm=np.linalg.norm(direction,axis=1,keepdims=True)
                    knorm=np.linalg.norm(k,axis=1,keepdims=True)
                    required=step_floor*knorm
                    along_rule=direction*required/np.maximum(gnorm,1e-10)
                    along_activity=k*required/np.maximum(knorm,1e-10)
                    if floor_mode==0:direction=np.where(gnorm<required,along_rule,direction)
                    elif floor_mode==1:direction=k*np.maximum(gnorm,required)/np.maximum(knorm,1e-10)
                    elif floor_mode==2:direction=np.where(gnorm<required,along_activity,direction)
                    elif floor_mode==3:direction=along_rule
                direction*=r.get("rls_step",1.)
            else:direction=r["eta"]*k
            if self.change=="permute_write":direction=np.roll(direction,1,axis=1)
            self.g[ids]=r["retention"]*self.g[ids]+error[:,None]*direction
            slot=self.count[ids]%p["capacity"]
            self.mem[ids,slot]=self.wlast[ids,side];self.signed[ids,slot]=y[ids];self.count[ids]+=1
        ctx=obs[:,food.CTX]
        if self.change=="other_context_read" and self.round>=self.start:
            show=ctx.any(1)&self.probes[:,self.round]
            if np.any(show):
                ctx=ctx.copy()
                tags=self.labels[show,self.round]
                ctx[show]=self.appearance[self.row[show],1-tags]
        observed=ctx.any(1);self.context[observed]=ctx[observed]
        scents=obs[:,food.LEFT.start:food.RIGHT.stop].reshape(n,2,4)*2.-1.
        self.h=self.h@p["matrix"]+scents@p["sensors"][:4]+(ctx@p["sensors"][4:])[:,None,:]
        w,b=p["ground"];odor=self.h@w+b
        norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        q=norm(odor);self.wlast=norm(odor@p["transform"])
        temp,null,_,_,_=p["scalars"]
        similarity=q@self.mem.transpose(0,2,1)
        raw_scores=temp*similarity
        if p.get("age_cost",0.):
            age=(self.count[:,None]-1-np.arange(p["capacity"])[None,:])%p["capacity"]
            raw_scores-=p["age_cost"]*age[:,None,:].astype(np.float32)
        logits=np.where(np.arange(p["capacity"])[None,None,:]<self.count[:,None,None],raw_scores,-1e4)
        logits=np.concatenate([logits,np.full((n,2,1),temp*null,np.float32)],2)
        score=np.exp(logits-logits.max(2,keepdims=True));score/=score.sum(2,keepdims=True)
        self.rlast=(score[:,:,:-1]*self.signed[:,None,:]).sum(2)
        self.code=self.address()
        mod=np.tanh(r["amp"]*(self.g*self.code).sum(1))
        val=mod[:,None]*self.rlast
        tonic,amplify,leak,noise,self_exc,opp,confidence=self.motion
        diff=val-val[:,::-1]
        drive=(tonic+confidence*np.abs(diff))*obs[:,world.ACTIVE,None]-.85*(1-obs[:,world.ACTIVE,None])
        drive+=amplify*np.tanh(diff)+noise*self._noise()+self_exc*self.motor-opp*self.motor[:,::-1]
        self.mv=np.maximum(-1.,leak*self.mv+drive)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor






