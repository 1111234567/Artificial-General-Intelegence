"""Bilateral recurrent olfactory circuits learn a private taste function from sampled rewards."""

from dataclasses import dataclass

import numpy as np

from . import mixture_world as game,world


@dataclass(frozen=True)
class Rule:
    representation:str="spikes"
    neurons:int=64
    latent:int=24
    rate:float=.88
    ridge:float=.90
    forget:float=1.
    covariance:bool=True
    leak:float=.58
    recurrence:float=.22
    trace:float=.55
    dendrite:float=.35
    sensor_gain:float=.68
    motor_gain:float=1.6
    motor_tonic:float=.39
    motor_noise:float=.13


class MixtureLearner:
    """The same sensory graph and value synapses serve either physical side."""

    def __init__(self,rule,net,seeds,*,cut=False,freeze=None,mask_context=False):
        self.rule,self.seeds,self.freeze,self.mask=rule,tuple(map(int,seeds)),freeze,mask_context
        if rule.representation not in ("spikes","linear_prior","quadratic_prior","additive","raw_no_context"):
            raise ValueError("Unknown cue representation")
        rng=np.random.default_rng(net);n=rule.neurons
        self.chem=(rng.normal(0,1,(4,n))*(rng.random((4,n))<.75)*rule.sensor_gain).astype(np.float32)
        self.ctx=rng.normal(.14,.50,(2,n)).astype(np.float32)
        mask=rng.random((n,n))<.10;np.fill_diagonal(mask,False)
        mask[np.arange(n),(np.arange(n)+1)%n]=True
        signs=np.where(rng.random(n)<.25,-1.8,1.)
        self.rec=(mask*signs[:,None]*rule.recurrence*rng.uniform(0.,1.,(n,n))*(not cut)).astype(np.float32)
        self.thresh=rng.uniform(.8,1.18,n).astype(np.float32)
        self.bias=rng.uniform(.09,.39,n).astype(np.float32)
        self.projection=(rng.normal(0,1,(n,rule.latent))/np.sqrt(n)).astype(np.float32)
        self.motor_back=(rng.normal(0,.035,(2,n))*(not cut)).astype(np.float32)

    def reset(self,n):
        typ=self.rule.representation;self.n=n
        k={"spikes":self.rule.latent,"linear_prior":10,"quadratic_prior":22,"raw_no_context":5,
           "additive":7}[typ]
        self.h=np.zeros((n,2,self.rule.neurons),np.float32);self.sp=np.zeros_like(self.h)
        self.z=np.zeros_like(self.h);self.mean=np.full((n,1,self.rule.neurons),.2,np.float32)
        self.motor=np.zeros((n,2),np.float32);self.mv=np.zeros_like(self.motor)
        self.weights=np.zeros((n,k),np.float32)
        self.p=np.broadcast_to(np.eye(k,dtype=np.float32)/self.rule.ridge,(n,k,k)).copy() if self.rule.covariance else None
        self.prior_features=np.zeros((n,2,k),np.float32);self.prior_value=self.motor.copy()
        self.rng=[np.random.default_rng(np.random.SeedSequence([s,745930])) for s in self.seeds]
        self.noise=np.empty((0,n,2),np.float32);self.it=0;self.round=-1
        self.writes=np.zeros(n,np.int32)

    def _noise(self):
        k=self.it%128
        if not k:self.noise=np.stack([r.standard_normal((128,2),dtype=np.float32) for r in self.rng],1)
        self.it+=1;return self.noise[k]

    def _write(self,obs):
        signal=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        ids=np.flatnonzero((signal!=0)&obs[:,world.CONTACT_ID].any(1))
        if not ids.size or not self.rule.rate or (self.freeze is not None and self.round>=self.freeze):return
        side=obs[ids,world.CONTACT_ID].argmax(1);key=self.prior_features[ids,side]
        residual=signal[ids]-self.prior_value[ids,side]
        if self.p is not None:
            p=self.p[ids]
            z=(p@key[:,:,None]).squeeze(-1)
            gain=z/(self.rule.forget+(z*key).sum(1,keepdims=True))
            self.p[ids]=(p-gain[:,:,None]*z[:,None,:])/self.rule.forget
        else:
            gain=key/(self.rule.ridge+(key*key).sum(1,keepdims=True))
        self.weights[ids]+=self.rule.rate*residual[:,None]*gain
        self.writes[ids]+=1

    def _features(self,x,ctx):
        r=self.rule
        if r.representation=="spikes":
            chemdrive=x@self.chem
            context=(ctx@self.ctx)[:,None,:]
            interaction=r.dendrite*context*chemdrive
            self.h=np.maximum(r.leak*self.h+self.bias+chemdrive+context+interaction+
                              self.sp.reshape(-1,r.neurons).dot(self.rec).reshape(self.h.shape)+
                              (self.motor@self.motor_back)[:,None,:],-2.5)
            self.sp=(self.h>=self.thresh).astype(np.float32);self.h*=1.-self.sp
            self.z+= (1.-r.trace)*(self.sp-self.z)
            key=(self.z-self.mean)@self.projection
            self.mean+=.012*(self.z.mean(1,keepdims=True)-self.mean)
        else:
            const=np.ones((*x.shape[:2],1),np.float32)
            if r.representation=="raw_no_context":key=np.concatenate((const,x),axis=2)
            elif r.representation=="additive":key=np.concatenate((const,x,np.broadcast_to(ctx[:,None],(*x.shape[:2],2))),2)
            else:
                if r.representation=="quadratic_prior":
                    x=np.concatenate((x,x[:,:,game.PAIR[:,0]]*x[:,:,game.PAIR[:,1]]),2)
                per_side=np.concatenate((const,x),axis=2)
                key=np.concatenate([per_side*ctx[:,None,j,None] for j in range(2)],axis=2)
        # Keep the same normalization across both physical motors.
        return key/np.sqrt(.25+(key*key).sum(2,keepdims=True))

    def act(self,obs):
        if np.all(obs[:,world.TRIAL_START]):self.round+=1
        self._write(obs)
        x=2.*np.stack((obs[:,game.LEFT],obs[:,game.RIGHT]),axis=1)-1.
        ctx=obs[:,game.CTX]
        if self.mask:ctx=np.full_like(ctx,.5)
        key=self._features(x,ctx)
        q=np.einsum("bi,bsi->bs",self.weights,key)
        active=obs[:,world.ACTIVE,None]
        drive=(self.rule.motor_tonic*active-.85*(1-active)+self.rule.motor_gain*np.tanh(q-q[:,::-1])+
               self.rule.motor_noise*self._noise()+1.08*self.motor-1.28*self.motor[:,::-1])
        self.mv=np.maximum(-1.,.78*self.mv+drive)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        self.prior_features=key;self.prior_value=q
        return self.motor
