"""Four-choice foraging with recurrent LIF cue cells and a single plastic bite response."""

from dataclasses import dataclass

import numpy as np

from . import ring_world as ring


@dataclass(frozen=True)
class Config:
    units:int=80
    latent:int=24
    leak:float=.45
    history:float=.64
    recurrence:float=.23
    receptor_decay:float=0.
    learning_rate:float=.75
    regularization:float=.09
    motor_gain:float=.85
    bite_bias:float=1.04
    exploration:float=.06
    competition:float=0.
    hunger_rate:float=0.
    flavor_overlap:float=0.
    method:str="delta"
    residual:bool=True


class Forager:
    """A tonic locomotor reflex plus a learned taste decision.

    The current fruit excites a population of taste receptors. A bite emits
    an eligibility event at their synapses. At the next tick, positive or
    negative feedback modifies these eligible synapses. The rule never
    receives a correct identity on bad outcomes, the preferred action or the layout.
    """
    def __init__(self,cfg,net_seed,life_seeds,*,no_recurrence=False,freeze_after=None):
        self.cfg=cfg;self.seeds=tuple(map(int,life_seeds));self.freeze_after=freeze_after
        if cfg.method not in ("delta","rls","diagonal"):raise ValueError("unknown plastic update")
        n=cfg.units;k=cfg.latent
        rng=np.random.default_rng(net_seed)
        self.win=(rng.normal(.22,.56,(6,n))*(rng.random((6,n))<.75)).astype(np.float32)
        self.bias=rng.uniform(.04,.13,n).astype(np.float32)
        self.thresh=rng.uniform(.85,1.1,n).astype(np.float32)
        mask=rng.random((n,n))<.10
        np.fill_diagonal(mask,False)
        mask[np.arange(n),(np.arange(n)+1)%n]=True
        sign=np.where(rng.random(n)<.3,-1.8,1.)
        self.rec=(mask*rng.uniform(.05,.95,(n,n))*cfg.recurrence*sign[:,None]*(not no_recurrence)).astype(np.float32)
        self.back=(rng.normal(0,.025,(3,n))*(rng.random((3,n))<.12)*(not no_recurrence)).astype(np.float32)
        self.project=(rng.standard_normal((n,k))/np.sqrt(n)).astype(np.float32)
        # Observed identity is supplied as a labelled sensory line by the world.
        # Zero overlap gives four independent afferents, not four action cells.
        self.odor=(np.eye(4)+cfg.flavor_overlap*rng.normal(0,1,(4,4))).astype(np.float32)
        self.odor/=np.linalg.norm(self.odor,axis=1,keepdims=True)

    def reset(self,n):
        if n!=len(self.seeds):raise ValueError("a seed per life is required")
        cfg=self.cfg;k=cfg.latent
        self.h=np.zeros((n,cfg.units),np.float32)
        self.s=np.zeros_like(self.h);self.z=np.zeros_like(self.h);self.mean=np.full_like(self.h,.18)
        self.sense=np.zeros((n,6),np.float32)
        self.motor=np.zeros((n,3),np.float32);self.mv=np.zeros_like(self.motor)
        self.w=np.zeros((n,4,k),np.float32)
        self.p=None
        if cfg.method=="rls":
            self.p=np.broadcast_to(np.eye(k,dtype=np.float32)/cfg.regularization,(n,4,k,k)).copy()
        elif cfg.method=="diagonal":
            self.p=np.full((n,4,k),cfg.regularization,np.float32)
        self.last_key=np.zeros((n,k),np.float32);self.last_odor=np.zeros((n,4),np.float32)
        self.last_pred=np.zeros(n,np.float32)
        self.hunger=np.zeros(n,np.float32)
        self.age=-1;self.writes=np.zeros(n,np.int32);self.spike_count=np.zeros(n,np.int64)
        self.rng=[np.random.default_rng(np.random.SeedSequence([s,88431])) for s in self.seeds]
        self.noise=np.empty((0,n),np.float32);self.noise_index=0

    def _explore(self):
        if self.noise_index==len(self.noise):
            self.noise=np.stack([r.random(128,dtype=np.float32) for r in self.rng],axis=1)
            self.noise_index=0
        outcome=self.noise[self.noise_index]<self.cfg.exploration
        self.noise_index+=1
        return outcome

    def _learn(self,obs):
        cfg=self.cfg
        reward=obs[:,ring.POSITIVE]-obs[:,ring.NEGATIVE]
        active=(reward!=0) & np.any(self.last_odor!=0,axis=1)
        if not active.any() or not cfg.learning_rate or (self.freeze_after is not None and self.age>=self.freeze_after):
            return
        who=np.flatnonzero(active)
        key=self.last_key[who]
        odor=self.last_odor[who]
        error=reward[who]-(self.last_pred[who] if cfg.residual else 0.)
        if cfg.method=="delta":
            direction=key/(cfg.regularization+(key*key).sum(1,keepdims=True))
            delta=(cfg.learning_rate*error[:,None,None])*odor[:,:,None]*direction[:,None,:]
            self.w[who]+=delta
        else:
            if cfg.flavor_overlap:raise ValueError("covariance comparison requires independent receptor lines")
            receptor=odor.argmax(1)
            old=self.p[who,receptor]
            if cfg.method=="diagonal":
                new=old+key*key
                self.p[who,receptor]=new
                direction=key/new
                direction/=1.+(key*direction).sum(1,keepdims=True)
            else:
                z=(old@key[:,:,None]).squeeze(-1)
                direction=z/(1.+(key*z).sum(1,keepdims=True))
                self.p[who,receptor]=old-direction[:,:,None]*z[:,None,:]
            self.w[who,receptor]+=cfg.learning_rate*error[:,None]*direction
        self.writes[who]+=1

    def act(self,obs):
        cfg=self.cfg
        if np.all(obs[:,ring.NEW]):self.age+=1
        self._learn(obs)
        self.sense*=cfg.receptor_decay
        np.maximum(self.sense,obs[:,:6],out=self.sense)
        # Each address is a distributed function of spike history. The fixed
        # projection is shared across contexts, raw cues, layouts and rewards.
        key=(self.z-self.mean)@self.project
        key/=np.sqrt(.01+(key*key).sum(1,keepdims=True))
        odor=obs[:,ring.FRUIT]@self.odor
        values=np.einsum("bfk,bk->bf",self.w,key,optimize=False)
        score=(odor*values).sum(1)
        drive=score
        if cfg.competition:
            # A competing sensory prediction inhibits the shared consummatory
            # output. No preferred destination or direction is passed to the body.
            drive=drive-cfg.competition*(values@self.odor.T).max(1)
        self.h=np.maximum(cfg.leak*self.h+self.sense@self.win+self.s@self.rec+self.motor@self.back+self.bias,-2.)
        self.s=(self.h>=self.thresh).astype(np.float32)
        self.h*=1-self.s;self.spike_count+=self.s.sum(1,dtype=np.int64)
        self.z*=cfg.history;self.z+=(1.-cfg.history)*self.s
        self.mean+=.008*(self.z-self.mean)
        # One low-level locomotion cell continuously scans the circular field
        # while active. There is no decoder ordering the agent to approach a color.
        active=obs[:,ring.ACTIVE]
        outcome=obs[:,ring.POSITIVE]+obs[:,ring.NEGATIVE]
        self.hunger=np.clip(.97*self.hunger+cfg.hunger_rate*active-outcome,0.,1.)
        self.mv*=.35
        self.mv[:,1]+=1.12*active-.3*(1-active)
        self.mv[:,2]+=(obs[:,ring.REACH]*(cfg.bite_bias+self.hunger+cfg.motor_gain*np.clip(drive,-2,2)+1.7*self._explore())
                      -.65*(1.-obs[:,ring.REACH]))
        np.maximum(self.mv,-.6,out=self.mv)
        self.motor=(self.mv>=1.).astype(np.float32)
        self.mv*=1-self.motor
        # This is an efference/sensory tag, not a true-outcome lookup. Only
        # a pulse near a visible fruit can form a nonzero tag.
        biting=(self.motor[:,2]!=0)&(obs[:,ring.REACH]!=0)&(active!=0)
        self.last_key[biting]=key[biting]
        self.last_odor[:]=odor*biting[:,None]
        self.last_pred[biting]=score[biting]
        return self.motor
