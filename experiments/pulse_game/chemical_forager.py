"""An online bilinear synaptic update equivariant to the current life's odor coordinates.

Learning uses the sensory receptor mixture which preceded a bite. The rule
never retrieves a discrete fruit identity or enumerates hypothetical odors.
"""

from dataclasses import dataclass,replace

import numpy as np

from . import ring_spikes, ring_world as r


@dataclass(frozen=True)
class Plasticity:
    geometry:str="factor"
    # Denominator is relative to the current preconditioner traces. This
    # keeps a frequently observed but surprising cue eligible for learning.
    damping:float=.14
    rate:float=.74
    covariance_rate:float=.17
    initial_scale:float=1.
    clip_direction:float=12.
    aspire:float=.4
    threshold:float=.15
    gain:float=1.65
    novelty:float=.055
    hunger:float=.017
    onset:bool=False
    local_release:float=0.
    global_release:bool=False
    repeated_surprise:bool=False
    surprise_threshold:float=.45


class ChemicalForager(ring_spikes.Forager):
    """Continuous spiking population, a single consummatory cell and tonic locomotion."""

    def __init__(self,config,plastic,net,seeds,*,cut=False,freeze=None):
        if plastic.geometry not in ("factor","diagonal","identity","scrambled","full"):
            raise ValueError("Unknown synaptic geometry")
        super().__init__(replace(config,method="delta",competition=0,exploration=plastic.novelty),
                         net,seeds,no_recurrence=cut,freeze_after=freeze)
        self.rule=plastic
        self.permute=np.random.default_rng(net+1917).permutation(config.latent)

    def reset(self,n):
        super().reset(n)
        k=self.cfg.latent;rule=self.rule
        self.pk=np.broadcast_to(np.eye(k,dtype=np.float32)*rule.initial_scale,(n,k,k)).copy()
        self.pf=np.broadcast_to(np.eye(4,dtype=np.float32)*rule.initial_scale,(n,4,4)).copy()
        self.joint=None
        if rule.geometry=="full":
            self.joint=np.broadcast_to(np.eye(4*k,dtype=np.float32)*rule.initial_scale,(n,4*k,4*k)).copy()
        self.inconsistency=np.zeros_like(self.w)
        self.peak=np.zeros(n,np.float32)
        self.onset=np.zeros_like(self.last_key)
        self.extra_write=np.zeros(n,np.float32)

    def _project(self,cov,features,who,release):
        p=cov[who]
        rule=self.rule
        if rule.local_release>0 and np.any(release):
            if rule.global_release:
                p+=release[:,None,None]*np.eye(p.shape[-1],dtype=np.float32)/p.shape[-1]
            else:
                p+=release[:,None,None]*features[:,:,None]*features[:,None,:]
        z=(p@features[:,:,None]).squeeze(-1)
        den=1.+rule.covariance_rate*(z*features).sum(1)
        cov[who]=p-(rule.covariance_rate/den)[:,None,None]*z[:,:,None]*z[:,None,:]
        if rule.geometry=="diagonal":
            z=np.diagonal(p,axis1=1,axis2=2)*features
        return z

    def _write(self,obs):
        rule=self.rule;rwd=obs[:,r.POSITIVE]-obs[:,r.NEGATIVE]
        who=np.flatnonzero((rwd!=0)&np.any(self.last_odor!=0,axis=1))
        if not who.size or rule.rate==0 or (self.freeze_after is not None and self.age>=self.freeze_after):
            return
        x=self.last_key[who]; y=self.last_odor[who]; pred=self.last_pred[who]
        y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1.e-7)
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1.e-7)
        observed=rwd[who]
        feature=y[:,:,None]*x[:,None,:]
        release=np.zeros(len(who),np.float32)
        if rule.local_release>0:
            surprise=np.maximum(0.,-observed*pred-rule.surprise_threshold)
            history=(self.inconsistency[who]*feature).sum((1,2))
            release=rule.local_release*surprise
            if rule.repeated_surprise:release*=np.clip(2.5*history,0.,1.)
            self.inconsistency[who]+=.45*(surprise-history)[:,None,None]*feature
            self.extra_write[who]+=release
        if rule.geometry=="identity":direction=feature
        elif rule.geometry=="full":
            f=feature.reshape(len(who),-1)
            direction=self._project(self.joint,f,who,release).reshape(feature.shape)
        else:
            zx=self._project(self.pk,x,who,release)
            zy=self._project(self.pf,y,who,np.zeros_like(release))
            if rule.geometry=="scrambled":
                scrambled=x[:,self.permute]
                zx=(self.pk[who]@scrambled[:,:,None]).squeeze(-1)[:,np.argsort(self.permute)]
            direction=zy[:,:,None]*zx[:,None,:]
        norm=np.sqrt((direction*direction).sum((1,2)))
        leverage=(feature*direction).sum((1,2))
        scale=np.minimum(1./np.maximum(leverage+rule.damping*norm,1e-6),
                         rule.clip_direction/np.maximum(norm,1e-6))
        self.w[who]+=(rule.rate*(observed-pred)*scale)[:,None,None]*direction
        self.writes[who]+=1

    def act(self,obs):
        if np.all(obs[:,r.NEW]):self.age+=1
        cfg,rule=self.cfg,self.rule
        self._write(obs)
        self.sense*=cfg.receptor_decay
        np.maximum(self.sense,obs[:,:6],out=self.sense)
        self.h=np.maximum(cfg.leak*self.h+self.sense@self.win+self.s@self.rec+self.motor@self.back+self.bias,-2.)
        self.s=(self.h>=self.thresh).astype(np.float32)
        self.h*=1.-self.s;self.spike_count+=self.s.sum(1,dtype=np.int64)
        self.z*=cfg.history;self.z+=(1-cfg.history)*self.s
        self.mean+=.008*(self.z-self.mean)
        if rule.onset:
            start=obs[:,r.NEW]!=0
            self.onset[start]=((self.s[start]+self.h[start])-.18)@self.project
            key=self.onset
        else:
            key=(self.z-self.mean)@self.project
        key=key/np.maximum(np.linalg.norm(key,axis=1,keepdims=True),1e-7)
        odor=obs[:,r.FRUIT]
        odor=odor/np.maximum(np.linalg.norm(odor,axis=1,keepdims=True),1e-7)
        score=np.einsum("bf,bfk,bk->b",odor,self.w,key,optimize=False)
        # The aspiration state sees rewards predicted for actually encountered
        # stimuli, not the true target nor the list of four possible stimuli.
        self.peak*=.947
        threshold=rule.threshold+rule.aspire*self.peak
        visible=odor.any(1)
        self.peak[visible]=np.maximum(self.peak[visible],np.clip(score[visible],0.,1.5))
        active=obs[:,r.ACTIVE]
        self.hunger=np.clip(.97*self.hunger+rule.hunger*active-obs[:,r.POSITIVE]-obs[:,r.NEGATIVE],0.,.9)
        toss=self._explore()
        self.mv*=.35
        self.mv[:,1]+=1.12*active-.3*(1-active)
        self.mv[:,2]+=obs[:,r.REACH]*(.86+self.hunger+rule.gain*(np.clip(score,-2.,2.)-threshold)+1.9*toss)-.65*(1-obs[:,r.REACH])
        np.maximum(self.mv,-.6,out=self.mv)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        biting=(self.motor[:,2]!=0)&(obs[:,r.REACH]!=0)&(active!=0)
        self.last_key[biting]=key[biting];self.last_odor[:]=odor*biting[:,None];self.last_pred[biting]=score[biting]
        return self.motor
