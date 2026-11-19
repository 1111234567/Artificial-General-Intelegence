"""Decide which prior memory read applies to this current sensory pattern, and protect the other rule."""
from dataclasses import dataclass
import numpy as np
from . import adaptive_local,world


@dataclass(frozen=True)
class Settings:
    evidence:float=7.
    prediction:float=1.
    global_rate:float=.7
    prior_limit:float=2.
    hold:float=.95
    read:bool=True
    protect:bool=True


class Agent(adaptive_local.LocalAdaptive):
    def __init__(self,front,route,seeds,*,motion,config,record_gate=False):
        self.conf=config;self.record_gate=record_gate
        super().__init__(front,route,seeds,motor=motion)
    def reset(self,n):
        super().reset(n)
        self.prior=np.zeros(n,np.float32)
        self.pred_i=np.zeros((n,2),np.float32);self.pred_t=np.zeros_like(self.pred_i)
        self.support_i=np.zeros_like(self.pred_i);self.support_t=np.zeros_like(self.pred_i)
        self.weights_at_start=[];self.responsibility_sum=np.zeros(n,np.float32);self.actual_updates=np.zeros(n,int)
    def act(self,obs):
        r=self.conf;p=self.perception;y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        ids=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        old_m=self.motor.copy();old_mv=self.mv.copy()
        if len(ids):
            s=obs[ids,world.CONTACT_ID].argmax(1)
            advantage=(y[ids]-self.pred_t[ids,s])**2-(y[ids]-self.pred_i[ids,s])**2
            activ=self.prior[ids]+r.evidence*(self.support_i[ids,s]-self.support_t[ids,s])+r.prediction*advantage
            plastic=1./(1.+np.exp(np.clip(activ,-30.,30.)))
            self.prior[ids]=np.clip(r.hold*self.prior[ids]+r.global_rate*advantage,-r.prior_limit,r.prior_limit)
            old_g=self.g[ids].copy();old_p=self.cov[ids].copy()
        pulse=super().act(obs)
        if len(ids):
            self.actual_updates[ids]+=1;self.responsibility_sum[ids]+=plastic
            if r.protect:
                self.g[ids]=old_g+plastic[:,None]*(self.g[ids]-old_g)
                self.cov[ids]=old_p+plastic[:,None,None]*(self.cov[ids]-old_p)
        temp,null=p["scalars"][:2]
        mask=np.arange(p["capacity"])[None,None,:]<self.count[:,None,None]
        age=(self.count[:,None]-1-np.arange(p["capacity"])[None,:])%p["capacity"]
        w,b=p["ground"];decoded=self.h@w+b
        decoded/=np.maximum(np.linalg.norm(decoded,axis=2,keepdims=True),1e-12)
        def read(key):
            raw=temp*(key@self.mem.transpose(0,2,1))-p.get("age_cost",0.)*age[:,None,:].astype(np.float32)
            score=np.concatenate((np.where(mask,raw,-1.e4),np.full((self.n,2,1),temp*null,np.float32)),2)
            z=np.exp(score-score.max(2,keepdims=True));z/=z.sum(2,keepdims=True)
            return (z[:,:,:-1]*self.signed[:,None,:]).sum(2),1.-z[:,:,-1]
        self.pred_i,self.support_i=read(self.wlast)
        _,self.support_t=read(decoded)
        mod=np.tanh(self.route["amp"]*(self.g*self.code).sum(1))
        self.pred_t=self.rlast*mod[:,None]
        evidence=r.evidence*(self.support_i-self.support_t) if r.read else 0.
        wij=1./(1.+np.exp(-np.clip(self.prior[:,None]+evidence,-30.,30.)))
        answer=wij*self.pred_i+(1.-wij)*self.pred_t
        if self.record_gate and obs[:,world.TRIAL_START].all():self.weights_at_start.append(wij.mean(1).copy())
        tonic,gain,leak,noise,exc,opp,confidence=self.motion;delta=answer-answer[:,::-1];act=obs[:,world.ACTIVE,None]
        drive=(tonic+confidence*np.abs(delta))*act-.85*(1.-act)
        drive+=gain*np.tanh(delta)+noise*self.noise[(self.it-1)%128]+exc*old_m-opp*old_m[:,::-1]
        self.mv=np.maximum(-1.,leak*old_mv+drive);self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor

