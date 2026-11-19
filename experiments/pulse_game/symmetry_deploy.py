"""Run the learned taste predictors through physical spiking motion at every tick."""
import numpy as np
from . import mixture_brain, mixture_world as game, world


class Forager(mixture_brain.MixtureLearner):
    def __init__(self,params,seeds,*,change="true",tonic=.39,gain=1.6):
        self.model_params=params
        self.change=change
        super().__init__(mixture_brain.Rule(neurons=params["matrix"].shape[0],latent=params["latent"],
                         covariance=False,motor_tonic=tonic,motor_gain=gain),13,seeds)

    def reset(self,n):
        super().reset(n)
        p=self.model_params
        self.s=np.zeros((n,2,p["matrix"].shape[0]),np.float32)
        self.fast=np.zeros((n,p["experts"],p["latent"]),np.float32)
        self.logit=np.zeros((n,p["experts"]),np.float32)
        self.last_key=np.zeros((n,2,p["experts"],p["latent"]),np.float32)
        self.prev_q=np.zeros((n,2,p["experts"]),np.float32)
        self.records=[]

    def act(self,obs):
        p=self.model_params
        if obs[:,world.TRIAL_START].all():self.round+=1
        real=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        rows=np.flatnonzero((real!=0)&obs[:,world.CONTACT_ID].any(1))
        if len(rows):
            side=obs[rows,world.CONTACT_ID].argmax(1)
            err=real[rows,None]-self.prev_q[rows,side]
            if self.change!="no_write":
                self.fast[rows]=p["forget"][None,:,None]*self.fast[rows]+(p["rate"]*err)[:,:,None]*self.last_key[rows,side]
            if self.change!="no_evidence":
                d=err**2
                self.logit[rows]=p["hold"]*self.logit[rows]-p["temp"]*(d-d.mean(1,keepdims=True))
        x=2.*np.stack([obs[:,game.LEFT],obs[:,game.RIGHT]],1)-1.
        ctx=obs[:,game.CTX]
        self.s=self.s@p["matrix"]+x@p["sensors"][:4]+(ctx@p["sensors"][4:])[:,None,:]
        def layer(m):
            w,b=p[m]
            return (self.s@w+b).reshape(self.n,2,p["experts"],p["width"])
        z=np.tanh(layer("a")*(1.+np.tanh(layer("b"))))
        k=np.einsum("bseh,ehk->bsek",z,p["out"],optimize=False)+p["offset"]
        self.last_key=k/np.sqrt(.13+(k*k).sum(-1,keepdims=True))
        self.prev_q=(self.last_key*self.fast[:,None]).sum(-1)
        logits=self.logit+p["prior"]
        if p["mechanism"]=="evidence" and self.change!="uniform":
            if self.change=="invert_evidence":logits=-logits
            w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        else:w=np.full_like(logits,1./p["experts"])
        q=(self.prev_q*w[:,None,:]).sum(2)
        if np.all(obs[:,world.TRIAL_START]):self.records.append(w.copy())
        active=obs[:,world.ACTIVE,None]
        drive=(self.rule.motor_tonic*active-.85*(1-active)+self.rule.motor_gain*np.tanh(q-q[:,::-1])+
               self.rule.motor_noise*self._noise()+1.08*self.motor-1.28*self.motor[:,::-1])
        self.mv=np.maximum(-1.,.78*self.mv+drive)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor

