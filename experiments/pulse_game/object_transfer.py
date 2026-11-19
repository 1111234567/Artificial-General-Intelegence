"""Use the model's two genuine sampled taste prediction errors to choose a memory similarity."""
import numpy as np
from . import adaptive_local,symmetry_world as angular,world


class Agent(adaptive_local.LocalAdaptive):
    def __init__(self,front,route,seeds,*,motion,kind="adaptive",hold=.96,rate=.9):
        super().__init__(front,route,seeds,motor=motion)
        self.kind,self.hold,self.rate=kind,hold,rate
    def reset(self,n):
        super().reset(n)
        self.gate=np.zeros(n,np.float32);self.same=np.zeros((n,2),np.float32)
        self.rel=np.zeros_like(self.same);self.weights_at_start=[]
    def act(self,obs):
        p=self.perception;now=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        ids=np.flatnonzero((now!=0)&obs[:,world.CONTACT_ID].any(1))
        if len(ids):
            sd=obs[ids,world.CONTACT_ID].argmax(1)
            error=(now[ids]-self.rel[ids,sd])**2-(now[ids]-self.same[ids,sd])**2
            self.gate[ids]=np.clip(self.hold*self.gate[ids]+self.rate*error,-12.,12.)
        old_motor=self.motor.copy();old_mv=self.mv.copy()
        out=super().act(obs)
        if self.kind=="native":return out
        # The same actual event memory supplies both comparisons. Transforming
        # the query in the same way as the stored candidate compares identity.
        temp,null=p["scalars"][:2]
        sim=self.wlast@self.mem.transpose(0,2,1)
        score=temp*sim
        if p.get("age_cost",0.):
            age=(self.count[:,None]-1-np.arange(p["capacity"])[None,:])%p["capacity"]
            score-=p["age_cost"]*age[:,None,:].astype(np.float32)
        score=np.where(np.arange(p["capacity"])[None,None,:]<self.count[:,None,None],score,-1e4)
        score=np.concatenate([score,np.full((self.n,2,1),temp*null,np.float32)],2)
        a=np.exp(score-score.max(2,keepdims=True));a/=a.sum(2,keepdims=True)
        self.same=(a[:,:,:-1]*self.signed[:,None,:]).sum(2)
        self.rel=np.tanh(self.route["amp"]*(self.g*self.code).sum(1))[:,None]*self.rlast
        w=1./(1.+np.exp(-self.gate))
        if self.kind=="identity":w.fill(1.)
        elif self.kind=="equal":w.fill(.5)
        elif self.kind=="reverse_gate":w=1.-w
        elif self.kind!="adaptive":raise ValueError(self.kind)
        if obs[:,world.TRIAL_START].all():self.weights_at_start.append(w.copy())
        val=self.same*w[:,None]+self.rel*(1.-w[:,None])
        tonic,amplify,leak,noise,exc,opp,certainty=self.motion;active=obs[:,world.ACTIVE,None]
        diff=val-val[:,::-1]
        drive=(tonic+certainty*np.abs(diff))*active-.85*(1.-active)
        drive+=amplify*np.tanh(diff)+noise*self.noise[(self.it-1)%128]+exc*old_motor-opp*old_motor[:,::-1]
        self.mv=np.maximum(-1.,leak*old_mv+drive);self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor
