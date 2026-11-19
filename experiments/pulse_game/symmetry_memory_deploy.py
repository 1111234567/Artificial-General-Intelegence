"""Deploy finite synaptic event memory and its learned modulation one physical tick at a time."""
import numpy as np
from . import mixture_brain, mixture_world as game, world


class Forager(mixture_brain.MixtureLearner):
    def __init__(self,p,seeds,*,intervention="true",boundary=0,tonic=.39,gain=1.6):
        self.model=p;self.intervention=intervention;self.boundary=boundary
        super().__init__(mixture_brain.Rule(neurons=p["matrix"].shape[0],latent=p["latent"],
                       covariance=False,motor_tonic=tonic,motor_gain=gain),13,seeds)

    def reset(self,n):
        super().reset(n);p=self.model
        self.state=np.zeros((n,2,p["matrix"].shape[0]),np.float32)
        self.addresses=np.zeros((n,p["capacity"],p["latent"]),np.float32)
        self.signs=np.zeros((n,p["capacity"]),np.float32)
        self.count=np.zeros(n,int)
        self.master=np.zeros(n,np.float32)
        self.previous_write=np.zeros((n,2,p["latent"]),np.float32)
        self.previous_value=np.zeros((n,2),np.float32)
        self.previous_confidence=np.zeros((n,2),np.float32)
        self.record=[]

    def event_age(self):
        cap=self.model["capacity"]
        return (self.count[:,None]-1-np.arange(cap)[None,:]) % cap

    def store(self,ids,side,sign):
        p=self.model
        where=self.count[ids]%p["capacity"]
        self.addresses[ids,where]=self.previous_write[ids,side]
        self.signs[ids,where]=sign
        self.count[ids]+=1

    def act(self,obs):
        p=self.model
        if obs[:,world.TRIAL_START].all():
            self.round+=1
            if self.round==self.boundary:
                n=self.n//2
                if self.intervention=="transplant_rule":self.master[:]=np.roll(self.master,n)
                if self.intervention=="clear_rule":self.master.fill(0.)
                if self.intervention=="transplant_events":
                    for v in (self.addresses,self.count,self.signs):v[:]=np.roll(v,n,axis=0)
            self.record.append(self.master.copy())
        y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        if self.intervention=="no_anchor" and self.round==self.boundary:y=np.zeros_like(y)
        ids=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        temp,null,rate,hold,scale=p["scalars"]
        if len(ids):
            side=obs[ids,world.CONTACT_ID].argmax(1)
            recalled=self.previous_value[ids,side]
            sign=y[ids]
            if self.intervention!="no_learning":
                if p["change"]=="delta":
                    self.master[ids]=hold*self.master[ids]+rate*recalled*(sign-self.master[ids]*recalled)
                elif p["change"]=="learned":
                    w0,b0=p["rule"][0];w1,b1=p["rule"][1]
                    x=np.stack((sign,recalled,sign*recalled,self.previous_confidence[ids,side]),1)
                    c=np.tanh(x@w0+b0)@w1+b1
                    self.master[ids]=(.55+.449/(1.+np.exp(-c[:,0])))*self.master[ids]+.9*np.tanh(c[:,1])
                self.store(ids,side,sign)
        x=2.*np.stack((obs[:,game.LEFT],obs[:,game.RIGHT]),1)-1.
        self.state=self.state@p["matrix"]+x@p["sensors"][:4]+(obs[:,game.CTX]@p["sensors"][4:])[:,None,:]
        def layer(v,name):
            w,b=p[name];return v@w+b
        norm=lambda k:k/np.maximum(np.sqrt((k*k).sum(-1,keepdims=True)),1e-12)
        if "ground" in p:
            chemical=layer(self.state,"ground")
            q=norm(chemical);self.previous_write=norm(chemical@p["transform"])
        else:
            z=np.tanh(layer(self.state,"a")*(1.+np.tanh(layer(self.state,"b"))))
            q=norm(layer(z,"read"));self.previous_write=norm(layer(z,"write"))
        sim=q@self.addresses.transpose(0,2,1)
        raw_scores=sim*temp
        if p.get("age_cost",0.)!=0.:
            age=self.event_age()
            raw_scores=raw_scores-p["age_cost"]*age[:,None,:].astype(np.float32)
        logits=np.where(np.arange(p["capacity"])[None,None,:]<self.count[:,None,None],raw_scores,-1e4)
        # The null slot reports an unrecognized odor without a signed taste.
        scores=np.concatenate((logits,np.full((*q.shape[:2],1),temp*null,np.float32)),axis=2)
        weights=np.exp(scores-scores.max(2,keepdims=True));weights/=weights.sum(2,keepdims=True)
        self.previous_value=(weights[:,:,:-1]*self.signs[:,None,:]).sum(2)
        self.previous_confidence=1.-weights[:,:,-1]
        gain=np.ones(self.n,np.float32) if p["change"]=="fixed" else np.tanh(scale*self.master)
        if self.intervention=="invert_rule":gain=-gain
        if self.intervention=="same_rule":gain[:]=1.
        if self.intervention=="opposite_rule":gain[:]=-1.
        values=self.previous_value*gain[:,None]
        active=obs[:,world.ACTIVE,None]
        motion=p.get("motion",(self.rule.motor_tonic,self.rule.motor_gain,.78,self.rule.motor_noise,1.08,1.28,0.))
        tonic,amplify,leak,noise,self_excitation,opposition,confidence=motion
        diff=values-values[:,::-1]
        drive=((tonic+confidence*np.abs(diff))*active-.85*(1-active)+amplify*np.tanh(diff)+
               noise*self._noise()+self_excitation*self.motor-opposition*self.motor[:,::-1])
        self.mv=np.maximum(-1.,leak*self.mv+drive)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor




