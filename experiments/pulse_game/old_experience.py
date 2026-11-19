"""Rehearse one stored, actually experienced association while the body is at rest."""
from dataclasses import dataclass,replace
import numpy as np
from . import habitat_deploy, world


@dataclass(frozen=True)
class Options:
    capacity:int=24
    storage:str="reservoir"
    choose:str="interference"
    target:str="actual"
    step:float=.12
    binding:str="factual"
    start_after:bool=False


class Agent(habitat_deploy.Forager):
    def __init__(self,front,route,seeds,*,start,motor,options):
        super().__init__(front,route,seeds,start=start,motor=motor)
        self.options=options
    def reset(self,n):
        super().reset(n)
        c=self.options.capacity;d=self.route["slots"]
        self.buffer=np.zeros((n,c,d),np.float32);self.target=np.zeros((n,c),np.float32)
        self.validity=np.zeros((n,c),np.float32);self.nevents=np.zeros(n,int);self.replays=np.zeros(n,int)
        self.replay_rng=[np.random.default_rng(np.random.SeedSequence([s,385929])) for s in self.seeds]
        self.replay_calls=0;self.draw=np.empty((0,n,2),np.float32)
    def _cash(self):
        j=self.replay_calls%128
        if not j:self.draw=np.stack([rng.random((128,2),dtype=np.float32) for rng in self.replay_rng],1)
        self.replay_calls+=1;return self.draw[j]
    def act(self,obs):
        coins=self._cash()
        y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        ids=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        if not len(ids):return super().act(obs)
        choose=obs[ids,world.CONTACT_ID].argmax(1)
        old=self.g[ids].copy();c=self.code[ids].copy();recalled=self.rlast[ids,choose].copy()
        # This self-generated sample is a rule estimate; its confidence is
        # reduced when the currently tasted address has no informative recall.
        opt=self.options;observed=y[ids].copy()
        pulse=super().act(obs)
        posterior=self.g[ids].copy()
        random=coins[ids]
        slots=opt.capacity;count=self.nevents[ids]
        if opt.step>0 and np.any(count>0) and (not opt.start_after or self.round>=self.start):
            eligible=np.arange(slots)[None,:]<count[:,None]
            bank=self.buffer[ids];target=self.target[ids];conf=self.validity[ids]
            before=(bank*old[:,None,:]).sum(2);after=(bank*self.g[ids,None,:]).sum(2)
            if opt.choose=="uniform":
                select=(random[:,0]*np.minimum(slots,count)).astype(int)
                active=count>0
            else:
                score=conf*((target-after)**2-(target-before)**2)
                score=np.where(eligible,score,-np.inf)
                select=score.argmax(1)
                active=(count>0)&(score[np.arange(len(ids)),select]>0)
                if opt.choose=="matched_uniform":select=(random[:,0]*np.minimum(slots,count)).astype(int)
            ii=np.flatnonzero(active)
            if len(ii):
                rows=ids[ii];idx=select[ii];key=bank[ii,idx]
                fixed=target[ii,idx]
                if opt.binding=="wrong":
                    # A different stored target of this SAME life. Same key,
                    # replay opportunity, learned covariance and availability.
                    next_slot=(idx+1)%np.minimum(slots,count[ii])
                    fixed=target[ii,next_slot]
                residual=fixed-(self.g[rows]*key).sum(1)
                z=(self.cov[rows]@key[:,:,None]).squeeze(2)
                # The covariance continues to count real evidence only. Replay
                # alters the remembered function without claiming a second receipt.
                direction=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-7)
                scale=opt.step*conf[ii,idx]*np.linalg.norm(key,axis=1)*np.clip(residual,-2.,2.)
                self.g[rows]+=scale[:,None]*direction;self.replays[rows]+=1
        if opt.target=="self":
            key=c;value=(posterior*c).sum(1);quality=np.abs(recalled).clip(0.,1.)
        else:key=c*recalled[:,None];value=observed;quality=np.ones_like(observed)
        if opt.storage=="fifo":slot=count%slots;store=np.ones(len(ids),bool)
        else:
            slot=np.where(count<slots,count,(random[:,1]*(count+1)).astype(int));store=slot<slots
        take=ids[store];j=slot[store]
        self.buffer[take,j]=key[store];self.target[take,j]=value[store];self.validity[take,j]=quality[store]
        self.nevents[ids]+=1
        return pulse

