"""Local loss of confidence after surprise in a noisy four-food reversal game."""

from dataclasses import asdict,dataclass,replace

import numpy as np

from . import ring_world as r,ring_spikes
from .ring_sensors import ChemicalInterface


@dataclass(frozen=True)
class Release:
    amount:float=0.
    threshold:float=.32
    hysteresis:float=0.
    scope:str="local"
    exploration:float=0.
    # Surprise estimates discard old evidence as new contacts are observed.
    surprise_rate:float=.38


class Relearning(ring_spikes.Forager):
    """Reopen covariance along a recent sensory direction after a surprising sign."""
    def __init__(self,cfg,rule,net,seeds):
        if cfg.method!="rls" or cfg.flavor_overlap:
            raise ValueError("Reversal experiment starts at the frozen independent-receptor RLS solution")
        super().__init__(cfg,net,seeds);self.rule=rule
        self.perm=np.random.default_rng(net+2227).permutation(cfg.latent)

    def reset(self,n):
        super().reset(n)
        self.alarm=np.zeros_like(self.w)
        self.extra=np.zeros(n,np.float32)
        self.nraise=np.zeros(n,np.int32)

    def _learn(self,obs):
        reward=obs[:,r.POSITIVE]-obs[:,r.NEGATIVE]
        who=np.flatnonzero((reward!=0)&np.any(self.last_odor!=0,axis=1))
        if who.size and (self.rule.amount>0 or self.rule.exploration>0):
            item=self.last_odor[who].argmax(1);x=self.last_key[who]
            prior=np.einsum("bi,bi->b",self.alarm[who,item],x)
            surprising=np.maximum(0.,np.clip(-reward[who]*self.last_pred[who],-1.,1.)-self.rule.threshold)
            gate=surprising*np.clip(prior/self.rule.hysteresis,0.,1.) if self.rule.hysteresis>0 else surprising
            self.alarm[who,item]+=self.rule.surprise_rate*(surprising-prior)[:,None]*x
            amount=self.rule.amount*gate
            if self.rule.scope=="uniform":
                n=x.shape[1]
                injection=np.eye(n,dtype=np.float32)[None]*((x*x).sum(1)/n)[:,None,None]
            else:
                direction=x[:,self.perm] if self.rule.scope=="permuted" else x
                injection=direction[:,:,None]*direction[:,None,:]
            self.p[who,item]+=amount[:,None,None]*injection
            self.extra[who]+=amount
            self.nraise[who]+=amount>1e-6
        return super()._learn(obs)

    def _explore(self):
        if not self.rule.exploration:
            return super()._explore()
        if self.noise_index==len(self.noise):
            self.noise=np.stack([rng.random(128,dtype=np.float32) for rng in self.rng],axis=1)
            self.noise_index=0
        key=(self.z-self.mean)@self.project
        key/=np.sqrt(.01+(key*key).sum(1,keepdims=True))
        confidence=np.einsum("bfk,bk->bf",self.alarm,key).max(1)
        probability=np.clip(self.cfg.exploration+self.rule.exploration*np.maximum(confidence,0),0.,.6)
        result=self.noise[self.noise_index]<probability
        self.noise_index+=1
        return result


def regime(seeds,trials=224,*,flip=True,restore=False):
    t=r.make_tape(seeds,trials)
    # Change exactly two cue assignments in both contexts. All public stimuli
    # follow the stationary tape, and both per-context maps remain permutations.
    pairs=np.stack([np.random.default_rng(np.random.SeedSequence([s,37053])).choice(4,2,replace=False)
                    for s in seeds])
    selected=(t.cue==pairs[:,0,None])|(t.cue==pairs[:,1,None])
    counter=np.where(t.cue==pairs[:,0,None],pairs[:,1,None],
                     np.where(t.cue==pairs[:,1,None],pairs[:,0,None],t.cue))
    new=t.preferences[np.arange(t.n)[:,None],t.ctx,counter]
    at=trials//2
    segment=np.arange(trials)[None,:]>=at
    if restore:segment &= np.arange(trials)[None,:]<3*trials//4
    if flip:t=replace(t,goal=np.where(segment,new,t.goal))
    return t,selected,at


class OutcomeChannel:
    def __init__(self,agent,seeds,trials,noise):
        self.agent=agent
        self.invert=np.stack([np.random.default_rng(np.random.SeedSequence([s,729451])).random(trials)<noise
                              for s in seeds])
    def reset(self,n):
        self.agent.reset(n);self.j=-1
    def act(self,obs):
        if np.all(obs[:,r.NEW]):self.j+=1
        frame=obs.copy()
        flip=self.invert[:,self.j]
        frame[flip,r.POSITIVE]=obs[flip,r.NEGATIVE];frame[flip,r.NEGATIVE]=obs[flip,r.POSITIVE]
        return self.agent.act(frame)


def rate_by_life(hit,mask):
    num=(hit&mask).sum(1);den=mask.sum(1)
    return {"hit":num.tolist(),"count":den.tolist(),
            "pooled":float(num.sum()/den.sum()) if den.sum() else None}


def evaluation(config,rule,graph,seeds,noise,*,trials=224,flip=True,restore=False):
    t,affected,at=regime(seeds,trials,flip=flip,restore=restore)
    p=Relearning(config,rule,graph,seeds)
    result=r.rollout(OutcomeChannel(p,seeds,trials,noise),t)
    tick=np.arange(trials)[None,:];early=tick<at;post=~early
    spans={"pre_last":np.broadcast_to(early&(tick>=at-32),affected.shape),
           "post_affected":post&affected,"post_untouched":post&~affected,
           "late_affected":(tick>=trials-32)&affected,"late_untouched":(tick>=trials-32)&~affected}
    # Rank the encounters of the exact (context, cue) independently, rather
    # than treating contact with B as direct experience of the new A reward.
    ranks=np.zeros_like(t.cue,dtype=int)
    for c in (0,1):
        for k in range(4):
            v=(t.ctx==c)&(t.cue==k)&affected&post
            ranks+=v*np.cumsum(v,1)
    for lo,hi in ((1,1),(2,3),(4,6),(7,1000)):
        spans[f"changed_exposure_{lo}_{hi}"]=(ranks>=lo)&(ranks<=hi)
    return {**{name:rate_by_life(result.first_correct,v) for name,v in spans.items()},
            "first_rate":result.first_correct.mean(1).tolist(),"coverage":(result.first>=0).mean(1).tolist(),
            "released":p.extra.tolist(),"release_events":p.nraise.tolist()}
