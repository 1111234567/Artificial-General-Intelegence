"""Experience-triggered removal of a contradicted association in a distributed spike code."""

from dataclasses import dataclass

import numpy as np

from . import ring_spikes,ring_world as ring


@dataclass(frozen=True)
class ResetRule:
    criterion:float=2.4
    decay:float=.93
    confidence:float=.24
    erase:float=.85
    reopen:float=2.0
    scope:str="cue"
    reset_score:float=0.
    trace_damping:float=.035


class SurpriseForager(ring_spikes.Forager):
    """No task or reversal ID. The change detector only sees its own valence errors."""
    def __init__(self,cfg,rule,net,seeds,*,matched_pulses=None):
        if cfg.method!="rls" or cfg.flavor_overlap:
            raise ValueError("Use the one-hot sensory reference")
        super().__init__(cfg,net,seeds);self.rule=rule
        if rule.scope not in ("cue","single_receptor","permuted","whole","unchanged"):
            raise ValueError(rule.scope)
        self.perm=np.random.default_rng(net+8133).permutation(cfg.latent)
        self.matched=matched_pulses

    def reset(self,n):
        super().reset(n)
        self.surprise=np.zeros_like(self.last_key)
        self.injections=np.zeros(n,np.int32)
        self.erased=np.zeros(n,np.float32)
        self.episodes=[]
        self.tick=-1

    def _learn(self,obs):
        self.tick+=1
        r=obs[:,ring.POSITIVE]-obs[:,ring.NEGATIVE]
        who=np.flatnonzero((r!=0)&self.last_odor.any(1))
        if who.size:
            rule=self.rule
            f=self.last_odor[who].argmax(1);x=self.last_key[who]
            pred=self.last_pred[who]
            cov=self.p[who,f];z=(cov@x[:,:,None]).squeeze(-1)
            lev=(z*x).sum(1)
            direction=z/(rule.trace_damping+lev[:,None])
            current=(self.surprise[who]*x).sum(1)
            # Contradictions count when the old model was confident. An
            # expected observation removes evidence for a change. No life clock.
            strength=np.clip(abs(pred),0.,1.)
            delta=np.where(r[who]*pred<0,1.,-1.)*np.maximum(0.,strength-rule.confidence)/(1.-rule.confidence)
            following=np.maximum(0.,rule.decay*current+delta)
            raise_alarm=following>=rule.criterion
            if self.matched is not None:
                raise_alarm=self.matched[who,self.tick]
            self.surprise[who]+=(np.where(raise_alarm,0.,following)-current)[:,None]*direction
            idx=who[raise_alarm]
            if idx.size:
                feature=x[raise_alarm];odor=f[raise_alarm]
                self.injections[idx]+=1
                self.episodes.extend((self.tick,int(i)) for i in idx)
                if rule.scope!="unchanged":
                    n=len(idx);k=feature.shape[1]
                    # Each receptor's sensitivity is expressed in its own
                    # coactivity geometry; use the same current neural code.
                    bank=self.p[idx]
                    src=feature[:,self.perm] if rule.scope=="permuted" else feature
                    zs=(bank@src[:,None,:,None]).squeeze(-1)
                    d=zs/(rule.trace_damping+(zs*src[:,None,:]).sum(2))[:,:,None]
                    old=self.w[idx]
                    change=rule.erase*(rule.reset_score-(old*src[:,None,:]).sum(2))[:,:,None]*d
                    if rule.scope=="single_receptor":
                        change*=np.eye(4,dtype=np.float32)[odor,:,None]
                    if rule.scope=="whole":
                        # Same expected L2 damage as the local rule, distributed
                        # over every old synapse in proportion to its magnitude.
                        budget=np.linalg.norm(change.reshape(n,-1),axis=1)
                        weight_norm=np.linalg.norm(old.reshape(n,-1),axis=1)
                        change=-old*(budget/np.maximum(weight_norm,1e-6))[:,None,None]
                    self.w[idx]=old+change
                    self.erased[idx]+=np.linalg.norm(change.reshape(n,-1),axis=1)
                    if rule.reopen:
                        if rule.scope=="whole":
                            self.p[idx]+=rule.reopen*np.eye(k,dtype=np.float32)[None,None]/k
                        else:
                            add=rule.reopen*src[:,:,None]*src[:,None,:]
                            if rule.scope=="single_receptor":self.p[idx,odor]+=add
                            else:self.p[idx]+=add[:,None,:,:]
                    # The incoming sample is now used once against the revised forecast.
                    self.last_pred[idx]=np.einsum("bi,bi->b",self.w[idx,odor],feature)
        return super()._learn(obs)
