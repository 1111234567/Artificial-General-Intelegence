"""Two local update timescales, selected by their errors on actually tasted outcomes."""

from dataclasses import dataclass

import numpy as np

from . import ring_world as r,ring_spikes


@dataclass(frozen=True)
class Gates:
    fast_rate:float=.9
    fast_ridge:float=.04
    gate_rate:float=.55
    gain:float=4.
    prior:float=-1.65
    scope:str="local"
    exploration:float=0.


class Arbiter(ring_spikes.Forager):
    def __init__(self,cfg,rules,net,seeds):
        if cfg.method!="rls" or cfg.flavor_overlap:raise ValueError("The reference stores one covariance per taste receptor")
        if rules.scope not in ("local","global","fixed","fast","slow","inverted"):raise ValueError("unknown arbitration")
        super().__init__(cfg,net,seeds);self.rules=rules

    def reset(self,n):
        super().reset(n)
        self.slow=np.zeros_like(self.w);self.quick=np.zeros_like(self.w)
        self.evidence=np.zeros_like(self.w);self.alarm=np.zeros_like(self.last_key)
        self.global_evidence=np.zeros(n,np.float32)
        self.last_mix=np.zeros_like(self.last_odor)
        self.sum_gate=np.zeros(n,np.float32);self.decision_count=np.zeros(n,np.int32)
        self.global_update=np.zeros(n,np.float32);self.gate_at_read=np.zeros(n,np.float32)

    def _learn(self,obs):
        rules=self.rules
        reward=obs[:,r.POSITIVE]-obs[:,r.NEGATIVE]
        who=np.flatnonzero((reward!=0)&np.any(self.last_odor!=0,axis=1))
        if who.size:
            idx=self.last_odor[who].argmax(1);x=self.last_key[who]
            sw=self.slow[who,idx];fw=self.quick[who,idx]
            qs=(sw*x).sum(1);qf=(fw*x).sum(1)
            evidence=self.evidence[who,idx];qe=(evidence*x).sum(1)
            # Payoff sign is external. Compare pre-update likelihoods; these
            # numbers are available internally before either predictor is changed.
            signal=reward[who]
            likelihood_s=np.clip((1.+signal*qs)*.5,.035,.965)
            likelihood_f=np.clip((1.+signal*qf)*.5,.035,.965)
            adv=np.log(likelihood_f)-np.log(likelihood_s)
            P=self.p[who,idx]
            z=(P@x[:,:,None]).squeeze(-1)
            leverage=(x*z).sum(1)
            self.p[who,idx]=P-z[:,:,None]*z[:,None,:]/(1.+leverage)[:,None,None]
            slow_step=self.cfg.learning_rate/(1.+leverage)
            fast_step=rules.fast_rate/(rules.fast_ridge+leverage)
            self.slow[who,idx]+=((signal-qs)*slow_step)[:,None]*z
            self.quick[who,idx]+=((signal-qf)*fast_step)[:,None]*z
            self.evidence[who,idx]+=(rules.gate_rate*(adv-qe)/(.035+leverage))[:,None]*z
            self.global_evidence[who]+=rules.gate_rate*(adv-self.global_evidence[who])
            if rules.exploration:
                surprise=np.maximum(0.,-signal*np.clip(qs,-1.,1.)-.35)
                alarm_prediction=(self.alarm[who]*x).sum(1)
                self.alarm[who]+=.5*(surprise-alarm_prediction)[:,None]*x
            self.writes[who]+=1
            self.sum_gate[who]+=self.last_mix[who,idx];self.decision_count[who]+=1
        # The base policy computes exactly this presynaptic feature before its
        # membrane update, then samples the real visible fruit using a motor spike.
        key=(self.z-self.mean)@self.project
        key/=np.sqrt(.01+(key*key).sum(1,keepdims=True))
        e=np.einsum("bfk,bk->bf",self.evidence,key)
        if rules.scope=="global":e=self.global_evidence[:,None]
        elif rules.scope=="fixed":e=np.zeros_like(e)
        elif rules.scope=="inverted":e=-e
        mix=1./(1.+np.exp(-np.clip(rules.prior+rules.gain*e,-16,16)))
        if rules.scope=="fast":mix=np.ones_like(e)
        elif rules.scope=="slow":mix=np.zeros_like(e)
        self.last_mix[:]=mix
        self.w[:]=self.slow+mix[:,:,None]*(self.quick-self.slow)
        if rules.exploration:self.gate_at_read[:]=np.maximum(0.,(self.alarm*key).sum(1))

    def _explore(self):
        if not self.rules.exploration:return super()._explore()
        if self.noise_index==len(self.noise):
            self.noise=np.stack([v.random(128,dtype=np.float32) for v in self.rng],axis=1)
            self.noise_index=0
        toss=self.noise[self.noise_index];self.noise_index+=1
        return toss<np.clip(self.cfg.exploration+self.rules.exploration*self.gate_at_read,0,.55)
