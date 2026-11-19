"""Observation interventions applied outside the physics and private reward mapping."""

import numpy as np

from . import world


class BriefCue:
    """Expose only the first specified ticks of the joint background/cue stimulus."""

    def __init__(self, policy, ticks):
        if not 0 <= ticks <= world.TICKS:
            raise ValueError("cue exposure outside the trial")
        self.policy,self.ticks=policy,ticks

    def reset(self,n):
        self.clock=-1
        self.policy.reset(n)

    def act(self,obs):
        self.clock=0 if np.all(obs[:,world.TRIAL_START]) else self.clock+1
        if self.clock>=self.ticks:
            obs=obs.copy()
            obs[:,world.CTX]=0
            obs[:,world.CUE]=0
        return self.policy.act(obs)


class ReceptorCurrent:
    """A saturating postsynaptic sensory current for each visible identity channel.

    No trial resets and no outcome input to this trace. Geometry, motor state
    and reward remain instantaneous. This is an explicit sensory-memory prior.
    """

    def __init__(self,policy,decay):
        if not 0<=decay<1:
            raise ValueError("sensory decay must be in [0,1)")
        self.policy,self.decay=policy,np.float32(decay)

    def reset(self,n):
        self.current=np.zeros((n,10),np.float32)
        self.policy.reset(n)

    def act(self,obs):
        self.current*=self.decay
        np.maximum(self.current,obs[:,:10],out=self.current)
        frame=obs.copy()
        frame[:,:10]=self.current
        return self.policy.act(frame)
