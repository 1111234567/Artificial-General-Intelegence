"""A directed spiking recurrent graph with local, outcome-modulated motor synapses.

There is no object-choice stage: the two output neurons fire directly into the
two body actuators. The outcome carries polarity only to the learning rule.
"""

from dataclasses import dataclass

import numpy as np

from . import world


@dataclass(frozen=True)
class MotorConfig:
    sensors: int = 94
    feature_leak: float = .62
    feature_scale: float = .56
    recurrence: float = .22
    pre_decay: float = .72
    background_rate: float = .018
    motor_leak: float = .82
    tonic: float = .36
    self_excitation: float = .95
    cross_inhibition: float = 1.10
    motor_noise: float = .15
    motor_gain: float = .75
    eligibility_decay: float = .88
    learning_rate: float = .52
    error_correct: bool = True
    onset_credit: bool = True


class MotorLife:
    """Population-batched LIF dynamics. Only reset() initializes a new life."""

    def __init__(self, cfg, net_seed, life_seeds, *, freeze_after=None, mute_recurrence=False):
        self.cfg = cfg
        self.seeds = tuple(int(s) for s in life_seeds)
        self.freeze_after = freeze_after
        self.net_seed = net_seed
        rng = np.random.default_rng(net_seed)
        n = cfg.sensors
        projection = rng.normal(0, cfg.feature_scale, (world.OBS_SIZE,n)) * (rng.random((world.OBS_SIZE,n)) < .66)
        # Complementary left/right labels and movement are sensory inputs. Their
        # relative scale is a birth prior, independent of the private preferences.
        projection[world.POSITION:] *= .12
        self.inputs = projection.astype(np.float32)
        self.threshold = rng.uniform(.8,1.2,n).astype(np.float32)
        self.bias = rng.normal(.13,.068,n).astype(np.float32)
        edges = rng.random((n,n)) < .09
        np.fill_diagonal(edges,False)
        edges[np.arange(n),(np.arange(n)+1)%n] = True
        signs = np.where(rng.random(n)<.3,-1.7,1.)
        rec = edges * rng.uniform(.2,1.8,(n,n)) * signs[:,None] * (cfg.recurrence / np.sqrt(.09*n))
        self.recurrent = (rec * (not mute_recurrence)).astype(np.float32)
        self.feedback = ((rng.random((2,n)) < .16)*rng.normal(0,.11,(2,n))*(not mute_recurrence)).astype(np.float32)
        self.birth = rng.normal(0,.024,(2,n)).astype(np.float32)
        self.plastic_mask = (rng.random((2,n)) < .84).astype(np.float32)

    def reset(self,n):
        if n!=len(self.seeds):
            raise ValueError("one noise seed per life required")
        units = self.cfg.sensors
        self.h = np.zeros((n,units),np.float32)
        self.spikes = np.zeros_like(self.h)
        self.pre = np.zeros_like(self.h)
        self.mean = np.full_like(self.h,.14)
        self.motor = np.zeros((n,2),np.float32)
        self.motor_v = np.zeros_like(self.motor)
        self.weights = np.zeros((n,2,units),np.float32)
        self.elig = np.zeros_like(self.weights)
        self.mass = np.zeros_like(self.motor)
        self.rngs = [np.random.default_rng(np.random.SeedSequence([s,0x50B8A1])) for s in self.seeds]
        self._noise = np.empty((0,n,2),np.float32)
        self._noise_i = 0
        self.tick = 0
        self.trial = -1
        self.spike_sum = np.zeros(n,np.int32)
        self.write_sum = np.zeros(n,np.int32)

    def _noise_at_tick(self):
        if self._noise_i >= len(self._noise):
            self._noise = np.stack([r.standard_normal((128,2),dtype=np.float32) for r in self.rngs],axis=1)
            self._noise_i = 0
        value = self._noise[self._noise_i]
        self._noise_i+=1
        return value

    def act(self,obs):
        cfg=self.cfg
        if np.all(obs[:,world.TRIAL_START]):
            self.trial+=1
        feedback = obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        events = feedback!=0
        if cfg.learning_rate and events.any() and (self.freeze_after is None or self.trial < self.freeze_after):
            # Synapses at each actual motor cell estimate return from their
            # own pre/post trace; no label, contact identity or chosen side is supplied.
            e = self.elig[events] / np.maximum(self.mass[events,:,None],.25)
            q = np.einsum("bki,bki->bk",self.weights[events],e) if cfg.error_correct else 0.
            error = feedback[events,None] - q
            norm = .12 + np.einsum("bki,bki->bk",e,e)
            change = cfg.learning_rate * (error/norm)[:,:,None] * self.plastic_mask * e
            self.weights[events] = np.clip(self.weights[events]+change,-1.25,1.25)
            self.write_sum[events] += 1
        features = self.pre-self.mean
        motor_value = features@self.birth.T + np.einsum("bki,bi->bk",self.weights,features)
        active = obs[:,world.ACTIVE,None]
        drive = (cfg.tonic*active - .8*(1-active) + cfg.motor_gain*np.tanh(motor_value)
                 + cfg.motor_noise*self._noise_at_tick() + cfg.self_excitation*self.motor
                 - cfg.cross_inhibition*self.motor[:,::-1])
        self.motor_v = np.maximum(cfg.motor_leak*self.motor_v + drive,-1.)
        next_motor = (self.motor_v >= 1.).astype(np.float32)
        self.motor_v *= 1.-next_motor
        self.h = np.maximum(cfg.feature_leak*self.h + self.bias + obs@self.inputs +
                            self.spikes@self.recurrent + self.motor@self.feedback,-2.)
        self.spikes = (self.h >= self.threshold).astype(np.float32)
        self.h *= 1.-self.spikes

        post = next_motor*(1.-self.motor) if cfg.onset_credit else next_motor
        self.elig *= cfg.eligibility_decay
        self.elig += post[:,:,None]*features[:,None,:]
        self.mass *= cfg.eligibility_decay
        self.mass += post
        self.pre *= cfg.pre_decay
        self.pre += (1-cfg.pre_decay)*self.spikes
        self.mean += cfg.background_rate*(self.pre-self.mean)
        self.motor=next_motor
        self.tick+=1
        self.spike_sum += self.spikes.sum(1,dtype=np.int32)
        return next_motor
