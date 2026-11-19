"""One-band synaptic memory with a locally learned cross-habitat write-to-read gain."""

from dataclasses import dataclass

import numpy as np

from . import mixture_world as taste, world


def normalized(x):
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, length, out=np.zeros_like(x), where=length > 1e-7)


@dataclass(frozen=True)
class Skill:
    encoding: str = "opponent"
    nodes: int = 32
    recurrence: float = .36
    signal: float = 1.36
    receptor: float = .45
    feature_trace: float = .67
    leak: float = .55
    read_clock: int = 5
    memory_hold: float = .84
    gain_hold: float = .965
    gain_rate: float = .30
    update_gain: str = "delta"
    fixed_gain: float | None = None
    share_habitats: bool = False
    address_power: int = 1
    centering: float = .12
    meta_fast_hold: float = 0.

    def __post_init__(self):
        if self.encoding not in ("cubic", "linear", "even", "opponent", "random", "centered", "voltage"):
            raise ValueError("unknown odor encoding")
        if self.update_gain not in ("delta", "correlation") or not 4 <= self.read_clock <= 7:
            raise ValueError("invalid readout mode or motor deadline")
        if self.address_power not in (1, 2) or not 0 <= self.centering <= 1:
            raise ValueError("invalid address expansion or homeostatic rate")
        if not 0 <= self.meta_fast_hold < 1:
            raise ValueError("invalid fast metaplastic hold")


class SpikingSkill:
    """Local reward-dependent updates of distributed weight vectors.

    The two public habitat receptors gate two projection fields. All contacts
    contribute to one gain synapse, unless it is explicitly held fixed. The
    spiking encoders do not receive a related-side marker or latent family.
    Cubic, linear, and even are analytic sensory references.
    """

    def __init__(self, skill=Skill(), *, birth=0, action_seeds=None):
        self.skill = skill
        self.action_seeds = None if action_seeds is None else tuple(action_seeds)
        self.birth = birth
        rng = np.random.default_rng(birth)
        h = skill.nodes
        self.aff = (rng.normal(0., .96, (4, h)) * (rng.random((4, h)) < .75)).astype(np.float32)
        self.bias = rng.uniform(.026, .104, h).astype(np.float32)
        self.threshold = rng.uniform(.86, 1.16, h).astype(np.float32)
        edges = rng.random((h, h)) < .16
        np.fill_diagonal(edges, False)
        edges[np.arange(h), (np.arange(h) + 1) % h] = True
        sign = np.where(rng.random(h) < .28, -1.65, 1.)
        self.rec = (edges * rng.uniform(.042, .31, (h, h)) * sign[:, None] * skill.recurrence).astype(np.float32)
        self.dims = {"cubic": 64, "linear": 4, "even": 16}.get(skill.encoding, h ** skill.address_power)
        self.spike_limbs = 2 if skill.encoding == "opponent" else 1

    def reset(self, n):
        self.weights = np.zeros((n, 2, self.dims), dtype=np.float32)
        self.fast_weights = np.zeros_like(self.weights) if self.skill.meta_fast_hold else None
        self.direction = np.zeros(n, dtype=np.float32)
        self.chem = np.zeros((n, 2, 4), dtype=np.float32)
        self.sense = np.zeros_like(self.chem)
        shape = (n, 2, self.spike_limbs, self.skill.nodes)
        self.v = np.zeros(shape, dtype=np.float32)
        self.spikes = np.zeros_like(self.v)
        self.z = np.zeros_like(self.v)
        self.mean = np.full((n, self.skill.nodes), .06, dtype=np.float32)
        self.gate = np.zeros(n, dtype=np.int8)
        self.last_field = self.gate.copy()
        self.pending = np.zeros(n, dtype=bool)
        self.key = np.zeros((n, 2, self.dims), dtype=np.float32)
        self.recall = np.zeros((n, 2), dtype=np.float32)
        self.meta_recall = np.zeros_like(self.recall)
        self.motor_v = np.zeros((n, 2), dtype=np.float32)
        self.motor_spikes = np.zeros_like(self.motor_v)
        self.picks = np.zeros(n, dtype=np.int8)
        self.clock, self.trial = -1, -1
        self.total_spikes = np.zeros(n, dtype=np.int64)
        self.receipts = np.zeros(n, dtype=np.int32)
        self.direction_history = []
        self.choices_history = []
        self.recall_energy = []
        seeds = np.random.SeedSequence((self.birth, 640)).spawn(n) if self.action_seeds is None else self.action_seeds
        if len(seeds) != n:
            raise ValueError("Require one motor-noise seed per life")
        self.rng = [np.random.default_rng(s) for s in seeds]

    def sensory(self, obs):
        chem = obs[:, taste.LEFT.start:taste.RIGHT.stop].reshape(len(obs), 2, 4) * 2. - 1.
        received = np.any(chem != 0, axis=(1, 2))
        if self.skill.encoding in ("linear", "cubic", "even"):
            self.chem[received] = chem[received]
        context = obs[:, taste.CTX]
        update = context.any(axis=1)
        self.gate[update] = context[update].argmax(axis=1)
        if self.skill.encoding not in ("linear", "cubic", "even"):
            self.sense *= self.skill.receptor
            self.sense += chem
            projection = self.skill.signal * (self.sense @ self.aff)
            signal = (np.stack((projection, -projection), axis=2) if self.spike_limbs == 2
                      else projection[:, :, None, :])
            self.v *= self.skill.leak
            self.v += self.bias + signal + self.spikes @ self.rec
            np.maximum(self.v, -2., out=self.v)
            np.greater_equal(self.v, self.threshold, out=self.spikes)
            self.v[self.spikes != 0] = 0.
            self.z *= self.skill.feature_trace
            self.z += (1 - self.skill.feature_trace) * self.spikes
            self.total_spikes += self.spikes.sum(axis=(1, 2, 3), dtype=np.int64)

    def address(self):
        e = self.skill.encoding
        if e == "linear":
            return normalized(self.chem)
        if e == "even":
            f = np.einsum("nbi,nbj->nbij", self.chem, self.chem, optimize=False)
        elif e == "cubic":
            f = np.einsum("nbi,nbj,nbk->nbijk", self.chem, self.chem, self.chem, optimize=False)
        elif e in ("opponent", "random", "centered", "voltage"):
            z = self.z[:, :, 0]
            if e == "opponent":
                z = z - self.z[:, :, 1]
            elif e in ("centered", "voltage"):
                if e == "voltage":
                    z = .64 * self.v[:, :, 0] + z
                center = self.mean
                current = z - center[:, None]
                center += self.skill.centering * (z.mean(1) - center)
                z = current
            else:
                z = z - .12
            z = normalized(z)
            if self.skill.address_power == 1:
                return z
            f = np.einsum("nbi,nbj->nbij", z, z, optimize=False)
        return normalized(f.reshape(len(f), 2, self.dims))

    def learn(self, obs):
        tag = obs[:, world.CONTACT_ID]
        sign = obs[:, world.REWARD_POS] - obs[:, world.REWARD_NEG]
        seen = tag.any(axis=1) & self.pending
        informative = seen & (sign != 0)
        rows = np.flatnonzero(informative)
        self.pending[seen] = False
        if not rows.size:
            return
        side = tag[rows].argmax(axis=1)
        earlier = np.clip(self.meta_recall[rows, side], -1.8, 1.8)
        score = sign[rows]
        self.adapt_direction(rows, side, score, earlier)
        field = self.last_field[rows]
        self.weights[rows, field] = self.skill.memory_hold * self.weights[rows, field] + score[:, None] * self.key[rows, side]
        if self.fast_weights is not None:
            self.fast_weights[rows, field] = (self.skill.meta_fast_hold * self.fast_weights[rows, field]
                                              + score[:, None] * self.key[rows, side])
        self.receipts[rows] += 1

    def adapt_direction(self, rows, side, score, earlier):
        if self.skill.fixed_gain is None:
            g = self.direction[rows]
            remainder = score - np.tanh(g) * earlier if self.skill.update_gain == "delta" else score
            self.direction[rows] = self.skill.gain_hold * g + self.skill.gain_rate * remainder * earlier

    def value(self, field):
        self.recall = np.einsum("nbk,nk->nb", self.key, self.weights[np.arange(len(field)), field], optimize=False)
        if self.fast_weights is None:
            self.meta_recall = self.recall
        else:
            recent = np.einsum("nbk,nk->nb", self.key, self.fast_weights[np.arange(len(field)), field], optimize=False)
            self.meta_recall = self.recall - .88 * recent
        gain = self.skill.fixed_gain if self.skill.fixed_gain is not None else np.tanh(self.direction)
        return self.recall * np.broadcast_to(gain, (len(field),))[:, None]

    def reverse_direction(self):
        self.direction *= -1

    def act(self, obs):
        self.clock += 1
        if obs[0, world.TRIAL_START]:
            self.clock = 0
            self.trial += 1
        self.learn(obs)
        self.sensory(obs)
        active = obs[:, world.ACTIVE] > 0
        self.motor_v *= .45
        self.motor_v += 1.18 * self.motor_spikes - .34 * self.motor_spikes[:, ::-1]
        if self.clock == self.skill.read_clock:
            field = np.zeros_like(self.gate) if self.skill.share_habitats else self.gate
            self.key = self.address()
            value = self.value(field)
            tie = np.array([r.integers(2) for r in self.rng], dtype=np.int8)
            self.picks = np.where(np.abs(value[:, 0] - value[:, 1]) <= 1e-7, tie, value[:, 1] > value[:, 0])
            self.motor_v[np.flatnonzero(active), self.picks[active]] += 2.45
            self.last_field[:] = field
            self.pending[:] = active
            self.direction_history.append(self.direction.copy())
            self.choices_history.append(self.picks.copy())
            self.recall_energy.append(np.abs(self.recall).max(axis=1))
        self.motor_v[~active | (obs[:, world.TRIAL_START] > 0)] -= 2.2
        np.maximum(self.motor_v, -1., out=self.motor_v)
        np.greater_equal(self.motor_v, 1., out=self.motor_spikes)
        self.motor_v[self.motor_spikes != 0] = 0.
        return self.motor_spikes.astype(bool)
