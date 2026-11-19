"""Vectorized receptor-gated LIF foraging with local one-bit taste credit."""

from dataclasses import dataclass

import numpy as np

from .ring_world import ACTIVE, CTX, CUE, FRUIT, NEGATIVE, NEW, POSITIVE, REACH, SIZE, TASTED


@dataclass(frozen=True)
class ForagingSpec:
    units: int = 64
    heads: int = 4
    representation: str = "random"
    outcomes: str = "signed"
    learning_rate: float = .10
    cov_rate: float = .055
    exploration: float = .12
    feedback_delay: int = 0
    trace_decay: float = .74
    recurrent_gain: float = 1.
    eligibility_sparsity: float = 0.
    birth_threshold: float = 1.
    scoring: str = "spiking"
    gait: str = "clockwise"
    drop_context: bool = False
    drop_cue: bool = False

    def __post_init__(self):
        if self.representation not in {"random", "coincidence"}:
            raise ValueError("invalid feature representation")
        if self.outcomes not in {"signed", "negative", "positive", "contact", "frozen"}:
            raise ValueError("invalid outcome mode")
        if self.scoring not in {"spiking", "linear", "blind"}:
            raise ValueError("invalid association readout")
        if self.gait not in {"clockwise", "counterclockwise", "still"}:
            raise ValueError("invalid gait")
        if self.units < 16 or self.heads < 1 or not 0 <= self.eligibility_sparsity < 1:
            raise ValueError("invalid population size or activity sparsity")
        if self.feedback_delay not in (0, 1, 2, 4):
            raise ValueError("unsupported credit-delay intervention")


class ReceptorForager:
    """An identity cohort may facilitate a bite only when that fruit is visible.

    The clockwise pattern generator ignores fruit identity. The four learned
    responses implement a shared inhibitory threshold for the visible cohort.
    On receipt, only the tasted cohort receives its stored eligibility update.
    """

    def __init__(self, spec=ForagingSpec(), *, birth_seed=0, action_seeds=None):
        self.spec = spec
        self.birth_seed = int(birth_seed)
        self.action_seeds = None if action_seeds is None else tuple(action_seeds)
        rng = np.random.default_rng(birth_seed)
        h = spec.units
        groups = rng.permutation(h) % 4
        inner = groups[:, None] == groups[None, :]
        edges = rng.random((h, h)) < np.where(inner, .18, .05)
        np.fill_diagonal(edges, False)
        signs = np.where(rng.random(h) < .23, -1.7, 1).astype(np.float32)
        self.recurrent = rng.uniform(.018, .075, (h, h)).astype(np.float32) * edges * signs
        self.recurrent *= spec.recurrent_gain
        # Only the six separate context and cue channels enter this reservoir.
        self.aff = rng.normal(.24, .54, (h, 6)).astype(np.float32)
        self.aff *= rng.random((h, 6)) < .64
        self.bias = rng.uniform(.035, .10, h).astype(np.float32)
        self.threshold = (rng.uniform(.9, 1.10, h) * spec.birth_threshold).astype(np.float32)
        self.mask = rng.random((SIZE, spec.heads, h)) < .72
        self.head_threshold = np.linspace(.71, 1.29, spec.heads, dtype=np.float32)
        self.coincidence = np.arange(h) % 8

    def reset(self, n):
        h, heads = self.spec.units, self.spec.heads
        self.v = np.zeros((n, h), dtype=np.float32)
        self.spike = np.zeros_like(self.v)
        self.trace = np.zeros_like(self.v)
        self.mean = np.full_like(self.v, .20)
        self.weights = np.zeros((n, SIZE, heads, h), dtype=np.float32)
        self.lateral = (np.tile(np.eye(h, dtype=np.float32), (n, 1, 1)) if self.spec.cov_rate else None)
        self.head_v = np.zeros((n, SIZE, heads), dtype=np.float32)
        self.head_spike = np.zeros_like(self.head_v)
        self.head_trace = np.zeros_like(self.head_v)
        self.efferent_v = np.zeros((n, 2), dtype=np.float32)
        self.tasted = np.zeros((n, SIZE), dtype=np.float32)
        self.last_motor_receptor = np.zeros_like(self.tasted)
        self.eligibility = np.zeros_like(self.v)
        self.reward_prediction = np.zeros((n, heads), dtype=np.float32)
        slots = max(1, self.spec.feedback_delay)
        self.delay_direction = np.zeros((slots, n, h), dtype=np.float32)
        self.delay_mask = np.zeros((slots, n, SIZE), dtype=np.float32)
        self.delay_prediction = np.zeros((slots, n, heads), dtype=np.float32)
        self.delay_receipt = np.zeros((slots, n), dtype=np.float32)
        self.last_coactivity = np.zeros(n, dtype=bool)
        self.recurrent_history = np.zeros(n, dtype=np.int64)
        self.association_history = np.zeros(n, dtype=np.int64)
        self.receipts = np.zeros((n, 2), dtype=np.int32)
        self.writes = np.zeros((n, 2), dtype=np.int32)
        self.write_norm_sum = np.zeros(n, dtype=np.float64)
        self.write_edges = np.zeros(n, dtype=np.int64)
        self.lateral_writes = np.zeros(n, dtype=np.int32)
        self.bite_pulse_count = np.zeros(n, dtype=np.int64)
        self.random_bias = np.zeros((n, SIZE), dtype=np.float32)
        self.clock = -1
        self.total_ticks = -1
        if self.action_seeds is None:
            self.noise = [np.random.default_rng(s) for s in np.random.SeedSequence((self.birth_seed, 716)).spawn(n)]
        elif len(self.action_seeds) == n:
            self.noise = [np.random.default_rng(s) for s in self.action_seeds]
        else:
            raise ValueError("one action seed required per life")

    def _new_input(self):
        self.clock = 0
        self.tasted.fill(0)
        self.last_coactivity.fill(False)
        draws = np.stack([rng.random(SIZE + 1, dtype=np.float32) for rng in self.noise])
        self.random_bias = draws[:, 1:] * .006
        self.random_bias[draws[:, 0] < self.spec.exploration] *= 1000.

    def _internal_feature(self, observation):
        stim = observation[:, :CUE.stop]
        if self.spec.drop_context or self.spec.drop_cue:
            stim = stim.copy()
            if self.spec.drop_context:
                stim[:, CTX] = 0
            if self.spec.drop_cue:
                stim[:, CUE] = 0
        self.v *= .55
        if self.spec.representation == "coincidence":
            pairs = (stim[:, CTX, None] * stim[:, None, CUE]).reshape(len(stim), 8)
            self.v += pairs[:, self.coincidence] * 1.24
        else:
            self.v += stim @ self.aff.T
        self.v += self.bias + self.spike @ self.recurrent.T
        np.greater_equal(self.v, self.threshold, out=self.spike)
        self.v[self.spike != 0] = 0
        self.trace *= self.spec.trace_decay
        self.trace += (1 - self.spec.trace_decay) * self.spike
        self.recurrent_history += self.spike.sum(axis=1, dtype=np.int64)
        return self.trace - self.mean

    def _receipt(self, observation):
        contact = observation[:, TASTED]
        physical = contact.any(axis=1)
        if not np.any(physical):
            return
        self.tasted[physical] = np.maximum(self.tasted[physical], contact[physical])
        aligned = (contact * self.last_motor_receptor).sum(axis=1) > 0
        sign = observation[:, POSITIVE] - observation[:, NEGATIVE]
        self.receipts[:, 0] += aligned & (sign > 0)
        self.receipts[:, 1] += aligned & (sign < 0)
        mode = self.spec.outcomes
        if mode == "positive":
            sign = np.maximum(sign, 0.)
        elif mode == "negative":
            sign = np.minimum(sign, 0.)
        elif mode == "contact":
            sign = physical.astype(np.float32)
        elif mode == "frozen":
            return
        use = aligned & (sign != 0)
        if np.any(use) and self.spec.feedback_delay:
            slot = self.total_ticks % self.spec.feedback_delay
            self.delay_receipt[slot, use] = sign[use]
            self.delay_direction[slot, use] = self.eligibility[use]
            self.delay_mask[slot, use] = contact[use]
            self.delay_prediction[slot, use] = self.reward_prediction[use]
        elif np.any(use):
            self._learn(use, contact, sign, self.eligibility, self.reward_prediction)

    def _delayed(self):
        if not self.spec.feedback_delay:
            return
        slot = self.total_ticks % self.spec.feedback_delay
        sign = self.delay_receipt[slot]
        use = sign != 0
        if np.any(use):
            fraction = .72**self.spec.feedback_delay
            self._learn(use, self.delay_mask[slot], sign, fraction * self.delay_direction[slot],
                        self.delay_prediction[slot])
            sign[use] = 0.

    def _learn(self, use, contact, sign, eligibility, prediction):
        rate = self.spec.learning_rate
        if rate == 0:
            return
        rows = np.flatnonzero(use)
        fruit = np.argmax(contact[rows], axis=1)
        delta = rate * (sign[rows, None] - prediction[rows])[:, :, None] * eligibility[rows, None, :]
        delta *= self.mask[fruit]
        initial = self.weights[rows, fruit]
        revised = np.clip(initial + delta, -.7, .7)
        self.weights[rows, fruit] = revised
        actual = (revised - initial).reshape(len(rows), -1)
        self.write_norm_sum[rows] += np.linalg.norm(actual, axis=1)
        self.write_edges[rows] += np.count_nonzero(actual, axis=1)
        self.writes[rows, 0] += sign[rows] > 0
        self.writes[rows, 1] += sign[rows] < 0

    def _capture(self, fire, feature, visible, current):
        rows = np.flatnonzero(fire)
        x = feature[rows]
        direction = x.copy()
        if self.lateral is not None:
            matrix = self.lateral[rows]
            z = np.einsum("nij,nj->ni", matrix, x, optimize=False)
            gain = np.linalg.norm(x, axis=1) / np.maximum(np.linalg.norm(z, axis=1), 1e-8)
            direction = z * gain[:, None]
            first = ~self.last_coactivity[rows]
            observed = np.maximum(np.einsum("ni,ni->n", x[first], z[first]), 0.)
            rate = self.spec.cov_rate
            z0 = z[first]
            transform = rate / (1 + rate * observed)
            self.lateral[rows[first]] -= transform[:, None, None] * z0[:, :, None] * z0[:, None, :]
            self.lateral_writes[rows[first]] += 1
        if self.spec.eligibility_sparsity:
            cutoff = max(1, int(direction.shape[1] * self.spec.eligibility_sparsity))
            threshold = np.partition(np.abs(direction), cutoff - 1, axis=1)[:, cutoff - 1]
            direction = np.where(np.abs(direction) > threshold[:, None], direction, 0.)
        self.eligibility[rows] = direction
        selected = np.einsum("nf,nfh->nh", visible[rows], current[rows], optimize=False)
        self.reward_prediction[rows] = np.tanh(2.5 * selected)
        self.last_motor_receptor[rows] = visible[rows]
        first = ~self.last_coactivity[rows]
        self.mean[rows[first]] += .018 * feature[rows[first]]
        self.last_coactivity[rows] = True
        self.bite_pulse_count[rows] += 1

    def _valence(self, current):
        if self.spec.scoring == "linear":
            return np.tanh(current).mean(axis=2)
        return self.head_trace.mean(axis=2) + .14 * np.tanh(current).mean(axis=2)

    def act(self, observation):
        self.clock += 1
        self.total_ticks += 1
        if observation[0, NEW]:
            self._new_input()
        self._delayed()
        self._receipt(observation)
        feature = self._internal_feature(observation)
        current = np.einsum("nfhd,nd->nfh", self.weights, feature, optimize=False)
        self.head_v *= .52
        self.head_v += .46 + current
        np.maximum(self.head_v, -1., out=self.head_v)
        np.greater_equal(self.head_v, self.head_threshold, out=self.head_spike)
        self.head_v[self.head_spike != 0] = 0
        self.head_trace *= .68
        self.head_trace += .32 * self.head_spike
        self.association_history += self.head_spike.sum(axis=(1, 2), dtype=np.int64)
        score = self._valence(current)
        score += self.random_bias
        score[self.tasted > .5] = -20
        visible = observation[:, FRUIT]
        acceptable = np.einsum("nf,nf->n", score, visible, optimize=False) >= score.max(axis=1) - 1e-6
        if self.spec.scoring == "blind":
            acceptable[:] = True
        active = observation[:, ACTIVE] > 0
        active_reach = active & (observation[:, REACH] > 0) & visible.any(axis=1)
        energize = active_reach & acceptable & (self.clock >= 3)
        self.efferent_v *= .4
        self.efferent_v[:, 0] += 1.23 * active
        self.efferent_v[:, 1] += 1.23 * energize
        motor = self.efferent_v >= 1
        self.efferent_v[motor] = 0
        self.last_motor_receptor.fill(0)
        if motor[:, 1].any():
            self._capture(motor[:, 1], feature, visible, current)
        left = motor[:, 0] & (self.spec.gait == "counterclockwise")
        right = motor[:, 0] & (self.spec.gait == "clockwise")
        return np.column_stack((left, right, motor[:, 1]))

    def stats(self):
        return {
            "recurrent_edges": int(np.count_nonzero(self.recurrent)),
            "association_edges": int(self.mask.sum()),
            "lateral_states_per_life": self.spec.units ** 2 if self.lateral is not None else 0,
            "reservoir_spikes_per_life": self.recurrent_history.tolist(),
            "association_spikes_per_life": self.association_history.tolist(),
            "receipts_per_life": self.receipts.tolist(),
            "writes_per_life": self.writes.tolist(),
            "l2_write_sum_per_life": self.write_norm_sum.tolist(),
            "effective_edge_writes_per_life": self.write_edges.tolist(),
            "lateral_writes_per_life": self.lateral_writes.tolist(),
            "bite_pulses_per_life": self.bite_pulse_count.tolist(),
        }
