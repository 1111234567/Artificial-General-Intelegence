"""Online pair allocation from corroborating receipt credit in a physical graph."""

from dataclasses import dataclass

import numpy as np

from . import mixture_world as food, world
from .dendrite_ports_graph import widen_birth
from .unified_live_graph import CONTACT, MOTOR, MOTOR_VECTOR, NO_CUT, SIZE, TASTE, Graph


@dataclass(frozen=True)
class ProbePolicy:
    slots: int = 6
    half_window: int = 6
    evidence: float = 2.0
    minimum_mean: float = .001
    l1: float = .003


PAIR_COUNTS = (
    "allocated", "legal", "probes", "probe_scores", "admitted", "shrank_to_zero",
    "lost_source", "learned_writes", "learned_writes_new_arc",
)


def _splitmix(bits):
    bits = (bits ^ (bits >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    bits = (bits ^ (bits >> 27)) * np.uint64(0x94d049bb133111eb)
    return bits ^ (bits >> 31)


def _key(lane, receiver, first, second):
    return ((lane * SIZE + receiver) * SIZE + first) * SIZE + second


class CorroboratingGraph(Graph):
    """Live Q records and a few per-receiver probes, all initially absent."""

    def __init__(self, seeds, recipe, *, world_seeds, policy, pair_rate, pair_decay):
        super().__init__(seeds, recipe)
        widen_birth(self, seeds)
        self.policy = policy
        self.pair_rate = np.float32(pair_rate)
        self.pair_decay = np.float32(pair_decay)
        self.live = np.empty((4, 0), dtype=np.intp)
        self.values = np.empty(0, dtype=np.float32)
        self.probe_key = np.asarray([
            np.random.SeedSequence([int(b), int(w), 9511093]).generate_state(1, dtype=np.uint64)[0]
            for b, w in zip(seeds, world_seeds, strict=True)
        ])
        shape = (len(seeds), SIZE, policy.slots)
        self.probe_first = np.full(shape, -1, dtype=np.intp)
        self.probe_second = np.full(shape, -1, dtype=np.intp)
        self.probe_age = np.zeros(shape, dtype=np.int16)
        self.probe_sum = np.zeros(shape, dtype=np.float32)
        self.probe_square = np.zeros(shape, dtype=np.float32)
        self.probe_first_half = np.zeros(shape, dtype=np.float32)
        self.pending_gradient = None
        self.pair_events = np.zeros((len(seeds), len(PAIR_COUNTS)), dtype=np.int16)

    def advance(self, obs, *, cut=NO_CUT):
        previous = self.state
        current = self.bias.copy()
        current[:, :8] += 1.25 * (2 * obs[:, food.LEFT.start:food.RIGHT.stop] - 1)
        current[:, MOTOR] += self.motor_noise[:, None] * MOTOR_VECTOR[MOTOR]
        sent = previous * cut
        lane, receiver, first, second = self.live
        branch = np.bincount(lane * SIZE + receiver,
                             weights=self.values * sent[lane, first] * sent[lane, second],
                             minlength=sent.size).reshape(sent.shape).astype(np.float32)
        base = current + np.matmul(self.weight, sent[..., None])[..., 0] + branch
        drive = base.copy()
        drive[:, CONTACT] += obs[:, world.CONTACT_ID]
        drive[:, TASTE] += 1.4 * obs[:, world.REWARD_POS:world.REWARD_NEG + 1]
        candidate = np.tanh(drive)
        self.state = (1 - self.rate) * previous + self.rate * candidate
        taste_delta = self.rate[:, TASTE] * (candidate[:, TASTE] - np.tanh(base[:, TASTE]))
        contact_delta = candidate[:, CONTACT] - np.tanh(base[:, CONTACT])
        self.receipt += taste_delta[:, 0] - taste_delta[:, 1]
        self.side += np.sign(contact_delta[:, 1] - contact_delta[:, 0])
        self.history.append((sent, 1 - candidate ** 2, self.state[:, MOTOR].copy(), obs[:, world.ACTIVE].copy(), cut))
        self.tick_delta = self.state - previous
        right = self.state[:, MOTOR.start + 1] > self.state[:, MOTOR.start]
        return np.column_stack((~right, right))

    def trial_credit(self):
        active = np.stack([row[3] for row in self.history], axis=1)
        count = active.sum(axis=1).clip(min=1)
        motor = np.stack([row[2] for row in self.history], axis=1)
        mean = ((motor[:, :, 1] - motor[:, :, 0]) * active).sum(axis=1) / count
        amplitude = self.receipt * self.side / (count * (1 + np.exp(2 * self.side * mean)))
        delay = np.zeros_like(self.state)
        credit = []
        lane, receiver, first, second = self.live
        endpoints = np.concatenate((lane * SIZE + first, lane * SIZE + second))
        for sent, derivative, _, tick_active, cut in reversed(self.history):
            delay += (amplitude * tick_active)[:, None] * MOTOR_VECTOR
            local = delay * self.rate * derivative
            credit.append(local)
            generated = local[lane, receiver] * self.values
            influence = np.concatenate((generated * sent[lane, second], generated * sent[lane, first]))
            pair_send = np.bincount(endpoints, weights=influence, minlength=sent.size)
            send = np.matmul(local[:, None, :], self.weight)[:, 0, :] + pair_send.reshape(sent.shape).astype(np.float32)
            delay = (1 - self.rate) * delay + send * cut
        return np.stack(credit[::-1], axis=-1), np.stack([row[0] for row in self.history], axis=1)

    def gradient(self):
        if self.pending_gradient is not None:
            return self.pending_gradient
        return np.matmul(*self.trial_credit())

    @staticmethod
    def pair_gradient(credit, sent, lane, receiver, first, second):
        return np.einsum("pt,pt,pt->p", credit[lane, receiver], sent[lane, :, first], sent[lane, :, second])

    def _tally(self, index, lanes):
        self.pair_events[:, index] = np.bincount(lanes, minlength=len(self.state))

    def _reserve_probes(self, fresh):
        indegree = self.mask.sum(axis=-1)
        lane, receiver, slot = np.nonzero((self.probe_first < 0) & fresh[:, None, None] & (indegree[:, :, None] >= 2))
        if not len(lane):
            return
        degree = indegree[lane, receiver].astype(np.uint64)
        counter = ((self.receipts[lane].astype(np.uint64) + 1) * SIZE + receiver.astype(np.uint64)) * self.policy.slots + slot.astype(np.uint64)
        entropy = self.probe_key[lane] + counter * np.uint64(0x9e3779b97f4a7c15)
        i = _splitmix(entropy) % degree
        j = _splitmix(entropy ^ np.uint64(0x82a2b175229d6a5b)) % (degree - 1)
        j += j >= i
        # Sampling two distinct ranks in local incoming arcs gives a uniform unordered opportunity.
        by_rank = np.argsort(~self.mask, axis=-1, kind="stable")
        first = by_rank[lane, receiver, np.minimum(i, j)]
        second = by_rank[lane, receiver, np.maximum(i, j)]
        proposed = _key(lane, receiver, first, second)
        old_lane, old_receiver, old_slot = np.nonzero(self.probe_first >= 0)
        occupied = _key(old_lane, old_receiver, self.probe_first[old_lane, old_receiver, old_slot],
                        self.probe_second[old_lane, old_receiver, old_slot])
        keys, index = np.unique(proposed, return_index=True)
        index = index[~np.isin(keys, np.concatenate((_key(*self.live), occupied)), assume_unique=True, kind="sort")]
        self.probe_first[lane[index], receiver[index], slot[index]] = first[index]
        self.probe_second[lane[index], receiver[index], slot[index]] = second[index]

    def _clear_probes(self, address):
        self.probe_first[address] = -1
        self.probe_second[address] = -1
        self.probe_age[address] = 0
        self.probe_sum[address] = 0
        self.probe_square[address] = 0
        self.probe_first_half[address] = 0

    def _evaluate_probes(self, fresh, credit, sent):
        self._reserve_probes(fresh)
        lane, receiver, slot = np.nonzero((self.probe_first >= 0) & fresh[:, None, None])
        address = (lane, receiver, slot)
        first = self.probe_first[address]
        second = self.probe_second[address]
        gradient = self.pair_gradient(credit, sent, lane, receiver, first, second)
        self._tally(3, lane)
        self.probe_age[address] += 1
        self.probe_sum[address] += gradient
        self.probe_square[address] += gradient ** 2
        first_half_mask = self.probe_age[address] <= self.policy.half_window
        self.probe_first_half[address] += gradient * first_half_mask
        length = 2 * self.policy.half_window
        expired = self.probe_age[address] >= length
        total = self.probe_sum[address][expired]
        first_half = self.probe_first_half[address][expired]
        qualifies = ((first_half * (total - first_half) > 0)
                     & (np.abs(total) >= self.policy.evidence * np.sqrt(self.probe_square[address][expired]))
                     & (np.abs(total) >= length * self.policy.minimum_mean))
        admitted = np.flatnonzero(expired)[qualifies]
        new_values = (self.pair_rate * total[qualifies] / np.sqrt(np.float32(length))).clip(-1.25, 1.25)
        next_live = np.stack((lane[admitted], receiver[admitted], first[admitted], second[admitted]))
        self._tally(4, next_live[0])
        born_on_edited_arc = ((self.generation[lane[admitted], receiver[admitted], first[admitted]] >= 0)
                              | (self.generation[lane[admitted], receiver[admitted], second[admitted]] >= 0))
        self.pair_events[:, 7] += np.bincount(next_live[0], minlength=len(self.state)).astype(np.int16)
        self.pair_events[:, 8] += np.bincount(next_live[0, born_on_edited_arc], minlength=len(self.state)).astype(np.int16)
        self._clear_probes(tuple(part[expired] for part in address))
        self.live = np.concatenate((self.live, next_live), axis=1)
        self.values = np.concatenate((self.values, new_values))

    def _learn_live(self, fresh, credit, sent):
        index = np.flatnonzero(fresh[self.live[0]])
        lane, receiver, first, second = self.live[:, index]
        gradient = self.pair_gradient(credit, sent, lane, receiver, first, second)
        old = self.values[index]
        baseline = self.pair_decay * old
        shrink = np.float32(self.policy.l1) * self.pair_rate
        step = baseline + self.pair_rate * gradient
        proposal = (np.sign(step) * np.maximum(np.abs(step) - shrink, 0)).clip(-1.25, 1.25)
        decay_only = np.sign(baseline) * np.maximum(np.abs(baseline) - shrink, 0)
        actual_write = (proposal != old) & (proposal != decay_only)
        new_arc = (self.generation[lane, receiver, first] >= 0) | (self.generation[lane, receiver, second] >= 0)
        self._tally(5, lane[proposal == 0])
        self._tally(7, lane[actual_write])
        self._tally(8, lane[actual_write & new_arc])
        self.values[index] = proposal
        keep = self.values != 0
        self.live = self.live[:, keep]
        self.values = self.values[keep]

    def _drop_illegal(self):
        lane, receiver, first, second = self.live
        keep = self.mask[lane, receiver, first] & self.mask[lane, receiver, second]
        self._tally(6, lane[~keep])
        self.live = self.live[:, keep]
        self.values = self.values[keep]
        lane, receiver, slot = np.nonzero(self.probe_first >= 0)
        address = (lane, receiver, slot)
        keep = self.mask[lane, receiver, self.probe_first[address]] & self.mask[lane, receiver, self.probe_second[address]]
        self._clear_probes(tuple(part[~keep] for part in address))

    def update(self, trial):
        self.pair_events.fill(0)
        fresh = (self.receipt != 0) & (self.side != 0)
        if fresh.any():
            credit, sent = self.trial_credit()
            self.pending_gradient = np.matmul(credit, sent)
            self._learn_live(fresh, credit, sent)
            self._evaluate_probes(fresh, credit, sent)
        try:
            result = super().update(trial)
        finally:
            self.pending_gradient = None
        if (result[2][:, 1] >= 0).any():
            self._drop_illegal()
        self._tally(0, self.live[0])
        indegree = self.mask.sum(axis=-1)
        self.pair_events[:, 1] = (indegree * (indegree - 1) // 2).sum(axis=1)
        self.pair_events[:, 2] = (self.probe_first >= 0).sum(axis=(1, 2))
        return result
