"""Receipt-sampled, receiver-local pair coefficients on the native port graph."""

from dataclasses import dataclass

import numpy as np

from . import mixture_world as food, world
from .dendrite_ports_graph import DendriteGraph, FIRST, SECOND, widen_birth
from .unified_live_graph import CONTACT, MOTOR, MOTOR_VECTOR, NO_CUT, SIZE, TASTE, Graph


WIDTH = len(FIRST)
PER_LANE = SIZE * WIDTH
RECORD = np.dtype([("key", "i4"), ("weight", "f4"), ("born", "i4"),
                   ("born_receipt", "i4"), ("writes", "i4")])
EVENT = np.dtype([("trial", "i4"), ("type", "i1"), *RECORD.descr, ("gradient", "f4")])
BIRTH, SUPPORT, EVIDENCE = 1, -1, -2


@dataclass(frozen=True)
class PairPolicy:
    samples: int = 24
    threshold: float = .035
    floor: float = .10
    grace: int = 72


class DenseWGraph(DendriteGraph):
    """Sealed dense Q with the same standalone W growth and retirement as sparse Q."""

    grow = Graph.grow
    retire = Graph.retire


def mix(values):
    """Stateless private SplitMix64 values with intentional unsigned wraparound."""
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def coordinates(keys):
    destination, index = np.divmod(keys, WIDTH)
    return destination // SIZE, destination % SIZE, FIRST[index], SECOND[index]


def legal(mask, keys):
    lane, receiver, first, second = coordinates(keys)
    return mask[lane, receiver, first] & mask[lane, receiver, second]


def absent(keys, active):
    if not len(active):
        return np.ones(len(keys), dtype=bool)
    position = active.searchsorted(keys).clip(max=len(active) - 1)
    return active[position] != keys


def event_rows(records, trial, kind, gradient=np.nan):
    events = np.empty(len(records), dtype=EVENT)
    for field in RECORD.names:
        events[field] = records[field]
    events["trial"], events["type"], events["gradient"] = trial, kind, gradient
    return events


def first_per_lane(keys, priorities, counts):
    lane = keys // PER_LANE
    order = np.lexsort((priorities, lane))
    ordered = lane[order]
    rank = np.arange(len(ordered)) - np.searchsorted(ordered, ordered)
    return order[rank < counts[ordered]]


class SparseDendriteGraph(Graph):
    """Packed authoritative Q records, with no trace or storage for unproposed Q."""

    def __init__(self, seeds, recipe, *, world_seeds, pair_rate, pair_decay,
                 policy=PairPolicy(), control="guided"):
        super().__init__(seeds, recipe)
        if control not in ("guided", "random"):
            raise ValueError("Unknown pair birth rule")
        if len(world_seeds) != len(seeds) or min(policy.samples, policy.threshold, policy.floor, policy.grace) < 0:
            raise ValueError("Invalid pair sampling policy or world seed shape")
        widen_birth(self, seeds)
        self.policy = policy
        self.control = control
        self.pair_rate = np.float32(pair_rate)
        self.pair_decay = np.float32(pair_decay)
        self.proposals_enabled = True
        seed_pairs = zip(seeds, world_seeds)
        self.private_keys = np.asarray([np.random.SeedSequence([int(a), int(b), 29040477])
                                        .generate_state(1, dtype=np.uint64)[0] for a, b in seed_pairs])
        self.sample_offsets = np.arange(policy.samples, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        self.pair_events = np.empty(0, dtype=EVENT)
        self.pending_gradient = None
        self.pair_writes = np.zeros_like(self.receipts)
        self.newborn_pair_writes = np.zeros_like(self.receipts)
        self.proposal_count = np.zeros_like(self.receipts)
        self.set_active(np.empty(0, dtype=RECORD))

    def set_active(self, records):
        """Replace a mature clone's packed records, sorted by unique physical address."""
        self.active = np.array(records, dtype=RECORD, copy=True)
        if len(self.active) > 1 and np.any(self.active["key"][1:] <= self.active["key"][:-1]):
            self.active = np.sort(self.active, order="key")
            if np.any(self.active["key"][1:] == self.active["key"][:-1]):
                raise ValueError("Duplicate interaction address")
        keys = self.active["key"]
        if len(keys) and (keys[0] < 0 or keys[-1] >= len(self.state) * PER_LANE or not legal(self.mask, keys).all()):
            raise ValueError("Interaction address lacks two active incoming arcs")
        self.lane, receiver, self.first, self.second = coordinates(keys)
        self.destination = self.lane * SIZE + receiver
        self.first_slot = self.lane * SIZE + self.first
        self.second_slot = self.lane * SIZE + self.second

    def active_counts(self):
        return np.bincount(self.lane, minlength=len(self.state))

    def pair_storage_bytes(self):
        """Persistent pair-authoritative, address-cache, and private sampling array bytes."""
        arrays = (self.active, self.lane, self.first, self.second, self.destination, self.first_slot,
                  self.second_slot, self.private_keys, self.sample_offsets)
        return sum(a.nbytes for a in arrays)

    def _node_sum(self, slots, values):
        return np.bincount(slots, values, minlength=self.state.size).astype(np.float32).reshape(self.state.shape)

    def advance(self, obs, *, cut=NO_CUT):
        previous = self.state
        current = self.bias.copy()
        current[:, :8] += 1.25 * (2 * obs[:, food.LEFT.start:food.RIGHT.stop] - 1)
        current[:, MOTOR] += self.motor_noise[:, None] * MOTOR_VECTOR[MOTOR]
        sent = previous * cut
        incoming = np.matmul(self.weight, sent[..., None])[..., 0]
        if len(self.active):
            pair_current = self.active["weight"] * sent[self.lane, self.first] * sent[self.lane, self.second]
            incoming += self._node_sum(self.destination, pair_current)
        base = current + incoming
        drive = base.copy()
        drive[:, TASTE] += 1.4 * obs[:, world.REWARD_POS:world.REWARD_NEG + 1]
        drive[:, CONTACT] += obs[:, world.CONTACT_ID]
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

    def receipt_credit(self):
        active = np.stack([record[3] for record in self.history], axis=1)
        count = active.sum(axis=1).clip(min=1)
        motor = np.stack([record[2] for record in self.history], axis=1)
        mean = ((motor[:, :, 1] - motor[:, :, 0]) * active).sum(axis=1) / count
        amplitude = self.receipt * self.side / (count * (1 + np.exp(2 * self.side * mean)))
        delay = np.zeros_like(self.state)
        time_credit = []
        for sent, derivative, _, tick_active, cut in reversed(self.history):
            delay += (amplitude * tick_active)[:, None] * MOTOR_VECTOR
            local = delay * self.rate * derivative
            time_credit.append(local)
            send_credit = np.matmul(local[:, None, :], self.weight)[:, 0, :]
            if len(self.active):
                credit = local.ravel()[self.destination] * self.active["weight"]
                slots = np.concatenate((self.first_slot, self.second_slot))
                increments = np.concatenate((credit * sent[self.lane, self.second], credit * sent[self.lane, self.first]))
                send_credit += self._node_sum(slots, increments)
            delay = (1 - self.rate) * delay + send_credit * cut
        local = np.stack(time_credit[::-1], axis=1)
        sent = np.stack([record[0] for record in self.history], axis=1)
        return local, sent

    def gradient(self):
        if self.pending_gradient is not None:
            return self.pending_gradient
        local, sent = self.receipt_credit()
        return np.matmul(local.swapaxes(1, 2), sent)

    @staticmethod
    def pair_gradient(keys, local, sent):
        lane, receiver, first, second = coordinates(keys)
        # Each iteration only allocates a vector of active or sampled entries.
        result = np.zeros(len(keys), dtype=np.float32)
        for tick in range(local.shape[1]):
            result += local[lane, tick, receiver] * (sent[lane, tick, first] * sent[lane, tick, second])
        return result

    def _sample(self, trial, fresh, active, batch=0):
        lanes = np.flatnonzero(fresh)
        label = np.uint64((trial * 0xCA5A826395121157 + batch * 0xBB67AE8584CAA73B) & ((1 << 64) - 1))
        bits = mix(self.private_keys[lanes, None] ^ label ^ self.sample_offsets)
        keys = (lanes[:, None] * PER_LANE + (bits % PER_LANE).astype(np.int32)).astype(np.int32)
        sampled, first = np.unique(keys, return_index=True)
        valid = legal(self.mask, sampled) & absent(sampled, active)
        return sampled[valid], bits.ravel()[first[valid]]

    def update(self, trial, *, match=None):
        self.pair_writes.fill(0)
        self.newborn_pair_writes.fill(0)
        self.proposal_count.fill(0)
        self.pair_events = np.empty(0, dtype=EVENT)
        fresh = (self.receipt != 0) & (self.side != 0)
        if not fresh.any() or not (self.learning or self.editing):
            return super().update(trial)
        if self.control == "random" and self.proposals_enabled and match is None:
            raise ValueError("Random placement requires the guided arm's current-trial counts")
        local, sent = self.receipt_credit()
        records = self.active.copy()
        live_keys = records["key"]
        candidates, priority = self._sample(trial, fresh, live_keys) if self.proposals_enabled and self.learning else (
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint64))
        if self.learning:
            self._update_active(records, local, sent, fresh)
        self.pending_gradient = np.matmul(local.swapaxes(1, 2), sent)
        try:
            ordinary = super().update(trial)
        finally:
            self.pending_gradient = None
        supported = legal(self.mask, live_keys)
        lost_support = event_rows(records[~supported], trial, SUPPORT)
        records = records[supported]
        retire = self._evidence_retire(records, lost_support["key"], fresh, trial, match)
        lost_evidence = event_rows(records[retire], trial, EVIDENCE)
        records = records[~retire]
        candidates, priority = self._post_edit_samples(candidates, priority, live_keys, fresh, ordinary[2], trial, match)
        candidate_gradient = self.pair_gradient(candidates, local, sent)
        births, birth_gradient = self._births(candidates, priority, candidate_gradient, trial, match)
        self.pair_events = np.concatenate((lost_support, lost_evidence, event_rows(births, trial, BIRTH, birth_gradient)))
        if len(self.pair_events):
            self.set_active(np.concatenate((records, births)))
        else:
            self.active["weight"] = records["weight"]
            self.active["writes"] = records["writes"]
        return ordinary

    def _update_active(self, records, local, sent, fresh):
        keys = records["key"]
        gradient = self.pair_gradient(keys, local, sent)
        old = records["weight"]
        decay = self.pair_decay * old
        proposal = (decay + self.pair_rate * gradient).clip(-1.25, 1.25)
        written = fresh[self.lane] & (proposal != old) & (proposal != decay)
        records["writes"] += written
        self.pair_writes[:] = np.bincount(self.lane[written], minlength=len(self.state))
        newborn = (self.generation[self.lane, self.destination % SIZE, self.first] >= 0) | (
            self.generation[self.lane, self.destination % SIZE, self.second] >= 0)
        self.newborn_pair_writes[:] = np.bincount(self.lane[written & newborn], minlength=len(self.state))
        old[:] = np.where(fresh[self.lane], proposal, old)

    def _evidence_retire(self, records, support, fresh, trial, match):
        lane = records["key"] // PER_LANE
        if self.control == "guided" or match is None:
            age = self.receipts[lane] - records["born_receipt"]
            return fresh[lane] & (age >= self.policy.grace) & (np.abs(records["weight"]) < self.policy.floor)
        eligible = np.flatnonzero(fresh[lane])
        keys = records["key"][eligible]
        priorities = mix(self.private_keys[lane[eligible]] ^ keys.astype(np.uint64) ^ np.uint64(trial))
        support_counts = np.bincount(support // PER_LANE, minlength=len(fresh))
        selected = first_per_lane(keys, priorities, np.maximum(0, match[1] - support_counts))
        retired = np.zeros(len(records), dtype=bool)
        retired[eligible[selected]] = True
        return retired

    def _post_edit_samples(self, candidates, priority, old, fresh, edits, trial, match):
        valid = legal(self.mask, candidates)
        candidates, priority = candidates[valid], priority[valid]
        if self.control == "random" and self.proposals_enabled:
            # Draw additional private random locations only to fill that trial's matched count.
            # A newly grown arc did not participate in this receipt, so skip it.
            for batch in range(1, 5):
                counts = np.bincount(candidates // PER_LANE, minlength=len(fresh))
                missing = fresh & (counts < match[0])
                if not missing.any():
                    break
                later, rank = self._sample(trial, missing, np.sort(np.concatenate((old, candidates))), batch=batch)
                lane, receiver, first, second = coordinates(later)
                edge = edits[lane, 0]
                kept = (receiver * SIZE + first != edge) & (receiver * SIZE + second != edge)
                candidates = np.concatenate((candidates, later[kept]))
                priority = np.concatenate((priority, rank[kept]))
        self.proposal_count[:] = np.bincount(candidates // PER_LANE, minlength=len(fresh))
        return candidates, priority

    def _births(self, candidates, priority, gradient, trial, match):
        initial = (self.pair_rate * gradient).clip(-1.25, 1.25)
        if self.control == "guided" or match is None:
            selected = np.flatnonzero((initial != 0) & (np.abs(gradient) >= self.policy.threshold))
        else:
            nonzero = np.flatnonzero(initial != 0)
            selected = nonzero[first_per_lane(candidates[nonzero], priority[nonzero], match[0])]
        keys = candidates[selected]
        born = np.empty(len(keys), dtype=RECORD)
        born["key"], born["weight"], born["born"], born["writes"] = keys, initial[selected], trial, 0
        born["born_receipt"] = self.receipts[keys // PER_LANE]
        return born, gradient[selected]
