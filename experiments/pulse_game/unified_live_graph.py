"""Physical ports on one plastic directed graph, with no learned output projection."""

from dataclasses import dataclass

import numpy as np

from . import mixture_world as food, world


SIZE = 16
MOTOR = slice(8, 10)
TASTE = slice(10, 12)
CONTACT = slice(12, 14)
NO_CUT = np.ones(SIZE, dtype=np.float32)
MOTOR_VECTOR = np.zeros(SIZE, dtype=np.float32)
MOTOR_VECTOR[MOTOR] = (-1, 1)


def strongly_connected(arcs):
    """Rows are destinations and columns are sources."""
    for edges in (arcs, arcs.T):
        reached = np.zeros(len(edges), dtype=bool)
        reached[0] = True
        while True:
            expanded = reached | edges[:, reached].any(axis=1)
            if np.array_equal(reached, expanded):
                break
            reached = expanded
        if not reached.all():
            return False
    return True


def births(seeds):
    """Random strong multicycle graphs with heterogeneous independent arcs and node rates."""
    masks, weights, rates, biases = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 490123]))
        # Reject disconnected samples; no permanent scaffold protects a subset of arcs.
        while True:
            arcs = rng.random((SIZE, SIZE)) < .31
            np.fill_diagonal(arcs, False)
            if strongly_connected(arcs) and arcs.sum() > 2 * SIZE:
                break
        masks.append(arcs)
        weights.append(arcs * rng.normal(0, .155, arcs.shape))
        rates.append(rng.uniform(.48, .88, SIZE))
        biases.append(rng.normal(0, .016, SIZE))
    return (np.asarray(masks), np.asarray(weights, dtype=np.float32),
            np.asarray(rates, dtype=np.float32), np.asarray(biases, dtype=np.float32))


@dataclass(frozen=True)
class Recipe:
    rate: float = .12
    exploration: float = .37
    decay: float = .995
    grow_every: int = 19
    retire_every: int = 17


class Graph:
    def __init__(self, seeds, recipe=Recipe(), *, learning=True, editing=True):
        self.mask, self.weight, self.rate, self.bias = births(seeds)
        self.initial_mask = self.mask.copy()
        self.initial_weight = self.weight.copy()
        self.recipe = recipe
        self.learning = learning
        self.editing = editing
        self.state = np.zeros_like(self.rate)
        self.dense_credit = np.zeros_like(self.weight)
        self.receipts = np.zeros(len(seeds), dtype=np.int32)
        self.generation = np.full(self.mask.shape, -1, dtype=np.int32)
        self.generation_writes = np.zeros(self.mask.shape, dtype=np.int32)
        self.tick_delta = np.zeros_like(self.rate)

    def start(self, motor_noise):
        # One private exploratory current per physical request, applied to the two motor registers.
        self.motor_noise = motor_noise
        self.history = []
        self.receipt = np.zeros(len(self.state), dtype=np.float32)
        self.side = np.zeros_like(self.receipt)

    def advance(self, obs, *, cut=NO_CUT):
        previous = self.state
        current = np.broadcast_to(self.bias, previous.shape).copy()
        current[:, :8] += 1.25 * (2 * obs[:, food.LEFT.start:food.RIGHT.stop] - 1)
        current[:, MOTOR] += self.motor_noise[:, None] * MOTOR_VECTOR[MOTOR]
        sent = previous * cut
        base = np.matmul(self.weight, sent[..., None])[..., 0] + current
        taste = 1.4 * obs[:, world.REWARD_POS:world.REWARD_NEG + 1]
        after_input = base.copy()
        after_input[:, TASTE] += taste
        after_input[:, CONTACT] += obs[:, world.CONTACT_ID]
        proposal = np.tanh(after_input)
        self.state = (1 - self.rate) * previous + self.rate * proposal
        if self.learning or self.editing:
            # Use the causal change of the two receipt nodes themselves for modulation.
            # The reference candidate has the exact same pre-tick state and a zero taste input.
            taste_delta = self.rate[:, TASTE] * (proposal[:, TASTE] - np.tanh(base[:, TASTE]))
            self.receipt += taste_delta[:, 0] - taste_delta[:, 1]
            contact_delta = proposal[:, CONTACT] - np.tanh(base[:, CONTACT])
            self.side += np.sign(contact_delta[:, 1] - contact_delta[:, 0])
            active = obs[:, world.ACTIVE].copy()
            self.history.append((sent, proposal, self.state[:, MOTOR].copy(), active))
        self.tick_delta = self.state - previous
        right = self.state[:, MOTOR.start + 1] > self.state[:, MOTOR.start]
        return np.column_stack((~right, right))

    def gradient(self):
        """Differentiate a receipt-weighted within-request motor contrast, holding its start fixed."""
        active = np.stack([row[3] for row in self.history], axis=1)
        count = active.sum(axis=1).clip(min=1)
        motor = np.stack([row[2] for row in self.history], axis=1)
        mean = ((motor[:, :, 1] - motor[:, :, 0]) * active).sum(axis=1) / count
        amplitude = self.receipt * self.side / (count * (1 + np.exp(2 * self.side * mean)))
        delay = np.zeros_like(self.state)
        gradient = np.zeros_like(self.weight)
        for sent, proposal, _, tick_active in reversed(self.history):
            delay += (amplitude * tick_active)[:, None] * MOTOR_VECTOR
            local = delay * self.rate * (1 - proposal ** 2)
            gradient += local[:, :, None] * sent[:, None, :]
            delay = (1 - self.rate) * delay + np.matmul(local[:, None, :], self.weight)[:, 0, :]
        return gradient

    def update(self, trial):
        changed = np.zeros_like(self.receipts)
        newborn_changed = np.zeros_like(self.receipts)
        events = np.full((len(self.state), 2), -1, dtype=np.int16)
        retired_writes = np.zeros_like(self.receipts)
        fresh = (self.receipt != 0) & (self.side != 0)
        if (self.learning or self.editing) and fresh.any():
            gradient = self.gradient()
            self.receipts += fresh
            if self.learning:
                old = self.weight.copy()
                proposal = self.recipe.decay * old + self.recipe.rate * gradient
                proposal = np.where(proposal != 0, proposal, np.copysign(1e-8, old))
                self.weight = np.where(fresh[:, None, None] & self.mask, proposal.clip(-1.25, 1.25), old)
                changed = (old != self.weight).sum(axis=(1, 2))
                # Count actual receipt-gradient contributions to the new generation, excluding decay alone.
                decay_only = (self.recipe.decay * old).clip(-1.25, 1.25)
                written = (old != self.weight) & (decay_only != self.weight) & (self.generation >= 0)
                newborn_changed = written.sum(axis=(1, 2))
                self.generation_writes += written
            if self.editing:
                self.dense_credit[fresh] = .90 * self.dense_credit[fresh] + .10 * gradient[fresh]
                for lane in np.flatnonzero(fresh):
                    if self.receipts[lane] % self.recipe.grow_every == 0:
                        events[lane, 0] = self.grow(lane, trial)
                    if self.receipts[lane] % self.recipe.retire_every == 0:
                        events[lane, 1], retired_writes[lane] = self.retire(lane)
        return changed, newborn_changed, events, retired_writes

    def grow(self, lane, trial):
        available = ~self.mask[lane].copy()
        np.fill_diagonal(available, False)
        score = np.where(available, np.abs(self.dense_credit[lane]), 0)
        index = int(score.argmax())
        if score.flat[index] <= 1e-7:
            return -1
        self.mask[lane].flat[index] = True
        self.weight[lane].flat[index] = .035 * np.sign(self.dense_credit[lane].flat[index])
        self.generation[lane].flat[index] = trial
        self.generation_writes[lane].flat[index] = 0
        return index

    def retire(self, lane):
        mask = self.mask[lane]
        weight = self.weight[lane]
        # A small coefficient or a sufficiently adverse signed gradient makes a candidate.
        score = np.where(mask, np.abs(weight) + 2 * weight * self.dense_credit[lane], np.inf)
        for index in np.argsort(score, axis=None):
            if score.flat[index] > .085:
                break
            mask.flat[index] = False
            if strongly_connected(mask & (weight != 0)):
                writes = int(self.generation_writes[lane].flat[index])
                weight.flat[index] = 0
                self.generation[lane].flat[index] = -1
                self.generation_writes[lane].flat[index] = 0
                return index, writes
            mask.flat[index] = True
        return -1, 0
