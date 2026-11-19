"""Train receiver-local pairs of incoming arcs on the native physical-port graph."""

import numpy as np

from . import mixture_world as food, world
from .unified_live_graph import CONTACT, MOTOR, MOTOR_VECTOR, NO_CUT, SIZE, TASTE, Graph, strongly_connected


FIRST, SECOND = np.triu_indices(SIZE, k=1)
LINEAR_SCALE = np.float32(.36)


def widen_birth(graph, seeds):
    """Use the same dense random birth for all arms without encoding any odor pairs."""
    for lane, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 16071571]))
        added = (rng.random((SIZE, SIZE)) < .52) & ~graph.mask[lane]
        np.fill_diagonal(added, False)
        graph.mask[lane] |= added
        graph.weight[lane] = .67 * graph.weight[lane] + added * rng.normal(0, .104, (SIZE, SIZE))
    graph.initial_mask = graph.mask.copy()
    graph.initial_weight = graph.weight.copy()


class DendriteGraph(Graph):
    """A pair parameter exists only while both source arcs enter the same receiver."""

    def __init__(self, seeds, recipe, *, product, pair_rate, pair_decay):
        super().__init__(seeds, recipe)
        widen_birth(self, seeds)
        self.product = product
        self.pair_rate = np.float32(pair_rate)
        self.pair_decay = np.float32(pair_decay)
        self.pair = np.zeros((*self.mask.shape[:2], len(FIRST)), dtype=np.float32)
        self.pair_credit = np.zeros_like(self.pair)
        self.symmetric = np.zeros((*self.mask.shape, SIZE), dtype=np.float32)
        self.pair_writes = np.zeros_like(self.receipts)
        self.newborn_pair_writes = np.zeros_like(self.receipts)
        self.pending_gradient = None
        self.refresh_pair_support()

    def refresh_pair_support(self):
        self.valid_pairs = self.mask[:, :, FIRST] & self.mask[:, :, SECOND]
        self.pair[~self.valid_pairs] = 0
        self.symmetric[:, :, FIRST, SECOND] = self.pair
        self.symmetric[:, :, SECOND, FIRST] = self.pair
        if not self.product:
            self.additive = LINEAR_SCALE * self.symmetric.sum(axis=-1)

    def advance(self, obs, *, cut=NO_CUT):
        previous = self.state
        current = self.bias.copy()
        current[:, :8] += 1.25 * (2 * obs[:, food.LEFT.start:food.RIGHT.stop] - 1)
        current[:, MOTOR] += self.motor_noise[:, None] * MOTOR_VECTOR[MOTOR]
        sent = previous * cut
        if self.product:
            branch = np.matmul(self.symmetric, sent[:, None, :, None])[..., 0]
            incoming = np.matmul(self.weight, sent[..., None])[..., 0] + .5 * (branch * sent[:, None, :]).sum(axis=-1)
        else:
            branch = self.additive
            incoming = np.matmul(self.weight + branch, sent[..., None])[..., 0]
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
        self.history.append((sent, branch, 1 - candidate ** 2, self.state[:, MOTOR].copy(), obs[:, world.ACTIVE].copy(), cut))
        self.tick_delta = self.state - previous
        right = self.state[:, MOTOR.start + 1] > self.state[:, MOTOR.start]
        return np.column_stack((~right, right))

    def gradients(self):
        active = np.stack([record[4] for record in self.history], axis=1)
        count = active.sum(axis=1).clip(min=1)
        motor = np.stack([record[3] for record in self.history], axis=1)
        mean = ((motor[:, :, 1] - motor[:, :, 0]) * active).sum(axis=1) / count
        amplitude = self.receipt * self.side / (count * (1 + np.exp(2 * self.side * mean)))
        delay = np.zeros_like(self.state)
        time_credit = []
        for _, branch, derivative, _, tick_active, cut in reversed(self.history):
            delay += (amplitude * tick_active)[:, None] * MOTOR_VECTOR
            local = delay * self.rate * derivative
            time_credit.append(local)
            send_credit = np.matmul(local[:, None, :], self.weight + branch)[:, 0, :]
            delay = (1 - self.rate) * delay + send_credit * cut
        local = np.stack(time_credit[::-1], axis=-1)
        sent = np.stack([record[0] for record in self.history], axis=1)
        pair_feature = (sent[:, :, FIRST] * sent[:, :, SECOND] if self.product
                        else LINEAR_SCALE * (sent[:, :, FIRST] + sent[:, :, SECOND]))
        return np.matmul(local, sent), np.matmul(local, pair_feature)

    def gradient(self):
        return self.pending_gradient if self.pending_gradient is not None else self.gradients()[0]

    def update(self, trial):
        self.pair_writes.fill(0)
        self.newborn_pair_writes.fill(0)
        fresh = (self.receipt != 0) & (self.side != 0)
        if not fresh.any():
            return super().update(trial)
        self.pending_gradient, pair_gradient = self.gradients()
        try:
            self.pair_credit[fresh] = .9 * self.pair_credit[fresh] + .1 * pair_gradient[fresh]
            decay = self.pair_decay * self.pair
            proposal = (decay + self.pair_rate * pair_gradient).clip(-1.25, 1.25)
            active = fresh[:, None, None] & self.valid_pairs
            written = active & (proposal != self.pair) & (proposal != decay)
            self.pair_writes[:] = written.sum(axis=(1, 2))
            born = self.generation >= 0
            self.newborn_pair_writes[:] = (written & (born[:, :, FIRST] | born[:, :, SECOND])).sum(axis=(1, 2))
            self.pair = np.where(active, proposal, self.pair)
            result = super().update(trial)
            self.refresh_pair_support()
            return result
        finally:
            self.pending_gradient = None

    def grow(self, lane, trial):
        mask = self.mask[lane]
        upper = mask[:, FIRST]
        lower = mask[:, SECOND]
        credit = np.abs(self.pair_credit[lane])
        unlock = np.zeros_like(self.weight[lane])
        np.maximum.at(unlock.T, FIRST, (credit * lower).T)
        np.maximum.at(unlock.T, SECOND, (credit * upper).T)
        scores = np.abs(self.dense_credit[lane]) + 3 * unlock
        scores[mask] = 0
        np.fill_diagonal(scores, 0)
        index = int(scores.argmax())
        if scores.flat[index] <= 1e-7:
            return -1
        mask.flat[index] = True
        self.weight[lane].flat[index] = np.copysign(.035, self.dense_credit[lane].flat[index])
        self.generation[lane].flat[index] = trial
        self.generation_writes[lane].flat[index] = 0
        return index

    def retire(self, lane):
        mask, weight = self.mask[lane], self.weight[lane]
        importance = np.zeros_like(weight)
        magnitude = np.abs(self.pair[lane]).T
        np.maximum.at(importance.T, FIRST, magnitude)
        np.maximum.at(importance.T, SECOND, magnitude)
        score = np.where(mask, np.abs(weight) + 2 * weight * self.dense_credit[lane] + .45 * importance, np.inf)
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
