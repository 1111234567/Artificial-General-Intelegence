"""Continuous interaction-only tastes and a per-node nonlinearity for the unified graph."""

from dataclasses import replace

import numpy as np

from . import mixture_world as food, world
from .unified_live_graph import CONTACT, MOTOR, MOTOR_VECTOR, NO_CUT, SIZE, TASTE, Graph


CROSS = np.array(((0, 2), (0, 3), (1, 2), (1, 3)))
TASTE_GAIN = 10.


def interactions(signals, pairs=CROSS):
    return signals[..., pairs[:, 0]] * signals[..., pairs[:, 1]]


def make_menu(seeds, mode, trials):
    if mode not in ("stable", "reverse"):
        raise ValueError("Unknown taste regime")
    tape = world.make_tape(seeds, trials=trials, distance_jitter=.13, wind_scale=.07)
    n = len(seeds)
    shape = (n, trials)
    chemicals = np.empty((*shape, 2, food.DIM), dtype=np.float32)
    uniforms = np.empty((*shape, 2), dtype=np.float32)
    weights = np.empty((n, len(CROSS)), dtype=np.float32)
    changes = np.empty(n, dtype=np.int16)
    motor_noise = np.empty(shape, dtype=np.float32)
    random_sign = np.empty(shape, dtype=np.int8)
    for lane, seed in enumerate(seeds):
        coeff, odor, taste, schedule, exploration, sham = (
            np.random.default_rng(branch) for branch in np.random.SeedSequence([int(seed), 69989137]).spawn(6)
        )
        coefficients = coeff.standard_normal(len(CROSS))
        weights[lane] = coefficients / np.linalg.norm(coefficients)
        chemicals[lane] = odor.uniform(0., 1., size=(trials, 2, food.DIM))
        uniforms[lane] = taste.random((trials, 2))
        changes[lane] = trials // 2 + schedule.integers(-trials // 24, trials // 24 + 1)
        motor_noise[lane] = exploration.logistic(size=trials).clip(-9, 9)
        random_sign[lane] = sham.choice((-1, 1), trials)
    utility = np.einsum("btik,bk->bti", interactions(chemicals * 2 - 1), weights)
    if mode == "reverse":
        utility = np.where(np.arange(trials)[None, :, None] >= changes[:, None, None], -utility, utility)
    probability = (1. / (1. + np.exp(-TASTE_GAIN * utility))).astype(np.float32)
    realized = np.where(uniforms < probability, 1, -1).astype(np.int8)
    menu = food.Foods(tape, chemicals, probability, realized, weights, "interaction", trials // 2)
    return menu, changes, motor_noise, random_sign


class SquareGraph(Graph):
    """Bounded even residual at sensory and internal nodes, with identical physical ports."""

    def __init__(self, seeds, recipe, *, scale, **kwargs):
        super().__init__(seeds, recipe, **kwargs)
        self.even = np.resize(np.array([scale, -scale], dtype=np.float32), SIZE)
        self.even[MOTOR] = self.even[TASTE] = self.even[CONTACT] = 0
        self.damping = 1. / (1. + np.abs(self.even))

    def advance(self, obs, *, cut=NO_CUT):
        previous = self.state
        current = self.bias.copy()
        current[:, :8] += 1.25 * (2 * obs[:, food.LEFT.start:food.RIGHT.stop] - 1)
        current[:, MOTOR] += self.motor_noise[:, None] * MOTOR_VECTOR[MOTOR]
        sent = previous * cut
        base = np.matmul(self.weight, sent[..., None])[..., 0] + current
        after_input = base.copy()
        after_input[:, TASTE] += 1.4 * obs[:, world.REWARD_POS:world.REWARD_NEG + 1]
        after_input[:, CONTACT] += obs[:, world.CONTACT_ID]
        unit = np.tanh(after_input)
        proposal = (unit + self.even * unit ** 2) * self.damping
        self.state = (1 - self.rate) * previous + self.rate * proposal
        if self.learning or self.editing:
            taste_delta = self.rate[:, TASTE] * (proposal[:, TASTE] - np.tanh(base[:, TASTE]))
            self.receipt += taste_delta[:, 0] - taste_delta[:, 1]
            contact_delta = proposal[:, CONTACT] - np.tanh(base[:, CONTACT])
            self.side += np.sign(contact_delta[:, 1] - contact_delta[:, 0])
            derivative = self.damping * (1. + 2. * self.even * unit) * (1. - unit ** 2)
            self.history.append((sent, derivative, self.state[:, MOTOR].copy(), obs[:, world.ACTIVE].copy(), cut))
        self.tick_delta = self.state - previous
        right = self.state[:, MOTOR.start + 1] > self.state[:, MOTOR.start]
        return np.column_stack((~right, right))

    def gradient(self):
        active = np.stack([row[3] for row in self.history], axis=1)
        count = active.sum(axis=1).clip(min=1)
        motor = np.stack([row[2] for row in self.history], axis=1)
        mean = ((motor[:, :, 1] - motor[:, :, 0]) * active).sum(axis=1) / count
        amplitude = self.receipt * self.side / (count * (1. + np.exp(2 * self.side * mean)))
        delay = np.zeros_like(self.state)
        gradient = np.zeros_like(self.weight)
        for sent, derivative, _, tick_active, cut in reversed(self.history):
            delay += (amplitude * tick_active)[:, None] * MOTOR_VECTOR
            local = delay * self.rate * derivative
            gradient += local[:, :, None] * sent[:, None, :]
            delay = (1. - self.rate) * delay + np.matmul(local[:, None, :], self.weight)[:, 0, :] * cut
        return gradient


class PublicRidge:
    """A visibility control, trained on this body's own contacted chemistry and shown taste."""

    def __init__(self, lanes, *, quadratic):
        self.quadratic = quadratic
        width = 1 + (len(food.PAIR) if quadratic else food.DIM)
        self.prior = .35 * np.eye(width)
        self.gram = np.broadcast_to(self.prior, (lanes, width, width)).copy()
        self.target = np.zeros((lanes, width), dtype=np.float64)

    def choose(self, initial_obs, noise, *, explore):
        chemicals = initial_obs[:, food.LEFT.start:food.RIGHT.stop].reshape(-1, 2, food.DIM)
        signals = 2. * chemicals - 1.
        columns = interactions(signals, food.PAIR) if self.quadratic else signals
        self.terms = np.concatenate((columns, np.ones((*columns.shape[:2], 1))), axis=-1)
        coefficient = np.linalg.solve(self.gram, self.target[..., None])[..., 0]
        score = np.einsum("bki,bi->bk", self.terms, coefficient)
        exploration = np.abs(noise) > 2.65 if explore else np.zeros(len(noise), dtype=bool)
        return np.where(exploration, noise > 0, score[:, 1] - score[:, 0] + 1e-8 * noise > 0)

    def learn(self, observations):
        contact = observations[:, world.CONTACT_ID]
        taste = observations[:, world.REWARD_POS] - observations[:, world.REWARD_NEG]
        heard = (taste != 0) & (contact.sum(axis=1) > 0)
        if not heard.any():
            return
        rows = np.flatnonzero(heard)
        features = self.terms[rows, contact[rows].argmax(axis=1)]
        self.gram[rows] = .982 * self.gram[rows] + .018 * self.prior + features[:, :, None] * features[:, None, :]
        self.target[rows] = .982 * self.target[rows] + features * taste[rows, None]
