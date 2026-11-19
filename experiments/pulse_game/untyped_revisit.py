"""Infer the two roles of a single public sensory port from distance and activity traces."""

from dataclasses import replace

import numpy as np

from . import mixture_world as food, symmetry_world as geometry, untyped_senses as base, world


class Receptor(base.Familiarity):
    def __init__(self, n, *, dot=False):
        super().__init__(n)
        self.dot = dot
        self.slow = np.zeros_like(self.trace)

    def features(self, rows, values, *, update=True):
        mean = geometry.unit(values.mean(2))
        activity = geometry.unit(np.exp(8. * (mean @ self.centers.T - 1.)).astype(np.float32))
        short = (activity * self.trace[rows, None]).sum(2)
        long = (activity * self.slow[rows, None]).sum(2)
        if self.dot:
            same = (values[:, :, 0] * values[:, :, 1]).sum(2)
        else:
            difference = values[:, :, 0] - values[:, :, 1]
            same = -.25 * np.square(difference).sum(2)
        count = self.count[rows].astype(np.float32)
        agree = (mean[:, :, None] * self.proto[rows, None]).sum(-1)
        confidence = np.square(count / (count + 3.))
        prototype = (np.exp(np.minimum(agree - 1., 0.) / .04) * confidence[:, None]).max(2)
        if update:
            signal = activity.mean(1)
            self.trace[rows] = .88 * self.trace[rows] + .12 * signal
            self.slow[rows] = .985 * self.slow[rows] + .015 * signal
        return np.stack((short, same, long, prototype), 2)

    def route(self, rows, values, weights, *, force=None):
        x = self.features(rows, values)
        scores = x @ weights
        if force is not None:
            scores = np.eye(2, dtype=np.float32)[force]
        role, taste, code = self.assign(rows, values, scores)
        return role, taste, code


def inputs(values):
    """Match the raw physical body's float32 transfer into the common port."""
    return 2. * (.5 * (values + 1.)) - 1.


def prepare(ep, packets, weights, *, dot=False, oracle="none"):
    n, tr = ep.arena.menu.body.ctx.shape
    rows = np.arange(n)
    rec = Receptor(n, dot=dot)
    weights = np.asarray(weights, np.float32)
    choices = np.empty((n, tr), np.int8)
    slots = np.empty((n, tr), np.int8)
    chem = np.empty_like(ep.arena.menu.chemicals)
    ctx = np.empty_like(ep.arena.appearance)
    switch = ep.arena.start_test
    a_probe = ep.checks["return_A"]
    for j in range(tr):
        force = (oracle == "all" or (oracle == "initial_A" and j < switch)
                 or (oracle == "return_pair" and j >= a_probe - 1)
                 or (oracle == "return_anchor" and j == a_probe - 1)
                 or (oracle == "return_query" and j == a_probe))
        role, taste, code = rec.route(rows, inputs(packets.values[:, j]), weights,
                                     force=packets.truth[:, j] if force else None)
        choices[:, j] = role
        slots[:, j] = (code[:, None] == rec.code[None]).all(2).argmax(1)
        chem[:, j] = .5 * (taste + 1.)
        ctx[:, j] = code
    arena = replace(ep.arena, menu=replace(ep.arena.menu, chemicals=chem), appearance=ctx)
    view = replace(ep, arena=arena)
    second = packets.times[:, :, 1].copy()
    view.sensory_timing = (second, second)
    return view, choices, slots, rec.overflow.copy()


class Live:
    """Receive only the environment's mixed sensory port and ordinary physical observations."""
    def __init__(self, brain, weights, *, dot=False):
        self.brain = brain
        self.weights = np.asarray(weights, np.float32)
        self.dot = dot

    def reset(self, n):
        self.brain.reset(n)
        self.rec = Receptor(n, dot=self.dot)
        self.first = np.empty((n, 2, 4), np.float32)
        self.seen = np.zeros(n, bool)
        self.role = []

    def act(self, frame):
        if frame[:, world.TRIAL_START].all():
            self.seen.fill(False)
            self.role.append(np.full(len(frame), -1, np.int8))
        x = 2. * frame[:, food.LEFT.start:food.RIGHT.stop].reshape(-1, 2, 4) - 1.
        pulse = x.any((1, 2))
        first = pulse & ~self.seen
        self.first[first] = x[first]
        rows = np.flatnonzero(pulse & self.seen)
        obs = frame.copy()
        obs[:, food.LEFT.start:food.RIGHT.stop] = .5
        if len(rows):
            raw = np.stack((self.first[rows], x[rows]), 1)
            role, taste, context = self.rec.route(rows, raw, self.weights)
            self.role[-1][rows] = role
            obs[rows, food.LEFT.start:food.RIGHT.stop] = (.5 * (taste + 1.)).reshape(len(rows), 8)
            obs[rows, food.CTX] = context
        self.seen |= pulse
        return self.brain.act(obs)
