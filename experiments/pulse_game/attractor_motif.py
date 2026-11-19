"""CPU intervention primitives for frozen recurrent delay encoders."""

from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np

from . import attractor_search as engine, ring_world


RULE = "attractor-overnight-mixed-full-437918"
GAP = 28
PHYSICAL = 40
WINDOWS = {"early": (1, 9), "middle": (10, 18), "late": (19, 27), "post": (29, 37), "all": (1, 28)}


def vector():
    path = engine.OUT / RULE / "search.json"
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["best"]["vector"], dtype=np.float64)


def curriculum(seeds, trials=63):
    tape = engine.task(seeds, "long", trials=trials)
    return replace(tape, query=tape.issue + GAP, length=np.full_like(tape.length, PHYSICAL), at=tape.issue + 14)


@dataclass
class State:
    membrane: np.ndarray
    adaptation: np.ndarray
    trace: np.ndarray
    synaptic: np.ndarray
    mean: np.ndarray

    def clone(self):
        return State(*(x.copy() for x in self.arrays()))

    def arrays(self):
        return self.membrane, self.adaptation, self.trace, self.synaptic, self.mean

    def duplicate(self, size):
        return State(*(np.repeat(x, size, axis=0) for x in self.arrays()))


class Encoder:
    def __init__(self, v, birth, *, exact=False):
        self.inp, self.offset, self.src, self.weight, self.proj, params, self.rate = engine.genome(v, birth)
        self.leak, self.syn_decay, self.trace_decay, self.adapt_decay, self.adapt, self.mean_rate = params
        self.matrix = self.to_matrix(self.weight)
        self.exact = exact

    def to_matrix(self, weight):
        matrix = np.zeros((engine.N, engine.N), dtype=np.float32)
        matrix[np.arange(engine.N)[:, None], self.src] = weight
        return matrix

    def start(self, n):
        z = np.zeros((n, engine.N), dtype=np.float32)
        return State(z.copy(), z.copy(), z.copy(), z.copy(), np.full_like(z, .12))

    def drive(self, synaptic):
        if not self.exact:
            return synaptic @ self.matrix.T
        current = np.zeros_like(synaptic)
        for k in range(engine.D):
            current += self.weight[:, k] * synaptic[:, self.src[:, k]]
        return current

    def read(self, q):
        value = q.trace - q.mean
        if not self.exact:
            return value @ self.proj.T
        out = np.zeros((len(value), engine.K), dtype=np.float32)
        for i in range(engine.N):
            out += value[:, i, None] * self.proj[:, i]
        return out

    def step(self, q, drive):
        q.membrane = np.maximum(-3., self.leak * q.membrane + drive)
        fired = (q.membrane >= self.offset[:, 1] + self.adapt * q.adaptation).astype(np.float32)
        q.membrane[fired > 0] = 0.
        q.adaptation = self.adapt_decay * q.adaptation + (1. - self.adapt_decay) * fired
        q.trace = self.trace_decay * q.trace + (1. - self.trace_decay) * fired
        q.mean += self.mean_rate * (q.trace - q.mean)
        q.synaptic = self.syn_decay * q.synaptic + (1. - self.syn_decay) * fired
        return fired

    def run(self, tape):
        n, count = tape.cue.shape
        q = self.start(n)
        features = np.empty((n, count, engine.K), dtype=np.float32)
        pulses = 0
        if not (tape.length == tape.length[0, 0]).all():
            raise ValueError("Prefix simulation requires one fixed native trial length")
        for trial in range(count):
            cue = self.inp[:, 2 + tape.cue[:, trial]].T
            ctx = self.inp[:, tape.ctx[:, trial]].T
            at = tape.raw_at[:, trial]
            early = ctx + np.where((at == 255)[:, None], cue, 0.) + self.inp[:, 6]
            noise = tape.distractor[:, trial]
            irrelevant = (self.inp[:, noise // 4] + self.inp[:, 2 + noise % 4]).T
            for tick in range(int(tape.length[0, 0])):
                drive = self.offset[:, 0] + self.drive(q.synaptic)
                drive += early * (tick == tape.issue[:, trial, None])
                drive += cue * (tick == at[:, None])
                drive += irrelevant * (tick == tape.at[:, trial, None])
                pulses += int(self.step(q, drive).sum())
                ready = tick == tape.query[:, trial]
                if ready.any():
                    features[ready, trial] = self.read(q)[ready]
        return q, features, pulses / (engine.physical_time(tape) * engine.N)


def normalize(x):
    return x * (1. / np.sqrt(.01 + np.square(x).sum(-1, keepdims=True)))


def readout_on_prefix(v, feats, tape):
    """Actual selected-identity signs train the four outputs before the fork."""
    n, count = tape.goal.shape
    eta = np.float32(engine.decode(v)["plastic_rate"])
    w = np.zeros((n, 4, engine.K), dtype=np.float32)
    cov = np.broadcast_to(np.eye(engine.K, dtype=np.float32) * 4., (*w.shape, engine.K)).copy()
    guesses = np.empty_like(tape.goal)
    ids = np.arange(n)
    for j in range(count):
        f = feats[:, j]
        mag = np.full(n, .01, dtype=np.float32)
        for k in range(engine.K):
            mag += f[:, k] * f[:, k]
        x = f * (1. / np.sqrt(mag[:, None]))
        val = np.zeros((n, 4), dtype=np.float32)
        for k in range(engine.K):
            val += w[:, :, k] * x[:, None, k]
        pick = (val + 1e-5 * tape.tie[:, j]).argmax(axis=1)
        guesses[:, j] = pick
        old = cov[ids, pick]
        z = np.zeros_like(x)
        scale = np.full(n, 1., dtype=np.float32)
        for k in range(engine.K):
            z += old[:, :, k] * x[:, k, None]
        for k in range(engine.K):
            scale += x[:, k] * z[:, k]
        scale = 1. / scale
        cov[ids, pick] = old - (scale[:, None, None] * z[:, :, None]) * z[:, None, :]
        reward = np.where(tape.goal[:, j] == pick, 1., -1.).astype(np.float32)
        w[ids, pick] += (eta * (reward - val[ids, pick]) * scale)[:, None] * z
    return w, guesses


@dataclass(frozen=True)
class Probes:
    start: State
    noise: np.ndarray
    labels: np.ndarray
    goals: np.ndarray
    tie: np.ndarray


def probes(q, seeds):
    n = len(seeds)
    pref = ring_world.make_tape(seeds, 1).preferences
    random = [np.random.default_rng(np.random.SeedSequence((s, 861749))) for s in seeds]
    noise = np.repeat([r.integers(8) for r in random], 8)
    tie = np.stack([r.random((8, 4), dtype=np.float32) for r in random])
    return Probes(q.duplicate(8), noise, np.tile(np.arange(8), n), pref.reshape(n, 8), tie)


def responses(enc, experiment, *, mask=None, window=None, operation="zero", baseline=None,
              swap=1, collect=False, time=PHYSICAL, kick=None, distractor=True, query_at=GAP):
    """One fixed-length cue sequence forked from the original life prefix."""
    q = experiment.start.clone()
    labels = experiment.labels
    drive0 = (enc.inp[:, labels // 4] + enc.inp[:, 2 + labels % 4] + enc.inp[:, 6, None]).T
    noise = experiment.noise
    distract = (enc.inp[:, noise // 4] + enc.inp[:, 2 + noise % 4]).T
    pulses = np.zeros((len(labels), time), dtype=np.int16)
    if collect:
        states = {name: np.empty((time, len(labels), engine.N), dtype=np.float32)
                  for name in ("synaptic", "spike", "trace", "membrane", "adaptation")}
        states["features"] = np.empty((time, len(labels), engine.K), dtype=np.float32)
    else:
        states = None
    matrix = None if mask is None else enc.to_matrix(enc.weight * mask)
    donor = (np.arange(len(labels)) // 8) * 8 + (labels ^ swap)
    drive_before = np.zeros((time, engine.N), dtype=np.float32)
    for tick in range(time):
        before = q.synaptic
        if collect:
            states["synaptic"][tick] = before
        current = enc.drive(before)
        if matrix is not None and window[0] <= tick < window[1]:
            if operation == "zero":
                current -= before @ matrix.T
            elif operation == "mean":
                current += (baseline[tick].mean(axis=0)[None] - before) @ matrix.T
            elif operation == "replay":
                current += (baseline[tick][donor] - before) @ matrix.T
            else:
                raise ValueError(operation)
        drive_before[tick] = current.mean(axis=0)
        drive = enc.offset[:, 0] + current
        if tick == 0:
            drive += drive0
        if tick == 14 and distractor:
            drive += distract
        if kick is not None and tick == kick[0]:
            q.membrane += kick[1]
        fired = enc.step(q, drive)
        pulses[:, tick] = fired.sum(axis=1)
        if collect:
            states["spike"][tick] = fired
            for name in ("trace", "membrane", "adaptation"):
                states[name][tick] = getattr(q, name)
            states["features"][tick] = enc.read(q)
        if tick == query_at:
            result = enc.read(q).reshape(len(labels) // 8, 8, engine.K)
    return {"address": result, "state": states, "pulses": pulses, "mean_input": drive_before}


def choices(w, result, test):
    val = np.einsum("nfk,njk->njf", w, normalize(result["address"]), optimize=False)
    return (val + 1e-5 * test.tie).argmax(axis=2)
