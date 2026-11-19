"""Exact training-stage cache for sealed relation tapes with symmetric teaching."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from . import relation_world, world
from .relation_learner import RelationLearner, _DIM, _SLOTS
from .trust_gate import FEATURES, covariance_of_others, read_prediction, update


class Recorder(RelationLearner):
    def reset(self, n):
        super().reset(n)
        self.collect = False
        self.record_noise = False
        self.noise_trace = []
        self.read_seen = np.zeros(n, dtype=bool)
        self.pairs = {}

    def act(self, obs):
        if obs[:, world.TRIAL_START].all():
            self.read_seen.fill(False)
        return super().act(obs)

    def _read(self, active):
        self.read_seen |= self.odor_seen & self.context_seen & active
        super()._read(active)

    def _noise(self):
        noise = super()._noise()
        if self.record_noise:
            self.noise_trace.append(noise.copy())
        return noise

    def _write(self, rows, slots, sensed, taste, sides):
        if not self.collect:
            return super()._write(rows, slots, sensed, taste, sides)
        # The true body's reached side and signed receipt cause this call. The
        # side is retained even though teaching presents identical endpoints.
        self.pairs.setdefault(self.round, []).append(
            (rows, slots, self.last_odor[rows, slots], sensed, self.last_taste[rows, slots], taste,
             sides, self.read_seen[rows]))


@dataclass
class Pair:
    rows: np.ndarray
    slots: np.ndarray
    previous: np.ndarray
    sensed: np.ndarray
    sign: np.ndarray
    taste: np.ndarray
    side: np.ndarray
    forecast_seen: np.ndarray
    covariance: np.ndarray


@dataclass
class Terminal:
    start_charge: np.ndarray
    start_motor: np.ndarray
    noise: np.ndarray
    distances: np.ndarray
    wind: np.ndarray
    ready: np.ndarray
    slots: np.ndarray
    previous: np.ndarray
    odor: np.ndarray
    taste: np.ndarray
    adjacent: np.ndarray
    truth: np.ndarray
    anchor: np.ndarray


def _probe_record(actor, stage):
    terminals = []
    for k in range(0, stage.menu.body.trials, 3):
        branch = deepcopy(actor)
        # Only the symmetric filler and tasted anchor precede a silent choice.
        branch.pairs.clear()
        setup = stage.slice(k, k + 2)
        _symmetric(setup)
        prefix = relation_world._play(branch, setup)
        if branch.pairs:
            raise ValueError("Probe prefix unexpectedly changed a fast matrix")
        terminal = stage.slice(k + 2, k + 3)
        charge, motor = branch.mv.copy(), branch.motor.copy()
        branch.record_noise = True
        relation_world._play(branch, terminal)
        rows = np.arange(branch.n)
        slots = branch.places.current
        terminals.append(Terminal(
            charge, motor, np.stack(branch.noise_trace),
            terminal.menu.body.distances[:, 0], terminal.menu.body.wind[:, 0],
            np.maximum(terminal.odor_tick[:, 0] + 1, terminal.context_tick[:, 0]),
            slots, branch.last_odor[rows, slots], branch.odor,
            branch.last_taste[rows, slots], branch.last_round[rows, slots] == branch.round - 1,
            terminal.truth[:, 0], (prefix.observed_reward[:, 1] != 0).any(axis=1)))
    return terminals


def _symmetric(stage):
    if not np.array_equal(stage.menu.chemicals[:, :, 0], stage.menu.chemicals[:, :, 1]):
        raise ValueError("Cache requires identical left and right teaching odors")
    if not np.array_equal(stage.menu.realized[:, :, 0], stage.menu.realized[:, :, 1]):
        raise ValueError("Cache requires identical left and right teaching tastes")


def prepare(protocol, front, motion, rule, *, old=True, initial=True):
    """Replay each physical teaching trajectory once; defer its matrix updates."""
    _symmetric(protocol.teaching)
    prepared = {}
    history_count = protocol.history["related"].menu.body.trials
    for history in relation_world.HISTORIES:
        stage = protocol.history[history]
        _symmetric(stage)
        actor = Recorder(front, protocol.seeds, motion=motion, rule=rule)
        actor.reset(len(protocol.seeds))
        hist = relation_world._play(actor, stage)
        matrix = actor.snapshot_matrix()
        actor.collect = True
        items, exposures = {}, dict(history_side=hist.contact_side,
                                     history_tick=hist.contact_tick,
                                     history_taste=hist.observed_reward.sum(axis=2))
        past = 0
        for dose in protocol.doses:
            if dose:
                segment = relation_world._play(actor, protocol.teaching.slice(3 * past // 2, 3 * dose // 2))
                for key, value in (("teaching_side", segment.contact_side), ("teaching_tick", segment.contact_tick),
                                   ("teaching_taste", segment.observed_reward.sum(axis=2))):
                    exposures.setdefault(key, []).append(value)
            pairs = []
            for request in range(history_count + 3 * past // 2, history_count + 3 * dose // 2):
                if request not in actor.pairs:
                    pairs.append(None)
                    continue
                batch = tuple(np.concatenate(columns) for columns in zip(*actor.pairs.pop(request)))
                rows, slots = batch[:2]
                pairs.append(Pair(*batch, covariance_of_others(matrix, rows, slots)))
            probes = {}
            if dose or initial:
                for metric in ("new", "old") if old else ("new",):
                    stage = protocol.probes[metric if metric == "new" else "old_" + history]
                    probes[metric] = _probe_record(actor, stage)
            items[dose] = (pairs, probes)
            past = dose
        for key in ("teaching_side", "teaching_tick", "teaching_taste"):
            exposures[key] = np.concatenate(exposures[key], axis=1)
        prepared[history] = (matrix, items, exposures)
    return prepared


def _choice(terminals, matrix, rule, motion):
    """Continue the same pulse actuation and contact dynamics for silent probes."""
    def stack(name):
        return np.stack([getattr(probe, name) for probe in terminals])

    shape = (len(terminals), len(terminals[0].slots))
    rows = np.arange(shape[1])
    signals = [read_prediction(rule, matrix[rows, t.slots], t.previous, t.odor, t.taste, t.adjacent)
               for t in terminals]
    prediction = np.stack(signals)
    diff = prediction - prediction[:, :, ::-1]
    mv, motor = stack("start_charge"), stack("start_motor")
    noise = stack("noise")
    distances, wind, onset = stack("distances"), stack("wind"), stack("ready")
    pos, vel = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    side, offset = np.full(shape, -1, dtype=np.int8), np.full(shape, -1, dtype=np.int8)
    tonic, gain, leak, fluctuation, excite, oppose, confidence = motion
    for tick in range(world.MOVEMENT_TICKS):
        active = side < 0
        if not active.any():
            break
        ready = ((tick >= onset) & active)[:, :, None]
        read = np.where(ready, diff, np.float32(0.))
        drive = (tonic + confidence * np.abs(read)) * active[:, :, None] - .85 * (~active[:, :, None])
        drive = drive.astype(np.float32, copy=False)
        drive += (gain * np.tanh(read) + fluctuation * noise[:, tick]
                  + excite * motor - oppose * motor[:, :, ::-1])
        mv = np.maximum(-1., leak * mv + drive)
        motor = (mv >= 1.).astype(np.float32)
        mv *= 1. - motor
        pressure = motor[:, :, 1] - motor[:, :, 0]
        vel = world.VELOCITY_DECAY * vel + world.PULSE_ACCELERATION * pressure + wind[:, :, tick]
        pos += vel
        left = active & (pos <= -distances[:, :, 0])
        right = active & (pos >= distances[:, :, 1])
        side[left], side[right] = 0, 1
        offset[left | right] = tick
    return dict(choice=side.T, tick=offset.T, correct=(side == stack("truth")).T, anchor=stack("anchor").T)


def evaluate(prepared, rule, motion, gate, *, overrides=None, record=False):
    """Evaluate paired physical outcomes, with optional intervention on gate values."""
    results = {}
    for history, (initial, items, exposures) in prepared.items():
        matrix = initial.copy()
        n = len(matrix)
        memory = np.zeros((n, _SLOTS, len(FEATURES)), dtype=np.float32)
        count = np.zeros((n, _SLOTS), dtype=np.int32)
        probes, trace_amount, trace_evidence = {}, [], []
        index = 0
        for dose, (pairs, terminals) in items.items():
            for pair in pairs:
                if record:
                    amounts = np.full(n, np.nan, dtype=np.float32)
                    features = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
                if pair is not None:
                    r, s = pair.rows, pair.slots
                    two_sides = np.repeat(pair.sensed[:, None, :], 2, axis=1)
                    predictions = read_prediction(rule, matrix[r, s], pair.previous, two_sides,
                                                  pair.sign, pair.forecast_seen)
                    pred = predictions[np.arange(len(r)), pair.side]
                    override = overrides[history][r, index] if overrides is not None else None
                    result = update(rule, gate, matrix[r, s], pair.covariance, pair.previous, pair.sensed,
                                    pair.sign, pair.taste, pred, memory[r, s], count[r, s], override)
                    matrix[r, s], memory[r, s], amount, evidence = result
                    count[r, s] += 1
                    if record:
                        amounts[r], features[r] = amount, evidence
                if record:
                    trace_amount.append(amounts)
                    trace_evidence.append(features)
                index += 1
            probes[dose] = {name: _choice(xs, matrix, rule, motion) for name, xs in terminals.items()}
        traces = {"gate": np.stack(trace_amount, axis=1), "evidence": np.stack(trace_evidence, axis=1)} if record else {}
        results[history] = (probes, traces, exposures)
    return results
