"""Local taste circuitry and explicit or reward-trained exploration policies."""

from dataclasses import dataclass
from functools import lru_cache
from copy import copy
from itertools import permutations

import numpy as np

from . import active_symmetry_world as game, mixture_world as food, world
from .symmetry_spike import Skill, SpikingSkill


BASE = Skill(encoding="voltage", update_gain="correlation")
SENSE_STATE = ("v", "spikes", "z", "sense", "chem", "mean", "gate", "motor_v", "motor_spikes")
ORDERS = np.array(tuple(permutations(range(4))))
NOISE = .04
P_AGREE = 1 - 2 * NOISE * (1 - NOISE)
LOGODDS = np.log(P_AGREE / (1 - P_AGREE))


class Phase:
    """A harvested dish starts a clean ephemeral anchor bank in the seen habitat."""

    def __init__(self, brain, explore):
        self.brain, self.explore = brain, explore

    def reset(self, n):
        self.local = -1

    def act(self, observation):
        self.local += int(observation[0, world.TRIAL_START] > 0)
        if not self.explore and self.local == 0:
            seen = observation[:, food.CTX]
            ready = np.flatnonzero(seen.any(axis=1))
            self.brain.weights[ready, seen[ready].argmax(axis=1)] = 0.
        return self.brain.act(observation)


class Cortex:
    """Persistent sensory state plus fixed-capacity local adaptation per arena."""

    def __init__(self, seeds, birth, *, public_reference=False):
        self.seeds = seeds
        self.public_reference = public_reference
        cfg = Skill(encoding="linear", gain_rate=0.) if public_reference else BASE
        self.prototype = SpikingSkill(cfg, birth=birth)
        n = len(seeds)
        self.prototype.reset(n)
        self.perception = {name: getattr(self.prototype, name).copy() for name in SENSE_STATE}
        self.memory = np.zeros((n, 2, 2, self.prototype.dims), dtype=np.float32)
        self.gain = np.zeros((n, 2), dtype=np.float32)
        self.evidence = np.zeros((n, 2), dtype=np.int16)
        self.receipts = np.zeros((n, 2), dtype=np.int32)

    def session(self, rows, arena, ticket, portion, explore):
        brain = copy(self.prototype)
        brain.action_seeds = [np.random.SeedSequence((self.seeds[i], ticket, arena, int(explore), 0xA41217)) for i in rows]
        brain.reset(len(rows))
        for name, backing in self.perception.items():
            setattr(brain, name, backing[rows].copy())
        brain.weights = self.memory[rows, arena].copy()
        if self.public_reference:
            brain.direction = (-.5 * LOGODDS * self.evidence[rows, arena]).astype(np.float32)
        else:
            brain.direction = self.gain[rows, arena].copy()
        trajectory = game.execute_taste(Phase(brain, explore), portion,
                                        explore=explore, record=self.public_reference and explore)
        for name, backing in self.perception.items():
            backing[rows] = getattr(brain, name)
        self.memory[rows, arena] = brain.weights
        self.gain[rows, arena] = brain.direction
        self.receipts[rows, arena] += brain.receipts
        if self.public_reference and explore:
            # A labelled sensory grammar is permitted only for this public
            # reference: match the two four-item groups by observed odors.
            od = ((trajectory.obs[..., food.LEFT] * 2. - 1.).sum(axis=2)).astype(np.float32)
            match = -od[:, :4] @ od[:, 4:].transpose(0, 2, 1)
            score = match[:, np.arange(4), ORDERS].sum(axis=2)
            positions = ORDERS[score.argmax(axis=1)]
            sy = trajectory.observed_reward.sum(axis=2)
            evidence = (sy[:, :4] * np.take_along_axis(sy[:, 4:], positions, axis=1)).sum(1)
            self.evidence[rows, arena] += evidence.astype(np.int16)
        return trajectory


def features(offer, gain):
    """Raw public quantities and an untrained nonlinear basis for a small gate."""
    ids = np.arange(len(gain))
    now = np.abs(gain[ids, offer.arena])
    other = np.abs(gain[ids, 1 - offer.arena])
    cost = offer.costs[ids, offer.arena]
    rest = offer.credits
    inputs = np.column_stack((np.tanh(now), np.tanh(other), cost, offer.costs[ids, 1-offer.arena],
                              rest / 14., offer.arena.astype(float)))
    # Frozen random expansion has no estimate of information value or teacher.
    return np.column_stack((np.ones(len(gain)), inputs, np.tanh(inputs @ BASIS + OFFSET)))


_rng = np.random.default_rng(8453291)
BASIS = _rng.normal(0., 1.4, (6, 12))
OFFSET = _rng.normal(0., .5, 12)
WIDTH = 1 + 6 + 12


def gate_randomness(seeds, length):
    return np.stack([np.random.default_rng(np.random.SeedSequence((s, 0xAB876))).random(length) for s in seeds])


def known(x):
    return 1. / (1. + np.exp(-np.clip(LOGODDS * x, -45, 45)))


@lru_cache(maxsize=None)
def ideal(cost0, cost1, horizon):
    """Public Bayes decision value assuming correct pairing of observed odors."""
    bound = 4 * horizon
    states = np.arange(-bound, bound + 1)
    prob = known(states)
    deltas = np.arange(0, 5).astype(np.int16) * 2 - 4
    combos = np.array([1., 4., 6., 4., 1.])
    p = combos * P_AGREE**np.arange(5) * (1 - P_AGREE)**np.arange(4, -1, -1)
    predictive = prob[:, None] * p + (1 - prob[:, None]) * p[::-1]
    new = np.clip(np.arange(len(states))[:, None] + deltas, 0, len(states) - 1)
    now = (2 * np.maximum(prob, 1 - prob) - 1) * (1 - 2 * NOISE)
    prior = np.zeros((len(states), len(states)), dtype=np.float64)
    margin = [None]
    beginning = [0.]
    for _ in range(horizon):
        current0 = now[:, None] + prior
        current1 = now[None, :] + prior
        probe0 = (prior[new, :] * predictive[:, :, None]).sum(1) - cost0
        probe1 = (prior.T[new, :] * predictive[:, :, None]).sum(1).T - cost1
        margin.append(np.stack((probe0-current0, probe1-current1)))
        prior = .5 * (np.maximum(probe0, current0) + np.maximum(probe1, current1))
        beginning.append(float(prior[bound, bound]))
    return margin, bound, beginning


@dataclass(frozen=True)
class Gate:
    name: str
    params: np.ndarray | None = None
    noisy: bool = False
    quantum: np.ndarray | None = None


def decide(gate, offer, gain, evidence, earlier, rows, tick, toss):
    design = features(offer, gain)
    ids = np.arange(len(gain))
    arena = offer.arena
    used = earlier[ids, arena]
    confidence = np.abs(np.tanh(gain[ids, arena]))
    if gate.name == "none":
        selected = np.zeros(len(gain), dtype=bool)
    elif gate.name == "once":
        selected = (used == 0) & (offer.credits > 1)
    elif gate.name == "twice":
        selected = (used < 2) & (offer.credits > 1)
    elif gate.name == "matched":
        selected = gate.quantum[rows, tick]
    elif gate.name == "heuristic":
        # Declared hand-set estimate of how much signed harvesting can improve.
        opportunity = .5 * (offer.credits - 1)
        available = .37 * (1 - confidence) * opportunity
        selected = available > offer.costs[ids, arena] + confidence
    elif gate.name == "public":
        selected = np.zeros(len(gain), dtype=bool)
        limit = max(int(offer.credits.max()), max(game.HORIZONS))
        for i, h in enumerate(offer.credits):
            margins, bound, _ = ideal(*map(float, offer.costs[i]), limit)
            k, j = np.clip(evidence[i] + bound, 0, 2 * bound)
            selected[i] = margins[int(h)][arena[i], k, j] > 0.
    elif gate.name in ("learned", "learned_blind"):
        if gate.name == "learned_blind":
            design = features(offer, np.zeros_like(gain))
        logit = design @ gate.params
        p = 1. / (1. + np.exp(-np.clip(logit, -16, 16)))
        return toss < p if gate.noisy else p >= .5, design, p
    else:
        raise ValueError(f"Unknown exploration policy {gate.name}")
    return selected, design, selected.astype(np.float64)


def lifetime(journey, gate, birth, *, toss=None):
    mode = gate.name == "public"
    cortex = Cortex(journey.seeds, birth, public_reference=mode)
    n, length = len(journey.seeds), journey.ticks
    if toss is None:
        toss = gate_randomness(journey.seeds, length)
    shape = (n, length)
    rewards = np.zeros(shape, dtype=np.float64)
    paid = np.zeros_like(rewards)
    traces = np.zeros((*shape, WIDTH), dtype=np.float32)
    probabilities = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    intentions = np.zeros(shape, dtype=bool)
    active_tickets = np.arange(length)[None, :] < journey.horizon[:, None]
    chosen_probes = np.zeros(shape, dtype=bool)
    harvested = np.zeros(shape, dtype=bool)
    landed = np.zeros(shape, dtype=bool)
    learned = np.zeros((n, 2), dtype=np.int16)
    for tick in range(length):
        rows = np.flatnonzero(active_tickets[:, tick])
        if not len(rows):
            continue
        staged = journey.arena[rows, tick]
        now = game.Offer(staged, journey.costs[rows], journey.horizon[rows] - tick, journey.probe_side[rows, tick])
        want, x, p = decide(gate, now, cortex.gain[rows], cortex.evidence[rows], learned[rows], rows, tick, toss[rows, tick])
        _, probe, arrived = game.reach_gate(journey, rows, tick, want)
        traces[rows, tick], probabilities[rows, tick], intentions[rows, tick] = x, p, want
        confidence[rows, tick] = abs(np.tanh(cortex.gain[rows, staged]))
        landed[rows, tick] = arrived
        chosen_probes[rows, tick] = probe
        harvested[rows, tick] = arrived & ~probe
        paid[rows[probe], tick] = journey.costs[rows[probe], staged[probe]]
        for arena in (0, 1):
            for explore in (True, False):
                selected = rows[arrived & (staged == arena) & (probe == explore)]
                if not len(selected):
                    continue
                portion = game.assembly(journey, arena, tick, selected, explore)
                out = cortex.session(selected, arena, tick, portion, explore)
                if explore:
                    learned[selected, arena] += 1
                else:
                    rewards[selected, tick] = out.reward[:, 1]
    return {"net": (rewards - paid).sum(1), "rewards": rewards, "paid": paid, "traces": traces,
            "probabilities": probabilities, "confidence": confidence, "intentions": intentions, "landed": landed,
            "harvest": harvested, "active": active_tickets, "probe": chosen_probes, "counts": learned}
