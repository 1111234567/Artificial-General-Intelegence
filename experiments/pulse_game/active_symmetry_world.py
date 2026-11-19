"""Embodied choices between buying balanced odor evidence and harvesting now."""

from dataclasses import dataclass, replace

import numpy as np

from . import mixture_flash, symmetry_world, world


COSTS = (.10, .36, .72)
HORIZONS = (4, 8, 14)
BLOCK = 8  # four pairs of physically tasted odors per successful probe


@dataclass(frozen=True)
class Journey:
    seeds: tuple
    arena: np.ndarray
    horizon: np.ndarray
    costs: np.ndarray
    probe_side: np.ndarray
    markers: world.Tape
    protocols: tuple
    flip: int

    @property
    def ticks(self):
        return self.arena.shape[1]


@dataclass(frozen=True)
class Offer:
    """All information displayed before choosing a gate. Future visits are hidden."""

    arena: np.ndarray
    costs: np.ndarray
    credits: np.ndarray
    probe_side: np.ndarray


def build(seeds, *, flip=1, horizons=HORIZONS, costs=COSTS, noise=.04):
    seeds = tuple(map(int, seeds))
    n = len(seeds)
    length = max(horizons)
    choices = np.zeros((n, length), dtype=np.int8)
    markers = np.empty_like(choices)
    h = np.empty(n, dtype=np.int16)
    c = np.empty((n, 2), dtype=np.float64)
    hidden = np.empty((n, 2), dtype=np.int8)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence((s, 0x018B4FE)))
        h[i] = rng.choice(horizons)
        c[i] = rng.choice(costs, size=2)
        hidden[i] = rng.choice((-1, 1), size=2)
        choices[i] = rng.integers(2, size=length, dtype=np.int8)
        markers[i] = rng.integers(2, size=length, dtype=np.int8)
    plate = world.make_tape([s + 0x120FE82A for s in seeds], length, distance_jitter=.10, wind_scale=.06)
    flavors = tuple(symmetry_world.make([s + 0x300E051B * (a + 1) for s in seeds], pairs=4 * length, probes=length,
                                        interleave=4, family=flip * hidden[:, a], noise=noise,
                                        balanced_stream=True) for a in range(2))
    return Journey(seeds, choices, h, c, markers, plate, flavors, flip)


def body_region(tape, rows, start, count, salt):
    region = slice(start, start + count)
    kept = {name: getattr(tape, name)[rows, region] for name in ("ctx", "cue", "left_id", "good_id", "distances", "wind")}
    return replace(tape, **kept, good_map=tape.good_map[rows], seeds=tuple(tape.seeds[i] + salt for i in rows))


def assembly(journey, arena, tick, rows, explore):
    protocol = journey.protocols[arena]
    native = protocol.menu
    if explore:
        lo, length = BLOCK * tick, BLOCK
    else:
        lo, length = protocol.exposed + 2 * tick, 2
    body = body_region(native.body, rows, lo, length, 17_281 * (tick + 1))
    cut = slice(lo, lo + length)
    return replace(native, body=body, chemicals=native.chemicals[rows, cut],
                   probability=native.probability[rows, cut], realized=native.realized[rows, cut])


class FlagMotor:
    """Transparent fixed routing from the selected sign to a physical side."""

    def __init__(self, mark):
        self.mark = mark

    def reset(self, n):
        if len(self.mark) != n:
            raise ValueError("mismatched bodies")

    def act(self, observation):
        drive = observation[:, world.ACTIVE] > 0
        return np.column_stack((drive & (self.mark == 0), drive & (self.mark == 1)))


def reach_gate(journey, rows, tick, wants_probe):
    offered = Offer(journey.arena[rows, tick], journey.costs[rows], journey.horizon[rows] - tick,
                    journey.probe_side[rows, tick])
    wanted = np.where(wants_probe, offered.probe_side, offered.probe_side ^ 1)
    stage = body_region(journey.markers, rows, tick, 1, 9247 * (tick + 1))
    seen = world.rollout(FlagMotor(wanted), stage, feedback="none")
    actual = seen.contact_side[:, 0]
    found = actual >= 0
    probe = found & (actual == offered.probe_side)
    return offered, probe, found


def execute_taste(policy, menu, *, explore, record=False):
    # Rewards are credited externally on harvest. Its first anchor taste still
    # emits a causal receipt. Probe tastes inform the learner without a payoff.
    return mixture_flash.rollout(policy, menu, hidden_after=None if explore else 1, record=record)
