"""Balanced temporal-order odors ahead of the existing pulse-driven body."""

from dataclasses import dataclass, replace

import numpy as np

from . import world


ODORS = 3
MARKER = 3
MOTOR = 4
SENSES = 5
RECOVERY = 3


@dataclass(frozen=True)
class Timing:
    sensory_ticks: int
    gap_low: int
    gap_high: int
    distractors: int


TIMING = {
    "standard": Timing(27, 6, 12, 6),
    "long": Timing(44, 18, 26, 12),
}


@dataclass(frozen=True)
class OrderTape:
    body: world.Tape
    observations: np.ndarray  # [request, clock, life, sense]
    target_ticks: np.ndarray  # [life, request, ordinal]
    target_symbols: np.ndarray  # [life, request, ordinal]
    target_vectors: np.ndarray  # [life, request, ordinal, odor channel]
    timing: Timing


def _sides(symbols):
    return (((symbols[..., 1] - symbols[..., 0]) % ODORS) == 1).astype(np.int8)


def make(roots, requests, profile):
    """Make random exogenous tapes and use the existing body for target contact."""
    roots = tuple(int(root) for root in roots)
    timing = TIMING[profile]
    body = world.make_tape(roots, trials=requests, distance_jitter=.19, wind_scale=.10)
    length = timing.sensory_ticks + world.TICKS + RECOVERY
    observations = np.zeros((requests, length, len(roots), SENSES), dtype=np.float32)
    observations[:, timing.sensory_ticks:timing.sensory_ticks + world.TICKS, :, MOTOR] = 1
    target_ticks = np.empty((len(roots), requests, 2), dtype=np.int8)
    target_symbols = np.empty_like(target_ticks)
    target_vectors = np.empty((*target_ticks.shape, ODORS), dtype=np.float32)
    for life, root in enumerate(roots):
        schedule, identities, noise = (np.random.default_rng(seed) for seed in
                                       np.random.SeedSequence([root, 40061507]).spawn(3))
        for req in range(requests):
            tick_1 = schedule.integers(2, 7)
            tick_2 = tick_1 + schedule.integers(timing.gap_low, timing.gap_high + 1)
            times = np.asarray([tick_1, tick_2])
            target_ticks[life, req] = times
            available = np.setdiff1d(np.arange(1, timing.sensory_ticks - 2), times)
            others = schedule.choice(available, timing.distractors, replace=False)
            markers = observations[req, :, life, MARKER]
            markers[times] = noise.uniform(.77, 1.08, 2)
            markers[others] = noise.uniform(.00, .24, len(others))
            background = observations[req, :timing.sensory_ticks, life, :ODORS]
            background[:] = noise.normal(0, .022, background.shape)

            # Timing, markings, layout, geometry, and distractors are independent of order.
            pair = identities.choice(ODORS, 2, replace=False)
            target_symbols[life, req] = pair
            vectors = noise.normal(0, .035, (2, ODORS)).astype(np.float32)
            vectors[np.arange(2), pair] += noise.uniform(.80, 1.22, 2)
            target_vectors[life, req] = vectors
            background[times] = vectors
            decoys = identities.integers(0, ODORS, len(others))
            background[others, decoys] += noise.uniform(.73, 1.25, len(others))

    good_side = _sides(target_symbols)
    good_id = np.where(good_side == 0, body.left_id, 1 - body.left_id).astype(np.int8)
    return OrderTape(replace(body, good_id=good_id), observations, target_ticks,
                     target_symbols, target_vectors, timing)


def fork(tape, change):
    """Replace only the final request and keep every earlier observation unchanged."""
    obs = tape.observations.copy()
    symbols = tape.target_symbols.copy()
    vectors = tape.target_vectors.copy()
    life = np.arange(tape.body.n)
    times = tape.target_ticks[:, -1]
    if change == "swap":
        symbols[:, -1] = symbols[:, -1, ::-1].copy()
        # Carry the whole odor vector. The unordered set of observed odor vectors is exact.
        vectors[:, -1] = vectors[:, -1, ::-1].copy()
        for slot in (0, 1):
            obs[-1, times[:, slot], life, :ODORS] = vectors[:, -1, slot]
    elif change in ("first_only", "second_only", "first_only_flip", "second_only_flip"):
        missing = 1 if change.startswith("first_only") else 0
        if change.endswith("_flip"):
            symbols[:, -1, missing] = ODORS * (ODORS - 1) // 2 - symbols[:, -1].sum(axis=-1)
        obs[-1, times[:, missing], life, :ODORS] = 0
    elif change in ("timing_only", "timing_only_flip"):
        obs[-1, :, :, :ODORS] = 0
        if change.endswith("_flip"):
            symbols[:, -1] = symbols[:, -1, ::-1].copy()
    else:
        raise ValueError(f"Unknown fork {change}")

    good_side = _sides(symbols)
    good_id = np.where(good_side == 0, tape.body.left_id, 1 - tape.body.left_id).astype(np.int8)
    return replace(tape, observations=obs, target_symbols=symbols,
                   target_vectors=vectors, body=replace(tape.body, good_id=good_id))


def physics(tape, right):
    """Convert each motor tick's binary controller output into a physical pulse."""
    expected = (tape.body.trials, world.TICKS, tape.body.n)
    if right.shape != expected:
        raise ValueError(f"Expected motor decisions shaped {expected}, received {right.shape}")
    body = world.PulseWorld(tape.body, feedback="true")
    for request in right:
        for tick in request:
            body.step(np.column_stack((~tick, tick)))
    return body.result()


def explore(tape):
    """Deploy a predeclared random open-loop policy and return its actual receipts."""
    decisions = np.asarray([
        np.random.default_rng(np.random.SeedSequence([root, 354791])).integers(
            0, 2, tape.body.trials, dtype=np.int8)
        for root in tape.body.seeds
    ], dtype=bool).T
    return physics(tape, np.broadcast_to(decisions[:, None, :],
                                         (tape.body.trials, world.TICKS, tape.body.n)))


def contact_receipts(result):
    """Return contacted side and publicly delivered signed taste, in request-major order."""
    taste = result.observed_reward.sum(axis=-1, dtype=np.int8)
    if not np.array_equal(taste, result.reward):
        raise RuntimeError("An observed contact failed to deliver its signed receipt")
    return result.contact_side.T, taste.T


def chronological_oracle(tape):
    """Decode visible pulses with the planted order rule, then drive the same body."""
    pulse_input = tape.observations[:, :tape.timing.sensory_ticks]
    certain = pulse_input[..., MARKER] > .5
    directions = np.zeros((tape.body.trials, tape.body.n), dtype=bool)
    for req in range(tape.body.trials):
        for life in range(tape.body.n):
            if certain[req, :, life].sum() == 2:
                decoded = pulse_input[req, certain[req, :, life], life, :ODORS].argmax(axis=-1)
                directions[req, life] = _sides(decoded)
    return physics(tape, np.broadcast_to(directions[:, None, :],
                                         (tape.body.trials, world.TICKS, tape.body.n)))
