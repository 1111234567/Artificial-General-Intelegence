"""Exhaustive paired odor queries with independently sampled public prefixes."""

from dataclasses import replace

import numpy as np

from . import world


ODORS = np.array([
    [1., 0., 0.], [0., .6, .8], [0., 1., 0.], [.8, 0., .6], [0., 0., 1.]
], dtype=np.float32)
SENSES = 5
PAIRS = np.array([(a, b) for a in range(len(ODORS)) for b in range(len(ODORS)) if a != b],
                 dtype=np.int8)
SWAP = np.array([np.flatnonzero(np.all(PAIRS == [b, a], axis=-1))[0] for a, b in PAIRS])
RULES = ("orientation", "adjacency", "lookup")
LOOKUP = ((1, 3), (0, 2), (1, 4), (2, 4), (0, 3))

# Sensory length, inclusive gap limits, distractor count
TIMING = {"standard": (24, 8, 12, 5), "long": (34, 18, 22, 8)}
PREFIX_REQUESTS = 1
RECOVERY = 2


def truth():
    first, second = PAIRS.T
    delta = (second - first) % len(ODORS)
    return np.stack((np.isin(delta, (1, 2)), np.isin(delta, (1, 4)),
                     np.array([b in LOOKUP[a] for a, b in PAIRS])))


def _request(rng, profile):
    sensory, gap_low, gap_high, count = TIMING[profile]
    start = int(rng.integers(2, 6))
    ticks = (start, start + int(rng.integers(gap_low, gap_high + 1)))
    obs = np.zeros((sensory + world.TICKS + RECOVERY, SENSES), dtype=np.float32)
    obs[sensory:sensory + world.TICKS, 4] = 1
    obs[:sensory, :3] = rng.normal(0, .018, (sensory, 3))
    eligible = np.setdiff1d(np.arange(1, sensory - 2), ticks)
    decoys = rng.choice(eligible, count, replace=False)
    obs[decoys, 3] = rng.uniform(.03, .24, count)
    decoy_id = rng.integers(len(ODORS), size=count)
    obs[decoys, :3] += ODORS[decoy_id] * rng.uniform(.75, 1.20, (count, 1))
    obs[list(ticks), 3] = rng.uniform(.80, 1.10, 2)
    # Each odor keeps its entire vector under swapping. All variants share this
    # realization as well as timing, decoys, and every other background value.
    vectors = ODORS * rng.uniform(.82, 1.22, (len(ODORS), 1))
    vectors += rng.normal(0, .022, vectors.shape)
    return obs, ticks, vectors.astype(np.float32)


def _place(obs, ticks, vectors):
    obs[..., list(ticks), :3] = vectors
    return obs


def sensory(roots, profile):
    """Return one public precursor and all 20 query forks for every episode root."""
    prefix = []
    query = []
    target_ticks = []
    for root in roots:
        identity, history, future = (np.random.default_rng(s) for s in
                                     np.random.SeedSequence([int(root), 6210731]).spawn(3))
        prior = []
        for _ in range(PREFIX_REQUESTS):
            start, ticks, scents = _request(history, "standard")
            pair = identity.choice(len(ODORS), 2, replace=False)
            prior.append(_place(start, ticks, scents[pair]))
        prefix.append(np.concatenate(prior))
        base, ticks, scents = _request(future, profile)
        forks = np.broadcast_to(base, (len(PAIRS), *base.shape)).copy()
        query.append(_place(forks, ticks, scents[PAIRS]))
        target_ticks.append(ticks)
    return np.stack(prefix), np.stack(query), np.asarray(target_ticks, dtype=np.int8)


def body_tape(roots):
    """One physical request per fork, with common geometry within every root."""
    base = world.make_tape(roots, trials=1, distance_jitter=.19, wind_scale=.10)
    count = len(PAIRS)
    return replace(
        base,
        seeds=tuple(seed for seed in base.seeds for _ in PAIRS),
        ctx=np.repeat(base.ctx, count, axis=0),
        cue=np.repeat(base.cue, count, axis=0),
        left_id=np.repeat(base.left_id, count, axis=0),
        good_id=np.repeat(base.good_id, count, axis=0),
        good_map=np.repeat(base.good_map, count, axis=0),
        distances=np.repeat(base.distances, count, axis=0),
        wind=np.repeat(base.wind, count, axis=0),
    )


def _drive(tape, right):
    motor = world.PulseWorld(tape, feedback="true")
    for tick in right:
        motor.step(np.column_stack((~tick, tick)))
    return motor.result()


def _with_truth(tape, row):
    direction = np.tile(row, tape.n // len(PAIRS))[:, None]
    return replace(tape, good_id=np.where(direction, 1 - tape.left_id, tape.left_id))


def exploration(tape, seed):
    """Randomly chosen physical sides and their actual publicly observed tastes."""
    right = np.random.default_rng(seed).integers(2, size=tape.n, dtype=np.int8).astype(bool)
    controls = np.broadcast_to(right, (world.TICKS, tape.n))
    results = [_drive(_with_truth(tape, row), controls) for row in truth()]
    tastes = np.stack([result.observed_reward[:, 0].sum(axis=-1, dtype=np.int8)
                       for result in results])
    contacted = results[0].contact_side[:, 0]
    if any(not np.array_equal(r.contact_side[:, 0], contacted) for r in results):
        raise RuntimeError("Exogenous exploration should have identical trajectories across rules")
    if any(not np.array_equal(r.reward[:, 0], t) for r, t in zip(results, tastes, strict=True)):
        raise RuntimeError("Only delivered public tastes may enter training")
    return (contacted.reshape(-1, len(PAIRS)),
            tastes.reshape(len(RULES), -1, len(PAIRS)))


def contacts(tape, right, rule):
    """Score every fork with its body's first physical contact, including timeouts."""
    result = _drive(_with_truth(tape, truth()[rule]), right)
    return np.stack((result.contact_side[:, 0], result.contact_tick[:, 0], result.reward[:, 0]),
                    axis=-1)


def _public_oracle(inputs):
    carriers = inputs[..., 3] > .5
    odors = inputs[carriers, :3].reshape(*inputs.shape[:2], 2, 3)
    decoded = (odors @ ODORS.T).argmax(axis=-1)
    return decoded


def oracle(tape, inputs, rule):
    decoded = _public_oracle(inputs)
    index_table = np.full((len(ODORS), len(ODORS)), -1)
    index_table[PAIRS[:, 0], PAIRS[:, 1]] = np.arange(len(PAIRS))
    cells = index_table[decoded[..., 0], decoded[..., 1]]
    side = (truth()[rule][cells] & (cells >= 0)).reshape(-1)
    choices = np.broadcast_to(side, (world.TICKS, tape.n))
    return contacts(tape, choices, rule)
