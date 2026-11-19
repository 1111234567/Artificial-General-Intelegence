"""Training-only private rule distributions for the unchanged Scout simulator."""

from dataclasses import dataclass, replace

import numpy as np

from .scout_world import ScoutTape
from .world import VELOCITY_DECAY


@dataclass(frozen=True)
class RangeTape(ScoutTape):
    range_override: float = 0.35

    @property
    def scout_distance(self):
        return self.range_override


def scaled_geometry(tape, scale, beacon):
    distances = tape.distances * np.float32(scale)
    wind = tape.wind * np.float32(scale)
    if distances.min() <= beacon or np.abs(wind).max() >= beacon * (1 - VELOCITY_DECAY):
        raise ValueError("Geometry must put the beacon before endpoints and keep gusts too small to trigger it alone")
    base = {**vars(tape), "distances": distances, "wind": wind, "range_override": beacon}
    return RangeTape(**base)


def varied_sensitive_count(tape, count):
    """Change private sensitivity prevalence while preserving public mechanics."""
    if not 0 <= count <= tape.good_map.shape[-1]:
        raise ValueError("Sensitive-cue count is out of range")
    sensitive = np.zeros_like(tape.sensitive)
    for row, seed in enumerate(tape.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6E559612]))
        sensitive[row, rng.permutation(tape.good_map.shape[-1])[:count]] = True
    good_map = tape.good_map.copy()
    good_map[:, 1] = good_map[:, 0] ^ sensitive
    good_id = good_map[np.arange(tape.n)[:, None], tape.ctx, tape.cue]
    return replace(tape, good_id=good_id, good_map=good_map, sensitive=sensitive)


def map_parity(tape, parity):
    """Sample uniformly on one half of the A-map truth tables."""
    if parity not in (0, 1):
        raise ValueError("A-map parity must be 0 or 1")
    maps = tape.good_map.copy()
    adjust = (maps[:, 0].sum(axis=1) % 2) != parity
    for row in np.flatnonzero(adjust):
        generator = np.random.default_rng(np.random.SeedSequence([tape.seeds[row], 0x5445BA33]))
        maps[row, 0, generator.integers(maps.shape[-1])] ^= 1
    maps[:, 1] = maps[:, 0] ^ tape.sensitive
    good = maps[np.arange(tape.n)[:, None], tape.ctx, tape.cue]
    return replace(tape, good_map=maps, good_id=good)


def randomly_change_one_context_rule(tape, probability):
    if not 0 <= probability <= 1:
        raise ValueError("Flip probability must be 0..1")
    changed = tape.good_id.copy()
    time = np.arange(tape.trials)
    for row, seed in enumerate(tape.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0xAF1873C2]))
        if rng.random() < probability:
            cue = rng.integers(tape.good_map.shape[-1])
            start = rng.integers(tape.trials // 3, 2 * tape.trials // 3 + 1)
            topic = (tape.cue[row] == cue) & (tape.ctx[row] == 1) & (time >= start)
            changed[row, topic] ^= 1
    return replace(tape, good_id=changed)
