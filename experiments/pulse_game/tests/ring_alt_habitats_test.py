"""Schedule and continuity checks for the long-life habitat experiment."""

import numpy as np

from experiments.pulse_game.ring_alt_chemical import ChemSpec
from experiments.pulse_game.ring_alt_habitats import (
    BETWEEN, WITHIN, Exposure, catalog, gather, hidden_preferences, new_actor, palette_collection, permutations,
)
from experiments.pulse_game.ring_world import rollout


def test_blocked_and_spaced_use_identical_trials_per_habitat():
    seeds = (821, 822, 823)
    hidden = hidden_preferences(seeds, "evaluation")
    panels = catalog(seeds, 16, 58192, hidden)
    order = permutations(seeds, 4)
    a, b = np.repeat(order, 8, axis=1), np.tile(order, (1, 8))
    tapes = gather(seeds, hidden, panels, a, offset=8), gather(seeds, hidden, panels, b, offset=8)
    for life in range(len(seeds)):
        for habitat in range(4):
            for field in ("ctx", "cue", "positions", "layout", "goal"):
                x, y = [getattr(t, field)[life] for t in tapes]
                np.testing.assert_array_equal(x[a[life] == habitat], y[b[life] == habitat])


def test_forked_review_preserves_the_continuous_brain():
    seeds = (8511, 8512)
    palettes, audit = palette_collection(seeds)
    for category in audit:
        assert np.min(category["within"]) >= WITHIN - 1e-6
        assert np.min(category["cross"]) >= BETWEEN - 1e-6
    prefs = hidden_preferences(seeds, "development")
    positions = catalog(seeds, 8, 12201, prefs)
    order = permutations(seeds, 2)
    start, review = np.repeat(order, 4, axis=1), np.tile(order, (1, 4))
    whole = np.column_stack((start, review))

    def brain():
        return new_actor(ChemSpec(cue_bank="growing", capacity=8, cue_capacity=8), seeds, 101, 32, "age")

    a, b = brain(), brain()
    together = rollout(Exposure(a, palettes, whole), gather(seeds, prefs, positions, whole, offset=0))
    first = rollout(Exposure(b, palettes, start), gather(seeds, prefs, positions, start, offset=0))
    last = rollout(Exposure(b, palettes, review, resume=True), gather(seeds, prefs, positions, review, offset=4))
    np.testing.assert_array_equal(together.first, np.column_stack((first.first, last.first)))
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.fruit.prototype, b.fruit.prototype)
