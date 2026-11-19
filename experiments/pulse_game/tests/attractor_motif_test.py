"""Parity and causal-order checks for delay interventions."""

import numpy as np

from experiments.pulse_game import attractor_search as frozen
from experiments.pulse_game.attractor_motif import Encoder, WINDOWS, curriculum, probes, readout_on_prefix, responses


def test_vectorized_reference_keeps_encoder_and_rls_semantics():
    genes = np.full(len(frozen.RANGES), .5)
    tape = curriculum([81071, 81072], trials=3)
    enc = Encoder(genes, 43, exact=True)
    _, observations, rate = enc.run(tape)
    z, root = frozen.cpu_features(genes, 43, tape)
    np.testing.assert_array_equal(observations, z)
    assert rate == root
    _, chosen = readout_on_prefix(genes, observations, tape)
    np.testing.assert_array_equal(chosen, frozen.cpu_answers(genes, observations, tape))


def test_query_is_unchanged_by_post_window_and_same_source_replay():
    genes = np.full(len(frozen.RANGES), .5)
    enc = Encoder(genes, 43)
    tape = curriculum([81071, 81072], trials=3)
    start, _, _ = enc.run(tape)
    branch = probes(start, tape.seeds)
    plate = np.ones_like(enc.weight)
    same = responses(enc, branch, collect=True)
    post = responses(enc, branch, mask=plate, window=WINDOWS["post"])
    identity = responses(enc, branch, mask=plate, window=WINDOWS["early"],
                         operation="replay", baseline=same["state"]["synaptic"], swap=0)
    np.testing.assert_array_equal(post["address"], same["address"])
    np.testing.assert_array_equal(identity["address"], same["address"])
