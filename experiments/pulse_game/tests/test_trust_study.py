"""Physical and no-feedback invariants for the contact-conditioned gate."""

import numpy as np
import pytest

from experiments.pulse_game import relation_world, trust_replay
from experiments.pulse_game.trust_gate import Gate
from experiments.pulse_game.trust_study import BIRTHS, REG, candidate_pool, factory, make, resources, root_ids, verify_closure


@pytest.fixture(scope="module")
def dev():
    protocol = make(root_ids(REG["roots"]["training"][0])[:8], "develop")
    front, motion, rule = resources(BIRTHS[0])
    prepared = trust_replay.prepare(protocol, front, motion, rule)
    return protocol, front, motion, rule, prepared


def test_registered_root_namespaces_are_disjoint():
    specs = [*REG["roots"]["training"], REG["roots"]["validation"], REG["roots"]["holdout"]]
    intervals = sorted((s["lower"], s["upper"]) for s in specs)
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    assert not any(lo < 470_000_000 and hi > 100_000 for lo, hi in intervals)


@pytest.mark.parametrize("mode", ["always", "local", "learned"])
def test_replay_equals_original_closure(dev, mode):
    protocol, front, motion, rule, prepared = dev
    gate = next(candidate_pool()) if mode == "learned" else Gate(mode)
    cached = trust_replay.evaluate(prepared, rule, motion, gate, record=True)
    checks = verify_closure(protocol, front, motion, rule, gate, cached, replay_base=mode != "learned")
    assert checks["probes"] == len(protocol.seeds) * 2 * 2 * len(protocol.doses) * REG["development_probes"]
    if mode == "learned":
        assert checks["writes"] > 0


def test_swapping_values_gives_crossed_gate_with_same_receipts(dev):
    protocol, front, motion, rule, prepared = dev
    gate = next(candidate_pool())
    natural = trust_replay.evaluate(prepared, rule, motion, gate, record=True)
    donor = {a: natural[b][1]["gate"] for a, b in (("related", "unrelated"), ("unrelated", "related"))}
    crossed = trust_replay.evaluate(prepared, rule, motion, gate, overrides=donor, record=True)
    counts = verify_closure(protocol, front, motion, rule, gate, crossed, schedules=donor)
    assert counts["writes"] > 0
    for history in relation_world.HISTORIES:
        masks = np.isfinite(natural[history][1]["gate"]) & np.isfinite(donor[history])
        np.testing.assert_array_equal(crossed[history][1]["gate"][masks], donor[history][masks])


def test_no_gate_or_matrix_write_from_silence(dev):
    protocol, *_ = dev
    actor = factory(protocol.seeds, BIRTHS[0], "explicit")
    actor.reset(len(protocol.seeds))
    relation_world._play(actor, protocol.history["related"])
    before = actor.snapshot_matrix()
    trace_before = set(actor.gate_trace)
    stage = protocol.teaching.slice(0, 3)
    from dataclasses import replace
    stage = replace(stage, taste=np.zeros_like(stage.taste))
    result = relation_world._play(actor, stage)
    assert (result.contact_side >= 0).any()
    assert not result.observed_reward.any()
    assert set(actor.gate_trace) == trace_before
    np.testing.assert_array_equal(actor.matrix, before)


def test_explicit_and_learned_gate_are_bounded_and_use_evidence():
    evidence = np.array([[.9, 1.3, .5, .5, .5], [.2, .2, .5, .5, .5]], dtype=np.float32)
    for gate in (Gate("explicit"), next(candidate_pool())):
        first, second = gate.amount(evidence, evidence)
        assert 0. <= second < first <= 1.
