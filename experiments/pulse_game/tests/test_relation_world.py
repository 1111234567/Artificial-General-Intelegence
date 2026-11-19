import numpy as np

from experiments.pulse_game import mixture_world, relation_world, world


def test_counterbalanced_zero_dose_has_same_public_future_and_opposite_targets():
    protocol = relation_world.make([6797, 536938], probes=3, doses=(0, 2))
    for stage in (*protocol.history.values(), protocol.teaching, *protocol.probes.values()):
        for first in (0, 4):
            twins = np.arange(first, first + 4)
            for field in (stage.menu.chemicals, stage.menu.body.distances, stage.menu.body.wind,
                          stage.appearance, stage.odor_tick, stage.context_tick):
                assert np.array_equal(field[twins], np.repeat(field[[first]], 4, axis=0))
    stage = protocol.probes["new"]
    assert np.array_equal(stage.menu.realized[::4, 1::3], stage.menu.realized[2::4, 1::3])
    assert np.array_equal(stage.truth[::4, 2::3], 1 - stage.truth[2::4, 2::3])
    assert np.array_equal(stage.truth[1::4, 2::3], 1 - stage.truth[3::4, 2::3])


def test_actual_tick_contacts_and_staged_replay_have_the_same_public_inputs():
    protocol = relation_world.make([98531, 56894], doses=(0, 2, 4))
    stage = protocol.teaching
    whole = relation_world._play(None, stage, record=True)
    first = relation_world._play(None, stage.slice(0, 3), record=True)
    last = relation_world._play(None, stage.slice(3, 6), record=True)
    # The geometry-only reference chooses alternating teaching sides, so
    # replay must account for its reset at the segment boundary.
    for name in ("obs", "observed_reward"):
        assert np.array_equal(getattr(whole, name)[:, :3], getattr(first, name))
    for field in (mixture_world.LEFT, mixture_world.RIGHT, mixture_world.CTX):
        assert np.array_equal(whole.obs[:, 3:, :, field], last.obs[:, :, :, field])
    for row in range(len(protocol.seeds)):
        for trial in range(stage.menu.body.trials):
            tick = whole.contact_tick[row, trial] + 1
            assert whole.obs[row, trial, tick, world.CONTACT_ID].sum() == 1
            shown = whole.observed_reward[row, trial]
            if stage.taste[row, trial]:
                assert shown[tick] == whole.reward[row, trial]
                assert np.count_nonzero(shown) == 1
            else:
                assert not shown.any()


def test_public_and_privileged_physical_oracles_are_distinct():
    protocol = relation_world.make(range(42327, 42332), probes=2, doses=(0, 2, 4))
    public = relation_world.public_oracle(protocol)
    physical = relation_world.physical_oracle(protocol)
    for history in relation_world.HISTORIES:
        for dose in protocol.doses:
            measured = public[history]["intact"][str(dose)]
            control = physical[history]["intact"][str(dose)]
            assert measured["new"].mean() == (.5 if dose == 0 else 1.)
            assert np.all(control["new"] == 1.)
            assert np.all(measured["old"] == 1.)
            assert np.all(measured["b_receipts"] == dose)
            assert np.all(measured["b_true_receipts"] == dose)
            assert np.all(measured["new_anchor_receipts"] == 2)
            assert np.all(measured["new_silent_receipts"] == 0)
            assert np.all(measured["new_contact"] == 1.)
            assert np.all(measured["old_contact"] == 1.)


def test_quarter_turn_is_identifiable_from_public_tastes_in_both_histories():
    # Diagnostic roots explicitly outside both development and scored ranges.
    roots = range(2**60, 2**60 + 32)
    protocol = relation_world.make(roots, partition="heldout", probes=3, doses=(0, 2, 4))
    public = relation_world.public_oracle(protocol)
    physical = relation_world.physical_oracle(protocol)
    related = public["related"]["intact"]
    unrelated = public["unrelated"]["intact"]
    assert np.all(related["2"]["new"] == 1.)
    assert .5 < unrelated["2"]["new"].mean() < 1.
    for condition in relation_world.HISTORIES:
        summaries = public[condition]["intact"]
        assert summaries["0"]["new_first"].mean() == .5
        assert np.isclose(summaries["0"]["new"].mean(), .5)
        assert np.all(summaries["4"]["new"] == 1.)
        for dose in protocol.doses:
            assert np.all(physical[condition]["intact"][str(dose)]["joint"] == 1.)
            assert np.all(summaries[str(dose)]["old"] == 1.)


def test_timeouts_remain_wrong_even_for_silent_probes():
    protocol = relation_world.make([18624], probes=2, doses=(0, 2))
    result = relation_world.evaluate(protocol, lambda _: world.NoopPolicy())
    for history in relation_world.HISTORIES:
        for values in result[history]["intact"].values():
            for key in ("new", "old", "joint", "new_anchor_receipts", "history_receipts", "b_receipts"):
                assert not values[key].any()
            for key in ("new_timeout", "old_timeout"):
                assert np.all(values[key])


def test_shuffle_preserves_sign_marginals_at_each_teaching_slot():
    protocol = relation_world.make([662, 663, 664], doses=(0, 2, 4, 8))
    shown = np.arange(len(protocol.seeds))[:, None] * 10 + np.arange(8)
    shuffled = relation_world._shuffled(protocol, shown)
    assert np.array_equal(np.sort(shown, axis=0), np.sort(shuffled, axis=0))
    assert not np.any(shuffled // 40 == protocol.row_root[:, None])
