from dataclasses import replace

import numpy as np

from experiments.pulse_game.twostep_agent import DirectedRoutePolicy, OraclePolicy
from experiments.pulse_game.twostep_distinct import fruit_score, make_distinct_tape
from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_world import (
    AVAILABLE, ENTRANCE_RECEIPT, ENTRANCE_STAGE, EPISODE_TICKS, ROOM_ID, ROOM_STAGE, STAGE_START, VISIT,
    make_tape, make_visit_replay, rollout, set_reversal,
)
from experiments.pulse_game.world import ACTIVE, CONTACT_ID, REWARD_NEG, REWARD_POS, TICKS, NoopPolicy


def test_contacts_gate_actual_room_and_terminal_reward():
    tape = make_tape(range(30, 46), acquisition=7, visits=4, rare_probability=0.2)
    result = rollout(OraclePolicy(tape), tape, record=True)
    rows = np.arange(tape.n)[:, None]
    acquisition = slice(0, tape.acquisition)
    expected_room = tape.common_room[rows, result.entry_id[:, acquisition]] ^ tape.rare[:, acquisition]
    np.testing.assert_array_equal(result.room_id[:, acquisition], expected_room)
    assert tape.rare[:, acquisition].any()
    np.testing.assert_array_equal(result.obs[..., :TICKS, REWARD_POS:REWARD_NEG + 1], 0)
    np.testing.assert_array_equal(result.obs[..., :TICKS, ROOM_ID], 0)
    np.testing.assert_array_equal(result.obs[:, acquisition, TICKS, ROOM_ID].argmax(axis=-1), expected_room)
    np.testing.assert_array_equal(result.obs[:, acquisition, TICKS, STAGE_START], 1)
    np.testing.assert_array_equal(result.obs[:, acquisition, 4, ENTRANCE_RECEIPT].argmax(axis=-1), result.entry_id[:, acquisition])
    np.testing.assert_array_equal(result.obs[..., TICKS:, ENTRANCE_RECEIPT], 0)
    np.testing.assert_array_equal(result.obs[:, acquisition, TICKS + 4, CONTACT_ID].argmax(axis=-1), result.fruit_id[:, acquisition])
    np.testing.assert_array_equal(
        result.shown[:, acquisition, TICKS + 4],
        np.where(expected_room == tape.initial_good_room[:, None], 1, -1),
    )
    np.testing.assert_array_equal(result.shown[:, tape.probe], 0)
    assert (result.fruit_id[:, tape.probe] >= 0).all()


def test_visit_physically_contacts_one_fruit_without_an_entrance():
    tape = make_tape([45, 46, 47], acquisition=3, visits=8, distance_jitter=0.22, wind_scale=0.12)
    result = rollout(OraclePolicy(tape), tape, record=True)
    visits = slice(tape.acquisition, tape.probe)
    np.testing.assert_array_equal(result.entry_id[:, visits], -1)
    np.testing.assert_array_equal(result.entry_tick[:, visits], -1)
    np.testing.assert_array_equal(result.obs[:, visits, :TICKS, ACTIVE], 0)
    np.testing.assert_array_equal(result.obs[:, visits, :TICKS, ENTRANCE_STAGE], 0)
    np.testing.assert_array_equal(result.obs[:, visits, TICKS:, ROOM_STAGE], 1)
    np.testing.assert_array_equal(result.obs[:, visits, :, VISIT], 1)
    np.testing.assert_array_equal(result.obs[:, visits, TICKS:, AVAILABLE].sum(axis=-1), 1)
    np.testing.assert_array_equal(result.fruit_id[:, visits], tape.offered_fruit[:, visits])
    np.testing.assert_array_equal((result.reward[:, visits] > 0).sum(axis=1), tape.visits // 4)
    assert result.visits_confirmed.all()

    inert = rollout(NoopPolicy(), tape)
    np.testing.assert_array_equal(inert.entry_id, -1)
    np.testing.assert_array_equal(inert.fruit_id, -1)
    np.testing.assert_array_equal(inert.shown, 0)
    assert not inert.visits_confirmed.any()


def test_revaluation_twins_match_prefix_and_body_visits():
    reverse = make_tape(range(30, 46), acquisition=32, visits=8, distance_jitter=0.2, wind_scale=0.1)
    same = set_reversal(reverse, False)
    flip = rollout(DirectedRoutePolicy(900), reverse, record=True)
    stay = rollout(DirectedRoutePolicy(900), same, record=True)
    acquire = slice(0, reverse.acquisition)
    visit = slice(reverse.acquisition, reverse.probe)
    np.testing.assert_array_equal(flip.obs[:, acquire], stay.obs[:, acquire])
    np.testing.assert_array_equal(flip.pulses[:, visit], stay.pulses[:, visit])
    visible = [index for index in range(flip.obs.shape[-1]) if index not in (REWARD_POS, REWARD_NEG)]
    np.testing.assert_array_equal(flip.obs[:, visit][..., visible], stay.obs[:, visit][..., visible])
    np.testing.assert_array_equal(flip.obs[:, reverse.probe, 0], stay.obs[:, reverse.probe, 0])
    assert np.all(flip.probe_entry != stay.probe_entry)
    assert flip.probe_accuracy == stay.probe_accuracy == 1
    frozen = rollout(DirectedRoutePolicy(900, cached_probe=True), reverse)
    assert frozen.probe_accuracy < 0.2


def test_room_blind_public_trace_is_invariant_under_balanced_hidden_permutation():
    original = make_tape(range(21, 28), acquisition=10, visits=8, reversal=False)
    forced_room = original.forced_room.copy()
    forced_room[:, original.acquisition:original.probe] ^= 1
    counterfactual = replace(set_reversal(original, True), forced_room=forced_room)
    a = rollout(DirectedRoutePolicy(86), original, hide_rooms="visits", record=True)
    b = rollout(DirectedRoutePolicy(86), counterfactual, hide_rooms="visits", record=True)
    np.testing.assert_array_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.pulses, b.pulses)
    np.testing.assert_array_equal(a.probe_entry, b.probe_entry)


def test_contact_locked_replay_changes_only_visit_sign_for_matched_motor_choices():
    tape = make_tape(range(150, 161), acquisition=2, visits=8)
    donor = rollout(OraclePolicy(tape), tape)
    replay = make_visit_replay(donor, seed=36)
    result = rollout(OraclePolicy(tape), tape, visit_replay=replay)
    visits = slice(tape.acquisition, tape.probe)
    np.testing.assert_array_equal(donor.reward, result.reward)
    np.testing.assert_array_equal(donor.fruit_tick, result.fruit_tick)
    np.testing.assert_array_equal(result.shown[:, visits].sum(axis=-1), replay.reward)
    np.testing.assert_array_equal(np.sort(replay.reward, axis=0), np.sort(donor.reward[:, visits], axis=0))
    np.testing.assert_array_equal(result.shown[:, visits, :TICKS], 0)
    assert result.shown.shape[-1] == EPISODE_TICKS


def test_distinct_fruit_requires_arrival_dependent_second_action():
    tape = make_distinct_tape(range(800, 856), rare_probability=0.2)
    _, stay, flipped = paired(lambda: DirectedRoutePolicy(78, mean_visits=True), tape)
    for reversed, result in ((False, stay), (True, flipped)):
        score = fruit_score(tape, result, reversed=reversed)
        assert score["poor_arrivals"] > 0
        assert score["optimal_fruit_in_poor_room"] == 1
        assert score["optimal_fruit_in_rich_room"] == 1
        np.testing.assert_array_equal(result.shown[:, tape.probe], 0)
    _, _, shortcut = paired(lambda: DirectedRoutePolicy(78, mean_visits=True, global_probe_fruit=True), tape)
    score = fruit_score(tape, shortcut, reversed=True)
    np.testing.assert_array_equal(shortcut.probe_entry, flipped.probe_entry)
    assert score["optimal_fruit_in_poor_room"] == 0
    assert score["optimal_fruit_in_rich_room"] == 1


def test_distinct_fruit_has_same_room_blind_trace_under_hidden_role_swap():
    tape = make_distinct_tape(range(801, 809), acquisition=8, reversal=False)
    forced_room = tape.forced_room.copy()
    forced_room[:, tape.acquisition:tape.probe] ^= 1
    flipped = replace(set_reversal(tape, True), forced_room=forced_room)
    a = rollout(DirectedRoutePolicy(97, mean_visits=True), tape, hide_rooms="visits", record=True)
    b = rollout(DirectedRoutePolicy(97, mean_visits=True), flipped, hide_rooms="visits", record=True)
    np.testing.assert_array_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.pulses, b.pulses)
