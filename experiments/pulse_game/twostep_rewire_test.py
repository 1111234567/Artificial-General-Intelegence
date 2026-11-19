import numpy as np
import torch

from experiments.pulse_game.twostep_agent import DirectedRoutePolicy
from experiments.pulse_game.twostep_latent import LatentPolicy, LatentTwoStep
from experiments.pulse_game.twostep_rewire import AdaptiveRoutePolicy, RewireWorld, condition, evaluate, make_rewire_tape, run
from experiments.pulse_game.twostep_untyped import UntypedLatent, UntypedWorld
from experiments.pulse_game.twostep_world import ENTRANCE_RECEIPT, ENTRANCE_STAGE, ROOM_STAGE, STAGE_START, VISIT
from experiments.pulse_game.world import CONTACT_ID


def test_landing_uses_each_worlds_hidden_switch_tick_and_actual_entry_contact():
    tape = make_rewire_tape(range(72, 84), rare_probability=0.2)
    result = run(lambda: AdaptiveRoutePolicy(95), tape)
    rows, episode = np.nonzero(result.entry_id >= 0)
    common = tape.common_room[rows, result.entry_id[rows, episode]]
    expected = common ^ tape.rare[rows, episode] ^ (episode >= tape.switch_at[rows])
    np.testing.assert_array_equal(result.room_id[rows, episode], expected)
    assert len(set(tape.switch_at)) > 1
    assert (result.entry_id[:, tape.ordinary:tape.acquisition] >= 0).all()
    np.testing.assert_array_equal(result.fruit_id[:, tape.ordinary:tape.acquisition], -1)
    np.testing.assert_array_equal(result.shown[:, tape.ordinary:tape.acquisition], 0)


def test_old_new_transition_and_value_information_compose_with_no_probe_feedback():
    tape = make_rewire_tape(range(41, 48))
    adaptive = evaluate(lambda: AdaptiveRoutePolicy(8), tape)
    assert adaptive["balanced_accuracy"] == adaptive["all_four_correct"] == adaptive["xor_signature"] == 1
    old = evaluate(lambda: DirectedRoutePolicy(8), tape)
    assert old["balanced_accuracy"] == 0.5
    assert old["all_four_correct"] == 0
    assert all(r["no_survey_or_probe_payoff_display"] for r in adaptive["cells"].values())


def test_hiding_unrewarded_destinations_keeps_route_flip_unobserved_before_probe():
    tape = make_rewire_tape(range(28, 35), ordinary=16, survey=12, rare_probability=0.2)
    stable = run(lambda: AdaptiveRoutePolicy(84), condition(tape, route=False, value=False), hide_survey_room=True, record=True)
    changed = run(lambda: AdaptiveRoutePolicy(84), condition(tape, route=True, value=False), hide_survey_room=True, record=True)
    np.testing.assert_array_equal(stable.obs[:, :tape.probe], changed.obs[:, :tape.probe])
    np.testing.assert_array_equal(stable.pulses[:, :tape.probe], changed.pulses[:, :tape.probe])
    np.testing.assert_array_equal(stable.probe_entry, changed.probe_entry)


def test_erasing_tags_does_not_change_physics_or_reveal_rewards():
    tape = make_rewire_tape(range(92, 97), ordinary=4, survey=8)
    direct = RewireWorld(tape)
    erased = RewireWorld(tape, erase_stage_tags=True)
    a, b = direct.observe(), erased.observe()
    altered = (ENTRANCE_STAGE, ROOM_STAGE, STAGE_START, VISIT)
    keep = [i for i in range(a.shape[-1]) if i not in altered]
    while a is not None:
        np.testing.assert_array_equal(a[:, keep], b[:, keep])
        np.testing.assert_array_equal(b[:, altered], 0)
        pulses = np.broadcast_to([1, 0], (tape.n, 2))
        a, b = direct.step(pulses), erased.step(pulses)
    np.testing.assert_array_equal(erased.result().shown[:, tape.probe], 0)


def test_both_real_tactile_contacts_share_one_public_channel():
    tape = make_rewire_tape(range(200, 205), ordinary=2, survey=8, visits=4)
    typed = RewireWorld(tape, erase_stage_tags=True)
    one_channel = UntypedWorld(tape)
    a, b = typed.observe(), one_channel.observe()
    count_entry = count_fruit = 0
    excluded = set(range(CONTACT_ID.start, CONTACT_ID.stop)) | set(range(ENTRANCE_RECEIPT.start, ENTRANCE_RECEIPT.stop))
    unchanged = [column for column in range(a.shape[1]) if column not in excluded]
    while a is not None:
        np.testing.assert_array_equal(a[:, unchanged], b[:, unchanged])
        np.testing.assert_array_equal(a[:, CONTACT_ID] + a[:, ENTRANCE_RECEIPT], b[:, CONTACT_ID])
        np.testing.assert_array_equal(b[:, ENTRANCE_RECEIPT], 0)
        count_entry += int(a[:, ENTRANCE_RECEIPT].sum())
        count_fruit += int(a[:, CONTACT_ID].sum())
        run_left = np.broadcast_to([1, 0], (tape.n, 2))
        a, b = typed.step(run_left), one_channel.step(run_left)
    assert count_entry > 0
    assert count_fruit > 0
    np.testing.assert_array_equal(typed.result().reward, one_channel.result().reward)


def test_contact_role_has_a_learning_signal_through_the_later_recall():
    model = UntypedLatent()
    events = torch.tensor([[[1., 0., 0., 0., 0., 0.], [0., 1., 0., 1., 1., 0.]]])
    room = torch.tensor([[0., 1.]])
    address = model.classify(events, room)
    prior = model.memory.initial(1)
    bank = model.memory.prepare()
    learned = model.memory.update(prior, bank, *address.unbind(1))
    later = model.memory.view(learned, bank)
    later.square().sum().backward()
    assert model.contact_role[-1].weight.grad.norm() > 0


def test_actor_stage_mask_keeps_every_nonstage_public_input():
    actor = LatentPolicy(LatentTwoStep(), seed=441, erase_stage_tags=True)
    tape = make_rewire_tape([2, 3], ordinary=2, survey=8)
    world = RewireWorld(tape)
    actor.reset(tape.n)
    obs = world.observe()
    raw = actor.public_frame(obs)
    dropped = (ENTRANCE_STAGE, ROOM_STAGE, STAGE_START, VISIT)
    kept = [column for column in range(obs.shape[1]) if column not in dropped]
    np.testing.assert_array_equal(raw[:, dropped], 0)
    np.testing.assert_array_equal(raw[:, kept], obs[:, kept])
    assert obs[:, ENTRANCE_STAGE].all()
