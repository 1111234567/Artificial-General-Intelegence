from copy import deepcopy

import numpy as np
import pytest

from experiments.pulse_game import habitat_world, lifelong_agent, mixture_world as food, relation_learner, world


def _observation(seeds, *, start=False, odor=None, context=None, taste=0, contacted=False):
    obs = np.zeros((len(seeds), world.OBS_SIZE), dtype=np.float32)
    obs[:, food.LEFT.start:food.RIGHT.stop] = .5
    obs[:, world.TRIAL_START] = start
    obs[:, world.ACTIVE] = not contacted
    if context is not None:
        obs[:, food.CTX] = context
    if odor is not None:
        obs[:, food.LEFT] = obs[:, food.RIGHT] = (np.asarray(odor) + 1.) / 2.
    obs[:, world.REWARD_POS] = taste > 0
    obs[:, world.REWARD_NEG] = taste < 0
    obs[:, world.CONTACT_ID] = [bool(contacted), False]
    return obs


def _contact(actor, odor, taste=1, *, contacted=True, context=(1., 0.)):
    seeds = actor.seeds
    actor.act(_observation(seeds, start=True, context=context))
    actor.act(_observation(seeds, odor=odor))
    actor.act(_observation(seeds))
    actor.act(_observation(seeds, taste=taste, contacted=contacted))


@pytest.mark.parametrize("mode", ["adaptive", "frozen"])
def test_only_physical_taste_updates_the_matrix(mode):
    actor = relation_learner.birth((10, 11), mode=mode)
    actor.reset(2)
    _contact(actor, [1, 0, 0, 0], taste=1, contacted=False)
    _contact(actor, [0, 1, 0, 0], taste=0)
    assert not actor.receipts.any()
    assert not actor.matrix.any()

    _contact(actor, [1, 0, 0, 0], taste=1)
    assert not actor.matrix.any()
    _contact(actor, [0, 1, 0, 0], taste=-1)
    assert np.array_equal(actor.receipts, [2, 2])
    assert np.array_equal(actor.last_taste[:, 0], [-1, -1])
    assert bool(actor.matrix.any()) == (mode == "adaptive")


def test_matrix_restore_leaves_ongoing_life_unchanged():
    actor = relation_learner.birth((10, 11))
    actor.reset(2)
    _contact(actor, [1, 0, 0, 0], taste=1)
    _contact(actor, [0, 1, 0, 0], taste=-1)
    saved = actor.snapshot_matrix()
    _contact(actor, [0, 0, 0, 1], taste=-1)
    state = deepcopy(actor)
    assert not np.array_equal(actor.matrix, saved)

    actor.restore_matrix(saved)
    np.testing.assert_array_equal(actor.matrix, saved)
    saved.fill(0)
    assert actor.matrix.any()
    for name, expected in vars(state).items():
        if isinstance(expected, np.ndarray) and name != "matrix":
            np.testing.assert_array_equal(getattr(actor, name), expected)
    for name in ("counts", "current", "prototypes", "overflows"):
        np.testing.assert_array_equal(getattr(actor.places, name), getattr(state.places, name))
    assert (actor.it, actor.round) == (state.it, state.round)
    assert [rng.bit_generator.state for rng in actor.rng] == [rng.bit_generator.state for rng in state.rng]
    with pytest.raises(ValueError, match="snapshot"):
        actor.restore_matrix(saved[:1])


def test_local_selective_control_is_the_original_actor():
    seeds = (10, 11)
    actor = relation_learner.birth(seeds, mode="local-selective")
    reference = lifelong_agent.new_lives(seeds, address="local-selective")
    assert type(actor) is type(reference)
    arena = habitat_world.make(seeds, pairs=(2, 2), visits=1)
    original = habitat_world.rollout(reference, arena, silent_probes=True, record=True)
    control = habitat_world.rollout(actor, arena, silent_probes=True, record=True)
    np.testing.assert_array_equal(original.pulses, control.pulses)
    np.testing.assert_array_equal(original.contact_side, control.contact_side)
    np.testing.assert_array_equal(original.observed_reward, control.observed_reward)


def test_continuous_sensory_graph_matches_original_for_same_public_input():
    seeds = (10, 11)
    arena = habitat_world.make(seeds, pairs=(2, 2), visits=1)
    sim = habitat_world.Body(arena, silent_probes=True)
    actor = relation_learner.birth(seeds)
    reference = lifelong_agent.new_lives(seeds, address="local-selective")
    actor.reset(len(seeds))
    reference.reset(len(seeds))
    obs = sim.observe()
    while obs is not None:
        reference.act(obs)
        pulses = actor.act(obs)
        np.testing.assert_array_equal(actor.h, reference.h)
        obs = sim.step(pulses)
    actual = sim.result()
    np.testing.assert_array_equal(actor.receipts, (actual.observed_reward != 0).sum(axis=(1, 2)))


def test_outer_parameters_are_isolated_from_life_and_caller():
    weights = relation_learner.Rule.default_vector()
    rule = relation_learner.Rule.from_vector(weights)
    assert weights.shape == relation_learner.Rule.mutation_scales().shape == (32,)
    weights[:] = 0
    original = rule.vector()
    actor = relation_learner.birth((10, 11), rule)
    actor.reset(2)
    _contact(actor, [1, 0, 0, 0], taste=1)
    _contact(actor, [0, 1, 0, 0], taste=-1)
    np.testing.assert_array_equal(rule.vector(), original)
    assert not np.shares_memory(rule.vector(), rule.weights)


def test_local_only_writes_ignore_other_contexts():
    adaptive = relation_learner.birth((10, 11))
    local = relation_learner.birth((10, 11), adaptive.rule, mode="local-only")
    for actor in (adaptive, local):
        actor.reset(2)
        _contact(actor, [1, 0, 0, 0])
        _contact(actor, [0, 1, 0, 0], taste=-1)
    np.testing.assert_array_equal(adaptive.matrix, local.matrix)

    empty_source = deepcopy(local)
    empty_source.restore_matrix(np.zeros_like(local.matrix))
    for actor in (adaptive, local, empty_source):
        _contact(actor, [0, 0, 1, 0], context=(0., 1.))
        _contact(actor, [1, 0, 0, 0], context=(0., 1.))
    assert local.matrix[:, 1].any()
    np.testing.assert_array_equal(local.matrix[:, 1], empty_source.matrix[:, 1])
    assert not np.allclose(adaptive.matrix[:, 1], local.matrix[:, 1])
