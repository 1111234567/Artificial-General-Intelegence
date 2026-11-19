import numpy as np

from experiments.pulse_game import mixture_world as food, world
from experiments.pulse_game.context_local import Addresses, BoundNativeForager, LocalForager
from experiments.pulse_game.context_local_multi import make_multi


def test_addresses_follow_pulse_similarity_and_order_without_semantic_labels():
    state = Addresses(2, 3, 0.09)
    a = np.array([1., 0.], np.float32)
    close = np.array([np.cos(.21), np.sin(.21)], np.float32)
    rows = np.arange(2)
    state.register(rows, np.stack((a, close)))
    np.testing.assert_array_equal(state.current, 0)
    state.register(rows, np.stack((close, a)))
    np.testing.assert_array_equal(state.current, 1)
    state.register(rows, np.stack((a, close)))
    np.testing.assert_array_equal(state.current, 0)
    np.testing.assert_array_equal(state.counts[:, :2], [[2, 1], [2, 1]])


def test_an_unallocated_pair_of_addresses_does_not_create_a_third_slot():
    state = Addresses(1, 2, 0.09)
    for angle in (.1, .4, .7):
        pulse = np.asarray([[np.cos(angle), np.sin(angle)]], dtype=np.float32)
        state.register(np.array([0]), pulse)
    assert (state.counts > 0).sum() == 2
    assert state.overflows[0] == 1


def actor(*, local: bool) -> LocalForager:
    perception = {
        "matrix": np.zeros((4, 4), np.float32), "sensors": np.zeros((6, 4), np.float32),
        "ground": (np.eye(4, dtype=np.float32), np.ones(4, np.float32)), "capacity": 4,
        "transform": np.negative(np.eye(4, dtype=np.float32)), "scalars": [5.0, 0.5, 0.8, 0.95, 2.0],
    }
    route = {"slots": 3, "routing": "public", "updater": "rls", "ridge": 0.8, "eta": 0.3, "amp": 1.0, "retention": 0.7}
    result = LocalForager(perception, route, [11], motor=np.array([.39, 1.6, .78, .13, 1.08, 1.28, 0.]), local_retention=local)
    result.reset(1)
    return result


def test_new_address_inherits_realized_default_but_the_previous_slot_retains_its_rule():
    assert food.CTX.stop < 27
    local, global_decay = actor(local=True), actor(local=False)
    for path in (local, global_decay):
        first = np.zeros((1, 27), np.float32)
        first[:, food.CTX] = [1., 0.]
        path.act(first)
        path.g[0, 0] = 0.75
        next_context = first.copy()
        next_context[:, food.CTX] = [.6, .8]
        path.act(next_context)
        np.testing.assert_array_equal(path.g[0, :2], [0.75, 0.75])
        actual_receipt = np.zeros_like(first)
        actual_receipt[:, world.REWARD_NEG] = 1
        actual_receipt[:, world.CONTACT_ID.start] = 1
        path.act(actual_receipt)
    np.testing.assert_allclose(local.g[0, :2], [0.75, 0.75 * 0.7])
    np.testing.assert_allclose(global_decay.g[0, :2], [0.75 * 0.7, 0.75 * 0.7])


def test_banking_preserves_both_original_vector_and_covariance_on_return():
    sample = actor(local=True)
    banked = BoundNativeForager(sample.perception, {**sample.route, "slots": 2}, [11], motor=sample.motion)
    banked.reset(1)
    obs = np.zeros((1, 27), np.float32)
    obs[:, food.CTX] = [1., 0.]
    banked.act(obs)
    original = np.array([.25, .75], np.float32)
    banked.g[0] = original
    banked.cov[0] *= 0.4
    covariance = banked.cov[0].copy()
    obs[:, food.CTX] = [0., 1.]
    banked.act(obs)
    np.testing.assert_array_equal(banked.g[0], original)
    np.testing.assert_allclose(banked.cov[0], np.eye(2) / banked.route["ridge"])
    banked.g[0] = -original
    obs[:, food.CTX] = [1., 0.]
    banked.act(obs)
    np.testing.assert_array_equal(banked.g[0], original)
    np.testing.assert_array_equal(banked.cov[0], covariance)


def test_sensor_alias_manipulation_keeps_the_physical_and_food_tape():
    roots = [121, 122, 123]
    clear = make_multi(roots, 4)
    merged = make_multi(roots, 4, collision=True)
    assert clear.arena.probe.sum() == merged.arena.probe.sum() == 3 * 4 * 3
    assert (clear.novel < clear.learned).all()
    assert (clear.returned > clear.learned.max()).all()
    np.testing.assert_array_equal(clear.arena.menu.chemicals, merged.arena.menu.chemicals)
    np.testing.assert_array_equal(clear.arena.menu.realized, merged.arena.menu.realized)
    np.testing.assert_array_equal(clear.arena.menu.body.wind, merged.arena.menu.body.wind)
    np.testing.assert_array_equal(merged.arena.tags[:, 0], merged.arena.tags[:, -1])
    assert not np.any(np.all(clear.arena.tags[:, 0] == clear.arena.tags[:, -1], axis=1))
