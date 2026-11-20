from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_eval as evaluation, local_trace_graph as local, world


def test_worlds_are_root_independent_and_twin_draws_are_matched():
    roots = evaluation.evaluation_roots("pilot", 6)
    assert not set(roots) & set(evaluation.evaluation_roots("main", 10))
    batch = evaluation.make_arena(roots, "new_layout")
    reversed_batch = evaluation.make_arena(roots[::-1], "new_layout")
    lane_order = np.arange(len(batch.roots)).reshape(len(roots), -1)[::-1].ravel()
    for name in ("actual_ears", "roots", "maps", "mark_law", "motor_draw", "edit_draw", "shuffle_draw", "gap"):
        np.testing.assert_array_equal(getattr(batch, name)[lane_order], getattr(reversed_batch, name))
    assert np.unique(batch.maps[..., 0], axis=0).shape == (4, 2)
    for other in ("cba", "b_only", "prime_zero", "prime_donor", "flipped"):
        counter = evaluation.make_arena(roots, "new_layout", other)
        np.testing.assert_array_equal(batch.menu.body.wind, counter.menu.body.wind)
        np.testing.assert_array_equal(batch.menu.body.distances, counter.menu.body.distances)
        np.testing.assert_array_equal(batch.actual_ears[:, -3:], counter.actual_ears[:, -3:])
        np.testing.assert_array_equal(batch.good_side[:, -3:], counter.good_side[:, -3:])


def test_replication_worlds_use_an_independent_root_and_draw_domain():
    pilot = evaluation.evaluation_roots("pilot", 32)
    main = evaluation.evaluation_roots("main", 96)
    replication = evaluation.evaluation_roots("replication", 96)
    assert not set(replication) & (set(pilot) | set(main))
    domain, donor_domain = evaluation.world_domains("replication")
    assert len({domain, donor_domain, *evaluation.world_domains("main")}) == 4
    common_root = replication[:3]
    legacy = evaluation.make_arena(common_root, case="prime_donor")
    fresh = evaluation.make_arena(common_root, case="prime_donor", domain=domain, donor_domain=donor_domain)
    assert set(legacy.menu.body.seeds).isdisjoint(fresh.menu.body.seeds)
    assert not np.array_equal(legacy.actual_ears, fresh.actual_ears)
    assert not np.array_equal(legacy.motor_draw, fresh.motor_draw)
    assert not np.array_equal(legacy.donor, fresh.donor)


def test_public_reference_solves_conflicting_returns_with_silent_actual_actions():
    roots = evaluation.evaluation_roots("pilot", 10)
    for profile in evaluation.PROFILES:
        arena = evaluation.make_arena(roots, profile)
        lookup = evaluation._PublicPolicy(arena.motor_draw)
        measurement, observed = evaluation.rollout(arena, lookup, record=True)
        output = evaluation.payload(arena, measurement)
        assert np.all(output["success/both"] & output["success/mark"])
        assert (measurement.side >= 0).all()
        assert (measurement.tick < world.MOVEMENT_TICKS).all()
        assert not observed[:, :, :, :2].any()
        if profile == "independent_odors":
            directions = arena.actual_ears[:, 1] - arena.actual_ears[:, 1].mean(axis=1, keepdims=True)
            assert np.linalg.norm(directions, axis=2).min() > .5
            centers = arena.actual_ears[:, 1].mean(axis=1)
            assert np.min(np.linalg.norm(centers[:, None] - evaluation.CUES[None], axis=2), axis=1).min() > .5
        rows = np.arange(len(arena.roots))
        for label in ("A", "B", "mark"):
            assert not measurement.observed[rows, arena.scored[label]].any()

    arena = evaluation.make_arena(roots)
    result, _ = evaluation.rollout(arena, evaluation._PublicPolicy(arena.motor_draw, learn=False))
    output = evaluation.payload(arena, result)
    assert all(output[f"success/{label}"].mean() == .5 for label in ("A", "B", "mark"))
    flipped = evaluation.make_arena(roots, case="flipped")
    f_result, _ = evaluation.rollout(flipped, evaluation._PublicPolicy(flipped.motor_draw))
    flipped_out = evaluation.payload(flipped, f_result)
    assert not flipped_out["success/A"].any()
    assert flipped_out["success/B"].all()


def test_hidden_reversal_and_actual_receipt_doses():
    roots = evaluation.evaluation_roots("pilot", 7)

    def read(case):
        arena = evaluation.make_arena(roots, case=case)
        policy = evaluation._PublicPolicy(arena.motor_draw)
        result, _ = evaluation.rollout(arena, policy)
        return arena, result, evaluation.payload(arena, result)

    natural, n_result, no = read("natural")
    stable, s_result, control = read("stable")
    k = natural.reversal_at
    assert np.all(natural.contexts[:, k - 1:] == evaluation.A)
    np.testing.assert_array_equal(natural.actual_ears, stable.actual_ears)
    np.testing.assert_array_equal(n_result.side[:, :k + 1], s_result.side[:, :k + 1])
    assert control["success/dose0"].all()
    for dose in evaluation.DOSES:
        aligned, _, ao = read(f"aligned_{dose}")
        assert np.all(no[f"changed_feedback_before/dose{dose}"] == dose)
        assert np.all(ao[f"changed_feedback_before/dose{dose}"] == dose)
        assert np.all(no[f"success/dose{dose}"] == (dose > 0))
        assert np.all(ao[f"success/dose{dose}"] == (dose > 0))
        np.testing.assert_array_equal(aligned.actual_ears[:, -1], natural.actual_ears[:, -1])
        np.testing.assert_array_equal(aligned.menu.body.distances[:, -1], natural.menu.body.distances[:, -1])


def test_early_taste_treatment_is_confined_to_the_matched_a_prime():
    roots = evaluation.evaluation_roots("pilot", 10)
    intact = evaluation.make_arena(roots)
    donor = evaluation.make_arena(roots, case="prime_donor")
    zero = evaluation.make_arena(roots, case="prime_zero")
    a_prime = (intact.contexts == evaluation.A) & intact.exposed
    np.testing.assert_array_equal(intact.actual_ears, donor.actual_ears)
    np.testing.assert_array_equal(intact.actual_ears, zero.actual_ears)
    np.testing.assert_array_equal(intact.menu.realized, donor.menu.realized)
    np.testing.assert_array_equal(intact.menu.realized, zero.menu.realized)
    np.testing.assert_array_equal(donor.donor != 0, a_prime)
    np.testing.assert_array_equal(intact.exposed != zero.exposed, a_prime)
    same_root = donor.donor.reshape(len(roots), -1, donor.donor.shape[1])
    np.testing.assert_array_equal(same_root, np.broadcast_to(same_root[:, :1], same_root.shape))
    pairs = []
    for arena in (intact, donor, zero):
        result, _ = evaluation.rollout(arena, evaluation._PublicPolicy(arena.motor_draw, learn=False))
        pairs.append(evaluation.payload(arena, result))
    np.testing.assert_array_equal(pairs[0]["side"], pairs[1]["side"])
    np.testing.assert_array_equal(pairs[0]["side"], pairs[2]["side"])
    np.testing.assert_array_equal(pairs[0]["observed_reward"][~a_prime], pairs[1]["observed_reward"][~a_prime])
    np.testing.assert_array_equal(pairs[1]["observed_reward"][a_prime], donor.donor[a_prime])
    assert not pairs[2]["observed_reward"][a_prime].any()


def test_graph_interventions_act_physically_and_disable_writes():
    torch.set_num_threads(1)
    roots = evaluation.evaluation_roots("pilot", 2)
    arena = evaluation.make_arena(roots)
    graph, rule = local.spawn(71), local.Rule(71)
    runs = {}
    for mode in ("active", "disabled", "shuffle"):
        policy = evaluation._GraphPolicy(graph, rule, arena, intervention=mode)
        result, _ = evaluation.rollout(arena, policy)
        runs[mode] = evaluation.payload(arena, result)
        assert np.all(result.side >= 0)
    assert runs["active"]["writes"].any() and runs["shuffle"]["writes"].any()
    assert not runs["disabled"]["writes"].any()
    assert runs["active"]["graph_digest"] != runs["shuffle"]["graph_digest"]
    assert np.all(runs["shuffle"]["shuffle_audit"][arena.exposed, :2] >= 0)

    # World guards cannot write into a graph or display a taste when no side physically contacts.
    unreachable = replace(arena.menu.body, distances=np.full_like(arena.menu.body.distances, 99.))
    no_hit = replace(arena, menu=replace(arena.menu, body=unreachable))
    policy = evaluation._GraphPolicy(graph, rule, no_hit)
    result, _ = evaluation.rollout(no_hit, policy)
    measured = evaluation.payload(no_hit, result)
    assert np.all(measured["side"] == -1)
    assert not measured["observed_reward"].any() and not measured["writes"].any()


def test_joint_retrieval_forks_share_a_prefix_and_remain_independent():
    torch.set_num_threads(1)
    arena = evaluation.make_arena(evaluation.evaluation_roots("pilot", 8))
    graph, rule = local.spawn(71), local.Rule(71)

    def run(value):
        result, _ = evaluation.rollout(value, evaluation._GraphPolicy(graph, rule, value))
        return evaluation.payload(value, result)

    baseline = run(arena)
    changed = arena.motor_draw.copy()
    rows = np.arange(len(arena.roots))
    changed[rows, arena.scored["B"]] = 1 - (baseline["side/B"] == 0)
    alternate = run(replace(arena, motor_draw=changed))
    np.testing.assert_array_equal(baseline["side/A"], alternate["side/A"])
    np.testing.assert_array_equal(baseline["side/mark"], alternate["side/mark"])
    np.testing.assert_array_equal(baseline["side/B"], 1 - alternate["side/B"])
    np.testing.assert_array_equal(baseline["success/both"], baseline["success/A"] & baseline["success/B"])


def test_per_birth_joint_table_uses_matching_uniform_and_actual_conflict(tmp_path):
    arena = evaluation.make_arena(evaluation.evaluation_roots("pilot", 6))
    for name, birth, learn in (("uniform", 0, False), ("local_active", 811, True)):
        measurement, _ = evaluation.rollout(arena, evaluation._PublicPolicy(arena.motor_draw, learn=learn))
        evaluation._atomic_npz(tmp_path / "familiar" / "aba" / f"{name}-{birth}.npz", evaluation.payload(arena, measurement))
    evaluation._atomic_json(tmp_path / "protocol.json", {"specs": [{"name": "local_active", "birth": 811}]})
    row, = evaluation.joint_by_birth(tmp_path)
    assert row["birth"] == 811 and row["trials"] == 24
    assert row["scores"]["both_modalities"] == row["serial_joint"] == 1.
    assert row["uniform_joint"] < 1.
    assert "| test | 811 | 100.00% |" in evaluation.joint_table((("test", tmp_path),))
