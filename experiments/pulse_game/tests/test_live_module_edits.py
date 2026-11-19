"""Exercise causality, SCC cut rejection, and independent growth timing."""

import numpy as np

from experiments.pulse_game import live_module_edits as lab
from experiments.pulse_game import live_module_edits_study as study, world


def population(schedule):
    ring = np.zeros((1, lab.WIDTH, lab.WIDTH), dtype=bool)
    nodes = np.arange(lab.WIDTH)
    ring[0, nodes, (nodes - 1) % lab.WIDTH] = True
    ring[0, nodes, (nodes - 2) % lab.WIDTH] = True
    rng = np.random.default_rng(8192)
    shapes = dict(drive=(1, 5, 2 * lab.WIDTH), bias=(1, 2 * lab.WIDTH),
                  gain=(1, lab.WIDTH), readout=(1, lab.WIDTH), offset=(1,))
    params = {key: rng.normal(0, .3, shape).astype(np.float32) for key, shape in shapes.items()}
    return lab.Population(params, ring, schedule, [81729])


def test_probe_only_observed_contact_receipt_and_preserve_state():
    brain = population(lab.Schedule(1, 5, 0.))
    brain.begin_request()
    active = np.ones(1, dtype=np.float32)
    cue = np.ones((1, 5), dtype=np.float32)
    for _ in range(8):
        brain.act(cue, active)
    before = brain.state.copy()
    assert np.all(brain.finish_request(np.array([-1]), np.array([-1])) < 0)
    assert np.all(brain.finish_request(np.array([1]), np.array([0])) < 0)
    change = brain.finish_request(np.array([1]), np.array([-1]))
    assert change[0, 0] >= 0 and change[0, 1] == -1
    assert brain.mask.sum() == 65 and brain.receive_count[0] == 1
    np.testing.assert_array_equal(brain.state, before)
    brain.begin_request()
    np.testing.assert_array_equal(brain.state, before)


def test_scc_guard_rejects_globally_vital_arc_despite_nonzero_degree():
    brain = population(lab.Schedule(1, 1, 0.))
    mask = brain.mask[0]
    mask.fill(False)
    half = lab.WIDTH // 2
    for start in (0, half):
        local = np.arange(half)
        for delay in (1, 2):
            mask[start + local, start + (local - delay) % half] = True
    mask[half, 0] = True
    mask[0, half] = True
    brain.retire_scores[0, half, 0] = 2.
    brain.retire_scores[0, half + 1, half] = 1.
    brain.refresh_matrices()
    ids = brain._edit(np.array([False]), np.array([True]), random=False)
    assert ids[0, 1] == (half + 1) * lab.WIDTH + half
    assert mask[half, 0] and not mask[half + 1, half]
    assert lab.strongly_connected(lab.WIDTH, zip(*np.nonzero(mask.T), strict=True))


def test_growth_and_retirement_have_no_configured_size_cutoffs():
    brain = population(lab.Schedule(1, 1, 0.))
    grow = np.array([[True, False]])
    retire = np.array([[False, True]])
    for _ in range(24):
        assert brain.random_edit(grow)[0, 0] >= 0
    assert brain.mask.sum() == 88
    # Retire secondary arcs by increasing their score, leaving the simple ring intact.
    idx = np.arange(lab.WIDTH)
    inner = np.zeros_like(brain.mask[0])
    inner[idx, (idx - 1) % lab.WIDTH] = True
    for _ in range(56):
        brain.retire_scores[0] = brain.mask[0] & ~inner
        assert brain._edit(np.array([False]), np.array([True]), random=False)[0, 1] >= 0
    assert brain.mask.sum() == lab.WIDTH
    assert brain.random_edit(retire)[0, 1] == -1


def test_public_receipt_reports_contacted_side_with_arbitrary_target_ids():
    obs = np.zeros((3, world.OBS_SIZE), dtype=np.float32)
    obs[:, world.LEFT_ID] = (0., 1.)
    obs[0, world.CONTACT_ID] = (0., 1.)
    obs[1, world.CONTACT_ID] = (1., 0.)
    obs[0, world.REWARD_NEG] = 1.
    obs[1, world.REWARD_POS] = 1.
    side, sign = study.receipt(obs)
    np.testing.assert_array_equal(side, [0, 1, -1])
    np.testing.assert_array_equal(sign, [-1, 1, 0])
