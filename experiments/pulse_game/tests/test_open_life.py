"""Regression checks for the receipt-trained contextual circuit."""

import numpy as np

from experiments.pulse_game import live_module_edits, open_life_model, open_life_study


def population(*, contextual=True):
    rng = np.random.default_rng(298)
    width = open_life_model.WIDTH
    parameters = {key: (rng.normal(size=shape) * .24).astype(np.float32) for key, shape in {
        "drive": (1, 5, 2 * width), "bias": (1, 2 * width), "readout": (1, width), "offset": (1,),
        "gain": (1, width),
    }.items()}
    mask = np.zeros((1, width, width), dtype=bool)
    target = np.arange(width)
    mask[0, target, (target - 1) % width] = True
    cfg = open_life_study.config(open_life_study.CANDIDATES[0])
    return open_life_model.Population(parameters, mask, live_module_edits.Schedule(3, 2, .5),
                                     (34,), config=cfg, learning_rate=.02, contextual=contextual)


def play(brain):
    brain.begin_request()
    frames = np.array([[1.0, 0, 0, .9, 0], [0, 0, 0, 0, 0], [0, .5, .8, .85, 0],
                       [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=np.float32)
    for tick, frame in enumerate(frames):
        active = np.array([True]) if tick >= 4 else None
        brain.act(frame[None], active, record=True)
    return (brain.logit_sum / brain.trace_ticks).item()


def test_context_and_head_use_signed_own_contact_and_keep_carried_state():
    brain = population()
    first = play(brain)
    assert first != 0
    current = brain.state.copy()
    initial = brain.head.copy()
    brain.finish_request(np.array([1]), np.array([-1]))
    np.testing.assert_array_equal(brain.state, current)
    assert (brain.head != initial).any()
    assert brain.key_mass.sum() == 1
    assert brain.coefficients[0, 0] < 0
    play(brain)
    assert brain.context.item() < 0
    snapshot = {"mass": brain.key_mass.copy(), "head": brain.head.copy(), "coefficient": brain.coefficients.copy(),
                "recurrent": brain.raw.copy(), "mask": brain.mask.copy()}
    rng_decoy = np.array([1])
    edits, _ = brain.finish_request(np.array([-1]), rng_decoy)
    np.testing.assert_array_equal(edits, -1)
    np.testing.assert_array_equal(brain.key_mass, snapshot["mass"])
    np.testing.assert_array_equal(brain.head, snapshot["head"])
    np.testing.assert_array_equal(brain.coefficients, snapshot["coefficient"])
    np.testing.assert_array_equal(brain.raw, snapshot["recurrent"])
    np.testing.assert_array_equal(brain.mask, snapshot["mask"])


def test_recurrent_gradient_includes_context_derived_from_public_markers():
    brain = population()
    play(brain)
    brain.finish_request(np.array([1]), np.array([-1]))
    carried = brain.state.copy()
    play(brain)
    assert np.linalg.norm(brain.key_derivative) > 0
    gradient = brain.gradients(1 / brain.trace_ticks)
    parameters = (brain.raw, *(brain.parameters[name] for name in ("drive", "bias", "readout", "offset")))
    for name, parameter, expected in zip(("raw", "drive", "bias", "readout", "offset"), parameters, gradient, strict=True):
        allowed = np.where(brain.mask, abs(expected), 0.) if name == "raw" else abs(expected)
        at = np.unravel_index(allowed.argmax(), parameter.shape)
        value = parameter[at]
        trial = []
        for direction in (1, -1):
            parameter[at] = value + direction * .002
            brain.state = carried.copy()
            brain.refresh_matrices()
            trial.append(play(brain))
        parameter[at] = value
        np.testing.assert_allclose((trial[0] - trial[1]) / .004, expected[at], rtol=.008, atol=8e-5)
