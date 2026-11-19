"""Tiny mathematical references only; no task training or formal-state reads."""
import math
import unittest

import torch

from experiments.continuity.model import AgentModel
from experiments.credit.local import ForwardEligibility, LocalConfig
from experiments.credit.tangent_audit import advance_tangents, pair_statistics, tensor_sha256, unit_directions


def fixture(diagonal=False):
    with torch.random.fork_rng(devices=[]):
        model = AgentModel(input_size=2, width=3, n_jobs=1, n_actions=3).double()
    with torch.no_grad():
        weight = torch.tensor([[.2, -.5, .1], [.3, -.1, .4], [-.2, .6, .15]], dtype=torch.float64)
        model.recurrent.copy_(torch.diag(weight.diag()) if diagonal else weight)
        model.encoder.weight.copy_(torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]], dtype=torch.float64))
    hidden = torch.tensor([[.2, -.3, .5], [-.4, .6, .1]], dtype=torch.float64)
    observations = [torch.tensor([[.3, -.2], [.1, .4]], dtype=torch.float64),
                    torch.zeros(2, 2, dtype=torch.float64),
                    torch.tensor([[-.2, .1], [.3, -.1]], dtype=torch.float64)]
    return model, hidden, observations, [.125, .5, .3]


class TangentAuditTests(unittest.TestCase):
    def test_exact_multistep_tangent_matches_tiny_autograd_jvp(self):
        model, hidden, observations, intervals = fixture()
        directions = unit_directions(2, 4, 3, torch.Generator().manual_seed(311))
        original_h, exact, diagonal = hidden.clone(), directions.clone(), directions.clone()
        for observation, dt in zip(observations, intervals):
            hidden, exact, diagonal = advance_tangents(model.recurrent, model.encoder.weight, hidden,
                                                       observation, exact, diagonal, dt)
        def trajectory(initial):
            state = initial
            for observation, dt in zip(observations, intervals):
                _, state = model.step(observation, state, dt)
            return state
        expected_h = trajectory(original_h)
        torch.testing.assert_close(hidden, expected_h, rtol=0, atol=0)
        for index in range(4):
            _, jvp = torch.autograd.functional.jvp(trajectory, original_h, directions[:, index])
            torch.testing.assert_close(exact[:, index], jvp, rtol=1e-13, atol=1e-13)
            _, feature_jvp = torch.autograd.functional.jvp(lambda initial: trajectory(initial).tanh(),
                                                          original_h, directions[:, index])
            torch.testing.assert_close((1-hidden.tanh().square())*exact[:, index], feature_jvp, rtol=1e-13, atol=1e-13)
        self.assertGreater(float((exact-diagonal).norm()), .01)

    def test_diagonal_weight_is_exact_positive_control(self):
        model, hidden, observations, intervals = fixture(diagonal=True)
        exact = unit_directions(2, 8, 3, torch.Generator().manual_seed(312))
        diagonal = exact.clone()
        for observation, dt in zip(observations, intervals):
            hidden, exact, diagonal = advance_tangents(model.recurrent, model.encoder.weight, hidden,
                                                       observation, exact, diagonal, dt)
        torch.testing.assert_close(exact, diagonal, rtol=1e-14, atol=1e-14)

    def test_diagonal_map_matches_frozen_helper_gain_and_preserves_inputs_rng(self):
        model, hidden, observations, _ = fixture()
        process_rng = torch.get_rng_state().clone()
        initial = unit_directions(2, 8, 3, torch.Generator().manual_seed(313))
        self.assertTrue(torch.equal(process_rng, torch.get_rng_state()))
        torch.testing.assert_close(torch.linalg.vector_norm(initial, dim=-1), torch.ones(2, 8, dtype=torch.float64), rtol=1e-14, atol=1e-14)
        inputs = [model.recurrent, model.encoder.weight, hidden, observations[0], initial]
        before = [tensor_sha256(value) for value in inputs]
        post, exact, diagonal = advance_tangents(model.recurrent, model.encoder.weight, hidden,
                                                observations[0], initial, initial, .6)
        helper = ForwardEligibility(2, 3, 2, LocalConfig(n_jobs=1, n_actions=3), dtype=torch.float64)
        with torch.no_grad():
            logits, actual_post = model.step(observations[0], hidden, .6)
        helper.observe(model, observations[0], hidden, actual_post, .6, logits, None, accumulate=False)
        torch.testing.assert_close(diagonal, initial*helper.last_gain[:, None, :], rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(post, actual_post, rtol=0, atol=0)
        self.assertEqual(before, [tensor_sha256(value) for value in inputs])
        self.assertFalse(exact.requires_grad)
        self.assertFalse(diagonal.requires_grad)
        self.assertTrue(torch.equal(process_rng, torch.get_rng_state()))

    def test_zero_elapsed_pulse_and_zero_sensitivity_conventions(self):
        model, hidden, observations, _ = fixture()
        initial = unit_directions(2, 2, 3, torch.Generator().manual_seed(314))
        post, exact, diagonal = advance_tangents(model.recurrent, model.encoder.weight, hidden,
                                                observations[0], initial, initial, 0.)
        torch.testing.assert_close(post, hidden+model.encoder(observations[0]), rtol=0, atol=0)
        torch.testing.assert_close(exact, initial, rtol=0, atol=0)
        torch.testing.assert_close(diagonal, initial, rtol=0, atol=0)
        stats = pair_statistics(torch.zeros_like(initial), torch.zeros_like(initial))
        self.assertTrue(bool(torch.isnan(stats["cosine"]).all()))
        self.assertTrue(bool(torch.isnan(stats["norm_ratio"]).all()))
        for bad in (-.1, math.nan):
            with self.assertRaises(ValueError):
                advance_tangents(model.recurrent, model.encoder.weight, hidden,
                                 observations[0], initial, initial, bad)


if __name__ == "__main__":
    unittest.main()
