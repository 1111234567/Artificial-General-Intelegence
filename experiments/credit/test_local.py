"""Tiny frozen-parameter references and continuation checks; no training jobs."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.continuity.model import AgentModel
from experiments.continuity.run import state_digest
from experiments.credit.local import ForwardEligibility, LocalConfig, analytic_head_gradients


def tiny_model(width=4, *, diagonal=False, device="cpu", dtype=torch.float64):
    model = AgentModel(input_size=3, n_jobs=2, n_actions=3, width=width).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.recurrent.copy_(.23*torch.arange(width*width, device=device, dtype=dtype).sin().view(width, width))
        if diagonal:
            model.recurrent.copy_(torch.diag(torch.linspace(-.3, .3, width, device=device, dtype=dtype)))
        model.encoder.weight.copy_(.11*torch.arange(width*3, device=device, dtype=dtype).cos().view(width, 3))
        model.head.weight.copy_(.17*torch.arange(6*width, device=device, dtype=dtype).sin().view(6, width))
        model.head.bias.copy_(.03*torch.arange(6, device=device, dtype=dtype).cos())
    return model


def sequence(*, device="cpu", dtype=torch.float64, length=7):
    observations, labels = [], []
    for index in range(length):
        x = (.6*torch.arange(6, device=device, dtype=dtype).add(index).cos()).view(2, 3)
        observations.append(x if index in {0, 2, 5} else torch.zeros_like(x))
        labels.append((torch.arange(4, device=device).view(2, 2)+index) % 3)
    return observations, labels, [0., .125, .25, .5, .3, .25, .125][:length]


def manual_gradients(model, helper, observations, labels, intervals, *, loss_start=0):
    state = model.initial_state(2)
    for index, (obs, target, dt) in enumerate(zip(observations, labels, intervals)):
        previous = state
        with torch.no_grad():
            output, state = model.step(obs, state, dt)
        helper.observe(model, obs, previous, state, dt, output,
                       target if index >= loss_start else None, accumulate=index >= loss_start)
    return helper.normalized_gradients()[0], state


def bptt_reference(model, observations, labels, intervals, *, loss_start=0):
    state, outputs = model.initial_state(2), []
    for index, (obs, dt) in enumerate(zip(observations, intervals)):
        output, state = model.step(obs, state, dt)
        if index >= loss_start:
            outputs.append(output)
    weight = state.new_tensor([.17, 1., 1.])
    loss = F.cross_entropy(torch.stack(outputs).reshape(-1, 3), torch.stack(labels[loss_start:]).reshape(-1), weight=weight)
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(named.values()))
    return dict(zip(named, gradients)), state


@torch.no_grad()
def full_rtrl_reference(model, observations, labels, intervals, *, loss_start=0):
    """Exact frozen-theta forward influence for the actual two-stage solver.

    P[b,k,p] retains ALL state-neuron/parameter influences; used only at tiny
    dimensions. W then encoder entries are flattened in post-by-pre order.
    """
    batch, width, inputs = 2, model.width, model.input_size
    n_weight, n_core = width**2, width**2+width*inputs
    state = model.initial_state(batch)
    influence = state.new_zeros(batch, width, n_core)
    core_gradient = state.new_zeros(n_core)
    selector = torch.eye(width, device=state.device, dtype=state.dtype)[None, :, :, None]
    head_gradient = {"head.weight": torch.zeros_like(model.head.weight), "head.bias": torch.zeros_like(model.head.bias)}
    mass = 0.
    for index, (obs, target, dt) in enumerate(zip(observations, labels, intervals)):
        n = max(1, math.ceil(dt/model.max_step))
        delta = dt/n
        for _ in range(n):
            rate = state.tanh()
            first = -state+F.linear(rate, model.recurrent)
            direct = state.new_zeros(batch, width, n_core)
            direct[:, :, :n_weight] = (selector*rate[:, None, None, :]).reshape(batch, width, n_weight)
            first_P = -influence+torch.einsum("ki,bip->bkp", model.recurrent, (1-rate.square())[:, :, None]*influence)+direct
            trial, trial_P = state+delta*first, influence+delta*first_P
            trial_rate = trial.tanh()
            second = -trial+F.linear(trial_rate, model.recurrent)
            direct_trial = state.new_zeros(batch, width, n_core)
            direct_trial[:, :, :n_weight] = (selector*trial_rate[:, None, None, :]).reshape(batch, width, n_weight)
            second_P = -trial_P+torch.einsum("ki,bip->bkp", model.recurrent, (1-trial_rate.square())[:, :, None]*trial_P)+direct_trial
            state = state+.5*delta*(first+second)
            influence = influence+.5*delta*(first_P+second_P)
        state = state+model.encoder(obs)
        influence[:, :, n_weight:] += (selector*obs[:, None, None, :]).reshape(batch, width, width*inputs)
        output = model.head(state.tanh()).reshape(batch, 2, 3)
        if index >= loss_start:
            head, stats = analytic_head_gradients(output, state, target, .17)
            signal = (stats["residual"].flatten(1) @ model.head.weight)*(1-state.tanh().square())
            core_gradient += torch.einsum("bi,bip->p", signal, influence)
            for name in head:
                head_gradient[name] += head[name]
            mass += float(stats["weight_mass"])
    return {"recurrent": core_gradient[:n_weight].view(width, width)/mass,
            "encoder.weight": core_gradient[n_weight:].view(width, inputs)/mass,
            **{name: value/mass for name, value in head_gradient.items()}}, state


class ForwardEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def helper(self, model, **changes):
        values = dict(n_jobs=2, n_actions=3, wait_weight=.17, seed=881)
        values.update(changes)
        return ForwardEligibility(2, model.width, 3, LocalConfig(**values), device=model.recurrent.device, dtype=model.recurrent.dtype)

    def test_analytic_head_is_exact_weighted_ce_gradient(self):
        model = tiny_model()
        h = torch.tensor([[.2, -.3, .1, .7], [-.4, .6, .1, -.2]], dtype=torch.float64)
        output = model.head(h.tanh()).reshape(2, 2, 3)
        target = torch.tensor([[0, 1], [2, 0]])
        raw, stats = analytic_head_gradients(output, h, target, .17)
        loss = F.cross_entropy(output.reshape(-1, 3), target.reshape(-1), weight=output.new_tensor([.17, 1., 1.]))
        reference = torch.autograd.grad(loss, (model.head.weight, model.head.bias))
        for name, expected in zip(("head.weight", "head.bias"), reference):
            torch.testing.assert_close(raw[name]/stats["weight_mass"], expected, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(float(stats["weight_mass"]), 2.34)
        self.assertAlmostEqual(float(stats["weighted_loss_sum"]/stats["weight_mass"]), float(loss.detach()), places=13)

    def test_full_forward_rtrl_matches_bptt_on_dense_heun_with_burnin_graph(self):
        model = tiny_model()
        data = sequence()
        for burnin in (0, 3):
            with self.subTest(burnin=burnin):
                expected, h = bptt_reference(model, *data, loss_start=burnin)
                actual, rtrl_h = full_rtrl_reference(model, *data, loss_start=burnin)
                torch.testing.assert_close(rtrl_h, h, rtol=0, atol=0)
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-13)

    def test_diagonal_weight_positive_control_matches_full_gradient(self):
        for width in (1, 4):
            for burnin in (0, 3):
                with self.subTest(width=width, burnin=burnin):
                    model = tiny_model(width, diagonal=True)
                    data = sequence()
                    expected, expected_h = bptt_reference(model, *data, loss_start=burnin)
                    helper = self.helper(model)
                    actual, h = manual_gradients(model, helper, *data, loss_start=burnin)
                    torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-13)

    def test_dense_projection_has_measurable_bias_not_claimed_equivalent(self):
        model = tiny_model()
        data = sequence()
        exact, _ = full_rtrl_reference(model, *data)
        approximate, _ = manual_gradients(model, self.helper(model), *data)
        for name in ("recurrent", "encoder.weight"):
            relative = torch.linalg.vector_norm(approximate[name]-exact[name])/torch.linalg.vector_norm(exact[name])
            self.assertGreater(float(relative), 1e-4)
            self.assertTrue(bool(torch.isfinite(relative)))
        for name in ("head.weight", "head.bias"):
            torch.testing.assert_close(approximate[name], exact[name], rtol=1e-13, atol=1e-13)

    def test_pulse_after_flow_and_zero_elapsed_time_have_correct_sources(self):
        model = tiny_model()
        helper = self.helper(model)
        x, targets, _ = sequence(length=1)
        h = model.initial_state(2)
        output, post = model.step(x[0], h, 0.)
        helper.observe(model, x[0], h, post, 0., output, targets[0])
        torch.testing.assert_close(helper.E, torch.zeros_like(helper.E), rtol=0, atol=0)
        torch.testing.assert_close(helper.F, x[0][:, None, :].expand(-1, model.width, -1), rtol=0, atol=0)
        expected, _ = bptt_reference(model, x, targets, [0.])
        actual, _ = helper.normalized_gradients()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-13, atol=1e-13)

    def test_random_feedback_row_scales_fixed_and_global_rng_untouched(self):
        model = tiny_model()
        helper = self.helper(model, feedback="random", shuffle=True)
        global_rng = torch.get_rng_state().clone()
        helper.initialize_feedback_(model.head.weight)
        expected = torch.linalg.vector_norm(model.head.weight.T, dim=1)
        torch.testing.assert_close(torch.linalg.vector_norm(helper.fixed_feedback, dim=1), expected, rtol=1e-13, atol=1e-13)
        before = helper.fixed_feedback.clone()
        manual_gradients(model, helper, *sequence())
        with torch.no_grad():
            model.head.weight.mul_(3)
        helper.initialize_feedback_(model.head.weight)
        helper.reset_eligibility_()
        torch.testing.assert_close(helper.fixed_feedback, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
        self.assertIsNotNone(helper.summary(model)["feedback"]["current_head_global_cosine"])

    def test_shuffle_changes_only_core_gradient_pairing_not_head_or_traces(self):
        model = tiny_model()
        ordinary, shuffled = self.helper(model), self.helper(model, shuffle=True)
        data = sequence()
        left, _ = manual_gradients(model, ordinary, *data)
        right, _ = manual_gradients(model, shuffled, *data)
        for name in ("head.weight", "head.bias"):
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        torch.testing.assert_close(ordinary.E, shuffled.E, rtol=0, atol=0)
        torch.testing.assert_close(ordinary.F, shuffled.F, rtol=0, atol=0)
        self.assertFalse(torch.equal(left["recurrent"], right["recurrent"]))
        self.assertGreater(shuffled.last_shift, 0)

    def test_no_backward_no_model_mutation_and_no_burnin_labels(self):
        model = tiny_model()
        helper = self.helper(model)
        snapshot = state_digest(model.state_dict())
        with patch("torch.autograd.grad", side_effect=AssertionError("reverse gradient")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("backward")):
            manual_gradients(model, helper, *sequence(), loss_start=3)
        self.assertEqual(snapshot, state_digest(model.state_dict()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(helper.E.requires_grad)
        self.assertFalse(helper.F.requires_grad)
        self.assertEqual(helper.loss_ticks, 4)
        self.assertEqual(helper.observed_ticks, 7)
        self.assertEqual(helper.summary()["reconstructed_flow_matvecs"], 18)

    def test_gain_observer_matches_actual_heun_substeps_and_is_resumable(self):
        model = tiny_model()
        with torch.no_grad():
            model.recurrent.zero_()
        helper = self.helper(model)
        state = model.initial_state(2)
        observation = state.new_zeros(2, 3)
        for dt in (0., .6):
            previous = state
            with torch.no_grad():
                output, state = model.step(observation, state, dt)
            helper.observe(model, observation, previous, state, dt, output, None, accumulate=False)
        # dt=.6 uses three .2 substeps, each with gain 1-.2+.5*.2^2.
        expected_gain = (1-.2+.5*.2**2)**3
        torch.testing.assert_close(helper.last_gain, torch.full_like(helper.last_gain, expected_gain), rtol=1e-14, atol=1e-14)
        helper.E.fill_(2.)
        helper.F.fill_(-3.)
        before = state_digest(helper.state_dict())
        report = helper.summary(model)
        self.assertEqual(before, state_digest(helper.state_dict()))
        self.assertEqual(report["E_rms"], 2.)
        self.assertEqual(report["F_rms"], 3.)
        self.assertEqual(report["E_absmax"], 2.)
        self.assertEqual(report["F_absmax"], 3.)
        gain = report["diagonal_homogeneous_gain"]
        self.assertAlmostEqual(gain["observed_min"], expected_gain, places=14)
        self.assertEqual(gain["observed_max"], 1.)
        self.assertEqual(gain["zero_gain_count"], 0)
        self.assertAlmostEqual(gain["mean_log_abs_gain_per_observation"], math.log(expected_gain)/2, places=14)
        self.assertAlmostEqual(gain["typical_log_abs_gain_over_64_observations"], 32*math.log(expected_gain), places=12)
        resumed = self.helper(model)
        resumed.load_state_dict(helper.state_dict())
        self.assertEqual(state_digest(helper.state_dict()), state_digest(resumed.state_dict()))

    def test_clear_gradients_retains_traces_reset_traces_retains_gradients(self):
        model = tiny_model()
        helper = self.helper(model)
        manual_gradients(model, helper, *sequence())
        e, f = helper.E.clone(), helper.F.clone()
        helper.clear_gradients()
        torch.testing.assert_close(helper.E, e, rtol=0, atol=0)
        torch.testing.assert_close(helper.F, f, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            helper.normalized_gradients()
        manual_gradients(model, helper, *sequence())
        gradient_hash = state_digest(helper.gradients)
        mass = helper.weight_mass.clone()
        helper.reset_eligibility_()
        self.assertEqual(state_digest(helper.gradients), gradient_hash)
        self.assertTrue(torch.equal(helper.weight_mass, mass))
        self.assertEqual(int(torch.count_nonzero(helper.E)), 0)
        self.assertEqual(int(torch.count_nonzero(helper.F)), 0)

    def _assert_exact_resume(self, device):
        model = tiny_model(device=device)
        helper = self.helper(model, feedback="random", shuffle=True)
        observations, labels, intervals = sequence(device=device)
        h = model.initial_state(2)
        for index in range(3):
            pre = h
            with torch.no_grad():
                output, h = model.step(observations[index], h, intervals[index])
            helper.observe(model, observations[index], pre, h, intervals[index], output, labels[index])
        saved = helper.state_dict()
        saved_hash = state_digest(saved)
        resumed = self.helper(model, feedback="random", shuffle=True)
        resumed.load_state_dict(saved)
        self.assertEqual(state_digest(resumed.state_dict()), saved_hash)
        for index in range(3, 7):
            pre = h
            with torch.no_grad():
                output, h = model.step(observations[index], h, intervals[index])
            for rule in (helper, resumed):
                rule.observe(model, observations[index], pre, h, intervals[index], output, labels[index])
        self.assertEqual(state_digest(helper.state_dict()), state_digest(resumed.state_dict()))
        for name, value in helper.normalized_gradients()[0].items():
            self.assertTrue(torch.equal(value, resumed.normalized_gradients()[0][name]))
        self.assertEqual(state_digest(saved), saved_hash)

    def test_exact_resume_pending_gradient_feedback_and_rng(self):
        self._assert_exact_resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_exact_cuda_resume(self):
        self._assert_exact_resume("cuda")

    def test_invalid_config_checkpoint_or_inputs_rejected(self):
        for values in ({"feedback": "other"}, {"wait_weight": 0}, {"seed": -1}, {"shuffle": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LocalConfig(**values)
        model = tiny_model()
        helper = self.helper(model)
        manual_gradients(model, helper, *sequence())
        for key, bad in (("width", 9), ("pending_loss_ticks", 99), ("last_shift", 2)):
            saved = helper.state_dict()
            saved[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.load_state_dict(saved)
        helper.E[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            helper.normalized_gradients()


if __name__ == "__main__":
    unittest.main()
