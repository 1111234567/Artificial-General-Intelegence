"""Numerical and continuation checks for the forward coactivity intervention."""
from copy import deepcopy
import math
import unittest

import torch

from experiments.continuity.model import AgentModel
from experiments.continuity.run import state_digest
from experiments.integration.local import LocalConfig, TemporalOja


def activity_pair(batch=3, width=4, *, device="cpu", seed=271):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    pre = torch.randn(batch, width, generator=rng, dtype=torch.float64).to(device)
    post = torch.randn(batch, width, generator=rng, dtype=torch.float64).to(device)
    return pre, post


class TemporalOjaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_step_matches_post_by_pre_rule_without_autograd(self):
        pre = torch.tensor([[0.7, -0.4], [0.2, 0.9]], dtype=torch.float64, requires_grad=True)
        post = torch.tensor([[-0.3, 0.5], [1.1, -0.1]], dtype=torch.float64, requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor([[0.3, -0.1], [0.2, 0.5]], dtype=torch.float64))
        weight.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        original_pre, original_post, original_grad = pre.detach().clone(), post.detach().clone(), weight.grad.clone()
        original = weight.detach().clone()
        config = LocalConfig(lr=0.03, tau=2.0)
        rule = TemporalOja(2, 2, config, dtype=torch.float64)
        expected_trace = (1 - math.exp(-0.4 / config.tau)) * original_pre.tanh()
        expected_c = original_post.tanh().T @ expected_trace / 2
        expected_s = original_post.tanh().square().mean(0)
        expected_delta = config.lr * (expected_c - expected_s[:, None] * original)
        rule.observe(pre, post, 0.4)
        delta, audit = rule.proposal(weight)
        self.assertEqual(rule.updates, 0)
        torch.testing.assert_close(weight, original, rtol=0, atol=0)
        torch.testing.assert_close(delta, expected_delta, rtol=1e-14, atol=1e-14)
        result = rule.apply_(weight)
        torch.testing.assert_close(weight, original + expected_delta, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(rule.trace, expected_trace, rtol=1e-14, atol=1e-14)
        for observed, expected in ((pre, original_pre), (post, original_post), (weight.grad, original_grad)):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertFalse(rule.trace.requires_grad)
        self.assertFalse(rule.coactivity_sum.requires_grad)
        self.assertFalse(delta.requires_grad)
        self.assertEqual(audit, result)
        self.assertEqual(result["batch_ticks"], 2)
        self.assertEqual(result["outer_product_terms"], 8)
        self.assertEqual(rule.pending_ticks, 0)
        self.assertEqual(rule.matrix_writes, 1)

    def test_trace_uses_elapsed_time_and_survives_chunk_application(self):
        pre, post = activity_pair()
        config = LocalConfig(lr=0.01, tau=16)
        whole = TemporalOja(3, 4, config, dtype=torch.float64)
        split = TemporalOja(3, 4, config, dtype=torch.float64)
        whole.observe(pre, post, 0.75)
        split.observe(pre, post, 0.25)
        expected_trace = split.trace.clone()
        split.apply_(torch.zeros(4, 4, dtype=torch.float64))
        torch.testing.assert_close(split.trace, expected_trace, rtol=0, atol=0)
        split.observe(pre, post, 0.5)
        torch.testing.assert_close(whole.trace, split.trace, rtol=1e-14, atol=1e-14)
        # After 16 environment units the old trace retains e^-1, independent of tick count.
        split.trace.fill_(1)
        split.observe(torch.zeros_like(pre), post, 16, accumulate=False)
        torch.testing.assert_close(split.trace, torch.full_like(pre, math.exp(-1)), rtol=1e-14, atol=1e-14)

    def test_accumulation_is_time_weighted_and_burnin_has_no_update_mass(self):
        rule = TemporalOja(3, 4, LocalConfig(tau=2), dtype=torch.float64)
        expected_c, expected_s = torch.zeros(4, 4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
        for index, (dt, accumulate) in enumerate(((0.0, True), (1.0, False), (0.2, True), (0.7, True))):
            pre, post = activity_pair(seed=index)
            rule.observe(pre, post, dt, accumulate=accumulate)
            if accumulate and dt:
                expected_c += dt * post.tanh().T @ rule.trace / 3
                expected_s += dt * post.tanh().square().mean(0)
        stats = rule.statistics()
        torch.testing.assert_close(stats["coactivity_sum"], expected_c, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(stats["post_second_sum"], expected_s, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(stats["elapsed"], 0.9)
        self.assertEqual(stats["ticks"], 2)
        summary = rule.summary()
        self.assertEqual(summary["observed_ticks"], 4)
        self.assertEqual(summary["plasticity_ticks"], 2)
        self.assertEqual(summary["plasticity_batch_ticks"], 6)
        self.assertEqual(summary["outer_product_terms"], 2 * 3 * 4**2)
        self.assertAlmostEqual(summary["plasticity_batch_elapsed"], 2.7)
        self.assertAlmostEqual(summary["observed_elapsed_per_trajectory"], 1.9)
        stats["coactivity_sum"].zero_()
        self.assertTrue(bool(rule.coactivity_sum.abs().sum() > 0))

    def test_mixture_normalizes_each_source_before_one_update(self):
        config = LocalConfig(lr=0.03, tau=2)
        live = TemporalOja(3, 4, config, dtype=torch.float64)
        old = TemporalOja(3, 4, LocalConfig(lr=0.03, tau=2, seed=17), dtype=torch.float64)
        live.observe(*activity_pair(seed=0), 0.1)
        old.observe(*activity_pair(seed=1), 0.25, accumulate=False)
        old.observe(*activity_pair(seed=2), 2.0)
        before_live_trace, before_old_trace = live.trace.clone(), old.trace.clone()
        c = 0.5 * (live.coactivity_sum / live.pending_elapsed + old.coactivity_sum / old.pending_elapsed)
        s = 0.5 * (live.post_second_sum / live.pending_elapsed + old.post_second_sum / old.pending_elapsed)
        weight = torch.arange(16, dtype=torch.float64).view(4, 4) / 10
        before = weight.clone()
        expected_delta = 0.03 * (c - s[:, None] * before)
        audit = live.apply_(weight, other=old)
        torch.testing.assert_close(weight, before + expected_delta, rtol=1e-14, atol=1e-14)
        self.assertEqual(audit["source_weights"], [0.5, 0.5])
        self.assertEqual(audit["source_ticks"], [1, 1])
        self.assertEqual(audit["source_elapsed_per_trajectory"], [0.1, 2.0])
        self.assertEqual(live.updates, 1)
        self.assertEqual(live.matrix_writes, 1)
        self.assertEqual(old.updates, 0)
        self.assertEqual(live.pending_ticks, 0)
        self.assertEqual(old.pending_ticks, 0)
        torch.testing.assert_close(live.trace, before_live_trace, rtol=0, atol=0)
        torch.testing.assert_close(old.trace, before_old_trace, rtol=0, atol=0)

    def test_shuffle_preserves_activity_marginals_and_decay_but_breaks_pairing(self):
        pre, post = activity_pair(batch=5, width=4)
        live = TemporalOja(5, 4, LocalConfig(tau=2), dtype=torch.float64)
        shuffled = TemporalOja(5, 4, LocalConfig(tau=2, shuffled=True, seed=7), dtype=torch.float64)
        live.observe(pre, post, 0.5)
        shuffled.observe(pre, post, 0.5)
        torch.testing.assert_close(shuffled.trace, live.trace, rtol=0, atol=0)
        torch.testing.assert_close(shuffled.post_second_sum, live.post_second_sum, rtol=0, atol=0)
        self.assertGreater(shuffled.last_shift, 0)
        self.assertLess(shuffled.last_shift, 5)
        shifted = torch.roll(shuffled.trace, shuffled.last_shift, dims=0)
        torch.testing.assert_close(shifted.sort(dim=0).values, live.trace.sort(dim=0).values, rtol=0, atol=0)
        expected_c = 0.5 * post.tanh().T @ shifted / 5
        torch.testing.assert_close(shuffled.coactivity_sum, expected_c, rtol=1e-14, atol=1e-14)
        self.assertFalse(torch.equal(live.coactivity_sum, shuffled.coactivity_sum))

    def test_private_shuffle_rng_never_uses_global_torch_rng(self):
        pre, post = activity_pair()
        global_state = torch.get_rng_state().clone()
        rule = TemporalOja(3, 4, LocalConfig(shuffled=True, seed=100), dtype=torch.float64)
        for _ in range(5):
            rule.observe(pre, post, 0.25)
        rule.apply_(torch.zeros(4, 4, dtype=torch.float64))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_state))

    def _assert_resume(self, device):
        config = LocalConfig(shuffled=True, seed=101, tau=16)
        original = TemporalOja(3, 4, config, device=device, dtype=torch.float64)
        for index in range(3):
            original.observe(*activity_pair(seed=index, device=device), 0.25)
        saved = original.state_dict()
        expected_saved_hash = state_digest(saved)
        resumed = TemporalOja(3, 4, config, device=device, dtype=torch.float64)
        resumed.load_state_dict(saved)
        self.assertEqual(expected_saved_hash, state_digest(resumed.state_dict()))
        for index in range(3, 8):
            pre, post = activity_pair(seed=index, device=device)
            original.observe(pre, post, 0.125 * index)
            resumed.observe(pre, post, 0.125 * index)
        weight_a = torch.eye(4, dtype=torch.float64, device=device)
        weight_b = weight_a.clone()
        original.apply_(weight_a)
        resumed.apply_(weight_b)
        self.assertTrue(torch.equal(weight_a, weight_b))
        self.assertEqual(state_digest(original.state_dict()), state_digest(resumed.state_dict()))
        self.assertEqual(expected_saved_hash, state_digest(saved))
        # Checkpointed traces/RNG/counters continue across a consumed chunk too.
        boundary = original.state_dict()
        resumed.load_state_dict(boundary)
        for rule in (original, resumed):
            rule.observe(*activity_pair(seed=37, device=device), 0.25)
        self.assertEqual(state_digest(original.state_dict()), state_digest(resumed.state_dict()))

    def test_exact_resume_including_pending_accumulator_and_private_rng(self):
        self._assert_resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_exact_cuda_resume(self):
        self._assert_resume("cuda")

    def test_history_restart_keeps_private_rng_and_live_trace_untouched(self):
        config = LocalConfig(shuffled=True, seed=11)
        live = TemporalOja(3, 4, config, dtype=torch.float64)
        old = TemporalOja(3, 4, config, dtype=torch.float64)
        for rule in (live, old):
            rule.observe(*activity_pair(), 0.25)
        live_snapshot = state_digest(live.state_dict())
        with self.assertRaises(RuntimeError):
            old.reset_history_()
        rng_before = old.rng.get_state().clone()
        old.clear_accumulator()
        old.reset_history_()
        self.assertTrue(torch.equal(rng_before, old.rng.get_state()))
        self.assertEqual(old.observed_ticks, 1)
        self.assertEqual(old.history_restarts, 1)
        self.assertEqual(int(torch.count_nonzero(old.trace)), 0)
        self.assertEqual(live_snapshot, state_digest(live.state_dict()))

    def test_zero_lr_is_parameter_identity_and_only_recurrent_weights_can_change(self):
        model = AgentModel(width=4).double()
        pre, post = activity_pair()
        snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        zero = TemporalOja(3, 4, LocalConfig(lr=0), dtype=torch.float64)
        version_before = model.recurrent._version
        zero.observe(pre, post, 0.25)
        audit = zero.apply_(model.recurrent)
        self.assertEqual(audit["delta_frob"], 0)
        self.assertEqual(zero.matrix_writes, 0)
        self.assertEqual(model.recurrent._version, version_before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        local = TemporalOja(3, 4, LocalConfig(lr=0.01), dtype=torch.float64)
        local.observe(pre.requires_grad_(), post.requires_grad_(), 0.25)
        local.apply_(model.recurrent)
        for name, value in model.state_dict().items():
            if name != "recurrent":
                torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.recurrent, snapshot["recurrent"]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_empty_or_zero_time_observation_has_no_plasticity(self):
        rule = TemporalOja(3, 4, dtype=torch.float64)
        before = torch.eye(4, dtype=torch.float64)
        weight = before.clone()
        rule.observe(*activity_pair(), 0.0)
        audit = rule.apply_(weight)
        self.assertFalse(audit["applied"])
        self.assertEqual(rule.updates, 0)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        torch.testing.assert_close(rule.trace, torch.zeros_like(rule.trace), rtol=0, atol=0)

    def test_invalid_time_dimensions_nonfinite_and_checkpoint_are_rejected(self):
        for changes in ({"lr": -1}, {"lr": float("nan")}, {"tau": 0}, {"tau": float("inf")},
                        {"seed": -1}, {"shuffled": 1}):
            with self.subTest(config=changes), self.assertRaises(ValueError):
                LocalConfig(**changes)
        with self.assertRaises(ValueError):
            TemporalOja(1, 4, LocalConfig(shuffled=True))
        rule = TemporalOja(3, 4, dtype=torch.float64)
        pre, post = activity_pair()
        for dt in (-0.1, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                rule.observe(pre, post, dt)
        with self.assertRaises(ValueError):
            rule.observe(pre[:, :2], post, 0.25)
        with self.assertRaises(ValueError):
            rule.observe(pre.float(), post, 0.25)
        rule.observe(pre, post, 0.25)
        state = rule.state_dict()
        for key, invalid in (("pending_ticks", -1), ("pending_elapsed", 0.0),
                             ("observed_ticks", 0), ("width", 9)):
            damaged = deepcopy(state)
            damaged[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                rule.load_state_dict(damaged)
        damaged = deepcopy(state)
        damaged["trace"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            rule.load_state_dict(damaged)
        weight = torch.eye(4, dtype=torch.float64)
        before = weight.clone()
        rule.coactivity_sum[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            rule.apply_(weight)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)

    def test_mixture_requires_two_independent_compatible_exposures(self):
        live = TemporalOja(3, 4, dtype=torch.float64)
        old = TemporalOja(3, 4, dtype=torch.float64)
        weight = torch.zeros(4, 4, dtype=torch.float64)
        live.observe(*activity_pair(), 0.25)
        with self.assertRaises(ValueError):
            live.apply_(weight, other=live)
        with self.assertRaises(ValueError):
            live.apply_(weight, other=old)
        wrong_tau = TemporalOja(3, 4, LocalConfig(tau=1), dtype=torch.float64)
        wrong_tau.observe(*activity_pair(), 0.25)
        with self.assertRaises(ValueError):
            live.apply_(weight, other=wrong_tau)


if __name__ == "__main__":
    unittest.main()
