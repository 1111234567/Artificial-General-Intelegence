"""Fast CPU checks for the controlled local-learning experiment."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from experiments.streaming import local_learning as local


class LocalLearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.task = local.make_task(42, input_dim=5, n_train=31, n_val=13, n_test=17)

    def test_context_conflict_is_resolvable_by_the_supplied_features(self):
        a, target_a = self.task.batch("test", context=0)
        b, target_b = self.task.batch("test", context=1)
        self.assertTrue(torch.all(target_a != target_b))
        self.assertTrue(torch.equal(target_b, (target_a + 1) % self.task.classes))
        self.assertTrue(torch.equal(a[:, 1:6], b[:, 1:6]))
        self.assertFalse(torch.equal(a, b))
        # Construct the exact linear teacher in feature space, rather than
        # requiring a training algorithm to serve as the task oracle.
        teacher_a = self.task.teacher
        teacher_b = teacher_a.roll(1, 0)
        weight = torch.zeros(self.task.classes, self.task.feature_dim)
        weight[:, 1:6] = (teacher_a + teacher_b) / 2
        weight[:, 7:] = (teacher_a - teacher_b) / (2 * self.task.context_scale)
        self.assertTrue(torch.equal((a @ weight.T).argmax(1), target_a))
        self.assertTrue(torch.equal((b @ weight.T).argmax(1), target_b))

    def test_splits_are_reproducible_independent_and_do_not_consume_global_rng(self):
        before = torch.random.get_rng_state().clone()
        duplicate = local.make_task(42, input_dim=5, n_train=31, n_val=13, n_test=17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name in ("train", "val", "test"):
            self.assertTrue(torch.equal(self.task.splits[name].x, duplicate.splits[name].x))
        self.assertFalse(torch.equal(self.task.splits["train"].x[:13],
                                     self.task.splits["val"].x))
        other_sizes = local.make_task(42, input_dim=5, n_train=99, n_val=13, n_test=17)
        self.assertTrue(torch.equal(self.task.splits["test"].x,
                                    other_sizes.splits["test"].x))
        self.assertTrue(torch.equal(self.task.splits["val"].x,
                                    other_sizes.splits["val"].x))

    def test_sampling_uses_only_the_explicit_rng(self):
        first = torch.Generator().manual_seed(19)
        second = torch.Generator().manual_seed(19)
        before = torch.random.get_rng_state().clone()
        a = local.sample_batch(self.task, "train", 1, 7, first)
        b = local.sample_batch(self.task, "train", 1, 7, second)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_analytic_delta_matches_one_layer_bp_over_multiple_updates(self):
        analytic = local.LocalClassifier(self.task.feature_dim, 3, 18).double()
        bp = copy.deepcopy(analytic)
        optimizer = torch.optim.SGD(bp.parameters(), lr=0.17)
        generator = torch.Generator().manual_seed(91)
        for context in (0, 1, 0, 1, 0):
            features, targets = local.sample_batch(self.task, "train", context, 11, generator)
            features = features.double()
            local.local_delta_update(analytic, features, targets, 0.17)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(bp(features), targets).backward()
            optimizer.step()
            torch.testing.assert_close(analytic.weight, bp.weight, rtol=1e-12, atol=1e-12)

    def test_local_paths_do_not_call_reverse_mode_autograd(self):
        features, targets = self.task.batch("train")
        for method in ("frozen", "frequency", "local_delta", "reward_local"):
            with self.subTest(method=method):
                model = local.LocalClassifier(self.task.feature_dim, 3, 18)
                with mock.patch.object(torch.Tensor, "backward",
                                       side_effect=AssertionError("backward called")), \
                     mock.patch.object(torch.autograd, "grad",
                                       side_effect=AssertionError("autograd.grad called")):
                    local.learning_step(method, model, features, targets,
                                        local.ActivityState.new(3), 0.2,
                                        torch.Generator().manual_seed(19))
                self.assertIsNone(model.weight.grad)

    def test_frequency_rule_is_label_invariant_and_only_changes_bias(self):
        original = local.LocalClassifier(self.task.feature_dim, 3, 18)
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        state_a, state_b = local.ActivityState.new(3), local.ActivityState.new(3)
        rng_a, rng_b = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
        features, targets = self.task.batch("train")
        for _ in range(9):
            local.learning_step("frequency", first, features, targets,
                                state_a, 0.2, rng_a)
            local.learning_step("frequency", second, features, (targets + 1) % 3,
                                state_b, 0.2, rng_b)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(state_a.rate_ema, state_b.rate_ema))
        self.assertTrue(torch.equal(first.weight[:, 1:], original.weight[:, 1:]))
        self.assertFalse(torch.equal(first.weight[:, 0], original.weight[:, 0]))

    def test_frozen_means_weights_bias_and_auxiliary_state_are_all_frozen(self):
        model = local.LocalClassifier(self.task.feature_dim, 3, 18)
        before = model.weight.detach().clone()
        state = local.ActivityState.new(3)
        initial_rate = state.rate_ema.clone()
        for context in (0, 1, 0):
            features, targets = self.task.batch("train", context=context)
            local.learning_step("frozen", model, features, targets, state, 0.2,
                                torch.Generator().manual_seed(19))
        self.assertTrue(torch.equal(before, model.weight))
        self.assertTrue(torch.equal(initial_rate, state.rate_ema))
        self.assertEqual(state.reward_baseline, 0.0)

    def test_frequency_feedback_pushes_an_overactive_output_down(self):
        model = local.LocalClassifier(self.task.feature_dim, 3, 18)
        state = local.ActivityState.new(3)
        before = model.weight.detach().clone()
        local.frequency_homeostasis_update(model, torch.zeros(20, dtype=torch.long),
                                            state, 0.2)
        self.assertLess(model.weight[0, 0].item(), before[0, 0].item())
        self.assertTrue(torch.all(model.weight[1:, 0] > before[1:, 0]))
        self.assertTrue(torch.equal(model.weight[:, 1:], before[:, 1:]))

    def test_reward_eligibility_uses_the_reward_sign_and_has_no_reward_no_update(self):
        model = local.LocalClassifier(self.task.feature_dim, 3, 18)
        features, _ = self.task.batch("train", torch.tensor([0]))
        probability = model(features).softmax(1).detach()
        action = torch.tensor([1])
        before = model.weight.detach().clone()
        no_reward = copy.deepcopy(model)
        local.reward_eligibility_update(no_reward, features, probability, action,
                                       torch.tensor([0.]), local.ActivityState.new(3), 0.2)
        self.assertTrue(torch.equal(before, no_reward.weight))
        rewarded, unrewarded = copy.deepcopy(model), copy.deepcopy(model)
        local.reward_eligibility_update(
            rewarded, features, probability, action, torch.tensor([1.]),
            local.ActivityState(torch.full((3,), 1/3), 0.5), 0.2)
        local.reward_eligibility_update(
            unrewarded, features, probability, action, torch.tensor([0.]),
            local.ActivityState(torch.full((3,), 1/3), 0.5), 0.2)
        self.assertGreater(rewarded(features).softmax(1)[0, 1].item(),
                           probability[0, 1].item())
        self.assertLess(unrewarded(features).softmax(1)[0, 1].item(),
                        probability[0, 1].item())
        torch.testing.assert_close(rewarded.weight - before, before - unrewarded.weight,
                                   rtol=1e-5, atol=1e-8)

    def test_expected_local_reward_update_equals_expected_reward_gradient(self):
        # Enumerating three actions avoids a noisy Monte Carlo assertion and
        # checks that the "local" rule really is a policy-gradient estimator.
        model = local.LocalClassifier(self.task.feature_dim, 3, 18).double()
        features, target = self.task.batch("train", torch.tensor([0]))
        features = features.double()
        probability = model(features).softmax(1)
        expected_reward = probability[0, target.item()]
        expected_gradient, = torch.autograd.grad(expected_reward, model.weight)
        averaged_update = torch.zeros_like(model.weight)
        for action in range(3):
            candidate = copy.deepcopy(model)
            state = local.ActivityState(torch.full((3,), 1/3), 0.23)
            local.reward_eligibility_update(
                candidate, features, probability.detach(), torch.tensor([action]),
                torch.tensor([float(action == target.item())]), state, 0.2)
            averaged_update += probability[0, action].detach() * (
                candidate.weight.detach() - model.weight.detach())
        torch.testing.assert_close(averaged_update, 0.2 * expected_gradient,
                                   rtol=1e-12, atol=1e-12)

    def test_pilot_never_evaluates_test_split(self):
        config = local.Config(seeds=(42,), input_dim=5, n_train=31, n_val=13,
                              n_test=17, phase_steps=2, batch_size=4,
                              eval_every=1, pilot=True)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(local, "evaluate", wraps=local.evaluate) as evaluate:
                result = local.run_regime("local_delta", "sequential", 42, self.task,
                                          config, Path(directory))
            self.assertTrue(all(call.args[2] != "test" for call in evaluate.call_args_list))
            self.assertNotIn("initial_test", result)
            self.assertTrue(all("test" not in phase for phase in result["phases"]))

    def test_existing_output_directory_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                local.run_experiment(local.Config(seeds=(0,), phase_steps=1), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
