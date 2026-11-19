"""Actual-loss trials, independent events, full Adam memory and continuation."""
from copy import deepcopy
import io
import unittest

import torch

from experiments.organization.plasticity import FunctionalConfig, FunctionalOperator, MODES


def make(weight=None, *, mode="functional_task", window=1, candidates=1, seed=14):
    weight = torch.nn.Parameter(torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
                                if weight is None else weight.clone().double())
    operator = FunctionalOperator(weight, FunctionalConfig(mode, window, candidates, seed))
    optimizer = torch.optim.Adam([weight], lr=.1)
    return weight, operator, optimizer


def advance(weight, operator, optimizer, gradient=None, loss=None):
    if gradient is None:
        gradient = torch.zeros_like(weight)
    optimizer.zero_grad(set_to_none=True)
    weight.grad = operator.before_gradient(gradient).clone()
    optimizer.step()
    operator.loss_fn = loss
    group = optimizer.param_groups[0]
    return operator.apply_after_optimizer(weight, optimizer.state[weight], gradient,
        1., group["lr"], group["betas"], group["eps"])


def closure(weight, target, counter=None):
    def loss():
        if counter is not None:
            counter.append(weight.detach().clone())
        value = float((weight - target).square().sum())
        return {"live": value, "history": value, "mixed": value}
    return loss


def equal_tree(test, left, right):
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right), (left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            equal_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class FunctionalTests(unittest.TestCase):
    def test_config_and_attachment_are_passive(self):
        for mode in MODES:
            weight = torch.tensor([[1., 0.], [0., 1.]])
            rng = torch.random.get_rng_state().clone()
            before = weight.clone()
            op = FunctionalOperator(weight, FunctionalConfig(mode=mode))
            self.assertTrue(torch.equal(weight, before))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertTrue(torch.equal(op.mask, weight.ne(0)))
            self.assertEqual(op.summary()["allowed"], 2)
            self.assertEqual(op.factor_start_updates, 0)
        for kwargs in ({"mode": "bad"}, {"window_updates": 0}, {"candidates_per_kind": 0},
                       {"structural_seed": -1}, {"structural_seed": 2**64}):
            with self.assertRaises(ValueError):
                FunctionalConfig(**kwargs)

    def test_full_gradient_and_dormant_moments_are_not_cleared(self):
        weight, op, adam = make(window=8)
        grad = torch.ones_like(weight)
        self.assertIs(op.before_gradient(grad), grad)
        event = advance(weight, op, adam, grad)
        state = adam.state[weight]
        self.assertGreater(float(state["exp_avg"][0, 1]), 0)
        self.assertGreater(float(state["exp_avg_sq"][0, 1]), 0)
        self.assertEqual(float(state["step"]), 1)
        self.assertEqual(float(weight.detach()[0, 1]), 0)
        self.assertEqual(event["births"], 0)

    def test_numerical_zero_does_not_silently_retire_an_eligible_edge(self):
        weight, op, adam = make(window=8)
        adam.zero_grad(set_to_none=True)
        weight.grad = torch.zeros_like(weight)
        adam.step()
        with torch.no_grad():
            weight[0, 0] = 0.
        group = adam.param_groups[0]
        event = op.apply_after_optimizer(weight, adam.state[weight], torch.zeros_like(weight),
            1., group["lr"], group["betas"], group["eps"])
        self.assertTrue(bool(op.allowed_mask[0, 0]))
        self.assertFalse(bool(op.support_mask[0, 0]))
        self.assertTrue(bool(op.survival_mask[0, 0]))
        self.assertEqual(event["deaths"], 0)
        self.assertEqual(event["numeric_support_deaths"], 1)
        restored = FunctionalOperator.from_state(weight, op.state_dict())
        self.assertTrue(torch.equal(restored.allowed_mask, op.allowed_mask))

    def test_float32_restore_at_birth_partial_window_and_pending_boundary(self):
        weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))
        op = FunctionalOperator(weight, FunctionalConfig(window_updates=3))
        adam = torch.optim.Adam([weight], lr=.1)
        for index in range(4):
            restored = FunctionalOperator.from_state(weight, op.state_dict())
            equal_tree(self, restored.state_dict(), op.state_dict())
            if index < 3:
                advance(weight, op, adam, torch.ones_like(weight))

    def test_proposals_are_delayed_and_exactly_four_probes(self):
        weight, op, adam = make(window=8)
        grad = torch.tensor([[0., -1.], [-2., 0.]], dtype=weight.dtype)
        target = torch.zeros_like(weight)
        calls = []
        for index in range(8):
            event = advance(weight, op, adam, grad)
            self.assertIsNone(event["functional_probe"])
            self.assertEqual(op.probe_calls, 0)
        self.assertEqual(op.pending["generated_update"], 8)
        self.assertEqual(op.pending["execute_update"], 9)
        self.assertEqual(op.pending["birth_coordinates"].tolist(), [[1, 0]])
        self.assertEqual(float(weight.detach()[1, 0]), 0.)
        advance(weight, op, adam, grad, closure(weight, target, calls))
        self.assertEqual(len(calls), 4)
        self.assertEqual(op.probe_calls, 4)
        self.assertIsNone(op.pending)
        self.assertIsNone(op.loss_fn)

    def test_zero_taylor_can_hide_necessary_edge_and_is_not_deleted(self):
        weight, op, adam = make(weight=torch.tensor([[1., 0.], [0., 0.]]))
        advance(weight, op, adam)
        self.assertEqual(op.pending["exit_scores"].tolist(), [0.])
        target = weight.detach().clone()
        calls = []
        event = advance(weight, op, adam, loss=closure(weight, target, calls))["functional_probe"]
        self.assertEqual(len(calls), 4)
        self.assertGreater(event["full_ablation_damage"]["mixed"], 0)
        self.assertGreater(event["half_exit_damage"]["mixed"], 0)
        self.assertFalse(event["exit_accepted"])
        self.assertEqual(float(weight.detach()[0, 0]), 1.)
        self.assertTrue(bool(op.allowed_mask[0, 0]))

    def test_incorrect_taylor_proposal_rolls_back_against_actual_loss(self):
        weight, op, adam = make(weight=torch.tensor([[1., 0.], [0., 0.]]))
        # The old gradient proposes lowering the live edge. The next task
        # instead prefers its current post-Adam value, so fading is harmful.
        grad = torch.tensor([[1., -1.], [0., 0.]], dtype=weight.dtype)
        advance(weight, op, adam, grad)
        self.assertLess(float(op.pending["exit_scores"][0]), 0)
        next_target = weight.detach().clone()
        with torch.no_grad():
            next_target[0, 0] -= .1
        event = advance(weight, op, adam, grad, closure(weight, next_target))["functional_probe"]
        self.assertFalse(event["exit_accepted"])
        self.assertGreater(event["full_ablation_damage"]["mixed"], 0)
        self.assertGreater(float(weight.detach()[0, 0]), .7)

    def test_slow_fading_retires_only_after_repeated_success(self):
        weight, op, adam = make(weight=torch.tensor([[1., 0.], [0., 0.]]))
        target = torch.zeros_like(weight)
        advance(weight, op, adam)
        event = advance(weight, op, adam, loss=closure(weight, target))
        self.assertEqual(float(weight.detach()[0, 0]), .5)
        self.assertEqual(event["deaths"], 0)
        self.assertEqual(op.cumulative_deaths, 0)
        for expected in (.25, .125, .0625, 0.):
            event = advance(weight, op, adam, loss=closure(weight, target))
            self.assertEqual(float(weight.detach()[0, 0]), expected)
        self.assertEqual(event["deaths"], 1)
        self.assertFalse(bool(op.allowed_mask[0, 0]))
        self.assertEqual(op.cumulative_births, 0)
        self.assertEqual(op.cumulative_deaths, 1)
        self.assertEqual(op.last_retire_update[0, 0], op.updates)

    def test_birth_and_exit_acceptance_are_independent(self):
        weight, op, adam = make()
        grad = torch.tensor([[0., -1.], [0., 0.]], dtype=weight.dtype)
        advance(weight, op, adam, grad)
        target = weight.detach().clone()
        target[0, 1] = .3
        event = advance(weight, op, adam, grad, closure(weight, target))
        probe = event["functional_probe"]
        self.assertTrue(probe["birth_accepted"])
        self.assertFalse(probe["exit_accepted"])
        self.assertEqual(event["births"], 1)
        self.assertEqual(event["deaths"], 0)
        self.assertEqual(int(op.allowed_mask.sum()), 3)
        self.assertEqual(probe["exit_baseline"], probe["birth_loss"])
        self.assertTrue(bool(op.ever_born[0, 1]))

    def test_fixed_spends_same_probes_but_never_changes_eligibility(self):
        weight, op, adam = make(mode="functional_fixed")
        grad = torch.tensor([[0., -1.], [0., 0.]], dtype=weight.dtype)
        advance(weight, op, adam, grad)
        target = torch.zeros_like(weight)
        target[0, 1] = .5
        calls = []
        event = advance(weight, op, adam, grad, closure(weight, target, calls))
        self.assertEqual(len(calls), 4)
        self.assertFalse(event["functional_probe"]["birth_accepted"])
        self.assertFalse(event["functional_probe"]["exit_accepted"])
        self.assertEqual(float(weight.detach()[0, 1]), 0.)
        self.assertEqual(float(weight.detach()[0, 0]), 1.)
        self.assertTrue(torch.equal(op.allowed_mask, op.initial_mask))

    def test_dummy_empty_birth_and_exit_keep_four_call_budget(self):
        for start in (torch.zeros(2, 2), torch.ones(2, 2)):
            weight, op, adam = make(weight=start)
            advance(weight, op, adam)
            calls = []
            advance(weight, op, adam, loss=closure(weight, torch.zeros_like(weight), calls))
            self.assertEqual(len(calls), 4)

    def test_task_and_magnitude_share_birth_score_but_rank_exit_differently(self):
        results = {}
        for mode in ("functional_task", "functional_magnitude", "functional_fixed"):
            weight, op, adam = make(weight=torch.tensor([[.2, 0.], [0., .8]]), mode=mode)
            grad = torch.tensor([[-1., -.2], [-.1, .01]], dtype=weight.dtype)
            advance(weight, op, adam, grad)
            results[mode] = deepcopy(op.pending)
            state = adam.state[weight]
            corrected = state["exp_avg"] / (1 - .9 ** float(state["step"]))
            variance = state["exp_avg_sq"] / (1 - .999 ** float(state["step"]))
            delta = -.1 * corrected / (variance.sqrt() + 1e-8)
            coords = op.pending["birth_coordinates"]
            self.assertTrue(torch.equal(op.pending["birth_delta"], op._values(delta, coords)))
            expected_score = -op._values(grad, coords) * op.pending["birth_delta"]
            self.assertTrue(torch.equal(op.pending["birth_scores"], expected_score))
        self.assertEqual(results["functional_task"]["exit_coordinates"].tolist(), [[1, 1]])
        self.assertEqual(results["functional_magnitude"]["exit_coordinates"].tolist(), [[0, 0]])
        self.assertEqual(results["functional_fixed"]["exit_coordinates"].tolist(), [[1, 1]])
        self.assertTrue(torch.equal(results["functional_task"]["birth_coordinates"],
                                    results["functional_magnitude"]["birth_coordinates"]))

    def test_retirement_preserves_live_moments_and_global_age(self):
        weight, op, adam = make(weight=torch.tensor([[.25, 0.], [0., 0.]]))
        grad = torch.tensor([[1., 0.], [0., 0.]], dtype=weight.dtype)
        advance(weight, op, adam, grad)
        adam.zero_grad(set_to_none=True)
        weight.grad = grad.clone()
        adam.step()
        expected = deepcopy(adam.state[weight])
        op.loss_fn = closure(weight, torch.zeros_like(weight))
        group = adam.param_groups[0]
        event = op.apply_after_optimizer(weight, adam.state[weight], grad, 1., group["lr"], group["betas"], group["eps"])
        self.assertEqual(event["deaths"], 1)
        self.assertEqual(float(weight.detach()[0, 0]), 0.)
        self.assertGreater(float(adam.state[weight]["exp_avg"][0, 0]), 0.)
        equal_tree(self, expected, adam.state[weight])

    def test_birth_preserves_nonzero_dormant_moments_and_global_age(self):
        weight, op, adam = make()
        grad = torch.tensor([[0., -1.], [0., 0.]], dtype=weight.dtype)
        advance(weight, op, adam, grad)
        adam.zero_grad(set_to_none=True)
        weight.grad = grad.clone()
        adam.step()
        expected = deepcopy(adam.state[weight])
        target = weight.detach().clone()
        target[0, 1] = .3
        op.loss_fn = closure(weight, target)
        group = adam.param_groups[0]
        event = op.apply_after_optimizer(weight, adam.state[weight], grad, 1., group["lr"], group["betas"], group["eps"])
        self.assertEqual(event["births"], 1)
        self.assertLess(float(adam.state[weight]["exp_avg"][0, 1]), 0.)
        equal_tree(self, expected, adam.state[weight])

    def test_structural_events_never_reset_optimizer_state(self):
        for target_value in (0., 1.):
            weight, op, adam = make(weight=torch.tensor([[.04, 0.], [0., 0.]]))
            grad = torch.tensor([[.001, -.001], [0., 0.]], dtype=weight.dtype)
            advance(weight, op, adam, grad)
            adam.zero_grad(set_to_none=True)
            weight.grad = grad.clone()
            adam.step()
            expected = deepcopy(adam.state[weight])
            op.loss_fn = closure(weight, torch.full_like(weight, target_value))
            group = adam.param_groups[0]
            op.apply_after_optimizer(weight, adam.state[weight], grad, 1., group["lr"], group["betas"], group["eps"])
            equal_tree(self, expected, adam.state[weight])

    def test_pending_and_rng_restore_produce_same_next_trials_and_proposals(self):
        for mode in MODES:
            weight, op, adam = make(mode=mode, window=2)
            grad = torch.tensor([[.01, -1.], [-2., .02]], dtype=weight.dtype)
            target = torch.tensor([[.2, .4], [.3, .2]], dtype=weight.dtype)
            for _ in range(2):
                advance(weight, op, adam, grad)
            packed = io.BytesIO()
            torch.save({"weight": weight.detach().clone(), "op": op.state_dict(), "adam": adam.state_dict()}, packed)
            packed.seek(0)
            saved = torch.load(packed, weights_only=False)
            restored_weight = torch.nn.Parameter(saved["weight"])
            restored = FunctionalOperator.from_state(restored_weight, saved["op"])
            restored_adam = torch.optim.Adam([restored_weight], lr=.1)
            restored_adam.load_state_dict(saved["adam"])
            for _ in range(5):
                a = advance(weight, op, adam, grad, closure(weight, target) if op.pending is not None else None)
                b = advance(restored_weight, restored, restored_adam, grad,
                            closure(restored_weight, target) if restored.pending is not None else None)
                equal_tree(self, a, b)
                equal_tree(self, op.state_dict(), restored.state_dict())
                self.assertTrue(torch.equal(weight, restored_weight))
                equal_tree(self, adam.state_dict(), restored_adam.state_dict())

    def test_loss_closure_not_serialized_and_cleared_on_probe_error(self):
        weight, op, adam = make()
        advance(weight, op, adam)
        op.loss_fn = lambda: {"mixed": 0.}
        with self.assertRaises(RuntimeError):
            op.state_dict()
        before = weight.detach().clone()
        with self.assertRaises(ValueError):
            advance(weight, op, adam, loss=lambda: {"mixed": 0.})
        self.assertIsNone(op.loss_fn)
        self.assertTrue(torch.equal(weight, before))
        self.assertIsNotNone(op.pending)

    def test_restore_rejects_corrupt_support_pending_age_and_traits(self):
        weight, op, adam = make(window=2)
        grad = torch.ones_like(weight)
        advance(weight, op, adam, grad)
        advance(weight, op, adam, grad)
        source = op.state_dict()
        for mutate in (
            lambda s: s["allowed_mask"].fill_(False),
            lambda s: s["pending"].update(execute_update=77),
            lambda s: s["traits"]["last_birth_update"].fill_(999),
            lambda s: s.update(probe_calls=4),
            lambda s: s["window_gradient_sum"].fill_(1.),
            lambda s: s["pending"]["birth_coordinates"].fill_(999),
        ):
            state = deepcopy(source)
            mutate(state)
            with self.assertRaises(ValueError):
                FunctionalOperator.from_state(weight, state)


if __name__ == "__main__":
    unittest.main()
