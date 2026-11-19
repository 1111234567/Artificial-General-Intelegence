"""Eligibility, inherited lifetime, exact event, and checkpoint controls."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.continuity.run import rng_state, state_digest
from experiments.factors.structure import FactorConfig, FactorOperator, MODES
from experiments.rewiring.plasticity import RewireConfig, RewireOperator


def moments(weight):
    return {"step": torch.tensor(3072.), "exp_avg": torch.full_like(weight, .25),
            "exp_avg_sq": torch.full_like(weight, .3), "max_exp_avg_sq": torch.full_like(weight, .4)}


def apply(operator, weight, gradient=None, state=None, *, lr=.03, betas=(.8, .9)):
    gradient = torch.ones_like(weight) if gradient is None else gradient
    state = moments(weight) if state is None else state
    return operator.apply_after_optimizer(weight, state, gradient, 1., lr, betas, 1e-8)


def inherited_fixture():
    weight = torch.tensor([[.08, 0.], [.5, .5]], dtype=torch.float64)
    operator = RewireOperator(weight, RewireConfig(arm="retire", shrink_step=.01,
                                                window_updates=3, structural_seed=731))
    state = moments(weight)
    for _ in range(2):
        apply(operator, weight, state=state)
    return weight, operator, state


def advance(operator, weight, optimizer, gradient):
    weight.grad = operator.before_gradient(gradient)
    optimizer.step()
    group = optimizer.param_groups[0]
    return operator.apply_after_optimizer(weight, optimizer.state[weight], gradient, 1.,
                                          group["lr"], group["betas"], group["eps"])


class FactorOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_config_and_invalid_inputs(self):
        for values in ({"mode": "other"}, {"idle_lambda": -1}, {"idle_lambda": float("nan")},
                       {"idle_lambda": True}, {"shrink_step": -1}, {"shrink_step": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FactorConfig(**values)
        config = FactorConfig()
        self.assertEqual(config.shrink_step, 2e-4)
        with self.assertRaises(FrozenInstanceError):
            config.mode = "grow"

    def test_attachment_preserves_live_weight_adam_global_rng_and_source_operator(self):
        weight, inherited, adam = inherited_fixture()
        before_weight = weight.clone()
        before_adam, before_source = state_digest(adam), state_digest(inherited.state_dict())
        for mode in MODES:
            with self.subTest(mode=mode):
                before_rng = state_digest(rng_state())
                operator = FactorOperator(weight, FactorConfig(mode=mode, shrink_step=.01),
                                          inherited_operator=inherited)
                self.assertEqual(before_rng, state_digest(rng_state()))
                self.assertTrue(torch.equal(before_weight, weight))
                self.assertEqual(before_adam, state_digest(adam))
                self.assertEqual(before_source, state_digest(inherited.state_dict()))
                current, source = operator.rewire.state_dict(), inherited.state_dict()
                for name in source:
                    if name != "config":
                        self.assertEqual(state_digest(current[name]), state_digest(source[name]))
                self.assertEqual(operator.updates, inherited.updates)
                self.assertEqual(operator.factor_start_updates, inherited.updates)
                self.assertEqual(operator.factor_updates, 0)
                self.assertEqual(operator.updates_in_window, inherited.updates_in_window)
                self.assertIs(operator.mask, operator.allowed_mask)
                self.assertIs(operator.allowed, operator.allowed_mask)
                restored = FactorOperator.from_state(weight, operator.state_dict())
                self.assertEqual(state_digest(operator.state_dict()), state_digest(restored.state_dict()))

    def test_fixed_allowed_zero_regrows_normally_but_retire_zero_stays_dormant(self):
        for mode in ("fixed_shrink", "retire"):
            with self.subTest(mode=mode):
                weight = torch.nn.Parameter(torch.tensor([[.005, 0.], [.5, .5]], dtype=torch.float64))
                old = RewireOperator(weight, RewireConfig(arm="retire", shrink_step=.01, window_updates=8))
                operator = FactorOperator(weight, FactorConfig(mode=mode, shrink_step=.01), inherited_operator=old)
                state = moments(weight)
                first = apply(operator, weight, state=state)
                self.assertEqual((first["deaths"], first["births"]), (1, 0))
                self.assertEqual(float(weight[0, 0].detach()), 0.)
                self.assertEqual(bool(operator.allowed_mask[0, 0]), mode == "fixed_shrink")
                if mode == "fixed_shrink":
                    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        self.assertNotEqual(float(state[name][0, 0]), 0.)
                else:
                    self.assertEqual(float(state["exp_avg"][0, 0]), 0.)
                optimizer = torch.optim.Adam([weight], lr=.1, betas=(.8, .9))
                optimizer.state[weight] = state
                second = advance(operator, weight, optimizer, torch.ones_like(weight))
                self.assertEqual(second["births"], int(mode == "fixed_shrink"))
                self.assertEqual(bool(weight[0, 0].detach().ne(0)), mode == "fixed_shrink")
                self.assertEqual(float(state["step"]), 3073.)

    def test_passive_allowed_zeros_keep_live_moments_and_forbidden_coordinates_clear(self):
        for mode in ("dense", "dense_shrink", "fixed", "fixed_shrink"):
            with self.subTest(mode=mode):
                weight = torch.tensor([[.005, 0.], [.5, .5]], dtype=torch.float64)
                operator = FactorOperator(weight, FactorConfig(mode=mode, shrink_step=.01))
                state = moments(weight)
                original = deepcopy(state)
                event = apply(operator, weight, state=state)
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    self.assertTrue(torch.equal(state[name][operator.allowed_mask], original[name][operator.allowed_mask]))
                    self.assertTrue(bool(state[name][~operator.allowed_mask].eq(0).all()))
                self.assertEqual(float(state["step"]), 3072.)
                self.assertEqual(event["deaths"], int(mode.endswith("shrink")))
                self.assertTrue(bool(weight[~operator.allowed_mask].eq(0).all()))

    def test_dense_natural_adam_birth_is_logged_without_clearing_source_shadow(self):
        weight, inherited, adam = inherited_fixture()
        weight = torch.nn.Parameter(weight)
        operator = FactorOperator(weight, FactorConfig(mode="dense"), inherited_operator=inherited)
        optimizer = torch.optim.Adam([weight], lr=.03, betas=(.8, .9), amsgrad=True)
        optimizer.state[weight] = adam
        source = deepcopy(operator.state_dict()["inherited_operator"])
        event = advance(operator, weight, optimizer, torch.ones_like(weight))
        self.assertEqual(event["births"], 1)
        self.assertEqual(event["factor_cumulative_births"], 1)
        self.assertTrue(bool(operator.support_mask.all()))
        for name in (*RewireOperator._FLOAT_BUFFERS, *RewireOperator._COUNT_BUFFERS):
            self.assertTrue(torch.equal(getattr(operator.rewire, name).cpu(), source[name]))
        self.assertGreater(float(optimizer.state[weight]["exp_avg"][0, 1]), 0.)
        # This active coordinate still holds frozen, unused source shadow state.
        self.assertGreater(int(operator.rewire.shadow_count[0, 1]), 0)
        restored = FactorOperator.from_state(weight, operator.state_dict())
        self.assertEqual(state_digest(operator.state_dict()), state_digest(restored.state_dict()))

    def test_dense_gradient_passthrough_and_fixed_eligibility_mask(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                weight, inherited, _ = inherited_fixture()
                operator = FactorOperator(weight, FactorConfig(mode=mode), inherited_operator=inherited)
                raw = torch.arange(4, dtype=weight.dtype).reshape_as(weight)
                before = raw.clone()
                masked = operator.before_gradient(raw)
                self.assertTrue(torch.equal(raw, before))
                if mode.startswith("dense"):
                    self.assertIs(masked, raw)
                else:
                    self.assertTrue(torch.equal(masked, raw * operator.allowed_mask))

    def test_source_retirement_history_and_fixed_source_support_remain_distinct(self):
        weight = torch.tensor([[.005, .5], [.5, .5]], dtype=torch.float64)
        source = RewireOperator(weight, RewireConfig(arm="retire", shrink_step=.01, window_updates=3))
        apply(source, weight)
        self.assertEqual((int(source.initial_mask.sum()), int(source.survival_mask.sum())), (4, 3))
        for mode in MODES:
            with self.subTest(mode=mode):
                factor = FactorOperator(weight, FactorConfig(mode=mode, shrink_step=.01), inherited_operator=source)
                self.assertEqual(factor.summary()["initial_survivors"], 3)
                self.assertEqual(factor.summary()["cumulative_deaths"], 1)
                self.assertEqual(factor.summary()["factor_cumulative_deaths"], 0)
                self.assertEqual(factor.summary()["factor_initial_active"], 3)
                self.assertTrue(torch.equal(factor.rewire.initial_mask, source.initial_mask))
                self.assertTrue(torch.equal(factor.rewire.survival_mask, source.survival_mask))
                if mode.startswith("fixed"):
                    self.assertFalse(bool(factor.allowed_mask[0, 0]))
                    self.assertTrue(torch.equal(factor.allowed_mask, weight.ne(0)))
                resumed = FactorOperator.from_state(weight, factor.state_dict())
                self.assertEqual(state_digest(factor.state_dict()), state_digest(resumed.state_dict()))

    def test_retire_and_grow_match_original_operator_continuation_exactly(self):
        for mode in ("retire", "grow"):
            with self.subTest(mode=mode):
                source_weight, inherited, adam = inherited_fixture()
                left, right = (torch.nn.Parameter(source_weight.clone()) for _ in range(2))
                factor = FactorOperator(left, FactorConfig(mode=mode, shrink_step=.01), inherited_operator=inherited)
                expected = RewireOperator.from_state(right, inherited.state_dict())
                expected.config = replace(expected.config, arm="learned" if mode == "grow" else "retire")
                left_optim = torch.optim.Adam([left], lr=.03, betas=(.8, .9), amsgrad=True)
                right_optim = torch.optim.Adam([right], lr=.03, betas=(.8, .9), amsgrad=True)
                left_optim.state[left], right_optim.state[right] = deepcopy(adam), deepcopy(adam)
                for tick in range(7):
                    raw = source_weight.new_tensor([[.5, 1.], [-1., .2]]) * (1 if tick % 2 else -1)
                    actual = advance(factor, left, left_optim, raw)
                    event = advance(expected, right, right_optim, raw)
                    for name in event:
                        self.assertEqual(actual[name], event[name], name)
                    self.assertTrue(torch.equal(left, right))
                    self.assertEqual(state_digest(left_optim.state_dict()), state_digest(right_optim.state_dict()))
                    self.assertEqual(state_digest(factor.rewire.state_dict()), state_digest(expected.state_dict()))
                self.assertEqual(factor.factor_updates, 7)
                self.assertEqual(factor.updates, inherited.updates + 7)
                self.assertTrue(torch.equal(factor.rewire.initial_mask, inherited.initial_mask))

    def test_all_modes_disk_checkpoint_preserves_next_update_and_partial_window(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                source_weight, inherited, adam = inherited_fixture()
                weight = torch.nn.Parameter(source_weight.clone())
                operator = FactorOperator(weight, FactorConfig(mode=mode, idle_lambda=.1, shrink_step=.01),
                                          inherited_operator=inherited)
                optimizer = torch.optim.Adam([weight], lr=.03, betas=(.8, .9), amsgrad=True)
                optimizer.state[weight] = deepcopy(adam)
                for _ in range(2):
                    advance(operator, weight, optimizer, torch.ones_like(weight))
                self.assertEqual(operator.updates_in_window, 1)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "factor.pt"
                    torch.save({"weight": weight.detach(), "adam": optimizer.state_dict(),
                                "operator": operator.state_dict()}, path)
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                before_rng = state_digest(rng_state())
                restored_weight = torch.nn.Parameter(saved["weight"])
                restored = FactorOperator.from_state(restored_weight, saved["operator"])
                self.assertEqual(before_rng, state_digest(rng_state()))
                restored_optim = torch.optim.Adam([restored_weight])
                restored_optim.load_state_dict(saved["adam"])
                self.assertEqual(state_digest(operator.state_dict()), state_digest(restored.state_dict()))
                for sign in (-1., 1., -1.):
                    raw = torch.full_like(weight, sign)
                    expected = advance(operator, weight, optimizer, raw)
                    actual = advance(restored, restored_weight, restored_optim, raw)
                    self.assertEqual(expected, actual)
                    self.assertTrue(torch.equal(weight, restored_weight))
                    self.assertEqual(state_digest(optimizer.state_dict()), state_digest(restored_optim.state_dict()))
                    self.assertEqual(state_digest(operator.state_dict()), state_digest(restored.state_dict()))
                saved["operator"]["allowed_mask"].zero_()
                self.assertTrue(bool(restored.allowed_mask.any()))

    def test_passive_resume_rejects_changed_frozen_shadow_eligibility_and_missing_history(self):
        for mode in ("dense", "fixed_shrink"):
            weight, inherited, _ = inherited_fixture()
            operator = FactorOperator(weight, FactorConfig(mode=mode, shrink_step=.01), inherited_operator=inherited)
            apply(operator, weight)
            for case in ("shadow", "eligibility", "start", "survival", "missing", "support"):
                with self.subTest(mode=mode, case=case):
                    broken = operator.state_dict()
                    if case == "shadow":
                        broken["operator"]["shadow_m"][0, 1] += 1
                    elif case == "eligibility":
                        broken["allowed_mask"][0, 0] = ~broken["allowed_mask"][0, 0]
                    elif case == "start":
                        broken["factor_start_updates"] += 1
                    elif case == "survival":
                        broken["factor_survival_mask"][0, 1] = True
                    elif case == "missing":
                        del broken["inherited_operator"]["shadow_count"]
                    else:
                        broken["operator"]["mask"][0, 0] = ~broken["operator"]["mask"][0, 0]
                    with self.assertRaises(ValueError):
                        FactorOperator.from_state(weight, broken)

    def test_invalid_runtime_is_rejected_before_mutation(self):
        weight, inherited, adam = inherited_fixture()
        operator = FactorOperator(weight, FactorConfig(mode="fixed_shrink"), inherited_operator=inherited)
        before = state_digest({"operator": operator.state_dict(), "weight": weight, "adam": adam})
        for raw in (torch.zeros(3), torch.full_like(weight, float("nan"))):
            with self.assertRaises((ValueError, FloatingPointError)):
                apply(operator, weight, raw, adam)
        self.assertEqual(before, state_digest({"operator": operator.state_dict(), "weight": weight, "adam": adam}))


if __name__ == "__main__":
    unittest.main()
