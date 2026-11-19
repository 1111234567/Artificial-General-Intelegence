"""Small exact-support, optimizer-lifetime, proposal and resume checks."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.continuity.model import AgentModel
from experiments.continuity.run import rng_state, state_digest
from experiments.rewiring.plasticity import ARMS, RewireConfig, RewireOperator


def moments(weight):
    return {"step": torch.tensor(11.), "exp_avg": torch.full_like(weight, .25),
            "exp_avg_sq": torch.full_like(weight, .3), "max_exp_avg_sq": torch.full_like(weight, .4)}


def apply(operator, weight, gradient=None, state=None, *, lr=.15, clip_scale=1., betas=(0., 0.), eps=1e-12):
    if gradient is None:
        gradient = torch.ones_like(weight)
    if state is None:
        state = moments(weight)
    return operator.apply_after_optimizer(weight, state, gradient, clip_scale, lr, betas, eps)


class RewireOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_attachment_preserves_weight_parameters_moments_and_global_rng(self):
        weight = torch.nn.Parameter(torch.tensor([[.3, -.2], [.1, .5]], dtype=torch.float64))
        state = moments(weight)
        reference, original_state = weight.detach().clone(), deepcopy(state)
        for arm in ARMS:
            before = state_digest(rng_state())
            operator = RewireOperator(weight, RewireConfig(arm=arm, shrink_step=.01, structural_seed=728))
            self.assertEqual(before, state_digest(rng_state()))
            self.assertTrue(torch.equal(weight, reference))
            self.assertEqual(state_digest(state), state_digest(original_state))
            self.assertTrue(torch.equal(operator.mask, weight.ne(0)))
            self.assertEqual(operator.summary()["active"], weight.numel())

    def test_dense_is_exact_gradient_weight_and_optimizer_passthrough(self):
        weight = torch.tensor([[0., -.2], [.1, .5]], dtype=torch.float64)
        state = moments(weight)
        operator = RewireOperator(weight, RewireConfig(arm="dense", shrink_step=100., window_updates=2))
        gradient = torch.arange(4, dtype=weight.dtype).reshape_as(weight)
        self.assertIs(operator.before_gradient(gradient), gradient)
        before_weight, before_state = weight.clone(), state_digest(state)
        before_rng = state_digest(rng_state())
        for _ in range(3):
            event = apply(operator, weight, gradient, state)
            self.assertEqual(event["births"], 0)
            self.assertEqual(event["deaths"], 0)
        self.assertTrue(torch.equal(weight, before_weight))
        self.assertEqual(before_state, state_digest(state))
        self.assertEqual(before_rng, state_digest(rng_state()))

    def test_only_exact_shrink_zeros_retire_and_same_step_evidence_is_excluded(self):
        weight = torch.tensor([[.05, -.5], [.5, .5]], dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="learned", shrink_step=.1, window_updates=1))
        state = moments(weight)
        active_before = deepcopy(state)
        event = apply(operator, weight, torch.full_like(weight, 100.), state)
        self.assertEqual((event["births"], event["deaths"], event["active"]), (0, 1, 3))
        self.assertEqual(event["nonzero_hard_cuts"], 0)
        self.assertEqual(float(weight[0, 0]), 0.)
        self.assertTrue(torch.equal(weight[operator.mask], torch.tensor([-.4, .4, .4], dtype=weight.dtype)))
        self.assertEqual(int(operator.shadow_count[0, 0]), 0)
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            self.assertEqual(float(state[name][0, 0]), 0.)
            self.assertTrue(torch.equal(state[name][operator.mask], active_before[name][operator.mask]))
        self.assertTrue(torch.equal(state["step"], active_before["step"]))

    def test_independent_birth_without_death_uses_actual_midwindow_evidence_count(self):
        weight = torch.tensor([[.05, .8], [.8, .8]], dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="learned", shrink_step=.1, window_updates=3))
        first = apply(operator, weight)
        self.assertEqual((first["births"], first["deaths"]), (0, 1))
        second = apply(operator, weight)
        self.assertEqual(int(operator.proposal_count[0, 0]), 1)
        self.assertEqual((second["births"], second["deaths"]), (0, 0))
        third = apply(operator, weight)
        self.assertEqual((third["births"], third["deaths"]), (1, 0))
        # Two dormant observations give -0.30 of shadow updates and 0.20 of
        # shrink. Charging the complete three-update window prevents birth.
        self.assertAlmostEqual(float(weight[0, 0]), -.1, places=10)
        self.assertEqual((operator.cumulative_births, operator.cumulative_deaths), (1, 1))
        self.assertEqual(operator.summary()["initial_survivors"], 3)
        self.assertEqual(operator.summary()["active"], 4)
        self.assertEqual(int(operator.shadow_count[0, 0]), 0)

    def test_shadow_adam_lifetime_survives_windows_while_proposal_counts_reset(self):
        weight = torch.eye(2, dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="retire", shrink_step=.01, window_updates=2))
        gradient = torch.tensor([[0., 2.], [-3., 0.]], dtype=weight.dtype)
        apply(operator, weight, gradient, lr=.03, clip_scale=.25, betas=(.8, .9))
        expected_m, expected_v = .2 * .5, .1 * .5 ** 2
        self.assertAlmostEqual(float(operator.shadow_m[0, 1]), expected_m)
        self.assertAlmostEqual(float(operator.shadow_v[0, 1]), expected_v)
        self.assertAlmostEqual(float(operator.proposal_sum[0, 1]), -.03, places=10)
        event = apply(operator, weight, gradient, lr=.03, clip_scale=.25, betas=(.8, .9))
        self.assertTrue(event["window"])
        self.assertEqual(event["births"], 0)
        self.assertEqual(event["proposal_candidates"], 2)
        self.assertEqual(int(operator.shadow_count[0, 1]), 2)
        self.assertEqual(int(operator.proposal_count.sum()), 0)
        self.assertEqual(float(operator.proposal_sum.abs().sum()), 0.)
        apply(operator, weight, gradient, lr=.03, clip_scale=.25, betas=(.8, .9))
        self.assertEqual(int(operator.shadow_count[0, 1]), 3)
        self.assertEqual(int(operator.proposal_count[0, 1]), 1)

    def test_dormant_adam_momentum_cannot_resurrect_or_contaminate_active_moments(self):
        weight = torch.nn.Parameter(torch.tensor([[0., .8], [.8, .8]], dtype=torch.float64))
        optimizer = torch.optim.Adam([weight], lr=.01)
        optimizer.state[weight] = {"step": torch.tensor(11.), "exp_avg": torch.full_like(weight, .4),
                                   "exp_avg_sq": torch.full_like(weight, .5)}
        operator = RewireOperator(weight, RewireConfig(arm="learned", shrink_step=.01, window_updates=8))
        raw = torch.ones_like(weight)
        masked = operator.before_gradient(raw)
        self.assertEqual(float(masked[0, 0]), 0.)
        self.assertEqual(float(raw[0, 0]), 1.)
        weight.grad = masked
        optimizer.step()
        self.assertNotEqual(float(weight[0, 0].detach()), 0.)
        state = optimizer.state[weight]
        active_moments = {name: value[operator.mask].clone() for name, value in state.items() if name != "step"}
        apply(operator, weight, raw, state, lr=.01, betas=(.9, .999), eps=1e-8)
        self.assertEqual(float(weight[0, 0].detach()), 0.)
        self.assertEqual(float(state["step"]), 12.)
        for name in active_moments:
            self.assertEqual(float(state[name][0, 0]), 0.)
            self.assertTrue(torch.equal(state[name][operator.mask], active_moments[name]))

    def test_random_births_match_own_informed_signed_values_and_per_row_counts(self):
        initial = torch.eye(5, dtype=torch.float64)
        gradient = torch.zeros_like(initial)
        gradient[:, 0] = 1.
        gradient[:, 4] = -2.
        learned, random = initial.clone(), initial.clone()
        informed = RewireOperator(learned, RewireConfig(arm="learned", shrink_step=.01, window_updates=1))
        shuffled = RewireOperator(random, RewireConfig(arm="random", shrink_step=.01, window_updates=1, structural_seed=179))
        before_rng = state_digest(rng_state())
        left = apply(informed, learned, gradient, lr=.1)
        right = apply(shuffled, random, gradient, lr=.1)
        self.assertEqual(before_rng, state_digest(rng_state()))
        self.assertEqual(left["births_per_row"], right["births_per_row"])
        self.assertEqual(left["births"], right["births"])
        for row in range(len(initial)):
            self.assertTrue(torch.equal(learned[row][informed.ever_born[row]].sort().values,
                                        random[row][shuffled.ever_born[row]].sort().values))
        self.assertFalse(torch.equal(informed.ever_born, shuffled.ever_born))

    def test_direct_zero_coordinate_bptt_derivative_is_real_and_only_optimizer_is_masked(self):
        model = AgentModel(input_size=2, n_jobs=1, n_actions=2, width=2).double()
        with torch.no_grad():
            model.recurrent.copy_(torch.tensor([[.3, 0.], [.1, -.2]], dtype=torch.float64))
        operator = RewireOperator(model.recurrent, RewireConfig(arm="learned"))
        observation = torch.tensor([[.2, -.1]], dtype=torch.float64)
        initial = torch.tensor([[.4, -.3]], dtype=torch.float64)

        def objective():
            hidden = initial
            for _ in range(3):
                output, hidden = model.step(observation, hidden, .25)
            return output.square().sum() + .2 * hidden.square().sum()

        gradient = torch.autograd.grad(objective(), model.recurrent)[0]
        eps = 1e-6
        with torch.no_grad():
            model.recurrent[0, 1] = eps
            plus = float(objective())
            model.recurrent[0, 1] = -eps
            minus = float(objective())
            model.recurrent[0, 1] = 0.
        self.assertGreater(abs(float(gradient[0, 1])), 1e-8)
        self.assertAlmostEqual(float(gradient[0, 1]), (plus - minus) / (2 * eps), places=7)
        self.assertEqual(float(operator.before_gradient(gradient)[0, 1]), 0.)

    def _assert_real_gradient_optimizer_and_saved_continuation(self, device):
        model = AgentModel(input_size=2, n_jobs=1, n_actions=2, width=2).to(device=device, dtype=torch.float64)
        with torch.no_grad():
            model.recurrent.copy_(torch.tensor([[.3, 0.], [.1, -.2]], device=device, dtype=torch.float64))
            model.encoder.weight.copy_(torch.tensor([[.4, .1], [-.2, .3]], device=device, dtype=torch.float64))
            model.head.weight.copy_(torch.tensor([[.5, -.3], [-.4, .2]], device=device, dtype=torch.float64))
            model.head.bias.copy_(torch.tensor([.1, -.05], device=device, dtype=torch.float64))
        weight = model.recurrent
        operator = RewireOperator(weight, RewireConfig(arm="learned", shrink_step=.001, window_updates=2))
        optimizer = torch.optim.Adam([weight], lr=.03, betas=(.8, .9), eps=1e-8)
        observation = weight.new_tensor([[.2, -.1]])
        initial = weight.new_tensor([[.4, -.3]])

        def objective():
            hidden = initial
            for _ in range(3):
                output, hidden = model.step(observation, hidden, .25)
            return output.square().sum() + .2 * hidden.square().sum()

        gradient = torch.autograd.grad(objective(), weight)[0]
        eps = 1e-6
        with torch.no_grad():
            weight[0, 1] = eps
            plus = float(objective())
            weight[0, 1] = -eps
            minus = float(objective())
            weight[0, 1] = 0.
        self.assertGreater(abs(float(gradient[0, 1])), 1e-8)
        self.assertAlmostEqual(float(gradient[0, 1]), (plus - minus) / (2 * eps), places=7)

        def advance(op, live_weight, live_optimizer, raw):
            live_weight.grad = op.before_gradient(raw)
            norm = torch.nn.utils.clip_grad_norm_([live_weight], .01)
            clip_scale = min(1., .01 / (float(norm) + 1e-6))
            live_optimizer.step()
            group = live_optimizer.param_groups[0]
            event = op.apply_after_optimizer(live_weight, live_optimizer.state[live_weight], raw,
                                            clip_scale, group["lr"], group["betas"], group["eps"])
            return event, clip_scale

        raw_copy = gradient.clone()
        first, clip_scale = advance(operator, weight, optimizer, gradient)
        self.assertLess(clip_scale, 1.)
        self.assertEqual((first["births"], first["deaths"]), (0, 0))
        self.assertTrue(torch.equal(gradient, raw_copy))
        self.assertEqual(float(weight[0, 1].detach()), 0.)
        self.assertAlmostEqual(float(operator.shadow_m[0, 1]), .2 * float(gradient[0, 1]) * clip_scale)
        self.assertAlmostEqual(float(operator.shadow_v[0, 1]), .1 * (float(gradient[0, 1]) * clip_scale) ** 2)
        for name in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(float(optimizer.state[weight][name][0, 1]), 0.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "operator.pt"
            torch.save({"weight": weight.detach(), "adam": optimizer.state_dict(),
                        "operator": operator.state_dict()}, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        resumed_weight = torch.nn.Parameter(saved["weight"].to(device))
        resumed = RewireOperator.from_state(resumed_weight, saved["operator"])
        resumed_optimizer = torch.optim.Adam([resumed_weight])
        resumed_optimizer.load_state_dict(saved["adam"])
        self.assertEqual(state_digest(operator.state_dict()), state_digest(resumed.state_dict()))
        following_gradient = torch.autograd.grad(objective(), weight)[0]
        expected, _ = advance(operator, weight, optimizer, following_gradient)
        actual, _ = advance(resumed, resumed_weight, resumed_optimizer, following_gradient)
        self.assertEqual(expected, actual)
        self.assertEqual(expected["births"], 1)
        self.assertTrue(torch.equal(weight, resumed_weight))
        self.assertEqual(state_digest(operator.state_dict()), state_digest(resumed.state_dict()))
        self.assertEqual(state_digest(optimizer.state_dict()), state_digest(resumed_optimizer.state_dict()))

    def test_cpu_real_bptt_gradient_adam_shrink_birth_and_disk_continuation(self):
        self._assert_real_gradient_optimizer_and_saved_continuation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_real_bptt_gradient_adam_shrink_birth_and_disk_continuation(self):
        self._assert_real_gradient_optimizer_and_saved_continuation("cuda")

    def test_full_state_roundtrip_preserves_partial_window_private_rng_and_future_events(self):
        weight = torch.eye(4, dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="random", shrink_step=.01, window_updates=3,
                                                     structural_seed=348))
        state = moments(weight)

        def advance(op, w, adam, tick):
            g = torch.zeros_like(w)
            g[:, tick % 4] = .5 if tick % 2 else -1.
            return apply(op, w, g, adam, lr=.04, betas=(.8, .9))

        for tick in range(4):
            advance(operator, weight, state, tick)
        saved = operator.state_dict()
        saved_weight, saved_adam = weight.clone(), deepcopy(state)
        self.assertEqual(operator.updates_in_window, 1)
        self.assertGreater(int(operator.proposal_count.sum()), 0)
        expected_events = [advance(operator, weight, state, tick) for tick in range(4, 10)]
        before_rng = state_digest(rng_state())
        resumed = RewireOperator.from_state(saved_weight, saved)
        self.assertEqual(before_rng, state_digest(rng_state()))
        self.assertEqual(state_digest(saved), state_digest(resumed.state_dict()))
        torch.rand(7)
        actual_events = [advance(resumed, saved_weight, saved_adam, tick) for tick in range(4, 10)]
        self.assertEqual(expected_events, actual_events)
        self.assertTrue(torch.equal(weight, saved_weight))
        self.assertEqual(state_digest(state), state_digest(saved_adam))
        self.assertEqual(state_digest(operator.state_dict()), state_digest(resumed.state_dict()))
        saved["mask"].zero_()
        self.assertFalse(torch.equal(saved["mask"], resumed.mask))

    def test_invalid_configs_inputs_and_corrupted_states_are_rejected(self):
        for values in ({"arm": "other"}, {"shrink_step": -1.}, {"shrink_step": float("nan")},
                       {"window_updates": 0}, {"window_updates": True}, {"structural_seed": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RewireConfig(**values)
        config = RewireConfig()
        with self.assertRaises(FrozenInstanceError):
            config.arm = "random"
        weight = torch.eye(2, dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="learned"))
        state = moments(weight)
        before_weight, before_adam, before_op = weight.clone(), state_digest(state), state_digest(operator.state_dict())
        for gradient in (torch.zeros(3), torch.full_like(weight, float("nan"))):
            with self.assertRaises((ValueError, FloatingPointError)):
                apply(operator, weight, gradient, state)
        with self.assertRaises(ValueError):
            apply(operator, weight, state=state, clip_scale=1.1)
        self.assertTrue(torch.equal(weight, before_weight))
        self.assertEqual(before_adam, state_digest(state))
        self.assertEqual(before_op, state_digest(operator.state_dict()))
        for field in ("mask", "shadow_v", "proposal_count", "cumulative_births", "generator_state"):
            with self.subTest(field=field):
                broken = operator.state_dict()
                if field == "mask":
                    broken[field][0, 0] = False
                elif field == "shadow_v":
                    broken[field][0, 1] = float("nan")
                elif field == "proposal_count":
                    broken[field][0, 1] = 1
                elif field == "cumulative_births":
                    broken[field] = 2
                else:
                    broken[field] = torch.tensor([1], dtype=torch.uint8)
                with self.assertRaises(ValueError):
                    RewireOperator.from_state(weight, broken)

    def test_incomplete_schema_and_impossible_lifetime_histories_are_rejected(self):
        weight = torch.eye(3, dtype=torch.float64)
        operator = RewireOperator(weight, RewireConfig(arm="learned", shrink_step=.01, window_updates=3))
        apply(operator, weight, torch.zeros_like(weight))
        saved = operator.state_dict()
        self.assertEqual(state_digest(saved), state_digest(RewireOperator.from_state(weight, saved).state_dict()))

        def corrupt(state, case):
            if case == "missing_buffer":
                del state["shadow_m"]
            elif case == "extra_buffer":
                state["unknown"] = torch.tensor(0)
            elif case == "missing_config":
                del state["config"]["window_updates"]
            elif case == "unobserved_dormant_update":
                state["proposal_count"][0, 1] = 0
            elif case == "birth_without_event":
                state["ever_born"][0, 1] = True
            elif case == "lost_survivor_without_birth":
                state["survival_mask"][0, 0] = False
            elif case == "too_many_events":
                state["births_per_row"][0] = state["deaths_per_row"][0] = 4
                state["cumulative_births"] = state["cumulative_deaths"] = 4
            elif case == "row_peak_without_birth":
                state["peak_per_row"][0] = 2
            elif case == "row_min_without_death":
                state["min_per_row"][0] = 0
            elif case == "total_peak_exceeds_row_peaks":
                state["peak_active"] = 4
            elif case == "total_min_below_row_minima":
                state["min_active"] = 2
            elif case == "unused_rng_advanced":
                state["generator_state"] = torch.Generator().manual_seed(937).get_state()

        for case in ("missing_buffer", "extra_buffer", "missing_config", "unobserved_dormant_update",
                     "birth_without_event", "lost_survivor_without_birth", "too_many_events",
                     "row_peak_without_birth", "row_min_without_death", "total_peak_exceeds_row_peaks",
                     "total_min_below_row_minima", "unused_rng_advanced"):
            with self.subTest(case=case):
                broken = deepcopy(saved)
                corrupt(broken, case)
                with self.assertRaises(ValueError):
                    RewireOperator.from_state(weight, broken)

    def test_dense_shadow_observations_and_retire_births_are_rejected(self):
        weight = torch.eye(3, dtype=torch.float64)
        dense = RewireOperator(weight, RewireConfig(arm="dense", window_updates=3))
        apply(dense, weight)
        broken = dense.state_dict()
        broken["shadow_count"][0, 1] = broken["proposal_count"][0, 1] = 1
        with self.assertRaises(ValueError):
            RewireOperator.from_state(weight, broken)
        retire = RewireOperator(weight, RewireConfig(arm="retire", window_updates=3))
        apply(retire, weight, torch.zeros_like(weight))
        broken = retire.state_dict()
        broken["ever_born"][0, 1] = True
        broken["births_per_row"][0] = broken["deaths_per_row"][0] = 1
        broken["cumulative_births"] = broken["cumulative_deaths"] = 1
        with self.assertRaises(ValueError):
            RewireOperator.from_state(weight, broken)


if __name__ == "__main__":
    unittest.main()
