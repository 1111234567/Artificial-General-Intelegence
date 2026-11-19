"""Verify native policy preservation and the four-arm gradient intervention."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.learn_remember.gru_train import GRUAgent as OriginalGRUAgent
from experiments.lifelong.world import ContextWorld
from experiments.remaining_time.model import (ARMS, RemainingTimeGRUAgent,
    choose, clip_gradients, experiment_loss, native_action_loss,
    remaining_time_targets, time_loss)


ROOT = Path(__file__).resolve().parents[2]


def clone_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class RemainingTimeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_policy_copy_counts_and_arm_trainability(self):
        source = OriginalGRUAgent()
        before = clone_state(source)
        expected_trainable = dict(probe=386, time=144386, action=148632,
                                  joint=148632)
        for arm in ARMS:
            model = RemainingTimeGRUAgent.from_original(source, arm)
            report = model.parameter_report()
            self.assertEqual(report["total_parameters"], 148632)
            self.assertEqual(report["total_parameters"] -
                             sum(p.numel() for p in source.parameters()), 386)
            self.assertEqual(report["time_head_parameters"], 386)
            self.assertEqual(report["trainable_parameters"], expected_trainable[arm])
            for name, original in source.named_parameters():
                copied = dict(model.named_parameters())[name]
                self.assertTrue(torch.equal(original, copied))
                self.assertNotEqual(original.data_ptr(), copied.data_ptr())
            self.assertTrue(all(p.count_nonzero() == 0
                                for p in model.time_head.parameters()))
            self.assertEqual(all(p.requires_grad for p in model.core.parameters()),
                             arm != "probe")
            self.assertEqual(all(p.requires_grad for p in model.head.parameters()),
                             arm in ("action", "joint"))
            restored = RemainingTimeGRUAgent(**model.config())
            restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(all(torch.equal(before[k], v)
                            for k, v in source.state_dict().items()))

    def test_initial_functions_bitwise_on_real_original_checkpoint_sequences(self):
        # These are the actual three sources and teacher histories, not a new
        # random source standing in for preservation of a pretrained policy.
        for seed in (0, 1, 2):
            path = ROOT / f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
            cp = torch.load(path, map_location="cpu", weights_only=False)
            source = OriginalGRUAgent(**cp["model_config"])
            source.load_state_dict(cp["model"], strict=True)
            with np.load(ROOT / f"data/action-split/streams-v1/seed_{seed:03d}/stream.npz",
                         allow_pickle=False) as archive:
                obs = torch.from_numpy(archive["obs"][:640].copy())
            hidden = source.initial_state(obs.shape[1])
            with torch.no_grad():
                _, anchor = source(obs[:512], hidden)
                expected_logits, expected_h = source(obs[512:], anchor)
                expected_step, expected_step_h = source.step(obs[512], anchor, dt=.25)
                for arm in ARMS:
                    model = RemainingTimeGRUAgent.from_original(source, arm)
                    actual_logits, actual_h = model(obs[512:], anchor)
                    logits, time_prediction, with_time_h = model.forward_with_time(
                        obs[512:], anchor)
                    self.assertTrue(torch.equal(actual_logits, expected_logits))
                    self.assertTrue(torch.equal(actual_h, expected_h))
                    self.assertTrue(torch.equal(logits, expected_logits))
                    self.assertTrue(torch.equal(with_time_h, expected_h))
                    self.assertEqual(int(time_prediction.count_nonzero()), 0)
                    self.assertTrue(torch.equal(choose(logits),
                                                expected_logits.argmax(-1)))
                    step, step_h = model.step(obs[512], anchor, dt=.25)
                    self.assertTrue(torch.equal(step, expected_step))
                    self.assertTrue(torch.equal(step_h, expected_step_h))

    def test_native_forward_does_not_even_compute_the_time_head(self):
        model = RemainingTimeGRUAgent(width=8, input_size=4)
        def fail(*args):
            raise AssertionError("native inference computed auxiliary output")
        handle = model.time_head.register_forward_hook(fail)
        try:
            model(torch.randn(5, 2, 4), model.initial_state(2))
            model.step(torch.randn(2, 4), model.initial_state(2), dt=.25)
        finally:
            handle.remove()

    def test_native_decoder_never_filters_illegal_actions(self):
        world = ContextWorld(seed=129, batch_size=1, actor_ids=(90310,), n_jobs=2,
            active_jobs=2, duration_range=(1., 2.), gap_range=(.5, 1.), dt=.25)
        world.observe()
        logits = torch.zeros(1, 2, 11)
        logits[..., 10] = 1.
        self.assertFalse(bool(world.started.any()))
        actions = choose(logits)
        self.assertEqual(actions.tolist(), [[10, 10]])
        world.act(actions)
        self.assertEqual(world.last_result.tolist(), [[-1, -1]])
        self.assertEqual(int(choose(torch.zeros(11))), 0)

    def test_signed_target_mask_and_first_legal_tick_use_physical_units(self):
        ready = torch.tensor([[16., float("inf"), 20., float("nan")],
                              [19.875, 19.875, 19.875, 19.875]],
                             dtype=torch.float64)
        now = torch.tensor([12., 20.], dtype=torch.float64)
        active = torch.tensor([[True, True, False, True],
                               [True, True, True, True]])
        started = torch.tensor([[True, False, True, True],
                                [True, True, True, False]])
        target, mask = remaining_time_targets(ready, now, active, started)
        self.assertEqual(mask.tolist(), [[True, False, False, False],
                                         [True, True, True, False]])
        self.assertEqual(float(target[0, 0]), .25)  # 4 physical units /16
        self.assertEqual(float(target[1, 0]), -.125 / 16)
        self.assertTrue(bool(torch.isfinite(target).all()))
        self.assertEqual(int(target[~mask].count_nonzero()), 0)
        with self.assertRaises(ValueError):
            remaining_time_targets(ready, float("nan"), active, started)
        # No future outcome or eligibility flag can enter this function. All
        # currently active/started finite slots above contribute identically.

    def test_time_mse_normalizer_and_masked_nonfinite_rows(self):
        prediction = torch.tensor([0., 1., float("nan"), 7.], requires_grad=True)
        target = torch.tensor([2., 4., float("inf"), float("nan")])
        mask = torch.tensor([True, True, False, False])
        losses = time_loss(prediction, target, mask)
        self.assertEqual(int(losses["rows"]), 2)
        self.assertEqual(float(losses["mse"].detach()), 6.5)
        losses["total"].backward()
        self.assertEqual(prediction.grad.tolist(), [-2., -3., 0., 0.])
        empty = torch.tensor([float("nan")], requires_grad=True)
        zero = time_loss(empty, empty.detach(), torch.tensor([False]))["total"]
        self.assertEqual(float(zero.detach()), 0.)
        zero.backward()
        self.assertEqual(empty.grad.tolist(), [0.])

    def test_native_action_loss_is_exact_original_weighted_ce(self):
        torch.manual_seed(230)
        logits = torch.randn(7, 3, 2, 11, dtype=torch.float64)
        target = torch.randint(0, 11, (7, 3, 2))
        weights = torch.ones(11, dtype=torch.float64)
        weights[0] = .25
        expected = torch.nn.functional.cross_entropy(logits.flatten(0, 2),
                                                     target.flatten(), weight=weights)
        self.assertTrue(torch.equal(native_action_loss(logits, target), expected))

    def test_probe_updates_preserve_entire_policy_and_time_arm_freezes_action_head(self):
        torch.manual_seed(231)
        source = OriginalGRUAgent(width=8, input_size=4)
        original = clone_state(source)
        obs = torch.randn(9, 3, 4)
        targets = torch.ones(9, 3, 2)
        mask = torch.ones_like(targets, dtype=torch.bool)
        actions = torch.zeros_like(targets, dtype=torch.long)
        for arm in ("probe", "time"):
            model = RemainingTimeGRUAgent.from_original(source, arm)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=.001)
            h = model.initial_state(3)
            for update in range(3):
                optimizer.zero_grad(set_to_none=True)
                logits, prediction, next_h = model.forward_with_time(obs, h)
                experiment_loss(logits, prediction, actions, targets, mask, arm)["total"].backward()
                if arm == "time" and update == 0:
                    self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0
                                        for p in model.core.parameters()))
                self.assertTrue(all(p.grad is None for p in model.head.parameters()))
                clip_gradients(model)
                optimizer.step()
                h = next_h.detach()
            for key, value in model.head.state_dict().items():
                self.assertTrue(torch.equal(value, source.head.state_dict()[key]))
            core_changed = any(not torch.equal(value, source.core.state_dict()[key])
                               for key, value in model.core.state_dict().items())
            self.assertEqual(core_changed, arm == "time")
            self.assertTrue(any(p.count_nonzero() > 0 for p in model.time_head.parameters()))
            if arm == "probe":
                with torch.no_grad():
                    expected, expected_h = source(obs, source.initial_state(3))
                    actual, actual_h = model(obs, model.initial_state(3))
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(expected_h, actual_h))
        self.assertTrue(all(torch.equal(original[key], value)
                            for key, value in source.state_dict().items()))

    def test_detached_auxiliary_loss_leaves_action_gradients_clipping_and_adam_exact(self):
        torch.manual_seed(232)
        source = OriginalGRUAgent(width=8, input_size=4)
        action_only = RemainingTimeGRUAgent.from_original(source, "action")
        with_aux = deepcopy(action_only)
        # A nonzero auxiliary head proves detachment rather than relying on
        # the zero-head first step; huge targets force a very different norm.
        with torch.no_grad():
            action_only.time_head.weight.fill_(1.)
            action_only.time_head.bias.fill_(1.)
            with_aux.load_state_dict(action_only.state_dict())
        optimizers = [torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                      lr=.001) for model in (action_only, with_aux)]
        obs = torch.randn(13, 3, 4)
        actions = torch.randint(0, 11, (13, 3, 2))
        targets = torch.full((13, 3, 2), 100000.)
        mask = torch.ones_like(targets, dtype=torch.bool)
        hs = [model.initial_state(3) for model in (action_only, with_aux)]
        for update in range(2):
            gradients = []
            clipped = []
            next_hs = []
            for index, (model, optimizer) in enumerate(zip((action_only, with_aux), optimizers)):
                optimizer.zero_grad(set_to_none=True)
                logits, prediction, next_h = model.forward_with_time(obs, hs[index])
                objective = native_action_loss(logits, actions)
                if index:
                    objective = objective + time_loss(prediction, targets, mask)["total"]
                objective.backward()
                gradients.append({key: p.grad.clone() for key, p in model.named_parameters()
                                  if key.startswith(("core.", "head."))})
                clip_gradients(model)
                clipped.append({key: p.grad.clone() for key, p in model.named_parameters()
                                if key.startswith(("core.", "head."))})
                optimizer.step()
                next_hs.append(next_h.detach())
            for key in gradients[0]:
                self.assertTrue(torch.equal(gradients[0][key], gradients[1][key]), key)
                self.assertTrue(torch.equal(clipped[0][key], clipped[1][key]), key)
                self.assertTrue(torch.equal(action_only.state_dict()[key],
                                            with_aux.state_dict()[key]), key)
            self.assertTrue(torch.equal(next_hs[0], next_hs[1]))
            hs = next_hs

    def test_joint_time_gradient_reaches_core_after_the_zero_head_update(self):
        torch.manual_seed(233)
        source = OriginalGRUAgent(width=8, input_size=4)
        model = RemainingTimeGRUAgent.from_original(source, "joint")
        with torch.no_grad():
            model.time_head.weight.fill_(1.)
        obs = torch.randn(5, 2, 4)
        _, prediction, _ = model.forward_with_time(obs, model.initial_state(2))
        time_loss(prediction, torch.ones_like(prediction),
                  torch.ones_like(prediction, dtype=torch.bool))["total"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() > 0
                            for p in model.core.parameters()))
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))


if __name__ == "__main__":
    unittest.main()
