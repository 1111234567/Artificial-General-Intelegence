"""Actual optimizer continuation, continuous h and independent gradient groups."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.run import state_digest
from experiments.learn_remember.gru_train import GRUAgent
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.remaining_time import train


def tiny_stream():
    rng = np.random.default_rng(829)
    ticks, batch = 2 + 3 * 4, 3
    actions = rng.integers(0, 11, size=(ticks, batch, 2), dtype=np.int64)
    actions[rng.random(actions.shape) < .6] = 0
    mask = rng.random(actions.shape) > .3
    target = rng.uniform(-.1, 1., size=actions.shape).astype(np.float32)
    target[~mask] = 0
    return dict(obs=rng.standard_normal((ticks, batch, 56), dtype=np.float32),
        actions=actions, time_target=target, time_mask=mask)


def make(arm="joint", folder=None, stream=None, device="cpu"):
    train.deterministic(55)
    original = GRUAgent(width=7)
    model = RemainingTimeGRUAgent.from_original(original, arm).to(device)
    return train.Learner(model, tiny_stream() if stream is None else stream, arm,
        warm=2, block=4, updates=3, folder=folder)


class TrainingTest(unittest.TestCase):
    def test_warm_has_no_loss_grad_or_optimizer_update(self):
        for arm in train.ARMS:
            learner = make(arm)
            self.assertTrue(learner.hidden.eq(0).all())
            before = state_digest(learner.model.state_dict())
            with torch.no_grad():
                _, expected = learner.model(torch.from_numpy(learner.stream["obs"][:2]), learner.hidden)
            learner.warm_start()
            self.assertTrue(torch.equal(expected, learner.hidden), arm)
            self.assertFalse(learner.hidden.eq(0).all())
            self.assertEqual(before, state_digest(learner.model.state_dict()))
            self.assertFalse(learner.optimizer.state)
            self.assertTrue(all(p.grad is None for p in learner.model.parameters()))
            self.assertEqual(learner.work["warm_loss_rows"], 0)
            self.assertEqual(learner.work["training_updates"], 0)
            self.assertEqual(learner.work["warm_lane_ticks"], 6)

    def test_h_carried_without_reset_or_postupdate_recompute(self):
        for arm in train.ARMS:
            learner = make(arm)
            learner.warm_start()
            for _ in range(3):
                start = learner.cursor
                obs = torch.from_numpy(learner.stream["obs"][start:start + 4])
                with torch.no_grad():
                    _, expected = learner.model(obs, learner.hidden)
                    _, reset = learner.model(obs, torch.zeros_like(learner.hidden))
                self.assertFalse(torch.equal(expected, reset), arm)
                row = learner.update()
                self.assertTrue(torch.equal(expected, learner.hidden), arm)
                self.assertIsNone(learner.hidden.grad_fn)
                self.assertEqual(row["hidden_after_sha256"], state_digest(expected))
            self.assertEqual(learner.cursor, 14)

    def test_real_arm_gradients_frozen_policy_and_exact_row_budgets(self):
        states, warm_states = [], []
        for arm in train.ARMS:
            learner = make(arm)
            states.append(state_digest(learner.model.state_dict()))
            learner.warm_start()
            warm_states.append(state_digest(learner.hidden))
            core, head = learner.initial_core_hash, learner.initial_head_hash
            for index in range(3):
                learner.update()
                if index == 0:
                    actual = state_digest(learner.model.core.state_dict())
                    self.assertEqual(actual == core, arm in ("probe", "time"), arm)
                    core_grads = [p.grad for p in learner.model.core.parameters()]
                    if arm == "probe":
                        self.assertTrue(all(g is None for g in core_grads))
                    elif arm == "time":
                        self.assertTrue(all(g is not None and not bool(g.any()) for g in core_grads))
                    else:
                        self.assertTrue(any(g is not None and bool(g.any()) for g in core_grads))
            self.assertEqual(state_digest(learner.model.core.state_dict()) == core, arm == "probe", arm)
            self.assertEqual(state_digest(learner.model.head.state_dict()) == head, arm in ("probe", "time"), arm)
            expected = train.summarize_rows(learner.stream, 2, 14)
            self.assertTrue(all(learner.work[key] == value for key, value in expected.items()))
            self.assertEqual(learner.work["training_updates"], 3)
            self.assertEqual(learner.work["training_ticks"], 12)
            self.assertEqual(learner.work["training_lane_ticks"], 36)
            self.assertEqual(learner.work["action_rows"], 72)
            self.assertEqual(learner.work["time_gradient_rows"], expected["time_rows"])
            self.assertEqual(learner.work["action_gradient_rows"], 72 if arm in ("action", "joint") else 0)
        self.assertEqual(len(set(states)), 1)
        self.assertEqual(len(set(warm_states)), 1)

    def test_action_policy_adam_independent_of_auxiliary_target_size(self):
        ordinary = make("action")
        extreme_stream = tiny_stream()
        extreme_stream["time_target"][extreme_stream["time_mask"]] *= 1e6
        extreme = make("action", stream=extreme_stream)
        ordinary.warm_start(); extreme.warm_start()
        saw_different_time_norm = False
        for _ in range(3):
            ordinary_row, extreme_row = ordinary.update(), extreme.update()
            saw_different_time_norm |= ordinary_row["preclip_grad_norm"]["time_head"] != extreme_row["preclip_grad_norm"]["time_head"]
            self.assertEqual(ordinary_row["preclip_grad_norm"]["policy"], extreme_row["preclip_grad_norm"]["policy"])
            self.assertEqual(state_digest(ordinary.model.core.state_dict()), state_digest(extreme.model.core.state_dict()))
            self.assertEqual(state_digest(ordinary.model.head.state_dict()), state_digest(extreme.model.head.state_dict()))
            self.assertTrue(torch.equal(ordinary.hidden, extreme.hidden))
            # One Adam owns all trainable parameters, but each policy parameter
            # has identical moments and updates despite the huge auxiliary loss.
            for left, right in zip(ordinary.model.policy_parameters(), extreme.model.policy_parameters()):
                self.assertEqual(state_digest(ordinary.optimizer.state[left]), state_digest(extreme.optimizer.state[right]))
        self.assertTrue(saw_different_time_norm)
        self.assertNotEqual(state_digest(ordinary.model.time_head.state_dict()), state_digest(extreme.model.time_head.state_dict()))

    def test_saved_actual_adam_and_h_resume_exactly_all_arms(self):
        for arm in train.ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "original"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                uninterrupted = make(arm, folder=folder, device=device)
                uninterrupted.warm_start(); uninterrupted.save("step000")
                uninterrupted.update(save_audit=True)
                uninterrupted.update(save_audit=True)
                path = folder / "resume.pt"
                torch.save(uninterrupted.payload(), path)
                cp = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(cp["step"], 2)
                self.assertTrue(cp["optimizer"]["state"])
                self.assertTrue(all(int(state["step"]) == 2 for state in cp["optimizer"]["state"].values()))
                self.assertGreater(sum(float(state["exp_avg"].abs().sum()) for state in cp["optimizer"]["state"].values()), 0.)
                resumed = train.Learner.restore(path, uninterrupted.stream, device=device)
                uninterrupted.update(); resumed.update()
                for key in ("model", "optimizer", "hidden", "stream_cursor", "step", "work"):
                    self.assertEqual(state_digest(uninterrupted.payload()[key]), state_digest(resumed.payload()[key]), (arm, key))
                before = torch.load(folder / "step002_before.pt", map_location="cpu", weights_only=False)
                after = torch.load(folder / "step002_after.pt", map_location="cpu", weights_only=False)
                gradients = torch.load(folder / "step002_gradients.pt", map_location="cpu", weights_only=False)
                self.assertEqual(before["stream_cursor"], 6)
                self.assertEqual(after["stream_cursor"], 10)
                self.assertTrue(torch.equal(gradients["next_hidden"], after["hidden"]))
                self.assertEqual(set(gradients["preclip_grad_norm"]), {"policy", "time_head"})
                self.assertNotEqual(state_digest(before["model"]), state_digest(after["model"]))

    def test_empty_time_mask_uses_zero_auxiliary_loss(self):
        stream = tiny_stream()
        stream["time_mask"][:] = False
        stream["time_target"][:] = 0
        for arm in train.ARMS:
            learner = make(arm, stream=stream)
            learner.warm_start()
            before = state_digest(learner.model.time_head.state_dict())
            row = learner.update()
            self.assertEqual(row["loss"]["time"], 0.)
            self.assertEqual(row["time_rows"], 0)
            self.assertEqual(before, state_digest(learner.model.time_head.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), "declared native runtime is CUDA float32")
    def test_real_original_checkpoints_preserved_at_initialization(self):
        for seed in train.SEEDS:
            cp = torch.load(train.original_checkpoint(seed), map_location="cpu", weights_only=False)
            original = GRUAgent(**cp["model_config"])
            original.load_state_dict(cp["model"])
            original = original.to("cuda")
            stream = train.raw_data.load_stream(seed, keys=("obs",))
            obs = torch.from_numpy(stream["obs"][:8, :3]).to("cuda")
            h = original.initial_state(3)
            with torch.no_grad():
                expected, expected_h = original(obs, h)
                step_expected, step_h = original.step(obs[0], h)
                for arm in train.ARMS:
                    model = train.initialize(seed, arm, device="cuda")
                    actual, actual_h = model(obs, h)
                    logits, time, aux_h = model.forward_with_time(obs, h)
                    step_actual, actual_step_h = model.step(obs[0], h)
                    self.assertTrue(torch.equal(expected, actual), (seed, arm))
                    self.assertTrue(torch.equal(expected, logits), (seed, arm))
                    self.assertTrue(torch.equal(expected_h, actual_h))
                    self.assertTrue(torch.equal(expected_h, aux_h))
                    self.assertTrue(torch.equal(step_expected, step_actual))
                    self.assertTrue(torch.equal(step_h, actual_step_h))
                    self.assertFalse(bool(time.any()))

    def test_real_original_cpu_deployment_function_with_common_flags(self):
        # Native CPU kernels can depend on requires_grad flags.  Actual rollout
        # freezes EVERY loaded policy, including the original reference.
        for seed in train.SEEDS:
            cp = torch.load(train.original_checkpoint(seed), map_location="cpu", weights_only=False)
            original = GRUAgent(**cp["model_config"])
            original.load_state_dict(cp["model"])
            original.eval().requires_grad_(False)
            obs = torch.from_numpy(train.raw_data.load_stream(seed, keys=("obs",))["obs"][:8, :3])
            h = original.initial_state(3)
            with torch.inference_mode():
                expected, expected_h = original(obs, h)
                step_expected, step_h = original.step(obs[0], h)
                for arm in train.ARMS:
                    model = train.initialize(seed, arm, device="cpu").eval().requires_grad_(False)
                    actual, actual_h = model(obs, h)
                    step_actual, actual_step_h = model.step(obs[0], h)
                    self.assertTrue(torch.equal(expected, actual), (seed, arm))
                    self.assertTrue(torch.equal(expected_h, actual_h))
                    self.assertTrue(torch.equal(step_expected, step_actual))
                    self.assertTrue(torch.equal(step_h, actual_step_h))


if __name__ == "__main__":
    unittest.main()
