"""Verify real optimizer continuation, numerical h carry, and exact row budgets."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.action_split.model import SplitGRUAgent
from experiments.action_split import train
from experiments.continuity.run import state_digest


def tiny_stream():
    rng = np.random.default_rng(582)
    ticks, batch = 2 + 3 * 4, 3
    mode = rng.integers(0, 3, size=(ticks, batch, 2), dtype=np.int64)
    active = rng.random(mode.shape) > .3
    active[mode != 0] = True
    begin = rng.integers(0, 2, size=mode.shape, dtype=np.int64)
    finish = rng.integers(0, 8, size=mode.shape, dtype=np.int64)
    begin[~active], finish[~active] = -1, -1
    return dict(obs=rng.standard_normal((ticks, batch, 56), dtype=np.float32),
        mode_target=mode, begin_target=begin, finish_target=finish, active=active)


def make(arm="full", folder=None):
    train.deterministic(5)
    model = SplitGRUAgent(width=7)
    return train.Learner(model, tiny_stream(), arm, warm=2, block=4, updates=3, folder=folder)


class TrainingTest(unittest.TestCase):
    def test_warm_has_no_loss_or_optimizer_update(self):
        learner = make()
        self.assertTrue(learner.hidden.eq(0).all())
        before = state_digest(learner.model.state_dict())
        with torch.no_grad():
            _, expected = learner.model(torch.from_numpy(learner.stream["obs"][:2]), learner.hidden)
        learner.warm_start()
        self.assertTrue(torch.equal(expected, learner.hidden))
        self.assertFalse(learner.hidden.eq(0).all())
        self.assertEqual(before, state_digest(learner.model.state_dict()))
        self.assertFalse(learner.optimizer.state)
        self.assertTrue(all(p.grad is None for p in learner.model.parameters()))
        self.assertEqual(learner.work["warm_loss_rows"], 0)
        self.assertEqual(learner.work["training_updates"], 0)
        self.assertEqual(learner.work["warm_lane_ticks"], 6)

    def test_numerical_hidden_carried_without_reset_or_postupdate_recompute(self):
        learner = make()
        learner.warm_start()
        for _ in range(3):
            start = learner.cursor
            obs = torch.from_numpy(learner.stream["obs"][start:start + 4])
            with torch.no_grad():
                _, expected = learner.model(obs, learner.hidden)
                _, reset = learner.model(obs, torch.zeros_like(learner.hidden))
            self.assertFalse(torch.equal(expected, reset))
            row = learner.update()
            self.assertTrue(torch.equal(expected, learner.hidden))
            self.assertIsNone(learner.hidden.grad_fn)
            self.assertEqual(row["hidden_after_sha256"], state_digest(expected))
        self.assertEqual(learner.cursor, 14)

    def test_frozen_exact_full_changes_and_paired_initial_states(self):
        frozen, full = make("frozen"), make("full")
        self.assertEqual(state_digest(frozen.model.state_dict()), state_digest(full.model.state_dict()))
        frozen.warm_start(); full.warm_start()
        self.assertTrue(torch.equal(frozen.hidden, full.hidden))
        initial_core = state_digest(full.model.core.state_dict())
        for index in range(3):
            frozen.update(); full.update()
            if index == 0:
                self.assertEqual(initial_core, state_digest(full.model.core.state_dict()))
        self.assertEqual(initial_core, state_digest(frozen.model.core.state_dict()))
        self.assertNotEqual(initial_core, state_digest(full.model.core.state_dict()))
        self.assertEqual(frozen.work, full.work)
        expected = train.summarize_rows(full.stream, 2, 14)
        self.assertTrue(all(full.work[key] == value for key, value in expected.items()))
        self.assertEqual(full.work["training_updates"], 3)
        self.assertEqual(full.work["training_ticks"], 12)
        self.assertEqual(full.work["training_lane_ticks"], 36)
        self.assertEqual(full.work["mode_rows"], 72)
        self.assertEqual(full.work["warm_loss_rows"], 0)

    def test_saved_real_adam_and_hidden_resume_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "original"
            uninterrupted = make(folder=folder)
            uninterrupted.warm_start()
            uninterrupted.save("step000")
            uninterrupted.update(save_audit=True)
            uninterrupted.update(save_audit=True)
            cp = uninterrupted.payload()
            self.assertEqual(cp["step"], 2)
            self.assertTrue(cp["optimizer"]["state"])
            self.assertTrue(all(int(state["step"]) == 2 for state in cp["optimizer"]["state"].values()))
            self.assertGreater(sum(float(state["exp_avg"].abs().sum())
                for state in cp["optimizer"]["state"].values()), 0.)
            resumed = train.Learner.restore(cp, uninterrupted.stream)
            uninterrupted.update(); resumed.update()
            for key in ("model", "optimizer", "hidden", "stream_cursor", "step", "work"):
                self.assertEqual(state_digest(uninterrupted.payload()[key]), state_digest(resumed.payload()[key]), key)
            before = torch.load(folder / "step002_before.pt", map_location="cpu", weights_only=False)
            after = torch.load(folder / "step002_after.pt", map_location="cpu", weights_only=False)
            gradients = torch.load(folder / "step002_gradients.pt", map_location="cpu", weights_only=False)
            self.assertEqual(before["stream_cursor"], 6)
            self.assertEqual(after["stream_cursor"], 10)
            self.assertTrue(torch.equal(gradients["next_hidden"], after["hidden"]))
            self.assertTrue(any(float(g.abs().sum()) > 0 for name, g in gradients["preclip"].items() if name.startswith("core.")))
            self.assertNotEqual(state_digest(before["model"]), state_digest(after["model"]))


if __name__ == "__main__":
    unittest.main()
