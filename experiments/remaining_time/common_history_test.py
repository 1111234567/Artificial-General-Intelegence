"""Actual teacher causality and read-only native replay continuity."""
import unittest

import numpy as np
import torch

from experiments.continuity.run import state_digest
from experiments.remaining_time import common_history as common
from experiments.remaining_time.model import RemainingTimeGRUAgent


class CommonHistoryTest(unittest.TestCase):
    def test_actual_teacher_receipts_and_new_request_denominators(self):
        cfg = common.evaluate.config(0, "long")
        cfg.update(seed=921773, batch_size=3, actor_ids=(98000, 98001, 98002),
            duration_range=(.5, 1.), gap_range=(.5, 1.), window_range=(.5, 1.))
        facts, worlds, external_hash = common.collect_native(0, "long", total=96, world_config=cfg)
        audit = common.verify_teacher(facts)
        report = common.base.summarize_facts(facts, 0, warm=8)
        self.assertEqual(facts["obs"].shape, (96, 3, 56))
        self.assertEqual(worlds["initial"]["tick"], -1)
        self.assertEqual(worlds["final"]["tick"], 95)
        self.assertEqual(worlds["final"]["event_hash"], external_hash)
        self.assertTrue(audit["native_actual_receipts_exact"])
        self.assertGreater(report["groups"]["all"]["matured"], 10)
        self.assertEqual(report["groups"]["all"]["ordinary"], report["groups"]["all"]["matured"])
        self.assertEqual(report["groups"]["all"]["strict"], report["groups"]["all"]["matured"])
        self.assertGreater(int((facts["active"] & facts["started"] & np.isfinite(facts["ready"])).sum()), 10)

    def test_replay_matches_native_steps_and_carries_h_from_origin(self):
        torch.set_num_threads(1)
        torch.manual_seed(730)
        model = RemainingTimeGRUAgent(width=7).eval().requires_grad_(False)
        with torch.no_grad():
            model.time_head.weight.normal_()
            model.time_head.bias.normal_()
        obs = np.random.default_rng(46).standard_normal((18, 3, 56), dtype=np.float32)
        before = state_digest(model.state_dict())
        original_obs = obs.copy()
        prediction, logits, final_h = common.replay(model, obs)
        with torch.inference_mode():
            h = model.initial_state(3)
            all_logits, all_time = [], []
            for row in obs:
                # Direct inherited forward is the exact original native path.
                scores, h = model(torch.from_numpy(row)[None], h)
                all_logits.append(scores[0].numpy().copy())
                all_time.append(model.time_head(h).numpy().copy())
            np.testing.assert_array_equal(logits, np.stack(all_logits))
            np.testing.assert_array_equal(prediction, np.stack(all_time))
            np.testing.assert_array_equal(final_h, h.numpy())
            reset = model.initial_state(3)
            for row in obs[8:]:
                _, reset = model.step(torch.from_numpy(row), reset)
            self.assertFalse(np.array_equal(final_h, reset.numpy()))
        np.testing.assert_array_equal(obs, original_obs)
        self.assertEqual(before, state_digest(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_head_outputs_do_not_change_replayed_native_policy_or_h(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = RemainingTimeGRUAgent(width=7)
        obs = np.random.default_rng(778).standard_normal((20, 2, 56), dtype=np.float32)
        time_before, logits_before, h_before = common.replay(model, obs)
        with torch.no_grad():
            model.time_head.weight.fill_(10000)
            model.time_head.bias.fill_(-10000)
        time_after, logits_after, h_after = common.replay(model, obs)
        self.assertFalse(np.array_equal(time_before, time_after))
        np.testing.assert_array_equal(logits_before, logits_after)
        np.testing.assert_array_equal(h_before, h_after)


if __name__ == "__main__":
    unittest.main()
