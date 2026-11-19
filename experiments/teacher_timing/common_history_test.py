"""Both actual teachers' histories and native read-only replay use fixed APIs."""
import unittest

import numpy as np
import torch

from experiments.continuity.run import state_digest
from experiments.remaining_time import common_history as previous
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.teacher_timing import common_history as common


class CrossedCommonHistoryTest(unittest.TestCase):
    def test_both_teachers_actual_receipts_request_coverage_and_exogenous_pairing(self):
        cfg = common.config(0)
        cfg.update(seed=921773, batch_size=3, actor_ids=(99000, 99001, 99002),
            duration_range=(.5, 1.), gap_range=(.5, 1.), window_range=(.5, 1.))
        histories, worlds = {}, {}
        for teacher in common.TEACHERS:
            facts, states, external = common.collect_native(0, teacher, total=96, world_config=cfg)
            histories[teacher], worlds[teacher] = facts, states
            audit = common.verify_teacher(facts, teacher)
            report = common.base.summarize_facts(facts, 0, warm=8)
            self.assertEqual(facts["obs"].shape, (96, 3, 56))
            self.assertEqual(states["initial"]["tick"], -1)
            self.assertEqual(states["final"]["tick"], 95)
            self.assertEqual(states["final"]["event_hash"], external)
            self.assertTrue(audit["native_actual_receipts_exact"])
            self.assertGreater(report["groups"]["all"]["matured"], 10)
            self.assertEqual(report["groups"]["all"]["ordinary"], report["groups"]["all"]["matured"])
            self.assertEqual(report["groups"]["all"]["strict"], report["groups"]["all"]["matured"])
        paired = common.data.compare_pair(histories["earliest"], histories["midpoint"])
        self.assertTrue(paired)
        self.assertGreater(int(histories["midpoint"]["time_mask"].sum()), int(histories["earliest"]["time_mask"].sum()))
        self.assertFalse(np.array_equal(histories["earliest"]["obs"], histories["midpoint"]["obs"]))
        self.assertEqual(worlds["earliest"]["final"]["event_hash"], worlds["midpoint"]["final"]["event_hash"])

    def test_exact_original_replay_never_uses_time_head_for_native_actions(self):
        self.assertIs(common.replay, previous.replay)
        torch.set_num_threads(1)
        torch.manual_seed(730)
        model = RemainingTimeGRUAgent(width=7)
        obs = np.random.default_rng(46).standard_normal((18, 3, 56), dtype=np.float32)
        before = state_digest(model.state_dict())
        time, logits, hidden = common.replay(model, obs)
        self.assertEqual(before, state_digest(model.state_dict()))
        with torch.no_grad():
            model.time_head.weight.fill_(10000)
            model.time_head.bias.fill_(-10000)
        changed_time, changed_logits, changed_hidden = common.replay(model, obs)
        self.assertFalse(np.array_equal(time, changed_time))
        np.testing.assert_array_equal(logits, changed_logits)
        np.testing.assert_array_equal(hidden, changed_hidden)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_crossed_paths_and_main_world_match_autonomous_evaluation(self):
        paths = {common.sidecar_path(seed, teacher, policy) for seed in common.SEEDS
            for teacher in common.TEACHERS for policy in common.POLICIES}
        self.assertEqual(len(paths), 24)
        for seed in common.SEEDS:
            cfg = common.config(seed)
            self.assertEqual(cfg, common.evaluate.config(seed, "long"))
            self.assertEqual(cfg["seed"], 854000 + seed)
            self.assertEqual(cfg["actor_ids"], tuple(range(99000, 99032)))
            self.assertEqual(cfg["duration_range"], (8., 16.))


if __name__ == "__main__":
    unittest.main()
