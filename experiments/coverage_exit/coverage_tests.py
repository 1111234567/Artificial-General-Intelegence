"""Actual-state integration checks for coverage continuation (opt-in slow test)."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
from copy import deepcopy

import torch
from experiments.coverage_exit import coverage
from experiments.undirected import run as native


class CoverageActualStateTests(unittest.TestCase):
    def test_unassisted_matches_archived_original_complete_trace(self):
        trainer = native.NativeTrainer.from_checkpoint(coverage.checkpoint(0), device="cpu")
        point, audit, counts, checks = coverage.one_update(trainer, "control")
        original = torch.load(coverage.checkpoint(0,196608),map_location="cpu",weights_only=False)["trace"][1024]
        self.assertEqual(point, original)
        self.assertEqual(counts["assisted"], 0)
        self.assertTrue(torch.equal(audit["model_actions"], audit["actual_actions"]))
        self.assertTrue(all(value is True for key,value in checks.items() if key.endswith("exact")))

    def test_assistance_creates_only_successful_begin_and_original_causal_labels(self):
        trainer = native.NativeTrainer.from_checkpoint(coverage.checkpoint(0), device="cpu")
        point, audit, counts, checks = coverage.one_update(trainer, "begin_quarter")
        changed = audit["actual_actions"].ne(audit["model_actions"])
        self.assertTrue(torch.equal(changed,audit["assisted"]))
        self.assertTrue(bool(audit["selected"][changed].all()))
        self.assertTrue(bool(audit["accepted"][changed].all()))
        self.assertTrue(bool((audit["actual_actions"][changed].gt(0)&audit["actual_actions"][changed].lt(3)).all()))
        self.assertEqual(counts["assisted"],6)
        self.assertGreater(counts["labeled_finish"],0)
        self.assertEqual(counts["actual_finish"],counts["model_finish"])
        self.assertEqual(point["actual_action_counts"],torch.bincount(audit["actual_actions"].reshape(-1),minlength=11).tolist())
        self.assertTrue(checks["original_teacher_exact"])

    def test_identity_selection_is_external_deterministic(self):
        expected = [coverage.selected_identity(0,7,1,1000+i,1) for i in range(100)]
        self.assertEqual(expected,[coverage.selected_identity(0,7,1,1000+i,1) for i in range(100)])
        self.assertGreater(sum(expected),10)
        self.assertLess(sum(expected),40)


if __name__=="__main__":
    unittest.main()
