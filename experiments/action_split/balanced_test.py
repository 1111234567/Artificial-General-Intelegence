"""Check the single normalization intervention and frozen-parent reuse."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.action_split import balanced_train as balanced, train as base
from experiments.action_split.model import SplitGRUAgent, split_loss
from experiments.continuity.run import state_digest


def targets(modes):
    active = torch.ones_like(modes, dtype=torch.bool)
    return modes % 2, modes % 8, active


class BalancedNormalizationTest(unittest.TestCase):
    def test_fixed_formula_and_unchanged_conditional_terms(self):
        torch.manual_seed(152)
        scores = torch.randn(9, 13, dtype=torch.float64, requires_grad=True)
        modes = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2])
        begin, finish, active = targets(modes)
        old = split_loss(scores, modes, begin, finish, active)
        new = balanced.balanced_loss(scores, modes, begin, finish, active)
        expected = sum(F.cross_entropy(scores[modes == k, :3], modes[modes == k])
                       for k in range(3)) / 3
        self.assertTrue(torch.equal(new["mode"], expected))
        self.assertTrue(torch.equal(old["begin"], new["begin"]))
        self.assertTrue(torch.equal(old["finish"], new["finish"]))
        self.assertTrue(torch.equal(new["total"], new["mode"] + old["begin"] + old["finish"]))
        self.assertEqual(float(new["mode_weight_sum"]), 3.)
        self.assertNotEqual(float(old["mode"].detach()), float(new["mode"].detach()))

    def test_repeating_class_rows_preserves_equal_class_influence(self):
        representatives = torch.tensor([[2., .1, -.2], [.2, 1., -.3], [.5, -.4, 1.5]], dtype=torch.float64)
        single = torch.zeros(3, 13, dtype=torch.float64)
        single[:, :3] = representatives
        single.requires_grad_()
        single_modes = torch.arange(3)
        reference = balanced.balanced_loss(single, single_modes, *targets(single_modes))["mode"]
        reference.backward()
        modes = torch.tensor([0] * 17 + [1] * 5 + [2])
        repeated = torch.zeros(len(modes), 13, dtype=torch.float64)
        repeated[:, :3] = representatives[modes]
        repeated.requires_grad_()
        result = balanced.balanced_loss(repeated, modes, *targets(modes))["mode"]
        result.backward()
        self.assertTrue(torch.allclose(result, reference, rtol=0, atol=1e-14))
        for mode in range(3):
            self.assertTrue(torch.allclose(repeated.grad[modes == mode, :3].sum(0),
                                           single.grad[mode, :3], rtol=0, atol=1e-14))

    def test_empty_class_is_rejected_without_fallback(self):
        modes = torch.tensor([0, 0, 1])
        with self.assertRaisesRegex(ValueError, "all three mode classes"):
            balanced.balanced_loss(torch.zeros(3, 13), modes, *targets(modes))
        with self.assertRaisesRegex(ValueError, "no tunable"):
            modes = torch.arange(3)
            balanced.balanced_loss(torch.zeros(3, 13), modes, *targets(modes), wait_weight=1.)

    def test_process_local_bindings_restore_and_raw_budgets_match(self):
        old = {name: getattr(base, name) for name in
               ("OUT", "PLAN", "FORMAT", "checked", "split_loss", "summarize_rows")}
        ticks, batch = 14, 3
        rng = np.random.default_rng(173)
        modes = np.arange(ticks * batch * 2).reshape(ticks, batch, 2) % 3
        stream = dict(obs=rng.standard_normal((ticks, batch, 56), dtype=np.float32),
            mode_target=modes, begin_target=modes % 2, finish_target=modes % 8,
            active=np.ones_like(modes, dtype=np.bool_))
        with balanced.bindings():
            self.assertIs(base.split_loss, balanced.balanced_loss)
            torch.manual_seed(96)
            frozen = base.Learner(SplitGRUAgent(width=7), stream, "frozen", warm=2, block=4, updates=3)
            torch.manual_seed(96)
            full = base.Learner(SplitGRUAgent(width=7), stream, "full", warm=2, block=4, updates=3)
            frozen.warm_start(); full.warm_start()
            initial = state_digest(full.model.core.state_dict())
            for update in range(3):
                frozen.update(); full.update()
                if update == 0:
                    self.assertEqual(state_digest(frozen.model.state_dict()), state_digest(full.model.state_dict()))
            self.assertEqual(initial, state_digest(frozen.model.core.state_dict()))
            self.assertNotEqual(initial, state_digest(full.model.core.state_dict()))
            self.assertEqual(frozen.work, full.work)
            self.assertEqual(full.work["mode_rows"], 72)
            self.assertEqual(full.work["mode_weight_sum"], 9.)
            self.assertEqual(full.work["begin_rows"], 24)
            self.assertEqual(full.work["finish_rows"], 72)
            self.assertEqual(full.cursor, 14)
        for name, value in old.items():
            self.assertIs(getattr(base, name), value)


if __name__ == "__main__":
    unittest.main()
