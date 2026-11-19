"""Check native decoding, dense supervision, and the frozen/full intervention."""
import math
import unittest

import torch

from experiments.action_split.model import SplitGRUAgent, choose, split_loss
from experiments.learn_remember.gru_train import GRUAgent as OriginalGRUAgent
from experiments.lifelong.world import ContextWorld


class SplitModelTest(unittest.TestCase):
    def test_parameter_counts_and_source_copy_are_exact(self):
        source = OriginalGRUAgent()
        before = {key: value.clone() for key, value in source.state_dict().items()}
        frozen = SplitGRUAgent.from_original(source, frozen_core=True)
        full = SplitGRUAgent.from_original(source, frozen_core=False)
        self.assertEqual(sum(p.numel() for p in source.parameters()), 148246)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 149018)
        self.assertEqual(sum(p.numel() for p in full.parameters()) -
                         sum(p.numel() for p in source.parameters()), 772)
        self.assertEqual(frozen.parameter_report()["trainable_parameters"], 5018)
        self.assertEqual(full.parameter_report()["trainable_parameters"], 149018)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        for a, b, c in zip(source.core.parameters(), frozen.core.parameters(),
                           full.core.parameters()):
            self.assertTrue(torch.equal(a, b) and torch.equal(a, c))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertNotEqual(b.data_ptr(), c.data_ptr())
            self.assertFalse(b.requires_grad)
            self.assertTrue(c.requires_grad)
        self.assertTrue(all(p.requires_grad for p in frozen.head.parameters()))
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in full.head.parameters()))
        restored = SplitGRUAgent(**full.config())
        restored.load_state_dict(full.state_dict())
        self.assertEqual(restored.config(), full.config())

    def test_zero_head_policies_match_and_wait_for_every_public_history(self):
        torch.manual_seed(71)
        source = OriginalGRUAgent(width=8, input_size=4)
        frozen = SplitGRUAgent.from_original(source, frozen_core=True)
        full = SplitGRUAgent.from_original(source, frozen_core=False)
        obs, anchor = torch.randn(9, 3, 4), torch.randn(3, 8)
        sf, hf = frozen(obs, anchor)
        st, ht = full(obs, anchor)
        self.assertTrue(torch.equal(sf, st) and torch.equal(hf, ht))
        self.assertEqual(tuple(sf.shape), (9, 3, 2, 13))
        self.assertEqual(int(choose(sf).sum()), 0)
        scores, h = full.step(obs[0], anchor, dt=.25)
        self.assertTrue(torch.equal(scores, st[0]))
        self.assertTrue(torch.equal(h, full(obs[:1], anchor)[1]))

    def test_group_first_decoding_covers_all_native_codes_and_ties(self):
        scores = torch.zeros(11, 13)
        for code in range(11):
            if code in (1, 2):
                scores[code, 1] = 1
                scores[code, 3 + code - 1] = 2
            elif code >= 3:
                scores[code, 2] = 1
                scores[code, 5 + code - 3] = 2
        self.assertTrue(torch.equal(choose(scores), torch.arange(11)))
        scores[0, 5:] = 10000  # conditional confidence cannot bypass WAIT mode
        self.assertEqual(int(choose(scores)[0]), 0)
        tied = torch.zeros(2, 13)
        tied[0, 1:3] = 1  # BEGIN precedes FINISH in a mode tie
        tied[1, 2] = 1  # first FINISH content is action 3
        self.assertEqual(choose(tied).tolist(), [1, 3])
        with self.assertRaises(ValueError):
            choose(torch.zeros(2, 11))

    def test_decoder_emits_illegal_native_actions_without_oracle_gate(self):
        world = ContextWorld(seed=127, batch_size=1, actor_ids=(90300,), n_jobs=2,
                             active_jobs=2, duration_range=(1., 2.),
                             gap_range=(.5, 1.), dt=.25)
        world.observe()
        self.assertFalse(bool(world.started.any()))
        scores = torch.zeros(1, 2, 13)
        scores[..., 2] = 1  # FINISH even though no job has begun
        scores[..., 12] = 2
        actions = choose(scores)
        self.assertEqual(actions.tolist(), [[10, 10]])
        world.act(actions)
        self.assertEqual(world.last_result.tolist(), [[-1, -1]])

    def test_losses_include_processing_content_and_use_declared_normalizers(self):
        scores = torch.zeros(4, 13, dtype=torch.float64, requires_grad=True)
        modes = torch.tensor([0, 1, 2, 0])
        begins = torch.tensor([-1, 1, -1, -1])
        finishes = torch.tensor([7, 0, 3, -1])
        active = torch.tensor([True, True, True, False])
        result = split_loss(scores, modes, begins, finishes, active)
        self.assertAlmostEqual(float(result["mode"].detach()), math.log(3), places=12)
        self.assertAlmostEqual(float(result["begin"].detach()), math.log(2), places=12)
        self.assertAlmostEqual(float(result["finish"].detach()), math.log(8), places=12)
        self.assertEqual([int(result[k]) for k in
                          ("mode_rows", "begin_rows", "finish_rows")], [4, 1, 3])
        self.assertEqual(float(result["mode_weight_sum"]), 2.5)
        result["total"].backward()
        expected_mode = torch.full((4, 3), 1 / 3, dtype=torch.float64)
        expected_mode[torch.arange(4), modes] -= 1
        expected_mode *= torch.tensor([.25, 1., 1., .25],
                                      dtype=torch.float64)[:, None] / 2.5
        self.assertTrue(torch.allclose(scores.grad[:, :3], expected_mode))
        # WAIT content (row 0) is supervised, even for a later-cancelled request;
        # inactive row 3 is ignored regardless of its sentinel target.
        self.assertLess(float(scores.grad[0, 5 + 7]), 0)
        self.assertTrue(torch.equal(scores.grad[3, 5:], torch.zeros(8)))
        self.assertTrue(torch.equal(scores.grad[[0, 2, 3], 3:5], torch.zeros(3, 2)))

    def test_empty_conditional_masks_are_finite_differentiable_zero(self):
        scores = torch.randn(5, 13, requires_grad=True)
        modes = torch.zeros(5, dtype=torch.long)
        absent = torch.full((5,), -1, dtype=torch.long)
        result = split_loss(scores, modes, absent, absent,
                            torch.zeros(5, dtype=torch.bool))
        self.assertEqual(float(result["begin"].detach()), 0)
        self.assertEqual(float(result["finish"].detach()), 0)
        self.assertTrue(bool(torch.isfinite(result["total"])))
        result["total"].backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertTrue(torch.equal(scores.grad[:, 3:], torch.zeros(5, 10)))

    def test_second_update_changes_only_the_permitted_core_and_not_source(self):
        torch.manual_seed(83)
        source = OriginalGRUAgent(width=8, input_size=4)
        before = {key: value.clone() for key, value in source.state_dict().items()}
        agents = [SplitGRUAgent.from_original(source, frozen_core=frozen)
                  for frozen in (True, False)]
        obs = torch.randn(7, 2, 4)
        modes = torch.full((7, 2, 2), 2, dtype=torch.long)
        begins = torch.full_like(modes, -1)
        finishes = torch.arange(modes.numel()).reshape_as(modes) % 8
        active = torch.ones_like(modes, dtype=torch.bool)
        for index, model in enumerate(agents):
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=.01)
            h = model.initial_state(2)
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                scores, next_h = model(obs, h)
                split_loss(scores, modes, begins, finishes, active)["total"].backward()
                if update == 0 and index == 1:
                    self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                                        for p in model.core.parameters()))
                if update == 1:
                    if index == 0:
                        self.assertTrue(all(p.grad is None for p in model.core.parameters()))
                    else:
                        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                                            for p in model.core.parameters()))
                optimizer.step()
                h = next_h.detach()  # preserve numerical state across updates
            equal = all(torch.equal(value, source.core.state_dict()[key])
                        for key, value in model.core.state_dict().items())
            self.assertEqual(equal, index == 0)
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in source.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
