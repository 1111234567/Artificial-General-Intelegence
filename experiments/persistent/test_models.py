"""CPU-only contracts for recurrence, fact memory, and label isolation.

These tests use forwards and small autograd probes, never optimizer steps.
Run with ``python -m unittest experiments.persistent.test_models -v``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from experiments.persistent.models import CompressedGRU, MemoryLoop, selected_logits
from experiments.persistent.run import build_model, call_model
from experiments.persistent.tasks import Batch, make_batch


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        # All examples and models remain on CPU, including on CUDA hosts.
        torch.manual_seed(1729)
        self.args = SimpleNamespace(vocab=23, dim=12, gru_hidden=16)
        self.batch = make_batch(
            5, 7, 0, 4, vocab=self.args.vocab, device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(541),
            fixed_hops=4,
        )

    def assert_same_output(self, first, second):
        for name in ("logits", "judgment", "attention"):
            left, right = getattr(first, name), getattr(second, name)
            if left is None or right is None:
                self.assertIs(left, right, name)
            else:
                self.assertTrue(torch.equal(left, right), f"{name} changed")

    def activate_update(self, model):
        # The final update layer starts at zero. Exercise a reachable nonzero
        # state so feedback tests cannot pass because that path is dormant.
        with torch.no_grad():
            model.update[-1].weight.normal_(0, 0.08)
            model.update[-1].bias.normal_(0, 0.02)
        return model.eval()

    def memory_pair(self):
        joint = self.activate_update(build_model("joint", self.args))
        detached = build_model("detach", self.args).eval()
        detached.load_state_dict(joint.state_dict())
        return joint, detached

    def test_joint_and_detach_are_bitwise_identical_in_forward(self):
        joint, detached = self.memory_pair()
        with torch.no_grad():
            self.assert_same_output(
                call_model(joint, self.batch, steps=4),
                call_model(detached, self.batch, steps=4),
            )

    def test_final_only_loss_has_different_cross_step_gradients(self):
        joint, detached = self.memory_pair()
        traces = []
        for model in (joint, detached):
            updates = []

            def retain_update(_module, _inputs, output):
                output.retain_grad()
                updates.append(output)

            handle = model.update.register_forward_hook(retain_update)
            try:
                output = call_model(model, self.batch, steps=4)
            finally:
                handle.remove()
            # Earlier predictions and all judgments are deliberately absent
            # from the loss. Any earlier update gradient must cross a step.
            F.cross_entropy(output.logits[:, -1], self.batch.target).backward()
            traces.append(updates)

        for step in range(3):
            with self.subTest(earlier_step=step):
                self.assertIsNotNone(traces[0][step].grad)
                self.assertGreater(traces[0][step].grad.abs().sum().item(), 1e-8)
                # Stacking all step logits can materialize explicit zero
                # gradients for unused slices rather than leaving grad=None.
                cut_gradient = traces[1][step].grad
                self.assertTrue(cut_gradient is None or
                                cut_gradient.count_nonzero().item() == 0)
        for updates in traces:
            self.assertGreater(updates[-1].grad.abs().sum().item(), 1e-8)
        self.assertFalse(torch.allclose(
            joint.query.weight.grad, detached.query.weight.grad,
            rtol=1e-5, atol=1e-7,
        ))

    def test_fact_permutation_preserves_predictions_and_judgment(self):
        reordered = self.batch.shuffle_records(
            torch.Generator(device="cpu").manual_seed(184)
        )
        self.assertFalse(torch.equal(self.batch.keys, reordered.keys))
        for mode in ("joint", "judge"):
            with self.subTest(mode=mode), torch.no_grad():
                model = self.activate_update(build_model(mode, self.args))
                expected = call_model(model, self.batch, steps=4)
                actual = call_model(model, reordered, steps=4)
                # Reductions over a differently ordered set can round
                # differently; set invariance is numerical, not bitwise.
                torch.testing.assert_close(actual.logits, expected.logits,
                                           rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(actual.judgment, expected.judgment,
                                           rtol=2e-5, atol=2e-5)
                # A constant predictor would pass the invariance assertion.
                # Changing pairings must affect the content-dependent read.
                corrupt = replace(self.batch, values=self.batch.values.roll(1, 1))
                changed = call_model(model, corrupt, steps=4)
                self.assertGreater(
                    (changed.logits - expected.logits).abs().max().item(), 1e-3
                )

    def test_runner_skips_noop_delay_for_lossless_memory(self):
        # This is an interface property: the archive does not process NOOPs.
        # Passing it must not be reported as learned long-sequence retention.
        for mode in ("single", "detach", "joint", "judge_aux", "judge"):
            with self.subTest(mode=mode), torch.no_grad():
                model = self.activate_update(build_model(mode, self.args))
                expected = call_model(model, self.batch.with_gap(0), steps=4)
                for gap in (1, 4096, 1_000_000):
                    with self.subTest(gap=gap):
                        self.assert_same_output(
                            expected, call_model(model, self.batch.with_gap(gap), steps=4)
                        )

    def test_call_model_never_uses_target_or_path_labels(self):
        poisoned = replace(
            self.batch,
            target=(self.batch.target + 1) % self.args.vocab,
            path_targets=(self.batch.path_targets + 7) % self.args.vocab,
        )
        self.assertTrue(torch.all(poisoned.target != self.batch.target))
        self.assertTrue(torch.all(poisoned.path_targets != self.batch.path_targets))
        for mode in ("single", "detach", "joint", "judge_aux", "judge", "gru"):
            with self.subTest(mode=mode), torch.no_grad():
                model = build_model(mode, self.args).eval()
                if isinstance(model, MemoryLoop):
                    self.activate_update(model)
                self.assert_same_output(
                    call_model(model, self.batch, steps=4),
                    call_model(model, poisoned, steps=4),
                )

    def test_predicted_judge_feedback_affects_only_later_steps(self):
        model = self.activate_update(build_model("judge", self.args))
        with torch.no_grad():
            normal = call_model(model, self.batch, steps=4)
            zeroed = call_model(model, self.batch, steps=4, feedback_override="zero")
        self.assertTrue(torch.equal(normal.logits[:, 0], zeroed.logits[:, 0]))
        self.assertTrue(torch.equal(normal.judgment[:, 0], zeroed.judgment[:, 0]))
        self.assertGreater(
            (normal.logits[:, 1:] - zeroed.logits[:, 1:]).abs().max().item(), 1e-4
        )

    def test_content_reader_follows_a_known_cycle_without_path_supervision(self):
        # Hand-set content addressing gives an independent semantic oracle:
        # a six-node cycle, presented in two different record orders.
        keys = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 3, 1, 4, 0, 2]])
        expected_path = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
        batch = Batch(
            keys=keys, values=(keys + 1) % 6, query=torch.tensor([0, 5]),
            hops=torch.tensor([4, 3]), target=torch.tensor([4, 2]),
            path_targets=expected_path, gap=0, vocab=8,
        )
        model = MemoryLoop(vocab=8, dim=8, mode="joint").eval()
        with torch.no_grad():
            model.embedding.weight.copy_(torch.eye(8))
            model.key.weight.copy_(torch.eye(8))
            model.query.weight.copy_(torch.eye(8))
            output = call_model(model, batch, steps=4)
        self.assertTrue(torch.equal(output.logits.argmax(-1), expected_path))
        self.assertTrue(torch.equal(selected_logits(output, batch.hops).argmax(-1),
                                    torch.tensor([4, 2])))

    def test_gru_chunking_with_carried_state_matches_full_sequence(self):
        model = CompressedGRU(vocab=23, dim=12, hidden=16).eval()
        records = torch.randn(3, 13, 2 * model.dim + 1)
        with torch.no_grad():
            full, full_state = model.encode_chunk(records)
            prefix, state = model.encode_chunk(records[:, :5])
            suffix, carried = model.encode_chunk(records[:, 5:], state)
            reset_suffix, _ = model.encode_chunk(records[:, 5:])
        torch.testing.assert_close(torch.cat((prefix, suffix), 1), full,
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(carried, full_state, rtol=1e-5, atol=1e-6)
        self.assertGreater((suffix - reset_suffix).abs().max().item(), 1e-4)

    def test_detaching_gru_state_keeps_memory_values_but_cuts_prefix_gradient(self):
        model = CompressedGRU(vocab=23, dim=12, hidden=16).eval()
        records = torch.randn(3, 11, 2 * model.dim + 1, requires_grad=True)
        split = 5
        _, state = model.encode_chunk(records[:, :split])
        detached = state.detach()
        self.assertTrue(torch.equal(state, detached))
        self.assertFalse(detached.requires_grad)
        self.assertIsNone(detached.grad_fn)
        live_output, live_state = model.encode_chunk(records[:, split:], state)
        cut_output, cut_state = model.encode_chunk(records[:, split:], detached)
        self.assertTrue(torch.equal(live_output, cut_output))
        self.assertTrue(torch.equal(live_state, cut_state))
        live_gradient, = torch.autograd.grad(
            live_output[:, -1].square().sum(), records, retain_graph=True
        )
        cut_gradient, = torch.autograd.grad(cut_output[:, -1].square().sum(), records)
        self.assertGreater(live_gradient[:, :split].abs().sum().item(), 1e-8)
        self.assertEqual(cut_gradient[:, :split].count_nonzero().item(), 0)
        self.assertGreater(cut_gradient[:, split:].abs().sum().item(), 1e-8)
        torch.testing.assert_close(live_gradient[:, split:], cut_gradient[:, split:],
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
