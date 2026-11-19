"""Small contracts for a uniform reservoir of real, past feedback records."""
from __future__ import annotations

import unittest

import torch

from experiments.consolidation.memory import Reservoir


def records(count: int, start: int = 0, device: str = "cpu"):
    targets = torch.arange(start, start + count, device=device)
    features = torch.stack((targets.float(), -targets.float()), dim=1)
    return features, targets


class ReservoirContracts(unittest.TestCase):
    def test_empty_and_partial_buffer_never_sample_unseen_rows(self):
        memory = Reservoir(50, 2, "cpu", seed=17)
        with self.assertRaisesRegex(ValueError, "empty"):
            memory.sample(1)
        features, targets = memory.sample(0)
        self.assertEqual(features.shape, (0, 2))
        self.assertEqual(targets.shape, (0,))
        self.assertEqual(memory.seen, 0)
        memory.add(*records(3, start=101))
        features, targets = memory.sample(200)
        self.assertEqual(len(memory), 3)
        self.assertEqual(memory.seen, 3)
        self.assertEqual(set(targets.tolist()), {101, 102, 103})
        torch.testing.assert_close(features[:, 0], targets.float())
        torch.testing.assert_close(features[:, 1], -targets.float())

    def test_only_add_exposes_true_feedback_to_replay(self):
        memory = Reservoir(10, 2, "cpu", seed=1)
        first_features = torch.tensor([[100., 0.], [100., 0.]])
        first_labels = torch.tensor([8, 9])  # Neither is feature argmax (zero).
        memory.add(first_features, first_labels)
        future_features, future_labels = records(3, start=20)
        _, old_targets = memory.sample(100)
        self.assertEqual(set(old_targets.tolist()), {8, 9})
        memory.add(future_features, future_labels)
        _, after_feedback = memory.sample(1000)
        self.assertEqual(set(after_feedback.tolist()), {8, 9, 20, 21, 22})

    def test_capacity_and_storage_accounting(self):
        memory = Reservoir(7, 2, "cpu", seed=2)
        for start, count in ((0, 4), (4, 31), (35, 80)):
            memory.add(*records(count, start=start))
            self.assertLessEqual(len(memory), 7)
            self.assertEqual(memory.seen, start + count)
        state = memory.state_dict()
        self.assertEqual(len(memory), 7)
        self.assertEqual(memory.bytes, 7 * (2 * 4 + 8 + 8))
        self.assertGreater(memory.replacements, 0)
        self.assertEqual(state["features"].shape, (7, 2))
        self.assertEqual(state["priorities"].shape, (7,))
        sampled_features, sampled_targets = memory.sample(50)
        self.assertTrue(bool(((sampled_targets >= 0) & (sampled_targets < 115)).all()))
        torch.testing.assert_close(sampled_features[:, 0], sampled_targets.float())

    def test_inputs_samples_and_state_are_detached_independent_copies(self):
        memory = Reservoir(10, 2, "cpu", seed=3)
        leaf = torch.arange(8, dtype=torch.float64, requires_grad=True)
        features = (leaf * 2).reshape(4, 2)
        targets = torch.arange(4, dtype=torch.float64, requires_grad=True) + 10
        expected_features, expected_targets = features.detach().clone(), targets.detach().clone()
        memory.add(features, targets)
        with torch.no_grad():
            features.fill_(-100)
            targets.fill_(-100)
        state = memory.state_dict()
        order = state["targets"].argsort()
        torch.testing.assert_close(state["features"][order], expected_features)
        torch.testing.assert_close(state["targets"][order], expected_targets)
        for name in ("features", "targets", "priorities"):
            self.assertFalse(state[name].requires_grad)
            self.assertIsNone(state[name].grad_fn)
            state[name].fill_(-999)
        sample_features, sample_targets = memory.sample(20)
        self.assertFalse(sample_features.requires_grad)
        self.assertFalse(sample_targets.requires_grad)
        self.assertIsNone(sample_features.grad_fn)
        sample_features.fill_(-888)
        sample_targets.fill_(-888)
        fresh = memory.state_dict()
        self.assertTrue(bool((fresh["features"] >= 0).all()))
        self.assertTrue(bool((fresh["targets"] >= 10).all()))
        self.assertIsNone(leaf.grad)

    def test_identical_seed_and_batch_partition_preserve_selection(self):
        features, targets = records(213)
        whole = Reservoir(19, 2, "cpu", seed=203)
        parts = Reservoir(19, 2, "cpu", seed=203)
        singleton = Reservoir(19, 2, "cpu", seed=203)
        whole.add(features, targets)
        for left, right in ((0, 1), (1, 10), (10, 87), (87, 213)):
            parts.add(features[left:right], targets[left:right])
        for index in range(213):
            singleton.add(features[index:index + 1], targets[index:index + 1])
        expected = whole.state_dict()
        for memory in (parts, singleton):
            actual = memory.state_dict()
            for name in ("features", "targets", "priorities", "rng_state", "sample_rng_state"):
                self.assertTrue(torch.equal(actual[name], expected[name]), name)
            self.assertEqual(memory.seen, whole.seen)
            expected_sample = whole.sample(11)
            actual_sample = memory.sample(11)
            for left, right in zip(expected_sample, actual_sample):
                self.assertTrue(torch.equal(left, right))
            # Reset whole's RNG before comparing the next independently built buffer.
            whole.load_state_dict(expected)

    def test_replay_frequency_does_not_change_retained_history(self):
        quiet = Reservoir(9, 2, "cpu", seed=42)
        replayed = Reservoir(9, 2, "cpu", seed=42)
        for start in range(0, 130, 10):
            quiet.add(*records(10, start=start))
            replayed.add(*records(10, start=start))
            replayed.sample(7)
            replayed.sample(19)
        first, second = quiet.state_dict(), replayed.state_dict()
        for name in ("features", "targets", "priorities", "rng_state"):
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assertFalse(torch.equal(first["sample_rng_state"], second["sample_rng_state"]))

    def test_rng_is_private_and_deterministic(self):
        global_before = torch.random.get_rng_state().clone()
        first = Reservoir(5, 2, "cpu", seed=77)
        second = Reservoir(5, 2, "cpu", seed=77)
        first.add(*records(30))
        second.add(*records(30))
        for _ in range(3):
            for left, right in zip(first.sample(9), second.sample(9)):
                self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_saved_state_restores_future_sampling_and_additions(self):
        original = Reservoir(8, 2, "cpu", seed=18)
        original.add(*records(40))
        original.sample(13)
        saved = original.state_dict()
        restored = Reservoir(8, 2, "cpu", seed=999)
        restored.load_state_dict(saved)
        saved["features"].fill_(-1)
        saved["rng_state"].zero_()
        for memory in (original, restored):
            memory.add(*records(60, start=40))
        self.assertEqual(original.seen, restored.seen)
        self.assertEqual(original.replacements, restored.replacements)
        for left, right in zip(original.sample(100), restored.sample(100)):
            self.assertTrue(torch.equal(left, right))

    def test_history_inclusion_and_replay_are_uniform_with_wide_tolerance(self):
        # Broad deterministic coverage checks catch recency/slot bias without
        # turning a small stochastic experiment into a fragile significance test.
        inclusion = torch.zeros(24, dtype=torch.long)
        for seed in range(512):
            memory = Reservoir(6, 2, "cpu", seed=seed)
            memory.add(*records(24))
            inclusion[memory.state_dict()["targets"]] += 1
        self.assertTrue(bool(((inclusion > 75) & (inclusion < 181)).all()))
        memory = Reservoir(3, 2, "cpu", seed=8)
        memory.add(*records(3))
        counts = torch.bincount(memory.sample(6000)[1], minlength=3)
        self.assertTrue(bool(((counts > 1700) & (counts < 2300)).all()))

    def test_zero_capacity_and_invalid_inputs(self):
        memory = Reservoir(0, 2, "cpu", seed=0)
        memory.add(*records(5))
        self.assertEqual(memory.seen, 5)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.bytes, 0)
        with self.assertRaises(ValueError):
            memory.sample(1)
        with self.assertRaises(ValueError):
            memory.sample(-1)
        with self.assertRaises(ValueError):
            memory.add(torch.zeros(2, 3), torch.zeros(2))
        with self.assertRaises(ValueError):
            memory.add(torch.zeros(2, 2), torch.zeros(3))
        with self.assertRaises(ValueError):
            Reservoir(-1, 2, "cpu", seed=0)
        with self.assertRaises(ValueError):
            Reservoir(1, 0, "cpu", seed=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_storage_keeps_cpu_seed_selection_and_detaches(self):
        cpu = Reservoir(11, 2, "cpu", seed=910)
        cuda = Reservoir(11, 2, "cuda", seed=910)
        features, targets = records(51)
        cpu.add(features, targets)
        cuda.add(features.to("cuda").requires_grad_(), targets.to("cuda"))
        for name in ("features", "targets", "priorities"):
            state = cuda.state_dict()[name]
            self.assertEqual(state.device.type, "cuda")
            self.assertFalse(state.requires_grad)
            self.assertTrue(torch.equal(state.cpu(), cpu.state_dict()[name]))
        for left, right in zip(cuda.sample(23), cpu.sample(23)):
            self.assertEqual(left.device.type, "cuda")
            self.assertTrue(torch.equal(left.cpu(), right))


if __name__ == "__main__":
    unittest.main()
