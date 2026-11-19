"""Real-history causality, uniform admission, RNG isolation, and exact resume."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import torch

from experiments.lifelong.replay import (
    HistoryReservoir, block_hash, combine_blocks, copied_block, validate_block,
)


def chunk(start=0, ticks=4, batch=2, inputs=3, jobs=2, hidden=5, family_context=True):
    observations = torch.arange(ticks * batch * inputs, dtype=torch.float32).reshape(ticks, batch, inputs)
    observations = observations / 10 + start
    # Deliberately unrelated actions and feedback detect accidental relabeling.
    targets = torch.full((ticks, batch, jobs), 7, dtype=torch.long)
    actions = torch.full((ticks, batch, jobs), 3, dtype=torch.long)
    result = {"observations": observations, "targets": targets, "actions": actions,
              "elapsed": (0.25,) * ticks, "start_h": torch.full((batch, hidden), float(start)),
              "start_tick": start, "end_tick": start + ticks,
              "family_counts": {0: ticks, 1: 2}, "source_id": f"own:{start}:{start + ticks}"}
    if family_context:
        result["family_context"] = torch.full_like(targets, (start // ticks) % 4)
    return result


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_startup_no_future_or_partial_block_is_sampled(self):
        memory = HistoryReservoir(32, 8, 19)
        for tick in (0, 4):
            self.assertEqual(memory.eligible_count(tick), 0)
            with self.assertRaisesRegex(ValueError, "no eligible past"):
                memory.sample(before_tick=tick)
            memory.add_chunk(chunk(tick))
        self.assertEqual(memory.blocks_seen, 1)
        self.assertEqual(memory.eligible_count(4), 0)
        self.assertEqual(memory.eligible_count(8), 1)
        historical = memory.sample(before_tick=8)
        self.assertEqual((historical["start_tick"], historical["end_tick"]), (0, 8))
        self.assertTrue(torch.equal(historical["targets"], torch.full((8, 2, 2), 7, dtype=torch.long)))
        self.assertTrue(torch.equal(historical["actions"], torch.full((8, 2, 2), 3, dtype=torch.long)))

    def test_consecutive_recent_window_keeps_actual_first_hidden_and_labels(self):
        previous, current = chunk(0), chunk(4)
        recent = combine_blocks(previous, current)
        self.assertEqual(recent["elapsed"], (0.25,) * 8)
        self.assertEqual(recent["family_counts"], {0: 8, 1: 4})
        self.assertEqual(recent["source_chunks"], ("own:0:4", "own:4:8"))
        self.assertTrue(torch.equal(recent["start_h"], previous["start_h"]))
        self.assertTrue(torch.equal(recent["observations"][:4], previous["observations"]))
        self.assertTrue(torch.equal(recent["observations"][4:], current["observations"]))
        self.assertTrue(torch.equal(recent["family_context"][:4], previous["family_context"]))
        recent["observations"].zero_()
        recent["start_h"].fill_(-100)
        self.assertFalse(torch.equal(recent["start_h"], previous["start_h"]))
        self.assertGreater(float(previous["observations"].sum()), 0)

    def test_own_feedback_never_recomputed_and_replay_never_admitted(self):
        memory = HistoryReservoir(3, 8, 7)
        memory.add_block(combine_blocks(chunk(0), chunk(4)))
        original = memory.summary()
        for _ in range(17):
            sample = memory.sample(8)
            self.assertEqual(set(sample["targets"].flatten().tolist()), {7})
            self.assertEqual(set(sample["actions"].flatten().tolist()), {3})
        after = memory.summary()
        for key in ("complete_blocks_seen", "real_chunks_seen", "real_ticks_seen", "stored_sha256", "stored_blocks"):
            self.assertEqual(after[key], original[key])
        self.assertEqual(after["samples_drawn"], 17)

    def test_storage_samples_and_checkpoint_are_detached_independent_copies(self):
        source = chunk(0, 8)
        source["observations"].requires_grad_()
        source["start_h"].requires_grad_()
        memory = HistoryReservoir(2, 8, 3)
        memory.add_block(source)
        expected_hash = memory.summary()["stored_sha256"]
        with torch.no_grad():
            source["observations"].fill_(-900)
            source["start_h"].fill_(-901)
        sample, saved = memory.sample(8), memory.state_dict()
        for key in ("observations", "targets", "actions", "start_h", "family_context"):
            self.assertFalse(sample[key].requires_grad)
            self.assertIsNone(sample[key].grad_fn)
            sample[key].fill_(0)
            saved["blocks"][0][key].fill_(0)
        self.assertEqual(memory.summary()["stored_sha256"], expected_hash)

    def test_replay_frequency_does_not_change_admission_or_global_rngs(self):
        random_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        quiet, replayed = HistoryReservoir(3, 8, 28), HistoryReservoir(3, 8, 28)
        for tick in range(0, 160, 8):
            for memory in (quiet, replayed):
                memory.add_block(chunk(tick, 8))
            replayed.sample(tick + 8)
            replayed.sample(tick + 8)
        left, right = quiet.state_dict(), replayed.state_dict()
        self.assertEqual([block_hash(item) for item in left["blocks"]],
                         [block_hash(item) for item in right["blocks"]])
        self.assertEqual(left["admission_rng_state"], right["admission_rng_state"])
        self.assertNotEqual(left["sample_rng_state"], right["sample_rng_state"])
        self.assertEqual(random.getstate(), random_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_direct_blocks_and_split_chunks_have_same_admission(self):
        direct, split = HistoryReservoir(5, 8, 32), HistoryReservoir(5, 8, 32)
        for tick in range(0, 200, 8):
            direct.add_block(combine_blocks(chunk(tick), chunk(tick + 4)))
            self.assertFalse(split.add_chunk(chunk(tick)))
            self.assertTrue(split.add_chunk(chunk(tick + 4)))
        self.assertEqual(direct.summary()["stored_sha256"], split.summary()["stored_sha256"])
        self.assertEqual(direct.blocks_seen, split.blocks_seen)
        self.assertEqual(direct.ticks_seen, split.ticks_seen)
        self.assertEqual(split.chunks_seen, 2 * direct.chunks_seen)
        self.assertEqual(direct.state_dict()["admission_rng_state"], split.state_dict()["admission_rng_state"])

    def test_uniform_inclusion_and_sampling_have_no_obvious_recency_or_slot_bias(self):
        inclusion = [0] * 12
        for seed in range(400):
            memory = HistoryReservoir(3, 1, seed)
            for tick in range(12):
                memory.add_block(chunk(tick, 1, batch=1, inputs=1, jobs=1, hidden=1))
            for record in memory.blocks:
                inclusion[record["start_tick"]] += 1
        # Expected inclusion 100; broad fixed tolerances detect a recency queue.
        self.assertTrue(all(60 < count < 145 for count in inclusion), inclusion)
        memory = HistoryReservoir(3, 1, 14)
        for tick in range(3):
            memory.add_block(chunk(tick, 1, batch=1, inputs=1, jobs=1, hidden=1))
        counts = [0, 0, 0]
        for _ in range(1200):
            counts[memory.sample(3)["start_tick"]] += 1
        self.assertTrue(all(330 < count < 470 for count in counts), counts)

    def test_disk_resume_continues_partial_admission_and_sample_rng_exactly(self):
        original = HistoryReservoir(3, 8, 19)
        for tick in range(0, 36, 4):
            original.add_chunk(chunk(tick))
        original.sample(36)
        saved = original.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actual-history.pt"
            torch.save(saved, path)
            restored = HistoryReservoir(3, 8, 999)
            restored.load_state_dict(torch.load(path, weights_only=False))
        saved["partial"]["observations"].zero_()
        for tick in range(36, 80, 4):
            for memory in (original, restored):
                memory.add_chunk(chunk(tick))
            self.assertEqual(block_hash(original.sample(tick + 4)), block_hash(restored.sample(tick + 4)))
        self.assertEqual(original.summary(), restored.summary())
        self.assertEqual(original.state_dict()["admission_rng_state"], restored.state_dict()["admission_rng_state"])
        self.assertEqual(original.state_dict()["sample_rng_state"], restored.state_dict()["sample_rng_state"])

    def test_capacity_zero_observes_real_history_without_retaining_or_sampling(self):
        memory = HistoryReservoir(0, 8, 3)
        for tick in range(0, 24, 4):
            memory.add_chunk(chunk(tick))
        self.assertEqual((len(memory), memory.blocks_seen, memory.chunks_seen, memory.ticks_seen), (0, 3, 6, 24))
        self.assertEqual(memory.bytes, 0)
        with self.assertRaisesRegex(ValueError, "no eligible"):
            memory.sample(24)
        restored = HistoryReservoir(0, 8)
        restored.load_state_dict(memory.state_dict())
        self.assertEqual(restored.summary(), memory.summary())

    def test_duplicate_skipped_overlong_and_nonbirth_history_rejected_without_mutation(self):
        memory = HistoryReservoir(3, 8)
        with self.assertRaisesRegex(ValueError, "birth"):
            memory.add_chunk(chunk(4))
        self.assertEqual(memory.ticks_seen, 0)
        memory.add_chunk(chunk(0))
        before = memory.summary()
        for record in (chunk(0), chunk(8), chunk(4, 8)):
            with self.subTest(start=record["start_tick"], end=record["end_tick"]), self.assertRaises(ValueError):
                memory.add_chunk(record)
        with self.assertRaisesRegex(ValueError, "partial"):
            memory.add_block(chunk(4, 8))
        self.assertEqual(memory.summary(), before)
        memory.add_chunk(chunk(4))
        self.assertEqual(memory.blocks_seen, 1)

    def test_schema_and_invalid_feedback_rejected(self):
        bad_records = []
        for key, value in (("targets", torch.zeros(4, 2, 2)),
                           ("elapsed", (0.25,)), ("end_tick", 9),
                           ("start_h", torch.zeros(3, 5)),
                           ("family_counts", {0: -1}), ("source_id", ""),
                           ("family_context", torch.full((4, 2, 2), 4, dtype=torch.long))):
            record = chunk()
            record[key] = value
            bad_records.append(record)
        for record in bad_records:
            with self.subTest(record=record.keys()), self.assertRaises(ValueError):
                validate_block(record)
        memory = HistoryReservoir(3, 8)
        memory.add_chunk(chunk())
        with self.assertRaisesRegex(ValueError, "schema"):
            memory.add_chunk(chunk(4, inputs=4))
        with self.assertRaisesRegex(ValueError, "optional"):
            combine_blocks(chunk(), chunk(4, family_context=False))

    def test_corrupt_checkpoint_counts_future_ranges_and_rngs_rejected(self):
        memory = HistoryReservoir(3, 8)
        memory.add_chunk(chunk())
        memory.add_chunk(chunk(4))
        memory.add_chunk(chunk(8))
        saved = memory.state_dict()
        alterations = (
            ("ticks_seen", 11), ("blocks_seen", 0), ("last_end_tick", 16),
            ("schema", {}), ("format", "unknown"), ("replacements", 9),
        )
        for key, value in alterations:
            state = deepcopy(saved)
            state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                HistoryReservoir(3, 8).load_state_dict(state)
        future = deepcopy(saved)
        future["blocks"][0]["start_tick"], future["blocks"][0]["end_tick"] = 8, 16
        with self.assertRaisesRegex(ValueError, "endpoint"):
            HistoryReservoir(3, 8).load_state_dict(future)
        self.assertEqual(memory.summary()["real_ticks_seen"], 12)

    def test_hash_changes_with_original_action_feedback_context_or_hidden(self):
        base = chunk()
        original = block_hash(base)
        self.assertEqual(original, block_hash(copied_block(base)))
        for key in ("observations", "targets", "actions", "start_h", "family_context"):
            altered = copied_block(base)
            altered[key].flatten()[0] += 1
            self.assertNotEqual(original, block_hash(altered), key)

    def test_checkpoint_rejects_missing_and_unknown_top_schema_fields(self):
        memory = HistoryReservoir(3, 8)
        memory.add_chunk(chunk())
        saved = memory.state_dict()
        for key in saved:
            state = deepcopy(saved)
            state.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                HistoryReservoir(3, 8).load_state_dict(state)
        state = deepcopy(saved)
        state["unexpected"] = 1
        with self.assertRaises(ValueError):
            HistoryReservoir(3, 8).load_state_dict(state)

    def test_lightweight_fingerprint_covers_content_partial_counts_and_both_rngs(self):
        memory = HistoryReservoir(3, 8, 15)
        first = memory.retention_fingerprint()
        memory.add_chunk(chunk())
        partial = memory.retention_fingerprint()
        self.assertNotEqual(first, partial)
        memory.add_chunk(chunk(4))
        full = memory.retention_fingerprint()
        self.assertNotEqual(partial, full)
        self.assertEqual(full, memory.retention_fingerprint())
        saved = memory.state_dict()
        restored = HistoryReservoir(3, 8)
        restored.load_state_dict(saved)
        self.assertEqual(full, restored.retention_fingerprint())
        memory.sample(8)
        self.assertNotEqual(full, memory.retention_fingerprint())
        for field in ("admission_rng_state", "sample_rng_state"):
            changed = deepcopy(saved)
            changed[field] = random.Random(881).getstate()
            restored.load_state_dict(changed)
            self.assertNotEqual(full, restored.retention_fingerprint())
        changed = deepcopy(saved)
        changed["blocks"][0]["actions"][0, 0, 0] += 1
        restored.load_state_dict(changed)
        self.assertNotEqual(full, restored.retention_fingerprint())
        self.assertNotEqual(full, block_hash(saved))


if __name__ == "__main__":
    unittest.main()
