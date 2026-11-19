"""Real-block persistence without learner side effects or history loss."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.run import state_digest
from experiments.undirected import run as native
from experiments.undirected.ledger import LiveLedger, content_hash, load_record, make_payload, read_ledger


def tiny():
    return native.NativeConfig(width=40, n_jobs=1, batch_size=1, initial_degree=4, credit="rtrl", course="warmup",
        chunk_ticks=2, main_ticks=8, warmup_ticks=4, withdrawal_ticks=4, return_ticks=2,
        macrocycle_ticks=8, block_ticks=2, replay_capacity=2, structural_window_updates=2, structural_candidates=1)


def binding(config):
    return dict(branch_id="mechanical", protocol_sha256="a" * 64, native_config=asdict(config))


class LiveLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_real_records_are_compressed_hashed_and_do_not_change_the_learner(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny())
            writer = LiveLedger(Path(folder) / "live_records", binding(trainer.config))
            for _ in range(3):
                trainer.train_update()
                before = state_digest(trainer.semantic_state())
                entry = writer(trainer)
                self.assertEqual(before, state_digest(trainer.semantic_state()))
                payload = load_record(writer.folder, entry, writer.binding_hash)
                self.assertEqual(content_hash(payload["record"]), trainer.trace[-1]["credit"]["record_hash"])
                self.assertTrue(torch.equal(payload["live"]["actions"], trainer.last_chunk["actions"]))
                self.assertTrue(torch.equal(payload["live"]["family_context"], trainer.last_chunk["family_context"]))
                self.assertNotIn("sensitivity", payload)
                self.assertEqual((writer.folder / entry["file"]).read_bytes()[4:8], b"\0\0\0\0")
            state = read_ledger(writer.folder, binding(trainer.config))
            self.assertEqual([entry["update_age"] for entry in state["entries"]], [1, 2, 3])
            self.assertEqual(writer.summary(updates=3, require_exact=True)["last_tick"], 6)
            before = (writer.folder / "manifest.jsonl").read_bytes()
            writer(trainer)
            self.assertEqual(before, (writer.folder / "manifest.jsonl").read_bytes())
            payload["record"]["end_h"].add_(1.)
            self.assertTrue(torch.equal(trainer.credit.tape[-1]["end_h"], trainer.h.cpu()))

    def test_gzip_and_manifest_are_reproducible_and_orphan_record_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny())
            trainer.train_update()
            first = LiveLedger(Path(folder) / "first", binding(trainer.config))
            entry = first(trainer)
            second = LiveLedger(Path(folder) / "second", binding(trainer.config))
            orphan = second.folder / entry["file"]
            orphan.write_bytes((first.folder / entry["file"]).read_bytes())
            self.assertEqual(second(trainer), entry)
            self.assertEqual((first.folder / "manifest.jsonl").read_bytes(), (second.folder / "manifest.jsonl").read_bytes())
            self.assertEqual((first.folder / entry["file"]).read_bytes(), orphan.read_bytes())

    def test_ahead_of_latest_records_are_verified_and_reused_after_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny())
            writer = LiveLedger(Path(folder) / "live_records", binding(trainer.config))
            trainer.train_update()
            writer(trainer)
            checkpoint = trainer.state_dict()
            for _ in range(2):
                trainer.train_update()
                writer(trainer)
            manifest = (writer.folder / "manifest.jsonl").read_bytes()
            restored = native.NativeTrainer.from_state_dict(checkpoint)
            resumed_writer = LiveLedger(writer.folder, binding(restored.config))
            resumed_writer(restored)
            for _ in range(2):
                restored.train_update()
                resumed_writer(restored)
            self.assertEqual(manifest, (writer.folder / "manifest.jsonl").read_bytes())
            self.assertEqual(state_digest(trainer.semantic_state()), state_digest(restored.semantic_state()))
            resumed_writer.summary(updates=3, require_exact=True)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                resumed_writer.summary(updates=2, require_exact=True)

    def test_missing_past_records_cannot_be_recreated_from_only_the_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny())
            trainer.run_updates(2)
            writer = LiveLedger(Path(folder) / "live_records", binding(trainer.config))
            with self.assertRaisesRegex(ValueError, "past live records"):
                writer(trainer)

    def test_foreign_binding_tampered_payload_bytes_and_manifest_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny())
            trainer.train_update()
            writer = LiveLedger(Path(folder) / "records", binding(trainer.config))
            entry = writer(trainer)
            with self.assertRaisesRegex(ValueError, "binding"):
                LiveLedger(writer.folder, {"branch_id": "foreign"})
            payload = make_payload(trainer, writer.binding_hash)
            payload["record"]["observations"].add_(.1)
            with self.assertRaisesRegex(ValueError, "trace"):
                writer.append_payload(payload)
            record_path = writer.folder / entry["file"]
            original = record_path.read_bytes()
            record_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                read_ledger(writer.folder)
            record_path.write_bytes(original)
            manifest = writer.folder / "manifest.jsonl"
            line = json.loads(manifest.read_text())
            line["update_age"] = 2
            manifest.write_text(json.dumps(line) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest"):
                read_ledger(writer.folder)

    def test_callback_io_failure_preserves_complete_checkpoint_and_resume_fills_last_record(self):
        with tempfile.TemporaryDirectory() as folder:
            out, config = Path(folder) / "run", tiny()
            trainer = native.NativeTrainer(config)
            trainer.save_checkpoint(out / "checkpoint_latest.pt")
            writer = LiveLedger(out / "live_records", binding(config))
            append = writer.append_payload

            def fail_second(payload):
                if payload["update_age"] == 2:
                    raise OSError("mechanical ledger write unavailable")
                return append(payload)

            with patch.object(writer, "append_payload", fail_second), self.assertRaises(OSError):
                native.train(config, out, resume=out / "checkpoint_latest.pt", stop_at=4, on_update=writer)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["status"], "execution_failure")
            self.assertEqual(failure["last_complete_tick"], 4)
            self.assertEqual(len(writer.entries), 1)
            restored = native.train(config, out, resume=out / "checkpoint_last_complete.pt", stop_at=4, on_update=writer)
            self.assertEqual(restored.ticks, 4)
            self.assertEqual(len(writer.entries), 2)
            writer.summary(updates=2, require_exact=True)


if __name__ == "__main__":
    unittest.main()
