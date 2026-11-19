"""Tiny local producer/cache checks; no formal study inputs are read."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.undirected import report, report_cache, study


class ReportCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        template = replace(study.formal_template(), width=40, n_jobs=1, batch_size=1, initial_degree=4,
            chunk_ticks=2, main_ticks=8, warmup_ticks=4, withdrawal_ticks=4, return_ticks=2,
            macrocycle_ticks=8, block_ticks=2, replay_capacity=2, structural_window_updates=2, structural_candidates=1)
        cls.protocol = study.prepare(cls.root / "evidence", cls.root / "raw", template=template,
                                     mechanical=True, evaluation_steps=2)
        cls.plan, cls.binding = report_cache._binding(cls.protocol, True, None)
        cls.item = cls.plan["branches"][0]
        study.train(cls.protocol, cls.item["id"])
        cls.evaluation = Path(cls.plan["raw_root"]) / "evaluation" / f"{cls.item['id']}.json"
        study.evaluate(cls.protocol, cls.item["id"], Path(cls.item["output"]) / "checkpoint_latest.pt", cls.evaluation)
        cls.result = report.audit_branch(cls.protocol, cls.item["id"], mechanical=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def new_cache(self):
        cache = Path(tempfile.mkdtemp(dir=self.root))
        report_cache._store(cache, self.item, self.binding, self.result)
        return cache

    def test_actual_branch_roundtrip_and_resume_never_replays(self):
        cache = self.new_cache()
        restored, artifacts = report_cache._load(cache, self.item, self.binding)
        self.assertEqual(restored, self.result)
        self.assertEqual(len(artifacts), 2)
        with patch.object(report, "audit_branch", side_effect=AssertionError("replay forbidden")):
            reused = report_cache._worker(cache, self.item, self.binding)
        self.assertEqual(reused, dict(id=self.item["id"], evidence="revalidated_cache"))
        sample = {3: (1, {"dict": [True, None]}), "tuple": (2.,)}
        self.assertEqual(report_cache._decode(report_cache._encode(sample)), sample)

    def test_changed_raw_input_or_producer_hash_is_rejected(self):
        cache = self.new_cache()
        original = self.evaluation.read_bytes()
        try:
            self.evaluation.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "inventory differs|changed"):
                report_cache._load(cache, self.item, self.binding)
        finally:
            self.evaluation.write_bytes(original)
        altered = deepcopy(self.binding)
        path = str(Path(report.__file__).resolve())
        altered["source_files"][path] = "0" * 64
        with self.assertRaises(ValueError):
            report_cache._verify_binding(altered)
        with self.assertRaisesRegex(ValueError, "receipt"):
            report_cache._load(cache, self.item, altered)

    def test_artifact_and_receipt_tampering_is_rejected(self):
        cache = self.new_cache()
        receipt_path = report_cache._receipt_path(cache, self.item["id"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifact = cache / receipt["artifact"]
        original = artifact.read_bytes()
        artifact.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            report_cache._load(cache, self.item, self.binding)
        artifact.write_bytes(original)
        receipt["identifier"] = "other_branch"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt"):
            report_cache._load(cache, self.item, self.binding)

    def test_missing_raw_fingerprint_is_rejected_even_with_rewritten_artifact(self):
        cache = Path(tempfile.mkdtemp(dir=self.root))
        altered = deepcopy(self.result)
        del altered["inputs"][str(self.evaluation.resolve())]
        report_cache._store(cache, self.item, self.binding, altered)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            report_cache._load(cache, self.item, self.binding)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report_cache._json('{"a":1,"a":2}')

    def test_late_parent_failure_keeps_atomic_branches_for_retry(self):
        cache, output = self.root / "restart-cache", self.root / "restart-report"
        reduced = {**self.plan, "branches": [self.item]}
        common = (patch.object(report_cache, "_binding", return_value=(reduced, self.binding)),
                  patch.object(report_cache, "ProcessPoolExecutor", ThreadPoolExecutor))
        with common[0], common[1], patch.object(report, "build_report", side_effect=RuntimeError("late parent failure")):
            with self.assertRaisesRegex(RuntimeError, "late parent failure"):
                report_cache.run(self.protocol, output, cache, workers=1, mechanical=True,
                                 min_available_gib=0., worker_reserve_gib=.001)
        self.assertTrue(report_cache._receipt_path(cache, self.item["id"]).is_file())
        state = json.loads((cache / "cache-status.json").read_text(encoding="utf-8"))
        self.assertFalse(state["complete"])
        self.assertIn("late parent failure", state["error"])
        captured = {}

        def aggregate(*args, **kwargs):
            captured.update(kwargs)
            return dict(counts={}, audit={})

        with patch.object(report_cache, "_binding", return_value=(reduced, self.binding)), \
             patch.object(report_cache, "ProcessPoolExecutor", ThreadPoolExecutor), \
             patch.object(report, "audit_branch", side_effect=AssertionError("replay forbidden")), \
             patch.object(report, "build_report", side_effect=aggregate), \
             patch.object(report, "write_artifacts", side_effect=lambda value, output: value):
            value = report_cache.run(self.protocol, output, cache, workers=1, mechanical=True,
                                     min_available_gib=0., worker_reserve_gib=.001)
        self.assertEqual(captured["cached_results"][0]["branch"], self.result["branch"])
        self.assertEqual(value["cache_provenance"]["revalidated_cached_branches"], 1)
        self.assertEqual(value["cache_provenance"]["fresh_branch_audits"], 0)
        self.assertTrue(json.loads((cache / "cache-status.json").read_text(encoding="utf-8"))["complete"])


if __name__ == "__main__":
    unittest.main()
