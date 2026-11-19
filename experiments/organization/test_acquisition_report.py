"""Small real lifetimes and frozen probes; no formal-score-based choices."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong import evaluate as observer, run as native
from experiments.loops.report import AuditError, _Checks
from experiments.organization import acquisition as acq
from experiments.organization import acquisition_report as report


def tiny():
    return native.NativeConfig(width=8, batch_size=2, chunk_ticks=4, initial_fan_in=2,
        replay_capacity=2, block_ticks=4, macrocycle_ticks=16, main_ticks=16,
        withdrawal_ticks=8, return_ticks=4, device="cpu")


def real_fixture(folder):
    folder = Path(folder)
    evidence, old_raw = folder / "old-evidence", folder / "old-raw"
    aggregate = evidence / "aggregate-v1"
    aggregate.mkdir(parents=True)
    config, ticks = tiny(), [0, 16, 24, 28]
    specs = acq.evaluation_specs(ticks, 8)
    cp_rows, matched, bindings, old_inputs = [], [], {}, {}
    for seed in (0, 1, 2):
        for variant in acq.VARIANTS:
            label = "__".join(variant[k] for k in acq.KEYS)
            identifier = f"seed_{seed:03d}__{label}"
            out = old_raw / "training" / f"seed_{seed:03d}" / label
            trainer = native.NativeTrainer(replace(config, seed=seed, **variant))
            item = dict(id=identifier, seed=seed, **variant)
            bindings[identifier], records = {}, []
            for tick in ticks:
                trainer.run_updates((tick - trainer.ticks) // config.chunk_ticks)
                cp = out / "checkpoints" / f"tick_{tick:06d}.pt"
                trainer.save_checkpoint(cp)
                result = report.legacy.audit_checkpoint_payload(trainer.state_dict(), tick=tick)
                result.pop("report")
                result.pop("checks")
                result.update(checkpoint=str(cp), checkpoint_sha256=report.legacy.file_sha256(cp))
                cp_rows.append(item | result)
                old_inputs[str(cp)] = result["checkpoint_sha256"]
                bindings[identifier][str(tick)] = {key: result[key] for key in report.BASELINE_ENTRY_FIELDS}
                for spec in (s for s in specs if s["tick"] == tick):
                    record = observer.evaluate(trainer.model, trainer.world, trainer.h, family=spec["family"],
                                               condition=spec["condition"], steps=8, device="cpu")
                    record["probe"] = dict(spec, purpose="mechanical")
                    records.append(record)
                    stub = dict(sources=acq.execution_files(), dt=.25, batch_size=2)
                    metrics = report.legacy.audit_native_record(record, result, plan=stub, spec=record["probe"])
                    matched.append(item | spec | dict(purpose="mechanical", probe_origin="primary",
                        primary=record["primary"], primary_censored=record["censored"]["primary"]["count"],
                        actual_family_exposures=record["actual_family_exposures"], source_family_recency=result["family_recency"],
                        source_clock=report.legacy.family_clock_metrics(result, spec["family"]),
                        within_training_support=report.legacy.within_training_support(variant["coverage"], spec["condition"]),
                        query_role="withheld_family" if spec["family"] == 0 else "other_family") | metrics)
            path = old_raw / "evaluation" / f"{identifier}.json"
            report.legacy._atomic_json(path, dict(format="lifelong-evaluation-v1", branch=item,
                                                 source_unchanged=True, records=records))
            old_inputs[str(path)] = report.legacy.file_sha256(path)
    report.legacy._csv(aggregate / "checkpoints.csv", cp_rows)
    report.legacy._csv(aggregate / "matched_memory_per_seed.csv", matched)
    report.legacy._csv(aggregate / "per_seed.csv", matched)
    report.legacy._csv(aggregate / "supplement_per_seed.csv", [])
    report.legacy._atomic_json(aggregate / "report.json", dict(inputs=old_inputs))
    artifacts = {name: report.legacy.file_sha256(aggregate / name) for name in (
        "report.json", "checkpoints.csv", "matched_memory_per_seed.csv", "per_seed.csv", "supplement_per_seed.csv")}
    report.legacy._atomic_json(aggregate / "artifacts.json", artifacts)
    report.legacy._atomic_json(evidence / "completion.json", dict(complete=True,
        audit=dict(status="passed", errors=[]), aggregate_artifacts=artifacts))
    files = {str(path): report.legacy.file_sha256(path) for path in [evidence / "completion.json", aggregate / "artifacts.json"] +
             [aggregate / name for name in ("checkpoints.csv", "matched_memory_per_seed.csv", "per_seed.csv", "supplement_per_seed.csv")]}
    baseline = dict(files=files, branches=bindings, shrink_step=2e-4, note="Immutable mechanical baseline.")
    path = acq.prepare(folder / "new-evidence", folder / "new-raw", template=config, baseline=baseline)
    plan = acq._json(path)
    for item in plan["branches"]:
        acq.train(path, item["id"])
        acq.evaluate(path, item["id"], Path(item["output"]) / "latest.pt",
                     folder / "new-raw/evaluation" / f"{item['id']}.json")
    return path, plan


class AcquisitionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        with _preserve_process_state():
            cls.path, cls.plan = real_fixture(cls.folder.name)
        cls.root = Path(cls.plan["raw_root"])

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_real_full_lifetimes_restore_and_paired_audit(self):
        value = report.build_report(self.path, self.root, mechanical=True, workers=1)
        self.assertTrue(value["complete"])
        self.assertEqual(value["audit"]["status"], "passed")
        self.assertEqual(value["counts"]["new_training_runs"], 9)
        self.assertEqual(value["counts"]["new_full_checkpoints"], 36)
        self.assertEqual(value["counts"]["new_evaluation_forks"], 216)
        self.assertEqual(value["counts"]["borrowed_old_evaluation_forks"], 216)
        self.assertEqual(len(value["paired_per_seed"]), 216)
        self.assertEqual(len(value["score_denominators"]), 432)
        self.assertTrue(all(r["equal"] for r in value["birth_equality"]))
        self.assertEqual({r["phase"] for r in value["per_seed"]}, {"main_acquisition", "post_withdrawal", "after_return"})
        self.assertTrue(all(m["n_total"] == 3 for r in value["paired_summary"] for m in r["metrics"].values()))
        destination = Path(self.folder.name) / "result-artifacts"
        files = report.write_artifacts(value, destination)
        self.assertIn("paired_summary.csv", files)
        with self.assertRaises(FileExistsError):
            report.write_artifacts(value, destination)

    def test_formal_report_rejects_toy_budgets(self):
        with self.assertRaisesRegex(AuditError, "FORMAL_NATIVE_DEFAULTS"):
            report.build_report(self.path, self.root)

    def test_real_nine_branch_serial_and_two_workers_are_identical(self):
        before = {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        serial = report.build_report(self.path, self.root, mechanical=True, workers=1)
        parallel = report.build_report(self.path, self.root, mechanical=True, workers=2)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial["inputs"], parallel["inputs"])
        self.assertEqual(serial["audit"], parallel["audit"])
        self.assertEqual(before, {name: os.environ.get(name) for name in before})

    def test_worker_failure_propagates_and_never_returns_partial_branches(self):
        plan = deepcopy(self.plan)
        plan["branches"][0]["id"] = "foreign-branch"
        with self.assertRaisesRegex(AuditError, "NEW_COMPLETION_BINDING"):
            report._new_branch_audits(plan, report.legacy.file_sha256(self.path), 2, {}, _Checks(), {})

    def test_worker_boundaries_reject_bool_zero_and_excessive_parallelism(self):
        for value in (True, 0, 7, 1.5):
            with self.subTest(workers=value), self.assertRaises(ValueError):
                report.build_report(self.path, self.root, mechanical=True, workers=value)

    def test_exact_configuration_rejects_lr_or_shrink_forgery(self):
        item = self.plan["branches"][0]
        cp = Path(item["output"]) / "checkpoints/tick_000000.pt"
        state = torch.load(cp, map_location="cpu", weights_only=False)
        for key, value in (("lr", .002), ("shrink_step", 2e-4)):
            changed = deepcopy(state)
            changed["config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(AuditError, "FULL_ACQUISITION_CONFIG"):
                report._config(changed, item, _Checks(), "changed", shrink=0.)
        changed = deepcopy(state)
        changed["operator"]["config"]["shrink_step"] = 2e-4
        with self.assertRaisesRegex(AuditError, "ACTUAL_ACQUISITION_SHRINK"):
            report._config(changed, item, _Checks(), "changed", shrink=0.)

    def test_source_inventory_and_protocol_branch_rejected(self):
        original = self.path.read_bytes()
        try:
            plan = deepcopy(self.plan)
            plan["sources"][next(iter(plan["sources"]))] = "0" * 64
            report.legacy._atomic_json(self.path, plan)
            with self.assertRaises((AuditError, ValueError)):
                report._protocol(self.path, self.root, _Checks(), {}, True)
        finally:
            self.path.write_bytes(original)

    def test_missing_or_duplicate_new_fork_rejected_before_statistics(self):
        item = self.plan["branches"][0]
        cp = report._new_checkpoints(self.plan, item, report.legacy.file_sha256(self.path), {}, _Checks())
        path = self.root / "evaluation" / f"{item['id']}.json"
        original, value = path.read_bytes(), acq._json(path)
        try:
            changed = deepcopy(value)
            changed["records"].pop()
            report.legacy._atomic_json(path, changed)
            with self.assertRaisesRegex(AuditError, "NEW_FORK_COUNT"):
                report._new_evaluation(self.plan, item, report.legacy.file_sha256(self.path), cp, {}, _Checks())
            changed = deepcopy(value)
            changed["records"][1] = deepcopy(changed["records"][0])
            report.legacy._atomic_json(path, changed)
            with self.assertRaisesRegex(AuditError, "PROBE_BINDING"):
                report._new_evaluation(self.plan, item, report.legacy.file_sha256(self.path), cp, {}, _Checks())
        finally:
            path.write_bytes(original)

    def test_pairing_and_sd_use_same_seed_differences(self):
        new, old = [], []
        for seed, a, b in ((0, .6, .1), (1, .2, .2), (2, .4, .3)):
            meta = dict(seed=seed, architecture="fixed_shrink", schedule="mixed", learning="history_replay",
                coverage="wide", tick=16, phase="main_acquisition", family=0, condition="short", steps=8,
                primary=dict(matured=10 + seed), primary_censored=seed)
            new.append(meta | dict(metrics=dict(strict_completion=a, undefined=None)))
            old.append(meta | dict(metrics=dict(strict_completion=b, undefined=None)))
        paired = report.paired_rows(new, old)
        summary = report.summaries(paired, paired=True)[0]["metrics"]
        self.assertAlmostEqual(summary["strict_completion"]["mean"], .2)
        self.assertAlmostEqual(summary["strict_completion"]["sample_sd"], (0.07) ** .5)
        self.assertEqual(summary["undefined"]["n_total"], 3)
        self.assertEqual(summary["undefined"]["n_undefined"], 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report.paired_rows(new + [new[0]], old)
        with self.assertRaisesRegex(ValueError, "missing"):
            report.paired_rows(new, old[:-1])

    def test_baseline_file_and_raw_bytes_are_bound(self):
        old_cps, csv_path, old_inputs = report._baseline(self.plan, {}, _Checks(), True)
        self.assertEqual(len(old_cps), 9)
        cp = Path(next(iter(old_cps.values()))[0]["checkpoint"])
        original = cp.read_bytes()
        try:
            cp.write_bytes(original + b"changed")
            with self.assertRaisesRegex(AuditError, "ACQUISITION_INPUT_SHA256"):
                report._baseline(self.plan, {}, _Checks(), True)
        finally:
            cp.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
