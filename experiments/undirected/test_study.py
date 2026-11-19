"""Small explicit mechanical budgets; never prepare or run the formal study."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import rng_state, state_digest
from experiments.undirected import run as native
from experiments.undirected import study
from experiments.undirected.queue import atomic_json, file_sha256, ready


def toy():
    return replace(study.formal_template(), width=40, n_jobs=1, batch_size=1, initial_degree=4,
        chunk_ticks=2, main_ticks=8, warmup_ticks=4, withdrawal_ticks=4, return_ticks=2,
        macrocycle_ticks=8, block_ticks=2, replay_capacity=2, structural_window_updates=2, structural_candidates=1)


def branch(plan, **changes):
    values = dict(interface="ports", mode="undirected", credit="rtrl", structure="fixed", course="warmup", seed=0)
    values.update(changes)
    return next(item for item in plan["branches"] if all(item[key] == value for key, value in values.items()))


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        before = state_digest(rng_state())
        cls.bindings = study.birth_bindings(toy())
        cls.birth_rng_unchanged = before == state_digest(rng_state())

    def prepare_toy(self, folder, *, steps=2):
        # Every actual variant was constructed above.  Reusing those independent
        # of output paths avoids repeating 42 newborn constructions per test.
        with patch.object(study, "birth_bindings", return_value=deepcopy(self.bindings)):
            return study.prepare(Path(folder) / "evidence", Path(folder) / "raw",
                                 template=toy(), mechanical=True, evaluation_steps=steps)

    def test_formal_inventory_is_explicit_and_source_list_excludes_future_reports(self):
        config = study.formal_template()
        self.assertEqual((config.width, config.batch_size, config.n_jobs, config.birth_gain, config.chunk_ticks), (80, 8, 2, .6, 128))
        self.assertEqual((config.dtype, config.device, config.world_device, config.cpu_threads), ("float64", "cpu", "cpu", 1))
        self.assertEqual(config.total_ticks, 237568)
        self.assertEqual(config.snapshot_ticks(), (0, 65536, 131072, 196608, 229376, 237568))
        self.assertEqual(len(study.VARIANTS), 14)
        self.assertEqual(len({tuple(variant[key] for key in study.KEYS) for variant in study.VARIANTS}), 14)
        self.assertEqual(len(study._branches(config, Path("unused"))), 42)
        specs = study.evaluation_specs(config.snapshot_ticks())
        self.assertEqual(len(specs), 40)
        self.assertEqual(42 * len(specs), 1680)
        self.assertEqual({row["steps"] for row in specs}, {2048})
        files = {Path(path).relative_to(study.ROOT).as_posix() for path in study.execution_files()}
        for name in ("model", "credit", "plasticity", "run", "study", "queue", "ledger"):
            self.assertIn(f"experiments/undirected/{name}.py", files)
            self.assertIn(f"experiments/undirected/test_{name}.py", files)
        self.assertFalse(any("report" in path or "plot" in path for path in files))
        self.assertFalse(any("sensitivity_audit" in path for path in files))
        self.assertIn("all three", study.REPORTING_POLICY["endpoint_mean"])

    def test_template_does_not_implicitly_change_mechanical_or_evaluation_budget(self):
        study._template_checks(study.formal_template(), False)  # read-only formal validation
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "mechanical"):
                study.prepare(Path(folder) / "evidence", Path(folder) / "raw", template=toy())
            with patch.object(study, "birth_bindings", return_value=deepcopy(self.bindings)):
                path = study.prepare(Path(folder) / "evidence", Path(folder) / "raw", template=toy(), mechanical=True)
            plan = study._json(path)
            self.assertTrue(plan["mechanical_test"])
            self.assertEqual(plan["evaluation_steps"], 2048)  # no hidden template=>8 rule
            self.assertFalse(Path(plan["raw_root"]).exists())
            with self.assertRaises(FileExistsError):
                study.prepare(path.parent, Path(folder) / "other", template=toy(), mechanical=True)

    def test_actual_births_match_physical_function_but_preserve_real_layouts_and_courses(self):
        self.assertTrue(self.birth_rng_unchanged)
        self.assertEqual(len(self.bindings["by_branch"]), 42)
        rows = study._branches(toy(), Path("unused"))
        for seed in study.SEEDS:
            signatures = [self.bindings["by_branch"][item["id"]] for item in rows if item["seed"] == seed]
            self.assertEqual(len({row["matrix_hash"] for row in signatures}), 1)
            self.assertEqual(len({row["hidden_hash"] for row in signatures}), 1)
            self.assertEqual(len({row["function_hash"] for row in signatures}), 1)
            self.assertGreater(len({row["model_hash"] for row in signatures}), 1)
            self.assertNotEqual(self.bindings["world_by_seed_course"][f"{seed}/wide"],
                                self.bindings["world_by_seed_course"][f"{seed}/warmup"])
        with _preserve_process_state():
            for item in rows[:2]:
                trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
                self.assertEqual(study._birth_signature(trainer), self.bindings["by_branch"][item["id"]])
                with torch.no_grad():
                    trainer.model.encoder.weight[0, 0] += .1
                with self.assertRaisesRegex(ValueError, "selector"):
                    study._function_signature(trainer.model)

    def test_toy_plans_pin_sources_and_emit_empty_baseline_and_terminal_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            self.assertEqual(plan["checkpoint_ticks"], [0, 4, 6, 8, 12, 14])
            self.assertEqual(plan["expected_counts"], dict(training_runs=42, full_checkpoints=252, forks_per_branch=40, frozen_forks=1680))
            for kind in ("training", "evaluation", "baseline"):
                manifest = study._json(path.parent / f"{kind}-commands-v1.json")
                self.assertEqual(manifest["format"], "undirected-command-plan-v1")
                self.assertEqual(len(manifest["jobs"]), 0 if kind == "baseline" else 42)
                self.assertEqual(manifest["file_hashes"][str(path)], file_sha256(path))
                for job in manifest["jobs"]:
                    self.assertEqual(job["command"][0], str(study.PYTHON.resolve()))
                    self.assertEqual(job["target_ticks"], 14)
                    self.assertTrue(job["terminal"].endswith("terminal.json"))
            for item in plan["branches"]:
                _, found, _ = study.read_plan(path, item["id"])
                self.assertEqual(found, item)

    def test_three_credits_and_interfaces_resume_then_fork_frozen_complete_states(self):
        selections = (dict(interface="core", credit="recorded512"), dict(credit="rtrl"), dict(credit="rtrl_reset"))
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            evaluation_jobs = study._json(path.parent / "evaluation-commands-v1.json")["jobs"]
            for selection in selections:
                item = branch(plan, **selection)
                with self.subTest(selection=selection):
                    pending = study.train(path, item["id"], stop_at=2)
                    self.assertEqual(pending["status"], "pending")
                    out = Path(item["output"])
                    self.assertFalse((out / "terminal.json").exists())
                    partial = native.NativeTrainer.from_checkpoint(out / "checkpoint_latest.pt")
                    self.assertEqual(partial.ticks, 2)
                    terminal = study.train(path, item["id"])
                    self.assertEqual(terminal["status"], "success")
                    self.assertTrue(terminal["endpoint_comparable"])
                    self.assertEqual(set(terminal["checkpoints"]), {"0", "4", "6", "8", "12", "14"})
                    job = next(row for row in evaluation_jobs if row["id"] == item["id"])
                    self.assertTrue(ready(job))
                    result = study.evaluate(path, item["id"], out / "checkpoint_latest.pt", job["outputs"][0])
                    self.assertEqual(len(result["records"]), 40)
                    self.assertTrue(result["source_unchanged"])
                    self.assertTrue(result["endpoint_comparable"])
                    for record in result["records"]:
                        self.assertEqual(record["undirected"]["credit"], item["credit"])
                        self.assertEqual(record["fork_tick"], record["probe"]["tick"] - 1)
                        self.assertEqual((record["world_resets"], record["hidden_resets"], record["parameter_updates"]), (0, 0, 0))
                    with patch.object(native, "train", side_effect=AssertionError("completed branch restarted")):
                        self.assertEqual(study.train(path, item["id"])["status"], "success")
                    existing_hash = file_sha256(job["outputs"][0])
                    with patch.object(study, "evaluate_fork", side_effect=AssertionError("verified forks restarted")):
                        reused = study.evaluate(path, item["id"], out / "checkpoint_latest.pt", job["outputs"][0])
                    self.assertEqual(reused["result_hash"], result["result_hash"])
                    self.assertEqual(file_sha256(job["outputs"][0]), existing_hash)

    def test_numerical_failure_is_terminal_and_evaluates_to_empty_records_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            item = branch(plan)
            original = native.NativeTrainer.train_update

            def fail_after_one(trainer):
                if trainer.updates == 1:
                    raise FloatingPointError("nonfinite mechanical-test sensitivity")
                return original(trainer)

            with patch.object(native.NativeTrainer, "train_update", fail_after_one):
                terminal = study.train(path, item["id"])
            self.assertEqual(terminal["status"], "numerical_failure")
            self.assertEqual((terminal["last_tick"], terminal["target_ticks"]), (2, 14))
            self.assertFalse(terminal["endpoint_comparable"])
            out = Path(item["output"])
            last = native.NativeTrainer.from_checkpoint(out / "checkpoint_last_complete.pt")
            self.assertEqual(last.ticks, 2)
            self.assertEqual(last.credit.updates, 1)
            job = next(row for row in study._json(path.parent / "evaluation-commands-v1.json")["jobs"] if row["id"] == item["id"])
            self.assertTrue(ready(job))
            result = study.evaluate(path, item["id"], out / "checkpoint_latest.pt", job["outputs"][0])
            self.assertEqual(result["records"], [])
            self.assertEqual(result["states"], {})
            self.assertEqual(len(result["omitted_specs"]), 40)
            self.assertFalse(result["endpoint_comparable"])
            self.assertNotIn("score", result)
            damaged = deepcopy(result)
            damaged["records"] = [{"score": 0}]
            atomic_json(job["outputs"][0], damaged)
            with self.assertRaisesRegex(ValueError, "content hash"):
                study.evaluate(path, item["id"], out / "checkpoint_latest.pt", job["outputs"][0])
            with patch.object(native, "train", side_effect=AssertionError("numerically failed branch retried")):
                self.assertEqual(study.train(path, item["id"])["status"], "numerical_failure")

    def test_execution_failure_remains_nonterminal_and_resumes_last_complete_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            item = branch(plan, credit="tbptt128")
            out = Path(item["output"])
            original = native.NativeTrainer.train_update

            def execution_error(trainer):
                if trainer.updates == 1:
                    raise RuntimeError("mechanical transient worker unavailable")
                return original(trainer)

            with patch.object(native.NativeTrainer, "train_update", execution_error), self.assertRaises(RuntimeError):
                study.train(path, item["id"])
            self.assertFalse((out / "terminal.json").exists())
            self.assertEqual(study._json(out / "failure.json")["status"], "execution_failure")
            self.assertEqual(native.NativeTrainer.from_checkpoint(out / "checkpoint_last_complete.pt").ticks, 2)
            self.assertEqual(study.train(path, item["id"])["status"], "success")

    def test_structural_pending_resume_preserves_the_next_real_probe_and_full_credit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            item = branch(plan, structure="both")
            out = Path(item["output"])
            study.train(path, item["id"], stop_at=4)
            partial = native.NativeTrainer.from_checkpoint(out / "checkpoint_latest.pt")
            self.assertIsNotNone(partial.operator.pending)
            self.assertEqual(partial.credit.updates, 2)
            with _preserve_process_state():
                expected = native.NativeTrainer.from_state_dict(partial.state_dict())
                expected.run_updates(expected.config.total_updates - expected.updates)
                expected_hash = state_digest(expected.semantic_state())
                expected_probes = expected.work["structural_probe_calls"]
            study.train(path, item["id"])
            actual = native.NativeTrainer.from_checkpoint(out / "checkpoint_latest.pt")
            self.assertEqual(state_digest(actual.semantic_state()), expected_hash)
            self.assertEqual(actual.work["structural_probe_calls"], expected_probes)
            self.assertGreater(expected_probes, 0)

    def test_final_callback_failure_recovers_last_record_and_promotes_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            item = branch(plan, credit="tbptt128")
            out = Path(item["output"])
            append = study.LiveLedger.append_payload

            def fail_final(writer, payload):
                if payload["update_age"] == 7:
                    raise OSError("mechanical final ledger callback unavailable")
                return append(writer, payload)

            with patch.object(study.LiveLedger, "append_payload", fail_final), self.assertRaises(OSError):
                study.train(path, item["id"])
            self.assertFalse((out / "terminal.json").exists())
            self.assertEqual(native.NativeTrainer.from_checkpoint(out / "checkpoint_last_complete.pt").ticks, 14)
            self.assertEqual(len(study.read_ledger(out / "live_records")["entries"]), 6)
            terminal = study.train(path, item["id"])
            self.assertEqual(terminal["status"], "success")
            self.assertEqual(terminal["ledger"]["records"], 7)
            self.assertEqual(native.NativeTrainer.from_checkpoint(out / "checkpoint_latest.pt").ticks, 14)

    def test_foreign_output_source_config_protocol_and_snapshot_bytes_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            plan = study._json(path)
            item = branch(plan)
            with self.assertRaises(ValueError):
                study.read_plan(path, "foreign")
            with patch.object(study, "execution_files", return_value={}):
                with self.assertRaises(ValueError):
                    study.read_plan(path, item["id"])
            damaged = deepcopy(plan)
            damaged["branches"][0]["native_config"]["lr"] *= 2
            atomic_json(path, damaged)
            with self.assertRaises(ValueError):
                study.read_plan(path, item["id"])
            atomic_json(path, plan)
            study.train(path, item["id"], stop_at=4)
            damaged = deepcopy(plan)
            damaged["created_utc"] = "amended"
            atomic_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "start binding"):
                study.train(path, item["id"])
            atomic_json(path, plan)
            cp = Path(item["output"]) / "checkpoints/tick_000004.pt"
            cp.write_bytes(b"foreign snapshot bytes")
            with self.assertRaisesRegex(ValueError, "snapshot"):
                study.train(path, item["id"])
        with tempfile.TemporaryDirectory() as folder:
            path = self.prepare_toy(folder)
            item = branch(study._json(path))
            out = Path(item["output"])
            out.mkdir(parents=True)
            (out / "foreign.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unbound"):
                study.train(path, item["id"])
            self.assertEqual((out / "foreign.txt").read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
