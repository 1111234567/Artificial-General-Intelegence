"""Finite organization protocol, matched newborns, continuation and provenance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, rng_state, state_digest
from experiments.credit.run import file_sha256
from experiments.lifelong.queue import ready
from experiments.organization import run as native
from experiments.organization import study


def tiny(**changes):
    config = native.NativeConfig(width=8, batch_size=2, chunk_ticks=4, initial_fan_in=2,
        replay_capacity=2, block_ticks=4, macrocycle_ticks=16, main_ticks=32,
        withdrawal_ticks=8, return_ticks=8, device="cpu")
    return replace(config, **changes)


def toy_plan(folder):
    return study.prepare(Path(folder) / "evidence", Path(folder) / "raw", template=tiny())


def branch(plan, field="plain", rule="passive", seed=0):
    return next(item for item in plan["branches"] if item["seed"] == seed
                and item["field_mode"] == field and item["structural_rule"] == rule)


class OrganizationStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_cpu_device_does_not_select_mechanical_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = study.prepare(root / "evidence", root / "raw", training_device="cpu")
            plan = study._json(path)
            self.assertIs(plan["mechanical_test"], False)
            self.assertEqual(plan["training_device"], "cpu")
            self.assertEqual(plan["target_ticks"], 172032)
            self.assertEqual(plan["evaluation_steps"], 2048)
            self.assertEqual(plan["latest_interval_ticks"], 32768)
            self.assertEqual(plan["native_template"]["batch_size"], 32)
            self.assertEqual(plan["native_template"]["width"], 128)
            self.assertTrue(all(spec["steps"] == 2048 for spec in plan["evaluation_specs"]))
            loaded, _, _ = study.read_plan(path, plan["branches"][0]["id"])
            self.assertEqual(loaded, plan)

    def test_formal_matrix_budget_and_explicit_source_inventory(self):
        config = native.NativeConfig(device="cuda")
        study._template_checks(config)
        self.assertEqual(config.total_ticks, 172032)
        self.assertEqual(config.checkpoint_ticks, (0, 131072, 163840, 172032))
        self.assertEqual((config.structural_candidates, config.structural_window_updates), (4, 8))
        rows = study._branches(config, study.DEFAULT_RAW)
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({item["id"] for item in rows}), 27)
        self.assertEqual(len(study.VARIANTS), 9)
        self.assertEqual(sum(v["structural_rule"] == "passive" for v in study.VARIANTS), 3)
        self.assertEqual(sum(v["field_mode"] == "plain" and v["structural_rule"] != "passive"
                             for v in study.VARIANTS), 4)
        specs = study.evaluation_specs(config.checkpoint_ticks)
        self.assertEqual(len(specs), 24)
        self.assertEqual(27 * len(specs), 648)
        self.assertEqual({row["steps"] for row in specs}, {2048})
        self.assertNotIn(0, {row["tick"] for row in specs})
        source_names = {Path(name).relative_to(study.ROOT).as_posix() for name in study.execution_files()}
        for name in ("model.py", "test_model.py", "run.py", "test_run.py", "plasticity.py",
                     "test_plasticity.py", "test_functional_run.py", "study.py", "test_study.py"):
            self.assertIn("experiments/organization/" + name, source_names)
        self.assertFalse(any("acquisition" in name or "report" in name for name in source_names))

    def test_toy_prepare_checks_all_actual_births_and_emits_queue_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            before = state_digest(rng_state())
            path = toy_plan(folder)
            after = state_digest(rng_state())
            self.assertEqual(before, after)
            plan = study._json(path)
            self.assertTrue(plan["mechanical_test"])
            self.assertEqual(plan["checkpoint_ticks"], [0, 32, 40, 48])
            self.assertEqual(len(plan["branches"]), 27)
            self.assertEqual(len(plan["birth_bindings"]["branch_birth_hashes"]), 27)
            self.assertEqual(set(plan["birth_bindings"]["shared_by_seed"]), {"0", "1", "2"})
            for seed in (0, 1, 2):
                binding = plan["birth_bindings"]["shared_by_seed"][str(seed)]
                self.assertEqual(binding["initial_nonzero"], 16)
                signatures = []
                with _preserve_process_state():
                    for item in (item for item in plan["branches"] if item["seed"] == seed):
                        trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
                        study.validate(trainer, plan, item, 0)
                        signatures.append(study._birth_signature(trainer))
                self.assertTrue(all(value == binding for value in signatures))
            for kind in ("training", "evaluation", "baseline"):
                manifest = study._json(path.parent / f"organization-{kind}-commands-v1.json")
                self.assertEqual(len(manifest["jobs"]), 0 if kind == "baseline" else 27)
                self.assertEqual(manifest["file_hashes"][str(path)], file_sha256(path))
            self.assertFalse(Path(plan["raw_root"]).exists())
            for item in plan["branches"]:
                restored, found, _ = study.read_plan(path, item["id"])
                self.assertEqual(found, item)
                self.assertEqual(restored["target_ticks"], 48)

    def test_passive_three_field_train_resume_four_snapshots_and_frozen_forks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = study._json(path)
            for field in ("plain", "parallel", "coupled"):
                with self.subTest(field=field):
                    item = branch(plan, field)
                    out = Path(item["output"])
                    study.train(path, item["id"], stop_at=12)
                    partial = native.NativeTrainer.from_checkpoint(out / "latest.pt", device="cpu")
                    self.assertEqual(partial.ticks, 12)
                    self.assertFalse((out / "launch-complete.json").exists())
                    study.train(path, item["id"])
                    done = study.completion(plan, item, file_sha256(path))
                    self.assertEqual(set(done["checkpoints"]), {"0", "32", "40", "48"})
                    for tick, entry in done["checkpoints"].items():
                        saved = native.NativeTrainer.from_checkpoint(entry["path"], device="cpu")
                        study.validate(saved, plan, item, int(tick))
                        self.assertEqual(study.normalized_hash(saved, "cpu"), entry["semantic_hash"])
                        self.assertEqual(saved.hidden_resets, 0)
                    manifest = study._json(path.parent / "organization-evaluation-commands-v1.json")
                    job = next(job for job in manifest["jobs"] if job["id"] == item["id"])
                    self.assertTrue(ready(job))
                    output = Path(job["outputs"][0])
                    study.evaluate(path, item["id"], out / "latest.pt", output)
                    value = study._json(output)
                    self.assertTrue(value["source_unchanged"])
                    self.assertEqual(len(value["records"]), 24)
                    for record in value["records"]:
                        self.assertEqual(record["model"]["field_mode"], field)
                        self.assertEqual(record["organization"]["structural_rule"], "passive")
                        self.assertEqual(record["fork_tick"], record["probe"]["tick"] - 1)
                        self.assertEqual((record["world_resets"], record["hidden_resets"], record["parameter_updates"]), (0, 0, 0))
                        entry = done["checkpoints"][str(record["probe"]["tick"])]
                        saved = native.NativeTrainer.from_checkpoint(entry["path"], device="cpu")
                        self.assertEqual(record["start_hidden_hash"], state_digest({"hidden": saved.h}))
                        self.assertEqual(record["start_world_state_hash"], state_digest(saved.world.state_dict()))
                    with patch.object(native.NativeTrainer, "train_update", side_effect=AssertionError("completed run restarted")):
                        study.train(path, item["id"])
                    with self.assertRaises(ValueError):
                        study.evaluate(path, item["id"], out / "latest.pt", output)

    def test_foreign_output_config_protocol_and_source_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = study._json(path)
            item = branch(plan)
            out = Path(item["output"])
            out.mkdir(parents=True)
            sentinel = out / "owned-by-another-run.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                study.train(path, item["id"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            with self.assertRaises(ValueError):
                study.read_plan(path, "foreign")
            with self.assertRaises(FileExistsError):
                study.prepare(path.parent, Path(folder) / "newraw", template=tiny())
            with _preserve_process_state():
                foreign = native.NativeTrainer(tiny(seed=99))
                with self.assertRaises(ValueError):
                    study.validate(foreign, plan, item)
            damaged = deepcopy(plan)
            damaged["branches"][0]["native_config"]["lr"] *= 2
            atomic_json(path, damaged)
            with self.assertRaises(ValueError):
                study.read_plan(path, item["id"])
            atomic_json(path, plan)
            damaged = deepcopy(plan)
            damaged["branches"][0]["output"] = str(Path(folder) / "foreign")
            atomic_json(path, damaged)
            with self.assertRaises(ValueError):
                study.read_plan(path, item["id"])
            atomic_json(path, plan)
            with patch.object(study, "execution_files", return_value={}):
                with self.assertRaises(ValueError):
                    study.read_plan(path, item["id"])

    def test_resume_rejects_amended_protocol_and_immutable_snapshot_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan, item, sha = study.read_plan(path, branch(study._json(path))["id"])
            trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
            out = Path(item["output"])
            study._snapshot(out, trainer, plan, item)
            cp = out / "checkpoints/tick_000000.pt"
            original = cp.read_bytes()
            torch.rand(1)
            with self.assertRaises(ValueError):
                study._snapshot(out, trainer, plan, item)
            self.assertEqual(cp.read_bytes(), original)
            item = branch(plan, "parallel")
            study.train(path, item["id"], stop_at=4)
            damaged = deepcopy(plan)
            damaged["primary"] += " amended"
            atomic_json(path, damaged)
            with self.assertRaises(ValueError):
                study.train(path, item["id"])
            with self.assertRaises(ValueError):
                study.verify_end(path, item["id"], sha)

    def test_evaluation_rejects_foreign_paths_and_changed_snapshot_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = study._json(path)
            item = branch(plan, "coupled")
            study.train(path, item["id"])
            out = Path(item["output"])
            output = Path(plan["raw_root"]) / "evaluation" / f"{item['id']}.json"
            with self.assertRaises(ValueError):
                study.evaluate(path, item["id"], out / "latest.pt", Path(folder) / "foreign.json")
            with self.assertRaises(ValueError):
                study.evaluate(path, item["id"], out / "checkpoints/tick_000048.pt", output)
            cp = out / "checkpoints/tick_000040.pt"
            original = cp.read_bytes()
            try:
                cp.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "snapshot bytes"):
                    study.evaluate(path, item["id"], out / "latest.pt", output)
                self.assertFalse(output.exists())
            finally:
                cp.write_bytes(original)

    def test_completed_functional_task_keeps_its_rule_and_extra_probe_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = study._json(path)
            item = branch(plan, "plain", "functional_task")
            study.train(path, item["id"], stop_at=32)
            out = Path(item["output"])
            partial = native.NativeTrainer.from_checkpoint(out / "latest.pt", device="cpu")
            self.assertIsNotNone(partial.operator.pending)
            study.train(path, item["id"])
            done = study.completion(plan, item, file_sha256(path))
            self.assertGreater(done["work"]["structural_probe_calls"], 0)
            self.assertGreater(done["work"]["structural_probe_forward_ticks"], 0)
            final = native.NativeTrainer.from_checkpoint(out / "latest.pt", device="cpu")
            self.assertEqual(final.config.structural_rule, "functional_task")
            study.validate(final, plan, item, 48)


if __name__ == "__main__":
    unittest.main()
