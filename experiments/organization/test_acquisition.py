"""Finite acquisition checks: zero shrink, matched birth, provenance and forks."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.run import atomic_json, state_digest
from experiments.lifelong import run as native
from experiments.lifelong.queue import ready
from experiments.organization import acquisition as acq


def tiny(**changes):
    config = native.NativeConfig(width=8, batch_size=2, chunk_ticks=4, initial_fan_in=2,
        replay_capacity=2, block_ticks=4, macrocycle_ticks=16, main_ticks=16,
        withdrawal_ticks=8, return_ticks=4, device="cpu")
    return replace(config, **changes)


def toy_plan(folder):
    config = tiny()
    bindings = {}
    for seed in (0, 1, 2):
        for variant in acq.VARIANTS:
            original = native.NativeTrainer(replace(config, seed=seed, **variant))
            old_id = f"seed_{seed:03d}__" + "__".join(variant[k] for k in acq.KEYS)
            bindings[old_id] = {"0": dict(initial_parameter_hash=original.initial_parameter_hash,
                                          birth_hash=state_digest(original.birth))}
    return acq.prepare(Path(folder) / "evidence", Path(folder) / "raw", template=config,
                       baseline=dict(files={}, branches=bindings, shrink_step=2e-4))


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_config_changes_only_declared_choices_and_shrink(self):
        with tempfile.TemporaryDirectory() as folder:
            old = {f"seed_{s:03d}__" + "__".join(v[k] for k in acq.KEYS): {}
                   for s in (0, 1, 2) for v in acq.VARIANTS}
            with patch.object(acq, "baseline_bindings", return_value=dict(files={}, branches=old, shrink_step=2e-4)):
                path = acq.prepare(Path(folder) / "e", Path(folder) / "raw")
            plan = acq._json(path)
            self.assertEqual(plan["checkpoint_ticks"], [0, 131072, 163840, 172032])
            self.assertEqual(plan["latest_interval_ticks"], 32768)
            self.assertEqual(len(plan["evaluation_specs"]), 24)
            self.assertEqual(len(plan["branches"]), 9)
            self.assertEqual(plan["threads"], 1)
            for item in plan["branches"]:
                expected = native.NativeConfig(seed=item["seed"], **{k: item[k] for k in acq.KEYS},
                                               device="cuda", shrink_step=0.)
                self.assertEqual(item["native_config"], acq._normalized(asdict(expected)))
                self.assertEqual(item["shrink_step"], 0.)
            for kind in ("training", "evaluation", "baseline"):
                manifest = acq._json(path.parent / f"acquisition-{kind}-commands-v1.json")
                self.assertEqual(len(manifest["jobs"]), 0 if kind == "baseline" else 9)
            pinned = set(plan["sources"])
            self.assertIn(str(Path(acq.__file__).resolve()), pinned)
            self.assertNotIn(str(Path(acq.__file__).with_name("model.py").resolve()), pinned)

    def test_zero_shrink_operator_really_preserves_small_allowed_weights(self):
        trainer = native.NativeTrainer(tiny(shrink_step=0.))
        with torch.no_grad():
            trainer.model.recurrent[trainer.operator.allowed_mask] = 1e-5
        before = trainer.model.recurrent.detach().clone()
        trainer.operator.apply_after_optimizer(trainer.model.recurrent, {}, torch.zeros_like(before),
                                               1., .001, (.9, .999), 1e-8)
        self.assertTrue(torch.equal(before, trainer.model.recurrent))
        self.assertEqual(trainer.operator.config.effective_shrink_step, 0.)
        self.assertEqual(trainer.operator.summary()["shrink_step"], 0.)

    def test_birth_matches_original_same_seed_across_variants(self):
        for seed in (0, 1, 2):
            for variant in acq.VARIANTS:
                with self.subTest(seed=seed, variant=variant):
                    original = native.NativeTrainer(tiny(seed=seed, **variant))
                    zero = native.NativeTrainer(tiny(seed=seed, **variant, shrink_step=0.))
                    self.assertEqual(original.initial_parameter_hash, zero.initial_parameter_hash)
                    self.assertEqual(state_digest(original.birth), state_digest(zero.birth))
                    self.assertEqual(state_digest(original.world.state_dict()), state_digest(zero.world.state_dict()))

    def test_checkpoint_restores_exact_next_update(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = native.NativeTrainer(tiny(shrink_step=0., schedule="mixed", learning="history_replay"))
            trainer.run_updates(3)
            cp = Path(folder) / "checkpoint.pt"
            trainer.save_checkpoint(cp)
            trainer.train_update()
            expected = state_digest(trainer.semantic_state())
            restored = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            restored.train_update()
            self.assertEqual(state_digest(restored.semantic_state()), expected)
            self.assertEqual(restored.hidden_resets, 0)

    def test_toy_train_resume_completion_and_unreset_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            item = acq._json(path)["branches"][0]
            acq.train(path, item["id"], stop_at=8)
            out = Path(item["output"])
            partial = native.NativeTrainer.from_checkpoint(out / "latest.pt")
            self.assertEqual(partial.ticks, 8)
            self.assertFalse((out / "launch-complete.json").exists())
            acq.train(path, item["id"])
            done = acq._json(out / "launch-complete.json")
            self.assertEqual(set(done["checkpoints"]), {"0", "16", "24", "28"})
            self.assertEqual(done["birth_hash"], state_digest(partial.birth))
            manifest = acq._json(path.parent / "acquisition-evaluation-commands-v1.json")
            job = next(j for j in manifest["jobs"] if j["id"] == item["id"])
            self.assertTrue(ready(job))
            output = Path(job["outputs"][0])
            acq.evaluate(path, item["id"], out / "latest.pt", output)
            value = acq._json(output)
            self.assertEqual(len(value["records"]), 24)
            self.assertTrue(value["source_unchanged"])
            for record in value["records"]:
                self.assertEqual(record["fork_tick"], record["probe"]["tick"] - 1)
                self.assertEqual(record["world_resets"], 0)
                self.assertEqual(record["hidden_resets"], 0)
                self.assertEqual(record["parameter_updates"], 0)
                cp = native.NativeTrainer.from_checkpoint(done["checkpoints"][str(record["probe"]["tick"])]["path"])
                self.assertEqual(record["start_hidden_hash"], state_digest({"hidden": cp.h}))
                self.assertEqual(record["start_world_state_hash"], state_digest(cp.world.state_dict()))
            with self.assertRaises(ValueError):
                acq.evaluate(path, item["id"], out / "latest.pt", output)

    def test_refuses_foreign_directory_checkpoint_and_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = acq._json(path)
            item = plan["branches"][0]
            out = Path(item["output"])
            out.mkdir(parents=True)
            sentinel = out / "belongs-to-another-run.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                acq.train(path, item["id"])
            self.assertEqual(sentinel.read_text(), "keep")
            with self.assertRaises(ValueError):
                acq.read_plan(path, "foreign")
            trainer = native.NativeTrainer(tiny(shrink_step=0., seed=5, **acq.VARIANTS[0]))
            with self.assertRaises(ValueError):
                acq.validate(trainer, plan, item)
            with self.assertRaises(FileExistsError):
                acq.prepare(path.parent, Path(folder) / "another-raw", template=tiny(), baseline=plan["baseline"])

    def test_rejects_protocol_change_and_foreign_native_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan, item, sha = acq.read_plan(path, acq._json(path)["branches"][0]["id"])
            modified = deepcopy(plan)
            modified["branches"][0]["native_config"]["lr"] *= 2
            atomic_json(path, modified)
            with self.assertRaises(ValueError):
                acq.read_plan(path, item["id"])
            atomic_json(path, plan)
            modified = deepcopy(plan)
            modified["interpretation"] += " changed"
            atomic_json(path, modified)
            with self.assertRaises(ValueError):
                acq.verify_end(path, item["id"], sha)

    def test_resume_refuses_changed_protocol_or_checkpoint_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan = acq._json(path)
            item = plan["branches"][0]
            acq.train(path, item["id"], stop_at=4)
            plan["interpretation"] += " amended"
            atomic_json(path, plan)
            with self.assertRaises(ValueError):
                acq.train(path, item["id"])
            with patch.object(acq, "execution_files", return_value={}):
                with self.assertRaises(ValueError):
                    acq.read_plan(path, item["id"])

    def test_existing_snapshot_is_immutable_including_process_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path = toy_plan(folder)
            plan, item, _ = acq.read_plan(path, acq._json(path)["branches"][0]["id"])
            trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
            out = Path(item["output"])
            acq._snapshot(out, trainer, plan, item)
            cp = out / "checkpoints/tick_000000.pt"
            old_bytes = cp.read_bytes()
            torch.rand(1)
            with self.assertRaises(ValueError):
                acq._snapshot(out, trainer, plan, item)
            self.assertEqual(cp.read_bytes(), old_bytes)


if __name__ == "__main__":
    unittest.main()
