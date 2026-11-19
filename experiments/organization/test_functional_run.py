"""Functional proposals must remain local to probes and survive exact resume."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.run import state_digest
from experiments.organization import run
from experiments.organization.plasticity import MODES
from experiments.organization.test_run import tiny


class FunctionalContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_birth_and_probe_budget_for_every_policy(self):
        starts = []
        work = []
        for mode in MODES:
            trainer = run.NativeTrainer(tiny(structural_rule=mode, structural_candidates=2))
            starts.append(state_digest({"model": trainer.model.state_dict(), "h": trainer.h,
                "world": trainer.world.state_dict(), "optimizer": trainer.optimizer.state_dict()}))
            trainer.run_updates(7)
            self.assertEqual(trainer.work["structural_probe_calls"], 8)
            self.assertEqual(trainer.work["structural_probe_forward_ticks"], 96)
            self.assertIsNone(trainer.operator.loss_fn)
            self.assertEqual(trainer.hidden_resets, 0)
            self.assertEqual(trainer.world.tick, 27)
            work.append(trainer.work)
        self.assertEqual(len(set(starts)), 1)
        self.assertTrue(all(item == work[0] for item in work))

    def test_pending_resume_exact_across_structural_and_schedule_boundaries(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for field in ("plain", "coupled"):
                for mode in MODES:
                    with self.subTest(device=device, field=field, mode=mode):
                        trainer = run.NativeTrainer(tiny(device=device, field_mode=field,
                            structural_rule=mode, structural_candidates=2))
                        trainer.run_updates(3)
                        self.assertIsNotNone(trainer.operator.pending)
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "pending.pt"
                            trainer.save_checkpoint(path)
                            trainer.run_updates(10)
                            expected = state_digest(trainer.semantic_state())
                            restored = run.NativeTrainer.from_checkpoint(path, device=device)
                            restored.run_updates(10)
                            self.assertEqual(state_digest(restored.semantic_state()), expected)

    def test_probe_has_no_live_or_history_side_effects(self):
        trainer = run.NativeTrainer(tiny(structural_rule="functional_task", structural_candidates=2))
        trainer.run_updates(3)
        original = trainer._structural_losses
        checks = []

        def checked(chunk):
            before = state_digest({"h": trainer.h, "world": trainer.world.state_dict(),
                "replay": trainer.replay.state_dict(), "rng": run.continuity.rng_state()})
            value = original(chunk)
            after = state_digest({"h": trainer.h, "world": trainer.world.state_dict(),
                "replay": trainer.replay.state_dict(), "rng": run.continuity.rng_state()})
            self.assertEqual(before, after)
            self.assertAlmostEqual(value["mixed"], .5 * (value["live"] + value["history"]))
            checks.append(value)
            return value

        with patch.object(trainer, "_structural_losses", side_effect=checked):
            point = trainer.train_update()
        self.assertEqual(len(checks), 4)
        sources = point["update_metrics"]["structure"]["probe_sources"]
        self.assertEqual((sources["live_start_tick"], sources["live_end_tick"]), (12, 16))
        self.assertEqual(sources["history_source_hash"], point["auxiliary"]["source_hash"])

    def test_probe_recomputes_warmup_and_suffix_from_saved_h(self):
        trainer = run.NativeTrainer(tiny(structural_rule="functional_fixed"))
        trainer.run_updates(3)
        _, _, live = trainer._live_chunk()
        trainer._auxiliary_gradients(live)
        durations = []
        original = trainer.model.step

        def traced(obs, hidden, dt, max_step=None):
            durations.append(float(dt))
            return original(obs, hidden, dt, max_step)

        with patch.object(trainer.model, "step", side_effect=traced):
            result = trainer._structural_losses(live)
        self.assertEqual(durations, list(live["elapsed"]) + list(trainer.last_auxiliary["block"]["elapsed"]))
        self.assertEqual(len(durations), 12)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def test_config_rejects_blanket_shrink_with_functional_policy(self):
        with self.assertRaises(ValueError):
            tiny(structural_rule="functional_task", shrink_step=2e-4)


if __name__ == "__main__":
    unittest.main()
