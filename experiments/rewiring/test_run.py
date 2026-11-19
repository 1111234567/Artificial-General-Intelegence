"""Small continuity, BPTT-potential and exact-resume checks, never a study."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.model import AgentModel
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.credit.run import CreditConfig, CreditTrainer
from experiments.rewiring.plasticity import RewireConfig
from experiments.rewiring.run import (
    ARMS, RewireTrainer, main, optimizer_parameter_names, run_to_target, write_artifacts,
)


def tiny_source(device="cpu", width=6):
    trainer = Trainer(Variant("rewiring-test", width=width, policy="online"),
        TrainConfig(seed=179, batch_size=2, tbptt=4, updates=2, device=device,
            train_dts=(.25,), replay_capacity=2, replay_block_ticks=8,
            replay_burnin=4, log_every=2))
    trainer.run_updates(2)
    return trainer


def trainer_for(arm="learned", device="cpu", **changes):
    values = dict(arm=arm, shrink_step=.02, window_updates=2, structural_seed=315)
    values.update(changes)
    return RewireTrainer(tiny_source(device), RewireConfig(**values), chunk_ticks=4, phase_ticks=12)


def disturb_rng():
    random.random()
    np.random.random(4)
    torch.rand(4)
    if torch.cuda.is_available():
        torch.rand(4, device="cuda")


class RewireRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_graft_retains_model_optimizer_hidden_history_and_rng_without_initial_prune(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                base = tiny_source()
                before = deepcopy(base.semantic_state())
                old_model, old_optimizer, old_world = base.model, base.optimizer, base.world
                parameters = dict(base.model.named_parameters())
                trainer = RewireTrainer(base, RewireConfig(arm=arm, shrink_step=.02), chunk_ticks=4)
                after = base.semantic_state()
                self.assertIs(type(trainer.model), AgentModel)
                self.assertIs(trainer.model, old_model)
                self.assertIs(trainer.optimizer, old_optimizer)
                self.assertIs(trainer.world, old_world)
                for name, parameter in parameters.items():
                    self.assertIs(dict(trainer.model.named_parameters())[name], parameter)
                    self.assertTrue(torch.equal(parameter, before["model"][name]))
                for key in ("model", "optimizer", "hidden", "rng", "replay", "ticks", "updates", "dt_history"):
                    self.assertEqual(state_digest(before[key]), state_digest(after[key]), key)
                for key in before["world"]:
                    if key not in {"regime", "regime_id", "regime_history"}:
                        self.assertEqual(state_digest(before["world"][key]), state_digest(after["world"][key]), key)
                self.assertEqual(int(trainer.operator.mask.sum()), trainer.model.width ** 2)
                self.assertEqual(trainer.operator.updates, 0)
                self.assertEqual(trainer.rewire_origin["inherited_semantic_hash"], state_digest(before))
                self.assertEqual(trainer.stage_ticks, 0)
                self.assertEqual(trainer.hidden_resets, 0)

    def _dense_matches_old_bp64(self, device):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            tiny_source(device).save_checkpoint(path)
            rewire = RewireTrainer(Trainer.from_checkpoint(path, device=device), RewireConfig(arm="dense", shrink_step=5e-5),
                                   chunk_ticks=64)
            rewire.run_updates(4)
            expected = deepcopy(CreditTrainer.semantic_state(rewire))
            old = CreditTrainer.from_base_checkpoint(path, CreditConfig(mode="bp64", chunk_ticks=64,
                duration_range=(4., 8.), gap_range=(4., 12.), regime_name="rewiring-B"), device=device)
            old.run_updates(4)
            actual = old.semantic_state()
            for key in ("model", "optimizer", "hidden", "world", "rng", "replay", "stats", "ticks", "updates"):
                self.assertEqual(state_digest(expected["continuation"][key]), state_digest(actual["continuation"][key]), key)
            for key in ("work", "update_norms", "sampler_rng", "regime_changes"):
                self.assertEqual(state_digest(expected[key]), state_digest(actual[key]), key)
            self.assertEqual(rewire.action_hash, old.action_hash)
            self.assertEqual(rewire.data_hash, old.data_hash)

    def test_dense_matches_old_bp64_for_four_complete_cpu_updates(self):
        self._dense_matches_old_bp64("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dense_matches_old_bp64_for_four_complete_cuda_updates(self):
        self._dense_matches_old_bp64("cuda")

    def test_attachment_roundtrip_retains_inherited_zero_weight_adam_moments(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                base = tiny_source()
                with torch.no_grad():
                    base.model.recurrent[0, 0] = 0.
                moments = base.optimizer.state[base.model.recurrent]
                self.assertNotEqual(float(moments["exp_avg"][0, 0]), 0.)
                trainer = RewireTrainer(base, RewireConfig(arm=arm), chunk_ticks=4)
                checkpoint = Path(folder) / "attachment.pt"
                trainer.save_checkpoint(checkpoint)
                expected = state_digest(trainer.semantic_state())
                resumed = RewireTrainer.from_checkpoint(checkpoint)
                self.assertEqual(expected, state_digest(resumed.semantic_state()))
                resumed.train_update()
                if arm != "dense":
                    live = resumed.optimizer.state[resumed.model.recurrent]
                    for name in ("exp_avg", "exp_avg_sq"):
                        self.assertEqual(int(torch.count_nonzero(live[name][~resumed.operator.mask])), 0)

    def test_dense_updated_zero_weight_may_retain_adam_second_moment(self):
        base = tiny_source()
        with torch.no_grad():
            base.model.recurrent[0, 0] = 0.
            base.optimizer.state[base.model.recurrent]["exp_avg"][0, 0] = 0.
        trainer = RewireTrainer(base, RewireConfig(arm="dense"), chunk_ticks=4)
        original = trainer._apply_gradients

        def with_zero_gradient(gradients):
            gradients["recurrent"][0, 0] = 0.
            return original(gradients)

        trainer._apply_gradients = with_zero_gradient
        trainer.train_update()
        self.assertEqual(float(trainer.model.recurrent.detach()[0, 0]), 0.)
        self.assertGreater(float(trainer.optimizer.state[trainer.model.recurrent]["exp_avg_sq"][0, 0]), 0.)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "dense.pt"
            trainer.save_checkpoint(checkpoint)
            expected = state_digest(trainer.semantic_state())
            resumed = RewireTrainer.from_checkpoint(checkpoint)
            self.assertEqual(expected, state_digest(resumed.semantic_state()))

    def test_zero_dormant_weights_have_full_bptt_potential_gradients_before_live_mask(self):
        trainer = trainer_for("retire", shrink_step=10., window_updates=1)
        zero = {name: torch.zeros_like(value) for name, value in trainer.model.named_parameters()}
        trainer._apply_gradients(zero)
        self.assertFalse(bool(trainer.operator.mask.any()))
        self.assertEqual(int(torch.count_nonzero(trainer.model.recurrent)), 0)
        gradients, _, _ = trainer._live_chunk()
        raw = gradients["recurrent"]
        self.assertGreater(int(torch.count_nonzero(raw)), 0)
        self.assertEqual(int(torch.count_nonzero(trainer.operator.before_gradient(raw))), 0)
        self.assertIs(type(trainer.model), AgentModel)

    def test_mask_precedes_joint_clipping_and_actual_delta_includes_shrink(self):
        trainer = trainer_for("retire", shrink_step=.15, window_updates=1)
        gradients = {name: torch.zeros_like(value) for name, value in trainer.model.named_parameters()}
        before = trainer.model.recurrent.detach().double().clone()
        update = trainer._apply_gradients(gradients)
        self.assertAlmostEqual(update["recurrent_delta"], float((trainer.model.recurrent.detach().double() - before).norm()))
        self.assertGreater(update["recurrent_delta"], 0.)
        self.assertGreater(int((~trainer.operator.mask).sum()), 0)
        gradients = {name: torch.full_like(value, .01) for name, value in trainer.model.named_parameters()}
        gradients["recurrent"][~trainer.operator.mask] = 1e6
        expected = torch.cat([value.detach().reshape(-1) if name != "recurrent" else
            trainer.operator.before_gradient(value).reshape(-1) for name, value in gradients.items()]).norm()
        update = trainer._apply_gradients(gradients)
        self.assertAlmostEqual(update["gradient_norm"], float(expected), places=5)
        self.assertGreater(update["potential_recurrent_gradient_norm"], 1e6)
        self.assertTrue(torch.equal(trainer.model.recurrent[~trainer.operator.mask],
                                    torch.zeros_like(trainer.model.recurrent[~trainer.operator.mask])))

    def test_surviving_active_adam_moments_and_shared_steps_are_never_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            tiny_source().save_checkpoint(path)
            trainer = RewireTrainer(Trainer.from_checkpoint(path), RewireConfig(arm="retire", shrink_step=.2,
                window_updates=1), chunk_ticks=4)
            reference = CreditTrainer.from_base_checkpoint(path, CreditConfig(chunk_ticks=4,
                regime_name="rewiring-B"))
            gradients = {name: torch.full_like(value, .01) for name, value in trainer.model.named_parameters()}
            trainer._apply_gradients(gradients)
            reference._apply_gradients(gradients)
            live = trainer.optimizer.state[trainer.model.recurrent]
            old = reference.optimizer.state[reference.model.recurrent]
            self.assertTrue(bool(trainer.operator.mask.any()))
            self.assertTrue(bool((~trainer.operator.mask).any()))
            self.assertTrue(torch.equal(live["step"], old["step"]))
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(live[name][trainer.operator.mask], old[name][trainer.operator.mask]))
                self.assertEqual(int(torch.count_nonzero(live[name][~trainer.operator.mask])), 0)
            for name in ("encoder.weight", "head.weight", "head.bias"):
                left = trainer.optimizer.state[dict(trainer.model.named_parameters())[name]]
                right = reference.optimizer.state[dict(reference.model.named_parameters())[name]]
                self.assertEqual(state_digest(left), state_digest(right))

    def _resume_case(self, arm, device):
        with tempfile.TemporaryDirectory() as folder:
            trainer = trainer_for(arm, device)
            trainer.train_update()  # Inside the first structural window.
            checkpoint = Path(folder) / "middle.pt"
            trainer.save_checkpoint(checkpoint)
            boundary = state_digest(trainer.semantic_state())
            trainer.run_updates(6)  # Cross structural and B -> A2 boundaries.
            expected = state_digest(trainer.semantic_state())
            disturb_rng()
            resumed = RewireTrainer.from_checkpoint(checkpoint, device=device)
            self.assertEqual(boundary, state_digest(resumed.semantic_state()))
            resumed.run_updates(6)
            self.assertEqual(expected, state_digest(resumed.semantic_state()))
            self.assertEqual(resumed.stage_ticks, 28)
            self.assertTrue(resumed.phase_returned)
            self.assertEqual(len(resumed.regime_changes), 2)

    def test_complete_cpu_resume_for_every_arm_including_midwindow_and_regime_return(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                self._resume_case(arm, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_complete_cuda_resume_for_every_arm_including_midwindow_and_regime_return(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                self._resume_case(arm, "cuda")

    def test_phase_checkpoint_is_before_return_and_resumes_change_exactly_once(self):
        trainer = trainer_for("random")
        with tempfile.TemporaryDirectory() as folder:
            with redirect_stdout(io.StringIO()):
                run_to_target(trainer, 24, folder, checkpoint_every=0)
            expected = state_digest(trainer.semantic_state())
            b = RewireTrainer.from_checkpoint(Path(folder) / "checkpoint_B.pt")
            self.assertEqual(b.stage_ticks, 12)
            self.assertFalse(b.phase_returned)
            self.assertEqual(b.world.regime.name, "rewiring-B")
            before = deepcopy(b.semantic_state())
            b._ensure_return_phase()
            after = b.semantic_state()
            for key in ("model", "optimizer", "hidden", "rng", "replay", "ticks", "updates", "dt_history"):
                self.assertEqual(state_digest(before["credit"]["continuation"][key]),
                                 state_digest(after["credit"]["continuation"][key]), key)
            self.assertEqual(state_digest(before["rewiring"]), state_digest(after["rewiring"]))
            b._ensure_return_phase()
            self.assertEqual(len(b.regime_changes), 2)
            b.run_updates(3)
            self.assertEqual(expected, state_digest(b.semantic_state()))
            final = RewireTrainer.from_checkpoint(Path(folder) / "checkpoint_A2.pt")
            self.assertEqual(expected, state_digest(final.semantic_state()))

    def test_artifacts_keep_every_update_and_complete_source_layers(self):
        trainer = trainer_for("learned")
        trainer.run_updates(3)
        with tempfile.TemporaryDirectory() as folder:
            write_artifacts(trainer, folder)
            folder = Path(folder)
            rows = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
            windows = [json.loads(line) for line in (folder / "structural_windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(windows), 1)
            self.assertEqual(rows[-1]["parameter_update"]["structure"]["active"], int(trainer.operator.mask.sum()))
            for name in ("sources/rewiring/run.py", "sources/rewiring/plasticity.py", "sources/credit/run.py",
                         "sources/continuity/model.py", "sources/dependencies/integration/run.py"):
                self.assertTrue((folder / name).is_file(), name)
            report = json.loads((folder / "report.json").read_text())
            self.assertNotIn("trace", report)
            self.assertNotIn("curve", report)
            self.assertEqual(report["model"]["state_scalars"], 6)
            # Reconciliation replaces any uncheckpointed suffix on resume.
            with (folder / "training.jsonl").open("a") as stream:
                stream.write('{"stale": true}\n')
            write_artifacts(trainer, folder)
            self.assertEqual(len((folder / "training.jsonl").read_text().splitlines()), 3)

    def test_invalid_gradients_do_not_mutate_weights_adam_or_structural_state(self):
        trainer = trainer_for()
        before = state_digest(trainer.semantic_state())
        gradients = {name: torch.zeros_like(value) for name, value in trainer.model.named_parameters()}
        missing = dict(gradients)
        missing.pop("recurrent")
        with self.assertRaisesRegex(ValueError, "cover every"):
            trainer._apply_gradients(missing)
        gradients["encoder.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "invalid gradient"):
            trainer._apply_gradients(gradients)
        self.assertEqual(before, state_digest(trainer.semantic_state()))

    def test_corrupt_checkpoint_tensors_sources_history_and_optimizer_order_rejected(self):
        trainer = trainer_for("retire", shrink_step=.2)
        trainer.run_updates(3)
        state = trainer.state_dict()
        for target in ("weight", "hidden", "adam-nan", "adam-shape", "adam-negative", "adam-order",
                       "sources", "dependencies", "progress", "windows", "return-history", "dormant-moment"):
            with self.subTest(target=target):
                broken = deepcopy(state)
                continuation = broken["credit"]["continuation"]
                if target == "weight":
                    continuation["model"]["recurrent"][0, 0] = float("inf")
                elif target == "hidden":
                    continuation["hidden"][0, 0] = float("nan")
                elif target == "adam-nan":
                    continuation["optimizer"]["state"][0]["exp_avg"][0, 0] = float("nan")
                elif target == "adam-shape":
                    continuation["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
                elif target == "adam-negative":
                    continuation["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1.
                elif target == "adam-order":
                    continuation["optimizer"]["param_groups"][0]["params"][0:2] = [1, 0]
                elif target == "sources":
                    broken["sources"]["run.py"] += "\n# corrupt\n"
                elif target == "dependencies":
                    broken["dependency_hashes"]["credit/run.py"] = "0" * 64
                elif target == "progress":
                    broken["phase_returned"] = True
                elif target == "windows":
                    broken["structural_windows"] = []
                elif target == "return-history":
                    broken["credit"]["regime_changes"].append(deepcopy(broken["credit"]["regime_changes"][0]))
                    broken["credit"]["regime_changes"][-1]["regime"]["name"] = "rewiring-A2"
                else:
                    mask = trainer.operator.mask.cpu()
                    self.assertTrue(bool((~mask).any()))
                    continuation["optimizer"]["state"][0]["exp_avg"][~mask] = 1.
                with self.assertRaises(ValueError):
                    RewireTrainer._from_state(broken)

    def test_helper_and_configuration_validation(self):
        for phase in (0, True, 3):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                RewireTrainer(tiny_source(), RewireConfig(arm="dense"), chunk_ticks=4, phase_ticks=phase)
        trainer = trainer_for("dense")
        self.assertEqual(optimizer_parameter_names(trainer.model, trainer.optimizer),
            [["recurrent", "encoder.weight", "head.weight", "head.bias"]])
        for target in (-1, 1, True):
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                run_to_target(trainer, target, folder)
        trainer.base._at_boundary = False
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(RuntimeError):
            trainer.save_checkpoint(Path(folder) / "partial.pt")

    def test_cli_zero_tick_start_resume_and_formal_width_restriction(self):
        with tempfile.TemporaryDirectory() as folder:
            source, narrow, output = Path(folder) / "A.pt", Path(folder) / "narrow.pt", Path(folder) / "run"
            tiny_source(width=128).save_checkpoint(source)
            tiny_source().save_checkpoint(narrow)
            with self.assertRaisesRegex(ValueError, "width 128"):
                RewireTrainer.from_source_checkpoint(narrow, RewireConfig(arm="dense"))
            original = source.read_bytes()
            with redirect_stdout(io.StringIO()):
                main(["--source", str(source), "--arm", "learned", "--output", str(output),
                      "--target-ticks", "0", "--device", "cpu"])
            trainer = RewireTrainer.from_checkpoint(output / "checkpoint.pt")
            expected = state_digest(trainer.semantic_state())
            self.assertEqual(trainer.stage_ticks, 0)
            main(["--output", str(output), "--resume", "--target-ticks", "0", "--device", "cpu"])
            self.assertEqual(expected, state_digest(RewireTrainer.from_checkpoint(output / "checkpoint.pt").semantic_state()))
            self.assertEqual(original, source.read_bytes())


if __name__ == "__main__":
    unittest.main()
