"""Read/write-order algebra, causal gradients, backend and activation checks."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.learn_remember.gru_train import GRUAgent
from experiments.learned_binding import common, pretrain, run, read_first
from experiments.learned_binding.model import NativeMemoryPolicy
from experiments.learned_binding.model_test import public_observations
from experiments.learned_binding.read_first import ReadFirstPolicy


class ReadFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9173)

    def pair(self):
        before = NativeMemoryPolicy()
        after = ReadFirstPolicy(**before.config())
        after.load_state_dict(before.state_dict(), strict=True)
        return before, after

    def test_same_seed_identical_parameters_config_state_and_only_report_extension(self):
        torch.manual_seed(372000)
        old = NativeMemoryPolicy()
        torch.manual_seed(372000)
        new = ReadFirstPolicy()
        self.assertEqual(old.config(), new.config())
        self.assertEqual(list(old.named_parameters())[0][0], list(new.named_parameters())[0][0])
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for name, tensor in old.state_dict().items():
            self.assertTrue(torch.equal(tensor, new.state_dict()[name]), name)
        old_report, new_report = old.parameter_report(), new.parameter_report()
        self.assertEqual(new_report.pop("read_write_order"), "read_then_write_job0_then_job1")
        self.assertEqual(new_report, old_report)
        self.assertEqual(old_report["trainable_parameters"], 8210)
        self.assertEqual(old_report["state_scalars"], 320)
        self.assertIs(ReadFirstPolicy.__init__, NativeMemoryPolicy.__init__)
        self.assertIs(ReadFirstPolicy.step, NativeMemoryPolicy.step)

    def test_disabled_write_is_identical_to_base_and_preserves_memory_object(self):
        old, new = self.pair()
        obs, state = public_observations(13, 3), old.initial_state(3)
        for value in state.values():
            value.normal_(std=.2)
        for read in (True, False):
            a, sa, da = old(obs, state, write_enabled=False, read_enabled=read, return_diagnostics=True)
            b, sb, db = new(obs, state, write_enabled=False, read_enabled=read, return_diagnostics=True)
            self.assertTrue(torch.equal(a, b))
            self.assertIs(sb["memory"], state["memory"])
            for name in state:
                self.assertTrue(torch.equal(sa[name], sb[name]), name)
            for name in da:
                self.assertTrue(torch.equal(da[name], db[name]), name)

    def test_zero_initial_memory_single_tick_reads_uniform_but_writes_same_next_memory(self):
        old, new = self.pair()
        obs, state = public_observations(1, 3), old.initial_state(3)
        a, sa, da = old(obs, state, return_diagnostics=True)
        b, sb, db = new(obs, state, return_diagnostics=True)
        self.assertTrue(torch.equal(db["read_probs"], torch.full((1, 3, 2, 4), .25)))
        self.assertTrue(torch.equal(sa["memory"], sb["memory"]))
        self.assertFalse(torch.equal(sb["memory"], state["memory"]))
        for name in ("q", "k", "value", "gate", "z"):
            self.assertTrue(torch.equal(da[name], db[name]), name)
        self.assertFalse(torch.equal(a, b))
        with torch.no_grad():
            new.gate.bias.fill_(3.)
            new.value.bias.copy_(torch.tensor([3., -1., 2., 0.]))
        changed_logits, changed, diagnostic = new(obs, state, return_diagnostics=True)
        self.assertTrue(torch.equal(changed_logits, b))
        self.assertTrue(torch.equal(diagnostic["read_probs"], db["read_probs"]))
        self.assertFalse(torch.equal(changed["memory"], sb["memory"]))

    def test_two_tick_old_writes_receive_future_ce_but_current_writes_do_not(self):
        model = ReadFirstPolicy().train()
        obs = public_observations(2, 2).reshape(2, 2, 2, 28)
        obs[0, ..., :3] = 0.
        obs[1, ..., 0] = 1.
        logits, _, diagnostic = model(obs.reshape(2, 2, 56), model.initial_state(2), return_diagnostics=True)
        for name in ("k", "value", "gate", "q"):
            diagnostic[name].retain_grad()
        F.cross_entropy(logits[1].reshape(-1, 11), torch.tensor([0, 3, 6, 9])).backward()
        for name in ("k", "value", "gate"):
            self.assertGreater(float(diagnostic[name].grad[0].abs().sum()), 0., name)
            self.assertEqual(float(diagnostic[name].grad[1].abs().sum()), 0., name)
        self.assertGreater(float(diagnostic["q"].grad[1].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.executor.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.encoder.parameters()))

    def test_read_disabled_identical_to_base_and_no_receipt_or_future_parser(self):
        old, new = self.pair()
        obs, state = public_observations(11, 2), old.initial_state(2)
        a, sa = old(obs, state, read_enabled=False)
        b, sb, diagnostic = new(obs, state, read_enabled=False, return_diagnostics=True)
        self.assertTrue(torch.equal(a, b))
        for name in state:
            self.assertTrue(torch.equal(sa[name], sb[name]), name)
        self.assertTrue(torch.equal(diagnostic["read_probs"], torch.full((11, 2, 2, 4), .25)))
        source = inspect.getsource(ReadFirstPolicy.forward)
        for forbidden in ("argmax", "detach", "PublicReceiptBinder", "event_kind", "success", "NEW", "REPLACE"):
            self.assertNotIn(forbidden, source)
        changed = obs.clone()
        changed[6:] = torch.randn_like(changed[6:])
        first, _ = new(obs, state)
        second, _ = new(changed, state)
        torch.testing.assert_close(first[:6], second[:6], rtol=2e-5, atol=2e-6)
        self.assertEqual(inspect.signature(ReadFirstPolicy.forward), inspect.signature(NativeMemoryPolicy.forward))

    def _step_sequence(self, device):
        model = ReadFirstPolicy().to(device).train()
        obs, state = public_observations(17, 3).to(device), model.initial_state(3)
        for write, read in ((True, True), (False, True), (True, False), (False, False)):
            with torch.no_grad():
                expected, final, diag = model(obs, state, write_enabled=write, read_enabled=read, return_diagnostics=True)
                current, rows, reads = state, [], []
                for frame in obs:
                    logits, current, step_diag = model.step(frame, current, write_enabled=write,
                        read_enabled=read, return_diagnostics=True)
                    rows.append(logits)
                    reads.append(step_diag["read_probs"])
                torch.testing.assert_close(torch.stack(rows), expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(torch.stack(reads), diag["read_probs"], rtol=2e-5, atol=2e-6)
                for name in state:
                    torch.testing.assert_close(current[name], final[name], rtol=2e-5, atol=2e-6)

    def test_cpu_training_mode_step_sequence(self):
        self._step_sequence("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_mode_step_sequence_and_cpu_gradients(self):
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            self._step_sequence("cuda")
            cpu = ReadFirstPolicy().train()
            cuda = deepcopy(cpu).cuda().train()
            obs = public_observations(13, 2)
            target = torch.arange(13 * 2 * 2).remainder(11)
            cpu_logits, cpu_state = cpu(obs, cpu.initial_state(2))
            cuda_logits, cuda_state = cuda(obs.cuda(), cuda.initial_state(2))
            torch.testing.assert_close(cuda_logits.cpu(), cpu_logits, rtol=2e-5, atol=2e-6)
            for name in cpu_state:
                torch.testing.assert_close(cuda_state[name].cpu(), cpu_state[name], rtol=2e-5, atol=2e-6)
            F.cross_entropy(cpu_logits.reshape(-1, 11), target).backward()
            F.cross_entropy(cuda_logits.reshape(-1, 11), target.cuda()).backward()
            for (name, a), (other, b) in zip(cpu.named_parameters(), cuda.named_parameters()):
                self.assertEqual(name, other)
                if a.requires_grad:
                    torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-4, atol=2e-6)
                else:
                    self.assertIsNone(a.grad)
                    self.assertIsNone(b.grad)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

    def test_activate_factory_and_checkpoint_loader_use_new_class_without_touching_v1(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(read_first, "OUT", folder), patch.object(common, "OUT"), \
                 patch.object(common, "NativeMemoryPolicy"), patch.object(pretrain, "OUT"), \
                 patch.object(run, "OUT"), patch.object(common, "executor", side_effect=lambda seed: GRUAgent().eval()):
                self.assertEqual(read_first.activate(), folder)
                self.assertIs(common.NativeMemoryPolicy, ReadFirstPolicy)
                self.assertEqual((common.OUT, pretrain.OUT, run.OUT), (folder,) * 3)
                model = common.new_native(0)
                self.assertIsInstance(model, ReadFirstPolicy)
                path = common.native_checkpoint(0)
                path.parent.mkdir(parents=True)
                torch.save(dict(model_config=model.config(), model=model.state_dict()), path)
                loaded = common.load_native(0)
                self.assertIsInstance(loaded, ReadFirstPolicy)
                for name, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))

    def test_manual_copy_is_byte_identical_and_records_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            old, new = repository / "v1", repository / "v2"
            for seed in range(3):
                for kind in ("manual_hard", "manual_soft"):
                    folder = old / f"seed_{seed:03d}/behavior" / kind
                    folder.mkdir(parents=True)
                    (folder / "completion.json").write_text(json.dumps(dict(seed=seed, kind=kind)), encoding="utf-8")
                    (folder / "fixture.bin").write_bytes(b"unchanged control")
            with patch.object(read_first, "ROOT", repository), patch.object(read_first, "OUT", new), \
                 patch.object(read_first, "V1_OUT", old), patch.object(read_first, "activate"), \
                 patch.object(read_first, "_parent_audit", return_value=dict(verified=True)), \
                 patch.object(run, "check_freeze"):
                manifest = read_first.reuse_manual_controls()
                self.assertEqual(len(manifest["copies"]), 6)
                self.assertFalse(manifest["controls_reexecuted"])
                for row in manifest["copies"]:
                    self.assertEqual(read_first._tree_hashes(repository / row["source"]),
                                     read_first._tree_hashes(repository / row["destination"]))
                    self.assertEqual(len(row["files_sha256"]), 2)
                self.assertTrue((new / "manual-reuse.json").is_file())

    def test_declare_temporary_fixture_preserves_data_manifest_and_discloses_followup(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            old, new = repository / "v1", repository / "v2"
            old.mkdir()
            frozen, source, plan = repository / "frozen.py", repository / "read_first.py", repository / "plan.md"
            for path in (frozen, source, plan):
                path.write_text("temporary fixture\n", encoding="utf-8")
            prototype = dict(settings=common.SETTINGS,
                             sources={"frozen.py": common.sha(frozen)}, origins={})
            (old / "protocol.json").write_text(json.dumps(prototype), encoding="utf-8")
            (old / "training-freeze.json").write_text("{}", encoding="utf-8")
            data_bytes = b'{"streams":[{"path":"v1/original.pt"}]}\n'
            (old / "data-freeze.json").write_bytes(data_bytes)
            (old / "audit.json").write_text(json.dumps(dict(probe_count=39, B_block_count=384,
                native_replay_count=9, independently_recomputed_pg_updates=6,
                all_world_actions_and_public_rewards_replayed=True,
                all_controller_live_states_preserved_across_optimizer_updates=True,
                all_source_executors_unchanged=True)), encoding="utf-8")
            def fake_declare():
                common.write_json(new / "protocol.json", prototype)
            with patch.object(read_first, "ROOT", repository), patch.object(read_first, "OUT", new), \
                 patch.object(read_first, "V1_OUT", old), patch.object(read_first, "SOURCE", source), \
                 patch.object(read_first, "PLAN", plan), patch.object(common, "declare", side_effect=fake_declare), \
                 patch.object(common, "OUT"), patch.object(common, "NativeMemoryPolicy"), \
                 patch.object(pretrain, "OUT"), patch.object(run, "OUT"):
                protocol = read_first.declare()
                self.assertEqual((new / "data-freeze.json").read_bytes(), data_bytes)
                self.assertTrue(protocol["followup"]["declared_after_v1_results_inspected"])
                self.assertFalse(protocol["manual_behavior_reuse"]["controls_reexecuted"])
                self.assertEqual(protocol["sources"]["read_first.py"], common.sha(source))
                self.assertEqual(protocol["sources"]["plan.md"], common.sha(plan))

    def test_audit_cli_switches_reconstruction_class_without_running_real_audit(self):
        from experiments.learned_binding import audit
        fixture = dict(probe_count=39, B_block_count=384, native_replay_count=9,
                       independently_recomputed_pg_updates=6)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(read_first, "OUT", folder), patch.object(read_first, "activate"), \
                 patch.object(audit, "NativeMemoryPolicy"), \
                 patch.object(audit, "audit_all", return_value=fixture) as mocked:
                result = read_first.audit_all()
                mocked.assert_called_once_with(folder)
                self.assertIs(audit.NativeMemoryPolicy, ReadFirstPolicy)
                self.assertEqual(result["probe_count"], 39)
                saved = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["variant"], "read-first-v2")


if __name__ == "__main__":
    unittest.main()
