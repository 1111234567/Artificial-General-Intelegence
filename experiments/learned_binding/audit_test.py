"""Independent-audit tests using only small temporary random-model artifacts."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.shared import SharedAssociationMemory
from experiments.learned_binding import audit, common, run
from experiments.learned_binding.interaction import Subject, probe
from experiments.learned_binding.model import NativeMemoryPolicy


def make_fixture(root, settings):
    """Produce the exact formal schema, with no training or external artifacts."""
    with redirect_stdout(io.StringIO()):
        for kind in common.KINDS:
            folder = root / "seed_000" / "behavior" / kind
            folder.mkdir(parents=True)
            subject = Subject(0, kind, stream_id=100)
            own = subject.support(settings["support_ticks"], teacher=False, sample=True)
            run.save_block(folder / "own_support", own, dict(teacher=False))
            run.save_probe(deepcopy(subject), folder / "own_immediate")
            if kind == "learned":
                learner = deepcopy(subject)
                run.b_phase(learner, folder / "outcome_B", learn=True)
                run.save_probe(deepcopy(learner), folder / "own_after_B_outcome")
            run.b_phase(subject, folder / "frozen_B", learn=False)
            run.save_probe(deepcopy(subject), folder / "own_after_B_frozen")
            teacher = Subject(0, kind, stream_id=100)
            shown = teacher.support(settings["support_ticks"], teacher=True, sample=False)
            run.save_block(folder / "teacher_support", shown, dict(teacher=True))
            run.save_probe(deepcopy(teacher), folder / "teacher_probe")
            if kind == "learned":
                cleared = deepcopy(teacher)
                cleared.memory.zero_()
                run.save_probe(cleared, folder / "teacher_clear_memory")
                cleared = deepcopy(teacher)
                cleared.state["z"].zero_()
                run.save_probe(cleared, folder / "teacher_clear_encoder")
                run.save_probe(deepcopy(teacher), folder / "teacher_no_read", read_enabled=False)
            common.write_json(folder / "completion.json", dict(seed=0, kind=kind,
                files={path.name: common.sha(path) for path in folder.glob("*.json")}))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(763213)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "fixture"
        cls.settings = dict(common.SETTINGS, seeds=[0], actors=2, support_ticks=16,
            probe_ticks=40, pg_blocks=2, pg_ticks=4,
            support_duration=[1., 1.], support_gap=[1., 1.])
        native = NativeMemoryPolicy(original_mappings=common.mapping_table(0, "related")).eval()
        manual = SharedAssociationMemory(kind="delta")
        manual_path = Path(cls.directory.name) / "manual.pt"
        torch.save(dict(model=manual.state_dict(), metadata=dict(model_config=dict(
            kind="delta", key_dim=16, n_keys=8, n_values=4))), manual_path)
        cls.patches = [patch.dict(common.SETTINGS, cls.settings),
            patch.object(common, "load_native", side_effect=lambda seed: deepcopy(native)),
            patch.object(common, "executor", side_effect=lambda seed: deepcopy(native.executor)),
            patch.object(common, "manual_checkpoint", return_value=manual_path)]
        for item in cls.patches:
            item.start()
        try:
            make_fixture(cls.root, cls.settings)
        except Exception:
            for item in reversed(cls.patches):
                item.stop()
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        for item in reversed(cls.patches):
            item.stop()
        cls.directory.cleanup()

    def prefix(self, kind="learned", name="teacher_probe"):
        return self.root / "seed_000" / "behavior" / kind / name

    def test_complete_temporary_experiment_audits_every_probe_block_and_pg_update(self):
        with redirect_stdout(io.StringIO()):
            result = audit.audit_all(self.root, settings=self.settings, strict_sources=False)
        self.assertEqual(result["probe_count"], 13)
        self.assertEqual(result["B_block_count"], 8)
        self.assertEqual(result["native_replay_count"], 3)
        self.assertEqual(result["independently_recomputed_pg_updates"], 2)
        self.assertTrue(result["oracle_paths"]["only_oracle_call_is_explicit_teacher_support"])
        for phase in result["B_phases"]:
            self.assertTrue(phase["clocks_pending_and_live_state_chain_verified"])
            for check in phase["pg_checks"]:
                self.assertTrue(check["model_and_adam_bitwise_equal"])

    def test_ledger_corruption_missing_failures_censoring_and_first_results_are_detected(self):
        report, arrays, states = audit.load_bundle(self.prefix())
        self.assertGreater(report["groups"]["all"]["n"], 0)
        audit.recount_probe(report, arrays, states)
        changed = deepcopy(report)
        changed["groups"]["all"]["n"] += 1
        with self.assertRaisesRegex(audit.AuditError, "denominator"):
            audit.recount_probe(changed, arrays, states, world_replay=False)
        changed = deepcopy(report)
        changed["requests"].pop()
        with self.assertRaisesRegex(audit.AuditError, "finished request ledger"):
            audit.recount_probe(changed, arrays, states, world_replay=False)
        changed = deepcopy(report)
        changed["first_requests"] = []
        with self.assertRaisesRegex(audit.AuditError, "first-request"):
            audit.recount_probe(changed, arrays, states, world_replay=False)

    def test_saved_actions_and_native_world_observations_are_independently_replayed(self):
        report, arrays, states = audit.load_bundle(self.prefix())
        modified = deepcopy(arrays)
        modified["obs"][0, 0, 11] = .25
        with self.assertRaisesRegex(audit.AuditError, "world public observation"):
            audit.replay_world(modified, states)
        modified = deepcopy(arrays)
        modified["actions"][0, 0, 0] = (int(modified["actions"][0, 0, 0]) + 1) % 11
        with self.assertRaises(audit.AuditError):
            audit.replay_world(modified, states)
        modified = deepcopy(arrays)
        modified["target"][0, 0, 0] = (modified["target"][0, 0, 0] + 1) % 4
        with self.assertRaisesRegex(audit.AuditError, "world target"):
            audit.replay_world(modified, states)

    def test_reward_alignment_including_final_pending_action_rejects_shift(self):
        prefix = self.prefix(name="outcome_B/block_000")
        report, arrays, states = audit.load_bundle(prefix)
        audit.check_block(report, arrays, states)
        modified = deepcopy(arrays)
        modified["rewards"][-1, 0, 0] += 1.
        with self.assertRaisesRegex(audit.AuditError, "reward alignment"):
            audit.check_block(report, modified, states)
        modified = deepcopy(arrays)
        modified["pending_obs"][0, 22] = 1. - modified["pending_obs"][0, 22]
        with self.assertRaises(audit.AuditError):
            audit.check_block(report, modified, states)

    def test_joint_actor_return_credits_an_action_for_the_other_jobs_reward(self):
        logits = torch.zeros(2, 2, 2, 11, requires_grad=True)
        actions = torch.zeros(2, 2, 2, dtype=torch.long)
        rewards = torch.zeros(2, 2, 2)
        rewards[1, 0, 1] = 1.  # Only actor0/job1 obtains an outcome.
        returns, advantage = audit.independent_returns_advantages(rewards, .5)
        self.assertTrue(torch.equal(returns[:, 0], torch.tensor([.5, 1.])))
        self.assertTrue(torch.equal(advantage[:, 0], torch.tensor([.5, 1.])))
        self.assertTrue(torch.equal(advantage[:, 1], torch.tensor([-.5, -1.])))
        joint = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1).sum(-1)
        (-(joint * advantage).mean()).backward()
        self.assertGreater(float(logits.grad[:, 0, 0].abs().sum()), 0.)
        self.assertTrue(torch.equal(logits.grad[:, 0, 0], logits.grad[:, 0, 1]))

    def test_independent_adam_recompute_detects_changed_parameter_or_optimizer(self):
        first = self.prefix(name="outcome_B/block_000")
        report, arrays, states = audit.load_bundle(first)
        audit.recompute_pg(report, arrays, states, self.settings)
        changed = deepcopy(states)
        changed["after_update"]["model"]["gate.bias"][0] += .01
        with self.assertRaisesRegex(audit.AuditError, "PG/Adam parameters"):
            audit.recompute_pg(report, arrays, changed, self.settings)
        changed = deepcopy(states)
        first_parameter = next(iter(changed["optimizer_after"]["state"].values()))
        first_parameter["exp_avg"].add_(.01)
        with self.assertRaisesRegex(audit.AuditError, "PG/Adam optimizer"):
            audit.recompute_pg(report, arrays, changed, self.settings)

    def test_optimizer_live_h_z_and_memory_transplants_are_detected_even_with_valid_hashes(self):
        folder = self.prefix(name="outcome_B")
        prefix = folder / "block_000"
        pt_path, json_path = prefix.with_suffix(".pt"), prefix.with_suffix(".json")
        original_pt, original_json = pt_path.read_bytes(), json_path.read_text(encoding="utf-8")
        _, _, support = audit.load_bundle(self.prefix(name="own_support"))
        try:
            for name in ("h", "z", "memory"):
                with self.subTest(state=name):
                    states = torch.load(io.BytesIO(original_pt), weights_only=False)
                    states["after_update"]["state"][name].reshape(-1)[0] += .01
                    if name == "memory":
                        states["after_update"]["memory_hash"] = audit.tensor_hash(states["after_update"]["state"]["memory"])
                    torch.save(states, pt_path)
                    report = json.loads(original_json)
                    report["states_sha256"] = audit.file_hash(pt_path)
                    json_path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaisesRegex(audit.AuditError, "optimizer transplanted/mutated live state"):
                        audit.audit_phase(folder, support["final"], self.settings, learn=True)
        finally:
            pt_path.write_bytes(original_pt)
            json_path.write_text(original_json, encoding="utf-8")

    def test_probability_replay_detects_changed_collection_logits(self):
        report, arrays, states = audit.load_bundle(self.prefix(name="outcome_B/block_000"))
        audit.native_probability_check(arrays, states)
        changed = deepcopy(arrays)
        changed["logits"][0, 0, 0, 0] += .01
        with self.assertRaisesRegex(audit.AuditError, "logits differ"):
            audit.native_probability_check(changed, states)

    def test_hash_damage_and_absent_artifacts_fail_closed(self):
        report_path = self.prefix().with_suffix(".json")
        original = report_path.read_text(encoding="utf-8")
        try:
            changed = json.loads(original)
            changed["arrays_sha256"] = "0" * 64
            report_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "arrays_sha256"):
                audit.load_bundle(self.prefix())
        finally:
            report_path.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(audit.AuditError, "missing artifact"):
            audit.load_bundle(self.root / "absent")

    def test_native_policy_replay_does_not_open_an_origin_or_native_checkpoint(self):
        report, arrays, states = audit.load_bundle(self.prefix())
        with patch.object(common, "load_native", side_effect=AssertionError("checkpoint forbidden")), \
             patch.object(common, "executor", side_effect=AssertionError("checkpoint forbidden")):
            result = audit.replay_probe(report, arrays, states)
        self.assertTrue(result["oracle_disabled"])
        self.assertTrue(result["full_final_state_bitwise"])

    def test_first_request_denominator_includes_failed_never_begin_requests(self):
        subject = Subject(0, "learned")
        with torch.no_grad():
            subject.executor.head.weight.zero_()
            subject.executor.head.bias.fill_(-1000.)
            subject.executor.head.bias.reshape(2, 11)[:, 0] = 0.
        result = probe(subject, ticks=64)
        counted = audit.recount_probe(result["report"], result["arrays"],
                                      dict(start=result["start"], final=result["final"]))
        self.assertGreater(counted["matured"], 0)
        self.assertEqual(counted["groups"]["all"]["ordinary"], 0)
        self.assertGreater(result["report"]["first_groups"]["all"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
