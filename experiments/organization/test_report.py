"""Small real organization states, mutation rejection and seed-level summaries.

The end-to-end fixture still declares all 27 variants; only model size and
lifetime/horizon are reduced using study.prepare's explicit mechanical mode.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import evaluate
from experiments.loops.report import AuditError, _Checks
from experiments.organization import report, run as native, study


def tiny(**changes):
    return replace(native.NativeConfig(width=4, batch_size=1, chunk_ticks=2, initial_fan_in=1,
        replay_capacity=2, block_ticks=2, macrocycle_ticks=8, main_ticks=16,
        withdrawal_ticks=4, return_ticks=4, device="cpu"), **changes)


def reseal(state):
    for key in state["state_hashes"]:
        if key != "semantic":
            state["state_hashes"][key] = state_digest(state[key])
    state["state_hashes"]["semantic"] = state_digest({key: state[key] for key in native._SEMANTIC_FIELDS})
    return state


class OrganizationCheckpointAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.states = {}
        with _preserve_process_state():
            for rule in ("passive", "functional_task", "functional_fixed"):
                trainer = native.NativeTrainer(tiny(structural_rule=rule))
                cls.states[(rule, 0)] = trainer.state_dict()
                for _ in range(12):
                    trainer.train_update()
                    if trainer.updates in (8, 9, 12):
                        cls.states[(rule, trainer.updates)] = trainer.state_dict()

    def test_real_organization_states_restore_and_old_trainer_is_never_called(self):
        with patch.object(report.legacy.native_run.NativeTrainer, "from_state_dict",
                          side_effect=AssertionError("organization audit used the old trainer")):
            for (rule, updates), state in self.states.items():
                with self.subTest(rule=rule, updates=updates):
                    result = report.audit_checkpoint_payload(state)
                    self.assertEqual(result["updates"], updates)
                    self.assertEqual(result["model_config"]["field_mode"], "plain")
                    self.assertEqual(result["work"]["structural_probe_calls"],
                        0 if rule == "passive" else 4 * (max(0, updates-1) // 8))

    def test_resealed_wrong_hash_chain_prefix_is_rejected(self):
        state = deepcopy(self.states[("passive", 12)])
        action = data = hashlib.sha256(f"lifelong-native-trainer-v1/{state['birth']['data_seed']}".encode()).hexdigest()
        for row in state["trace"]:
            action = hashlib.sha256(bytes.fromhex(action) + bytes.fromhex(row["action_chunk_hash"])).hexdigest()
            data = hashlib.sha256(bytes.fromhex(data) + bytes.fromhex(row["data_chunk_hash"])).hexdigest()
            row["action_hash"], row["data_hash"] = action, data
        state["action_hash"], state["data_hash"] = action, data
        with self.assertRaises(AuditError):
            report.audit_checkpoint_payload(reseal(state))

    def test_pending_delta_checked_against_real_full_adam_slots(self):
        state = deepcopy(self.states[("functional_task", 8)])
        self.assertGreater(len(state["operator"]["pending"]["birth_delta"]), 0)
        state["operator"]["pending"]["birth_delta"][0] += .01
        record = state["trace"][-1]["update_metrics"]["structure"]["pending_generated"]
        record["birth_delta"] = state["operator"]["pending"]["birth_delta"].tolist()
        state["trace"][-1]["operator_hash"] = state_digest(state["operator"])
        with self.assertRaises(AuditError):
            report.audit_checkpoint_payload(reseal(state))

    def test_birth_source_raw_labels_and_probe_sources_reject_mutations(self):
        for mutation in (
            lambda s: s["birth"].update(initial_topology="invented"),
            lambda s: s["last_chunk"]["targets"].fill_(10),
            lambda s: s["trace"][8]["update_metrics"]["structure"]["probe_sources"].update(history_start_tick=999),
            lambda s: s["operator"]["traits"]["activation_count"].add_(1),
            lambda s: s["schedule_history"].append(dict(tick=999, phase="invented")),
            lambda s: s["stats"]["confusion"].add_(1),
        ):
            state = deepcopy(self.states[("functional_task", 12)])
            mutation(state)
            with self.assertRaises(AuditError):
                report.audit_checkpoint_payload(reseal(state))

    def test_full_probe_damage_and_same_next_block_pending_are_bound(self):
        for mutation in (
            lambda p: p.update(execute_update=100),
            lambda p: p["full_ablation_damage"].update(history=987.),
            lambda p: p.update(actual_birth_coordinates=[[3, 3]]),
        ):
            state = deepcopy(self.states[("functional_task", 12)])
            mutation(state["trace"][8]["update_metrics"]["structure"]["functional_probe"])
            with self.assertRaises(AuditError):
                report.audit_checkpoint_payload(reseal(state))

    def test_lifetimes_include_completed_and_censored_eligibility_spells(self):
        initial = torch.tensor([[True, False], [False, False]])
        state = dict(config=dict(structural_rule="functional_task", chunk_ticks=2, width=2), updates=12,
            operator=dict(initial_mask=initial, event_history=[
                dict(executed_update=3, actual_birth_coordinates=[[0, 1]], actual_retire_coordinates=[]),
                dict(executed_update=5, actual_birth_coordinates=[], actual_retire_coordinates=[[0, 0]]),
                dict(executed_update=9, actual_birth_coordinates=[[0, 0]], actual_retire_coordinates=[])]))
        rows = report.edge_lifetimes(state)
        self.assertEqual(len(rows), 3)
        completed = next(row for row in rows if not row["right_censored"])
        self.assertEqual((completed["start_update"], completed["end_update"], completed["duration_ticks"]), (0, 5, 10))
        self.assertEqual(sum(row["right_censored"] for row in rows), 2)
        bridge = next(row for row in rows if row["cross_bank"])
        self.assertEqual(bridge["start_update"], 3)
        self.assertFalse(bridge["initial_coordinate"])

    def test_functional_structure_outside_initial_and_cross_bank_counts(self):
        state = self.states[("functional_task", 12)]
        result = report.structural_history(state)
        self.assertEqual(result["event_semantics"], "eligibility allocation changes")
        self.assertEqual(result["active_edges"], int(state["model"]["recurrent"].ne(0).sum()))
        self.assertEqual(result["unique_edges_ever_born_outside_initial"],
            int((state["operator"]["ever_born"] & ~state["operator"]["initial_mask"]).sum()))

    def test_frozen_denominators_and_field_identity_are_strict(self):
        state = self.states[("functional_fixed", 12)]
        cp = report.audit_checkpoint_payload(state)
        with _preserve_process_state():
            trainer = native.NativeTrainer.from_state_dict(state, device="cpu")
            spec = dict(tick=24, family=0, condition="short", steps=8)
            record = evaluate(trainer.model, trainer.world, trainer.h, family=0, condition="short", steps=8, device="cpu")
        record["probe"] = spec
        record["organization"] = dict(field_mode="plain", structural_rule="functional_fixed", structural_updates_during_fork=0)
        branch = dict(field_mode="plain", structural_rule="functional_fixed", native_config=state["config"])
        sources = {str((report.ROOT / name).resolve()): digest for name, digest in record["source_hashes"].items()}
        plan = dict(sources=sources, dt=.25)
        metrics = report.audit_native_record(record, cp, plan=plan, spec=spec, branch=branch)
        self.assertIn("missed_finish_despite_legal", metrics)
        self.assertIn("legal_finish_omission_rate", metrics)
        for mutate in (lambda r: r["primary"].update(clean_completion_rate=.1234),
                       lambda r: r["model"].update(field_mode="coupled"),
                       lambda r: r.update(inactive_actions_per_1000_idle_opportunities=-1),
                       lambda r: r["content_by_cohort_phase_age"][0].update(executed_finish_attempts=10**9)):
            damaged = deepcopy(record)
            mutate(damaged)
            with self.assertRaises(AuditError):
                report.audit_native_record(damaged, cp, plan=plan, spec=spec, branch=branch)


class OrganizationStatisticsTests(unittest.TestCase):
    def test_worker_count_limits_and_pickle_safe_branch_failures(self):
        for workers in (0, 7, True, 1.5):
            with self.assertRaises(ValueError):
                report.build_report("unused.json", workers=workers)
        item = dict(id="toy-branch")
        with patch.object(report, "_branch", side_effect=AuditError("TOY_REJECTION", "fixture")):
            result = report._branch_worker(({}, item, "unused"))
        self.assertEqual(result["failure"]["errors"][0]["code"], "TOY_REJECTION")
        self.assertEqual(result["branch"], item)

    def test_complete_flag_cannot_publish_missing_runs_or_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "must-not-exist"
            claimed = dict(complete=True, audit=dict(status="passed", errors=[]),
                counts=dict(training_runs=27, trained_seeds=3, full_checkpoints=108, evaluation_files=27, frozen_forks=648),
                header=dict(complete=True))
            with self.assertRaises(ValueError):
                report.write_artifacts(claimed, out)
            self.assertFalse(out.exists())

    def test_cli_failure_artifact_handles_path_inventory_sets(self):
        with tempfile.TemporaryDirectory() as folder:
            error = AuditError("MISSING_BRANCH", "fixture", dict(actual={Path(folder) / "missing"}, expected=set()))
            with patch.object(report, "build_report", side_effect=error):
                with self.assertRaises(AuditError):
                    report.main(["--protocol", "unused.json", "--output", folder])
            saved = json.loads((Path(folder) / "audit-failed.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertFalse((Path(folder) / "report.json").exists())

    def test_three_seed_sample_sd_and_undefined_denominators(self):
        value = report.summarize_seed_values([dict(seed=i, value=i) for i in (0, 1, 2)])
        self.assertEqual((value["n"], value["mean"], value["sample_sd"]), (3, 1, 1))
        self.assertEqual(report.summarize_seed_values([dict(seed=i, value=None) for i in (0, 1, 2)])["n_undefined"], 3)
        for rows in ([dict(seed=0, value=1)], [dict(seed=0, value=1)] * 3):
            with self.assertRaises(ValueError):
                report.summarize_seed_values(rows)

    def test_family_averages_precede_seed_aggregation_and_did_is_paired(self):
        rows = []
        for seed in (0, 1, 2):
            for variant_index, variant in enumerate(study.VARIANTS):
                for family in range(4):
                    rows.append(dict(seed=seed, **variant, tick=16, family=family, condition="short", steps=8,
                                     metrics=dict(score=seed + 10*variant_index + family)))
        family = report.family_averages(rows)
        self.assertEqual(len(family), 27)
        reference = next(row for row in family if row["seed"] == 0 and row["field_mode"] == "plain"
                         and row["structural_rule"] == "passive")
        self.assertEqual(reference["metrics"]["score"], 1.5)
        contrasts = report.paired_contrasts(rows)
        self.assertEqual(len(contrasts), 3 * 4 * 10)
        did = [row for row in contrasts if row["contrast"].endswith("difference_in_differences")]
        self.assertTrue(all(row["metrics"]["score"] == 0 for row in did))
        with self.assertRaises(ValueError):
            report.family_averages(rows[:-1])
        with self.assertRaises(ValueError):
            report.paired_contrasts(rows[:-4])

    def test_formal_extra_work_is_reported_separately(self):
        passive = report.expected_work(172032, 128, 32, rule="passive")
        functional = report.expected_work(172032, 128, 32, rule="functional_task")
        self.assertEqual(passive["structural_probe_calls"], 0)
        self.assertEqual(functional["structural_probe_calls"], 668)
        self.assertEqual(functional["structural_probe_forward_ticks"], 668 * 3 * 128)
        self.assertEqual(functional["backward_calls"], passive["backward_calls"])


class OrganizationCompleteReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        with _preserve_process_state():
            cls.protocol = study.prepare(root / "evidence", root / "raw", template=tiny())
            cls.plan = study._json(cls.protocol)
            for item in cls.plan["branches"]:
                study.train(cls.protocol, item["id"])
                study.evaluate(cls.protocol, item["id"], Path(item["output"]) / "latest.pt",
                    Path(cls.plan["raw_root"]) / "evaluation" / f"{item['id']}.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _serial_report(cls):
        if not hasattr(cls, "_serial_cache"):
            cls._serial_cache = report.build_report(cls.protocol, workers=1)
        return cls._serial_cache

    def test_all_27_runs_108_checkpoints_and_648_forks_and_artifacts(self):
        result = self._serial_report()
        self.assertEqual(result["counts"]["training_runs"], 27)
        self.assertEqual(result["counts"]["full_checkpoints"], 108)
        self.assertEqual(result["counts"]["frozen_forks"], 648)
        self.assertEqual(result["audit"]["status"], "passed")
        self.assertTrue(all(metric["n_total"] == 3 for row in result["main_summary"] for metric in row["metrics"].values()))
        out = Path(self.temporary.name) / "aggregate"
        files = report.write_artifacts(result, out)
        self.assertIn("header.json", files)
        self.assertIn("edge_lifetimes.csv", files)
        header = json.loads((out / "header.json").read_text(encoding="utf-8"))
        self.assertEqual(len(header["training_per_seed"]), 27)
        self.assertTrue(all("per_seed" not in metric for row in header["main_summary"] for metric in row["metrics"].values()))

    def test_missing_branch_is_fail_closed_and_does_not_emit_partial_result(self):
        path = Path(self.plan["raw_root"]) / "evaluation" / f"{self.plan['branches'][0]['id']}.json"
        temporary = path.with_suffix(".withheld")
        path.rename(temporary)
        try:
            with self.assertRaises(AuditError):
                report.build_report(self.protocol, workers=1)
        finally:
            temporary.rename(path)

    def test_incomplete_artifact_is_refused(self):
        with self.assertRaises(ValueError):
            report.write_artifacts(dict(complete=False), Path(self.temporary.name) / "invalid")

    def test_real_27_branch_serial_and_two_worker_reports_are_identical(self):
        serial = deepcopy(self._serial_report())
        parallel = report.build_report(self.protocol, workers=2)
        self.assertEqual(serial.pop("checkpoint_workers"), 1)
        self.assertEqual(parallel.pop("checkpoint_workers"), 2)
        # This covers input bindings, header, all summaries, source audits,
        # event coordinates/lifetimes, work and the seed-level statistics.
        self.assertEqual(report.json_hash(serial), report.json_hash(parallel))

    def test_rejection_in_a_real_worker_propagates_without_partial_report(self):
        marker = Path(self.plan["branches"][0]["output"]) / "launch-complete.json"
        original = marker.read_bytes()
        damaged = json.loads(original)
        damaged["target"] -= self.plan["native_template"]["chunk_ticks"]
        marker.write_text(json.dumps(damaged), encoding="utf-8")
        try:
            with self.assertRaises(AuditError) as caught:
                report.build_report(self.protocol, workers=2)
            self.assertEqual(caught.exception.audit["errors"][0]["code"], "COMPLETION_BINDING")
        finally:
            marker.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
