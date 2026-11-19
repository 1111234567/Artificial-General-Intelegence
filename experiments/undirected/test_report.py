"""Explicit tiny analysis fixtures; no formal checkpoint or score is read."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.run import state_digest
from experiments.undirected import plot, report, run as native, study
from experiments.undirected.queue import atomic_json, file_sha256


def toy():
    return replace(study.formal_template(), width=40, n_jobs=1, batch_size=1, initial_degree=4,
        chunk_ticks=2, main_ticks=8, warmup_ticks=4, withdrawal_ticks=4, return_ticks=2,
        macrocycle_ticks=8, block_ticks=2, replay_capacity=2, structural_window_updates=2, structural_candidates=1)


def rehash(state):
    state["state_hashes"] = {name: state_digest(state[name]) for name in native.HASH_FIELDS}
    state["state_hashes"]["semantic"] = state_digest({name: state[name] for name in native.SEMANTIC_FIELDS})


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.protocol = study.prepare(root / "evidence", root / "raw", template=toy(), mechanical=True, evaluation_steps=2)
        cls.plan = json.loads(cls.protocol.read_text(encoding="utf-8"))
        cls.success = next(item for item in cls.plan["branches"] if item["seed"] == 0 and
            report._key(item) == report.variant(structure="both"))
        cls.window = next(item for item in cls.plan["branches"] if item["seed"] == 0 and
            report._key(item) == report.variant(credit="recorded512"))
        cls.fixed = next(item for item in cls.plan["branches"] if item["seed"] == 0 and report._key(item) == report.variant())
        cls.fixed_probe = next(item for item in cls.plan["branches"] if item["seed"] == 0 and report._key(item) == report.variant(structure="fixed_probe"))
        successful = (cls.success, cls.window, cls.fixed, cls.fixed_probe)
        cls.failure = next(item for item in cls.plan["branches"] if item not in successful)
        for item in cls.plan["branches"]:
            if item in successful:
                study.train(cls.protocol, item["id"])
            else:
                # This is an explicitly injected mechanical fixture, not an
                # observed failure or performance result of a formal seed.
                with patch.object(native.NativeTrainer, "train_update", side_effect=FloatingPointError("injected empty-history mechanical fixture")):
                    study.train(cls.protocol, item["id"])
            study.evaluate(cls.protocol, item["id"], Path(item["output"]) / "checkpoint_latest.pt",
                           Path(cls.plan["raw_root"]) / "evaluation" / f"{item['id']}.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_branch_audit_replays_actual_world_labels_and_binds_all_layouts(self):
        result = report.audit_branch(self.protocol, self.success["id"], mechanical=True)
        self.assertEqual(result["branch"]["status"], "success")
        self.assertEqual((len(result["checkpoints"]), len(result["per_seed"])), (6, 40))
        self.assertEqual(result["branch"]["world_teacher_replayed_ticks"], 14)
        self.assertEqual(result["branch"]["records"], 7)
        self.assertGreater(result["checks"]["CAUSAL_TEACHER_LABEL"], 0)
        self.assertEqual(result["branch"]["sensitivity_audit"]["status"], "pending_independent_jvp")
        self.assertEqual({row["role"] for row in result["graphs"]}, {"birth", "last_valid"})
        self.assertTrue(all(row["downsampled"] is False for row in result["graphs"]))
        self.assertTrue(all(row["width"] == 40 for row in result["graphs"]))
        self.assertTrue(result["probes"])

    def test_all_42_accounted_failure_is_unscored_and_no_seed_is_dropped(self):
        audited = [report.audit_branch(self.protocol, item["id"], mechanical=True) for item in self.plan["branches"]]
        with patch.object(report, "_worker", side_effect=AssertionError("cached reports must not repeat branch replay")):
            value = report.build_report(self.protocol, workers=1, mechanical=True, cached_results=audited)
        self.assertEqual(value["audit"]["branch_evidence"], "revalidated_branch_audit_cache")
        self.assertEqual(value["audit"]["scope"], "provenance_and_accounting")
        fresh = report.build_report(self.protocol, workers=1, mechanical=True)
        for name in ("branches", "checkpoints", "per_seed", "graphs", "fixed_probe_comparison", "inputs",
                     "family_average_per_seed", "family_average_summary", "family_average_paired_contrasts"):
            self.assertEqual(value[name], fresh[name], name)
        with self.assertRaisesRegex(report.AuditError, "CACHED_BRANCH_RESULT_DECLARED_ORDER"):
            report.build_report(self.protocol, mechanical=True, cached_results=list(reversed(audited)))
        self.assertEqual(value["counts"], dict(declared_branches=42, accounted_branches=42,
            successful_branches=4, numerical_failure_branches=38, declared_forks=1680, actual_forks=160))
        self.assertEqual(len(value["branches"]), 42)
        self.assertEqual(len(value["family_average_per_seed"]), 42 * 10)
        self.assertTrue(all(row["mean"] is None and row["sample_sd"] is None for row in value["family_average_summary"]))
        with tempfile.TemporaryDirectory() as folder:
            header = report.write_artifacts(value, folder)
            self.assertEqual(header["audit"]["status"], "passed")
            plot._comparison_contract(header)
            self.assertEqual(header["tables"]["branches"]["rows"], 42)
            self.assertEqual(header["tables"]["family_average_per_seed"]["rows"], 420)
            self.assertEqual(header["graph_artifact"]["rows"], 84)
            graphs = json.loads(Path(header["graph_artifact"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(graphs["format"], "undirected-audited-graphs-v1")
            self.assertEqual(file_sha256(header["graph_artifact"]["path"]), header["graph_artifact"]["sha256"])
            plot._check_graphs(graphs["graphs"], value["checkpoints"])
            derived = plot.family_values(value["branches"], value["per_seed"], value["checkpoint_ticks"][1:])
            plot._check_family_tables(derived, value["family_average_per_seed"], value["family_average_summary"])
            self.assertLess((Path(folder) / "report.json").stat().st_size, 100000)
        with self.assertRaises(ValueError):
            report.build_report(self.protocol, workers=7, mechanical=True)
        with self.assertRaisesRegex(report.AuditError, "explicit"):
            report.build_report(self.protocol)

    def _paired_audits(self):
        if not hasattr(type(self), "paired_audits"):
            type(self).paired_audits = [report.audit_branch(self.protocol, item["id"], mechanical=True)
                for item in self.plan["branches"] if report._key(item) in (report.variant(), report.variant(structure="fixed_probe"))]
        return deepcopy(type(self).paired_audits)

    def test_exact_learning_difference_remains_failed_without_hiding_behavior(self):
        original = self._paired_audits()
        baseline = report._fixed_probe_comparison(original)
        self.assertEqual(baseline["status"], "passed")
        target = next(row for row in original if row["branch"]["id"] == self.fixed_probe["id"])
        point = target["trace"][1]
        point["parameter_hash"] = "0" * 64
        point["physical_matrix"]["operator_norm"] = math.nextafter(point["physical_matrix"]["operator_norm"], float("inf"))
        finding = report._fixed_probe_comparison(original)
        row = finding["per_seed"][0]["training_trace"]
        self.assertEqual((finding["status"], row["status"], row["mismatch_count"]), ("failed", "failed", 1))
        self.assertEqual(row["first_mismatch_tick"], point["tick"])
        self.assertEqual(row["fields"]["parameter_hash"]["mismatch_count"], 1)
        self.assertEqual(row["fields"]["physical_matrix"]["mismatch_count"], 1)
        self.assertEqual(row["groups"]["learning_state"]["status"], "different")
        self.assertEqual(row["groups"]["behavior"]["status"], "equal")
        self.assertEqual(row["groups"]["external_stream"]["status"], "equal")
        self.assertEqual(finding["per_seed"][0]["frozen_evaluation"]["status"], "passed")
        from experiments.undirected import plot, results
        header = dict(audit=dict(scope="provenance_and_accounting", scopes=dict(
            fixed_probe_exact_training_equality="failed", fixed_probe_frozen_evaluation_comparison="equal")),
            fixed_probe_comparison=finding)
        plot._comparison_contract(header)
        prose = "\n".join(results.audit_disclosure(header))
        self.assertIn("原预期未通过", prose)
        self.assertIn("事后修订", prose)
        self.assertIn("原因尚未确定", prose)
        header["fixed_probe_comparison"]["status"] = "passed"
        with self.assertRaisesRegex(plot.PlotDataError, "silently relabeled"):
            plot._comparison_contract(header)

    def test_training_actions_fork_actions_and_internal_diagnostic_counts_are_distinct(self):
        original = self._paired_audits()
        target = next(row for row in original if row["branch"]["id"] == self.fixed_probe["id"])
        target["trace"][1]["action_hash"] = "0" * 64
        target["trace"][1]["live_world_hash"] = "1" * 64
        finding = report._fixed_probe_comparison(original)
        row = finding["per_seed"][0]
        self.assertEqual(row["training_trace"]["groups"]["behavior"]["status"], "different")
        self.assertEqual(row["training_trace"]["groups"]["learning_state"]["status"], "equal")
        self.assertEqual(row["frozen_evaluation"]["status"], "passed")
        original = self._paired_audits()
        target = next(row for row in original if row["branch"]["id"] == self.fixed_probe["id"])
        target["evaluation_trace"][0]["action_hash"] = "0" * 64
        finding = report._fixed_probe_comparison(original)
        self.assertEqual(finding["status"], "passed")
        self.assertEqual(finding["per_seed"][0]["frozen_evaluation"]["groups"]["actions"]["status"], "different")
        original = self._paired_audits()
        target = next(row for row in original if row["branch"]["id"] == self.fixed_probe["id"])
        target["evaluation_trace"][0]["request_diagnosis"]["primary"]["requests"] += 1
        target["evaluation_trace"][0]["request_diagnosis"]["primary"]["observed_first_error_age_sum"] += .125
        finding = report._fixed_probe_comparison(original)
        evaluated = finding["per_seed"][0]["frozen_evaluation"]
        self.assertEqual(finding["status"], "passed")
        self.assertEqual(evaluated["groups"]["actions"]["status"], "equal")
        self.assertEqual(evaluated["groups"]["world_and_behavior_counts"]["status"], "equal")
        self.assertEqual(evaluated["groups"]["request_and_content_diagnostics"]["mismatch_count"], 1)
        self.assertEqual(evaluated["numeric_leaf_differences"]["integer"]["record_count"], 1)
        self.assertEqual(evaluated["numeric_leaf_differences"]["float"]["record_count"], 1)

    def test_external_command_mismatch_still_fails_parent_audit(self):
        paired = self._paired_audits()
        fixed = {row["branch"]["seed"]: row for row in paired if report._key(row["branch"]) == report.variant()}
        matching = {row["branch"]["id"]: row for row in paired}
        # Parent-check fixture uses the original real audited traces.  Other
        # identities only exercise the complete parent inventory contract.
        rows = [matching.get(item["id"], dict(branch={**item, "status": "success"},
            trace=deepcopy(fixed[item["seed"]]["trace"]), evaluation_trace=[])) for item in self.plan["branches"]]
        report._parent_checks(rows, self.plan, report.Checks())
        target = next(row for row in rows if row["branch"]["id"] == self.fixed_probe["id"])
        target["trace"][1]["event_hash"] = "0" * 64
        with self.assertRaisesRegex(report.AuditError, "MATCHED_REAL_EXTERNAL_COMMAND_STREAM"):
            report._parent_checks(rows, self.plan, report.Checks())

    def test_zero_score_cannot_be_inserted_into_failed_evaluation(self):
        path = Path(self.plan["raw_root"]) / "evaluation" / f"{self.failure['id']}.json"
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value["records"] = [dict(score=0)]
            value["actual_forks"] = 1
            value["result_hash"] = state_digest({key: item for key, item in value.items() if key != "result_hash"})
            atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, "empty unscored"):
                report.audit_branch(self.protocol, self.failure["id"], mechanical=True)
        finally:
            path.write_bytes(original)

    def test_rehashed_evaluation_rates_and_fake_old_input_layout_are_rejected(self):
        path = Path(self.plan["raw_root"]) / "evaluation" / f"{self.success['id']}.json"
        original = path.read_bytes()
        for kind in ("count", "layout", "evaluator_source"):
            with self.subTest(kind=kind):
                try:
                    value = json.loads(original)
                    row = value["records"][0]
                    if kind == "count":
                        row["primary"]["clean_completed"] = row["primary"]["matured"] + 1
                    elif kind == "layout":
                        row["model"]["input_size"] = 112
                    else:
                        row["source_hashes"] = {"old_native.py": "0" * 64}
                    value["result_hash"] = state_digest({key: item for key, item in value.items() if key != "result_hash"})
                    atomic_json(path, value)
                    with self.assertRaises(report.AuditError):
                        report.audit_branch(self.protocol, self.success["id"], mechanical=True)
                finally:
                    path.write_bytes(original)

    def test_latest_cannot_diverge_from_canonical_even_with_valid_new_hashes(self):
        folder = Path(self.success["output"])
        latest, terminal_path = folder / "checkpoint_latest.pt", folder / "terminal.json"
        original_cp, original_terminal = latest.read_bytes(), terminal_path.read_bytes()
        try:
            state = torch.load(latest, map_location="cpu", weights_only=False)
            reserved = set()
            for event in state["operator"]["event_history"]:
                reserved.update(event["proposal"]["birth_indices"].tolist())
                reserved.update(event["proposal"]["fade_indices"].tolist())
            index = next(i for i in state["operator"]["mask"].nonzero().flatten().tolist() if i not in reserved)
            state["model"]["edge_shared"][index] += .001
            loaded = native.NativeTrainer.from_checkpoint(latest)
            loaded.model.load_state_dict(state["model"])
            state["trace"][-1]["physical_matrix"] = loaded._matrix_diagnostics()
            state["trace"][-1]["parameter_hash"] = state_digest(state["model"])
            point = state["trace"][-1]
            point["continuity_hash"] = state_digest({name: point[name] for name in (
                "tick", "parameter_hash", "optimizer_hash", "rng_hash", "live_hidden_hash", "live_world_hash",
                "credit_hash", "operator_hash", "reservoir_hash", "action_hash", "data_hash")})
            rehash(state)
            torch.save(state, latest)
            terminal = json.loads(original_terminal)
            terminal["checkpoint_sha256"] = file_sha256(latest)
            terminal["checkpoint_semantic_hash"] = state_digest({key: state[key] for key in native.SEMANTIC_FIELDS})
            atomic_json(terminal_path, terminal)
            with self.assertRaisesRegex(report.AuditError, "LATEST_EQUALS_CANONICAL|CANONICAL_TRACE_IS_ORIGINAL_PREFIX"):
                report.audit_branch(self.protocol, self.success["id"], mechanical=True)
        finally:
            latest.write_bytes(original_cp)
            terminal_path.write_bytes(original_terminal)

    def test_independent_jvp_evidence_needs_real_projection_rows_and_inputs(self):
        from experiments.undirected import sensitivity_audit
        terminal_path = Path(self.success["output"]) / "terminal.json"
        evidence = sensitivity_audit.audit_terminal(terminal_path)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            atomic_json(path, evidence)
            result = report.audit_branch(self.protocol, self.success["id"], mechanical=True, sensitivity=path)
            self.assertTrue(result["branch"]["sensitivity_audit"]["calibrated"])
            damaged = deepcopy(evidence)
            damaged["checkpoints"] = []
            atomic_json(path, damaged)
            with self.assertRaisesRegex(report.AuditError, "JVP_EXACT_CHECKPOINT_INVENTORY"):
                report.audit_branch(self.protocol, self.success["id"], mechanical=True, sensitivity=path)
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            for kind in ("direction", "dynamic_layout", "analysis_source", "runtime_fingerprint", "duplicate_checkpoint", "age", "bool_ratio", "records"):
                with self.subTest(kind=kind):
                    damaged = deepcopy(evidence)
                    if kind == "direction":
                        damaged["direction_hash"] = "0" * 64
                    elif kind == "dynamic_layout":
                        damaged["dynamic_names"] = []
                    elif kind == "analysis_source":
                        damaged["analysis_sources"].pop(str((study.ROOT / "experiments/undirected/ledger.py").resolve()))
                    elif kind == "runtime_fingerprint":
                        damaged["input_fingerprints"].pop(str((study.ROOT / "experiments/undirected/model.py").resolve()))
                    elif kind == "duplicate_checkpoint":
                        damaged["checkpoints"].append(deepcopy(damaged["checkpoints"][0]))
                        damaged["checkpoint_count"] += 1
                    elif kind == "age":
                        damaged["checkpoints"][-1]["update_age"] += 1
                    elif kind == "bool_ratio":
                        damaged["checkpoints"][-1]["projection_comparisons"][0]["max_tolerance_ratio"] = True
                    else:
                        damaged["records"] = []
                    atomic_json(path, damaged)
                    with self.assertRaises(report.AuditError):
                        report._sensitivity(path, terminal_path, terminal, report.Checks(), {})

    def test_window_audit_is_primal_only_and_cannot_omit_its_history(self):
        from experiments.undirected import sensitivity_audit
        terminal_path = Path(self.window["output"]) / "terminal.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        evidence = sensitivity_audit.audit_terminal(terminal_path)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            atomic_json(path, evidence)
            result = report._sensitivity(path, terminal_path, terminal, report.Checks(), {})
            self.assertFalse(result["calibrated"])
            self.assertIn("no full saved S", result["scope"])
            damaged = deepcopy(evidence)
            damaged["records"] = []
            atomic_json(path, damaged)
            with self.assertRaisesRegex(report.AuditError, "JVP_RECORD_ROWS"):
                report._sensitivity(path, terminal_path, terminal, report.Checks(), {})

    def test_command_inventory_is_checked_before_raw_branch_audits(self):
        path = self.protocol.parent / "training-commands-v1.json"
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value["jobs"][0]["command"][-1] = "foreign-branch"
            atomic_json(path, value)
            with patch.object(report, "_worker", side_effect=AssertionError("raw branch opened too early")), \
                 self.assertRaisesRegex(report.AuditError, "EXACT_SOURCE_BOUND_COMMAND"):
                report.build_report(self.protocol, mechanical=True)
        finally:
            path.write_bytes(original)


class StatisticTests(unittest.TestCase):
    def test_three_seed_summary_is_fail_closed(self):
        rows = [dict(seed=seed, status="success", value=float(seed)) for seed in study.SEEDS]
        result = report.summarize_declared(rows)
        self.assertEqual((result["mean"], result["sample_sd"]), (1., 1.))
        rows[2].update(status="numerical_failure", value=None)
        result = report.summarize_declared(rows)
        self.assertEqual(result["failure_count"], 1)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["sample_sd"])
        with self.assertRaises(report.AuditError):
            report.summarize_declared(rows[:2])

    def test_family_mean_is_within_seed_and_requires_all_four_families(self):
        branches, values = [], []
        for seed in study.SEEDS:
            for v in study.VARIANTS:
                identifier = f"{seed}/{report._key(v)}"
                branches.append(dict(id=identifier, seed=seed, status="success", **v))
                for family in range(4):
                    values.append(dict(id=identifier, seed=seed, **v, tick=8, family=family, condition="short", steps=2,
                        primary_matured=1 if family == 0 else 1000, **{metric: seed + family / 10 for metric in report.METRICS}))
        specs = [dict(tick=8, family=family, condition="short", steps=2) for family in range(4)]
        individuals, summary, contrasts = report._family_tables(branches, values, dict(evaluation_specs=specs))
        row = next(row for row in individuals if row["seed"] == 0 and report._key(row) == report.variant())
        self.assertAlmostEqual(row["strict_completion"], .15)
        result = next(row for row in summary if report._key(row) == report.variant() and row["metric"] == "strict_completion")
        self.assertAlmostEqual(result["mean"], 1.15)
        self.assertAlmostEqual(result["sample_sd"], 1.)
        self.assertEqual(len(contrasts), len(report.CONTRASTS) * len(report.METRICS))
        values = [row for row in values if not (row["seed"] == 2 and report._key(row) == report.variant() and row["family"] == 3)]
        _, summary, _ = report._family_tables(branches, values, dict(evaluation_specs=specs))
        result = next(row for row in summary if report._key(row) == report.variant() and row["metric"] == "strict_completion")
        self.assertEqual(result["n_success"], 3)
        self.assertEqual(result["n_defined"], 2)
        self.assertIsNone(result["mean"])


class AuditScopeContractTests(unittest.TestCase):
    """Tiny neural-only audit fixtures; no 42-branch setup or study data."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, folder, method):
        from experiments.undirected.test_sensitivity_audit import _fixture
        from experiments.undirected import ledger, sensitivity_audit
        branch = dict(id="mechanical-report-primal-contract", credit=method,
                      native_config=dict(credit=method, chunk_ticks=2))
        binding = dict(protocol_sha256="a" * 64, branch=branch,
                       recording_convention="committed-live-block-pre-parameter-trajectory-v1")
        _, checkpoints, _ = _fixture(folder, method=method, binding=binding)
        writer = ledger.LiveLedger(Path(folder) / "live_records", binding)
        terminal = dict(format="undirected-branch-terminal-v1", branch=branch, protocol_sha256="a" * 64,
            status="success", completed=True, endpoint_comparable=True, last_tick=6, target_ticks=6,
            checkpoint_path=str(checkpoints[6]), checkpoint_sha256=ledger.file_sha256(checkpoints[6]),
            checkpoints={str(tick): dict(path=str(path), sha256=ledger.file_sha256(path)) for tick, path in checkpoints.items()},
            ledger=writer.summary(updates=3, require_exact=True))
        terminal_path = Path(folder) / "terminal.json"
        atomic_json(terminal_path, terminal)
        evidence = sensitivity_audit.audit_terminal(terminal_path)
        output = Path(folder) / "audit.json"
        atomic_json(output, evidence)
        return terminal_path, terminal, output, evidence

    def test_no_saved_s_primal_contract_uses_one_forward_and_zero_tangents(self):
        for method in ("tbptt128", "recorded512", "recorded1024"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                terminal_path, terminal, output, evidence = self.fixture(folder, method)
                self.assertEqual(evidence["work"], dict(jvp_calls=0, model_step_calls=6,
                    credit_engine_jacobian_calls=0, optimizer_steps=0, world_steps=0))
                self.assertTrue(all(len(row["primal_comparisons"]) == 1 and row["tangent_norms"] == []
                                    for row in evidence["records"]))
                result = report._sensitivity(output, terminal_path, terminal, report.Checks(), {})
                self.assertFalse(result["calibrated"])
                self.assertIn("zero JVP", result["scope"])
                for kind in ("old_jvp_work", "bool_work", "float_work", "missing_primal", "duplicate_primal",
                             "tangent", "tangent_reset", "false_calibration"):
                    with self.subTest(kind=kind):
                        damaged = deepcopy(evidence)
                        if kind == "old_jvp_work":
                            damaged["work"].update(jvp_calls=6, model_step_calls=12)
                        elif kind == "bool_work":
                            damaged["work"]["jvp_calls"] = False
                        elif kind == "float_work":
                            damaged["work"]["model_step_calls"] = 6.
                        elif kind == "missing_primal":
                            damaged["records"][0]["primal_comparisons"] = []
                        elif kind == "duplicate_primal":
                            damaged["records"][0]["primal_comparisons"].append(deepcopy(damaged["records"][0]["primal_comparisons"][0]))
                        elif kind == "tangent":
                            damaged["records"][0]["tangent_norms"] = [dict(l2=1.)]
                        elif kind == "tangent_reset":
                            damaged["records"][0]["tangent_reset"] = True
                        else:
                            damaged["sensitivity_calibrated"] = True
                        atomic_json(output, damaged)
                        with self.assertRaises(report.AuditError):
                            report._sensitivity(output, terminal_path, terminal, report.Checks(), {})

    def test_saved_s_projection_contract_keeps_two_directions_and_work(self):
        for method in ("rtrl", "rtrl_reset"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                terminal_path, terminal, output, evidence = self.fixture(folder, method)
                self.assertEqual(evidence["work"]["jvp_calls"], 6)
                self.assertEqual(evidence["work"]["model_step_calls"], 12)
                self.assertTrue(all(len(row["primal_comparisons"]) == len(row["tangent_norms"]) == 2
                                    for row in evidence["records"]))
                self.assertTrue(report._sensitivity(output, terminal_path, terminal, report.Checks(), {})["calibrated"])
                for kind in ("primal", "tangent", "zero_jvp_work"):
                    damaged = deepcopy(evidence)
                    if kind == "primal":
                        damaged["records"][0]["primal_comparisons"].pop()
                    elif kind == "tangent":
                        damaged["records"][0]["tangent_norms"] = []
                    else:
                        damaged["work"].update(jvp_calls=0, model_step_calls=6)
                    atomic_json(output, damaged)
                    with self.assertRaises(report.AuditError):
                        report._sensitivity(output, terminal_path, terminal, report.Checks(), {})


if __name__ == "__main__":
    unittest.main()
