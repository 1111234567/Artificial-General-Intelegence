"""Small real-state fixtures; no formal training or mocked performance claims."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong import evaluate, report, run
from experiments.lifelong.replay import HistoryReservoir
from experiments.loops.report import AuditError, _Checks, file_sha256


def tiny(**changes):
    values = dict(width=8, batch_size=2, chunk_ticks=4, initial_fan_in=2,
        replay_capacity=2, block_ticks=4, macrocycle_ticks=16, main_ticks=32,
        withdrawal_ticks=16, return_ticks=8, structural_window_updates=3)
    values.update(changes)
    return run.NativeConfig(**values)


def rehash(state):
    for key in state["state_hashes"]:
        if key != "semantic":
            state["state_hashes"][key] = state_digest(state[key])
    state["state_hashes"]["semantic"] = state_digest({key: state[key] for key in run._SEMANTIC_FIELDS})


def prepare_original_protocol(folder, raw):
    """Build synthetic fixtures from the original frozen execution source set."""
    from experiments.lifelong import study
    native = (study.ROOT / "experiments/lifelong").resolve()
    original_glob = Path.glob

    def original_sources(path, pattern):
        if path.resolve() == native and pattern == "*.py":
            return (native / name for name in report.MAIN_NATIVE_SOURCE_FILES)
        return original_glob(path, pattern)

    # Later analysis scripts must not change the historical protocol fixture.
    with patch.object(Path, "glob", autospec=True, side_effect=original_sources):
        study.prepare(folder, raw)


class NativeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with _preserve_process_state():
            trainer = run.NativeTrainer(tiny(learning="history_replay"))
            cls.born = trainer.state_dict()
            trainer.run_updates(6)
            cls.state = trainer.state_dict()
            cls.cp = report.audit_checkpoint_payload(cls.state, tick=24)
            cls.plan = dict(dt=.25, batch_size=2,
                sources={str((report.ROOT / key).resolve()): file_sha256(report.ROOT / key)
                         for key in report.EVALUATION_SOURCE_FILES})
            cls.spec = dict(tick=24, family=1, condition="long_wait", steps=80, purpose="fixture")
            cls.record = evaluate.evaluate(trainer.model, trainer.world, trainer.h,
                family=1, condition="long_wait", steps=80, device="cpu")
            cls.record["probe"] = cls.spec

    def test_audit_real_native_birth_and_trained_checkpoint(self):
        born = report.audit_checkpoint_payload(self.born, tick=0)
        trained = report.audit_checkpoint_payload(self.state, tick=24)
        self.assertEqual(born["birth_hash"], trained["birth_hash"])
        self.assertEqual(trained["work"], report.expected_work(24, 4, 2))
        self.assertEqual(trained["checks"]["status"], "passed")
        self.assertEqual(trained["replay"]["complete_blocks_seen"], 3)
        self.assertEqual(trained["auxiliary_stats"]["history_replay"], 4)
        self.assertEqual(trained["model_config"]["input_size"], 112)

    def test_native_adapter_preserves_long_wait_range_and_cue_accounting(self):
        before = report.json_hash(self.record)
        audit = _Checks()
        metrics = report.audit_native_record(self.record, self.cp, plan=self.plan, spec=self.spec, audit=audit)
        self.assertIn("strict_completion", metrics)
        self.assertIn("inactive_per_1000_idle", metrics)
        self.assertEqual(audit.result()["status"], "passed")
        self.assertGreater(audit.counts["CUE_PAIR_COUNTS"], 0)
        self.assertEqual(self.record["regime"]["duration_range"], (8., 16.))
        self.assertEqual(report.json_hash(self.record), before)

    def test_wrong_native_model_rule_source_or_origin_rejected(self):
        mutations = (("family_mapping", [0, 1, 2, 3]), ("context_policy", [0]),
                     ("format", "integration-own-trace-v2"), ("parameter_hash", "0" * 64))
        for key, value in mutations:
            record = deepcopy(self.record)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(AuditError):
                report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)
        record = deepcopy(self.record)
        record["model"]["input_size"] = 96
        with self.assertRaisesRegex(AuditError, "NATIVE_MODEL_CONFIG"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)
        record = deepcopy(self.record)
        record["source_hashes"][next(iter(record["source_hashes"]))] = "0" * 64
        with self.assertRaisesRegex(AuditError, "SOURCE_BINDING"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)

    def test_denominator_and_selected_family_forgery_rejected(self):
        record = deepcopy(self.record)
        record["request_diagnosis"]["primary"]["cue_legal_pair_count"] += 1
        with self.assertRaisesRegex(AuditError, "CUE_PAIR_PARTITION"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)
        record = deepcopy(self.record)
        record["actual_family_exposures"]["new"][0] += 1
        record["actual_family_exposures"]["commands"][0] += 1
        with self.assertRaisesRegex(AuditError, "NEW_ONLY_SELECTED"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)

    def test_unknown_record_field_and_current_or_future_history_rejected(self):
        record = deepcopy(self.record)
        record["unreviewed"] = True
        with self.assertRaisesRegex(AuditError, "EXACT_SCHEMA"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)
        state = deepcopy(self.state)
        state["trace"][2]["auxiliary"]["end_tick"] = 12
        state["trace"][2]["auxiliary"]["start_tick"] = 4
        rehash(state)
        with self.assertRaisesRegex(AuditError, "NO_FUTURE_REPLAY"):
            report.audit_checkpoint_payload(state)

    def test_original_labels_rehash_to_live_chunks_even_after_claimed_digests_repaired(self):
        state = deepcopy(self.state)
        state["replay"]["blocks"][0]["targets"][0, 0, 0] = 10
        memory = HistoryReservoir(2, 8)
        memory.load_state_dict(state["replay"])
        state["trace"][-1]["reservoir_hash"] = memory.retention_fingerprint()
        rehash(state)
        with self.assertRaisesRegex(AuditError, "HISTORY_ORIGINAL_LABELLED_DATA|original causal live digest"):
            report.audit_checkpoint_payload(state)

    def test_original_family_context_cannot_be_relabelled_after_retention(self):
        state = deepcopy(self.state)
        original = state["replay"]["blocks"][0]["family_context"][0, 0, 0].item()
        state["replay"]["blocks"][0]["family_context"][0, 0, 0] = (original + 1) % 4
        memory = HistoryReservoir(2, 8)
        memory.load_state_dict(state["replay"])
        state["trace"][-1]["reservoir_hash"] = memory.retention_fingerprint()
        rehash(state)
        with self.assertRaisesRegex(AuditError, "HISTORY_ORIGINAL_LABELLED_DATA|original causal live digest"):
            report.audit_checkpoint_payload(state)

    def test_admission_and_sampling_rngs_are_replayed_from_their_private_birth_seed(self):
        for key, expected_error in (("admission_rng_state", "PRIVATE_ADMISSION_RNG_EXACT"),
                                    ("sample_rng_state", "PRIVATE_SAMPLING_RNG_EXACT")):
            state = deepcopy(self.state)
            state["replay"][key] = random.Random(919).getstate()
            memory = HistoryReservoir(2, 8)
            memory.load_state_dict(state["replay"])
            state["trace"][-1]["reservoir_hash"] = memory.retention_fingerprint()
            rehash(state)
            with self.subTest(key=key), self.assertRaisesRegex(AuditError, expected_error):
                report.audit_checkpoint_payload(state)

    def test_omitted_optimizer_step_or_work_rejected(self):
        for key, value in (("aux_forward_ticks", 24), ("aux_warmup_ticks", 0), ("optimizer_steps", 5)):
            state = deepcopy(self.state)
            state["work"][key] = value
            rehash(state)
            with self.subTest(key=key), self.assertRaises(AuditError):
                report.audit_checkpoint_payload(state)

    def test_seed_is_only_statistical_replicate_and_undefined_values_retained(self):
        values = [dict(seed=seed, value=value) for seed, value in enumerate((.2, None, .8))]
        result = report.summarize_seed_values(values)
        self.assertEqual((result["n"], result["n_total"], result["n_undefined"]), (2, 3, 1))
        self.assertAlmostEqual(result["mean"], .5)
        self.assertEqual(len(result["per_seed"]), 3)
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            report.summarize_seed_values(values + [dict(seed=0, value=.3)])

    def test_paired_effects_use_seed_differences_not_difference_of_pooled_opportunities(self):
        rows = []
        for seed in range(3):
            for learning, value in (("recent_repeat", .1 + .1 * seed), ("history_replay", .4 + .05 * seed)):
                rows.append(dict(seed=seed, architecture="fixed_shrink", schedule="mixed", learning=learning,
                    coverage="wide", tick=24, purpose="fixture", family=0, condition="long_both", steps=1024,
                    metrics=dict(strict_completion=value, inactive_per_1000_idle=None)))
        contrasts = report.paired_contrasts(rows)
        self.assertEqual(len(contrasts), 3)
        summary = report._contrast_summaries(contrasts)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["metrics"]["strict_completion"]["mean"], .25)
        self.assertEqual(summary[0]["metrics"]["inactive_per_1000_idle"]["n_undefined"], 3)

    def test_family_average_precedes_seed_statistics(self):
        rows = [dict(seed=seed, architecture="fixed_shrink", schedule="mixed", learning="history_replay",
            coverage="wide", tick=24, purpose="fixture", family=family, condition="long_both", steps=1024,
            metrics=dict(strict_completion=.1 * family + .01 * seed)) for seed in range(3) for family in range(4)]
        average = report._family_averages(rows)
        self.assertEqual(len(average), 3)
        summary = report._summaries(average, (*report.BRANCH_KEYS, "tick", "purpose", "condition", "steps"))
        self.assertEqual(summary[0]["metrics"]["strict_completion"]["n"], 3)
        self.assertAlmostEqual(summary[0]["metrics"]["strict_completion"]["mean"], .16)

    def test_partial_protocol_and_excess_workers_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = root / "protocol.json"
            protocol.write_text('{"format":"lifelong-study-protocol-v1"}', encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "EXACT_SCHEMA"):
                report.build_report(protocol, root, supplement=root / "supplement.json", workers=1)
            with self.assertRaisesRegex(ValueError, "supplement protocol is required"):
                report.build_report(protocol, root, workers=1)
            with self.assertRaises(ValueError):
                report.build_report(protocol, root, workers=7)
            with self.assertRaisesRegex(ValueError, "complete passed"):
                report.write_artifacts(dict(complete=False, audit=dict(status="passed")), root / "bad")
            self.assertFalse((root / "bad").exists())

    def test_branch_worker_loads_full_disk_snapshots_report_sources_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary, _preserve_process_state():
            folder = Path(temporary) / "training" / "fixture"
            config = tiny(learning="history_replay", seed=5)
            trainer = run.NativeTrainer(config)
            checkpoints = (0, 8, 24, 56)
            plan = asdict(config)
            plan.update(training_device="cpu", target_ticks=56, checkpoint_ticks=list(checkpoints))
            item = dict(id="tiny_fixture", seed=5, architecture=config.architecture, schedule=config.schedule,
                        learning=config.learning, coverage=config.coverage, output=str(folder))
            entries, protocol_hash = {}, "a" * 64
            for tick in checkpoints:
                trainer.run_updates((tick - trainer.ticks) // config.chunk_ticks)
                path = folder / "checkpoints" / f"tick_{tick:06d}.pt"
                trainer.save_checkpoint(path)
                entries[str(tick)] = dict(path=str(path), sha256=file_sha256(path),
                                          semantic_hash=state_digest(trainer.semantic_state()))
            trainer.save_checkpoint(folder / "latest.pt")
            run.write_run(folder, trainer)
            marker = dict(format="lifelong-launch-complete-v1", branch=item, protocol_sha256=protocol_hash,
                target=56, stage_ticks=56, checkpoint_sha256=file_sha256(folder / "latest.pt"),
                checkpoints=entries, hidden_resets=0)
            report._atomic_json(folder / "launch-complete.json", marker)
            report._atomic_json(folder.with_name(folder.name + ".launch-start.json"),
                dict(format="lifelong-launch-start-v1", branch=item, protocol_sha256=protocol_hash))
            result = report._branch_worker((plan, item, protocol_hash))
            self.assertNotIn("failure", result, result.get("failure"))
            self.assertEqual(set(result["checkpoints"]), set(checkpoints))
            self.assertEqual(result["audit"]["status"], "passed")
            self.assertGreater(result["audit"]["checks_by_code"]["HISTORY_ORIGINAL_LABELLED_DATA"], 0)
            self.assertIn(str(folder / "trace.jsonl"), result["inputs"])
            marker["unrecognized"] = True
            report._atomic_json(folder / "launch-complete.json", marker)
            rejected = report._branch_worker((plan, item, protocol_hash))
            self.assertEqual(rejected["failure"]["errors"][0]["code"], "EXACT_SCHEMA")

    def test_birth_is_reconstructed_from_private_seed_and_not_only_a_claimed_hash(self):
        state = deepcopy(self.born)
        state["model"]["encoder.weight"][0, 0] += .01
        state["birth"]["initial_parameter_hash"] = state_digest(state["model"])
        rehash(state)
        with self.assertRaisesRegex(AuditError, "REGENERATED_RANDOM_BIRTH"):
            report.audit_checkpoint_payload(state)

    def test_formal_protocol_preparation_and_source_set_are_independently_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, raw = Path(temporary) / "evidence", Path(temporary) / "raw"
            with redirect_stdout(io.StringIO()):
                prepare_original_protocol(folder, raw)
            path = folder / "study-protocol-v1.json"
            audit, inputs = _Checks(), {}
            plan, digest = report._formal_protocol(path, raw, audit, inputs)
            self.assertEqual((len(plan["branches"]), len(plan["checkpoint_ticks"]), len(plan["evaluation_specs"])), (30, 13, 46))
            self.assertEqual(digest, file_sha256(path))
            self.assertEqual(len(plan["sources"]), 29)
            for name in ("plot.py", "finalize.py"):
                mutated = deepcopy(plan)
                mutated["sources"][str((report.ROOT / "experiments/lifelong" / name).resolve())] = "0" * 64
                report._atomic_json(path, mutated)
                with self.subTest(source=name), self.assertRaisesRegex(AuditError, "COMPLETE_PROTOCOL_SOURCE_SET"):
                    report._formal_protocol(path, raw, _Checks(), {})
            plan["sources"].pop(next(iter(plan["sources"])))
            report._atomic_json(path, plan)
            with self.assertRaisesRegex(AuditError, "COMPLETE_PROTOCOL_SOURCE_SET"):
                report._formal_protocol(path, raw, _Checks(), {})

    def test_malformed_nested_record_is_a_structured_rejection(self):
        record = deepcopy(self.record)
        record["primary"] = None
        with self.assertRaisesRegex(AuditError, "MALFORMED_NATIVE_EVALUATION"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)

    def test_diagnostic_oracle_cannot_be_published_as_native_autonomous_performance(self):
        record = deepcopy(self.record)
        record.update(intervention="finish_content_oracle", action_policy="diagnostic_counterfactual",
                      action_hash="0" * 64, observation_hash="0" * 64)
        with self.assertRaisesRegex(AuditError, "NATIVE_AUTONOMOUS_INTERVENTION"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)
        record = deepcopy(self.record)
        record["action_policy"] = "diagnostic_counterfactual"
        with self.assertRaisesRegex(AuditError, "NATIVE_AUTONOMOUS_POLICY"):
            report.audit_native_record(record, self.cp, plan=self.plan, spec=self.spec)

    def test_birth_auxiliary_hidden_shape_is_authenticated_against_native_model(self):
        with _preserve_process_state():
            trainer = run.NativeTrainer(tiny())
            trainer.run_updates(1)
            state = trainer.state_dict()
        state["last_auxiliary"]["block"]["start_h"] = torch.zeros(2, 99)
        state["last_auxiliary"]["info"]["source_hash"] = report.block_hash(state["last_auxiliary"]["block"])
        rehash(state)
        with self.assertRaisesRegex(AuditError, "HISTORY_NATIVE_MODEL_SCHEMA"):
            report.audit_checkpoint_payload(state)

    def test_last_auxiliary_metadata_cannot_contradict_the_original_trace(self):
        state = deepcopy(self.state)
        state["last_auxiliary"]["info"]["loss"] += .5
        rehash(state)
        with self.assertRaisesRegex(AuditError, "LAST_AUXILIARY_TRACE_BINDING"):
            report.audit_checkpoint_payload(state)

    @unittest.skipUnless(torch.cuda.is_available(), "Project CUDA runtime unavailable")
    def test_real_cuda_checkpoint_and_saved_report_reconstruct_exactly_on_cpu(self):
        with _preserve_process_state():
            trainer = run.NativeTrainer(tiny(device="cuda", learning="history_replay"))
            trainer.run_updates(3)
            state, saved_report = trainer.state_dict(), trainer.report()
            checked = report.audit_checkpoint_payload(state)
            recreated = deepcopy(checked["report"])
            recreated["config"]["device"] = "cuda"
            recreated["runtime"] = state["runtime"]
            self.assertEqual(report.json_hash(saved_report), report.json_hash(recreated))
            self.assertEqual(checked["semantic_hash"], state["state_hashes"]["semantic"])

    def test_supplement_protocol_authenticates_parent_sources_specs_and_no_training_changes(self):
        from experiments.lifelong import supplement
        with tempfile.TemporaryDirectory() as temporary:
            folder, raw = Path(temporary) / "evidence", Path(temporary) / "raw"
            with redirect_stdout(io.StringIO()):
                prepare_original_protocol(folder, raw)
                supplement.prepare(folder / "study-protocol-v1.json", folder, Path(temporary) / "coverage")
            parent = folder / "study-protocol-v1.json"
            path = folder / "coverage-probe-supplement-v1.json"
            plan, digest = report._supplement_protocol(path, parent, file_sha256(parent), _Checks(), {})
            self.assertEqual(len(plan["evaluation_specs"]), 16)
            self.assertEqual(digest, file_sha256(path))
            changes = (("parent_protocol_sha256", "0" * 64, "SUPPLEMENT_PARENT_SHA256"),
                       ("training_changed", True, "SUPPLEMENT_NO_TRAINING_CHANGE"),
                       ("evaluation_specs", plan["evaluation_specs"][:-1], "SUPPLEMENT_EXACT_16_PROBES"),
                       ("sources", {}, "SUPPLEMENT_SOURCE_SET"))
            for key, value, code in changes:
                mutated = deepcopy(plan)
                mutated[key] = value
                report._atomic_json(path, mutated)
                with self.subTest(key=key), self.assertRaisesRegex(AuditError, code):
                    report._supplement_protocol(path, parent, file_sha256(parent), _Checks(), {})

    def test_all_supplement_files_are_required_before_any_expensive_checkpoint_audit(self):
        from experiments.lifelong import supplement
        with tempfile.TemporaryDirectory() as temporary:
            folder, raw = Path(temporary) / "evidence", Path(temporary) / "raw"
            with redirect_stdout(io.StringIO()):
                prepare_original_protocol(folder, raw)
                supplement.prepare(folder / "study-protocol-v1.json", folder, Path(temporary) / "coverage")
            parent = folder / "study-protocol-v1.json"
            plan = report._read_json(parent)
            for branch in plan["branches"]:
                report._atomic_json(raw / "evaluation" / f"{branch['id']}.json", {})
            with self.assertRaisesRegex(AuditError, "SUPPLEMENT_EVALUATION_FILE_SET"):
                report.build_report(parent, raw, supplement=folder / "coverage-probe-supplement-v1.json", workers=1)

    def test_supplement_wrapper_binds_exact_parent_and_complete_native_checkpoint(self):
        checkpoint = deepcopy(self.cp)
        checkpoint.update(checkpoint="fixture.pt", checkpoint_sha256="a" * 64)
        item = dict(id="fixture", output="unused", **{key: self.state["config"][key] for key in ("seed", *report.BRANCH_KEYS)})
        wrapper = report.normalize(dict(format="lifelong-coverage-evaluation-v1", branch=item,
            protocol_sha256="b" * 64, parent_protocol_sha256="c" * 64, source_unchanged=True,
            states={str(self.spec["tick"]): dict(checkpoint=checkpoint["checkpoint"],
                checkpoint_sha256=checkpoint["checkpoint_sha256"], state_hash=checkpoint["cpu_semantic_hash"],
                training_device_normalized_state_hash=checkpoint["semantic_hash"], family_recency=checkpoint["family_recency"])},
            records=[self.record]))
        kwargs = dict(plan=self.plan, protocol_hash="b" * 64, parent_protocol_hash="c" * 64,
                      origin="supplement", where="fixture")
        rows, idle, content, external = report._evaluation_rows(wrapper, item, {24: checkpoint}, [self.spec], audit=_Checks(), **kwargs)
        self.assertEqual(rows[0]["probe_origin"], "supplement")
        self.assertEqual(external[0][0][-1], self.spec["steps"])
        changed = deepcopy(wrapper)
        changed["parent_protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(AuditError, "SUPPLEMENT_EVALUATION_PARENT"):
            report._evaluation_rows(changed, item, {24: checkpoint}, [self.spec], audit=_Checks(), **kwargs)

    def test_matched_probe_selector_keeps_horizons_conditions_and_seed_units_separate(self):
        from experiments.lifelong import study, supplement
        rows = []
        for seed in range(3):
            for origin, specs in (("primary", study.evaluation_specs()), ("supplement", supplement.specs())):
                for spec in specs:
                    rows.append(dict(seed=seed, architecture="fixed_shrink", schedule="mixed", learning="history_replay",
                        coverage="narrow", **spec, probe_origin=origin,
                        within_training_support=report.within_training_support("narrow", spec["condition"]),
                        metrics=dict(strict_completion=.2 + seed * .01 + spec["tick"] / 1e6),
                        primary=dict(matured=10 + seed), primary_censored=1,
                        actual_family_exposures={key: [0] * 4 for key in ("new", "replace", "commands")},
                        source_clock=dict(world_tick=spec["tick"] - 1)))
        matched = report.matched_memory_rows(rows)
        self.assertEqual(len(matched), 84)
        self.assertEqual(sum(row["family"] == 0 for row in matched), 30)
        self.assertTrue(all(row["steps"] == 2048 for row in matched))
        changes = report.matched_temporal_contrasts(matched)
        self.assertEqual(len(changes), 60)
        summary = report._matched_contrast_summaries(changes)
        self.assertTrue(all(row["metrics"]["strict_completion"]["n"] == 3 for row in summary))
        self.assertTrue(all(row["within_training_support"] is False for row in changes if row["condition"] == "long_both"))
        with self.assertRaisesRegex(ValueError, "duplicate semantic"):
            report.matched_memory_rows(rows + [deepcopy(rows[0])])
        unaligned = deepcopy(matched)
        unaligned[0]["steps"] = 1024
        with self.assertRaisesRegex(ValueError, "2048tick"):
            report.matched_temporal_contrasts(unaligned)

    def test_family_clock_ages_use_the_correct_world_or_optimizer_boundary_basis(self):
        checkpoint = dict(world_tick=9, tick=10, world_time=2.25, family_recency=dict(
            last_command_tick=[4, -1, -1, -1], last_command_time=[1., -1., -1., -1.],
            last_live_family_tick=[8, -1, -1, -1], last_aux_family_tick=[8, -1, -1, -1],
            live_supervised_family_positions=[7, 0, 0, 0], aux_supervised_family_positions=[6, 0, 0, 0]))
        clocks = report.family_clock_metrics(checkpoint, 0)
        self.assertEqual(clocks["real_command_lag_world_ticks"], 5)
        self.assertEqual(clocks["live_supervision_lag_world_ticks"], 1)
        self.assertEqual(clocks["auxiliary_supervision_lag_boundary_ticks"], 2)
        self.assertEqual(clocks["real_command_lag_physical_time"], 1.25)
        self.assertIsNone(report.family_clock_metrics(checkpoint, 1)["real_command_lag_world_ticks"])

    def test_structure_distinguishes_new_coordinates_initial_returns_and_repeat_events(self):
        initial = torch.tensor([[True, False], [False, True]])
        support = torch.tensor([[True, True], [False, True]])
        ever = torch.tensor([[True, True], [False, False]])
        state = dict(model=dict(recurrent=support.float()), birth=dict(initial_mask_hash=state_digest(initial)),
            operator=dict(inherited_operator=dict(mask=initial),
            operator=dict(mask=support, initial_mask=initial, cumulative_births=5), factor_ever_born=ever,
            factor_survival_mask=torch.tensor([[False, False], [False, True]])))
        result = report.structural_history(state)
        self.assertEqual(result["unique_edges_ever_born_outside_initial"], 1)
        self.assertEqual(result["initial_edges_ever_reactivated"], 1)
        self.assertEqual(result["reactivation_birth_events"], 4)
        self.assertEqual(result["birth_events_minus_unique_coordinates"], 3)
        self.assertEqual(result["active_edges_outside_initial"], 1)
        self.assertEqual(result["active_initial_coordinates_reactivated"], 1)

    def test_temporal_changes_keep_horizons_and_family0_before_new_feedback(self):
        rows = []
        for seed in range(3):
            for tick, value, steps in ((131072, .5, 1024), (163840, .3, 2048),
                                       (164864, .4, 1024), (167936, .6, 1024), (172032, .7, 2048)):
                rows.append(dict(seed=seed, architecture="fixed_shrink", schedule="mixed", learning="history_replay",
                    coverage="wide", tick=tick, purpose="fixture", family=0, condition="long_both", steps=steps,
                    source_family_recency=dict(last_real_command_tick=[tick - 100] * 4),
                    metrics=dict(strict_completion=value)))
        differences = report.temporal_contrasts(rows)
        self.assertEqual(len(differences), 12)
        withdrawal = next(row for row in differences if row["contrast"] == "withdrawal_change")
        self.assertAlmostEqual(withdrawal["metrics"]["strict_completion"], -.2)
        self.assertEqual((withdrawal["before_steps"], withdrawal["after_steps"]), (1024, 2048))
        final = next(row for row in differences if row["contrast"] == "return_after_8192")
        self.assertAlmostEqual(final["metrics"]["strict_completion"], .4)
        self.assertEqual(final["live_ticks_between"], 8192)


if __name__ == "__main__":
    unittest.main()
