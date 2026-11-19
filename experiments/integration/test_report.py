"""Synthetic audit rejection tests; no learner or evaluation is advanced."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from experiments.integration.report import (
    ARMS, CONDITIONS, aggregate, json_digest, load_evaluations, load_reports, main, summarize, write_artifacts,
)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _cohort(n, completed, clean, fingerprint, batch=64):
    return dict(matured=n, completed=completed, clean_completed=clean,
                completion_rate=completed/n if n else None, clean_completion_rate=clean/n if n else None,
                request_fingerprint=fingerprint,
                per_actor={"matured": [n//batch] * batch,
                           "completed": [1] * completed + [0] * (batch-completed),
                           "clean_completed": [1] * clean + [0] * (batch-clean)})


def _details(n, completed, clean):
    result = dict(requests=n, completed=completed, clean_completed=clean,
                with_recorded_or_inherited_error=8 if n else 0,
                first_error_patterns={"early+wrong_payload": 5, "wrong_payload": 3} if n else {},
                first_error_causes={"early": 5, "wrong_payload": 3} if n else {},
                observed_first_error_age_sum=4.0 if n else 0., observed_first_error_age_count=8 if n else 0,
                began=n-2 if n else 0, with_observed_legal_opportunity=32 if n else 0,
                first_observed_legal_count=32 if n else 0, first_observed_legal_content_correct=26 if n else 0,
                active_finish_attempts=100 if n else 0, legally_timed_finish_attempts=80 if n else 0,
                finish_attempts_with_wrong_content=12 if n else 0,
                uncompleted_no_begin=2 if n else 0, uncompleted_began_but_no_legal_opportunity=1 if n else 0,
                uncompleted_despite_legal_opportunity=n-completed-3 if n else 0)
    counts = dict(cue_observed_count=n, cue_content_correct_count=52 if n else 0,
                  cue_legal_pair_count=32 if n else 0, cue_legal_both_correct=24 if n else 0,
                  cue_legal_lost=4 if n else 0, cue_legal_gained=2 if n else 0, cue_legal_both_wrong=2 if n else 0,
                  cue_legal_cue_correct=28 if n else 0, cue_legal_legal_correct=26 if n else 0,
                  cue_without_legal_count=32 if n else 0, legal_without_cue_count=0, neither_cue_nor_legal_count=0)
    counts.update(cue_content_accuracy=52/64 if n else None, cue_legal_cue_accuracy=28/32 if n else None,
                  cue_legal_legal_accuracy=26/32 if n else None, cue_legal_accuracy_given_cue_correct=24/28 if n else None)
    return result | counts


def _record(report, condition, intervention="none"):
    seed, mode = report["config"]["seed"], report["integration_config"]["mode"]
    group = mode if mode.startswith("coverage_") else "main"
    clean = 40 + ARMS.index(mode) + seed
    adjustment = {"none": 0, "finish_content_oracle": 1, "timing_oracle": 2,
                  "completion_receipt_ablation": -3, "cancel_command_ablation": -2}[intervention]
    clean += adjustment
    completed, n = clean+5, 64
    total, active, inactive_actions = 2048 * 64 * 4, 4000, 100 + ARMS.index(mode)
    idle = total-active
    action_hash = digest((seed, mode, condition, intervention, "actions"))
    observation_hash = digest((seed, mode, condition, intervention, "observation"))
    regimes = {"iid": ([1., 4.], [.5, 2.]), "long_wait": ([4., 8.], [.5, 2.]),
               "longer": ([8., 16.], [.5, 2.]), "sparse": ([1., 4.], [4., 12.]),
               "long_sparse": ([4., 8.], [4., 12.])}
    content = []
    for phase, opportunities, correct, first, first_correct in (("processing", 3000, 2200, 0, 0),
                                                               ("legal_finish", 1000, 800, 32, 26)):
        content.append(dict(cohort="primary", phase=phase, age_bin="[4,8)", age_lower=4., age_upper=8.,
                            opportunities=opportunities, finish_choice_correct=correct,
                            version_choice_correct=correct, payload_choice_correct=correct,
                            model_finish_attempts=100, executed_finish_attempts=100,
                            conditional_finish_accuracy=correct/opportunities,
                            first_legal_opportunities=first, first_legal_choice_correct=first_correct,
                            first_legal_conditional_accuracy=first_correct/first if first else None,
                            mean_correct_logit_margin=.4))
    result = dict(format="integration-own-trace-v2", condition=condition, intervention=intervention,
        action_policy="model" if intervention == "none" else "diagnostic_counterfactual",
        world_resets=0, hidden_resets=0, parameter_updates=0, steps=2048, dt=.25, batch_size=64,
        n_jobs=4, active_jobs=4, actor_ids=list(range(64)), case_ids=[f"{seed}/{i}" for i in range(64)],
        fork_tick=report["ticks"]-1, fork_time=(report["ticks"]-1)*.25,
        end_tick=report["ticks"]-1+2048, end_time=(report["ticks"]-1+2048)*.25,
        evaluation_regime_id=3, regime=dict(duration_range=regimes[condition][0], gap_range=regimes[condition][1], window_range=[1.5, 2.5], cancel=.2, replace=.3),
        age_edges=[0., 1., 2., 4., 8., 16., 32.], primary_eligible_issued=68,
        carryover_contract_eligible_issued=0, initial_eligible_pending=0,
        censored={"primary": dict(count=4, completed=1, clean_completed=1),
                  "inherited": dict(count=0, completed=0, clean_completed=0),
                  "carryover_contract": dict(count=0, completed=0, clean_completed=0)},
        request_diagnosis={"primary": _details(n, completed, clean), "inherited": _details(0, 0, 0),
                           "carryover_contract": _details(0, 0, 0)},
        content_by_cohort_phase_age=content,
        opportunities=dict(opportunities=total, active_opportunities=active, inactive_opportunities=idle,
            inactive_actions=inactive_actions, inactive_begin_actions=30, inactive_finish_actions=inactive_actions-30,
            inactive_finish_matches_retired_content=50, disabled_slot_opportunities=0, disabled_slot_actions=0,
            model_action_changes=0 if intervention == "none" else 20, observation_ablation_opportunities=0,
            active_finish_attempts=100, legally_timed_finish_attempts=80,
            legally_timed_wrong_content_attempts=10, illegally_timed_finish_attempts=20,
            content_correct_but_illegally_timed_finish_attempts=10),
        inactive_actions_per_idle_opportunity=inactive_actions/idle,
        inactive_actions_per_1000_idle_opportunities=1000*inactive_actions/idle,
        inactive_actions_per_1000_total_opportunities=1000*inactive_actions/total,
        inactive_history=[dict(cause="completion", time_since_terminal_bin="[4,8)", opportunities=idle,
            actions=inactive_actions, begin_actions=30, finish_actions=inactive_actions-30, finish_matches_retired_content=50,
            known_retired_content_opportunities=idle, conditional_finish_matches_retired_content=300000,
            model_wait_preferences=idle-inactive_actions, model_finish_preferences=inactive_actions-30,
            finish_preferred_to_wait=inactive_actions, retired_exact_finish_preferred_to_wait=50,
            retired_content_readout_with_wait=100000, finish_minus_wait_logit_sum=.2*idle)],
        post_fork_counters=dict(false_action=inactive_actions, completed_all=completed, finish_attempted=100, eligible=68),
        post_fork_finish_precision=completed/100,
        start_world_state_hash=report["world_hash"], end_world_state_hash=digest((seed, mode, condition, intervention, "endworld")),
        start_hidden_hash=digest((seed, mode, "hidden")), end_hidden_hash=digest((seed, mode, condition, intervention, "endh")),
        start_scheduler_hash=digest((seed, group, "scheduler")), end_scheduler_hash=digest((seed, group, condition, "scheduler")),
        start_event_hash=report["event_hash"], end_event_hash=digest((seed, group, condition, "events")),
        parameter_hash=report["final_parameter_hash"], source_continuation_hash=digest((seed, mode, "source")),
        action_hash=action_hash, model_action_hash=action_hash if intervention == "none" else digest((action_hash, "model")),
        observation_hash=observation_hash, raw_observation_hash=observation_hash if intervention == "none" else digest((observation_hash, "raw")))
    for cohort, count in (("primary", n), ("inherited", 0), ("carryover_contract", 0)):
        result[cohort] = _cohort(count, completed if count else 0, clean if count else 0,
                                 digest((seed, group, condition, cohort)))
    if intervention != "none":
        result["paired_baseline_action_hash"] = digest((seed, mode, condition, "none", "actions"))
    return result


def study_fixture(*, seeds=(0, 1, 2), modes=ARMS):
    sources = {"continuity": {name: digest(name) for name in ("world.py", "model.py", "run.py")},
               "integration": {name: digest("integration/"+name) for name in ("run.py", "local.py")},
               "diagnose": {"diagnose.py": digest("diagnose.py")}}
    diagnostic_modes = [mode for mode in ("frozen", "bp64", "replay64", "coact_mix") if mode in modes]
    manifest = dict(seeds=list(seeds), arms=list(modes), conditions=list(CONDITIONS), stage_ticks=128,
                    batch_size=64, n_jobs=4, width=128, total_parameters=34348, dt=.25, eval_steps=2048,
                    local_lr=.001, local_tau=16., origin_ticks=131072, source_hashes=sources,
                    diagnostic_arms=diagnostic_modes, diagnostic_interventions={
                        "longer": ["none", "finish_content_oracle", "timing_oracle"],
                        "long_sparse": ["none", "finish_content_oracle", "timing_oracle"],
                        "sparse": ["none", "completion_receipt_ablation", "cancel_command_ablation"]})
    reports, evaluations = [], []
    for seed in seeds:
        origin = {name: digest((seed, name)) for name in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash")}
        origin.update(tick=131072, update=2048, inherited_source_hashes=sources["continuity"],
                      checkpoint=f"F:/synthetic/A/seed_{seed:03d}/checkpoint.pt", checkpoint_sha256=digest((seed, "Afile")))
        for mode in modes:
            live = 128 if mode in {"accum128", "bp128"} else 64
            updates, bp, old, local = 128//live, mode not in {"frozen", "coact_only_mix"}, mode == "replay64" or mode.startswith("coact_"), mode.startswith("coact_")
            extra = "current" if mode == "repeat64" else "replay" if mode == "replay64" else None
            passes = int(bp)+int(extra is not None)
            flags = dict(current_gradient=bp, extra_gradient=extra,
                         local_source="live" if mode == "coact_live" else "mix" if local else None,
                         shuffled=mode == "coact_mix_shuffled")
            config = dict(mode=mode, live_ticks=live, gradient_horizon=128 if mode == "bp128" else 64,
                          duration_range=[1., 4.] if mode == "coverage_short_sparse" else [4., 8.],
                          gap_range=[.5, 2.] if mode == "coverage_long_dense" else [4., 12.], local_lr=.001, local_tau=16., regime_name="integration")
            trace = []
            for index in range(updates):
                point = dict(stage="B", stage_ticks=(index+1)*live, gradient_horizon_this_chunk=config["gradient_horizon"] if bp else 0,
                             gradient_updates_this_chunk=passes, local_updates_this_chunk=int(local),
                             bp_recurrent_delta_frob=.02 if bp else None, bp_all_delta_frob=.03 if bp else None,
                             extra_bp_recurrent_delta_frob=.025 if extra else None, extra_bp_all_delta_frob=.035 if extra else None,
                             local=dict(source_weights=[1.] if mode == "coact_live" else [.5, .5], delta_frob=.001, lr=.001, direction_frob=1.) if local else None,
                             old_block_index=index if old else None, replay_start_tick=index*128 if old else None)
                trace.append(point)
            norms = dict(bp_recurrent_frob_sum=updates*((.02 if bp else 0)+(.025 if extra else 0)),
                         bp_recurrent_frob_squared_sum=updates*((.02**2 if bp else 0)+(.025**2 if extra else 0)),
                         bp_all_frob_sum=updates*((.03 if bp else 0)+(.035 if extra else 0)),
                         bp_all_frob_squared_sum=updates*((.03**2 if bp else 0)+(.035**2 if extra else 0)),
                         local_recurrent_frob_sum=updates*.001 if local else 0.,
                         local_recurrent_frob_squared_sum=updates*.001**2 if local else 0.)
            supervised = (128 if bp else 0)+(128 if extra == "current" else updates*64 if extra == "replay" else 0)
            group = mode if mode.startswith("coverage_") else "main"
            report = dict(format="integration-trainer-v1", config=dict(seed=seed, batch_size=64, n_jobs=4, train_dts=[.25],
                replay_capacity=32, replay_block_ticks=128, replay_burnin=64), integration_config=config, arm=flags,
                variant=dict(name="online128", width=128, kind="learned", policy="online", reset_every=0), stage="B", adaptation_mode=mode,
                model=dict(kind="learned", state_scalars=128, input_size=96, n_jobs=4, n_actions=11, total_parameters=34348),
                stage_ticks=128, stage_updates=updates, stage_gradient_updates=updates*passes, local_updates=updates if local else 0,
                stage_batch_observations=128*64, stage_action_labels_observed=128*64*4,
                origin=deepcopy(origin), ticks=131200, updates=2048+updates, world_resets=0, hidden_resets=0,
                source_hashes=sources["continuity"], integration_source_hashes=sources["integration"],
                final_parameter_hash=origin["model_hash"] if mode == "frozen" else digest((seed, mode, "parameters")),
                world_hash=digest((seed, mode, "world")), event_hash=digest((seed, group, "training_events")), data_seed=seed,
                work=dict(live_forward_ticks=128, current_reprocess_forward_ticks=128 if extra == "current" else 0,
                    old_forward_ticks=updates*128 if old else 0, old_burnin_ticks=updates*64 if old else 0,
                    old_samples=updates if old else 0, supervised_ticks=supervised, backward_horizon_ticks=supervised,
                    local_contributing_ticks=(128 if local else 0)+(updates*64 if local and mode != "coact_live" else 0)),
                parameter_updates=dict(adam_parameters=34348 if bp else 0, local_parameters=16384 if local else 0),
                replay=dict(stored_blocks=32, capacity_blocks=32, block_ticks=128, stored_tick_ranges=[[i*128, (i+1)*128] for i in range(32)]),
                update_norms=norms, trace=trace, curve=[{"not_copied": "raw curve"}], local_live=None, local_replay=None)
            if local:
                for namespace in ("live", "replay"):
                    report["local_"+namespace] = dict(config=dict(lr=.001, tau=16., shuffled=flags["shuffled"], seed=seed),
                        observed_ticks=128 if namespace == "live" else updates*128,
                        plasticity_ticks=128 if namespace == "live" else updates*64,
                        local_updates=updates if namespace == "live" else 0, matrix_writes=updates if namespace == "live" else 0,
                        pending_ticks=0, pending_elapsed=0., history_restarts=0 if namespace == "live" else updates,
                        trace_bytes=32768, buffer_bytes=98816, private_rng_bytes=5056)
            report["_provenance"] = dict(path=f"F:/synthetic/main/seed_{seed:03d}/{mode}/report.json", sha256=digest((seed, mode, "reportfile")))
            reports.append(report)
            checkpoint = f"F:/synthetic/main/seed_{seed:03d}/{mode}/checkpoint.pt"
            baseline = dict(format="integration-own-trace-v2", checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint),
                checkpoint_state_hash=digest((seed, mode, "semantic")), training_config=report["config"], training_variant=report["variant"],
                training_stage="B", adaptation_mode=mode, training_ticks=report["ticks"], training_updates=report["updates"],
                model=report["model"], checkpoint_source_hashes=sources["integration"], source_hashes=sources["diagnose"],
                evaluation_args=dict(conditions=list(CONDITIONS), interventions=["none"], steps=2048, dt=.25, strict_sources=True,
                                     age_edges=[0., 1., 2., 4., 8., 16., 32.]),
                records=[_record(report, condition) for condition in CONDITIONS],
                _provenance=dict(path=f"F:/synthetic/evaluation/seed_{seed:03d}/{mode}.json", sha256=digest((seed, mode, "evalfile"))))
            evaluations.append(baseline)
            if mode in diagnostic_modes:
                for condition, interventions in manifest["diagnostic_interventions"].items():
                    diagnostic = deepcopy(baseline)
                    diagnostic["evaluation_args"]["conditions"] = [condition]
                    diagnostic["evaluation_args"]["interventions"] = interventions
                    diagnostic["records"] = [_record(report, condition, intervention) for intervention in interventions]
                    diagnostic["_provenance"] = dict(path=f"F:/synthetic/diagnosis/seed_{seed:03d}/{mode}_{condition}.json", sha256=digest((seed, mode, condition, "diagfile")))
                    evaluations.append(diagnostic)
    return reports, evaluations, manifest


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.evaluations, cls.manifest = study_fixture()

    def run_fixture(self, reports=None, evaluations=None, manifest=None, **kwargs):
        return aggregate(deepcopy(self.reports if reports is None else reports),
                         deepcopy(self.evaluations if evaluations is None else evaluations),
                         manifest=deepcopy(self.manifest if manifest is None else manifest), **kwargs)

    def test_complete_protocol_passes_and_duplicate_baselines_are_not_replicates(self):
        result = self.run_fixture()
        self.assertEqual(result["audit"]["error_count"], 0, result["audit"]["issues"][:3])
        self.assertEqual(result["audit"]["status"], "passed")
        self.assertEqual(len(result["training"]), 36)
        self.assertEqual(len(result["per_seed"]), 180)
        self.assertEqual(len(result["diagnostic_per_seed"]), 72)
        row = next(row for row in result["summary"] if row["arm"] == "bp64" and row["condition"] == "iid")
        stat = row["metrics"]["strict_completion"]
        self.assertEqual(stat["n"], 3)
        self.assertAlmostEqual(stat["mean"], 42/64)
        self.assertAlmostEqual(stat["sample_sd"], 1/64)
        for row in result["comparisons"]:
            self.assertEqual(row["event_cohort_paired"], not row["arm"].startswith("coverage_"))

    def test_summary_uses_seed_values_and_handles_undefined_denominators(self):
        result = summarize([dict(seed=0, value=.5, requests=10), dict(seed=1, value=.75, requests=1000), dict(seed=2, value=None)])
        self.assertEqual(result["mean"], .625)
        self.assertEqual(result["sample_sd"], statistics.stdev([.5, .75]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_total"], 3)
        self.assertEqual(result["n_undefined"], 1)
        self.assertIsNone(summarize([dict(seed=0, value=None)])["sample_sd"])

    def test_oracle_scores_enter_only_separate_diagnostic_deltas(self):
        result = self.run_fixture()
        row = next(row for row in result["diagnostic_deltas"] if row["arm"] == "bp64" and row["condition"] == "long_sparse" and row["intervention"] == "timing_oracle")
        self.assertEqual(row["classification"], "diagnostic_counterfactual_only")
        self.assertAlmostEqual(row["metric_deltas"]["strict_completion"]["mean"], 2/64)
        model = [row for row in result["per_seed"] if row["arm"] == "bp64" and row["condition"] == "long_sparse"]
        self.assertEqual(len(model), 3)
        self.assertTrue(all("intervention" not in row for row in model))

    def test_integrity_failures_are_not_relaxed_by_incomplete_mode(self):
        cases = (
            ("TRAINING_BUDGET", lambda r,e,m: r[0].__setitem__("stage_ticks", 64)),
            ("INTEGRATION_SOURCES", lambda r,e,m: r[0]["integration_source_hashes"].__setitem__("run.py", "0"*64)),
            ("COMPLETE_ORIGIN_PAIRING", lambda r,e,m: r[1]["origin"].__setitem__("hidden_hash", "0"*64)),
            ("TRAINING_EVENT_PAIRING", lambda r,e,m: r[1].__setitem__("event_hash", "0"*64)),
            ("PARAMETER_COUNT", lambda r,e,m: r[0]["model"].__setitem__("total_parameters", 34349)),
            ("RATE_DENOMINATOR", lambda r,e,m: e[0]["records"][0]["primary"].__setitem__("completion_rate", .1)),
            ("INACTIVE_DENOMINATOR", lambda r,e,m: e[0]["records"][0].__setitem__("inactive_actions_per_1000_idle_opportunities", .1)),
            ("FAILURE_PARTITION", lambda r,e,m: e[0]["records"][0]["request_diagnosis"]["primary"].__setitem__("uncompleted_no_begin", 999)),
            ("EVALUATION_EVENT_COHORT_PAIRING", lambda r,e,m: e[4]["records"][0].__setitem__("end_scheduler_hash", "0"*64)),
            ("ACTION_POLICY", lambda r,e,m: e[0]["records"][0].__setitem__("action_policy", "teacher")),
            ("CENSOR_ACCOUNTING", lambda r,e,m: e[0]["records"][0]["censored"]["primary"].__setitem__("count", 90)),
            ("TRACE_GRADIENT", lambda r,e,m: r[1]["trace"][0].__setitem__("gradient_horizon_this_chunk", 128)),
            ("NORM_TRACE_TOTAL", lambda r,e,m: r[1]["update_norms"].__setitem__("bp_recurrent_frob_sum", 99.)),
            ("CUE_PAIR_PARTITION", lambda r,e,m: e[0]["records"][0]["request_diagnosis"]["primary"].__setitem__("cue_legal_lost", 30)),
            ("CUE_RATE_DENOMINATOR", lambda r,e,m: e[0]["records"][0]["request_diagnosis"]["primary"].__setitem__("cue_legal_accuracy_given_cue_correct", .1)),
            ("CONDITION_START_PAIRING", lambda r,e,m: e[0]["records"][0].__setitem__("start_hidden_hash", "0"*64)),
        )
        for code, mutate in cases:
            with self.subTest(code=code):
                reports, evaluations, manifest = deepcopy(self.reports), deepcopy(self.evaluations), deepcopy(self.manifest)
                mutate(reports, evaluations, manifest)
                result = aggregate(reports, evaluations, manifest=manifest, allow_incomplete=True)
                self.assertEqual(result["audit"]["status"], "failed")
                self.assertIn(code, {row["code"] for row in result["audit"]["issues"]})

    def test_duplicate_fork_semantic_mismatch_is_rejected(self):
        evaluations = deepcopy(self.evaluations)
        diagnostic = next(item for item in evaluations if item["evaluation_args"]["conditions"] == ["longer"])
        diagnostic["records"][0]["end_hidden_hash"] = "0"*64
        result = self.run_fixture(evaluations=evaluations)
        self.assertIn("DUPLICATE_FORK_CONTENT", {row["code"] for row in result["audit"]["issues"]})

    def test_duplicate_wall_time_is_ignored_but_matched_loss_uses_cue_correct_pairs(self):
        evaluations = deepcopy(self.evaluations)
        diagnostic = next(item for item in evaluations if item["evaluation_args"]["conditions"] == ["longer"])
        diagnostic["records"][0]["wall_seconds"] = 123.4
        result = self.run_fixture(evaluations=evaluations)
        self.assertEqual(result["audit"]["error_count"], 0, result["audit"]["issues"][:3])
        row = next(row for row in result["summary"] if row["arm"] == "bp64" and row["condition"] == "long_sparse")
        self.assertAlmostEqual(row["metrics"]["matched_loss_given_cue_correct"]["mean"], 4/28)

    def test_missing_prespecified_diagnostic_is_explicitly_incomplete_only_when_allowed(self):
        evaluations = [item for item in self.evaluations if not (item["adaptation_mode"] == "bp64" and
                       item["evaluation_args"]["conditions"] == ["longer"])]
        strict = self.run_fixture(evaluations=evaluations)
        progress = self.run_fixture(evaluations=evaluations, allow_incomplete=True)
        self.assertEqual(strict["audit"]["status"], "failed")
        self.assertEqual(progress["audit"]["status"], "incomplete")
        self.assertEqual(progress["audit"]["error_count"], 0)
        self.assertIn("MISSING_DIAGNOSTIC_FORK", {row["code"] for row in progress["audit"]["issues"]})

    def test_shadow_coactivity_and_pure_coactivity_budgets_stay_distinct(self):
        reports = deepcopy(self.reports)
        pure = next(row for row in reports if row["integration_config"]["mode"] == "coact_only_mix")
        pure["parameter_updates"]["adam_parameters"] = 34348
        result = self.run_fixture(reports=reports)
        self.assertIn("PARAMETER_UPDATES", {row["code"] for row in result["audit"]["issues"]})

    def test_loaders_export_compact_audit_and_all_three_figures(self):
        reports, evaluations, manifest = study_fixture(seeds=(0, 1), modes=("frozen", "bp64", "replay64", "coact_mix"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, report in enumerate(reports):
                folder = root / "training" / str(index)
                folder.mkdir(parents=True)
                (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "evaluation.json").write_text(json.dumps(dict(format="integration-own-trace-v2", evaluations=evaluations)), encoding="utf-8")
            loaded_reports = load_reports([root / "training"])
            loaded_evaluations = load_evaluations([root / "evaluation.json"])
            result = aggregate(loaded_reports, loaded_evaluations, manifest=manifest)
            self.assertEqual(result["audit"]["error_count"], 0, result["audit"]["issues"][:3])
            output = root / "aggregate"
            paths = write_artifacts(result, output, synthetic=True)
            saved = json.loads(Path(paths["summary.json"]).read_text(encoding="utf-8"))
            self.assertTrue(saved["synthetic_layout_check"])
            self.assertNotIn("content_by_phase_age", saved)
            self.assertNotIn("trace", json.dumps(saved["training"]))
            self.assertEqual(len(saved["figures"]), 3)
            self.assertEqual(len(saved["table_files"]), 11)
            self.assertLess((output / "audit.json").stat().st_size, 100000)
            for figure in saved["figures"]:
                self.assertGreater((output / figure["path"]).stat().st_size, 10000)

    def test_cli_verifies_origin_sources_and_checkpoint_bytes(self):
        reports, evaluations, manifest = study_fixture(seeds=(0,), modes=("frozen",))
        report = reports[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "main" / "seed_000" / "frozen"
            run.mkdir(parents=True)
            for prefix in ("continuity", "integration"):
                actual = {}
                for name in manifest["source_hashes"][prefix]:
                    source = run / "sources" / prefix / name
                    source.parent.mkdir(parents=True, exist_ok=True)
                    content = f"# synthetic {prefix}/{name}\n# CRLF disk snapshot\n"
                    source.write_bytes(content.replace("\n", "\r\n").encode())
                    actual[name] = hashlib.sha256(content.encode()).hexdigest()
                manifest["source_hashes"][prefix] = actual
                report["source_hashes" if prefix == "continuity" else "integration_source_hashes"] = actual
            origin = root / "A" / "checkpoint.pt"
            origin.parent.mkdir()
            origin.write_bytes(b"synthetic source checkpoint; never loaded")
            report["origin"].update(checkpoint=str(origin.resolve()), checkpoint_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
                                     inherited_source_hashes=report["source_hashes"])
            original_report = dict(config={"seed": 0}, variant={"name": "online128", "policy": "online"}, stage="A",
                ticks=131072, updates=2048, final_parameter_hash=report["origin"]["model_hash"], world_hash=report["origin"]["world_hash"])
            source_report = origin.parent / "report.json"
            source_report.write_text(json.dumps(original_report), encoding="utf-8")
            manifest["source_checkpoints"] = {"0": dict(path=str(origin.resolve()), sha256=report["origin"]["checkpoint_sha256"],
                report_sha256=hashlib.sha256(source_report.read_bytes()).hexdigest())}
            final = run / "checkpoint.pt"
            final.write_bytes(b"synthetic final checkpoint; never loaded")
            for evaluation in evaluations:
                evaluation.update(checkpoint=str(final.resolve()), checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                                  checkpoint_source_hashes=report["integration_source_hashes"])
            (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps(dict(format="integration-own-trace-v2", evaluations=evaluations)), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            arguments = ["--root", str(run), "--evaluations", str(evaluation_path), "--manifest", str(manifest_path), "--no-figures"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments + ["--output", str(root / "aggregate")]), 0)
            final.write_bytes(b"changed checkpoint")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments + ["--output", str(root / "rejected")]), 1)
            rejected = json.loads((root / "rejected" / "audit.json").read_text(encoding="utf-8"))
            self.assertIn("CHECKPOINT_FILE", {row["code"] for row in rejected["issues"]})
            self.assertFalse((root / "rejected" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
