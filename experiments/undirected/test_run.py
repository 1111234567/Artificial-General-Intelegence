"""Runner-level tests: real worlds, parameter trajectories and full continuation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.continuity import run as continuity
from experiments.undirected import run


def toy(**changes):
    values = dict(width=40, n_jobs=1, batch_size=1, initial_degree=4,
        chunk_ticks=4, main_ticks=16, withdrawal_ticks=8, return_ticks=4,
        warmup_ticks=8, macrocycle_ticks=16, block_ticks=4, credit="tbptt128")
    values.update(changes)
    return run.NativeConfig(**values)


def rehash(state):
    state["state_hashes"] = {name: continuity.state_digest(state[name]) for name in run.HASH_FIELDS}
    state["state_hashes"]["semantic"] = continuity.state_digest({name: state[name] for name in run.SEMANTIC_FIELDS})
    return state


class NativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_defaults_and_limits_are_explicit(self):
        c = run.NativeConfig()
        self.assertEqual((c.width, c.batch_size, c.n_jobs, c.input_size, c.chunk_ticks, c.dtype), (80, 8, 2, 56, 128, "float64"))
        self.assertEqual(c.snapshot_ticks(), (0, 65536, 131072, 196608, 229376, 237568))
        self.assertEqual(c.total_ticks, 237568)
        self.assertEqual(c.birth_gain, .6)
        self.assertEqual(c.coverage_at(0), "wide")
        self.assertEqual(replace(c, course="warmup").coverage_at(65535), "narrow")
        self.assertEqual(replace(c, course="warmup").coverage_at(65536), "wide")
        for change in (dict(width=77), dict(dtype="float16"), dict(chunk_ticks=127), dict(initial_degree=80), dict(warmup_ticks=3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(c, **change)

    def test_four_model_births_match_function_world_and_global_rng(self):
        ambient = continuity.rng_state()
        trainers = [run.NativeTrainer(toy(interface=interface, mode=mode))
                    for interface in ("core", "ports") for mode in ("undirected", "untied")]
        self.assertEqual(continuity.state_digest(ambient), continuity.state_digest(continuity.rng_state()))
        reference = trainers[0]
        observation = torch.arange(28, dtype=torch.float64).reshape(1, 28) * .001
        with torch.no_grad():
            expected_logits, expected_h = reference.model(observation, reference.h, .25)
            for t in trainers:
                self.assertTrue(torch.equal(reference.model.recurrent_matrix(), t.model.recurrent_matrix()))
                self.assertEqual(continuity.state_digest(reference.world.state_dict()), continuity.state_digest(t.world.state_dict()))
                logits, hidden = t.model(observation, t.h, .25)
                self.assertTrue(torch.equal(logits, expected_logits))
                self.assertTrue(torch.equal(hidden, expected_h))
        self.assertNotEqual(sum(p.numel() for p in trainers[0].model.parameters()), sum(p.numel() for p in trainers[2].model.parameters()))

    def test_live_records_real_end_state_and_primal_reconstruction(self):
        t = run.NativeTrainer(toy())
        point = t.train_update()
        self.assertEqual(t.ticks, 4)
        self.assertEqual(t.world.tick, 3)
        self.assertFalse(t.h.requires_grad)
        self.assertEqual(t.last_chunk["observations"].shape, (4, 1, 28))
        self.assertEqual(t.last_chunk["observations"].dtype, torch.float64)
        self.assertTrue(torch.equal(t.last_chunk["end_h"], t.h.cpu()))
        self.assertTrue(torch.equal(t.credit.tape[-1]["end_h"], t.h.cpu()))
        self.assertEqual(point["credit"]["primal_max_abs_error"], 0.)
        self.assertEqual(point["credit"]["loss"], point["live_loss"])
        self.assertEqual(t.work["optimizer_steps"], 1)
        self.assertEqual(point["auxiliary"]["source"], "startup")

    def test_all_potential_edges_receive_adam_history_but_stay_zero(self):
        t = run.NativeTrainer(toy(mode="untied", interface="core"))
        t.run_updates(3)
        mask = t.operator.mask.to(t.device)
        for name in ("edge_shared", "edge_skew"):
            parameter = dict(t.model.named_parameters())[name]
            self.assertTrue(torch.equal(parameter[~mask], torch.zeros_like(parameter[~mask])))
            self.assertEqual(float(t.optimizer.state[parameter]["step"]), 3.)
            self.assertGreater(float(t.optimizer.state[parameter]["exp_avg"][~mask].norm()), 0.)
        self.assertEqual(t.operator.summary()["cumulative_births"], 0)
        self.assertEqual(t.operator.summary()["cumulative_deaths"], 0)

    def test_replay_is_causal_and_does_not_replace_live_state(self):
        t = run.NativeTrainer(toy())
        t.run_updates(3)
        info = t.last_auxiliary["info"]
        self.assertEqual(info["source"], "history_replay")
        self.assertLessEqual(info["end_tick"], t.last_chunk["start_tick"])
        self.assertEqual((info["warmup_ticks"], info["loss_ticks"]), (4, 4))
        self.assertEqual(t.replay.ticks_seen, 12)
        self.assertEqual(t.replay.samples_drawn, 1)
        hidden, world = continuity.state_digest(t.h), continuity.state_digest(t.world.state_dict())
        t._replay_loss(t.last_auxiliary["block"], 4)
        self.assertEqual(continuity.state_digest(t.h), hidden)
        self.assertEqual(continuity.state_digest(t.world.state_dict()), world)

    def test_all_credit_methods_resume_next_update_exactly_without_birth(self):
        for method in run.CREDIT_METHODS:
            with self.subTest(method=method):
                t = run.NativeTrainer(toy(credit=method))
                t.run_updates(2)
                state = t.state_dict()
                with patch.object(run.NativeTrainer, "__init__", side_effect=AssertionError("must not rebirth")), \
                     patch.object(run.PortGraphModel, "initialize_birth", side_effect=AssertionError("must not redraw graph")):
                    restored = run.NativeTrainer.from_state_dict(state)
                self.assertEqual(continuity.state_digest(t.semantic_state()), continuity.state_digest(restored.semantic_state()))
                actual, expected = t.train_update(), restored.train_update()
                self.assertEqual(continuity.state_digest(actual), continuity.state_digest(expected))
                self.assertEqual(continuity.state_digest(t.semantic_state()), continuity.state_digest(restored.semantic_state()))
                self.assertEqual(restored.credit.updates, 3)
                if method.startswith("rtrl"):
                    self.assertGreater(float(restored.credit.sensitivity.norm()), 0.)
                    self.assertGreater(restored.work["credit_jacobian_reverse_rows"], 0)

    def test_core_untied_float32_continuation_layout(self):
        t = run.NativeTrainer(toy(interface="core", mode="untied", dtype="float32"))
        t.run_updates(2)
        restored = run.NativeTrainer.from_state_dict(t.state_dict())
        self.assertEqual(continuity.state_digest(t.train_update()), continuity.state_digest(restored.train_update()))

    def test_restore_and_next_update_do_not_mutate_callers_saved_state(self):
        t = run.NativeTrainer(toy(interface="core", credit="rtrl", structure="both",
                                  structural_window_updates=1, structural_candidates=1))
        t.run_updates(3)
        saved = t.state_dict()
        original_hash = continuity.state_digest(saved)
        restored = run.NativeTrainer.from_state_dict(saved)
        self.assertEqual(continuity.state_digest(saved), original_hash)
        restored.train_update()
        self.assertEqual(continuity.state_digest(saved), original_hash)
        self.assertEqual(saved["ticks"], 12)
        self.assertEqual(restored.ticks, 16)

    def test_warmup_withdrawal_return_change_future_only(self):
        t = run.NativeTrainer(toy(course="warmup"))
        t.run_updates(2)
        self.assertEqual(t.world.regime.duration_range, (1., 4.))
        self.assertEqual(t.course_history, [{"tick": 0, "coverage": "narrow"}])
        state = t.state_dict()
        restored = run.NativeTrainer.from_state_dict(state)
        self.assertEqual(continuity.state_digest(t.train_update()), continuity.state_digest(restored.train_update()))
        self.assertEqual(t.course_history, [{"tick": 0, "coverage": "narrow"}, {"tick": 8, "coverage": "wide"}])
        t.run_updates(1)
        self.assertEqual(tuple(t.world.family_allowed), (0, 1, 2, 3))
        t.train_update()
        self.assertEqual(tuple(t.world.family_allowed), (1, 2, 3))
        t.run_updates(1)
        t.train_update()
        self.assertEqual(tuple(t.world.family_allowed), (0,))
        self.assertEqual(t.ticks, 28)
        self.assertEqual(t.hidden_resets, 0)
        self.assertEqual(t.world.tick, 27)
        with self.assertRaises(ValueError):
            t.train_update()

    def test_checkpoint_rejects_missing_hash_dtype_clock_and_layout(self):
        t = run.NativeTrainer(toy(credit="rtrl"))
        t.run_updates(2)
        original = t.state_dict()
        mutations = [lambda s: s.pop("credit"), lambda s: s["hidden"].add_(1),
            lambda s: s["model"].update(edge_shared=s["model"]["edge_shared"].float()),
            lambda s: s.update(ticks=s["ticks"] + 4), lambda s: s["parameter_layout"].append(("foreign", (1,), "torch.float64")),
            lambda s: s["work"].update(real_ticks_admitted=999),
            lambda s: s["last_chunk"].update(end_h=torch.ones_like(s["hidden"])),
            lambda s: s["operator"].update(updates=99)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                state = deepcopy(original)
                mutate(state)
                if index >= 2:
                    rehash(state)
                with self.assertRaises(ValueError):
                    run.NativeTrainer.from_state_dict(state)

    def test_rehash_does_not_allow_corrupt_adam_or_source(self):
        t = run.NativeTrainer(toy())
        t.train_update()
        original = t.state_dict()
        for kind in ("moment_dtype", "age", "source", "source_hash", "dependency"):
            with self.subTest(kind=kind):
                state = deepcopy(original)
                entry = next(iter(state["optimizer"]["state"].values()))
                if kind == "moment_dtype":
                    entry["exp_avg"] = entry["exp_avg"].float()
                    rehash(state)
                elif kind == "age":
                    entry["step"] += 1
                    rehash(state)
                elif kind in ("source", "source_hash"):
                    state["sources"]["model.py"] += "\n# foreign implementation\n"
                    if kind == "source":
                        state["source_hashes"] = run.source_hashes(state["sources"])
                else:
                    state["dependencies"]["lifelong/world.py"] += "\n# foreign world\n"
                    state["dependency_hashes"] = run.source_hashes(state["dependencies"])
                with self.assertRaises(ValueError):
                    run.NativeTrainer.from_state_dict(state)

    def test_rehash_does_not_allow_conflicting_copies_of_real_history(self):
        t = run.NativeTrainer(toy(credit="rtrl"))
        t.run_updates(3)
        original = t.state_dict()
        for target in ("last_chunk", "last_auxiliary", "replay", "credit"):
            with self.subTest(target=target):
                state = deepcopy(original)
                if target == "last_chunk":
                    state["last_chunk"]["observations"][0, 0, 0] += 1
                elif target == "last_auxiliary":
                    state["last_auxiliary"]["block"]["observations"][0, 0, 0] += 1
                    state["last_auxiliary"]["info"]["source_hash"] = run.block_hash(state["last_auxiliary"]["block"])
                elif target == "replay":
                    state["replay"]["blocks"][0]["observations"][0, 0, 0] += 1
                else:
                    state["credit"]["tape"][-1]["observations"][0, 0, 0] += 1
                    state["credit"]["state_hash"] = run.credit_state_digest({key: value for key, value in state["credit"].items() if key != "state_hash"})
                rehash(state)
                with self.assertRaises(ValueError):
                    run.NativeTrainer.from_state_dict(state)

    def test_four_structure_modes_probe_equally_and_restore_pending(self):
        births = []
        for structure in ("fixed_probe", "grow", "retire", "both"):
            with self.subTest(structure=structure):
                t = run.NativeTrainer(toy(structure=structure, structural_window_updates=1,
                                          structural_candidates=1, credit="rtrl"))
                births.append(t.birth["initial_recurrent_matrix_hash"])
                t.train_update()
                restored = run.NativeTrainer.from_state_dict(t.state_dict())
                for _ in range(2):
                    self.assertEqual(continuity.state_digest(t.train_update()), continuity.state_digest(restored.train_update()))
                self.assertEqual(t.work["structural_probe_calls"], 8)
                self.assertEqual(t.world.tick, 11)
                self.assertEqual(t.hidden_resets, 0)
                self.assertEqual(continuity.state_digest(t.semantic_state()), continuity.state_digest(restored.semantic_state()))
        self.assertEqual(len(set(births)), 1)

    def test_numerical_failure_saves_last_complete_state_without_retry(self):
        original_update = run.NativeTrainer.train_update

        def fail_after_three(trainer):
            if trainer.updates == 3:
                raise FloatingPointError("nonfinite sensitivity: injected numerical-failure fixture")
            return original_update(trainer)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run.NativeTrainer, "train_update", fail_after_three), self.assertRaises(FloatingPointError):
                run.train(toy(credit="rtrl"), directory, stop_at=16)
            failure = json.loads((Path(directory) / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "algorithm_numerical_failure")
            self.assertEqual(failure["kind"], "numerical_failure")
            self.assertFalse(failure["completed"])
            self.assertEqual(failure["last_complete_tick"], 12)
            saved = run.NativeTrainer.from_checkpoint(Path(directory) / "checkpoint_last_complete.pt")
            self.assertEqual(saved.ticks, 12)
            self.assertEqual(saved.credit.updates, 3)
            self.assertGreater(float(saved.credit.sensitivity.norm()), 0.)
            self.assertEqual(failure["last_complete_checkpoint_sha256"], run.file_sha256(Path(directory) / "checkpoint_last_complete.pt"))

    def test_newborn_restore_binds_actual_model_optimizer_hidden_world(self):
        original = run.NativeTrainer(toy()).state_dict()
        for target in ("model", "optimizer", "hidden", "world"):
            with self.subTest(target=target):
                state = deepcopy(original)
                if target == "model":
                    state["model"]["edge_shared"][state["operator"]["initial_mask"].nonzero()[0]] += .01
                elif target == "optimizer":
                    state["optimizer"]["param_groups"][0]["lr"] *= 2
                    state["config"]["lr"] *= 2
                elif target == "hidden":
                    state["hidden"] += 1
                else:
                    state["world"]["context"]["tensors"]["raw_payload"].fill_(1)
                rehash(state)
                with self.assertRaisesRegex(ValueError, "actual birth"):
                    run.NativeTrainer.from_state_dict(state)

    def test_actual_nonfinite_hidden_logits_sensitivity_and_gradient_are_typed(self):
        for target in ("hidden", "logits", "sensitivity", "gradient", "norm"):
            with self.subTest(target=target):
                t = run.NativeTrainer(toy(credit="rtrl"))
                if target in ("hidden", "logits"):
                    original_step = t.model.step

                    def broken_step(observation, hidden, dt, *, selected=target):
                        logits, state = original_step(observation, hidden, dt)
                        if selected == "hidden":
                            state = state * float("nan")
                        else:
                            logits = logits * float("nan")
                        return logits, state

                    with patch.object(t.model, "step", broken_step), self.assertRaises(FloatingPointError):
                        t.train_update()
                elif target == "sensitivity":
                    t.train_update()
                    t.credit.sensitivity.fill_(float("nan"))
                    with self.assertRaises(FloatingPointError):
                        t.train_update()
                else:
                    gradients = {name: torch.full_like(parameter, float("nan") if target == "gradient" else 1e308)
                                 for name, parameter in t.model.named_parameters()}
                    with self.assertRaises(FloatingPointError):
                        t._apply_gradients(gradients)

    def test_text_nonfinite_oom_schema_and_interrupt_remain_execution_failures(self):
        fixtures = (RuntimeError("CUDA out of memory; nonfinite nan in message"),
                    ValueError("nonfinite schema data"), RuntimeError("nan backend failure"), KeyboardInterrupt("user stopped"))
        for error in fixtures:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                with patch.object(run.NativeTrainer, "train_update", side_effect=error), self.assertRaises(type(error)):
                    run.train(toy(), directory, stop_at=4)
                failure = json.loads((Path(directory) / "failure.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["status"], "execution_failure")
                self.assertEqual(failure["kind"], "execution_failure")
                self.assertEqual(failure["failed_tick"], 0)
                self.assertEqual(failure["last_complete_checkpoint_sha256"], run.file_sha256(Path(directory) / "checkpoint_last_complete.pt"))
        t = run.NativeTrainer(toy())
        finite = {name: torch.ones_like(parameter) for name, parameter in t.model.named_parameters()}
        with patch.object(t.optimizer, "step", side_effect=RuntimeError("nan backend allocator")), self.assertRaises(RuntimeError):
            t._apply_gradients(finite)

    def test_finite_1e200_gradient_clips_stably_before_adam(self):
        t = run.NativeTrainer(toy(interface="core"))
        gradients = {name: torch.full_like(parameter, 1e200) for name, parameter in t.model.named_parameters()}
        captured = {}
        original_step = t.optimizer.step

        def step():
            captured.update({name: parameter.grad.clone() for name, parameter in t.model.named_parameters()})
            return original_step()

        with patch.object(t.optimizer, "step", step):
            metrics = t._apply_gradients(gradients)
        expected_norm = 1e200 * sum(value.numel() for value in gradients.values()) ** .5
        self.assertAlmostEqual(metrics["gradient_norm"] / expected_norm, 1., places=14)
        self.assertAlmostEqual(float(torch.cat([value.reshape(-1) for value in captured.values()]).norm()), 1., places=14)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in t.model.parameters()))
        for state in t.optimizer.state.values():
            self.assertTrue(bool(torch.isfinite(state["exp_avg"]).all()))
            self.assertTrue(bool(torch.isfinite(state["exp_avg_sq"]).all()))

    def test_stable_global_clip_matches_ordinary_range_reference(self):
        for dtype, tolerance in (("float64", 2e-14), ("float32", 2e-7)):
            with self.subTest(dtype=dtype):
                t = run.NativeTrainer(toy(interface="core", dtype=dtype))
                generator = torch.Generator().manual_seed(119)
                gradients = {name: torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                             for name, parameter in t.model.named_parameters()}
                reference = [torch.nn.Parameter(torch.zeros_like(value)) for value in gradients.values()]
                for parameter, value in zip(reference, gradients.values()):
                    parameter.grad = value.clone()
                expected_norm = float(torch.nn.utils.clip_grad_norm_(reference, t.config.clip, error_if_nonfinite=True))
                captured = {}
                original_step = t.optimizer.step

                def step():
                    captured.update({name: parameter.grad.clone() for name, parameter in t.model.named_parameters()})
                    return original_step()

                with patch.object(t.optimizer, "step", step):
                    metrics = t._apply_gradients(gradients)
                self.assertLess(abs(metrics["gradient_norm"] / expected_norm - 1.), tolerance)
                for actual, expected in zip(captured.values(), reference):
                    torch.testing.assert_close(actual, expected.grad, rtol=tolerance, atol=tolerance)

    def test_report_and_trace_are_small_json_safe_committed_progress(self):
        t = run.NativeTrainer(toy(credit="rtrl", structure="both", structural_window_updates=1, structural_candidates=1))
        newborn = t.report()
        self.assertEqual(newborn["ticks"], 0)
        self.assertEqual(newborn["source_hashes"], run.source_hashes())
        t.run_updates(3)
        with tempfile.TemporaryDirectory() as directory:
            expected = t.write_run(directory)
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report, expected)
            self.assertEqual((report["ticks"], report["updates"], report["trace_records"]), (12, 3, 3))
            self.assertEqual(report["last_boundary_hashes"]["continuity_hash"], t.trace[-1]["continuity_hash"])
            self.assertEqual(report["credit"]["sensitivity"]["shape"], list(t.credit.sensitivity.shape))
            self.assertEqual(report["credit"]["sensitivity"]["bytes"], t.credit.sensitivity.numel() * t.credit.sensitivity.element_size())
            self.assertNotIn("parameters", report["credit"])
            self.assertNotIn("tape", report["credit"])
            self.assertEqual([row["tick"] for row in records], [4, 8, 12])
            self.assertLess((Path(directory) / "report.json").stat().st_size, 100000)
            json.dumps(report, allow_nan=False)
            with self.assertRaises(TypeError):
                run._json_safe({"S": t.credit.sensitivity})

    def test_nonempty_new_output_is_refused_and_normal_stops_are_not_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            foreign = Path(directory) / "unbound-source.txt"
            foreign.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                run.train(toy(), directory, stop_at=4)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")
        with tempfile.TemporaryDirectory() as directory:
            run.train(toy(), directory, stop_at=4)
            self.assertFalse((Path(directory) / "failure.json").exists())
            self.assertEqual(json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["ticks"], 4)
            run.train(toy(), directory, resume=Path(directory) / "checkpoint_latest.pt", stop_at=8)
            self.assertFalse((Path(directory) / "failure.json").exists())
            records = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["tick"] for row in records], [4, 8])

    def test_cli_train_and_resume_keep_declared_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            c = toy()
            t = run.train(c, directory, stop_at=8)
            latest = Path(directory) / "checkpoint_latest.pt"
            self.assertTrue(latest.exists())
            self.assertTrue((Path(directory) / "checkpoint_tick_000000.pt").exists())
            self.assertTrue((Path(directory) / "checkpoint_tick_000008.pt").exists())
            with self.assertRaises(ValueError):
                run.train(c, directory, stop_at=12)
            restored = run.train(c, directory, resume=latest, stop_at=12)
            self.assertEqual(restored.ticks, 12)
            self.assertEqual(restored.config.total_ticks, 28)
            self.assertEqual(t.ticks, 8)
            with self.assertRaises(ValueError):
                run.train(replace(c, lr=.002), directory, resume=latest, stop_at=16)

    def test_audit_callback_runs_before_publication_and_reoffers_resumed_record(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            c = toy(credit="rtrl")

            def audit(trainer):
                self.assertTrue(trainer._at_boundary)
                self.assertEqual(trainer.credit.updates, trainer.updates)
                self.assertEqual(trainer.credit.tape[-1]["end_tick"], trainer.ticks)
                self.assertTrue(torch.equal(trainer.last_chunk["end_h"], trainer.credit.tape[-1]["end_h"]))
                cp = torch.load(Path(directory) / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
                report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
                if not calls or trainer.ticks != calls[-1]:
                    self.assertLess(cp["ticks"], trainer.ticks)
                    self.assertLess(report["ticks"], trainer.ticks)
                calls.append(trainer.ticks)

            run.train(c, directory, stop_at=8, on_update=audit)
            actual = run.train(c, directory, resume=Path(directory) / "checkpoint_latest.pt", stop_at=12, on_update=audit)
            self.assertEqual(calls, [4, 8, 8, 12])
            expected = run.NativeTrainer(c)
            expected.run_updates(3)
            self.assertEqual(continuity.state_digest(actual.semantic_state()), continuity.state_digest(expected.semantic_state()))

    def test_audit_io_failure_preserves_new_complete_credit_and_resume_repairs_record(self):
        for error in (OSError("audit disk unavailable"), FloatingPointError("audit computation failed")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                c = toy(credit="rtrl")

                def failing_audit(trainer):
                    if trainer.updates == 3:
                        raise error

                with self.assertRaises(type(error)):
                    run.train(c, directory, stop_at=16, on_update=failing_audit)
                failure = json.loads((Path(directory) / "failure.json").read_text(encoding="utf-8"))
                self.assertEqual((failure["kind"], failure["stage"], failure["last_complete_tick"]),
                                 ("execution_failure", "audit_update_callback", 12))
                last = Path(directory) / "checkpoint_last_complete.pt"
                saved = run.NativeTrainer.from_checkpoint(last)
                self.assertEqual((saved.updates, saved.credit.updates, saved.credit.tape[-1]["end_tick"]), (3, 3, 12))
                self.assertGreater(float(saved.credit.sensitivity.norm()), 0.)
                self.assertEqual(json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["ticks"], 12)
                expected = run.NativeTrainer.from_state_dict(saved.state_dict())
                expected.train_update()
                repaired = []

                def repair(trainer):
                    repaired.append(trainer.ticks)

                actual = run.train(c, directory, resume=last, stop_at=16, on_update=repair)
                self.assertEqual(repaired, [12, 16])
                self.assertEqual(continuity.state_digest(actual.semantic_state()), continuity.state_digest(expected.semantic_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fixed_rtrl_checkpoint_continuation(self):
        t = run.NativeTrainer(toy(device="cuda", credit="rtrl", interface="core"))
        t.train_update()
        restored = run.NativeTrainer.from_state_dict(t.state_dict())
        self.assertEqual(continuity.state_digest(t.train_update()), continuity.state_digest(restored.train_update()))
        self.assertEqual(continuity.state_digest(t.semantic_state()), continuity.state_digest(restored.semantic_state()))


if __name__ == "__main__":
    unittest.main()
