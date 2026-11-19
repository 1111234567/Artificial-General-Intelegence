"""Native birth, compute matching, gradient locality and exact continuation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.continuity import run as continuity
from experiments.continuity.model import AgentModel
from experiments.lifelong import run
from experiments.lifelong.replay import block_hash, combine_blocks


def tiny(**changes):
    values = dict(width=8, batch_size=2, chunk_ticks=4, initial_fan_in=2,
        replay_capacity=2, block_ticks=4, macrocycle_ticks=16, main_ticks=32,
        withdrawal_ticks=16, return_ticks=8, structural_window_updates=3)
    values.update(changes)
    return run.NativeConfig(**values)


def digest(trainer):
    return continuity.state_digest(trainer.semantic_state())


class NativeBirthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_dimensions_and_budget_are_native(self):
        config = run.NativeConfig()
        self.assertEqual(config.total_ticks, 172032)
        self.assertEqual(config.total_updates, 1344)
        self.assertEqual((config.width, config.batch_size, config.chunk_ticks), (128, 32, 128))
        trainer = run.NativeTrainer(config)
        self.assertEqual(trainer.model.input_size, 112)
        self.assertEqual(trainer.model.n_actions, 11)
        self.assertEqual(trainer.birth["source"], "from_scratch")
        self.assertEqual(len(trainer.optimizer.state), 0)
        self.assertEqual(trainer.replay.ticks_seen, 0)
        self.assertEqual(trainer.operator.updates, 0)
        self.assertTrue(torch.equal(trainer.h, torch.zeros_like(trainer.h)))
        self.assertEqual(trainer.config.wait_weight, .1)
        self.assertEqual(trainer.operator.config.idle_lambda, 0.)
        trainer._validate_continuation()

    def test_all_architectures_have_same_random_initial_function(self):
        trainers = [run.NativeTrainer(tiny(architecture=architecture, seed=19))
                    for architecture in run.ARCHITECTURES]
        expected = trainers[0]
        for trainer in trainers[1:]:
            self.assertEqual(trainer.initial_parameter_hash, expected.initial_parameter_hash)
            self.assertEqual(continuity.state_digest(trainer.model.state_dict()),
                             continuity.state_digest(expected.model.state_dict()))
            self.assertEqual(continuity.state_digest(trainer.world.state_dict()),
                             continuity.state_digest(expected.world.state_dict()))
            self.assertEqual(continuity.state_digest(trainer.birth), continuity.state_digest(expected.birth))
        observation = torch.linspace(-1., 1., 2 * 112).reshape(2, 112)
        outputs = [trainer.model.step(observation, trainer.h, .25) for trainer in trainers]
        for logits, hidden in outputs[1:]:
            self.assertTrue(torch.equal(logits, outputs[0][0]))
            self.assertTrue(torch.equal(hidden, outputs[0][1]))
        self.assertEqual([int(trainer.operator.allowed_mask.sum()) for trainer in trainers], [16, 16, 64])

    def test_gaussian_is_masked_by_private_fanin_then_rescaled(self):
        config = run.NativeConfig(seed=7)
        trainer = run.NativeTrainer(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(trainer.birth["initialization_seed"])
            dense = AgentModel(input_size=112, width=128)
        mask = trainer.model.recurrent.detach().ne(0)
        self.assertTrue(torch.equal(mask.sum(-1), torch.full((128,), 32)))
        self.assertEqual(trainer.birth["initial_recurrent_rescale"], 2.)
        self.assertTrue(torch.equal(trainer.model.recurrent, dense.recurrent * mask * 2))
        self.assertTrue(torch.equal(trainer.model.encoder.weight, dense.encoder.weight))
        self.assertTrue(torch.equal(trainer.model.head.weight, dense.head.weight))
        self.assertTrue(torch.equal(trainer.model.head.bias, dense.head.bias))

    def test_private_mask_and_world_sources_are_unchanged_across_methods(self):
        trainers = [run.NativeTrainer(tiny(seed=8, learning=learning, coverage=coverage))
                    for coverage in ("narrow", "wide") for learning in run.LEARNING_METHODS]
        self.assertEqual(len({trainer.birth["initial_parameter_hash"] for trainer in trainers}), 1)
        self.assertEqual(len({trainer.birth["data_seed"] for trainer in trainers}), 1)
        self.assertEqual(len({trainer.birth["mask_seed"] for trainer in trainers}), 1)
        for left, right in (trainers[:2], trainers[2:]):
            self.assertEqual(continuity.state_digest(left.world.state_dict()),
                             continuity.state_digest(right.world.state_dict()))
            left.run_updates(4)
            right.run_updates(4)
            self.assertEqual(left.world.event_hash, right.world.event_hash)


class NativeLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_startup_fallback_and_one_adam_step(self):
        trainer = run.NativeTrainer(tiny(learning="history_replay"))
        first = trainer.train_update()
        self.assertEqual(first["auxiliary"]["source"], "startup")
        self.assertEqual((first["auxiliary"]["forward_ticks"], first["auxiliary"]["warmup_ticks"]), (4, 0))
        self.assertEqual(len(trainer.replay.blocks), 0)
        self.assertEqual(trainer.replay.partial["end_tick"], 4)
        second = trainer.train_update()
        self.assertEqual(second["auxiliary"]["source"], "fallback_recent")
        self.assertEqual((second["auxiliary"]["forward_ticks"], second["auxiliary"]["warmup_ticks"]), (8, 4))
        self.assertEqual(len(trainer.replay.blocks), 1)
        third = trainer.train_update()
        self.assertEqual(third["auxiliary"]["source"], "history_replay")
        self.assertLessEqual(third["auxiliary"]["end_tick"], 8)
        self.assertEqual(trainer.gradient_updates, 3)
        self.assertEqual(trainer.work["optimizer_steps"], 3)
        self.assertEqual(trainer.work["backward_calls"], 6)
        for state in trainer.optimizer.state.values():
            self.assertEqual(float(state["step"]), 3.)

    def test_repeat_and_replay_match_all_compute_counters(self):
        repeat = run.NativeTrainer(tiny(learning="recent_repeat", seed=5))
        replay = run.NativeTrainer(tiny(learning="history_replay", seed=5))
        # First two updates have identical selected experience and parameters.
        for _ in range(2):
            repeat.train_update()
            replay.train_update()
            self.assertEqual(continuity.state_digest(repeat.model.state_dict()),
                             continuity.state_digest(replay.model.state_dict()))
        repeat.run_updates(5)
        replay.run_updates(5)
        self.assertEqual(repeat.work, replay.work)
        self.assertEqual(repeat.replay.blocks_seen, replay.replay.blocks_seen)
        self.assertEqual(repeat.replay.admission_rng.getstate(), replay.replay.admission_rng.getstate())
        self.assertEqual(repeat.work["live_forward_ticks"], 28)
        self.assertEqual(repeat.work["aux_forward_ticks"], 52)
        self.assertEqual(repeat.work["aux_warmup_ticks"], 24)
        self.assertEqual(repeat.work["bp_graph_ticks"], 56)
        self.assertEqual(repeat.work["live_loss_positions"], 28 * 2 * 4)
        self.assertEqual(repeat.work["aux_loss_positions"], 28 * 2 * 4)
        self.assertEqual(replay.replay.samples_drawn, 5)
        self.assertEqual(repeat.replay.samples_drawn, 0)

    def test_auxiliary_replays_original_data_and_preserves_live_continuation(self):
        trainer = run.NativeTrainer(tiny(learning="history_replay"))
        trainer.run_updates(2)
        _, _, current = trainer._live_chunk()
        before_h = trainer.h.clone()
        before_world = continuity.state_digest(trainer.world.state_dict())
        before_weights = continuity.state_digest(trainer.model.state_dict())
        block = deepcopy(trainer.replay.blocks[0])
        gradients, info = trainer._auxiliary_gradients(current)
        self.assertEqual(info["source_hash"], block_hash(block))
        self.assertEqual(before_world, continuity.state_digest(trainer.world.state_dict()))
        self.assertEqual(before_weights, continuity.state_digest(trainer.model.state_dict()))
        self.assertTrue(torch.equal(before_h, trainer.h))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients.values()))
        # Auxiliary targets are the retained causal labels even if a private
        # oracle is now unavailable. No observe/oracle/act route is needed.
        with patch.object(trainer.world, "observe", side_effect=AssertionError("observed during replay")), \
             patch.object(trainer.world, "oracle", side_effect=AssertionError("relabeled history")), \
             patch.object(trainer.world, "act", side_effect=AssertionError("acted during replay")):
            trainer._auxiliary_gradients(current)

    def test_warmup_is_no_grad_suffix_is_full_graph_at_same_weights(self):
        trainer = run.NativeTrainer(tiny())
        trainer.train_update()
        _, _, current = trainer._live_chunk()
        seen = []
        original = trainer.model.step

        def observe_step(*args, **kwargs):
            seen.append(torch.is_grad_enabled())
            return original(*args, **kwargs)

        with patch.object(trainer.model, "step", side_effect=observe_step):
            gradients, info = trainer._auxiliary_gradients(current)
        self.assertEqual(seen, [False] * 4 + [True] * 4)
        block = combine_blocks(trainer.last_chunk, current)
        hidden = block["start_h"].clone()
        with torch.no_grad():
            for observation, dt in zip(block["observations"][:4], block["elapsed"][:4]):
                _, hidden = trainer.model.step(observation, hidden, dt)
        logits = []
        for observation, dt in zip(block["observations"][4:], block["elapsed"][4:]):
            output, hidden = trainer.model.step(observation, hidden, dt)
            logits.append(output)
        loss = trainer._loss(torch.stack(logits), block["targets"][4:])
        expected = torch.autograd.grad(loss, tuple(trainer.model.parameters()))
        for name, value in zip(dict(trainer.model.named_parameters()), expected):
            self.assertTrue(torch.equal(gradients[name], value))
        self.assertEqual(info["loss_ticks"], 4)

    def test_gradients_are_averaged_then_eligibility_masked_then_joint_clipped(self):
        trainer = run.NativeTrainer(tiny(clip=.0001))
        original = trainer._apply_gradients
        captured = {}

        def fixed_gradients(current):
            del current
            gradients = {name: torch.full_like(value, 3.) for name, value in trainer.model.named_parameters()}
            return gradients, {"loss": 0., "source": "test"}

        live = {name: torch.full_like(value, 1.) for name, value in trainer.model.named_parameters()}
        _, loss, chunk = trainer._live_chunk()
        with patch.object(trainer, "_live_chunk", return_value=(live, loss, chunk)), \
             patch.object(trainer, "_auxiliary_gradients", side_effect=fixed_gradients):
            def capture(gradients):
                captured.update({key: value.clone() for key, value in gradients.items()})
                return original(gradients)
            with patch.object(trainer, "_apply_gradients", side_effect=capture):
                point = trainer.train_update()
        self.assertTrue(all(torch.equal(value, torch.full_like(value, 2.)) for value in captured.values()))
        raw = captured["recurrent"]
        eligible = raw * trainer.operator.allowed_mask
        expected_square = sum(float(value.double().square().sum()) for name, value in captured.items() if name != "recurrent")
        expected_square += float(eligible.double().square().sum())
        self.assertAlmostEqual(point["update_metrics"]["gradient_norm"], expected_square ** .5, places=4)
        self.assertTrue(torch.equal(trainer.model.recurrent[~trainer.operator.allowed_mask], torch.zeros_like(raw[~trainer.operator.allowed_mask])))

    def test_fixed_eligible_zero_can_regrow_but_ineligible_stays_zero(self):
        trainer = run.NativeTrainer(tiny())
        allowed = trainer.operator.allowed_mask
        row, col = allowed.nonzero()[0].tolist()
        # A real first shrink retires an allowed nonzero; a subsequent ordinary
        # Adam step can exceed shrink and regrow it without a shadow proposal.
        with torch.no_grad():
            trainer.model.recurrent[row, col] = 1e-5
        zero = {name: torch.zeros_like(parameter) for name, parameter in trainer.model.named_parameters()}
        retired = trainer._apply_gradients(zero)
        self.assertEqual(float(trainer.model.recurrent[row, col].detach()), 0.)
        self.assertFalse(bool(trainer.operator.support_mask[row, col]))
        self.assertTrue(bool(trainer.operator.allowed_mask[row, col]))
        self.assertGreaterEqual(retired["structure"]["deaths"], 1)
        gradients = {name: torch.ones_like(parameter) for name, parameter in trainer.model.named_parameters()}
        born = trainer._apply_gradients(gradients)
        self.assertNotEqual(float(trainer.model.recurrent[row, col].detach()), 0.)
        self.assertTrue(bool(trainer.operator.allowed_mask[row, col]))
        self.assertGreaterEqual(born["structure"]["births"], 1)
        self.assertTrue(torch.equal(trainer.model.recurrent[~allowed], torch.zeros_like(trainer.model.recurrent[~allowed])))

    def test_phases_change_future_policy_once_on_next_live_update(self):
        trainer = run.NativeTrainer(tiny(learning="history_replay"))
        trainer.run_updates(8)
        self.assertEqual(trainer.ticks, 32)
        self.assertNotEqual(tuple(trainer.world.family_allowed), (1, 2, 3))
        # Saving at the exact boundary must not pre-apply the future policy.
        state = trainer.state_dict()
        world_before = continuity.state_digest(state["world"])
        self.assertEqual(world_before, continuity.state_digest(trainer.world.state_dict()))
        hidden, history_before = trainer.h.clone(), len(trainer.schedule_history)
        trainer.train_update()
        self.assertEqual(tuple(trainer.world.family_allowed), (1, 2, 3))
        self.assertEqual(len(trainer.schedule_history), history_before + 1)
        self.assertFalse(torch.equal(trainer.h, torch.zeros_like(hidden)))
        trainer.run_updates(3)
        self.assertEqual(trainer.ticks, 48)
        self.assertEqual(tuple(trainer.world.family_allowed), (1, 2, 3))
        trainer.train_update()
        self.assertEqual(tuple(trainer.world.family_allowed), (0,))
        self.assertEqual(trainer.hidden_resets, 0)
        self.assertEqual(trainer.work["optimizer_steps"], trainer.updates)

    def test_live_and_auxiliary_family_recency_use_distinct_clocks(self):
        trainer = run.NativeTrainer(tiny())
        self.assertEqual(trainer.last_live_family_tick, [-1] * 4)
        self.assertEqual(trainer.last_aux_family_tick, [-1] * 4)
        self.assertEqual(trainer.live_supervised_family_positions, [0] * 4)
        for _ in range(3):
            before = list(trainer.live_supervised_family_positions)
            trainer.train_update()
            chunk = trainer.last_chunk
            context = chunk["family_context"]
            for family in range(4):
                count = int(context.eq(family).sum())
                self.assertEqual(trainer.live_supervised_family_positions[family], before[family] + count)
                positions = context.eq(family).any(-1).any(-1).nonzero().flatten()
                if len(positions):
                    self.assertEqual(trainer.last_live_family_tick[family], chunk["start_tick"] + int(positions[-1]))
                    self.assertEqual(trainer.last_aux_family_tick[family], trainer.ticks)
        report = trainer.family_recency()
        self.assertEqual(report["last_command_tick"], trainer.world.family_recency()["last_command_tick"])
        self.assertEqual(report["last_live_family_tick"], trainer.last_live_family_tick)
        self.assertEqual(report["last_aux_family_tick"], trainer.last_aux_family_tick)
        self.assertEqual(report["live_supervised_family_positions"], trainer.live_supervised_family_positions)

    def test_live_data_digest_binds_original_family_context(self):
        trainer = run.NativeTrainer(tiny())
        point = trainer.train_update()
        chunk = deepcopy(trainer.last_chunk)
        expected = continuity.state_digest({key: chunk[key] for key in run.DATA_HASH_FIELDS})
        self.assertEqual(point["data_chunk_hash"], expected)
        chunk["family_context"][0, 0, 0] = (int(chunk["family_context"][0, 0, 0]) + 1) % 4
        changed = continuity.state_digest({key: chunk[key] for key in run.DATA_HASH_FIELDS})
        self.assertNotEqual(changed, expected)

    def test_replay_supervision_never_refreshes_real_cue_or_live_feedback_recency(self):
        trainer = run.NativeTrainer(tiny(learning="history_replay", replay_capacity=8))
        trainer.run_updates(8)
        source = deepcopy(trainer.replay.blocks[0])
        family = int(source["family_context"][4:][source["family_context"][4:] >= 0][0])
        trainer.run_updates(3)
        _, _, current = trainer._live_chunk()
        world_before = continuity.state_digest(trainer.world.state_dict())
        live_before = list(trainer.last_live_family_tick)
        cue_before = trainer.world.family_recency()
        with patch.object(trainer.replay, "sample", return_value=source), \
             patch.object(trainer.replay, "eligible_count", return_value=1):
            trainer._auxiliary_gradients(current)
        self.assertEqual(trainer.last_aux_family_tick[family], trainer.ticks)
        self.assertEqual(trainer.last_live_family_tick, live_before)
        self.assertEqual(trainer.world.family_recency(), cue_before)
        self.assertEqual(continuity.state_digest(trainer.world.state_dict()), world_before)


class NativeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _resume_case(self, device, architecture, split):
        config = tiny(device=device, architecture=architecture, learning="history_replay", seed=11)
        uninterrupted = run.NativeTrainer(config)
        uninterrupted.run_updates(split)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "native.pt"
            uninterrupted.save_checkpoint(checkpoint)
            checkpoint_digest = continuity.state_digest(uninterrupted.semantic_state())
            uninterrupted.run_updates(config.total_updates - split)
            expected = digest(uninterrupted)
            with patch.object(continuity, "seed_all", side_effect=AssertionError("reseeded during restoration")), \
                 patch.object(run.NativeTrainer, "__init__", side_effect=AssertionError("replayed random birth")):
                resumed = run.NativeTrainer.from_checkpoint(checkpoint, device=device)
            self.assertEqual(digest(resumed), checkpoint_digest)
            resumed.run_updates(config.total_updates - split)
            self.assertEqual(digest(resumed), expected)
            self.assertEqual(resumed.hidden_resets, 0)
            self.assertEqual(resumed.replay.chunks_seen, config.total_updates)
            self.assertEqual(resumed.operator.updates, config.total_updates)
            resumed._validate_continuation()

    def test_cpu_exact_resume_mid_reservoir_and_shadow_window(self):
        for architecture in run.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self._resume_case("cpu", architecture, 5)

    def test_cpu_exact_resume_before_phase_policy_changes(self):
        for split in (8, 12):
            with self.subTest(split=split):
                self._resume_case("cpu", "grow_shrink", split)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_exact_resume_full_native_state(self):
        for architecture in run.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self._resume_case("cuda", architecture, 5)
        self._resume_case("cuda", "grow_shrink", 12)

    def test_newborn_native_checkpoint_has_no_optimizer_moments(self):
        trainer = run.NativeTrainer(tiny())
        restored = run.NativeTrainer.from_state_dict(trainer.state_dict())
        self.assertEqual(digest(trainer), digest(restored))
        self.assertEqual(len(restored.optimizer.state), 0)
        self.assertEqual(restored.birth["source"], "from_scratch")

    def test_restore_reinstates_numeric_runtime_without_rng_reseeding(self):
        trainer = run.NativeTrainer(tiny())
        trainer.run_updates(3)
        state = trainer.state_dict()
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with patch.object(continuity, "seed_all", side_effect=AssertionError("reseeded")):
            resumed = run.NativeTrainer.from_state_dict(state)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(digest(resumed), continuity.state_digest({key: state[key] for key in run._SEMANTIC_FIELDS}))

    def test_checkpoint_rejects_incomplete_and_tampered_native_state(self):
        trainer = run.NativeTrainer(tiny(architecture="grow_shrink"))
        trainer.run_updates(5)
        original = trainer.state_dict()
        mutations = []
        for field in ("operator", "last_chunk", "rng", "sources", "work"):
            damaged = deepcopy(original)
            damaged.pop(field)
            mutations.append(damaged)
        damaged = deepcopy(original)
        damaged["config"].pop("coverage")
        mutations.append(damaged)
        damaged = deepcopy(original)
        damaged["model_config"]["input_size"] = 96
        mutations.append(damaged)
        damaged = deepcopy(original)
        damaged["ticks"] += 4
        mutations.append(damaged)
        damaged = deepcopy(original)
        damaged["last_auxiliary"]["block"]["targets"][0, 0, 0] += 1
        mutations.append(damaged)
        damaged = deepcopy(original)
        damaged["sources"]["world.py"] += "\n# tamper\n"
        mutations.append(damaged)
        for damaged in mutations:
            with self.subTest(keys=tuple(damaged)):
                with self.assertRaises((ValueError, KeyError)):
                    run.NativeTrainer.from_state_dict(damaged)

    def test_strict_sources_check_every_runtime_and_frozen_dependency(self):
        trainer = run.NativeTrainer(tiny())
        original = trainer.state_dict()
        for field, hashes_field in (("sources", "source_hashes"), ("dependencies", "dependency_hashes")):
            for name in original[field]:
                damaged = deepcopy(original)
                damaged[field][name] += "\n# changed implementation\n"
                damaged[hashes_field] = run.source_hashes(damaged[field])
                with self.subTest(field=field, name=name):
                    with self.assertRaises(ValueError):
                        run.NativeTrainer.from_state_dict(damaged)
        self.assertEqual(set(original["sources"]), set(run.RUNTIME_SOURCES))
        self.assertEqual(set(original["dependencies"]), set(run.DEPENDENCIES))

    def test_restore_binds_retained_context_to_original_trace_after_outer_rehash(self):
        trainer = run.NativeTrainer(tiny(learning="history_replay"))
        trainer.run_updates(5)
        original = trainer.state_dict()
        for name in ("replay", "last_chunk"):
            damaged = deepcopy(original)
            block = damaged["replay"]["blocks"][0] if name == "replay" else damaged["last_chunk"]
            context = block["family_context"]
            context[0, 0, 0] = (int(context[0, 0, 0]) + 1) % 4
            for key in damaged["state_hashes"]:
                if key != "semantic":
                    damaged["state_hashes"][key] = continuity.state_digest(damaged[key])
            damaged["state_hashes"]["semantic"] = continuity.state_digest({key: damaged[key] for key in run._SEMANTIC_FIELDS})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "original causal live digest"):
                run.NativeTrainer.from_state_dict(damaged)

    def test_json_report_and_owned_last_chunk(self):
        trainer = run.NativeTrainer(tiny())
        trainer.run_updates(3)
        state = trainer.state_dict()
        original_hash = block_hash(trainer.last_chunk)
        state["last_chunk"]["observations"].zero_()
        self.assertEqual(block_hash(trainer.last_chunk), original_hash)
        with tempfile.TemporaryDirectory() as folder:
            run.write_run(folder, trainer)
            self.assertTrue((Path(folder) / "report.json").is_file())
            self.assertTrue((Path(folder) / "trace.jsonl").is_file())
            for name in run.RUNTIME_SOURCES:
                self.assertTrue((Path(folder) / "sources" / name).is_file())


class NativeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_latest_and_report_exist_before_first_update(self):
        trainer = run.NativeTrainer(tiny())
        original = trainer.train_update
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"

            def update():
                state = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
                self.assertEqual(state["ticks"], 0)
                self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8"))["ticks"], 0)
                self.assertTrue((output / "checkpoints" / "tick_000000.pt").is_file())
                return original()

            with patch.object(run, "NativeTrainer", return_value=trainer), \
                 patch.object(trainer, "train_update", side_effect=update), \
                 redirect_stdout(io.StringIO()):
                run.main(["--output", str(output), "--stop-at", "4"])
            latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(latest["ticks"], 4)
            self.assertEqual(torch.get_num_threads(), 1)

    def test_new_run_refuses_existing_output_before_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(run, "NativeTrainer") as constructor, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run.main(["--output", folder])
                self.assertEqual(caught.exception.code, 2)
                constructor.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_interruption_saves_only_completed_boundary(self):
        for completed in (False, True):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as folder:
                trainer = run.NativeTrainer(tiny())
                original = trainer.train_update
                output = Path(folder) / "new-run"

                def interrupted():
                    if completed:
                        original()
                    else:
                        trainer._at_boundary = False
                    raise KeyboardInterrupt()

                with patch.object(run, "NativeTrainer", return_value=trainer), \
                     patch.object(trainer, "train_update", side_effect=interrupted), \
                     redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt):
                        run.main(["--output", str(output), "--stop-at", "8"])
                state = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
                self.assertEqual(state["ticks"], 4 if completed else 0)
                self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8"))["ticks"], state["ticks"])

    def test_progress_is_emitted_at_128_updates(self):
        config = tiny(main_ticks=512)
        fake = SimpleNamespace(config=config, ticks=0, updates=0, device="cpu", seconds=0., _at_boundary=True)

        def update():
            fake.updates += 1
            fake.ticks += config.chunk_ticks
            fake.seconds += 1.
            return {"phase": "main", "live_loss": 1., "auxiliary": {"loss": 2.},
                    "update_metrics": {"structure": {"density": .25}}}

        fake.train_update = update
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            output_text = io.StringIO()
            with patch.object(run, "NativeTrainer", return_value=fake), \
                 patch.object(run, "write_run"), patch.object(run.torch, "set_num_threads") as threads:
                fake.save_checkpoint = lambda path: None
                with redirect_stdout(output_text):
                    run.main(["--output", str(output), "--stop-at", "512"])
                threads.assert_called_once_with(1)
            self.assertIn("update=128/", output_text.getvalue())
            self.assertIn("live=1.00000 aux=2.00000 density=0.2500", output_text.getvalue())


if __name__ == "__main__":
    unittest.main()
