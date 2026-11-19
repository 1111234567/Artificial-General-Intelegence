"""Synthetic runner checks; these tests never score a trained held-out policy."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.associative_state import evaluate, train
from experiments.associative_state.bridge import translate
from experiments.associative_state.bridge_test import blank, cue, receipt
from experiments.associative_state.model import AssociationMemory
from experiments.learn_remember.gru_train import GRUAgent
from experiments.teacher_timing import data as source_data


def subject_for_test(kind="delta"):
    """An untrained small executor, independent of all final checkpoint files."""
    model = AssociationMemory(kind)
    executor = GRUAgent(width=8)
    executor.eval().requires_grad_(False)
    with patch.object(evaluate, "load", return_value=(model, executor)):
        return evaluate.Subject(0, kind)


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sample_queries_target_last_visible_write_not_hidden_revision(self):
        batch, events = 32, 48
        keys, values, queries, targets = train.sample(np.random.default_rng(877), batch, events)
        known = np.full((batch, 8), -1, dtype=np.int64)
        first_table = np.full((batch, 8), -1, dtype=np.int64)
        rows = np.arange(batch)
        stale_but_authoritative = 0
        refreshed = np.zeros((batch, 8), dtype=bool)
        for tick, (key, value, query, target) in enumerate(zip(keys.numpy(), values.numpy(),
                                                              queries.numpy(), targets.numpy())):
            known[rows, key] = value
            if tick < 8:
                first_table[rows, key] = value
            if tick >= events // 2:
                refreshed[rows, key] = True
                stale_but_authoritative += int(((query < 2) & ~refreshed[rows, query]).sum())
            expected = known[rows, query]
            self.assertTrue(np.all(expected >= 0))
            np.testing.assert_array_equal(target, expected)
        self.assertGreater(stale_but_authoritative, 0)
        for lane in range(batch):
            self.assertEqual(sorted(keys[:8, lane].tolist()), list(range(8)))
            for context in range(2):
                self.assertEqual(sorted(first_table[lane, 4*context:4*context+4]), list(range(4)))

    def test_train_seeds_align_initial_parameters_and_actual_sample_bytes(self):
        initial = {}
        reports = {}
        original = AssociationMemory

        def factory(*args, **kwargs):
            model = original(*args, **kwargs)
            initial[model.kind] = deepcopy(model.state_dict())
            return model

        with patch.dict(train.SETTINGS, batch=3, events=12), \
                patch.object(train, "AssociationMemory", side_effect=factory), \
                redirect_stdout(io.StringIO()):
            for kind in train.ARMS:
                reports[kind] = train.train(29, kind, pilot=True, steps=2)
        self.assertEqual(len({row["train_sample_sha256"] for row in reports.values()}), 1)
        self.assertEqual(set(initial["add"]), set(initial["delta"]))
        for name, value in initial["add"].items():
            self.assertTrue(torch.equal(value, initial["delta"][name]))
        for kind in train.ARMS:
            self.assertTrue(torch.equal(initial[kind]["key_embedding.weight"],
                                        initial["add"]["key_embedding.weight"]))

    def test_executor_load_filters_only_auxiliary_head_and_retains_native_dimensions(self):
        symbol = AssociationMemory("delta")
        executor = GRUAgent(width=192, input_size=56, n_jobs=2, n_actions=11)
        old = dict(model_config=executor.config(), model=deepcopy(executor.state_dict()))
        old["model"].update({"time_head.weight": torch.randn(2, 192),
                              "time_head.bias": torch.randn(2)})
        cp = dict(metadata=dict(model_config=symbol.config()), model=symbol.state_dict())
        protocol = dict(origins={"0": dict(path="unused-source.pt", sha256="source-digest")})
        with patch.object(evaluate.torch, "load", side_effect=(cp, old)), \
                patch.object(evaluate, "check_protocol", return_value=protocol), \
                patch.object(evaluate, "sha", return_value="source-digest"):
            actual, loaded = evaluate.load(0, "delta")
        self.assertEqual(tuple(actual.initial_state(2).shape), (2, 64))
        self.assertEqual(tuple(loaded.initial_state(2).shape), (2, 192))
        scores, _ = loaded.step(torch.zeros(2, 56), loaded.initial_state(2))
        self.assertEqual(tuple(scores.shape), (2, 2, 11))
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        for name, value in executor.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))

    def test_translation_uses_the_source_teachers_custom_seed_rules(self):
        for seed in range(3):
            source = source_data.mapping_table(seed, "related")
            self.assertEqual(source, evaluate.mapping_table(seed, "related"))
            contexts = (evaluate.SEEDS[seed]["history_cue"], evaluate.SEEDS[seed]["B_cue"])
            for family in contexts:
                for value in range(4):
                    obs = cue(blank(), family=family, raw=0)
                    translated = translate(obs, torch.tensor([[value, 0]]), source)
                    canonical = int(translated.reshape(1, 2, 28)[0, 0, 5:9].argmax())
                    self.assertEqual(source[family][canonical], value)

    def test_evaluation_mappings_have_exact_shared_conflict_structure(self):
        for seed in range(3):
            table, revised = evaluate.mappings(seed, 11)
            same, _ = evaluate.mappings(seed, 11)
            self.assertTrue(torch.equal(table, same))
            self.assertTrue(torch.equal(table[:, 0, :2], revised[:, 0, :2]))
            self.assertTrue(torch.equal(table[:, 1], revised[:, 1]))
            self.assertTrue(torch.equal((table[:, 0] != table[:, 1]).sum(-1),
                                        torch.full((11,), 2)))
            self.assertTrue(torch.equal((table[:, 0] != revised[:, 0]).sum(-1),
                                        torch.full((11,), 2)))

    def test_public_receipt_is_the_only_write_and_no_write_clear_no_read_are_effective(self):
        with patch.dict(evaluate.SETTINGS, actors=1):
            subject = subject_for_test()
            theta = evaluate.params_hash(subject.model)
            initial_hidden = subject.hidden.clone()
            subject.observe_and_update(cue(blank(), raw=2, family=subject.contexts[0]))
            self.assertEqual(int(subject.memory.count_nonzero()), 0)
            success = receipt(blank(), 6)  # public successful FINISH for value 3
            subject.observe_and_update(success, write=False)
            self.assertEqual(int(subject.memory.count_nonzero()), 0)
            self.assertEqual(subject.work["public_examples"], 1)
            subject.observe_and_update(cue(blank(), raw=2, family=subject.contexts[0]))
            subject.observe_and_update(success, write=True)
            self.assertGreater(float(subject.memory.norm()), 0.)
            self.assertEqual(int(subject.model.read(subject.memory, torch.tensor([2])).argmax()), 3)
            subject.memory.zero_()
            self.assertEqual(int(subject.memory.count_nonzero()), 0)
            self.assertEqual(evaluate.params_hash(subject.model), theta)
            self.assertTrue(torch.equal(subject.hidden, initial_hidden))
            subject.memory.normal_()
            self.assertTrue(torch.equal(subject.predictions(blank(), no_read=True),
                                        torch.zeros(1, 2, dtype=torch.long)))

    def test_online_read_loss_has_only_address_and_scale_gradients_and_keeps_theta_only_state(self):
        with patch.dict(evaluate.SETTINGS, actors=1):
            for kind in train.ARMS:
                subject = subject_for_test(kind)
                subject.memory.normal_()
                memory = subject.memory.clone()
                subject.observe_and_update(cue(blank(), raw=2, family=subject.contexts[0]))
                optimizer = torch.optim.Adam(subject.model.parameters(), lr=.003)
                subject.observe_and_update(receipt(blank(), 6), write=False, optimizer=optimizer)
                self.assertTrue(torch.equal(memory, subject.memory))
                self.assertEqual(subject.work["parameter_updates"], 1)
                self.assertEqual(subject.work["gradient_examples"], 1)
                for name, parameter in subject.model.named_parameters():
                    if name in ("read_scale_raw", "key_embedding.weight"):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    else:
                        self.assertIsNone(parameter.grad)

    def test_probe_counts_all_new_eligible_mature_requests_and_separates_censoring(self):
        # Scripted pulses explicitly include an inherited mature request, an
        # ineligible command, a failed eligible request, an end-censored request,
        # and maturation plus a new command on the same job and tick.
        subject = ScriptedSubject()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(evaluate.SETTINGS, actors=1, eval_ticks=6), \
                redirect_stdout(io.StringIO()):
            report = evaluate.probe(subject, Path(directory), "immediate")
        self.assertEqual(report["groups"]["all"]["n"], 2)
        self.assertEqual(report["groups"]["all"]["strict"], 1)
        self.assertEqual(report["groups"]["all"]["ordinary"], 1)
        self.assertEqual(sorted(row["rid"] for row in report["requests"]), [10, 12])
        self.assertEqual([row["rid"] for row in report["right_censored"]], [13])
        self.assertEqual(report["command_translations"], 4)
        self.assertEqual(report["idle_opportunities"], 12)
        self.assertEqual(report["idle_actions"], 1)
        self.assertTrue(report["memory_readonly"])
        self.assertTrue(report["parameters_frozen_during_probe"])
        self.assertEqual(subject.advance_args, [(False, False)]*6)


class ScriptedWorld:
    def __init__(self):
        self.tick = 99
        self.private_mappings = torch.tensor([[[0, 1, 2, 3], [1, 0, 2, 3]]])

    def set_family_policy(self, allowed):
        self.allowed = allowed

    def act(self, actions):
        pass

    def emit(self, index):
        self.tick += 1
        for name in ("event_kind", "event_request_id", "event_family", "raw_event_payload", "event_payload",
                     "matured_request_id"):
            setattr(self, name, torch.zeros(1, 2, dtype=torch.long))
        for name in ("event_eligible", "matured_mask", "matured_completed", "matured_clean_completed", "active"):
            setattr(self, name, torch.zeros(1, 2, dtype=torch.bool))
        events = {0: (0, 10, True), 1: (1, 11, False), 3: (0, 12, True), 4: (1, 13, True)}
        if index in events:
            job, rid, eligible = events[index]
            self.event_kind[0, job] = 1
            self.event_request_id[0, job] = rid
            self.event_eligible[0, job] = eligible
            self.event_family[0, job] = job
        mature = {2: (1, 9, True), 3: (0, 10, True), 5: (0, 12, False)}
        if index in mature:
            job, rid, completed = mature[index]
            self.matured_mask[0, job] = True
            self.matured_request_id[0, job] = rid
            self.matured_completed[0, job] = completed
            self.matured_clean_completed[0, job] = completed


class ScriptedSubject:
    def __init__(self):
        self.seed, self.arm = 777, "delta"
        self.contexts = (0, 1)
        self.world = ScriptedWorld()
        self.model = AssociationMemory(self.arm)
        self.memory = self.model.initial_state(1)
        self.coverage = torch.zeros(1, 8, dtype=torch.long)
        self.work = dict(ticks=0)
        self.advance_args = []

    def snapshot(self):
        return dict(memory=self.memory.clone(), parameter_hash=evaluate.params_hash(self.model),
                    world=dict(tick=self.world.tick), coverage=self.coverage.clone(),
                    work=deepcopy(self.work))

    def advance(self, write, no_read):
        self.advance_args.append((write, no_read))
        index = len(self.advance_args)-1
        self.world.emit(index)
        actions = torch.zeros(1, 2, dtype=torch.long)
        if index == 4:
            actions[0, 1] = 3
        return torch.zeros(1, 56), torch.zeros(1, 2, dtype=torch.long), actions, torch.zeros(1, 2, 11)


if __name__ == "__main__":
    unittest.main()
