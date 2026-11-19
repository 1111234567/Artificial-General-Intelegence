"""Exact complete source restoration and native B-only retention updates."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments.coverage_retention import train as frozen
from experiments.coverage_retention.train_test import tiny_stream
from experiments.episodic_review.train_test import make as make_primary, numerical_fields
from experiments.episodic_review import retention_train as train
from experiments.remaining_time.model import RemainingTimeGRUAgent


def tiny_primary():
    learner = make_primary("replay_A")
    learner.update(); learner.update()
    cp = learner.payload()
    for state in cp["optimizer"]["state"].values():
        state["step"].fill_(512)
    cp.update(step=256,stream_cursor=32768,cumulative_stream_cursor=66048)
    cp["work"].update(training_updates=256,training_ticks=32768)
    cp["cumulative_work"] = {key:cp["source_work"][key]+cp["work"][key] for key in cp["work"]}
    return cp


def make(arm="none",source=None):
    source = tiny_primary() if source is None else source
    model = RemainingTimeGRUAgent(**source["model_config"])
    model.load_state_dict(source["model"])
    return train.Learner(model,tiny_stream(ticks=16),arm,source=source,block=4,updates=4)


class RetentionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.path_patch = patch.object(train,"states_path",return_value=Path("retention-test-world.pt"))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_complete_primary_source_copy_counters_and_no_replay(self):
        source = tiny_primary()
        for arm in train.ARMS:
            learner = make(arm,source)
            cp = learner.payload()
            for key in ("model","optimizer","hidden","torch_rng","numpy_rng"):
                train.assert_exact(cp[key],source[key],key)
            train.assert_exact(cp["source_work"],source["cumulative_work"],"full512_work")
            train.assert_exact(cp["primary_work"],source["work"],"local_primary_work")
            train.assert_exact(cp["source_replay_work"],source["replay_work"],"archived_replay_work")
            self.assertEqual(cp["source_stream_cursor"],66048)
            self.assertEqual(cp["cumulative_stream_cursor"],66048)
            self.assertEqual(cp["study_step"],256)
            self.assertEqual(cp["optimizer_state_steps"],[512])
            self.assertFalse(cp["replay"])
            self.assertTrue(all(value==0 for value in learner.work.values()))
            with self.assertRaisesRegex(RuntimeError,"no warm"):
                learner.warm_start()

    def test_real_default_budget_is_64_blocks_without_main_budget_leak(self):
        source = tiny_primary()
        model = RemainingTimeGRUAgent(**source["model_config"])
        model.load_state_dict(source["model"])
        learner = train.Learner(model,tiny_stream(ticks=8192),"none",source=source)
        self.assertEqual(learner.payload()["settings"],dict(warm=0,block=128,updates=64))

    def test_identical_frozen_action_update_and_live_hidden(self):
        source = tiny_primary()
        learner = make("replay_A",source)
        model = RemainingTimeGRUAgent(**source["model_config"])
        model.load_state_dict(source["model"])
        adapted = deepcopy(source)
        adapted.update(work=deepcopy(source["cumulative_work"]),stream_cursor=66048)
        reference = frozen.Learner(model,learner.stream,"withdraw",source=adapted,block=4,updates=4)
        for _ in range(4):
            obs = torch.from_numpy(learner.stream["obs"][learner.cursor:learner.cursor+4])
            with torch.no_grad():
                _,expected_h = learner.model(obs,learner.hidden)
            learner.update(); reference.update()
            train.assert_exact(learner.hidden,expected_h,"live_B_h")
            train.assert_exact(numerical_fields(learner.payload()),numerical_fields(reference.payload()),"frozen_update")
        self.assertEqual(train.optimizer_steps(learner.optimizer),[516])
        self.assertEqual(learner.payload()["study_step"],260)
        self.assertEqual(learner.payload()["cumulative_stream_cursor"],66064)

    def test_suffix_resume_exact_preserves_primary_replay_accounting(self):
        source = tiny_primary()
        for arm in train.ARMS:
            learner = make(arm,source)
            learner.update(); learner.update()
            resumed = train.Learner.restore(learner.payload(),learner.stream)
            for _ in range(2):
                learner.update(); resumed.update()
            train.assert_exact(numerical_fields(learner.payload()),numerical_fields(resumed.payload()),"resume")
            train.assert_exact(resumed.source_replay_work,source["replay_work"],"no_new_replay")


if __name__=="__main__":
    unittest.main()
