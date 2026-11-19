"""Gradient masks, current-prefix anchors, live h, exact state and no-replay math."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.run import cpu_tree, state_digest
from experiments.coverage_retention import train as frozen
from experiments.coverage_retention.train_test import tiny_stream, tiny_source
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.episodic_review import train


def tiny_bank():
    group = tiny_stream(ticks=4, batch=3)
    group["action_mask"] = np.zeros((4,3,2), bool)
    group["action_mask"][1:,np.arange(3),np.arange(3)%2] = True
    group["actor_lane"] = np.array([2,0,2], np.int64)
    group["anchor_start"] = np.array([5,9,11], np.int64)
    return dict(A=deepcopy(group), B=deepcopy(group),
        schedule=np.tile(np.arange(3,dtype=np.int64),(32,1)))


def make(arm="none", *, source=None, stream=None, bank=None, folder=None, device="cpu"):
    source = tiny_source(device) if source is None else source
    model = RemainingTimeGRUAgent(**source["model_config"])
    model.load_state_dict(source["model"])
    history = tiny_stream(ticks=16)["obs"]
    return train.Learner(model.to(device),tiny_stream(ticks=16) if stream is None else stream,
        arm,source=source,bank=tiny_bank() if bank is None else bank,history=history,
        block=4,updates=4,replay_every=2,prefix_chunk=3,folder=folder)


def numerical_fields(cp):
    return {key:cp[key] for key in ("model","optimizer","hidden","work","stream_cursor","step",
        "torch_rng","cuda_rng","numpy_rng","source_work","cumulative_work")}


class EpisodicReviewTrainingTest(unittest.TestCase):
    def test_masked_labels_outside_target_cannot_change_any_gradient(self):
        source = tiny_source()
        a = make("replay_A",source=source)
        bank = tiny_bank()
        group = bank["A"]
        group["actions"][~group["action_mask"]] = 10
        group["time_target"][~group["action_mask"]] = np.nan
        group["time_mask"][~group["action_mask"]] = True
        b = make("replay_A",source=source,bank=bank)
        for _ in range(2):
            a.update(); b.update()
        train.assert_exact(numerical_fields(a.payload()),numerical_fields(b.payload()),"outside_mask")
        for (name,left),(_,right) in zip(a.model.named_parameters(),b.model.named_parameters()):
            train.assert_exact(left.grad,right.grad,name)

    def test_replay_never_replaces_live_hidden_and_live_data_always_consumed(self):
        learner = make("replay_A")
        learner.update()
        obs = torch.from_numpy(learner.stream["obs"][4:8])
        with torch.no_grad():
            _,expected = learner.model(obs,learner.hidden)
        with tempfile.TemporaryDirectory() as directory:
            learner.folder = Path(directory)
            learner.update(save_audit=True)
            saved = torch.load(Path(directory)/"step002_gradients.pt",weights_only=False)
            train.assert_exact(expected,learner.hidden,"exact_live_h")
            self.assertFalse(torch.equal(learner.hidden,saved["replay_next_hidden"]))
            train.assert_exact(learner.hidden,saved["next_hidden"])
            after = torch.load(Path(directory)/"step002_after.pt",weights_only=False)
            self.assertEqual(after["replay_work"]["live_forward_updates"],2)
        self.assertEqual(learner.cursor,8)
        self.assertEqual(learner.replay_work["live_forward_ticks"],8)
        self.assertEqual(learner.replay_work["historical_gradient_updates"],1)
        self.assertEqual(learner.work["action_rows"],8*3*2)
        self.assertEqual(learner.work["action_gradient_rows"],4*3*2+3*3)

    def test_current_prefix_reconstruction_uses_full_path_from_zero(self):
        learner = make("replay_A")
        group = learner.bank["A"]
        lanes,starts = group["actor_lane"],group["anchor_start"]
        stale,_ = train.reconstruct_anchors(learner.model,learner.history,lanes,starts,3)
        with torch.no_grad():
            learner.model.core.weight_hh_l0.add_(.3)
        actual,work = train.reconstruct_anchors(learner.model,learner.history,lanes,starts,3)
        unique = np.unique(lanes)
        hidden = learner.model.initial_state(len(unique))
        expected = torch.empty_like(actual)
        # Independently step through every old public observation, even before
        # the window's prelude; no bank labels or archived h are available here.
        with torch.no_grad():
            for tick in range(int(starts.max())+1):
                for row in np.flatnonzero(starts == tick):
                    expected[row] = hidden[np.flatnonzero(unique == lanes[row])[0]]
                if tick < int(starts.max()):
                    obs = torch.from_numpy(learner.history[tick:tick+1,unique].copy())
                    _,hidden = learner.model(obs,hidden)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
        self.assertFalse(torch.equal(actual,stale))
        self.assertEqual(work["prefix_lane_ticks"],int(starts.max())*len(unique))
        self.assertIsNone(actual.grad_fn)
        self.assertTrue(learner.model.training)

    def test_source_model_adam_hidden_rng_copied_and_resume_exact(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = tiny_source(device)
        for arm in train.ARMS:
            learner = make(arm,source=source,device=device)
            for key in ("model","optimizer","hidden","torch_rng","cuda_rng","numpy_rng"):
                train.assert_exact(learner.payload()[key],source[key],key)
            self.assertEqual(train.optimizer_steps(learner.optimizer),[256])
            learner.update(); learner.update()
            resumed = train.Learner.restore(learner.payload(),learner.stream,
                bank=learner.bank,history=learner.history,device=device)
            # Timings are measured quantities, not resumable numerical state.
            for _ in range(2):
                learner.update(); resumed.update()
            train.assert_exact(numerical_fields(learner.payload()),numerical_fields(resumed.payload()),arm)
            for key,value in learner.replay_work.items():
                if not key.endswith("seconds"):
                    self.assertEqual(value,resumed.replay_work[key],key)
            self.assertEqual(train.optimizer_steps(resumed.optimizer),[260])

    def test_none_exact_frozen_continuation_math_and_first_seven_updates(self):
        source = tiny_source()
        stream = tiny_stream(ticks=32)
        model = RemainingTimeGRUAgent(**source["model_config"])
        model.load_state_dict(source["model"])
        old = frozen.Learner(model,stream,"withdraw",source=source,block=4,updates=8)
        learners = []
        for arm in train.ARMS:
            learner = make(arm,source=source)
            learner.stream,learner.updates,learner.replay_every = stream,8,8
            learners.append(learner)
        for step in range(8):
            old.update()
            for learner in learners:
                learner.update()
            train.assert_exact(numerical_fields(old.payload()),numerical_fields(learners[0].payload()),"none_exact")
            if step < 7:
                for learner in learners[1:]:
                    train.assert_exact(numerical_fields(old.payload()),numerical_fields(learner.payload()),"first_seven")
        self.assertNotEqual(state_digest(learners[0].model.state_dict()),state_digest(learners[1].model.state_dict()))

    def test_all_real_initial_states_and_no_warm(self):
        for seed in train.SEEDS:
            source = train.load_source(seed)
            for arm in train.ARMS:
                learner = train.learner_from_source(seed,arm,tiny_stream(ticks=128,batch=32),
                    device="cpu",block=128,updates=1,bank=tiny_bank(),history=tiny_stream()["obs"])
                for key in ("model","optimizer","hidden","torch_rng","numpy_rng"):
                    train.assert_exact(learner.payload()[key],source[key],f"{seed}.{arm}.{key}")
                with self.assertRaisesRegex(RuntimeError,"no warm"):
                    learner.warm_start()


if __name__ == "__main__":
    unittest.main()
