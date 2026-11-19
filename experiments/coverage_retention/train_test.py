"""Full learned-state restoration, no warm, exact resumed Adam and carried h."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.continuity.run import cpu_tree, state_digest
from experiments.learn_remember.gru_train import GRUAgent
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.remaining_time import train as previous
from experiments.coverage_retention import train


def tiny_stream(ticks=12, batch=3):
    rng = np.random.default_rng(7441)
    shape = (ticks, batch, 2)
    action = rng.integers(0, 11, size=shape, dtype=np.int64)
    action[rng.random(shape) < .6] = 0
    mask = rng.random(shape) > .25
    target = rng.uniform(-.4, 1.5, size=shape).astype(np.float32)
    target[~mask] = 0
    return dict(obs=rng.standard_normal((ticks, batch, 56), dtype=np.float32),
        actions=action, time_target=target, time_mask=mask)


def tiny_source(device="cpu"):
    """Real learned model/moments/h; set archived Adam age to source256."""
    train.deterministic(671)
    model = RemainingTimeGRUAgent.from_original(GRUAgent(width=7), "action").to(device)
    old = previous.Learner(model, tiny_stream(ticks=14), "action", warm=2, block=4, updates=3)
    old.warm_start()
    for _ in range(3):
        old.update()
    source = old.payload()
    # This tests exact state loading for the real source age, independently of
    # the much longer native stream. Real archived source256 is checked below.
    for value in source["optimizer"]["state"].values():
        value["step"].fill_(256)
    source["step"], source["stream_cursor"] = 256, 33280
    source["work"].update(training_updates=256, training_ticks=32768)
    return source


def make(arm="control", *, stream=None, source=None, folder=None, device="cpu"):
    source = tiny_source(device) if source is None else source
    model = RemainingTimeGRUAgent(**source["model_config"])
    model.load_state_dict(source["model"])
    return train.Learner(model.to(device), tiny_stream() if stream is None else stream,
        arm, source=source, folder=folder, block=4, updates=3)


class ContinuationTest(unittest.TestCase):
    def test_complete_state_copy_no_warm_no_observation_consumption(self):
        source = tiny_source()
        for arm in train.ARMS:
            learner = make(arm, source=source)
            cp = learner.payload()
            for field in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng"):
                train.assert_exact(cp[field], source[field], field)
            self.assertEqual(cp["stream_cursor"], 0)
            self.assertEqual(cp["step"], 0)
            self.assertEqual(cp["optimizer_state_steps"], [256])
            self.assertEqual(cp["cumulative_optimizer_step"], 256)
            self.assertTrue(all(value == 0 for value in learner.work.values()))
            self.assertTrue(all(p.requires_grad for p in learner.model.parameters()))
            self.assertTrue(all(p.grad is None for p in learner.model.parameters()))
            with self.assertRaisesRegex(RuntimeError, "no warm"):
                learner.warm_start()
            # Initializing/restoring does not invoke either recurrent API.
            with patch.object(learner.model, "forward", side_effect=AssertionError("unexpected warm")), \
                 patch.object(learner.model, "forward_with_time", side_effect=AssertionError("unexpected new obs")):
                copied = train.Learner(learner.model, learner.stream, arm, source=source, block=4, updates=3)
                train.assert_exact(copied.hidden.cpu(), source["hidden"])

    def test_preupdate_numerical_hidden_carried_without_recompute(self):
        learner = make()
        for _ in range(3):
            obs = torch.from_numpy(learner.stream["obs"][learner.cursor:learner.cursor + 4])
            with torch.no_grad():
                _, expected = learner.model(obs, learner.hidden)
                _, reset = learner.model(obs, torch.zeros_like(learner.hidden))
            self.assertFalse(torch.equal(expected, reset))
            old_model = state_digest(learner.model.state_dict())
            learner.update()
            self.assertTrue(torch.equal(learner.hidden, expected))
            self.assertIsNone(learner.hidden.grad_fn)
            self.assertNotEqual(old_model, state_digest(learner.model.state_dict()))
            self.assertEqual(train.optimizer_steps(learner.optimizer), [256 + learner.step])
        self.assertEqual(learner.cursor, 12)
        cp = learner.payload()
        self.assertEqual(cp["cumulative_stream_cursor"], 33292)
        for key in learner.work:
            self.assertEqual(cp["cumulative_work"][key], cp["source_work"][key] + cp["work"][key])

    def test_auxiliary_detached_and_separate_clip_preserves_policy_adam(self):
        source = tiny_source()
        ordinary = make(source=source)
        stream = tiny_stream()
        stream["time_target"][stream["time_mask"]] *= 1e6
        extreme = make(source=source, stream=stream)
        observed = False
        for _ in range(3):
            a, b = ordinary.update(), extreme.update()
            self.assertEqual(a["preclip_grad_norm"]["policy"], b["preclip_grad_norm"]["policy"])
            observed |= a["preclip_grad_norm"]["time_head"] != b["preclip_grad_norm"]["time_head"]
            train.assert_exact(cpu_tree(ordinary.model.core.state_dict()), cpu_tree(extreme.model.core.state_dict()))
            train.assert_exact(cpu_tree(ordinary.model.head.state_dict()), cpu_tree(extreme.model.head.state_dict()))
            train.assert_exact(ordinary.hidden, extreme.hidden)
            for left, right in zip(ordinary.model.policy_parameters(), extreme.model.policy_parameters()):
                train.assert_exact(cpu_tree(ordinary.optimizer.state[left]), cpu_tree(extreme.optimizer.state[right]))
        self.assertTrue(observed)
        self.assertNotEqual(state_digest(ordinary.model.time_head.state_dict()), state_digest(extreme.model.time_head.state_dict()))

    def test_disk_resume_exact_model_moments_hidden_rng_and_work(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = tiny_source(device)
        for arm in train.ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "actual"
                continuous = make(arm, source=source, folder=folder, device=device)
                continuous.save("step000")
                continuous.update(save_audit=True)
                continuous.update(save_audit=True)
                path = folder / "disk.pt"
                torch.save(continuous.payload(), path)
                resumed = train.Learner.restore(path, continuous.stream, device=device)
                continuous.update(); resumed.update()
                for key in ("model", "optimizer", "hidden", "stream_cursor", "step", "work", "source_work", "cumulative_work", "torch_rng", "cuda_rng", "numpy_rng"):
                    train.assert_exact(continuous.payload()[key], resumed.payload()[key], key)
                before = torch.load(folder / "step002_before.pt", map_location="cpu", weights_only=False)
                after = torch.load(folder / "step002_after.pt", map_location="cpu", weights_only=False)
                gradients = torch.load(folder / "step002_gradients.pt", map_location="cpu", weights_only=False)
                self.assertEqual(before["stream_cursor"], 4)
                self.assertEqual(after["stream_cursor"], 8)
                self.assertEqual(before["optimizer_state_steps"], [257])
                self.assertEqual(after["optimizer_state_steps"], [258])
                self.assertTrue(torch.equal(gradients["next_hidden"], after["hidden"]))
                self.assertEqual(set(gradients["preclip_grad_norm"]), {"policy", "time_head"})

    def test_all_nine_real_initial_states_preserve_learned_source(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for seed in train.SEEDS:
            source = train.load_source(seed)
            states = []
            for arm in train.ARMS:
                learner = train.learner_from_source(seed, arm, tiny_stream(batch=32),
                    device=device, block=4, updates=3)
                cp = learner.payload()
                for key in ("model", "optimizer", "hidden", "torch_rng", "cuda_rng", "numpy_rng"):
                    train.assert_exact(cp[key], source[key], f"{seed}.{arm}.{key}")
                self.assertEqual(cp["source_work"], source["work"])
                states.append({key: state_digest(cp[key]) for key in ("model", "optimizer", "hidden")})
                self.assertEqual(sum(p.numel() for p in learner.model.parameters()), 148632)
                self.assertEqual(cp["step"], 0)
            self.assertEqual(states[0], states[1])
            self.assertEqual(states[1], states[2])

    @unittest.skipUnless(torch.cuda.is_available(), "actual continuation runtime is CUDA float32")
    def test_independent_plain_gru_reconstructs_real_source_adam_continuation(self):
        from experiments.coverage_retention.train_audit import reconstruct_update
        train.deterministic(671)
        stream = tiny_stream(ticks=256, batch=32)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "source256"
            learner = train.learner_from_source(0, "control", stream,
                device="cuda", block=128, updates=2, folder=folder)
            for step in (1, 2):
                learner.update(save_audit=True)
                report = reconstruct_update(folder, stream, step, device="cuda")
                self.assertTrue(report["passed"])
                self.assertEqual(report["optimizer_step_before"], 256 + step - 1)
                self.assertEqual(report["optimizer_step_after"], 256 + step)


if __name__ == "__main__":
    unittest.main()
