"""The new experiment changes stream identity, never the frozen learner recipe."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.remaining_time import train as previous
from experiments.remaining_time.train_test import make
from experiments.teacher_timing import train


class TeacherTrainingTest(unittest.TestCase):
    def test_native_training_functions_are_original_objects(self):
        for name in ("Learner", "initialize", "original_checkpoint", "validate_stream",
                     "summarize_rows", "stream_identity", "OPTIMIZER"):
            self.assertIs(getattr(train, name), getattr(previous, name), name)
        self.assertEqual(train.OBJECTIVES, ("action", "joint"))
        self.assertEqual((train.WARM, train.TRAIN_STEPS, train.BLOCK, train.BATCH, train.UPDATES),
                         (512, 32768, 128, 32, 256))

    def test_teacher_objective_snapshot_paths_are_unambiguous(self):
        paths = {train.checkpoint_path(seed, teacher, objective, step)
            for seed in train.SEEDS for teacher in train.TEACHERS
            for objective in train.OBJECTIVES for step in train.SNAPSHOTS}
        self.assertEqual(len(paths), 36)
        self.assertEqual(train.checkpoint_path(1, "midpoint", "joint", 64),
            train.OUT / "midpoint" / "seed_001" / "joint" / "step064.pt")
        for args in ((3, "earliest", "action", 0), (0, "latest", "action", 0),
                     (0, "earliest", "probe", 0), (0, "earliest", "action", 1)):
            with self.assertRaises(AssertionError):
                train.checkpoint_path(*args)

    def test_stream_dispatch_uses_actual_teacher_data(self):
        expected = object()
        with patch.object(train.data, "load_stream", return_value=expected) as mocked:
            self.assertIs(train.load_stream(2, "midpoint"), expected)
            mocked.assert_called_once_with(2, "midpoint")

    def test_exact_checker_detects_dtype_shape_keys_and_small_tensor_changes(self):
        original = dict(tensor=torch.tensor([1., 2.]), state=[0, (1, 2)])
        train.assert_exact(original, deepcopy(original))
        variants = [dict(original, extra=1), dict(original, tensor=original["tensor"].double()),
                    dict(original, tensor=original["tensor"][None]),
                    dict(original, state=[0, [1, 2]])]
        tiny = deepcopy(original)
        tiny["tensor"][0] = torch.nextafter(tiny["tensor"][0], torch.tensor(float("inf")))
        variants.append(tiny)
        for variant in variants:
            with self.assertRaises(AssertionError):
                train.assert_exact(original, variant)

    def test_historical_comparison_requires_all_numerical_fields_exact(self):
        learner = make("action")
        learner.warm_start(); learner.update(); learner.update()
        original = learner.payload()
        self.assertTrue(original["optimizer"]["state"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(train, "OUT", root / "new"), patch.object(previous, "OUT", root / "old"):
                new_path = train.checkpoint_path(0, "earliest", "action", 64)
                old_path = previous.checkpoint_path(0, "action", 64)
                for path in (new_path, old_path):
                    path.parent.mkdir(parents=True)
                torch.save(original, old_path)
                metadata_only = deepcopy(original)
                metadata_only["protocol_hash"] = "different-new-protocol"
                metadata_only["teacher"] = "earliest"
                torch.save(metadata_only, new_path)
                result = train.compare_historical(0, "action", 64)
                self.assertTrue(result["passed"])
                self.assertEqual(set(result["fields"]), set(train.REGRESSION_FIELDS))
                for key in train.REGRESSION_FIELDS:
                    changed = deepcopy(metadata_only)
                    if key == "model":
                        changed[key]["head.bias"][0] += .001
                    elif key == "optimizer":
                        first = next(iter(changed[key]["state"].values()))
                        first["exp_avg"].view(-1)[0] += .001
                    elif key == "hidden":
                        changed[key][0, 0] += .001
                    elif key == "work":
                        changed[key]["action_rows"] += 1
                    else:
                        changed[key] += 1
                    torch.save(changed, new_path)
                    with self.assertRaises(AssertionError, msg=key):
                        train.compare_historical(0, "action", 64)


if __name__ == "__main__":
    unittest.main()
