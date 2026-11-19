"""Finite curriculum boundaries and private permutation contracts."""
from copy import deepcopy
import random
import unittest

from experiments.lifelong.schedule import CHECKPOINT_TICKS, Schedule, ScheduleConfig


class ScheduleTests(unittest.TestCase):
    def test_default_finite_phases_and_checkpoints(self):
        schedule = Schedule("mixed", seed=11)
        self.assertEqual(schedule.total_ticks, 172032)
        self.assertEqual(schedule.return_start, 163840)
        self.assertEqual(schedule.mode_for_tick(0)["phase"], "main")
        self.assertEqual(schedule.mode_for_tick(131071)["future_family_allowed"], (0, 1, 2, 3))
        self.assertEqual(schedule.mode_for_tick(131072)["phase"], "withdrawal")
        self.assertEqual(schedule.mode_for_tick(163839)["future_family_allowed"], (1, 2, 3))
        self.assertEqual(schedule.mode_for_tick(163840)["phase"], "return")
        self.assertEqual(schedule.mode_for_tick(172031)["future_family_allowed"], (0,))
        self.assertEqual(schedule.mode_for_tick(172032)["phase"], "complete")
        self.assertTrue(all(tick % 128 == 0 for tick in CHECKPOINT_TICKS))
        self.assertEqual(CHECKPOINT_TICKS[-1], schedule.total_ticks)

    def test_blocked_exact_nominal_exposure_and_rotation(self):
        schedule = Schedule("blocked", seed=91)
        exposure = {family: 0 for family in range(4)}
        for tick in range(0, schedule.main_ticks, 128):
            mode = schedule.mode_for_tick(tick)
            family, = mode["future_family_allowed"]
            exposure[family] += 128
            cycle = tick // 32768
            block = (tick % 32768) // 8192
            self.assertEqual(family, schedule.order_for_cycle(cycle)[block])
        self.assertEqual(exposure, dict.fromkeys(range(4), 32768))
        self.assertEqual({schedule.order_for_cycle(cycle)[-1] for cycle in range(4)}, set(range(4)))
        self.assertEqual({schedule.order_for_cycle(cycle)[0] for cycle in range(4)}, set(range(4)))

    def test_instruction_changes_align_with_boundaries(self):
        schedule = Schedule("blocked", seed=8)
        boundaries = schedule.boundaries()
        self.assertTrue(all(tick % 128 == 0 for tick in boundaries))
        self.assertEqual(boundaries, tuple(range(0, 131072, 8192)) + (131072, 163840, 172032))
        for tick in boundaries[1:]:
            left, right = schedule.mode_for_tick(tick - 1), schedule.mode_for_tick(tick)
            self.assertNotEqual((left["name"], left["phase"]), (right["name"], right["phase"]))
        self.assertEqual(Schedule("mixed").boundaries(), (0, 131072, 163840, 172032))

    def test_seed_is_private_and_lookup_order_independent(self):
        state = random.getstate()
        first, second = Schedule("blocked", 29), Schedule("blocked", 29)
        forward = [first.mode_for_tick(tick) for tick in range(0, first.total_ticks + 1, 128)]
        backward = [second.mode_for_tick(tick) for tick in reversed(range(0, second.total_ticks + 1, 128))]
        self.assertEqual(forward, list(reversed(backward)))
        self.assertEqual(random.getstate(), state)
        orders = {Schedule("blocked", seed).initial_permutation for seed in range(40)}
        self.assertGreater(len(orders), 12)

    def test_scaled_schedule_preserves_four_rotated_cycles(self):
        config = ScheduleConfig(mode="blocked", seed=4, main_ticks=64, macrocycle_ticks=16,
                                block_ticks=4, withdrawal_ticks=16, return_ticks=8, chunk_ticks=4)
        schedule = Schedule(config)
        self.assertEqual(schedule.total_ticks, 88)
        self.assertEqual(schedule.mode_for_tick(64)["phase"], "withdrawal")
        self.assertEqual(schedule.mode_for_tick(80)["future_family_allowed"], (0,))
        self.assertEqual(schedule.boundaries(), tuple(range(0, 64, 4)) + (64, 80, 88))

    def test_state_roundtrip_checks_seeded_permutation(self):
        schedule = Schedule("blocked", -15)
        state = schedule.state_dict()
        restored = Schedule.from_state_dict(state)
        self.assertEqual(schedule.summary(), restored.summary())
        for tick in schedule.boundaries():
            self.assertEqual(schedule.mode_for_tick(tick), restored.mode_for_tick(tick))
        altered = deepcopy(state)
        altered["initial_permutation"] = tuple(reversed(altered["initial_permutation"]))
        with self.assertRaisesRegex(ValueError, "permutation"):
            Schedule.from_state_dict(altered)

    def test_invalid_config_and_tick_rejected(self):
        for changes in ({"mode": "unknown"}, {"seed": True}, {"main_ticks": 131073},
                        {"block_ticks": 4096}, {"chunk_ticks": 129}, {"return_ticks": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ScheduleConfig(**changes)
        schedule = Schedule()
        for tick in (-1, True, 0.5, schedule.total_ticks + 1):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                schedule.mode_for_tick(tick)

    def test_checkpoint_rejects_missing_and_unknown_schema_fields(self):
        saved = Schedule("blocked", 2).state_dict()
        for key in saved:
            state = deepcopy(saved)
            state.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                Schedule.from_state_dict(state)
        state = deepcopy(saved)
        state["unexpected"] = 1
        with self.assertRaises(ValueError):
            Schedule.from_state_dict(state)
        for key in saved["config"]:
            state = deepcopy(saved)
            state["config"].pop(key)
            with self.subTest(missing_config=key), self.assertRaises(ValueError):
                Schedule.from_state_dict(state)
        state = deepcopy(saved)
        state["config"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            Schedule.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
