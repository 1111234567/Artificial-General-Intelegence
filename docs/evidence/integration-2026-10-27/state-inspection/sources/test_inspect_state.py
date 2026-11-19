"""Deterministic synthetic-array checks; no real checkpoint tensors are read."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.integration.inspect_state import (
    StudyNotReady, activity_statistics, inspect_checkpoint, matrix_statistics, phase_statistics,
    process_rng_fingerprints, summarize_records, validate_final_reports,
)


ARMS = ("frozen", "bp64", "repeat64", "replay64", "accum128", "bp128", "coact_live",
        "coact_mix", "coact_mix_shuffled", "coact_only_mix", "coverage_long_dense", "coverage_short_sparse")


def synthetic_world():
    return {
        "spec": dict(batch_size=2, n_jobs=4, active_jobs=4), "tick": 99, "now": 10., "elapsed": .25,
        "acted": True, "event_hash": "0"*64,
        "tensors": {
            "active": torch.tensor([[False, True, True, True], [True, True, False, False]]),
            "started": torch.tensor([[True, False, False, True], [True, True, False, False]]),
            "duration": torch.tensor([[2., 2., 4., 4.], [4., 4., 0., 0.]]),
            "deadline": torch.tensor([[20., 13., 13., 14.], [11., 9., 0., 0.]]),
            "ready_time": torch.tensor([[11., math.inf, math.inf, 12.], [9., 8., math.inf, math.inf]]),
        },
    }


class InspectionTests(unittest.TestCase):
    def test_known_weight_norms(self):
        result = matrix_statistics(torch.diag(torch.tensor([3., 4., 0.])))
        self.assertEqual(result, dict(frobenius_norm=5., spectral_norm=4.))
        self.assertEqual(matrix_statistics(torch.zeros(2, 2)), dict(frobenius_norm=0., spectral_norm=0.))

    def test_zero_and_identical_rows_have_zero_centred_effective_ranks(self):
        for hidden in (torch.zeros(4, 3, dtype=torch.float64),
                       torch.tensor([[.11, -.23, .37]], dtype=torch.float64).repeat(4, 1)):
            with self.subTest(hidden=hidden.tolist()):
                result = activity_statistics(hidden)
                covariance = result["covariance"]
                self.assertEqual(covariance["trace"], 0.)
                self.assertEqual(covariance["participation_ratio"], 0.)
                self.assertEqual(covariance["entropy_effective_rank"], 0.)
                self.assertEqual(covariance["numerical_rank"], 0)
                self.assertEqual(covariance["eigenvalues_descending"], [0.] * 3)
                self.assertEqual(result["within_actor_centred_batch_variance_mean"], 0.)
        zero = activity_statistics(torch.zeros(4, 3))
        self.assertEqual(zero["mean_tanh_derivative"], 1.)
        self.assertEqual(zero["common_mode_energy_fraction"], 0.)
        nonzero = activity_statistics(torch.tensor([[.11, -.23, .37]], dtype=torch.float64).repeat(4, 1))
        self.assertAlmostEqual(nonzero["common_mode_energy_fraction"], 1., places=14)

    def test_diverse_equal_energy_contrasts_have_known_covariance_rank(self):
        rates = .25 * torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]], dtype=torch.float64)
        result = activity_statistics(rates.atanh())
        covariance = result["covariance"]
        torch.testing.assert_close(torch.tensor(covariance["eigenvalues_descending"], dtype=torch.float64),
                                   torch.full((3,), 1/12, dtype=torch.float64), rtol=1e-13, atol=1e-13)
        self.assertEqual(covariance["centred_sample_rank_bound"], 3)
        self.assertEqual(covariance["numerical_rank"], 3)
        self.assertAlmostEqual(covariance["participation_ratio"], 3., places=13)
        self.assertAlmostEqual(covariance["entropy_effective_rank"], 3., places=13)
        self.assertAlmostEqual(result["mean_tanh_derivative"], 15/16, places=14)
        self.assertAlmostEqual(result["common_mode_energy_fraction"], 0., places=14)

    def test_one_direction_and_actor_wide_offsets_remain_distinct(self):
        rates = torch.tensor([[.1, .2, .3], [-.1, -.2, -.3], [.2, .4, .6], [-.2, -.4, -.6]], dtype=torch.float64)
        covariance = activity_statistics(rates.atanh())["covariance"]
        self.assertEqual(covariance["numerical_rank"], 1)
        self.assertAlmostEqual(covariance["trace"], 10*.14/3, places=13)
        self.assertAlmostEqual(covariance["participation_ratio"], 1., places=13)
        self.assertAlmostEqual(covariance["entropy_effective_rank"], 1., places=13)
        offsets = torch.tensor([[0.], [.25], [.5], [.75]], dtype=torch.float64).expand(-1, 3)
        result = activity_statistics(offsets.atanh())
        self.assertEqual(result["within_actor_variance_mean"], 0.)
        self.assertEqual(result["within_actor_centred_batch_variance_mean"], 0.)
        self.assertGreater(result["covariance"]["mean_diagonal_variance"], 0.)
        self.assertEqual(result["covariance"]["numerical_rank"], 1)

    def test_64_actor_128_unit_sample_cannot_exceed_rank_63(self):
        hadamard = torch.ones(1, 1, dtype=torch.float64)
        for _ in range(6):
            hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                                  torch.cat((hadamard, -hadamard), 1)), 0)
        rates = torch.cat((.1*hadamard[:, 1:], torch.zeros(64, 65, dtype=torch.float64)), 1)
        covariance = activity_statistics(rates.atanh())["covariance"]
        self.assertEqual(covariance["centred_sample_rank_bound"], 63)
        self.assertEqual(covariance["numerical_rank"], 63)
        self.assertAlmostEqual(covariance["participation_ratio"], 63., places=10)
        self.assertAlmostEqual(covariance["entropy_effective_rank"], 63., places=10)

    def test_saturation_and_derivative_are_prespecified_pointwise_statistics(self):
        hidden = torch.tensor([[10., -10., 10.], [-10., 10., -10.]], dtype=torch.float64)
        result = activity_statistics(hidden)
        self.assertEqual(result["saturation_threshold"], .95)
        self.assertEqual(result["saturation_fraction"], 1.)
        self.assertLess(result["mean_tanh_derivative"], 1e-7)
        moderate = activity_statistics(torch.zeros(4, 2))
        self.assertEqual(moderate["saturation_fraction"], 0.)

    def test_phase_partition_exposes_inactive_and_raw_started_flags(self):
        world = synthetic_world()
        result = phase_statistics(world)
        self.assertEqual(result["enabled_slots"], 8)
        self.assertEqual(result["active_slots"], 5)
        self.assertEqual(result["raw_started_flag_slots"], 4)
        self.assertEqual(result["active_started_slots"], 3)
        self.assertEqual(result["phase_counts"], dict(inactive=3, active_unstarted_feasible=1,
            active_unstarted_missed_start=1, active_processing=1, active_legal_finish=1, active_past_deadline=1))
        self.assertEqual(result["active_jobs_per_actor_histogram"], [0, 0, 1, 1, 0])
        self.assertEqual(result["active_started_jobs_per_actor_histogram"], [0, 1, 1, 0, 0])
        self.assertAlmostEqual(sum(result["phase_fractions"].values()), 1.)
        world["spec"]["active_jobs"] = 2
        restricted = phase_statistics(world)
        self.assertEqual(restricted["enabled_slots"], 4)
        self.assertEqual(sum(restricted["phase_counts"].values()), 4)
        self.assertEqual(restricted["active_slots"], 3)

    def test_inputs_and_all_cpu_rng_states_are_unchanged(self):
        weight = torch.diag(torch.tensor([3., 4., 0.], dtype=torch.float64, requires_grad=True))
        hidden = torch.tensor([[.1, -.2, .3], [-.4, .5, -.6], [.7, -.8, .9], [-.2, .3, -.4]], dtype=torch.float64, requires_grad=True)
        world = synthetic_world()
        originals = [weight.detach().clone(), hidden.detach().clone(), deepcopy(world)]
        rng_before = process_rng_fingerprints()
        with patch("torch.rand", side_effect=AssertionError("random draw")), patch("torch.randn", side_effect=AssertionError("random draw")), \
             patch("random.random", side_effect=AssertionError("random draw")), patch("numpy.random.random", side_effect=AssertionError("random draw")):
            matrix_statistics(weight)
            activity_statistics(hidden)
            phase_statistics(world)
        self.assertEqual(process_rng_fingerprints(), rng_before)
        torch.testing.assert_close(weight, originals[0], rtol=0, atol=0)
        torch.testing.assert_close(hidden, originals[1], rtol=0, atol=0)
        for name, tensor in world["tensors"].items():
            torch.testing.assert_close(tensor, originals[2]["tensors"][name], rtol=0, atol=0)
        self.assertIsNone(hidden.grad)

    def test_invalid_arrays_and_world_boundary_are_rejected(self):
        with self.assertRaises(ValueError):
            matrix_statistics(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            activity_statistics(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            activity_statistics(torch.full((4, 3), float("nan")))
        world = synthetic_world()
        world["acted"] = False
        with self.assertRaises(ValueError):
            phase_statistics(world)

    def test_complete_final_report_gate_precedes_any_checkpoint_read(self):
        plan = {"runs": dict(seeds=[0, 1, 2], arms=list(ARMS), required_stage_ticks=65536,
                             batch_size=64, width=128, total_parameters=34348)}
        manifest = dict(stage_ticks=65536, batch_size=64, width=128, total_parameters=34348,
                        source_hashes={"continuity": {"run.py": "0"*64}, "integration": {"run.py": "1"*64}}, runs=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(3):
                for arm in ARMS:
                    folder = root / f"seed_{seed:03d}" / arm
                    folder.mkdir(parents=True)
                    config = dict(mode=arm, live_ticks=128 if arm in {"bp128", "accum128"} else 64)
                    report = dict(stage="B", stage_ticks=65536, stage_updates=65536//config["live_ticks"], config={"seed": seed},
                        adaptation_mode=arm, integration_config=config, model=dict(state_scalars=128, total_parameters=34348),
                        hidden_resets=0, world_resets=0, source_hashes=manifest["source_hashes"]["continuity"],
                        integration_source_hashes=manifest["source_hashes"]["integration"])
                    (folder / "checkpoint.pt").write_bytes(b"synthetic placeholder; must never be opened")
                    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
                    manifest["runs"].append(dict(seed=seed, mode=arm, output=str(folder), integration_config=config))
            with patch("torch.load", side_effect=AssertionError("checkpoint read before readiness")):
                self.assertEqual(len(validate_final_reports(manifest, plan)), 36)
                report_path = root / "seed_000" / "frozen" / "report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["stage_ticks"] = 65472
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(StudyNotReady):
                    validate_final_reports(manifest, plan)

    def test_statistical_unit_is_seed(self):
        records = []
        for seed, diagonal in enumerate((1., 2., 3.)):
            records.append(dict(seed=seed, arm="synthetic", weight=matrix_statistics(torch.eye(3)*diagonal),
                                activity=activity_statistics(torch.zeros(4, 3)), world_phase=phase_statistics(synthetic_world())))
        scalar, summary = summarize_records(records)
        self.assertEqual(len(scalar), 3)
        spectral = next(row for row in summary if row["metric"] == "weight/spectral_norm")
        self.assertEqual(spectral["n_seeds"], 3)
        self.assertEqual(spectral["mean"], 2.)
        self.assertEqual(spectral["sample_sd"], 1.)

    def test_raw_checkpoint_tuple_config_matches_json_report_without_mutation(self):
        from experiments.integration.inspect_state import file_sha256
        config = dict(mode="bp64", live_ticks=4, gradient_horizon=4, duration_range=(4., 8.), gap_range=(4., 12.))
        manifest = dict(stage_ticks=4, batch_size=2, width=3, total_parameters=9,
                        source_hashes={"continuity": {"run.py": "0"*64}, "integration": {"run.py": "1"*64}})
        state = dict(format="integration-trainer-v1", integration_config=config, source_hashes=manifest["source_hashes"]["integration"],
            continuation=dict(stage="B", ticks=100, phase_start_tick=96, updates=6, config={"seed": 2}, adaptation_mode="bp64",
                hidden=torch.zeros(2, 3), model={"recurrent": torch.eye(3)}, world=synthetic_world(),
                source_hashes=manifest["source_hashes"]["continuity"]))
        with tempfile.TemporaryDirectory() as directory:
            path, report_path = Path(directory) / "checkpoint.pt", Path(directory) / "report.json"
            torch.save(state, path)
            report = dict(integration_config=json.loads(json.dumps(config)), ticks=100, updates=6)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            before = file_sha256(path)
            rng_before = process_rng_fingerprints()
            result = inspect_checkpoint(dict(checkpoint=path, report=report, report_path=report_path,
                report_sha256=file_sha256(report_path), seed=2, arm="bp64"), manifest)
            self.assertEqual(result["checkpoint_sha256_before"], before)
            self.assertEqual(result["checkpoint_sha256_after"], before)
            self.assertEqual(file_sha256(path), before)
            self.assertEqual(process_rng_fingerprints(), rng_before)


if __name__ == "__main__":
    unittest.main()
