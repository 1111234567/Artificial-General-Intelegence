"""Verify the hybrid trainer and benchmark it against three frozen sources."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import torch

from experiments.learned_binding import data
from experiments.seed_evolution import core, device_benchmark, hybrid_backend, large_runner
from experiments.seed_evolution.encoder_cache_test import _assert_bits


class HybridBackendTest(unittest.TestCase):
    def test_cpu_cache_preserves_checkpoints_log_and_original_metadata(self):
        source = data.generate(0, batch=3, ticks=32, duration_range=(1., 2.),
                               gap_range=(1., 2.), stream_id=0)
        streams = ((source["obs"], source["actions"]),) * 16
        genome = core.new_genome(0, 710000)
        original_factory = core.model_from_genome
        with patch.object(core, "_training_streams", return_value=streams):
            expected = core.inner_train(0, genome, steps=9, checkpoints=(0, 9))
            for cache in (False, True):
                with self.subTest(cache=cache):
                    result = hybrid_backend.train(0, genome, 9, (0, 9), cache=cache)
                    execution = result["metadata"].pop("execution")
                    self.assertEqual(execution["device_topology"],
                                     dict(policy="cpu", optimizer="cpu", executor="cpu"))
                    stats = result.pop("cache_metrics")
                    if cache:
                        self.assertEqual((stats["hits"], stats["misses"], stats["bypasses"]), (8, 1, 0))
                    else:
                        self.assertIsNone(stats)
                    _assert_bits(result, expected)
                    self.assertIs(core.model_from_genome, original_factory)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_exception_restores_cuda_flags_env_and_factory(self):
        previous = _flags()
        original_factory = core.model_from_genome

        def fail(*args, **kwargs):
            self.assertNotEqual(core.model_from_genome, original_factory)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.enabled)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertFalse(torch.is_autocast_enabled("cuda"))
            raise RuntimeError("deliberate failure")

        with patch.object(core, "inner_train", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                hybrid_backend.train(0, {}, 0, (), backend="hybrid")
        self.assertIs(core.model_from_genome, original_factory)
        self.assertEqual(_flags(), previous)


def _flags():
    return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def _genome(source_seed):
    path = large_runner.OUT / f"search/rep_{source_seed}/shared_g00_i00"
    row = device_benchmark.read(path / "candidate.json")
    genome = device_benchmark.load(path / "initial.pt")
    assert row["source_seed"] == source_seed
    assert row["genome_hash"] == core.genome_hash(genome)
    _assert_bits(core.new_genome(source_seed, row["init_seed"]), genome)
    return genome, row["init_seed"]


def _measure(source_seed, genome, steps, checkpoints, *, backend, cache):
    torch.set_num_threads(1)
    core._training_streams(source_seed)
    previous_flags = _flags()
    start = time.perf_counter()
    result = hybrid_backend.train(source_seed, genome, steps, checkpoints, backend=backend, cache=cache)
    elapsed = time.perf_counter() - start
    assert _flags() == previous_flags
    stats = dict(backend=backend, cache=cache, steps=steps, elapsed_seconds=elapsed,
                 cache_metrics=result["cache_metrics"], pid=os.getpid())
    if backend == "hybrid":
        stats.update(torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / device_benchmark.MIB,
                     torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / device_benchmark.MIB)
    return result, stats


def _compare_three_sources():
    for source_seed in range(3):
        genome, init_seed = _genome(source_seed)
        options = dict(source_seed=source_seed, genome=genome, steps=8, checkpoints=(0, 8))
        cpu, cpu_timing = _measure(**options, backend="cpu", cache=False)
        hybrid, hybrid_timing = _measure(**options, backend="hybrid", cache=False)
        _assert_bits(hybrid["checkpoints"][0], cpu["checkpoints"][0])
        delta = device_benchmark.compare_checkpoints(
            cpu["checkpoints"][8], hybrid["checkpoints"][8], cpu["log"], hybrid["log"])
        assert delta["model"]["nonzero"] == delta["trainable"]["nonzero"]
        assert delta["step"]["nonzero"] == 0
        assert delta["config_matches"] and delta["optimizer_groups_match"] and delta["updates_match"]
        assert delta["trainable"]["max_abs"] < 1e-4, delta["trainable"]["max_abs"]
        print(json.dumps(dict(event="cpu_hybrid_comparison", source_seed=source_seed, init_seed=init_seed,
            cpu=cpu_timing, hybrid=hybrid_timing,
            differences={name: {k: v for k, v in delta[name].items() if k != "by_tensor"}
                         for name in ("trainable", "exp_avg", "exp_avg_sq")},
            logs_cuda_minus_cpu=delta["log_cuda_minus_cpu"])), flush=True)


def _measure_spawned(steps, cache, result_path):
    genome, _ = _genome(0)
    checkpoints = (0, 64) if steps == 64 else (0, 1, 64, 65)
    result, stats = _measure(0, genome, steps, checkpoints, backend="hybrid", cache=cache)
    torch.save(result, result_path)
    print(json.dumps(stats), flush=True)


def _verify_fresh_processes():
    results, stats = {}, {}
    with TemporaryDirectory() as directory:
        for label, steps, cache in (("uncached_a", 64, False), ("cached", 64, True),
                                    ("uncached_b", 64, False), ("longer_prefix", 65, True)):
            result_path = Path(directory) / f"{label}.pt"
            proc = subprocess.run([sys.executable, "-B", "-m", __spec__.name, "--measure",
                                   str(steps), str(int(cache)), str(result_path)], check=True,
                                  capture_output=False, stdout=subprocess.PIPE, text=True, cwd=core.ROOT)
            results[label] = device_benchmark.load(result_path)
            stats[label] = json.loads(proc.stdout)
            print(json.dumps(dict(event="hybrid_trial", label=label, **stats[label])), flush=True)
    first = results["uncached_a"]
    second = results["uncached_b"]
    _assert_bits(first, second, "independent uncached hybrid64 processes")
    cached = results["cached"]
    _assert_bits(first["checkpoints"], cached["checkpoints"], "hybrid64 cache off/on checkpoints")
    _assert_bits(first["log"], cached["log"], "hybrid64 cache off/on logs")
    longer = results["longer_prefix"]
    _assert_bits(cached["checkpoints"][64], longer["checkpoints"][64], "hybrid64 prefix of 65 updates")
    _assert_bits(cached["log"], longer["log"][:2], "hybrid64 log prefix of 65 updates")
    uncached_time = (stats["uncached_a"]["elapsed_seconds"] + stats["uncached_b"]["elapsed_seconds"]) / 2
    cached_time = stats["cached"]["elapsed_seconds"]
    print(json.dumps(dict(event="verification", independent_hybrid64_bits_match=True,
                          cache_off_on_hybrid64_bits_match=True, hybrid65_prefix64_bits_match=True,
                          uncached_mean_seconds=uncached_time, cached_seconds=cached_time,
                          cache_speedup=uncached_time / cached_time,
                          measured_faster_cache_setting=cached_time < uncached_time)), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--benchmark"]:
        _compare_three_sources()
        _verify_fresh_processes()
    elif sys.argv[1:2] == ["--measure"]:
        _measure_spawned(int(sys.argv[2]), bool(int(sys.argv[3])), Path(sys.argv[4]))
    else:
        unittest.main()
