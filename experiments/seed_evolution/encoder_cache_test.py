"""Exercise trace reuse and benchmark one archived CPU candidate on demand."""
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from experiments.learned_binding import common, data
from experiments.seed_evolution import core, device_benchmark
from experiments.seed_evolution.encoder_cache import attach_frozen_encoder_cache


def _assert_bits(actual, expected, path="result"):
    assert type(actual) is type(expected), path
    if isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        left = actual.detach().reshape(-1).contiguous().view(torch.uint8)
        right = expected.detach().reshape(-1).contiguous().view(torch.uint8)
        assert torch.equal(left, right), path
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_bits(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_bits(left, right, f"{path}.{index}")
    elif isinstance(actual, float):
        assert actual.hex() == expected.hex(), path
    else:
        assert actual == expected, path


def _inner_train(source_seed, genome, steps, *, cached):
    cache = None
    original = core.model_from_genome
    with ExitStack() as stack:
        if cached:
            def model_with_cache(*args, **kwargs):
                nonlocal cache
                model = original(*args, **kwargs)
                cache = stack.enter_context(attach_frozen_encoder_cache(model))
                return model

            stack.enter_context(patch.object(core, "model_from_genome", model_with_cache))
        result = core.inner_train(source_seed, genome, steps=steps, checkpoints=(0, steps))
        return result, cache.stats if cache is not None else None


class EncoderCacheTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = nn.Module()
        self.model.encoder = nn.GRU(4, 5).requires_grad_(False)
        self.encoder = self.model.encoder
        self.inputs = torch.randn(6, 2, 4)
        self.hidden = torch.zeros(1, 2, 5)

    def test_hits_for_fresh_views_and_owns_cached_tensors(self):
        before = self.model.state_dict()
        parameter_ids = [id(p) for p in self.model.parameters()]
        expected = self.encoder(self.inputs, self.hidden)
        with attach_frozen_encoder_cache(self.model) as cache:
            first = self.encoder(self.inputs.view_as(self.inputs), self.hidden)
            for tensor in first:
                tensor.fill_(float("nan"))
            again = self.encoder(self.inputs.view_as(self.inputs), torch.zeros_like(self.hidden))
            _assert_bits(again, expected)
            self.assertEqual(cache.stats["hits"], 1)
            self.assertGreater(cache.stats["bytes"], 0)
            for actual, reference in zip(again, expected):
                self.assertEqual(actual.stride(), reference.stride())
                self.assertEqual(actual.requires_grad, reference.requires_grad)
                self.assertIsNone(actual.grad_fn)
            _assert_bits(self.model.state_dict(), before)
            self.assertEqual([id(p) for p in self.model.parameters()], parameter_ids)
        self.assertNotIn("forward", vars(self.encoder))
        self.assertEqual(cache.stats["bytes"], 0)
        self.assertGreater(cache.stats["peak_bytes"], 0)

    def test_versions_storage_offsets_and_strides_distinguish_inputs(self):
        storage = torch.randn(12, 2, 4)
        views = (storage[:6], storage[6:], storage.as_strided((6, 2, 4), (4, 24, 1)))
        with attach_frozen_encoder_cache(self.model) as cache:
            for view in views:
                _assert_bits(self.encoder(view), nn.GRU.forward(self.encoder, view))
            storage.add_(.5)
            _assert_bits(self.encoder(views[0]), nn.GRU.forward(self.encoder, views[0]))
            self.encoder.weight_ih_l0.add_(.25)
            _assert_bits(self.encoder(views[0]), nn.GRU.forward(self.encoder, views[0]))
            self.assertEqual(cache.stats["misses"], 5)
            self.assertEqual(cache.stats["streams"], 1)

    def test_nonzero_hidden_eval_and_gradients_bypass_reuse(self):
        with attach_frozen_encoder_cache(self.model) as cache:
            self.encoder(self.inputs, self.hidden)
            for hidden in (torch.ones_like(self.hidden), -self.hidden):
                _assert_bits(self.encoder(self.inputs, hidden),
                             nn.GRU.forward(self.encoder, self.inputs, hidden))
            self.model.eval()
            self.encoder.train()
            self.encoder(self.inputs, self.hidden)
            self.model.train()
            self.encoder.eval()
            self.encoder(self.inputs, self.hidden)
            self.encoder.train()
            with torch.no_grad():
                self.encoder(self.inputs, self.hidden)
            self.inputs.requires_grad_(True)
            self.encoder(self.inputs, self.hidden)[0].sum().backward()
            self.assertIsNotNone(self.inputs.grad)
            self.inputs.requires_grad_(False)
            self.hidden.requires_grad_(True)
            self.encoder(self.inputs, self.hidden)[0].sum().backward()
            self.assertIsNotNone(self.hidden.grad)
            self.hidden.requires_grad_(False)
            self.encoder.weight_hh_l0.requires_grad_(True)
            self.encoder(self.inputs, self.hidden)[0].sum().backward()
            self.assertIsNotNone(self.encoder.weight_hh_l0.grad)
            self.assertEqual(cache.stats["hits"], 0)
            self.assertEqual(cache.stats["bypasses"], 8)

    def test_independent_candidates_and_bounded_streams(self):
        other = nn.Module()
        other.encoder = nn.GRU(4, 5).requires_grad_(False)
        with attach_frozen_encoder_cache(self.model) as first:
            self.encoder(self.inputs, self.hidden)
            with attach_frozen_encoder_cache(other) as second:
                actual = other.encoder(self.inputs, self.hidden)
                _assert_bits(actual, nn.GRU.forward(other.encoder, self.inputs, self.hidden))
                self.assertEqual(second.stats["hits"], 0)
                self.assertEqual(second.stats["streams"], 1)
            for _ in range(18):
                self.encoder(self.inputs.clone(), self.hidden)
            self.assertEqual(first.stats["streams"], 16)

    def test_inner_train_repeats_keep_model_adam_and_log_bitwise(self):
        fixtures = []
        for index in range(8):
            sample = data.generate(0, batch=3, ticks=32, duration_range=(1., 2.),
                                   gap_range=(1., 2.), stream_id=index)
            fixtures.append((sample["obs"], sample["actions"]))
        genome = core.new_genome(0, 710000)
        with patch.object(core, "_training_streams", return_value=tuple(fixtures)):
            baseline, _ = _inner_train(0, genome, 16, cached=False)
            cached, stats = _inner_train(0, genome, 16, cached=True)
        _assert_bits(cached, baseline)
        self.assertEqual((stats["hits"], stats["misses"], stats["bypasses"]), (8, 8, 0))


def _peak_rss_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("peak_wset", "wset", "peak_paged_pool", "paged_pool", "peak_nonpaged_pool",
                 "nonpaged_pool", "pagefile", "peak_pagefile")
            ]

        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        info.restype = wintypes.BOOL
        counters = Counters()
        counters.size = ctypes.sizeof(counters)
        if info(current_process(), ctypes.byref(counters), counters.size):
            return counters.peak_wset
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _measure(steps, mode, result_path):
    source_seed = device_benchmark.SOURCE_SEED
    genome, archived, training = device_benchmark.reference()
    torch.set_num_threads(1)
    core._training_streams(source_seed)
    peak_before = _peak_rss_bytes()
    started = datetime.now(timezone.utc).isoformat()
    begin = perf_counter()
    result, stats = _inner_train(source_seed, genome, steps, cached=mode == "cache")
    elapsed = perf_counter() - begin
    peak_after = _peak_rss_bytes()
    if steps == 64:
        _assert_bits(result["checkpoints"][64], archived, "archived checkpoint")
        _assert_bits(result["metadata"], training["metadata"], "archived training metadata")
        _assert_bits(result["log"], training["log"], "archived training log")
    torch.save(result, result_path)
    print(json.dumps(dict(steps=steps, mode=mode, started_utc=started,
                          elapsed_seconds=elapsed, updates_per_second=steps / elapsed,
                          peak_rss_before_bytes=peak_before, peak_rss_after_bytes=peak_after,
                          cache=stats, matches_archived_cpu=(True if steps == 64 else None))))


def _benchmark():
    out = core.ROOT / "data/seed-evolution/runtime-2026-11-11/encoder-cache.json"
    _, _, archived = device_benchmark.reference()
    report = dict(
        reference=dict(path=archived["reference_path"], artifacts=archived["artifacts"],
                       archived_cpu_seconds=archived["train_seconds"]),
        environment=dict(platform=platform.platform(), torch=torch.__version__,
                         logical_cpu_count=os.cpu_count(), torch_threads=1, device="cpu",
                         background="Existing large search with 14 CPU workers"),
        timing_scope="One fresh process per trial. Preload verified training streams outside the timer.",
        rss_scope="Process-lifetime peak resident/working set before and after the timed training call.",
        trials=[], pairs=[],
    )
    with TemporaryDirectory() as temp:
        for steps, modes in ((8, ("cache", "baseline")), (16, ("baseline", "cache")),
                             (64, ("cache", "baseline"))):
            results, seconds = {}, {}
            for mode in modes:
                result_path = Path(temp) / f"{mode}-{steps}.pt"
                run = subprocess.run([sys.executable, "-m", __spec__.name, "--measure",
                                      str(steps), mode, str(result_path)],
                                     check=True, stdout=subprocess.PIPE, text=True, cwd=core.ROOT)
                trial = json.loads(run.stdout)
                print(json.dumps(trial), flush=True)
                report["trials"].append(trial)
                results[mode] = device_benchmark.load(result_path)
                seconds[mode] = trial["elapsed_seconds"]
            _assert_bits(results["cache"], results["baseline"], f"matched {steps} updates")
            report["pairs"].append(dict(steps=steps, model_and_adam_bits_match=True,
                                        log_and_metadata_match=True,
                                        speedup=seconds["baseline"] / seconds["cache"]))
    common.write_json(out, report)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--benchmark"]:
        _benchmark()
    elif sys.argv[1:2] == ["--measure"]:
        _measure(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        main()
