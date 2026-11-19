"""Verify lazy source isolation and replay the archived candidate on demand."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest import TestCase, main
from unittest.mock import patch

import torch

from experiments.learned_binding import common
from experiments.seed_evolution import core, device_benchmark
from experiments.seed_evolution.encoder_cache_test import _assert_bits, _inner_train, _peak_rss_bytes
from experiments.seed_evolution.lazy_training_streams import LazyTrainingStreams


_OUTPUT = core.ROOT / "data/seed-evolution/runtime-2026-11-11/lazy-training-streams.json"


class LazyTrainingStreamsTests(TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.root = Path(stack.enter_context(TemporaryDirectory()))
        self.manifest = self.root / "data-freeze.json"
        self.rows = []
        stack.enter_context(patch.object(core, "ROOT", self.root))
        stack.enter_context(patch.object(core, "DATA_FREEZE", self.manifest))
        stack.enter_context(patch.dict(common.SETTINGS, train_ticks=4, train_batch=2))
        for seed in (0, 1):
            for index in range(16):
                path = self.root / f"{seed}-{index}.pt"
                torch.save(dict(obs=torch.full((4, 2, 56), seed * 100. + index),
                                actions=torch.zeros(4, 2, 2, dtype=torch.long),
                                metadata=dict(source_seed=seed, stream_id=index, split="train")), path)
                self.rows.append(dict(seed=seed, stream=index, path=path.name, sha256=common.sha(path)))
        common.write_json(self.manifest, dict(streams=list(reversed(self.rows))))

    def test_dense_schedule_loads_eight_then_wide_access_loads_remaining(self):
        store = LazyTrainingStreams()
        with store.install():
            streams = core._training_streams(0)
            self.assertEqual(store.stats["loaded_streams"], 0)
            for update in range(64):
                index = core._stream_index(update)
                self.assertEqual(streams[index][0][0, 0, 0].item(), index)
            self.assertEqual(store.stats["loaded_indices"], list(range(8)))
            first = streams[0]
        with store.install():
            streams = core._training_streams(0)
            self.assertIs(streams[0], first)
            self.assertEqual(store.stats["file_reads"], 8)
            for update in range(256):
                streams[core._stream_index(update)]
            self.assertEqual(store.stats["loaded_indices"], list(range(16)))
            self.assertEqual(store.stats["file_reads"], 16)

    def test_sources_are_isolated_and_switches_drop_retained_tensors(self):
        store = LazyTrainingStreams()
        with store.install():
            old = core._training_streams(0)
            self.assertEqual(old[0][0][0, 0, 0].item(), 0.)
            old[1]
            with self.assertRaisesRegex(RuntimeError, "within a candidate"):
                core._training_streams(1)
        with store.install():
            streams = core._training_streams(1)
            self.assertEqual(store.stats["loaded_streams"], 0)
            self.assertEqual(streams[0][0][0, 0, 0].item(), 100.)
            self.assertEqual(store.stats["loaded_indices"], [0])
            with self.assertRaisesRegex(RuntimeError, "within their candidate"):
                old[0]
        with store.install():
            self.assertEqual(core._training_streams(0)[0][0][0, 0, 0].item(), 0.)
            self.assertEqual(store.stats["file_reads"], 4)

    def test_validation_is_deferred_to_the_requested_file(self):
        path = self.root / self.rows[8]["path"]
        path.write_bytes(b"invalid")
        store = LazyTrainingStreams()
        with store.install():
            streams = core._training_streams(0)
            streams[0]
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                streams[8]
            self.assertEqual(store.stats["loaded_indices"], [0])

    def test_valid_hash_does_not_hide_incompatible_tensors_or_identity(self):
        payload = torch.load(self.root / self.rows[0]["path"], weights_only=False)
        cases = (
            (dict(obs=payload["obs"].double()), "incompatible"),
            (dict(actions=payload["actions"][..., :1]), "incompatible"),
            (dict(metadata=dict(source_seed=1, stream_id=0, split="train")), "identity"),
        )
        for changes, error in cases:
            with self.subTest(error=error):
                path = self.root / self.rows[0]["path"]
                torch.save({**payload, **changes}, path)
                self.rows[0]["sha256"] = common.sha(path)
                common.write_json(self.manifest, dict(streams=self.rows))
                store = LazyTrainingStreams()
                with store.install(), self.assertRaisesRegex(ValueError, error):
                    core._training_streams(0)[0]

    def test_manifest_and_scope_restore_on_error(self):
        store = LazyTrainingStreams()
        common.write_json(self.manifest, dict(streams=self.rows + [self.rows[0]]))
        original = core._training_streams
        with self.assertRaisesRegex(ValueError, "duplicate"), store.install():
            with self.assertRaisesRegex(RuntimeError, "already installed"), store.install():
                pass
            core._training_streams(0)
        self.assertIs(core._training_streams, original)


def _benchmark():
    genome, archived, training = device_benchmark.reference()
    torch.set_num_threads(1)
    store = LazyTrainingStreams()
    before = _peak_rss_bytes()
    started = datetime.now(timezone.utc).isoformat()
    begin = perf_counter()
    with store.install():
        result, encoder = _inner_train(0, genome, 64, cached=True)
    elapsed = perf_counter() - begin
    after = _peak_rss_bytes()
    _assert_bits(result["checkpoints"][64], archived, "archived checkpoint")
    _assert_bits(result["metadata"], training["metadata"], "archived training metadata")
    _assert_bits(result["log"], training["log"], "archived training log")
    assert store.stats["loaded_indices"] == list(range(8))
    assert store.stats["file_reads"] == 8 and encoder["hits"] == 56
    assert not core._training_streams.cache_info().currsize
    root = core.ROOT / "data/seed-evolution/runtime-2026-11-11"
    eager_benchmark = device_benchmark.read(root / "encoder-cache.json")
    eager = next(row for row in eager_benchmark["trials"]
                 if row["steps"] == 64 and row["mode"] == "cache")
    stream_bytes = store.stats["bytes"] // store.stats["loaded_streams"]
    progress = device_benchmark.large_runner.status()
    report = dict(started_utc=started, elapsed_seconds=elapsed, torch=str(torch.__version__),
                  device="cpu", torch_threads=1,
                  latest_search_progress_utc=progress.get("updated_utc"),
                  archive=dict(path=training["reference_path"], artifacts=training["artifacts"]),
                  checkpoint_and_log_bits_match=True, training_streams=store.stats, encoder_cache=encoder,
                  peak_rss_before_bytes=before, peak_rss_after_bytes=after,
                  earlier_eager_encoder_cache_64=dict(
                      loaded_streams=common.SETTINGS["training_streams"],
                      stream_bytes=common.SETTINGS["training_streams"] * stream_bytes,
                      **eager,
                  ),
                  timing_scope="The lazy measurement includes cold stream verification and loading. "
                               "The earlier eager measurement preloaded streams outside the timer.",
                  rss_scope="Process-lifetime peak resident/working set. The two samples are from "
                            "separate process launches with uncontrolled background load.",
                  estimated_retained_stream_bytes_3_eager_sources=(
                      len(common.SETTINGS["seeds"]) * common.SETTINGS["training_streams"] * stream_bytes))
    common.write_json(_OUTPUT, report)
    print(report, flush=True)


def _measure_stream_cache(mode):
    torch.set_num_threads(1)
    store = LazyTrainingStreams()
    retained = 0
    samples = []
    before = _peak_rss_bytes()
    begin = perf_counter()
    indices = sorted({core._stream_index(step) for step in range(64)})
    for seed in common.SETTINGS["seeds"]:
        if mode == "eager":
            streams = core._training_streams(seed)
            retained += sum(tensor.numel() * tensor.element_size()
                            for pair in streams for tensor in pair)
            loaded = len(streams)
        else:
            with store.install():
                streams = core._training_streams(seed)
                for index in indices:
                    streams[index]
                retained = store.stats["bytes"]
                loaded = store.stats["loaded_streams"]
        samples.append(dict(source_seed=seed, streams_loaded_for_source=loaded,
                            total_retained_stream_bytes=retained, peak_rss_bytes=_peak_rss_bytes()))
    print(json.dumps(dict(mode=mode, elapsed_seconds=perf_counter() - begin,
                          peak_rss_before_bytes=before, sources=samples)))


def _compare_stream_caches():
    report = device_benchmark.read(_OUTPUT)
    samples = []
    for mode in ("eager", "lazy"):
        run = subprocess.run([sys.executable, "-m", __spec__.name, "--measure-cache", mode],
                             check=True, stdout=subprocess.PIPE, text=True, cwd=core.ROOT)
        samples.append(json.loads(run.stdout))
    report["three_source_cache_probe"] = dict(
        scope="Data-only probe with no training. Each mode used a separate fresh process.",
        samples=samples,
    )
    common.write_json(_OUTPUT, report)
    print(json.dumps(report["three_source_cache_probe"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--benchmark"]:
        _benchmark()
    elif sys.argv[1:] == ["--compare-stream-caches"]:
        _compare_stream_caches()
    elif sys.argv[1:2] == ["--measure-cache"]:
        _measure_stream_cache(sys.argv[2])
    else:
        main()
