"""Exercise execution-only routing and recovery without a formal search."""
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

from experiments.seed_evolution import core, execution_runtime as runtime, large_runner as runner
from experiments.seed_evolution.lazy_training_streams import LazyTrainingStreams


class ExecutionRuntimeTests(TestCase):
    def setUp(self):
        self.out = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(runtime, "OUT", self.out))
        self.enterContext(patch.object(runner, "OUT", self.out))
        self.amendment = self.out / runtime.AMENDMENT
        protocol = self.out / "protocol.json"
        runner._write(protocol, {"sealed": True})
        self.actual = {
            "cpu": dict(backend="cpu", dtype="float32", amp=False, tf32=False),
            "hybrid": dict(backend="hybrid", dtype="float32", amp=False, tf32=False),
        }
        templates = {
            mode: dict(meta, lazy_streams=True, lazy_source_sha="lazy", backend_source_sha="backend")
            for mode, meta in self.actual.items()
        }
        self.spec = dict(format="seed-evolution-execution-v1", sources={},
                         original_protocol_sha256=runtime.sha(protocol), start_generation=4,
                         hybrid_indices=list(range(8)), backend_metadata=templates,
                         prefix_workers=20, cpu_workers=18, gpu_workers=2)
        runner._write(self.amendment, self.spec)
        load = runtime.load_amendment
        self.enterContext(patch.object(runtime, "load_amendment", side_effect=lambda: load(self.out)))

    def _trained(self, mode):
        backend = "hybrid" if mode == "hybrid" else "cpu"
        device = "cpu+cuda_executor" if mode == "hybrid" else "cpu"
        return dict(metadata=dict(device=device, steps=64, execution=deepcopy(self.actual[backend])),
                    cache_metrics=dict(hits=56), checkpoints={}, log=[])

    def test_metadata_matches_template_and_preserves_legacy_cpu_keys(self):
        prefix = runtime._execution(self._trained("legacy_cpu"), "legacy_cpu", self.spec)
        self.assertEqual(prefix["metadata"], dict(device="cpu", steps=64))
        for mode in ("cpu", "hybrid"):
            with self.subTest(mode=mode):
                original = self._trained(mode)
                device = original["metadata"]["device"]
                result = runtime._execution(original, mode, self.spec)
                expected = dict(self.spec["backend_metadata"][mode],
                                amendment_sha=runtime.sha(self.amendment))
                self.assertEqual(result["metadata"], dict(device=device, steps=64, execution=expected))
        wrong = self._trained("hybrid")
        wrong["metadata"]["execution"]["tf32"] = True
        with self.assertRaisesRegex(AssertionError, "actual backend environment"):
            runtime._execution(wrong, "hybrid", self.spec)

    def test_dual_pool_routes_corresponding_arms_and_preserves_all_jobs(self):
        pools = []

        class Pool:
            def __init__(self, max_workers, initializer, initargs):
                self.max_workers = max_workers
                self.initializer = initializer
                self.initargs = initargs
                self.submissions = []
                self.closed = False
                pools.append(self)

            def submit(self, fn, job, role, mode):
                self.submissions.append((fn, job, role, mode))
                future = Future()
                future.set_result((job, role, mode))
                return future

            def shutdown(self, *, wait, cancel_futures):
                self.closed = wait and cancel_futures

        suffix = [dict(rep=rep, id=f"{arm}_g04_i{index:02d}")
                  for rep in (0, 1) for arm in ("evolution", "random") for index in range(12)]
        prefix = [dict(rep=0, id=f"{arm}_g03_i00") for arm in ("evolution", "random")]
        with patch.object(runtime, "ProcessPoolExecutor", Pool):
            result = list(runtime._parallel(suffix, runner._candidate_job, 20))
            legacy = list(runtime._parallel(prefix, runner._candidate_job, 20))
            final = list(runtime._parallel([suffix[0]], runner._train_final, 6))
            tests = list(runtime._parallel([dict(winner=suffix[0])], runner._test_final, 6))

        self.assertEqual([pool.max_workers for pool in pools], [18, 2, 20, 2, 6])
        self.assertTrue(all(pool.closed for pool in pools))
        self.assertTrue(all(pool.initializer is runtime._init_worker and pool.initargs == (self.spec,)
                            for pool in pools))
        self.assertCountEqual([job for job, _, _ in result], suffix)
        routes = {(job["rep"], job["id"]): mode for job, role, mode in result if role == "candidate"}
        for rep in (0, 1):
            for index in range(12):
                expected = "hybrid" if index < 8 else "cpu"
                self.assertEqual(routes[rep, f"evolution_g04_i{index:02d}"], expected)
                self.assertEqual(routes[rep, f"random_g04_i{index:02d}"], expected)
        for pool, expected in zip(pools, ("cpu", "hybrid", "legacy_cpu", "hybrid", "legacy_cpu")):
            self.assertTrue(all(fn is runtime._worker and mode == expected
                                for fn, _, _, mode in pool.submissions))
        self.assertCountEqual(legacy, [(job, "candidate", "legacy_cpu") for job in prefix])
        self.assertEqual(final, [(suffix[0], "final", "hybrid")])
        self.assertEqual(tests, [(dict(winner=suffix[0]), "test", "legacy_cpu")])

    def test_worker_calls_captured_trainer_and_restores_patches(self):
        job = dict(rep=0, id="evolution_g04_i00")
        location = runner._candidate_dir(0, job["id"])
        store = LazyTrainingStreams()
        original = Mock(return_value=self._trained("hybrid"))
        previous_trainer = core.inner_train
        previous_loader = core._training_streams
        previous_committed = runner._committed_candidate
        genome = object()

        def backend(seed, initial, steps, checkpoints, *, backend, cache):
            self.assertEqual((backend, cache), ("hybrid", True))
            self.assertIs(core.inner_train, original)
            self.assertIs(core._training_streams.__self__, store)
            return core.inner_train(seed, initial, steps, checkpoints, device="cpu")

        def candidate(descriptor):
            self.assertIsNot(core.inner_train, original)
            self.assertIs(runner._committed_candidate, runtime._committed)
            self.assertIs(core._training_streams, previous_loader)
            result = core.inner_train(descriptor["rep"], genome)
            runner._write(location / "training.json", dict(metadata=result["metadata"]))
            return descriptor

        with (patch.object(runtime, "_SPEC", self.spec), patch.object(runtime, "_STORE", store),
              patch.object(runtime, "_ORIGINAL_TRAIN", original),
              patch.object(runtime.hybrid_backend, "train", side_effect=backend) as hybrid,
              patch.object(runner, "_candidate_job", side_effect=candidate)):
            self.assertEqual(runtime._worker(job, "candidate", "hybrid"), job)
            original.assert_called_once_with(0, genome, 64, (0, 64), device="cpu")
            hybrid.side_effect = RuntimeError("trainer failed")
            with self.assertRaisesRegex(RuntimeError, "trainer failed"):
                runtime._worker(job, "candidate", "hybrid")
        self.assertIs(core.inner_train, previous_trainer)
        self.assertIs(core._training_streams, previous_loader)
        self.assertIs(runner._committed_candidate, previous_committed)
        execution = runtime.read(location / "execution.json")
        self.assertEqual(execution["backend"], "hybrid")
        self.assertEqual(execution["training_calls"][0]["cache"], {"hits": 56})

    def test_resume_rejects_wrong_backend_before_candidate_or_final_work(self):
        job = dict(rep=0, arm="evolution", id="evolution_g04_i00")
        correct = runtime._execution(self._trained("hybrid"), "hybrid", self.spec)
        wrong = runtime._execution(self._trained("cpu"), "cpu", self.spec)
        for role, folder, worker in (("candidate", runner._candidate_dir(0, job["id"]), "_candidate_job"),
                                     ("final", runner._final_dir(job), "_train_final")):
            path = folder / "training.json"
            with self.subTest(role=role), patch.object(runtime, "_SPEC", self.spec):
                runner._write(runner._pending(path), dict(metadata=wrong["metadata"]))
                with patch.object(runner, worker) as job_worker:
                    with self.assertRaisesRegex(AssertionError, "resumed training backend"):
                        runtime._worker(job, role, "hybrid")
                    job_worker.assert_not_called()
                runner._write(path, dict(metadata=correct["metadata"]))
                runtime._validate_training(path, "hybrid", self.spec)
                correct_backend_wrong_device = dict(correct["metadata"], device="cpu")
                runner._write(path, dict(metadata=correct_backend_wrong_device))
                with self.assertRaisesRegex(ValueError, "resumed device differs"):
                    runtime._validate_training(path, "hybrid", self.spec)
        with patch.object(runtime, "_SPEC", self.spec):
            with self.assertRaisesRegex(ValueError, "task backend assignment changed"):
                runtime._worker(job, "candidate", "cpu")


if __name__ == "__main__":
    main()
