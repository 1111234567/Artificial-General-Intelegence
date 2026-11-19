"""Test continuation using temporary files and mocked CE and world scoring."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import torch

from experiments.seed_evolution import large_runner as runner


class RecoveryTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.out = self.root / "run"
        self.enterContext(patch.object(runner, "ROOT", self.root))
        self.enterContext(patch.object(runner, "OUT", self.out))
        self.enterContext(patch.object(runner, "_check", return_value={}))
        self.genomes = self.enterContext(patch.object(runner.core, "new_genome", side_effect=self.fake_genome))
        self.mutations = self.enterContext(patch.object(runner.core, "mutate_genome", side_effect=self.fake_mutation))
        self.train = self.enterContext(patch.object(runner.core, "inner_train", side_effect=self.fake_train))
        self.score = self.enterContext(patch.object(runner.core, "score_model", side_effect=self.fake_score))

    @staticmethod
    def fake_genome(rep, seed):
        return {"weight": torch.tensor([seed, rep], dtype=torch.long)}

    @staticmethod
    def fake_mutation(parent, seed, sigma):
        return {"weight": parent["weight"] + torch.tensor([seed, 0])}

    @staticmethod
    def fake_train(rep, genome, steps, checkpoints, device):
        result = {}
        for update in checkpoints:
            model = {"weight": genome["weight"] + torch.tensor([update, 0])}
            result[update] = dict(model=model, optimizer={}, updates=update, source_seed=rep)
        return dict(checkpoints=result, metadata=dict(genome_hash=runner.core.genome_hash(genome)), log=[])

    @staticmethod
    def fake_score(rep, state, *, archive=False, **options):
        score = dict(source_seed=rep, parameter_hash=runner.base.model_hash(state),
                     fitness=int(state["weight"][0]) % 97 / 100)
        if archive:
            score["archives"] = {options["stream_ids"][0]: {"model": state}}
        return score

    def test_full_budget_and_seed_ranges(self):
        self.assertEqual(runner._unique_budget(), 12096)
        self.assertEqual(runner.CONFIG["population"] * runner.CONFIG["generations"], 2048)
        records = {rep: {} for rep in range(3)}
        pools = {}
        first = runner._plan(0, records, pools)
        self.assertEqual(len(first["candidates"]), 192)
        self.assertTrue(all(row["parent_id"] is None for row in first["candidates"]))
        for entry in first["entries"]:
            rep = entry["rep"]
            for index, cid in enumerate(entry["new_ids"]):
                records[rep][cid] = dict(fitness=index)
            for arm in runner.ARMS:
                pools[rep, arm] = entry["new_ids"]
        last = runner._plan(31, records, pools)
        self.assertEqual(len(last["candidates"]), 384)
        self.assertEqual(last["candidates"][-1]["init_seed"], 733163)
        for desc in last["candidates"]:
            if desc["arm"] == "evolution":
                self.assertIsNone(desc["init_seed"])
                self.assertIn(desc["parent_id"], pools[desc["rep"], "evolution"][-8:])
            else:
                self.assertIsNone(desc["mutation_seed"])
                self.assertIsNone(desc["parent_id"])

    def test_candidate_resumes_after_initial_and_after_ce(self):
        descriptor = runner._plan(0, {rep: {} for rep in range(3)}, {})["candidates"][0]
        folder = runner._candidate_dir(0, descriptor["id"])
        self.train.side_effect = RuntimeError("interrupted before checkpoint")
        with self.assertRaisesRegex(RuntimeError, "before checkpoint"):
            runner._candidate_job(descriptor)
        self.assertTrue((folder / "initial.pt").is_file())
        first_initial = (folder / "initial.pt").read_bytes()
        self.assertFalse((folder / "candidate.json").exists())
        self.train.side_effect = self.fake_train
        self.score.side_effect = RuntimeError("interrupted before score")
        with self.assertRaisesRegex(RuntimeError, "before score"):
            runner._candidate_job(descriptor)
        first_checkpoint = (folder / "inner64.pt").read_bytes()
        self.assertFalse((folder / "candidate.json").exists())

        self.score.side_effect = self.fake_score
        result = runner._candidate_job(descriptor)
        self.assertEqual(result["attempt"], 3)
        self.assertEqual(result["train_attempt"], 2)
        self.assertEqual(result["eval_attempt"], 3)
        self.assertEqual(result["reused_artifacts"], ["initial.pt", "inner64.pt"])
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual((folder / "initial.pt").read_bytes(), first_initial)
        self.assertEqual((folder / "inner64.pt").read_bytes(), first_checkpoint)
        self.assertEqual(runner._candidate_job(descriptor), result)
        self.assertEqual(self.score.call_count, 2)
        (folder / "score.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash differs"):
            runner._candidate_job(descriptor)
        self.assertEqual(self.score.call_count, 2)

    def test_pair_recovery_from_each_publication_boundary(self):
        for data_already_published in (False, True):
            with self.subTest(data_already_published=data_already_published):
                folder = self.out / str(data_already_published)
                data_path = folder / "weights.pt"
                manifest_path = folder / "weights.json"
                pending_data = data_path if data_already_published else runner._pending(data_path)
                runner._save(pending_data, {"weight": torch.zeros(2)})
                manifest = dict(sha256=runner.base.sha(pending_data))
                runner._write(runner._pending(manifest_path), manifest)
                self.assertEqual(runner._recover_pair(data_path, manifest_path, "sha256"), manifest)
                self.assertTrue(data_path.is_file())
                self.assertTrue(manifest_path.is_file())
                self.assertFalse(runner._pending(manifest_path).exists())
                self.assertEqual(runner._recover_pair(data_path, manifest_path, "sha256"), manifest)

    def test_interrupted_generation_keeps_plan_and_skips_completed_candidates(self):
        config = dict(runner.CONFIG, repeats=1, generations=2, population=2, parents=1, source_seeds=[0])
        self.enterContext(patch.object(runner, "CONFIG", config))
        dispatched = []
        fail = True

        def execute(jobs, worker, workers):
            for job in jobs:
                dispatched.append(job["id"])
                if fail and job["id"] == "evolution_g01_i01":
                    raise RuntimeError("interrupted generation")
                yield worker(job)

        self.enterContext(patch.object(runner, "_parallel", side_effect=execute))
        with self.assertRaisesRegex(RuntimeError, "interrupted generation"):
            runner.search(workers=1)
        self.assertEqual(runner.status()["state"], "failed")
        self.assertEqual(runner.status()["completed_candidates"], 3)
        self.assertEqual(runner.base.read(self.out / "failure.json")["phase"], "search")
        self.assertTrue((self.out / "generation_0.json").is_file())
        self.assertFalse((self.out / "generation_1.json").exists())
        self.assertFalse((self.out / "search.json").exists())
        sealed_plan = runner._plan_path(1).read_bytes()
        fail = False
        dispatched.clear()
        summary = runner.search(workers=1)
        self.assertEqual(dispatched, ["evolution_g01_i01", "random_g01_i00", "random_g01_i01"])
        self.assertEqual(sealed_plan, runner._plan_path(1).read_bytes())
        self.assertEqual(summary["unique_candidates"], 6)
        self.assertEqual(len(summary["winners"]), 2)
        self.assertEqual(len(summary["history"]), 4)
        self.assertEqual(runner.status()["state"], "complete")
        self.assertEqual([row["n"] for row in runner.status()["generation_distributions"]], [2, 2, 2])
        dispatched.clear()
        self.assertEqual(runner.search(workers=1), runner.base.read(self.out / "search.json"))
        self.assertEqual(dispatched, [])

    def test_final_and_heldout_partial_pairs_resume(self):
        descriptor = runner._plan(0, {rep: {} for rep in range(3)}, {})["candidates"][0]
        runner._candidate_job(descriptor)
        winner = dict(rep=0, arm="evolution", id=descriptor["id"])
        frozen = runner._train_final(winner)
        self.assertEqual(runner._train_final(winner), frozen)
        self.assertEqual(self.train.call_count, 2)

        def interrupt(rep, state, **options):
            if options.get("teacher_support"):
                raise RuntimeError("interrupted held-out conditions")
            return self.fake_score(rep, state, **options)

        self.score.side_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "interrupted held-out conditions"):
            runner._test_final(frozen)
        dest = runner._final_dir(winner) / "evaluations"
        committed = {path.name: path.read_bytes() for path in dest.glob("*_own.json")}
        self.assertEqual(len(committed), 3)
        self.score.reset_mock(side_effect=True)
        self.score.side_effect = self.fake_score
        report = runner._test_final(frozen)
        self.assertEqual(len(report["conditions"]), 6)
        self.assertEqual(self.score.call_count, 3)
        self.assertEqual(committed, {path.name: path.read_bytes() for path in dest.glob("*_own.json")})
        self.assertEqual(runner._test_final(frozen), report)
        self.assertEqual(self.score.call_count, 3)


if __name__ == "__main__":
    main()
