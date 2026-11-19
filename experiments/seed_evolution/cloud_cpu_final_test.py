"""Check that terminal CE continues the committed state and Adam counters."""
from contextlib import nullcontext

import torch

from experiments.seed_evolution import audit, cloud_cpu_final, core, large_runner as runner


def test_final_continues_the_archived_model_and_adam(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "run")
    monkeypatch.setattr(cloud_cpu_final.LazyTrainingStreams, "install", lambda self: nullcontext())
    generator = torch.Generator().manual_seed(819207)
    streams = [(torch.randn(2, 1, 56, generator=generator),
                torch.randint(11, (2, 1, 2), generator=generator)) for _ in range(16)]
    monkeypatch.setattr(core, "_training_streams", lambda seed: streams)

    seed, cid = 0, "shared_g00_i00"
    winner = dict(rep=seed, arm="evolution", id=cid)
    candidate = runner._candidate_dir(seed, cid)
    genome = core.new_genome(seed, 710000)
    prefix = core.inner_train(seed, genome, steps=64)
    origin = prefix["checkpoints"][64]
    execution = dict(backend="hybrid", amendment_sha="historical-amendment")
    runner._save(candidate / "initial.pt", genome)
    source = runner._commit_pair(candidate / "inner64.pt", candidate / "training.json", origin,
                                 dict(metadata=dict(prefix["metadata"], device="cpu+cuda_executor",
                                                    execution=execution)), "checkpoint_sha256")
    artifacts = {name: runner.base.sha(candidate / name) for name in
                 ("initial.pt", "inner64.pt", "training.json")}
    runner._write(candidate / "candidate.json",
                  dict(rep=seed, id=cid, artifacts=artifacts, genome_hash=core.genome_hash(genome),
                       trained_hash=runner.base.model_hash(origin["model"])))

    result = cloud_cpu_final.train_final(winner)
    trained = runner._load(runner.ROOT / result["path"])
    checkpoints, metadata = trained["checkpoints"], trained["metadata"]
    assert set(checkpoints) == {0, 64, 65, 511, 512}
    audit.same_tree(checkpoints[0], prefix["checkpoints"][0], "reconstructed initialization")
    audit.same_tree(checkpoints[64], origin, "archived search state")
    assert metadata["origin64_sha256"] == source["checkpoint_sha256"]
    assert metadata["origin64_training_sha256"] == artifacts["training.json"]
    assert metadata["origin64_execution"] == execution
    assert metadata["origin64_device"] == "cpu+cuda_executor"
    assert metadata["device"] == "cpu"
    assert metadata["fresh_optimizer"] is False
    assert metadata["continuation_start_update"] == 65
    assert metadata["continuation_updates"] == 448
    assert metadata["stream_indices"] == [audit.historical_stream_at(step) for step in range(64, 512)]
    log = {row["update"]: row for row in trained["log"]}
    assert set(log) == {65, 128, 192, 256, 320, 384, 448, 511, 512}
    for step in (64, 511):
        obs, actions = streams[audit.historical_stream_at(step)]
        replay = audit.replay_checkpoint_update(checkpoints[step], checkpoints[step + 1],
                                                dict(obs=obs, actions=actions))
        assert log[step + 1]["loss"] == replay["loss"]
        assert log[step + 1]["grad_norm"] == replay["grad_norm"]
    assert runner.base.sha(candidate / "inner64.pt") == artifacts["inner64.pt"]
    monkeypatch.setattr(core, "_training_streams", None)
    assert cloud_cpu_final.train_final(winner) == result
