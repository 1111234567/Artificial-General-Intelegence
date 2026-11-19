"""Check execution allocation and independent hybrid CE replay on small streams."""
from copy import deepcopy
import json

import pytest
import torch

from experiments.learned_binding.audit import AuditError
from experiments.seed_evolution import audit, core, execution_audit, hybrid_backend, large_audit


@pytest.fixture
def training_inputs(monkeypatch):
    generator = torch.Generator().manual_seed(819207)
    streams = tuple((torch.randn(6, 2, 56, generator=generator),
                     torch.randint(11, (6, 2, 2), generator=generator)) for _ in range(2))
    monkeypatch.setattr(core, "_training_streams", lambda _seed: streams)
    return core.new_genome(0, 710000), streams


def test_execution_allocation_is_paired_and_excludes_started_generations():
    execution = execution_audit.Execution(4, tuple(range(8)), {}, "amendment")
    assert execution.backend("shared_g00_i00") == "legacy_cpu"
    assert execution.backend("evolution_g03_i07") == "legacy_cpu"
    for arm in ("evolution", "random"):
        assert execution.backend(f"{arm}_g04_i00") == "hybrid"
        assert execution.backend(f"{arm}_g31_i07") == "hybrid"
        assert execution.backend(f"{arm}_g04_i08") == "cpu"
        assert execution.backend(f"{arm}_g04_i63") == "cpu"
    with pytest.raises(AuditError, match="invalid shared candidate identity"):
        execution.backend("shared_g04_i00")


def test_execution_training_report_checks_all_fields(training_inputs):
    genome, _ = training_inputs
    executor = audit.common.executor(0).state_dict()
    trained = hybrid_backend.train(0, genome, 2, (0, 2), backend="cpu", cache=True)
    metadata = trained["metadata"]
    expected = dict(metadata["execution"], amendment_sha="sealed")
    metadata["execution"] = deepcopy(expected)
    execution_audit.audit_training_report(trained, 0, 2, genome, executor, expected_execution=expected)

    metadata["execution"]["amendment_sha"] = "altered"
    with pytest.raises(AuditError, match="amendment_sha"):
        execution_audit.audit_training_report(trained, 0, 2, genome, executor, expected_execution=expected)
    metadata["execution"] = deepcopy(expected)
    metadata["device"] = "cpu+cuda_executor"
    with pytest.raises(AuditError, match="device"):
        execution_audit.audit_training_report(trained, 0, 2, genome, executor, expected_execution=expected)
    metadata["device"] = "cpu"
    metadata["stream_indices"] = [0, 0]
    with pytest.raises(AuditError, match="stream_indices"):
        execution_audit.audit_training_report(trained, 0, 2, genome, executor, expected_execution=expected)


def test_metadata_mode_routes_training_report_by_plan_and_restores_hooks(tmp_path, monkeypatch):
    templates = {backend: {"backend": backend} for backend in ("cpu", "hybrid")}
    execution = execution_audit.Execution(4, tuple(range(8)), templates, "sealed")
    monkeypatch.setattr(execution_audit, "_load_execution", lambda _root: execution)
    expected = {"shared_g00_i00": None, "evolution_g04_i00": dict(backend="hybrid", amendment_sha="sealed"),
                "random_g04_i10": dict(backend="cpu", amendment_sha="sealed")}
    original_candidate = audit.audit_candidate

    def training_report(training, *_args, expected_execution):
        assert expected_execution == expected[training]

    def candidate(_root, record, *_):
        audit.audit_training_report(record["id"])
        return []

    def frozen_audit(root, _workers, *, metadata_only):
        assert metadata_only
        for cid in expected:
            audit.audit_candidate(root, dict(rep=0, id=cid, arm=cid.split("_")[0]), None, None, {})
        return dict(candidates_provenance_checked=3, full_audit_passed=False)

    monkeypatch.setattr(execution_audit, "audit_training_report", training_report)
    monkeypatch.setattr(execution_audit, "_BASE_CANDIDATE", candidate)
    monkeypatch.setattr(large_audit, "audit_all", frozen_audit)
    result = execution_audit.audit_all(tmp_path, metadata_only=True)
    assert result["legacy_cpu_candidates"] == 1
    assert result["suffix_backend_counts"]["0"] == {"evolution": {"cpu": 0, "hybrid": 1},
                                                    "random": {"cpu": 1, "hybrid": 0}}
    assert result["execution_amendment_sha256"] == "sealed"
    assert audit.audit_candidate is original_candidate


def test_runtime_amendment_accepts_renamed_prefix_manifest(tmp_path, training_inputs):
    from experiments.seed_evolution import execution_runtime, lazy_training_streams

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    plan = tmp_path / "plans/generation_00.json"
    plan.parent.mkdir()
    plan.write_text("{}", encoding="utf-8")
    folder = audit.candidate_directory(tmp_path, 0, "shared_g00_i00")
    folder.mkdir(parents=True)
    training = folder / "training.json"
    training.write_text("{}", encoding="utf-8")
    genome, _ = training_inputs
    trained = hybrid_backend.train(0, genome, 1, (0, 1), backend="cpu", cache=True)
    sha = audit.common.sha
    cpu = dict(trained["metadata"]["execution"], backend_source_sha=sha(hybrid_backend.__file__),
               lazy_streams=True, lazy_source_sha=sha(lazy_training_streams.__file__))
    hybrid = dict(cpu, backend="hybrid", device_topology=dict(cpu["device_topology"], executor="cuda:0"))
    spec = dict(format="seed-evolution-execution-v1", start_generation=1, hybrid_indices=list(range(8)),
                original_protocol_sha256=sha(protocol), sources={}, cache_encoder=True, lazy_training=True,
                issued_plans={"plans/generation_00.json": sha(plan)},
                existing_prefix_inventory={"search/rep_0/shared_g00_i00/.training.json.pending": sha(training)},
                backend_metadata=dict(cpu=cpu, hybrid=hybrid))
    amendment = tmp_path / execution_runtime.AMENDMENT
    amendment.write_text(json.dumps(spec), encoding="utf-8")
    execution = execution_audit._load_execution(tmp_path)
    assert execution.expected_metadata("cpu") == dict(cpu, amendment_sha=sha(amendment))
    training.write_text("{\"updated\": true}", encoding="utf-8")
    with pytest.raises(AuditError, match="artifact SHA-256 changed"):
        execution_audit._load_execution(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA independent replay requires a GPU")
def test_hybrid_ce_replay_matches_production_model_and_adam(training_inputs, tmp_path, monkeypatch):
    genome, streams = training_inputs
    trained = hybrid_backend.train(0, genome, 2, (0, 1, 2), backend="hybrid", cache=True)
    metadata = trained["metadata"]["execution"]
    execution = execution_audit.Execution(4, tuple(range(8)), {"hybrid": metadata}, "sealed")
    checkpoints = trained["checkpoints"]
    with (execution_audit._cuda_environment(execution),
          execution_audit._override(audit, checkpoint_model=execution_audit._hybrid_checkpoint_model,
                                    independent_ce_update=execution_audit._hybrid_update)):
        for step, (obs, actions) in enumerate(streams):
            metrics = audit.replay_checkpoint_update(checkpoints[step], checkpoints[step + 1],
                                                     dict(obs=obs, actions=actions))
            assert metrics["model_and_adam_bitwise_equal"]
            assert metrics["loss"] == trained["log"][step]["loss"]
            assert metrics["grad_norm"] == trained["log"][step]["grad_norm"]

    cid = "random_g04_i00"
    folder = audit.candidate_directory(tmp_path, 0, cid)
    folder.mkdir(parents=True)
    torch.save(genome, folder / "initial.pt")
    torch.save(checkpoints[2], folder / "inner64.pt")
    score = dict(fitness=.125, streams=[])
    (folder / "score.json").write_text(json.dumps(score), encoding="utf-8")
    monkeypatch.setattr(execution_audit, "_load_execution", lambda _root: execution)
    monkeypatch.setattr(audit, "load_source_streams", lambda *_: {
        step: dict(obs=obs, actions=actions) for step, (obs, actions) in enumerate(streams)})
    monkeypatch.setattr(core, "score_model", lambda *_args, **_kwargs: dict(score, archives={}))
    config = dict(inner_steps=2, dev_streams=[], actors=2, support_ticks=6, probe_ticks=6, random_seed=0)
    replay = execution_audit.replay_candidate((tmp_path, 0, cid, {}, config, True))
    assert replay["execution_backend"] == "hybrid"
    assert replay["independent_ce_updates"] == 2
