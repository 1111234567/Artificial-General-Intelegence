"""Tiny full-checkpoint audit regressions; these are not formal study runs."""
from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.credit.run import CreditConfig, CreditTrainer, file_sha256, load_checkpoint
from experiments.integration.diagnose import diagnose_fork
from experiments.loops import audit_run
from experiments.loops.interventions import INTERVENTIONS


@pytest.fixture
def tiny_audit(tmp_path):
    """Real saved Adam/replay/clock histories, with only a few synthetic updates."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with _preserve_process_state():
        config = TrainConfig(seed=173, batch_size=2, n_jobs=2, tbptt=4, updates=4,
                             device="cpu", train_dts=(.25,), replay_capacity=2,
                             replay_block_ticks=8, replay_burnin=4, log_every=4)
        trainer = Trainer(Variant("loops-fixture", width=6, policy="online"), config)
        trainer.run_updates(4)
        assert trainer.optimizer.state and trainer.replay.blocks
        source = tmp_path / "frozen-source.py"
        source.write_text("# synthetic protocol input\n", encoding="utf-8")
        checkpoints = []
        path = tmp_path / "A.pt"
        trainer.save_checkpoint(path)
        checkpoints.append(dict(id="A", path=str(path), sha256=file_sha256(path)))
        for mode in ("bp64", "bp_replay"):
            branch = CreditTrainer.from_base_checkpoint(
                path, CreditConfig(mode=mode, chunk_ticks=4))
            branch.run_updates(1)
            checkpoint = tmp_path / f"{mode}.pt"
            branch.save_checkpoint(checkpoint)
            checkpoints.append(dict(id=mode, path=str(checkpoint), sha256=file_sha256(checkpoint)))
        protocol = tmp_path / "protocol.json"
        plan = dict(format="loops-existing-audit-protocol-v1", checkpoints=checkpoints,
                    sources={str(source): file_sha256(source)}, mask_seeds=[11, 29, 47],
                    conditions=["iid", "long_sparse", "longer"], steps=96, dt=.25,
                    state_samples=2)
        protocol.write_text(json.dumps(plan), encoding="utf-8")
    try:
        yield protocol, plan, source
    finally:
        torch.set_num_threads(threads)


def _components(trainer):
    """Check actual loaded source objects as well as the semantic-state guard."""
    return {"model": state_digest(trainer.model.state_dict()),
            "optimizer": state_digest(trainer.optimizer.state_dict()),
            "hidden": state_digest(trainer.h),
            "world": state_digest(trainer.world.state_dict()),
            "replay": state_digest(trainer.replay.state_dict()),
            "training": trainer.model.training,
            "grad_flags": tuple(parameter.requires_grad for parameter in trainer.model.parameters()),
            "gradients": state_digest({name: parameter.grad
                                        for name, parameter in trainer.model.named_parameters()})}


@pytest.mark.parametrize("source_id", ["A", "bp64", "bp_replay"])
def test_full_audit_preserves_source_and_matches_original_observer(
        tiny_audit, tmp_path, monkeypatch, source_id):
    protocol, plan, source = tiny_audit
    checkpoint = next(row for row in plan["checkpoints"] if row["id"] == source_id)
    checkpoint_bytes = Path(checkpoint["path"]).read_bytes()
    source_bytes, protocol_bytes = source.read_bytes(), protocol.read_bytes()
    loaded = []

    def track_load(*args, **kwargs):
        trainer = load_checkpoint(*args, **kwargs)
        # Gradients are not serialized; attach one to detect live-source changes
        # that would be invisible to semantic_state's checkpoint representation.
        trainer.model.recurrent.grad = torch.full_like(trainer.model.recurrent, .375)
        loaded.append((trainer, _components(trainer)))
        return trainer

    monkeypatch.setattr(audit_run, "load_checkpoint", track_load)
    caller_rng = state_digest(rng_state())
    result = audit_run.run_job(protocol, source_id, 11, tmp_path / f"{source_id}-first.json")
    assert caller_rng == state_digest(rng_state())
    assert result["source_semantic_hash_before"] == result["source_semantic_hash_after"]
    assert result["checkpoint_bytes_unchanged"]
    assert result["protocol_sha256"] == file_sha256(protocol)
    assert result["source"] == checkpoint
    assert len(loaded) == 1
    trainer, before = loaded[0]
    assert _components(trainer) == before
    assert trainer.optimizer.state and trainer.replay.blocks
    assert result["structure"] is not None
    assert result["source_files"] == plan["sources"]
    assert len(result["records"]) == 8 * len(plan["conditions"])

    for condition in plan["conditions"]:
        rows = [row for row in result["records"] if row["condition"] == condition]
        assert tuple(row["graph_intervention"] for row in rows) == INTERVENTIONS
        baseline = diagnose_fork(trainer.model, trainer.world, trainer.h,
                                 condition=condition, steps=plan["steps"], dt=plan["dt"], device="cpu")
        assert baseline["primary"]["matured"] > 0
        recorded = {key: value for key, value in rows[0].items()
                    if key not in {"graph_intervention", "mask_seed", "graph_parameter_hash", "graph_action_policy"}}
        assert recorded == baseline
        assert rows[0]["graph_parameter_hash"] == baseline["parameter_hash"]
        assert rows[0]["graph_action_policy"] == "unmodified_model"
        for row in rows:
            assert row["action_hash"] == row["model_action_hash"]
            assert row["hidden_resets"] == row["world_resets"] == row["parameter_updates"] == 0
            assert row["opportunities"]["model_action_changes"] == 0
            for key in ("start_hidden_hash", "start_world_state_hash", "start_scheduler_hash",
                        "end_scheduler_hash", "end_event_hash", "primary_eligible_issued"):
                assert row[key] == baseline[key]
            for cohort in ("primary", "inherited", "carryover_contract"):
                assert row[cohort]["request_fingerprint"] == baseline[cohort]["request_fingerprint"]
                assert row[cohort]["per_actor"]["matured"] == baseline[cohort]["per_actor"]["matured"]
            if row["graph_intervention"] != "none":
                assert row["graph_action_policy"] == "parameter_intervention_counterfactual"

    assert Path(checkpoint["path"]).read_bytes() == checkpoint_bytes
    assert source.read_bytes() == source_bytes and protocol.read_bytes() == protocol_bytes
    assert _components(trainer) == before
    assert caller_rng == state_digest(rng_state())
    saved = json.loads((tmp_path / f"{source_id}-first.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(result, allow_nan=False))


def test_later_mask_draws_have_six_variants_and_matching_exogenous_cohorts(tiny_audit, tmp_path):
    protocol, plan, _ = tiny_audit
    results = [audit_run.run_job(protocol, "A", seed, tmp_path / f"mask-{seed}.json")
               for seed in plan["mask_seeds"]]
    expected_names = tuple(name for name in INTERVENTIONS if name not in ("none", "diagonal_only"))
    assert [len(result["records"]) for result in results] == [24, 18, 18]
    assert sum(len(result["records"]) for result in results) == 60
    assert all(result["structure"] is None for result in results[1:])
    for result in results[1:]:
        assert Counter(row["condition"] for row in result["records"]) == Counter(
            {condition: 6 for condition in plan["conditions"]})
        for condition in plan["conditions"]:
            rows = [row for row in result["records"] if row["condition"] == condition]
            assert tuple(row["graph_intervention"] for row in rows) == expected_names
            baseline = next(row for row in results[0]["records"]
                            if row["condition"] == condition and row["graph_intervention"] == "none")
            for row in rows:
                for cohort in ("primary", "inherited", "carryover_contract"):
                    assert row[cohort]["request_fingerprint"] == baseline[cohort]["request_fingerprint"]
                    assert row[cohort]["matured"] == baseline[cohort]["matured"]


@pytest.mark.parametrize("target", ["source", "checkpoint"])
def test_modified_frozen_input_is_rejected_before_loading(tiny_audit, tmp_path, monkeypatch, target):
    protocol, plan, source = tiny_audit
    path = source if target == "source" else tmp_path / "A.pt"
    path.write_bytes(path.read_bytes() + b"\nchanged")

    def no_load(*args, **kwargs):
        pytest.fail("tampered inputs must be rejected before loading")

    monkeypatch.setattr(audit_run, "load_checkpoint", no_load)
    output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="frozen input/source changed|historical checkpoint changed"):
        audit_run.run_job(protocol, "A", plan["mask_seeds"][0], output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["protocol", "source", "checkpoint"])
def test_midrun_tampering_rejects_publication_and_restores_rng(
        tiny_audit, tmp_path, monkeypatch, target):
    protocol, plan, source = tiny_audit
    plan["steps"] = 1
    protocol.write_text(json.dumps(plan), encoding="utf-8")
    real_diagnose = audit_run.diagnose_fork
    modified = False

    def tamper(*args, **kwargs):
        nonlocal modified
        record = real_diagnose(*args, **kwargs)
        if not modified:
            path = {"protocol": protocol, "source": source, "checkpoint": tmp_path / "A.pt"}[target]
            path.write_bytes(path.read_bytes() + b"\n")
            modified = True
        return record

    monkeypatch.setattr(audit_run, "diagnose_fork", tamper)
    before_rng = state_digest(rng_state())
    output = tmp_path / "rejected.json"
    with pytest.raises((AssertionError, ValueError), match="changed"):
        audit_run.run_job(protocol, "A", plan["mask_seeds"][1], output)
    assert modified and not output.exists()
    assert state_digest(rng_state()) == before_rng
