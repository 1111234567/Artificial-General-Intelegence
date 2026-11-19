"""Real finite launcher/evaluator and foreign continuation refusal checks."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import Trainer, TrainConfig, Variant, atomic_json
from experiments.credit.run import file_sha256
from experiments.rewiring import study


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("rewiring-launch")
    with _preserve_process_state():
        base = Trainer(Variant("rewiring-launch-fixture", width=128, policy="online"),
            TrainConfig(seed=71, batch_size=2, n_jobs=4, tbptt=64, updates=2, device="cpu",
                train_dts=(.25,), replay_capacity=2, replay_block_ticks=128, replay_burnin=64))
        base.run_updates(2)
        source = root / "A.pt"
        base.save_checkpoint(source)
    output = root / "training"
    item = dict(id="tiny", seed=71, condition_id="learned_low", arm="learned", shrink_step=.0002,
                structural_seed=47, source=str(source), source_sha256=file_sha256(source), output=str(output))
    plan = dict(format="rewiring-study-protocol-v1", sources={str(Path(study.__file__)):file_sha256(study.__file__)},
        branches=[item], phase_ticks=64, target_ticks=128, window_updates=1, chunk_ticks=64,
        training_device="cpu", evaluation_device="cpu", evaluation_steps=24, dt=.25, conditions=["iid"],
        regimes={"B":{"duration":[4.,8.],"gap":[4.,12.]},"A2":{"duration":[1.,4.],"gap":[.5,2.]}})
    protocol = root / "protocol.json"
    atomic_json(protocol, plan)
    study.train(protocol, "tiny")
    return root, protocol, plan, item


def test_real_launcher_two_phases_and_side_effect_free_evaluation(tiny):
    root, protocol, plan, item = tiny
    from experiments.rewiring.run import RewireTrainer
    output = root / "evaluation.json"
    with _preserve_process_state():
        b = RewireTrainer.from_checkpoint(Path(item["output"])/"checkpoint_B.pt", device="cpu")
        a = RewireTrainer.from_checkpoint(Path(item["output"])/"checkpoint_A2.pt", device="cpu")
        assert b.stage_ticks == 64 and not b.phase_returned
        assert a.stage_ticks == 128 and a.phase_returned
        assert a.hidden_resets == b.hidden_resets == 0
    study.evaluate(protocol, "tiny", Path(item["output"])/"checkpoint.pt", output)
    result = json.loads(output.read_text())
    assert result["source_unchanged"] and len(result["records"]) == 2
    assert {r["learning_phase"] for r in result["records"]} == {"B", "A2"}
    with pytest.raises(ValueError, match="existing"):
        study.evaluate(protocol, "tiny", Path(item["output"])/"checkpoint.pt", output)


def test_completed_launch_is_idempotent(tiny, monkeypatch):
    from experiments.rewiring import run
    monkeypatch.setattr(run, "main", lambda _: pytest.fail("completed training was rerun"))
    study.train(tiny[1], "tiny")


@pytest.mark.parametrize("field", ["seed", "arm", "shrink_step", "structural_seed", "source_sha256"])
def test_wrong_branch_continuation_refused(tiny, field):
    from experiments.rewiring.run import RewireTrainer
    _, _, plan, item = tiny
    wrong = deepcopy(item)
    wrong[field] = {"seed":72,"arm":"random","shrink_step":.1,"structural_seed":48,"source_sha256":"0"*64}[field]
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(Path(item["output"])/"checkpoint_A2.pt", device="cpu")
        with pytest.raises(ValueError):
            study.validate(trainer, plan, wrong, 128)


@pytest.mark.parametrize("field", ["target", "live_regime", "phase_flag", "resets", "future_schedule"])
def test_incomplete_or_wrong_live_state_refused(tiny, field):
    from experiments.rewiring.run import RewireTrainer
    _, _, plan, item = tiny
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(Path(item["output"])/"checkpoint_A2.pt", device="cpu")
        target = 128
        if field == "target": target = 64
        elif field == "live_regime": trainer.world.set_regime(duration_range=(9.,10.))
        elif field == "phase_flag": trainer.phase_returned = False
        elif field == "resets": trainer.base.hidden_resets = 1
        elif field == "future_schedule": trainer.return_regime["duration_range"] = (9.,10.)
        with pytest.raises(ValueError):
            study.validate(trainer, plan, item, target)


def test_foreign_complete_marker_refused_before_evaluation(tiny, tmp_path):
    marker = Path(tiny[3]["output"])/"launch-complete.json"
    original = marker.read_bytes()
    bad = json.loads(original)
    bad["branch"]["seed"] += 1
    try:
        atomic_json(marker, bad)
        with pytest.raises(ValueError, match="identity"):
            study.evaluate(tiny[1], "tiny", Path(tiny[3]["output"])/"checkpoint.pt", tmp_path/"bad.json")
        assert not (tmp_path/"bad.json").exists()
    finally:
        marker.write_bytes(original)
