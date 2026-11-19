"""The frozen observer and revisitation orchestration must not perturb training."""
from copy import deepcopy
import json

import torch

from experiments.continuity.run import state_digest
from experiments.integration.test_run import old_trainer
from experiments.integration.diagnose import diagnose_fork
from experiments.credit.run import CreditTrainer, CreditConfig
from experiments.credit.evaluate import evaluate_trainer
from experiments.credit.cycle import run_cycles
from experiments.credit.launch import launch


def test_observer_is_identical_and_preserves_complete_local_continuation():
    torch.set_num_threads(1)
    trainer = CreditTrainer(old_trainer(), CreditConfig(mode="diag_sym_replay", chunk_ticks=8))
    trainer.run_updates(1)
    before = state_digest(trainer.semantic_state())
    direct = diagnose_fork(trainer.model, trainer.world, trainer.h, condition="long_sparse", steps=16)
    wrapped = evaluate_trainer(trainer, conditions=("long_sparse",), steps=16,
                               interventions=("finish_content_oracle", "timing_oracle"))
    assert wrapped["records"][0] == direct
    assert state_digest(trainer.semantic_state()) == before


def test_cycle_matches_uninterrupted_manual_schedule_and_resume_is_idempotent(tmp_path):
    torch.set_num_threads(1)
    trainer = CreditTrainer(old_trainer(), CreditConfig(mode="diag_sym_replay", chunk_ticks=8))
    trainer.run_updates(1)
    source = tmp_path/"source.pt"
    trainer.save_checkpoint(source)
    protocol = {"cycles": dict(modes=["diag_sym_replay"], seeds=[173], start_stage_ticks=8,
        sequence=["A", "B"], segment_ticks=16, milestones=[8,16],
        regimes={"A": [[1,4],[.5,2]], "B": [[4,8],[4,12]]},
        eval_conditions=["iid", "long_sparse"], eval_steps=8)}
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    out = tmp_path/"cycles"
    run_cycles(source, out, path, "cpu")
    result = json.loads((out/"complete.json").read_text())
    assert len(result["completed"]) == 4
    actual = CreditTrainer.from_checkpoint(result["completed"][-1]["checkpoint"], device="cpu")
    expected = CreditTrainer.from_checkpoint(source, device="cpu")
    for index,name in enumerate(("A", "B")):
        duration,gap=protocol["cycles"]["regimes"][name]
        expected.change_regime(duration,gap,f"revisit/{index:02d}/{name}")
        expected.run_updates(2)
    assert state_digest(actual.semantic_state()) == state_digest(expected.semantic_state())
    run_cycles(source, out, path, "cpu")
    resumed=json.loads((out/"complete.json").read_text())
    assert resumed["completed"] == result["completed"]


def test_manifest_launcher_resumes_complete_state_and_preserves_attempt_logs(tmp_path):
    torch.set_num_threads(1)
    source=tmp_path/"source.pt"
    old_trainer().save_checkpoint(source)
    out=tmp_path/"branch"
    launch(source,out,"diag_sym",64,"cpu",0)
    checkpoint=out/"checkpoint.pt"
    expected=CreditTrainer.from_checkpoint(checkpoint,device="cpu")
    expected.run_updates(1)
    expected_hash=state_digest(expected.semantic_state())
    launch(source,out,"diag_sym",128,"cpu",0)
    actual=CreditTrainer.from_checkpoint(checkpoint,device="cpu")
    assert state_digest(actual.semantic_state()) == expected_hash
    assert (out/"training_attempt_001.jsonl").exists()
    assert len((out/"launches.jsonl").read_text().splitlines()) == 2
