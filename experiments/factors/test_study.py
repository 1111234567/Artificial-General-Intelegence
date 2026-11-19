from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import TrainConfig,Trainer,Variant,atomic_json
from experiments.credit.run import file_sha256
from experiments.rewiring.run import RewireTrainer
from experiments.rewiring.plasticity import RewireConfig
from experiments.factors import study


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp("factor-study")
    source=study.ROOT/"data/rewiring/main-v1/training/seed_000/retire_high/checkpoint_B.pt"
    out=root/"training"
    item=dict(id="tiny",seed=0,mode="grow",idle_lambda=.1,shrink_step=.0002,source=str(source),source_sha256=file_sha256(source),output=str(out))
    origin=dict(id="source",seed=0,checkpoint=str(source),checkpoint_sha256=file_sha256(source))
    plan=dict(format="factors-study-protocol-v1",sources={str(Path(study.__file__)):file_sha256(study.__file__)},branches=[item],origins=[origin],target_ticks=128,chunk_ticks=64,source_stage_ticks=65536,training_device="cpu",evaluation_device="cpu",evaluation_steps=24,dt=.25,conditions=["iid"],return_regime=dict(duration=[1.,4.],gap=[.5,2.],name="rewiring-A2"))
    protocol=root/"protocol.json";atomic_json(protocol,plan)
    study.train(protocol,"tiny")
    return root,protocol,plan,item


def test_actual_training_source_and_endpoint_evaluation(tiny):
    root,protocol,plan,item=tiny
    study.baseline(protocol,"source",root/"baseline.json")
    study.evaluate(protocol,"tiny",Path(item["output"])/"checkpoint.pt",root/"evaluation.json")
    for name in ["baseline","evaluation"]:
        value=json.loads((root/f"{name}.json").read_text())
        assert value["source_unchanged"] and len(value["records"])==1
    with pytest.raises(ValueError,match="existing"):
        study.evaluate(protocol,"tiny",Path(item["output"])/"checkpoint.pt",root/"evaluation.json")


def test_completed_run_skips_training(tiny,monkeypatch):
    from experiments.factors import run
    monkeypatch.setattr(run,"main",lambda _:pytest.fail("reran completed branch"))
    study.train(tiny[1],"tiny")


@pytest.mark.parametrize("field",["mode","idle_lambda","shrink_step","seed","source_sha256"])
def test_foreign_branch_identity_refused(tiny,field):
    from experiments.factors.run import FactorTrainer
    item=deepcopy(tiny[3]);item[field]={"mode":"fixed","idle_lambda":1.,"shrink_step":.1,"seed":74,"source_sha256":"0"*64}[field]
    with _preserve_process_state():
        t=FactorTrainer.from_checkpoint(Path(item["output"])/"checkpoint.pt",device="cpu")
        with pytest.raises(ValueError):study.validate(t,tiny[2],item,complete=True)


@pytest.mark.parametrize("field",["target","regime","reset"])
def test_wrong_progress_or_live_state_refused(tiny,field):
    from experiments.factors.run import FactorTrainer
    with _preserve_process_state():
        t=FactorTrainer.from_checkpoint(Path(tiny[3]["output"])/"checkpoint.pt",device="cpu")
        plan=deepcopy(tiny[2])
        if field=="target":plan["target_ticks"]=64
        elif field=="regime":t.world.set_regime(duration_range=(9.,10.))
        else:t.base.hidden_resets=1
        with pytest.raises(ValueError):study.validate(t,plan,tiny[3],complete=True)


def test_foreign_completion_marker_refused(tiny,tmp_path):
    path=Path(tiny[3]["output"])/"launch-complete.json";original=path.read_bytes();value=json.loads(original);value["branch"]["idle_lambda"]=9.
    try:
        atomic_json(path,value)
        with pytest.raises(ValueError,match="completion"):study.evaluate(tiny[1],"tiny",path.parent/"checkpoint.pt",tmp_path/"bad.json")
        assert not (tmp_path/"bad.json").exists()
    finally:path.write_bytes(original)
