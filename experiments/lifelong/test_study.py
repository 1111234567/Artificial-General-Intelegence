from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json
from experiments.credit.run import file_sha256
from experiments.lifelong.run import NativeConfig,NativeTrainer,write_run
from experiments.lifelong import study


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp("native-study");out=root/"run"
    config=NativeConfig(seed=13,architecture="grow_shrink",schedule="mixed",learning="history_replay",coverage="wide",
        width=8,batch_size=2,chunk_ticks=4,initial_fan_in=4,replay_capacity=3,
        main_ticks=32,macrocycle_ticks=16,block_ticks=4,withdrawal_ticks=8,return_ticks=8)
    ticks=[0,4,32,40,44,48];entries={}
    with _preserve_process_state():
        trainer=NativeTrainer(config)
        for tick in ticks:
            trainer.run_updates((tick-trainer.ticks)//config.chunk_ticks)
            cp=out/"checkpoints"/f"tick_{tick:06d}.pt";trainer.save_checkpoint(cp)
            entries[str(tick)]=dict(path=str(cp),sha256=file_sha256(cp),semantic_hash=study.normalized_hash(trainer,"cpu"))
        trainer.save_checkpoint(out/"latest.pt");write_run(out,trainer)
    item={key:getattr(config,key) for key in ("seed","architecture","schedule","learning","coverage")}
    item.update(id="tiny",output=str(out))
    plan={key:getattr(config,key) for key in ("width","batch_size","chunk_ticks","initial_fan_in","main_ticks","withdrawal_ticks","return_ticks")}
    plan.update(format="lifelong-study-protocol-v1",sources={str(Path(study.__file__)):file_sha256(study.__file__)},branches=[item],
        target_ticks=48,training_device="cpu",evaluation_device="cpu",dt=.25,checkpoint_ticks=ticks,
        evaluation_specs=[dict(tick=40,family=0,condition="short",steps=16,purpose="before_return"),
                          dict(tick=48,family=0,condition="short",steps=16,purpose="final")])
    protocol=root/"protocol.json";atomic_json(protocol,plan);sha=file_sha256(protocol)
    atomic_json(out/"launch-complete.json",dict(format="lifelong-launch-complete-v1",branch=item,protocol_sha256=sha,
        target=48,stage_ticks=48,checkpoint_sha256=file_sha256(out/"latest.pt"),checkpoints=entries,hidden_resets=0))
    return root,protocol,plan,item


def test_native_two_point_fork_and_completed_skip(tiny,monkeypatch):
    from experiments.lifelong import run
    root,path,plan,item=tiny
    monkeypatch.setattr(run,"main",lambda _:pytest.fail("completed native run restarted"))
    study.train(path,"tiny")
    result=root/"eval.json";study.evaluate(path,"tiny",Path(item["output"])/"latest.pt",result)
    value=json.loads(result.read_text());assert value["source_unchanged"] and len(value["records"])==2
    assert {r["probe"]["tick"] for r in value["records"]}=={40,48}
    with pytest.raises(ValueError,match="existing"):study.evaluate(path,"tiny",Path(item["output"])/"latest.pt",result)


@pytest.mark.parametrize("field",["seed","architecture","schedule","learning","coverage"])
def test_foreign_native_condition_refused(tiny,field):
    item=deepcopy(tiny[3]);item[field]={"seed":14,"architecture":"fixed_shrink","schedule":"blocked","learning":"recent_repeat","coverage":"narrow"}[field]
    with _preserve_process_state():
        t=NativeTrainer.from_checkpoint(Path(item["output"])/"latest.pt",device="cpu")
        with pytest.raises(ValueError):study.validate(t,tiny[2],item,48)


def test_snapshot_byte_change_refused_before_publish(tiny,tmp_path):
    path=Path(tiny[3]["output"])/"checkpoints/tick_000040.pt";original=path.read_bytes()
    try:
        path.write_bytes(original+b"altered")
        with pytest.raises(ValueError,match="bytes"):study.evaluate(tiny[1],"tiny",Path(tiny[3]["output"])/"latest.pt",tmp_path/"bad.json")
        assert not (tmp_path/"bad.json").exists()
    finally:path.write_bytes(original)


def test_formal_probe_coverage_and_no_cold_clock_fork():
    specs=study.evaluation_specs()
    assert len(specs)==46 and all(s["tick"]>0 for s in specs)
    assert len({(s["tick"],s["family"],s["condition"]) for s in specs})==46
    assert {s["family"] for s in specs if s["purpose"]=="before_return"}=={0,1,2,3}
