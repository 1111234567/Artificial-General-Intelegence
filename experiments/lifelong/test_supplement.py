import json
from pathlib import Path
import pytest
from experiments.continuity.run import atomic_json
from experiments.credit.run import file_sha256
from experiments.lifelong import supplement,study
from experiments.lifelong.test_study import tiny


def test_matched_horizon_plan_has_no_duplicate_original_forks():
    extra=supplement.specs();original=study.evaluation_specs()
    key=lambda x:(x["tick"],x["family"],x["condition"],x["steps"])
    assert len(extra)==16 and len({key(x) for x in extra})==16
    assert not ({key(x) for x in extra}&{key(x) for x in original})
    all_keys={key(x) for x in original+extra}
    for tick in (131072,163840,164864,167936,172032):
        assert (tick,0,"short",2048) in all_keys and (tick,0,"long_both",2048) in all_keys


def test_real_native_supplement_binding_and_source_preservation(tiny,tmp_path):
    _,parent,_,item=tiny
    path=tmp_path/"supplement.json"
    atomic_json(path,dict(format="lifelong-coverage-supplement-v1",parent_protocol=str(parent),
        parent_protocol_sha256=file_sha256(parent),sources={str(Path(supplement.__file__)):file_sha256(supplement.__file__)},
        evaluation_specs=[dict(tick=40,family=0,condition="short",steps=16,purpose="aligned_memory")]))
    output=tmp_path/"result.json"
    supplement.evaluate(path,"tiny",Path(item["output"])/"latest.pt",output)
    result=json.loads(output.read_text());assert result["source_unchanged"] and len(result["records"])==1
    with pytest.raises(ValueError,match="existing"):supplement.evaluate(path,"tiny",Path(item["output"])/"latest.pt",output)
    bad=json.loads(path.read_text());bad["parent_protocol_sha256"]="0"*64;atomic_json(path,bad)
    with pytest.raises(ValueError,match="parent"):supplement.read_plan(path,"tiny")
