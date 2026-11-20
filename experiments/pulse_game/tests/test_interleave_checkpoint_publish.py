import json

import pytest

from experiments.pulse_game import interleave_checkpoint_publish as publish


def test_publication_only_change_keeps_the_original_optimizer_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(publish,"EVIDENCE",tmp_path)
    old=dict(births=[7],code="old",budget=8)
    fresh=dict(births=[7],code="new",budget=8)
    file=tmp_path/"optimizer"/"run.json"
    identity=publish.locked_plan(file,old,"comparison","code")
    content=file.read_bytes()
    (tmp_path/"checkpoint-publication.json").write_text(json.dumps({"transitions":{"comparison":["old","new"]}}))
    assert publish.locked_plan(file,fresh,"comparison","code")==identity
    assert file.read_bytes()==content
    for changed in ({**fresh,"budget":9},{**fresh,"code":"different"}):
        with pytest.raises(ValueError):publish.locked_plan(file,changed,"comparison","code")


def test_partly_exported_final_stage_is_visible_to_restart(tmp_path):
    births=[731,732]
    assert not publish.needs_export(tmp_path,"odor","candidate",births,7,(0,8))
    for root in births:
        assert publish.needs_export(tmp_path,"odor","candidate",births,8,(0,8))
        f=publish.model_io.checkpoint_path(tmp_path,"odor","candidate",root,8)
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes(b"complete")
    assert not publish.needs_export(tmp_path,"odor","candidate",births,8,(0,8))
