"""Regression for the observed Windows concurrent-reader status-write failure."""
import json
from pathlib import Path
from unittest.mock import patch

from experiments.credit.windows_runner import save_with_retry, prepare_resume


def test_transient_replace_denial_retries_and_preserves_complete_json(tmp_path):
    path=tmp_path/"execution.json"
    path.write_text('{"old":true}',encoding="utf-8")
    actual=Path.replace
    calls=[]
    def temporarily_denied(source,target):
        calls.append(1)
        if len(calls)<3:
            raise PermissionError("simulated concurrent Windows reader")
        return actual(source,target)
    with patch.object(Path,"replace",temporarily_denied), patch("experiments.credit.windows_runner.time.sleep"):
        save_with_retry(path,{"new":True})
    assert len(calls)==3
    assert json.loads(path.read_text())=={"new":True}


def test_resume_archives_failure_evidence_and_only_requeues_unfinished_jobs(tmp_path):
    path=tmp_path/"execution.json"
    original={"plan_digest":"fixed", "jobs":{"done":{"returncode":0},"killed":{"returncode":1}},
              "finished":123.,"complete":False}
    path.write_text(json.dumps(original),encoding="utf-8")
    prepare_resume(path)
    updated=json.loads(path.read_text())
    archived=json.loads((tmp_path/"execution-attempt-001.json").read_text())
    assert archived==original
    assert updated["jobs"]["done"]==original["jobs"]["done"]
    assert updated["jobs"]["killed"]["returncode"] is None
    assert updated["jobs"]["killed"]["status"]=="queued_for_resume"
    assert updated["resume_history"][0]["jobs"]==["killed"]


def test_completed_manifest_rerun_is_not_recorded_as_a_failure(tmp_path):
    path=tmp_path/"execution.json"
    original=json.dumps({"jobs":{"done":{"returncode":0}},"finished":123.,"complete":True}).encode()
    path.write_bytes(original)
    prepare_resume(path)
    assert path.read_bytes()==original
    assert not (tmp_path/"execution-attempt-001.json").exists()
