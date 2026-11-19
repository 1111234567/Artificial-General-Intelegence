"""Real subprocess queue recovery and scientific-terminal dependency checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.undirected import queue


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _terminal_value(folder, protocol_sha, *, status="success", identifier="one"):
    return {"format": queue.TERMINAL_FORMAT, "branch": {"id": identifier},
            "protocol_sha256": protocol_sha, "status": status, "target_ticks": 8,
            "last_tick": 8 if status == "success" else 2,
            "completed": status == "success", "endpoint_comparable": status == "success",
            "checkpoint_path": str(folder / "checkpoint_latest.pt"), "checkpoint_sha256": "0" * 64,
            "checkpoints": {}, "failure": None if status == "success" else
            {"path": str(folder / "failure.json"), "sha256": "0" * 64}}


WRITE_TERMINAL = """
import hashlib,json,sys
from pathlib import Path
value=json.loads(sys.argv[1]); marker=Path(sys.argv[2])
marker.parent.mkdir(parents=True,exist_ok=True)
checkpoint=Path(value['checkpoint_path']); checkpoint.write_bytes(b'complete-state')
value['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if value['failure'] is not None:
    path=Path(value['failure']['path']); path.write_text('typed numerical failure',encoding='utf-8')
    value['failure']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
marker.write_text(json.dumps(value),encoding='utf-8')
"""

WRITE_EVALUATION = """
import json,sys
from pathlib import Path
terminal=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
output=Path(sys.argv[2]);output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'records':[] if terminal['status']=='numerical_failure' else [{'evaluated':True}],
    'status':terminal['status'],'endpoint_comparable':terminal['endpoint_comparable']}),encoding='utf-8')
"""


def plans(tmp_path, *, status="success", with_training=True, with_evaluation=True, fail_once=False):
    protocol = tmp_path / "protocol.json"
    queue.atomic_json(protocol, {"format": "mechanical-queue-fixture", "target_ticks": 8})
    protocol_sha = queue.file_sha256(protocol)
    folder = tmp_path / "branch"
    marker = folder / "terminal.json"
    value = _terminal_value(folder, protocol_sha, status=status)
    metadata = {"id": "one", "branch_id": "one", "branch": {"id": "one"},
                "terminal": str(marker), "protocol_sha256": protocol_sha, "target_ticks": 8}
    code = WRITE_TERMINAL
    if fail_once:
        flag = tmp_path / "first-execution-failed"
        code = ("from pathlib import Path\nimport sys\n"
                f"flag=Path({str(flag)!r})\n"
                "if not flag.exists():\n    flag.write_text('transient execution failure')\n    sys.exit(7)\n" + code)
    training = {**metadata, "command": [sys.executable, "-c", code, json.dumps(value), str(marker)],
                "outputs": [value["checkpoint_path"], str(marker)], "log": str(tmp_path / "train.log")}
    evaluation_output = tmp_path / "evaluation_result.json"
    evaluation = {**metadata, "command": [sys.executable, "-c", WRITE_EVALUATION, str(marker), str(evaluation_output)],
                  "outputs": [str(evaluation_output)], "log": str(tmp_path / "eval.log")}
    manifest_paths = {}
    for kind, jobs in (("training", [training] if with_training else []), ("baseline", []),
                       ("evaluation", [evaluation] if with_evaluation else [])):
        path = tmp_path / f"{kind}.json"
        queue.atomic_json(path, {"format": queue.MANIFEST_FORMAT, "kind": kind, "cwd": str(tmp_path),
            "protocol": str(protocol), "protocol_sha256": protocol_sha,
            "file_hashes": {str(protocol): protocol_sha}, "jobs": jobs})
        manifest_paths[kind] = path
    limits = tmp_path / "limits.json"
    queue.atomic_json(limits, {"max_workers": 2, "training": 1, "evaluation": 1, "min_available_bytes": 0})
    return manifest_paths, limits, training, evaluation


def write_dependency(job, *, status="success"):
    marker = Path(job["terminal"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    value = _terminal_value(marker.parent, job["protocol_sha256"], status=status, identifier=job["id"])
    checkpoint = Path(value["checkpoint_path"])
    checkpoint.write_bytes(b"checkpoint")
    value["checkpoint_sha256"] = queue.file_sha256(checkpoint)
    if value["failure"] is not None:
        failure = Path(value["failure"]["path"])
        failure.write_text("typed numerical failure", encoding="utf-8")
        value["failure"]["sha256"] = queue.file_sha256(failure)
    queue.atomic_json(marker, value)
    return value


def test_successful_subprocesses_and_completed_jobs_are_preserved(tmp_path):
    manifests, limits, _, evaluation = plans(tmp_path)
    state_root = tmp_path / "queue"
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    before = {kind: _json(state_root / f"{kind}-execution.json")["jobs"] for kind in manifests}
    assert _json(evaluation["outputs"][0])["records"] == [{"evaluated": True}]
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    assert all(_json(state_root / f"{kind}-execution.json")["jobs"] == rows for kind, rows in before.items())
    assert all(_json(state_root / f"{kind}-execution.json")["complete"] for kind in manifests)
    assert _json(state_root / "training-execution.json")["terminal_counts"] == {"success": 1, "numerical_failure": 0}


def test_numerical_terminal_launches_empty_evaluation_and_completes_queue(tmp_path):
    manifests, limits, training, evaluation = plans(tmp_path, status="numerical_failure")
    state_root = tmp_path / "queue"
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    assert queue.ready(evaluation)
    value = _json(evaluation["outputs"][0])
    assert value == {"records": [], "status": "numerical_failure", "endpoint_comparable": False}
    row = _json(state_root / "training-execution.json")["jobs"][training["id"]]
    assert row["returncode"] == 0
    assert row["terminal_status"] == "numerical_failure"
    assert _json(training["terminal"])["last_tick"] < 8
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0


def test_execution_failure_stays_nonzero_and_resumes_same_manifest(tmp_path):
    manifests, limits, training, _ = plans(tmp_path, fail_once=True)
    state_root = tmp_path / "queue"
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 1
    first = _json(state_root / "training-execution.json")
    assert first["jobs"][training["id"]]["returncode"] == 7
    assert "terminal_status" not in first["jobs"][training["id"]]
    assert not first["complete"]
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    resumed = _json(state_root / "training-execution.json")
    assert resumed["jobs"][training["id"]]["returncode"] == 0
    assert resumed["jobs"][training["id"]]["attempts"][0]["returncode"] == 7
    assert resumed["resume_history"]
    assert Path(resumed["resume_history"][0]["archive"]).exists()


def test_dead_pending_job_is_resumed_but_live_pid_is_refused(tmp_path):
    manifests, limits, training, _ = plans(tmp_path)
    state_root = tmp_path / "queue"
    state_root.mkdir()
    plan = _json(manifests["training"])
    state = {"format": queue.QUEUE_FORMAT, "kind": "training", "plan": str(manifests["training"].resolve()),
        "plan_digest": queue.digest(plan), "jobs": {training["id"]: {"returncode": None, "pid": os.getpid(),
            "command_digest": queue.digest(training["command"])}}}
    path = state_root / "training-execution.json"
    queue.atomic_json(path, state)
    with pytest.raises(RuntimeError, match="existing live job"):
        queue.execute(manifests, state_root, limits, poll_interval=.01)
    state["jobs"][training["id"]]["pid"] = 2 ** 30
    queue.atomic_json(path, state)
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    assert _json(path)["jobs"][training["id"]]["attempts"][0]["returncode"] is None


@pytest.mark.parametrize("change", ("branch", "protocol", "target", "progress", "bytes", "status", "bool_tick"))
def test_ready_refuses_foreign_or_corrupt_success_terminal(tmp_path, change):
    _, _, _, job = plans(tmp_path)
    assert not queue.ready(job)
    value = write_dependency(job)
    if change == "branch":
        value["branch"]["id"] = "foreign"
    elif change == "protocol":
        value["protocol_sha256"] = "f" * 64
    elif change == "target":
        value["target_ticks"] = 9
    elif change == "progress":
        value["last_tick"] = 7
    elif change == "bytes":
        Path(value["checkpoint_path"]).write_bytes(b"changed")
    elif change == "bool_tick":
        value["last_tick"] = True
    else:
        value["status"] = "execution_failure"
    queue.atomic_json(job["terminal"], value)
    with pytest.raises(ValueError):
        queue.ready(job)


def test_ready_requires_numerical_failure_bytes_and_incomparable_endpoint(tmp_path):
    _, _, _, job = plans(tmp_path)
    value = write_dependency(job, status="numerical_failure")
    assert queue.ready(job)
    Path(value["failure"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="failure bytes"):
        queue.ready(job)
    value = write_dependency(job, status="numerical_failure")
    value["endpoint_comparable"] = True
    queue.atomic_json(job["terminal"], value)
    with pytest.raises(ValueError, match="endpoint semantics"):
        queue.ready(job)


def test_missing_terminal_without_a_producer_fails_instead_of_waiting_forever(tmp_path):
    manifests, limits, _, _ = plans(tmp_path, with_training=False)
    state_root = tmp_path / "queue"
    started = time.monotonic()
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 1
    assert time.monotonic() - started < 2
    state = _json(state_root / "evaluation-execution.json")
    assert state["blocked_jobs"] == ["one"]
    assert "No remaining producer" in state["coordinator_error"]


def test_protocol_source_manifest_and_completed_output_changes_are_refused(tmp_path):
    manifests, limits, _, evaluation = plans(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    for path in manifests.values():
        plan = _json(path)
        plan["file_hashes"][str(source)] = queue.file_sha256(source)
        queue.atomic_json(path, plan)
    original = source.read_bytes()
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source"):
        queue.execute(manifests, tmp_path / "state", limits, poll_interval=.01)
    source.write_bytes(original)
    state_root = tmp_path / "state"
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    output = Path(evaluation["outputs"][0])
    original_output = output.read_bytes()
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="output bytes"):
        queue.execute(manifests, state_root, limits, poll_interval=.01)
    output.write_bytes(original_output)
    plan = _json(manifests["training"])
    plan["jobs"][0]["command"] += ["foreign-arg"]
    queue.atomic_json(manifests["training"], plan)
    with pytest.raises(ValueError, match="manifest or execution binding"):
        queue.execute(manifests, state_root, limits, poll_interval=.01)


def test_dynamic_limits_are_reloaded_and_legacy_limits_remain_explicit(tmp_path, monkeypatch):
    manifests, limits, _, _ = plans(tmp_path)
    queue.atomic_json(limits, {"total": 1, "training": 1, "cpu": 1, "min_available_bytes": 0})
    original_popen = queue.subprocess.Popen
    seen = []

    def start(*args, **kwargs):
        seen.append(kwargs)
        process = original_popen(*args, **kwargs)
        if len(seen) == 1:
            queue.atomic_json(limits, {"max_workers": 2, "training": 2, "evaluation": 1})
        return process

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    state_root = tmp_path / "state"
    assert queue.execute(manifests, state_root, limits, poll_interval=.01) == 0
    history = _json(state_root / "training-execution.json")["limits_history"]
    assert [entry["effective"]["max_workers"] for entry in history] == [1, 2]
    assert seen[0]["env"]["OMP_NUM_THREADS"] == "1"
    if os.name == "nt":
        assert seen[0]["creationflags"] & subprocess.CREATE_NO_WINDOW
    with pytest.raises(ValueError, match="conflicting"):
        queue._limits({"max_workers": 2, "total": 3})


def test_manifest_changes_after_launch_are_refused(tmp_path, monkeypatch):
    manifests, limits, _, _ = plans(tmp_path)
    original_popen = queue.subprocess.Popen

    def start(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        value = _json(manifests["evaluation"])
        value["jobs"][0]["command"] += ["foreign-arg"]
        queue.atomic_json(manifests["evaluation"], value)
        return process

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    with pytest.raises(ValueError, match="manifest changed during execution"):
        queue.execute(manifests, tmp_path / "state", limits, poll_interval=.01)


def test_outputs_cannot_overwrite_manifests_or_pinned_sources(tmp_path):
    manifests, limits, _, _ = plans(tmp_path)
    value = _json(manifests["evaluation"])
    value["jobs"][0]["outputs"] = [str(manifests["evaluation"])]
    queue.atomic_json(manifests["evaluation"], value)
    with pytest.raises(ValueError, match="collides"):
        queue.execute(manifests, tmp_path / "state", limits, poll_interval=.01)


@pytest.mark.skipif(os.name != "nt", reason="Windows spawned-worker process tree")
def test_termination_covers_own_spawned_worker(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess,sys,time;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            "Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)")
    process = subprocess.Popen([sys.executable, "-c", code, str(pidfile)], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        child = int(pidfile.read_text())
        assert queue.live_pid(child)
        queue.terminate_tree(process)
        assert not queue.live_pid(child)
    finally:
        queue.terminate_tree(process)
