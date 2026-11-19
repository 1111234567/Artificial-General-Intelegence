import json
from pathlib import Path
import sys
import pytest
from experiments.credit.run import file_sha256
from experiments.continuity.run import atomic_json
from experiments.rewiring import evaluate_ready as queue


def case(tmp_path, name="one"):
    folder=tmp_path/name
    folder.mkdir()
    checks={}
    for phase in ("B","A2"):
        path=folder/f"checkpoint_{phase}.pt"
        path.write_bytes((name+phase).encode())
        checks[phase]={"path":str(path),"sha256":file_sha256(path)}
    marker=folder/"launch-complete.json"
    atomic_json(marker,{"format":"rewiring-launch-complete-v1","branch":{"id":name},"checkpoints":checks})
    out=folder/"result.txt"
    job={"id":name,"command":[sys.executable,"-c","from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",str(out),"--checkpoint",str(folder/"checkpoint.pt")],"outputs":[str(out)],"log":str(folder/"log.txt")}
    return job,marker


def test_two_phase_ready_and_missing_marker(tmp_path):
    job,marker=case(tmp_path)
    assert queue.ready(job)==file_sha256(marker)
    marker.unlink()
    assert queue.ready(job) is None


@pytest.mark.parametrize("field",["format","branch","path","bytes"])
def test_corrupt_ready_provenance_refused(tmp_path,field):
    job,marker=case(tmp_path)
    data=json.loads(marker.read_text())
    if field=="format": data["format"]="wrong"
    elif field=="branch": data["branch"]["id"]="foreign"
    elif field=="path": data["checkpoints"]["B"]["path"]=str(tmp_path/"foreign.pt")
    else: (marker.parent/"checkpoint_B.pt").write_bytes(b"changed")
    atomic_json(marker,data)
    with pytest.raises(ValueError): queue.ready(job)


def test_finite_queue_two_jobs_and_resume(tmp_path):
    jobs=[case(tmp_path,name)[0] for name in ("one","two")]
    plan=tmp_path/"plan.json"
    atomic_json(plan,{"cwd":str(tmp_path),"file_hashes":{},"jobs":jobs})
    state=tmp_path/"state.json"
    upstream=tmp_path/"training.json"
    atomic_json(upstream,{"jobs":{},"complete":True})
    assert queue.run(plan,state,upstream,2)==0
    first=json.loads(state.read_text())
    assert first["complete"] and len(first["jobs"])==2
    assert queue.run(plan,state,upstream,2)==0
    assert json.loads(state.read_text())["jobs"]==first["jobs"]
    Path(jobs[0]["outputs"][0]).unlink()
    with pytest.raises(ValueError,match="missing"): queue.run(plan,state,upstream,2)
