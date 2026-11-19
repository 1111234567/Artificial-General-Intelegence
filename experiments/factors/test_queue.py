import json
import sys
import os
import subprocess
import time
from pathlib import Path
import pytest
from experiments.continuity.run import atomic_json
from experiments.credit.run import file_sha256
from experiments.factors import queue


def test_finite_queue_preserves_completed_jobs_and_resumes(tmp_path):
    cp=tmp_path/"checkpoint.pt";cp.write_bytes(b"checkpoint")
    atomic_json(tmp_path/"launch-complete.json",dict(format="factors-launch-complete-v1",branch=dict(id="eval"),stage_ticks=1,target=1,checkpoint_sha256=file_sha256(cp)))
    plans={}
    for kind in ["training","baseline","evaluation"]:
        name="eval" if kind=="evaluation" else kind
        output=tmp_path/f"{kind}.result"
        command=[sys.executable,"-c","from pathlib import Path;import sys;Path(sys.argv[1]).write_text('ok')",str(output)]
        if kind=="evaluation":command += ["--checkpoint",str(cp)]
        path=tmp_path/f"{kind}.json";atomic_json(path,dict(cwd=str(tmp_path),file_hashes={},jobs=[dict(id=name,command=command,outputs=[str(output)],log=str(tmp_path/f'{kind}.log'))]));plans[kind]=path
    limits=tmp_path/"limits.json";atomic_json(limits,dict(training=1,cpu=2,total=3,min_available_bytes=0))
    state=tmp_path/"state"
    assert queue.execute(plans,state,limits)==0
    before={k:json.loads((state/f"{k}-execution.json").read_text())["jobs"] for k in plans}
    assert queue.execute(plans,state,limits)==0
    assert all(json.loads((state/f"{k}-execution.json").read_text())["jobs"]==v for k,v in before.items())


@pytest.mark.parametrize("change",["identity","bytes","progress"])
def test_ready_rejects_foreign_completion(tmp_path,change):
    cp=tmp_path/"checkpoint.pt";cp.write_bytes(b"source")
    value=dict(format="factors-launch-complete-v1",branch=dict(id="one"),stage_ticks=1,target=1,checkpoint_sha256=file_sha256(cp))
    if change=="identity":value["branch"]["id"]="two"
    elif change=="progress":value["stage_ticks"]=0
    else:cp.write_bytes(b"changed")
    atomic_json(tmp_path/"launch-complete.json",value)
    with pytest.raises(ValueError):queue.ready(dict(id="one",command=["--checkpoint",str(cp)]))


@pytest.mark.skipif(os.name!="nt",reason="Windows venv proxy process tree")
def test_termination_covers_spawned_worker(tmp_path):
    pidfile=tmp_path/"child.pid"
    code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)"
    p=subprocess.Popen([sys.executable,"-c",code,str(pidfile)],creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert pidfile.exists()
        child=int(pidfile.read_text());assert queue.live_pid(child)
        queue.terminate_tree(p)
        assert not queue.live_pid(child)
    finally:
        queue.terminate_tree(p)
