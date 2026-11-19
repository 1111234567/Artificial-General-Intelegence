"""Finite native lifelong queues sharing a live process budget."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.credit.batch import digest
from experiments.credit.run import file_sha256
from experiments.credit.windows_runner import save_with_retry as save
from experiments.loops.audit_run import verify_files


def available_memory():
    if os.name != "nt": return float("inf")
    class Memory(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ("total","available","page_total","page_available","virtual_total","virtual_available","extended")]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)): raise OSError("memory status unavailable")
    return value.available


def ready(job):
    command=job["command"]
    cp=Path(command[command.index("--checkpoint")+1])
    marker=cp.parent/"launch-complete.json"
    if not marker.exists(): return False
    value=json.loads(marker.read_text(encoding="utf-8"))
    if (value.get("format")!="lifelong-launch-complete-v1" or value.get("branch",{}).get("id")!=job["id"]
            or value.get("stage_ticks")!=value.get("target") or value.get("checkpoint_sha256")!=file_sha256(cp)):
        raise ValueError("untrusted completion marker")
    return True


def live_pid(pid):
    if not pid: return False
    if os.name=="nt":
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def terminate_tree(process):
    if process.poll() is None:
        if os.name=="nt":
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        else: process.terminate()
    process.wait()


def execute(plans, folder, limits_path):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    states,pending={},{}
    for kind,path in plans.items():
        plan=json.loads(Path(path).read_text(encoding="utf-8"));verify_files(plan["file_hashes"])
        statepath=folder/f"{kind}-execution.json"
        state=json.loads(statepath.read_text()) if statepath.exists() else {"plan":str(Path(path).resolve()),"plan_digest":digest(plan),"jobs":{}}
        if state["plan_digest"]!=digest(plan):raise ValueError("command manifest changed")
        for identifier,row in state["jobs"].items():
            if row.get("returncode") is None and live_pid(row.get("pid")):raise RuntimeError(f"existing live job {identifier}")
        unfinished=[job for job in plan["jobs"] if state["jobs"].get(job["id"],{}).get("returncode")!=0]
        for job in plan["jobs"]:
            if job not in unfinished and not all(Path(p).exists() for p in job["outputs"]):raise ValueError("successful job outputs missing")
        if statepath.exists() and not state.get("complete"):
            index=1
            while statepath.with_name(f"{kind}-execution-attempt-{index:03d}.json").exists():index+=1
            archive=statepath.with_name(f"{kind}-execution-attempt-{index:03d}.json");archive.write_bytes(statepath.read_bytes())
            state.setdefault("resume_history",[]).append({"archive":str(archive),"sha256":file_sha256(archive),"time":time.time()})
        state.pop("finished",None);state["complete"]=False
        state["coordinator_sha256"]=file_sha256(__file__)
        states[kind]=(state,plan,statepath);pending[kind]=unfinished
        save(statepath,state)
    active={};failed=False;last_limits=None
    env=os.environ.copy();env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    try:
        while any(pending.values()) or active:
            limits=json.loads(Path(limits_path).read_text())
            if any(type(limits.get(k)) is not int or limits[k]<1 for k in ("training","cpu","total")):raise ValueError("invalid limits")
            if limits!=last_limits:
                for state,_,statepath in states.values():
                    state.setdefault("limits_history",[]).append({"time":time.time(),"limits":limits})
                    save(statepath,state)
                last_limits=limits
            for kind in ("training","baseline","evaluation"):
                state,plan,statepath=states[kind]
                for job in tuple(pending[kind]):
                    train_count=sum(k[0]=="training" for k in active)
                    cpu_count=len(active)-train_count
                    if failed or len(active)>=limits["total"] or (kind=="training" and train_count>=limits["training"]) or (kind!="training" and cpu_count>=limits["cpu"]):break
                    if available_memory()<limits.get("min_available_bytes",0):break
                    if kind=="evaluation" and not ready(job):continue
                    logpath=Path(job["log"]);logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open("ab")
                    process=subprocess.Popen(job["command"],cwd=plan["cwd"],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                    state["jobs"][job["id"]]={"started":time.time(),"pid":process.pid,"returncode":None,"command_digest":digest(job["command"]),"log":str(logpath)}
                    active[(kind,job["id"])]=(job,process,log);pending[kind].remove(job)
                    print(json.dumps({"started":job["id"],"kind":kind,"pid":process.pid}),flush=True);save(statepath,state)
            for key,(job,process,log) in tuple(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();missing=[p for p in job["outputs"] if not Path(p).exists()]
                if code==0 and missing:code=-1
                kind,identifier=key;state,_,statepath=states[kind]
                state["jobs"][identifier].update(returncode=code,finished=time.time(),missing_outputs=missing)
                failed |= code!=0;del active[key]
                print(json.dumps({"finished":identifier,"kind":kind,"returncode":code}),flush=True);save(statepath,state)
            if failed and not active:break
            if any(pending.values()) or active:time.sleep(1)
    finally:
        for (kind,identifier),(job,process,log) in active.items():
            terminate_tree(process);log.close()
            states[kind][0]["jobs"][identifier].update(returncode=process.returncode,finished=time.time())
        for kind,(state,_,statepath) in states.items():
            state.update(finished=time.time(),complete=not pending[kind] and all(r.get("returncode")==0 for r in state["jobs"].values()))
            save(statepath,state)
    return 0 if all(s[0]["complete"] for s in states.values()) else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("training-plan","evaluation-plan","baseline-plan","status-root","limits"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(execute({"training":args.training_plan,"evaluation":args.evaluation_plan,"baseline":args.baseline_plan},args.status_root,args.limits))
