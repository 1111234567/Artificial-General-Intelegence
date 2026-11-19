"""Explicit execution-only amendment: exact CPU caches and paired hybrid lanes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import traceback
from unittest.mock import patch
import torch
from experiments.seed_evolution import core,large_runner as runner,hybrid_backend,encoder_cache,lazy_training_streams
from experiments.learned_binding import common

ROOT=runner.ROOT
OUT=runner.OUT
AMENDMENT="execution-amendment.json"
_ORIGINAL_TRAIN=core.inner_train
_ORIGINAL_PARALLEL=runner._parallel
_STORE=None
_SPEC=None
_META_GPU=None
_ORIGINAL_COMMITTED=runner._committed_candidate

def sha(p):return common.sha(p)
def read(p):return runner.base.read(p)

def load_amendment(root=OUT):
    root=Path(root);spec=read(root/AMENDMENT)
    if spec["format"]!="seed-evolution-execution-v1":raise ValueError("unknown execution amendment")
    if sha(root/"protocol.json")!=spec["original_protocol_sha256"]:raise ValueError("original protocol changed")
    for name,h in spec["sources"].items():
        if sha(ROOT/name)!=h:raise ValueError(f"execution source changed: {name}")
    if spec["hybrid_indices"]!=list(range(8)):raise ValueError("device assignment changed")
    return spec

def backend_for(cid,spec):
    match=re.fullmatch(r"(?:shared|evolution|random)_g(\d{2})_i(\d{2})",cid)
    if not match:raise ValueError(cid)
    generation,index=map(int,match.groups())
    if generation<spec["start_generation"]:return "legacy_cpu"
    return "hybrid" if index in spec["hybrid_indices"] else "cpu"

def _execution(result,mode,spec):
    metadata=result["metadata"];actual=metadata.pop("execution")
    key="hybrid" if mode=="hybrid" else "cpu"
    expected=dict(spec["backend_metadata"][key]);expected.pop("lazy_streams",None);expected.pop("lazy_source_sha",None);expected.pop("backend_source_sha",None)
    common.assert_tree_equal(actual,expected,"actual backend environment")
    if mode!="legacy_cpu":
        metadata["execution"]=dict(spec["backend_metadata"][key],amendment_sha=sha(OUT/AMENDMENT))
    return result

def _init_worker(spec):
    global _SPEC,_STORE
    runner._init_worker();_SPEC=load_amendment();common.assert_tree_equal(_SPEC,spec,"worker amendment")
    _STORE=lazy_training_streams.LazyTrainingStreams()

def _job_mode(job,role,spec):
    if role=="candidate":cid=job["id"]
    elif role=="final":cid=job["id"]
    else:return "legacy_cpu"
    return backend_for(cid,spec)

def _validate_training(path,mode,spec):
    selected=path if path.is_file() else runner._pending(path)
    if not selected.is_file():return
    meta=read(selected)["metadata"]
    if mode=="legacy_cpu":
        if meta["device"]!="cpu" or "execution" in meta:raise ValueError("legacy CPU provenance changed")
    else:
        common.assert_tree_equal(meta.get("execution"),dict(spec["backend_metadata"][mode],amendment_sha=sha(OUT/AMENDMENT)),"resumed training backend")
        if meta["device"]!=("cpu+cuda_executor" if mode=="hybrid" else "cpu"):raise ValueError("resumed device differs")

def _committed(desc):
    result=_ORIGINAL_COMMITTED(desc)
    if result is not None:
        spec=load_amendment();_validate_training(runner._candidate_dir(desc["rep"],desc["id"])/"training.json",backend_for(desc["id"],spec),spec)
    return result

def _worker(job,role,mode):
    spec=_SPEC
    if _job_mode(job,role,spec)!=mode:raise ValueError("task backend assignment changed")
    records=[]
    def train(seed,genome,steps=64,checkpoints=(0,64),device="cpu"):
        # The captured trainer prevents recursion through the production wrapper.
        with patch.object(core,"inner_train",_ORIGINAL_TRAIN),_STORE.install():
            result=hybrid_backend.train(seed,genome,steps,checkpoints,backend="hybrid" if mode=="hybrid" else "cpu",cache=True)
        result=_execution(result,mode,spec)
        records.append(dict(backend=mode,cache=result.get("cache_metrics"),streams=_STORE.stats))
        return result
    fn={"candidate":runner._candidate_job,"final":runner._train_final,"test":runner._test_final}[role]
    location=runner._candidate_dir(job["rep"],job["id"]) if role=="candidate" else runner._final_dir(job) if role=="final" else None
    if location is not None:_validate_training(location/"training.json",mode,spec)
    with patch.object(core,"inner_train",train),patch.object(runner,"_committed_candidate",_committed):result=fn(job)
    if location is not None:_validate_training(location/"training.json",mode,spec)
    if records:
        location=(runner._candidate_dir(job["rep"],job["id"]) if role=="candidate" else runner._final_dir(job))
        runner._write(location/"execution.json",dict(amendment_sha=sha(OUT/AMENDMENT),backend=mode,training_calls=records))
    return result

def _parallel(jobs,worker,workers):
    if not jobs:return
    spec=load_amendment()
    role={"_candidate_job":"candidate","_train_final":"final","_test_final":"test"}.get(worker.__name__)
    if role is None:raise ValueError(f"unrecognized sealed worker: {worker.__name__}")
    grouped={"cpu":[],"hybrid":[]}
    for job in jobs:
        mode=_job_mode(job,role,spec);grouped["hybrid" if mode=="hybrid" else "cpu"].append((job,mode))
    only_prefix=all(mode=="legacy_cpu" for job,mode in grouped["cpu"]) and not grouped["hybrid"]
    cpu_limit=(spec["prefix_workers"] if only_prefix else spec["cpu_workers"]) if role=="candidate" else min(workers,6)
    limits={"cpu":cpu_limit,"hybrid":spec["gpu_workers"]}
    pools={};pending={};queues={k:iter(v) for k,v in grouped.items()}
    try:
        for lane,items in grouped.items():
            if not items:continue
            pools[lane]=ProcessPoolExecutor(max_workers=limits[lane],initializer=_init_worker,initargs=(spec,))
            for _ in range(limits[lane]):
                item=next(queues[lane],None)
                if item is None:break
                job,mode=item;pending[pools[lane].submit(_worker,job,role,mode)]=lane
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            batch=[(pending.pop(future),future.result()) for future in finished]
            for lane,result in batch:
                yield result
                item=next(queues[lane],None)
                if item is not None:
                    job,mode=item;pending[pools[lane].submit(_worker,job,role,mode)]=lane
    except BaseException:
        for p in pools.values():
            if hasattr(p,"terminate_workers"):p.terminate_workers()
        raise
    finally:
        for p in pools.values():p.shutdown(wait=True,cancel_futures=True)

def prepare(prefix_workers=20,cpu_workers=18,gpu_workers=2):
    path=OUT/AMENDMENT
    with runner._lock(OUT/".pipeline.lock"),runner._lock(OUT/".run.lock"):
        if path.exists():raise FileExistsError(path)
        runner._activate();runner._check()
        plans=sorted((OUT/"plans").glob("generation_*.json"))
        start=max(int(p.stem.split("_")[-1]) for p in plans)+1
        if start>=32:raise ValueError("search has no unissued generation for backend change")
        sources={}
        for name in ["execution_runtime.py","execution_audit.py","execution_results.py","hybrid_backend.py","encoder_cache.py","lazy_training_streams.py"]:
            f=Path(__file__).with_name(name);sources[str(f.relative_to(ROOT))]=sha(f)
        templates={}
        genome=core.new_genome(0,710000)
        for mode in ("cpu","hybrid"):
            result=hybrid_backend.train(0,genome,steps=1,checkpoints=(0,1),backend=mode,cache=True)
            template=result["metadata"]["execution"]
            template.update(backend_source_sha=sha(hybrid_backend.__file__),lazy_streams=True,lazy_source_sha=sha(lazy_training_streams.__file__))
            templates[mode]=template
        inventory={str(p.relative_to(OUT)):sha(p) for pattern in ("search/rep_*/*/candidate.json","search/rep_*/*/training.json","search/rep_*/*/.training.json.pending") for p in OUT.glob(pattern)}
        spec=dict(format="seed-evolution-execution-v1",created_utc=datetime.now(timezone.utc).isoformat(),
            original_protocol_sha256=sha(OUT/"protocol.json"),start_generation=start,hybrid_indices=list(range(8)),
            issued_plans={str(p.relative_to(OUT)):sha(p) for p in plans},existing_prefix_inventory=inventory,
            prefix_workers=prefix_workers,cpu_workers=cpu_workers,gpu_workers=gpu_workers,
            backend_metadata=templates,sources=sources,cache_encoder=True,lazy_training=True,
            disclosure="User-authorized execution amendment. FP32 unchanged; CUDA arithmetic is not bitwise CPU. Issued generations remain CPU; subsequent corresponding E/R indices use the same backend. No budget or selection changes.")
        runner._write(path,spec)
        return dict(start_generation=start,committed_candidates=sum(1 for p in OUT.glob("search/rep_*/*/candidate.json")),cpu_workers=cpu_workers,gpu_workers=gpu_workers)

def run():
    from experiments.seed_evolution import execution_audit,execution_results
    with runner._lock(OUT/".pipeline.lock"):
        runner._activate();spec=load_amendment()
        runner._write(OUT/"execution-process.json",dict(pid=__import__('os').getpid(),started_utc=datetime.now(timezone.utc).isoformat(),amendment_sha=sha(OUT/AMENDMENT)))
        stage="search"
        try:
            with patch.object(runner,"_parallel",_parallel),patch.object(runner,"_committed_candidate",_committed):
                runner.search(spec["prefix_workers"])
                stage="finals";runner.finals(6)
                stage="test";runner.tests(6)
            stage="audit";runner._progress(stage,"running")
            result=execution_audit.audit_all(OUT,workers=9)
            if result.get("full_audit_passed") is not True:raise ValueError("execution audit failed")
            runner._write(OUT/"audit.json",result)
            if result.get("execution_amendment_sha256")!=sha(OUT/AMENDMENT):raise ValueError("execution amendment changed")
            stage="results";summary=execution_results.aggregate(OUT)
            runner._write(OUT/"pipeline-completion.json",dict(full_audit_passed=True,result=summary,execution_amendment_sha256=sha(OUT/AMENDMENT)))
            runner._progress("complete","complete",completed_candidates=12096,total_candidates=12096)
        except BaseException as e:
            runner._write(OUT/"execution-failure.json",dict(stage=stage,error=repr(e),traceback=traceback.format_exc()))
            runner._progress(stage,"failed",error=repr(e));raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","run"));p.add_argument("--prefix-workers",type=int,default=20);p.add_argument("--cpu-workers",type=int,default=18);p.add_argument("--gpu-workers",type=int,default=2)
    a=p.parse_args()
    if a.command=="prepare":print(json.dumps(prepare(a.prefix_workers,a.cpu_workers,a.gpu_workers)))
    else:run()
