"""Durable one-shot pipeline for the long local experiment, with no scheduler."""
import argparse
from datetime import datetime,timezone
import json
import os
import traceback
from experiments.seed_evolution import large_runner as runner,large_audit,large_results

def run(workers=14):
    with runner._lock(runner.OUT/".pipeline.lock"):
        runner._activate()
        runner._write(runner.OUT/"pipeline-process.json",dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),workers=workers))
        stage="search"
        try:
            runner.search(workers)
            stage="finals";runner.finals(6)
            stage="test";runner.tests(6)
            stage="audit";runner._progress(stage,"running")
            audit=large_audit.audit_all(runner.OUT,workers=9,metadata_only=False)
            if audit.get("full_audit_passed") is not True:
                raise ValueError("independent audit did not pass; publication stopped")
            runner._write(runner.OUT/"audit.json",audit)
            stage="results";runner._progress(stage,"running")
            result=large_results.aggregate(runner.OUT)
            runner._write(runner.OUT/"pipeline-completion.json",dict(full_audit_passed=True,result=result,completed_utc=datetime.now(timezone.utc).isoformat()))
            runner._progress("complete","complete",completed_candidates=12096,total_candidates=12096)
        except BaseException as exc:
            runner._write(runner.OUT/"pipeline-failure.json",dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
            runner._progress(stage,"failed",error=repr(exc));raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=14);a=p.parse_args();run(a.workers)
