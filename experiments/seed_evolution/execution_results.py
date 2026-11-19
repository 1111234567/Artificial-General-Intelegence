"""Publish the sealed results with truthful mixed-backend provenance."""
import argparse
import json
from pathlib import Path
from experiments.seed_evolution import large_results,execution_runtime
from experiments.learned_binding import common

def aggregate(root=execution_runtime.OUT,dest=large_results.DEST):
    root,dest=Path(root),Path(dest)
    spec=execution_runtime.load_amendment(root)
    audit=large_results.read(root/"audit.json")
    expected=common.sha(root/"execution-amendment.json")
    if audit.get("execution_amendment_sha256")!=expected or not audit.get("suffix_backend_counts"):
        raise ValueError("complete execution-aware audit required before publication")
    source=Path(__file__).with_name("execution_audit.py")
    if audit.get("execution_audit_sha256")!=common.sha(source):raise ValueError("execution audit implementation changed")
    result=large_results.aggregate(root,dest)
    report=large_results.read(dest/"results.json")
    report["execution"]=dict(amendment_sha256=expected,start_generation=spec["start_generation"],
        hybrid_indices=spec["hybrid_indices"],backend_metadata=spec["backend_metadata"],
        suffix_backend_counts=audit["suffix_backend_counts"],
        note="The original CPU-only device fields are superseded by this execution amendment. FP32 budgets/selection unchanged; CUDA arithmetic is not bitwise CPU.")
    common.write_json(dest/"results.json",report)
    with (dest/"report.md").open("a",encoding="utf-8") as handle:
        handle.write(f"\n执行后端修订：第{spec['start_generation']+1}代起，每组相同序号使用同一CPU/混合后端；已发计划的代保留CPU。精度与预算不变，CUDA舍入不等同CPU逐位结果。详见results.json的execution与审计中的后端计数。\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=execution_runtime.OUT);p.add_argument("--output",type=Path,default=large_results.DEST);a=p.parse_args();print(json.dumps(aggregate(a.root,a.output)))
