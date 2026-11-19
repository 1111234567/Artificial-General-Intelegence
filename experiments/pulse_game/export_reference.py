"""Export one validation-selected replicate and verify the portable NumPy game from a clean directory."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
from . import independent_births,unified_physical as physics,lifelong_agent as portable,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def choose():
    src=OUT/"independent-lifelong-births-v1"
    reg=json.loads((src/"registered.json").read_text())
    scores=[]
    for seed in reg["fixed_birth_seeds"]:
        folder=src/f"birth-{seed}"
        fit=next(folder.glob("*--full-*/frozen.json"))
        d=json.loads(fit.read_text())
        scores.append((d["best"]["score"],seed,fit))
    if len(scores)!=12:raise RuntimeError("Twelve completed policy searches required")
    return sorted(scores)[5],scores,reg


def exact_bundle(folder,path,meta):
    root=Path(__file__).parent
    sources=["world","mixture_world","mixture_flash","symmetry_world","habitat_world","unified_world","longrun_world",
             "habitat_deploy","context_local","adaptive_local","content_routing","object_world","recall_world","lifelong_agent"]
    files={"lifelong/__init__.py":b"","lifelong/reference-agent.npz":path.read_bytes()}
    for key in sources:
        src=(root/(key+".py")).read_bytes()
        if key=="mixture_flash":
            # Omit the training-data builder and its Torch-based training import.
            # The physical body and online tick implementation are kept verbatim.
            src=src.decode("utf-8-sig").split("def public_examples",1)[0]
            src="\n".join(line for line in src.splitlines() if line.strip()!="from .mixture_metric import Explorer")+"\n"
            src=src.encode("utf-8")
        files["lifelong/"+key+".py"]=src
    files["README.md"]=(folder/"README.md").read_bytes()
    files["lifelong/readout-options.json"]=(folder/"readout-options.json").read_bytes()
    contents={f:hashlib.sha256(v).hexdigest() for f,v in files.items()}
    files["manifest.json"]=json.dumps({"selected":meta,"sha256":contents},ensure_ascii=False,indent=2).encode("utf-8")
    target=folder/"continuous-learning-cpu.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,content in sorted(files.items()):z.writestr(name,content)
    return target


def verify(zip_path,npz):
    results=[]
    with tempfile.TemporaryDirectory(prefix="agi_lifelong_") as tmp:
        temp=Path(tmp)
        with zipfile.ZipFile(zip_path) as bundle:bundle.extractall(temp)
        # Explicitly reject Torch imports to exercise the NumPy-only delivery.
        child="""import builtins,json,sys
sys.path.insert(0,sys.argv[1])
old=builtins.__import__
def block(name,*a,**kw):
    if name.split('.')[0]=='torch':raise RuntimeError('Portable inference imported Torch')
    return old(name,*a,**kw)
builtins.__import__=block
from lifelong.lifelong_agent import DEFAULT,run_suite
roots=[51171,51173,51179,51181,51193,51200,51204,51205]
print(json.dumps(run_suite(sys.argv[2],roots,DEFAULT,32,.05,True,sys.argv[3]),sort_keys=True))
"""
        ss=[51171,51173,51179,51181,51193,51200,51204,51205]
        for scenario,mode in (("transfer","distributed"),("long","distributed"),("transfer","local"),("long","local"),("cross-game","local-selective")):
            expected=portable.run_suite(scenario,ss,npz,32,.05,True,mode)
            cmd=[sys.executable,"-I","-c",child,str(temp),scenario,mode]
            out=subprocess.run(cmd,cwd=temp,capture_output=True,encoding="utf-8",check=True,timeout=180)
            observed=json.loads(out.stdout)
            if expected!=observed:raise ValueError("Extracted physical game differs")
            results.append(dict(scenario=scenario,address=mode,exact_native_aggregate=True,outcome=observed))
    return results


def main():
    folder=OUT/"morning-20261119";folder.mkdir(exist_ok=True)
    (score,seed,fit),all_scores,reg=choose()
    d=json.loads(fit.read_text());study=fit.parent.parent
    f,r,m=independent_births.resources(study)
    source=next(study.glob("rls-learned-s*.pt"))
    f,r,m=physics.materialize(f,r,m,np.asarray(d["best"]["genome"]),"full")
    r["release"]=reg["reopening_rule"]
    identity=dict(birth_seed=seed,criterion="lower median developmental validation",development_score=score,
       graph_source=audit.fingerprint(study/"receptor.pt"),address_source=audit.fingerprint(source),
       policy_source=audit.fingerprint(fit),search_generations=d["settings"]["gens"],search_width=d["settings"]["width"])
    path=folder/"reference-agent.npz";portable.save(path,f,r,m,identity)
    adjacency=np.abs(f["matrix"])>1e-7
    stats=dict(recurrent_cells_per_hemisphere=len(adjacency),hemispheres=2,motor_cells=2,
       directed_nonself_edges_per_hemisphere=int(adjacency.sum()-np.trace(adjacency)),
       self_connections_per_hemisphere=int(np.trace(adjacency)),bidirectional_pairs_per_hemisphere=int(np.triu(adjacency&adjacency.T,1).sum()),
       episodic_slots=f["capacity"],context_weights=r["slots"],covariance_shape=[r["slots"],r["slots"]],
       input_width=20,outputs=2)
    gated=OUT/"object-content-v1/frozen.json"
    _write(folder/"readout-options.json",dict(params=json.loads(gated.read_text())["params"],
            source=audit.fingerprint(gated),purpose="optional identity and transformed read responsibility, actual sampled validation"))
    z=exact_bundle(folder,path,identity)
    checks=verify(z,path)
    _write(folder/"portable-verification.json",dict(identity=identity,topology=stats,
        archive_sha256=audit.fingerprint(z),archive_bytes=z.stat().st_size,weights_sha256=audit.fingerprint(path),
        scores=[dict(seed=s,validation=v) for v,s,_ in sorted(all_scores)],examples=checks,
        python=platform.python_version(),numpy=np.__version__))
    print(identity,stats,z,flush=True)


if __name__=="__main__":main()


