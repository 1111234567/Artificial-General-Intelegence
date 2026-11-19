"""Matched end-to-end training of isolated leaky sensory cells versus directed graph interactions."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import time
import numpy as np
import torch
from . import independent_births as original,unified_train as teach,unified_evolution as evolve,habitat_meta as routes
from . import symmetry_grounded as front,symmetry_memory as structure,mixture_circuit_train as tensor
from . import unified_world as task,unified_physical as physics,adaptive_local_gpu as arms
from . import unified_audit as setting,unified_confirm as score,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def make_diagonal(src,dest,seed,reg):
    rng=np.random.default_rng(seed);sensor_seed,graph,policy_seed,learn_seed,train_seed,valid_seed=rng.integers(1,400_000_000,6).tolist()
    s=reg["training"];cfg=structure.Config(graph_seed=graph,graph="diagonal");receptor=dest/"receptor.pt"
    model=f"rls-learned-s{learn_seed}";path=dest/(model+".pt")
    d=teach.raw(src,train_seed,s["lives"]);v=teach.raw(src,valid_seed,800)
    if not receptor.exists():
        torch.manual_seed(sensor_seed);net=front.Grounded(cfg).cuda()
        error,secs=front.public_fit(net,tensor.torch_data(d,"cuda"),360,"cuda",sensor_seed)
        matrix,sv,cross=front.measured_transform(d,20)
        with torch.no_grad():net.transform.copy_(torch.from_numpy(matrix).cuda())
        torch.save(dict(spec=asdict(cfg),state=net.state_dict(),step=360,seed=sensor_seed,
                        graph_seed=graph,reconstruction_training_mse=error),receptor)
    if not path.exists() or json.loads((dest/(model+"-history.json")).read_text())[-1]["step"]!=s["steps"]:
        net=routes.receptor(receptor)
        z=teach.precompute(net,d,"cuda");val=teach.precompute(net,v,"cuda")
        teach.train(z,val,learn_seed,routes.Config(routing="learned",updater="rls"),
                    SimpleNamespace(steps=s["steps"],device="cuda"),dest,receptor)
        del net,z,val
    del d,v
    res=original.resources(dest)
    evolve.optimize(dest,model,policy_seed,"full",SimpleNamespace(width=s["width"],gens=s["gens"],lives=80),
                    frozen_inputs=res,model_path=path)


def configuration(folder):
    f,r,m=original.resources(folder)
    frozen=next(folder.glob("*--full-*/frozen.json"));v=json.loads(frozen.read_text())["best"]["genome"]
    return f,r,m,physics.phenotype(physics.params(f,r,m),v,"full")[0],frozen


def verify_pair(src,compare,seed,alpha,roots,out):
    full=configuration(src);diagonal=configuration(compare)
    h1=json.loads(full[-1].with_name("checkpoint.json").read_text())["history"]
    h2=json.loads(diagonal[-1].with_name("checkpoint.json").read_text())["history"]
    if len(h1)!=len(h2) or any(a["roots"]!=b["roots"] or a["environment"]!=b["environment"] for a,b in zip(h1,h2,strict=True)):
        raise RuntimeError("The evolutionary worlds are not paired")
    f,r,m,v,_=full;matrix=f["matrix"];perm=np.random.default_rng(seed+90511).permutation(len(matrix))
    variants={
        "intact":(f,r,m,v),"independently_trained_diagonal":diagonal[:4],
        "remove_only_cross_edges":({**f,"matrix":np.diag(matrix.diagonal())},r,m,v),
        "diagonal_norm_matched":({**f,"matrix":np.diag(np.sign(matrix.diagonal())*np.linalg.norm(matrix,axis=0))},r,m,v),
        "rewire_keep_spectrum":({**f,"matrix":matrix[np.ix_(perm,perm)]},r,m,v),
        "remove_all_recurrence":({**f,"matrix":np.zeros_like(matrix)},r,m,v)}
    n=len(roots);families=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0);table={}
    for kind in ("standard","long_competing_run","similar_habitats"):
        ep=task.make(roots*4,force=families,nonce=4938673,**setting.SETTINGS[kind])
        table[kind]={}
        for name,(ff,rr,mm,par) in variants.items():
            arch=arms.Cohort(ep,ff,rr,mm)
            answer=arch.play(par,alpha)
            table[kind][name]=score.data(ep,answer,n)
    _write(out,dict(seed=seed,worlds=roots,rows=table,training_lifetimes=4800,evolution_width=1024,generations=len(h1),
              trained_sources=[audit.fingerprint(x[-1]) for x in (full,diagonal)],paired_search_worlds=True))
    print("matched birth confirmed",seed,flush=True)


def main():
    torch.set_num_threads(1);base=OUT/"independent-lifelong-births-v1";reg=json.loads((base/"registered.json").read_text())
    directory=OUT/"matched-diagonal-births";directory.mkdir(exist_ok=True)
    dest=audit.OUT/"matched-sensory-topology";dest.mkdir(exist_ok=True)
    seeds=reg["fixed_birth_seeds"][:6]
    worlds=np.random.default_rng(21789551).integers(8_000_000,475_000_000,256).tolist()
    _write(directory/"registered.json",dict(seeds=seeds,worlds=worlds,settings=reg["training"],parent_registration=audit.fingerprint(base/"registered.json")))
    for seed in seeds:
        src=base/f"birth-{seed}";target=directory/src.name;target.mkdir(exist_ok=True)
        output=dest/(src.name+".json")
        if output.exists():continue
        make_diagonal(src,target,seed,reg)
        verify_pair(src,target,seed,reg["reopening_rule"],worlds,output)
    _write(dest/"complete.json",dict(seeds=seeds,worlds=worlds))


if __name__=="__main__":main()
