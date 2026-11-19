"""Matched, externally seeded closed-loop tests on never-repeated continuous foods."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import time

import numpy as np
from experiments.seed_evolution.large_runner import _write
from . import mixture_brain as brain, mixture_world as game, research_audit as audit
from .attractor_search import OUT


def evaluate(spec):
    family, distribution, candidate, graph, seeds, trials, switches, folder = spec
    label = f"{family}-{distribution}-{candidate}-{graph}"
    path = folder / (label+".json")
    if path.is_file():
        return
    menu = game.make(seeds, trials, rule=family, distribution=distribution)
    if candidate.startswith("model/"):
        raise ValueError("Use checkpoint stem without path")
    if candidate.startswith(("dense-", "chemical_product-", "compartment-")):
        from . import mixture_metric as metric
        modelpath = OUT/"mixture-meta-v1"/(candidate+".pt")
        params = metric.load(modelpath)
        checkpoint = audit.fingerprint(modelpath)
        def factory(writes=True, masked=False):
            return metric.LearnedForager(params,graph,seeds,learn=writes,mask_context=masked)
    else:
        checkpoint = None
        def factory(writes=True,masked=False):
            return brain.MixtureLearner(replace(brain.Rule(),representation=candidate, rate=.88*int(writes)),
                                        graph,seeds,mask_context=masked)
    true = game.rollout(factory(),menu)
    rows = {"true":game.metrics(menu,true),
            "silent_second":game.metrics(menu,game.rollout(factory(),menu,hidden_after=menu.counter))}
    if switches:
        rows.update(no_write=game.metrics(menu,game.rollout(factory(False),menu)),
                    context_hidden=game.metrics(menu,game.rollout(factory(masked=True),menu)),
                    other_life=game.metrics(menu,game.rollout(audit.ContactMatched(factory(),true.reward[np.arange(len(seeds))^1]),menu)))
        if candidate == "spikes":
            cut = brain.MixtureLearner(brain.Rule(),graph,seeds,cut=True)
            rows["cut_recurrence"] = game.metrics(menu,game.rollout(cut, menu))
    _write(path,dict(candidate=candidate, source=checkpoint, rule=family,distribution=distribution,
                     graph=graph,seeds=seeds,trials=trials,rows={k:{m:v.tolist() for m,v in d.items()} for k,d in rows.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--worlds",type=int,default=128)
    args=p.parse_args()
    folder=audit.OUT/"mixture-visible-independent"
    folder.mkdir(exist_ok=True)
    seeds=[int(np.random.SeedSequence([s,43099681]).generate_state(1)[0]) for s in audit.split("replication")[:args.worlds]]
    models=[p.stem for p in sorted((OUT/"mixture-meta-v1").glob("*.pt"))]
    if len(models)!=6:raise RuntimeError("Expected both predeclared seeds for each of three neural designs")
    jobs=[]
    for family in ("linear","interaction","mixed"):
        for dist in ("continuous","components","disjoint_octants"):
            for model in models+["linear_prior","quadratic_prior","additive","spikes"]:
                graphs=(17,73) if model=="spikes" else (17,)
                for graph in graphs:
                    controls=(model in ("spikes","quadratic_prior","compartment-s3361") and dist=="continuous")
                    jobs.append((family,dist,model,graph,seeds,128,controls,folder))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(evaluate,job) for job in jobs]
        for i,f in enumerate(as_completed(tasks),1):
            f.result()
            if i%12==0:print(i,len(tasks),round(time.perf_counter()-start),"s",flush=True)
    print("complete",len(jobs),flush=True)


if __name__=="__main__":main()
