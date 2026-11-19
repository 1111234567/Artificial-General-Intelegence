"""Check all birth graphs, calibration selections and outcomes; replay one selected group."""
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import module_graph_study as study, module_graphs


def bridge_density(blocks):
    cube=np.stack(blocks)
    groups,_,_,roots=cube.shape
    seed=5569104
    rng=np.random.default_rng(seed)
    births=rng.multinomial(groups,np.full(groups,1/groups),20000)/groups
    worlds=rng.multinomial(roots,np.full(roots,1/roots),20000)/roots
    result={}
    for i,rule in enumerate(study.env.RULES):
        def score(arm):
            return cube[:,i,study.ARMS.index(arm)]
        module=score("modules64")-score("modules_min")
        single=score("single64")-score("single_min")
        for name,x in (("modular extra bridges",module),("single extra edges",single),("interaction",module-single)):
            draws=((births@x)*worlds).sum(axis=1)
            result[f"{rule}/{name}"]=dict(estimate=float(x.mean()),ci=np.quantile(draws,[.025,.975]).tolist())
    study.study.write(study.EVIDENCE/"bridge-density.json",dict(bootstrap_seed=seed,
          scope="Post hoc descriptive contrasts at 1200/standard, ordinary paired 95% intervals.",results=result))


def main():
    torch.set_num_threads(1)
    frozen=study.seal(); reg=study.study.read(study.BASE/"registered.json")
    graphs=study.study.read(study.BASE/"graphs.json")
    assert set(graphs)=={str(i) for pair in reg["births"] for i in pair}
    scc={}
    for key,count in (("modules64",64),("modules_min","minimum")):
        connected=0
        for seed,arms in graphs.items():
            record=arms[key]
            sampled=module_graphs.sample(int(seed),count)
            assert np.array_equal(record["internal_arcs"],sampled.within)
            assert np.array_equal(record["inter_ring_arcs"],sampled.bridges)
            assert min(record["ring_in_degree"])>=1 and min(record["ring_out_degree"])>=1
            assert np.all(np.bincount(np.array(sampled.within)[:,0],minlength=32)==1)
            assert np.all(np.bincount(np.array(sampled.within)[:,1],minlength=32)==1)
            connected+=int(sampled.record()["strongly_connected"])
        scc[key]=connected
    checks=0
    replay=0
    standard=[]
    for g,group in enumerate(frozen["groups"]):
        for record in group["records"]:
            chosen=np.asarray(record["selected"])
            assert np.array_equal(chosen,np.asarray(record["validation"]).argmax(0))
            j,k=np.indices(chosen.shape)
            picks=np.ravel_multi_index((chosen,j,k),(2,len(study.env.RULES),len(study.ARMS))).ravel()
            rows=[group["rows"][i] for i in picks]
            assert all(r[0]==g and r[2]==reg["births"][g][r[1]] and
                       r[3]==l//len(study.ARMS) and r[4]==study.ARMS[l%len(study.ARMS)] for l,r in enumerate(rows))
            if g==0:
                bank=study.make_bank(rows)
                with np.load(study.BASE/"checkpoints"/record["file"]) as params:
                    bank.load_state_dict({key:torch.from_numpy(params[key][picks]) for key in study.KEYS})
            with np.load(study.BASE/f"scores/group-{g:02}.npz") as arrays:
                for profile in ("standard","long"):
                    x=arrays[f"{record['updates']}/{profile}"]
                    side,taste=x[...,0],x[...,2]
                    assert np.array_equal(taste>0,(side>=0)&(side==study.env.truth()[:,None,None,:]))
                    assert np.array_equal(side<0,taste==0)
                    checks+=side.size
                    if profile=="standard" and record["updates"]==1200:
                        standard.append((taste>0).mean(axis=-1))
                    if g==0:
                        again=study.study.score(bank,rows,study.episode(profile))
                        assert np.array_equal(again.reshape(x.shape),x)
                        replay+=side.size
    archive=study.EVIDENCE/"raw.zip"
    summary=study.study.read(study.EVIDENCE/"summary.json")
    assert study.study.sha(archive)==summary["raw_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():
            assert z.read(n)==(study.BASE/n).read_bytes()
        members=len(z.namelist())
    bridge_density(standard)
    study.study.write(study.EVIDENCE/"verification.json",dict(connected=scc,checked_physical_outcomes=checks,
          replayed_bit_identical_outcomes=replay,archive_members=members,source_sha256=study.study.sha(Path(__file__))))
    print(scc,checks,replay,members)


if __name__=="__main__":
    main()
