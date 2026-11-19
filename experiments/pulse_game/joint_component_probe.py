"""After joint learning, probe the signed action readout and sensory/recurrent core separately."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from . import joint_live_model, live_edge_probe, live_module_edits_study as life
from . import ring_logic_study as io, ring_logic_world as sense


OUT = io.REPO / "docs/evidence/free-graph-2026-11-19/joint-components"
REG = io.read(OUT / "protocol.json")


def scorer(rows, reverse):
    seeds = REG["probe_roots"]
    _, x, _ = sense.sensory(seeds, "standard")
    rules = np.repeat([row[3] for row in rows],len(seeds))
    indices = np.tile(np.arange(len(seeds)),len(rows))
    return SimpleNamespace(rule=rules, root_index=indices, roots=np.asarray(seeds)[indices],
                           tape=sense.body_tape(seeds), direction=sense.truth()[rules]^reverse,
                           stimuli=x.reshape(-1,*x.shape[2:]))


def evaluate(source):
    initial, _, rows, _ = life.heads(life.TEST_GROUPS)
    seed_count=len(REG["probe_roots"])
    initial_flat={name:np.repeat(x,seed_count,axis=0) for name,x in initial.items()}
    raw_hashes, outcomes = {}, {}
    for condition in REG["conditions"]:
        path=source/f"{condition}.npz"
        raw_hashes[condition]=io.sha(path)
        with np.load(path,allow_pickle=False) as data:
            keys=("mask","raw","state","newborn_trial","newborn_raw","drive","bias","readout","offset")
            final={f"final_{k}":live_edge_probe.flatten_lives(data[f"joint/final_{k}"]) for k in keys}
            original_raw=np.repeat(data["initial_raw"],seed_count,axis=0)
        final["starting_gain"]=initial_flat["gain"]
        reference=np.where(final["final_mask"],np.where(final["final_newborn_trial"]>=0,
                           final["final_newborn_raw"],original_raw),0)
        assessor=scorer(rows,condition=="reversal")
        for arm in REG["arms"]:
            branch=dict(final)
            raw=final["final_raw"]
            if arm in ("restore_core","restore_both"):
                branch["final_drive"]=initial_flat["drive"]
                branch["final_bias"]=initial_flat["bias"]
                raw=reference
            if arm in ("restore_motor","restore_both"):
                branch["final_readout"]=initial_flat["readout"]
                branch["final_offset"]=initial_flat["offset"]
            result=live_edge_probe.score_probes(joint_live_model.Population,branch,assessor,raw)
            outcomes[condition+"/"+arm]=result.reshape(len(life.TEST_GROUPS),len(sense.RULES),seed_count,
                                                       len(sense.PAIRS),3)
    return raw_hashes, outcomes


def report(source):
    hashes,observed=evaluate(source)
    groups,_,roots,_,_=next(iter(observed.values())).shape
    rng=np.random.default_rng(190397)
    gb=rng.multinomial(groups,np.full(groups,1/groups),20000)/groups
    wb=rng.multinomial(roots,np.full(roots,1/roots),20000)/roots
    metrics={}
    for condition in REG["conditions"]:
        rates={name:(observed[condition+"/"+name][...,2]>0).mean(axis=(1,3)) for name in REG["arms"]}
        diff={}
        for arm in ("restore_motor","restore_core"):
            delta=rates["learned"]-rates[arm]
            draws=((gb@delta)*wb).sum(axis=1)
            diff[arm]=dict(mean=float(delta.mean()),ci=np.quantile(draws,[.025/6,1-.025/6]).tolist())
        metrics[condition]=dict(accuracy={k:float(v.mean()) for k,v in rates.items()},intact_minus_restored=diff)
    archive=OUT/"probes.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as out:
        for key,arr in observed.items():
            buffer=BytesIO()
            np.save(buffer,arr,allow_pickle=False)
            out.writestr(key+".npy",buffer.getvalue())
    io.write(OUT/"summary.json",dict(results=metrics,source_run=source.name,source_npz=hashes,
             protocol_sha256=io.sha(OUT/"protocol.json"), code_sha256=io.sha(Path(__file__)),
             probe_code_sha256=io.sha(Path(live_edge_probe.__file__)),archive_sha256=io.sha(archive)))
    print(metrics)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    report(parser.parse_args().source)
