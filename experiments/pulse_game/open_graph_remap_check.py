"""Check paired remapping and a public-odor, last-contact ideal reference."""
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from . import open_graph_remap as study, world


def run_reference(parts, rows, twin):
    """The ideal reader knows the public pulse alphabet, but not the private food values."""
    tape,observations,lanes,_,_,_=study.scoring_tape(parts,rows,twin)
    body=world.PulseWorld(tape,feedback="true")
    values=np.zeros((len(lanes),len(study.sensory.ODORS),len(study.sensory.ODORS)),np.int8)
    idx=np.arange(len(lanes))
    for trial in range(tape.trials):
        pulse=observations[:,trial]
        bright=pulse[:,:,:3][pulse[:,:,3]>.5].reshape(-1,2,3)
        identified=(bright@study.sensory.ODORS.T).argmax(-1)
        first,second=identified[lanes].T
        choice=values[idx,first,second]>0
        move=np.stack((~choice,choice),axis=-1)
        for _ in range(world.TICKS):
            side,taste=study.life.receipt(body.observe())
            delivered=(side>=0)&(taste!=0)
            values[idx[delivered],first[delivered],second[delivered]]=(2*side[delivered]-1)*taste[delivered]
            body.step(move)
    result=body.result()
    return np.stack((result.contact_side,result.contact_tick,result.reward),-1).reshape(-1,3,len(parts[3]),tape.trials,3)


def main():
    manifest=study.io.read(study.EVIDENCE/"summary.json")
    assert manifest["source_sha256"]==study.io.sha(Path(study.__file__))
    assert manifest["raw_sha256"]==study.io.sha(study.EVIDENCE/"remap-raw.zip")
    with zipfile.ZipFile(study.EVIDENCE/"remap-raw.zip") as z:
        assert z.testzip() is None
        for name in z.namelist(): assert z.read(name)==(study.DATA/name).read_bytes()
    head=study.life.heads(tuple(study.REG["groups"]))
    roots=list(range(study.REG["world_start"],study.REG["world_start"]+study.REG["world_count"]))
    parts=study.design(roots,head[2])
    cases=[run_reference(parts,head[2],tag) for tag in (0,1)]
    both=study.matched_twins(cases,parts)
    observed=[study.summarize_arm(x,parts,head[2],tag) for tag,x in enumerate(cases)]
    rates={k:float(np.mean([r[k] for r in observed])) for k in observed[0]}
    count=0
    for arm in study.REG["conditions"]:
        with np.load(study.DATA/f"{arm}.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["mask"],parts[6])
            for tag in (0,1):
                x=raw[f"physical_{tag}"]
                tape=study.scoring_tape(parts,head[2],tag)[0]
                truth=(tape.good_id!=tape.left_id).reshape(x.shape[:-1])
                side,tick,taste=x[...,0],x[...,1],x[...,2]
                assert np.array_equal(taste>0,(side==truth)&(side>=0))
                assert np.array_equal(taste==0,side<0)
                assert np.array_equal(tick<0,side<0)
                count+=side.size
            if arm=="placebo":
                np.testing.assert_array_equal(raw["physical_0"][...,:2],raw["physical_1"][...,:2])
    study.io.write(study.EVIDENCE/"checks.json",dict(public_reader=rates,paired_both_B3=float(both.mean()),
                  all_opportunities_checked=int(count),placebo_side_and_tick_identical_across_opposing_truths=True,
                  source_sha256=study.io.sha(Path(__file__))))
    print(f"Checked {count} physical outcomes, B3 public reader {rates['B3']:.4f}.")


if __name__=="__main__": main()
