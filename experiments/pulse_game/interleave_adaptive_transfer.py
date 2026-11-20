"""Deploy the three fitted constant gains unchanged in independent mixed, 256-contact lives."""

from hashlib import sha256
import json

import numpy as np
import torch

from . import interleave_integrated_stability as evaluation, interleave_stability_report as paired
from . import interleave_stream as stream
from .interleave_interference import DATA, EVIDENCE


def run():
    torch.set_num_threads(1)
    fitted=json.loads((EVIDENCE/"adaptive-reference.json").read_text())
    coefficients=[v[0][0] for v in fitted["coefficients"].values()]
    gains=[float(torch.tensor(v,dtype=torch.float32).sigmoid()) for v in coefficients]
    branches=[("fixed",.12,.02),*[(f"fit{i}",gain,.02) for i,gain in enumerate(gains)]]
    results={};hashes={}
    for name,source in (("joint",stream.Source("joint",DATA/"joint-continuation-replication/full-direct","local_active",8000)),
                        ("uniform",stream.Source("objective",DATA/"retention-objective-confirmation","uniform",8000))):
        old_folder=DATA/"integrated-stability"/(name+"-confirmation")
        destination=DATA/"adaptive-reference"/"deployment"/name
        destination.mkdir(parents=True,exist_ok=True)
        groups=[]
        for b in range(811,817):
            f=destination/f"{b}.npz"
            if not f.exists():
                g,r=source.load(b)
                seed=9750121
                tape=stream.tape_for_worlds(b,64,max(evaluation.BLOCKS),seed)
                raw=evaluation.evaluate(g,r,tape,b,seed,evaluation.BLOCKS,branches)
                with np.load(old_folder/f"{b}.npz") as z:
                    for key in ("probe","earned","distance"):
                        np.testing.assert_array_equal(raw["fixed/"+key],z["slow/"+key])
                np.savez_compressed(f,**raw)
            hashes[name+"/"+f.name]=sha256(f.read_bytes()).hexdigest()
            values={}
            with np.load(f) as z:
                for label,_,_ in branches:
                    for key in ("fresh","old_after_fresh","ctx_both_mark","natural_quiz"):
                        ag=[]
                        for block in paired.STAGES:
                            cut=evaluation.BLOCKS.index(block)
                            if key=="natural_quiz":
                                lo=4*evaluation.BLOCKS[cut-1];hi=4*block
                                take=np.arange(lo,hi)%4>=2
                                v=z[label+"/earned"][:,lo:hi][:,take].mean(1)
                            else:
                                v=z[label+"/probe"][cut,evaluation.PROBES.index(key)].mean(1)
                            if key=="ctx_both_mark":
                                mask=z["ctx_conflict"]&(z["context_firsts"]<4*block).all(1)
                                v=np.where(mask,v,np.nan)
                            ag.append(v.reshape(-1,2))
                        values[label+"/"+key]=np.stack(ag)
            groups.append(values)
            print(name,b,flush=True)
        rng=np.random.default_rng(942817)
        nboot=4000
        bw=rng.integers(6,size=(nboot,6))
        ww=rng.multinomial(64,np.full(64,1/64),size=(6,nboot))
        sw=rng.multinomial(3,np.full(3,1/3),size=nboot)/3
        scores={}
        for key in ("fresh","old_after_fresh","ctx_both_mark","natural_quiz"):
            for comp in ("fixed","fitted","fitted minus fixed"):
                per_seed=[];means=[]
                for s in range(3):
                    boot_roots=[];points=[]
                    for b,bank in enumerate(groups):
                        fixed=bank["fixed/"+key];learned=bank[f"fit{s}/"+key]
                        z=fixed if comp=="fixed" else learned if comp=="fitted" else learned-fixed
                        numerator=ww[b]@np.nan_to_num(z).sum(2).T
                        denominator=ww[b]@np.isfinite(z).sum(2).T
                        if (denominator==0).any():raise ValueError("Empty deployment stratum")
                        boot_roots.append((numerator/denominator).mean(1))
                        points.append(np.nanmean(z,axis=(1,2)).mean())
                    per_seed.append(np.stack(boot_roots)[bw,np.arange(nboot)[:,None]].mean(1))
                    means.append(float(np.mean(points)))
                sampled=(np.stack(per_seed,1)*sw).sum(1)
                scores[comp+"/"+key]=dict(mean=float(np.mean(means)),by_external_seed=means,
                                          ci95=np.quantile(sampled,[.025,.975]).tolist())
        results[name]=scores
    out=dict(gains=gains,cohort="811–816",calibration="none on these tasks, learned rates used unchanged",
             ages_in_contacts=[64,128,256],results=results,raw_sha256=hashes,
             notes="Unchanged beta .02 for fixed and learned constants. Use each fitted scalar logit with its "
                   "ordinary sigmoid. Rare conflict probes use only histories where the three cues have been seen. "
                   "Three outer fitting seeds, six deployment bodies, per-body independent world roots; paired "
                   "private sign lanes remain together. Fixed comparator reproduces prior mixed-life archive.")
    (EVIDENCE/"adaptive-deployment.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    run()
