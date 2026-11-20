"""Let one real state run for different numbers of zero-input ticks and then ask both questions."""

import numpy as np
import torch

from . import native_rehearsal_eval as ev
from . import native_rehearsal_fit as fit
from . import native_rehearsal_graph as live
from . import noon_market_stats as stats


TIMES=(0,3,6,12,24,48)
SEED=27159371
ROOTS=192


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    scene=live.scenes(SEED,ROOTS,evaluation=True)
    angles=scene.angle.numpy().reshape(ROOTS,16)[:,0]
    agree=(scene.laws[:,0]==scene.laws[:,2]).numpy().reshape(ROOTS,16)
    outcome={}
    for kind in fit.KINDS:
        cohort={str(t):[] for t in TIMES}
        hashes={}
        for birth in fit.BIRTHS:
            model,path=ev.load(kind,birth)
            paid=[]
            for i in range(0,len(scene.laws),256):
                e=ev.subset(scene,i,i+256)
                _,s,_,_=live.support(model,e)
                zero=s.h.new_zeros((len(s.h),10))
                batch=[]
                for tick in range(max(TIMES)+1):
                    if tick in TIMES:
                        _,_,pays,_=live.queries(model,s,e)
                        batch.append(pays.to(torch.int8).numpy())
                    if tick!=max(TIMES):
                        s,_=model.tick(s,zero)
                paid.append(np.stack(batch,axis=1))
            paid=np.concatenate(paid,axis=0).reshape(ROOTS,16,len(TIMES),2)
            raw=fit.DATA/"wait"/f"{kind}-{birth}.npz"
            fit.storage._write_raw(raw,pay=paid,angles=angles,agree=agree,steps=TIMES,seed=SEED,
                                   source=fit.digest(__file__),body=fit.digest(path))
            hashes[raw.name]=fit.digest(raw)
            for j,t in enumerate(TIMES):
                a,b=np.moveaxis((paid[:,:,j,:]>0)[~agree].reshape(ROOTS,8,2),2,0)
                focus=np.isin(angles,(15,30))
                cohort[str(t)].append({name:x[focus].mean(1) for name,x in (("A",a),("B",b),("both",a&b))})
        banks={k:{m:np.stack([rep[m] for rep in vals]) for m in ("A","B","both")} for k,vals in cohort.items()}
        out=stats.summarize(banks,int(focus.sum()),fit.BIRTHS,SEED+57,[(str(t),"12") for t in TIMES if t!=12])
        outcome[kind]=dict(results=out,hashes=hashes)
        print(kind,[(t,round(100*out[str(t)]["both"]["mean"],1)) for t in TIMES],flush=True)
    fit.storage._write_json(fit.EVIDENCE/"wait.json",dict(seed=SEED,roots=ROOTS,close=int(focus.sum()),scores=outcome))


if __name__=="__main__":
    main()
