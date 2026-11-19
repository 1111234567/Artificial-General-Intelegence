"""One lifetime deciding when a learned habitat rule transfers and when new evidence overturns it."""
from dataclasses import dataclass,replace
import numpy as np

from . import habitat_world as habitats, mixture_world as food, world


@dataclass
class Episode:
    arena:habitats.MultiWorld
    checks:dict
    order:np.ndarray
    steps:tuple


def make(seeds,*,initial=12,depths=(0,2,4,8,12),span=2,noise=.04,jitter=.08,appearance="smooth",
         same_prior=.75,force=None,nonce=617751):
    seeds=list(map(int,seeds));n=len(seeds)
    if depths[0]!=0 or any(a>=b for a,b in zip(depths,depths[1:])):raise ValueError("Increasing evidence counts from zero")
    if any(v%span for v in depths) or initial%2:raise ValueError("A complete matched exposure group is required")
    # Underlying semantic context labels are randomly permuted physically by
    # the habitat codebook. The current/probe context pulses are the only cue.
    if force is None:
        rules=[]
        for s in seeds:
            rng=np.random.default_rng(np.random.SeedSequence([s,nonce,165]))
            a=int(rng.choice([-1,1]));same=rng.random()<same_prior
            rules.append([a,a if same else -a])
        rules=np.asarray(rules)
    else:rules=np.asarray(force)
    base=habitats.make(seeds,pairs=(initial,depths[-1]),visits=len(depths)+1,span=span,noise=noise,jitter=jitter,
                       relations=rules,appearance=appearance,novel_seed=nonce)
    teaching=base.start_test
    test_count=len(depths)+1
    length=2*(initial+depths[-1])+2*test_count
    order=np.empty((n,length),int)
    checks={**{f"B_after_{j}":2*initial+2*j+2*k+1 for k,j in enumerate(depths)},"return_A":length-1}
    for i in range(n):
        ctx=base.menu.body.ctx[i]
        tA=np.flatnonzero((np.arange(len(ctx))<teaching)&(ctx==0))
        tB=np.flatnonzero((np.arange(len(ctx))<teaching)&(ctx==1))
        probesA=np.flatnonzero(base.probe[i]&(ctx==0))
        probesB=np.flatnonzero(base.probe[i]&(ctx==1))
        seq=tA.tolist();start=0
        for k,depth in enumerate(depths):
            seq.extend(tB[2*start:2*depth].tolist())
            j=probesB[k];seq.extend([j-1,j]);start=depth
        j=probesA[0];seq.extend([j-1,j]);order[i]=seq
    row=np.arange(n)[:,None]
    b=base.menu.body
    def ordered(x):return x[row,order]
    body=replace(b,ctx=ordered(b.ctx),cue=ordered(b.cue),left_id=ordered(b.left_id),good_id=ordered(b.good_id),
                 wind=ordered(b.wind),distances=ordered(b.distances))
    menu=food.Foods(body,ordered(base.menu.chemicals),ordered(base.menu.probability),ordered(base.menu.realized),
                    np.empty(0),"share_or_separate",2*initial)
    first=np.broadcast_to([checks["return_A"],checks["B_after_0"]],(n,2)).copy()
    arena=habitats.MultiWorld(menu,base.tags,ordered(base.appearance),rules,first,ordered(base.probe),2*initial,
                              menu.probability.argmax(2))
    return Episode(arena,checks,order,tuple(depths))


def measures(episode,r):
    w=episode.arena
    correct=(r.contact_side>=0)&(r.contact_side==w.truths)
    last=episode.checks[f"B_after_{episode.steps[-1]}"]
    a=correct[:,episode.checks["return_A"]]
    out={name:correct[:,ix].astype(float) for name,ix in episode.checks.items()}
    out.update(both=(correct[:,last]&a).astype(float),coverage=(r.contact_side[:,list(episode.checks.values())]>=0).mean(1),
               B_initial_side=r.contact_side[:,episode.checks["B_after_0"]].astype(float),
               B_final_side=r.contact_side[:,last].astype(float),A_return_side=r.contact_side[:,episode.checks["return_A"]].astype(float))
    return out
