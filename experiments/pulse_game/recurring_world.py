"""Retrospective probes after recurring, novel, noisy or partly conflicting physical tastes."""
import numpy as np
from . import symmetry_audit as paired, symmetry_world as original


def make(seeds,*,gap=32,distinct=8,query="random",recurring=True,spread=.06,noise=.10,
         conflicts=False,pairs=12,probes=2,nonce=543729):
    if query not in ("anchor","random","rare"):raise ValueError(query)
    n=len(seeds);seeds=list(map(int,seeds))
    w=original.with_distractors(paired.joint(seeds,pairs=pairs,probes=probes,span=4,noise=.04,jitter=.08,matched=True),gap)
    if distinct<1 or distinct>gap:raise ValueError("Need at least one appearance per catalogue item")
    old=w.menu;cut=w.exposed
    records=[]
    for row,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,nonce,12727]))
        labels=[]
        for z in range(probes):
            first=cut+z*(gap+2)
            j=first+gap+1
            center=original.unit(rng.normal(size=(distinct,4)))
            if conflicts and distinct>=4:
                # Two odors are near one another but can have distinct tastes.
                center[1]=original.unit(center[0]+.14*rng.normal(size=4))
            values=rng.choice([-1,1],size=distinct)
            if conflicts:values[1]=-values[0]
            order=np.r_[np.arange(distinct),rng.integers(distinct,size=gap-distinct)]
            rng.shuffle(order)
            chosen_unique=original.unit(rng.normal(size=(gap,4)))
            perturb=rng.normal(size=(gap,4))
            coords=original.unit(center[order]+spread*perturb) if recurring else chosen_unique
            signed=values[order];feedback=np.where(rng.random(gap)<noise,-signed,signed)
            for branch in (0,1):
                i=row+branch*n
                interval=slice(first+1,j)
                old.chemicals[i,interval]=((coords+1.)*.5)[:,None,:]
                old.realized[i,interval]=feedback[:,None]
                old.probability[i,interval]=(signed[:,None]>0)
            # The item is selected after the sequence has been generated, without
            # any advance cue about what will be tested. In the novel condition
            # selection is uniform over distinct experiences; in the recurring
            # condition it is uniform over identities (including the anchor).
            used=rng.choice(np.r_[-1,np.arange(distinct if recurring else gap)])
            if query=="anchor":used=-1
            elif query=="rare":
                counts=np.bincount(order,minlength=distinct)
                if recurring:used=int(rng.choice(np.flatnonzero(counts==counts.min())))
                else:used=int(rng.integers(gap))
            if used<0:
                x=old.chemicals[row,first,0]*2.-1.
                latent=int(1 if old.probability[row,first,0] else -1)
            elif recurring:x=center[used];latent=int(values[used])
            else:x=chosen_unique[used];latent=int(signed[used])
            side=int(w.related_side[row,j])
            antipode=original.unit(-x+.075*rng.normal(size=4))
            distract=original.unit(rng.normal(size=4))
            for branch in (0,1):
                i=row+branch*n;f=w.family[i];goal=int(latent*f)
                old.chemicals[i,j,side]=(antipode+1.)*.5;old.chemicals[i,j,1-side]=(distract+1.)*.5
                old.realized[i,j,side]=goal;old.realized[i,j,1-side]=-goal
                old.probability[i,j,side]=goal>0;old.probability[i,j,1-side]=goal<0
                w.anchor_sign[i,j]=latent
            labels.append(int(used))
        records.append(labels)
    return w,np.array(records)
