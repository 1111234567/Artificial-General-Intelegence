"""Taste-rule transfer across habitats, with random signs on never-repeated odor pairs.

A transformed odor has the same or opposite taste across both habitats in this
life. Absolute tastes are independent even for homologous cues in different
habitats. Every outcome depends on the physically reached food; no family bit,
probe flag, bite target, transformed-side label or phase index is observed.
"""
from dataclasses import dataclass, replace
import numpy as np

from . import mixture_world as foods, mixture_flash as flash, world


@dataclass
class Protocol:
    menu: foods.Foods
    family: np.ndarray
    related_side: np.ndarray
    query: np.ndarray
    anchor_sign: np.ndarray
    pairmates: np.ndarray
    exposed: int


def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)


def make(seeds,*,pairs=12,probes=8,interleave=1,noise=.04,jitter=.08,family=None,permute_signs=False,
         association=1.,target_flip=False,balanced_stream=False):
    seeds=tuple(map(int,seeds))
    if pairs<1 or probes<1 or not 1<=interleave<=pairs:raise ValueError("Invalid exposure schedule")
    count=2*(pairs+probes)
    body=world.make_tape(seeds,count,distance_jitter=.13,wind_scale=.07)
    shapes=(len(seeds),count)
    chem=np.empty((*shapes,2,4),np.float32)
    sign=np.empty((*shapes,2),np.int8)
    prob=np.empty_like(sign,dtype=np.float32)
    ctx=np.zeros(shapes,np.int8)
    kind=np.empty(len(seeds),np.int8)
    related=np.full(shapes,-1,np.int8);query=np.zeros(shapes,bool)
    anchor=np.zeros(shapes,np.int8);mates=np.full(shapes,-1,np.int16)
    for row,seed in enumerate(seeds):
        rmenu,rtruth,rnoise=(np.random.default_rng(s) for s in np.random.SeedSequence([seed,0xF77A0]).spawn(3))
        original_family=rtruth.choice([-1,1])
        route=original_family if family is None else (family if np.ndim(family)==0 else family[row])
        kind[row]=route
        habitat=rmenu.integers(2)
        ctx[row,:2*pairs]=habitat;ctx[row,2*pairs:]=1-habitat
        index=0
        for start in range(0,pairs,interleave):
            span=min(interleave,pairs-start)
            scents=unit(rmenu.normal(size=(span,4)))
            tastes=rtruth.choice([-1,1],size=span)
            if balanced_stream:
                if span%2 or association!=1.:
                    raise ValueError("A stream-matched group needs equal sign counts and exact latent symmetry")
                tastes=rtruth.permutation(np.tile([-1,1],span//2))
                replay_sign=rtruth.permutation(np.tile([-1,1],span//2))
                transformed=unit(-scents+jitter*rmenu.normal(size=scents.shape))
            # Generate random numbers even when the route is overridden; all
            # future geometry and noise stay identical under family intervention.
            decoys=rtruth.choice([-1,1],size=span)
            reliable=rtruth.random(span)<association
            matches=np.where(reliable,route,decoys)
            for i,(x,y) in enumerate(zip(scents,tastes,strict=True)):
                chem[row,index+i]=x
                sign[row,index+i]=y
            order=rmenu.permutation(span)
            if balanced_stream:
                buckets={s:list(order[tastes[order]==s]) for s in (-1,1)}
                order=[buckets[s*route].pop() for s in replay_sign]
            for offset,i in enumerate(order):
                j=index+span+offset
                related_x=transformed[i] if balanced_stream else unit(-scents[i]+jitter*rmenu.normal(size=4))
                chem[row,j]=related_x
                sign[row,j]=tastes[i]*matches[i]
                mates[row,j]=index+i
            index+=2*span
        if permute_signs:
            # Break the paired association while preserving the whole evidence
            # multiset in each life, on exactly the same physical odors.
            sign[row,:index]=sign[row,np.random.default_rng(np.random.SeedSequence([seed, 937515])).permutation(index)].copy()
        for k in range(probes):
            x=unit(rmenu.normal(size=4))
            y=rtruth.choice([-1,1])
            # Related food is a previously unseen noisy transformation. The
            # alternative is independent of tastes and randomly placed.
            which=int(rmenu.integers(2))
            z=unit(rmenu.normal(size=4))
            other=unit(-x+jitter*rmenu.normal(size=4))
            chem[row,index]=x;sign[row,index]=y
            j=index+1
            chem[row,j,which]=other;chem[row,j,1-which]=z
            target=y*route*(-1 if target_flip else 1)
            sign[row,j,which]=target;sign[row,j,1-which]=-target
            related[row,j]=which;query[row,j]=True
            anchor[row,j]=y;mates[row,j]=index
            index+=2
        prob[row]=(sign[row]>0).astype(np.float32)
        # Report noisy taste in exposure bouts and test anchors. The unknown
        # target in the query is deterministic, independent of this sensor noise.
        corrupt=(rnoise.random(count)<noise)&~query[row]
        sign[row,corrupt]*=-1
    physical=replace(body,ctx=ctx)
    menu=foods.Foods(physical,(chem+1.)*.5,prob,sign,np.empty(0),"abstract_symmetry",2*pairs)
    return Protocol(menu,kind,related,query,anchor,mates,2*pairs)


def metrics(protocol,result):
    mask=protocol.query
    side=result.contact_side
    hit=side>=0
    goal=protocol.menu.probability.argmax(2)
    signed=(side==goal)&hit
    count=len(side)
    correct=signed[mask].reshape(count,-1)
    related=(side==protocol.related_side)[mask].reshape(count,-1)
    expect=(goal==protocol.related_side)[mask].reshape(count,-1)
    return dict(first=correct[:,0].astype(float),all=correct.mean(1),last_four=correct[:,-4:].mean(1),
                contact=hit[mask].reshape(count,-1).mean(1),first_related=related[:,0].astype(float),
                first_should_approach=expect[:,0].astype(float),family=protocol.family.astype(float),
                first_signed=result.reward[:,np.flatnonzero(protocol.query[0])[0]].astype(float))


def public_rows(protocol,result):
    """Pack public pulses and actual bites. Offline indicators are excluded from this API."""
    tr=protocol.menu.body.trials
    if (result.contact_side<0).any():raise ValueError("Missing explorer contact")
    xy=result.obs[:,:,:,foods.LEFT.start:foods.RIGHT.stop].reshape(len(result.seeds),tr,world.TICKS,2,4)*2.-1.
    ctx=result.obs[:,:,:,foods.CTX]
    ot=xy.any((3,4)).argmax(2)
    ct=ctx.any(3).argmax(2)
    if not np.all((xy.any((3,4)).sum(2)==1)&(ctx.any(3).sum(2)==1)):raise ValueError("Expected pulse pair")
    return dict(odor=xy.sum(2),ctx=ctx.sum(2),ot=ot.astype(np.int8),ct=ct.astype(np.int8),
                side=result.contact_side,touch=result.contact_tick,reward=result.observed_reward.sum(2).astype(np.float32))



def with_distractors(original, gap, *, nonce=517733, same_habitat=True):
    """Insert real unsigned-by-design-independent foods; each physical touch still yields a sign."""
    if gap==0:return original
    if gap<0:raise ValueError(gap)
    old=original.menu;seeds=old.body.seeds;n=len(seeds)
    order=[]
    for j,probe in enumerate(original.query[0]):
        if probe:order.extend([-1]*gap)
        order.append(j)
    order=np.asarray(order)
    kept=order>=0
    src=order[kept]
    extra=~kept
    trials=len(order)
    # Original requests keep their geometry across delays. New physical
    # requests receive independent wind/geometry with the same distribution.
    ss=[int(np.random.SeedSequence([s,nonce,79]).generate_state(1)[0]) for s in seeds]
    new_body=world.make_tape(ss,trials,distance_jitter=.13,wind_scale=.07)
    distances=new_body.distances.copy();wind=new_body.wind.copy()
    distances[:,kept]=old.body.distances[:,src];wind[:,kept]=old.body.wind[:,src]
    ctx=np.zeros((n,trials),np.int8);ctx[:,kept]=old.body.ctx[:,src]
    ctx[:,extra]=old.body.ctx[:,original.exposed,None] if same_habitat else 1-old.body.ctx[:,original.exposed,None]
    new_body=replace(new_body,seeds=seeds,ctx=ctx,distances=distances,wind=wind)
    xy=np.zeros((n,trials,2,4),np.float32);sgn=np.empty((n,trials,2),np.int8)
    pr=np.zeros((n,trials,2),np.float32)
    xy[:,kept]=old.chemicals[:,src];sgn[:,kept]=old.realized[:,src];pr[:,kept]=old.probability[:,src]
    for i,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,nonce,440833]))
        count=int(extra.sum());z=unit(rng.normal(size=(count,4)))
        xy[i,extra]=((z+1.)*.5)[:,None,:]
        signs=rng.choice([-1,1],size=count)
        sgn[i,extra]=signs[:,None];pr[i,extra]=(signs[:,None]>0)
    query=np.zeros((n,trials),bool);query[:,kept]=original.query[:,src]
    side=np.full((n,trials),-1,np.int8);side[:,kept]=original.related_side[:,src]
    anchor=np.zeros((n,trials),np.int8);anchor[:,kept]=original.anchor_sign[:,src]
    mapping=np.flatnonzero(kept)
    mates=np.full((n,trials),-1,np.int16)
    old_m=original.pairmates[:,src];mates[:,kept]=np.where(old_m>=0,mapping[old_m.clip(min=0)],-1)
    menu=foods.Foods(new_body,xy,pr,sgn,np.empty(0),old.rule,old.counter)
    return Protocol(menu,original.family.copy(),side,query,anchor,mates,original.exposed)

