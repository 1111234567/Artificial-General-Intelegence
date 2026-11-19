"""Out-of-training-family object discrimination, tested without any supplied problem identity."""
from dataclasses import replace
import numpy as np
from . import habitat_world as habitats, mixture_world as food, symmetry_world as symbol,world


def make(seeds,problems=40,*,noise=.0,variable=False,nonce=819093):
    n=len(seeds)
    lengths=np.array([6]*problems) if not variable else np.random.default_rng(nonce).integers(3,10,size=problems)
    size=int(lengths.sum());ids=np.arange(n)[:,None]
    base=world.make_tape(seeds,size,distance_jitter=.13,wind_scale=.07)
    chem=np.empty((n,size,2,4),np.float32)
    actual=np.empty((n,size,2),np.int8);target=np.empty_like(actual,dtype=np.float32)
    habitat=np.empty((n,2,2),np.float32);ctx=np.zeros((n,size),np.int8)
    for i,seed in enumerate(seeds):
        rng,taste,flip=(np.random.default_rng(s) for s in np.random.SeedSequence([seed,nonce,559]).spawn(3))
        a=rng.uniform(0,.8);angles=np.array([a,a+.55])
        habitat[i]=np.column_stack((np.cos(angles),np.sin(angles)))
        items=symbol.unit(rng.normal(size=(problems,2,4)))
        good=taste.integers(2,size=problems);which=rng.integers(2,size=size)
        problem=np.repeat(np.arange(problems),lengths);j=np.arange(size)
        chem[i,:,0]=items[problem,which];chem[i,:,1]=items[problem,1-which]
        target[i,:,0]=which==good[problem];target[i,:,1]=1.-target[i,:,0]
        actual[i]=np.where(target[i]>0,1,-1)
        # Separate outcomes of the two foods. Only one can be encountered.
        actual[i]*=np.where(flip.random((size,2))<noise,-1,1).astype(np.int8)
    tape=replace(base,ctx=ctx)
    menu=food.Foods(tape,(chem+1.)*.5,target,actual,np.empty(0),"object_learning_set",0)
    w=habitats.MultiWorld(menu,habitat,habitat[ids,ctx],np.zeros((n,2),int),np.zeros((n,2),int),
                         np.zeros((n,size),bool),0,target.argmax(2))
    return w,lengths


def metrics(w,lengths,r):
    chosen=r.contact_side;hit=chosen>=0;correct=(chosen==w.truths)&hit
    starts=np.r_[0,np.cumsum(lengths)[:-1]];n=len(chosen)
    res={f"trial_{i+1}":correct[:,starts[lengths>i]+i].mean(1).tolist() for i in range(min(6,max(lengths)))}
    early=starts[:max(4,len(starts)//4)];late=starts[-max(4,len(starts)//4):]
    res.update(early_second=correct[:,early+1].mean(1).tolist(),late_second=correct[:,late+1].mean(1).tolist(),
               all=correct.mean(1).tolist(),contact=hit.mean(1).tolist())
    y=r.reward[:,starts]
    for sign in (-1,1):
        m=y==sign
        res[f"second_after_{sign}"]={"hit":(correct[:,starts+1]&m).sum(1).tolist(),"opportunities":m.sum(1).tolist()}
    return res
