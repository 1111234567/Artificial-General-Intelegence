"""Fresh anchor tastes followed by real choices without additional relation teaching."""
from dataclasses import replace
import numpy as np
from . import habitat_world as habits, mixture_world as tastes, symmetry_world as signed, world


def probe(seeds,tags,rules,*,flip=False,rounds=8):
    n=len(seeds);count=2*rounds;base=world.make_tape(seeds,count,distance_jitter=.13,wind_scale=.07)
    ctx=np.zeros((n,count),np.int8);x=np.zeros((n,count,2,4),np.float32)
    y=np.zeros((n,count,2),np.int8);prob=np.zeros_like(y,np.float32)
    for i,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,187351]))
        visits=np.concatenate([rng.permutation(2) for _ in range(rounds//2)])
        ctx[i]=np.repeat(visits,2)
        odor=signed.unit(rng.normal(size=(rounds,4)))
        other=signed.unit(rng.normal(size=(rounds,4)));anti=signed.unit(-odor+.08*rng.normal(size=odor.shape))
        side=rng.integers(2,size=rounds);sign=rng.choice([-1,1],size=rounds)
        j=2*np.arange(rounds)
        x[i,j]=odor[:,None,:];y[i,j]=sign[:,None]
        x[i,j+1,side]=anti;x[i,j+1,1-side]=other
        latent=rules[i,visits]*np.where((visits==1)&flip,-1,1)
        y[i,j+1,side]=sign*latent;y[i,j+1,1-side]=-sign*latent
        prob[i]=(y[i]>0)
        noise=rng.random(rounds)<.05;y[i,j]*=np.where(noise,-1,1)[:,None].astype(np.int8)
    body=replace(base,ctx=ctx);menu=tastes.Foods(body,(x+1.)*.5,prob,y,np.empty(0),"recall_without_instruction",0)
    mask=np.zeros((n,count),bool);mask[:,1::2]=True
    return habits.MultiWorld(menu,tags,tags[np.arange(n)[:,None],ctx],rules,np.zeros((n,2),int),mask,0,prob.argmax(2))


