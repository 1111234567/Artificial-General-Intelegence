"""A single continuous life with recurring habitats, unannounced reversals and returns."""
from dataclasses import dataclass,replace
import numpy as np
from . import habitat_world as habitats,mixture_world as tastes,symmetry_world as geometry,world


@dataclass
class Stream:
    arena:habitats.MultiWorld
    checks:dict
    headline_probes:tuple
    block_context:np.ndarray
    truth:np.ndarray
    boundary:np.ndarray
    query_at:np.ndarray
    mode:str


def make(seeds,*,blocks=96,span=4,mode="reverse_return",order="clustered",noise=.05,appearance="smooth",nonce=6191701,
         private_rules=None,public_habitats=None,absolute_switches=None,block_offset=0):
    if blocks<12 or span%2:raise ValueError("Need long streams with matched even-sized evidence groups")
    if mode not in ("stable","reverse","reverse_return"):raise ValueError(mode)
    seeds=list(map(int,seeds));n=len(seeds);length=4+2*span;t=blocks*length
    xy=np.empty((n,blocks,length,2,4),np.float32);known=np.empty((n,blocks,length,2),np.float32)
    taste=np.empty((n,blocks,length,2),np.int8)
    ctx=np.empty((n,blocks),np.int8);relations=np.empty((n,blocks),np.int8)
    breaks=np.empty((n,2),int);appearance_key=np.empty((n,2,2),np.float32)
    for i,s in enumerate(seeds):
        times,scene,private,err=(np.random.default_rng(g) for g in np.random.SeedSequence([s,nonce,319807]).spawn(4))
        if order=="clustered":
            a=int(times.integers(2));seq=[]
            while len(seq)<blocks:
                seq.extend([a]*int(times.integers(2,10)));a=1-a
            ctx[i]=seq[:blocks]
        elif order=="iid":ctx[i]=times.integers(2,size=blocks)
        else:raise ValueError(order)
        cuts=np.array([blocks//3,2*blocks//3])+times.integers(-max(1,blocks//18),max(2,blocks//18+1),2)
        if absolute_switches is not None:cuts=np.asarray(absolute_switches[i],int)-block_offset
        breaks[i]=cuts
        if appearance=="discrete":appearance_key[i]=np.eye(2)[times.permutation(2)]
        else:
            start=times.uniform(0,.45);delta=times.uniform(.68,1.0) if appearance=="smooth" else times.uniform(.18,.31)
            angles=np.array([start,start+delta]);appearance_key[i]=np.column_stack((np.cos(angles),np.sin(angles)))[times.permutation(2)]
        r=private.choice([-1,1],2)
        if private_rules is not None:r=np.asarray(private_rules[i])
        if public_habitats is not None:appearance_key[i]=public_habitats[i]
        f=r[ctx[i]]
        ix=np.arange(blocks)
        switched=(ctx[i]==1)&(ix>=cuts[0])
        if mode=="stable":switched[:]=False
        if mode=="reverse_return":switched&=ix<cuts[1]
        f=f*np.where(switched,-1,1);relations[i]=f
        odors=geometry.unit(scene.normal(size=(blocks,span,4)))
        echoed=geometry.unit(-odors+.08*scene.normal(size=odors.shape))
        rand=private.random((blocks,span));labels=np.where(np.argsort(np.argsort(rand,1),1)<span//2,-1,1)
        rand=private.random((blocks,span));replayed=np.where(np.argsort(np.argsort(rand,1),1)<span//2,-1,1)
        # Both relations produce exactly the same reward time series. Only the
        # permutation of actual sensory experiences indexes the sign relation.
        needs=replayed*f[:,None]
        position=np.where(needs<0,np.cumsum(needs<0,1)-1,span//2+np.cumsum(needs>0,1)-1)
        ordered=np.argsort((labels>0)+scene.random((blocks,span))*.7,axis=1)
        picked=np.take_along_axis(ordered,position,1)
        repeat=echoed[np.arange(blocks)[:,None],picked]
        xy[i,:,2:2+span]=odors[:,:,None,:];xy[i,:,2+span:2+2*span]=repeat[:,:,None,:]
        taste[i,:,2:2+span]=labels[:,:,None];taste[i,:,2+span:2+2*span]=replayed[:,:,None]
        for begin in (0,length-2):
            x=geometry.unit(scene.normal(size=(blocks,4)));other=geometry.unit(scene.normal(size=(blocks,4)))
            related=geometry.unit(-x+.08*scene.normal(size=(blocks,4)));side=scene.integers(2,size=blocks)
            sign=private.choice([-1,1],size=blocks)
            xy[i,:,begin]=x[:,None,:];taste[i,:,begin]=sign[:,None]
            j=begin+1
            xy[i,ix,j,side]=related;xy[i,ix,j,1-side]=other
            taste[i,ix,j,side]=f*sign;taste[i,ix,j,1-side]=-f*sign
        known[i]=(taste[i]>0)
        corrupt=err.random((blocks,length))<noise
        corrupt[:,[1,length-1]]=False
        taste[i]*=np.where(corrupt,-1,1)[:,:,None].astype(np.int8)
    ids=np.arange(n)[:,None];flat_ctx=np.repeat(ctx,length,axis=1)
    base=replace(world.make_tape(seeds,t,distance_jitter=.13,wind_scale=.07),ctx=flat_ctx)
    menu=tastes.Foods(base,((xy.reshape(n,t,2,4)+1.)*.5).astype(np.float32),known.reshape(n,t,2),taste.reshape(n,t,2),
                      np.empty(0),"long_changing_habitats",0)
    offset=np.arange(blocks)*length
    at=np.column_stack((offset+1,offset+length-1))
    probes=np.zeros((n,t),bool);probes[:,at.ravel()]=True
    arena=habitats.MultiWorld(menu,appearance_key,appearance_key[ids,flat_ctx],relations,np.zeros((n,2),int),
                              probes,0,menu.probability.argmax(2))
    names={f"visit_{k}_{phase}":int(at[k,j]) for k in range(blocks) for j,phase in enumerate(("pre","post"))}
    return Stream(arena,names,(int(at[0,0]),int(at[-1,0]),int(at[-1,1])),ctx,relations,breaks,at,mode)


def count_regions(stream,output):
    w=stream.arena;pos=stream.query_at
    hits=(output.contact_side==w.truths)&(output.contact_side>=0)
    pre=hits[:,pos[:,0]];post=hits[:,pos[:,1]]
    b=stream.block_context==1
    j=np.arange(len(pos))[None,:]
    first,second=stream.boundary.T
    within=(j>=first[:,None])&(j<second[:,None])
    later=j>=second[:,None]
    out={}
    def store(name,mask):
        out[name]={"count":mask.sum(1).tolist(),"pre":(pre&mask).sum(1).tolist(),"post":(post&mask).sum(1).tolist()}
    store("stable_A",~b&(j>=first[:,None]))
    store("old_B",b&(j<first[:,None])&(j>=np.maximum(0,first-20)[:,None]))
    for phase,section in (("change",within),("return",later)):
        mask=section&b;rank=np.cumsum(mask,1)
        for lo,hi in ((1,1),(2,2),(3,4),(5,8),(9,16),(17,9999)):
            store(f"{phase}_{lo}_{hi}",mask&(rank>=lo)&(rank<=hi))
    store("whole_after_warmup",j>=np.ones((len(pre),1))*8)
    return out

