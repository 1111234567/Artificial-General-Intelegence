"""Two same-port pulses per request: infer recurring background versus current food from public experience."""
from dataclasses import dataclass,replace
import numpy as np
from . import symmetry_world as geometry,mixture_flash,habitat_world as habitats,mixture_world as food,world


@dataclass
class Packets:
    values:np.ndarray
    times:np.ndarray
    truth:np.ndarray


def present(ep,*,noise=0.,informative=True):
    arena=ep.arena;n,tr=arena.menu.body.ctx.shape
    o,c=mixture_flash.timing(arena.menu.body.seeds,tr)
    first=(c<o)
    taste=2*arena.menu.chemicals-1.
    ambient=np.empty((n,tr,2,4),np.float32)
    for row,seed in enumerate(arena.menu.body.seeds):
        rng=np.random.default_rng(np.random.SeedSequence([seed,52793]))
        a,b=geometry.unit(rng.normal(size=(2,4)))
        # A bounded separation changes sensory ambiguity independently of which
        # private relation each of those new backgrounds was assigned.
        if abs(a@b)>.6:
            b=geometry.unit(b-(a@b)*a)
        proto=np.stack([a,b])
        if informative:z=proto[arena.menu.body.ctx[row]]
        else:z=geometry.unit(rng.normal(size=(tr,4)))
        z=geometry.unit(z+noise*rng.normal(size=(tr,4)))
        ambient[row]=z[:,None,:]
    v=np.stack((np.where(first[:,:,None,None],ambient,taste),np.where(first[:,:,None,None],taste,ambient)),2)
    return Packets(v,np.stack([np.minimum(o,c),np.maximum(o,c)],2),np.where(first,0,1).astype(np.int8))


class Familiarity:
    def __init__(self,n,*,seed=8191,nodes=64,slots=6,radius=.21):
        rng=np.random.default_rng(seed)
        self.centers=geometry.unit(rng.normal(size=(nodes,4))).astype(np.float32)
        self.trace=np.zeros((n,nodes),np.float32)
        self.slots=slots;self.threshold=np.cos(radius)
        self.proto=np.zeros((n,slots,4),np.float32)
        self.count=np.zeros((n,slots),np.int32)
        self.overflow=np.zeros(n,int)
        angles=np.array([0.,1.48,.74,.37,1.11,.18,.55,1.31][:slots])
        self.code=np.stack((np.cos(angles),np.sin(angles)),1).astype(np.float32)

    def features(self,rows,v,*,update=True):
        # v (lives,2 incoming packets,2 sides,4); no semantic channel,
        # physical event type, private preference, score or family is available.
        mean=geometry.unit(v.mean(2))
        a=np.exp(8.*(mean@self.centers.T-1.)).astype(np.float32)
        a=geometry.unit(a)
        familiarity=(a*self.trace[rows,None,:]).sum(2)
        bilateral=(v[:,:,0]*v[:,:,1]).sum(2)
        if update:self.trace[rows]=.88*self.trace[rows]+.12*a.mean(1)
        return np.stack((familiarity,bilateral),2)

    def assign(self,rows,v,scores):
        context_index=scores.argmax(1);i=np.arange(len(rows))
        selected=geometry.unit(v[i,context_index].mean(1))
        occupied=self.count[rows]>0
        match=np.where(occupied,(self.proto[rows]*selected[:,None,:]).sum(2),-np.inf)
        which=match.argmax(1);new=match.max(1)<self.threshold
        vacant=(~occupied).any(1);alloc=new&vacant
        which[alloc]=(~occupied[alloc]).argmax(1);self.overflow[rows[new&~vacant]]+=1
        counts=self.count[rows,which]
        self.proto[rows,which]=geometry.unit(self.proto[rows,which]*counts[:,None]+selected)
        self.count[rows,which]+=1
        return context_index,v[i,1-context_index],self.code[which]


def prepare(ep,packets,weights,*,ablate="true"):
    n,tr=ep.arena.menu.body.ctx.shape;row=np.arange(n)
    rec=Familiarity(n)
    choice=np.empty((n,tr),np.int8);chem=np.empty_like(ep.arena.menu.chemicals);indicator=np.empty_like(ep.arena.appearance)
    for j in range(tr):
        v=packets.values[:,j];x=rec.features(row,v,update=ablate!="no_history")
        score=x@np.asarray(weights[:2],np.float32)+np.array([weights[2],0.],np.float32)
        if ablate=="oracle":score=np.eye(2,dtype=np.float32)[packets.truth[:,j]]
        elif ablate=="shuffled_density":score=x[row^1,:,0]*weights[0]+x[:,:,1]*weights[1]+[weights[2],0.]
        pick,taste,ctx=rec.assign(row,v,score)
        choice[:,j]=pick;chem[:,j]=.5*(taste+1.);indicator[:,j]=ctx
    arena=replace(ep.arena,menu=replace(ep.arena.menu,chemicals=chem),appearance=indicator)
    view=replace(ep,arena=arena)
    # Both routed currents can be sent only after the actual second physical
    # pulse. No early cue is reconstructed before the animal could see it.
    second=packets.times[:,:,1].copy()
    view.sensory_timing=(second,second)
    return view,choice,rec


class RawBody(habitats.Body):
    def __init__(self,ep,packets):
        self.packets=packets
        super().__init__(ep.arena,silent_probes=True)
    def observe(self):
        obs=super().observe();j,t=self.trial_index,self.tick_index
        obs[:,food.CTX]=0.;obs[:,food.LEFT.start:food.RIGHT.stop]=.5
        for k in range(2):
            rows=np.flatnonzero(self.packets.times[:,j,k]==t)
            if len(rows):obs[rows,food.LEFT.start:food.RIGHT.stop]=.5*(self.packets.values[rows,j,k].reshape(len(rows),8)+1.)
        return obs


class Live:
    def __init__(self,underlying,weights):
        self.brain=underlying;self.weights=np.asarray(weights,np.float32)
    def reset(self,n):
        self.brain.reset(n);self.familiar=Familiarity(n)
        self.first=np.empty((n,2,4),np.float32);self.seen=np.zeros(n,bool)
        self.assignments=[]
    def act(self,frame):
        if frame[:,world.TRIAL_START].all():self.seen.fill(False)
        x=2*frame[:,food.LEFT.start:food.RIGHT.stop].reshape(-1,2,4)-1.
        pulse=x.any((1,2));first=pulse&~self.seen;self.first[first]=x[first]
        rows=np.flatnonzero(pulse&self.seen)
        obs=frame.copy();obs[:,food.LEFT.start:food.RIGHT.stop]=.5
        if len(rows):
            raw=np.stack((self.first[rows],x[rows]),1)
            feat=self.familiar.features(rows,raw)
            scores=feat@self.weights[:2]+np.array([self.weights[2],0.],np.float32)
            k,odor,ctx=self.familiar.assign(rows,raw,scores)
            obs[rows,food.LEFT.start:food.RIGHT.stop]=(.5*(odor+1.)).reshape(len(rows),8)
            obs[rows,food.CTX]=ctx
        self.seen|=pulse
        return self.brain.act(obs)

