"""Revisit real signed olfactory experiences during physical idle ticks to infer a private transform."""
from dataclasses import dataclass,replace
import numpy as np
from . import mixture_brain,mixture_world as scents,world,signed_replay as inference
from .symmetry_spike_exposure import random_rotations
from .symmetry_spike_matrix import alternate


def geometry(p,kind):
    if kind=="antipodal":return p
    if kind=="rotation":return random_rotations((p,))[0]
    if kind=="quarter":return alternate((p,),"quarter-turn")[0]
    if kind=="similar":return alternate((p,),"near-identical")[0]
    if kind=="stretched":
        v=random_rotations((p,))[0]
        chem=v.menu.chemicals.copy()
        changed=(v.pairmates>=0)&(np.arange(v.menu.body.trials)[None,:]<v.exposed)
        m=np.repeat(changed[:,:,None],2,axis=2)
        i,j=np.nonzero(v.query);m[i,j,v.related_side[i,j]]=True
        x=chem*2.-1.
        # Same invertible non-orthogonal distortion after each private rotation.
        stretch=inference.norm(x*np.array([.38,.72,1.43,2.2],np.float32))
        chem[m]=(.5+.5*stretch[m]).clip(0.,1.)
        return replace(v,menu=replace(v.menu,chemicals=chem))
    raise ValueError(kind)


def strength(history,labels,train_count,full_count,op,cfg):
    """Held-out signed odor likelihood using preceding public observations in the same life."""
    legal,_=inference.compatible(history,labels,full_count,cfg)
    sim=inference.norm(history@op)@history.transpose(0,2,1)
    plus=np.sqrt(1.+np.where(legal,np.exp(cfg.temperature*(sim-cfg.threshold)),0.).sum(1))
    minus=np.sqrt(1.+np.where(legal,np.exp(cfg.temperature*(-sim-cfg.threshold)),0.).sum(1))
    # Logistic proper score for the sign that was actually experienced.
    loss=np.log1p(minus/plus)
    t=np.arange(history.shape[1])
    test=(t[None,:]>=train_count[:,None])&(t[None,:]<full_count[:,None])
    return -(loss*test).sum(1)/np.maximum(test.sum(1),1)


@dataclass(frozen=True)
class Motion:
    tonic:float=.48
    gain:float=3.5
    leak:float=.68
    noise:float=.15
    excite:float=1.1
    oppose:float=1.12


class Actor(mixture_brain.MixtureLearner):
    def __init__(self,receptor,cfg,seeds,*,passes=4,adaptive=False,control="true",motion=Motion(),at_query=False):
        self.front,self.cfg,self.moves=receptor,cfg,motion
        self.passes,self.adaptive,self.control,self.at_query=passes,adaptive,control,at_query
        super().__init__(mixture_brain.Rule(neurons=receptor["matrix"].shape[0],latent=4,covariance=False),19,seeds)

    def reset(self,n):
        super().reset(n);capacity=self.cfg.capacity
        self.s=np.zeros((n,2,self.front["matrix"].shape[0]),np.float32)
        self.history=np.zeros((n,capacity,4),np.float32)
        self.fields=np.full((n,capacity),-1,np.int8);self.count=np.zeros(n,int)
        self.matrix=np.zeros((n,4,4),np.float32);self.check=np.zeros_like(self.matrix)
        self.valid_score=np.zeros(n,np.float32);self.train_end=np.zeros(n,int)
        self.remaining=np.zeros(n,int);self.accepts=np.zeros(n,int);self.work=np.zeros(n,int)
        self.ctx=np.zeros(n,int);self.latest=np.zeros((n,2,4),np.float32)
        self.last_x=np.zeros((n,2,4),np.float32);self.last_field=np.zeros(n,int)
        self.saw_options=np.zeros(n,bool);self.rr=np.arange(n);self.replay_at_choice=[]
        self.anchors=[]
    def _view(self,rows):
        # Chronological buffer, dropping the oldest public event when full.
        size=max(1,int(self.count[rows].max()))
        return self.history[rows,:size],self.fields[rows,:size],self.count[rows]
    def _init(self,rows):
        if not len(rows):return
        x,c,n=self._view(rows)
        self.matrix[rows]=inference.initialize(x,c,n,self.cfg)
        if self.adaptive:
            prefix=np.maximum(2,(.70*n).astype(int)).clip(max=n)
            self.check[rows]=inference.initialize(x,c,prefix,self.cfg)
            self.valid_score[rows]=strength(x,c,prefix,n,self.check[rows],self.cfg)
            self.train_end[rows]=prefix
        self.remaining[rows]=self.passes
    def _think(self,rows):
        if not len(rows):return
        x,c,n=self._view(rows)
        update=inference.revise(x,c,n,self.matrix[rows],self.cfg)
        mask=np.ones(len(rows),bool)
        if self.adaptive:
            prefix=self.train_end[rows]
            test=inference.revise(x,c,prefix,self.check[rows],self.cfg)
            score=strength(x,c,prefix,n,test,self.cfg)
            mask=(n>=6)&(score>=self.valid_score[rows]+.00005)
            self.check[rows[mask]]=test[mask];self.valid_score[rows[mask]]=score[mask]
        self.matrix[rows[mask]]=update[mask]
        self.accepts[rows[mask]]+=1;self.work[rows]+=1
        self.remaining[rows]-=1;self.remaining[rows[~mask]]=0
    def act(self,obs):
        f=self.front;cfg=self.cfg
        if obs[:,world.TRIAL_START].all():self.round+=1;self.saw_options[:]=False
        y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        rows=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        if len(rows):
            side=obs[rows,world.CONTACT_ID].argmax(1)
            signed=self.last_x[rows,side]*y[rows,None]
            field=self.last_field[rows];self.latest[rows,field]=signed
            full=rows[self.count[rows]==cfg.capacity]
            if len(full):
                self.history[full,:-1]=self.history[full,1:].copy()
                self.fields[full,:-1]=self.fields[full,1:].copy()
            slot=self.count[rows].clip(max=cfg.capacity-1)
            self.history[rows,slot]=self.last_x[rows,side] if self.control=="unsigned_history" else signed
            self.fields[rows,slot]=field;self.count[rows]=np.minimum(self.count[rows]+1,cfg.capacity)
            if not self.at_query:self._init(rows)
        sensor=obs[:,scents.CTX]
        present=sensor.any(1);self.ctx[present]=sensor[present].argmax(1)
        x=2.*obs[:,scents.LEFT.start:scents.RIGHT.stop].reshape(self.n,2,4)-1.
        choice=(np.abs(x[:,0]-x[:,1]).sum(1)>.02)&~self.saw_options
        if self.at_query:
            new=np.flatnonzero(choice)
            self._init(new)
        self.saw_options|=choice
        if self.at_query:self._think(np.flatnonzero(self.saw_options&(self.remaining>0)))
        else:self._think(np.flatnonzero((obs[:,world.ACTIVE]==0)&(self.remaining>0)))
        self.s=self.s@f["matrix"]+x@f["sensors"][:4]+(sensor@f["sensors"][4:])[:,None]
        dec,b=f["ground"];self.last_x=inference.norm(self.s@dec+b)
        self.last_field=self.ctx.copy()
        plan=(self.latest[self.rr,self.ctx,None,:]@self.matrix).squeeze(1)
        if self.control=="invert":plan=-plan
        if self.control=="cut_cross_dimensions":plan=self.latest[self.rr,self.ctx]*np.diagonal(self.matrix,axis1=1,axis2=2)
        v=(self.last_x*plan[:,None,:]).sum(2)
        m=self.moves;active=obs[:,world.ACTIVE,None]
        drive=m.tonic*active-.85*(1.-active)+m.gain*np.tanh(v-v[:,::-1])+m.noise*self._noise()+m.excite*self.motor-m.oppose*self.motor[:,::-1]
        self.mv=np.maximum(-1.,m.leak*self.mv+drive)
        self.motor=(self.mv>=1.).astype(np.float32);self.mv*=1.-self.motor
        return self.motor
