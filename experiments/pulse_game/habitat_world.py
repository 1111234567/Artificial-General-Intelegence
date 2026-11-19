"""Recall two independently learned relations after an actual period of conflicting experience."""
from dataclasses import dataclass, replace
import numpy as np

from . import symmetry_world as relation, mixture_world as game, mixture_flash as signals, world


@dataclass
class MultiWorld:
    menu:game.Foods
    tags:np.ndarray
    appearance:np.ndarray
    relation:np.ndarray
    first:np.ndarray
    probe:np.ndarray
    start_test:int
    truths:np.ndarray


def make(seeds,*,pairs=(12,12),visits=3,span=4,noise=.04,jitter=.08,relations=None,appearance="discrete",
         teach_first=True,novel_seed=76913):
    n=len(seeds);seeds=list(map(int,seeds))
    if any(p%2 for p in pairs):raise ValueError("Even numbers of matched pairs")
    if relations is None:
        relations=np.stack([np.random.default_rng(np.random.SeedSequence([s,novel_seed,300])).
                            choice([-1,1],size=2) for s in seeds])
    relations=np.asarray(relations)
    if relations.shape!=(n,2):raise ValueError("One private relation for each habitat")
    teachers=[]
    for c in range(2):
        ssub=[int(np.random.SeedSequence([s,novel_seed,17+c]).generate_state(1)[0]) for s in seeds]
        teachers.append(relation.make(ssub,pairs=pairs[c],probes=visits,interleave=min(span,pairs[c]),noise=noise,
                            jitter=jitter,family=relations[:,c],balanced_stream=True))
    pre=2*sum(pairs) if teach_first else 2*pairs[0]
    length=pre+4*visits
    max_=world.make_tape(seeds,length,distance_jitter=.13,wind_scale=.07)
    ctx=np.zeros((n,length),np.int8)
    xy=np.zeros((n,length,2,4),np.float32);r=np.zeros((n,length,2),np.int8);p=np.zeros_like(r,dtype=np.float32)
    tags=np.zeros((n,2,2),np.float32);out=np.zeros((n,length,2),np.float32)
    mask=np.zeros((n,length),bool);first=np.full((n,2),-1,int)
    for row,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,novel_seed,831]))
        if appearance=="discrete":basis=np.eye(2,dtype=np.float32)
        elif appearance in ("smooth","nearby"):
            lo=rng.uniform(0.,.50 if appearance=="smooth" else 1.00)
            sep=rng.uniform(.67,1.0) if appearance=="smooth" else rng.uniform(.19,.31)
            angle=np.array([lo,lo+sep]);basis=np.stack((np.cos(angle),np.sin(angle)),1)
        else:raise ValueError(appearance)
        tags[row]=basis[rng.permutation(2)]
        order=rng.permutation(2).tolist() if teach_first else [0]
        events=[]
        for c in order:events.extend((c,j,False) for j in range(2*pairs[c]))
        for v in range(visits):
            for c in rng.permutation(2):
                j=2*pairs[c]+2*v
                events.extend([(c,j,False),(c,j+1,True)])
        for k,(c,j,probe) in enumerate(events):
            teach=teachers[c].menu
            xy[row,k]=teach.chemicals[row,j];r[row,k]=teach.realized[row,j];p[row,k]=teach.probability[row,j]
            ctx[row,k]=c;out[row,k]=tags[row,c]
            mask[row,k]=probe
            if probe and first[row,c]<0:first[row,c]=k
    physical=replace(max_,ctx=ctx)
    return MultiWorld(game.Foods(physical,xy,p,r,np.empty(0),"two_private_rule",pre),tags,out,relations,first,
                      mask,pre,p.argmax(2))


class Body(signals.FlashBody):
    def __init__(self,t,*,silent_probes=False,**kw):
        self.habitats=t;self.silent_probes=silent_probes
        super().__init__(t.menu,**kw)
    def observe(self):
        obs=super().observe()
        t=self.trial_index
        show=obs[:,game.CTX].any(1)
        obs[show,game.CTX]=self.habitats.appearance[show,t]
        return obs
    def _shown_reward(self,receipt):
        value=super()._shown_reward(receipt)
        return np.where(self.habitats.probe[:,self.trial_index],0,value).astype(np.int8) if self.silent_probes else value


def rollout(agent,t,**options):
    sim=Body(t,**options);agent.reset(t.menu.body.n);obs=sim.observe()
    while obs is not None:obs=sim.step(agent.act(obs))
    return sim.result()


def metrics(t,res):
    side=res.contact_side
    hit=side>=0
    correct=hit&(side==t.truths)
    row=np.arange(len(side))
    a=correct[row,t.first[:,0]];b=correct[row,t.first[:,1]]
    last_context=t.menu.body.ctx[:,t.start_test-1]
    old=np.where(last_context==1,a,b);fresh=np.where(last_context==1,b,a)
    return dict(first=(a.astype(float)+b)*.5,both=(a&b).astype(float),earlier_habitat=old.astype(float),
                recent_habitat=fresh.astype(float),A=a.astype(float),B=b.astype(float),
                late=correct[t.probe].reshape(len(row),-1).mean(1),
                contact=hit[t.probe].reshape(len(row),-1).mean(1),
                same_family=(t.relation[:,0]==t.relation[:,1]).astype(float),
                A_first_side=side[row,t.first[:,0]].astype(float),B_first_side=side[row,t.first[:,1]].astype(float))
