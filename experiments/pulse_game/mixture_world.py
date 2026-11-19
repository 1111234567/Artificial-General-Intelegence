"""Two never-repeated continuous foods and an unknown preference rule per life.

The agent receives a noisy taste only for the food its body reaches. The upper
bound and regret use hidden expected utilities in the evaluator, never on an input.
"""

from dataclasses import dataclass
import numpy as np

from . import world as body


CTX=slice(0,2)
LEFT=slice(2,6)
RIGHT=slice(6,10)
DIM=4
PAIR=np.array([(i,j) for i in range(DIM) for j in range(i+1,DIM)])


@dataclass
class Foods:
    body:body.Tape
    chemicals:np.ndarray
    probability:np.ndarray
    realized:np.ndarray
    weights:np.ndarray
    rule:str
    counter:int


def make(seeds,trials=128,*,rule="linear",distribution="continuous",change=False,noise=0.):
    if rule not in ("linear","interaction","mixed") or distribution not in ("continuous","components","disjoint_octants"):
        raise ValueError("Unknown preference game")
    t=body.make_tape(seeds,trials=trials,distance_jitter=.13,wind_scale=.07)
    n=t.n;cut=trials//2
    chem=np.zeros((n,trials,2,DIM),np.float32)
    prob=np.zeros((n,trials,2),np.float32);rew=np.zeros_like(prob,dtype=np.int8)
    w_all=[]
    for life,s in enumerate(seeds):
        family,menu,uniform=(np.random.default_rng(x) for x in np.random.SeedSequence([s,0x092DBE]).spawn(3))
        # Both contexts have independent novel utilities; opposite-by-construction
        # used in the deterministic binary task is not presumed here.
        w=family.normal(size=(2,DIM+len(PAIR)))
        w[:,:DIM]/=np.linalg.norm(w[:,:DIM],axis=1,keepdims=True)
        w[:,DIM:]/=np.linalg.norm(w[:,DIM:],axis=1,keepdims=True)
        x=menu.uniform(-1.,1.,size=(trials,2,DIM))
        if distribution=="components":
            dims=menu.integers(DIM,size=(cut,2))
            values=menu.uniform(.50,1.,size=(cut,2))*menu.choice([-1,1],size=(cut,2))
            x[:cut]=np.eye(DIM)[dims]*values[...,None]
        elif distribution=="disjoint_octants":
            # Train and test occupy different conjunctions, with individually
            # symmetric coordinate marginals in each half.
            sign=menu.choice([-1,1],size=(trials,2,DIM-1))
            last=np.prod(sign,axis=-1)
            last[cut:]*=-1
            x=np.abs(x)*np.concatenate((sign,last[:,:,None]),axis=2)
        graph=w[t.ctx[life]]
        one=(x*graph[:,None,:DIM]).sum(2)
        interactions=x[:,:,PAIR[:,0]]*x[:,:,PAIR[:,1]]
        two=(interactions*graph[:,None,DIM:]).sum(2)
        logit=one if rule=="linear" else 1.8*two if rule=="interaction" else .67*one+1.25*two
        if change:
            # Only context B changes. No sensory marker of the reversal.
            alter=(np.arange(trials)>=cut)&(t.ctx[life]==1)
            logit[alter]*=-1
        p=1./(1.+np.exp(-4.8*logit))
        p=noise+(1.-2.*noise)*p
        chem[life]=(x+1.)/2.;prob[life]=p
        rew[life]=np.where(uniform.random((trials,2))<p,1,-1)
        w_all.append(w)
    # The base body's original private identity target is completely ignored
    # by the environment below, including the externally displayed reward.
    return Foods(t,chem,prob,rew,np.stack(w_all),rule,cut)


class Body(body.PulseWorld):
    def __init__(self,menu,*,hidden_after=None,**kw):
        self.menu=menu
        self.hidden_after=hidden_after
        super().__init__(menu.body,**kw)

    def _shown_reward(self,receipt):
        if self.hidden_after is not None and self.trial_index>=self.hidden_after:
            return np.zeros(self.tape.n,np.int8)
        return super()._shown_reward(receipt)

    def _load_trial(self):
        super()._load_trial()
        self._background[:,LEFT]=self.menu.chemicals[:,self.trial_index,0]
        self._background[:,RIGHT]=self.menu.chemicals[:,self.trial_index,1]

    def _move(self,pulses):
        super()._move(pulses)
        k=self.trial_index
        hit=self._contact_tick[:,k]==self.tick_index
        rows=np.flatnonzero(hit);side=self._contact_side[rows,k]
        self._contact_id[rows,k]=side
        self._reward[rows,k]=self.menu.realized[rows,k,side]


def rollout(agent,menu,**kw):
    sim=Body(menu,**kw);agent.reset(menu.body.n);obs=sim.observe()
    while obs is not None:obs=sim.step(agent.act(obs))
    return sim.result()


def metrics(t,res):
    side=res.contact_side
    chosen=np.take_along_axis(t.probability,side.clip(min=0)[:,:,None],axis=2)[:,:,0]
    hit=side>=0
    expected=chosen*hit
    best=t.probability.max(2)
    tied=np.isclose(t.probability[:,:,0],t.probability[:,:,1],atol=1e-7,rtol=0)
    optimal=hit & (chosen>=best-1e-7)
    # For tied foods give half a hit to a contacting policy, so lack of
    # information cannot inflate choice accuracy in the interaction-only control.
    correct=np.where(tied,.5*hit,optimal)
    return {"first_8":correct[:,:8].mean(1),"tail_32":correct[:,-32:].mean(1),"whole":correct.mean(1),
            "test_phase":correct[:,t.counter:].mean(1),"transfer_0_16":correct[:,t.counter:t.counter+16].mean(1),
            "second_A":np.divide((correct[:,t.counter:]*(t.body.ctx[:,t.counter:]==0)).sum(1),
                                    (t.body.ctx[:,t.counter:]==0).sum(1).clip(min=1)),
            "second_B":np.divide((correct[:,t.counter:]*(t.body.ctx[:,t.counter:]==1)).sum(1),
                                    (t.body.ctx[:,t.counter:]==1).sum(1).clip(min=1)),
            "expected_good":expected.mean(1),"tail_expected_good":expected[:,-32:].mean(1),
            "oracle_expected_good":best.mean(1),"tail_regret":(best[:,-32:]-expected[:,-32:]).mean(1),
            "contact":hit.mean(1),"positive":(res.reward>0).mean(1)}
