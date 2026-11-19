"""Wait briefly only when the public evidence for a feedback binding is ambiguous."""
from dataclasses import dataclass
import numpy as np
from . import delayed_receipts as timing


@dataclass(frozen=True)
class Rule:
    wait:int=8
    margin:float=.65
    prediction:float=0.
    echo_gain:float=4.
    prior_gap:float=.1
    joint:bool=True

    def make_controller(self,brain,channel):
        return Resolver(brain,self,channel)


class Resolver(timing.Pending):
    def __init__(self,brain,rule,ch):
        super().__init__(brain)
        self.rule,self.channel=rule,ch
        n=brain.n;k=6;self.queue=k
        self.arrived=np.full((n,k),-1,int)
        self.reported_side=np.zeros((n,k),int)
        self.sign=np.zeros((n,k),np.float32);self.glimpse=np.zeros((n,k,4),np.float32)
        # Evaluator-only identity. Never consulted when selecting what to write.
        self.meter_source=np.full((n,k),-1,int)
        self.expected=np.zeros_like(self.when,np.float32);self.wait_sum=np.zeros(n,int)
    def touch(self,rows,side,j,t):
        super().touch(rows,side,j,t)
        if len(rows):
            idx=(self.origin[rows]==j).argmax(1)
            g=(self.brain.g[rows]*self.brain.code[rows]).sum(1)
            self.expected[rows,idx]=np.tanh(self.brain.route["amp"]*g)*self.brain.rlast[rows,side]
    def receive(self,rows,side,echo,sign,source,now):
        self.delivered[rows]+=1
        free=self.arrived[rows]<0
        if not free.any(1).all():raise RuntimeError("An unconsumed receipt was overwritten")
        q=free.argmax(1);self.arrived[rows,q]=now;self.reported_side[rows,q]=side
        self.sign[rows,q]=sign;self.glimpse[rows,q]=echo;self.meter_source[rows,q]=source
    def scores(self,rows,slots):
        age=self.arrived[rows,slots,None]-self.when[rows]
        k=self.channel.echo_dims;r=self.rule
        diff=self.odor[rows,:,:k]-self.glimpse[rows,slots,:k][:,None,:]
        score=-r.echo_gain*(diff*diff).sum(2)+r.prediction*(self.sign[rows,slots,None]*self.expected[rows])
        score-=r.prior_gap*np.abs(age-self.channel.lag*.5)/max(1,self.channel.lag)
        # The reported physical side and that a touch precedes its receipt are
        # exact public facts. The upper transport delay is a supplied prior.
        score-=4.*(self.side[rows]!=self.reported_side[rows,slots,None])
        score-=3.*np.maximum(0.,age-self.channel.lag-2.)/max(1,self.channel.lag)
        legal=(self.origin[rows]>=0)&(age>=0)
        return np.where(legal,score,-1e5)
    def commit(self,now):
        active=np.flatnonzero((self.arrived>=0).any(1)&(self.origin>=0).any(1))
        if not len(active):return active,active,np.zeros(0,np.float32)
        arrival=self.arrived[active]
        order=np.argsort(np.where(arrival>=0,arrival,np.iinfo(np.int32).max),1,kind="stable")
        q0,q1=order[:,0],order[:,1]
        s0=self.scores(active,q0)
        count=(arrival>=0).sum(1)
        if self.rule.joint:
            pair=np.flatnonzero(count>=2)
            if len(pair):
                s1=self.scores(active[pair],q1[pair])
                # For the oldest report, compare assignments that reserve a
                # different plausible contact for the second report.
                order2=np.argsort(-s1,1,kind="stable")
                best=s1[np.arange(len(pair)),order2[:,0]]
                runner=s1[np.arange(len(pair)),order2[:,1]]
                extra=np.broadcast_to(best[:,None],s1.shape).copy()
                extra[np.arange(len(pair)),order2[:,0]]=runner
                s0[pair]+=extra
        ranked=np.argsort(-s0,1,kind="stable");winner=ranked[:,0]
        gap=s0[np.arange(len(active)),winner]-s0[np.arange(len(active)),ranked[:,1]]
        wait=now-arrival[np.arange(len(active)),q0]
        # At least one quantum of external time passes before reconsidering an
        # ambiguous binding. Already-arrived signals are the only extra evidence.
        write=(gap>=self.rule.margin)|(wait>=self.rule.wait)|(count>=3)
        a=active[write];q=q0[write];b=winner[write]
        self.wait_sum[a]+=wait[write];self.written[a]+=1
        self.matched[a]+=self.origin[a,b]==self.meter_source[a,q]
        sd=self.reported_side[a,q];y=self.sign[a,q].copy()
        self.brain.code[a]=self.code[a,b];self.brain.rlast[a,sd]=self.recalled[a,b]
        self.brain.wlast[a,sd]=self.write[a,b]
        self.origin[a,b]=-1;self.arrived[a,q]=-1
        return a,sd,y
