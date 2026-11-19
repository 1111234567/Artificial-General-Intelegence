"""An actual touch creates one unresolved eligibility; delayed public taste need not arrive in order."""
from dataclasses import dataclass
import numpy as np
from . import habitat_world,mixture_world as inputs,symmetry_world as features,world


@dataclass(frozen=True)
class Channel:
    lag:int=40
    ordered:bool=False
    echo_dims:int=2
    echo_noise:float=.12
    wait:int=56


class Pending:
    """Forward eligibility for a bounded number of physically touched items."""
    def __init__(self,brain,capacity=8):
        self.brain=brain;self.capacity=capacity
        n=brain.n;d=brain.route["slots"]
        self.origin=np.full((n,capacity),-1,int);self.when=np.zeros_like(self.origin)
        self.side=np.zeros_like(self.origin);self.odor=np.zeros((n,capacity,4),np.float32)
        self.write=np.zeros((n,capacity,4),np.float32);self.code=np.zeros((n,capacity,d),np.float32)
        self.recalled=np.zeros((n,capacity),np.float32)
        self.max_pending=np.zeros(n,int);self.overflow=np.zeros(n,int)
        self.matched=np.zeros(n,int);self.delivered=np.zeros(n,int);self.written=np.zeros(n,int)
    def touch(self,rows,side,j,t):
        if not len(rows):return
        c=self.brain
        free=self.origin[rows]<0
        new=np.where(free.any(1),free.argmax(1),self.when[rows].argmin(1))
        self.overflow[rows]+=~free.any(1)
        self.origin[rows,new]=j;self.when[rows,new]=t;self.side[rows,new]=side
        w,b=c.perception["ground"]
        self.odor[rows,new]=features.unit(c.h[rows,side]@w+b)
        self.write[rows,new]=c.wlast[rows,side];self.code[rows,new]=c.code[rows]
        self.recalled[rows,new]=c.rlast[rows,side]
        self.max_pending[rows]=np.maximum(self.max_pending[rows],(self.origin[rows]>=0).sum(1))
    def accept(self,rows,side,echo,receipt_origin,now,mode,dimensions,weights=None):
        if not len(rows):return
        self.delivered[rows]+=1
        legal=self.origin[rows]>=0
        earlier=self.when[rows];age=now-earlier
        same=self.side[rows]==side[:,None]
        if mode=="oracle":
            # Diagnostic lower bound on misattribution: the chronological source
            # is never supplied to any of the learned/public address policies.
            score=(self.origin[rows]==receipt_origin[:,None]).astype(float)
        elif mode in ("fifo","fifo_side"):
            score=age.astype(float)+(10000.*same if mode=="fifo_side" else 0.)
        elif mode=="latest" or mode=="time_local":score=-age.astype(float)
        elif mode=="match":
            # Only the portion of the public echoed sensory vector that was shown.
            if dimensions:
                diff=self.odor[rows,:,:dimensions]-echo[:,None,:dimensions]
                distance=(diff*diff).sum(2)
            else:distance=np.zeros_like(age,np.float32)
            a,b,c=weights
            score=-a*distance+b*same-c*np.abs(age-20.)/40.
        else:raise ValueError(mode)
        index=np.where(legal,score,-np.inf).argmax(1)
        have=legal.any(1);r=rows[have];slot=index[have]
        self.matched[r]+=self.origin[r,slot]==receipt_origin[have]
        self.written[r]+=1
        if mode!="time_local":
            self.brain.code[r]=self.code[r,slot]
            self.brain.rlast[r,side[have]]=self.recalled[r,slot]
            self.brain.wlast[r,side[have]]=self.write[r,slot]
        self.origin[r,slot]=-1


def run(ep,actor,seeds,ch,mode,weights=None):
    n=len(seeds);tr=ep.arena.menu.body.trials
    if ch.lag<0 or ch.lag>63:raise ValueError("Bound the diagnostic delivery buffer")
    rng=[np.random.default_rng(np.random.SeedSequence([s,554716])) for s in seeds]
    lag=np.stack([r.integers(0,ch.lag+1,tr) for r in rng])
    jitter=np.stack([r.normal(size=(tr,4)) for r in rng]).astype(np.float32)
    sim=habitat_world.Body(ep.arena,silent_probes=False);actor.reset(n)
    deferred=hasattr(mode,"make_controller")
    detector=mode.make_controller(actor,ch) if deferred else Pending(actor)
    size=128
    calendar=np.full((n,size),-1,int);sign=np.zeros((n,size),np.float32)
    sidebank=np.zeros((n,size),int);echoes=np.zeros((n,size,4),np.float32)
    last_due=np.full(n,-1,int);now=0;rr=np.arange(n)
    received=np.zeros((n,tr),bool);anchor_seen=np.zeros((n,tr),bool);lags=[]
    idle=np.zeros((n,world.OBS_SIZE),np.float32);idle[:,inputs.LEFT.start:inputs.RIGHT.stop]=.5
    def step(raw,trial):
        nonlocal now
        view=raw.copy();y=raw[:,world.REWARD_POS]-raw[:,world.REWARD_NEG]
        rows=np.flatnonzero((y!=0)&raw[:,world.CONTACT_ID].any(1))
        if len(rows):
            side=raw[rows,world.CONTACT_ID].argmax(1)
            detector.touch(rows,side,trial,now)
            date=now+lag[rows,trial]
            if ch.ordered:date=np.maximum(date,last_due[rows]+1)
            # Every actual receipt is delivered once. A full public channel
            # delays collisions; it never silently drops a second signed taste.
            for _ in range(size):
                busy=calendar[rows,date%size]>=0
                if not busy.any():break
                date[busy]+=1
            if np.any(date-now>=size):raise RuntimeError("Receipt calendar exhausted")
            at=date%size;calendar[rows,at]=trial;sign[rows,at]=y[rows];sidebank[rows,at]=side
            odor=ep.arena.menu.chemicals[rows,trial,side]*2.-1.
            echoes[rows,at]=odor+ch.echo_noise*jitter[rows,trial]
            last_due[rows]=date;lags.extend((date-now).tolist())
        view[:,world.REWARD_POS:world.REWARD_NEG+1]=0.;view[:,world.CONTACT_ID]=0.
        at=now%size;due=np.flatnonzero(calendar[:,at]>=0)
        if len(due):
            origin=calendar[due,at].copy();sd=sidebank[due,at].copy();y=sign[due,at]
            if deferred:
                detector.receive(due,sd,echoes[due,at],y,origin,now)
            else:
                detector.accept(due,sd,echoes[due,at],origin,now,mode,ch.echo_dims,weights)
                # The source request index is reserved for a diagnostic oracle.
                view[due,world.REWARD_POS]=y>0;view[due,world.REWARD_NEG]=y<0
                view[due,world.CONTACT_ID.start+sd]=1
            received[due,origin]=True;calendar[due,at]=-1
        if deferred:
            writing,sd,signal=detector.commit(now)
            view[writing,world.REWARD_POS]=signal>0;view[writing,world.REWARD_NEG]=signal<0
            view[writing,world.CONTACT_ID.start+sd]=1
        movement=actor.act(view);now+=1;return movement
    obs=sim.observe()
    while obs is not None:
        j=sim.trial_index
        if sim.tick_index==0 and ep.arena.probe[0,j]:
            for _ in range(ch.wait):step(idle,j)
            anchor_seen[:,j]=received[:,j-1]
        obs=sim.step(step(obs,j))
    res=sim.result();q=ep.arena.probe[0]
    return res,dict(extra_commit_wait=(detector.wait_sum/detector.written.clip(min=1) if deferred else np.zeros(n)),
                     correct_binding=detector.matched/detector.delivered.clip(min=1),
                     packets=detector.delivered,peak_pending=detector.max_pending,
                     overflow=detector.overflow,anchor_available=anchor_seen[:,q].mean(1),
                     actual_delay_mean=float(np.mean(lags)),signed_receipts_created=len(lags),
                     written=detector.written)

