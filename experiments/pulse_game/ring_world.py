"""Four fruit stations on a circle. Movement controls what is visible; biting samples one fruit."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np


SIZE=4
TICKS=40
OBS=21
CTX=slice(0,2)
CUE=slice(2,6)
FRUIT=slice(6,10)
OFFSET=10
VELOCITY=11
REACH=12
POSITIVE=13
NEGATIVE=14
TASTED=slice(15,19)
NEW=19
ACTIVE=20
PERMS=np.array(list(permutations(range(SIZE))),dtype=np.int8)
DERANGED=PERMS[(PERMS!=np.arange(SIZE)).all(1)]
ONE4=np.eye(4,dtype=np.float32)
ONE2=np.eye(2,dtype=np.float32)


@dataclass(frozen=True)
class Tape:
    seeds: tuple
    ctx: np.ndarray
    cue: np.ndarray
    positions: np.ndarray
    layout: np.ndarray
    preferences: np.ndarray
    goal: np.ndarray

    @property
    def n(self):return len(self.seeds)

    @property
    def trials(self):return self.ctx.shape[1]


def make_tape(seeds,trials=96):
    seeds=tuple(map(int,seeds))
    if not seeds or trials<1:
        raise ValueError("Need world seeds and at least one trial")
    contexts=[];cues=[];starts=[];layouts=[];maps=[]
    for seed in seeds:
        rngs=[np.random.default_rng(s) for s in np.random.SeedSequence([seed,0x96A205]).spawn(5)]
        a=PERMS[rngs[0].integers(len(PERMS))]
        b=a[DERANGED[rngs[0].integers(len(DERANGED))]]
        maps.append(np.stack((a,b)))
        contexts.append(rngs[1].integers(2,size=trials,dtype=np.int8))
        cues.append(rngs[2].integers(4,size=trials,dtype=np.int8))
        starts.append((rngs[3].integers(4,size=trials)+rngs[3].uniform(.40,.60,trials)).astype(np.float32) % SIZE)
        layouts.append(PERMS[rngs[4].integers(len(PERMS),size=trials)])
    ctx,cue,pos,layout,maps=map(np.stack,(contexts,cues,starts,layouts,maps))
    goal=maps[np.arange(len(seeds))[:,None],ctx,cue]
    return Tape(seeds,ctx,cue,pos,layout,maps,goal)


@dataclass
class Result:
    first: np.ndarray
    first_tick: np.ndarray
    solved: np.ndarray
    bites: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    waveform: np.ndarray
    first_correct: np.ndarray
    motion: np.ndarray


def rollout(policy,tape,*,retry=False,cue_ticks=TICKS,feedback="true",yoke=None,motion=.09,
            hidden_after=None):
    if not 0<=cue_ticks<=TICKS or feedback not in ("true","none","yoked","invert"):
        raise ValueError("Invalid perception/feedback protocol")
    n,trials=tape.n,tape.trials
    if feedback=="yoked" and (yoke is None or np.shape(yoke)!=(n,trials,TICKS)):
        raise ValueError("Need donor signal for each tick")
    rows=np.arange(n)
    first=np.full((n,trials),-1,np.int8);first_tick=np.full_like(first,-1)
    bites=np.zeros_like(first);win=np.zeros_like(first,dtype=bool)
    pos_count=np.zeros_like(first);neg_count=np.zeros_like(first)
    waveform=np.zeros((n,trials,TICKS),np.int8)
    motion_pulses=np.zeros((n,trials,2),np.int16)
    policy.reset(n)
    for j in range(trials):
        angle=tape.positions[:,j].copy();speed=np.zeros(n,np.float32)
        eaten=np.zeros((n,SIZE),bool)
        finished=np.zeros(n,bool)
        receipt=np.zeros(n,np.int8);last_id=np.zeros((n,4),np.float32)
        for tick in range(TICKS):
            active=~finished & (tick<TICKS-1)
            sector=np.floor(angle+.5).astype(np.int32)%SIZE
            offset=(angle-sector.astype(np.float32)+2.)%SIZE-2.
            intact=~eaten[rows,sector]
            visible=intact & (np.abs(offset)<=.34)
            reach=intact & (np.abs(offset)<=.26)
            items=tape.layout[rows,j,sector]
            obs=np.zeros((n,OBS),np.float32)
            if tick<cue_ticks:
                obs[:,CTX]=ONE2[tape.ctx[:,j]]
                obs[:,CUE]=ONE4[tape.cue[:,j]]
            obs[visible,FRUIT]=ONE4[items[visible]]
            obs[visible,OFFSET]=offset[visible]
            obs[:,VELOCITY]=speed;obs[:,REACH]=reach;obs[:,ACTIVE]=active;obs[:,NEW]=tick==0
            shown=yoke[:,j,tick] if feedback=="yoked" else (-receipt if feedback=="invert" else receipt)
            if feedback=="none" or (hidden_after is not None and j>=hidden_after):
                shown=np.zeros(n,np.int8)
            obs[:,POSITIVE]=shown>0;obs[:,NEGATIVE]=shown<0;obs[:,TASTED]=last_id
            waveform[:,j,tick]=shown
            pulses=np.asarray(policy.act(obs))
            if pulses.shape!=(n,3) or (pulses.dtype!=np.bool_ and not ((pulses==0)|(pulses==1)).all()):
                raise ValueError("Expected [N,3] binary left/right/bite pulses")
            if tick==TICKS-1: break
            # A bite samples the fruit currently under the proboscis, independent of a motion command.
            touched=active & reach & pulses[:,2].astype(bool)
            correct=touched & (items==tape.goal[:,j])
            wrong=touched & ~correct
            was_first=touched & (first[:,j]<0)
            first[was_first,j]=items[was_first];first_tick[was_first,j]=tick
            bites[:,j]+=touched;win[:,j]|=correct
            pos_count[:,j]+=correct;neg_count[:,j]+=wrong
            eaten[rows[touched],sector[touched]]=True
            finished |= correct if retry else touched
            receipt=correct.astype(np.int8)-wrong.astype(np.int8)
            last_id.fill(0.)
            last_id[touched]=ONE4[items[touched]]
            moving=active & ~finished
            motion_pulses[:,j]+=pulses[:,:2].astype(np.int16)*moving[:,None]
            speed=np.where(moving,.70*speed+motion*(pulses[:,1].astype(np.float32)-pulses[:,0]),0.)
            angle=(angle+speed)%SIZE
    return Result(first,first_tick,win,bites,pos_count,neg_count,waveform,first==tape.goal,motion_pulses)


class PublicForager:
    """Reference policies; only 'oracle' receives the private target array."""

    def __init__(self,mode="eliminate",goals=None):
        if mode not in ("oracle","first","eliminate","silent"):raise ValueError(mode)
        self.mode=mode;self.goals=goals

    def reset(self,n):
        self.possible=np.ones((n,2,4,4),bool)
        self.key=np.zeros((n,2),np.int8); self.trial=-1

    def act(self,obs):
        n=len(obs);i=np.arange(n)
        if np.all(obs[:,NEW]):
            self.trial+=1
            self.key[:,0]=obs[:,CTX].argmax(1);self.key[:,1]=obs[:,CUE].argmax(1)
        ctx,cue=self.key.T
        heard=obs[:,TASTED].any(1)
        yes=heard & (obs[:,POSITIVE]>0);no=heard & (obs[:,NEGATIVE]>0)
        identity=obs[:,TASTED].argmax(1)
        if self.mode=="eliminate":
            self.possible[i[yes],ctx[yes],cue[yes]]=False
            self.possible[i[yes],ctx[yes],cue[yes],identity[yes]]=True
            self.possible[i[no],ctx[no],cue[no],identity[no]]=False
        present=obs[:,FRUIT].argmax(1);go=obs[:,ACTIVE]>0;reach=obs[:,REACH]>0
        if self.mode=="oracle":
            bite=present==self.goals[:,self.trial]
        elif self.mode=="eliminate":
            bite=self.possible[i,ctx,cue,present]
        else:
            bite=np.ones(n,bool)
        if self.mode=="silent":go=np.zeros(n,bool)
        return np.column_stack((np.zeros(n,bool),go,go & reach & bite))
