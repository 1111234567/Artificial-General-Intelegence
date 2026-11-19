"""A signed odor compresses one actual taste and its public continuous sensory address."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Settings:
    capacity:int=56
    lookback:int=9
    temperature:float=7.
    threshold:float=.54
    polar:bool=True
    balance:bool=True
    mix:float=.66
    timelag:float=.015


def norm(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-7)


def project(c,polar):
    if polar:
        u,_,v=np.linalg.svd(c,full_matrices=False)
        out=u@v
    else:
        out=c/np.maximum(np.sqrt((c*c).sum((1,2),keepdims=True)/c.shape[-1]),1e-7)
    return np.where((np.abs(c).sum((1,2))>1e-7)[:,None,None],out,0).astype(np.float32)


def compatible(history,labels,count,cfg):
    length=history.shape[1];idx=np.arange(length)
    delay=idx[None,:]-idx[:,None]
    legal=(delay>0)&(delay<=cfg.lookback)
    legal=legal[None,:,:] & (labels[:,:,None]==labels[:,None,:]) & (idx[None,None,:]<count[:,None,None])
    # Pure public chronological and habitat constraints. No true source/target
    # correspondence is provided, even when the environment knows its own pairs.
    return legal,delay


def initialize(history,labels,count,cfg):
    legal,lag=compatible(history,labels,count,cfg)
    weight=np.where(legal,np.exp(-cfg.timelag*lag),0.).astype(np.float32)
    cross=history.transpose(0,2,1)@weight@history
    return project(cross,cfg.polar)


def revise(history,labels,count,matrix,cfg):
    """A current operator proposes soft links between previously observed signed odors."""
    legal,lag=compatible(history,labels,count,cfg)
    predictions=norm(history@matrix)
    sim=predictions@history.transpose(0,2,1)
    exponent=cfg.temperature*(sim-cfg.threshold)-cfg.timelag*lag
    weight=np.exp(np.where(legal,exponent,-60.).astype(np.float32))*legal
    if cfg.balance:
        # Both a possible earlier and later event can stay unmatched; repeated
        # links to one chance neighbor are moderated symmetrically.
        left=1.+weight.sum(2,keepdims=True)
        right=1.+weight.sum(1,keepdims=True)
        strength=weight/np.sqrt(left*right)
    else:strength=weight/(1.+weight.sum(1,keepdims=True))
    cross=history.transpose(0,2,1)@strength@history
    update=project(cross,cfg.polar)
    return project((1.-cfg.mix)*matrix+cfg.mix*update,cfg.polar)
