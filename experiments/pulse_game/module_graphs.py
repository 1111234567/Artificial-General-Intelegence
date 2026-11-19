"""Sample disjoint directed cycles, then add edges only between distinct cycles."""
from dataclasses import dataclass
from functools import cache
import random

import numpy as np

from .free_graph_model import WIDTH, strongly_connected


@dataclass(frozen=True)
class ModuleGraph:
    rings: tuple
    within: tuple
    bridges: tuple

    def arcs(self):
        return np.asarray(sorted(self.within+self.bridges,key=lambda x:(x[1],x[0])),dtype=np.int64)

    def record(self):
        n=sum(map(len,self.rings))
        module=np.empty(n,dtype=int)
        for i,ring in enumerate(self.rings): module[list(ring)]=i
        src,dst=np.asarray(self.bridges).T
        arcs=self.arcs()
        return dict(rings=self.rings, internal_arcs=self.within, inter_ring_arcs=self.bridges,
                    ring_in_degree=np.bincount(module[dst],minlength=len(self.rings)).tolist(),
                    ring_out_degree=np.bincount(module[src],minlength=len(self.rings)).tolist(),
                    node_in_degree=np.bincount(arcs[:,1],minlength=n).tolist(),
                    node_out_degree=np.bincount(arcs[:,0],minlength=n).tolist(),
                    strongly_connected=strongly_connected(n,arcs))


@cache
def _cycles(seed,n):
    if n<4: raise ValueError("Need at least two non-self cycles")
    rng=random.Random((int(seed)<<18)^0x78CE216)
    successor=list(range(n))
    while True:
        rng.shuffle(successor)
        if any(i==dst for i,dst in enumerate(successor)): continue
        pending=set(range(n)); rings=[]
        while pending:
            i=min(pending); ring=[]
            while i in pending:
                ring.append(i);pending.remove(i);i=successor[i]
            rings.append(tuple(ring))
        if len(rings)>=2:return tuple(rings),tuple(enumerate(successor))


@cache
def sample(seed,count=64,n=WIDTH):
    """The primary sampler accepts exactly the requested per-ring incoming/outgoing floor."""
    rings,within=_cycles(seed,n)
    k=len(rings)
    if count=="minimum":
        # Minimum globally communicating case: one incoming and one outgoing bridge per ring.
        rng=random.Random((int(seed)<<18)^0x6B4A15)
        order=list(range(k));rng.shuffle(order)
        bridges=tuple((rng.choice(rings[source]),rng.choice(rings[order[(j+1)%k]]))
                      for j,source in enumerate(order))
    else:
        rng=random.Random((int(seed)<<18)^0x471A109)
        mid={node:i for i,ring in enumerate(rings) for node in ring}
        possible=[(s,t) for s in range(n) for t in range(n) if mid[s]!=mid[t]]
        extra=count-n
        if not k<=extra<=len(possible):raise ValueError("Unattainable number of inter-ring edges")
        while True:
            bridges=tuple(rng.sample(possible,extra))
            if len({mid[s] for s,t in bridges})==k and len({mid[t] for s,t in bridges})==k:
                break
    return ModuleGraph(rings,within,bridges)


def graph_for(seed,count=64):
    """Canonical [source,target] arcs for use in the fixed and adaptive models."""
    return sample(int(seed),count).arcs()


def single_sparse(seed):
    """Single backbone with exactly as many extra arcs as the matched minimum bridge arm."""
    n=WIDTH
    extra=len(_cycles(int(seed),n)[0])
    backbone=[((t-1)%n,t) for t in range(n)]
    remaining=[(s,t) for t in range(n) for s in range(n) if s!=t and s!=(t-1)%n]
    rng=random.Random((int(seed)<<18)^0x6810F3)
    return np.asarray(sorted(backbone+rng.sample(remaining,extra),key=lambda x:(x[1],x[0])),dtype=np.int64)
