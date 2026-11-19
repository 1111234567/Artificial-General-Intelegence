"""Replace labelled odors by stable but entirely new receptor mixtures in each life."""

import numpy as np

from . import ring_world as ring


def codebooks(seeds,bank,style):
    if style not in ("permutation","anchored","novel","noisy"):
        raise ValueError("Unknown receptor family")
    matrices=[]
    for s in seeds:
        rng=np.random.default_rng(np.random.SeedSequence([int(s),0xA2801,bank]))
        basis=np.eye(4,dtype=np.float32)[rng.permutation(4)]
        if style=="permutation":
            matrices.append(basis);continue
        if style=="anchored":
            p=.55*basis+.45*rng.dirichlet(np.ones(4),size=4)
        else:
            palette=[]
            # Four arbitrary nonnegative odors. Separation is a sensory
            # resolvability condition shared by all comparators, not a label.
            while len(palette)<4:
                v=rng.dirichlet(np.full(4,.85))
                v=v/np.linalg.norm(v)
                if all(np.linalg.norm(v-q)>.50 for q in palette):palette.append(v)
            p=np.stack(palette)
        matrices.append((p/np.linalg.norm(p,axis=1,keepdims=True)).astype(np.float32))
    return np.stack(matrices)


class ChemicalInterface:
    """The same receptor mixture is returned at visual sampling and at taste."""

    def __init__(self,child,seeds,style,*,cue=True,fruit=True):
        self.child,self.seeds,self.style=child,tuple(seeds),style
        self.cues=codebooks(seeds,1,style) if cue else None
        self.fruits=codebooks(seeds,2,style) if fruit else None

    def reset(self,n):
        if n!=len(self.seeds):raise ValueError("Chemical-interface life count differs")
        self.child.reset(n)
        self.rng=[np.random.default_rng(np.random.SeedSequence([s,0x6A641])) for s in self.seeds]
        self.noise=None;self.tick=0

    def act(self,obs):
        x=obs.copy()
        for sl,mat in ((ring.CUE,self.cues),(ring.FRUIT,self.fruits),(ring.TASTED,self.fruits)):
            if mat is not None:
                x[:,sl]=np.einsum("bi,bij->bj",obs[:,sl],mat)
        if self.style=="noisy":
            index=self.tick%64
            if index==0:
                self.noise=np.stack([r.standard_normal((64,8),dtype=np.float32) for r in self.rng],axis=1)
            # Sensor noise applies only while a physical cue or fruit is present.
            for sl,noise_slice in ((ring.CUE,slice(0,4)),(ring.FRUIT,slice(4,8))):
                x[:,sl]=np.maximum(0,x[:,sl]+.045*self.noise[index,:,noise_slice]*obs[:,sl].any(1)[:,None])
        self.tick+=1
        return self.child.act(x)
