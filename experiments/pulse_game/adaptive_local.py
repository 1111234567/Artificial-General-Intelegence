"""Use public, self-organized habitat addresses while maintaining local responsiveness to new outcomes."""
import numpy as np
from . import habitat_deploy,context_local,mixture_world as food,world


class LocalAdaptive(habitat_deploy.Forager):
    def __init__(self,front,route,seeds,*,motor,slots=6,radius=.09,inherit=True):
        super().__init__(front,{**route,"slots":slots},seeds,start=-1,motor=motor)
        self.slots=slots;self.radius=radius;self.inherit=inherit
    def reset(self,n):
        super().reset(n)
        self.places=context_local.Addresses(n,self.slots,self.radius)
        self.show=np.empty(0,int)
    def address(self):
        if len(self.show):
            born,slot=self.places.register(self.show,self.context[self.show])
            if self.inherit:self.g[born,slot]=self.g[born,0]
        return self.places.code()
    def act(self,obs):
        self.show=np.flatnonzero(obs[:,food.CTX].any(1))
        received=np.flatnonzero((obs[:,world.REWARD_POS]!=obs[:,world.REWARD_NEG])&obs[:,world.CONTACT_ID].any(1))
        # Only the addressed local association is modified by this outcome.
        old=self.g[received].copy();address=self.code[received].copy()
        pulses=super().act(obs)
        if len(received):self.g[received]=np.where(address==0,old,self.g[received])
        return pulses


def public_addresses(appearance,slots,radius,clock):
    n,tr,_=appearance.shape
    a=context_local.Addresses(n,slots,radius)
    ids=np.arange(n);code=np.empty((n,tr,16),np.uint8)
    prev=np.full(n,255,np.uint8)
    born=np.full((n,tr),255,np.uint8)
    for j in range(tr):
        new,slot=a.register(ids,appearance[:,j]);born[new,j]=slot.astype(np.uint8)
        after=a.current.astype(np.uint8)
        code[:,j]=np.where(np.arange(16)[None,:]>=clock[:,j,None],after[:,None],prev[:,None])
        prev=after.copy()
    return code,born,a


def add_sensor_noise(arena,seeds,sd):
    if sd==0:return
    for i,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,1850723]))
        z=arena.appearance[i]
        angle=np.arctan2(z[:,1],z[:,0])+rng.normal(0.,sd,size=len(z))
        # A noisy public receptor is still bounded in the original positive quadrant.
        angle=np.clip(angle,0.,np.pi/2)
        z[:]=np.column_stack((np.cos(angle),np.sin(angle)))
