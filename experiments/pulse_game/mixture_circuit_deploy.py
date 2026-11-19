"""Physical deployment of a frozen recurrent sensory graph and fast synapses."""
import numpy as np
from . import mixture_brain, mixture_world as game, world


class FlashForager(mixture_brain.MixtureLearner):
    def __init__(self,params,net,seeds,*,writes=True,cut_fast=False,freeze=None,tonic=.39,gain=1.6,
                 disrupt=None):
        self.params=params
        self.allow_writes=writes
        self.cut_fast=cut_fast
        self.disrupt=disrupt
        rule=mixture_brain.Rule(representation="spikes",neurons=params["nodes"],latent=params["latent"],rate=1.,
                                ridge=params["ridge"],covariance=params["learner"]=="ridge",
                                motor_tonic=tonic,motor_gain=gain)
        super().__init__(rule,net,seeds,freeze=freeze)

    def reset(self,n):
        super().reset(n)
        self.state=np.zeros((n,2,self.params["nodes"]),np.float32)
        self.prev_write=np.zeros_like(self.prior_features)
        self.prev_coeff=np.zeros((n,2,3),np.float32)
        self.tick=-1

    def _write(self,obs):
        if not self.allow_writes or (self.freeze is not None and self.round>=self.freeze):return
        if self.params["learner"]=="ridge":return super()._write(obs)
        y=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        rows=np.flatnonzero((y!=0)&obs[:,world.CONTACT_ID].any(1))
        if len(rows)==0:return
        side=obs[rows,world.CONTACT_ID].argmax(1)
        coeff=self.prev_coeff[rows,side]
        pred=self.prior_value[rows,side]
        change=coeff[:,0]*y[rows]+coeff[:,1]*pred+coeff[:,2]
        self.weights[rows]+=change[:,None]*self.prev_write[rows,side]
        self.writes[rows]+=1

    def _features(self,x,ctx):
        p=self.params
        def layer(v,key):
            w,b=p[key];return v@w+b
        def norm(k):return k/np.sqrt(.16+(k*k).sum(-1,keepdims=True))
        observations=x@p["input"][:4]+(ctx@p["input"][4:])[:,None,:]
        self.tick=(self.tick+1)%world.TICKS
        if self.disrupt=="silence" and self.tick in (4,5,6,7):
            self.state.fill(0)
        self.state=self.state@p["graph"]+observations
        a=layer(self.state,"a")
        hidden=np.tanh(a*(1.+np.tanh(layer(self.state,"b"))))
        raw=layer(hidden,"read")
        key=norm(raw)
        if p["learner"]=="local":
            self.prev_write=norm(raw+.30*layer(hidden,"write"))
            coeff=layer(hidden,"coeff")
            self.prev_coeff=np.stack((1.65*np.tanh(coeff[:,:,0]),-1.65/(1.+np.exp(-coeff[:,:,1])),
                                      .30*np.tanh(coeff[:,:,2])),2)
        elif p["learner"]=="delta":
            rate=1.8/(1.+np.exp(-p["coeff"][1][0]))
            self.prev_write=key
            self.prev_coeff[:]=[rate,-rate,0.]
        return key if not self.cut_fast else np.zeros_like(key)
