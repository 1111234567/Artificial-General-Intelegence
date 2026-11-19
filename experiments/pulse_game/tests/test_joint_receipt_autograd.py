"""Independent autograd check of online credit for a live sparse recurrent graph."""
import numpy as np
import torch

from experiments.pulse_game.joint_live_model import Population, WEIGHT_NAMES
from experiments.pulse_game.live_module_edits import Schedule


def test_autograd_reaches_early_odors_and_individual_newborn_coefficients():
    rng = np.random.default_rng(84101)
    n, width, channels = 2, 32, 5
    def draw(shape, scale=.3):
        return (rng.normal(size=shape)*scale).astype(np.float32)
    params = dict(drive=draw((n,channels,2*width)), bias=draw((n,2*width)),
                  readout=draw((n,width)), offset=draw((n,)), gain=draw((n,width)))
    ix = np.arange(width)
    support = np.zeros((n,width,width),bool)
    support[:,ix,(ix-1)%width] = True
    support[:,ix,(ix+5)%width] = True
    actor = Population(params,support,Schedule(9,6,.5),[143,146],learning_rate=.03)
    actor.state = draw((n,width))
    actor.begin_request(learn_edges=True)
    # A new active contact can have zero weight yet receive a gradient.
    target, source = 5, 15
    actor.mask[:,target,source] = True
    actor.newborn_trial[:,target,source] = 0
    actor.refresh_matrices()
    assert np.all(actor.raw[:,target,source] == 0)
    start = actor.state.copy()
    signals = np.zeros((13,n,channels),np.float32)
    signals[1:4,:,:4] = draw((3,n,4),1.)
    signals[7:,:,:] = 0
    signals[7:,:,4] = 1
    active = np.zeros((len(signals),n),np.float32)
    active[7:11,0] = 1
    active[7:13,1] = 1
    for t,x in enumerate(signals):
        actor.act(x,active[t] if t>=7 else None,record=True)
    sides = np.array([0,1]); tastes = np.array([1,1])
    sign = (2*sides-1)*tastes
    credit = sign/(1+np.exp(sign*actor.logit_sum/actor.trace_ticks))/actor.trace_ticks
    gradients = actor.gradients(credit.astype(np.float32))

    def tensor(a, train=False):
        return torch.tensor(a,dtype=torch.float64,requires_grad=train)
    raw = tensor(actor.raw,True)
    drive,bias,readout,offset = (tensor(params[k],True) for k in WEIGHT_NAMES[1:])
    matrix = raw*tensor(actor.mask)/tensor(np.sqrt(actor.degree))[...,None]
    state = tensor(start)
    logits = []
    for x in signals:
        candidate,retain = (torch.einsum("bi,bij->bj",tensor(x),drive)+bias).chunk(2,dim=-1)
        keep=retain.sigmoid()
        state=keep*state+(1-keep)*torch.tanh(candidate+(matrix@state.unsqueeze(-1)).squeeze(-1))
        logits.append((readout*state).sum(axis=1)+offset)
    mean=(torch.stack(logits)*tensor(active)).sum(axis=0)/tensor(active.sum(axis=0))
    log_probability=torch.nn.functional.logsigmoid(tensor(sign)*mean).sum()
    reference=torch.autograd.grad(log_probability,(raw,drive,bias,readout,offset))
    for actual,expected in zip(gradients,reference,strict=True):
        np.testing.assert_allclose(actual,expected.detach().numpy(),atol=1.2e-7,rtol=2e-5)
    assert np.all(np.linalg.norm(gradients[1][:,:4],axis=(1,2))>1e-5)
    assert np.all(np.abs(gradients[0][:,target,source])>1e-7)
    actor.finish_request(sides,tastes)
    assert np.all(np.sign(actor.raw[:,target,source])==np.sign(gradients[0][:,target,source]))
    assert np.all(actor.input_step[:,:4]>0)
    assert np.all(actor.weight_step>0)
