import numpy as np
import torch

from .. import ring_metric as m, ring_world


def test_batched_triangular_prediction_matches_causal_updates_and_gradients():
    torch.manual_seed(21)
    cfg=m.Settings(neurons=12,hidden=16,latent=7)
    net=m.Metric(cfg).double()
    x=torch.randn(5,17,cfg.inputs,dtype=torch.float64)
    y=torch.where(torch.rand(5,17)>.5,1.,-1.).double()
    valid=(torch.rand(5,17)>.2).double()
    a=net.predict_events(x,y,valid)
    b=net.predict_events(x,y,valid,parallel=False)
    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    pa=tuple(p for name,p in net.named_parameters() if name!="logit_gain")
    ga=torch.autograd.grad((a*valid).square().sum(),pa)
    gb=torch.autograd.grad((b*valid).square().sum(),pa)
    for i,j in zip(ga,gb,strict=True):torch.testing.assert_close(i,j,rtol=1e-10,atol=1e-10)
    yh=y.clone();yh[:,8:]*=-1
    q=net.predict_events(x,yh,valid)
    torch.testing.assert_close(q[:,:9],a[:,:9],rtol=0,atol=0)


def test_numpy_actor_and_offline_event_targets_agree_at_actual_bites():
    torch.manual_seed(7);s=m.Settings(neurons=12,hidden=10,latent=7)
    net=m.Metric(s);seeds=list(range(8))
    class Recording(m.MetricForager):
        def reset(self,n):
            super().reset(n)
            self.predictions=[[] for _ in range(n)]
        def act(self,o):
            event=(o[:,ring_world.POSITIVE]+o[:,ring_world.NEGATIVE])>0
            for i in np.flatnonzero(event&self.last_bite):
                self.predictions[i].append(float(self.last_pred[i]))
            return super().act(o)
    agent=Recording(s,net.export(),seeds,explorer=True,appetite=m.Appetite(random_bite=.5),record=True)
    tape=ring_world.make_tape(seeds,12)
    data=ring_world.rollout(agent,tape,retry=True)
    x,y,valid=m.event_batch(agent.events,48)
    assert int(valid.sum())==int(data.bites.sum())
    with torch.no_grad():q=net.predict_events(*map(torch.from_numpy,(x,y,valid))).numpy()
    for i,row in enumerate(agent.predictions):
        np.testing.assert_allclose(row,q[i,:len(row)],atol=5e-6)
