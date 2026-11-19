import unittest

import torch

from experiments.coverage_exit.fast_window import WindowBlock
from experiments.undirected.model import PortGraphModel, UndirectedModel


class WindowEquivalence(unittest.TestCase):
    def test_state_logits_and_derivatives_with_variable_dt(self):
        for kind in (UndirectedModel,PortGraphModel):
            for mode in ("undirected","untied"):
                with self.subTest(kind=kind.__name__,mode=mode), torch.random.fork_rng():
                    torch.manual_seed(5394)
                    kwargs = dict(input_size=3,n_jobs=1,n_actions=2,width=7,mode=mode,input_scale=(1.,.25,2.))
                    model = kind(**kwargs).double()
                    with torch.no_grad():
                        for param in model.parameters():
                            param.normal_(std=.15)
                    x = torch.randn(5,2,3,dtype=torch.float64,requires_grad=True)
                    h = torch.randn(2,7,dtype=torch.float64,requires_grad=True)
                    steps = (0.,.25,.05,.62,.38)
                    ordinary, state = [], h
                    for obs,dt in zip(x,steps):
                        logits,state = model.step(obs,state,dt)
                        ordinary.append(logits)
                    packed, end = WindowBlock(model)(x,h,steps,True,True)
                    native = torch.stack(ordinary)
                    torch.testing.assert_close(packed,native,rtol=0,atol=0)
                    torch.testing.assert_close(end,state,rtol=0,atol=0)
                    leaves = (x,h,*model.parameters())
                    expected = torch.autograd.grad(native.square().sum()+state.sum(),leaves)
                    actual = torch.autograd.grad(packed.square().sum()+end.sum(),leaves)
                    for a,b in zip(actual,expected):
                        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
