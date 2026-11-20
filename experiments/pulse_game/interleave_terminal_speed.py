"""Measure the terminal motor shortcut in isolation and in complete physical life evaluations."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
import torch

from . import interleave_adaptive_reference as learner, local_trace_graph as net
from . import revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


def benchmark(pairs):
    torch.set_num_threads(1)
    birth = 731
    graph, rule = learner.old.load_checkpoint(DATA/"retention-objective","odor","uniform",birth,8000)
    fast_act = motor.act
    intervals = {}
    for workload, worlds, age, terminal in (("six_snapshot_actions",128,0,False),
                                            ("four_contact_gap_and_snapshot",48,4,False),
                                            ("sixteen_contact_gap_and_snapshot",48,16,False),
                                            ("sixteen_contact_gap_and_one_paid_action",48,16,True)):
        order = np.random.default_rng(889711)
        layouts = learner.slow.public_tapes(worlds, 643291, 751737)
        batch = 8*worlds
        view = rule.expand(batch)
        init = net.start(graph, batch)
        q = order.permutation(np.tile(np.arange(3), worlds//3)) if terminal else None
        menu = layouts[0].objects[:,:3].repeat_interleave(8,0)
        draws = layouts[0].probe_draws.repeat_interleave(8,0)
        def ordinary(state,left,right,draw):
            return net.act(view,state,left,right,draw)
        def call(actor):
            if age==0:
                values=[]
                for i in range(3):
                    for orientation in (1.,-1.):
                        x=menu[:,i]*orientation
                        result=actor(init,-x,x,draws[:,i])
                        values.append(torch.stack((result.side,result.motor_logit,result.motor_log_prob)))
                return torch.stack(values).numpy()
            with patch.object(learner.motor,"act",actor):
                return learner.life(graph,rule,layouts,np.array([[learner.GAIN0,0.,0.,0.]]),
                                    [age],"absent_a",4,query=q)
        timings=[]
        with torch.no_grad():
            np.testing.assert_array_equal(call(ordinary),call(fast_act))
            orders = order.permutation(np.tile([0,1],pairs//2))
            for first in orders:
                row=[0.,0.]
                for idx in (first,1-first):
                    t=perf_counter_ns()
                    call((ordinary,fast_act)[idx])
                    row[idx]=(perf_counter_ns()-t)/1e6
                timings.append(row)
        dt=np.asarray(timings)
        sampled=order.integers(pairs,size=(5000,pairs))
        speed=dt[:,0].mean()/dt[:,1].mean()
        boot=dt[sampled,0].mean(1)/dt[sampled,1].mean(1)
        intervals[workload]=dict(worlds=worlds,paired_ms_ordinary_fast=timings,
                                  order_first=orders.tolist(),mean_ms=dt.mean(0).tolist(),
                                  speed_ratio=float(speed),ci95=np.quantile(boot,[.025,.975]).tolist())
        print(workload, round(speed,3), flush=True)

    # The terminal decision contributes the same derivative to the earlier physical state
    # and inherited graph at that decision. Native traces still persist during study.
    for optimized in (False,True):
        body=net.Graph(graph.weights.detach().clone().requires_grad_(),
                       graph.bias.detach().clone().requires_grad_(),graph.present)
        state=net.start(body,8)
        h=state.hidden.detach().clone().requires_grad_()
        state=replace(state,hidden=h)
        left=torch.randn(8,3,generator=torch.Generator().manual_seed(911))
        draw=torch.linspace(.05,.95,8)
        chosen=fast_act(state,left,-left,draw) if optimized else net.act(rule.expand(8),state,left,-left,draw)
        gradients=torch.autograd.grad(chosen.motor_log_prob.sum(),(body.weights,body.bias,h))
        values=[chosen.side.detach(),chosen.motor_logit.detach(),chosen.motor_log_prob.detach(),*gradients]
        if not optimized:
            expected=values
        else:
            for a,b in zip(expected,values,strict=True):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
    report=dict(paired_repeats=pairs,torch_version=str(torch.__version__),threads=torch.get_num_threads(),
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),results=intervals,
                notes="CPU, ordinary fp32, single PyTorch worker, other training can run concurrently. "
                      "Balanced random order, paired ratios of mean wall times on identical input. The native "
                      "five-tick state to action and its gradients are identical. Only a terminal discarded "
                      "branch uses the shortcut. Every within-life support contact still updates its traces.")
    (EVIDENCE/"terminal-speed.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",type=int,default=32)
    args=parser.parse_args()
    if args.pairs<4 or args.pairs%2:
        raise ValueError("Use an even number of paired repeats")
    benchmark(args.pairs)
