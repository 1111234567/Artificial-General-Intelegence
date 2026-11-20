import numpy as np
import torch

from experiments.pulse_game import interleave_integrated_stability as integrated, interleave_stream as stream
from experiments.pulse_game import local_trace_graph as native


def test_unmodified_integrated_branch_is_the_same_continuous_stream():
    torch.set_num_threads(1)
    tape=stream.tape_for_worlds(731,4,4,743)
    graph=native.spawn(731);rule=native.Rule(731)
    old=stream.evaluate(graph,rule,tape,743,731,(0,1,2,4),("carry",))
    new=integrated.evaluate(graph,rule,tape,731,743,(0,1,2,4))
    np.testing.assert_array_equal(old["carry/credit"],new["native/earned"])
    np.testing.assert_array_equal(old["carry/read"][:,:5],new["native/probe"][:,:5])


def test_context_queries_have_same_odor_but_distinct_visible_scene():
    tape=stream.tape_for_worlds(732,4,16,753)
    contexts,valid,conflict,_=integrated.context_questions(tape,732,753)
    l1,r1,*_=contexts[0];l2,r2,*_=contexts[1]
    assert valid.any()
    torch.testing.assert_close((r1-l1)[valid],(r2-l2)[valid],atol=5e-6,rtol=1e-6)
    assert ((r1+l1-r2-l2)[valid].abs().sum(1)>.5).all()
