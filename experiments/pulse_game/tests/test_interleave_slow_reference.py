import numpy as np
import torch

from experiments.pulse_game import interleave_homeostasis as original, interleave_slow_reference as slow
from experiments.pulse_game import local_trace_graph as net


def test_slow_rate_zero_reproduces_established_birth_reference():
    torch.set_num_threads(1)
    tapes=slow.public_tapes(3,441,739)
    g,r=net.spawn(731),net.Rule(731)
    measured=slow.live(g,r,tapes,"absent_a",1,.12,0)
    source=original.replay(g,r,tapes[0],"absent_a",slow.AGES,"birth",.12)
    reference=original.correctness(source)
    np.testing.assert_array_equal(measured["correct"][:,:,[3,0],::-1,:5],reference)
    assert not measured["distance"][...,0].any()


def test_swapping_frozen_slow_memory_cannot_change_recall():
    torch.set_num_threads(1)
    tapes=slow.public_tapes(2,418,730)
    g,r=net.spawn(732),net.Rule(732)
    a=slow.live(g,r,tapes,"absent_a",4,.12,0.)
    b=slow.live(g,r,tapes,"absent_a",4,.12,0.,swap_slow=True)
    np.testing.assert_array_equal(a["correct"],b["correct"])
