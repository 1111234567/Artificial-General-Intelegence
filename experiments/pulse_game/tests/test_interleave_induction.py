import numpy as np
import torch

from experiments.pulse_game import interleave_induction as induction
from experiments.pulse_game import interleave_induction_train as training
from experiments.pulse_game import local_trace_graph as net


def test_independent_training_targets_are_unknowable_on_first_Y_even_given_complete_history():
    cfg=training.Config(worlds=16)
    w=training.tape(cfg,731,42,8,"unrelated",training.DOMAIN)
    prefix=w["factor"].reshape(16,8,-1)[:,:,:-1]
    target=w["factor"].reshape(16,8,-1)[:,:,-1]
    novel=w["kind"].reshape(16,8)[:,0]==1
    assert novel.any()
    np.testing.assert_array_equal(prefix[novel,::2],prefix[novel,1::2])
    np.testing.assert_array_equal(target[novel,::2],-target[novel,1::2])
    assert not (w["right"][:,-1][w["kind"]==1]==w["right"][:,-2][w["kind"]==1]).all(1).any()


def test_heldout_exception_and_rule_flip_cannot_leak_to_first_x_action():
    t=induction.world(17,731,6,"linear",.2,4)
    z=induction.evaluate(net.spawn(731),net.Rule(731),t,(0,4))
    for k in (0,4):
        choices=z[f"coherent/{k}/X_side"].reshape(-1,4)
        np.testing.assert_array_equal(choices[:,0],choices[:,1])
        np.testing.assert_array_equal(choices[:,2],choices[:,3])
        normal=t["x_sign"].numpy()>0
        assert z[f"unrelated_history/{k}"][normal,0].mean()==.5
