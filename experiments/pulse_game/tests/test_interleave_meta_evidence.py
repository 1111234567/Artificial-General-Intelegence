import numpy as np
import torch

from experiments.pulse_game import interleave_meta_evidence as trust, local_trace_graph as native


def test_two_old_observations_per_object_are_not_information_about_the_regime():
    torch.set_num_threads(1)
    cfg=trust.Config(worlds=7)
    t=trust.batch(cfg,[731],12,2,"direct")
    with torch.no_grad():
        _,_,_,hist=trust.replay(native.spawn(731),native.Rule(731),t,2,cfg,return_history=True)
    _,log_likelihood=trust.reference(t,hist,2,trust.P,trust.H)
    np.testing.assert_allclose(log_likelihood,0,atol=1e-7)


def test_new_objects_own_history_is_physically_identical_in_yoked_test():
    torch.set_num_threads(1)
    cfg=trust.Config(worlds=3)
    g,r=native.spawn(731),native.Rule(731)
    t=trust.batch(cfg,[731],6,5,"direct")
    with torch.no_grad():
        _,reward,_,hist=trust.replay(g,r,t,5,cfg,mode="same_new_contacts",return_history=True)
        _,blank,_,_=trust.replay(g,r,t,5,cfg,mode="blank_prefix")
    cut=hist[:,10:13].reshape(cfg.worlds,2,4,3,2)
    assert torch.equal(cut[:,0],cut[:,1])
    assert (blank>0).float().mean().item()==.5
    assert set(reward.tolist())=={-1.,1.}
