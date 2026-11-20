import torch

from experiments.pulse_game import interleave_age_train as age, interleave_rule_search as search


def test_paired_rule_search_recovers_known_linear_resource_direction():
    directions=torch.tensor([[[1.,1.],[1.,-1.]]])
    grad=torch.tensor([.4,-.8])
    sigma=.04
    returned=sigma*(directions@grad)
    scores=torch.stack((returned,-returned),dim=2)
    actual=search.direction_gradient(scores,directions,sigma)
    torch.testing.assert_close(actual[0],grad)


def test_antithetic_replicas_have_identical_sensory_and_sampling_tapes():
    cfg=age.Config(pairs=3)
    tape=age.make_tape(cfg,[101,112],20,2)
    for directional in (False,True):
        copied=search.lives(tape,cfg,[101,112],20,2,directional)
        left=copied.left.reshape(2,3,2,2,8,3)
        assert torch.equal(left[:,:,0],left[:,:,1])
        draws=copied.draws.reshape(2,3,2,2,8,5)
        assert torch.equal(draws[:,:,:,0],draws[:,:,:,1])
        assert torch.equal(draws[:,:,0],draws[:,:,1])==directional
