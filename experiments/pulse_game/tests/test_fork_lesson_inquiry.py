import torch

from experiments.pulse_game import fork_lesson_world as world
from experiments.pulse_game import fork_lesson_whatif as imagined
from experiments.pulse_game.fork_lesson_whatif_eval import matched_direction


def test_question_label_can_only_affect_primary_through_the_physically_delivered_request():
    body=world.ForkGraph(731).requires_grad_(False)
    learner=imagined.WhatIfGraph(body).requires_grad_(False)
    e=world.scenes(14837,7)
    s=imagined.start_trial(learner,e)
    left=torch.zeros(7,dtype=torch.long)
    a=imagined.perform_trial(learner,s,left)
    b=imagined.perform_trial(learner,s,1-left,driven=left)
    torch.testing.assert_close(a.forecast,b.forecast,rtol=0,atol=0)
    torch.testing.assert_close(a.final_state.h,b.final_state.h,rtol=0,atol=0)
    neutral=imagined.perform_trial(learner,s,1-left,blank=True)
    old=imagined.perform_trial(learner,s,left,blank=True)
    torch.testing.assert_close(neutral.forecast,old.forecast,rtol=0,atol=0)
    assert torch.count_nonzero(a.pulse[:, :10])==0 and torch.count_nonzero(a.pulse[:, 12:])==0


def test_energy_matching_uses_a_public_axis_for_an_entirely_silent_donor():
    current=torch.tensor([[[0.,1.],[-3.,4.]]])
    donor=torch.tensor([[[0.,0.],[1.,0.]]])
    fixed=matched_direction(current,donor)
    torch.testing.assert_close(current.norm(dim=-1),fixed.norm(dim=-1))
    torch.testing.assert_close(fixed,torch.tensor([[[1.,0.],[5.,0.]]]))


def test_if_all_outgoing_ordinary_current_is_cut_the_prediction_cannot_teach_those_axons():
    body=world.ForkGraph(733)
    body.requires_grad_(False);body.return_weight.requires_grad_(True)
    e=world.scenes(812,8)
    with torch.no_grad():s=world.cue(body,world.support(body,e),e)
    _,hs,_,_=world.choice_window(body,s,cut_outbound=True,return_outgoing=True)
    assert not body.forecast(hs[-3]).requires_grad
