"""A stored receipt must survive swapping physical sides, and future action draws stay private."""

from dataclasses import replace

import torch

from experiments.pulse_game import edge_site_replay as replay


def test_same_remembered_food_under_mirrored_presentation_gives_the_same_forecast():
    g,r=replay.parents.net.spawn(731),replay.parents.net.Rule(731)
    world=replay.interference.worlds(9921,0,2,30)
    prefix=replay.sites.prepare(g,r,world)
    local=prefix.action_a.side*prefix.experience_a.taste
    menu=world.a*world.orders[:,0,None]
    state=prefix.native_b.state
    first=replay.expected_scores(state,((menu,local),))
    same=replay.expected_scores(state,((-menu,-local),))
    wrong=replay.expected_scores(state,((menu,-local),))
    torch.testing.assert_close(first,same,atol=0,rtol=0)
    torch.testing.assert_close(first+wrong,torch.ones_like(first),atol=1e-6,rtol=0)


def test_actual_test_tape_is_independent_of_number_of_search_groups_and_keeps_past_intact():
    world=replay.interference.worlds(2113,0,2,15)
    _,narrow=replay.random_fields(2,15,3)
    _,wide=replay.random_fields(2,15,17)
    for a,b in zip(narrow,wide,strict=True):
        assert torch.equal(a,b)
    past=replace(world,orders=world.orders.clone(),draws=world.draws.clone())
    downstream,order=replay.test_world(world,narrow)
    assert torch.equal(downstream.orders[:,:3],past.orders[:,:3])
    assert torch.equal(downstream.draws[:,:3],past.draws[:,:3])
    assert torch.equal(world.orders,past.orders) and torch.equal(world.draws,past.draws)
    assert torch.equal(downstream.draws[:,3:5,0],narrow[-1])
    assert torch.equal(order,narrow[1])
