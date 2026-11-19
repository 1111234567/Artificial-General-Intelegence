from dataclasses import replace

import torch

from experiments.pulse_game import revisit_choice_train as train


def differentiable_births(*births):
    graphs = [train.fw.spawn(seed, 8) for seed in births]
    return train.fw.Graph(torch.cat([graph.weights for graph in graphs]).requires_grad_(),
                          torch.cat([graph.bias for graph in graphs]).requires_grad_(),
                          torch.cat([graph.present for graph in graphs]))


def test_tape_keeps_same_odor_and_private_taste_under_independent_side_orders():
    config = train.Config(batch=30)
    tape, donors = train.world_for_update(config, 731, 1)
    orientation = (tape.first_left * tape.return_left).sum(1).sign()
    torch.testing.assert_close(tape.return_left, tape.first_left * orientation[:, None])
    torch.testing.assert_close(tape.return_outcome, tape.first_outcome * orientation)
    assert set(orientation.tolist()) == {-1, 1}
    assert set(tape.first_outcome.tolist()) == {-1, 1}
    assert donors.unique().numel() == config.batch
    assert (donors != torch.arange(config.batch)).all()
    other, _ = train.world_for_update(config, 732, 1)
    assert not torch.equal(tape.first_left, other.first_left)


def test_shuffle_keeps_whole_joint_outcomes_and_ties_have_no_directional_gradient():
    outcomes = torch.tensor([[[1., 1., 1.], [1., -1., -1.], [-1., -1., 1.]]])
    donors = torch.tensor([[2, 0, 1]])
    pair, gate = train.preference_targets(outcomes, donors, shuffled=False)
    pair_shuffled, gate_shuffled = train.preference_targets(outcomes, donors, shuffled=True)
    torch.testing.assert_close(pair, torch.tensor([[0., 1., 0.]]))
    torch.testing.assert_close(gate, torch.tensor([[[0., 0.], [1., 0.], [-1., -1.]]]))
    torch.testing.assert_close(pair_shuffled, pair[:, donors[0]])
    torch.testing.assert_close(gate_shuffled, gate[:, donors[0]])
    scores = torch.zeros_like(pair, requires_grad=True)
    train.preference_loss(scores, pair).sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[0., -.5, 0.]]))


def test_forks_replay_own_actions_with_shared_world_and_unseen_return_taste():
    config = train.Config(batch=9)
    graph = train.fw.spawn(731, config.writers)
    tape, _ = train.world_for_update(config, 731, 9)
    native = train.native
    start = native.start(graph, config.batch)
    action = native.act(start, tape.first_left, -tape.first_left, tape.first_draw)
    experience = native.taste(action, tape.first_left, -tape.first_left, action.side * tape.first_outcome)
    proposals = native.propose(experience, tape.proposal_draws)
    forks = train.return_branches(experience, proposals, tape, max_wait=config.max_wait, keep_connected=True)
    for branch in range(3):
        decision = native.choose(proposals, override=branch)
        committed = native.commit(experience, proposals, decision)
        after_wait = native.wait(committed.state, tape.wait)
        returned = native.act(after_wait, tape.return_left, -tape.return_left, tape.return_draw)
        torch.testing.assert_close(forks.logit[:, branch], returned.motor_logit)
        torch.testing.assert_close(forks.taste[:, branch], returned.side * tape.return_outcome)
    other = train.return_branches(experience, proposals, replace(tape, return_outcome=-tape.return_outcome),
                                  max_wait=config.max_wait, keep_connected=True)
    torch.testing.assert_close(forks.logit, other.logit)
    torch.testing.assert_close(forks.side, other.side)
    torch.testing.assert_close(forks.taste, -other.taste)


def test_batched_births_preserve_loss_and_gradient_independence():
    torch.set_num_threads(1)
    births = (731, 732)
    config = train.Config(batch=5, max_wait=2)
    tapes, donors = zip(*(train.world_for_update(config, birth, 17) for birth in births), strict=True)
    grouped = differentiable_births(*births)
    total, metrics = train.training_step(grouped, train.Tape.concatenate(tapes), torch.stack(donors),
                                         "shuffled", config)
    group_grads = torch.autograd.grad(total, (grouped.weights, grouped.bias))
    for lane, (birth, tape, donor) in enumerate(zip(births, tapes, donors, strict=True)):
        one = differentiable_births(birth)
        loss, trace = train.training_step(one, tape, donor, "shuffled", config)
        grads = torch.autograd.grad(loss, (one.weights, one.bias))
        torch.testing.assert_close(torch.from_numpy(metrics[lane:lane + 1]), torch.from_numpy(trace))
        torch.testing.assert_close(tuple(grad[lane:lane + 1] for grad in group_grads), grads,
                                   atol=1e-7, rtol=2e-5)
