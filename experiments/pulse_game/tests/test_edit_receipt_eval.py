import numpy as np
import torch

from experiments.pulse_game import edit_receipt_eval as ev
from experiments.pulse_game import edit_receipt_graph as native
from experiments.pulse_game import edit_receipt_train as training
from experiments.pulse_game import self_param_writer as fw


def test_paired_tapes_share_public_inputs_and_exogenous_draws():
    settings = ev.Settings(worlds=7, holdout_worlds=7, contacts=20, phase_contacts=12)
    tapes = ev.make_tapes(settings, 137, {"contacts": 8, "noise": .1})
    stable, reversal = (tapes["deployment"][name] for name in ev.SCENARIOS)
    flip = ev.reversal_at(settings)
    assert torch.equal(stable.left, reversal.left)
    assert torch.equal(stable.draws, reversal.draws)
    assert torch.equal(stable.truth[:, :flip], reversal.truth[:, :flip])
    assert torch.equal(stable.outcome[:, flip:], -reversal.outcome[:, flip:])
    assert torch.equal(stable.outcome != stable.truth, reversal.outcome != reversal.truth)

    a, c, b, ret = (tapes["return"][name] for name in ("A_first", "C_first", "B", "A_return"))
    assert torch.equal(a.left, c.left) and torch.equal(a.draws, c.draws)
    assert torch.equal(a.truth, -c.truth) and torch.equal(a.outcome, -c.outcome)
    # Orient each sampled odor by its target sign to compare its latent assignment.
    a_bank = torch.stack([(a.left * a.truth[..., None])[i, a.cue[i].argsort(stable=True)[::4]]
                          for i in range(settings.worlds)])
    ret_bank = torch.stack([(ret.left * ret.truth[..., None])[i, ret.cue[i].argsort(stable=True)[::4]]
                            for i in range(settings.worlds)])
    assert torch.equal(a_bank, ret_bank)
    assert not torch.equal(a.left, b.left)
    donor = tapes["yoked_donor"]["stable_noise"]
    paired = stable.interleave(donor)
    assert torch.equal(paired.draws[::2], stable.draws)
    assert torch.equal(paired.draws[1::2], donor.draws)
    assert not torch.equal(donor.draws, stable.draws)
    assert not torch.equal(stable.draws, tapes["fork_holdout"]["stable_noise"].draws)


@torch.no_grad()
def test_evaluation_fork_matches_training_targets_and_actual_sides():
    settings = ev.Settings(worlds=8, contacts=14)
    tape = ev.environment_tapes(settings, 137, "deployment")["reversal"]
    focal, horizon = 6, 3
    state = native.start(fw.spawn(137, 4), settings.worlds)
    policy = ev.policy_settings("causal", {})
    proposal = ev.propose(state, tape, focal, policy)
    ordinary = ev.commit(state, proposal, tape.draws[:, focal], policy)
    measured = ev.fork_contact(state, proposal, ordinary, tape, focal, policy, horizon)
    trained = training.future_fork(
        state, proposal, training.Tape(tape.left, tape.truth, tape.outcome, tape.draws),
        focal, horizon, sensing="own", keep_connected=True,
    )
    torch.testing.assert_close(measured["applied"], trained.eligible)
    torch.testing.assert_close(measured["delta_positive_taste"], trained.taste)
    torch.testing.assert_close(measured["delta_p_positive_taste"], trained.expected_taste)
    torch.testing.assert_close(measured["delta_correct"], trained.physical)
    torch.testing.assert_close(measured["delta_p_correct"], trained.expected)
    torch.testing.assert_close(measured["action_changed"].float().mean(1), trained.changed_side)


@torch.no_grad()
def test_disabled_deployment_and_graft_preserve_frozen_body():
    settings = ev.Settings(worlds=4, contacts=12)
    tape = ev.environment_tapes(settings, 3, "deployment")["stable_noise"]
    cold = native.start(fw.spawn(3, 4), settings.worlds)
    policy = ev.policy_settings("sensing", {})
    final, record, _ = ev.deploy(cold, tape, policy, intervention="disabled")
    assert not record["applied"].any()
    for key in ("weights", "bias", "present"):
        assert torch.equal(getattr(final.graph, key), getattr(cold.graph, key))
    donor_graph = fw.Graph(cold.graph.weights + cold.graph.present * .1, cold.graph.bias + .2, cold.graph.present)
    donor = native.State(donor_graph, torch.full_like(cold.hidden, .7))
    graft, valid = ev.writer_graft(cold, donor)
    assert valid.all() and torch.equal(graft.hidden, cold.hidden)
    for key in ("weights", "bias", "present"):
        actual = getattr(graft.graph, key)
        assert torch.equal(actual[:, :fw.BODY], getattr(cold.graph, key)[:, :fw.BODY])
        assert torch.equal(actual[:, fw.BODY:], getattr(donor.graph, key)[:, fw.BODY:])


def test_calibration_distinguishes_zero_teacher_from_wrong_positive_prediction():
    target = np.array([0., 0., .25, -.25])
    prediction = np.array([.2, .2, .25, .25])
    score = ev.calibrate(prediction, target, constant=.01)
    assert score["count"] == 4 and score["nonzero_count"] == 2
    assert score["sign_accuracy_on_nonzero"] == .5
    assert score["mse"] > score["zero_baseline_mse"]
    assert score["development_constant_mse"] > score["zero_baseline_mse"]
