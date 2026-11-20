"""Check the direction, timing and isolation of the self-issued trial interface."""

from dataclasses import fields
from itertools import product

import torch

from . import fork_lesson_whatif as graph
from . import fork_lesson_whatif_fit as fit
from . import interleave_revisit_train as storage


def equal_state(left, right):
    return all(torch.equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))


def equal_rollout(left, right):
    return all(torch.equal(getattr(left, name), getattr(right, name)) for name in
               ("forecast", "motor_logit", "action", "actual_payoff")) and (
        equal_state(left.trial.forecast_state, right.trial.forecast_state)
        and equal_state(left.trial.final_state, right.trial.final_state)
    )


def reproduce_start(kind, birth):
    saved = torch.load(fit.destination(kind, birth), weights_only=True, map_location="cpu")
    initialized = fit.make_graph(birth)
    optimizer = torch.optim.Adam([parameter for parameter in initialized.parameters() if parameter.requires_grad],
                                 lr=fit.LEARNING_RATE)
    curve, trace, _ = fit.update(initialized, optimizer, kind, birth, 1, 0.0)
    torch.testing.assert_close(curve, saved["curve"][0], atol=0, rtol=0)
    torch.testing.assert_close(trace, saved["trace"][0], atol=0, rtol=0)


@torch.no_grad()
def check_interfaces(model, kind, birth):
    episode = graph.world.scenes([7823201, birth], 64)
    prepared = graph.start_trial(model, episode)
    kwargs = {"prepared": prepared, "shuffle_trial": kind == "misbound"}
    direct = graph.rollout(model, episode, episode.draws, 1.0 - episode.draws, **kwargs)
    reinjected = graph.rollout(model, episode, episode.draws, 1.0 - episode.draws,
                               feedback_override=direct.trial.feedback, **kwargs)
    if not equal_rollout(direct, reinjected):
        raise ValueError("Reinjecting the body's own received current altered its activity or actual action")

    fixed_drive = direct.driven_query
    kwargs = {"prepared": prepared, "drive_query": fixed_drive}
    nominal_left = graph.rollout(model, episode, episode.draws, 1.0 - episode.draws, forced_query=0, **kwargs)
    nominal_right = graph.rollout(model, episode, episode.draws, 1.0 - episode.draws, forced_query=1, **kwargs)
    if not equal_rollout(direct, nominal_left) or not equal_rollout(nominal_left, nominal_right):
        raise ValueError("Nominal query side leaked into the primary predictor or motor state")

    # Each published forecast must be fixed before the actual side is drawn or its consequence is emitted.
    actual = tuple(graph.rollout(model, episode, episode.draws, 1.0 - episode.draws, forced_action=side, **kwargs)
                   for side in (0, 1))
    for side, row in enumerate(actual):
        if (row.action != side).any() or row.immediate.any():
            raise ValueError("The forced route did not execute one door with zero immediate return")
        if not torch.equal(row.forecast, direct.forecast) or not equal_state(row.trial.final_state, direct.trial.final_state):
            raise ValueError("Executed route or its future payoff entered the trial predictor")

    localized = direct.trial.pulse.clone()
    localized[:, graph.MOTOR_PORTS] = 0
    if localized.any() or direct.trial.feedback.shape != (graph.TRIAL_TICKS, len(fixed_drive), graph.world.REGION):
        raise ValueError("The isolated clone trial violated its two-port input or five-tick return shape")
    return {"rows": len(fixed_drive), "return_reinjection_bitwise": True, "nominal_side_independence_bitwise": True,
            "later_external_side_independence_bitwise": True, "stimulus_only_at_clone_ports_30_and_31": True}


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence = {}
    for kind, birth in product(fit.KINDS, fit.BIRTHS):
        model, _, sha = fit.load_final(kind, birth)
        reproduce_start(kind, birth)
        evidence[f"{kind}/{birth}"] = check_interfaces(model, kind, birth) | {
            "checkpoint_sha256": sha, "first_update_curve_and_trace_match_bitwise": True,
        }
    output = fit.original.EVIDENCE / "whatif-fit-audit.json"
    storage._write_json(output, {"protocol": fit.register(), "fits": evidence})
    print(output, flush=True)


if __name__ == "__main__":
    run()
