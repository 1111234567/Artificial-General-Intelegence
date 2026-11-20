"""Audit single-route receipts, paired scene streams and the first actual gradient updates."""

from hashlib import sha256
from itertools import product

import torch

from . import fork_lesson_fit as fit
from . import fork_lesson_world as world
from . import interleave_revisit_train as storage


def matching_scene_columns():
    return [fit.EVENTS.index("route_draw"), *range(fit.EVENTS.index("visit1_object"), len(fit.EVENTS))]


def numeric_digest(tensors):
    digest = sha256()
    for name, tensor in tensors.items():
        digest.update(name.encode("utf8"))
        digest.update(str(tensor.shape).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_curve(curve, route):
    def column(name):
        return route[:, :, fit.EVENTS.index(name)]

    def curve_column(name):
        return curve[:, fit.METRICS.index(name)]

    right = column("action_right").bool()
    actual = column("actual_delayed_return")
    teacher = column("applied_teacher_target")
    teacher = torch.where(teacher.isnan(), actual, teacher)
    probability = column("right_propensity")
    left_estimate = column("estimated_delayed_left")
    right_estimate = column("estimated_delayed_right")
    chosen = torch.where(right, right_estimate, left_estimate)
    selected = torch.where(right, probability, 1 - probability)
    score = torch.where(right, probability.log(), torch.log1p(-probability))
    advantage = (actual - actual.mean(1, keepdim=True)) * fit.BATCH / (fit.BATCH - 1)
    checks = {
        "delayed_return": actual.mean(1),
        "right_fraction": right.float().mean(1),
        "selected_propensity": selected.mean(1),
        "observed_mse": (chosen - actual).square().mean(1),
        "teacher_mse": (chosen - teacher).square().mean(1),
        "policy_loss": -(score * advantage).mean(1),
        "forecast_span": (right_estimate - left_estimate).abs().mean(1),
    }
    for metric, expected in checks.items():
        torch.testing.assert_close(curve_column(metric), expected, atol=2e-7, rtol=1e-6)
    return len(checks)


def verify_first_update(kind, birth, original):
    torch.manual_seed(birth)
    started = world.ForkGraph(birth)
    initial = numeric_digest(started.state_dict())
    optimizer = torch.optim.Adam(started.parameters(), lr=fit.LEARNING_RATE)
    first_curve, first_route = fit.update(started, optimizer, kind, birth, 1, fit.BATCH)
    torch.testing.assert_close(first_curve, original["curve"][0], atol=0, rtol=0)
    torch.testing.assert_close(first_route, original["routes"][0], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(started.mask, original["state"]["mask"], atol=0, rtol=0)
    return initial, sum(parameter.numel() for parameter in started.parameters())


def verify_bridge_credit(birth):
    model = world.ForkGraph(birth)
    episode = world.scenes([fit.DOMAIN, birth, 1], fit.BATCH)
    history = world.support(model, episode)
    ready, activities, _ = world.choice_window(model, world.cue(model, history, episode))
    action = world.choose(model, ready, episode, exploration=fit.EXPLORATION)
    rows = torch.arange(fit.BATCH)
    chosen_forecast = fit.forecast_scores(activities)[rows, action.side]
    with torch.no_grad():
        _, outcome = world.commit(model, ready, episode, action)
        payoff = outcome.payoff
        advantage = (payoff - payoff.mean()) * fit.BATCH / (fit.BATCH - 1)
    policy = -(action.score * advantage).mean()
    teaching = (chosen_forecast - payoff).square().mean()
    named = dict(model.named_parameters())
    prior = torch.autograd.grad(policy, tuple(named.values()), retain_graph=True)
    fit.update_bridge(model, policy, teaching)
    allowed_slices = {
        "weight": (slice(None), slice(world.REGION, None)),
        "return_weight": (slice(None), slice(None)),
        "bias": slice(12, 14),
        "leak_logit": slice(12, 14),
    }
    modified = {}
    for (name, parameter), baseline in zip(named.items(), prior, strict=True):
        changed = parameter.grad.ne(baseline)
        if name in allowed_slices:
            selected = allowed_slices[name]
            modified[name] = int(changed[selected].sum())
            changed[selected] = False
        if changed.any():
            raise ValueError(f"The bridge condition sent auxiliary credit outside its assigned slice: {name}")
    if not all(modified.values()):
        raise ValueError(f"A declared bridge credit path had no measurable gradient at birth {birth}")
    return modified


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    signature = fit.register()
    births = {}
    results = {}
    for birth, kind in product(fit.BIRTHS, fit.KINDS):
        target = fit.destination(kind, birth)
        row = torch.load(target, map_location="cpu", weights_only=True)
        if row["signature"] != {"protocol": signature, "kind": kind, "birth": birth} or row["step"] != fit.STEPS:
            raise ValueError(f"Fit does not match the original protocol: {target}")
        routes = row["routes"]
        verified = fit.verify_routes(kind, routes)
        curves = verify_curve(row["curve"], routes)
        for name, tensor in row["state"].items():
            if not tensor.isfinite().all():
                raise ValueError(f"Nonfinite value in {target} parameter {name}")
        paired_scene = numeric_digest({"known_history_inputs": routes[:, :, matching_scene_columns()]})
        initial, parameters = verify_first_update(kind, birth, row)
        pair = (paired_scene, initial, parameters)
        if births.setdefault(str(birth), pair) != pair:
            raise ValueError(f"Paired fit has different physical input, initialization or capacity: {target}")
        results[f"{kind}/{birth}"] = {"audit_routes": verified, "curve_columns_reconstructed": curves,
                                      "first_update_curve_and_routes_match_bitwise": True,
                                      "checkpoint": fit.digest(target)}
    output = fit.EVIDENCE / "fit-audit.json"
    storage._write_json(output, {
        "protocol": signature,
        "audited_routes": sum(row["audit_routes"] for row in results.values()),
        "paired_initializations": {
            birth: dict(zip(("scenes", "initial_state", "allocated_parameter_count"), data, strict=True))
            for birth, data in births.items()
        },
        "first_step_bridge_credit_nonzero_entries": {
            str(birth): verify_bridge_credit(birth) for birth in fit.BIRTHS
        },
        "fits": results,
    })
    print(output, flush=True)


if __name__ == "__main__":
    run()
