"""Fit matched native 40-node requests from identical whole-body endpoints."""

from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import torch

from . import fork_lesson_fit as prior
from . import fork_lesson_native_request as native


WORLD = native.WORLD
DATA = prior.DATA.parent / "native-request"
EVIDENCE = prior.EVIDENCE
PLAN = EVIDENCE / "native-request-plan.md"
KINDS = ("binding", "misbound")
BIRTHS = prior.BIRTHS
DOMAIN = 69278203
STEPS = 1536
BATCH = 64
LEARNING_RATE = .005
LEDGER = ("query_draw", "query_right_prob", "query", "driven", "action_draw", "action_right_prob", "action",
          "forecast", "settled", "forecast_weight", "pulse_left", "pulse_right")
CURVE = ("matched", "settled", "forecast_mse", "policy_loss", "gradient_norm")


def artifact(kind, birth):
    return DATA / "fit" / kind / f"birth-{birth}.pt"


def make_model(birth):
    anchor, _, parent = prior.load_final("whole", birth)
    model = native.NativeRequest(birth)
    missing, extra = model.load_state_dict(anchor.state_dict(), strict=False)
    if missing != ["trial_gain"] or extra or model.width != 40:
        raise AssertionError("Native body architecture differs from its whole checkpoint")
    return model, parent


def signature(kind, birth, parent):
    return dict(kind=kind, birth=birth, parent=parent, plan=prior.digest(PLAN), steps=STEPS,
                fit_source=prior.digest(__file__), model_source=prior.digest(native.__file__),
                world_source=prior.digest(WORLD.__file__), reference_source=prior.digest(native.protocol.__file__))


def episode(birth, step):
    scene = WORLD.scenes([DOMAIN, birth, step], BATCH)
    rng = np.random.default_rng([DOMAIN, birth, step, 253])
    query_draw, action_draw = torch.from_numpy(rng.random((2, BATCH), dtype=np.float32))
    return scene, query_draw, action_draw


def update(model, optimizer, kind, birth, step, average):
    scene, query_draw, action_draw = episode(birth, step)
    execution = native.roll(model, scene, query_draw, action_draw, shuffle=kind == "misbound")
    query, action = execution.requested, execution.action
    policy = -((query.loglikelihood + action.loglikelihood) * (execution.paid - average)).mean()
    with torch.no_grad():
        match = query.side == action.side
        selected_probability = torch.where(action.side.bool(), action.probability, 1 - action.probability)
        weight = match / selected_probability
    forecast = ((execution.trial.forecast - execution.paid).square() * weight).sum() / weight.sum().clamp_min(1.0)
    optimizer.zero_grad(set_to_none=True)
    (policy + forecast).backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        average = 0.98 * average + 0.02 * float(execution.paid.mean())
        curve = torch.stack((match.float().mean(), execution.paid.mean(), forecast, policy, gradient))
        trace = torch.column_stack((query_draw, query.probability, query.side, execution.driven,
                                    action_draw, action.probability, action.side, execution.trial.forecast,
                                    execution.paid, weight, execution.trial.pulse))
    return curve, trace, average


def audit(kind, birth, trace):
    if trace.ndim != 3 or trace.shape[1:] != (BATCH, len(LEDGER)) or not torch.isfinite(trace).all():
        raise AssertionError("Incomplete or non-finite training ledger")
    q_draw, q_prob, query, driven, a_draw, a_prob, action, estimate, earned, weight, left, right = trace.unbind(2)
    expected = query if kind == "binding" else query.roll(-1, dims=1)
    chosen_probability = torch.where(action.bool(), a_prob, 1 - a_prob)
    for observed, warranted in ((query.bool(), q_draw < q_prob), (action.bool(), a_draw < a_prob),
                                 (driven, expected), (weight, (query == action) / chosen_probability)):
        if not torch.equal(observed, warranted):
            raise AssertionError("A query, executed action or teacher weight failed its actual training rule")
    if not (((torch.where(driven.bool(), left, right) + .8).abs() <= 1e-8).all()
            and (torch.where(driven.bool(), right, left) >= 0).all() and (estimate.abs() <= 1).all()):
        raise AssertionError("Unexpected clone stimulus or out-of-range native readout activity")
    for i, actual in enumerate(action.long()):
        scene, first_draw, second_draw = episode(birth, i + 1)
        if not torch.equal(q_draw[i], first_draw) or not torch.equal(a_draw[i], second_draw):
            raise AssertionError("Action draw did not use the shared training tape")
        if not torch.equal(earned[i], WORLD.oracle(scene).gather(1, actual[:, None])[:, 0]):
            raise AssertionError("Actual received payout failed independent environment reconstruction")
    return dict(real_routes=int(action.numel()), calibrated_routes=int(weight.ne(0).sum()),
                matched_fraction=float((query == action).float().mean()))


def fit(kind, birth, smoke=False):
    model, parent = make_model(birth)
    wanted = signature(kind, birth, parent)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    target = artifact(kind, birth)
    length = 8 if smoke else STEPS
    curves = torch.empty((length, len(CURVE)))
    routes = torch.empty((length, BATCH, len(LEDGER)))
    begin, average = 0, 0.
    if not smoke and target.is_file():
        previous = torch.load(target, weights_only=True, map_location="cpu")
        if previous["signature"] != wanted:
            raise ValueError(f"Existing checkpoint does not match the experiment: {target}")
        model.load_state_dict(previous["body"])
        optimizer.load_state_dict(previous["optimizer"])
        begin, average = previous["end"], previous["average"]
        curves[:begin], routes[:begin] = previous["curve"], previous["ledger"]
    then = perf_counter()
    for step in range(begin + 1, length + 1):
        curves[step - 1], routes[step - 1], average = update(model, optimizer, kind, birth, step, average)
        if step % 256 == 0 or step == length:
            if not smoke:
                checks = audit(kind, birth, routes) if step == length else None
                changes = {key: int(value.ne(initial[key]).sum()) for key, value in model.state_dict().items()}
                pack = dict(signature=wanted, end=step, body=model.state_dict(), optimizer=optimizer.state_dict(),
                            curve=curves[:step].clone(), ledger=routes[:step].clone(), average=average,
                            audit=checks, changed_entries=changes)
                prior.storage._atomic(target, lambda temporary: torch.save(pack, temporary))
            print(kind, birth, step, curves[max(0, step - 64):step].mean(0).tolist(),
                  round(perf_counter() - then, 2), flush=True)
    if smoke:
        print(audit(kind, birth, routes), flush=True)


def load(kind, birth):
    model, parent = make_model(birth)
    path = artifact(kind, birth)
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["end"] != STEPS or saved["signature"] != signature(kind, birth, parent):
        raise AssertionError(f"Unfinished or incompatible native request fit: {path}")
    model.load_state_dict(saved["body"])
    return model.requires_grad_(False).eval(), saved, prior.digest(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("birth", type=int, choices=BIRTHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fit(args.kind, args.birth, smoke=args.smoke)


if __name__ == "__main__":
    main()
