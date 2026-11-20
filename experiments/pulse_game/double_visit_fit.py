"""Fit two matched native graphs from their own routes in the double-visit world."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import double_visit_world as world
from . import reroute_fit as prior_fit
from . import reroute_fit_transfer as inherited


ROOT = prior_fit.ROOT
FIT_ROOT = ROOT / "data/double-visit-20261120"
EVIDENCE = ROOT / "docs/evidence/double-visit-20261120"
PROTOCOL = EVIDENCE / "fit-protocol.json"
BIRTHS = prior_fit.BIRTHS
CONDITIONS = ("baseline", "identity")
TRAINING_MAPS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
HELDOUT_MAPS = ((0, 2, 1), (1, 0, 2), (2, 1, 0))
DOMAIN = 19367129
SMOKE_DOMAIN = 19367239
STEPS, BATCH, EASY_STEPS = 1792, 64, 256
CHECKPOINT_EVERY = 128
QUERY_EXPLORATION, ACTION_EXPLORATION = .45, .30
LR, CLIP, IDENTITY_WEIGHT = .003, 2., 1.
LEDGER_WORLD = (
    "left_room", "right_room", "prior_room0", "prior_room1", "first_inspected_room",
    "second_inspected_room", "first_observed_endpoint", "second_observed_endpoint",
    "first_visited_door", "second_visited_door", "goal_endpoint0", "goal_endpoint1", "goal_endpoint2",
)
LEDGER_TRACE = (
    "query_draw", "query_right_probability", "query", "action_draw", "action_right_probability", "action",
    "actual_room", "actual_endpoint", "paid", "predicted_room", "predicted_payoff", "endpoint_logit0",
    "endpoint_logit1", "endpoint_logit2", "eligible", "calibration_weight", "endpoint_teacher_target",
    "query_logit", "action_logit",
)
METRICS = ("paid", "eligible", "payoff_mse", "room_mse", "room_accuracy", "identity_ce",
           "identity_accuracy", "policy_loss", "gradient_norm")
digest = prior_fit.digest
state_digest = prior_fit.state_digest
atomic = prior_fit.storage._atomic


def source_weight_file(birth):
    return inherited.destination("next_room", birth)


def specification():
    prior_protocol = inherited.register()
    prior_dependencies = json.loads(prior_fit.PROTOCOL.read_text(encoding="utf8"))["source_sha256"]
    paths = {Path(__file__), Path(world.__file__), Path(inherited.__file__)}
    paths.update(ROOT / relative for relative in prior_dependencies)
    return {
        "plan_sha256": digest(EVIDENCE / "fit-plan.md"), "births": BIRTHS, "conditions": CONDITIONS,
        "updates": STEPS, "batch": BATCH, "easy_updates": EASY_STEPS,
        "checkpoint_interval": CHECKPOINT_EVERY, "world_seed": [DOMAIN, "birth", "step", 0],
        "motor_seed": [DOMAIN, "birth", "step", 1], "smoke_domain": SMOKE_DOMAIN,
        "training_maps": TRAINING_MAPS, "heldout_maps": HELDOUT_MAPS,
        "inheritance": {"condition": "next_room", "update": inherited.STEPS,
                        "transfer_protocol_sha256": prior_protocol,
                        "optimizer_inherited": False, "files": {
                            str(birth): {"path": source_weight_file(birth).relative_to(ROOT).as_posix(),
                                         "sha256": digest(source_weight_file(birth))} for birth in BIRTHS}},
        "clock": {"history": world.HISTORY_TICKS, "goal": world.GOAL_TICKS,
                  "preparation": world.PREPARATION_TICKS, "trial": world.TRIAL_TICKS,
                  "commit": world.COMMIT_TICKS},
        "calibration_clock": {"room_trial_tick": world.MID_TICK, "payoff_trial_tick": world.FORECAST_TICK,
                              "identity_trial_tick": world.ENDPOINT_TICK},
        "endpoint_readout": list(range(world.ENDPOINT_READOUT.start, world.ENDPOINT_READOUT.stop)),
        "endpoint_logit_gain": world.ENDPOINT_GAIN,
        "exploration": {"trial": QUERY_EXPLORATION, "commit": ACTION_EXPLORATION},
        "objective": {"policy": "two mixture log-likelihoods with actual payoff and leave-one-out centering",
                      "room_and_payoff": "MSE on the physically taken, self-tried route, divided by its action propensity",
                      "identity": "additional CE on the same eligible route, physical endpoint 0–2 at the three existing primary units",
                      "identity_coefficient": IDENTITY_WEIGHT, "normalize_by_weight_sum": True,
                      "settlement_after_action_is_gradient_free": True, "detach_prefix": False},
        "optimizer": {"name": "Adam", "lr": LR, "gradient_norm_max": CLIP, "decay": 0.,
                      "fresh": True, "foreach": False, "fused": False},
        "backend": "cpu", "threads_per_worker": 1, "deterministic": True, "dtype": "torch.float32",
        "torch": str(torch.__version__), "numpy": np.__version__,
        "world_columns": LEDGER_WORLD, "trace_columns": LEDGER_TRACE, "curve_columns": METRICS,
        "source_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)},
    }


def register():
    content = json.dumps(specification(), indent=2, ensure_ascii=False) + "\n"
    if PROTOCOL.exists():
        if PROTOCOL.read_text(encoding="utf8") != content:
            raise ValueError(f"Current source or settings differ from registration {PROTOCOL}")
    else:
        atomic(PROTOCOL, lambda path: path.write_text(content, encoding="utf8"))
    return digest(PROTOCOL)


def initial_model(birth, model_device="cpu"):
    old, _, _ = inherited.load_final("next_room", birth)
    model = world.DoubleVisitGraph(birth).to(device=model_device, dtype=torch.float32)
    model.load_state_dict(old.state_dict(), strict=True)
    return model


def episode(birth, step, *, smoke=False):
    domain = SMOKE_DOMAIN if smoke else DOMAIN
    scene = world.scenes([domain, birth, step, 0], BATCH, easy=step <= EASY_STEPS)
    if world.heldout_map(scene).any():
        raise AssertionError("Held-out old map in the fit world")
    draws = np.random.default_rng([domain, birth, step, 1]).random((2, BATCH), dtype=np.float32)
    return scene, *torch.from_numpy(draws)


def world_rows(scene):
    return torch.column_stack((scene.door_rooms, scene.prior_endpoints, scene.inspected_rooms,
                               scene.observed_endpoints, scene.visit_order, scene.goals)).to(torch.int8)


def weighted(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.)


def gradient_evidence(shared, trial, objectives):
    checks = {name: prior_fit.witness(objective, shared) for name, objective in objectives.items()}
    if "identity" in objectives:
        activity = trial.states[world.ENDPOINT_TICK - 1].h
        direct, = torch.autograd.grad(objectives["identity"], activity, retain_graph=True)
        untrained_sites = direct.clone()
        untrained_sites[:, world.ENDPOINT_READOUT] = 0.
        if untrained_sites.count_nonzero().item() or direct[:, world.ENDPOINT_READOUT].norm().item() <= 0.:
            raise AssertionError("Direct identity gradient left the three primary neurons")
        checks["identity_direct_neuron_norms"] = direct.norm(dim=0).tolist()
    return checks


def update(model, optimizer, condition, scene, query_draw, action_draw, *, check_grad=False):
    shared = world.history(model, scene)
    ready = world.prepare(model, scene, shared=shared)
    query = world.select(model, ready, query_draw, QUERY_EXPLORATION)
    trial = world.try_side(model, ready, query.side)
    action = world.select(model, trial.final, action_draw, ACTION_EXPLORATION)
    with torch.no_grad():
        outcome = world.settle(model, trial.final, scene, action)
        eligible = query.side == action.side
        propensity = torch.where(query.side.bool(), action.probability, 1. - action.probability)
        weights = eligible.to(propensity) / propensity
        room_target = (2 * outcome.room - 1).to(trial.mid)
        # Ignore ineligible rows inside cross-entropy too. Zero is a dummy, masked class.
        endpoint_target = outcome.endpoint.masked_fill(~eligible, 0)
        advantage = (outcome.paid - outcome.paid.mean()) * len(scene) / (len(scene) - 1)
    policy = -((query.loglikelihood + action.loglikelihood) * advantage).mean()
    payoff = weighted((trial.forecast - outcome.paid).square(), weights)
    room = weighted((trial.mid - room_target).square(), weights)
    if condition == "identity":
        identity = weighted(F.cross_entropy(trial.endpoint_logits, endpoint_target, reduction="none"), weights)
        objective = policy + payoff + room + IDENTITY_WEIGHT * identity
    else:
        with torch.no_grad():
            identity = weighted(F.cross_entropy(trial.endpoint_logits, endpoint_target, reduction="none"), weights)
        objective = policy + payoff + room
    gradient = None
    if check_grad:
        objectives = {"payoff": payoff, "room": room, "policy": policy}
        if condition == "identity":
            objectives["identity"] = identity
        gradient = gradient_evidence(shared, trial, objectives)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    with torch.no_grad():
        room_accuracy = weighted((trial.mid.ge(0) == outcome.room.bool()).to(weights), weights)
        identity_accuracy = weighted((trial.endpoint_logits.argmax(1) == outcome.endpoint).to(weights), weights)
        curve = torch.stack((outcome.paid.mean(), eligible.float().mean(), payoff, room, room_accuracy,
                             identity, identity_accuracy, policy, grad_norm))
        targets = outcome.endpoint.masked_fill(~eligible, -1)
        if condition == "baseline":
            targets.fill_(-1)
        trace = torch.column_stack((query_draw, query.probability, query.side, action_draw,
                                   action.probability, action.side, outcome.room, outcome.endpoint,
                                   outcome.paid, trial.mid, trial.forecast, trial.endpoint_logits,
                                   eligible, weights, targets, query.logit, action.logit))
    return curve, trace, gradient


def destination(birth, condition, root=FIT_ROOT):
    return Path(root) / f"fit-{condition}-{birth}.pt"


def ledger_path(birth, condition, root=FIT_ROOT):
    return Path(root) / f"fit-ledger-{condition}-{birth}.pt"


def signature(birth, condition, protocol):
    return {"birth": birth, "condition": condition, "protocol_sha256": protocol}


def train(birth, condition, *, smoke=False):
    prior_fit.configure("cpu")
    identity = signature(birth, condition, None if smoke else digest(PROTOCOL))
    model = initial_model(birth)
    beginning = state_digest(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False, fused=False)
    total = 2 if smoke else STEPS
    curve = torch.empty((total, len(METRICS)))
    worlds = torch.empty((total, BATCH, len(LEDGER_WORLD)), dtype=torch.int8)
    trace = torch.empty((total, BATCH, len(LEDGER_TRACE)))
    gradients = {}
    start, elapsed = 0, 0.
    ledger = ledger_path(birth, condition)
    final = destination(birth, condition)
    if not smoke and ledger.is_file():
        saved = torch.load(ledger, weights_only=True, map_location="cpu")
        if saved["signature"] != identity or saved["initial_sha256"] != beginning:
            raise ValueError(f"Incompatible training state in {ledger}")
        start = saved["step"]
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        curve[:start], worlds[:start], trace[:start] = saved["curve"], saved["worlds"], saved["trace"]
        gradients, elapsed = saved["gradients"], saved["seconds"]

    running = perf_counter()
    for step in range(start + 1, total + 1):
        scene_step = EASY_STEPS + 1 if smoke and step == 2 else step
        scene, query_draw, action_draw = episode(birth, scene_step, smoke=smoke)
        curve[step - 1], trace[step - 1], evidence = update(
            model, optimizer, condition, scene, query_draw, action_draw,
            check_grad=scene_step in (1, EASY_STEPS + 1),
        )
        worlds[step - 1] = world_rows(scene)
        if evidence is not None:
            gradients[str(scene_step)] = evidence
        if step % CHECKPOINT_EVERY == 0 or step == total:
            seconds = elapsed + perf_counter() - running
            if not smoke:
                payload = dict(signature=identity, initial_sha256=beginning, step=step, state=model.state_dict(),
                               optimizer=optimizer.state_dict(), seconds=seconds, gradients=gradients,
                               curve=curve[:step].clone(), worlds=worlds[:step].clone(), trace=trace[:step].clone())
                atomic(ledger, lambda path: torch.save(payload, path))
            recent = curve[max(0, step - 64):step].mean(0).tolist()
            print(condition, birth, step, [round(value, 5) for value in recent], round(seconds, 2), "s", flush=True)
    if smoke:
        print(gradients, flush=True)
    else:
        payload = dict(signature=identity, step=total, state=model.state_dict(), initial_sha256=beginning,
                       state_sha256=state_digest(model.state_dict()), ledger_sha256=digest(ledger))
        atomic(final, lambda path: torch.save(payload, path))


def load_final(birth, condition, root=FIT_ROOT, device="cpu"):
    if birth not in BIRTHS or condition not in CONDITIONS:
        raise ValueError("Unknown double-visit birth or training condition")
    target = destination(birth, condition, root)
    saved = torch.load(target, weights_only=True, map_location="cpu")
    if saved["signature"] != signature(birth, condition, digest(PROTOCOL)) or saved["step"] != STEPS:
        raise ValueError(f"Incomplete or incompatible final state in {target}")
    model = world.DoubleVisitGraph(birth).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["state"], strict=True)
    if saved["state_sha256"] != state_digest(model.state_dict()):
        raise ValueError(f"State digest mismatch in {target}")
    metadata = {name: value for name, value in saved.items() if name != "state"}
    return model.requires_grad_(False).eval(), metadata, digest(target)


def run_all(jobs):
    register()
    logs = FIT_ROOT / "fit-logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(pair):
        birth, condition = pair
        log = logs / f"{condition}-{birth}.txt"
        with log.open("a", encoding="utf8") as stream:
            command = (sys.executable, "-m", __spec__.name, condition, str(birth), "--worker")
            task = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        print(condition, birth, task.returncode, log, flush=True)
        return task.returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        status = list(pool.map(run, [(birth, condition) for birth in BIRTHS for condition in CONDITIONS]))
    if any(status):
        raise RuntimeError("At least one fit failed. See its per-fit log.")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("condition", nargs="?", choices=CONDITIONS)
    parser.add_argument("birth", nargs="?", type=int, choices=BIRTHS)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help="Use the parent's validated protocol")
    args = parser.parse_args()
    if args.register:
        print(register())
    elif args.jobs:
        run_all(args.jobs)
    elif args.condition and args.birth:
        if not (args.smoke or args.worker):
            register()
        train(args.birth, args.condition, smoke=args.smoke)
    else:
        parser.error("Provide a condition and birth, --register, or --jobs")


if __name__ == "__main__":
    main()
