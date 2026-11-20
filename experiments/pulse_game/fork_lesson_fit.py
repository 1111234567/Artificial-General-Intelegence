"""Fit matched 40-node bodies with realized delayed returns and optional prediction credit."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from torch.nn import functional as F

from . import edge_site_train_fit as ancestors
from . import fork_lesson_world as world
from . import interleave_revisit_train as storage
from . import native_rehearsal_graph as graph
from . import native_rehearsal_trial as trial


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/fork-lesson-20261120/fit"
EVIDENCE = ROOT / "docs/evidence/fork-lesson-20261120"
PROTOCOL = EVIDENCE / "fit-protocol.json"
KINDS = ("reward", "whole", "sham", "bridge", "cut_out", "cut_back")
BIRTHS = (731, 733, 735)
DOMAIN = 76312417
STEPS = 1536
BATCH = 64
CHECKPOINT_EVERY = 256
EXPLORATION = 0.25
LEARNING_RATE = 0.003
FORECAST_WEIGHT = 1.0
AFTER_FORECAST_TICKS = 2
METRICS = ("delayed_return", "right_fraction", "selected_propensity", "observed_mse", "teacher_mse",
           "policy_loss", "gradient_norm", "forecast_span", "return_current_rms")
EVENTS = (
    "route_draw", "right_propensity", "action_right", "main_motor_logit",
    "estimated_delayed_left", "estimated_delayed_right", "actual_delayed_return", "applied_teacher_target",
    "visit1_object", "visit1_x", "visit1_y", "visit1_z", "visit1_side", "visit1_actual_taste",
    "visit2_object", "visit2_x", "visit2_y", "visit2_z", "visit2_side", "visit2_actual_taste",
    "door_left_x", "door_left_y", "door_left_z", "door_right_x", "door_right_y", "door_right_z",
    "door_left_multiplier", "door_right_multiplier",
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def specification():
    inputs = [Path(__file__), Path(world.__file__), Path(graph.__file__), Path(trial.__file__),
              Path(graph.parent.__file__), Path(storage.__file__)]
    cohort = ancestors.parent_paths()
    return {
        "conditions": {
            "reward": "Realized delayed return and policy likelihood.",
            "whole": "Reward objective and realized-route whole-body early forecast loss.",
            "sham": "Same as whole, with the following independent row's realized return as the forecast target.",
            "bridge": ("Same as whole, with auxiliary credit applied to both directions' base bridge weights "
                       "and the bias and leak of primary neurons 12 and 13. Policy credit reaches the entire graph."),
            "cut_out": "Same as whole, with the primary-to-replica current bridge disabled from door cue to choice.",
            "cut_back": "Same as whole, with the replica-to-primary current bridge disabled from door cue to choice.",
        },
        "graph": "One 40-node TrialGraph with two 20-node populations, the same architecture in every fit.",
        "current_cuts": "The specified current bridge is zero. Shared recurrence, draft and adoption still run.",
        "births": BIRTHS,
        "training_world_seed": [DOMAIN, "birth", "optimizer_step"],
        "steps": STEPS,
        "routes_per_step": BATCH,
        "checkpoint_interval": CHECKPOINT_EVERY,
        "forecast": ("Read primary neurons 12 and 13 two recurrent ticks before motor neurons 10 and 11 commit. "
                     "Forecast loss is squared error at the selected door's own delayed ±1 payoff. "
                     "The graph runs the final two ticks normally subject to the assigned current intervention."),
        "forecast_weight": FORECAST_WEIGHT,
        "feedback": "Each training route physically commits just one door and receives that door's delayed taste.",
        "exploration": {"uniform_mixture_weight": EXPLORATION, "record": "Exact behavior right propensity and draw."},
        "policy": "Actual delayed return centered by a leave-one-route-out batch baseline, scored at the mixture likelihood.",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "max_gradient_norm": 2.0},
        "curve_columns": METRICS,
        "route_ledger_columns": EVENTS,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "parent_sha256": {str(birth): digest(cohort[str(birth)]) for birth in BIRTHS},
    }


def register():
    spec = specification()
    encoded = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    if PROTOCOL.is_file():
        if PROTOCOL.read_text(encoding="utf8") != encoded:
            raise ValueError(f"Frozen training protocol differs from the current configuration: {PROTOCOL}")
    else:
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
        storage._atomic(PROTOCOL, lambda temp: temp.write_text(encoded, encoding="utf8"))
    return digest(PROTOCOL)


def destination(kind, birth):
    return DATA / kind / f"seed-{birth}.pt"


def forecast_scores(activities):
    return activities[-1 - AFTER_FORECAST_TICKS, :, 12:14]


def window_flags(kind):
    return {"cut": kind == "cut_back", "cut_outbound": kind == "cut_out"}


def update_bridge(model, policy_loss, teaching):
    parameters = (model.weight, model.return_weight, model.bias, model.leak_logit)
    auxiliary = torch.autograd.grad(FORECAST_WEIGHT * teaching, parameters, retain_graph=True)
    policy_loss.backward()
    with torch.no_grad():
        model.weight.grad[:, 20:].add_(auxiliary[0][:, 20:])
        model.return_weight.grad.add_(auxiliary[1])
        model.bias.grad[12:14].add_(auxiliary[2][12:14])
        model.leak_logit.grad[12:14].add_(auxiliary[3][12:14])


def record_visit(visit):
    return torch.column_stack((visit.object_id, visit.cue, visit.side, visit.payoff))


def update(model, optimizer, kind, birth, step, batch):
    episode = world.scenes([DOMAIN, birth, step], batch)
    support, visits = world.support(model, episode, return_receipts=True)
    intervention = window_flags(kind)
    presented = world.cue(model, support, episode, **intervention)
    ready, activities, current = world.choice_window(model, presented, **intervention)
    estimates = model.forecast(activities[-1 - AFTER_FORECAST_TICKS])
    action = world.choose(model, ready, episode, exploration=EXPLORATION)
    right_prob = (1.0 - EXPLORATION) * action.logit.sigmoid() + EXPLORATION / 2.0
    selected = torch.where(action.side == 1, right_prob, 1.0 - right_prob)
    chosen_estimate = estimates.gather(1, action.side[:, None]).squeeze(1)

    with torch.no_grad():
        _, outcome = world.commit(model, ready, episode, action)
        realized = outcome.payoff.detach()
        advantage = (realized - realized.mean()) * batch / (batch - 1)
        target = realized.roll(-1) if kind == "sham" else realized

    policy_loss = -(action.score * advantage).mean()
    teaching = F.mse_loss(chosen_estimate, target)
    optimizer.zero_grad(set_to_none=True)
    if kind == "bridge":
        update_bridge(model, policy_loss, teaching)
    else:
        loss = policy_loss if kind == "reward" else policy_loss + FORECAST_WEIGHT * teaching
        loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
    optimizer.step()

    with torch.no_grad():
        curve = torch.stack((realized.mean(), action.side.float().mean(), selected.mean(),
                             F.mse_loss(chosen_estimate, realized), teaching, policy_loss, gradient,
                             (estimates[:, 1] - estimates[:, 0]).abs().mean(), current.square().mean().sqrt()))
        ledger_target = torch.full_like(realized, torch.nan) if kind == "reward" else target
        route = torch.column_stack((episode.draws, right_prob, action.side, action.logit, estimates,
                                    realized, ledger_target, *map(record_visit, visits),
                                    world.door_cues(episode)[:, :8]))
        if route.shape != (batch, len(EVENTS)):
            raise ValueError(f"Unexpected training ledger shape: {route.shape}")
        return curve, route


def fit(kind, birth, *, smoke=False, protocol=None):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(birth)
    model = world.ForkGraph(birth)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    target = destination(kind, birth)
    signature = {"protocol": None if smoke else (protocol or register()), "kind": kind, "birth": birth}
    steps = 8 if smoke else STEPS
    curve = torch.empty((steps, len(METRICS)))
    routes = torch.empty((steps, BATCH, len(EVENTS)))
    start = 0
    if not smoke and target.is_file():
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError(f"Saved fit has a different training signature: {target}")
        start = saved["step"]
        if not 0 <= start <= steps:
            raise ValueError(f"Unexpected fit position: {target}")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        curve[:start] = saved["curve"]
        routes[:start] = saved["routes"]

    print(f"{kind}/{birth} continuing at {start}/{steps}", flush=True)
    began = perf_counter()
    for step in range(start + 1, steps + 1):
        curve[step - 1], routes[step - 1] = update(model, optimizer, kind, birth, step, BATCH)
        if step % CHECKPOINT_EVERY == 0 or step == steps:
            if not smoke:
                target.parent.mkdir(parents=True, exist_ok=True)
                pack = dict(signature=signature, step=step, state=model.state_dict(), optimizer=optimizer.state_dict(),
                            curve=curve[:step].clone(), routes=routes[:step].clone())
                storage._atomic(target, lambda temp: torch.save(pack, temp))
            measures = curve[max(0, step - 64):step, :6].mean(0).numpy().round(4).tolist()
            print(kind, birth, step, measures, f"{perf_counter() - began:.2f}s", flush=True)
    if smoke:
        verify_routes(kind, routes)
    return curve


def load_final(kind, birth):
    target = destination(kind, birth)
    saved = torch.load(target, map_location="cpu", weights_only=True)
    identity = {"protocol": register(), "kind": kind, "birth": birth}
    if saved["signature"] != identity or saved["step"] != STEPS:
        raise ValueError(f"Fit has an incomplete or unexpected signature: {target}")
    model = world.ForkGraph(birth)
    model.load_state_dict(saved["state"])
    return model.eval(), saved["curve"], digest(target)


def verify_routes(kind, routes):
    def column(name):
        return routes[:, :, EVENTS.index(name)]

    draw = column("route_draw")
    propensity = column("right_propensity")
    chosen = column("action_right")
    if not torch.equal(chosen.bool(), draw < propensity):
        raise ValueError("A logged action disagrees with its recorded draw and propensity")
    expected = (1.0 - EXPLORATION) * column("main_motor_logit").sigmoid() + EXPLORATION / 2.0
    torch.testing.assert_close(propensity, expected)
    realized = column("actual_delayed_return")
    if not (realized.abs() == 1).all():
        raise ValueError("Delayed payoff outside the declared support")
    selected_x = torch.where(chosen.bool(), column("door_right_x"), column("door_left_x"))
    selected_multiplier = torch.where(chosen.bool(), column("door_right_multiplier"), column("door_left_multiplier"))
    first_visited = selected_x == column("visit1_x")
    known_object = first_visited | (selected_x == column("visit2_x"))
    selected_history = torch.where(first_visited, column("visit1_actual_taste"), column("visit2_actual_taste"))
    if not known_object.all() or not torch.equal(realized, selected_history * selected_multiplier):
        raise ValueError("The realized return does not match the logged physical route")
    target = column("applied_teacher_target")
    if kind == "reward":
        if not target.isnan().all():
            raise ValueError("Reward-only fit logged an auxiliary target")
    else:
        expected = realized.roll(-1, dims=1) if kind == "sham" else realized
        if not torch.equal(target, expected):
            raise ValueError("A logged target came from a different route source")
    return int(chosen.numel())


def publish():
    identity = register()
    fitted = {}
    total_routes = 0
    for kind, birth in product(KINDS, BIRTHS):
        target = destination(kind, birth)
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved["signature"] != {"protocol": identity, "kind": kind, "birth": birth} or saved["step"] != STEPS:
            raise ValueError(f"Fit has an incomplete or unexpected signature: {target}")
        curve, routes = saved["curve"], saved["routes"]
        if curve.shape != (STEPS, len(METRICS)) or not curve.isfinite().all():
            raise ValueError(f"Incomplete or nonfinite curve: {target}")
        if routes.shape != (STEPS, BATCH, len(EVENTS)):
            raise ValueError(f"Incomplete route ledger: {target}")
        verified = verify_routes(kind, routes)
        total_routes += verified
        fitted[f"{kind}/{birth}"] = {
            "file": target.relative_to(ROOT).as_posix(), "sha256": digest(target),
            "steps": STEPS, "verified_routes": verified, "present_edges": int(saved["state"]["mask"].sum()),
            "first_64": dict(zip(METRICS, curve[:64].mean(0).tolist(), strict=True)),
            "last_64": dict(zip(METRICS, curve[-64:].mean(0).tolist(), strict=True)),
        }
    path = EVIDENCE / "fit-results.json"
    summary = {"protocol_sha256": identity, "verified_routes": total_routes, "fits": fitted}
    storage._write_json(path, summary)
    print(f"Verified {total_routes} training routes and wrote {path}", flush=True)


def run_all(jobs):
    register()
    log_dir = DATA / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def worker(entry):
        kind, birth = entry
        log = log_dir / f"{kind}-{birth}.txt"
        with log.open("a", encoding="utf8") as output:
            process = subprocess.run([sys.executable, "-m", __spec__.name, kind, str(birth)], cwd=ROOT,
                                     env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        print(f"{kind}/{birth} exit={process.returncode} {log}", flush=True)
        return process.returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        status = list(pool.map(worker, product(KINDS, BIRTHS)))
    if any(status):
        raise RuntimeError("At least one fit failed; see the fitted-condition logs")
    publish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("birth", nargs="?", type=int, choices=BIRTHS)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.register:
        print(register())
    elif args.publish:
        publish()
    elif args.jobs:
        run_all(args.jobs)
    elif args.kind is not None and args.birth is not None:
        fit(args.kind, args.birth, smoke=args.smoke)
    else:
        parser.error("Supply a kind and birth, or use --jobs, --register, or --publish")


if __name__ == "__main__":
    main()
