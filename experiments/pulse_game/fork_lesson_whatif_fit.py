"""Train self-issued replica probes against the few outcomes actually experienced."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from . import fork_lesson_fit as original
from . import fork_lesson_whatif as graph
from . import interleave_revisit_train as storage


ROOT = original.ROOT
DATA = original.DATA.parent / "whatif-fit"
PROTOCOL = original.EVIDENCE / "whatif-fit-protocol.json"
KINDS = ("binding", "misbound")
BIRTHS = original.BIRTHS
DOMAIN = 69274021
STEPS = 1280
BATCH = 64
LEARNING_RATE = 0.005
CHECKPOINT_EVERY = 256
TRACE = ("query_draw", "query_prob_right", "query", "driven_query", "action_draw", "action_prob_right", "action",
         "forecast", "actual_payoff", "forecast_weight", "clone_motor_left_input", "clone_motor_right_input")
METRICS = ("matched_fraction", "driven_match_fraction", "delayed_return", "weighted_forecast_mse", "policy_loss",
           "gradient_norm", "query_right_fraction", "action_right_fraction", "ema_baseline")


def specification():
    files = [__file__, graph.__file__, graph.world.__file__, original.__file__]
    return {
        "fixed_anchor": "Three complete whole checkpoints. All body parameters and persistent masks remain frozen.",
        "conditions": {
            "binding": "The replica motor pulse encodes the primary population's self-chosen trial side.",
            "misbound": "Each row receives the following independent row's self-chosen trial side, cyclically. "
                        "Each batch has exactly the same number of sent left and right pulses as its own requests.",
        },
        "trainable": "Two motor-axon input columns, feedback from the two replica motor neurons into primary neurons, "
                     "and primary-only query, forecast and action readouts.",
        "sources": {Path(file).name: original.digest(file) for file in files},
        "anchors": {str(birth): original.digest(original.destination("whole", birth)) for birth in BIRTHS},
        "training_world_seed": [DOMAIN, "birth", "optimizer_step"],
        "steps": STEPS, "batch": BATCH, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "query_exploration": graph.QUERY_EXPLORATION, "action_exploration": graph.ACTION_EXPLORATION,
        "forecast_criterion": "Squared error to the actual signed delayed outcome only when the primary's queried and "
                              "actually executed sides coincide. Stabilized inverse action propensity weighting.",
        "policy_gradient": "Both query and action likelihoods are weighted by the actual delayed outcome minus a "
                           "baseline from previous batches.",
        "forecast_tick": graph.FORECAST_TICK, "additional_ticks_after_forecast": graph.AFTER_FORECAST_TICKS,
        "trace_columns": TRACE, "curve_columns": METRICS,
    }


def register():
    content = json.dumps(specification(), indent=2) + "\n"
    if PROTOCOL.is_file():
        if PROTOCOL.read_text(encoding="utf8") != content:
            raise ValueError(f"Self-issued trial protocol differs from configuration: {PROTOCOL}")
    else:
        storage._atomic(PROTOCOL, lambda path: path.write_text(content, encoding="utf8"))
    return original.digest(PROTOCOL)


def destination(kind, birth):
    return DATA / kind / f"seed-{birth}.pt"


def make_graph(birth):
    torch.manual_seed(birth)
    body, _, _ = original.load_final("whole", birth)
    return graph.WhatIfGraph(body)


def update(model, optimizer, kind, birth, step, baseline):
    episode = graph.world.scenes([DOMAIN, birth, step], BATCH)
    rng = np.random.default_rng([DOMAIN, birth, step, 253])
    query_draw, action_draw = torch.from_numpy(rng.random((2, BATCH), dtype=np.float32))
    result = graph.rollout(model, episode, query_draw, action_draw, shuffle_trial=kind == "misbound")
    actual = result.actual_payoff
    policy = -((result.query_score + result.action_score) * (actual - baseline)).mean()
    with torch.no_grad():
        matched = result.query == result.action
        selected_propensity = torch.where(result.action.bool(), result.action_right, 1 - result.action_right)
        weight = matched / selected_propensity
    forecast = ((result.forecast - actual).square() * weight).sum() / weight.sum().clamp_min(1.0)
    loss = policy + forecast
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        baseline = 0.98 * baseline + 0.02 * actual.mean().item()
        metrics = torch.stack((matched.float().mean(), (result.query == result.driven_query).float().mean(),
                               actual.mean(), forecast, policy, gradient, result.query.float().mean(),
                               result.action.float().mean(), actual.new_tensor(baseline)))
        trace = torch.column_stack((query_draw, result.query_right, result.query, result.driven_query,
                                     action_draw, result.action_right, result.action, result.forecast,
                                     actual, weight, result.trial.pulse[:, graph.MOTOR_PORTS]))
    return metrics, trace, baseline


def audit(kind, trace):
    def column(key):
        return trace[..., TRACE.index(key)]

    query = column("query")
    motor = column("action")
    driven = query if kind == "binding" else query.roll(-1, dims=1)
    if not torch.equal(column("driven_query"), driven):
        raise ValueError("Recorded replica inputs do not follow their arm's side assignment")
    for decision, draw, probability in ((query, "query_draw", "query_prob_right"),
                                        (motor, "action_draw", "action_prob_right")):
        if not torch.equal(decision.bool(), column(draw) < column(probability)):
            raise ValueError("Decision did not follow the recorded policy and independent random draw")
    probability = column("action_prob_right")
    propensity = torch.where(motor.bool(), probability, 1 - probability)
    if not torch.equal(column("forecast_weight"), (query == motor) / propensity):
        raise ValueError("The forecast update included a route other than the actual trial-matched one")
    if not trace.isfinite().all() or not (column("actual_payoff").abs() == 1).all():
        raise ValueError("Found a nonfinite trace or wrong payoff scale")
    return int((query == motor).sum())


def fit(kind, birth, smoke=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    signature = {"protocol": None if smoke else register(), "birth": birth, "kind": kind}
    model = make_graph(birth)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    target = destination(kind, birth)
    count = 8 if smoke else STEPS
    curve = torch.empty((count, len(METRICS)))
    history = torch.empty((count, BATCH, len(TRACE)))
    step_start = 0
    baseline = 0.0
    if not smoke and target.is_file():
        saved = torch.load(target, weights_only=True, map_location="cpu")
        if saved["signature"] != signature:
            raise ValueError(f"Different fit identity in {target}")
        step_start = saved["step"]
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        curve[:step_start], history[:step_start], baseline = saved["curve"], saved["trace"], saved["baseline"]
    tic = perf_counter()
    for step in range(step_start + 1, count + 1):
        curve[step - 1], history[step - 1], baseline = update(model, optimizer, kind, birth, step, baseline)
        if step % CHECKPOINT_EVERY == 0 or step == count:
            if not smoke:
                state = {"signature": signature, "step": step, "state": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "baseline": baseline,
                         "curve": curve[:step].clone(), "trace": history[:step].clone()}
                storage._atomic(target, lambda path: torch.save(state, path))
            print(kind, birth, step, curve[max(0, step - 64):step, :6].mean(0).tolist(),
                  f"{perf_counter() - tic:.2f}s", flush=True)
    if smoke:
        audit(kind, history)


def load_final(kind, birth):
    saved = torch.load(destination(kind, birth), weights_only=True, map_location="cpu")
    if saved["signature"] != {"protocol": register(), "birth": birth, "kind": kind} or saved["step"] != STEPS:
        raise ValueError("Unfinished or inconsistent self-issued trial checkpoint")
    model = make_graph(birth)
    model.load_state_dict(saved["state"])
    return model.eval(), saved["curve"], original.digest(destination(kind, birth))


def publish():
    results = {}
    for kind, birth in product(KINDS, BIRTHS):
        model, curve, digest = load_final(kind, birth)
        anchor, _, _ = original.load_final("whole", birth)
        if any(not torch.equal(value, anchor.state_dict()[name]) for name, value in model.body.state_dict().items()):
            raise ValueError(f"Frozen 40-node body changed during {kind}/{birth}")
        if any(parameter.requires_grad for parameter in model.body.parameters()) or not curve.isfinite().all():
            raise ValueError(f"Invalid fit or unfrozen body at {kind}/{birth}")
        target = destination(kind, birth)
        saved = torch.load(target, weights_only=True, map_location="cpu")
        matching = audit(kind, saved["trace"])
        results[f"{kind}/{birth}"] = {
            "checkpoint": target.relative_to(ROOT).as_posix(), "sha256": digest,
            "routes": STEPS * BATCH, "matched_real_routes": matching,
            "body_equal_to_anchor_bitwise": True,
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "first64": dict(zip(METRICS, curve[:64].mean(0).tolist(), strict=True)),
            "last64": dict(zip(METRICS, curve[-64:].mean(0).tolist(), strict=True)),
        }
    output = original.EVIDENCE / "whatif-fit-results.json"
    storage._write_json(output, {"protocol": register(), "fits": results})
    print(output, flush=True)


def run_all(jobs):
    register()
    logs = DATA / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_pair(pair):
        kind, birth = pair
        log = logs / f"{kind}-{birth}.txt"
        with log.open("a", encoding="utf8") as stream:
            process = subprocess.run((sys.executable, "-m", __spec__.name, kind, str(birth)), cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        print(kind, birth, process.returncode, log, flush=True)
        return process.returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        status = list(pool.map(run_pair, product(KINDS, BIRTHS)))
    if any(status):
        raise RuntimeError("At least one trial fit failed. See the per-fit logs.")
    publish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("birth", nargs="?", type=int, choices=BIRTHS)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish()
    elif args.jobs:
        run_all(args.jobs)
    elif args.kind is not None and args.birth is not None:
        fit(args.kind, args.birth, smoke=args.smoke)
    else:
        parser.error("Supply a condition and birth, or use --jobs or --publish")


if __name__ == "__main__":
    main()
