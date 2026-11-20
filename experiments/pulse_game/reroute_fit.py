"""Learn a room and an outcome from traversed routes in a 40-node body."""

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

import numpy as np
import torch

from . import edge_site_train_fit as parents
from . import fork_lesson_bit_relay, fork_lesson_native_request, fork_lesson_whatif, fork_lesson_world
from . import interleave_revisit_train as storage
from . import native_rehearsal_graph, native_rehearsal_trial
from . import reroute_world as world


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/reroute-20261120"
EVIDENCE = ROOT / "docs/evidence/reroute-20261120"
PROTOCOL = EVIDENCE / "fit-protocol.json"
BIRTHS = (731, 732, 733, 734, 735, 736)
KINDS = ("terminal", "next_room", "misbound")
DOMAIN = 18692341
SMOKE_DOMAIN = 18692863
EASY_STEPS = 256
STEPS = 1792
BATCH = 64
CHECKPOINT_EVERY = 128
QUERY_EXPLORATION = .45
ACTION_EXPLORATION = .30
LR = .003
CLIP = 2.
MID_WEIGHT = 1.
TRACE = (
    "query_draw", "query_right_probability", "query", "action_draw", "action_right_probability", "action",
    "actual_room", "actual_endpoint", "paid", "forecast_room", "forecast_pay", "actual_route_weight",
    "room_teacher_source", "room_teacher_target", "query_logit", "action_logit",
)
SHARED = ("query_draw", "action_draw")
METRICS = (
    "paid", "query_action_match", "weighted_terminal_mse", "weighted_real_room_mse",
    "weighted_room_teacher_mse", "weighted_real_room_acc", "policy_loss", "gradient_norm",
    "query_right", "action_right", "used_room_target_count",
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def specification(backend):
    sources = (
        __file__, world.__file__, fork_lesson_native_request.__file__, fork_lesson_whatif.__file__,
        fork_lesson_world.__file__, fork_lesson_bit_relay.__file__, native_rehearsal_graph.__file__,
        native_rehearsal_trial.__file__, parents.gate.net.__file__, parents.gate.storage.__file__,
    )
    source_paths = sorted({Path(source).resolve() for source in sources})
    return {
        "plan_sha256": digest(EVIDENCE / "fit-plan.md"), "births": BIRTHS, "kinds": KINDS,
        "steps": STEPS, "easy_steps": EASY_STEPS, "batch": BATCH, "checkpoint_every": CHECKPOINT_EVERY,
        "world_seed": [DOMAIN, "birth", "step", 0],
        "draw_seed": [DOMAIN, "birth", "step", 1], "smoke_domain": SMOKE_DOMAIN,
        "query_exploration": QUERY_EXPLORATION, "action_exploration": ACTION_EXPLORATION,
        "optimizer": {"name": "Adam", "learning_rate": LR, "max_gradient_norm": CLIP, "foreach": False,
                      "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0.},
        "objective": {
            "actor": "Both native draws are self-selected. Both mixture log-likelihoods get actual return minus "
                     "the leave-one-other-world-out batch baseline.",
            "terminal": "The native terminal readout forecasts the signed payout of the tried door. Apply "
                        "stabilized inverse action propensity weighting only on rows which took the tried door.",
            "next_room": "Also teach the preceding native room readout the realized room ID as -1 or +1, using "
                         "the same weights and the same eligible rows as the terminal loss.",
            "misbound": "Same extra room computation and eligibility. Apply a cyclic derangement of the same "
                        "eligible rows' actual room targets. Used targets have identical marginal counts "
                        "to the eligible routes' actual room targets within each batch.",
            "terminal_coefficient": 1., "room_coefficient": MID_WEIGHT,
            "loss_scope": "Full BPTT through all physical prior visits and internal discussion. Read targets "
                          "only after actual settlement, outside the differentiation graph.",
        },
        "paired_experience": "Identical birth, world tape and both draw columns. Actual doors may diverge as "
                             "the bodies' native motor policies change.",
        "backend": backend, "dtype": "torch.float32", "torch": str(torch.__version__),
        "deterministic": True, "cuda_tf32": False, "torch_threads": 1,
        "curve_columns": METRICS, "route_columns": TRACE,
        "source_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
        "parent_sha256": {str(birth): digest(parents.parent_paths()[str(birth)]) for birth in BIRTHS},
    }


def register(backend=None):
    if PROTOCOL.is_file():
        content = PROTOCOL.read_text(encoding="utf8")
        if backend is None:
            backend = json.loads(content)["backend"]
    else:
        content = None
    if backend is None:
        raise ValueError("Choose the fit backend when registering the protocol")
    encoded = json.dumps(specification(backend), indent=2, ensure_ascii=False) + "\n"
    if content is None:
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
        storage._atomic(PROTOCOL, lambda temporary: temporary.write_text(encoded, encoding="utf8"))
    elif content != encoded:
        raise ValueError(f"Existing registration differs from fit settings: {PROTOCOL}")
    return digest(PROTOCOL)


def configure(backend):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if backend == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.fp32_precision = "ieee"


def episode(birth, step, backend="cpu", *, smoke=False):
    domain = SMOKE_DOMAIN if smoke else DOMAIN
    easy = step <= EASY_STEPS
    scene = world.scenes([domain, birth, step, 0], BATCH, device=backend, easy=easy)
    tape = np.random.default_rng([domain, birth, step, 1]).random((2, BATCH), dtype=np.float32)
    draws = torch.from_numpy(tape).to(backend)
    return scene, *draws


def teacher(kind, query, action, actual_room):
    active = (query == action).nonzero().flatten()
    source = torch.full_like(actual_room, -1)
    assigned = torch.zeros_like(actual_room)
    if kind == "terminal" or len(active) == 0:
        return source, assigned
    # An unpaired row cannot receive a deranged target. This affects both room arms.
    if len(active) == 1:
        return source, assigned
    source[active] = active if kind == "next_room" else active.roll(-1)
    assigned[active] = 2 * actual_room[source[active]] - 1
    return source, assigned


def squared(predicted, target, weight):
    return ((predicted - target).square() * weight).sum() / weight.sum().clamp_min(1.)


def witness(loss, shared):
    tensors = (shared.h, shared.fast, shared.trace, shared.candidate)
    gradient = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    norms = [None if grad is None else float(grad.norm()) for grad in gradient]
    if any(value is None or not np.isfinite(value) for value in norms) or sum(norms) <= 0.:
        raise AssertionError(f"Teaching gradient did not reach the completed past visits: {norms}")
    return norms


def update(model, optimizer, kind, scene, query_draw, action_draw, *, probe_gradients=False):
    shared = world.history(model, scene)
    ready = world.prepare(model, scene, shared=shared)
    query = world.select(model, ready, query_draw, QUERY_EXPLORATION)
    trial = world.try_side(model, ready, query.side)
    action = world.select(model, trial.final, action_draw, ACTION_EXPLORATION)
    with torch.no_grad():
        outcome = world.settle(model, trial.final, scene, action)
        real_mid = (2 * outcome.room - 1).to(trial.mid)
        eligible = query.side == action.side
        propensity = torch.where(query.side.bool(), action.probability, 1 - action.probability)
        weight = eligible / propensity
        source, assigned = teacher(kind, query.side, action.side, outcome.room)
        advantage = (outcome.paid - outcome.paid.mean()) * BATCH / (BATCH - 1)
    policy = -((query.loglikelihood + action.loglikelihood) * advantage).mean()
    terminal = squared(trial.forecast, outcome.paid, weight)
    room_teacher = squared(trial.mid, assigned, weight * source.ge(0))
    objective = policy + terminal + MID_WEIGHT * room_teacher
    evidence = None
    if probe_gradients:
        evidence = {"terminal": witness(terminal, shared)}
        if kind != "terminal":
            evidence["room"] = witness(room_teacher, shared)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    with torch.no_grad():
        room_acc = ((trial.mid.ge(0) == outcome.room.bool()).to(weight) * weight).sum() / weight.sum().clamp_min(1.)
        curve = torch.stack((outcome.paid.mean(), eligible.float().mean(), terminal, squared(trial.mid, real_mid, weight),
                             room_teacher, room_acc, policy, norm, query.side.float().mean(),
                             action.side.float().mean(), source.ge(0).sum()))
        trace = torch.column_stack((
            query_draw, query.probability, query.side, action_draw, action.probability, action.side,
            outcome.room, outcome.endpoint, outcome.paid, trial.mid, trial.forecast, weight, source, assigned,
            query.logit, action.logit,
        ))
    return curve.cpu(), trace.cpu(), evidence


def destination(kind, birth):
    return DATA / f"fit-{kind}" / f"birth-{birth}.pt"


def state_digest(state):
    result = sha256()
    for name, tensor in state.items():
        result.update(f"{name} {tensor.dtype} {tuple(tensor.shape)}\n".encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def make_graph(birth, backend="cpu"):
    torch.manual_seed(birth)
    return world.RerouteGraph(birth).to(device=backend, dtype=torch.float32)


def fit(kind, birth, backend, *, smoke=False, smoke_steps=16):
    configure(backend)
    signature = dict(protocol=None if smoke else register(backend), kind=kind, birth=birth)
    model = make_graph(birth, backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False, fused=False)
    beginning = state_digest(model.state_dict())
    target = destination(kind, birth)
    total = smoke_steps if smoke else STEPS
    curve = torch.empty((total, len(METRICS)))
    ledger = torch.empty((total, BATCH, len(TRACE)))
    gradients = {}
    start = 0
    elapsed = 0.
    if target.is_file() and not smoke:
        saved = torch.load(target, weights_only=True, map_location="cpu")
        if saved["signature"] != signature or saved["birth_sha256"] != beginning:
            raise ValueError(f"Wrong training identity: {target}")
        start = saved["step"]
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        curve[:start], ledger[:start] = saved["curve"], saved["ledger"]
        gradients, elapsed = saved["through_history_gradient_norms"], saved["seconds"]

    clock = perf_counter()
    for step in range(start + 1, total + 1):
        scene_step = step if not smoke or step % 2 else EASY_STEPS + step
        curve[step - 1], ledger[step - 1], evidence = update(
            model, optimizer, kind, *episode(birth, scene_step, backend, smoke=smoke),
            probe_gradients=step == 1 or step == EASY_STEPS + 1 or (smoke and step == 2),
        )
        if evidence is not None:
            gradients[str(scene_step)] = evidence
        if step % CHECKPOINT_EVERY == 0 or step == total:
            seconds = elapsed + perf_counter() - clock
            if not smoke:
                payload = dict(signature=signature, birth_sha256=beginning, step=step, state=model.state_dict(),
                               optimizer=optimizer.state_dict(), curve=curve[:step].clone(),
                               ledger=ledger[:step].clone(), through_history_gradient_norms=gradients,
                               seconds=seconds)
                storage._atomic(target, lambda temporary: torch.save(payload, temporary))
            recent = dict(zip(METRICS, curve[max(0, step - 64):step].mean(0).tolist(), strict=True))
            print(kind, birth, step, {key: round(recent[key], 5) for key in METRICS[:8]},
                  round(seconds, 2), "s", flush=True)
    if smoke:
        print("through-history gradient L2 by state (activity, fast, eligibility, proposal):", gradients, flush=True)


def load_final(kind, birth):
    target = destination(kind, birth)
    saved = torch.load(target, weights_only=True, map_location="cpu")
    signature = dict(protocol=register(), kind=kind, birth=birth)
    if saved["signature"] != signature or saved["step"] != STEPS:
        raise ValueError(f"Unfinished or incompatible reproduction: {target}")
    model = make_graph(birth)
    if saved["birth_sha256"] != state_digest(model.state_dict()):
        raise ValueError(f"Invalid birth digest: {target}")
    model.load_state_dict(saved["state"])
    return model.requires_grad_(False).eval(), saved, digest(target)


def run_all(backend, jobs):
    register(backend)
    logs = DATA / "fit-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(pair):
        kind, birth = pair
        log = logs / f"{kind}-{birth}.txt"
        with log.open("a", encoding="utf8") as stream:
            process = subprocess.run(
                (sys.executable, "-m", __spec__.name, kind, str(birth), "--backend", backend),
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
        print(kind, birth, process.returncode, log, flush=True)
        return process.returncode

    # Interleave all three arms at each birth so early completion can be compared.
    queue = [(kind, birth) for birth, kind in product(BIRTHS, KINDS)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        status = list(pool.map(run, queue))
    if any(status):
        raise RuntimeError("At least one room fit failed. See its per-fit log.")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("birth", nargs="?", type=int, choices=BIRTHS)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=16)
    args = parser.parse_args()
    if args.register:
        print(register(args.backend))
    elif args.jobs:
        run_all(args.backend, args.jobs)
    elif args.kind and args.birth:
        fit(args.kind, args.birth, args.backend, smoke=args.smoke, smoke_steps=args.smoke_steps)
    else:
        parser.error("Provide kind and birth, or use --register or --jobs")


if __name__ == "__main__":
    main()
