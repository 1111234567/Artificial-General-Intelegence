"""Fit the draft-rehearsal selector from sampled routes with matched experience."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import draft_rehearse as draft
from . import edge_site_train as sites
from . import interleave_interference as environment
from . import interleave_revisit_train as storage
from . import noon_ticket as parents


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/draft-rehearsal-20261120"
EVIDENCE = ROOT / "docs/evidence/draft-rehearsal-20261120"
PROTOCOL = EVIDENCE / "fit-protocol.json"
BIRTHS = tuple(parents.births("original"))
ARMS = ("pre", "post", "wrong")
SEEDS = (21294703, 21294711)
DOMAIN = 73111943
ANGLES = (15, 30, 45, 60, 75)
DESIGN = np.array([(angle, a, b, first) for angle in ANGLES for a in (-1, 1)
                   for b in (-1, 1) for first in (0, 1)], dtype=np.int16)
WORLDS = 2 * len(DESIGN)
CHECKPOINTS = (0, 128, 256, 512, 768)
STEPS = CHECKPOINTS[-1]
SEGMENT = 64
REVISIT_WAIT = 4
LR = 0.012
DECAY = 0.00025
COEFFICIENT_BOUND = 4
# The two sign blocks are padded to their respective batch-wide maxima.
MAX_GROUPS = 2 * (sites.net.NODES ** 2 // sites.GROUP_SIZE)
EVENTS = ("c_side", "c_taste", "a_side", "a_taste", "b_side", "b_taste",
          "rehearsed_first_receipt_side", "rehearsed_first_receipt_taste",
          "rehearsed_b_side", "rehearsed_b_taste", "first_real_revisit_object",
          "revisit_a_side", "revisit_b_side", "revisit_a_payout", "revisit_b_payout",
          "rehearsed_retained_before_b")
METRICS = ("correct_a", "correct_b", "correct_both", "reward", "groups_per_route",
           "mean_l1_deferred", "maximum_signed_dose_error", "mean_group_log_prob",
           "gradient_l2", "coefficient_l2")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def protocol():
    sources = (__file__, draft.__file__, sites.__file__, sites.net.__file__,
               sites.net.native.__file__, sites.storage.__file__, parents.__file__,
               environment.__file__, storage.__file__)
    bodies = {str(birth): sites.storage.checkpoint_path(parents.source("original"), "odor", "uniform",
                                                       birth, 8000) for birth in BIRTHS}
    return {
        "plan_sha256": digest(EVIDENCE / "fit-plan.md"),
        "training_births": BIRTHS, "heldout_births": parents.births("independent"),
        "fit_seeds": SEEDS, "arms": ARMS, "feature_names": draft.FEATURES,
        "geometry_domain": DOMAIN, "steps": STEPS, "checkpoints": CHECKPOINTS,
        "worlds_per_birth_per_update": WORLDS, "angles_degrees": ANGLES,
        "true_revisit_order": "exactly crossed within every birth and update",
        "revisit_wait_ticks": REVISIT_WAIT, "objective": {"A": 1, "B": 1, "both": 1},
        "routed_training": "root-major; shared tapes across arms, disjoint substreams for geometry and "
                           "actual support, true revisits, trial/rehearsal, and address sampling",
        "optimization": {"baseline": "leave-one-out mean within birth", "optimizer": "Adam",
                         "learning_rate": LR, "l2_gradient": DECAY,
                         "coefficient_bound": COEFFICIENT_BOUND, "initial_coefficients": 0,
                         "torch_threads": 1, "torch_deterministic": True},
        "event_columns": EVENTS, "event_shape": "step × root-major route × field",
        "address_log": "Flattened receiver * 20 + sender in candidate group order; -1 means invalid or padding.",
        "source_sha256": {str(Path(source).relative_to(ROOT).as_posix()): digest(Path(source))
                          for source in sources},
        "parent_sha256": {birth: digest(path) for birth, path in bodies.items()},
    }


def register():
    spec = protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text(encoding="utf8")) != json.loads(json.dumps(spec)):
            raise ValueError("Registered fit protocol differs from the current source or settings")
    else:
        storage._write_json(PROTOCOL, spec)
    return digest(PROTOCOL)


def training_world(seed, step, worlds):
    n = len(BIRTHS) * worlds
    geometry = np.random.default_rng([DOMAIN, seed, step, 0])
    support = np.random.default_rng([DOMAIN, seed, step, 1])
    revisit = np.random.default_rng([DOMAIN, seed, step, 2])
    address = np.random.default_rng([DOMAIN, seed, step, 3])
    indices = np.resize(np.arange(len(DESIGN)), worlds)
    design = DESIGN[np.concatenate([geometry.permutation(indices) for _ in BIRTHS])]
    angles, law_a, law_b, first = design.T
    a, tangent = geometry.standard_normal((2, n, 3), dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    tangent -= np.einsum("ij,ij->i", a, tangent)[:, None] * a
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    theta = np.deg2rad(angles).astype(np.float32)[:, None]
    b = a * np.cos(theta) + tangent * np.sin(theta)
    orders = np.ones((n, 5), dtype=np.float32)
    orders[:, :2] = 2 * support.integers(0, 2, size=(n, 2)) - 1
    orders[:, 3:] = 2 * revisit.integers(0, 2, size=(n, 2)) - 1
    draws = np.zeros((n, 5, 5), dtype=np.float32)
    draws[:, :2] = support.random((n, 2, 5), dtype=np.float32)
    draws[:, 3:, 0] = revisit.random((n, 2), dtype=np.float32)
    tensor = torch.from_numpy
    world = environment.World(tensor(1.35 * a), tensor(1.35 * b),
                              tensor(law_a.astype(np.float32)), tensor(law_b.astype(np.float32)),
                              tensor(orders), tensor(draws), np.empty((n, 0)))
    choice = tensor(address.random((n, MAX_GROUPS), dtype=np.float32))
    return world, tensor(first.astype(np.int64)), choice


def contact_log(*contacts):
    return torch.stack([value for contact in contacts for value in
                        (contact.action.side, contact.experience.taste)], dim=1).to(torch.int8)


def verify_dose(prefix, state):
    native = prefix.native_b.state.graph
    for field in ("bias", "present"):
        if not torch.equal(getattr(native, field), getattr(state.graph, field)):
            raise AssertionError(f"Selected graph changed the native {field}")
    for field in ("hidden", "sender_trace", "receiver_trace"):
        if not torch.equal(getattr(prefix.live, field), getattr(state, field)):
            raise AssertionError(f"Selected graph changed live {field} before the real revisit")
    delta = (native.weights - state.graph.weights).flatten(1)
    group = prefix.candidates
    if not torch.equal(delta.ne(0).sum(1), group.valid.sum(1)):
        raise AssertionError("Selected graph did not withhold exactly one address per valid group")
    positive = (delta.clamp_min(0).sum(1) - group.amount.clamp_min(0).sum(1)).abs().max()
    negative = (delta.clamp_max(0).sum(1) - group.amount.clamp_max(0).sum(1)).abs().max()
    error = torch.maximum(positive, negative)
    if error > 3e-6:
        raise AssertionError(f"Selected graph did not match the prescribed signed write budgets: {error}")
    return error


@torch.no_grad()
def update_arm(prefix, world, first, draw, shared_log, weights, optimizer, worlds, arm):
    rehearsal = draft.rehearse(prefix, mode=arm)
    group = prefix.candidates
    draw = draw[:, :group.valid.shape[1]]
    selected = draft.choose(prefix, rehearsal, weights, draw=draw, gradient=True)
    live = draft.apply(prefix, selected)
    dose_error = verify_dose(prefix, live)
    result = draft.probe(prefix.rule, live, world, wait=REVISIT_WAIT, order=first)
    a, b = result.correct.float().unbind(1)
    both = a * b
    reward = a + b + both
    by_birth = torch.stack((a, b, both, reward), dim=-1).reshape(len(BIRTHS), worlds, 4)
    payout = by_birth[..., 3]
    centered = ((payout - payout.mean(1, keepdim=True)) * worlds / (worlds - 1)).flatten()
    optimizer.zero_grad(set_to_none=True)
    weights.grad = -(centered[:, None] * selected.score_gradient).mean(0) + DECAY * weights
    if not torch.isfinite(weights.grad).all():
        raise FloatingPointError(f"Nonfinite {arm} selector gradient")
    optimizer.step()
    weights.clamp_(-COEFFICIENT_BOUND, COEFFICIENT_BOUND)
    outcomes = by_birth.mean(1)
    statistics = torch.cat((outcomes.mean(0), torch.stack((
        group.valid.sum(1).float().mean(), group.amount.abs().sum(1).mean(), dose_error,
        selected.log_prob.sum() / group.valid.sum().clamp_min(1), weights.grad.norm(), weights.norm(),
    ))))

    replay_log = contact_log(rehearsal.retained, rehearsal.current)
    actions = torch.cat((shared_log, replay_log, first[:, None].to(torch.int8),
                         result.side.to(torch.int8), result.payout.to(torch.int8),
                         prefix.retained_first[:, None].to(torch.int8)), dim=1).numpy()
    addresses = np.full((len(actions), MAX_GROUPS), -1, dtype="<i2")
    chosen = group.address.gather(2, selected.index[..., None]).squeeze(-1)
    addresses[:, :chosen.shape[1]] = torch.where(group.valid, chosen, -1).numpy()
    stamp = sha256(actions.tobytes() + addresses.tobytes()).hexdigest()
    return statistics.tolist(), outcomes.tolist(), stamp, actions, addresses


@torch.no_grad()
def prepare_step(g, rule, seed, step, worlds):
    world, first, draw = training_world(seed, step, worlds)
    prefix = draft.prepare(g, rule, world, seed=[DOMAIN, seed, step, 4])
    shared_log = contact_log(prefix.study_c, prefix.study_a, prefix.study_b)
    return prefix, world, first, draw, shared_log


def folder(replica):
    return DATA / f"fit-replica-{replica}"


def fit(replica, *, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed = SEEDS[replica]
    g, rule = parents.inherited("original")
    destination = folder(replica)
    steps, worlds = (5, 40) if smoke else (STEPS, WORLDS)
    identity = None if smoke else register()
    weights = {arm: torch.nn.Parameter(torch.zeros(len(draft.FEATURES))) for arm in ARMS}
    optimizers = {arm: torch.optim.Adam([weights[arm]], lr=LR) for arm in ARMS}
    ledger = {arm: [] for arm in ARMS}
    by_birth = {arm: [] for arm in ARMS}
    stamps = {arm: [] for arm in ARMS}
    events = {arm: [] for arm in ARMS}
    addresses = {arm: [] for arm in ARMS}
    timings = {name: [] for name in ("prefix", *ARMS)}
    start = 0
    latest = destination / "resume.pt"
    if not smoke and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if (saved["protocol"], saved["replica"]) != (identity, replica):
            raise ValueError(f"Resume has a different identity: {latest}")
        start = saved["step"]
        ledger, by_birth, stamps = saved["ledger"], saved["by_birth"], saved["stamps"]
        timings = saved["timings"]
        for arm in ARMS:
            with torch.no_grad():
                weights[arm].copy_(saved["weights"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])

    def save(step):
        if step and events[ARMS[0]]:
            storage._write_raw(destination / f"routes-through-{step:04d}.npz", arms=ARMS,
                               step=np.arange(step - len(events[ARMS[0]]) + 1, step + 1),
                               births=BIRTHS, seed=seed, columns=EVENTS, protocol=identity,
                               actions=np.asarray(list(events.values()), dtype=np.int8),
                               addresses=np.asarray(list(addresses.values()), dtype="<i2"))
            for arm in ARMS:
                events[arm].clear()
                addresses[arm].clear()
        if step in CHECKPOINTS:
            storage._write_raw(destination / f"step-{step:04d}.npz", step=step, seed=seed, replica=replica,
                               **{arm: weights[arm].detach().numpy() for arm in ARMS},
                               feature_names=draft.FEATURES, protocol=identity)
        if step == STEPS:
            storage._write_raw(destination / "ledger.npz", step=np.arange(1, step + 1), arms=ARMS,
                               columns=METRICS, metrics=np.asarray(list(ledger.values())),
                               outcomes_by_birth=np.asarray(list(by_birth.values())),
                               event_sha256=np.asarray(list(stamps.values())), timings_columns=list(timings),
                               step_seconds=np.asarray(list(timings.values())),
                               routes_per_step=len(BIRTHS) * worlds, births=BIRTHS, seed=seed, protocol=identity)
        storage._atomic(latest, lambda file: torch.save(dict(
            protocol=identity, replica=replica, step=step, ledger=ledger, by_birth=by_birth,
            stamps=stamps, timings=timings, weights={arm: w.detach() for arm, w in weights.items()},
            optimizers={arm: opt.state_dict() for arm, opt in optimizers.items()}), file))

    if not smoke and start in (0, steps):
        save(start)
    started = perf_counter()
    for step in range(start + 1, steps + 1):
        before = perf_counter()
        batch = prepare_step(g, rule, seed, step, worlds)
        timings["prefix"].append(perf_counter() - before)
        cycle = (step + replica - 1) % len(ARMS)
        for arm in ARMS[cycle:] + ARMS[:cycle]:
            before = perf_counter()
            row, birth_row, stamp, action_log, address_log = update_arm(
                *batch, weights[arm], optimizers[arm], worlds, arm)
            timings[arm].append(perf_counter() - before)
            ledger[arm].append(row)
            by_birth[arm].append(birth_row)
            stamps[arm].append(stamp)
            events[arm].append(action_log)
            addresses[arm].append(address_log)
        if not smoke and step % SEGMENT == 0:
            save(step)
        if step % SEGMENT == 0 or smoke:
            report = {arm: np.mean(ledger[arm][-SEGMENT:], axis=0)[:4].round(4).tolist() for arm in ARMS}
            print(replica, step, report, f"{perf_counter() - started:.2f}s", flush=True)


def publish():
    identity = register()
    heads = {arm: [] for arm in ARMS}
    summary = {"protocol_sha256": identity, "replicas": []}
    for replica, seed in enumerate(SEEDS):
        dest = folder(replica)
        checkpoints = [dest / f"step-{step:04d}.npz" for step in CHECKPOINTS]
        routes = [dest / f"routes-through-{step:04d}.npz" for step in range(SEGMENT, STEPS + 1, SEGMENT)]
        ledger = dest / "ledger.npz"
        info = {"replica": replica, "seed": seed, "fits": {}, "files": {
            path.relative_to(ROOT).as_posix(): digest(path) for path in (*checkpoints, *routes, ledger)}}
        with np.load(checkpoints[-1]) as final, np.load(ledger) as curve:
            if str(final["protocol"]) != identity or str(curve["protocol"]) != identity:
                raise ValueError("Final fit files belong to a different training protocol")
            if int(final["step"]) != STEPS or not np.array_equal(curve["step"], np.arange(1, STEPS + 1)):
                raise ValueError("Unfinished selector fit")
            metrics = curve["metrics"]
            for index, arm in enumerate(ARMS):
                coefficients = final[arm]
                if coefficients.shape != (len(draft.FEATURES),) or not np.isfinite(coefficients).all():
                    raise ValueError(f"Invalid coefficient row for {arm}")
                heads[arm].append(coefficients)
                info["fits"][arm] = {
                    "routes": int(curve["routes_per_step"]) * STEPS,
                    "first_64": dict(zip(METRICS, metrics[index, :SEGMENT].mean(0).tolist(), strict=True)),
                    "last_64": dict(zip(METRICS, metrics[index, -SEGMENT:].mean(0).tolist(), strict=True)),
                    "max_signed_dose_error": float(metrics[index, :, 6].max()),
                    "arm_compute_seconds": float(curve["step_seconds"][index + 1].sum()),
                    "shared_prefix_compute_seconds": float(curve["step_seconds"][0].sum()),
                }
        summary["replicas"].append(info)
    artifact = DATA / "fit-heads.npz"
    storage._write_raw(artifact, **{arm: np.stack(rows) for arm, rows in heads.items()},
                       feature_names=draft.FEATURES, replica_seeds=SEEDS, final_step=STEPS,
                       protocol_sha256=identity)
    summary["heads"] = artifact.relative_to(ROOT).as_posix()
    summary["heads_sha256"] = digest(artifact)
    storage._write_json(EVIDENCE / "fit-summary.json", summary)
    print(artifact, summary["heads_sha256"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "fit", "publish", "smoke"))
    parser.add_argument("--replica", type=int, choices=range(len(SEEDS)), default=0)
    args = parser.parse_args()
    if args.phase == "register":
        print(register())
    elif args.phase == "publish":
        publish()
    else:
        fit(args.replica, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()
