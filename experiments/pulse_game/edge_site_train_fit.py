"""Fit shared edge selectors from single-route stochastic experience."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import edge_site_train as gate
from . import interleave_revisit_train as storage
from . import noon_ticket


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "edge-site-20261120"
EVIDENCE = ROOT / "docs" / "evidence" / "edge-site-20261120"
BIRTHS = (731, 732, 733, 734, 735, 736)
CONFIRM = (811, 812, 813, 814, 815, 816)
TRAIN_DOMAIN = 73110827
SEEDS = (21294563, 21294601)
ARMS = ("full", "no_old")
CHECKPOINTS = (0, 256, 512, 1024)
STEPS = CHECKPOINTS[-1]
WORLDS = 128
ANGLES = (15, 30, 45, 60, 75)
MOTOR_ORDER = 0
LR = 0.012
DECAY = 0.00025
PAY = (1, 2, 1)
METRICS = ("a", "b", "both", "payout", "valid_groups_per_life", "deferred_l1_per_life",
           "log_probability_per_group", "gradient_norm", "weight_l2")
PROTOCOL = EVIDENCE / "training-protocol.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def parent_paths():
    return {str(b): gate.storage.checkpoint_path(noon_ticket.source(cohort), "odor", "uniform", b, 8000)
            for cohort, births in (("original", BIRTHS), ("independent", CONFIRM)) for b in births}


def protocol():
    """Return the protocol and precise frozen provenance used by every fit."""
    return {
        "training_births": BIRTHS, "heldout_births": CONFIRM, "replica_seeds": SEEDS,
        "rollout_domain": TRAIN_DOMAIN, "steps_per_fit": STEPS,
        "routes_per_birth_per_update": WORLDS, "arms": ARMS, "checkpoints": CHECKPOINTS,
        "angles_degrees": ANGLES, "gap_ticks": 4, "probe_wait_ticks": 4,
        "probe_orders": "A then B; unpaid contact and a neutral wait occur after the first action",
        "parent_checkpoint": "odor/uniform at step 8000",
        "role_of_parent": "Frozen parameters and original active local write process. Each life starts cold once.",
        "laws": "Independent unbiased signs, sampled separately for every route and never provided to the selector.",
        "objective": {"correct_A": PAY[0], "correct_B": PAY[1], "joint_correct": PAY[2]},
        "B_preservation": "Double weight on realized B alone, joint reward, a small fixed fraction deferred, and "
                          "a prespecified independent paired test of B. Objective is a soft incentive.",
        "rule": {
            "learned_parameters": gate.FEATURES, "no_old_active_parameters": gate.CURRENT_FEATURES,
            "candidate_group_size": gate.GROUP_SIZE, "minimum_absolute_native_change": gate.MIN_PROPOSAL,
            "common_deferred_fraction": gate.DEFER_FRACTION,
            "proposal_partition": "Same-sign surviving native B proposals sorted by magnitude, nonoverlapping "
                                  "consecutive quadruples. Up to three unmatched proposals per sign pass unchanged.",
            "dose": "At one of four addresses withhold 75% of the minimum group magnitude, preserving residuals "
                    "at all four, native writes elsewhere, bias, birth and retirement. Per-group signed dose is "
                    "constant across selected-address controls, subject only to float32 subtraction roundoff.",
            "old_record": "Pre-receipt motor emission endpoint activities and the last actual common-edge "
                          "coefficient change from A. Node activities are physically stored once per node.",
            "training_update": "Sample once from the shared categorical head per valid group. Center actual serial "
                               "return by a leave-one-route-out mean within birth. Apply the analytical "
                               "score-function derivative over sampled groups, with Adam on head coefficients.",
            "optimizer": {"name": "Adam", "learning_rate": LR, "l2": DECAY,
                          "coefficient_bounds": [-4, 4]},
            "isolation": "Arms use disjoint random streams. No route continuation yields another reward. "
                         "The native commit helper calculates two neutral ticks while constructing the B "
                         "receipt; its hidden result is discarded in fitted routes. Only the sampled B write "
                         "enters either return action. The optimizer sees only realized payoffs and local scores.",
        },
        "source_sha256": {Path(p).name: digest(p) for p in (__file__, gate.__file__, gate.net.__file__,
                                                            gate.net.native.__file__, gate.storage.__file__)},
        "parent_sha256": {b: digest(p) for b, p in parent_paths().items()},
    }


def register():
    spec = protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text(encoding="utf8")) != json.loads(json.dumps(spec)):
            raise ValueError("The registered training protocol differs from current source or settings")
    else:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        storage._write_json(PROTOCOL, spec)
    return digest(PROTOCOL)


def training_worlds(seed, step, arm, worlds):
    """Independent geometries, laws and draws in root-major order."""
    rng = np.random.default_rng([TRAIN_DOMAIN, seed, step, ARMS.index(arm)])
    n = len(BIRTHS) * worlds
    a, tangent = rng.standard_normal((2, n, 3), dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    tangent -= np.einsum("ij,ij->i", a, tangent)[:, None] * a
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    theta = np.deg2rad(rng.choice(ANGLES, size=n)).astype(np.float32)[:, None]
    b = a * np.cos(theta) + tangent * np.sin(theta)
    orders = 2.0 * rng.integers(0, 2, (n, 5), dtype=np.int8).astype(np.float32) - 1
    laws = 2.0 * rng.integers(0, 2, (2, n), dtype=np.int8).astype(np.float32) - 1
    random = rng.random((n, 5, 5), dtype=np.float32)
    as_tensor = torch.from_numpy
    world = gate.environment.World(as_tensor(1.35 * a), as_tensor(1.35 * b),
                                   as_tensor(laws[0]), as_tensor(laws[1]), as_tensor(orders),
                                   as_tensor(random), np.empty((n, 0)))
    return world, rng


def one_update(g, rule, weights, optimizer, seed, step, arm, worlds):
    world, rng = training_worlds(seed, step, arm, worlds)
    prefix = gate.prepare(g, rule, world)
    draw = torch.from_numpy(rng.random(tuple(prefix.candidates.valid.shape), dtype=np.float32))
    selected = gate.choose(prefix, weights.detach(), mode=arm, draw=draw, gradient=True)
    after = gate.apply(prefix, selected)
    realized = gate.probe(prefix.rule, after, world, order=MOTOR_ORDER)
    correct_a, correct_b = realized.correct.float().unbind(1)
    joint = correct_a * correct_b
    reward = PAY[0] * correct_a + PAY[1] * correct_b + PAY[2] * joint
    by_birth = reward.reshape(len(BIRTHS), worlds)
    centered = (by_birth - by_birth.mean(1, keepdim=True)) * worlds / (worlds - 1)
    valid = prefix.candidates.valid
    active = torch.arange(len(gate.FEATURES)) < (
        len(gate.FEATURES) if arm == "full" else len(gate.CURRENT_FEATURES))
    optimizer.zero_grad(set_to_none=True)
    weights.grad = (-(centered.flatten()[:, None] * selected.score_gradient).mean(0)
                    + DECAY * weights.detach()) * active
    optimizer.step()
    with torch.no_grad():
        weights.clamp_(-4, 4)
    stats = (correct_a.mean(), correct_b.mean(), joint.mean(), reward.mean(), valid.sum(1).float().mean(),
             prefix.candidates.amount.abs().sum(1).mean(), selected.log_prob.sum() / valid.sum().clamp_min(1),
             weights.grad.norm(), weights.detach().norm())
    event = torch.cat((prefix.action_a.side[:, None].to(torch.int8),
                       prefix.experience_a.taste[:, None].to(torch.int8),
                       prefix.action_b.side[:, None].to(torch.int8),
                       prefix.experience_b.taste[:, None].to(torch.int8),
                       realized.payout.to(torch.int8),
                       torch.where(valid, selected.index, -1).to(torch.int8)), dim=1)
    return torch.stack(stats).tolist(), sha256(event.numpy().tobytes()).hexdigest()


def folder(replica, arm):
    return DATA / "training" / f"replica-{replica}" / arm


def fit(replica, arm, *, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed = SEEDS[replica]
    g, rule = noon_ticket.inherited("original")
    destination = folder(replica, arm)
    steps, worlds = (4, 16) if smoke else (STEPS, WORLDS)
    identity = None if smoke else register()
    weights = torch.nn.Parameter(torch.zeros(len(gate.FEATURES)))
    optimizer = torch.optim.Adam([weights], lr=LR)
    ledger, event_digests = [], []
    start = 0
    latest = destination / "resume.pt"
    if not smoke:
        destination.mkdir(parents=True, exist_ok=True)
        if latest.exists():
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            if (saved["protocol"], saved["replica"], saved["arm"]) != (identity, replica, arm):
                raise ValueError(f"Resume state has a different identity: {latest}")
            start, ledger, event_digests = saved["step"], saved["ledger"], saved["event_digests"]
            with torch.no_grad():
                weights.copy_(saved["coefficients"])
            optimizer.load_state_dict(saved["optimizer"])

    def save(step):
        if step in CHECKPOINTS:
            storage._write_raw(destination / f"step-{step:04d}.npz", coefficients=weights.detach().numpy(),
                               step=step, seed=seed, replica=replica, arm=arm, protocol=identity)
        storage._atomic(latest, lambda f: torch.save(
            dict(protocol=identity, replica=replica, arm=arm, step=step, ledger=ledger,
                 event_digests=event_digests, coefficients=weights.detach(), optimizer=optimizer.state_dict()), f))
        if step == STEPS:
            storage._write_raw(destination / "ledger.npz", steps=np.arange(1, step + 1), columns=METRICS,
                               metrics=ledger, event_sha256=event_digests, routes_per_step=len(BIRTHS) * worlds,
                               seed=seed, replica=replica, arm=arm, protocol=identity)

    if not smoke and (not latest.exists() or start == steps):
        save(start)
    started = perf_counter()
    for step in range(start + 1, steps + 1):
        stats, stamp = one_update(g, rule, weights, optimizer, seed, step, arm, worlds)
        ledger.append(stats)
        event_digests.append(stamp)
        if not smoke and step % 64 == 0:
            save(step)
        if step % 64 == 0 or smoke:
            metrics = np.mean(ledger[-64:], axis=0)[:4].round(4)
            print(replica, arm, step, metrics.tolist(), f"{perf_counter() - started:.2f}s", flush=True)


def publish():
    identity = register()
    heads, info = {}, {}
    for arm in ARMS:
        coefficients = []
        for replica in range(len(SEEDS)):
            fit_root = folder(replica, arm)
            saved = fit_root / f"step-{STEPS:04d}.npz"
            ledger = fit_root / "ledger.npz"
            with np.load(saved) as row, np.load(ledger) as data:
                if str(row["protocol"]) != identity or str(data["protocol"]) != identity or int(row["step"]) != STEPS:
                    raise ValueError("A fitted row or ledger belongs to a different training protocol")
                values = row["coefficients"]
                coefficients.append(values)
                if not np.isfinite(values).all() or len(data["steps"]) != STEPS:
                    raise ValueError("Unfinished or nonfinite fit")
                if arm == "no_old" and np.count_nonzero(values[len(gate.CURRENT_FEATURES):]):
                    raise ValueError("The no-old fit has nonzero historical coefficients")
                info[f"{arm}/{replica}"] = {
                    "routes": int(data["routes_per_step"]) * STEPS,
                    "checkpoint": str(saved.relative_to(ROOT)), "checkpoint_sha256": digest(saved),
                    "ledger": str(ledger.relative_to(ROOT)), "ledger_sha256": digest(ledger),
                    "first_64": dict(zip(METRICS, data["metrics"][:64].mean(0).tolist(), strict=True)),
                    "last_64": dict(zip(METRICS, data["metrics"][-64:].mean(0).tolist(), strict=True)),
                }
        heads[arm] = np.stack(coefficients)
    output = DATA / "training-heads.npz"
    storage._write_raw(output, **heads, feature_names=gate.FEATURES, replica_seeds=SEEDS, final_step=STEPS,
                       protocol_sha256=identity)
    storage._write_json(EVIDENCE / "training-summary.json",
                        dict(protocol_sha256=identity, heads_sha256=digest(output),
                             heads=str(output.relative_to(ROOT)), fits=info))
    print(output, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "fit", "publish", "smoke"))
    parser.add_argument("--arm", choices=ARMS, default="full")
    parser.add_argument("--replica", type=int, choices=range(len(SEEDS)), default=0)
    args = parser.parse_args()
    if args.phase == "register":
        print(register())
    elif args.phase == "publish":
        publish()
    else:
        fit(args.replica, args.arm, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()
