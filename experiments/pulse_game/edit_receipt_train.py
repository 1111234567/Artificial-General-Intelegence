"""Train matched pulse graphs with and without credit from a one-edit future fork."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import batched
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import edit_receipt_graph as native
from . import self_param_writer as fw


ARMS = ("baseline", "sensing", "causal", "shuffled")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/edit-receipt-20261119"
TRAIN_DOMAIN = 51793
CHECKPOINT = "birth_{birth}/{arm}/checkpoint_{step}.npz"
METRICS = (
    "actor_score_loss", "writer_score_loss", "motor_path_loss", "local_effect_loss",
    "future_benefit_loss", "physical_accuracy", "expected_accuracy", "mean_taste",
    "proposed", "attempted", "applied", "fork_contact", "fork_eligible", "fork_used",
    "fork_physical_mean", "fork_physical_abs", "fork_expected_mean", "fork_expected_abs",
    "fork_taste_mean", "fork_taste_abs", "fork_expected_taste_mean", "fork_expected_taste_abs",
    "fork_changed_side", "fork_benefit_mean", "grad_norm",
)


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    batch: int = 12
    contacts: int = 8
    horizon: int = 3
    writers: int = 8
    noise: float = .1
    world_seed: int = 43197
    lr: float = .003
    credit_target: str = "taste"
    writer_weight: float = .25
    motor_path_weight: float = 1.
    effect_weight: float = .25
    benefit_weight: float = 2.
    keep_connected: bool = True


@dataclass(frozen=True)
class Tape:
    x: torch.Tensor
    truth: torch.Tensor
    outcome: torch.Tensor
    draws: torch.Tensor

    def contact(self, step):
        left = self.x[:, step]
        return left, -left, self.outcome[:, step], self.draws[:, step]

    def fork_batch(self):
        return self.concatenate((self, self))

    @classmethod
    def concatenate(cls, tapes):
        return cls(*(torch.cat([getattr(tape, name) for tape in tapes])
                     for name in ("x", "truth", "outcome", "draws")))


def make_worlds(seed, count, contacts, *, noise=.1, layout="mixed"):
    """Create two separable odors whose independent assignments can change and return.

    Each two-contact block presents both odors. The motor draw and delivered-taste
    slip are exogenous and independent of the graph's history and choices.
    """
    layouts = {"stable": 0, "change": 1, "return": 2}
    if layout not in (*layouts, "mixed") or not 0 <= noise <= 1 or count < 1 or contacts < 4:
        raise ValueError("Expected a supported layout, count >= 1, contacts >= 4 and noise in [0, 1]")
    rng = np.random.default_rng(seed)
    cues = rng.normal(size=(count, 2, 3))
    cues /= np.linalg.norm(cues, axis=-1, keepdims=True)
    # Two nearly collinear odors would be ambiguous after a left/right swap.
    close = np.abs((cues[:, 0] * cues[:, 1]).sum(-1)) > .75
    while close.any():
        new = rng.normal(size=(int(close.sum()), 3))
        cues[close, 1] = new / np.linalg.norm(new, axis=-1, keepdims=True)
        close = np.abs((cues[:, 0] * cues[:, 1]).sum(-1)) > .75

    phases = rng.choice(3, p=(.5, .3, .2), size=count)
    if layout != "mixed":
        phases.fill(layouts[layout])
    flip = rng.choice(3, size=count)
    changed_cues = np.stack((flip != 1, flip != 0), axis=1)
    swap_at = rng.integers(2, max(3, contacts - 2), size=count)
    return_at = np.minimum(swap_at + rng.integers(2, 5, size=count), contacts - 1)
    times = np.arange(contacts)[None, :]
    changed = (phases[:, None] != 0) & (times >= swap_at[:, None])
    changed &= (phases[:, None] != 2) | (times < return_at[:, None])

    assignments = rng.choice((-1, 1), size=(count, 2))
    order = rng.random((count, (contacts + 1) // 2, 2)).argsort(-1).reshape(count, -1)[:, :contacts]
    rows = np.arange(count)[:, None]
    orientation = rng.choice((-1, 1), size=(count, contacts))
    slips = np.where(rng.random((count, contacts)) < noise, -1, 1)
    x = (1.35 * cues[rows, order] * orientation[..., None]).astype(np.float32)
    truth = assignments[rows, order] * orientation * np.where(changed & changed_cues[rows, order], -1, 1)
    draws = rng.random((count, contacts, 4), dtype=np.float32)
    return Tape(*(torch.from_numpy(values) for values in
                  (x, truth.astype(np.float32), (truth * slips).astype(np.float32), draws)))


def world_for_update(config, birth, update):
    seed = [TRAIN_DOMAIN, config.world_seed, birth, update]
    tape = make_worlds([*seed, 0], config.batch, config.contacts, noise=config.noise)
    focal_rng = np.random.default_rng([TRAIN_DOMAIN, config.world_seed, update, 1])
    focal = int(focal_rng.integers(config.contacts - config.horizon))
    shuffle = torch.from_numpy(np.random.default_rng([*seed, 2]).permutation(config.batch))
    return tape, focal, shuffle


@dataclass(frozen=True)
class Fork:
    eligible: torch.Tensor
    physical: torch.Tensor
    expected: torch.Tensor
    taste: torch.Tensor
    expected_taste: torch.Tensor
    changed_side: torch.Tensor


def combine_states(first, second):
    a, b = first.graph, second.graph
    graph = fw.Graph(torch.cat((a.weights, b.weights)), torch.cat((a.bias, b.bias)),
                     torch.cat((a.present, b.present)))
    return native.State(graph, torch.cat((first.hidden, second.hidden)))


@torch.no_grad()
def future_fork(state, proposal, tape, focal, horizon, *, sensing, keep_connected):
    """Try or suppress one fixed current proposal, then let both continuations act."""
    draws = tape.draws[:, focal]
    execute = native.commit(state, proposal, draws, sensing=sensing,
                            keep_connected=keep_connected, accept_override=True)
    suppress = native.commit(state, proposal, draws, sensing=sensing,
                             keep_connected=keep_connected, accept_override=False)
    branches = combine_states(execute.state, suppress.state)
    paired = tape.fork_batch()
    eligible = execute.receipt.status == fw.Status.APPLIED
    physical = torch.zeros_like(proposal.benefit)
    expected = torch.zeros_like(physical)
    taste = torch.zeros_like(physical)
    expected_taste = torch.zeros_like(physical)
    changed_side = torch.zeros_like(physical)
    for step in range(focal + 1, focal + horizon + 1):
        left, right, outcome, draws = paired.contact(step)
        if step < focal + horizon:
            choice = native.propose(branches, left, right, outcome, draws, sensing=sensing)
            motor_logit, side, earned = choice.motor_logit, choice.side, choice.taste
            branches = native.commit(branches, choice, draws, sensing=sensing,
                                     keep_connected=keep_connected).state
        else:
            # Later edits cannot affect any motor choice inside the scoring window.
            _, motor_logit = native.motor(branches, left, right)
            side = (draws[:, 2] < motor_logit.sigmoid()).float() * 2 - 1
            earned = side * outcome
        truth = paired.truth[:, step]
        a, b = (side == truth).chunk(2)
        pa, pb = (motor_logit * truth).sigmoid().chunk(2)
        ta, tb = earned.chunk(2)
        tpa, tpb = (motor_logit * (earned * side)).sigmoid().chunk(2)
        side_a, side_b = side.chunk(2)
        physical += a.float() - b.float()
        expected += pa - pb
        taste += (ta - tb) / 2
        expected_taste += tpa - tpb
        changed_side += (side_a != side_b).float()
    return Fork(eligible, *(values / horizon for values in
                            (physical, expected, taste, expected_taste, changed_side)))


def paired_targets(target, eligible, shuffle, *, shuffled):
    """Rotate eligible targets in random order with no lane receiving itself."""
    ordered = shuffle[eligible[shuffle]]
    if len(ordered) < 2:
        return target, torch.zeros_like(eligible)
    if not shuffled:
        return target, eligible
    misassigned = target.clone()
    misassigned[ordered] = target[ordered.roll(1)]
    return misassigned, eligible


def mean_masked(values, mask, dims):
    return (values * mask).sum(dims) / mask.sum(dims).clamp_min(1)


def leave_one_out_advantage(returns):
    count = returns.shape[1]
    return (returns - returns.mean(1, keepdim=True)) * (count / (count - 1))


def arm_sensing(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown training arm {arm!r}")
    return "blind" if arm == "baseline" else "own"


def training_step(initial, tape, focal, shuffle, arm, config):
    """Compute one loss/log record for each of one or more independent birth fits."""
    sensing = arm_sensing(arm)
    roots = len(initial.bias)
    batch = config.batch
    if roots == 1:
        state = native.start(initial, batch)
    else:
        expanded = fw.Graph(*(parameter.repeat_interleave(batch, dim=0) for parameter in
                              (initial.weights, initial.bias, initial.present)))
        state = native.State(expanded, torch.zeros_like(expanded.bias))
    proposed, attempted, applied, effect_losses = [], [], [], []
    tastes, sides, actor, address, operation, gate, motors, physical = [], [], [], [], [], [], [], []
    for step in range(config.contacts):
        proposal = native.propose(state, *tape.contact(step), sensing=sensing)
        if step == focal:
            fork = future_fork(state, proposal, tape, focal, config.horizon, sensing=sensing,
                               keep_connected=config.keep_connected)
            prediction = proposal.benefit

        result = native.commit(state, proposal, tape.draws[:, step], sensing=sensing,
                               keep_connected=config.keep_connected)
        state = result.state
        suggested = proposal.command.op != fw.Op.NONE
        tried = result.accepted & suggested
        proposed.append(suggested)
        attempted.append(tried)
        applied.append(result.receipt.status == fw.Status.APPLIED)
        effect_losses.append((proposal.effect_prediction - result.receipt.effect_target.detach()).square().mean(1))
        tastes.append(proposal.taste)
        sides.append(proposal.side)
        actor.append(proposal.motor_log_prob)
        address.append(proposal.address_log_prob)
        operation.append(proposal.op_log_prob)
        gate.append(result.gate_log_prob)
        motors.append(proposal.motor_logit)
        physical.append(proposal.side == tape.truth[:, step])

    def by_contact(records):
        return torch.stack(records, dim=1).reshape(roots, batch, config.contacts)

    def by_birth(values):
        return values.reshape(roots, batch)

    tastes = by_contact(tastes).detach()
    actor, address, operation, gate, motors = (by_contact(items) for items in
                                               (actor, address, operation, gate, motors))
    # reward_to_go works on the contact dimension of a lanes × time tensor.
    flattened_tastes = tastes.flatten(end_dim=1)
    full = fw.reward_to_go(flattened_tastes).reshape_as(tastes)
    short = fw.reward_to_go(flattened_tastes, config.horizon).reshape_as(tastes)
    motor_future = full - tastes
    next_future = short[..., 1:] / config.horizon
    actor_loss = -(actor * leave_one_out_advantage(motor_future / config.contacts)).mean((1, 2))
    writer_loss = -((address + operation + gate)[..., :-1]
                    * leave_one_out_advantage(next_future)).mean((1, 2))
    # Once actual taste arrives, the signed taste and physically contacted side
    # identify the delivered binary preference. This gives a pathwise motor loss
    # on the realized graph without leaking the clean private answer.
    delivered_preference = by_contact(sides) * tastes
    mean_taste_now = delivered_preference * (2 * motors.sigmoid() - 1)
    motor_path_loss = -mean_taste_now.mean((1, 2)) / config.contacts
    effect_loss = mean_masked(by_contact(effect_losses), by_contact(attempted), (1, 2))
    raw_target = fork.taste if config.credit_target == "taste" else fork.expected_taste
    eligible = by_birth(fork.eligible)
    paired = [paired_targets(values, mask, order, shuffled=arm == "shuffled")
              for values, mask, order in zip(by_birth(raw_target), eligible, by_birth(shuffle), strict=True)]
    target, benefit_mask = map(torch.stack, zip(*paired, strict=True))
    prediction = by_birth(prediction)
    benefit_loss = mean_masked((prediction - target).square(), benefit_mask, 1)
    benefit_weight = config.benefit_weight if arm in ("causal", "shuffled") else 0.
    loss = (actor_loss + config.writer_weight * writer_loss + config.motor_path_weight * motor_path_loss
            + config.effect_weight * effect_loss + benefit_weight * benefit_loss)
    with torch.no_grad():
        metrics = torch.stack((
            actor_loss, writer_loss, motor_path_loss, effect_loss, benefit_loss,
            by_contact(physical).float().mean((1, 2)),
            (motors * tape.truth.reshape_as(motors)).sigmoid().mean((1, 2)), tastes.mean((1, 2)),
            *(by_contact(records).float().mean((1, 2)) for records in (proposed, attempted, applied)),
            prediction.new_full((roots,), focal), eligible.sum(1), benefit_mask.sum(1) * (benefit_weight > 0),
            *(mean_masked(by_birth(values), eligible, 1) for values in
              (fork.physical, fork.physical.abs(), fork.expected, fork.expected.abs(), fork.taste,
               fork.taste.abs(), fork.expected_taste, fork.expected_taste.abs(), fork.changed_side)),
            mean_masked(prediction, eligible, 1),
        ), dim=1).numpy()
    return loss.sum(), metrics


def checkpoint_path(path, arm, birth, step):
    return Path(path) / CHECKPOINT.format(arm=arm, birth=birth, step=step)


def write_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def save_checkpoint(path, arm, birth, step, graph):
    write_arrays(checkpoint_path(path, arm, birth, step), arm=arm, birth=birth, step=step,
                 weights=graph.weights.detach().numpy(), bias=graph.bias.detach().numpy(),
                 present=graph.present.numpy())


def load_checkpoint(path, arm, birth, step):
    """Load one single-lane birth template saved at a frozen training stage."""
    filename = checkpoint_path(path, arm, birth, step)
    with np.load(filename, allow_pickle=False) as data:
        if (data["arm"].item(), int(data["birth"]), int(data["step"])) != (arm, birth, step):
            raise ValueError(f"Checkpoint identity mismatch at {filename}")
        return fw.Graph(*(torch.from_numpy(data[key]) for key in ("weights", "bias", "present")))


def clip_birth_gradients(parameters, maximum):
    with torch.no_grad():
        sum_squares = sum(parameter.grad.flatten(1).square().sum(1) for parameter in parameters)
        norms = sum_squares.sqrt()
        if not norms.isfinite().all():
            raise RuntimeError("Training produced a non-finite per-birth gradient norm")
        factors = (maximum / (norms + 1e-6)).clamp(max=1)
        for parameter in parameters:
            parameter.grad.mul_(factors.reshape((-1,) + (1,) * (parameter.ndim - 1)))
    return norms


def fit(path, config, births, arm, checkpoints, progress):
    torch.set_num_threads(1)
    born = [fw.spawn(birth, config.writers) for birth in births]
    weights = nn.Parameter(torch.cat([graph.weights for graph in born]))
    bias = nn.Parameter(torch.cat([graph.bias for graph in born]))
    initial = fw.Graph(weights, bias, torch.cat([graph.present for graph in born]))
    optimizer = torch.optim.Adam((weights, bias), lr=config.lr)
    trace = np.empty((len(births), config.updates, len(METRICS)), dtype=np.float32)
    started = perf_counter()
    if 0 in checkpoints:
        for birth, graph in zip(births, born, strict=True):
            save_checkpoint(path, arm, birth, 0, graph)
    for step in range(1, config.updates + 1):
        tapes, foci, shuffles = zip(*(world_for_update(config, birth, step) for birth in births), strict=True)
        tape = Tape.concatenate(tapes) if len(births) > 1 else tapes[0]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_step(initial, tape, foci[0], torch.cat(shuffles), arm, config)
        loss.backward()
        grad_norm = clip_birth_gradients((weights, bias), 2.)
        optimizer.step()
        trace[:, step - 1, :-1] = metrics
        trace[:, step - 1, -1] = grad_norm.numpy()
        elapsed = perf_counter() - started
        if step in checkpoints:
            for lane, birth in enumerate(births):
                group = slice(lane, lane + 1)
                graph = fw.Graph(weights[group], bias[group], initial.present[group])
                save_checkpoint(path, arm, birth, step, graph)
                filename = checkpoint_path(path, arm, birth, step).with_name("training.npz")
                write_arrays(filename, values=trace[lane, :step], metrics=METRICS, seconds=elapsed)
        if step == 1 or step in checkpoints or (progress and step % progress == 0):
            accuracy = trace[:, max(0, step - 100):step, METRICS.index("physical_accuracy")].mean(1)
            for birth, value in zip(births, accuracy, strict=True):
                print(f"birth={birth} arm={arm} step={step}/{config.updates} "
                      f"accuracy={value:.3f} seconds={elapsed:.1f}", flush=True)


def validate(config):
    if config.batch < 2 or config.contacts < 4 or not 1 <= config.horizon < config.contacts:
        raise ValueError("Require batch >= 2 and 1 <= horizon < contacts with contacts >= 4")
    if config.updates < 1 or config.lr <= 0 or not 0 <= config.noise <= 1:
        raise ValueError("Require positive updates/rate and noise in [0, 1]")
    if config.credit_target not in ("taste", "expected_taste"):
        raise ValueError("Expected taste or expected_taste as the training credit target")


def manifest(path, config, births, arms, checkpoints, group_size):
    sources = (Path(__file__), Path(native.__file__), Path(fw.__file__))
    return {
        "schema_version": 1, "config": asdict(config), "births": births, "arms": arms,
        "checkpoints": checkpoints, "metrics": METRICS, "checkpoint_template": CHECKPOINT,
        "trace_template": "birth_{birth}/{arm}/training.npz",
        "train_world_history_count_per_fit": config.updates * config.batch,
        "main_contacts_per_fit": config.updates * config.batch * config.contacts,
        "fork_future_contacts_per_fit": 2 * config.updates * config.batch * config.horizon,
        "arm_sensing": {arm: arm_sensing(arm) for arm in arms},
        "world_key": [TRAIN_DOMAIN, config.world_seed, "birth", "update", 0],
        "focal_key": [TRAIN_DOMAIN, config.world_seed, "update", 1],
        "shuffle_key": [TRAIN_DOMAIN, config.world_seed, "birth", "update", 2],
        "fit_group_size": group_size,
        "layout_probabilities": {"stable": .5, "change": .3, "return": .2},
        "sources_sha256": {source.name: sha256(source.read_bytes()).hexdigest() for source in sources},
        "output": str(Path(path).resolve()),
    }


def run(path, config, births, arms, checkpoints, *, workers=1, group_size=1, progress=250):
    validate(config)
    if (workers < 1 or group_size < 1 or progress < 0 or not births or len(set(births)) != len(births)
            or min(births) < 0 or not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS)):
        raise ValueError("Require unique nonnegative births, distinct known arms, positive workers/group size and nonnegative progress")
    checkpoints = sorted({0, config.updates, *checkpoints})
    if checkpoints[0] < 0 or checkpoints[-1] > config.updates:
        raise ValueError("Checkpoints must fall between 0 and the final update")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    plan = manifest(path, config, births, arms, checkpoints, group_size)
    (path / "run.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    jobs = [(path, config, group, arm, checkpoints, progress)
            for group in batched(births, group_size) for arm in arms]
    if workers == 1:
        for job in jobs:
            fit(*job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job in [pool.submit(fit, *args) for args in jobs]:
                job.result()
    return plan


def main():
    parser = ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--births", type=int, nargs="+", default=[421, 422, 423, 424])
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--checkpoints", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1, help="Independent births sharing one batched graph runtime.")
    parser.add_argument("--progress", type=int, default=250, help="Print every N updates, or 0 for checkpoints only.")
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--contacts", type=int, default=defaults.contacts)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--writers", type=int, choices=fw.WRITER_WIDTHS, default=defaults.writers)
    parser.add_argument("--noise", type=float, default=defaults.noise)
    parser.add_argument("--world-seed", type=int, default=defaults.world_seed)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--credit-target", choices=("taste", "expected_taste"), default=defaults.credit_target)
    parser.add_argument("--allow-disconnection", action="store_true")
    args = parser.parse_args()
    args.keep_connected = not args.allow_disconnection
    config = Config(**{name: getattr(args, name) for name in asdict(defaults) if hasattr(args, name)})
    checkpoints = args.checkpoints if args.checkpoints is not None else [
        stage for stage in (500, 2000, 8000) if stage <= args.updates]
    run(args.out, config, args.births, args.arms, checkpoints, workers=args.workers,
        group_size=args.group_size, progress=args.progress)


if __name__ == "__main__":
    main()
