"""Bootstrap native edit choice using actual taste at a single odor's return."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from itertools import batched
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import edit_receipt_train as storage
from . import revisit_choice_graph as native
from . import self_param_writer as fw


ARMS = ("ranked", "shuffled", "outcome_only")
BIRTHS = (731, 732, 733, 734, 735, 736)
STEPS = (0, 500, 2000, 8000)
TRAIN_DOMAIN = 62787
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data/revisit-choice-20261119"
METRICS = (
    "first_positive", "return_positive", "return_expected_positive", "chosen_applied",
    "chosen_0", "chosen_1", "chosen_skip", "branch_0_positive", "branch_1_positive",
    "branch_skip_positive", "branch_0_applied", "branch_1_applied", "identical_commands",
    "pair_ties", "pair_non_ties", "all_ties", "better_than_skip", "worse_than_skip",
    "tied_with_skip", "rank_hits", "selection_hits", "gate_hits", "mean_wait",
    "world_side_changed", "actor_score_loss", "proposal_score_loss", "choice_score_loss",
    "motor_path_loss", "rank_loss", "selector_rank_loss", "gate_loss", "grad_norm",
)


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    batch: int = 16
    writers: int = 8
    min_wait: int = 0
    max_wait: int = 8
    world_seed: int = 43197
    lr: float = .003
    actor_weight: float = .25
    proposal_weight: float = .25
    choice_weight: float = .25
    first_path_weight: float = .25
    rank_weight: float = .5
    gate_weight: float = .5
    keep_connected: bool = True


@dataclass(frozen=True)
class Tape:
    first_left: torch.Tensor
    return_left: torch.Tensor
    first_outcome: torch.Tensor
    return_outcome: torch.Tensor
    wait: torch.Tensor
    first_draw: torch.Tensor
    proposal_draws: torch.Tensor
    choice_draw: torch.Tensor
    return_draw: torch.Tensor

    @classmethod
    def concatenate(cls, tapes):
        return cls(*(torch.cat([getattr(tape, field.name) for tape in tapes])
                     for field in fields(cls)))


@dataclass(frozen=True)
class Branches:
    side: torch.Tensor
    taste: torch.Tensor
    logit: torch.Tensor
    applied: torch.Tensor


def world_for_update(config, birth, update):
    """Generate a fresh odor and private stable taste, with independent side orders."""
    key = [TRAIN_DOMAIN, config.world_seed, birth, update]
    rng = np.random.default_rng([*key, 0])
    batch = config.batch
    odor = rng.normal(size=(batch, 3)).astype(np.float32)
    odor *= 1.35 / np.linalg.norm(odor, axis=1, keepdims=True)
    first_side, return_side, preference = rng.integers(0, 2, size=(3, batch)).astype(np.float32) * 2 - 1
    wait = rng.integers(config.min_wait, config.max_wait + 1, size=batch)
    draws = rng.random((batch, 7), dtype=np.float32)
    tape = Tape(*(torch.from_numpy(values) for values in (
        odor * first_side[:, None], odor * return_side[:, None], first_side * preference,
        return_side * preference, wait, draws[:, 0], draws[:, 1:5].reshape(batch, 2, 2),
        draws[:, 5], draws[:, 6],
    )))
    # Rotate a uniformly permuted world batch to preserve entire joint consequence vectors.
    ordered = np.random.default_rng([*key, 1]).permutation(batch)
    donors = np.empty(batch, dtype=np.int64)
    donors[ordered] = np.roll(ordered, 1)
    return tape, torch.from_numpy(donors)


def return_branches(experience, proposals, tape, *, max_wait, keep_connected):
    """Apply each endogenous candidate or skip, with a time-matched shared future."""
    lanes = len(tape.wait)
    branch = torch.arange(3, device=tape.wait.device).repeat_interleave(lanes)
    decision = native.Decision(branch, torch.zeros_like(tape.choice_draw).repeat(3))
    committed = native.commit(experience.fork_batch(3), proposals.fork_batch(3), decision,
                              keep_connected=keep_connected)
    state = native.wait(committed.state, tape.wait.repeat(3), max_ticks=max_wait)
    left = tape.return_left.repeat(3, 1)
    action = native.act(state, left, -left, tape.return_draw.repeat(3))
    taste = action.side * tape.return_outcome.repeat(3)

    def by_lane(values):
        return values.reshape(3, lanes).transpose(0, 1)

    return Branches(*(by_lane(values) for values in (
        action.side, taste, action.motor_logit, committed.raw_receipt.status == fw.Status.APPLIED,
    )))


def preference_targets(tastes, donors, *, shuffled):
    """Keep ties and each donor's joint pair/skip outcomes intact within a birth."""
    if shuffled:
        tastes = tastes.gather(1, donors[:, :, None].expand_as(tastes))
    return ((tastes[:, :, 0] - tastes[:, :, 1]).sign(),
            (tastes[:, :, :2] - tastes[:, :, 2:]).sign())


def preference_loss(score, sign):
    """Retain tied visits in accounting but assign them zero directional loss."""
    eligible = sign != 0
    dims = tuple(range(1, score.ndim))
    return (F.softplus(-score * sign) * eligible).sum(dims) / eligible.sum(dims).clamp_min(1)


def training_step(initial, tape, donors, arm, config):
    """One outer update, with one independent loss per stacked birth."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    roots = len(initial.bias)
    batch = config.batch
    graph = fw.Graph(*(tensor.repeat_interleave(batch, dim=0) for tensor in
                       (initial.weights, initial.bias, initial.present)))
    state = native.State(graph, torch.zeros_like(graph.bias))
    first = native.act(state, tape.first_left, -tape.first_left, tape.first_draw)
    first_taste = first.side * tape.first_outcome
    experience = native.taste(first, tape.first_left, -tape.first_left, first_taste)
    proposals = native.propose(experience, tape.proposal_draws)
    choice = native.choose(proposals, tape.choice_draw)
    branches = return_branches(experience, proposals, tape, max_wait=config.max_wait,
                               keep_connected=config.keep_connected)

    def by_birth(values):
        return values.reshape(roots, batch, *values.shape[1:])

    def actual(values):
        return by_birth(values.gather(1, choice.index[:, None])[:, 0])

    actual_taste = actual(branches.taste).detach()
    actual_side = actual(branches.side)
    actual_logit = actual(branches.logit)
    advantage = storage.leave_one_out_advantage(actual_taste)
    # Joint sampling score includes both nominees because either can change the choice.
    actor_loss = -(by_birth(first.motor_log_prob) * advantage).mean(1)
    nomination_score = proposals.address_log_prob + proposals.op_log_prob
    proposal_loss = -(by_birth(nomination_score.sum(1)) * advantage).mean(1)
    choice_loss = -(by_birth(choice.log_prob) * advantage).mean(1)
    # The delivered taste and executed side reveal the current binary preference
    # only after the choice. Use that sensory identity on the realized branch.
    delivered = (actual_taste * actual_side).detach()
    first_delivered = by_birth((first_taste * first.side).detach())
    motor_loss = (F.softplus(-actual_logit * delivered).mean(1)
                  + config.first_path_weight * F.softplus(-by_birth(first.motor_logit) * first_delivered).mean(1))
    branch_taste = by_birth(branches.taste).detach()
    pair_target, gate_target = preference_targets(branch_taste, donors.reshape(roots, batch),
                                                  shuffled=arm == "shuffled")
    benefit = by_birth(proposals.benefit[:, 0] - proposals.benefit[:, 1])
    selector = by_birth(proposals.logits[:, 0] - proposals.logits[:, 1])
    gate = by_birth(proposals.logits[:, :2] - proposals.logits[:, 2:])
    rank_loss = preference_loss(benefit, pair_target)
    selector_rank_loss = preference_loss(selector, pair_target)
    gate_loss = preference_loss(gate, gate_target)
    loss = (motor_loss + config.actor_weight * actor_loss + config.proposal_weight * proposal_loss
            + config.choice_weight * choice_loss)
    if arm != "outcome_only":
        loss = loss + config.rank_weight * (rank_loss + selector_rank_loss) + config.gate_weight * gate_loss

    with torch.no_grad():
        physical_pair, physical_gate = preference_targets(branch_taste, None, shuffled=False)
        cmd = proposals.command
        same = by_birth((cmd.address[:, 0] == cmd.address[:, 1]) & (cmd.op[:, 0] == cmd.op[:, 1])
                        & (cmd.payload[:, 0] == cmd.payload[:, 1]))
        selected = by_birth(choice.index)
        metrics = torch.stack((
            by_birth(first_taste > 0).float().mean(1), (actual_taste > 0).float().mean(1),
            (actual_logit * delivered).sigmoid().mean(1), actual(branches.applied).float().mean(1),
            *((selected == arm_index).float().mean(1) for arm_index in range(3)),
            *((branch_taste[:, :, arm_index] > 0).float().mean(1) for arm_index in range(3)),
            *(by_birth(branches.applied[:, arm_index]).float().mean(1) for arm_index in range(2)),
            same.sum(1), (physical_pair == 0).sum(1), (physical_pair != 0).sum(1),
            (physical_gate == 0).all(2).sum(1), (physical_gate > 0).sum((1, 2)),
            (physical_gate < 0).sum((1, 2)), (physical_gate == 0).sum((1, 2)),
            ((benefit.sign() == physical_pair) & (physical_pair != 0)).sum(1),
            ((selector.sign() == physical_pair) & (physical_pair != 0)).sum(1),
            ((gate.sign() == physical_gate) & (physical_gate != 0)).sum((1, 2)),
            by_birth(tape.wait).float().mean(1),
            by_birth(tape.first_outcome != tape.return_outcome).float().mean(1),
            actor_loss, proposal_loss, choice_loss, motor_loss, rank_loss, selector_rank_loss, gate_loss,
        ), dim=1).numpy()
    return loss.sum(), metrics


def load_checkpoint(path, arm, birth, step):
    """Load the single-lane graph saved at one longitudinal checkpoint."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    return storage.load_checkpoint(path, arm, birth, step)


def fit(path, config, births, arm, checkpoints, progress):
    torch.set_num_threads(1)
    born = [fw.spawn(birth, config.writers) for birth in births]
    weights = nn.Parameter(torch.cat([graph.weights for graph in born]))
    bias = nn.Parameter(torch.cat([graph.bias for graph in born]))
    initial = fw.Graph(weights, bias, torch.cat([graph.present for graph in born]))
    optimizer = torch.optim.Adam((weights, bias), lr=config.lr)
    trace = np.empty((len(births), config.updates, len(METRICS)), dtype=np.float32)
    started = perf_counter()
    for birth, graph in zip(births, born, strict=True):
        storage.save_checkpoint(path, arm, birth, 0, graph)
    for step in range(1, config.updates + 1):
        tapes, donors = zip(*(world_for_update(config, birth, step) for birth in births), strict=True)
        tape = Tape.concatenate(tapes) if len(births) > 1 else tapes[0]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_step(initial, tape, torch.stack(donors), arm, config)
        loss.backward()
        grad_norm = storage.clip_birth_gradients((weights, bias), 2.)
        optimizer.step()
        trace[:, step - 1, :-1] = metrics
        trace[:, step - 1, -1] = grad_norm.numpy()
        elapsed = perf_counter() - started
        if step in checkpoints:
            for lane, birth in enumerate(births):
                group = slice(lane, lane + 1)
                graph = fw.Graph(weights[group], bias[group], initial.present[group])
                storage.save_checkpoint(path, arm, birth, step, graph)
                filename = storage.checkpoint_path(path, arm, birth, step).with_name("training.npz")
                storage.write_arrays(filename, values=trace[lane, :step], metrics=METRICS, seconds=elapsed)
        if step == 1 or step in checkpoints or (progress and step % progress == 0):
            rec = trace[:, max(0, step - 100):step]
            positive = rec[:, :, METRICS.index("return_positive")].mean(1)
            untied = rec[:, :, METRICS.index("pair_non_ties")].sum(1) / (config.batch * rec.shape[1])
            for birth, value, fraction in zip(births, positive, untied, strict=True):
                print(f"birth={birth} arm={arm} step={step}/{config.updates} "
                      f"return_positive={value:.3f} non_tied={fraction:.3f} seconds={elapsed:.1f}", flush=True)


def run(path, config, births=BIRTHS, arms=ARMS, checkpoints=None, *, workers=1, group_size=1, progress=250):
    births, arms = tuple(births), tuple(arms)
    if (config.updates < 0 or config.batch < 2 or not 0 <= config.min_wait <= config.max_wait
            or config.world_seed < 0 or config.writers not in fw.WRITER_WIDTHS or config.lr <= 0):
        raise ValueError("Require nonnegative updates, seed and delays, supported writers, and batch >= 2")
    if (workers < 1 or group_size < 1 or progress < 0 or not births or len(set(births)) != len(births)
            or min(births) < 0 or not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS)):
        raise ValueError("Require unique births and known arms, positive workers/group size and nonnegative progress")
    if checkpoints is None:
        checkpoints = [step for step in STEPS if step <= config.updates]
    checkpoints = sorted({0, config.updates, *checkpoints})
    if checkpoints[0] < 0 or checkpoints[-1] > config.updates:
        raise ValueError("Checkpoints must fall between 0 and the final update")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__), Path(native.__file__), Path(fw.__file__), Path(storage.__file__))
    plan = {
        "schema_version": 1, "config": asdict(config), "births": births, "arms": arms,
        "checkpoints": checkpoints, "checkpoint_template": storage.CHECKPOINT,
        "trace_template": "birth_{birth}/{arm}/training.npz", "metrics": METRICS,
        "world_key": [TRAIN_DOMAIN, config.world_seed, "birth", "update", 0],
        "shuffle_key": [TRAIN_DOMAIN, config.world_seed, "birth", "update", 1],
        "worlds_per_fit": config.batch * config.updates, "branch_returns_per_world": 3,
        "fit_group_size": group_size,
        "sources_sha256": {source.name: sha256(source.read_bytes()).hexdigest() for source in sources},
    }
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--births", type=int, nargs="+", default=BIRTHS)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--checkpoints", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--progress", type=int, default=250)
    defaults = Config()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if name == "keep_connected":
            parser.add_argument(flag, action="store_true", default=value)
            parser.add_argument("--allow-disconnection", dest=name, action="store_false")
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in asdict(defaults)})
    run(args.out, config, args.births, args.arms, args.checkpoints, workers=args.workers,
        group_size=args.group_size, progress=args.progress)


if __name__ == "__main__":
    main()
