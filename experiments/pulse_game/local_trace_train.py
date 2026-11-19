"""Bootstrap synapse-local plasticity from the resource earned by a sampled final action."""

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

from . import local_trace_graph as native
from . import revisit_choice_graph as legacy
from . import self_param_writer as fw
from .edit_receipt_train import clip_birth_gradients, write_arrays


TASKS = ("odor", "feeder")
ARMS = ("active", "independent")
CONTROLLERS = ("local", "legacy")
BIRTHS = (731, 732, 733, 734, 735, 736)
STEPS = (0, 1000, 4000, 8000)
TRAIN_DOMAIN = 529731
CHECKPOINT = "birth_{birth}/{task}/{arm}/checkpoint_{step}.npz"
TRACE = "birth_{birth}/{task}/{arm}/training.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data/local-trace-20261119"
MARK_AXIS = np.array([1.35, 0., 0.], dtype=np.float32)
METRICS = (
    "terminal_resource", "terminal_positive", "positive_law_success", "negative_law_success",
    "both_twin_positive", "terminal_motor_score_loss", "support_motor_score_loss", "edit_score_loss",
    "contacts_with_numeric_change", "edges_born_per_life", "edges_retired_per_life", "grad_norm",
)


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    pairs: int = 16
    min_wait: int = 0
    max_wait: int = 8
    world_seed: int = 43197
    lr: float = .003
    grad_clip: float = 2.0
    controller: str = "local"


@dataclass(frozen=True)
class Tape:
    """Environment-only data, with adjacent opposite-law lanes per tape."""

    left: torch.Tensor
    right: torch.Tensor
    support_factor: torch.Tensor
    donor_factor: torch.Tensor
    terminal_factor: torch.Tensor
    law: torch.Tensor
    wait: torch.Tensor
    draws: torch.Tensor

    @classmethod
    def concatenate(cls, tapes):
        return cls(*(torch.cat([getattr(tape, field.name) for tape in tapes]) for field in fields(cls)))


@dataclass(frozen=True)
class Rollout:
    terminal_resource: torch.Tensor
    terminal_motor_score: torch.Tensor
    support_motor_score: torch.Tensor
    edit_score: torch.Tensor
    terminal_side: torch.Tensor
    terminal_logit: torch.Tensor
    numeric_contacts: torch.Tensor
    births: torch.Tensor
    retirements: torch.Tensor


def odor_sensors(odor, orientation):
    left = odor * orientation[..., None]
    return left, -left


def feeder_mark(sign):
    """The two icons in the existing feeder-mark calibration benchmark."""
    return np.asarray(sign, dtype=np.float32)[..., None] * MARK_AXIS


def world_for_update(config, task, birth, update):
    """Share inputs and draws across laws and treatments, with a separate donor stream."""
    key = [TRAIN_DOMAIN, config.world_seed, TASKS.index(task), birth, update]
    rng = np.random.default_rng([*key, 0])
    donor_rng = np.random.default_rng([*key, 1])
    pairs = config.pairs
    first_law = rng.choice((-1., 1.), size=pairs).astype(np.float32)
    first_donor = donor_rng.choice((-1., 1.), size=pairs).astype(np.float32)
    law = np.stack((first_law, -first_law), axis=1)
    donor = np.stack((first_donor, -first_donor), axis=1)
    first_sign, last_sign = rng.choice((-1., 1.), size=(2, pairs)).astype(np.float32)

    if task == "odor":
        odor = rng.normal(size=(pairs, 3)).astype(np.float32)
        odor *= 1.35 / np.linalg.norm(odor, axis=1, keepdims=True)
        orientation = np.stack((first_sign, last_sign), axis=1)
        left, right = odor_sensors(odor[:, None, :], orientation)
        support_sign = first_sign[:, None]
        wait = rng.integers(config.min_wait, config.max_wait + 1, size=pairs)
    else:
        support_sign = np.stack((first_sign, -first_sign), axis=1)
        support_mark, right_mark = feeder_mark(support_sign), feeder_mark(last_sign)
        left = np.concatenate((support_mark, -right_mark[:, None, :]), axis=1)
        right = np.concatenate((support_mark, right_mark[:, None, :]), axis=1)
        wait = np.zeros(pairs, dtype=np.int64)

    def twins(values):
        return np.repeat(values[:, None, ...], 2, axis=1)

    # Legacy uses one additional discrete draw, with common motors and available edit uniforms.
    draws = rng.random((pairs, left.shape[1], 6), dtype=np.float32)
    return Tape(*(torch.from_numpy(values) for values in (
        twins(left), twins(right), law[..., None] * support_sign[:, None, :],
        donor[..., None] * support_sign[:, None, :], law * last_sign[:, None],
        law, twins(wait), twins(draws),
    )))


class LocalController:
    def __init__(self, graph, rule, lives):
        self.state = native.start(graph, lives)
        self.rule = rule.expand(lives)

    def act(self, left, right, draws):
        return native.act(self.rule, self.state, left, right, draws[:, 0])

    def contact(self, action, left, right, resource, draws):
        observed = native.observe(self.rule, action, left, right, resource)
        result = native.commit(self.rule, observed, draws[:, 1:5])
        self.state = result.state
        return result.edit_log_prob

    def wait(self, durations, max_ticks):
        self.state = native.wait(self.rule, self.state, durations, max_ticks=max_ticks)


class LegacyController:
    def __init__(self, graph, rule, lives):
        expanded = fw.Graph(*(tensor.repeat_interleave(lives, dim=0)
                              for tensor in (graph.weights, graph.bias, graph.present)))
        self.state = legacy.State(expanded, torch.zeros_like(expanded.bias))

    def act(self, left, right, draws):
        return legacy.act(self.state, left, right, draws[:, 0])

    def contact(self, action, left, right, resource, draws):
        observed = legacy.taste(action, left, right, resource)
        candidates = legacy.propose(observed, draws[:, 1:5].reshape(-1, 2, 2))
        decision = legacy.choose(candidates, draws[:, 5])
        self.state = legacy.commit(observed, candidates, decision).state
        return (candidates.address_log_prob + candidates.op_log_prob).sum(1) + decision.log_prob

    def wait(self, durations, max_ticks):
        self.state = legacy.wait(self.state, durations, max_ticks=max_ticks)


def rollout(initial, rule, tape, task, arm, config):
    """Act before reading a physical receipt. Score the unobserved final payout separately."""
    if task not in TASKS or arm not in ARMS:
        raise ValueError(f"Unknown task or arm {task!r}, {arm!r}")
    lives = 2 * config.pairs
    controller = {"local": LocalController, "legacy": LegacyController}[config.controller](initial, rule, lives)
    left, right, draws = (tensor.flatten(0, 1) for tensor in (tape.left, tape.right, tape.draws))
    outcomes = (tape.support_factor if arm == "active" else tape.donor_factor).flatten(0, 1)
    support_scores, edit_scores = [], []
    with torch.no_grad():
        numeric = torch.zeros_like(controller.state.graph.bias[:, 0])
        births = torch.zeros_like(numeric)
        retirements = torch.zeros_like(numeric)

    for contact in range(left.shape[1] - 1):
        a, b, u = left[:, contact], right[:, contact], draws[:, contact]
        action = controller.act(a, b, u)
        earned = outcomes[:, contact]
        if task == "odor":
            earned = earned * action.side
        before = controller.state.graph
        support_scores.append(action.motor_log_prob)
        edit_scores.append(controller.contact(action, a, b, earned, u))
        with torch.no_grad():
            after = controller.state.graph
            shared = before.present & after.present
            numeric += ((before.weights.ne(after.weights) & shared).flatten(1).any(1)
                        | (before.bias != after.bias).any(1))
            births += (~before.present & after.present).sum((1, 2))
            retirements += (before.present & ~after.present).sum((1, 2))

    if task == "odor" and config.max_wait:
        controller.wait(tape.wait.flatten(), config.max_wait)
    terminal = controller.act(left[:, -1], right[:, -1], draws[:, -1])
    terminal_resource = (terminal.side * tape.terminal_factor.flatten()).detach()
    return Rollout(terminal_resource, terminal.motor_log_prob, sum(support_scores), sum(edit_scores),
                   terminal.side, terminal.motor_logit, numeric, births, retirements)


def leave_pair_out_advantage(returns):
    """Remove the mean resource of the other independent pairs of this birth."""
    if returns.ndim != 3 or returns.shape[1] < 2 or returns.shape[2] != 2:
        raise ValueError("Expected [birth, independent pair, twin] resource with at least two pairs")
    rewards = returns.detach()
    pair_reward = rewards.mean(2, keepdim=True)
    baseline = (pair_reward.sum(1, keepdim=True) - pair_reward) / (returns.shape[1] - 1)
    return rewards - baseline


def score_loss(log_probability, advantage):
    return -(log_probability * advantage).mean((1, 2))


def training_step(initial, rule, tape, task, arm, config):
    """REINFORCE on the realized trajectory, with one loss per independent birth."""
    rolled = rollout(initial, rule, tape, task, arm, config)

    def paired(values):
        return values.reshape(len(initial.bias), config.pairs, 2)

    resource = paired(rolled.terminal_resource)
    advantage = leave_pair_out_advantage(resource)
    scores = (rolled.terminal_motor_score, rolled.support_motor_score, rolled.edit_score)
    losses = tuple(score_loss(paired(score), advantage) for score in scores)

    with torch.no_grad():
        law = paired(tape.law)
        success = resource > 0
        by_law = tuple((success & (law == sign)).sum((1, 2)) / config.pairs for sign in (1, -1))
        metrics = torch.stack((
            resource.mean((1, 2)), success.float().mean((1, 2)), *by_law,
            success.all(2).float().mean(1), *losses,
            *(paired(value).mean((1, 2)) for value in (rolled.numeric_contacts, rolled.births,
                                                       rolled.retirements)),
        ), dim=1)
    return sum(losses).sum(), metrics.numpy()


def checkpoint_path(path, task, arm, birth, step):
    return Path(path) / CHECKPOINT.format(task=task, arm=arm, birth=birth, step=step)


def save_checkpoint(path, task, arm, birth, step, graph, rule, lane):
    slot = slice(lane, lane + 1)
    rule_state = {} if rule is None else {f"rule/{name}": value[slot].detach().numpy()
                                           for name, value in rule.state_dict().items()}
    write_arrays(checkpoint_path(path, task, arm, birth, step), task=task, arm=arm, birth=birth, step=step,
                 controller="legacy" if rule is None else "local",
                 weights=graph.weights[slot].detach().numpy(), bias=graph.bias[slot].detach().numpy(),
                 present=graph.present[slot].numpy(), **rule_state)


def load_checkpoint(path, task, arm, birth, step):
    """Load a frozen birth graph and the local rule where present."""
    path = Path(path)
    filename = checkpoint_path(path.parent if path.is_file() else path, task, arm, birth, step)
    with np.load(filename, allow_pickle=False) as data:
        identity = (data["task"].item(), data["arm"].item(), int(data["birth"]), int(data["step"]))
        if identity != (task, arm, birth, step):
            raise ValueError(f"Checkpoint identity mismatch at {filename}")
        graph = fw.Graph(*(torch.from_numpy(data[key]) for key in ("weights", "bias", "present")))
        rule = None
        if data["controller"].item() == "local":
            rule = native.Rule(birth)
            rule.load_state_dict({key.removeprefix("rule/"): torch.from_numpy(data[key])
                                  for key in data.files if key.startswith("rule/")})
            rule.eval().requires_grad_(False)
    return graph, rule


def birth_graph(births, controller):
    if controller == "local":
        return native.spawn(births)
    graphs = tuple(fw.spawn(birth, 8) for birth in births)
    return fw.Graph(*(torch.cat([getattr(graph, name) for graph in graphs])
                      for name in ("weights", "bias", "present")))


def fit(path, config, births, task, arm, checkpoints, progress):
    torch.set_num_threads(1)
    born = birth_graph(births, config.controller)
    initial = fw.Graph(nn.Parameter(born.weights), nn.Parameter(born.bias), born.present)
    rule = native.Rule(births) if config.controller == "local" else None
    parameters = (initial.weights, initial.bias, *(rule.parameters() if rule is not None else ()))
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    trace = np.empty((len(births), config.updates, len(METRICS)), dtype=np.float32)
    started = perf_counter()

    def save(step):
        for lane, birth in enumerate(births):
            save_checkpoint(path, task, arm, birth, step, initial, rule, lane)
            filename = path / TRACE.format(task=task, arm=arm, birth=birth)
            write_arrays(filename, values=trace[lane, :step], metrics=METRICS, seconds=perf_counter() - started)

    save(0)
    for step in range(1, config.updates + 1):
        tape = Tape.concatenate(tuple(world_for_update(config, task, birth, step) for birth in births))
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_step(initial, rule, tape, task, arm, config)
        loss.backward()
        norms = clip_birth_gradients(tuple(p for p in parameters if p.grad is not None), config.grad_clip)
        optimizer.step()
        trace[:, step - 1, :-1] = metrics
        trace[:, step - 1, -1] = norms.numpy()
        if step in checkpoints:
            save(step)
        if step == 1 or step in checkpoints or (progress and step % progress == 0):
            recent = trace[:, max(0, step - 100):step, 1:4].mean(1)
            for birth, (pooled, positive, negative) in zip(births, recent, strict=True):
                print(f"birth={birth} task={task} arm={arm} step={step}/{config.updates} "
                      f"positive={pooled:.3f} laws={positive:.3f}/{negative:.3f} "
                      f"seconds={perf_counter() - started:.1f}", flush=True)


def run(path, config, births=BIRTHS, tasks=TASKS, arms=ARMS, *, workers=4, group_size=3, progress=500):
    if (config.updates < 1 or config.pairs < 2 or not 0 <= config.min_wait <= config.max_wait
            or config.lr <= 0 or config.grad_clip <= 0 or config.controller not in CONTROLLERS
            or workers < 1 or group_size < 1 or progress < 0):
        raise ValueError("Invalid fit configuration")
    for selected, allowed in ((births, None), (tasks, TASKS), (arms, ARMS)):
        if not selected or len(set(selected)) != len(selected) or (allowed and set(selected) - set(allowed)):
            raise ValueError("Select distinct supported births, tasks, and arms")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted({step for step in STEPS if step <= config.updates} | {0, config.updates})
    modules = [Path(mod.__file__) for mod in (fw, native, legacy)] + [Path(__file__)]
    plan = dict(config=asdict(config), births=list(births), tasks=list(tasks), arms=list(arms),
                checkpoints=checkpoints, metrics=METRICS, episodes_per_fit=config.updates*config.pairs*2,
                target="Sampled terminal resource, leave-own-pair-out baseline. Sum the log probabilities "
                       "of all sampled actions and edits; the earlier observed taste enters the graph only.",
                sources={p.name: sha256(p.read_bytes()).hexdigest() for p in modules})
    (path / "run.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf8")
    jobs = [(path,config,group,task,arm,checkpoints,progress)
            for group in batched(births,group_size) for task in tasks for arm in arms]
    if workers == 1:
        for job in jobs:
            fit(*job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(fit,*job) for job in jobs]:
                future.result()


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--controller", choices=CONTROLLERS, default="local")
    p.add_argument("--births", type=int, nargs="+", default=BIRTHS)
    p.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    p.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--progress", type=int, default=500)
    a=p.parse_args()
    run(a.out, Config(updates=a.updates,pairs=a.pairs,controller=a.controller),
        a.births,a.tasks,a.arms,workers=a.workers,group_size=a.group_size,progress=a.progress)
