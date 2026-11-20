"""Fitness of jointly completing three physical choices before any terminal taste."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_eval as physical, interleave_train as matched
from . import local_trace_graph as local, self_param_writer as writer, world
from .edit_receipt_train import clip_birth_gradients


ARMS = ("conjunction", "mean")
BIRTHS = (731, 732, 733, 734, 735, 736)
HOME = physical.RAW / "terminal-quest"
EVIDENCE = physical.EVIDENCE / "terminal-quest"
PARENT = physical.RAW / "joint-continuation/full-direct"
TRAIN_DOMAIN, DEV_DOMAIN, VALIDATION_DOMAIN, PHYSICAL_DOMAIN, PHYSICAL_DONOR_DOMAIN = range(0x7EA210, 0x7EA215)
SUPPORT, CONTACTS = 6, 9
A, B, C, MARK_PLUS, MARK_MINUS, MARK_CHOICE = range(6)
LAWS = np.asarray(list(product((-1, 1), repeat=3)), np.int8)
ORDERS = np.asarray(list(permutations(range(SUPPORT))), np.int8)
METRICS = ("joint", "conflict_joint", "agree_joint", "A", "B", "mark", "conflict_A", "conflict_B",
           "conflict_mark", "advantage", "gradient_norm")


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    worlds: int = 4
    lr: float = .0015
    clip: float = 2.
    save_every: int = 500
    seed: int = 722513
    max_wait: int = 8


@dataclass(frozen=True)
class Tape:
    # Shape starts with birth, independent world, private law, physical contact.
    left: torch.Tensor
    right: torch.Tensor
    resources: torch.Tensor
    draws: torch.Tensor
    waits: torch.Tensor
    contact_tick: torch.Tensor
    order: torch.Tensor
    quest: torch.Tensor
    law: torch.Tensor

    @property
    def shape(self):
        return self.left.shape[:3]


@dataclass(frozen=True)
class Experience:
    resource: torch.Tensor
    taste: torch.Tensor
    side: torch.Tensor
    tick: torch.Tensor
    motor_score: torch.Tensor
    edit_score: torch.Tensor


def travel_ticks(distance, wind):
    """First actual endpoint touch for each of two sustained native pulse actions."""
    direction = np.array([-1, 1], dtype=np.float32)
    ticks = np.full(distance.shape, -1, np.int8)
    position = np.zeros_like(distance)
    velocity = np.zeros_like(distance)
    for tick in range(world.MOVEMENT_TICKS):
        velocity = world.VELOCITY_DECAY * velocity + world.PULSE_ACCELERATION * direction + wind[..., tick, None]
        position = position + velocity
        landed = (position * direction >= distance) & (ticks < 0)
        ticks[landed] = tick
    if (ticks < 0).any() or np.any(np.abs(wind) >= world.PULSE_ACCELERATION):
        raise AssertionError("The sampled arena must allow both sampled physical directions to land")
    return ticks


def _unit(rng, n):
    vector = rng.normal(size=(n, 3)).astype(np.float32)
    return 1.35 * vector / np.linalg.norm(vector, axis=1, keepdims=True)


def make_tape(births, config, update, *, domain=TRAIN_DOMAIN, novel=False, shared=False):
    if config.worlds < 2 or len(births) == 0:
        raise ValueError("Use at least two independent worlds per birth")
    if novel and domain == TRAIN_DOMAIN:
        raise ValueError("Holdout support layouts have their own evaluation domains")
    shape = (config.worlds, len(LAWS), CONTACTS)
    counts = config.worlds
    def one(birth):
        rng = [np.random.default_rng([config.seed, domain, 0 if shared else int(birth), update, j]) for j in range(6)]
        pattern, latent, ears, motor, edit, movement = rng
        repeat = np.where(pattern.random((counts, 2)) < .5,
                          pattern.choice((A, B, MARK_PLUS, MARK_MINUS), size=(counts, 2)), C)
        supports = np.concatenate((np.broadcast_to([A, B, MARK_PLUS, MARK_MINUS], (counts, 4)), repeat), axis=1)
        eligible = (supports[:, ORDERS[:, :2]] >= MARK_PLUS).all(axis=2) == novel
        support_order = np.stack([ORDERS[mask][pattern.integers(mask.sum())] for mask in eligible])
        supports = np.take_along_axis(supports, support_order.astype(int), axis=1)
        terminal = np.asarray([pattern.permutation([A, B, MARK_CHOICE]) for _ in range(counts)])
        order = np.concatenate((supports, terminal), axis=1)
        laws = np.broadcast_to(LAWS[None], (counts, len(LAWS), 3))
        side_sgn = ears.choice((-1, 1), size=(counts, CONTACTS)).astype(np.float32)
        scent, distractor = _unit(ears, counts), _unit(ears, counts)
        cue = np.stack([physical.CUES[ears.permutation(3)] for _ in range(counts)])
        ctx = np.minimum(order, C)
        odorous = order <= C
        background = cue[np.arange(counts)[:, None], ctx] * odorous[..., None]
        direction = np.broadcast_to(scent[:, None], (counts, CONTACTS, 3)).copy()
        direction = np.where((order == C)[..., None], distractor[:, None], direction)
        direction = np.where(odorous[..., None], direction, physical.MARK_AXIS)
        mark_demo = (order == MARK_PLUS) | (order == MARK_MINUS)
        side_sgn = np.where(mark_demo, np.where(order == MARK_PLUS, 1., -1.), side_sgn).astype(np.float32)
        right = background + side_sgn[..., None] * direction
        left = np.where(mark_demo[..., None], right, background - side_sgn[..., None] * direction)
        task_law = np.where(odorous[:, None], np.where(ctx[:, None] == C,
                            latent.choice((-1, 1), size=(counts, 1, 1)),
                            np.take_along_axis(laws, np.broadcast_to(np.minimum(ctx, B)[:, None], shape), axis=2)),
                            laws[..., 2, None])
        right_value = (task_law * side_sgn[:, None]).astype(np.float32)
        left_value = np.where(mark_demo[:, None], right_value, -right_value)
        value = np.stack((left_value, right_value), axis=-1)

        draws = np.concatenate((motor.random((counts, CONTACTS, 1), dtype=np.float32),
                                edit.random((counts, CONTACTS, local.EDIT_DRAWS), dtype=np.float32)), axis=-1)
        wait = movement.integers(config.max_wait + 1, size=(counts, CONTACTS), dtype=np.int64)
        jitter, gust = (.34, .11) if novel else (.13, .07)
        distance = movement.uniform(1 - jitter, 1 + jitter, size=(counts, CONTACTS, 2)).astype(np.float32)
        wind = np.zeros((counts, CONTACTS, world.MOVEMENT_TICKS), np.float32)
        wind[np.arange(counts)[:, None], np.arange(CONTACTS), movement.integers(world.MOVEMENT_TICKS, size=order.shape)] = (
            movement.uniform(-gust, gust, size=order.shape))
        ticks = travel_ticks(distance, wind)

        def across_laws(values):
            return np.repeat(values[:, None], len(LAWS), axis=1)

        return (across_laws(left), across_laws(right), value, across_laws(draws), across_laws(wait),
                across_laws(ticks), across_laws(order), across_laws(terminal), laws)

    per_birth = [one(birth) for birth in births] if not shared else [one(0)] * len(births)
    return Tape(*(torch.from_numpy(np.stack(group)) for group in zip(*per_birth, strict=True)))


def play(graph, rule, tape, max_wait, *, intervention="active"):
    """Keep activity, traces and writes between all actual visits, even when taste is silent."""
    lives = tape.shape[1] * tape.shape[2]
    state = local.start(graph, lives)
    expanded = rule.expand(lives)
    left, right, available, draws, waits, landing = (
        value.flatten(0, 2).transpose(0, 1) for value in
        (tape.left, tape.right, tape.resources, tape.draws, tape.waits, tape.contact_tick)
    )
    earned, shown, choices, ticks, motor_score, edit_score = [], [], [], [], [], []
    for visit, (a, b, u) in enumerate(zip(left, right, draws, strict=True)):
        if visit and max_wait:
            state = local.wait(expanded, state, waits[visit], max_ticks=max_wait)
        action = local.act(expanded, state, a, b, u[:, 0])
        side = (action.side > 0).long()[:, None]
        contact_tick = landing[visit].gather(1, side)[:, 0]
        reward = available[visit].gather(1, side)[:, 0]
        taste = reward if visit < SUPPORT else torch.zeros_like(reward)
        contact = local.observe(expanded, action, a, b, taste)
        committed = local.commit(expanded, contact, u[:, 1:], intervention=intervention)
        state = committed.state
        for record, value in zip((earned, shown, choices, ticks, motor_score, edit_score),
                                 (reward, taste, action.side, contact_tick, action.motor_log_prob,
                                  committed.edit_log_prob), strict=True):
            record.append(value)
    return Experience(*(torch.stack(v, dim=-1).reshape(*tape.shape, CONTACTS)
                        for v in (earned, shown, choices, ticks, motor_score, edit_score)))


def achieved(experience, tape):
    terminal = experience.resource[..., SUPPORT:] > 0
    # These addresses are used in the scorer, after all actions have occurred.
    a = (terminal & (tape.quest == A)).any(dim=-1)
    b = (terminal & (tape.quest == B)).any(dim=-1)
    mark = (terminal & (tape.quest == MARK_CHOICE)).any(dim=-1)
    return a, b, mark


def score(experience, tape, arm):
    outcomes = experience.resource[..., SUPPORT:].detach()
    if arm == "conjunction":
        reward = (outcomes > 0).all(dim=-1).float()
    elif arm == "mean":
        reward = outcomes.mean(dim=-1)
    else:
        raise ValueError(f"Unknown objective {arm!r}")
    pair_mean = reward.mean(dim=2, keepdim=True)
    centered = reward - (pair_mean.sum(dim=1, keepdim=True) - pair_mean) / (reward.shape[1] - 1)
    logits = experience.motor_score.sum(dim=-1) + experience.edit_score[..., :-1].sum(dim=-1)
    loss = -(logits * centered).mean((1, 2))
    with torch.no_grad():
        a, b, mark = achieved(experience, tape)
        joint = a & b & mark
        conflict = tape.law[..., 0] != tape.law[..., 1]

        def average(value, mask):
            return (value & mask).sum((1, 2)) / mask.sum((1, 2))

        values = torch.stack((joint.float().mean((1, 2)), average(joint, conflict), average(joint, ~conflict),
                              *(item.float().mean((1, 2)) for item in (a, b, mark)),
                              *(average(item, conflict) for item in (a, b, mark)),
                              centered.abs().mean((1, 2))), dim=1)
    return loss.sum(), values


def _parameters(parent, births):
    parents = [matched.load_checkpoint(parent, "local_active", birth, 8000) for birth in births]
    pieces = {field.name: torch.cat([getattr(graph, field.name) for graph, _ in parents])
              for field in fields(writer.Graph)}
    graph = writer.Graph(nn.Parameter(pieces["weights"]), nn.Parameter(pieces["bias"]), pieces["present"])
    rule = local.Rule(births)
    rules = [one.state_dict() for _, one in parents]
    rule.load_state_dict({key: torch.cat([state[key] for state in rules]) for key in rules[0]})
    return graph, rule


def _fit(path, parent, births, config, arm, plan_hash):
    torch.set_num_threads(1)
    graph, rule = _parameters(parent, births)
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    history = torch.empty((len(births), config.updates, len(METRICS)), dtype=torch.float32)
    prior = sorted((path / arm).glob("step-*.pt"))
    step, earlier = 0, 0.
    if prior:
        state = torch.load(prior[-1], map_location="cpu", weights_only=True)
        if (state["arm"], state["births"], state["plan_hash"]) != (arm, list(births), plan_hash):
            raise ValueError(f"Saved checkpoint differs from the declared {arm} run")
        step, earlier = state["step"], state["seconds"]
        with torch.no_grad():
            for name, value in state["graph"].items():
                getattr(graph, name).copy_(value)
        rule.load_state_dict(state["rule"])
        optimizer.load_state_dict(state["optimizer"])
        history[:, :step] = state["metrics"]
    started = perf_counter()

    def save(index):
        payload = dict(arm=arm, step=index, births=list(births), plan_hash=plan_hash,
                       graph={field.name: getattr(graph, field.name).detach() for field in fields(writer.Graph)},
                       rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=history[:, :index].clone(),
                       seconds=earlier + perf_counter() - started)
        matched._atomic_save(matched.checkpoint_path(path, arm, index), lambda p: torch.save(payload, p))

    if not prior:
        save(0)
    print(f"quest {arm} start {step} / {config.updates}", flush=True)
    for index in range(step + 1, config.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        tape = make_tape(births, config, index)
        rolled = play(graph, rule, tape, config.max_wait)
        loss, measures = score(rolled, tape, arm)
        loss.backward()
        norm = clip_birth_gradients(parameters, config.clip)
        if not torch.isfinite(norm).all():
            raise FloatingPointError(f"{arm} invalid optimizer step {index}")
        optimizer.step()
        history[:, index - 1, :-1] = measures
        history[:, index - 1, -1] = norm
        del rolled, loss
        if index % config.save_every == 0 or index == config.updates:
            save(index)
        if index == 1 or index % 100 == 0 or index == config.updates:
            vals = history[:, max(0, index - 100):index, :6].mean((0, 1)).tolist()
            print(f"quest {arm} step {index} joint={vals[0]:.3f} conflict={vals[1]:.3f} "
                  f"agree={vals[2]:.3f} A={vals[3]:.3f} B={vals[4]:.3f} mark={vals[5]:.3f} "
                  f"seconds={earlier + perf_counter() - started:.1f}", flush=True)


def run(path, parent, births, config, *, workers=2):
    if (min(config.updates, config.save_every, workers) < 1 or config.worlds < 2
            or config.lr <= 0 or config.clip <= 0 or config.max_wait < 0 or len(set(births)) != len(births)):
        raise ValueError("Invalid quest fit configuration")
    source = matched.checkpoint_path(parent, "local_active", 8000)
    manifest = dict(config=asdict(config), arms=ARMS, births=tuple(births), parent=str(source),
                    parent_sha256=physical._file_digest(source), domains=[TRAIN_DOMAIN, DEV_DOMAIN,
                        VALIDATION_DOMAIN, PHYSICAL_DOMAIN, PHYSICAL_DONOR_DOMAIN],
                    support=SUPPORT, query_contacts=3,
                    holdout_layout="Both first support visits are marker demonstrations. Training excludes this observable order.",
                    source_sha256={Path(name).name: physical._file_digest(Path(name))
                                   for name in (__file__, local.__file__, matched.__file__, world.__file__)},
                    objectives={"conjunction": "1 if all three earned resources are positive, 0 otherwise",
                                "mean": "arithmetic mean of the three earned {-1,+1} resources"},
                    protocol="Interleaved signed support includes one A visit, one B visit, both mark demos and two "
                    "independently drawn repeats or visits to the independent C menu. A/B/mark queries are independently "
                    "permuted and all tastes are hidden. The physical body lands via sustained pulses. Native contact "
                    "activity uses common neutral gaps and its usual contact clock. At every landing "
                    "the graph receives side, active trace processing and an endogenous graph proposal. Support tastes "
                    "are exposed. Only terminal resources contribute to outer credit, with the same leave-world-out "
                    "baseline. The final edit follows the monitored payoff and has no score credit. Contact and update "
                    "draws, clocks, policies at initialization and optimizer budgets match across the objective arms.")
    text = json.dumps(manifest, sort_keys=True)
    target = path / "run.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != json.loads(text):
        raise ValueError("Output already holds another frozen quest protocol")
    physical._atomic_json(target, manifest)
    digest = sha256(text.encode()).hexdigest()
    if workers == 1:
        for arm in ARMS:
            _fit(path, parent, tuple(births), config, arm, digest)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(ARMS))) as pool:
            jobs = [pool.submit(_fit, path, parent, tuple(births), config, arm, digest) for arm in ARMS]
            for job in jobs:
                job.result()


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HOME)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--updates", default=Config.updates, type=int)
    parser.add_argument("--worlds", default=Config.worlds, type=int)
    parser.add_argument("--save-every", default=Config.save_every, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--births", nargs="+", type=int, default=BIRTHS)
    args = parser.parse_args(argv)
    config = replace(Config(), updates=args.updates, worlds=args.worlds, save_every=args.save_every)
    run(args.out, args.parent, args.births, config, workers=args.workers)


if __name__ == "__main__":
    main()
