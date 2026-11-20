"""Separate the number of studied items from contact time in a 20-node frozen-source fit."""

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_capacity as capacity, interleave_train as common
from . import local_trace_graph as net, local_trace_train as prior
from .interleave_development import estimate
from .interleave_revisit_train import _gradients, _write_json, _write_raw


ROOT = Path(__file__).resolve().parents[2]
DATA = common.DEFAULT_OUTPUT.parent
OUT = DATA / "wide-capacity"
SOURCE = DATA / "retention-objective"
ARMS = ("wide_any", "wide_recent", "few_time")
TRAIN_DOMAIN = 2201177
BLOCK = 4
LAYOUT = torch.tensor((1., -1., 1., -1.))
LAW_SIGN = torch.tensor((1., 1., -1., -1.))
STAGES = ((2000, (2, 4, 8)), (4000, (2, 4, 8, 16)), (8000, (2, 4, 8, 16, 32)))
EVAL_SEEDS = {"matched": 871831, "disjoint": 9124459, "confirmation": 9124467}
CAPACITIES = (1, 2, 4, 8, 16, 32)
FIT_METRICS = ("resource", "accuracy", "prior_motor_score_loss", "prior_edit_score_loss", "terminal_score_loss",
               "older_hits", "older_trials", "gradient_norm", "float64_recovery")
EVAL_METRICS = ("A", "latest", "A_and_latest", "all", "mean_number_correct", "mean_fraction_correct")


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    worlds: int = 4
    world_seed: int = 415397
    max_wait: int = 8
    lr: float = .003
    grad_clip: float = 2.
    checkpoint_every: int = 500


@dataclass(frozen=True)
class Tape:
    odor: torch.Tensor
    law: torch.Tensor
    repeat_visits: torch.Tensor
    queries: torch.Tensor
    orientation: torch.Tensor
    waits: torch.Tensor
    draws: torch.Tensor

    @property
    def contacts(self):
        return self.odor.shape[2]


def contact_count(config, step):
    for end, choices in STAGES:
        if step <= end:
            rng = np.random.default_rng([TRAIN_DOMAIN, config.world_seed, step, 0])
            return int(rng.choice(choices))
    raise ValueError("The prescribed contact schedule ends at step 8000")


def _sign(rng, shape):
    return (rng.integers(2, size=shape, dtype=np.int8) * 2 - 1).astype(np.float32)


def make_tape(config, births, step, *, contacts=None):
    """Draw once per world; all arms share geometries, timing, laws and uniforms."""
    size = contact_count(config, step) if contacts is None else contacts
    worlds = config.worlds
    few = min(size, 4)
    rows = []
    for birth in births:
        rng = np.random.default_rng([TRAIN_DOMAIN, config.world_seed, step, 1, birth])
        odors = rng.normal(size=(worlds, size, 3)).astype(np.float32)
        odors *= 1.35 / np.linalg.norm(odors, axis=-1, keepdims=True)
        law = _sign(rng, (worlds, size))
        repeats = np.c_[np.tile(np.arange(few), (worlds, 1)), rng.integers(few, size=(worlds, size - few))]
        u = rng.random(worlds)
        queries = np.stack(((u * size).astype(np.int64), size - 2 + (u * 2).astype(np.int64),
                            (u * few).astype(np.int64)), axis=1)
        rows.append((odors, law, repeats, queries, _sign(rng, (worlds, size + 1)),
                     rng.integers(config.max_wait + 1, size=(worlds, size)),
                     rng.random((worlds, size + 1, net.EDIT_DRAWS + 1), dtype=np.float32)))
    return Tape(*(torch.from_numpy(np.stack(values)) for values in zip(*rows, strict=True)))


def physical(tape, arm):
    roots, worlds, _ = tape.law.shape
    query = tape.queries[..., ARMS.index(arm)]
    visits = tape.repeat_visits if arm == "few_time" else torch.arange(tape.contacts).expand_as(tape.repeat_visits)
    events = torch.cat((visits, query[..., None]), dim=2)
    vector = tape.odor.gather(2, events[..., None].expand(-1, -1, -1, 3))
    right = vector * tape.orientation[..., None]
    factor = tape.law.gather(2, events) * tape.orientation

    def expand(tensor):
        return tensor[:, :, None].expand(-1, -1, 4, *tensor.shape[2:])

    def by_contact(tensor):
        return tensor.flatten(0, 2).transpose(0, 1).contiguous()

    right = expand(right) * LAYOUT.view(1, 1, 4, 1, 1)
    factor = expand(factor) * (LAYOUT * LAW_SIGN).view(1, 1, 4, 1)
    old = ~(visits[..., -2:] == query[..., None]).any(dim=-1)
    return (by_contact(right), by_contact(factor), by_contact(expand(tape.draws)),
            by_contact(expand(tape.waits)), old)


def trajectory(graph, rule, tape, arm, max_wait, *, truncate=True):
    menu, truth, draws, wait, old = physical(tape, arm)
    worlds = tape.law.shape[1]
    view = rule.expand(4 * worlds)
    state = net.start(graph, 4 * worlds)
    motor = edit = 0.
    for index in range(tape.contacts):
        if truncate and index and index % BLOCK == 0:
            state = state.detach()
        right, u = menu[index], draws[index]
        action = net.act(view, state, -right, right, u[:, 0])
        earned = (action.side * truth[index]).detach()
        observed = net.observe(view, action, -right, right, earned)
        committed = net.commit(view, observed, u[:, 1:])
        state = net.wait(view, committed.state, wait[index], max_ticks=max_wait)
        motor = motor + action.motor_log_prob
        edit = edit + committed.edit_log_prob
    right = menu[-1]
    terminal = net.act(view, state, -right, right, draws[-1, :, 0])
    resource = (terminal.side * truth[-1]).detach()
    shape = (*tape.law.shape[:2], 4)
    return tuple(value.reshape(shape) for value in (resource, motor, edit, terminal.motor_log_prob)), old


def objective(graph, rule, tape, arm, max_wait):
    (earned, motor, edit, terminal), older = trajectory(graph, rule, tape, arm, max_wait)
    world = earned.mean(2, keepdim=True)
    baseline = (world.sum(1, keepdim=True) - world) / (world.shape[1] - 1)
    advantage = earned - baseline
    scores = [-(value * advantage).mean((1, 2)) for value in (motor, edit, terminal)]
    with torch.no_grad():
        success = earned > 0
        mask = older[:, :, None].expand_as(success)
        health = torch.stack((earned.mean((1, 2)), success.float().mean((1, 2)), *scores,
                              (mask & success).sum((1, 2)), mask.sum((1, 2))), dim=1)
    return sum(scores).sum(), health


def _step(graph, rule, optimizer, tape, arm, config):
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss, values = objective(graph, rule, tape, arm, config.max_wait)
    norm = None
    if loss.isfinite():
        loss.backward()
        norm = _gradients(parameters, config.grad_clip)
    recovered = norm is None
    if recovered:
        precise_graph = net.Graph(nn.Parameter(graph.weights.detach().double()),
                                  nn.Parameter(graph.bias.detach().double()), graph.present)
        precise_rule = deepcopy(rule).double()
        loss, values = objective(precise_graph, precise_rule, tape, arm, config.max_wait)
        loss.backward()
        precise = (precise_graph.weights, precise_graph.bias, *precise_rule.parameters())
        norm = _gradients(precise, config.grad_clip)
        if not loss.isfinite() or norm is None:
            raise FloatingPointError("Non-finite gradient after retrying identical worlds in float64")
        for target, source in zip(parameters, precise, strict=True):
            target.grad = None if source.grad is None else source.grad.to(target.dtype)
    optimizer.step()
    return torch.cat((values.double(), norm[:, None], torch.full_like(norm[:, None], float(recovered))), 1)


def initialize(source, births, parent_step):
    graphs, rules = zip(*(prior.load_checkpoint(source, "odor", "uniform", birth, parent_step) for birth in births),
                        strict=True)
    w, b = (nn.Parameter(torch.cat([getattr(g, name) for g in graphs])) for name in ("weights", "bias"))
    graph = net.Graph(w, b, torch.cat([g.present for g in graphs]))
    rule = net.Rule(births)
    rule.load_state_dict({name: torch.cat([r.state_dict()[name] for r in rules]) for name in rule.state_dict()})
    return graph, rule


def make_plan(path, config, births, source, parent_step):
    plan = dict(config=asdict(config), births=list(births), arms=ARMS, source=str(Path(source).resolve()),
                parent_step=parent_step, checkpoints={str(birth): common._source_digest(
                    prior.checkpoint_path(source, "odor", "uniform", birth, parent_step)) for birth in births},
                code={Path(f).name: common._source_digest(f) for f in
                      (__file__, net.__file__, prior.__file__, capacity.__file__)}, schedule=STAGES,
                domain=TRAIN_DOMAIN, train_metrics=FIT_METRICS, eval_metrics=EVAL_METRICS, eval_seeds=EVAL_SEEDS,
                eval_worlds=192, eval_capacities=CAPACITIES, training_crossing="Four lanes per independent world: "
                "private sign and left/right mirror crossed, all physical draws shared. World is the baseline unit.",
                prefix="Wide arms have the identical exogenous full prefix, each item encountered once. Few-time "
                       "uses the same first four or fewer items as wide, then repeats only those IDs. All arms use "
                       "exactly the same number of full contacts and neutral gaps. A common query uniform selects "
                       "any wide item, only the last two wide items, or any seen few-time item.",
                gradient="Earned terminal unit resource supplies REINFORCE to every sampled study motor, edit "
                         "proposal, and terminal motor. Only pathwise derivatives through hidden, traces and graph "
                         "are truncated between four-study-contact blocks. Terminal read stays in the final block.")
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    target = path / "run.json"
    if target.exists() and target.read_text(encoding="utf8") != payload:
        raise ValueError(f"Immutable training plan disagrees at {target}")
    if not target.exists():
        common._atomic_save(target, lambda file: file.write_text(payload, encoding="utf8"))
    return sha256(payload.encode("utf8")).hexdigest()


def fit(path, config, births, source, parent_step, arm, stop, progress):
    if (not 1 <= stop <= config.updates <= 8000 or config.worlds < 2 or len(set(births)) != len(births)
            or config.checkpoint_every < 1 or config.max_wait < 0 or not births):
        raise ValueError("Invalid fit limits")
    torch.set_num_threads(1)
    digest = make_plan(path, config, births, source, parent_step)
    graph, rule = initialize(source, births, parent_step)
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=config.lr)
    progress_metrics = torch.empty((len(births), config.updates, len(FIT_METRICS)), dtype=torch.float64)
    size = torch.empty(config.updates, dtype=torch.uint8)
    files = sorted((path / arm).glob("step-*.pt"))
    current, elapsed = 0, 0.
    if files:
        state = torch.load(files[-1], weights_only=True, map_location="cpu")
        current, elapsed = state["step"], state["seconds"]
        if (state["births"], state["arm"], state["plan_hash"]) != (list(births), arm, digest):
            raise ValueError("Checkpoint and fit plan disagree")
        if common.checkpoint_path(path, arm, current) != files[-1]:
            raise ValueError("Checkpoint name and step disagree")
        with torch.no_grad():
            for name in capacity.GRAPH_FIELDS:
                getattr(graph, name).copy_(state["graph"][name])
        rule.load_state_dict(state["rule"])
        optimizer.load_state_dict(state["optimizer"])
        progress_metrics[:, :current] = state["metrics"]
        size[:current] = state["contacts"]
    if current > stop:
        raise ValueError("Requested stop precedes the latest checkpoint")
    started = perf_counter()

    def save(step):
        state = dict(arm=arm, births=list(births), step=step, plan_hash=digest,
                     graph={name: getattr(graph, name).detach() for name in capacity.GRAPH_FIELDS},
                     rule=rule.state_dict(), optimizer=optimizer.state_dict(),
                     metrics=progress_metrics[:, :step].clone(), contacts=size[:step].clone(),
                     world_seed=config.world_seed, seconds=elapsed + perf_counter() - started)
        common._atomic_save(common.checkpoint_path(path, arm, step), lambda file: torch.save(state, file))

    if not files:
        save(0)
    print(f"{arm} start={current} stop={stop} worlds={config.worlds}", flush=True)
    for index in range(current + 1, stop + 1):
        tape = make_tape(config, births, index)
        progress_metrics[:, index - 1] = _step(graph, rule, optimizer, tape, arm, config)
        size[index - 1] = tape.contacts
        if index == stop or index % config.checkpoint_every == 0:
            save(index)
        if index in (1, stop) or progress and index % progress == 0:
            recent = progress_metrics[:, max(0, index - 100):index]
            print(f"{arm} step={index}/{stop} acc={recent[:, :, 1].mean():.3f} "
                  f"fallbacks={int(recent[..., -1].sum())} secs={elapsed + perf_counter() - started:.1f}", flush=True)


def _scores(raw):
    by_branch = {}
    for index, branch in enumerate(capacity.BRANCHES):
        cells = []
        for cut, k in enumerate(raw["capacities"]):
            target = raw["right_value"][:, :k + 1, -1].T
            success = raw["side"][index, cut, :k + 1] == target
            a, latest = success[0], success[-1]
            score = np.stack((a, latest, a & latest, success.all(axis=0),
                              success.sum(axis=0), success.mean(axis=0)), axis=-1)
            cells.append(score.reshape(-1, 4, len(EVAL_METRICS)).mean(axis=1))
        by_branch[branch] = np.stack(cells)
    return by_branch


def _verify_standard(tape, source, birth):
    path = source / "evaluation/matched/008000" / f"capacity-uniform-{birth}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as raw:
        for field in fields(tape):
            name = field.name
            if name not in raw.files:
                continue
            old = raw[name]
            new = getattr(tape, name).numpy()[:, :old.shape[1]]
            if not np.array_equal(old, new):
                raise ValueError(f"Standard tape differs in {name} at {path}")
    return common._source_digest(path)


def evaluate(path, births, source, parent_step, stages, splits, worlds, bootstrap, arms):
    torch.set_num_threads(1)
    for split in splits:
        seed = EVAL_SEEDS[split]
        tape = capacity.make_tape("odor", seed, worlds, max(CAPACITIES))
        standard_hash = _verify_standard(tape, Path(source), births[0]) if split == "matched" and worlds == 192 else None
        directory = path / f"eval-{split}"
        rng = np.random.default_rng([TRAIN_DOMAIN, seed, 7])
        boot_b = rng.integers(len(births), size=(bootstrap, len(births)))
        boot_w = rng.integers(worlds, size=(bootstrap, worlds))
        values, hashes, checkpoints = {}, {}, {}
        models = [("parent", prior.load_checkpoint, parent_step)] + [
            (arm, common.load_checkpoint, step) for step in stages for arm in arms]
        for arm, loader, step in models:
            label = arm if arm == "parent" else f"{arm}/{step}"
            rows = []
            for birth in births:
                if arm == "parent":
                    graph, rule = loader(source, "odor", "uniform", birth, step)
                    checkpoint = prior.checkpoint_path(source, "odor", "uniform", birth, step)
                else:
                    graph, rule = loader(path, arm, birth, step)
                    checkpoint = common.checkpoint_path(path, arm, step)
                raw = capacity.evaluate(graph, rule, tape, CAPACITIES, snapshot_worlds=0)
                filename = directory / label / f"birth-{birth}.npz"
                _write_raw(filename, **raw)
                hashes[filename.relative_to(directory).as_posix()] = common._source_digest(filename)
                checkpoints[f"{label}/{birth}"] = common._source_digest(checkpoint)
                rows.append(_scores(raw))
            values[label] = {branch: np.stack([row[branch] for row in rows]) for branch in capacity.BRANCHES}
            print(f"{split} {label} evaluated", flush=True)

        def summarize(array):
            return {str(k): {metric: estimate(array[:, i, :, j], boot_b, boot_w)
                             for j, metric in enumerate(EVAL_METRICS)} for i, k in enumerate(CAPACITIES)}

        results = {label: {branch: summarize(row) for branch, row in by_branch.items()}
                   for label, by_branch in values.items()}
        contrasts = {}
        for label, branches in values.items():
            for branch in capacity.BRANCHES[1:]:
                contrasts[f"{label}/write_minus_{branch}"] = summarize(branches["write"] - branches[branch])
            if label != "parent":
                contrasts[f"{label}/minus_parent"] = summarize(branches["write"] - values["parent"]["write"])
        for step in stages:
            for a, b in (("wide_any", "wide_recent"), ("wide_any", "few_time"), ("wide_recent", "few_time")):
                if {a, b} <= set(arms):
                    contrasts[f"{step}/{a}_minus_{b}"] = summarize(
                        values[f"{a}/{step}"]["write"] - values[f"{b}/{step}"]["write"])
        _write_json(directory / "summary.json", dict(split=split, seed=seed, births=list(births), worlds=worlds,
                                                     age_unit="intervening odors after A", capacities=CAPACITIES,
                                                     bootstrap=bootstrap, archived_standard_tape_sha256=standard_hash,
                                                     exact_prefix=standard_hash is not None, raw_sha256=hashes,
                                                     checkpoint_sha256=checkpoints, results=results, contrasts=contrasts))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent-step", type=int, default=8000)
    parser.add_argument("--births", nargs="+", type=int, default=prior.BIRTHS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--stages", nargs="+", type=int, default=(8000,))
    parser.add_argument("--splits", nargs="+", choices=tuple(EVAL_SEEDS), default=("matched", "disjoint"))
    parser.add_argument("--eval-worlds", type=int, default=192)
    parser.add_argument("--bootstrap", type=int, default=2500)
    for field in fields(Config):
        value = getattr(Config(), field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    if args.command == "fit":
        if args.arm is None:
            parser.error("Each fit process requires one --arm")
        cfg = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
        fit(args.out, cfg, args.births, args.source, args.parent_step, args.arm, args.stop or cfg.updates, args.progress)
    else:
        evaluate(args.out, args.births, args.source, args.parent_step, args.stages, args.splits,
                 args.eval_worlds, args.bootstrap, args.arms)


if __name__ == "__main__":
    main()
