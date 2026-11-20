"""Continue contextual and multi-object specialists on the same short-to-long joint lives."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_train as train


BASE = train.DEFAULT_OUTPUT.parent
OUT = BASE / "joint-continuation"
DOMAIN = 2201161
ARM = "local_active"
SOURCES = ("full-direct", "uniform")
MODES = ("joint4", "mark3", "context3", "full14")
STAGE = ((0., .20), (.125, .30), (.25, .55), (.50, 1.))
SHORT_PROPORTION = (.60, .25, .15)
EVENT_FIELDS = ("left", "right", "right_factor", "fixed_factor", "donor_right_factor",
                "donor_fixed_factor", "context", "probe")
MIXED_ORDERS = torch.tensor(((0, 2, 1, 3), (0, 2, 3, 1), (2, 0, 1, 3), (2, 0, 3, 1)))


def mode_for_step(config, step):
    progress, full = np.array(STAGE).T
    full_probability = np.interp(step / config.updates, progress, full)
    draw = np.random.default_rng([DOMAIN, config.world_seed, step]).random()
    thresholds = np.cumsum(SHORT_PROPORTION) * (1 - full_probability)
    return MODES[int(np.searchsorted(thresholds, draw, side="right"))]


def _first(mask):
    return mask.to(torch.int8).argmax(dim=-1)


def _short_indices(tape, mode, rng):
    context, probe = (value[:, :, 0] for value in (tape.context, tape.probe))
    mark_probe = _first(probe == 3)
    mark_demo = probe.eq(-1) & context.eq(train.MARK_CONTEXT)
    if mode == "joint4":
        focus = torch.from_numpy(rng.integers(3, size=context.shape[:2]))[..., None]
        events = torch.stack((_first(context == focus), _first(probe == focus), _first(mark_demo),
                              mark_probe), dim=-1)
        order = MIXED_ORDERS[rng.integers(len(MIXED_ORDERS), size=context.shape[:2])]
        return events.gather(-1, order)
    if mode == "context3":
        order = torch.stack((_first(context == 0), _first(context == 1)), dim=-1).sort(dim=-1).values
        focus = torch.from_numpy(rng.integers(2, size=context.shape[:2]))[..., None]
        return torch.cat((order, _first(probe == focus)[..., None]), dim=-1)
    if mode != "mark3":
        raise ValueError(f"Unknown short-life mode {mode!r}")
    sign = tape.left[:, :, 0, :, 0] > 0
    demos = torch.stack((_first(mark_demo & sign), _first(mark_demo & ~sign)), dim=-1).sort(dim=-1).values
    return torch.cat((demos, mark_probe[..., None]), dim=-1)


def short_life(tape, mode, rng):
    """Take valid study/return subhistories, with positional clock and motor draws."""
    index = _short_indices(tape, mode, rng)[..., None, :]
    count = index.shape[-1]

    def select(value):
        expanded = index[..., None] if value.ndim == 5 else index
        return value.gather(3, expanded.expand(*value.shape[:3], count, *value.shape[4:]))

    events = {name: select(getattr(tape, name)) for name in EVENT_FIELDS}
    scored = events["probe"] >= 0
    objective = scored.float() / scored.sum(-1, keepdim=True)
    return replace(tape, **events, objective_weight=objective,
                   draws=tape.draws[..., :count, :], waits=tape.waits[..., :count])


def world(births, config, step):
    tape = train.make_tape(births, config, step)
    mode = mode_for_step(config, step)
    if mode != "full14":
        tape = short_life(tape, mode, np.random.default_rng([DOMAIN, config.world_seed, step, 1]))
    return tape, mode


def _short_rollout(graph, rule, tape, max_wait, *, intervention="active"):
    """Reuse the native controller and keep its state through every short-life contact."""
    count = tape.left.shape[3]
    lives = tape.batch_shape[1] * tape.batch_shape[2]
    controller = train.previous.LocalController(graph, rule, lives)
    left, right, draws, wait, earned_side, earned_fixed = (
        value.flatten(0, 2).transpose(0, 1) for value in
        (tape.left, tape.right, tape.draws, tape.waits, tape.right_factor, tape.fixed_factor)
    )
    resources, motor, edits, sides = [], [], [], []
    numeric = torch.zeros_like(controller.state.graph.bias[:, 0])
    born, retired = torch.zeros_like(numeric), torch.zeros_like(numeric)
    for step, (a, b, u) in enumerate(zip(left, right, draws, strict=True)):
        if max_wait and step:
            controller.wait(wait[step], max_wait)
        action = controller.act(a, b, u)
        reward = earned_side[step] * action.side + earned_fixed[step]
        before = controller.state.graph
        if intervention == "active":
            edit_score = controller.contact(action, a, b, reward, u)
        else:
            seen = train.local.observe(controller.rule, action, a, b, reward)
            committed = train.local.commit(controller.rule, seen, u[:, 1:5], intervention=intervention)
            controller.state, edit_score = committed.state, committed.edit_log_prob
        edits.append(edit_score)
        resources.append(reward)
        motor.append(action.motor_log_prob)
        sides.append(action.side)
        with torch.no_grad():
            after = controller.state.graph
            numeric += ((before.weights.ne(after.weights) & (before.present & after.present)).flatten(1).any(1)
                        | before.bias.ne(after.bias).any(1))
            born += (~before.present & after.present).sum((1, 2))
            retired += (before.present & ~after.present).sum((1, 2))

    def stack(values):
        return torch.stack(values, dim=1).reshape(*tape.batch_shape, count)

    received = stack(resources)
    return train.Rollout(received, received, stack(motor), stack(edits), stack(sides),
                         *(value.reshape(tape.batch_shape) for value in (numeric, born, retired)), controller.state)


def training_step(graph, rule, tape, mode, config):
    rolled = (train.rollout(graph, rule, tape, ARM, config.max_wait) if mode == "full14"
              else _short_rollout(graph, rule, tape, config.max_wait))
    loss, metrics = train.score_and_measure(rolled, tape)
    if mode in ("mark3", "context3"):
        correct, empty = (3, 2) if mode == "mark3" else (2, 3)
        with torch.no_grad():
            metrics[:, correct] = metrics[:, 1]
            metrics[:, empty] = float("nan")
    return loss, metrics


def parent_inputs(source, direct, uniform, births, step):
    if source == "full-direct":
        return (train.checkpoint_path(direct, "full_direct", step),)
    return tuple(train.previous.checkpoint_path(uniform, "odor", "uniform", birth, step) for birth in births)


def initialize(source, direct, uniform, births, step):
    if source == "full-direct":
        parents = [train.load_checkpoint(direct, "full_direct", birth, step) for birth in births]
    else:
        parents = [train.previous.load_checkpoint(uniform, "odor", "uniform", birth, step) for birth in births]
    values = {field.name: torch.cat([getattr(graph, field.name) for graph, _ in parents])
              for field in fields(train.fw.Graph)}
    graph = train.fw.Graph(nn.Parameter(values["weights"]), nn.Parameter(values["bias"]), values["present"])
    rules = [rule.state_dict() for _, rule in parents]
    rule = train.local.Rule(births)
    rule.load_state_dict({key: torch.cat([state[key] for state in rules]) for key in rules[0]})
    return graph, rule


def fit(path, config, births, source, direct, uniform, parent_step, digest, progress):
    torch.set_num_threads(1)
    graph, rule = initialize(source, direct, uniform, births, parent_step)
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    metrics = torch.empty((len(births), config.updates, len(train.METRICS)), dtype=torch.float32)
    modes = torch.empty(config.updates, dtype=torch.uint8)
    snapshots = sorted((path / ARM).glob("step-*.pt"))
    step, previous_seconds = 0, 0.
    if snapshots:
        saved = torch.load(snapshots[-1], map_location="cpu", weights_only=True)
        step, previous_seconds = saved["step"], saved["seconds"]
        if (saved["plan_hash"], saved["births"], saved["initializer"], saved["arm"]) != (
                digest, list(births), source, ARM) or snapshots[-1] != train.checkpoint_path(path, ARM, step):
            raise ValueError(f"Invalid resume snapshot {snapshots[-1]}")
        with torch.no_grad():
            for field in fields(train.fw.Graph):
                getattr(graph, field.name).copy_(saved["graph"][field.name])
        rule.load_state_dict(saved["rule"])
        optimizer.load_state_dict(saved["optimizer"])
        metrics[:, :step] = saved["metrics"]
        modes[:step] = saved["modes"]
    started = perf_counter()

    def elapsed():
        return previous_seconds + perf_counter() - started

    def save(index):
        state = dict(arm=ARM, births=list(births), step=index, plan_hash=digest, initializer=source,
                     world_seed=config.world_seed, world_domain=train.TRAIN_DOMAIN, schedule_domain=DOMAIN,
                     graph={field.name: getattr(graph, field.name).detach() for field in fields(train.fw.Graph)},
                     rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics[:, :index].clone(),
                     modes=modes[:index].clone(), seconds=elapsed())
        train._atomic_save(train.checkpoint_path(path, ARM, index), lambda temporary: torch.save(state, temporary))

    if not snapshots:
        save(0)
    print(f"initializer={source} begin={step} updates={config.updates}", flush=True)
    for index in range(step + 1, config.updates + 1):
        tape, mode = world(births, config, index)
        optimizer.zero_grad(set_to_none=True)
        loss, values = training_step(graph, rule, tape, mode, config)
        loss.backward()
        norm = train.clip_birth_gradients(tuple(p for p in parameters if p.grad is not None), config.grad_clip)
        optimizer.step()
        metrics[:, index - 1, :-1] = values
        metrics[:, index - 1, -1] = norm
        del loss
        modes[index - 1] = MODES.index(mode)
        saved = index % config.checkpoint_every == 0 or index == config.updates
        if saved:
            save(index)
        if index == 1 or saved or (progress and index % progress == 0):
            begin = max(0, index - 100)
            same = modes[begin:index] == modes[index - 1]
            joint, odor, mark = metrics[:, begin:index, 1:4][:, same].mean((0, 1)).tolist()
            print(f"initializer={source} step={index}/{config.updates} mode={mode} joint={joint:.3f} "
                  f"odor={odor:.3f} mark={mark:.3f} seconds={elapsed():.1f}", flush=True)


def run(path, config, births, sources, direct, uniform, parent_step, workers, progress):
    births, sources = tuple(births), tuple(sources)
    if (config.updates < 1 or config.pairs < 2 or config.checkpoint_every < 1 or config.lr <= 0
            or not births or len(set(births)) != len(births) or not sources or set(sources) - set(SOURCES)
            or len(set(sources)) != len(sources) or workers < 1 or progress < 0):
        raise ValueError("Invalid joint continuation configuration")
    path, direct, uniform = Path(path), Path(direct).resolve(), Path(uniform).resolve()
    inputs = {name: {str(file): train._source_digest(file)
                     for file in parent_inputs(name, direct, uniform, births, parent_step)} for name in sources}
    modules = (__file__, train.__file__, train.local.__file__, train.fw.__file__, train.previous.__file__)
    scheduled = Counter(mode_for_step(config, index) for index in range(1, config.updates + 1))
    contacts_per_mode = dict(zip(MODES, (4, 3, 3, 14), strict=True))
    lives_per_update = 2 * config.pairs
    contacts_per_birth = lives_per_update * sum(scheduled[mode] * count for mode, count in contacts_per_mode.items())
    plan = dict(config=asdict(config), births=births, initializers=sources, parent_step=parent_step,
                parent_checkpoint_sha256=inputs, source_sha256={Path(f).name: train._source_digest(f) for f in modules},
                modes=MODES, full_probability=STAGE, short_proportion=SHORT_PROPORTION,
                scheduled_updates=scheduled, lives_per_birth=lives_per_update * config.updates,
                active_ticks_per_contact=17, contacts_per_birth=contacts_per_birth,
                metrics=train.METRICS, schedule_domain=DOMAIN, tape_domain=train.TRAIN_DOMAIN,
                exposure="Both continuations use the same mode, same number of contacts, identical laws, gaps and "
                         "motor/edit draws each update. Four-contact mixed lives teach both modalities and score "
                         "one return from each, using a single mark demonstration and an odor study. Three-contact "
                         "lives separately study both marks or both odor contexts before one scored return. "
                         "The last half uses exact primary 14-contact worlds at identical update coordinates.",
                objective="Causal REINFORCE on physically earned choice payouts, with independent pair baseline. "
                          "Every contact updates the same graph and recurrent/trace state inside a life. "
                          "Mixed modes weight the two tasks equally; single-task lives score the one return.",
                prior_budget="Full-direct initialization includes additional context-specialist pretraining. "
                             "Both continuations initialize fresh Adam and share the entire continuation budget.")
    text = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    filename = path / "run.json"
    if filename.exists() and filename.read_text(encoding="utf8") != text:
        raise ValueError(f"Immutable continuation plan differs at {filename}")
    if not filename.exists():
        train._atomic_save(filename, lambda temporary: temporary.write_text(text, encoding="utf8"))
    digest = sha256(text.encode("utf8")).hexdigest()
    jobs = [(path / source, config, births, source, direct, uniform, parent_step, digest, progress)
            for source in sources]
    if workers == 1:
        for job in jobs:
            fit(*job)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for result in [pool.submit(fit, *job) for job in jobs]:
                result.result()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--direct-root", type=Path, default=BASE / "context-timing")
    parser.add_argument("--uniform-root", type=Path, default=BASE / "retention-objective")
    parser.add_argument("--parent-step", type=int, default=8000)
    parser.add_argument("--initializers", choices=SOURCES, nargs="+", default=SOURCES)
    parser.add_argument("--births", type=int, nargs="+", default=train.BIRTHS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()
    fit_config = train.Config(updates=args.updates, pairs=args.pairs, checkpoint_every=args.checkpoint_every)
    run(args.out, fit_config, args.births, args.initializers, args.direct_root, args.uniform_root,
        args.parent_step, args.workers, args.progress)
