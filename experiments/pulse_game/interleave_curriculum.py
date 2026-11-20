"""Fit short context retrievals before the same randomized joint-life endpoint."""

from argparse import ArgumentParser
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_train as train


DOMAIN = 2201141
ARM = "local_active"
OUT = train.DEFAULT_OUTPUT.parent / "curriculum-warm-odor"
PHASES = ((0., .95, .05), (.15, .60, .35), (.30, .10, .50), (.4375, 0., 0.))
EVENT_FIELDS = ("left", "right", "right_factor", "fixed_factor", "donor_right_factor",
                "donor_fixed_factor", "context", "probe", "objective_weight")


def schedule_probabilities(step, total):
    progress, immediate, interference = np.array(PHASES).T
    direct, crossed = (float(np.interp(step / total, progress, value)) for value in (immediate, interference))
    return direct, crossed


def arrange(tape, modes):
    """Move odor events to short alternating retrievals without moving physical time or marks."""
    context = tape.context[:, :, 0]
    odor = (context != train.MARK_CONTEXT).nonzero()[:, 2].reshape(*context.shape[:2], 7)
    current = context.gather(-1, odor)
    expected, reviews = current[..., :3], current[..., 3:]
    review_match = expected[..., None] == reviews[..., None, :]
    r0, r1, r2 = (3 + review_match.to(torch.int8).argmax(dim=-1)).unbind(-1)
    extra = 18 - r0 - r1 - r2  # The return event indices sum to 3 + 4 + 5 + 6.
    zero, one, two = (torch.full_like(extra, j) for j in range(3))
    direct = torch.stack((zero, r0, one, r1, two, r2, extra), dim=-1)
    crossed = torch.stack((zero, one, r0, two, r1, r2, extra), dim=-1)
    order = torch.where(modes[..., None] == 0, direct, crossed)
    order = torch.where(modes[..., None] == 2, torch.arange(7), order)
    physical = torch.arange(train.CONTACTS).expand_as(context).scatter(-1, odor, odor.gather(-1, order))

    def gather(value):
        index = physical[..., None, :]
        if value.ndim == 5:
            index = index[..., None]
        return value.gather(3, index.expand_as(value))

    return replace(tape, **{name: gather(getattr(tape, name)) for name in EVENT_FIELDS})


def world(births, config, step):
    tape = train.make_tape(births, config, step)
    direct, crossed = schedule_probabilities(step, config.updates)
    if direct + crossed == 0:
        return tape, (0., 0., 1.)
    draws = np.stack([np.random.default_rng([DOMAIN, config.world_seed, birth, step]).random(config.pairs)
                      for birth in births])
    mode = (draws >= direct).astype(np.int8) + (draws >= (direct + crossed)).astype(np.int8)
    fraction = tuple(float((mode == i).mean()) for i in range(3))
    return arrange(tape, torch.from_numpy(mode)), fraction


def run(path, config, births, warm_root, warm_step, progress):
    if (config.updates < 1 or config.pairs < 2 or config.checkpoint_every < 1 or progress < 0 or
            not births or len(set(births)) != len(births) or min(births) < 0):
        raise ValueError("Invalid curriculum configuration")
    torch.set_num_threads(1)
    path, warm_root = Path(path), Path(warm_root).resolve()
    warm = (str(warm_root), "odor", warm_step)
    parents = [train.previous.checkpoint_path(warm_root / "local", "odor", "active", birth, warm_step)
               for birth in births]
    sources = (__file__, train.__file__, train.local.__file__, train.legacy.__file__, train.fw.__file__,
               train.previous.__file__)
    manifest = dict(arms=[ARM], births=births, config=asdict(config), warm=warm, metrics=train.METRICS,
                    training_domain=train.TRAIN_DOMAIN, scaffold_domain=DOMAIN,
                    phase_schedule=dict(columns=("fraction_of_updates", "direct", "one_interference"), values=PHASES),
                    source_sha256={Path(source).name: train._source_digest(source) for source in sources},
                    warm_sha256={str(birth): train._source_digest(parent) for birth, parent in zip(births, parents)},
                    protocol="The same sampled joint lives, laws, contact ticks, waits, optimizer and objective as "
                             "the warm-odor fit. During the scaffold, only the seven odor events are moved. One "
                             "mode pairs every introduction with its first odor return and then adds the extra "
                             "return. A second mode provides one intervening odor between the first introduction "
                             "and return. Random context, modality positions and A/B agreement remain intact. "
                             "After 43.75% of updates the tape equals the warm-odor reference at every step.")
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    filename = path / "run.json"
    if filename.exists() and filename.read_text(encoding="utf8") != text:
        raise ValueError(f"Immutable curriculum protocol differs at {filename}")
    if not filename.exists():
        train._atomic_save(filename, lambda temporary: temporary.write_text(text, encoding="utf8"))
    plan_hash = sha256(text.encode("utf8")).hexdigest()

    graph, rule = train._birth(births, ARM, warm)
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    metrics = torch.empty((len(births), config.updates, len(train.METRICS)), dtype=torch.float32)
    dose = torch.empty((config.updates, 3), dtype=torch.float32)
    checkpoints = sorted((path / ARM).glob("step-*.pt"))
    step, previous_seconds = 0, 0.
    if checkpoints:
        snapshot = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        step, previous_seconds = snapshot["step"], snapshot["seconds"]
        if (snapshot["plan_hash"] != plan_hash or snapshot["arm"] != ARM or snapshot["births"] != list(births)
                or checkpoints[-1] != train.checkpoint_path(path, ARM, step)):
            raise ValueError("Resume checkpoint differs from curriculum manifest")
        with torch.no_grad():
            for field in fields(train.fw.Graph):
                getattr(graph, field.name).copy_(snapshot["graph"][field.name])
        rule.load_state_dict(snapshot["rule"])
        optimizer.load_state_dict(snapshot["optimizer"])
        metrics[:, :step] = snapshot["metrics"]
        dose[:step] = snapshot["schedule"]

    started = perf_counter()

    def elapsed():
        return previous_seconds + perf_counter() - started

    def save(index):
        state = dict(arm=ARM, births=list(births), step=index, plan_hash=plan_hash,
                     world_seed=config.world_seed, world_domain=train.TRAIN_DOMAIN, scaffold_domain=DOMAIN,
                     graph={field.name: getattr(graph, field.name).detach() for field in fields(train.fw.Graph)},
                     rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics[:, :index].clone(),
                     schedule=dose[:index].clone(), seconds=elapsed())
        train._atomic_save(train.checkpoint_path(path, ARM, index), lambda temporary: torch.save(state, temporary))

    if not checkpoints:
        save(0)
    print(f"curriculum start={step} updates={config.updates}", flush=True)
    for index in range(step + 1, config.updates + 1):
        tape, fraction = world(births, config, index)
        optimizer.zero_grad(set_to_none=True)
        rolled = train.rollout(graph, rule, tape, ARM, config.max_wait)
        loss, values = train.score_and_measure(rolled, tape)
        loss.backward()
        norms = train.clip_birth_gradients(tuple(p for p in parameters if p.grad is not None), config.grad_clip)
        optimizer.step()
        metrics[:, index - 1, :-1] = values
        metrics[:, index - 1, -1] = norms
        dose[index - 1] = torch.as_tensor(fraction)
        del rolled, loss
        saved = index % config.checkpoint_every == 0 or index == config.updates
        if saved:
            save(index)
        if index == 1 or saved or (progress and index % progress == 0):
            joint, odor, mark = metrics[:, max(0, index - 100):index, 1:4].mean((0, 1)).tolist()
            print(f"curriculum step={index}/{config.updates} joint={joint:.3f} odor={odor:.3f} marks={mark:.3f} "
                  f"full={fraction[2]:.2f} seconds={elapsed():.1f}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--births", type=int, nargs="+", default=train.BIRTHS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--warm-root", type=Path, default=train.previous.DEFAULT_OUTPUT)
    parser.add_argument("--warm-step", type=int, default=8000)
    args = parser.parse_args()
    fit_config = train.Config(updates=args.updates, pairs=args.pairs, checkpoint_every=args.checkpoint_every)
    run(args.out, fit_config, args.births, args.warm_root, args.warm_step, args.progress)
