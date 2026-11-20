"""Measure the realized return of each route across physically cued planning horizons."""

from argparse import ArgumentParser
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_paired_lesson as lesson, noon_review_world as world
from . import local_trace_graph as net, local_trace_train as weights
from . import interleave_revisit_train as storage, noon_value_report as intervals


DATA = world.DATA / "horizon"
EVIDENCE = world.EVIDENCE
CAL_PLAN = EVIDENCE / "horizon-calibration.md"
COHORTS = world.COHORTS
CAL_SEEDS = {"original": 268201032, "independent": 268201038}
CAL_LENGTHS = (0, 1, 2, 4, 6, 8, 12)
CAL_WORLDS = 256
SOURCE_STEP = 12000
CUE_TICKS = 3
TAIL_TICKS = net.REACTION_TICKS + world.ECHO_TICKS


@dataclass(frozen=True)
class Opportunity:
    target: torch.Tensor
    orientation: torch.Tensor
    motor: torch.Tensor
    cue_noise: torch.Tensor

    def subset(self, index):
        return Opportunity(*(getattr(self, f.name)[index] for f in fields(self)))


def opportunities(seed, births, update, worlds, length):
    roots = ([0x486f5041, seed, birth, update] for birth in births) if births else ([0x486f5042, seed],)

    def generate(domain):
        rng = np.random.default_rng(domain)
        starts = rng.integers(0, 2, (worlds, (length + 1) // 2))
        target = np.stack((starts, 1 - starts), axis=-1).reshape(worlds, -1)[:, :length]
        sign = (2 * rng.integers(0, 2, (worlds, length)) - 1).astype(np.float32)
        motor = rng.random((worlds, length), dtype=np.float32)
        cue = (2 * rng.integers(0, 2, worlds) - 1).astype(np.float32)
        return tuple(np.repeat(value, 4, axis=0) for value in (target, sign, motor, cue))

    return Opportunity(*(torch.from_numpy(np.concatenate(parts)) for parts in zip(*(generate(x) for x in roots), strict=True)))


def initial(graph, rule, tape, missing):
    count = tape.lanes // len(graph.bias)
    state = net.start(graph, count)
    coeff = rule.expand(count)
    history = []
    for item in range(2):
        orientation = tape.study_orientation[:, item]
        left = tape.odor[:, item] * orientation[:, None]
        factor = torch.where(missing == item, 0., tape.law[:, item] * orientation)
        contact = world.collect(coeff, state, left, -left, factor, tape.study_draw[:, item])
        state = contact.state
        history.append(torch.stack((contact.motor.side, contact.earned), 1))
    return state, torch.stack(history, 1)


def signal(coeff, state, tape, cue):
    left, right = world.candidate_menu(tape)
    mode = state.graph.bias.new_tensor((0., 0., .85))[None].expand_as(left)
    state = net._advance(coeff, state, net.native.sensory(state.graph, mode, mode), world.MODE_TICKS)
    sensory = mode.clone()
    sensory[:, 0] = cue * 1.35
    state = net._advance(coeff, state, net.native.sensory(state.graph, sensory, sensory), CUE_TICKS)
    return net.act(coeff, state, left, right, tape.decision_draw)


def paid_tail(coeff, clocked, tape, future, horizon):
    n = tape.lanes
    rows = torch.arange(n)
    observed = []
    for i in range(future.target.shape[1]):
        if i >= horizon:
            observed.append(torch.zeros((n, 2), dtype=clocked.hidden.dtype))
            clocked = net.wait(coeff, clocked, TAIL_TICKS)
            continue
        item = future.target[:, i]
        orient = future.orientation[:, i]
        left, right = world.sensory_for(tape, item, orient)
        action = net.act(coeff, clocked, left, right, future.motor[:, i])
        earned = action.side * tape.law[rows, item] * orient
        clocked = world.echo_choice(coeff, action.state, action.side)
        observed.append(torch.stack((action.side, earned), 1))
    return torch.stack(observed, 1), clocked


@torch.no_grad()
def branches(graph, rule, state, tape, future, horizon, *, write=True, echo=True):
    if not isinstance(horizon, int) or not 0 <= horizon <= future.target.shape[1]:
        raise ValueError("A rollout has one realized integer number of paid future opportunities")
    rows = torch.arange(tape.lanes).repeat_interleave(2)
    branch = lesson.expand(tape, rows)
    odds = future.subset(rows)
    side = state.hidden.new_tensor((-1., 1.)).repeat(tape.lanes)
    coeff = rule.expand(branch.lanes // len(graph.bias))
    state = state.subset(rows)
    if echo:
        state = world.echo_choice(coeff, state, side)
    else:
        state = net.wait(coeff, state, world.ECHO_TICKS)
    item = world.selected_item(branch, side)
    left, right = world.sensory_for(branch, item, branch.review_orientation)
    truth = branch.law[torch.arange(branch.lanes), item] * branch.review_orientation
    contact = world.collect(coeff, state, left, right, truth, branch.review_draw, write=write)
    history, after = paid_tail(coeff, contact.state, branch, odds, horizon)
    events = torch.cat((torch.stack((contact.motor.side, contact.earned), 1)[:, None], history), dim=1)
    return events.reshape(tape.lanes, 2, -1, 2), after


def checkpoint(cohort, birth, stage=SOURCE_STEP, arm="paired", path=None):
    directory = (lesson.ROOT if path is None else path) / cohort
    return weights.checkpoint_path(directory, "odor", arm, birth, stage)


def load(cohort, birth, stage=SOURCE_STEP, arm="paired", path=None):
    directory = (lesson.ROOT if path is None else path) / cohort
    return weights.load_checkpoint(directory, "odor", arm, birth, stage)


@torch.no_grad()
def evaluate(graph, rule, tape, missing, future):
    state, prefix = initial(graph, rule, tape, missing)
    n = tape.lanes
    cue = state.hidden.new_tensor((-1., 1.)).repeat(n)
    ids = torch.arange(n).repeat_interleave(2)
    duplicate = lesson.expand(tape, ids)
    coeff = rule.expand(2 * n)
    decision = signal(coeff, state.subset(ids), duplicate, cue)
    rewards, _ = branches(graph, rule, decision.state, duplicate, future.subset(ids), future.target.shape[1])
    return dict(events=rewards.reshape(n, 2, 2, -1, 2).numpy().astype(np.int8),
                action=decision.side.reshape(n, 2).numpy().astype(np.int8),
                motor_logit=decision.motor_logit.reshape(n, 2).numpy(),
                known_history=prefix.numpy().astype(np.int8), candidate=tape.candidates.numpy(),
                no_taste=(tape.candidates[:, 1] == missing).long().numpy(), queries=future.target.numpy())


def digest(path):
    return lesson.sha256(Path(path).read_bytes()).hexdigest()


def calibration(cohort, worlds=CAL_WORLDS, births=None):
    seed = CAL_SEEDS[cohort]
    planned = COHORTS[cohort] if births is None else births
    directory = DATA / "calibration-integer" / cohort
    directory.mkdir(parents=True, exist_ok=True)
    tape, missing = lesson.prepare(seed, (), 0, worlds, 2, omissions=True, heldout=True)
    future = opportunities(seed, (), 0, worlds, max(CAL_LENGTHS))
    verified = {str(b): digest(checkpoint(cohort, b)) for b in planned}
    code = {Path(p).name: digest(p) for p in (__file__, lesson.__file__, world.__file__, net.__file__, net.native.__file__)}
    signature = str((verified, code, digest(CAL_PLAN), worlds, seed))
    archive, per_world = {}, []
    for b in planned:
        path = directory / f"{worlds}-{b}.npz"
        if not path.exists():
            graph, rule = load(cohort, b)
            chunks = []
            for begin in range(0, tape.lanes, 128):
                idx = torch.arange(begin, min(tape.lanes, begin + 128))
                chunks.append(evaluate(graph, rule, lesson.expand(tape, idx), missing[idx], future.subset(idx)))
            raw = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
            storage._write_raw(path, **raw, signature=signature)
        with np.load(path) as loaded:
            if str(loaded["signature"]) != signature:
                raise ValueError(f"Wrong calibration cache {path}")
            items = np.arange(tape.lanes)[:, None]
            cue = np.arange(2)[None]
            latent = loaded["no_taste"][:, None]
            events = loaded["events"][..., 1]
            direct = events[items, cue, latent].cumsum(-1)
            known = events[items, cue, 1-latent].cumsum(-1)
            metrics = {}
            for j, name in enumerate(("minus", "plus")):
                for h in CAL_LENGTHS:
                    for label, value in (("unknown", direct[:, j, h]), ("known", known[:, j, h]),
                                         ("unknown_minus_known", direct[:, j, h] - known[:, j, h])):
                        metrics[f"{name}/{h}/{label}"] = value.reshape(worlds, 4).mean(1)
                probability = 1 / (1 + np.exp(-loaded["motor_logit"][:, j]))
                metrics[f"{name}/initial_unknown_probability"] = np.where(latent[:, 0], probability,
                                                                            1-probability).reshape(worlds,4).mean(1)
            per_world.append(metrics)
        archive[str(path.relative_to(world.ROOT))] = digest(path)
        print(cohort, b, flush=True)
    table = {key: np.stack([row[key] for row in per_world]) for key in per_world[0]}
    summary = dict(births=list(planned), seed=seed, worlds=worlds, exact_paid_horizons=CAL_LENGTHS,
                   checkpoints=verified, code=code, registered_plan_sha256=digest(CAL_PLAN), files=archive,
                   scores=intervals.PairedCrossed(len(planned), worlds, seed+109).summary(table))
    filename = f"horizon-calibration-{cohort}.json" if worlds == CAL_WORLDS and births is None else f"horizon-calibration-{cohort}-smoke.json"
    storage._write_json(EVIDENCE/filename, summary)


if __name__ == "__main__":
    args = ArgumentParser(description=__doc__)
    args.add_argument("--cohort", choices=COHORTS, required=True)
    args.add_argument("--worlds", type=int, default=CAL_WORLDS)
    args.add_argument("--births", type=int, nargs="+")
    cli = args.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    calibration(cli.cohort, cli.worlds, cli.births)
    print(round(perf_counter() - started, 2), flush=True)
