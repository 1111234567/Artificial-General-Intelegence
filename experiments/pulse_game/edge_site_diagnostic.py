"""Replay a real B receipt while changing only selected native edge values."""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import interleave_interference as geometry
from . import interleave_revisit_train as io
from . import local_trace_graph as net
from . import noon_value as prior


ROOT = geometry.ROOT
EVIDENCE = ROOT / "docs/evidence/edge-site-20261120"
DATA = ROOT / "data/edge-site-20261120/diagnostic"
PLAN = EVIDENCE / "diagnostic-plan.md"
COHORTS = {"original": (731, 346112031), "independent": (811, 346112087)}
ANGLES = (15, 30, 90)
WORLDS = 256
CONTROLS = 8
BUDGETS = (1, 2, 4)
METHODS = ("dose", "joint", "protect_a", "joint_or_skip", "protect_a_or_skip")
POST_TICKS = net.POST_WRITE_TICKS + 4
N = net.NODES
SITES = N * N


@dataclass(frozen=True)
class Prepared:
    ready: net.State
    full: net.Commit
    eligible: torch.Tensor
    delta: torch.Tensor
    side: torch.Tensor
    taste: torch.Tensor


def pack_worlds(seed, count, angle):
    """The second seed coordinate is fixed so all births share the world bank."""
    return geometry.worlds(seed, 0, count, angle)


def shortlist_groups(delta, ranked):
    magnitudes = delta.abs()
    dose = magnitudes.argsort(descending=True, stable=True)
    rows = magnitudes.reshape(-1, N, N).argsort(descending=True, stable=True)[..., :4]
    row_addresses = torch.arange(N)[None, :, None] * N + rows
    transposed = magnitudes.reshape(-1, N, N).transpose(1, 2)
    columns = transposed.argsort(descending=True, stable=True)[..., :4]
    column_addresses = columns * N + torch.arange(N)[None, :, None]
    union = torch.cat(ranked, dim=1)
    choose_pair = torch.combinations(torch.arange(union.shape[1]), r=2)
    pairs = union[:, choose_pair]
    pairs = torch.cat((dose[:, None, :2], pairs), dim=1)
    fours = torch.cat((dose[:, None, :4], row_addresses, column_addresses,
                       torch.stack(ranked, dim=1)), dim=1)
    return {2: address_order(pairs), 4: address_order(fours)}


def address_order(groups):
    groups = groups.sort(dim=2).values
    place = SITES ** torch.arange(groups.shape[2] - 1, -1, -1)
    rank = (groups * place).sum(2).argsort(stable=True)
    return groups.gather(1, rank[..., None].expand_as(groups))


@torch.no_grad()
def prepare(graph, rule, world):
    coeff = rule.expand(len(world.a))
    cold = net.start(graph, len(world.a))
    updated_a = geometry.commits(coeff, cold, world.a, world.law_a, world, 0)
    after_a = net.wait(coeff, updated_a.state, 4)
    receipt_b = geometry.experience(coeff, after_a, world.b, world.law_b, world, 1)
    draw = world.draws[:, 1, 1:]
    full = net.commit(coeff, receipt_b, draw)
    ready = net.wait(coeff, receipt_b.state, net.PRE_WRITE_TICKS)
    delta = full.receipt.weight_delta.flatten(1)
    eligible = (ready.graph.present & full.state.graph.present).flatten(1)
    if (delta[~eligible] != 0).any():
        raise AssertionError("Native receipt moved a non-surviving edge")
    return Prepared(ready, full, eligible, delta, receipt_b.side, receipt_b.taste)


def advance_many(graph, hidden, stimulus, ticks, addresses, delta):
    """Run one recurrent graph with small additive interventions per candidate."""
    weights = (graph.weights * graph.present).transpose(1, 2)
    sender = addresses % N
    receiver = addresses // N
    drive = (graph.bias + stimulus)[:, None, :]
    for _ in range(ticks):
        current = torch.bmm(hidden, weights)
        current.scatter_add_(2, receiver, hidden.gather(2, sender) * delta)
        hidden = (1.0 - net.native.LEAK) * hidden + net.native.LEAK * (current + drive).tanh()
    return hidden


def score_options(ready, delta, addresses, axes, laws):
    """Integrate over public test-menu orientations without using the test tape."""
    graph = ready.graph
    change = delta.gather(1, addresses.flatten(1)).reshape_as(addresses)
    hidden = ready.hidden[:, None, :].expand(-1, addresses.shape[1], -1)
    state = advance_many(graph, hidden, torch.zeros_like(graph.bias), POST_TICKS, addresses, change)
    probabilities = []
    for axis, law in zip(axes, laws, strict=True):
        stimulus = net.native.sensory(graph, -axis, axis)
        odds = []
        for sign in (1, -1):
            h = advance_many(graph, state, sign * stimulus, net.REACTION_TICKS, addresses, change)
            odds.append(((2.5 * sign * law[:, None]) * (h[..., 11] - h[..., 10])).sigmoid())
        probabilities.append((odds[0] + odds[1]) / 2)
    return torch.stack(probabilities, dim=2)


def select_options(options, ok, p, blank, delta, k):
    truth = p.prod(dim=2)
    largest = delta.abs().argsort(descending=True, stable=True)[:, :k]
    largest = largest.sort(dim=1).values
    joint = truth.masked_fill(~ok, -torch.inf).argmax(1)
    protected = ok & (p[..., 0] >= (blank[:, 0, None] - .02))
    protect_a = p[..., 1].masked_fill(~protected, -torch.inf).argmax(1)
    lane = torch.arange(len(delta))
    have_dose = (delta.gather(1, largest) != 0).all(1)
    have_joint = ok.any(1)
    have_protect = protected.any(1)
    joint_skip = have_joint & (truth[lane, joint] > blank.prod(1))
    protect_skip = have_protect & (p[lane, protect_a, 1] > blank[:, 1])
    location = torch.stack((largest, options[lane, joint], options[lane, protect_a],
                            options[lane, joint], options[lane, protect_a]), dim=1)
    enabled = torch.stack((have_dose, have_joint, have_protect, joint_skip, protect_skip), dim=1)
    both_improve = (ok & ((p - blank[:, None, :]) >= .01).all(2)).any(1)
    return location, enabled, p[lane, joint], p[lane, protect_a], both_improve


def select(prepared, axes, laws):
    ready, delta = prepared.ready, prepared.delta
    lanes = len(delta)
    single = torch.arange(SITES).reshape(1, SITES, 1).expand(lanes, -1, -1)
    blank = score_options(ready, torch.zeros_like(delta), single[:, :1], axes, laws)[:, 0]
    score = score_options(ready, delta, single, axes, laws)
    single_ok = delta != 0
    ranked = [metric.masked_fill(~single_ok, -torch.inf).argsort(descending=True, stable=True)[:, :4]
              for metric in (score[..., 0], score[..., 1], score.prod(2))]
    candidates = {1: single, **shortlist_groups(delta, ranked)}
    out = {}
    for k, options in candidates.items():
        valid = (delta.gather(1, options.flatten(1)).reshape_as(options) != 0).all(2)
        if k > 1:
            valid &= (options.diff(dim=2) != 0).all(2)
        p = score if k == 1 else score_options(ready, delta, options, axes, laws)
        out[k] = select_options(options, valid, p, blank, delta, k)
    return out


def matches(prepared, locations, enabled, draws):
    """Relocate each payload using an independent uniform draw per available set."""
    old = prepared.ready.graph.weights.flatten(1).numpy()
    eligible = prepared.eligible.numpy()
    delta = prepared.delta.numpy()
    location = locations.numpy()
    on = enabled.numpy()
    lanes, methods, k = location.shape
    row = np.arange(lanes)[:, None, None]
    cursor = np.zeros((lanes, methods, CONTROLS, k), dtype=np.int16)
    replacement = np.zeros_like(cursor, dtype=np.float32)
    fallback = np.full(cursor.shape, -1, dtype=np.int8)
    unavailable = np.zeros((lanes, methods, CONTROLS, SITES), dtype=bool)
    np.put_along_axis(unavailable, np.broadcast_to(location[:, :, None, :], cursor.shape), True, axis=-1)
    lower, upper = np.minimum(old, -net.WEIGHT_LIMIT), np.maximum(old, net.WEIGHT_LIMIT)
    for j in range(k):
        source = location[:, :, j, None]
        signed = delta[row, source]
        update = old[:, None, :] + signed
        bounds = (update >= lower[:, None, :]) & (update <= upper[:, None, :])
        safe = bounds & (update != old[:, None, :]) & (np.abs(update) >= net.MIN_WEIGHT)
        available = (eligible[:, None, :] & safe)[:, :, None, :] & ~unavailable
        direction = np.signbit(delta[:, None, :]) == np.signbit(signed)
        direction &= delta[:, None, :] != 0
        preferred = available & direction[:, :, None, :]
        have_match = preferred.any(-1)
        choice = np.where(have_match[..., None], preferred, available)
        count = choice.sum(-1)
        rank = (draws[:, None, :, j] * count).astype(np.int32)
        picked = (choice.cumsum(-1) > rank[..., None]).argmax(-1)
        destination = np.where(on[:, :, None] & (count > 0), picked, source)
        level = np.where(have_match, 0, np.where(count > 0, 1, 2))
        fallback[..., j] = np.where(on[:, :, None], level, -1)
        cursor[..., j] = destination
        value = np.take_along_axis(update[:, :, None, :], destination[..., None], -1)[..., 0]
        original = prepared.full.state.graph.weights.flatten(1).numpy()[row, source]
        unchanged = old[row, source]
        replacement[..., j] = np.where(on[:, :, None], np.where(level < 2, value, original), unchanged)
        np.put_along_axis(unavailable, destination[..., None], True, axis=-1)
    realized = replacement - old[:, None, None, :][np.arange(lanes)[:, None, None, None], 0, 0, cursor]
    intended = delta[row, location] * on[..., None]
    if not np.array_equal(np.sign(realized), np.broadcast_to(np.sign(intended[:, :, None, :]), realized.shape)):
        raise AssertionError("Relocation lost the sign or count of physical writes")
    return torch.from_numpy(cursor).long(), torch.from_numpy(replacement), realized, fallback


def replay(rule, ready, graphs, world):
    """Take actual native motor actions and report separately integrated odds."""
    lanes, arms = len(ready.hidden), len(graphs.bias) // len(ready.hidden)
    lane = torch.arange(lanes).repeat_interleave(arms)
    coeff = rule.expand(len(lane))
    selected = replace(ready.subset(lane), graph=graphs)
    state = net.wait(coeff, selected, POST_TICKS)
    side, correct, probability = [], [], []
    for axis, law, t in ((world.a, world.law_a, 3), (world.b, world.law_b, 4)):
        vector = axis[lane]
        orientation = world.orders[lane, t]
        goal = law[lane] * orientation
        right = vector * orientation[:, None]
        action = net.act(coeff, state, -right, right, world.draws[lane, t, 0])
        side.append(action.side.to(torch.int8))
        correct.append(action.side == goal)
        plus = prior.motor_logit(state, -vector, vector)
        minus = prior.motor_logit(state, vector, -vector)
        probability.append(((plus * law[lane]).sigmoid() + (-minus * law[lane]).sigmoid()) / 2)
    return {name: torch.stack(values, dim=1).reshape(lanes, arms, 2).numpy()
            for name, values in (("side", side), ("correct", correct), ("probability", probability))}


def variant_graph(graph, locations, replacements):
    lanes = len(graph.bias)
    arms = locations.shape[1]
    base = graph.weights.flatten(1)[:, None, :].expand(lanes, arms, SITES)
    weights = base.scatter(2, locations, replacements).reshape(-1, N, N)
    return net.Graph(weights, graph.bias.repeat_interleave(arms, dim=0),
                     graph.present.repeat_interleave(arms, dim=0))


def baseline_graphs(prepared):
    previous, full = prepared.ready.graph, prepared.full.state.graph
    edge_full = torch.where(prepared.eligible.reshape(-1, N, N), full.weights, previous.weights)
    lanes = len(previous.bias)
    return net.Graph(torch.stack((previous.weights, edge_full, full.weights), dim=1).reshape(-1, N, N),
                     torch.stack((previous.bias, previous.bias, full.bias), dim=1).reshape(-1, N),
                     torch.stack((previous.present, previous.present, full.present), dim=1).reshape(lanes * 3, N, N))


@torch.no_grad()
def evaluate(graph, rule, world, draws):
    prepared = prepare(graph, rule, world)
    options = select(prepared, (world.a, world.b), (world.law_a, world.law_b))
    ready, full = prepared.ready, prepared.full.state
    base_graphs = baseline_graphs(prepared)
    base = replay(rule, ready, base_graphs, world)
    coeff = rule.expand(len(ready.hidden))
    state = net.wait(coeff, full, 4)
    native_side, native_correct, _ = geometry.probe(coeff, state, world)
    if not np.array_equal(native_side.numpy(), base["side"][:, 2]) or not np.array_equal(
            native_correct.numpy(), base["correct"][:, 2]):
        raise AssertionError("Fully native replay changed the actual action")
    result = {f"baseline/{name}": x for name, x in base.items()}
    result.update({"receipt/delta": prepared.delta.numpy(),
                   "receipt/side": prepared.side.numpy().astype(np.int8),
                   "receipt/taste": prepared.taste.numpy().astype(np.int8),
                   "receipt/born": prepared.full.receipt.birth.numpy().astype(np.int16),
                   "receipt/retired": prepared.full.receipt.retired.numpy().astype(np.int16),
                   "receipt/bias_l1": prepared.full.receipt.bias_delta.abs().sum(1).numpy()})
    lanes = len(ready.hidden)
    old = ready.graph.weights.flatten(1)
    new = full.graph.weights.flatten(1)
    for k, (locations, enabled, joint_p, protect_p, both_improve) in options.items():
        index = locations.flatten(1)
        unchanged = old.gather(1, index).reshape_as(locations)
        selected = new.gather(1, index).reshape_as(locations)
        selected = torch.where(enabled[..., None], selected, unchanged)
        permute, values, actual_delta, fallback = matches(prepared, locations, enabled, draws)
        all_dest = torch.cat((locations, permute.reshape(lanes, -1, k)), dim=1)
        all_values = torch.cat((selected, values.reshape(lanes, -1, k)), dim=1)
        graph = variant_graph(ready.graph, all_dest, all_values)
        outcomes = replay(rule, ready, graph, world)
        count = len(METHODS)
        result.update({f"{k}/native/{name}": value[:, :count] for name, value in outcomes.items()})
        result.update({f"{k}/random/{name}": value[:, count:].reshape(lanes, count, CONTROLS, 2)
                       for name, value in outcomes.items()})
        result.update({f"{k}/native/address": locations.numpy().astype(np.int16),
                       f"{k}/native/delta": (selected - unchanged).numpy(),
                       f"{k}/native/enabled": enabled.numpy(),
                       f"{k}/random/address": permute.numpy().astype(np.int16),
                       f"{k}/random/delta": actual_delta,
                       f"{k}/random/fallback": fallback,
                       f"{k}/feasible_both_gain": both_improve.numpy()})
        expected_oracles = torch.stack((joint_p, protect_p, joint_p, protect_p), dim=1).numpy()
        valid = enabled[:, 1:].numpy()
        error = np.abs(expected_oracles - outcomes["probability"][:, 1:count])
        if valid.any() and error[valid].max() > 3e-6:
            raise AssertionError("Sparse candidate scoring diverged from native replay")
        result[f"{k}/score_replay_max_error"] = (error * valid[..., None]).max(axis=(1, 2))
    return result


def execute(cohort, worlds, chunk, out):
    first, seed = COHORTS[cohort]
    source, arm, step = prior.source(cohort, "uniform")
    dest = out / cohort
    dest.mkdir(parents=True, exist_ok=True)
    signatures = dict(plan=prior.digest(PLAN), runner=prior.digest(Path(__file__)))
    rng = np.random.default_rng([0x45444745, seed])
    draws = rng.random((worlds, CONTROLS, max(BUDGETS)), dtype=np.float32).repeat(4, axis=0)
    for i in range(first, first + 6):
        checkpoint = prior.fitting.checkpoint_path(source, "odor", arm, i, step)
        hashes = {**signatures, "checkpoint": prior.digest(checkpoint)}
        graph, rule = prior.fitting.load_checkpoint(source, "odor", arm, i, step)
        for angle in ANGLES:
            path = dest / f"{i}-{angle}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as raw:
                    valid = all(str(raw[f"sha256/{key}"]) == value for key, value in hashes.items())
                    if not valid or int(raw["worlds"]) != worlds:
                        raise ValueError(f"Wrong frozen replay {path}")
                continue
            world = pack_worlds(seed, worlds, angle)
            banks = []
            for lo in range(0, worlds, chunk):
                section = slice(lo * 4, (lo + chunk) * 4)
                sliced = geometry.World(*(value[section] for value in vars(world).values()))
                banks.append(evaluate(graph, rule, sliced, draws[section]))
            sample = {key: np.concatenate([bank[key] for bank in banks]) for key in banks[0]}
            sample.update({f"sha256/{key}": value for key, value in hashes.items()})
            sample.update(worlds=worlds, cohort=cohort, seed=seed, angle=angle, birth=i,
                          checkpoint=checkpoint.relative_to(ROOT).as_posix(), laws=torch.stack(
                              (world.law_a, world.law_b), dim=1).numpy().astype(np.int8),
                          orders=world.orders.numpy().astype(np.int8), a=world.a.numpy(), b=world.b.numpy())
            io._write_raw(path, **sample)
            print(cohort, i, angle, worlds, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=COHORTS)
    parser.add_argument("--out", type=Path, default=DATA)
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    execute(args.cohort, args.worlds, args.chunk, args.out)
