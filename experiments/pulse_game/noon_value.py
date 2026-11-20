"""Replay frozen candidates with common random numbers and sealed observation budgets."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_cross_skill_eval as six
from . import interleave_revisit_train as io, local_trace_graph as net
from . import local_trace_train as fitting, revisit_choice_graph as final_motor
from .interleave_interference import DATA as OLD_DATA, ROOT


DATA = ROOT / "data/active-review-20261120/value"
EVIDENCE = ROOT / "docs/evidence/active-review-20261120"
PLAN = EVIDENCE / "value-plan.md"
COHORTS = {"original": (731, 246112013), "independent": (811, 246112071)}
ASSAYS = {"odor4": 160, "odor8": 160, "angle15": 96, "angle45": 96, "angle75": 96, "joint6": 64}
DIAGNOSTICS = ("blind", "all", "half")
SAMPLES_PER_FOLD = 8
THRESHOLD = .01
POST_STUDY_WAIT = POST_REVIEW_WAIT = 4


@dataclass(frozen=True)
class Tape:
    axis: torch.Tensor
    cue: torch.Tensor
    law: torch.Tensor
    support: torch.Tensor
    support_order: torch.Tensor
    support_sign: torch.Tensor
    support_draw: torch.Tensor
    support_wait: torch.Tensor
    preview_order: torch.Tensor
    preview_sign: torch.Tensor
    priority: torch.Tensor
    scramble: torch.Tensor
    review_sign: torch.Tensor
    review_draw: torch.Tensor
    terminal_sign: torch.Tensor
    terminal_draw: torch.Tensor
    paid_objects: torch.Tensor
    paid_draw: torch.Tensor
    paid_wait: torch.Tensor

    def __getitem__(self, world_slice):
        return Tape(*(value[world_slice] for value in vars(self).values()))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source(cohort, body):
    if body == "uniform":
        relative = "retention-objective" if cohort == "original" else "retention-objective-confirmation"
        return OLD_DATA / relative, "uniform", 8000
    if body == "mixed_repair":
        return OLD_DATA / "one-life-repair" / cohort, "mixed", 2000
    raise ValueError(f"Unknown frozen body {body}")


def make_tape(assay, seed, worlds):
    # Controlled-angle runs share all streams and rotate only the second direction.
    code = 1 if assay == "odor4" else 2 if assay == "odor8" else 3 if assay == "joint6" else 4
    seeds = np.random.SeedSequence([0x415652, seed, code]).spawn(7)
    layout, preview, policy, fold_a, fold_b, terminal, paid = (np.random.default_rng(key) for key in seeds)
    layout_seed = int(layout.integers(2**31))
    if assay == "joint6":
        world = six.make(layout_seed, worlds)
        axis = (world["query_right"] - world["query_left"]) * world["query_orientation"][..., None] / 2
        cue = (world["query_left"] + world["query_right"]) / 2
        law = np.broadcast_to(six.LAWS.astype(np.float32), (worlds, 64, 6)).copy()
        order = world["orders"][:, 2]
        support = np.minimum(order, 5)
        support_sign = np.take_along_axis(world["study_orientation"], order, axis=1)
        support_draw = world["draws"][np.arange(worlds)[:, None], order]
        support_wait = np.take_along_axis(world["waits"], order, axis=1)
        support_wait[:, 0] = 0
    else:
        count = 8 if assay == "odor8" else 4
        world = cap.make_tape("odor", layout_seed, worlds, count - 1)
        axis = -world.directions.numpy()
        cue = np.zeros_like(axis)
        law = world.law.numpy().reshape(worlds, 4, count)
        order = np.broadcast_to(np.arange(count), (worlds, count)).copy()
        support = order
        support_sign = np.sign((world.right[::4, :, 0].numpy() * axis).sum(2))
        support_draw = np.concatenate((world.motor_draw[::4, :, 0, None].numpy(),
                                       world.edit_draw[::4, :, 0].numpy()), axis=2)
        support_wait = np.zeros_like(order)
        if assay.startswith("angle"):
            angle = np.deg2rad(int(assay.removeprefix("angle")))
            direction = axis[:, 0] / np.linalg.norm(axis[:, 0], axis=1, keepdims=True)
            tangent = axis[:, 1] - (axis[:, 1] * direction).sum(1, keepdims=True) * direction
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            axis[:, 1] = 1.35 * (direction * np.cos(angle) + tangent * np.sin(angle))
    count = axis.shape[1]
    candidates = np.broadcast_to(np.arange(count), (worlds, count))
    preview_order = preview.permuted(candidates, axis=1)
    preview_sign = signs(preview, (worlds, count))
    priority = policy.random((worlds, count), dtype=np.float32)
    scramble = policy.random((worlds, count), dtype=np.float32).argsort(1)
    balanced = np.tile([-1., 1.], SAMPLES_PER_FOLD // 2).astype(np.float32)
    balanced = np.broadcast_to(balanced, (worlds, SAMPLES_PER_FOLD))
    review_sign = np.concatenate([fold.permuted(balanced, axis=1) for fold in (fold_a, fold_b)], axis=1)
    review_draw = np.concatenate([fold.random((worlds, SAMPLES_PER_FOLD, 5), dtype=np.float32)
                                  for fold in (fold_a, fold_b)], axis=1)
    terminal_sign = signs(terminal, (worlds, count))
    terminal_draw = terminal.random((worlds, count), dtype=np.float32)
    paid_objects = paid.permuted(np.column_stack((paid.integers(0, 3, worlds), paid.integers(3, 6, worlds))), axis=1)
    paid_draw = paid.random((worlds, 2, 5), dtype=np.float32)
    paid_wait = paid.integers(0, 9, (worlds, 2))
    values = (axis, cue, law, support, order, support_sign, support_draw, support_wait, preview_order,
              preview_sign, priority, scramble, review_sign, review_draw, terminal_sign, terminal_draw,
              paid_objects, paid_draw, paid_wait)
    return Tape(*(torch.from_numpy(value) for value in values))


def signs(rng, shape):
    return rng.choice(np.array([-1., 1.], dtype=np.float32), shape)


def menus(axis, cue, sign):
    turn = axis * sign[..., None]
    return cue - turn, cue + turn


def begin(graph, rule, tape, public, law, joint):
    state = net.start(graph, len(public))
    rows = torch.arange(len(public))
    for slot in range(tape.support.shape[1]):
        if joint:
            state = net.wait(rule, state, tape.support_wait[public, slot], max_ticks=8)
        item = tape.support[public, slot]
        sign = tape.support_sign[public, slot]
        left, right = menus(tape.axis[public, item], tape.cue[public, item], sign)
        feeder = tape.support_order[public, slot] >= 5 if joint else torch.zeros_like(item, dtype=torch.bool)
        left = torch.where(feeder[:, None], right, left)
        draws = tape.support_draw[public, slot]
        action = net.act(rule, state, left, right, draws[:, 0])
        factor = law[rows, item] * sign
        taste = torch.where(feeder, factor, action.side * factor)
        if slot == 0 and not torch.equal(action.side.reshape(tape.law.shape[:2]),
                                         action.side.reshape(tape.law.shape[:2])[:, :1].expand_as(tape.law[:, :, 0])):
            raise AssertionError("First action depended on an unrevealed private map")
        observed = net.observe(rule, action, left, right, taste)
        state = net.commit(rule, observed, draws[:, 1:]).state
    return net.wait(rule, state, POST_STUDY_WAIT)


def preview(rule, start, tape, public):
    count = tape.axis.shape[1]
    idle = net.wait(rule, start, count * net.REACTION_TICKS)
    state = start
    logits = torch.zeros((len(public), count))
    rows = torch.arange(len(public))
    half = (count + 1) // 2
    snapshots = {}
    for slot in range(count):
        item = tape.preview_order[public, slot]
        sign = tape.preview_sign[public, item]
        left, right = menus(tape.axis[public, item], tape.cue[public, item], sign)
        action = net.act(rule, state, left, right, stochastic=False)
        state = action.state
        logits[rows, item] = action.motor_logit
        if slot + 1 == half:
            seen = torch.zeros_like(logits, dtype=torch.bool)
            seen.scatter_(1, tape.preview_order[public, :half], True)
            snapshots["half"] = (net.wait(rule, state, (count - half) * net.REACTION_TICKS),
                                 logits.masked_fill(~seen, torch.nan), seen)
    snapshots["all"] = (state, logits, torch.ones_like(logits, dtype=torch.bool))
    # A diagnostic external reader interrogates separate copies of the idle state.
    # The physical branch keeps the original idle state throughout.
    plus, minus = antipodal_logits(idle, tape, public)
    snapshots["blind"] = (idle, (plus - minus) / 2, torch.zeros_like(logits, dtype=torch.bool))
    return snapshots


def motor_logit(state, left, right):
    # Same reaction as net.act. The unused terminal traces and score are discarded.
    stimulus = net.native.sensory(state.graph, left, right)
    hidden = net.native.advance(state.graph, state.hidden, stimulus, net.REACTION_TICKS)
    return 2.5 * (hidden[:, 11] - hidden[:, 10])


def antipodal_logits(state, tape, public):
    plus, minus = [], []
    for item in range(tape.axis.shape[1]):
        axis, cue = tape.axis[public, item], tape.cue[public, item]
        left, right = cue - axis, cue + axis
        plus.append(motor_logit(state, left, right))
        minus.append(motor_logit(state, right, left))
    return torch.stack(plus, 1), torch.stack(minus, 1)


def probe(state, tape, public, law):
    plus, minus = antipodal_logits(state, tape, public)
    signs = tape.terminal_sign[public]
    odds = torch.where(signs > 0, plus, minus).sigmoid()
    side = tape.terminal_draw[public] < odds
    actual = side == (signs * law > 0)
    expected = ((plus * law).sigmoid() + (-minus * law).sigmoid()) / 2
    return expected, actual


def paid_probe(state, coeff, tape, public, law):
    rows = torch.arange(len(public))
    earned = torch.zeros((len(public), 2), dtype=torch.bool)
    for i in range(2):
        state = net.wait(coeff, state, tape.paid_wait[public, i], max_ticks=8)
        item = tape.paid_objects[public, i]
        sign = tape.terminal_sign[public, item]
        draws = tape.paid_draw[public, i]
        left, right = menus(tape.axis[public, item], tape.cue[public, item], sign)
        action = (net.act(coeff, state, left, right, draws[:, 0]) if i == 0
                  else final_motor.act(state, left, right, draws[:, 0]))
        ok = action.side == sign * law[rows, item]
        earned[rows, (item >= 3).long()] = ok
        if i == 0:
            experience = net.observe(coeff, action, left, right, torch.zeros_like(action.side))
            state = net.commit(coeff, experience, draws[:, 1:]).state
    return torch.column_stack((earned.all(1), earned))


def endpoint(value):
    value = value.float()
    return torch.column_stack((value.prod(1), value.mean(1), value))


def evaluate_panel(rule, coeff, snapshot, tape, public, law, joint, samples_per_fold):
    count = tape.axis.shape[1]
    n = len(public)
    samples = 2 * samples_per_fold
    if samples_per_fold != SAMPLES_PER_FOLD:
        sample_idx = torch.tensor([0, SAMPLES_PER_FOLD])  # smoke uses one repeat per fold
    else:
        sample_idx = torch.arange(samples)
    indices = torch.arange(n).repeat_interleave(samples)
    repeated = snapshot.subset(indices)
    expanded = rule.expand(len(indices))
    rep_world, rep_law = public[indices], law[indices]
    repeat = sample_idx.repeat(n)
    signs = tape.review_sign[rep_world, repeat]
    draws = tape.review_draw[rep_world, repeat]
    first = torch.arange(n) * samples
    clocks = cap.CONTACT_TICKS + POST_REVIEW_WAIT
    idle = net.wait(coeff, snapshot, clocks)
    idle_expected, idle_actual = probe(idle, tape, public, law)
    idle_paid = paid_probe(idle, coeff, tape, public, law) if joint else torch.empty((n, 0), dtype=torch.bool)
    pre, _ = probe(snapshot, tape, public, law)
    first_actual, folds, first_paid, first_contact, topology = [], [], [], [], []
    for item in range(count):
        left, right = menus(tape.axis[rep_world, item], tape.cue[rep_world, item], signs)
        feeder = joint and item == 5
        if feeder:
            left = right
        action = net.act(expanded, repeated, left, right, draws[:, 0])
        factor = rep_law[:, item] * signs
        taste = factor if feeder else action.side * factor
        experience = net.observe(expanded, action, left, right, taste)
        edit = net.commit(expanded, experience, draws[:, 1:])
        first_contact.append(torch.column_stack((action.side[first], taste[first])).to(torch.int8))
        changed = (edit.receipt.birth >= 0) | (edit.receipt.retired >= 0)
        topology.append(changed.float().reshape(n, 2, samples_per_fold).mean(2))
        actual, expected, paid = [], [], []
        for mode in ("active", "sham"):
            # Native sham holds graph and traces through the two commit wait segments.
            written = (edit.state if mode == "active" else
                       net.wait(expanded, experience.state, net.PRE_WRITE_TICKS + net.POST_WRITE_TICKS))
            waited = net.wait(expanded, written, POST_REVIEW_WAIT)
            prob, ok = probe(waited, tape, rep_world, rep_law)
            expected.append(endpoint(prob).reshape(n, 2, samples_per_fold, count + 2).mean(2))
            actual.append(endpoint(ok[first]))
            if joint:
                paid.append(paid_probe(waited.subset(first), coeff, tape, public, law))
        folds.append(torch.stack(expected, 1))
        first_actual.append(torch.stack(actual, 1))
        if joint:
            first_paid.append(torch.stack(paid, 1))
    folds = torch.stack(folds, 3)
    first_actual = torch.stack(first_actual, 2)
    first_paid = torch.stack(first_paid, 2) if joint else torch.empty((n, 2, count, 0), dtype=torch.bool)
    return {
        "fold": folds,
        "actual": first_actual,
        "paid": first_paid,
        "idle_expected": endpoint(idle_expected),
        "idle_actual": endpoint(idle_actual),
        "idle_paid": idle_paid,
        "prior_expected": endpoint(pre),
        "first_contact": torch.stack(first_contact, 1),
        "topology_rate": torch.stack(topology, 2),
    }


@torch.no_grad()
def evaluate(graph, rule, tape, joint, samples_per_fold=SAMPLES_PER_FOLD):
    worlds, maps, count = tape.law.shape
    public = torch.arange(worlds).repeat_interleave(maps)
    law = tape.law.reshape(-1, count)
    coeff = rule.expand(len(public))
    initial = begin(graph, coeff, tape, public, law, joint)
    result = {}
    for name, (state, confidence, seen) in preview(coeff, initial, tape, public).items():
        panel = evaluate_panel(rule, coeff, state, tape, public, law, joint, samples_per_fold)
        for key, value in {**panel, "logit": confidence, "visible": seen}.items():
            result[f"{name}/{key}"] = value.reshape(worlds, maps, *value.shape[1:]).numpy()
    return result


def verify_native(graph, rule, tape):
    coeff = rule.expand(len(tape.axis))
    state = net.start(graph, len(tape.axis))
    left, right = -tape.axis[:, 0], tape.axis[:, 0]
    contact = net.act(coeff, state, left, right, tape.support_draw[:, 0, 0])
    torch.testing.assert_close(motor_logit(state, left, right), contact.motor_logit, rtol=0, atol=0)
    experience = net.observe(coeff, contact, left, right, contact.side)
    exact = net.commit(coeff, experience, tape.support_draw[:, 0, 1:], intervention="sham").state
    skip = net.wait(coeff, experience.state, net.PRE_WRITE_TICKS + net.POST_WRITE_TICKS)
    for name in ("hidden", "sender_trace", "receiver_trace"):
        torch.testing.assert_close(getattr(exact, name), getattr(skip, name), rtol=0, atol=0)
    if exact.graph is not skip.graph:
        raise AssertionError("A no-write replay modified the original graph")


def run(cohort, body, assays, births, smoke=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    first, seed = COHORTS[cohort]
    births = list(range(first, first + 6)) if births is None else births
    protocol = digest(PLAN)
    signature_paths = (Path(__file__), Path(net.__file__), Path(net.native.__file__), Path(cap.__file__),
                       Path(six.__file__), Path(final_motor.__file__))
    implementation = "\n".join(f"{path.name} {digest(path)}" for path in signature_paths)
    path = DATA / ("smoke" if smoke else "sealed") / cohort / body
    path.mkdir(parents=True, exist_ok=True)
    location, arm, step = source(cohort, body)
    for assay in assays:
        assay_start = perf_counter()
        joint = assay == "joint6"
        worlds = 2 if smoke else ASSAYS[assay]
        chunk_size = 4 if joint else 32
        tape = make_tape(assay, seed, worlds)
        for birth in births:
            file = path / f"{assay}-{birth}.npz"
            checkpoint = fitting.checkpoint_path(location, "odor", arm, birth, step)
            identity = digest(checkpoint)
            if file.exists():
                with np.load(file) as cache:
                    if any(str(cache[k]) != v for k, v in (("plan_sha256", protocol), ("source_sha256", identity),
                                                          ("implementation", implementation))):
                        raise ValueError(f"Mismatched frozen replay cache {file}")
                continue
            graph, rule = fitting.load_checkpoint(location, "odor", arm, birth, step)
            verify_native(graph, rule, tape[:2])
            chunks = [evaluate(graph, rule, tape[i:i + chunk_size], joint, 1 if smoke else SAMPLES_PER_FOLD)
                      for i in range(0, worlds, chunk_size)]
            raw = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            raw.update({key: value.numpy() for key, value in vars(tape).items()})
            raw.update(source_sha256=identity, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                       plan_sha256=protocol, implementation=implementation, seed=seed)
            io._write_raw(file, **raw)
            print(cohort, body, assay, birth, round(perf_counter() - assay_start, 2), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--body", required=True, choices=("uniform", "mixed_repair"))
    parser.add_argument("--assay", choices=ASSAYS, nargs="+", default=list(ASSAYS))
    parser.add_argument("--births", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.cohort, args.body, args.assay, args.births, args.smoke)
