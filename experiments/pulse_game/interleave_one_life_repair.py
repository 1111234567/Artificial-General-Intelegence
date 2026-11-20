"""Assign credit to short or long actual outcomes across the same experienced lives."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_one_life as prior
from . import local_trace_graph as net, revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


ARMS = ("long_only", "mixed")
COHORTS = ("original", "independent")
STAGES = (500, 1000, 1500, 2000)
DEST = DATA / "one-life-repair"
PLAN = EVIDENCE / "one-life-repair-plan.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def setup(cohort):
    original = cohort == "original"
    start = 731 if original else 811
    return list(range(start, start + 6)), 466219 if original else 466237


def inherited(cohort, births):
    parents = [prior.tri.old_fit.load_checkpoint(prior.DEST / cohort, "odor", "interspersed", b, 8000)
               for b in births]
    weights, bias, present = (torch.cat([getattr(g, k) for g, _ in parents]) for k in cap.GRAPH_FIELDS)
    graph = net.Graph(torch.nn.Parameter(weights), torch.nn.Parameter(bias), present)
    rule = net.Rule(births)
    rule.load_state_dict({k: torch.cat([r.state_dict()[k] for _, r in parents]) for k in rule.state_dict()})
    return graph, rule


def short_ticket(seed, birth, step):
    world, query = np.random.SeedSequence([427719, seed, birth, step]).spawn(2)
    data = cap.make_tape("odor", int(np.random.default_rng(world).integers(2**31)), 8, 1)
    ids = torch.from_numpy(np.random.default_rng(query).integers(2, size=8)).repeat_interleave(4)
    rows = torch.arange(data.lanes)
    return dict(left=data.left[:, :, 0], right=data.right[:, :, 0], truth=data.right_value[:, :, 0],
                draw=data.motor_draw[:, :, 0], edit=data.edit_draw[:, :, 0],
                query_left=data.left[rows, ids, -1], query_right=data.right[rows, ids, -1],
                query_draw=data.motor_draw[rows, ids, -1], query_truth=data.right_value[rows, ids, -1])


def short_life(graph, rule, tape):
    state = net.start(graph, 32)
    rule = rule.expand(32)
    score = 0.
    for i in range(2):
        left, right = tape["left"][:, i], tape["right"][:, i]
        action = net.act(rule, state, left, right, tape["draw"][:, i])
        taste = action.side * tape["truth"][:, i]
        experience = net.observe(rule, action, left, right, taste)
        result = net.commit(rule, experience, tape["edit"][:, i])
        state = net.wait(rule, result.state, 4)
        score += action.motor_log_prob + result.edit_log_prob
    choice = motor.act(state, tape["query_left"], tape["query_right"], tape["query_draw"])
    return score + choice.motor_log_prob, (choice.side * tape["query_truth"] > 0)


def long_life(graph, rule, tape):
    score, resources, _, _ = prior.perform(graph, rule, tape)
    return score, (resources > 0).all(1)


def join(tapes):
    return {key: torch.cat([t[key] for t in tapes]) for key in tapes[0]}


def update(graph, rule, optimizer, credited, tape, worlds):
    ordered = (graph.weights, graph.bias, *rule.parameters())
    for precise in (False, True):
        optimizer.zero_grad(set_to_none=True)
        if precise:
            model = net.Graph(torch.nn.Parameter(graph.weights.detach().double()),
                              torch.nn.Parameter(graph.bias.detach().double()), graph.present)
            local = deepcopy(rule).double()
        else:
            model, local = graph, rule
        score, outcomes = credited(model, local, tape)
        earned = outcomes.float().reshape(len(graph.bias), worlds, -1)
        means = earned.mean(2, keepdim=True)
        advantage = earned - (means.sum(1, keepdim=True) - means) / (worlds - 1)
        loss = -(score.reshape_as(earned) * advantage).mean((1, 2)).sum()
        if not loss.isfinite():
            continue
        loss.backward()
        origin = (model.weights, model.bias, *local.parameters())
        norms = prior.tri.revisit._gradients(origin, 2.)
        if norms is None:
            continue
        if precise:
            for p, q in zip(ordered, origin, strict=True):
                p.grad = None if q.grad is None else q.grad.to(p.dtype)
        optimizer.step()
        return earned.mean((1, 2)), norms, precise
    raise FloatingPointError("Nonfinite gradient in paired repair")


def run(cohort, arm):
    torch.set_num_threads(1)
    births, seed = setup(cohort)
    path = DEST / cohort
    folder = path / arm
    folder.mkdir(parents=True, exist_ok=True)
    old = prior.tri.old_fit
    files = (Path(__file__), Path(prior.__file__), Path(prior.env.__file__), Path(cap.__file__),
             Path(net.__file__), Path(motor.__file__), Path(net.native.__file__))
    manifest = dict(seed=seed, arm=arm, births=births, stages=list(STAGES), lr=.003,
                    short_worlds=8, long_worlds=4, plan_sha256=digest(PLAN),
                    numpy=np.__version__, torch=str(torch.__version__),
                    source_sha256={p.name: digest(p) for p in files},
                    parent_sha256={str(b): digest(old.checkpoint_path(prior.DEST / cohort, "odor", "interspersed", b, 8000))
                                   for b in births})
    lock = folder / "run.json"
    if lock.exists() and json.loads(lock.read_text()) != manifest:
        raise ValueError("Repair protocol or parent changed")
    prior.tri.revisit._write_json(lock, manifest)
    graph, rule = inherited(cohort, births)
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=manifest["lr"])
    step = 0
    trace = []
    resume = folder / "resume.pt"
    key = digest(lock)
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["manifest"] != key:
            raise ValueError("Resume data does not match the repair protocol")
        with torch.no_grad():
            for field in cap.GRAPH_FIELDS:
                getattr(graph, field).copy_(saved["graph"][field])
        rule.load_state_dict(saved["rule"])
        optimizer.load_state_dict(saved["optimizer"])
        step, trace = saved["step"], saved["trace"].tolist()
    def save():
        if step in STAGES:
            for i, b in enumerate(births):
                old.save_checkpoint(path, "odor", arm, b, step, graph, rule, i)
        payload = dict(manifest=key, step=step, trace=torch.as_tensor(trace),
                       graph={k: getattr(graph, k).detach() for k in cap.GRAPH_FIELDS},
                       rule=rule.state_dict(), optimizer=optimizer.state_dict())
        prior.tri.revisit._atomic(resume, lambda dest: torch.save(payload, dest))

    start, began = step, perf_counter()
    for step in range(start + 1, STAGES[-1] + 1):
        block, slot = divmod(step - 1, 2)
        short_is_credited = arm == "mixed" and slot == np.random.default_rng([903911, seed, block]).integers(2)
        short = join([short_ticket(seed, b, step) for b in births])
        long = join([prior.ticket(seed, b, step, 4, "interspersed", domain=955103) for b in births])
        with torch.no_grad():
            idle = (long_life(graph, rule, long) if short_is_credited else short_life(graph, rule, short))[1]
        if short_is_credited:
            short_rate, norms, precise = update(graph, rule, optimizer, short_life, short, 8)
            long_rate = idle.reshape(len(births), 32).float().mean(1)
        else:
            long_rate, norms, precise = update(graph, rule, optimizer, long_life, long, 4)
            short_rate = idle.reshape(len(births), 32).float().mean(1)
        trace.append(torch.stack((short_rate, long_rate, norms,
                                  torch.full_like(norms, short_is_credited), torch.full_like(norms, precise)), 1).tolist())
        if step % 50 == 0:
            print(cohort, arm, step, "seconds/update", round((perf_counter() - began) / (step - start), 4), flush=True)
        if step % 100 == 0:
            save()
    credited = np.array(trace)[:, :, 3].sum(axis=0)
    if not (credited == (1000 if arm == "mixed" else 0)).all():
        raise AssertionError("Wrong number of credited short tickets")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    run(args.cohort, args.arm)
