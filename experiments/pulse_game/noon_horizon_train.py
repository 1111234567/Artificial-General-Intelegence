"""Train a native route decision from two realized physical planning returns."""

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_horizon as env


PLAN = env.EVIDENCE / "horizon-plan.json"
ARMS = ("informative", "uninformative")


def load_plan():
    plan = json.loads(PLAN.read_text(encoding="utf8"))
    for cohort, signature in plan["calibration_sha256"].items():
        if env.digest(env.EVIDENCE / f"horizon-calibration-{cohort}.json") != signature:
            raise ValueError("The conditions used to choose the physical planning period changed")
    return plan


def inherited(cohort):
    births = env.COHORTS[cohort]
    values = [env.load(cohort, birth) for birth in births]
    weight = torch.nn.Parameter(torch.cat([g.weights for g, _ in values]))
    bias = torch.nn.Parameter(torch.cat([g.bias for g, _ in values]))
    present = torch.cat([g.present for g, _ in values])
    rule = env.net.Rule(births)
    rule.load_state_dict({key: torch.cat([r.state_dict()[key] for _, r in values]) for key in rule.state_dict()})
    return env.net.Graph(weight, bias, present), rule


def objective(graph, rule, tape, missing, future, arm, length):
    state, _ = env.initial(graph, rule, tape, missing)
    ids = torch.arange(tape.lanes).repeat_interleave(2)
    long = (torch.arange(len(ids)) % 2) == 1
    assigned = (2 * long.float() - 1)
    cue = assigned if arm == "informative" else assigned * future.cue_noise[ids]
    twin = env.lesson.expand(tape, ids)
    view = rule.expand(twin.lanes // len(graph.bias))
    action = env.signal(view, state.subset(ids), twin, cue)
    scores = []
    for i, horizon in enumerate((0, length)):
        index = torch.arange(i, twin.lanes, 2)
        rewards, _ = env.branches(graph,rule,action.state.subset(index),tape,future,horizon)
        scores.append(rewards[..., 1].sum(-1))
    scores = torch.stack(scores, dim=1).flatten(0,1)
    probability = action.motor_logit.sigmoid()
    credit = probability * (scores[:, 1] - scores[:, 0])
    loss = -credit.reshape(len(graph.bias), -1).mean(1).sum()
    with torch.no_grad():
        actual = scores[torch.arange(twin.lanes), (action.side > 0).long()]
        chance = scores.mean(1)
        omitted_right = twin.candidates[:, 1] == missing[ids]
        select_unknown = torch.where(omitted_right, probability, 1-probability)
        raw = torch.stack((select_unknown, actual, chance, actual-chance), 1).reshape(len(graph.bias), -1, 2, 4)
        metrics = raw.mean(1)
    return loss, metrics


def update(graph, rule, optimizer, tape, missing, future, arm, length):
    optimizer.zero_grad(set_to_none=True)
    loss, outcomes = objective(graph, rule, tape, missing, future, arm, length)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite route objective")
    loss.backward()
    parameters = (graph.weights, graph.bias, *rule.parameters())
    norm = env.storage._gradients(parameters, 2.)
    if norm is None:
        raise FloatingPointError("Nonfinite native route gradient")
    optimizer.step()
    return outcomes, norm


def fit(cohort, arm, benchmark=None):
    plan = load_plan()
    roots = env.COHORTS[cohort]
    seed = plan["training_seeds"][cohort]
    worlds = plan["geometries_per_update"]
    length = plan["paid_tail_length"]
    stages, stop = tuple(plan["checkpoints"]), plan["final_update"]
    cutoff = datetime.fromisoformat(plan["compute_deadline_beijing"] + "+08:00")
    if not (stages[0] == 0 and stages[-1] == stop):
        raise ValueError("Training must end at a registered checkpoint")
    graph, rule = inherited(cohort)
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=plan["lr"])
    directory = env.DATA / "fit" / cohort / arm
    meta = dict(cohort=cohort, arm=arm, births=list(roots), plan_sha256=env.digest(PLAN),
                initial={str(b): env.digest(env.checkpoint(cohort,b)) for b in roots},
                code={Path(p).name: env.digest(p) for p in (__file__, env.__file__, env.lesson.__file__, env.world.__file__,
                                                            env.net.__file__, env.net.native.__file__)})
    step, trace = 0, []
    if benchmark is None:
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "run.json"
        if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf8")) != meta:
            raise ValueError("Training provenance changed")
        if not meta_path.exists():
            env.storage._write_json(meta_path, meta)
        optimizer_file = directory / "resume.pt"
        if optimizer_file.exists():
            loaded = torch.load(optimizer_file, map_location="cpu", weights_only=True)
            if loaded["plan_sha256"] != meta["plan_sha256"]:
                raise ValueError("Training checkpoint changed plan")
            with torch.no_grad():
                graph.weights.copy_(loaded["weights"])
                graph.bias.copy_(loaded["bias"])
            rule.load_state_dict(loaded["rule"])
            optimizer.load_state_dict(loaded["optimizer"])
            step, trace = loaded["step"], loaded["trace"].tolist()

    def save(step, trace):
        if step in stages:
            for row, birth in enumerate(roots):
                env.weights.save_checkpoint(env.DATA/"fit"/cohort,"odor",arm,birth,step,graph,rule,row)
        state = dict(step=step, weights=graph.weights.detach(), bias=graph.bias.detach(), rule=rule.state_dict(),
                     optimizer=optimizer.state_dict(), trace=torch.tensor(trace), plan_sha256=meta["plan_sha256"])
        env.storage._atomic(optimizer_file, lambda f: torch.save(state, f))

    if benchmark is None and not optimizer_file.exists():
        save(step, trace)
    before = step
    now = perf_counter()
    target = stop if benchmark is None else benchmark
    for step in range(before+1, target+1):
        domain = seed if benchmark is None else seed+8131
        tape, missing = env.lesson.prepare(domain, roots, step, worlds, 2, omissions=True)
        future = env.opportunities(domain, roots, step, worlds, length)
        outcomes, norm = update(graph, rule, optimizer, tape, missing, future, arm, length)
        if benchmark is None:
            trace.append(torch.cat((outcomes.flatten(1), norm[:, None]), dim=1).tolist())
            if step % 250 == 0 or step in stages:
                save(step, trace)
                rolling = np.asarray(trace[-250:])[:, :, :8].mean((0, 1)).reshape(2, 4).round(3).tolist()
                print(cohort, arm, step, round((perf_counter()-now)/(step-before), 3), rolling, flush=True)
                if datetime.now(cutoff.tzinfo) >= cutoff and step < stop:
                    print(cohort, arm, "compute cutoff", step, flush=True)
                    break
    if benchmark is not None:
        print(cohort, arm, benchmark, round((perf_counter()-now)/benchmark, 4), flush=True)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--cohort", choices=env.COHORTS, required=True)
    cli.add_argument("--arm", choices=ARMS, required=True)
    cli.add_argument("--benchmark", type=int)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fit(**vars(args))
