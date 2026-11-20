"""Matched outcome-only continuation with informative versus unrelated physical previews."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as score
from . import noon_review_preview as previewing
from . import noon_review_train as core
from . import noon_review_world as world


WARM_STAGE = 8000
STAGES = (0, 1000, 2000, 4000, 8000)
SEED = 76202433
CONFIG = core.Config(lr=.001)
METRICS = ("accuracy", "paired_twins_accuracy", "selection_right", "selection_loss", "gradient_norm", "recovered")


def load_warm(source, cohort):
    births = world.COHORTS[cohort]
    ancestry = [world.load_frozen(cohort, "native", birth, WARM_STAGE, path=source) for birth in births]
    weights, bias, support = [torch.cat([getattr(graph, field) for graph, _ in ancestry])
                              for field in ("weights", "bias", "present")]
    graph = native.Graph(nn.Parameter(weights), nn.Parameter(bias), support)
    rule = native.Rule(births)
    rule.load_state_dict({name: torch.cat([row.state_dict()[name] for _, row in ancestry])
                          for name in rule.state_dict()})
    return graph, rule


def objective(graph, rule, tape, preview, *, informative, retention, pairs):
    view = rule.expand(pairs * 2)
    state, contacts = world.introduce(view, native.start(graph, pairs * 2), tape)
    select_score = side = torch.zeros_like(tape.decision_draw, dtype=state.graph.bias.dtype)
    if not retention:
        state, _ = previewing.sweep(view, state, tape, preview, informative)
        selected = world.select(view, state, *world.candidate_menu(tape), tape.decision_draw)
        state, contact = world.review(view, selected.state, tape, selected.side)
        contacts.append(contact)
        select_score = selected.motor_log_prob
        side = (selected.side > 0).float()
    final, truth = world.final_probe(view, state, tape, tape.final_item)

    def paired(items):
        return items.reshape(len(graph.bias), pairs, 2)

    actual = paired((final.side * truth).detach())
    advantage = score.leave_pair_out_advantage(actual)
    actions = (final.motor_log_prob, sum(row.motor.motor_log_prob for row in contacts),
               sum(row.edit.edit_log_prob for row in contacts), select_score)
    losses = [score.score_loss(paired(choice), advantage) for choice in actions]
    with torch.no_grad():
        correct = actual > 0
        metrics = torch.stack((correct.float().mean((1, 2)), correct.all(2).float().mean(1),
                               paired(side).mean((1, 2)), losses[-1]), dim=1)
    return sum(losses).sum(), metrics


def update(graph, rule, optimizer, tape, preview, *, informative, retention):
    def backward(g, r):
        loss, metrics = objective(g, r, tape, preview, informative=informative, retention=retention, pairs=CONFIG.pairs)
        if not loss.isfinite():
            return None, metrics
        loss.backward()
        return archive._gradients((g.weights, g.bias, *r.parameters()), CONFIG.grad_clip), metrics

    optimizer.zero_grad(set_to_none=True)
    norm, metrics = backward(graph, rule)
    recovered = norm is None
    if recovered:
        exact = native.Graph(nn.Parameter(graph.weights.detach().double()),
                             nn.Parameter(graph.bias.detach().double()), graph.present)
        exact_rule = deepcopy(rule).double()
        norm, metrics = backward(exact, exact_rule)
        if norm is None:
            raise FloatingPointError("Invalid preview update after float64 world replay")
        for target, original in zip((graph.weights, graph.bias, *rule.parameters()),
                                    (exact.weights, exact.bias, *exact_rule.parameters()), strict=True):
            target.grad = None if original.grad is None else original.grad.to(target.dtype)
    optimizer.step()
    return torch.cat((metrics.double(), norm[:, None], torch.full_like(norm[:, None], recovered)), dim=1)


def fit(path, source, cohort, arm, target, manifest_sha256):
    torch.set_num_threads(1)
    graph, rule = load_warm(source, cohort)
    births = world.COHORTS[cohort]
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=CONFIG.lr)
    output = path / cohort / "training" / arm
    resume = output / "resume.pt"
    metrics = torch.empty((len(births), target, len(METRICS)), dtype=torch.float64)
    start = 0
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (state["cohort"], state["arm"], state["manifest_sha256"]) != (cohort, arm, manifest_sha256):
            raise ValueError(f"Unexpected preview run identity at {resume}")
        start = state["step"]
        if start >= target:
            if not all(core.checkpoint_path(path, cohort, arm, birth, target).is_file() for birth in births):
                raise FileNotFoundError("Preview stage was missing from previously saved run")
            return
        with torch.no_grad():
            for key in ("weights", "bias", "present"):
                getattr(graph, key).copy_(state[key])
        rule.load_state_dict(state["rule"])
        optimizer.load_state_dict(state["optimizer"])
        metrics[:, :start] = state["metrics"]

    def save(step):
        for lane, birth in enumerate(births):
            score.save_checkpoint(path / cohort, "odor", arm, birth, step, graph, rule, lane)
        payload = dict(cohort=cohort, arm=arm, manifest_sha256=manifest_sha256, step=step,
                       weights=graph.weights.detach(), bias=graph.bias.detach(), present=graph.present,
                       rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics[:, :step].clone())
        archive._atomic(output / f"optimizer-{step:06d}.pt", lambda tmp: torch.save(payload, tmp))
        archive._atomic(resume, lambda tmp: torch.save(payload, tmp))
        archive._write_raw(output / f"trace-{step:06d}.npz", births=births, keys=METRICS, values=metrics[:, :step].numpy())

    if start == 0:
        save(0)
    timer = perf_counter()
    for step in range(start + 1, target + 1):
        retention, count = core.practice_step(step, CONFIG)
        tape = world.make_tape(SEED, CONFIG.pairs, count, births=births, update=step)
        preview = None if retention else previewing.make_preview(SEED, CONFIG.pairs, births=births, step=step)
        metrics[:, step - 1] = update(graph, rule, optimizer, tape, preview,
                                     informative=(arm == "informative"), retention=retention)
        if step in STAGES:
            save(step)
        if step % 250 == 0:
            results = metrics[:, max(step - 128, start):step, 0]
            speed = (step - start) / (perf_counter() - timer)
            print(f"preview {cohort} {arm} {step}: accuracy {results.mean().item():.4f}; {speed:.2f} updates/s", flush=True)


def manifest(path, source):
    modules = tuple(Path(name) for name in (__file__, previewing.__file__, core.__file__, world.__file__, native.__file__))
    inputs = {cohort: {birth: prior.digest(core.checkpoint_path(source, cohort, "native", birth, WARM_STAGE))
                       for birth in births} for cohort, births in world.COHORTS.items()}
    plan = dict(config=asdict(CONFIG), source_stage=WARM_STAGE, source_checkpoints_sha256=inputs,
                source_manifest_sha256=prior.digest(source / "run.json"), train_seed=SEED, arms=previewing.FIT_ARMS,
                stages=STAGES, metric=METRICS, heldout_seeds=previewing.FIT_SEEDS,
                freeze_date="2026-11-20", source_sha256={file.name: prior.digest(file) for file in modules},
                protocol_sha256=prior.digest(world.EVIDENCE / "noon-review-preview-plan.md"))
    dest = path / "run.json"
    if dest.is_file():
        if json.loads(dest.read_text(encoding="utf8")) != json.loads(json.dumps(plan)):
            raise ValueError(f"Preview resume would change the fixed protocol at {dest}")
    else:
        archive._write_json(dest, plan)
    return prior.digest(dest)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=world.DATA / "core")
    p.add_argument("--out", type=Path, default=previewing.FIT_PATH)
    p.add_argument("--until", type=int, choices=STAGES, required=True)
    args = p.parse_args()
    digest = manifest(args.out, args.source)
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fit, args.out, args.source, cohort, arm, args.until, digest)
                for cohort in world.COHORTS for arm in previewing.FIT_ARMS]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
