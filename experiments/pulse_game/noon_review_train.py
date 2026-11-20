"""Continue the same 20-node graph on native, random, and frozen-parent review selection."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as previous
from . import noon_review_world as world
from . import self_param_writer as body


METRICS = ("terminal_accuracy", "terminal_twins", "terminal_score_loss", "collection_score_loss",
           "write_score_loss", "selector_score_loss", "review_right", "review_oldest",
           "earned_review_taste", "review_write", "gradient_norm", "float64_recovery")


@dataclass(frozen=True)
class Config:
    pairs: int = 16
    lr: float = .002
    grad_clip: float = 2.
    retention_interval: int = 4


def load_cohort(cohort):
    births = world.COHORTS[cohort]
    parents = [world.load_parent(cohort, birth) for birth in births]
    values = [torch.cat([getattr(graph, field) for graph, _ in parents])
              for field in ("weights", "bias", "present")]
    graph = native.Graph(nn.Parameter(values[0]), nn.Parameter(values[1]), values[2])
    rule = native.Rule(births)
    rule.load_state_dict({name: torch.cat([parent.state_dict()[name] for _, parent in parents])
                          for name in rule.state_dict()})
    return graph, rule


def practice_step(step, config):
    retention = step % config.retention_interval == 0
    return retention, 2 if retention and (step // config.retention_interval) % 2 else 4


def objective(graph, rule, tape, pairs, arm, *, retention=False, external_side=None):
    state = native.start(graph, 2 * pairs)
    rule = rule.expand(2 * pairs)
    state, contacts = world.introduce(rule, state, tape)
    lanes = torch.zeros_like(tape.decision_draw, dtype=state.graph.bias.dtype)
    selection = right = oldest = taste = written = lanes

    if not retention:
        choice = world.select(rule, state, *world.candidate_menu(tape), tape.decision_draw)
        if arm == "native":
            side = choice.side
            selection = choice.motor_log_prob
        elif arm == "random":
            side = world.random_side(tape)
        else:
            if external_side is None:
                raise ValueError("Frozen-parent arm requires its own comparison selection")
            side = external_side
        state, contact = world.review(rule, choice.state, tape, side)
        contacts.append(contact)
        right = (side > 0).float()
        oldest = (world.selected_item(tape, side) == tape.candidates.min(1).values).float()
        taste = contact.earned
        receipt = contact.edit.receipt
        written = ((receipt.birth >= 0) | (receipt.retired >= 0) | receipt.bias_delta.ne(0).any(1)
                   | receipt.weight_delta.flatten(1).ne(0).any(1)).float()

    final, truth = world.final_probe(rule, state, tape, tape.final_item)

    def paired(values):
        return values.reshape(len(graph.bias), pairs, 2)

    terminal = paired((final.side * truth).detach())
    advantage = previous.leave_pair_out_advantage(terminal)
    collection_score = sum(contact.motor.motor_log_prob for contact in contacts)
    edit_score = sum(contact.edit.edit_log_prob for contact in contacts)
    losses = [previous.score_loss(paired(score), advantage) for score in
              (final.motor_log_prob, collection_score, edit_score, selection)]
    with torch.no_grad():
        correct = terminal > 0
        metrics = torch.stack((correct.float().mean((1, 2)), correct.all(2).float().mean(1), *losses,
                               *(paired(value).mean((1, 2)) for value in (right, oldest, taste, written))), dim=1)
    return sum(losses).sum(), metrics


def fit_step(graph, rule, optimizer, tape, arm, config, *, retention, external_side):
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer.zero_grad(set_to_none=True)

    def backward(g, r):
        loss, metrics = objective(g, r, tape, config.pairs, arm, retention=retention,
                                  external_side=external_side)
        if not loss.isfinite():
            return metrics, None
        loss.backward()
        return metrics, archive._gradients((g.weights, g.bias, *r.parameters()), config.grad_clip)

    metrics, norms = backward(graph, rule)
    recovered = norms is None
    if recovered:
        precise_graph = native.Graph(nn.Parameter(graph.weights.detach().double()),
                                     nn.Parameter(graph.bias.detach().double()), graph.present)
        precise_rule = deepcopy(rule).double()
        metrics, norms = backward(precise_graph, precise_rule)
        if norms is None:
            raise FloatingPointError("Review step became nonfinite on same-world float64 replay")
        for dest, source in zip(parameters, (precise_graph.weights, precise_graph.bias,
                                             *precise_rule.parameters()), strict=True):
            dest.grad = None if source.grad is None else source.grad.to(dest.dtype)
    optimizer.step()
    return torch.cat((metrics.double(), norms[:, None], torch.full_like(norms[:, None], recovered)), dim=1)


def checkpoint_path(path, cohort, arm, birth, step):
    return previous.checkpoint_path(path / cohort, "odor", arm, birth, step)


def fit_job(path, cohort, arm, target, config, manifest_digest):
    torch.set_num_threads(1)
    births = world.COHORTS[cohort]
    graph, rule = load_cohort(cohort)
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=config.lr)
    directory = path / cohort / "training" / arm
    resume = directory / "resume.pt"
    start = 0
    sample = torch.empty((len(births), target, len(METRICS)), dtype=torch.float64)
    if resume.is_file():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved["manifest_sha256"], saved["cohort"], saved["arm"]) != (manifest_digest, cohort, arm):
            raise ValueError(f"Run identity changed at {resume}")
        start = saved["step"]
        if start >= target:
            if not all(checkpoint_path(path, cohort, arm, birth, target).is_file() for birth in births):
                raise FileNotFoundError(f"Resume is beyond incomplete checkpoint {target} at {resume}")
            return
        with torch.no_grad():
            graph.weights.copy_(saved["weights"])
            graph.bias.copy_(saved["bias"])
            graph.present.copy_(saved["present"])
        rule.load_state_dict(saved["rule"])
        optimizer.load_state_dict(saved["optimizer"])
        sample[:, :start] = saved["metrics"]

    def save(step):
        for lane, birth in enumerate(births):
            previous.save_checkpoint(path / cohort, "odor", arm, birth, step, graph, rule, lane)
        payload = dict(manifest_sha256=manifest_digest, cohort=cohort, arm=arm, step=step,
                       weights=graph.weights.detach(), bias=graph.bias.detach(), present=graph.present,
                       rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=sample[:, :step].clone())
        archive._atomic(directory / f"optimizer-{step:06d}.pt", lambda tmp: torch.save(payload, tmp))
        archive._atomic(resume, lambda tmp: torch.save(payload, tmp))
        archive._write_raw(directory / f"trace-{step:06d}.npz", births=births, keys=METRICS,
                           values=sample[:, :step].numpy())

    if start == 0:
        save(0)
    parent, parent_rule = None, None
    if arm == "uncertain":
        parent, frozen = load_cohort(cohort)
        parent = parent.detach()
        parent_rule = frozen.requires_grad_(False).expand(2 * config.pairs)

    started = perf_counter()
    for step in range(start + 1, target + 1):
        retention, count = practice_step(step, config)
        tape = world.make_tape(world.TRAIN_SEED, config.pairs, count, births=births, update=step)
        external_side = None
        if not retention and parent is not None:
            with torch.no_grad():
                parent_state, _ = world.introduce(parent_rule, native.start(parent, 2 * config.pairs), tape)
                external_side, _ = world.parent_uncertainty(parent_rule, parent_state, tape)
        sample[:, step - 1] = fit_step(graph, rule, optimizer, tape, arm, config, retention=retention,
                                       external_side=external_side)
        if step in world.CHECKPOINTS:
            save(step)
        if step in world.CHECKPOINTS or step % 250 == 0:
            left = max(step - 128, start)
            recent = sample[:, left:step]
            review = torch.as_tensor([index % config.retention_interval != 0 for index in range(left + 1, step + 1)])
            good, side, older = (recent[:, review, i].mean().item() for i in (0, 6, 7))
            speed = (step - start) / (perf_counter() - started)
            print(f"{cohort} {arm} {step}: review accuracy {good:.4f}, right {side:.3f}, "
                  f"oldest {older:.3f}; {speed:.2f} updates/s", flush=True)


def manifest(path, config):
    protocol = world.EVIDENCE / "noon-review-protocol.md"
    parent_digests = {cohort: {birth: prior.digest(previous.checkpoint_path(world.parent_path(cohort),
                                      "odor", "uniform", birth, world.PARENT_STAGE)) for birth in births}
                      for cohort, births in world.COHORTS.items()}
    modules = (__file__, world.__file__, native.__file__, previous.__file__, body.__file__, archive.__file__)
    record = dict(config=asdict(config), training_seed=world.TRAIN_SEED, training_domain=world.TRAIN_DOMAIN,
                  eval_domain=world.EVAL_DOMAIN, eval_seeds=world.EVAL_SEEDS, cohorts=world.COHORTS,
                  arms=world.ARMS, checkpoints=world.CHECKPOINTS, protocol_sha256=prior.digest(protocol),
                  parent_sha256=parent_digests, source_sha256={Path(name).name: prior.digest(Path(name))
                                                              for name in modules}, metrics=METRICS)
    destination = path / "run.json"
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf8")) != json.loads(json.dumps(record)):
            raise ValueError(f"Declared protocol changed at {destination}")
    else:
        archive._write_json(destination, record)
    return prior.digest(destination)


def run(path, target, config=Config(), *, cohorts=tuple(world.COHORTS), arms=world.ARMS, workers=6):
    if target not in world.CHECKPOINTS or workers < 1 or config.pairs < 2:
        raise ValueError("Invalid worker count, matched-pair count, or checkpoint target")
    torch.set_num_threads(1)
    digest = manifest(path, config)
    jobs = [(cohort, arm) for cohort in cohorts for arm in arms]
    if len(jobs) == 1:
        cohort, arm = jobs[0]
        fit_job(path, cohort, arm, target, config, digest)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(fit_job, path, cohort, arm, target, config, digest) for cohort, arm in jobs]
        for future in futures:
            future.result()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=world.DATA / "core")
    parser.add_argument("--until", type=int, choices=world.CHECKPOINTS, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=world.COHORTS, default=list(world.COHORTS))
    parser.add_argument("--arms", nargs="+", choices=world.ARMS, default=world.ARMS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.out, args.until, cohorts=args.cohorts, arms=args.arms, workers=args.workers)


if __name__ == "__main__":
    main()
