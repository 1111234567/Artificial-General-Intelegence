"""Retune only the 17 shared coefficients using one actual continuous life per world."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_repeated as life
from . import noon_review_world as world
from . import noon_ticket as parents


SEED = 76202517
WORLDS = 8
STAGES = (800, 1600)
LR = .003
METRICS = ("total", "review", "seen", "new", "final_seen", "later", "readout_loss", "divergence_from_fair")


def credit(actual, arm):
    if arm not in life.FIT_ARMS:
        raise ValueError(f"Unknown credit condition {arm}")
    last = actual["final"]
    review, seen, new = actual["earned"].unbind(dim=2)
    by_turn = seen + new + (review if arm == "total" else 0)
    subsequent = by_turn.flip(1).cumsum(1).flip(1) + last[:, None]
    scale = 3 * torch.arange(life.ROUNDS, 0, -1) + 1
    return subsequent.double() / scale


def run_step(g, rule, model, optimizer, tape, arm, worlds=WORLDS):
    actual = life.run_life(g, rule, tape, "learned", model)
    chance = actual["policy_probability_right"]
    targets = credit(actual, arm).reshape(len(g.bias), worlds, 2, life.ROUNDS)
    means = targets.mean(2, keepdim=True)
    baseline = (means.sum(1, keepdim=True) - means) / (worlds - 1)
    centered = (targets - baseline).reshape_as(chance).detach()
    logp = torch.where(actual["selected_side"] > 0, chance, 1 - chance).clamp_min(1e-14).log()
    divergence = (chance * (2 * chance).clamp_min(1e-12).log()
                  + (1 - chance) * (2 * (1 - chance)).clamp_min(1e-12).log()).mean()
    regularization = torch.cat([param.flatten() for param in model.parameters()]).square().mean()
    loss = -(logp * centered).mean() + .002 * divergence + .0001 * regularization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(torch.isfinite(value.grad).all() for value in model.parameters()):
        raise FloatingPointError("Nonfinite online readout gradient")
    optimizer.step()
    by_phase = life.resources(actual)
    return [float(by_phase[key].mean()) for key in METRICS[:-2]] + [float(loss.detach()), float(divergence.detach())]


def fit_manifest(target):
    births = world.COHORTS["original"]
    warm_sha = {birth: prior.digest(world.checkpoints.checkpoint_path(world.parent_path("original"), "odor", "uniform",
                                                                       birth, world.PARENT_STAGE)) for birth in births}
    sources = (__file__, life.__file__, life.preview.__file__, life.readouts.__file__, world.__file__)
    plan = dict(target=target, maximum=STAGES[-1], stages=STAGES, worlds_per_birth=WORLDS, seed=SEED,
                domain=life.TRAIN_DOMAIN, births=births, coefficients=17, lr=LR, arms=life.FIT_ARMS,
                metrics=METRICS, original_readout_sha256=prior.digest(life.SOURCE), warm_sha256=warm_sha,
                source_sha256={Path(name).name: prior.digest(Path(name)) for name in sources},
                plan_sha256=prior.digest(life.PLAN), heldout=life.SEEDS["trained"])
    path = life.DATA / "fit" / "run.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf8")) != json.loads(json.dumps(plan)):
            raise ValueError(f"Matched-fit protocol differs at {path}")
    else:
        archive._write_json(path, plan)
    return prior.digest(path)


def benchmark(updates):
    torch.set_num_threads(1)
    g, rule = parents.inherited("original")
    g, rule = g.detach(), rule.requires_grad_(False)
    model = life.controller()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    saved = [p.detach().clone() for p in model.parameters()]
    # Explicitly disjoint runtime benchmark domain. These weights are discarded.
    start = perf_counter()
    for i in range(1, updates + 1):
        tape = life.make_tape(76202529, WORLDS, births=world.COHORTS["original"], step=i)
        _ = run_step(g, rule, model, optimizer, tape, "total")
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters(), strict=True))
    print(f"{updates} actual updates in {perf_counter()-start:.3f} seconds", flush=True)


def fit(arm, target, manifest):
    torch.set_num_threads(1)
    g, rule = parents.inherited("original")
    g, rule = g.detach(), rule.requires_grad_(False)
    model = life.controller()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    root = life.DATA / "fit" / arm
    resume = root / "resume.pt"
    start, trace = 0, []
    if resume.is_file():
        history = torch.load(resume, map_location="cpu", weights_only=True)
        if history["manifest_sha256"] != manifest or history["arm"] != arm:
            raise ValueError("Optimizer does not belong to this independent fit arm")
        model.load_state_dict(history["weights"])
        optimizer.load_state_dict(history["optimizer"])
        start, trace = history["step"], history["trace"].tolist()

    begin = perf_counter()
    for step in range(start + 1, target + 1):
        tape = life.make_tape(SEED, WORLDS, births=world.COHORTS["original"], step=step)
        trace.append(run_step(g, rule, model, optimizer, tape, arm))
        if step % 100 == 0 or step in STAGES:
            measured = torch.tensor(trace, dtype=torch.float64)
            state = dict(step=step, arm=arm, weights=model.state_dict(), optimizer=optimizer.state_dict(),
                         trace=measured, manifest_sha256=manifest)
            archive._atomic(resume, lambda file: torch.save(state, file))
            if step in STAGES:
                archive._atomic(root / f"step-{step}.pt", lambda file: torch.save(state, file))
            velocity = (step - start) / (perf_counter() - begin)
            print(arm, step, measured[-100:, :6].mean(0).tolist(), f"{velocity:.2f} updates/s", flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--until", type=int, choices=STAGES)
    action.add_argument("--benchmark", type=int)
    args = p.parse_args()
    if args.benchmark:
        benchmark(args.benchmark)
        return
    digest = fit_manifest(args.until)
    with ProcessPoolExecutor(max_workers=len(life.FIT_ARMS)) as pool:
        jobs = [pool.submit(fit, arm, args.until, digest) for arm in life.FIT_ARMS]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
