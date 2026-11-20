"""Teach the same persistent ticket body with one-step or realized future credit."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn

from . import interleave_revisit_train as io
from . import noon_coverage as coverage, noon_ticket as ticket
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_lifetime as lifetime


ROOT = card.DATA / "transfer" / "continuous-credit"
PLAN = card.OUT / "transfer-continuous-credit-plan.json"
ARMS = ("current", "future")
NET = ticket.net
WORLDS = 4
WINDOW = 8
TRAIN_DOMAIN = 2057786247
TRAIN_SEEDS = {"original": 2057713701, "independent": 2057713723}
PILOT_SEED = 2057713741
LEARNING_RATE = .003
CLIP = 2.
TRACE = ("net_per_opportunity", "net_per_life", "gross_per_opportunity", "buy_fraction",
         "gradient_norm", "float64_retry")


def train_tape(cohort, births, update, *, pilot=False):
    setup = archive.read(lifetime.PLAN)
    setup.update(domain=TRAIN_DOMAIN, opportunities=WINDOW, recurring_objects=3, block_size=4,
                 new_targets_per_block=1, distractor_contact_every=4, yoke_replicates=1)
    parent = PILOT_SEED if pilot else TRAIN_SEEDS[cohort]
    sources = []
    for birth in births:
        seed = int(np.random.SeedSequence([parent, birth, update]).generate_state(1)[0])
        sources.append(lifetime.make_tape(setup, seed, WORLDS))
    used = ("initial_x", "initial_value", "initial_draw", "price", "ticket_side", "distractor_x", "distractor_value",
            "gaps", "preview", "step_draw", "offer_left", "offer_right", "review", "review_value", "final", "final_value",
            "delayed_draw")
    return {k: torch.cat([s[k] for s in sources]) for k in used}


def settle(view, action, left, right, taste, draws):
    event = NET.observe(view, action, left, right, taste)
    written = NET.commit(view, event, draws)
    return written.state, written.edit_log_prob


def life(g, r, t):
    """Preserve every native state for eight decisions and every gradient link."""
    lanes, steps = t["price"].shape
    batch = lanes // len(g.bias)
    view = r.expand(batch)
    state = NET.start(g, batch)
    zero = g.bias.new_zeros(lanes)
    initial_score = zero
    for j in range(t["initial_x"].shape[1]):
        state, before, _, _ = ticket.encounter(view, state, t["initial_x"][:, j],
                                               t["initial_value"][:, j], t["initial_draw"][:, j])
        initial_score = initial_score + before
    scores, harvests, edits = [], [], []
    realized, physical = [], []
    for i in range(steps):
        u = t["step_draw"][:, i]
        before = initial_score if i == 0 else zero
        if (i+1) % 4 == 0:
            j = (i+1) // 4 - 1
            state, part, _, _ = ticket.encounter(view, state, t["distractor_x"][:, j],
                                                 t["distractor_value"][:, j], u[:, 0])
            before = before + part
        state = NET.wait(view, state, t["gaps"][:, i, 0], max_ticks=6)
        x = t["preview"][:, i]
        state = NET._advance(view, state, NET.native.sensory(state.graph, -x, x), NET.REACTION_TICKS)
        left, right = t["offer_left"][:, i], t["offer_right"][:, i]
        choice = NET.act(view, state, left, right, u[:, 1, 0])
        buy = choice.side == t["ticket_side"][:, i]
        fee = t["price"][:, i].to(g.bias.dtype) * buy / 100
        state, routing_edit = settle(view, choice, left, right, -fee, u[:, 1, 1:])
        x = t["review"][:, i]
        state, review_score, visit, taste = ticket.encounter(view, state, x, t["review_value"][:, i] * buy, u[:, 2])
        scores.append(before + choice.motor_log_prob + routing_edit + review_score)

        state = NET.wait(view, state, t["gaps"][:, i, 1], max_ticks=6)
        x = t["final"][:, i]
        harvest = NET.act(view, state, -x, x, u[:, 3, 0])
        true = harvest.side * t["final_value"][:, i]
        gross = (true > 0).to(g.bias.dtype)
        state, after_edit = settle(view, harvest, -x, x, zero, u[:, 3, 1:])
        harvests.append(harvest.motor_log_prob)
        edits.append(after_edit)
        realized.append(torch.stack((gross - fee, gross, buy.to(g.bias.dtype)), 1))
        physical.append(torch.stack((choice.side, visit, taste, harvest.side, true), 1))
    with torch.no_grad():
        settled = state.detach()
        for i in range(steps):
            x, draws = t["final"][:, i], t["delayed_draw"][:, i]
            side, taste = physical[i][:, 3], physical[i][:, 4]
            action = NET.act(view, settled, -x, x, draws[:, 0], side_override=side)
            settled, _ = settle(view, action, -x, x, taste, draws[:, 1:])
    return (tuple(torch.stack(row, 1) for row in (scores, harvests, edits)),
            torch.stack(realized, 1).detach(), torch.stack(physical, 1).detach(), settled)


def targets(realized, arm):
    net, gross = realized[:, :, 0], realized[:, :, 1]
    future = net.flip([1]).cumsum(1).flip([1]) - net if arm == "future" else torch.zeros_like(net)
    return net + future, gross + future, future


def advantages(values, roots):
    shaped = values.reshape(roots, WORLDS, 4, WINDOW)
    baseline_world = shaped.mean(2, keepdim=True)
    other_worlds = (baseline_world.sum(1, keepdim=True) - baseline_world) / (WORLDS - 1)
    return shaped - other_worlds


def loss(scores, realized, roots, arm):
    rewards = targets(realized, arm)
    terms = [-(score.reshape(roots, WORLDS, 4, WINDOW) * advantages(value, roots)).mean((1, 2, 3))
             for score, value in zip(scores, rewards, strict=True)]
    return sum(terms).sum()


def parameters(g, r):
    return (g.weights, g.bias, *r.parameters())


def update(g, r, optimizer, tape, arm):
    for precise in (False, True):
        optimizer.zero_grad(set_to_none=True)
        if precise:
            body = NET.Graph(nn.Parameter(g.weights.detach().double()), nn.Parameter(g.bias.detach().double()), g.present)
            writer = deepcopy(r).double()
        else:
            body, writer = g, r
        scores, realized, _, _ = life(body, writer, tape)
        value = loss(scores, realized, len(g.bias), arm)
        if not value.isfinite():
            continue
        value.backward()
        norm = io._gradients(parameters(body, writer), CLIP)
        if norm is None:
            continue
        if precise:
            for x, y in zip(parameters(g, r), parameters(body, writer), strict=True):
                x.grad = y.grad.to(x.dtype) if y.grad is not None else None
        optimizer.step()
        rows = realized.reshape(len(g.bias), -1, WINDOW, 3).mean((1, 2)).T
        return torch.stack((rows[0], rows[0]*WINDOW, rows[1], rows[2], norm, torch.full_like(norm, precise)), 1)
    raise FloatingPointError(f"Nonfinite long credit update in {arm}, including same-tape float64 replay")


@torch.no_grad()
def physical_parity(cohort):
    births = ticket.births(cohort)
    state_a = coverage.inherited(cohort)
    state_b = coverage.inherited(cohort)
    sample = train_tape(cohort, births, 0, pilot=True)
    _, results_a, receipts_a, actor_a = life(*state_a, sample)
    _, results_b, receipts_b, actor_b = life(*state_b, sample)
    for a, b in ((results_a, results_b), (receipts_a, receipts_b), (actor_a.hidden, actor_b.hidden),
                 (actor_a.graph.weights, actor_b.graph.weights), (actor_a.sender_trace, actor_b.sender_trace)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    u = results_a[:, :, 0]
    short = targets(results_a, "current")
    long = targets(results_a, "future")
    torch.testing.assert_close(short[0], u, rtol=0, atol=0)
    torch.testing.assert_close(long[0][:, 0], u.sum(1), rtol=0, atol=2e-6)
    torch.testing.assert_close(long[0][:, -1], u[:, -1], rtol=0, atol=2e-6)
    torch.testing.assert_close(long[2][:, -1], torch.zeros_like(u[:, -1]), rtol=0, atol=2e-6)


def benchmark(cohort, arm, count):
    torch.set_num_threads(1)
    physical_parity(cohort)
    births = ticket.births(cohort)
    g, r = coverage.inherited(cohort)
    opt = torch.optim.Adam(parameters(g, r), lr=LEARNING_RATE)
    times, recoveries = [], []
    begin = perf_counter()
    for i in range(1, count+1):
        start = perf_counter()
        tape = train_tape(cohort, births, i, pilot=True)
        row = update(g, r, opt, tape, arm)
        times.append(perf_counter()-start)
        recoveries.append(bool(row[:, -1].any()))
    print(json.dumps(dict(cohort=cohort, arm=arm, steps=count, wall_seconds=perf_counter()-begin,
                          median_seconds=float(np.median(times)), p90_seconds=float(np.quantile(times, .9)),
                          float64_recoveries=sum(recoveries), physical_parity=True)), flush=True)


def fit(cohort, arm):
    torch.set_num_threads(1)
    plan = archive.read(PLAN)
    target, stages = plan["updates"], plan["checkpoints"]
    births = ticket.births(cohort)
    if plan["training"] != {"worlds_per_birth": WORLDS, "opportunities": WINDOW, "domain": TRAIN_DOMAIN,
                            "seeds": TRAIN_SEEDS, "learning_rate": LEARNING_RATE, "clip": CLIP}:
        raise ValueError("Frozen plan differs from the training constants")
    source = card.OUT / f"transfer-ticket-coverage-{cohort}-8000.json"
    original = archive.read(source)["model_custody"]["ticket_native"]["models"]
    for b in births:
        row = original[str(b)]
        archive.verify(card.ROOT / row["path"], row["sha256"])
    prefix = ROOT / cohort
    directory = prefix / "training" / arm
    directory.mkdir(parents=True, exist_ok=True)
    run = directory / "run.json"
    modules = (__file__, coverage.__file__, ticket.__file__, lifetime.__file__, NET.__file__, NET.native.__file__)
    manifest = {"cohort": cohort, "arm": arm, "births": births, "plan": custody.record(PLAN),
                "parents": original, "source_sha256": {Path(f).name: archive.digest(Path(f)) for f in modules},
                "target": target, "versions": {"torch": str(torch.__version__), "numpy": np.__version__}}
    if run.is_file():
        if archive.read(run) != manifest:
            raise ValueError(f"Optimizer protocol changed at {run}")
    else:
        io._write_json(run, manifest)
    lock = archive.digest(run)
    g, r = coverage.inherited(cohort)
    optimizer = torch.optim.Adam(parameters(g, r), lr=LEARNING_RATE)
    progress = []
    resume = directory / "resume.pt"
    step = 0
    if resume.is_file():
        stored = torch.load(resume, map_location="cpu", weights_only=True)
        if stored["manifest_sha256"] != lock:
            raise ValueError(f"Optimizer identity mismatch at {resume}")
        step = stored["step"]
        if step == target and all(card.checkpoint.checkpoint_path(prefix, "odor", arm, b, step).is_file() for b in births):
            return
        with torch.no_grad():
            for name in ("weights", "bias", "present"):
                getattr(g, name).copy_(stored["graph"][name])
        r.load_state_dict(stored["rule"])
        optimizer.load_state_dict(stored["optimizer"])
        progress = stored["trace"].tolist()

    def save(frozen):
        payload = dict(step=step, manifest_sha256=lock, graph={k: getattr(g, k).detach() for k in ("weights", "bias", "present")},
                       rule=r.state_dict(), optimizer=optimizer.state_dict(), trace=torch.tensor(progress, dtype=torch.float64))
        if frozen:
            for i, birth in enumerate(births):
                card.checkpoint.save_checkpoint(prefix, "odor", arm, birth, step, g, r, i)
            io._atomic(directory / f"optimizer-{step:06d}.pt", lambda path: torch.save(payload, path))
            io._write_raw(directory / f"trace-{step:06d}.npz", births=births, labels=TRACE, values=payload["trace"].numpy())
        io._atomic(resume, lambda path: torch.save(payload, path))

    if not resume.is_file():
        save(0 in stages)
    initial = step
    started = perf_counter()
    for step in range(step+1, target+1):
        tape = train_tape(cohort, births, step)
        row = update(g, r, optimizer, tape, arm)
        progress.append(row.tolist())
        if step in stages or step % 100 == 0:
            save(step in stages)
            score = np.asarray(progress[-100:]).mean((0, 1))
            speed = (step-initial)/(perf_counter()-started)
            print(cohort, arm, step, "net, gross, buy", np.round(score[[0, 2, 3]], 5).tolist(), f"{speed:.3f}/s", flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=tuple(TRAIN_SEEDS), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--benchmark", type=int)
    args = p.parse_args()
    if args.benchmark:
        benchmark(args.cohort, args.arm, args.benchmark)
    else:
        fit(args.cohort, args.arm)
