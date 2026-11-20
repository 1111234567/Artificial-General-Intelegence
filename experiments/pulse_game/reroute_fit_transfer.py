"""Repeat the route fit with half of the absolute room-to-endpoint maps excluded."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from itertools import product
import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from . import reroute_fit as fit
from . import reroute_world as world


PROTOCOL = fit.EVIDENCE / "fit-transfer-protocol.json"
BIRTHS = fit.BIRTHS
KINDS = ("terminal", "next_room")
DOMAIN = 18693173
METRICS = fit.METRICS
TRACE = fit.TRACE
STEPS = fit.STEPS
BATCH = fit.BATCH
EASY_STEPS = fit.EASY_STEPS


def heldout_map(scene):
    first, second = scene.prior_endpoints.unbind(1)
    third = 3 - first - second
    inversions = (first > second).long() + (first > third).long() + (second > third).long()
    return inversions.remainder(2).bool()


def episode(birth, step, backend="cpu"):
    accepted = []
    batch, block = 0, 0
    while batch < BATCH:
        sample = world.scenes([DOMAIN, birth, step, 0, block], 2 * BATCH,
                              easy=step <= EASY_STEPS, device=backend)
        eligible = sample[~heldout_map(sample)]
        accepted.append(eligible)
        batch += len(eligible)
        block += 1
    scene = world.Episode(*(torch.cat([getattr(chunk, field.name) for chunk in accepted])[:BATCH]
                            for field in fields(world.Episode)))
    tape = np.random.default_rng([DOMAIN, birth, step, 1]).random((2, BATCH), dtype=np.float32)
    return scene, *torch.from_numpy(tape).to(backend)


def register():
    content = json.dumps({
        "main_protocol_sha256": fit.register("cpu"), "training_domain": DOMAIN, "kinds": KINDS, "births": BIRTHS,
        "steps": STEPS, "easy_steps": EASY_STEPS, "batch": BATCH, "backend": "cpu",
        "source_sha256": fit.digest(__file__), "plan_sha256": fit.digest(fit.EVIDENCE / "fit-transfer-plan.md"),
        "map_tuple": ["prior endpoint from room 0", "prior endpoint from room 1", "remaining endpoint"],
        "training_maps": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        "heldout_maps": [[0, 2, 1], [1, 0, 2], [2, 1, 0]],
        "samples": "Generate from the main environment and retain only even maps. Each update uses the first "
                   "64 accepted rows in deterministic blocks. The exploration draws have a separate substream.",
        "curve_columns": METRICS, "route_columns": TRACE,
    }, indent=2) + "\n"
    if not PROTOCOL.is_file():
        fit.storage._atomic(PROTOCOL, lambda path: path.write_text(content, encoding="utf8"))
    elif PROTOCOL.read_text(encoding="utf8") != content:
        raise ValueError(f"Existing transfer protocol differs from the current source: {PROTOCOL}")
    return fit.digest(PROTOCOL)


def destination(kind, birth):
    return fit.DATA / f"fit-transfer-{kind}" / f"birth-{birth}.pt"


def train(kind, birth, total=STEPS):
    fit.configure("cpu")
    signature = dict(protocol=register(), kind=kind, birth=birth)
    model = fit.make_graph(birth)
    optimizer = torch.optim.Adam(model.parameters(), lr=fit.LR, foreach=False, fused=False)
    beginning = fit.state_digest(model.state_dict())
    target = destination(kind, birth)
    curve = torch.empty((total, len(METRICS)))
    ledger = torch.empty((total, BATCH, len(TRACE)))
    gradients = {}
    start, elapsed = 0, 0.
    if target.is_file():
        saved = torch.load(target, weights_only=True, map_location="cpu")
        if saved["signature"] != signature or saved["birth_sha256"] != beginning:
            raise ValueError(f"Wrong transfer training identity: {target}")
        start = saved["step"]
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        curve[:start], ledger[:start] = saved["curve"], saved["ledger"]
        gradients, elapsed = saved["through_history_gradient_norms"], saved["seconds"]

    clock = perf_counter()
    for step in range(start + 1, total + 1):
        curve[step - 1], ledger[step - 1], evidence = fit.update(
            model, optimizer, kind, *episode(birth, step),
            probe_gradients=step in (1, EASY_STEPS + 1),
        )
        if evidence is not None:
            gradients[str(step)] = evidence
        if step % fit.CHECKPOINT_EVERY == 0 or step == total:
            seconds = elapsed + perf_counter() - clock
            payload = dict(signature=signature, birth_sha256=beginning, step=step, state=model.state_dict(),
                           optimizer=optimizer.state_dict(), curve=curve[:step].clone(),
                           ledger=ledger[:step].clone(), through_history_gradient_norms=gradients,
                           seconds=seconds)
            fit.storage._atomic(target, lambda temporary: torch.save(payload, temporary))
            recent = curve[max(0, step - 64):step, :8].mean(0).tolist()
            print(kind, birth, step, [round(value, 5) for value in recent], round(seconds, 2), "s", flush=True)


def load_final(kind, birth):
    target = destination(kind, birth)
    saved = torch.load(target, weights_only=True, map_location="cpu")
    signature = dict(protocol=register(), kind=kind, birth=birth)
    if saved["signature"] != signature or saved["step"] != STEPS:
        raise ValueError(f"Unfinished or incompatible transfer fit: {target}")
    model = fit.make_graph(birth)
    if saved["birth_sha256"] != fit.state_digest(model.state_dict()):
        raise ValueError(f"Wrong transfer birth: {target}")
    model.load_state_dict(saved["state"])
    return model.requires_grad_(False).eval(), saved, fit.digest(target)


def run_all(jobs, total):
    register()
    logs = fit.DATA / "fit-transfer-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(pair):
        kind, birth = pair
        log = logs / f"{kind}-{birth}.txt"
        with log.open("a", encoding="utf8") as stream:
            process = subprocess.run(
                (sys.executable, "-m", __spec__.name, kind, str(birth), "--total", str(total)),
                cwd=fit.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
        print(kind, birth, process.returncode, log, flush=True)
        return process.returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        status = list(pool.map(run, [(kind, birth) for birth, kind in product(BIRTHS, KINDS)]))
    if any(status):
        raise RuntimeError("At least one transfer fit failed. See its log.")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("birth", nargs="?", type=int, choices=BIRTHS)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--total", type=int, default=STEPS)
    args = parser.parse_args()
    if not 1 <= args.total <= STEPS:
        parser.error(f"Total updates must lie in [1, {STEPS}]")
    if args.jobs:
        run_all(args.jobs, args.total)
    elif args.kind and args.birth:
        train(args.kind, args.birth, args.total)
    else:
        parser.error("Provide kind and birth, or use --jobs")


if __name__ == "__main__":
    main()
