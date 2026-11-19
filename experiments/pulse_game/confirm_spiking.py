"""Run externally chosen world splits with frozen spiking subjects and feedback controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit, spike_policy as spike, world


def _job(job):
    config, net, seeds = job
    tape = world.make_tape(seeds)
    def factory(cfg=config):
        return spike.BatchSpikingPolicy(cfg, network_seed=net, action_seeds=[
            np.random.SeedSequence([s, 0x4C49464C]) for s in seeds])
    conditions = audit.normal_and_controls(factory, tape)
    for name, cfg in (("no_write", replace(config, learning_rate=0.)),
                      ("no_recurrent_edges", replace(config, recurrence_gain=0.))):
        conditions[name] = audit.life_rows(tape, world.rollout(factory(cfg), tape))
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("diagnostic", "confirmation", "replication"), default="confirmation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 19, 37, 53])
    parser.add_argument("--representation", choices=("random", "engineered"), default="random")
    args = parser.parse_args()
    config = (spike.SpikeConfig(use_residual=True) if args.representation == "random" else
              spike.SpikeConfig(representation="engineered"))
    seeds = audit.split(args.split)
    tag = f"spike-{args.representation}-{args.split}"
    subdir = audit.OUT / f"{tag}-shards"
    meta = {"source_sha256": audit.fingerprint(spike.__file__), "config": asdict(config),
            "analysis_sha256": audit.fingerprint(audit.__file__), "world_sha256": audit.fingerprint(world.__file__),
            "split_sha256": audit.SPLITS_SHA256, "network_seeds": args.seeds}
    frozen = subdir / "frozen.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != meta:
        raise ValueError("The selected release has changed")
    if not frozen.exists():
        _write(frozen, meta)
    width = 16
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        todo = []
        for net in args.seeds:
            for i in range(0, len(seeds), width):
                path = subdir / f"g{net}-{i:04d}.json"
                if not path.exists():
                    todo.append((path, pool.submit(_job, (config, net, seeds[i:i+width]))))
        for n, (path, future) in enumerate(todo, 1):
            _write(path, future.result())
            if n % 8 == 0:
                print(f"{tag} {n}/{len(todo)} shards {time.perf_counter()-start:.1f}s", flush=True)
    summaries = {}
    for net in args.seeds:
        by_condition = {}
        for i in range(0,len(seeds),width):
            data = json.loads((subdir / f"g{net}-{i:04d}.json").read_text(encoding="utf-8"))
            for name, rows in data.items():
                columns = by_condition.setdefault(name,{})
                for key, vals in rows.items():
                    columns.setdefault(key,[]).extend(vals)
        path = audit.write_audit(f"{tag}-g{net}", {**meta, "network_seed": net}, world.make_tape(seeds), by_condition)
        record = json.loads(path.read_text(encoding="utf-8"))
        summaries[str(net)] = {k: record[k] for k in ("summary","true_minus_control")}
    _write(audit.OUT / f"{tag}.json", {**meta, "world_count": len(seeds), "by_graph": summaries})
    print({k: round(v["summary"]["true"]["late_accuracy"]["mean"], 4) for k,v in summaries.items()},flush=True)


if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    main()
