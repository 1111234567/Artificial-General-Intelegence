"""Independent native world and plain PyTorch policy replay of held-out runs.

The audit does not call the evaluator, its scoring helpers, policy classes,
decoder, or an oracle.  Its protocol and all source/input hashes are frozen
before results are produced.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from experiments.action_split.data import sha, read, write_new
from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.timing_boundary.data import FACT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/action-split/eval-v1"
OUT = ROOT / "data/action-split/eval-audit-v1"
PLAN = OUT / "replay_plan.json"


def source_files():
    return [Path(__file__), ROOT / "experiments/action_split/data.py",
            ROOT / "experiments/experience_transfer/train_transfer.py",
            ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
            ROOT / "experiments/timing_boundary/data.py"]


def declare():
    completed = read(INPUT / "completion.json")
    assert completed["complete"] and completed["runs"] == 30
    rows = []
    for name, digest in sorted(completed["files"].items()):
        path = ROOT / name
        assert sha(path) == digest
        metadata = read(path)
        raw = path.with_suffix(".npz")
        checkpoint = ROOT / metadata["checkpoint_path"]
        assert sha(raw) == metadata["facts_sha256"]
        assert sha(checkpoint) == metadata["checkpoint_sha256"]
        rows.append(dict(seed=metadata["seed"], policy=metadata["policy"], condition=metadata["condition"],
            metadata=name, metadata_sha256=digest,
            facts=str(raw.relative_to(ROOT)), facts_sha256=sha(raw),
            checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint)))
    assert len(rows) == 30
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        format="action-split-independent-evaluation-replay-v1", runs=rows,
        input_protocol_sha256=sha(INPUT / "protocol.json"),
        input_freeze_sha256=sha(INPUT / "freeze.json"),
        input_completion_sha256=sha(INPUT / "completion.json"),
        sources={str(p.relative_to(ROOT)): sha(p) for p in source_files()},
        checks="All30 runs; each4608tick native world from archived JSONconfig under related mapping. Compare every publicobs, privateFACTFIELD, executedaction, native receipt, clock and finaleventhash. No oracle call.",
        policy_replay="Independent plain nn.GRU+nn.Linear loaded directly from state_dict; initialh=zeros; publicobs only; old11wayargmax or own13score modeargmax then conditionalargmax. Compare everyaction and finalh exactly. No teacher, legality or timing gate.",
        scoring="Per-request metrics are independently audited by a separate audit; this audit validates their actual native trajectories.")
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", runs=len(rows), plan_sha256=sha(PLAN))), flush=True)


def checked():
    plan = read(PLAN)
    for name, digest in plan["sources"].items():
        assert sha(ROOT / name) == digest, name
    for key, path in (("input_protocol_sha256", INPUT / "protocol.json"),
                      ("input_freeze_sha256", INPUT / "freeze.json"),
                      ("input_completion_sha256", INPUT / "completion.json")):
        assert sha(path) == plan[key], path
    return plan


@torch.inference_mode()
def audit(index):
    torch.set_num_threads(1)
    plan = checked()
    row = plan["runs"][index]
    for name in ("metadata", "facts", "checkpoint"):
        assert sha(ROOT / row[name]) == row[name+"_sha256"], row[name]
    metadata = read(ROOT / row["metadata"])
    cp = torch.load(ROOT / row["checkpoint"], weights_only=False, map_location="cpu")
    cfg = cp["model_config"]
    assert cfg["input_size"] == 56 and cfg["n_jobs"] == 2
    n_actions = int(cfg["n_actions"])
    assert n_actions == (11 if row["policy"] == "original" else 13)
    core = nn.GRU(cfg["input_size"], cfg["width"], num_layers=1).eval().requires_grad_(False)
    head = nn.Linear(cfg["width"], cfg["n_jobs"]*n_actions).eval().requires_grad_(False)
    core.load_state_dict({k[len("core."):]: v for k,v in cp["model"].items() if k.startswith("core.")})
    head.load_state_dict({k[len("head."):]: v for k,v in cp["model"].items() if k.startswith("head.")})
    with np.load(ROOT / row["facts"], allow_pickle=False) as archive:
        facts = {name: archive[name] for name in archive.files}
    assert facts["obs"].dtype == np.float32 and facts["obs"].shape == (4608, 32, 56)
    assert facts["actions"].shape == (4608, 32, 2)
    expected_hashes = {name: hashlib.sha256(np.ascontiguousarray(facts[name])).hexdigest()
                       for name in FACT_FIELDS}
    replay_hashes = {name: hashlib.sha256() for name in FACT_FIELDS}
    kwargs = metadata["world_config"]
    assert kwargs["seed"] == 851000+row["seed"]
    assert kwargs["actor_ids"] == list(range(96000, 96032))
    assert kwargs["dt"] == .25
    assert kwargs["duration_range"] == ([8., 16.] if row["condition"] == "long" else [16., 32.])
    state = torch.zeros(1, 32, cfg["width"], dtype=head.weight.dtype)
    with mapping_scope(mapping_table(row["seed"], "related")):
        world = ContextWorld(**kwargs)
        for tick in range(4608):
            obs = world.observe()
            assert np.array_equal(obs.numpy(), facts["obs"][tick]), (index, tick, "obs")
            for name, attr in FACT_FIELDS.items():
                replay_hashes[name].update(getattr(world, attr).numpy())
            assert world.now == facts["now"][tick] == tick*.25
            values, state = core(obs[None], state)
            scores = head(values[0]).reshape(32, 2, n_actions)
            if n_actions == 11:
                chosen = scores.argmax(-1)
            else:
                mode = scores[..., :3].argmax(-1)
                begin = 1+scores[..., 3:5].argmax(-1)
                finish = 3+scores[..., 5:13].argmax(-1)
                chosen = torch.zeros_like(mode)
                chosen[mode == 1] = begin[mode == 1]
                chosen[mode == 2] = finish[mode == 2]
            assert np.array_equal(chosen.numpy(), facts["actions"][tick]), (index, tick, "policy action")
            # Execute the archived actions, whose policy derivation was just
            # independently verified, through unchanged native act().
            world.act(torch.from_numpy(facts["actions"][tick]))
            assert np.array_equal(world.last_result.numpy(), facts["receipts"][tick]), (index, tick, "receipt")
        assert world.event_hash == metadata["external_hash"]
    for name in FACT_FIELDS:
        assert replay_hashes[name].hexdigest() == expected_hashes[name], (index, name)
    assert np.array_equal(state[0].numpy(), facts["final_h"]), (index, "final_h")
    result = dict(index=index, seed=row["seed"], policy=row["policy"], condition=row["condition"],
        passed=True, ticks=4608, opportunities=4608*32*2, public_observations="bitwise exact",
        private_fields={name: "bitwise exact" for name in FACT_FIELDS},
        actions="independent plainGRU+linear and explicit decoder; all bitwise exact",
        receipts="native exact everytick", final_h="bitwise exact", clock="exact everytick",
        external_hash=world.event_hash, input_files=row, plan_sha256=sha(PLAN),
        auditor_sha256=sha(__file__))
    write_new(OUT / "replay_runs" / f"run_{index:03d}.json", result)
    print(json.dumps(dict(index=index, seed=row["seed"], policy=row["policy"],
                          condition=row["condition"], passed=True)), flush=True)


def launch(workers=6):
    plan = checked()
    assert 1 <= workers <= 6
    logs = OUT / "replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(index):
        path = logs / f"run_{index:03d}.log"
        with path.open("x", encoding="utf-8") as handle:
            done = subprocess.run([sys.executable, "-m", "experiments.action_split.eval_replay",
                "run", "--index", str(index)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        record = dict(index=index, exit_code=done.returncode, log=str(path.relative_to(ROOT)))
        print(json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=workers) as pool:
        workers_done = list(pool.map(job, range(len(plan["runs"]))))
    assert all(row["exit_code"] == 0 for row in workers_done), workers_done
    results = [read(OUT / "replay_runs" / f"run_{index:03d}.json") for index in range(30)]
    write_new(OUT / "replay.json", dict(passed=True, runs=30,
        ticks_compared=sum(row["ticks"] for row in results),
        action_opportunities_compared=sum(row["opportunities"] for row in results),
        native_observations_receipts_privatefacts="all bitwise exact",
        independent_policy_actions_finalh="all bitwise exact",
        plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), workers=workers_done, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "all"))
    parser.add_argument("--index", type=int, choices=range(30))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "run":
        audit(args.index)
    else:
        launch(args.workers)
