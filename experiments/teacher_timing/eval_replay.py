"""Independent native world, original policy and auxiliary linear replay.

Load plain PyTorch GRU/Linear tensors directly; never instantiate the study's
model or call its evaluator, action chooser, target helper, scorer or oracle.
The auxiliary prediction is recomputed from the same post-observation hidden
state after verifying the unchanged eleven-action policy independently.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import json

import numpy as np
import torch
from torch import nn

from experiments.action_split.data import sha, read, write_new
from experiments.experience_transfer.train_transfer import mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.timing_boundary.data import FACT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/teacher-timing/eval-v1"
OUT = ROOT / "data/teacher-timing/eval-audit-v1"
PLAN = OUT / "replay_plan.json"
POLICIES = ("original", "earliest_action", "earliest_joint", "midpoint_action", "midpoint_joint")
CONDITIONS = ("long", "longer")
TOTAL, BATCH, WIDTH = 4608, 32, 192


def source_files():
    paths = [Path(__file__), ROOT / "experiments/action_split/data.py",
             ROOT / "experiments/experience_transfer/train_transfer.py",
             ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
             ROOT / "experiments/timing_boundary/data.py"]
    # Freeze all evaluation source bindings too, without importing its code.
    paths.extend(ROOT / name for name in read(INPUT / "protocol.json")["sources"])
    return list(dict.fromkeys(paths))


def declare():
    protocol = read(INPUT / "protocol.json")
    assert protocol["world_seed"] == "854000+seed"
    assert protocol["actor_ids"] == list(range(99000, 99032))
    assert set(protocol["policies"]) == set(POLICIES)
    assert set(protocol["conditions"]) == set(CONDITIONS)
    assert (protocol["warm"], protocol["test"], protocol["dt"]) == (512, 4096, .25)
    completed = read(INPUT / "completion.json")
    assert completed["complete"] and completed["runs"] == 30
    assert completed["sidecars"] == 24 and len(completed["time_files"]) == 24
    assert len(completed["files"]) == 30
    for name, digest in completed["time_files"].items():
        assert sha(ROOT / name) == digest, name
    rows = []
    # Main result JSON paths are the completion manifest.  Sidecars have their
    # own hashes and are bound separately below, with no scoring interpretation.
    for name, digest in sorted(completed["files"].items()):
        path = ROOT / name
        assert sha(path) == digest, name
        if path.stem.endswith("_time"):
            continue
        metadata = read(path)
        if "policy" not in metadata or "world_config" not in metadata:
            continue
        raw = path.with_suffix(".npz")
        cp = ROOT / metadata["checkpoint_path"]
        assert sha(raw) == metadata["facts_sha256"]
        assert sha(cp) == metadata["checkpoint_sha256"]
        row = dict(seed=metadata["seed"], policy=metadata["policy"], condition=metadata["condition"],
            metadata=name, metadata_sha256=digest, facts=str(raw.relative_to(ROOT)), facts_sha256=sha(raw),
            checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=sha(cp))
        assert row["policy"] in POLICIES and row["condition"] in CONDITIONS
        if row["policy"] != "original":
            sidecar = path.with_name(path.stem + "_time.npz")
            side_meta = path.with_name(path.stem + "_time.json")
            assert completed["time_files"][str(side_meta.relative_to(ROOT))] == sha(side_meta)
            row.update(time_predictions=str(sidecar.relative_to(ROOT)), time_predictions_sha256=sha(sidecar),
                       time_metadata=str(side_meta.relative_to(ROOT)), time_metadata_sha256=sha(side_meta))
        rows.append(row)
    assert len(rows) == 30
    assert {(row["seed"], row["policy"], row["condition"]) for row in rows} == {
        (seed, policy, condition) for seed in range(3) for policy in POLICIES for condition in CONDITIONS}
    plan = dict(format="actual-teacher-timing-independent-evaluation-replay-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), runs=rows,
        input_protocol_sha256=sha(INPUT / "protocol.json"), input_freeze_sha256=sha(INPUT / "freeze.json"),
        input_completion_sha256=sha(INPUT / "completion.json"),
        sources={str(p.relative_to(ROOT)): sha(p) for p in source_files()},
        checks="All30 runs: all4608ticks native publicobs/private facts/native receipts/clock/eventhash; no native oracle call",
        policy="Independent plain nn.GRU(56,192)+nn.Linear(192,22), original11way argmax, zero initialh, continuouswarm+test; everyaction/finalh bitwise exact",
        auxiliary="All24 newpolicy sidecars: independent nn.Linear(192,2) on post-observationh; never influences action or recurrent update; compare every prediction and all11 native logits bitwise",
        scoring="Request denominators and outcome metrics are independently reconstructed by the separate state-machine scorer")
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", runs=30, auxiliary_sidecars=24, plan_sha256=sha(PLAN))), flush=True)


def checked():
    plan = read(PLAN)
    for name, digest in plan["sources"].items():
        assert sha(ROOT / name) == digest, name
    for key, name in (("input_protocol_sha256", "protocol.json"), ("input_freeze_sha256", "freeze.json"),
                      ("input_completion_sha256", "completion.json")):
        assert sha(INPUT / name) == plan[key], name
    return plan


@torch.inference_mode()
def audit(index):
    torch.set_num_threads(1)
    plan = checked()
    row = plan["runs"][index]
    for name in ("metadata", "facts", "checkpoint", "time_predictions", "time_metadata"):
        if name in row:
            assert sha(ROOT / row[name]) == row[name + "_sha256"], row[name]
    metadata = read(ROOT / row["metadata"])
    cp = torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=False)
    cfg = cp["model_config"]
    assert (cfg["input_size"], cfg["width"], cfg["n_jobs"], cfg["n_actions"]) == (56, WIDTH, 2, 11)
    core = nn.GRU(56, WIDTH).eval().requires_grad_(False)
    head = nn.Linear(WIDTH, 22).eval().requires_grad_(False)
    core.load_state_dict({name.removeprefix("core."): value for name, value in cp["model"].items() if name.startswith("core.")})
    head.load_state_dict({name.removeprefix("head."): value for name, value in cp["model"].items() if name.startswith("head.")})
    time_head = None
    prediction_hash = None
    if row["policy"] != "original":
        time_head = nn.Linear(WIDTH, 2).eval().requires_grad_(False)
        time_head.load_state_dict({name.removeprefix("time_head."): value for name, value in cp["model"].items() if name.startswith("time_head.")})
        with np.load(ROOT / row["time_predictions"], allow_pickle=False) as archive:
            predictions = archive["time_prediction"]
            native_logits = archive["native_logits"]
        assert predictions.dtype == np.float32 and predictions.shape == (TOTAL, BATCH, 2)
        assert native_logits.dtype == np.float32 and native_logits.shape == (TOTAL, BATCH, 2, 11)
        prediction_hash = hashlib.sha256()
        expected_prediction_hash = hashlib.sha256(np.ascontiguousarray(predictions)).hexdigest()
    with np.load(ROOT / row["facts"], allow_pickle=False) as archive:
        facts = {name: archive[name] for name in archive.files}
    assert facts["obs"].dtype == np.float32 and facts["obs"].shape == (TOTAL, BATCH, 56)
    assert facts["actions"].dtype == np.int64 and facts["actions"].shape == (TOTAL, BATCH, 2)
    expected = {name: hashlib.sha256(np.ascontiguousarray(facts[name])).hexdigest() for name in FACT_FIELDS}
    replay = {name: hashlib.sha256() for name in FACT_FIELDS}
    kwargs = metadata["world_config"]
    assert kwargs["seed"] == 854000 + row["seed"]
    assert kwargs["actor_ids"] == list(range(99000, 99032))
    assert kwargs["dt"] == .25
    assert kwargs["duration_range"] == ([8., 16.] if row["condition"] == "long" else [16., 32.])
    assert kwargs["gap_range"] == [16., 32.]
    state = torch.zeros(1, BATCH, WIDTH, dtype=head.weight.dtype)
    with mapping_scope(mapping_table(row["seed"], "related")):
        world = ContextWorld(**kwargs)
        for tick in range(TOTAL):
            obs = world.observe()
            assert np.array_equal(obs.numpy(), facts["obs"][tick]), (index, tick, "native public observation")
            for name, attr in FACT_FIELDS.items():
                replay[name].update(getattr(world, attr).numpy())
            assert world.now == facts["now"][tick] == tick*.25
            values, state = core(obs.unsqueeze(0), state)
            scores = head(values[0]).reshape(BATCH, 2, 11)
            chosen = scores.argmax(-1)
            assert np.array_equal(chosen.numpy(), facts["actions"][tick]), (index, tick, "independent policy action")
            if time_head is not None:
                assert np.array_equal(scores.numpy(), native_logits[tick]), (index, tick, "independent native logits")
                prediction = time_head(values[0]).numpy()
                assert np.array_equal(prediction, predictions[tick]), (index, tick, "independent auxiliary readout")
                prediction_hash.update(prediction)
            world.act(torch.from_numpy(facts["actions"][tick]))
            assert np.array_equal(world.last_result.numpy(), facts["receipts"][tick]), (index, tick, "native receipt")
        assert world.event_hash == metadata["external_hash"]
    for name, digest in replay.items():
        assert digest.hexdigest() == expected[name], (index, name)
    assert np.array_equal(state[0].numpy(), facts["final_h"]), (index, "final hidden state")
    if prediction_hash is not None:
        assert prediction_hash.hexdigest() == expected_prediction_hash
    result = dict(index=index, seed=row["seed"], policy=row["policy"], condition=row["condition"], passed=True,
        ticks=TOTAL, opportunities=TOTAL*BATCH*2, public_observations="bitwise exact",
        private_fields={name: "bitwise exact" for name in FACT_FIELDS},
        actions="plain original GRU+linear11way argmax; everyaction bitwise exact",
        receipts="native exact everytick", clock="native exact everytick", final_h="bitwise exact",
        auxiliary_predictions="all bitwise exact" if time_head is not None else "original has no auxiliary head",
        native_logits="all bitwise exact" if time_head is not None else "original has no logits sidecar",
        external_hash=world.event_hash, input_files=row, plan_sha256=sha(PLAN), auditor_sha256=sha(__file__))
    write_new(OUT / "replay_runs" / f"run_{index:03d}.json", result)
    print(json.dumps(dict(index=index, policy=row["policy"], condition=row["condition"], passed=True)), flush=True)


def launch(workers=6):
    plan = checked()
    assert 1 <= workers <= 6
    logs = OUT / "replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(index):
        path = logs / f"run_{index:03d}.log"
        with path.open("x", encoding="utf-8") as handle:
            done = subprocess.run([sys.executable, "-m", "experiments.teacher_timing.eval_replay", "run", "--index", str(index)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        record = dict(index=index, exit_code=done.returncode, log=str(path.relative_to(ROOT)))
        print(json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finished = list(pool.map(job, range(len(plan["runs"]))))
    assert all(row["exit_code"] == 0 for row in finished), finished
    results = [read(OUT / "replay_runs" / f"run_{index:03d}.json") for index in range(len(plan["runs"]))]
    write_new(OUT / "replay.json", dict(passed=True, runs=len(results), auxiliary_sidecars=24,
        ticks_compared=sum(row["ticks"] for row in results),
        action_opportunities_compared=sum(row["opportunities"] for row in results),
        native_observations_receipts_privatefacts="all bitwise exact",
        independent_policy_actions_finalh="all bitwise exact", independent_auxiliary_predictions="all bitwise exact",
        independent_native_logits="all bitwise exact", auxiliary_prediction_values_compared=24*TOTAL*BATCH*2,
        native_logit_values_compared=24*TOTAL*BATCH*2*11,
        plan_sha256=sha(PLAN), auditor_sha256=sha(__file__), workers=finished, results=results))


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

