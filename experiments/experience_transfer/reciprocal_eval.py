"""Independent heldout Bprime continuations through the frozen generic evaluator."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.continuity.run import state_digest
from experiments.experience_transfer import reciprocal_train as r
from experiments.experience_transfer import eval_transfer as e
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import CONDITIONS

OUT = r.OUT / "evaluation"
PLAN = OUT / "protocol.json"


def checked():
    plan = e.read(PLAN)
    assert plan["protocol_hash"] == state_digest({k: v for k, v in plan.items() if k != "protocol_hash"})
    for path, expected in plan["source_files"].items():
        assert e.sha(path) == expected, f"frozen evaluation source changed: {path}"
    r.checked()
    return plan


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    training = r.checked()
    e.checked()
    sources = [Path(__file__).resolve(), Path(e.__file__).resolve(), Path(r.__file__).resolve(),
               Path(r.t.__file__).resolve(), ROOT / "experiments/learn_remember/evaluate_gru.py",
               ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
               ROOT / "experiments/lifelong/evaluate.py"]
    plan = dict(format="reciprocal-Bprime-autonomous-evaluation-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        training_protocol_hash=training["protocol_hash"], original_eval_protocol_hash=e.checked()["protocol_hash"],
        source_files={str(p): e.sha(p) for p in sources}, seeds=[0, 1, 2], arms=list(r.ARMS),
        snapshots=list(r.SNAPSHOTS), steps=2048, warmup_ticks=1024,
        heldout=dict(seed_offset=730000, actors=list(range(730000, 730008)), batch=8, n_jobs=2, dt=.25),
        history="Rebuild exact original heldout WAIT world under the Bprime mapping. Alternating history/B cues every128 ticks;512short followed by512long. All snapshots and A/C histories receive identical public observations. Private mapped targets may differ.",
        comparisons="Within each seed, compare original A/C histories under the same Bprime future raw requests, at B0/1/2/4/8. Preserve original A/B shared/conflict raw names unchanged for both histories. External digest and request identity must also equal original B-only WAIT reference, enabling fixed-common-conflict cross-target comparisons.",
        action="Frozen model native11-way argmax only, no teacher or gating. After one deployment birth and warmup, no reset at any request.",
        evidence="Every stage keeps full public obs/logits/actions, private scorer facts, receipts, complete endpoint states, matured request and semantic action rows, checkpoint work (including actual cumulative B NEW count), original-history source hashes.")
    plan["protocol_hash"] = state_digest(plan)
    e.write(PLAN, plan)
    for seed in range(3):
        histories = [common(seed, arm, create=True) for arm in r.ARMS]
        assert histories[0]["observation_hash"] == histories[1]["observation_hash"]
    print(json.dumps(dict(protocol_hash=plan["protocol_hash"], path=str(PLAN))), flush=True)


def common(seed, arm, create=False):
    path = OUT / f"common_seed_{seed:03d}_{arm}.pt"
    plan = e.read(PLAN)
    values = r.table(seed, arm)
    spec = r.t.SEEDS[seed]
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["protocol_hash"] == plan["protocol_hash"] and saved["table"] == values
        return saved
    if not create:
        raise FileNotFoundError(path)
    with r.t.mapping_scope(values):
        world = ContextWorld(seed=730000+seed, actor_ids=list(range(730000, 730008)),
            batch_size=8, n_jobs=2, active_jobs=2, dt=.25, future_family=spec["history_cue"])
        birth = world.state_dict()
        observations, elapsed = [], []
        for tick in range(1024):
            if tick % 128 == 0:
                world.set_family(spec["history_cue"] if (tick // 128) % 2 == 0 else spec["B_cue"])
                condition = "short" if tick < 512 else "long_both"
                world.set_regime(**CONDITIONS[condition], name=f"common/{tick // 128}", dt=.25)
            observations.append(world.observe().clone())
            elapsed.append(world.elapsed)
            world.act(torch.zeros((8, 2), dtype=torch.long))
        saved = dict(protocol_hash=plan["protocol_hash"], seed=seed, arm=arm, table=values,
            birth_world=birth, world=world.state_dict(), observations=torch.stack(observations),
            elapsed=elapsed, observation_hash=state_digest(torch.stack(observations)))
    old = e.common(seed, r.t.mapping_table(seed, arm), spec["history_cue"], spec["B_cue"], arm=arm)
    assert torch.equal(saved["observations"], old["observations"])
    assert saved["elapsed"] == old["elapsed"]
    saved["original_common_path"] = str(e.OUT / f"common_seed_{seed:03d}_{arm}.pt")
    saved["original_public_history_exact"] = True
    torch.save(saved, path)
    return saved


@torch.inference_mode()
def run(seed):
    torch.set_num_threads(1)
    plan = checked()
    training = r.checked()
    spec = r.t.SEEDS[seed]
    groups = training["spec"][str(seed)]["frozen_raw_groups"]
    records = []
    for arm in r.ARMS:
        shared = common(seed, arm)
        values = r.table(seed, arm)
        for snapshot in r.SNAPSHOTS:
            path = r.OUT / f"seed_{seed:03d}" / arm / f"{snapshot}.pt"
            if not path.exists():
                continue
            stem = OUT / f"seed_{seed:03d}" / f"{arm}_{snapshot}_Bprime"
            if stem.with_suffix(".json").exists():
                records.append(e.read(stem.with_suffix(".json")))
                continue
            began = time.perf_counter()
            model, _, _, cp = r.t.load_checkpoint(path, device="cpu")
            model.eval()
            assert cp["mapping_table"] == values
            assert cp["transfer"]["spec"]["B"] == r.bprime(seed)
            hidden = e.original.warmup(model, shared)
            with r.t.mapping_scope(values):
                world = ContextWorld.from_state_dict(shared["world"], device="cpu")
                scores, frames, states = e.rollout(model, world, hidden, families=(spec["B_cue"],),
                    steps=plan["steps"], table=values, history_family=spec["history_cue"],
                    b_family=spec["B_cue"], raw_groups=groups)
            reference = e.reference(seed, arm, "B")
            for key in ("external_digest", "primary_identity_hash", "start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash"):
                assert scores[key] == reference[key], (seed, arm, snapshot, key)
            stem.parent.mkdir(parents=True, exist_ok=True)
            rows, actions = scores.pop("primary_requests"), scores.pop("action_requests")
            np.savez_compressed(stem.with_suffix(".npz"), **frames)
            torch.save(states, stem.with_suffix(".pt"))
            e.write(stem.with_suffix(".requests.json"), dict(matured=rows, actions=actions))
            scores.update(format=plan["format"], protocol_hash=plan["protocol_hash"], seed=seed, arm=arm,
                history_rule="A" if arm == "related" else "C", target_rule="Bprime", target_related_history="C",
                snapshot=snapshot, B_round=int(snapshot[1:]), scenario="Bprime",
                checkpoint_path=str(path), checkpoint_sha256=e.sha(path), checkpoint_work=cp["work"],
                checkpoint_transfer=cp["transfer"], mapping_table=values, history_family=spec["history_cue"],
                B_family=spec["B_cue"], warmup_observation_hash=shared["observation_hash"],
                original_B_reference_external_digest=reference["external_digest"],
                elapsed_seconds=time.perf_counter()-began,
                artifacts={str(stem.with_suffix(ext)): e.sha(stem.with_suffix(ext)) for ext in (".npz", ".pt", ".requests.json")})
            e.write(stem.with_suffix(".json"), scores)
            records.append(scores)
            print(json.dumps(dict(event="reciprocal_eval", seed=seed, arm=arm, snapshot=snapshot,
                counts=scores["counts"], B_NEW=cp["work"]["B_unique_new_commands"])), flush=True)
    e.write(OUT / f"seed_{seed:03d}.json", dict(seed=seed, protocol_hash=plan["protocol_hash"], records=records))
    checked()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--seed", type=int, choices=range(3))
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.seed)
