"""Cross the target rule while reusing the exact archived A/C histories."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.continuity.run import state_digest
from experiments.experience_transfer import train_transfer as t

OUT = ROOT / "data/experience-transfer/reciprocal-v1"
PLAN = OUT / "protocol.json"
ARMS = ("related", "disjoint")  # Preserve original history names: A and C.
SNAPSHOTS = ("B000", "B001", "B002", "B004", "B008")


def bprime(seed):
    spec = t.SEEDS[seed]
    return [spec["B"][raw] if raw in spec["conflict_raw"] else spec["C"][raw] for raw in range(4)]


def table(seed, arm):
    values = deepcopy(t.mapping_table(seed, arm))
    values[t.SEEDS[seed]["B_cue"]] = bprime(seed)
    return values


def origin(seed, arm):
    return t.OUT / f"seed_{seed:03d}" / arm / "B000.pt"


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["protocol_hash"] == state_digest({k: v for k, v in plan.items() if k != "protocol_hash"})
    for path, value in plan["source_files"].items():
        assert t.sha(path) == value, f"frozen source changed: {path}"
    t.checked()
    return plan


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    specs = {}
    for seed, spec in t.SEEDS.items():
        target = bprime(seed)
        eq = lambda left, right: sum(a == b for a, b in zip(left, right))
        assert sorted(target) == [0, 1, 2, 3]
        assert eq(spec["C"], target) == 2 and eq(spec["A"], target) == 0
        assert all(target[r] == spec["B"][r] and spec["A"][r] == spec["C"][r] for r in spec["conflict_raw"])
        assert all(target[r] == spec["C"][r] and target[r] != spec["A"][r] for r in spec["shared_raw"])
        specs[str(seed)] = dict(original=deepcopy(spec), B_prime=target,
            mapping_tables={arm: table(seed, arm) for arm in ARMS},
            frozen_raw_groups=dict(all=[0, 1, 2, 3], shared=spec["shared_raw"], conflict=spec["conflict_raw"]),
            history_origins={arm: str(origin(seed, arm)) for arm in ARMS})
    sources = [Path(__file__).resolve(), Path(t.__file__).resolve(),
               ROOT / "experiments/learn_remember/gru_train.py", ROOT / "experiments/lifelong/world.py",
               ROOT / "experiments/continuity/world.py"]
    plan = dict(format="experience-transfer-reciprocal-Bprime-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        declaration="Target crossing specified by parent before training outcomes; operational freeze before any Bprime training/evaluation. All3 seeds, both existing histories, all fixed checkpoints retained.",
        source_files={str(p): t.sha(p) for p in sources}, original_training_protocol_sha256=t.sha(t.PLAN),
        seeds=[0, 1, 2], arms=list(ARMS), snapshots=list(SNAPSHOTS), spec=specs,
        intervention="Reuse exact original A/C B000 complete checkpoints; no repeated history or direct arm. Bprime keeps original B correct answers on the original A/B conflict raw group, where A and C wrong answers are identical. On the original shared raw group use C answers. Thus C has2 correct associations and A0, reversing the original relatedness while holding common conflict difficulty fixed.",
        training="Learner.from_checkpoint restores exact model, Adam, numeric h, full world/scheduler/receipts and RNG from original B000. Only never-issued B cue mapping row changes, plus descriptive spec.B. Call unchanged B_training on A/C pair for8 rounds of1024ticks64actors, current-round32epochs128block; exact B masks/labels/actions paired. No history replay, no new history training. Save B000/B001/B002/B004/B008; unchanged helper also emits early snapshots, not primary here.",
        evaluation="Frozen snapshot standalone long_both Bprime free-action evaluation with unchanged eval_transfer.rollout. Independent heldout seed730000+seed,8actors730000..730007,1024 WAIT ticks alternating old/B cue every128 with512short then512long, followed by2048long ticks. Rebuild private worlds under each Bprime table; require identical public warmup and future request/external digests. Standard newborn h warmed per frozen snapshot, no transplanted h in this crossing.",
        metrics="All/shared/conflict based on frozen ORIGINAL A/B raw groups for both targets and both histories. Retain ordinary/strict mature counts, actual idle actions/opportunities, valid/feasible BEGIN and semantic FINISH counts, full raw/request facts, checkpoint work including B_unique_new_commands and updates. Pair seedwise A-minus-C atB0/1/2/4/8; no winner or duration selection.",
        audit="Bind every branch to original B000 hash and compare complete restored model/Adam/h/world/RNG. Independently verify archived B demonstrations and training budgets, cross-arm exact labels/actions/masks. Recompute evaluation outcomes from raw receipts; native evaluator world replay atB000/B008, all stages model recurrence and request identity checks.",
        limits=["Original related/disjoint filenames retain A/C history identity; for Bprime their semantic relation reverses.",
                "Three seeds, fixed permutations and one heldout stream per seed; not a general transfer claim.",
                "B training receives oracle demonstrations; heldout behavior is true model argmax.",
                "The common conflict group's correct and historical wrong answers are fixed across B/Bprime, but non-conflict answers and their contextual interference change."])
    plan["protocol_hash"] = state_digest(plan)
    t.write(PLAN, plan)
    print(json.dumps(dict(protocol_hash=plan["protocol_hash"], path=str(PLAN))), flush=True)


def run(seed):
    torch.set_num_threads(1)
    plan = checked()
    folder = OUT / f"seed_{seed:03d}"
    if (folder / "complete.json").exists():
        raise FileExistsError(folder / "complete.json")
    learners, bindings = [], []
    began = time.perf_counter()
    for arm in ARMS:
        source = origin(seed, arm)
        original = torch.load(source, map_location="cpu", weights_only=False)
        spec = t.SEEDS[seed]
        assert original["world"]["context"]["tensors"]["family_new_counts"][..., spec["B_cue"]].sum() == 0
        assert original["world"]["context"]["tensors"]["family_replace_counts"][..., spec["B_cue"]].sum() == 0
        learner = t.Learner.from_checkpoint(source, folder / arm, table=table(seed, arm))
        learner.spec["B"] = bprime(seed)
        learner.binding["reciprocal_protocol_hash"] = plan["protocol_hash"]
        expected_world = deepcopy(original["world"])
        expected_world["context"]["mappings"] = table(seed, arm)
        assert state_digest(learner.world_state()) == state_digest(expected_world)
        assert state_digest(learner.model.state_dict()) == state_digest(original["model"])
        assert state_digest(learner.optimizer.state_dict()) == state_digest(original["optimizer"])
        assert torch.equal(learner.hidden, original["hidden"])
        assert state_digest(learner.rng.bit_generator.state) == state_digest(original["numpy_rng"])
        assert learner.work == original["work"]
        bindings.append(dict(seed=seed, arm=arm, source_checkpoint=str(source), source_sha256=t.sha(source),
            model_hash=state_digest(original["model"]), optimizer_hash=state_digest(original["optimizer"]),
            hidden_hash=state_digest(original["hidden"]), old_world_hash=state_digest(original["world"]),
            new_world_hash=state_digest(expected_world), initial_work=deepcopy(learner.work),
            B_prime=bprime(seed), complete_restoration_except_unissued_mapping=True))
        learners.append(learner)
    t.write(folder / "branch_bindings.json", bindings)
    report = t.B_training(learners, rounds=8)
    checked()
    for item in bindings:
        assert t.sha(item["source_checkpoint"]) == item["source_sha256"]
    t.write(folder / "complete.json", dict(seed=seed, protocol_hash=plan["protocol_hash"], passed=True,
        elapsed_seconds=time.perf_counter()-began, B_pairing=report,
        work={v.arm: v.work for v in learners}, bindings=bindings))
    print(json.dumps(dict(event="reciprocal_complete", seed=seed, seconds=time.perf_counter()-began)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--seed", type=int, choices=range(3))
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.seed)
