"""Frozen-state, real-history input clamping followed by autonomous release.

Only observe() enters the controller. Each history is driven by one real
eleven-action argmax policy; both initial hidden states consume that same
history, including its real action receipts. The other hidden state's proposed
actions are recorded, never applied during clamping.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

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
from experiments.learn_remember import evaluate_gru as e
from experiments.learn_remember.gru_train import load_checkpoint
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import CONDITIONS

OUT = ROOT / "data/experience-transfer/state-v1"
ARCHES = ("plain", "symmetric")
STATES = ("standard", "transplant")
POINTS = (0, 32, 128, 512, 1024)
FORMAT = "experience-transfer-same-input-state-release-v1"


def checkpoint(arch, seed):
    return ROOT / "data/learn-remember" / ("gru-v1" if arch == "plain" else "symmetric-v1") / f"seed_{seed:03d}/wide.pt"


def checked():
    plan = e.read(OUT / "protocol.json")
    assert plan["protocol_hash"] == state_digest({k: v for k, v in plan.items() if k != "protocol_hash"})
    for path, expected in plan["source_files"].items():
        assert e.sha(path) == expected, f"frozen source changed: {path}"
    for path, expected in plan["checkpoints"].items():
        assert e.sha(path) == expected, f"checkpoint changed: {path}"
    return plan


def declare():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("immutable state protocol already exists")
    sources = [Path(__file__).resolve(), ROOT / "experiments/learn_remember/evaluate_gru.py",
               ROOT / "experiments/learn_remember/gru_train.py", ROOT / "experiments/lifelong/world.py",
               ROOT / "experiments/lifelong/evaluate.py", ROOT / "experiments/continuity/world.py"]
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration="Frozen before generating any same-input/release results. Posthoc follow-up to learn-remember/state-transfer-v1; no new training or state/checkpoint selection.",
        seeds=[0, 1, 2], architectures=list(ARCHES), states=list(STATES), clamp_points=list(POINTS),
        checkpoints={str(checkpoint(a, s)): e.sha(checkpoint(a, s)) for a in ARCHES for s in range(3)},
        source_files={str(p): e.sha(p) for p in sources},
        common_history="Reuse original evaluation-v1/common_seed_NNN.pt: eight unfamiliar actors, the same 1024-tick WAIT warmup, complete endpoint world. Standard h is recomputed from that exact public history with each frozen model; transplant h is saved native training endpoint first eight lanes in original order, without selection.",
        playback="Two separate real free-action trajectories start from the identical heldout world, one driven by standard h and one driven by transplant h. Set future family3/long_both once. Both h consume exactly the same observe() tensor at every tick on a given trajectory; only the named driver's eleven-way argmax acts in the world. Receipts are actual driver receipts. Labels never enter the model, and oracle is not called.",
        checkpoints_semantics="At k, exactly k public observations have updated both h and k driver actions have occurred. Save complete shared world and each evolved h. Distance is measured before the next public input. Next eleven-action disagreement uses one additional shared observation and cloned h, without advancing the saved h/world. Also retain per-tick h distance and both proposed argmax choices.",
        release="At k>0 clone the corresponding playback endpoint world twice, one per evolved h; then execute 1024 actual observation/action ticks independently with frozen weights and true eleven-way argmax. Change only future family/regime via original evaluator. At k=0 reuse the original common endpoint without a preceding extra regime switch; execute2048 ticks to exactly reproduce original state-transfer counts. k=0 is common to both histories and counted once.",
        request_matching="Within each seed/architecture/history/k, both h release branches have identical starting full world, future external events, scheduler/context RNG states, and mature primary request identities. Scoring includes all eligible post-release requests with mature deadlines, regardless of action/success. Inherited requests are excluded from primary completion, but idle counts use every actual inactive enabled opportunity. Pairing across histories is descriptive total-history comparison, not an h-only intervention.",
        metrics=["per-tick h RMS/L2 per lane, relative-to-k0 RMS, cosine similarity; mean/max and all lanes",
                 "same-input next eleven-action argmax disagreement, by all/active/idle slots and lane",
                 "within-history released ordinary/strict completion counts and paired differences, actual idle numerator/denominator and rates",
                 "all three seed values and equal-seed means; individual residual/disagreement cases retained"],
        release_steps_nonzero=1024, release_steps_zero=2048,
        mature_sample="Expect roughly80 mature family3 requests per seed per nonzero branch (~240 across3); report actual denominators. k0 retains original ~170/seed. No extension or cutoff chosen by score.",
        compute="Three seed workers, CPU only, torch and BLAS threads1; no optimizer, gradient or parameter update. Frozen checkpoint/model hashes checked at start/end.",
        audit="Independent saved-array scoring and model recurrence; actual world replays of both clamped histories; original lifelong native evaluator on all zero branches and selected longest-release branches. Record audit coverage and exact hashes.",
        limits=["Artificial initial-h transplant and externally clamped observations are causal probes, not autonomous deployment success.",
                "The two real histories may differ in world state and receipts; only two h releases within one history/k isolate initial h.",
                "Finite same-input persistence or convergence is not proof of an attractor, global basin, or asymptotic recovery.",
                "Three trained seeds and one unfamiliar stream per seed; plain/symmetric differ in parameterization and previous training trajectories."])
    plan["protocol_hash"] = state_digest(plan)
    e.write(OUT / "protocol.json", plan)
    print(json.dumps(dict(protocol_hash=plan["protocol_hash"], created_utc=plan["created_utc"])), flush=True)


def distance(a, b, initial_rms):
    delta = (a.double() - b.double())
    rms = delta.square().mean(-1).sqrt()
    norm = delta.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)
    return dict(rms_mean=float(rms.mean()), rms_max=float(rms.max()),
                l2_mean=float(norm.mean()), l2_max=float(norm.max()),
                cosine_mean=float(cos.mean()), per_lane_rms=rms.tolist(),
                per_lane_l2=norm.tolist(), per_lane_cosine=cos.tolist(),
                relative_initial_rms=float(rms.mean()) / initial_rms)


@torch.inference_mode()
def next_disagreement(model, world_state, hs):
    world = ContextWorld.from_state_dict(world_state, device="cpu")
    obs = world.observe()
    choices = {name: model.step(obs, hs[name].clone(), world.elapsed)[0].argmax(-1) for name in STATES}
    diff = choices["standard"] != choices["transplant"]
    active = world.active
    idle = ~active
    return dict(disagree=int(diff.sum()), slots=diff.numel(), active_disagree=int((diff & active).sum()),
                active_slots=int(active.sum()), idle_disagree=int((diff & idle).sum()), idle_slots=int(idle.sum()),
                per_lane_disagree=diff.sum(-1).tolist(),
                standard_actions=choices["standard"].tolist(), transplant_actions=choices["transplant"].tolist(),
                observation_sha=state_digest(obs))


def old_record(arch, seed):
    path = ROOT / f"data/learn-remember/state-transfer-v1/seed_{seed:03d}.json"
    return next(x for x in e.read(path) if x["arch"] == arch)


def release(model, world, h, seed, arch, driver, point, state, plan):
    stem = OUT / f"seed_{seed:03d}" / arch / f"{driver}_k{point:04d}_{state}"
    steps = plan["release_steps_zero"] if point == 0 else plan["release_steps_nonzero"]
    scores, frames, states = e.rollout(model, world, h, family=3, condition="long_both", steps=steps)
    if point == 0:
        old = old_record(arch, seed)
        assert scores["counts"] == old["counts_original" if state == "standard" else "counts_transplant"]
        assert scores["start_h_hash"] == old["from_h_hash" if state == "standard" else "donor_h_hash"]
        assert scores["start_world_hash"] == old["world_hash"]
        assert scores["primary_identity_hash"] == old["request_digest"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), **frames)
    torch.save(states, stem.with_suffix(".pt"))
    scores.update(seed=seed, arch=arch, driver=driver, clamp_ticks=point, entry_state=state,
                  protocol_hash=plan["protocol_hash"], checkpoint_sha256=e.sha(checkpoint(arch, seed)),
                  old_zero_reproduced=(point == 0))
    scores["artifacts"] = {str(stem.with_suffix(ext)): e.sha(stem.with_suffix(ext)) for ext in (".npz", ".pt")}
    e.write(stem.with_suffix(".json"), scores)
    return scores


@torch.inference_mode()
def run(seed):
    torch.set_num_threads(1)
    plan = checked()
    seed_path = OUT / f"seed_{seed:03d}.json"
    if seed_path.exists():
        raise FileExistsError(seed_path)
    started = time.perf_counter()
    common = e.common_stream(seed)
    records, probes = [], []
    for arch in ARCHES:
        model, native, trained_h, _ = load_checkpoint(checkpoint(arch, seed))
        model.eval()
        model_hash = state_digest(model.state_dict())
        base_world = ContextWorld.from_state_dict(common["world"], device="cpu")
        assert base_world.seed != native.seed and set(base_world.actor_ids).isdisjoint(native.actor_ids)
        initial = dict(standard=e.warmup(model, common), transplant=trained_h[:8].clone())
        initial_rms = float((initial["standard"].double() - initial["transplant"].double()).square().mean(-1).sqrt().mean())
        assert initial_rms > 0
        zero_probe = dict(seed=seed, arch=arch, driver="shared", clamp_ticks=0,
            distance=distance(initial["standard"], initial["transplant"], initial_rms),
            next_action=next_disagreement(model, common["world"], initial))
        probes.append(zero_probe)
        zero = [release(model, base_world, initial[name], seed, arch, "shared", 0, name, plan) for name in STATES]
        assert_matched(*zero)
        records.extend(zero)
        for driver in STATES:
            live = ContextWorld.from_state_dict(common["world"], device="cpu")
            hs = {name: initial[name].clone() for name in STATES}
            endpoints = {0: dict(world=live.state_dict(), hs={name: h.clone() for name, h in hs.items()})}
            live.set_family(3)
            live.set_regime(**CONDITIONS["long_both"], name="evaluation/3/long_both", dt=.25)
            frames = {key: [] for key in ("obs", "actions", "result", "active", "standard_actions", "transplant_actions", "hidden_rms", "elapsed")}
            for tick in range(1, max(POINTS) + 1):
                obs = live.observe()
                proposals = {}
                for name in STATES:
                    logits, hs[name] = model.step(obs, hs[name], live.elapsed)
                    assert torch.isfinite(hs[name]).all() and torch.isfinite(logits).all()
                    proposals[name] = logits.argmax(-1)
                frames["obs"].append(e.array(obs))
                frames["elapsed"].append(live.elapsed)
                frames["active"].append(e.array(live.active))
                for name in STATES:
                    frames[name + "_actions"].append(e.array(proposals[name]))
                frames["actions"].append(e.array(proposals[driver]))
                frames["hidden_rms"].append(e.array((hs["standard"].double() - hs["transplant"].double()).square().mean(-1).sqrt()))
                live.act(proposals[driver])
                frames["result"].append(e.array(live.last_result))
                if tick in POINTS:
                    endpoints[tick] = dict(world=live.state_dict(), hs={name: h.clone() for name, h in hs.items()})
                    probe = dict(seed=seed, arch=arch, driver=driver, clamp_ticks=tick,
                        world_hash=state_digest(endpoints[tick]["world"]),
                        hidden_hashes={name: state_digest(h) for name, h in hs.items()},
                        distance=distance(hs["standard"], hs["transplant"], initial_rms),
                        next_action=next_disagreement(model, endpoints[tick]["world"], hs))
                    probes.append(probe)
            history_stem = OUT / f"seed_{seed:03d}" / arch / f"history_{driver}"
            frames = {key: np.asarray(value) for key, value in frames.items()}
            np.savez_compressed(history_stem.with_suffix(".npz"), **frames)
            torch.save(dict(endpoints=endpoints, driver=driver, initial_h=initial, model_hash=model_hash), history_stem.with_suffix(".pt"))
            e.write(history_stem.with_suffix(".json"), dict(seed=seed, arch=arch, driver=driver,
                protocol_hash=plan["protocol_hash"], steps=max(POINTS), model_hash=model_hash,
                initial_world_hash=state_digest(common["world"]),
                artifacts={str(history_stem.with_suffix(ext)): e.sha(history_stem.with_suffix(ext)) for ext in (".npz", ".pt")}))
            for point in POINTS[1:]:
                endpoint = endpoints[point]
                world = ContextWorld.from_state_dict(endpoint["world"], device="cpu")
                pair = [release(model, world, endpoint["hs"][name], seed, arch, driver, point, name, plan) for name in STATES]
                assert_matched(*pair)
                records.extend(pair)
                print(f"seed={seed} arch={arch} driver={driver} k={point} strict=" + "/".join(str(x["counts"]["clean_completed"]) for x in pair), flush=True)
        assert state_digest(model.state_dict()) == model_hash
    e.write(seed_path, dict(format=FORMAT, seed=seed, protocol_hash=plan["protocol_hash"],
        elapsed_seconds=time.perf_counter() - started, probes=probes, records=records))
    checked()
    print(f"seed={seed} complete seconds={time.perf_counter()-started:.1f}", flush=True)


def assert_matched(a, b):
    for key in ("start_world_hash", "end_scheduler_hash", "end_context_hash", "end_event_hash", "primary_identity_hash", "model_hash"):
        assert a[key] == b[key], key
    assert a["counts"]["matured"] == b["counts"]["matured"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--seed", type=int, choices=range(3))
    args = parser.parse_args()
    declare() if args.command == "declare" else run(args.seed)
