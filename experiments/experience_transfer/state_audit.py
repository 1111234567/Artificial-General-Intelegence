"""Independent checks of saved same-input histories and autonomous releases.

The scorer below does not call the producer's scorer. Native evaluator replays
cover all zero branches and k1024 branches. All saved release arrays additionally
receive direct GRU recurrence and independent request/idle accounting checks.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.experience_transfer import state_clamp as s
from experiments.continuity.run import state_digest
from experiments.learn_remember.gru_train import load_checkpoint
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import evaluate, CONDITIONS


def arr(x):
    return x.detach().cpu().numpy().copy()


def digest_id(rows):
    values = sorted((r["actor"], r["job"], r["request_id"], r["family"], r["raw"], r["issued_time"], r["duration"]) for r in rows)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def independent_counts(frames, states, record):
    initial, end = states["initial_world"], states["end_world"]
    tensors, ctx = initial["tensors"], initial["context"]["tensors"]
    trackers = {key: arr(value) for key, value in dict(
        request_id=tensors["request_id"], regime=tensors["current_regime_id"],
        raw=ctx["raw_payload"], family=ctx["current_family"],
        issued_time=tensors["issued_time"], duration=tensors["duration"]).items()}
    jobs = initial["spec"]["n_jobs"]
    enabled = np.broadcast_to(np.arange(jobs) < initial["spec"]["active_jobs"], trackers["request_id"].shape)
    count = Counter()
    rows = []
    for index in range(len(frames["actions"])):
        mature = frames["matured_mask"][index]
        assert np.all(frames["matured_request_id"][index][mature] == trackers["request_id"][mature])
        primary = mature & (trackers["request_id"] >= 0) & (trackers["request_id"] // jobs > record["fork_tick"]) & (trackers["regime"] == record["evaluation_regime_id"])
        count["matured"] += np.count_nonzero(primary)
        count["completed"] += np.count_nonzero(primary & frames["matured_completed"][index])
        count["clean_completed"] += np.count_nonzero(primary & frames["matured_clean_completed"][index])
        for actor, job in zip(*np.nonzero(primary)):
            rows.append(dict(actor=int(initial["spec"]["actor_ids"][actor]), job=int(job),
                request_id=int(trackers["request_id"][actor, job]), family=int(trackers["family"][actor, job]),
                raw=int(trackers["raw"][actor, job]), issued_time=float(trackers["issued_time"][actor, job]),
                duration=float(trackers["duration"][actor, job]), matured_tick=int(frames["tick"][index]),
                completed=bool(frames["matured_completed"][index, actor, job]),
                clean_completed=bool(frames["matured_clean_completed"][index, actor, job])))
        inactive = enabled & ~frames["active"][index]
        actions = frames["actions"][index]
        count["idle_opportunities"] += np.count_nonzero(inactive)
        count["idle_actions"] += np.count_nonzero(inactive & (actions != 0))
        count["idle_begin"] += np.count_nonzero(inactive & (actions >= 1) & (actions <= 2))
        count["idle_finish"] += np.count_nonzero(inactive & (actions >= 3))
        command = np.isin(frames["event_kind"][index], (NEW, REPLACE))
        source_keys = dict(request_id="request_id", regime="current_regime_id", raw="raw_payload", family="current_family", issued_time="issued_time", duration="duration")
        for key, source in source_keys.items():
            trackers[key][command] = frames[source][index][command]
        count["primary_issued"] += np.count_nonzero(command & frames["event_eligible"][index] & (trackers["regime"] == record["evaluation_regime_id"]))
    final = end["tensors"]
    pending = arr(final["_score_pending"]) & (arr(final["_score_request_id"]) // jobs > record["fork_tick"]) & (arr(final["current_regime_id"]) == record["evaluation_regime_id"])
    count["primary_censored"] = np.count_nonzero(pending)
    observed = {key: int(count[key]) for key in record["counts"]}
    assert observed == record["counts"], (observed, record["counts"])
    assert rows == record["primary_requests"]
    assert digest_id(rows) == record["primary_identity_hash"]
    return observed


def no_oracle(*args, **kwargs):
    raise AssertionError("teacher/oracle called during frozen replay")


@torch.inference_mode()
def verify_history(model, stem, initial, probes):
    meta = s.e.read(stem.with_suffix(".json"))
    saved = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
    for path, expected in meta["artifacts"].items():
        assert s.e.sha(path) == expected
    hs = {key: value.clone() for key, value in initial.items()}
    world = ContextWorld.from_state_dict(saved["endpoints"][0]["world"], device="cpu")
    assert state_digest(world.state_dict()) == meta["initial_world_hash"]
    world.set_family(3)
    world.set_regime(**CONDITIONS["long_both"], name="evaluation/3/long_both", dt=.25)
    with np.load(stem.with_suffix(".npz")) as frames:
        assert len(frames["obs"]) == 1024
        for tick in range(1, 1025):
            obs = world.observe()
            assert np.array_equal(arr(obs), frames["obs"][tick-1]), (stem, tick, "obs")
            assert np.array_equal(arr(world.active), frames["active"][tick-1])
            proposals = {}
            for name in s.STATES:
                logits, hs[name] = model.step(obs, hs[name], world.elapsed)
                proposals[name] = logits.argmax(-1)
                assert np.array_equal(arr(proposals[name]), frames[name + "_actions"][tick-1])
            assert np.array_equal(arr(proposals[meta["driver"]]), frames["actions"][tick-1])
            rms = (hs["standard"].double()-hs["transplant"].double()).square().mean(-1).sqrt()
            assert np.array_equal(arr(rms), frames["hidden_rms"][tick-1])
            world.act(proposals[meta["driver"]])
            assert np.array_equal(arr(world.last_result), frames["result"][tick-1])
            if tick in saved["endpoints"]:
                endpoint = saved["endpoints"][tick]
                assert state_digest(world.state_dict()) == state_digest(endpoint["world"])
                for name in s.STATES:
                    assert torch.equal(hs[name], endpoint["hs"][name])
                probe = next(p for p in probes if p["driver"] == meta["driver"] and p["clamp_ticks"] == tick)
                assert probe["distance"]["rms_mean"] == float(rms.mean())
                # Independent endpoint action check, including active/idle strata.
                preview = ContextWorld.from_state_dict(endpoint["world"], device="cpu")
                obs_next = preview.observe()
                a = [model.step(obs_next, hs[name].clone(), preview.elapsed)[0].argmax(-1) for name in s.STATES]
                diff = a[0] != a[1]
                assert int(diff.sum()) == probe["next_action"]["disagree"]
                assert int((diff & preview.active).sum()) == probe["next_action"]["active_disagree"]
                assert int((diff & ~preview.active).sum()) == probe["next_action"]["idle_disagree"]
    return dict(arch=meta["arch"], driver=meta["driver"], ticks=1024,
                end_world_hash=state_digest(world.state_dict()),
                end_h_hashes={name: state_digest(h) for name, h in hs.items()})


@torch.inference_mode()
def verify(seed):
    torch.set_num_threads(1)
    ContextWorld.oracle = no_oracle
    plan = s.checked()
    began = time.perf_counter()
    bundle = s.e.read(s.OUT / f"seed_{seed:03d}.json")
    assert bundle["protocol_hash"] == plan["protocol_hash"]
    checked_releases, native_checks, history_checks = [], [], []
    for arch in s.ARCHES:
        model, _, trained_h, _ = load_checkpoint(s.checkpoint(arch, seed))
        model.eval()
        initial = dict(standard=s.e.warmup(model, s.e.common_stream(seed)), transplant=trained_h[:8].clone())
        arch_path = s.OUT / f"seed_{seed:03d}" / arch
        probes = [p for p in bundle["probes"] if p["arch"] == arch]
        for driver in s.STATES:
            history_checks.append(verify_history(model, arch_path / f"history_{driver}", initial, probes))
        records = [r for r in bundle["records"] if r["arch"] == arch]
        for record in records:
            stem = arch_path / f"{record['driver']}_k{record['clamp_ticks']:04d}_{record['entry_state']}"
            for path, expected in record["artifacts"].items():
                assert s.e.sha(path) == expected
            states = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
            with np.load(stem.with_suffix(".npz")) as frames:
                counts = independent_counts(frames, states, record)
                h = states["initial_h"].clone()
                assert state_digest(h) == record["start_h_hash"]
                # Direct forward recurrence at the original one-tick boundaries
                # avoids the different floating-point kernel used by whole-
                # sequence CPU GRU. Require exact logits/actions/final state.
                for index, obs in enumerate(frames["obs"]):
                    logits, h = model(torch.from_numpy(obs)[None], h)
                    assert np.array_equal(arr(logits[0]), frames["logits"][index])
                    assert np.array_equal(arr(logits[0].argmax(-1)), frames["actions"][index])
                assert torch.equal(h, states["end_h"])
                assert state_digest(states["initial_world"]) == record["start_world_hash"]
                assert state_digest(states["end_world"]) == record["end_world_hash"]
            checked_releases.append(dict(arch=arch, driver=record["driver"], clamp_ticks=record["clamp_ticks"], state=record["entry_state"], counts=counts))
            if record["clamp_ticks"] in (0, 1024):
                world = ContextWorld.from_state_dict(states["initial_world"], device="cpu")
                native = evaluate(model, world, states["initial_h"], family=3, condition="long_both", steps=record["steps"], device="cpu")
                for key in ("matured", "completed", "clean_completed"):
                    assert native["primary"][key] == counts[key]
                assert native["opportunities"]["inactive_actions"] == counts["idle_actions"]
                assert native["opportunities"]["inactive_opportunities"] == counts["idle_opportunities"]
                assert native["end_world_state_hash"] == record["end_world_hash"]
                assert native["end_hidden_hash"] == state_digest({"hidden": states["end_h"]})
                native_checks.append(dict(arch=arch, driver=record["driver"], clamp_ticks=record["clamp_ticks"], state=record["entry_state"], action_hash=native["action_hash"], end_world_hash=native["end_world_state_hash"]))
        print(f"audit seed={seed} arch={arch} releases={len(records)} histories=2", flush=True)
    result = dict(format="state-same-input-independent-audit-v1", seed=seed,
        protocol_hash=plan["protocol_hash"], audit_source_sha256=s.e.sha(__file__),
        oracle_calls=0, elapsed_seconds=time.perf_counter()-began,
        coverage=dict(release_arrays=len(checked_releases), release_recurrence=len(checked_releases),
                      native_complete_releases=len(native_checks), real_playback_histories=len(history_checks)),
        release_checks=checked_releases, native_checks=native_checks, history_checks=history_checks)
    s.e.write(s.OUT / f"audit_seed_{seed:03d}.json", result)
    print(f"audit seed={seed} complete seconds={time.perf_counter()-began:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(3), required=True)
    verify(parser.parse_args().seed)
