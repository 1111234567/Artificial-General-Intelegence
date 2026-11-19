"""Signed remaining-time labels for the immutable native earliest streams.

Private ready/current clocks are used only to derive labels.  No new world is
collected, no old archive is changed, and eligibility is diagnostic only.  A
separate accepted-BEGIN ledger rebuilds native ready times from public command
durations and actual accepted actions before comparing every label.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import torch

from experiments.action_split import data as source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/remaining-time/targets-v1"
PLAN = OUT / "protocol.json"
WARM, TRAIN_STEPS, TOTAL, BLOCK, BATCH, DT = (
    source.WARM, source.TRAIN_STEPS, source.TOTAL, source.BLOCK, source.BATCH, source.DT)
SEEDS = source.SEEDS
TIME_SCALE = 16.
sha, read, write_new = source.sha, source.read, source.write_new
TARGET_KEYS = ("time_target", "time_mask")
FACT_KEYS = ("ready", "now", "tick", "active", "started", "duration", "eligible",
             "rid", "version", "payload", "deadline", "actions", "receipts",
             "event_kind", "event_id", "event_version", "event_payload",
             "event_duration", "event_window")
DURATION_STRATA = (("duration_lt8", None, 8.), ("duration_8_10", 8., 10.),
                   ("duration_10_12", 10., 12.), ("duration_12_14", 12., 14.),
                   ("duration_14_16", 14., 16.), ("duration_ge16", 16., None))


def target_path(seed):
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    return OUT / f"seed_{int(seed):03d}" / "targets.npz"


def load_targets(seed, keys=None):
    with np.load(target_path(seed), allow_pickle=False) as archive:
        return {key: archive[key] for key in (archive.files if keys is None else keys)}


def derive_targets(ready, now, active, started):
    """Float64 physical subtraction, float32 output, finite zero off-mask.

    Eligibility, deadlines and future outcomes are deliberately absent from
    the interface.  Even masked +/-inf and NaN ready sentinels are never used
    in a subtraction, so an invalid arithmetic warning cannot be hidden.
    """
    ready = np.asarray(ready, dtype=np.float64)
    active, started = np.asarray(active), np.asarray(started)
    if active.shape != ready.shape or started.shape != ready.shape:
        raise ValueError("ready, active and started shapes must match")
    if active.dtype != np.bool_ or started.dtype != np.bool_:
        raise TypeError("active and started must be bool")
    now = np.asarray(now, dtype=np.float64)
    if not np.isfinite(now).all():
        raise ValueError("current clock must be finite")
    if now.ndim == 1 and ready.ndim > 1 and now.shape[0] == ready.shape[0]:
        now = now.reshape((now.shape[0],) + (1,) * (ready.ndim - 1))
    now = np.broadcast_to(now, ready.shape)
    mask = active & started & np.isfinite(ready)
    raw_time = np.zeros(ready.shape, dtype=np.float64)
    np.subtract(ready, now, out=raw_time, where=mask)
    raw_time /= TIME_SCALE
    return raw_time.astype(np.float32), mask


def counts(facts, targets, start=0, stop=None):
    stop = len(targets["time_mask"]) if stop is None else stop
    take = slice(start, stop)
    mask = targets["time_mask"][take]
    values = targets["time_target"][take]
    eligible = facts["eligible"][take]
    duration = facts["duration"][take]
    result = dict(ticks=int(stop-start), opportunities=int(mask.size),
                  valid=int(mask.sum()), masked=int((~mask).sum()),
                  negative=int((mask & (values < 0)).sum()),
                  zero=int((mask & (values == 0)).sum()),
                  positive=int((mask & (values > 0)).sum()),
                  eligible_mask_diagnostic=int((mask & eligible).sum()),
                  ineligible_mask_diagnostic=int((mask & ~eligible).sum()))
    for name, low, high in DURATION_STRATA:
        stratum = np.ones(mask.shape, dtype=bool)
        if low is not None:
            stratum &= duration >= low
        if high is not None:
            stratum &= duration < high
        result[name] = int((mask & stratum).sum())
    assert result["valid"] == result["negative"] + result["zero"] + result["positive"]
    assert result["valid"] == sum(result[name] for name, _, _ in DURATION_STRATA)
    assert result["valid"] == result["eligible_mask_diagnostic"] + result["ineligible_mask_diagnostic"]
    return result


def accepted_begin_ledger(facts):
    """Independent row-by-row native state and accepted-BEGIN reconstruction.

    This implementation does not use archived ready values, the target helper,
    the model helper, an oracle, or eligibility to construct its expected
    values.  Commands reset a slot; actual accepted BEGIN records start_time
    plus the command's native float32 duration promoted to float64.  Accepted
    FINISH, cancellation and expiration preserve native ready-clock semantics.
    """
    n, batch, jobs = facts["actions"].shape
    shape = batch, jobs
    active = np.zeros(shape, bool)
    started = np.zeros(shape, bool)
    version = np.zeros(shape, np.int64)
    payload = np.zeros(shape, np.int64)
    rid = np.full(shape, -1, np.int64)
    duration = np.zeros(shape, np.float64)
    deadline = np.zeros(shape, np.float64)
    ready = np.full(shape, np.inf, np.float64)
    begin_time = np.full(shape, np.nan, np.float64)
    expected = np.zeros((n, batch, jobs), np.float32)
    masks = np.zeros((n, batch, jobs), bool)
    begins = finishes = commands = canceled = 0
    for tick in range(n):
        now = float(facts["now"][tick])
        assert int(facts["tick"][tick]) == tick
        assert now == tick * DT
        active &= now <= deadline + 1e-6
        kind = facts["event_kind"][tick]
        command = (kind == 1) | (kind == 2)
        cancel = (kind == 3) & (facts["event_version"][tick] == version)
        active = np.where(command, True, active & ~cancel)
        started[command | cancel] = False
        ready[command | cancel] = np.inf
        begin_time[command | cancel] = np.nan
        for previous, key in ((version, "event_version"), (payload, "event_payload"),
                              (rid, "event_id"), (duration, "event_duration")):
            previous[command] = facts[key][tick][command]
        deadline[command] = (now + facts["event_duration"][tick][command].astype(np.float64)
                             + facts["event_window"][tick][command].astype(np.float64))
        commands += int(command.sum())
        canceled += int(cancel.sum())
        for key, value in (("active", active), ("started", started), ("rid", rid),
                           ("version", version), ("payload", payload),
                           ("duration", duration), ("deadline", deadline), ("ready", ready)):
            np.testing.assert_array_equal(value, facts[key][tick], err_msg=f"ledger {key} tick {tick}")
        mask = active & started & np.isfinite(ready)
        masks[tick] = mask
        # Separate scalar-order arithmetic from derive_targets' ufunc pipeline.
        if mask.any():
            np.testing.assert_array_equal(ready[mask], begin_time[mask] + duration[mask])
            expected[tick][mask] = ((begin_time[mask] + duration[mask] - now) / 16.).astype(np.float32)
        action, receipt = facts["actions"][tick], facts["receipts"][tick]
        begin = (action == 1) | (action == 2)
        finish = (action >= 3) & (action <= 10)
        accepted_begin = begin & active & ~started & (action - 1 == version)
        accepted_finish = (finish & active & started
                           & ((action - 3) // 4 == version) & ((action - 3) % 4 == payload)
                           & (now >= ready - 1e-6) & (now <= deadline + 1e-6))
        expected_receipts = (accepted_begin | accepted_finish).astype(np.int64)
        expected_receipts[(action != 0) & (expected_receipts == 0)] = -1
        np.testing.assert_array_equal(receipt, expected_receipts, err_msg=f"ledger receipts tick {tick}")
        begin_time[accepted_begin] = now
        ready[accepted_begin] = begin_time[accepted_begin] + duration[accepted_begin]
        started = (started | accepted_begin) & ~accepted_finish
        active &= ~accepted_finish
        begins += int(accepted_begin.sum())
        finishes += int(accepted_finish.sum())
    return dict(time_target=expected, time_mask=masks), dict(
        checked_ticks=n, checked_opportunities=int(expected.size), commands=commands,
        matching_cancellations=canceled, accepted_BEGIN=begins, accepted_FINISH=finishes,
        reconstructed_fields=["active", "started", "rid", "version", "payload", "duration", "deadline", "ready"],
        native_ready_clock="every row bitwise exact; accepted_BEGIN float64 start_time + native duration",
        target_comparison="bitwise exact float32; masks exact", eligibility_used=False)


def source_files():
    return [Path(__file__), Path(__file__).with_name("targets_test.py"),
            ROOT / "data/action-split/streams-v1/audit_raw.py", *source.source_files()]


def declare():
    original = source.checked()
    inputs = {}
    for seed in SEEDS:
        path = source.stream_path(seed)
        meta = read(path.with_suffix(".json"))
        prior = read(path.with_name("raw_audit.json"))
        assert sha(path) == meta["stream_sha256"] == prior["stream_sha256"]
        assert prior["native_replay_ticks"] == TOTAL
        assert prior["observations_actions_receipts"] == "exact native replay every tick"
        assert prior["protocol_sha256"] == sha(source.PLAN) == meta["protocol_sha256"]
        assert prior["source_files"] == original["source_files"]
        assert prior["auditor_sha256"] == sha(ROOT / "data/action-split/streams-v1/audit_raw.py")
        paths = (path, path.with_suffix(".json"), path.with_name("raw_audit.json"), path.with_name("world_states.pt"))
        assert sha(paths[-1]) == meta["world_states_sha256"]
        inputs[str(seed)] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    plan = dict(format="remaining-time-targets-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), warm=WARM, train_steps=TRAIN_STEPS, total_ticks=TOTAL,
        block=BLOCK, updates=TRAIN_STEPS // BLOCK, batch_size=BATCH, dt=DT,
        source_files={str(p.relative_to(ROOT)): sha(p) for p in source_files()},
        source_protocol_path=str(source.PLAN.relative_to(ROOT)), source_protocol_sha256=sha(source.PLAN),
        input_sha256=inputs, time_scale=TIME_SCALE,
        target="signed (native ready_time - native current now) /16 in physical time units; compute float64 then save float32",
        mask="active & started & isfinite(native ready_time); current eligibility never selects or weights labels",
        nonmask="finite float32 zero; subtraction evaluated only where mask true",
        inputs="unchanged immutable original public56 observation only; constant dt=.25; no private time or target input",
        continuity="original native world uninterrupted33280ticks, first512 warm, then256 blocks of128; no reset, no collection, no replay as learner exposure",
        budget="counts_training and every one of256 block_counts exclude the512 warm ticks",
        duration_strata={name: dict(low_inclusive=low, high_exclusive=high) for name, low, high in DURATION_STRATA},
        audit="Bind prior full33280tick native replay then independently reconstruct commands, requests, acceptedBEGIN start_time+duration, active/started/ready/receipt every row. Compare independent ledger targets and model helper to float32 labels.",
        negative_coverage="Native earliest finishes at first sampled tick at/after readiness (EPS tolerance). Negative labels generally cover only a sub-dt final slice; this dataset does not teach a long overdue trajectory.")
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), sha256=sha(PLAN))), flush=True)
    return plan


def checked():
    plan = read(PLAN)
    assert (plan["warm"], plan["train_steps"], plan["total_ticks"], plan["block"]) == (WARM, TRAIN_STEPS, TOTAL, BLOCK)
    assert plan["source_protocol_sha256"] == sha(source.PLAN)
    for name, digest in plan["source_files"].items():
        assert sha(ROOT / name) == digest, name
    for bindings in plan["input_sha256"].values():
        for name, digest in bindings.items():
            assert sha(ROOT / name) == digest, name
    return plan


def derive(seed):
    checked()
    path = target_path(seed)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    f = source.load_stream(seed, FACT_KEYS)
    assert f["ready"].dtype == f["duration"].dtype == f["now"].dtype == np.float64
    value, mask = derive_targets(f["ready"], f["now"], f["active"], f["started"])
    targets = dict(time_target=value, time_mask=mask)
    ledger, audit = accepted_begin_ledger(f)
    for key in TARGET_KEYS:
        np.testing.assert_array_equal(targets[key], ledger[key])
    from experiments.remaining_time.model import remaining_time_targets
    helper_target, helper_mask = remaining_time_targets(torch.from_numpy(f["ready"]),
        torch.from_numpy(f["now"]), torch.from_numpy(f["active"]), torch.from_numpy(f["started"]))
    np.testing.assert_array_equal(value, helper_target.float().numpy())
    np.testing.assert_array_equal(mask, helper_mask.numpy())
    audit["model_helper_comparison"] = "native float64 ready/current input then float32: bitwise exact"
    audit["model_helper_source_sha256"] = sha(ROOT / "experiments/remaining_time/model.py")
    blocks = [dict(update=i+1, start_tick=WARM+i*BLOCK, stop_tick=WARM+(i+1)*BLOCK,
                   **counts(f, targets, WARM+i*BLOCK, WARM+(i+1)*BLOCK))
              for i in range(TRAIN_STEPS // BLOCK)]
    training_counts = counts(f, targets, WARM, TOTAL)
    for key, val in training_counts.items():
        assert sum(row[key] for row in blocks) == val, key
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **targets)
    valid_seconds = value[mask].astype(np.float64) * 16.
    metadata = dict(format="remaining-time-targets-seed-v1", seed=int(seed),
        protocol_sha256=sha(PLAN), targets_sha256=sha(path), source_stream_sha256=sha(source.stream_path(seed)),
        arrays={key: dict(shape=list(arr.shape), dtype=str(arr.dtype)) for key, arr in targets.items()},
        counts_all=counts(f, targets), counts_warm=counts(f, targets, 0, WARM),
        counts_training=training_counts, block_counts=blocks, audit=audit,
        signed_physical_range_all=dict(min=float(valid_seconds.min()), max=float(valid_seconds.max())),
        negative_coverage="Native earliest finishes immediately at the first ready sample. Negative rows are short post-ready samples, not a long overdue trajectory.")
    write_new(path.with_suffix(".json"), metadata)
    print(json.dumps(dict(event="derived", seed=seed, counts_training=training_counts,
                          targets_sha256=sha(path))), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "derive", "all"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "declare":
        declare()
    elif args.command == "derive":
        derive(args.seed)
    else:
        checked()
        rows = [derive(seed) for seed in SEEDS]
        write_new(OUT / "completion.json", dict(complete=True, seeds=list(SEEDS), protocol_sha256=sha(PLAN),
            files={str(target_path(seed).relative_to(ROOT)): sha(target_path(seed)) for seed in SEEDS},
            counts_training={str(row["seed"]): row["counts_training"] for row in rows}))


if __name__ == "__main__":
    main()
