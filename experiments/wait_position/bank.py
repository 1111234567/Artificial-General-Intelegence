"""Add real processing-WAIT candidates to the unchanged retained A bank.

The original 256 requests, numeric anchors, observation windows, labels and
three-row masks are copied exactly.  New fields are private selection/audit
metadata and are never arguments to the recurrent policy.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.review_selection import bank as old
from experiments.experience_transfer import train_transfer as source

OUT = ROOT / "data/wait-position/bank-v1"
FORMAT = "wait-position-fixed-A-eligible-bank-v1"
PRIVATE_FIELDS = ("eligible_mask", "eligible_time_to_ready", "eligible_processing_fraction", "eligible_metadata")


def load(seed, root=OUT):
    return torch.load(Path(root) / f"seed_{seed:03d}" / "bank.pt", map_location="cpu", weights_only=False)


def summary(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return dict(n=values.numel(), min=float(values.min()), q25=float(values.quantile(.25)),
                median=float(values.median()), mean=float(values.mean()),
                q75=float(values.quantile(.75)), max=float(values.max()))


def build(seed):
    torch.set_num_threads(1)
    began = time.perf_counter()
    original_folder = old.OUT / f"seed_{seed:03d}"
    folder = OUT / f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    data = old.load(seed)
    assert data["obs"].shape == (128, 256, 56)
    original_manifest = json.loads((original_folder / "manifest.json").read_text(encoding="utf-8"))
    assert old.sha(original_folder / "bank.pt") == original_manifest["bank_sha256"]
    eligible = torch.zeros_like(data["mask"])
    gap = torch.full(eligible.shape, float("nan"), dtype=torch.float64)
    fraction = torch.full_like(gap, float("nan"))
    eligible_meta = [None] * 256
    by_source = defaultdict(list)
    for i, row in enumerate(data["metadata"]):
        by_source[row["source"]].append(i)
    sources = {}
    replay_checks = Counter()
    for path, indices in sorted(by_source.items()):
        digest = old.sha(ROOT / path)
        assert digest == original_manifest["sources"][path]
        sources[path] = digest
        trajectory = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        assert trajectory["teacher_probability"] == 0 and not trajectory["teacher_lanes"].any()
        accepted = {}
        with source.mapping_scope(source.mapping_table(seed, "related")):
            world = source.worlds.ContextWorld.from_state_dict(trajectory["initial_world"], device="cpu")
            for tick in range(len(trajectory["obs"])):
                obs = world.observe()
                labels = world.oracle("midpoint")
                assert torch.equal(obs, trajectory["obs"][tick]), (seed, path, tick, "obs")
                assert torch.equal(labels, trajectory["labels"][tick]), (seed, path, tick, "labels")
                for i in indices:
                    row = data["metadata"][i]
                    if not row["start_tick"] <= tick < row["stop_tick"]:
                        continue
                    lane, slot = row["lane"], row["slot"]
                    same = int(world.request_id[lane, slot]) == row["request_id"]
                    begin = accepted.get((lane, slot, row["request_id"]))
                    is_candidate = (same and begin is not None and row["cue_tick"] <= begin["tick"] < tick
                        and bool(world.active[lane, slot]) and bool(world.started[lane, slot])
                        and world.now < float(world.ready_time[lane, slot]) and int(labels[lane, slot]) == 0)
                    if not is_candidate:
                        continue
                    t = tick - row["start_tick"]
                    assert int(world.actor_ids[lane]) == row["actor_id"]
                    assert int(world.raw_payload[lane, slot]) == row["raw"]
                    assert int(world.version[lane, slot]) == row["version"]
                    assert int(world.current_family[lane, slot]) == row["context"]
                    assert int(data["labels"][t, i, slot]) == 0
                    eligible[t, i, slot] = True
                    gap[t, i, slot] = float(world.ready_time[lane, slot]) - world.now
                    fraction[t, i, slot] = (world.now - begin["now"]) / float(world.duration[lane, slot])
                    if eligible_meta[i] is None:
                        eligible_meta[i] = dict(sample_index=i, id=row["id"], source=path,
                            source_sha256=digest, actor_id=row["actor_id"], lane=lane, slot=slot,
                            request_id=row["request_id"], accepted_begin_tick=begin["tick"],
                            accepted_begin_world_tick=begin["world_tick"], accepted_begin_time=begin["now"],
                            ready_time=float(world.ready_time[lane, slot]), duration=float(world.duration[lane, slot]),
                            window_start_tick=row["start_tick"], window_stop_tick=row["stop_tick"])
                actions = trajectory["actions"][tick]
                begin_ok = ((actions == 1 + world.version) & world.active & ~world.started)
                records = [(int(lane), int(slot), int(world.request_id[lane, slot]))
                           for lane, slot in begin_ok.nonzero().tolist()]
                clock, world_tick = world.now, world.tick
                world.act(actions)
                assert bool((world.last_result[begin_ok] == 1).all())
                for lane, slot, rid in records:
                    accepted[lane, slot, rid] = dict(tick=tick, world_tick=world_tick, now=clock)
                replay_checks["native_world_ticks"] += 1
                replay_checks["native_obs_label_rows"] += labels.numel()
        for i in indices:
            row, extra = data["metadata"][i], eligible_meta[i]
            assert extra is not None, (seed, i, "no eligible processing WAIT")
            slot = row["slot"]
            positions = eligible[:, i, slot].nonzero().flatten().tolist()
            assert positions and row["positions"][1] in positions
            assert row["positions"][0] not in positions and row["positions"][2] not in positions
            assert positions == list(range(positions[0], positions[-1]+1))
            assert not eligible[:, i, 1-slot].any()
            extra.update(eligible_positions=positions, count=len(positions),
                eligible_first_position=positions[0], eligible_last_position=positions[-1],
                eligible_first_source_tick=positions[0]+row["start_tick"],
                eligible_last_source_tick=positions[-1]+row["start_tick"],
                midpoint_position=row["positions"][1],
                time_to_ready=[float(gap[t, i, slot]) for t in positions],
                processing_fraction=[float(fraction[t, i, slot]) for t in positions])
    result = dict(data)
    result.update(eligible_format=FORMAT, eligible_mask=eligible, eligible_time_to_ready=gap,
                  eligible_processing_fraction=fraction, eligible_metadata=eligible_meta)
    for key in data:
        if torch.is_tensor(data[key]):
            assert torch.equal(result[key], data[key]), key
        else:
            assert result[key] == data[key], key
    assert bool((data["labels"][eligible] == 0).all())
    assert bool((gap[eligible] > 0).all()) and bool(((fraction[eligible] > 0) & (fraction[eligible] < 1)).all())
    torch.save(result, folder / "bank.pt")
    old.write(folder / "eligible_metadata.json", eligible_meta)
    counts = eligible.sum((0,2))
    distribution = dict(total_candidates=int(eligible.sum()), candidates_per_request=summary(counts),
        candidate_count_histogram=dict(sorted(Counter(str(int(n)) for n in counts).items(), key=lambda v:int(v[0]))),
        time_to_ready=summary(gap[eligible]), processing_fraction=summary(fraction[eligible]),
        midpoint_fraction=summary([fraction[row['positions'][1], i, row['slot']] for i,row in enumerate(data['metadata'])]),
        per_raw={str(raw):dict(requests=int((data['raw']==raw).sum()),
            total_candidates=int(counts[data['raw']==raw].sum()), candidates_per_request=summary(counts[data['raw']==raw])) for raw in range(4)})
    manifest = dict(format=FORMAT, seed=seed, created_utc=datetime.now(timezone.utc).isoformat(),
        original_bank=str((original_folder/"bank.pt").relative_to(ROOT)),
        original_bank_sha256=old.sha(original_folder/"bank.pt"),
        original_manifest_sha256=old.sha(original_folder/"manifest.json"),
        original_audit_sha256=old.sha(original_folder/"audit.json"),
        original_fields_exact=list(data), samples=256, sequence_ticks=128, sources=sources,
        candidate_rule="Original selected request and slot; original128-tick window; strictly after native accepted BEGIN for that request; native active and started; now < ready_time; native midpoint teacher WAIT0. No outcomes or scores used to select requests.",
        private_fields=list(PRIVATE_FIELDS),
        private_schema="eligible_mask bool[128,256,2]; eligible_time_to_ready and eligible_processing_fraction float64 same shape, NaN outside mask. eligible_metadata is audit only. Model receives only original obs and anchor.",
        original_midpoint_included=True, original_BEGIN_FINISH_excluded=True,
        request_reselection=False, native_replay_checks=dict(replay_checks), distribution=distribution,
        implementation_sha256=old.sha(__file__), world_sha256=old.sha(source.worlds.__file__),
        base_world_sha256=old.sha(ROOT/"experiments/continuity/world.py"),
        bank_sha256=old.sha(folder/"bank.pt"), metadata_sha256=old.sha(folder/"eligible_metadata.json"),
        elapsed_seconds=time.perf_counter()-began)
    old.write(folder/"manifest.json", manifest)
    print(json.dumps(dict(seed=seed, samples=256, **distribution, seconds=time.perf_counter()-began)), flush=True)
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(3))
    args=parser.parse_args()
    for seed in range(3) if args.seed is None else (args.seed,):
        build(seed)
