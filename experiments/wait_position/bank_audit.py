"""Independent frame-based native-world audit of fixed-bank WAIT candidates."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from experiments.wait_position import bank as builder
from experiments.review_selection import bank as old
from experiments.experience_transfer import train_transfer as source


def audit(seed):
    torch.set_num_threads(1)
    began=time.perf_counter()
    folder=builder.OUT/f"seed_{seed:03d}"
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    data=builder.load(seed)
    original=old.load(seed)
    assert old.sha(ROOT/manifest["original_bank"])==manifest["original_bank_sha256"]
    assert old.sha(folder/"bank.pt")==manifest["bank_sha256"]
    assert old.sha(folder/"eligible_metadata.json")==manifest["metadata_sha256"]
    assert old.sha(builder.__file__)==manifest["implementation_sha256"]
    assert old.sha(source.worlds.__file__)==manifest["world_sha256"]
    assert old.sha(ROOT/"experiments/continuity/world.py")==manifest["base_world_sha256"]
    original_folder=old.OUT/f"seed_{seed:03d}"
    for name, key in (("manifest.json","original_manifest_sha256"),("audit.json","original_audit_sha256")):
        assert old.sha(original_folder/name)==manifest[key]
    original_manifest=json.loads((original_folder/"manifest.json").read_text(encoding="utf-8"))
    original_audit=json.loads((original_folder/"audit.json").read_text(encoding="utf-8"))
    assert original_audit["passed"] and original_audit["bank_sha256"]==manifest["original_bank_sha256"]
    for key, value in original.items():
        assert (torch.equal(data[key],value) if torch.is_tensor(value) else data[key]==value),key
    assert data["obs"].shape==(128,256,56) and data["anchor"].shape==(256,192)
    assert data["eligible_mask"].shape==(128,256,2) and data["eligible_mask"].dtype==torch.bool
    assert data["eligible_time_to_ready"].dtype==data["eligible_processing_fraction"].dtype==torch.float64
    assert len(set(data["ids"]))==256
    assert torch.bincount(data["raw"]).tolist()==[64]*4
    assert int(data["mask"].sum())==768
    by_source=defaultdict(list)
    for i,row in enumerate(data["metadata"]):
        by_source[row["source"]].append(i)
    checkpoint=ROOT/original_manifest["checkpoint"]
    assert old.sha(checkpoint)==original_manifest["checkpoint_sha256"]
    model,_,_,_=source.load_checkpoint(checkpoint,"cpu")
    model.eval()
    total=Counter()
    for src_path,indices in sorted(by_source.items()):
        assert old.sha(ROOT/src_path)==manifest["sources"][src_path]==original_manifest["sources"][src_path]
        trace=torch.load(ROOT/src_path,map_location="cpu",weights_only=False)
        assert trace["teacher_probability"]==0 and not trace["teacher_lanes"].any()
        starts=defaultdict(list)
        for i in indices:
            row=data["metadata"][i]
            starts[row["start_tick"]].append(i)
            lane,start,stop=row["lane"],row["start_tick"],row["stop_tick"]
            assert stop-start==128 and row["cue_tick"]-start==32
            assert torch.equal(data["obs"][:,i],trace["obs"][start:stop,lane])
            assert torch.equal(data["actions"][:,i],trace["actions"][start:stop,lane])
        frames=defaultdict(list)
        h=trace["anchor"].clone()
        with source.mapping_scope(source.mapping_table(seed,"related")),torch.no_grad():
            world=source.worlds.ContextWorld.from_state_dict(trace["initial_world"],device="cpu")
            for tick in range(len(trace["obs"])):
                for i in starts[tick]:
                    assert torch.equal(h[data["metadata"][i]["lane"]],data["anchor"][i]),(src_path,tick,i,"anchor")
                    total["numeric_anchor_rebuilt_exact"]+=1
                obs=world.observe()
                labels=world.oracle("midpoint")
                assert torch.equal(obs,trace["obs"][tick])
                assert torch.equal(labels,trace["labels"][tick])
                for field in ("request_id","active","started","ready_time","duration","version","raw_payload","current_family"):
                    frames[field].append(getattr(world,field).clone())
                frames["now"].append(world.now)
                frames["world_tick"].append(world.tick)
                frames["teacher"].append(labels.clone())
                _,h=model.step(obs,h)
                world.act(trace["actions"][tick])
                frames["result"].append(world.last_result.clone())
                total["native_world_ticks"]+=1
                total["native_obs_label_rows"]+=labels.numel()
        f={key:(torch.tensor(value,dtype=torch.float64) if key=="now" else
                    torch.tensor(value,dtype=torch.long) if key=="world_tick" else torch.stack(value))
           for key,value in frames.items()}
        actual_begin=((trace["actions"]==1+f["version"]) & f["active"] & ~f["started"])
        assert bool((f["result"][actual_begin]==1).all())
        for i in indices:
            row=data["metadata"][i]
            extra=data["eligible_metadata"][i]
            lane,slot,rid=row["lane"],row["slot"],row["request_id"]
            start,stop=row["start_tick"],row["stop_tick"]
            identities=f["request_id"][:,lane,slot]==rid
            begin_ticks=(actual_begin[:,lane,slot] & identities).nonzero().flatten()
            assert len(begin_ticks)==1 and int(begin_ticks[0])>=row["cue_tick"]
            accepted_tick=int(begin_ticks[0])
            ticks=torch.arange(len(trace["obs"]))
            allowed=(identities & (ticks>accepted_tick) & f["active"][:,lane,slot]
                & f["started"][:,lane,slot] & (f["now"]<f["ready_time"][:,lane,slot])
                & (f["teacher"][:,lane,slot]==0))
            expected=torch.zeros((128,2),dtype=torch.bool)
            expected[:,slot]=allowed[start:stop]
            assert torch.equal(expected,data["eligible_mask"][:,i]),(seed,i,"complete admissible set")
            positions=expected[:,slot].nonzero().flatten()
            assert len(positions)>0 and bool(expected[row["positions"][1],slot])
            assert not bool(expected[row["positions"][0],slot]) and not bool(expected[row["positions"][2],slot])
            assert torch.equal(data["labels"][:,i,slot],f["teacher"][start:stop,lane,slot])
            assert bool((f["version"][start:stop,lane,slot][expected[:,slot]]==row["version"]).all())
            assert bool((f["raw_payload"][start:stop,lane,slot][expected[:,slot]]==row["raw"]).all())
            assert bool((f["current_family"][start:stop,lane,slot][expected[:,slot]]==row["context"]).all())
            physical_gap=f["ready_time"][start:stop,lane,slot]-f["now"][start:stop]
            processing_fraction=(f["now"][start:stop]-f["now"][accepted_tick])/f["duration"][start:stop,lane,slot]
            m=expected[:,slot]
            assert torch.equal(physical_gap[m],data["eligible_time_to_ready"][:,i,slot][m])
            assert torch.equal(processing_fraction[m],data["eligible_processing_fraction"][:,i,slot][m])
            assert torch.isnan(data["eligible_time_to_ready"][:,i][~expected]).all()
            assert torch.isnan(data["eligible_processing_fraction"][:,i][~expected]).all()
            assert extra["eligible_positions"]==positions.tolist() and extra["count"]==len(positions)
            assert extra["accepted_begin_tick"]==accepted_tick
            assert extra["accepted_begin_world_tick"]==int(f["world_tick"][accepted_tick])
            assert extra["accepted_begin_time"]==float(f["now"][accepted_tick])
            assert extra["time_to_ready"]==physical_gap[m].tolist()
            assert extra["processing_fraction"]==processing_fraction[m].tolist()
            total["requests"]+=1
            total["eligible_processing_WAIT"]+=int(expected.sum())
            total["original_midpoint_included"]+=1
            total["BEGIN_FINISH_excluded"]+=2
    assert total["requests"]==256 and total["numeric_anchor_rebuilt_exact"]==256
    result=dict(seed=seed,passed=True,checks=dict(total),all_original_fields_exact=True,
        complete_candidate_mask_independently_rebuilt=True,accepted_BEGIN_receipts_verified=True,
        request_actor_slot_version_raw_context_exact=True,strictly_before_real_ready=True,
        original_midpoint_included=True,original_BEGIN_FINISH_excluded=True,
        within_original_128tick_windows=True,private_fields_only_selection_and_audit=True,
        anchor_independent_stepwise_exact=True,anchor_tolerance=dict(atol=0.,rtol=0.),
        bank_sha256=old.sha(folder/"bank.pt"),manifest_sha256=old.sha(folder/"manifest.json"),
        implementation_sha256=old.sha(__file__),elapsed_seconds=time.perf_counter()-began)
    old.write(folder/"audit.json",result)
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    for seed in range(3) if args.seed is None else (args.seed,):
        audit(seed)
