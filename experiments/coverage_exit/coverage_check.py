"""Read-only post-run checks and compact tables for the coverage study."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import csv
import gzip
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
from experiments.coverage_exit import coverage as c
from experiments.continuity.run import state_digest


def verify(path):
    protocol=c.check_protocol(path)
    out=Path(protocol["output"])
    results=c.summary(path)
    checks=[]
    files={}
    for seed in range(3):
        archived=torch.load(c.checkpoint(seed,196608),map_location="cpu",weights_only=False)
        for arm in c.ARMS:
            folder=out/"training"/f"seed_{seed:03d}"/arm
            terminal=c.read(folder/"terminal.json")
            cp=Path(terminal["checkpoint"])
            if c.sha(cp)!=terminal["checkpoint_sha256"]:
                raise ValueError("endpoint bytes differ from terminal")
            state=torch.load(cp,map_location="cpu",weights_only=False)
            count=0
            audits=sorted((folder/"audit").glob("*.pt.gz"))
            if len(audits)!=c.UPDATES:
                raise ValueError("not exactly 128 actual live audit blocks")
            for offset,audit_path in enumerate(audits):
                payload=torch.load(io.BytesIO(gzip.decompress(audit_path.read_bytes())),map_location="cpu",weights_only=False)
                files[str(audit_path)]=c.sha(audit_path)
                actions=payload["actions"]
                live=payload["live"]
                point=state["trace"][1024+offset]
                if payload["trace"]!=point or payload["protocol_sha256"]!=c.sha(path):
                    raise AssertionError("audit differs from complete endpoint/protocol")
                for name in ("observations","targets","start_h","end_h"):
                    if not torch.equal(payload["record"][name],live[name]):
                        raise AssertionError("credit record differs from actual live chunk")
                if state_digest(live["actions"])!=point["action_chunk_hash"]:
                    raise AssertionError("actual actions do not match original trace hash")
                if not torch.equal(live["actions"],actions["actual_actions"]) or not torch.equal(live["targets"],actions["targets"]):
                    raise AssertionError("sidecar action/label arrays differ")
                changed=actions["actual_actions"].ne(actions["model_actions"])
                if not torch.equal(changed,actions["assisted"]):
                    raise AssertionError("changes are not exactly counted as assists")
                if not bool(actions["selected"][changed].all()) or not bool(actions["accepted"][changed].all()):
                    raise AssertionError("assist was not externally selected and accepted")
                if not bool((actions["actual_actions"][changed].gt(0)&actions["actual_actions"][changed].lt(3)).all()):
                    raise AssertionError("teacher supplied another action")
                if any(value is not True for key,value in payload["checks"].items() if key.endswith("exact")):
                    raise AssertionError("runtime independent original replay check failed")
                count+=int(changed.sum())
            if count!=terminal["counts"]["assisted"]:
                raise AssertionError("teacher count differs from all recorded action changes")
            control_exact=None
            if arm=="control":
                control_exact=state["trace"][1024:1152]==archived["trace"][1024:1152]
                if not control_exact:
                    raise AssertionError("control original continuation differs from archived original training")
            files[str(cp)]=c.sha(cp)
            checks.append(dict(seed=seed,arm=arm,updates=128,actual_action_change_count=count,
                original_replay_exact_blocks=128,recorded_credit_matches_actual_live=True,
                control_all_128_original_archival_trace_points_exact=control_exact))
    # External inputs match; arm-specific receipts, h and oracle labels may differ.
    for seed in range(3):
        a=torch.load(out/"training"/f"seed_{seed:03d}"/"control"/"checkpoint_final.pt",map_location="cpu",weights_only=False)
        b=torch.load(out/"training"/f"seed_{seed:03d}"/"begin_quarter"/"checkpoint_final.pt",map_location="cpu",weights_only=False)
        for pa,pb in zip(a["trace"][1024:],b["trace"][1024:]):
            if pa["event_hash"]!=pb["event_hash"] or pa["actual_family_exposures"]!=pb["actual_family_exposures"]:
                raise AssertionError("paired external command streams differ")
    c.write(out/"verification.json",dict(format=c.FORMAT,protocol_sha256=c.sha(path),
        checks=checks,external_streams_exact_between_arms=True,files=files,
        recording_fields=dict(explicit_actual_actions="last_chunk.actions, retained replay block.actions, native trace action hashes/counts, coverage actions.actual_actions",
            credit_tape="Original schema has no actions field. It stores observations including actual-action receipts, causal targets, h and recorded parameters.",
            model_proposals="coverage actions.model_actions, before in-place BEGIN substitution",
            clarification="The frozen protocol's tape/trace phrase refers to actual causal experience; explicit action tensors follow the original native schema above."),
        analysis_source=dict(path=str(Path(__file__).resolve()),sha256=c.sha(__file__))))
    training=[]
    for terminal in results["training"]:
        counts=terminal["counts"]
        training.append(dict(seed=terminal["seed"],arm=terminal["arm"],
            commands=counts["commands"],selected=counts["selected_commands"],
            selected_fraction=counts["selected_commands"]/counts["commands"],
            actual_assists=counts["assisted"],assists_fraction=counts["assisted"]/counts["commands"],
            eligible_assists=counts["assisted_eligible"],
            successful_begin=counts["accepted_begin"],model_begin=counts["model_begin"],
            labeled_begin=counts["labeled_begin"],labeled_finish=counts["labeled_finish"],
            aux_labeled_begin=counts["aux_labeled_begin"],aux_labeled_finish=counts["aux_labeled_finish"],
            model_finish=counts["model_finish"],actual_finish=counts["actual_finish"],
            legal_finish_attempts=counts["legally_timed_finish_attempts"],
            accepted_finish=counts["accepted_finish"],
            training_seconds=terminal["training_seconds"],total_seconds=terminal["total_seconds"]))
    with (out/"training.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(training[0]))
        writer.writeheader();writer.writerows(training)
    evalrows=[]
    for row in results["rows"]:
        rates={key:value for key,value in row.items() if key!="counts"}
        rates["begin_rate"]=rates.pop("began")
        counts=dict(row["counts"])
        counts["began_requests"]=counts.pop("began")
        evalrows.append(rates|counts)
    with (out/"evaluation.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(evalrows[0]))
        writer.writeheader();writer.writerows(evalrows)
    return dict(checks=checks,training=training,paired=results["paired"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default=str(c.PROTOCOL))
    parser.add_argument("--affinity",type=int,default=12)
    args=parser.parse_args()
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype=wintypes.BOOL
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1<<args.affinity):
            raise OSError("could not assign requested analysis CPU affinity")
    print(verify(args.protocol))
