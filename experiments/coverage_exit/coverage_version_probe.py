"""Post-hoc, held-out version decoding on A's original unassisted trajectories.

No behavior is changed and no new world continuation is evaluated. All inputs
are existing original live records. Prepare freezes rules before extraction.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.continuity.run import state_digest
from experiments.lifelong.world import ContextWorld, FEATURE_LAYOUT, FEATURES_PER_JOB
from experiments.coverage_exit import coverage as a
from experiments.undirected import run as native

torch.set_num_threads(1)
FORMAT = "coverage-version-probe-posthoc-v1"
OUT = Path(a.RAW) / "version-probe-posthoc-v1"
PROTOCOL = OUT / "protocol.json"


def pin_affinity(cpu):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype = wintypes.BOOL
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise OSError("requested single logical CPU affinity could not be set")


def prepare(path=PROTOCOL):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("post-hoc protocol is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    original = a.check_protocol(a.PROTOCOL)
    verified_path = a.RAW / "verification.json"
    verified = a.read(verified_path)
    if verified["protocol_sha256"] != a.sha(a.PROTOCOL):
        raise ValueError("A has not been independently verified")
    inputs = {}
    for seed in range(3):
        folder = a.RAW / "training" / f"seed_{seed:03d}" / "control"
        files = sorted((folder / "audit").glob("*.pt.gz"))
        if len(files) != 128:
            raise ValueError("exact original control inventory required")
        for file in files:
            if a.sha(file) != verified["files"][str(file.resolve())]:
                raise ValueError("A control audit input changed")
            inputs[str(file.resolve())] = a.sha(file)
    protocol = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        timing="Added after observing A's free WAIT behavior; not a prospective intervention or preplanned outcome of A.",
        question="After a command pulse, is its version linearly readable from native state when an autonomous BEGIN is still feasible?",
        a_protocol=dict(path=str(a.PROTOCOL),sha256=a.sha(a.PROTOCOL)),
        a_verification=dict(path=str(verified_path),sha256=a.sha(verified_path)),
        original_sources=original["source_pins"], starts=original["starts"], audit_inputs=inputs,
        reader_source=dict(path=str(Path(__file__).resolve()),sha256=a.sha(__file__)),
        output=str(path.parent), seeds=[0,1,2], jobs=[0,1],
        sample_rule=dict(target="original midpoint oracle target is BEGIN 1 or 2; binary label target-1",
            no_pulse="all this-job public kind[0:3], version[3:5], payload, duration, window and context pulse coordinates exactly zero",
            current_input="raw entire public observation in its original 56-coordinate order, without oracle/readiness/request metadata",
            state="post-step/post-observation-pulse h, original recorded parameters, all80 nodes in original index order",
            weights="one row per selected live tick/job, including repeated positions of a request; not independent-request samples"),
        splits=dict(training="updates0..63 and actor positions0..3 only",
            test="updates64..127 and actor positions4..7 only",
            excluded="all other time/actor combinations; no data-driven subset choices",
            request_identity="original case_id + job + request_id; exact no train/test intersection required",
            fitting="separate seed/job classifier; no cross-seed/job fitting"),
        features=[dict(name="public_obs",dimension=56,order="original public coordinates0..55"),
            dict(name="all_state",dimension=80,order="native node indices0..79"),
            dict(name="non_input_state",dimension=24,order="ascending complement of model.input_ports; excludes all56 input port nodes"),
            dict(name="native_begin",dimension=0,order="argmax of native logits[...,1:3]; no fitted parameters")],
        ridge=dict(dtype="CPU float64",target="-1 for version0, +1 for version1",
            standardization="unweighted training-row population mean and population standard deviation only",
            zero_variance="exact zero training variance columns excluded; effective standardized coefficient fixed zero",
            objective="sum_i [1/(2*n_train_class_i)]*(y_i-b-z_i@beta)^2 + lambda*sum_j beta_j^2",
            class_balance="each training class total loss weight1/2; total weight1",
            intercept="fitted, not penalized",lambda_value=0.001,
            fitting="closed-form float64 linear solve, no tuning, no search, no threshold fitting",
            prediction="score>=0 predicts version1, else version0; native ties use original torch.argmax version0"),
        metrics=dict(primary="held-out balanced accuracy = mean of per-class recalls",
            save="class denominators, confusion counts, ordinary accuracy, recalls, number of requests, observed pulse ages, fixed standardization/coefficients",
            uncertainty="six seed/job descriptive results; actors, repeated ticks and jobs are not independent training seeds"),
        reconstruction="one original step per recorded tick; original model-action, block start_h/end_h, world observation/oracle/request-version and world end hash must all match",
        limitations=["Decoding uses external fitted readouts; it does not show autonomous action ability.",
            "Current cue is excluded, but earlier pulse history remains in native state; this is not a long-memory test.",
            "All labels and request identities are analysis-only and never model/probe features.",
            "Time and actor changes both separate held-out data; the probe is not optimized on the held-out set."])
    a.write(path,protocol)
    return protocol


def check(path):
    protocol=a.read(path)
    if protocol["format"] != FORMAT or a.sha(__file__) != protocol["reader_source"]["sha256"]:
        raise ValueError("post-hoc reader/protocol mismatch")
    for field in ("a_protocol","a_verification"):
        pin=protocol[field]
        if a.sha(pin["path"]) != pin["sha256"]:
            raise ValueError("bound A provenance changed")
    for source,expected in protocol["original_sources"].items():
        if a.sha(source)!=expected:
            raise ValueError("frozen source changed")
    for start in protocol["starts"]:
        if a.sha(start["path"]) != start["sha256"]:
            raise ValueError("original starting state changed")
    for source,expected in protocol["audit_inputs"].items():
        if a.sha(source)!=expected:
            raise ValueError("original control record changed")
    return protocol


def metrics(labels,predictions):
    matrix=torch.bincount(2*labels+predictions,minlength=4).reshape(2,2)
    counts=matrix.sum(1)
    if not bool(counts.gt(0).all()):
        raise ValueError("both classes must exist in every declared test group")
    recall=matrix.diag().double()/counts
    return dict(n=int(counts.sum()),class_counts=counts.tolist(),confusion_truth_by_prediction=matrix.tolist(),
        recall=recall.tolist(),balanced_accuracy=float(recall.mean()),
        accuracy=float(matrix.diag().sum().double()/counts.sum().double()))


def fit_predict(train,test,labels):
    if train.dtype!=torch.float64 or test.dtype!=torch.float64 or not bool(torch.isfinite(train).all()) or not bool(torch.isfinite(test).all()):
        raise ValueError("probe features must be finite CPU FP64")
    mean=train.mean(0)
    scale=(train-mean).square().mean(0).sqrt()
    keep=scale.ne(0)
    z=(train[:,keep]-mean[keep])/scale[keep]
    ztest=(test[:,keep]-mean[keep])/scale[keep]
    design=torch.cat((torch.ones((len(train),1),dtype=torch.float64),z),1)
    test_design=torch.cat((torch.ones((len(test),1),dtype=torch.float64),ztest),1)
    counts=torch.bincount(labels,minlength=2)
    if not bool(counts.gt(0).all()):
        raise ValueError("both classes required in every training group")
    weights=1/(2*counts[labels].double())
    if abs(float(weights.sum())-1)>1e-12:
        raise AssertionError("class-balanced training weight sum differs from1")
    target=2*labels.double()-1
    penalty=torch.eye(design.shape[1],dtype=torch.float64)*0.001
    penalty[0,0]=0
    lhs=design.T@(weights[:,None]*design)+penalty
    rhs=design.T@(weights*target)
    beta=torch.linalg.solve(lhs,rhs)
    residual=float((lhs@beta-rhs).abs().max())
    if residual>1e-10 or not bool(torch.isfinite(beta).all()):
        raise ArithmeticError("ridge normal equation check failed")
    coefficients=torch.zeros(train.shape[1],dtype=torch.float64)
    coefficients[keep]=beta[1:]
    score=test_design@beta
    return score.ge(0).long(),dict(train_mean=mean.tolist(),train_std=scale.tolist(),
        zero_variance_columns=(~keep).nonzero().flatten().tolist(),kept_columns=keep.nonzero().flatten().tolist(),
        standardized_coefficients=coefficients.tolist(),intercept=float(beta[0]),
        train_class_counts=counts.tolist(),normal_equation_max_abs_residual=residual,
        test_score_min=float(score.min()),test_score_max=float(score.max()))


def extract(seed,protocol):
    start=next(entry for entry in protocol["starts"] if entry["seed"]==seed)
    trainer=native.NativeTrainer.from_checkpoint(start["path"],device="cpu")
    if state_digest(trainer.semantic_state())!=start["semantic_hash"]:
        raise ValueError("not the declared complete starting state")
    model,world=trainer.model,trainer.world
    if (model.input_size,model.width,model.n_jobs)!=(56,80,2):
        raise ValueError("original H80 B8 two-job model required")
    input_ports=model.input_ports.tolist()
    non_input=[index for index in range(80) if index not in set(input_ports)]
    if len(set(input_ports))!=56 or len(non_input)!=24:
        raise ValueError("invalid original port layout")
    cases=list(world.case_ids)
    actors=list(world.actor_ids)
    batches={(job,split):[] for job in range(2) for split in ("train","test")}
    sampling=Counter()
    files=sorted((a.RAW/"training"/f"seed_{seed:03d}"/"control"/"audit").glob("*.pt.gz"))
    hidden=trainer.h.clone()
    for offset,file in enumerate(files):
        audit=torch.load(io.BytesIO(gzip.decompress(file.read_bytes())),map_location="cpu",weights_only=False)
        record=audit["record"]
        if not torch.equal(hidden,record["start_h"]):
            raise AssertionError("original block start_h changed")
        # Loading only recorded tensors does not initialize or perturb the model.
        model.load_state_dict({**record["parameters"],**record["buffers"]},strict=True)
        block_logits=[]
        with torch.no_grad():
            for tick_index,(obs,elapsed) in enumerate(zip(record["observations"],record["elapsed"])):
                replay_obs=world.observe().to(dtype=torch.float64)
                targets=world.oracle("midpoint")
                if not torch.equal(replay_obs,obs) or not torch.equal(targets,record["targets"][tick_index]):
                    raise AssertionError("original public observation/oracle changed")
                logits,hidden=model.step(obs,hidden,elapsed)
                block_logits.append(logits)
                if not torch.equal(logits.argmax(-1),audit["actions"]["model_actions"][tick_index]):
                    raise AssertionError("native action reconstruction differs")
                for job in range(2):
                    local=obs[:,job*FEATURES_PER_JOB:(job+1)*FEATURES_PER_JOB]
                    pulse=torch.cat((local[:,:11],local[:,24:28]),1)
                    no_pulse=pulse.eq(0).all(1)
                    begin=targets[:,job].gt(0)&targets[:,job].lt(3)
                    selected=begin&no_pulse
                    sampling[f"job{job}_all_begin_positions"]+=int(begin.sum())
                    sampling[f"job{job}_no_pulse_begin_positions"]+=int(selected.sum())
                    for actor_index in selected.nonzero().flatten().tolist():
                        split="train" if offset<64 and actor_index<4 else "test" if offset>=64 and actor_index>=4 else None
                        if split is None:
                            continue
                        label=int(targets[actor_index,job])-1
                        if int(world.version[actor_index,job])!=label or int(world.event_kind[actor_index,job])!=0:
                            raise AssertionError("label/version or zero-pulse feature rule differs")
                        identity=(cases[actor_index],job,int(world.request_id[actor_index,job]))
                        age_ticks=world.tick-int(world.request_id[actor_index,job])//world.n_jobs
                        if age_ticks<1:
                            raise AssertionError("selected a current command pulse")
                        batches[(job,split)].append(dict(obs=obs[actor_index].clone(),h=hidden[actor_index].clone(),
                            label=label,native=int(logits[actor_index,job,1:3].argmax()),
                            request=identity,actor_position=actor_index,tick=world.tick,age_ticks=age_ticks,
                            family=int(world.current_family[actor_index,job])))
                world.act(audit["actions"]["actual_actions"][tick_index])
        if not torch.equal(hidden,record["end_h"]) or not torch.equal(torch.stack(block_logits).argmax(-1),audit["actions"]["actual_actions"]):
            raise AssertionError("original end_h/actions do not match")
        if state_digest(world.state_dict())!=audit["trace"]["live_world_hash"]:
            raise AssertionError("original world endpoint changed")
    packed={}
    for (job,split),samples in batches.items():
        if not samples:
            raise ValueError("empty declared group")
        packed[f"job{job}_{split}"]={key:torch.stack([s[key] for s in samples]) for key in ("obs","h")}
        packed[f"job{job}_{split}"].update({key:torch.tensor([s[key] for s in samples],dtype=torch.long) for key in ("label","native","actor_position","tick","age_ticks","family")})
        packed[f"job{job}_{split}"]["request"]=[s["request"] for s in samples]
    return dict(seed=seed,case_ids=cases,actor_ids=actors,input_ports=input_ports,non_input_indices=non_input,
        public_feature_layout=FEATURE_LAYOUT,public_features_per_job=FEATURES_PER_JOB,
        sampling=dict(sampling),groups=packed,reconstruction=dict(blocks=128,ticks=16384,
            block_start_h_exact=True,block_end_h_exact=True,native_actions_exact=True,
            original_observations_exact=True,original_oracle_exact=True,original_world_endpoints_exact=True))


def request_summary(group):
    requests=sorted(set(group["request"]))
    ages=group["age_ticks"].double()
    serialized=json.dumps(requests,separators=(",",":"),ensure_ascii=False)
    return dict(positions=len(group["label"]),class_counts=torch.bincount(group["label"],minlength=2).tolist(),
        request_count=len(requests),request_set_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        actor_positions=sorted(set(group["actor_position"].tolist())),
        first_tick=int(group["tick"].min()),last_tick=int(group["tick"].max()),
        pulse_age_ticks=dict(min=int(ages.min()),median=float(torch.quantile(ages,.5)),max=int(ages.max())))


def run(path=PROTOCOL):
    protocol=check(path)
    out=Path(protocol["output"])
    if (out/"results.json").exists():
        raise FileExistsError("probe results already exist")
    started=time.perf_counter()
    rows=[]
    datasets=[]
    for seed in range(3):
        dataset=extract(seed,protocol)
        target=out/f"seed_{seed:03d}_dataset.pt.gz"
        buffer=io.BytesIO()
        torch.save(dataset,buffer)
        target.write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
        datasets.append(dict(seed=seed,path=str(target),sha256=a.sha(target),reconstruction=dataset["reconstruction"],
            input_ports=dataset["input_ports"],non_input_indices=dataset["non_input_indices"],
            actor_ids=dataset["actor_ids"],case_ids=dataset["case_ids"],sampling=dataset["sampling"]))
        for job in range(2):
            train=dataset["groups"][f"job{job}_train"]
            test=dataset["groups"][f"job{job}_test"]
            overlap=set(train["request"])&set(test["request"])
            if overlap or set(train["actor_position"].tolist())&set(test["actor_position"].tolist()):
                raise AssertionError("train/test request or actor overlap")
            row=dict(seed=seed,job=job,train=request_summary(train),test=request_summary(test),
                no_request_overlap=True,no_actor_overlap=True,prediction={})
            feature_sets=dict(public_obs=(train["obs"],test["obs"]),all_state=(train["h"],test["h"]),
                non_input_state=(train["h"][:,dataset["non_input_indices"]],test["h"][:,dataset["non_input_indices"]]))
            for name,(train_x,test_x) in feature_sets.items():
                predicted,fit=fit_predict(train_x,test_x,train["label"])
                row["prediction"][name]=dict(heldout=metrics(test["label"],predicted),fit=fit)
            row["prediction"]["native_begin"]=dict(heldout=metrics(test["label"],test["native"]))
            rows.append(row)
        print(json.dumps(dict(seed=seed,reconstructed_blocks=128,seconds=time.perf_counter()-started)),flush=True)
    check(path)
    result=dict(format=FORMAT,post_hoc=True,protocol_sha256=a.sha(path),
        source_sha256=a.sha(__file__),seconds=time.perf_counter()-started,datasets=datasets,rows=rows,
        no_new_learning_or_behavior_evaluation=True,
        aggregation="Per seed/job held-out label positions. Balanced accuracy averages class recalls; repeated positions and the two jobs do not create extra independent training seeds.")
    a.write(out/"results.json",result)
    compact=[dict(seed=row["seed"],job=row["job"],train_counts=row["train"]["class_counts"],
        test_counts=row["test"]["class_counts"],age_ticks=row["test"]["pulse_age_ticks"],
        balanced_accuracy={name:value["heldout"]["balanced_accuracy"] for name,value in row["prediction"].items()}) for row in rows]
    a.write(out/"compact.json",dict(format=FORMAT,protocol_sha256=a.sha(path),rows=compact))
    return compact


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--affinity",type=int,default=12)
    args=parser.parse_args()
    pin_affinity(args.affinity)
    if args.command=="prepare":
        prepare(args.protocol)
        print(json.dumps(dict(protocol=str(args.protocol),sha256=a.sha(args.protocol))),flush=True)
    else:
        print(json.dumps(run(args.protocol)),flush=True)
