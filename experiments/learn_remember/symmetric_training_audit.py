"""Independent read-only audit of frozen symmetric training and review artifacts.

No optimizer step, collection action, training run, or source-file mutation is
performed. Worlds are detached reconstructions driven by recorded executions.
Final output waits for all three fixed review endpoints and source audits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.learn_remember import gru_train as plain, symmetric_gru as symmetric
from experiments.continuity.run import state_digest
from experiments.lifelong.world import ContextWorld, NEW, REPLACE

OUT = ROOT / "data/learn-remember/symmetric-v1"
ENDPOINTS = ("random", "short_mid", "short", "wide_mid", "wide", "withdrawal_mid", "withdrawal", "review_mid", "review")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_for(paths, timeout):
    deadline = time.monotonic() + timeout
    while not all(Path(p).exists() for p in paths):
        if time.monotonic() >= deadline:
            raise TimeoutError([str(p) for p in paths if not Path(p).exists()])
        time.sleep(5)


def restore_packed(checkpoint):
    """Reconstruct packed model and Adam independently, without a trainer."""
    with torch.random.fork_rng(devices=[]):
        model = symmetric.constrain(plain.GRUAgent(**checkpoint["model_config"]))
    native = model.state_dict()
    rule = model.core.parametrizations.weight_hh_l0[0]
    for name, value in checkpoint["model"].items():
        if name == "core.weight_hh_l0":
            native["core.parametrizations.weight_hh_l0.original"] = rule.right_inverse(value)
        else:
            native[name] = value
    model.load_state_dict(native, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["config"]["lr"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    check(state_digest(optimizer.state_dict()) == state_digest(checkpoint["optimizer"]), "Adam round-trip differs")
    return model, optimizer


def audit_endpoint(folder, name):
    path = folder / f"{name}.pt"
    cp = load(path)
    width = cp["model_config"]["width"]
    layer_names = sorted(k for k in cp["model"] if "weight_hh_l" in k)
    check(layer_names == ["core.weight_hh_l0"], "unexpected recurrent layers")
    matrix = cp["model"][layer_names[0]].reshape(3, width, width)
    check(torch.isfinite(matrix).all().item(), "nonfinite recurrent matrices")
    check(torch.equal(matrix, matrix.transpose(-1, -2)), "asymmetric checkpoint")
    model, optimizer = restore_packed(cp)
    pairs = model.core.parametrizations.weight_hh_l0.original
    rule = model.core.parametrizations.weight_hh_l0[0]
    multiplicity = torch.bincount(rule.index.flatten())
    check(torch.equal(rule.index, rule.index.T), "reverse directions have different coordinates")
    check(int((multiplicity == 1).sum()) == width and int((multiplicity == 2).sum()) == width*(width-1)//2,
          "unordered-pair index multiplicity")
    check(tuple(pairs.shape) == (3, width*(width+1)//2), "not packed unique pairs")
    trainable_count = sum(p.numel() for p in model.parameters())
    dense_count = sum(p.numel() for p in cp["model"].values())
    check(trainable_count == 93238 and pairs.numel() == 55584 and dense_count == 148246, "parameter-count mismatch")
    dense = symmetric.deploy(model)
    check(all(torch.equal(value, cp["model"][key]) for key, value in dense.state_dict().items()), "dense deployment restore mismatch")
    Adam = []
    expected_step = cp["work"]["gradient_updates"]
    for pname, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if expected_step == 0:
            check(state is None, "untrained endpoint has Adam moments")
            continue
        check(state is not None, "missing Adam state " + pname)
        check(int(state["step"]) == expected_step, "Adam step mismatch " + pname)
        for key in ("exp_avg", "exp_avg_sq"):
            check(state[key].shape == parameter.shape and torch.isfinite(state[key]).all().item(), "wrong Adam moment " + pname)
        check((state["exp_avg_sq"] >= 0).all().item(), "negative Adam second moment")
        Adam.append(dict(name=pname, shape=list(parameter.shape), step=int(state["step"])))
    # Observe on an independent clone; the saved endpoint remains untouched.
    probe = ContextWorld.from_state_dict(cp["world"], device="cpu")
    obs = probe.observe()
    with torch.no_grad():
        logits1, h1 = model.step(obs, cp["hidden"])
        logits2, h2 = dense.step(obs, cp["hidden"])
    check(torch.equal(logits1, logits2) and torch.equal(h1, h2), "CPU packed/dense forward parity")
    meta = read(folder / f"{name}.symmetry.json")
    check(meta["checkpoint_sha256"] == plain.sha(path), "endpoint metadata hash mismatch")
    check(meta["trainable_parameters"] == trainable_count and meta["unique_hidden_coordinates"] == pairs.numel(), "metadata count differs")
    packed_path = folder / f"{name}.packed.pt"
    packed_proof = None
    if packed_path.exists():
        packed = load(packed_path)
        check(packed["deployment_sha256"] == plain.sha(path), "packed deployment hash")
        for key, value in model.state_dict().items():
            check(torch.equal(value, packed["model"][key]), "native packed model differs " + key)
        for key in ("optimizer", "world", "hidden", "work"):
            check(state_digest(cp[key]) == state_digest(packed[key]), "packed endpoint differs " + key)
        packed_proof = dict(path=str(packed_path), sha256=plain.sha(packed_path), native_model_optimizer_world_hidden_exact=True)
    return dict(path=str(path), sha256=plain.sha(path), stage=cp["stage"], round=cp["round"],
                saved_time=cp["saved_time"], world_tick=cp["world"]["tick"], world_time=cp["world"]["now"],
                work=cp["work"], true_trainable_parameters=trainable_count, deployed_dense_coordinates=dense_count,
                independent_pair_coordinates=pairs.numel(), layers=[dict(layer=0, gates=["reset", "update", "new"],
                all_three_exactly_symmetric=True, unique_per_gate=pairs.shape[1], unique_diagonal_per_gate=width,
                unique_nondiagonal_per_gate=width*(width-1)//2)], packed_Adam=Adam,
                optimizer_restore_hash=state_digest(optimizer.state_dict()), dense_packed_next_forward_bit_exact=True,
                native_packed_endpoint=packed_proof), cp


def blank_history():
    return dict(all_tick_slots=0, active_tick_slots=0, family_tick_slots=[0]*4,
                active_family_tick_slots=[0]*4, nonwait_family_labels=[0]*4, command_families=[0]*4)


def add_history(stat, world, labels):
    stat["all_tick_slots"] += labels.numel()
    stat["active_tick_slots"] += int(world.active.sum())
    command = (world.event_kind == NEW) | (world.event_kind == REPLACE)
    for f in range(4):
        mask = world.current_family == f
        stat["family_tick_slots"][f] += int(mask.sum())
        stat["active_family_tick_slots"][f] += int((mask & world.active).sum())
        stat["nonwait_family_labels"][f] += int((mask & (labels != 0)).sum())
        stat["command_families"][f] += int(((world.event_family == f) & command).sum())


def blank_residual():
    return dict(commands=0, nonwait_labels=0, active_tick_slots=0, teacher_nonwait_labels=0,
                labeled_requests=set(), public_command_requests=set())


def add_residual(stat, world, labels, teachers):
    old = world.current_family == 3
    commands = (world.event_family == 3) & ((world.event_kind == NEW) | (world.event_kind == REPLACE))
    nonwait = old & (labels != 0)
    stat["commands"] += int(commands.sum())
    stat["nonwait_labels"] += int(nonwait.sum())
    stat["active_tick_slots"] += int((old & world.active).sum())
    stat["teacher_nonwait_labels"] += int((nonwait & teachers[:,None]).sum())
    for lane, job in nonwait.nonzero().tolist():
        stat["labeled_requests"].add((lane, job, int(world.request_id[lane,job])))
    for lane, job in commands.nonzero().tolist():
        stat["public_command_requests"].add((lane, job, int(world.request_id[lane,job])))


def finish_residual(stat, epochs):
    stat["labeled_requests"] = [list(x) for x in sorted(stat["labeled_requests"])]
    stat["public_command_requests"] = [list(x) for x in sorted(stat["public_command_requests"])]
    stat["nonwait_gradient_label_repetitions"] = stat["nonwait_labels"] * epochs
    return stat


def audit_seed(args):
    seed, timeout = args
    torch.set_num_threads(1)
    folder = OUT / f"seed_{seed:03d}"
    wait_for([folder / "withdrawal.symmetry.json"], timeout)
    checkpoints, raw = {}, {}
    for name in ENDPOINTS[:7]:
        checkpoints[name], raw[name] = audit_endpoint(folder, name)
    cfg = raw["random"]["config"]
    check(cfg["seed"] == seed, "seed mismatch")
    rows = [json.loads(line) for line in (folder/"updates.jsonl").read_text().splitlines() if line]
    check(len(rows) == 34, "wrong main round count")
    world = ContextWorld.from_state_dict(raw["random"]["world"], device="cpu")
    history, residual = {}, blank_residual()
    records, boundaries, trajectories = {}, [], []
    previous_hidden = raw["random"]["hidden"]
    previous_checkpoint = "random"
    total_ticks = 0
    expected_round_updates = cfg["epochs"] * cfg["ticks_per_round"] // cfg["block"]
    for stage, nrounds, allowed, duration, gap in (
        ("short",6,(3,),(1.,4.),(.5,2.)),
        ("wide",16,(0,1,2,3),(1.,16.),(.5,32.)),
        ("withdrawal",12,(0,1,2),(1.,16.),(.5,32.))):
        world.set_family_policy(allowed)
        world.set_regime(duration_range=duration, gap_range=gap, name=f"gru-v1/{stage}")
        stage_rows = [r for r in rows if r["stage"] == stage]
        check([r["round"] for r in stage_rows] == list(range(nrounds)), "missing stage round")
        for index in range(nrounds):
            path = folder / f"trajectory_{stage}_{index:03d}.pt"
            data = load(path)
            check(tuple(data["obs"].shape) == (1024,64,56), "public observation dimensions")
            check(tuple(data["labels"].shape) == tuple(data["actions"].shape) == (1024,64,2), "action dimensions")
            if previous_hidden is not None:
                check(torch.equal(previous_hidden, data["anchor"]), "hidden boundary reset")
                boundaries.append(dict(checkpoint=previous_checkpoint, next_trajectory=path.name, hidden_exact=True))
            before_counts = {k:int(v.sum()) for k,v in world.counts.items()}
            stat = blank_history() if stage == "wide" and index in (12,13,14,15) else None
            row = stage_rows[index]
            beta = max(0.,1.-index/max(1,nrounds//2))
            check(row["teacher_probability"] == beta == data["teacher_probability"], "teacher probability")
            check(row["teacher_lanes"] == int(data["teacher_lanes"].sum()), "teacher lane number")
            for tick in range(1024):
                obs = world.observe(); labels = world.oracle("midpoint")
                check(torch.equal(obs,data["obs"][tick]), f"public observation replay {seed}/{stage}/{index}/{tick}")
                check(torch.equal(labels,data["labels"][tick]), "oracle target replay")
                teachers = data["teacher_lanes"]
                check(torch.equal(data["actions"][tick][teachers],labels[teachers]), "teacher lane execution")
                if stat is not None:add_history(stat,world,labels)
                if stage == "withdrawal":add_residual(residual,world,labels,teachers)
                world.act(data["actions"][tick])
            delta = {k:int(v.sum())-before_counts[k] for k,v in world.counts.items()}
            check(delta == data["metrics_delta"] == row["execution_metrics"], "actual execution count mismatch")
            total_ticks += 1024
            check(row["work"]["environment_ticks"] == total_ticks, "environment budget progression")
            check(row["work"]["gradient_updates"] == total_ticks//1024*expected_round_updates, "Adam budget progression")
            check(row["work"]["gradient_lane_ticks"] == total_ticks*64*32, "training position budget progression")
            check(data["exposure_after"] == world.family_exposures(), "family exposure progression")
            if stat is not None:history[str(index)] = stat
            trajectories.append(dict(path=str(path),sha256=plain.sha(path),teacher_lanes=int(teachers.sum())))
            name = stage+"_mid" if index+1 == nrounds//2 else stage if index+1 == nrounds else None
            previous_hidden = None
            if name:
                check(state_digest(world.state_dict()) == state_digest(raw[name]["world"]), "saved full world mismatch " + name)
                previous_hidden = raw[name]["hidden"]
                previous_checkpoint = name
        start_name = {"short":"random","wide":"short","withdrawal":"wide"}[stage]
        start_cp, end_cp = raw[start_name], raw[stage]
        records[stage] = dict(environment_ticks=end_cp["work"]["environment_ticks"]-start_cp["work"]["environment_ticks"],
            physical_world_time=end_cp["world"]["now"]-start_cp["world"]["now"],
            lane_ticks=end_cp["work"]["lane_ticks"]-start_cp["work"]["lane_ticks"],
            Adam_updates=end_cp["work"]["gradient_updates"]-start_cp["work"]["gradient_updates"],
            gradient_lane_ticks=end_cp["work"]["gradient_lane_ticks"]-start_cp["work"]["gradient_lane_ticks"],
            sum_recorded_round_seconds=sum(r["seconds"] for r in stage_rows),
            checkpoint_elapsed_wall_seconds=end_cp["saved_time"]-start_cp["saved_time"],
            round_count=nrounds, full_world_replay_exact=True)
    print(json.dumps(dict(seed=seed,main_replay="passed",ticks=total_ticks,waiting_for="review")),flush=True)
    wait_for([folder/"review_audit.json",folder/"review.packed.pt",folder/"review.symmetry.json"],timeout)
    for name in ENDPOINTS[7:]:checkpoints[name],raw[name]=audit_endpoint(folder,name)
    review_plan = read(folder/"review_plan.json")
    review_source_audit = read(folder/"review_audit.json")
    check(review_plan["historical_rounds"] == [12,13,14,15], "history selection changed")
    check(history == review_plan["historical_stats"] == review_source_audit["historical_stats"], "independent historical counts differ")
    for path, expected in review_plan["bindings"].items():
        check(plain.sha(path) == expected, "review source binding modified")
    wide = raw["wide"]
    restored, optimizer = restore_packed(wide)
    check(review_plan["restoration"]["trainable_parameters"] == sum(p.numel() for p in restored.parameters()), "review restoration parameter count")
    check(all(row["Adam_step"] == wide["work"]["gradient_updates"] for row in review_plan["restoration"]["Adam"]), "review initial Adam steps")
    review_world = ContextWorld.from_state_dict(wide["world"],device="cpu")
    review_world.set_family_policy((0,1,2))
    review_world.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name="gru-v1/withdrawal")
    rng = np.random.default_rng();rng.bit_generator.state=copy.deepcopy(wide["numpy_rng"])
    review_rows = [json.loads(line) for line in (folder/"review_updates.jsonl").read_text().splitlines() if line]
    check([r["round"] for r in review_rows] == list(range(12)), "review rounds")
    review_residual = blank_residual()
    doses = dict(review_updates=0,live_updates=0,all_historical_tick_slots=0,family3_historical_tick_slots=0,
                 active_historical_tick_slots=0,family3_active_historical_tick_slots=0,
                 historical_nonwait_labels=0,family3_historical_nonwait_labels=0)
    for index,row in enumerate(review_rows):
        data = load(folder/f"trajectory_review_{index:03d}.pt")
        main_data = load(folder/f"trajectory_withdrawal_{index:03d}.pt")
        beta=max(0.,1.-index/6)
        expected_lanes=torch.from_numpy(rng.random(64)<beta)
        check(torch.equal(expected_lanes,data["teacher_lanes"]) and torch.equal(expected_lanes,main_data["teacher_lanes"]), "teacher RNG schedule does not continue wide")
        if index in (0,6):
            source = wide if index == 0 else raw["review_mid"]
            check(torch.equal(source["hidden"],data["anchor"]), "review h did not continue source")
            boundaries.append(dict(checkpoint="wide" if index==0 else "review_mid", next_trajectory=f"trajectory_review_{index:03d}.pt",hidden_exact=True))
        live_public=data["obs"].reshape(1024,64,2,28)
        main_public=main_data["obs"].reshape(1024,64,2,28)
        check(torch.equal(live_public[...,:11],main_public[...,:11]) and torch.equal(live_public[...,24:28],main_public[...,24:28]), "review external public opportunities differ")
        check(data["exposure_after"] == main_data["exposure_after"], "review family opportunity differs")
        before_counts={k:int(v.sum()) for k,v in review_world.counts.items()}
        for tick in range(1024):
            obs=review_world.observe();labels=review_world.oracle("midpoint")
            check(torch.equal(obs,data["obs"][tick]) and torch.equal(labels,data["labels"][tick]),"review actual world/target replay")
            add_residual(review_residual,review_world,labels,expected_lanes)
            check(torch.equal(data["actions"][tick][expected_lanes],labels[expected_lanes]), "review teacher execution")
            review_world.act(data["actions"][tick])
        delta={k:int(v.sum())-before_counts[k] for k,v in review_world.counts.items()}
        check(delta == data["metrics_delta"] == row["execution_metrics"], "review execution metrics")
        check(row["fit"]["updates"] == {"live":224,"review":32}, "review budget allocation")
        source_index=12+index%4
        check(row["fit"]["historical_source_round"] == source_index,"historical round schedule")
        source_path=folder/f"trajectory_wide_{source_index:03d}.pt"
        check(row["fit"]["historical_sha256"] == plain.sha(source_path), "wrong historical data input")
        stat=history[str(source_index)]
        doses["review_updates"]+=32;doses["live_updates"]+=224
        for field,statfield in (("all_historical_tick_slots","all_tick_slots"),("active_historical_tick_slots","active_tick_slots")):
            doses[field]+=4*stat[statfield]
        doses["family3_historical_tick_slots"]+=4*stat["family_tick_slots"][3]
        doses["family3_active_historical_tick_slots"]+=4*stat["active_family_tick_slots"][3]
        doses["historical_nonwait_labels"]+=4*sum(stat["nonwait_family_labels"])
        doses["family3_historical_nonwait_labels"]+=4*stat["nonwait_family_labels"][3]
        for key in ("gradient_updates","gradient_lane_ticks","environment_ticks","lane_ticks"):
            check(row["work"][key] == [r for r in rows if r["stage"]=="withdrawal"][index]["work"][key],"review total progression differs " + key)
        name="review_mid" if index==5 else "review" if index==11 else None
        if name:
            check(state_digest(review_world.state_dict())==state_digest(raw[name]["world"]),"review saved full world differs")
    for key in ("gradient_updates","gradient_lane_ticks","environment_ticks","lane_ticks"):
        check(raw["review"]["work"][key] == raw["withdrawal"]["work"][key],"review/no-review total budget differs")
    check(doses["review_updates"] == raw["review"]["work"]["review_updates"] == 384,"review update total")
    check(doses["family3_historical_tick_slots"] == raw["review"]["work"]["review_family3_gradient_tick_slots"], "review old positions")
    check(doses["family3_historical_nonwait_labels"] == raw["review"]["work"]["review_family3_nonwait_labels"],"review old labels")
    check(rng.bit_generator.state == raw["review"]["numpy_rng"],"review teacher RNG endpoint")
    doses["review_update_fraction"] = doses["review_updates"]/(doses["review_updates"]+doses["live_updates"])
    doses["old_family_fraction_of_history_positions"] = doses["family3_historical_tick_slots"]/doses["all_historical_tick_slots"]
    doses["old_family_fraction_of_total_gradient_positions"] = doses["family3_historical_tick_slots"]/(12288*64*2*32)
    records["review"] = dict(environment_ticks=12288,physical_world_time=3072.,Adam_updates=3072,round_count=12,
        sum_recorded_round_seconds=sum(r["seconds"] for r in review_rows),full_world_replay_exact=True,
        all12_external_event_fields_exact=True,all12_teacher_lane_masks_exact=True,
        initial_world_source="wide.pt exact state used for independent replay",initial_hidden_bit_exact=True,
        reconstructed_source_optimizer_hash=state_digest(optimizer.state_dict()),
        initial_optimizer_evidence="Frozen constructor directly loads wide packed Adam; independent restoration reproduces its entire state exactly and matches recorded initial per-parameter steps. There is no separate raw before-first-update review optimizer snapshot.")
    for name in ("withdrawal","review"):
        rr = residual if name=="withdrawal" else review_residual
        check(rr["commands"] == raw[name]["work"]["withdrawal_residual_family3_commands"] and
              rr["nonwait_labels"] == raw[name]["work"]["withdrawal_residual_family3_labels"] and
              rr["active_tick_slots"] == raw[name]["work"]["withdrawal_residual_family3_active_tick_slots"],"residual old request count")
    result=dict(seed=seed,passed=True,endpoints=checkpoints,stages=records,hidden_boundaries=boundaries,
        historical_rounds=history,review_dose=doses,
        no_review_boundary_residual=finish_residual(residual,32),review_boundary_residual=finish_residual(review_residual,28),
        main_trajectories=trajectories,review_source_audit_sha256=plain.sha(folder/"review_audit.json"),
        review_plan_sha256=plain.sha(folder/"review_plan.json"))
    print(json.dumps(dict(seed=seed,passed=True,all_endpoints=9,total_detached_world_replay_ticks=47104)),flush=True)
    return result


def audit_sources():
    plan=read(OUT/"protocol.json");review=read(OUT/"review_protocol.json")
    check(plan["seeds"]==review["seeds"]==[0,1,2],"seed selection")
    check(plain.sha(symmetric.__file__)==plan["source_sha256"],"frozen symmetric trainer changed")
    check(plain.sha(plain.__file__)==plan["parent_sha256"],"frozen plain trainer changed")
    check(plain.sha(OUT/"protocol.json")==review["parent_protocol_sha256"],"parent protocol changed")
    replacements={}
    amendment=read(OUT/"review-implementation-amendment.json")
    for item in amendment["changed_sources"]:
        check(review["source_files"][item["path"]]==item["original_sha256"]==plain.sha(item["original_snapshot"]),"unpreserved review original")
        replacements[item["path"]]=item["amended_sha256"]
    bindings={}
    for path,expected in review["source_files"].items():
        actual=plain.sha(path);check(actual==replacements.get(path,expected),"review frozen source changed")
        bindings[path]=actual
    return dict(main_protocol_sha256=plain.sha(OUT/"protocol.json"),review_protocol_sha256=plain.sha(OUT/"review_protocol.json"),
                review_amendment_sha256=plain.sha(OUT/"review-implementation-amendment.json"),source_bindings=bindings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout",type=int,default=1800)
    args=parser.parse_args()
    torch.set_num_threads(1)
    started=time.time();sources=audit_sources()
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(audit_seed,[(seed,args.timeout) for seed in (0,1,2)]))
    check(sources==audit_sources(),"sources changed during independent audit")
    output=dict(format="symmetric-independent-training-audit-v1",passed=True,seeds=[0,1,2],
        source=dict(path=str(Path(__file__).resolve()),sha256=plain.sha(__file__)),source_bindings=sources,
        started_time=started,finished_time=time.time(),per_seed={str(r["seed"]):r for r in results},
        scope="Read-only independent endpoint/optimizer reconstruction; exact detached replay of all recorded34,816 main and12,288 review ticks per seed; no optimizer steps or new training.",
        parameter_conclusion="93,238 genuine trainable scalars;55,584 unique recurrent pair coordinates including576 self weights; three192x192 symmetric gate matrices in one recurrent layer;148,246 coordinates only in the dense deployment representation.",
        limits=["Every historical per-round policy weight was not saved. World replay verifies actual observations, labels, executions and counts; it does not independently regenerate every historical own-action argmax.",
                "Adam step/moments and constructor restoration are checked. No separate raw optimizer snapshot immediately before the review branch's first update was retained.",
                "Elapsed round seconds include collection, fitting, I/O and resource scheduling; they are not GPU-only timings.",
                "Review checkpoint legacy replay_family3_during_withdrawal=0 is inherited main metadata; review_* counters and independently counted review_dose are authoritative.",
                "Old-task withdrawal stops future NEW family3, while already-issued requests can produce residual public commands and supervised labels. Both residual and explicit historical review are reported separately.",
                "Historical-position fractions describe sample/update allocation, not measured gradient magnitude. Training used labels from private causal oracle, never private model inputs.",
                "Symmetric gate matrices do not by themselves imply symmetric effective GRU influences; behavioral reciprocity is audited separately."])
    plain.write_json(OUT/"training-audit.json",output)
    print(str(OUT/"training-audit.json"),flush=True)


if __name__=="__main__":main()
