"""Prospective paired BEGIN-coverage continuation of frozen native checkpoints.

This file is an opt-in observer/intervention around the unchanged native
single-update trainer. Only its world.act argument is changed, in place, so
native recorded experience contains actual actions and consequent receipts.
"""
from __future__ import annotations

import os
for _name, _value in {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
    os.environ[_name] = _value

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.continuity.run import state_digest
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import evaluate
from experiments.undirected import run as native

FORMAT = "coverage-begin-continuation-v1"
START = 131072
UPDATES = 128
END = START + 128 * UPDATES
ARMS = ("control", "begin_quarter")
OLD_PROTOCOL = ROOT / "docs/evidence/undirected-2026-10-30/protocol-v1.json"
RAW = ROOT / "data/coverage-exit/coverage-v1"
PROTOCOL = RAW / "protocol.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    native._atomic_json(path, value)


def sha(path):
    return native.file_sha256(path)


def selected_identity(seed, actor, job, request, kind):
    identity = f"{FORMAT}/{seed}/{actor}/{job}/{request}/{kind}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 4 == 0


def checkpoint(seed, tick=START):
    branch = f"ports__undirected__recorded512__fixed__warmup"
    folder = ROOT / "data/undirected/main-v1/training" / f"seed_{seed:03d}" / branch
    return folder / "checkpoints" / f"tick_{tick:06d}.pt"


def source_pins():
    old = read(OLD_PROTOCOL)
    pins = old["sources"]
    if len(pins) != 26:
        raise ValueError("expected 26 original frozen source pins")
    for name, expected in pins.items():
        if sha(name) != expected:
            raise ValueError(f"original frozen source changed: {name}")
    return dict(pins)


def prepare(path=PROTOCOL):
    path = Path(path)
    if path.exists():
        raise FileExistsError("protocol is immutable; use the committed protocol")
    path.parent.mkdir(parents=True, exist_ok=True)
    pins = source_pins()
    pins[str(Path(__file__).resolve())] = sha(__file__)
    starts, baselines = [], []
    for seed in range(3):
        cp = checkpoint(seed)
        entry = read(cp.parent.parent / "snapshots.json")["entries"][str(START)]
        if str(cp.resolve()) != entry["path"] or sha(cp) != entry["sha256"]:
            raise ValueError("canonical starting checkpoint binding failed")
        starts.append(dict(seed=seed, **entry))
        baseline = ROOT / "data/undirected/main-v1/evaluation" / f"seed_{seed:03d}__ports__undirected__recorded512__fixed__warmup.json"
        baselines.append(dict(seed=seed,path=str(baseline),sha256=sha(baseline)))
    protocol = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        scientific_question="Does occasional successful BEGIN experience break the WAIT/absent-live-FINISH-label closed loop?",
        status="precommitted_before_formal_continuations", starts=starts, baseline_inputs=baselines,
        original_protocol=dict(path=str(OLD_PROTOCOL), sha256=sha(OLD_PROTOCOL)),
        source_pins=pins, output=str(path.parent.resolve()), seeds=[0, 1, 2], arms=list(ARMS),
        start_tick=START, end_tick=END, updates=UPDATES, live_ticks=END-START,
        batch_size=8, width=80, dtype="float64", cpu_threads=1,
        intervention=dict(frequency="one quarter of cue identities in expectation, deterministic external SHA256",
            identity="coverage-begin-continuation-v1/seed/actor_id/job/request_id/event_kind",
            hash="first 8 SHA256 bytes interpreted little-endian, modulo4==0",
            scope="NEW/REPLACE cue tick only, target is feasible correct BEGIN and model action differs",
            execution="change action tensor in place before original world.act; original tape/trace records actual action",
            supplied_actions="BEGIN only; every executed FINISH retains model choice",
            oracle="original midpoint oracle; same existing live labels in both arms; no FINISH demonstration"),
        learning=dict(entrypoint="unchanged NativeTrainer.train_update", live_credit="original recorded512 including inherited tape",
            replay="original reservoir retained, warmup128+suffix128, one auxiliary call per update",
            gradient="original equal-weight live/replay plus original Adam, clipping, lr and moments",
            state="retain original model, h, world/scheduler, Adam, replay, credit, RNG, counters and trace",
            course="remaining original wide stage; no schedule or configuration changes"),
        evaluation=dict(conditions=["short", "long_both"], families=[0,1,2,3], steps=2048,
            baseline="reuse exact archived CP131072 frozen evaluations only after checking byte/source/semantic/continuation bindings",
            endpoint="fresh frozen original evaluator, no assistance, all eleven actions are model argmax",
            independent_repetitions=3, group_unit="seed, equal family weighting; no pooling families as independent seeds"),
        audit="Each update replayed independently under original model/world/oracle with actual actions; exact observation, target, model-action, actual-action, hidden and world-state equality required.",
        stopping="128 updates per arm regardless of interim behavior; preserve observed failures, no replacement/reset",
        implementation="original execution path; no runtime adapter", benchmark_only=False,
        runtime=dict(python=sys.executable,torch=torch.__version__,
            environment={name:os.environ[name] for name in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}))
    write(path, protocol)
    return protocol


def check_protocol(path):
    protocol = read(path)
    if protocol["format"] != FORMAT or protocol["updates"] != UPDATES:
        raise ValueError("foreign protocol")
    for name, expected in protocol["source_pins"].items():
        if sha(name) != expected:
            raise ValueError(f"prospectively pinned source changed: {name}")
    for entry in protocol["starts"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("canonical input checkpoint changed")
    for entry in protocol["baseline_inputs"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("precommitted autonomous baseline input changed")
    return protocol


class BeginObserver:
    def __init__(self, trainer, arm):
        self.trainer, self.arm = trainer, arm
        self.original_act = trainer.world.act
        self.frames = []
        self.counts = Counter()

    def __enter__(self):
        self.trainer.world.act = self.act
        return self

    def __exit__(self, *unused):
        # Remove instance attribute to restore exactly the original class method.
        del self.trainer.world.act

    def act(self, action):
        world = self.trainer.world
        model_action = action.clone()
        target = world.oracle("midpoint").cpu()
        command = world.event_kind.eq(NEW) | world.event_kind.eq(REPLACE)
        selected = torch.zeros_like(command)
        for actor_index, job in command.nonzero().tolist():
            selected[actor_index, job] = selected_identity(self.trainer.config.seed,
                world.actor_ids[actor_index], job, int(world.event_request_id[actor_index,job]),
                int(world.event_kind[actor_index,job]))
        feasible_begin = target.eq(1) | target.eq(2)
        assist = selected & feasible_begin & model_action.ne(target) & (self.arm == "begin_quarter")
        legal = world.active & world.started & (world.now >= world.ready_time - 1e-6) & (world.now <= world.deadline + 1e-6)
        idle = ~world.active
        action[assist] = target[assist]
        if bool((action.ge(3) & action.ne(model_action)).any()):
            raise AssertionError("teacher supplied a FINISH")
        self.original_act(action)
        accepted = world.last_result.eq(1)
        if not bool(accepted[assist].all()):
            raise AssertionError("forced BEGIN was not successful")
        frame = dict(model_actions=model_action, actual_actions=action.clone(), targets=target,
            selected=selected.clone(), assisted=assist.clone(), accepted=accepted.clone())
        self.frames.append(frame)
        masks = dict(job_positions=torch.ones_like(command), commands=command,
            selected_commands=selected, eligible_commands=command & world.current_eligible,
            selected_eligible_commands=selected & world.current_eligible,
            feasible_begin_positions=feasible_begin, labeled_begin=target.gt(0)&target.lt(3),
            labeled_finish=target.ge(3), assisted=assist, assisted_eligible=assist&world.current_eligible,
            model_begin=model_action.gt(0)&model_action.lt(3), model_finish=model_action.ge(3),
            model_wait=model_action.eq(0), actual_begin=action.gt(0)&action.lt(3),
            actual_finish=action.ge(3), actual_wait=action.eq(0),
            accepted_begin=accepted&action.gt(0)&action.lt(3), accepted_finish=accepted&action.ge(3),
            legal_finish_opportunities=legal, legally_timed_finish_attempts=legal&action.ge(3),
            idle_opportunities=idle, idle_actions=idle&action.ne(0))
        for key, mask in masks.items():
            self.counts[key] += int(mask.sum())

    def stacked(self):
        return {key: torch.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}


def verify_actual_block(trainer, model_before, world_before, block, audit):
    """Original unmodified model/world replay; checks labels before actual action."""
    replay_world = ContextWorld.from_state_dict(world_before, device="cpu")
    hidden = block["start_h"].clone()
    with torch.no_grad():
        for index, dt in enumerate(block["elapsed"]):
            obs = replay_world.observe().to(dtype=torch.float64)
            target = replay_world.oracle("midpoint")
            if not torch.equal(obs, block["observations"][index]) or not torch.equal(target, block["targets"][index]):
                raise AssertionError("original world/teacher replay differs")
            output, hidden = model_before.step(obs, hidden, dt)
            if not torch.equal(output.argmax(-1), audit["model_actions"][index]):
                raise AssertionError("separate model action replay differs")
            if not torch.equal(block["actions"][index], audit["actual_actions"][index]):
                raise AssertionError("original native trace did not record actual actions")
            replay_world.act(block["actions"][index])
    if not torch.equal(hidden, trainer.h) or state_digest(replay_world.state_dict()) != state_digest(trainer.world.state_dict()):
        raise AssertionError("independent actual-action replay endpoint differs")
    return dict(checked_ticks=len(block["elapsed"]), hidden_exact=True, world_exact=True,
                observations_exact=True, original_teacher_exact=True, model_actions_exact=True, actual_actions_exact=True)


def one_update(trainer, arm, verify=True):
    before_model = deepcopy(trainer.model)
    before_world = trainer.world.state_dict()
    with BeginObserver(trainer, arm) as observer:
        point = trainer.train_update()
    audit = observer.stacked()
    if point["actual_action_counts"] != torch.bincount(audit["actual_actions"].reshape(-1), minlength=11).tolist():
        raise AssertionError("native action counts do not represent actual actions")
    if point["target_action_counts"] != torch.bincount(audit["targets"].reshape(-1), minlength=11).tolist():
        raise AssertionError("native label counts differ")
    checks = verify_actual_block(trainer, before_model, before_world, trainer.last_chunk, audit) if verify else None
    counts = dict(observer.counts)
    auxiliary = trainer.last_auxiliary
    suffix = auxiliary["block"]["targets"][auxiliary["info"]["warmup_ticks"]:]
    counts.update(aux_labeled_begin=int((suffix.gt(0)&suffix.lt(3)).sum()),
                  aux_labeled_finish=int(suffix.ge(3).sum()),
                  aux_label_positions=suffix.numel())
    return point, audit, counts, checks


def archived_baseline(trainer, source):
    name = f"seed_{trainer.config.seed:03d}__ports__undirected__recorded512__fixed__warmup.json"
    path = ROOT / "data/undirected/main-v1/evaluation" / name
    archive = read(path)
    records = [record for record in archive["records"] if record["fork_tick"] == START - 1]
    # Native tick is the count of admitted observations; world.tick is count-1.
    if len(records) != 8:
        records = [record for record in archive["records"] if record.get("probe", {}).get("tick") == START]
    expected = {(family, condition) for family in range(4) for condition in ("short", "long_both")}
    if {(r["family"],r["condition"]) for r in records} != expected or len(records) != 8:
        raise ValueError("archive lacks exact baseline inventory")
    bound = state_digest(dict(model=trainer.model.state_dict(), world=trainer.world.state_dict(), hidden=trainer.h))
    for record in records:
        if record["source_continuation_hash"] != bound or record["steps"] != 2048:
            raise ValueError("archive baseline differs from canonical complete continuation")
        if record["action_hash"] != record["model_action_hash"] or record["intervention"] != "none" or any(
                record[key] for key in ("world_resets", "hidden_resets", "parameter_updates")):
            raise ValueError("archive baseline is not autonomous frozen evaluation")
        if any(sha(ROOT / name) != expected for name, expected in record["source_hashes"].items()):
            raise ValueError("archive evaluator source differs")
    return dict(format=FORMAT, phase="baseline", archive=str(path), archive_sha256=sha(path),
        checkpoint=source, source_continuation_hash=bound, records=records,
        reuse="exact canonical source-continuation hash, 2048 tick budget, 8 forks and frozen source/action checks")


def endpoint_evaluation(trainer):
    before = state_digest(trainer.semantic_state())
    with _preserve_process_state(), torch.no_grad():
        records = [evaluate(trainer.model, trainer.world, trainer.h, family=family,
                   condition=condition, steps=2048, device="cpu")
                   for family in range(4) for condition in ("short", "long_both")]
    if state_digest(trainer.semantic_state()) != before:
        raise AssertionError("evaluation changed complete continuation")
    return dict(format=FORMAT, phase="endpoint", records=records, semantic_hash=before,
                source_unchanged=True)


def run(path, seed, arm, *, smoke_updates=None):
    protocol = check_protocol(path)
    source = next(entry for entry in protocol["starts"] if entry["seed"] == seed)
    folder = Path(protocol["output"]) / ("smoke" if smoke_updates else "training") / f"seed_{seed:03d}" / arm
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "audit").mkdir()
    trainer = native.NativeTrainer.from_checkpoint(source["path"], device="cpu")
    semantic_start = state_digest(trainer.semantic_state())
    if semantic_start != source["semantic_hash"] or trainer.ticks != START:
        raise ValueError("starting complete state differs from canonical snapshot")
    config = trainer.config
    if (config.interface,config.mode,config.credit,config.structure,config.course,config.width,config.batch_size,config.dtype,config.cpu_threads) != (
            "ports","undirected","recorded512","fixed","warmup",80,8,"float64",1):
        raise ValueError("wrong scientific configuration")
    baseline = archived_baseline(trainer, source)
    write(folder / "baseline.json", baseline)
    counts = Counter()
    started = time.perf_counter()
    initial_work = deepcopy(trainer.work)
    updates = smoke_updates if smoke_updates else UPDATES
    last_complete = trainer.state_dict()
    try:
        for offset in range(updates):
            point, audit, row, checks = one_update(trainer, arm)
            last_complete = trainer.state_dict()
            counts.update(row)
            payload = dict(format=FORMAT, seed=seed, arm=arm, protocol_sha256=sha(path),
                update_offset=offset+1, record=deepcopy(trainer.credit.tape[-1]),
                live=deepcopy(trainer.last_chunk), actions=audit, counts=row,
                checks=checks, trace=deepcopy(point))
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            audit_file = folder / "audit" / f"tick_{trainer.ticks:06d}.pt.gz"
            audit_file.write_bytes(gzip.compress(buffer.getvalue(), compresslevel=1, mtime=0))
            progress = dict(format=FORMAT, seed=seed, arm=arm, completed_updates=offset+1,
                ticks=trainer.ticks, seconds=time.perf_counter()-started, counts=dict(counts),
                last_audit=str(audit_file), last_audit_sha256=sha(audit_file))
            write(folder / "progress.json", progress)
            if (offset+1) % 16 == 0 or offset == 0:
                print(json.dumps(progress), flush=True)
        elapsed_training = time.perf_counter() - started
        trainer._validate_continuation()
        trainer.save_checkpoint(folder / "checkpoint_final.pt")
        trainer.write_run(folder)
        endpoint = None if smoke_updates else endpoint_evaluation(trainer)
        if endpoint is not None:
            write(folder / "evaluation.json", endpoint)
        work_delta = {key: value-initial_work[key] for key, value in trainer.work.items()}
        if (work_delta["optimizer_steps"] != updates or work_delta["live_loss_positions"] != updates*128*8*2
                or work_delta["aux_loss_positions"] != updates*128*8*2
                or work_delta["aux_warmup_ticks"] != updates*128):
            raise AssertionError("matched original live/replay training budget failed")
        if counts["labeled_finish"] != sum(sum(p["target_action_counts"][3:]) for p in trainer.trace[-updates:]):
            raise AssertionError("FINISH-label totals differ from original trainer trace")
        terminal = dict(format=FORMAT, status="success", seed=seed, arm=arm,
            protocol_sha256=sha(path), source=source, semantic_start=semantic_start,
            semantic_end=state_digest(trainer.semantic_state()), target_updates=updates,
            completed_updates=updates, tick=trainer.ticks, counts=dict(counts), work_delta=work_delta,
            training_seconds=elapsed_training, total_seconds=time.perf_counter()-started,
            checkpoint=str((folder / "checkpoint_final.pt").resolve()),
            checkpoint_sha256=sha(folder / "checkpoint_final.pt"),
            original_sources_unchanged=source_pins() == read(OLD_PROTOCOL)["sources"])
        write(folder / "terminal.json", terminal)
        return terminal
    except BaseException as error:
        torch.save(last_complete, folder / "checkpoint_last_complete.pt")
        write(folder / "failure.json", dict(format=FORMAT, seed=seed, arm=arm,
            status="algorithm_numerical_failure" if isinstance(error,FloatingPointError) else "execution_failure",
            exception_type=type(error).__name__, message=str(error), ticks=trainer.ticks,
            completed_updates=trainer.updates-START//128, counts=dict(counts)))
        raise


def summary(path):
    protocol = check_protocol(path)
    out = Path(protocol["output"])
    rows = []
    training = []
    for seed in protocol["seeds"]:
        for arm in ARMS:
            folder = out / "training" / f"seed_{seed:03d}" / arm
            terminal = read(folder / "terminal.json")
            if terminal["completed_updates"] != UPDATES or terminal["tick"] != END:
                raise ValueError("incomplete formal continuation")
            training.append(terminal)
            for phase, name in (("before","baseline.json"),("after","evaluation.json")):
                evaluation = read(folder / name)
                for condition in ("short","long_both"):
                    records = [r for r in evaluation["records"] if r["condition"] == condition]
                    counts = Counter()
                    completions, strict, begins = [], [], []
                    for r in records:
                        p, d, o = r["primary"], r["request_diagnosis"]["primary"], r["opportunities"]
                        for key,value in dict(matured=p["matured"], completed=p["completed"], strict=p["clean_completed"],
                                began=d["began"], legal_requests=d["with_observed_legal_opportunity"],
                                finish_attempts=r["post_fork_counters"]["finish_attempted"],
                                legal_finish_attempts=o["legally_timed_finish_attempts"],
                                idle_opportunities=o["inactive_opportunities"], idle_actions=o["inactive_actions"],
                                opportunities=o["opportunities"]).items():
                            counts[key] += value
                        completions.append(p["completed"]/p["matured"])
                        strict.append(p["clean_completed"]/p["matured"])
                        begins.append(d["began"]/p["matured"])
                    rows.append(dict(seed=seed,arm=arm,phase=phase,condition=condition,counts=dict(counts),
                        completion=statistics.mean(completions), strict_completion=statistics.mean(strict),
                        began=statistics.mean(begins), idle_per1000=1000*counts["idle_actions"]/counts["idle_opportunities"]))
        pair = [t for t in training if t["seed"] == seed]
        if pair[0]["work_delta"] != pair[1]["work_delta"]:
            raise AssertionError("paired training compute budgets differ")
        for key in ("commands", "selected_commands", "eligible_commands", "selected_eligible_commands", "job_positions"):
            if pair[0]["counts"][key] != pair[1]["counts"][key]:
                raise AssertionError("paired external command/selection budgets differ")
    aggregates = []
    for condition in ("short","long_both"):
        for arm in ARMS:
            for phase in ("before","after"):
                group=[r for r in rows if (r["condition"],r["arm"],r["phase"])==(condition,arm,phase)]
                aggregates.append(dict(condition=condition,arm=arm,phase=phase,n_seeds=3,
                    **{metric:dict(mean=statistics.mean(r[metric] for r in group),sd=statistics.stdev(r[metric] for r in group))
                       for metric in ("completion","strict_completion","began","idle_per1000")}))
    paired=[]
    for condition in ("short","long_both"):
        deltas=[]
        for seed in range(3):
            selected=[r for r in rows if (r["seed"],r["condition"],r["phase"])==(seed,condition,"after")]
            assisted=next(r for r in selected if r["arm"]=="begin_quarter")
            control=next(r for r in selected if r["arm"]=="control")
            deltas.append(dict(seed=seed,**{metric:assisted[metric]-control[metric] for metric in ("completion","strict_completion","began","idle_per1000")}))
        paired.append(dict(condition=condition,per_seed=deltas,**{metric:dict(mean=statistics.mean(r[metric] for r in deltas),sd=statistics.stdev(r[metric] for r in deltas)) for metric in ("completion","strict_completion","began","idle_per1000")}))
    result=dict(format=FORMAT,protocol_sha256=sha(path),rows=rows,aggregates=aggregates,paired=paired,training=training,
        definitions=dict(eval_completion="equal-weight mean of 4 family completed/matured rates per seed",
            eval_strict="equal-weight mean of 4 family clean_completed/matured rates per seed",
            eval_begin="matured primary requests with successful BEGIN, same family weighting",
            eval_attempts="all post-fork job action opportunities, including inherited/carryover; legal counts ignore content correctness",
            training_labels="original causal midpoint labels at each live actor-job tick, replay labels separately budgeted",
            uncertainty="sample SD across 3 independently trained seeds; family/actor forks do not increase n"))
    write(out / "results.json", result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","summary"))
    parser.add_argument("--protocol",default=str(PROTOCOL))
    parser.add_argument("--seed",type=int,choices=(0,1,2),default=0)
    parser.add_argument("--arm",choices=ARMS,default="control")
    parser.add_argument("--smoke-updates",type=int)
    args=parser.parse_args()
    if args.command=="prepare":
        value=prepare(args.protocol)
        print(json.dumps(dict(protocol=args.protocol,sha256=sha(args.protocol),starts=value["starts"])),flush=True)
    elif args.command=="run":
        print(json.dumps(run(args.protocol,args.seed,args.arm,smoke_updates=args.smoke_updates)),flush=True)
    else:
        result=summary(args.protocol)
        print(json.dumps(dict(paired=result["paired"],aggregates=result["aggregates"])),flush=True)


if __name__=="__main__":
    main()
