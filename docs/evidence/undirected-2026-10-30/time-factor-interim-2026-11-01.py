"""Post-hoc seed-zero duration/idle measurement; never changes formal artifacts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import CONDITIONS, evaluate
from experiments.undirected import run as native, study
from experiments.undirected.queue import atomic_json, available_memory, file_sha256, verify_files

torch.set_num_threads(1)
EVIDENCE = Path(__file__).resolve().parent
PROTOCOL = EVIDENCE / "time-factor-interim-2026-11-01.protocol.json"
SUMMARY = EVIDENCE / "time-factor-interim-2026-11-01.json"
MARKDOWN = EVIDENCE / "time-factor-interim-2026-11-01.md"
OUTPUT = ROOT / "data/undirected/interim-time-factor-v1"
TICK, STEPS, RESERVE = 196608, 2048, 2 * 1024 ** 3
FACTORS = ("long_wait", "long_idle")
FORMAL = ("short", "long_both")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def sole_writer():
    import msvcrt
    with (OUTPUT / "runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare():
    if PROTOCOL.exists() or OUTPUT.exists():
        raise FileExistsError("post-hoc preparation requires new protocol and output paths")
    original = EVIDENCE / "protocol-v1.json"
    plan = read(original)
    branches = [item for item in plan["branches"] if item["seed"] == 0 and item["credit"] == "recorded512"
                and item["structure"] == "fixed" and item["course"] == "warmup"]
    assert len(branches) == 4
    models = []
    for item in branches:
        study.read_plan(original, item["id"])
        index = Path(item["output"]) / "snapshots.json"
        entry = read(index)["entries"][str(TICK)]
        assert file_sha256(entry["path"]) == entry["sha256"]
        models.append(dict(branch=item, checkpoint=entry, snapshot_index=str(index), snapshot_index_sha256=file_sha256(index)))
    sources = dict(plan["sources"])
    for path in (Path(__file__), ROOT / "experiments/lifelong/evaluate.py", ROOT / "experiments/continuity/evaluate.py"):
        sources[str(path.resolve())] = file_sha256(path)
    grid = [dict(branch_id=model["branch"]["id"], family=family, condition=condition, steps=STEPS, tick=TICK)
            for model in models for condition in FACTORS for family in range(4)]
    value = dict(format="undirected-interim-time-factor-protocol-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        classification="Post-hoc supplemental measurement, chosen after seeing seed0 core long_both failure; outside the original 42 branches/1680 forks.",
        rationale="Separate future request processing duration from intertask idle after the two core seed0 models made zero legally timed FINISH attempts in formal long_both. No endpoint ranking or pooled-seed inference is authorized.",
        original_protocol=str(original), original_protocol_sha256=file_sha256(original), sources=sources, models=models,
        seed=0, independent_trained_seeds=1, checkpoint_tick=TICK, steps_per_fork=STEPS, planned_additional_forks=32, grid=grid,
        conditions={name: CONDITIONS[name] for name in (*FORMAL, *FACTORS)}, dt=plan["dt"], output=str(OUTPUT),
        optional_formal_comparators=dict(conditions=list(FORMAL), tick=TICK, rule="Reuse matching existing frozen formal outputs only; bind their bytes at extraction and never rerun or modify them."),
        resources=dict(device="cpu", dtype="float64", cpu_threads=1, own_processes=1, loaded_checkpoints_at_once=1, min_available_bytes_before_each_model=RESERVE),
        frozen_contract="Existing evaluator/teacher, model eleven-way argmax, copied original world/RNG/contracts and original h; only future NEW duration/gap and family change. No training or formal exposure/parameter changes.",
        integrity="Validate original plan and canonical CP bytes/full restore/semantic digest before each model, before and after every fork; verify source bytes and CP bytes afterward. Preserve all raw fork diagnostics.",
        reporting="Report every planned fork including execution/resource deferral. Four-family rates and raw denominators; actor/family/fork are not independent trained seeds. Primary matured requests and all-opportunity idle diagnostics are separate denominators.")
    atomic_json(PROTOCOL, value)
    OUTPUT.mkdir(parents=True)
    print(json.dumps(dict(protocol=str(PROTOCOL), protocol_sha256=file_sha256(PROTOCOL), planned_forks=len(grid))), flush=True)


def load_protocol():
    value = read(PROTOCOL)
    verify_files(value["sources"])
    assert file_sha256(value["original_protocol"]) == value["original_protocol_sha256"]
    assert value["planned_additional_forks"] == len(value["grid"]) == 32
    return value


def fork_path(spec):
    return OUTPUT / "forks" / f"{spec['branch_id']}__{spec['condition']}__family_{spec['family']}.json"


def checked_record(value, spec, model, protocol_sha):
    assert value["protocol_sha256"] == protocol_sha and value["spec"] == spec and value["checkpoint"] == model["checkpoint"]
    assert value["result_hash"] == state_digest({key: item for key, item in value.items() if key != "result_hash"})
    r = value["record"]
    assert value["source_world_tick"] == TICK-1  # Native world indexes completed actions from zero.
    assert (r["family"], r["condition"], r["steps"], r["fork_tick"]) == (spec["family"], spec["condition"], STEPS, value["source_world_tick"])
    assert all(r[key] == 0 for key in ("world_resets", "hidden_resets", "parameter_updates"))
    assert value["source_unchanged"] is True
    return r


def training_coverage(trainer, model):
    ranges = ((0,65536),(65536,131072),(131072,196608))
    points = [point for point in trainer.trace if point["tick"] <= TICK]
    assert len(points) == TICK//128 and [point["tick"] for point in points] == list(range(128,TICK+1,128))
    rows = []
    for start,end in ranges:
        selected = [point for point in points if start < point["tick"] <= end]
        counts = {key:[0]*11 for key in ("actual_action_counts","target_action_counts")}
        for point in selected:
            for key in counts:
                values = point[key]
                assert len(values)==11 and all(type(v) is int and v>=0 for v in values) and sum(values)==2048
                counts[key] = [left+right for left,right in zip(counts[key],values)]
        assert len(selected)==(end-start)//128
        assert all(sum(values)==len(selected)*2048 for values in counts.values())
        rows.append(dict(start_tick=start,end_tick=end,blocks=len(selected),positions=len(selected)*2048,
                         **{key:dict(classes=values,WAIT=values[0],BEGIN_1=values[1],BEGIN_2=values[2],
                                     FINISH_by_class=values[3:],FINISH_total=sum(values[3:])) for key,values in counts.items()}))
    result=dict(format="undirected-interim-live-training-coverage-v1",branch_id=model["branch"]["id"],checkpoint=model["checkpoint"],
                source="Already loaded canonical main-end trainer.trace; live training only, excludes auxiliary replay.",
                actual_BEGIN_definition="Model action attempts, not accepted starts; target counts follow each model's own realized training world.",
                positions_per_block=2048,ranges=rows,causal_claim=False)
    atomic_json(OUTPUT/"training-coverage"/f"{model['branch']['id']}.json",result)
    return result


def run():
    value, sha = load_protocol(), file_sha256(PROTOCOL)
    with sole_writer():
        state = dict(format="undirected-interim-time-factor-execution-v1", protocol_sha256=sha, complete=False, status="running", models={})
        atomic_json(OUTPUT / "execution.json", state)
        for model in value["models"]:
            item = model["branch"]
            specs = [row for row in value["grid"] if row["branch_id"] == item["id"]]
            if all(fork_path(spec).exists() for spec in specs):
                for spec in specs:
                    checked_record(read(fork_path(spec)), spec, model, sha)
                state["models"][item["id"]] = dict(status="complete_existing", forks=8)
                continue
            available = available_memory()
            if available <= RESERVE:
                state.update(status="deferred_resources", available_bytes=available, deferred_model=item["id"])
                atomic_json(OUTPUT / "execution.json", state)
                print(json.dumps(dict(status="deferred_resources", model=item["id"], available_bytes=available)), flush=True)
                summarize()
                return 2
            plan, bound, _ = study.read_plan(value["original_protocol"], item["id"])
            assert bound == item
            cp = model["checkpoint"]
            assert file_sha256(cp["path"]) == cp["sha256"]
            with _preserve_process_state(), torch.no_grad():
                trainer = native.NativeTrainer.from_checkpoint(cp["path"], device="cpu")
                study.validate(trainer, plan, item, TICK)
                before = state_digest(trainer.semantic_state())
                assert before == cp["semantic_hash"]
                training_coverage(trainer,model)
                for spec in specs:
                    path = fork_path(spec)
                    if path.exists():
                        checked_record(read(path), spec, model, sha)
                        continue
                    assert state_digest(trainer.semantic_state()) == before
                    r = evaluate(trainer.model, trainer.world, trainer.h, family=spec["family"], condition=spec["condition"],
                                 steps=STEPS, dt=value["dt"], device="cpu")
                    assert state_digest(trainer.semantic_state()) == before and file_sha256(cp["path"]) == cp["sha256"]
                    assert r["parameter_hash"] == state_digest(trainer.model.state_dict())
                    assert r["start_hidden_hash"] == state_digest(dict(hidden=trainer.h))
                    assert r["start_world_state_hash"] == state_digest(trainer.world.state_dict())
                    result = dict(format="undirected-interim-time-factor-fork-v1", protocol_sha256=sha, spec=spec,
                                  checkpoint=cp, source_world_tick=trainer.world.tick, source_semantic_hash=before, source_unchanged=True, record=r)
                    result["result_hash"] = state_digest(result)
                    checked_record(result, spec, model, sha)
                    atomic_json(path, result)
                    print(json.dumps(dict(branch=item["id"], family=spec["family"], condition=spec["condition"],
                                          strict=r["primary"]["clean_completed"], matured=r["primary"]["matured"])), flush=True)
                assert state_digest(trainer.semantic_state()) == before
                del trainer
            gc.collect()
            verify_files(value["sources"])
            assert file_sha256(cp["path"]) == cp["sha256"]
            state["models"][item["id"]] = dict(status="complete", forks=8, before_after_semantic_hash=before, available_bytes_before_load=available)
            atomic_json(OUTPUT / "execution.json", state)
            print(json.dumps(dict(model_complete=item["id"], forks=8)), flush=True)
        state.update(complete=True, status="complete", additional_forks=32)
        atomic_json(OUTPUT / "execution.json", state)
        summarize()
    return 0


def row(item, condition, family, record, path):
    d = record["request_diagnosis"].get("primary", {})
    return dict(branch_id=item["id"], interface=item["interface"], mode=item["mode"], condition=condition, family=family,
        primary=record["primary"], matured_request_diagnosis=d, opportunities=record["opportunities"],
        raw_path=str(path.resolve()), raw_sha256=file_sha256(path), observed_terminal_events=record["observed_terminal_events"],
        inactive_history=record["inactive_history"], actual_family_exposures=record["actual_family_exposures"])


def fraction(a, b):
    return f"{a}/{b} ({100*a/b:.1f}%)" if b else f"{a}/0 (未定义)"


def summarize():
    value, sha = load_protocol(), file_sha256(PROTOCOL)
    rows, bindings, pending = [], {}, []
    for model in value["models"]:
        item = model["branch"]
        for spec in (s for s in value["grid"] if s["branch_id"] == item["id"]):
            path = fork_path(spec)
            if not path.exists():
                pending.append(spec)
                continue
            r = checked_record(read(path), spec, model, sha)
            rows.append(row(item, spec["condition"], spec["family"], r, path))
        path = Path(value["original_protocol"]).parent
        formal = Path(read(value["original_protocol"])["raw_root"]) / "evaluation" / f"{item['id']}.json"
        if not formal.exists():
            continue
        result = read(formal)
        plan, bound, original_sha = study.read_plan(value["original_protocol"], item["id"])
        terminal = read(Path(item["output"]) / "terminal.json")
        study._checked_evaluation(result, plan, bound, original_sha, terminal)
        assert result["states"][str(TICK)]["checkpoint_sha256"] == model["checkpoint"]["sha256"]
        bindings[str(formal)] = file_sha256(formal)
        for r in result["records"]:
            spec = r["probe"]
            if spec["tick"] == TICK and spec["condition"] in FORMAL:
                rows.append(row(item, spec["condition"], spec["family"], r, formal))
    output = dict(format="undirected-interim-time-factor-summary-v1", protocol=str(PROTOCOL), protocol_sha256=sha,
                  independent_trained_seeds=1, additional_forks_completed=32-len(pending), planned_additional_forks=32,
                  pending_additional_forks=pending, formal_comparator_bindings=bindings, rows=rows,
                  complete=len(pending)==0, scope="Post-hoc seed0-only frozen main-end snapshot supplement, outside original formal counts; no pooled-seed or causal mediation claim.")
    output["live_training_coverage"]={}
    for model in value["models"]:
        path=OUTPUT/"training-coverage"/f"{model['branch']['id']}.json"
        if path.exists():
            output["live_training_coverage"][model["branch"]["id"]]=dict(path=str(path),sha256=file_sha256(path),**read(path))
    atomic_json(OUTPUT / "summary.json", output)
    atomic_json(SUMMARY, output)
    lines = ["# 主训练末时间因素补充测量（事后、seed 0）", "", "只使用四个 recorded512 / fixed / warmup 分支的 tick 196608 完整冻结快照。新增 long_wait 和 long_idle 共 32 个闭环分叉，每个 2,048 tick；short 和 long_both 只引用已经完成的正式分叉。独立训练种子 n=1，四个 family、八个 actor 和分叉不增加独立样本数。", "", "在看到两个 core 模型于 long_both 没有合法时机 FINISH 后才提出本检查。它分开未来 NEW 的处理时长与任务间隔；继承合同、原世界/RNG 和 h 保留。无训练、权重更新或正式暴露计数改动。", "", "| 条件 | duration | gap |", "|---|---|---|", "| short | 1–4 | 0.5–2 |", "| long_wait | 8–16 | 0.5–2 |", "| long_idle | 1–4 | 16–32 |", "| long_both | 8–16 | 16–32 |", "", f"新增分叉：{32-len(pending)}/32。未出现的正式比较项仍待其原队列完成。", "", "严格完成：分母为当前条件下已成熟的 primary 请求。四家族平均为本种子四个比例的等权描述；括号计数为总量，不作跨种子推断。", "", "| 模型 | 条件 | F0 严格完成 | F1 | F2 | F3 | 四家族均值 | 总严格完成/成熟 |", "|---|---|---|---|---|---|---:|---|"]
    grouped = []
    for model in value["models"]:
        item = model["branch"]
        for condition in ("short", "long_wait", "long_idle", "long_both"):
            group = sorted([r for r in rows if r["branch_id"]==item["id"] and r["condition"]==condition], key=lambda r:r["family"])
            if len(group)!=4:
                continue
            grouped.append((item, condition, group))
            ratios=[r["primary"]["clean_completed"]/r["primary"]["matured"] if r["primary"]["matured"] else None for r in group]
            average=f"{100*sum(ratios)/4:.1f}%" if all(r is not None for r in ratios) else "未定义"
            family=[fraction(r["primary"]["clean_completed"],r["primary"]["matured"]) for r in group]
            lines.append(f"| {item['interface']}/{item['mode']} | {condition} | " + " | ".join(family) + f" | {average} | {sum(r['primary']['clean_completed'] for r in group)}/{sum(r['primary']['matured'] for r in group)} |")
    lines += ["", "BEGIN、内容、时机诊断属于已成熟 primary 请求；合法 FINISH 为尝试次数，不能把它当作请求成功率。闲置机会覆盖整个观测分叉（含继承请求），与 primary 成熟分母不同。", "", "| 模型 | 条件 | 已 BEGIN/成熟请求 | 有合法机会/成熟请求 | 首个合法机会内容正确 | 合法 FINISH/活跃 FINISH 尝试 | 未BEGIN失败 | BEGIN但无合法机会失败 | 有机会仍失败 | 实际闲置机会 | 闲置动作/闲置机会 |", "|---|---|---|---|---|---|---:|---:|---:|---:|---|"]
    for item, condition, group in grouped:
        total=lambda key:sum(r["matured_request_diagnosis"].get(key,0) for r in group)
        mature=sum(r["primary"]["matured"] for r in group)
        idle=sum(r["opportunities"]["inactive_opportunities"] for r in group)
        actions=sum(r["opportunities"]["inactive_actions"] for r in group)
        lines.append(f"| {item['interface']}/{item['mode']} | {condition} | {fraction(total('began'),mature)} | {fraction(total('with_observed_legal_opportunity'),mature)} | {fraction(total('first_observed_legal_content_correct'),total('first_observed_legal_count'))} | {total('legally_timed_finish_attempts')}/{total('active_finish_attempts')} | {total('uncompleted_no_begin')} | {total('uncompleted_began_but_no_legal_opportunity')} | {total('uncompleted_despite_legal_opportunity')} | {idle} | {fraction(actions,idle)} |")
    lines += ["", "所有逐家族原始诊断、取消/到期事件、真实闲置机会与暴露计数都保存在 JSON 与分叉原件中。条件会改变闭环轨迹与请求数；观察到的差异不能单独识别内在记忆机制，也不能推广到三个训练种子。", "", f"协议：[protocol](<{PROTOCOL.name}>)；逐家族数据：[JSON](<{SUMMARY.name}>)。协议 SHA-256：`{sha}`。"]
    lines += ["", "## 训练中的实际动作和目标覆盖", "", "来源为已经加载的 tick 196608 完整快照的 trainer.trace。仅统计 live 训练，不包含辅助历史回放；实际 BEGIN 是尝试数而非成功数。每块两套计数均核对为 128×8×2=2,048 个位置。完整八类 FINISH 分项保存在 JSON。这里只描述各自闭环训练标签覆盖，不推断因果。", "", "| 模型 | live tick范围 | 位置总数 | 实际 WAIT | 实际 BEGIN1 / BEGIN2 | 实际 FINISH | 目标 WAIT | 目标 BEGIN1 / BEGIN2 | 目标 FINISH |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in value["models"]:
        item=model["branch"]
        coverage=output["live_training_coverage"].get(item["id"])
        if coverage:
            for r in coverage["ranges"]:
                a,t=r["actual_action_counts"],r["target_action_counts"]
                lines.append(f"| {item['interface']}/{item['mode']} | {r['start_tick']}–{r['end_tick']} | {r['positions']} | {a['WAIT']} | {a['BEGIN_1']} / {a['BEGIN_2']} | {a['FINISH_total']} | {t['WAIT']} | {t['BEGIN_1']} / {t['BEGIN_2']} | {t['FINISH_total']} |")
    MARKDOWN.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(summary=str(SUMMARY), markdown=str(MARKDOWN), additional_forks=32-len(pending), formal_files=len(bindings))),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","summarize"))
    action=parser.parse_args().action
    if action=="prepare": prepare()
    elif action=="run": raise SystemExit(run())
    else: summarize()
