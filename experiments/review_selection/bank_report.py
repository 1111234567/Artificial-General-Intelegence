"""Describe bank eligibility and actual own-action outcomes without changing it."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.experience_transfer import train_transfer as source
from experiments.review_selection import bank


def diagnose(seed):
    folder = bank.OUT / f"seed_{seed:03d}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data = bank.load(seed)
    selected_keys = {(m["source_round"],m["lane"],m["slot"],m["request_id"]) for m in data["metadata"]}
    all_new, selected = Counter(), Counter()
    for index in bank.ROUNDS:
        original = torch.load(source.OUT/f"seed_{seed:03d}"/"related"/f"trajectory_history_{index:03d}.pt",map_location="cpu",weights_only=False)
        live = {}
        rows = []
        with source.mapping_scope(source.mapping_table(seed,"related")):
            world = source.worlds.ContextWorld.from_state_dict(original["initial_world"],device="cpu")
            for tick in range(1024):
                obs = world.observe()
                labels = world.oracle("midpoint")
                assert torch.equal(obs,original["obs"][tick])
                assert torch.equal(labels,original["labels"][tick])
                for lane,slot in (world.event_kind==source.worlds.NEW).nonzero().tolist():
                    row=dict(teacher_finish=False,own_complete=False,own_complete_before_teacher_finish=False,
                             selected=(index,lane,slot,int(world.request_id[lane,slot])) in selected_keys,
                             request_id=int(world.request_id[lane,slot]),raw=int(world.raw_payload[lane,slot]))
                    live[lane,slot]=row
                    rows.append(row)
                for key,row in tuple(live.items()):
                    lane,slot=key
                    if int(world.request_id[lane,slot])!=row["request_id"]:
                        del live[key]
                    elif labels[lane,slot]>=3:
                        row["teacher_finish"]=True
                world.act(original["actions"][tick])
                for key,row in live.items():
                    lane,slot=key
                    if original["actions"][tick,lane,slot]>=3 and world.last_result[lane,slot]==1:
                        row["own_complete"]=True
                        row["own_complete_before_teacher_finish"] |= not row["teacher_finish"]
            for row in rows:
                for target in (all_new,selected) if row["selected"] else (all_new,):
                    target["NEW_requests"]+=1
                    target["ever_midpoint_teacher_FINISH"]+=int(row["teacher_finish"])
                    target["own_completed_in_source_window"]+=int(row["own_complete"])
                    target["own_completed_before_first_midpoint_teacher_FINISH"]+=int(row["own_complete_before_teacher_finish"])
    assert all_new["NEW_requests"]==manifest["total_source_counts"]["NEW_requests"]
    assert selected["NEW_requests"]==256 and selected["ever_midpoint_teacher_FINISH"]==256
    assert selected["own_completed_before_first_midpoint_teacher_FINISH"]==0
    return dict(seed=seed,all_source_NEW=dict(all_new),retained_NEW=dict(selected),
                fixed_window_eligible=manifest["total_source_counts"]["fixed_window_eligible"],
                bank_sha256=manifest["bank_sha256"],audit=json.loads((folder/"audit.json").read_text(encoding="utf-8")))


def main():
    rows=[diagnose(seed) for seed in range(3)]
    report=dict(format="review-selection-bank-audit-summary-v1",seeds=rows,
                samples=768,fixed_supervised_rows=2304,source_world_ticks=12288,
                native_oracle_label_rows=1572864,unittests="bank_test.py: 4 passed",
                anchor_independent_stepwise_exact=True,
                selection_bias="Eligibility requires a real processing WAIT and a legal archived midpoint-oracle FINISH. It excludes requests completed by the old policy before that teacher opportunity, as well as canceled/replaced/unstarted cases and incomplete windows. Thus it favors already-begun requests that remain active until midpoint. Candidate priority itself never reads completion, model loss, or B/review results.",
                limits="Native-world replay validates all archived obs, labels, actions, and identities. Historical own argmax is not independently rederived because intermediate policy weights were not archived. Fixed numeric replay anchors belong to a frozen A endpoint public reconstruction, not the changing B policy.")
    bank.write(bank.OUT/"summary.json",report)
    lines=["# 真实旧 A 请求复习库", "", "三个种子各保留256条请求，原始 raw 各64条。每条是末四轮完全自主行动轨迹中的真实 NEW 请求，固定128 tick公开观察（cue前32、cue后96），只在同一请求、同一槽的教师 BEGIN、处理中 WAIT、首次合法 midpoint FINISH 监督，共3行。另一槽的原生公开观察保留，其训练标签完全屏蔽。", "", "| seed | 源 NEW | 有合法教师 FINISH | 完整128窗候选 | 留存 | 源中自主提前完成 | 留存中自主完成 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        total,retained=r["all_source_NEW"],r["retained_NEW"]
        lines.append(f"| {r['seed']} | {total['NEW_requests']} | {total['ever_midpoint_teacher_FINISH']} | {r['fixed_window_eligible']} | 256 | {total['own_completed_before_first_midpoint_teacher_FINISH']} | {retained['own_completed_in_source_window']} |")
    lines += ["", "‘自主提前完成’指在源窗口内、第一次 midpoint 教师 FINISH 前已由实际自主动作合法完成。这些请求没有后续的真实 midpoint FINISH 标签，因此不会进入此库。取消、替换、未启动或没有完整时间窗的请求也可能被排除。银行有明确的行为条件，不能代表全部 A 请求。资格判断不读取 B/复习实验结果；资格内按固定 SHA256 优先级、每轮每 raw 16条取样，不按完成率或 NLL 选取。", "", "h 从每条历史的真实数值 anchor 出发，使用冻结 A B000 模型沿该条原生公开流重建至段首。独立逐 tick GRU 重建与库中全部768个 h 逐位相等。它是固定 A 表示，不会随 B 训练更新。", "", "独立审计重放全部12,288个世界 tick、1,572,864个原生 oracle 槽标签，逐位验证公开观察、真实动作回执、标签、请求身份和三个监督点的状态合法性；A/C在两个冲突 raw 上答案相同、A/B不同。四项单测通过。未用未存档的历史中间权重重算每次旧自主 argmax。", "", "文件接口：`obs[128,256,56] float32`、`labels/mask[128,256,2]`、`anchor[256,192]`，以及raw、conflict、target_slot、positions、IDs与metadata。`request_mask`额外保存完整目标请求标签范围，但主实验使用固定三点mask。", "", "| seed | bank.pt SHA256 |", "|---|---|"]
    lines.extend(f"| {r['seed']} | `{r['bank_sha256']}` |" for r in rows)
    lines += ["", "生成：`.venv/Scripts/python.exe experiments/review_selection/bank.py`（目标目录须不存在）；核验：`bank_audit.py`、`bank_test.py`；本报告：`bank_report.py`。历史源未修改。", ""]
    (bank.OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"diagnostics":[{k:v for k,v in r.items() if k!="audit"} for r in rows],"summary":str(bank.OUT/"summary.json")}),flush=True)


if __name__=="__main__":
    main()
