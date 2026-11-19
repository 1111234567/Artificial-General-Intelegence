"""Readable raw curves for the crossed target, retaining original group names."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.experience_transfer import reciprocal_train as r, reciprocal_eval as q


def export():
    rows, records, audits, gradient_audits = [], [], [], []
    for seed in range(3):
        bundle = q.e.read(q.OUT / f"seed_{seed:03d}.json")
        assert len(bundle["records"]) == 10
        records.extend(bundle["records"])
        audit = q.e.read(r.OUT / f"audit_seed_{seed:03d}.json")
        assert audit["passed"]
        audits.append(audit)
        gradient = q.e.read(r.OUT / f"gradient_audit_seed_{seed:03d}.json")
        assert gradient["passed"]
        gradient_audits.append(gradient)
        for record in bundle["records"]:
            cp = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)
            bf = record["B_family"]
            actual_B = int(cp["world"]["context"]["tensors"]["family_new_counts"][..., bf].sum())
            original_B = r.t.OUT / f"seed_{seed:03d}" / record["arm"] / f"{record['snapshot']}.pt"
            old_cp = torch.load(original_B, map_location="cpu", weights_only=False)
            assert actual_B == int(old_cp["world"]["context"]["tensors"]["family_new_counts"][..., bf].sum())
            for group in ("all", "shared", "conflict"):
                scores = record["strata"][str(bf)][group]
                row = dict(seed=seed, history=record["history_rule"], arm=record["arm"], target="Bprime",
                    snapshot=record["snapshot"], B_round=record["B_round"], group=group,
                    raws=json.dumps(scores["raws"]), actual_B_NEW_total=actual_B,
                    B_supervised_NEW_work=cp["work"]["B_unique_new_commands"],
                    B_boundary_NEW_not_in_work=actual_B-cp["work"]["B_unique_new_commands"],
                    B_unique_label_positions=cp["work"]["B_unique_label_positions"],
                    B_gradient_label_positions=cp["work"]["B_gradient_label_positions"],
                    gradient_updates=cp["work"]["gradient_updates"],
                    overall_idle_actions=record["counts"]["idle_actions"],
                    overall_idle_opportunities=record["counts"]["idle_opportunities"],
                    overall_idle_actions_per1000=record["metrics"]["idle_actions_per1000"],
                    checkpoint_path=record["checkpoint_path"], checkpoint_sha256=record["checkpoint_sha256"],
                    result_path=str(q.OUT / f"seed_{seed:03d}" / f"{record['arm']}_{record['snapshot']}_Bprime.json"),
                    request_path=str(q.OUT / f"seed_{seed:03d}" / f"{record['arm']}_{record['snapshot']}_Bprime.requests.json"),
                    facts_path=str(q.OUT / f"seed_{seed:03d}" / f"{record['arm']}_{record['snapshot']}_Bprime.npz"),
                    primary_identity_hash=record["primary_identity_hash"], external_digest=record["external_digest"])
                for key in ("matured", "completed", "clean_completed", "idle_actions", "idle_opportunities", "primary_issued", "begin_attempts", "valid_begin", "feasible_begin", "finish_attempts", "legal_finish"):
                    row[key] = scores["counts"].get(key, 0)
                for key in ("ordinary_completion_rate", "strict_completion_rate", "idle_actions_per1000"):
                    row[key] = scores["metrics"][key]
                rows.append(row)
    for seed in range(3):
        for snapshot in r.SNAPSHOTS:
            pair = [v for v in records if v["seed"] == seed and v["snapshot"] == snapshot]
            assert len(pair) == 2
            for key in ("external_digest", "primary_identity_hash", "start_scheduler_hash", "end_scheduler_hash", "warmup_observation_hash"):
                assert pair[0][key] == pair[1][key]
    q.e.write(r.OUT / "curves.json", dict(rows=rows, records=records,
        protocol_hash=r.checked()["protocol_hash"], evaluation_protocol_hash=q.checked()["protocol_hash"],
        semantics="A/C filenames preserve original arm identity; C is related to Bprime. shared/conflict always refer to ORIGINAL A/B groups. actual_B_NEW_total includes the common boundary WAIT tick; supervised_NEW_work excludes it.",
        audit_coverage={key: sum(v["coverage"][key] for v in audits) for key in audits[0]["coverage"]},
        exact_gradient_audit=dict(seeds=3, branches=6, online_updates_per_branch=8, checkpoints=[2, 8],
            all_model_Adam_hidden_exact=True, artifacts=[str(r.OUT / f"gradient_audit_seed_{seed:03d}.json") for seed in range(3)]),
        export_source_sha256=q.e.sha(__file__)))
    with (r.OUT / "curves.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report(rows, audits)
    print(json.dumps(dict(rows=len(rows), branches=len(records), path=str(r.OUT / "curves.json"))), flush=True)


def report(rows, audits):
    mean = statistics.mean
    lines = ["# 交换相关前史的 B′ 目标对照", "",
        "复用原A/C的完整前史端点，只改变从未发行的B提示对应规则。B′在原A/B冲突raw保留B正确答案，A与C在该组的历史错误答案也相同；在原共享raw改用C答案。因此C与B′有两个关联重合，A与B′零重合。两臂各继续相同8轮B示范/更新，无direct，无重复前史。", "",
        "文件名related仍指原A前史，disjoint仍指原C前史；在本目标下C才是相关前史。下表三个种子等权平均。共享/冲突标签始终指原A/B分组；不能随着目标重命名或重新划分。", "",
        "| B轮 | 组 | A前史严格完成 | C前史严格完成 | C−A/百分点 | A普通完成 | C普通完成 | A B槽idle/千机会 | C B槽idle/千机会 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in (0, 1, 2, 4, 8):
        for group in ("all", "shared", "conflict"):
            a = [v for v in rows if v["B_round"] == stage and v["group"] == group and v["history"] == "A"]
            c = [v for v in rows if v["B_round"] == stage and v["group"] == group and v["history"] == "C"]
            av = mean(v["strict_completion_rate"] for v in a); cv = mean(v["strict_completion_rate"] for v in c)
            lines.append(f"| {stage} | {group} | {100*av:.2f}% | {100*cv:.2f}% | {100*(cv-av):+.2f} | {100*mean(v['ordinary_completion_rate'] for v in a):.2f}% | {100*mean(v['ordinary_completion_rate'] for v in c):.2f}% | {mean(v['idle_actions_per1000'] for v in a):.2f} | {mean(v['idle_actions_per1000'] for v in c):.2f} |")
    lines.extend(["", "## 每个种子的完整严格完成曲线", "", "| seed | 前史 | 组 | B0 | B1 | B2 | B4 | B8 |", "|---:|---|---|---:|---:|---:|---:|---:|"])
    for seed in range(3):
        for history in ("A", "C"):
            for group in ("all", "shared", "conflict"):
                curve = [next(v for v in rows if (v["seed"], v["history"], v["group"], v["B_round"]) == (seed, history, group, stage)) for stage in (0, 1, 2, 4, 8)]
                values = [f"{v['clean_completed']}/{v['matured']}" for v in curve]
                lines.append(f"| {seed} | {history} | {group} | " + " | ".join(values) + " |")
    lines.extend(["", "## 原始证据与核验", "",
        "30个heldout分支均在相同的1024tick WAIT公开背景后自由argmax行动2048tick，按新发起且成熟请求计分。B′和原B的公共暖启、未来外部事件、成熟请求身份与scheduler哈希完全匹配；私有目标和实际政策回执按真实规则演化。原始观察/logits/动作/世界字段/回执位于evaluation下各.npz，完整初末world/h位于.pt，逐请求位于.requests.json。", "",
        "分组idle按空闲槽当前保留的B cue和raw计数，raw是结束后保留的内容，不是成熟请求分母；上表给出该B槽分层率。完整实际世界的整体idle分子分母另存CSV/JSON的overall_idle_actions与overall_idle_opportunities，也原样保存在各分叉结果counts中。", "",
        "[curves.csv](curves.csv) 与 [curves.json](curves.json) 给出每seed/前史/阶段/组的分子分母、实际累计B_NEW、教师配对NEW、标签/更新数及原始文件路径。actual_B_NEW_total从checkpoint完整world中计数，包含统一边界WAIT tick上发出的新请求；B_supervised_NEW_work是原训练work的计数，不包含该边界。两者分开保留。", "",
        "独立审计核对六个B0分支的完整恢复、24个双臂配对训练轮、12条首尾真实训练轨迹重放、30个heldout分支的逐位GRU推理与独立成熟/idle/语义回执计数；原生lifelong evaluator重放全部B0/B8共12个分支。另用训练审计的独立梯度实现，从六个B0分支各重做前八次在线Adam更新，在第2/8次更新点逐位复现model、Adam与实际公开历史h。所有分支的实际B_NEW与原B阶段相同。审计没有独立重做全部2048次B梯度更新；后续预算与示范数据均保存核对。", "",
        "本对照只交换目标关联结构，三个种子不能保证一般的迁移性质。原共享组的目标变化会改变它对共同冲突组的训练干扰；共同冲突组本身的正确答案和历史错误相同，但不能把其跨目标分差视作只改了相关性单一标量。B0初始差异及后续增益应分开读；跨目标的早期曲线分析由总报告按seed配对。", "",
        "[冻结训练方案](protocol.json) · [冻结heldout方案](evaluation/protocol.json) · [审计0](audit_seed_000.json) · [审计1](audit_seed_001.json) · [审计2](audit_seed_002.json) · [梯度审计0](gradient_audit_seed_000.json) · [梯度审计1](gradient_audit_seed_001.json) · [梯度审计2](gradient_audit_seed_002.json)", ""])
    (r.OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    export()
