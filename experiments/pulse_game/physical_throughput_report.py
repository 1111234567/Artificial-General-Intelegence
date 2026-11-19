"""Summarize the raw paired physical throughput and trajectory evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


STAGES = {
    "resident_device_event": "CUDA event，设备数据常驻且输出预分配",
    "resident_scoring_equal_batch": "常驻世界评分，同为 64 候选/分块",
    "resident_scoring_native_batch": "常驻世界评分，沿用各自分块设置",
    "prefetch_pipeline": "含世界生成与预取的流水线，沿用各自分块设置",
}
CPU_LABELS = {"numpy_serial": "NumPy，一进程", "numpy_four_processes": "NumPy，四进程",
              "gpu_reference": "旧 CUDA 详细评分", "gpu_fused": "融合 + warp CUDA 评分"}


def read(root, name):
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def factor(summary):
    return (f"{summary['speedup_a_over_b']:.3f}× "
            f"[{summary['speedup_95_ci'][0]:.3f}, {summary['speedup_95_ci'][1]:.3f}]")


def parity(diagnostics):
    result = {}
    for label, fields in diagnostics.items():
        result[label] = {field: {"different": sum(d[field]["different"] for d in fields),
                                 "elements": sum(d[field]["elements"] for d in fields),
                                 "max_abs": max(d[field]["max_abs"] for d in fields)} for field in fields[0]}
    return result


def disagreement(row):
    return f"{row['different']:,}/{row['elements']:,}"


def hardware_context(stages):
    detail = {}
    for name, row in stages.items():
        entries = [row[f"context_{phase}"]["nvidia_smi"]["device"][0].split(",") for phase in ("before", "after")]
        ix = row["context_before"]["device_fields"]
        detail[name] = {"cpu_busy": row["whole_system_cpu_busy_fraction"],
                        "gpu_util_brackets": [int(x[ix.index("utilization.gpu")]) for x in entries],
                        "sm_clock_brackets_mhz": [int(x[ix.index("clocks.sm")]) for x in entries],
                        "temperature_brackets_c": [float(x[ix.index("temperature.gpu")]) for x in entries],
                        "time_brackets": [row[f"context_{phase}"]["local_time"] for phase in ("before", "after")]}
    return detail


def pipeline_breakdown(blocks):
    measures = {}
    for label in "AB":
        runs = [entry for block in blocks for arm, entry in zip(block["sequence"], block["positions"], strict=True)
                if arm == label]
        epochs = [stage for group in runs for stage in group["generations"]]
        timing = {f: float(np.median([row[f] for row in epochs])) for f in
                  ("host_world_seconds", "host_world_wait_seconds", "cohort_and_decode_seconds", "scores_and_rank_seconds")}
        measures[label] = {"median_of_generations": timing, "enclosing_wall_seconds": float(np.median([e["seconds"] for e in runs]))}
    return measures


def summarize(root):
    gpu, traced, cpu, quiet = (read(root, n) for n in ("gpu", "parity", "cpu", "matched"))
    gs = gpu["stages"]
    native = quiet["stages"]["matched_native_runtime"]
    traces = list(traced["stages"].values())
    p = parity({"trace_old_vs_uninstrumented_old": [s["traced_reference_vs_reference"] for s in traces],
                "fused_trace_vs_old": [s["fused_trace_vs_reference"] for s in traces],
                "warp_vs_old": [s["warp_vs_reference"] for s in traces],
                "numpy_vs_old": [s["numpy_vs_reference"] for s in traces],
                "raw_counts": [{k: v for k, v in s.items() if k.startswith("raw_fused_counts")
                                or k == "fused_traced_vs_uninstrumented_counts"} for s in traces],
                "pipeline": [{k: v for k, v in s.items() if k not in {"settings", "shape", "world_sha256"}}
                             for s in gs["pipeline_parity"]["cases"]]})
    cpu_hash = cpu["stages"]["native_numpy_1_vs_4"]["score_numerators_sha256"]
    matched_hash = {stage["numerators_sha256"] for block in native["raw"] for stage in block["positions"]
                    if "numerators_sha256" in stage}
    if matched_hash != {cpu_hash} or cpu_hash != traces[0]["cpu_score_numerators_sha256"]:
        raise ValueError("The matched timing and diagnostic workloads differ")
    args = gpu["metadata"]["args"]
    runs = {k: gs[k]["summary"] for k in STAGES}
    host_gpu = hardware_context({k: gs[k] for k in STAGES})
    host_cpu = hardware_context({**cpu["stages"], **quiet["stages"]})
    breakdown = pipeline_breakdown(gs["prefetch_pipeline"]["raw"])
    result = {"gpu": runs, "parity": p, "native": native["contrast"], "contexts": {**host_gpu, **host_cpu},
              "native_median_seconds": native["summary"], "gpu_resident": gpu["resident_case"],
              "pipeline_worlds": [r["shape"] for r in gs["pipeline_parity"]["cases"]],
              "pipeline_breakdown": breakdown, "args": args, "score_numerators_sha256": cpu_hash}
    width, pairs, blocks = (args[key] for key in ("width", "world_pairs", "blocks"))
    ms = 1000
    lines = ["# 实际物理评分吞吐独立复核", "",
             f"RTX 4060 Ti（8 GB）、i5-14600KF（14 核、20 逻辑线程）、Windows。GPU 计时每批 {width:,} 个"
             f"来自同一冻结遗传检查点的候选，每个场景 {pairs} 对同生反事实世界，固定场景每个身体 {gpu['resident_case']['shape'][1]} 回合。"
             f"两路输入、动作物理和奖励完全相同。每级做 {blocks} 个随机平衡 ABBA/BAAB 块。"
             "表中是块内均值的中位数、几何成对加速比和按顺序分层重抽整块的 95% 区间。区间反映此机此组输入的重复运行波动。", "",
             "| 计时范围 | 旧路径中位 | 新路径中位 | 旧耗时 ÷ 新耗时［95% 区间］ |",
             "|---|---:|---:|---:|"]
    for key, what in STAGES.items():
        row = runs[key]
        a, b = row["median_seconds_a_b"]
        unit, scale = ("s", 1) if key == "prefetch_pipeline" else ("ms", ms)
        lines.append(f"| {what} | {a*scale:.3f} {unit} | {b*scale:.3f} {unit} | {factor(row)} |")
    end_a, end_b = runs["prefetch_pipeline"]["median_seconds_a_b"]
    timing_a, timing_b = (breakdown[k]["median_of_generations"] for k in "AB")
    total_candidates = width * args["pipeline_generations"]
    lines.extend(["", f"端到端每次依次评分 {args['pipeline_generations']} 批，每个世界的回合数为 "
                  f"{', '.join(str(s[1]) for s in result['pipeline_worlds'])}。含一个 CPU 工作者的世界生成、容量为 "
                  f"{args['prefetch']} 的预取队列、填充和排空、CUDA 世界与感受器生成、基因解码、评分还原和选择排名。"
                  "常规分块上限为旧 64、新 2048，故新路径每次整批提交。"
                  "两边都预热，计时不纳入首次 JIT、遗传更新、独立验证和检查点落盘。"
                  "CUDA event 用常驻世界、已解码候选和预分配输出，仍可能包括连续 launch 的间隔。"
                  "常驻评分还含基因上传、设备分配、归约和下载分数。", "",
                  f"端到端吞吐约 {total_candidates / end_a:,.0f} → {total_candidates / end_b:,.0f} 候选/秒。"
                  f"每批 CPU 世界准备的中位耗时为旧 {timing_a['host_world_seconds']*ms:.2f} ms、新 {timing_b['host_world_seconds']*ms:.2f} ms，"
                  f"主线程等世界分别 {timing_a['host_world_wait_seconds']*ms:.2f} ms 与 {timing_b['host_world_wait_seconds']*ms:.2f} ms。"
                  "准备与设备执行可重叠。", "",
                  "## 逐拍与计分一致性", "",
                  "| 对照 | 双通道逐拍动作不同 | 末端位置/速度不同 | 接触方向、拍、奖励各自不同 |",
                  "|---|---:|---:|---:|"])
    for key, label in (("warp_vs_old", "CUDA warp 诊断副本 ↔ 旧 CUDA"), ("numpy_vs_old", "原生 NumPy ↔ 旧 CUDA")):
        row = p[key]
        discrete = " / ".join(disagreement(row[field]) for field in ("side", "contact_tick", "reward"))
        lines.append(f"| {label} | {disagreement(row['motor_pulse_pair'])} | {disagreement(row['final_position_velocity'])} | {discrete} |")
    ftrace = p["fused_trace_vs_old"]
    lines.append(f"| 融合 + warp 诊断副本 ↔ 旧 CUDA | {disagreement(ftrace['motor_pulse_pair'])} | "
                 f"{disagreement(ftrace['final_position_velocity'])} | 仅留整数计数归约 |")
    np_gain = p["numpy_vs_old"]["rule_gain_start"]
    native_counts = p["raw_counts"]["raw_fused_counts_vs_numpy"]
    raw = p["pipeline"]
    gpu_busy = ", ".join(f"{v['cpu_busy']:.1%}" for v in host_gpu.values())
    lines.extend(["", f"三套动作对拍各用 {traces[0]['work']['candidates']} 个候选 × {traces[0]['work']['independent_pairs']} 对世界。"
                  f"旧 CUDA 改为仅记录动作与末端的诊断副本，在已有输出字段的结果不同计数均为 "
                  f"{max(v['different'] for v in p['trace_old_vs_uninstrumented_old'].values())}。NumPy 规则增益最大绝对误差 {np_gain['max_abs']:.3g}。"
                  f"融合分数的原始整数计数与 NumPy 不同 {disagreement(native_counts)}，加入诊断输出前后不同 "
                  f"{disagreement(p['raw_counts']['fused_traced_vs_uninstrumented_counts'])}。", "",
                  f"完整流水线世界的融合与旧实现原始整数分数不同 {disagreement(raw['counts'])}，最终浮点分数不同 "
                  f"{disagreement(raw['float_scores'])}（最大绝对差 {raw['float_scores']['max_abs']:.3g}）。"
                  f"稳定排序的候选名次不同 {disagreement(raw['stable_island_rank'])}，入选集合不同 {disagreement(raw['top_selection_membership'])} "
                  f"（按每岛候选位置计），获胜候选不同 {disagreement(raw['island_winners'])}，获胜候选的精确整数适应度不同 "
                  f"{disagreement(raw['winner_exact_utility'])}。", "",
                  "## 小批量原生动作吞吐", "",
                  f"同一场景和个体，{traces[0]['work']['candidates']} 候选 × {traces[0]['work']['independent_pairs']} 世界对 × "
                  f"{traces[0]['work']['trials']} 回合。四方法轮换各跑 {len(native['raw'])} 次。NumPy 每个进程只用 1 个 BLAS 线程，"
                  "Windows 调度器可迁移进程。GPU 时段从已构建的设备常驻世界评分到分数回到 CPU。NumPy 用原生身体闭环并汇总同样计数。"
                  "各方法在计时前构建同一世界，进程启动与环境构建耗时另外保存。", "",
                  "| 方法 | 中位每批 |", "|---|---:|"])
    for key, label in CPU_LABELS.items():
        lines.append(f"| {label} | {native['summary'][key]*ms:.3f} ms |")
    cpu1v4 = native["contrast"]["numpy_serial_over_numpy_four_processes"]
    cpu4vfused = native["contrast"]["numpy_four_processes_over_gpu_fused"]
    lines.extend(["", f"NumPy 四进程对单进程为 {factor(cpu1v4)}。GPU 融合评分对四进程为 {factor(cpu4vfused)}，"
                  "评分从已准备好的世界开始。完整生成到计分的实际加速使用上表流水线。"
                  f"小批量世界生成 {native['host_world_seconds']*ms:.1f} ms，首次 CUDA 建立和编译 "
                  f"{native['first_gpu_setup_seconds']*ms:.1f} ms，四进程冷启动加首次评分 {native['warm']['B']['seconds']:.2f} s。", "",
                  f"GPU 计时期间整机 CPU 忙比例各阶段为 {gpu_busy}，小批量四方法轮换为 "
                  f"{host_cpu['matched_native_runtime']['cpu_busy']:.1%}。JSON 含每阶段起止时刻的 GPU 利用率、时钟、温度、"
                  "系统进程列表及每次计时。窗口观测包含桌面等整机负载，未锁频或绑核。"
                  f"另有一次研究 GPU 并发期间的 NumPy 对照，四进程／单进程为 {factor(cpu['stages']['native_numpy_1_vs_4']['summary'])}。", "",
                  "原始数据位于同目录 `gpu.json`、`parity.json`、`matched.json` 和 `cpu.json`。`quick-concurrent.json`"
                  " 与 `smoke-concurrent.json` 是并发背景下的短验收。", "",
                  "```powershell", "$out = 'data/pulse-game/research-2026-11-19/physical-throughput'",
                  "foreach ($part in @('gpu', 'cpu', 'parity', 'matched')) {",
                  "    .venv/Scripts/python.exe -m experiments.pulse_game.physical_throughput_bench --part $part --out \"$out/$part.json\"",
                  "}", ".venv/Scripts/python.exe -m experiments.pulse_game.physical_throughput_report $out", "```", ""])
    if raw["counts"]["different"] or native_counts["different"]:
        lines[2:2] = ["**原始整数分数存在差异。下面的耗时只能用于诊断。**", ""]
    return result, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result, report = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
