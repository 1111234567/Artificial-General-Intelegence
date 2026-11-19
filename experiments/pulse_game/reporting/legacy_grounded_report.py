"""Summarize independent original-world transfer audits and the replay release."""

import json

import numpy as np
import torch

from .. import research_audit as audit
from ..legacy_grounded import read, sha
from ..legacy_grounded_graph_control import bootstrap_difference


ORDER = (
    ("previous_bridge", "上一桥接"),
    ("trained_factor", "训练图＋公开感知校准＋因子地址"),
    ("random_graph_factor", "独立随机图＋相同公开校准与因子地址"),
    ("source_full_affine", "原嗅觉仿射读出＋显式情境因子"),
    ("actual", "训练图＋实际 ACK/REJ 学得地址"),
    ("cold_actual", "同出生冷图＋实际 ACK/REJ 学得地址"),
    ("initial", "训练图＋未训练地址头"),
    ("shuffled", "训练图＋打乱 ACK/REJ 学得地址"),
    ("no_write", "实际回执头＋停写"),
    ("wrong_binding", "实际回执头＋错地址"),
    ("actual_cut_sensory_loop", "实际回执头＋切感知循环"),
    ("actual_hide_context_input", "实际回执头＋去情境输入"),
    ("random_actions", "随机合法动作"),
    ("table_once", "公开查表，一次作答"),
    ("table_retry", "公开查表，允许补答"),
)


def rate(rows, key):
    v = np.asarray([x[key] for x in rows], dtype=np.float64)
    ix = np.random.default_rng(16183694).integers(len(v), size=(10000,len(v)))
    return {"mean":float(v.mean()), "ci95":np.quantile(v[ix].mean(1), [.025,.975]).tolist(), "worlds":len(v)}


def gather(folder):
    paths = {name:folder/("legacy_grounded-"+name+".json")
             for name in ("selected","confirmed","outcomes","graph-control","replay","sensory-interventions")}
    selected, first, actual, cold, replay, removal = [read(p) for p in paths.values()]
    cp = folder/"legacy_grounded-release.pt"
    record = torch.load(cp,map_location="cpu",weights_only=True)
    if (sha(cp) != replay["checkpoint_sha256"] or sha(cp) != removal["release_sha256"]
            or sha(paths["outcomes"]) != record["original_closed_loop_report_sha256"]):
        raise ValueError("The released checkpoint and the underlying evaluations do not match")
    sets = {}
    for name in ("original32","new32"):
        roots = first["sets"][name]["roots"]
        if any(row["sets"][name]["roots"] != roots for row in (actual,cold,replay,removal)):
            raise ValueError("The held-out evaluations do not pair")
        mode = first["sets"][name]["rows"].copy()
        mode.update(actual["sets"][name]["rows"])
        mode.update(cold["sets"][name]["rows"])
        mode.update(removal["sets"][name]["rows"])
        if not replay["sets"][name]["matches_original_rows_exactly"] or replay["sets"][name]["rows"] != mode["actual"]:
            raise ValueError("Replay did not reproduce the selected address")
        keys = ("strict","pair_strict_rate","completion","ready_coverage","premature_request_rate")
        estimates={arm:{key:rate(rows,key) for key in keys} for arm,rows in mode.items()}
        pairs={arm:{key:bootstrap_difference(mode["actual"],rows,key) for key in keys} for arm,rows in mode.items()
               if arm!="actual"}
        sets[name]={"roots":roots,"estimates":estimates,"paired_actual_minus_control":pairs}
    return {"source_files_sha256":{str(p):sha(p) for p in paths.values()},
            "release":{"path":str(cp),"sha256":sha(cp),"replayed_each_world_exactly":True},
            "sensory_graph":{"checkpoint":first["source_checkpoint"],
                             "checkpoint_sha256":first["source_sha256"],
                             "nodes":first["graph_nodes"],"directed_nonzero_edges":first["directed_nonzero_edges"],
                             "same_birth_mask":cold["same_direction_mask"],
                             "pretraining_max_absolute_changes":cold["public_fitting_max_change"]},
            "source_public_fit":{"steps":360,"objective":"reconstruct public observed odor"},
            "factor_selection":{"settings":selected["best"]["setting"],"development_seeds":selected["development_seeds"],
                                "public_calibration_seeds":selected["calibration_seeds"]},
            "actual_address_selection":record["receipt_selection"],"namespaces":record["world_namespaces"],
            "learning_rule":"frozen sensory graph; local RLS indexed by actual issued action; MLP address learned on receipt prefixes",
            "paired_estimator":"paired worlds, 10000 bootstrap draws, fixed checkpoints; no claims over training seeds",
            "sets":sets}


def format_pair(row):
    p=row["difference"]*100
    lo,hi=(x*100 for x in row["ci95"])
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def report_text(data):
    old,new=(data["sets"][k] for k in ("original32","new32"))
    def key(arm, measurement):
        return [100*row["estimates"][arm][measurement]["mean"] for row in (old,new)]

    lines=[
      "# 公开嗅觉图迁移到原符号世界",
      "",
      "旧模拟器每生命 64 道请求，按原来的 READY、接受、拒绝与严格完成记分。双回访计同一线索在 A/B 中的两个合格配对回访都严格完成。原 32 世界与另 32 个新世界各包含每世界 8 个生命。区间以世界为配对单位，自举 10,000 次，适用于表中固定检查点。",
      "",
      "| 冻结策略 | 原 32 严格 | 原 32 双回访 | 新 32 严格 | 新 32 双回访 |",
      "|---|---:|---:|---:|---:|",
    ]
    for arm,label in ORDER:
        strict=key(arm,"strict");dual=key(arm,"pair_strict_rate")
        lines.append(f"| {label} | {strict[0]:.2f}% | {dual[0]:.2f}% | {strict[1]:.2f}% | {dual[1]:.2f}% |")
    old_p,new_p=(r["paired_actual_minus_control"] for r in (old,new))
    seal=data["actual_address_selection"]
    source=data["sensory_graph"]
    lines.extend([
      "",
      ("源图有 24 个连续活动节点和 "
       f'{source["directed_nonzero_edges"]} 条非零有向边。此前通过 360 步公开嗅觉重建训练。'
       f'与同出生冷态相比，循环权重最大改变量为 {source["pretraining_max_absolute_changes"]["dynamics"]:.3f}，'
       f'输入权重为 {source["pretraining_max_absolute_changes"]["afferents"]:.3f}，方向掩码相同。'
       "原世界的公开 raw4 和情境2进入冻结图，状态跨请求延续。输出仍走 READY 调制的四通道 LIF 运动器。"
       "因子地址明确预设 cue 与情境分开。公开校准仅重建早先公开的感知。"),
      "",
      (f'实际回执地址头的训练与开发分别用 {seal["train_actual_receipts"]:,} 和 '
       f'{seal["development_actual_receipts"]:,} 条真实随机合法作答的 ACK/REJ。'
       "只用该次实际选择的签名，状态预测只读此前回执。没有给私有正确符号或其余候选的标签。"
       f'开发集签名前缀预测误差最小在第 {seal["best_development_step"]} 步（{seal["best_development_error"]:.5f}）。'
       "该地址头先保存，随后才分别评估原 32 和新 32。"),
      "",
      (f'实际回执头相对上一桥接的严格差，原／新分别为 {format_pair(old_p["previous_bridge"]["strict"])} 和 '
       f'{format_pair(new_p["previous_bridge"]["strict"])} 个百分点。'
       f'相对可补答查表的严格差为 {format_pair(old_p["table_retry"]["strict"])} 和 '
       f'{format_pair(new_p["table_retry"]["strict"])}，双回访差为 {format_pair(old_p["table_retry"]["pair_strict_rate"])} 和 '
       f'{format_pair(new_p["table_retry"]["pair_strict_rate"])}。'
       "实际回执头和可补答查表在两批世界的完成率及 READY 覆盖都是 100%。原世界目标表为每情境的置换，"
       "A/B 在每个 raw 上异值。分槽查表是实测策略参照。"),
      "",
      (f'冻结训练图相对同出生冷图的严格配对增益为 {format_pair(old_p["cold_actual"]["strict"])} 和 '
       f'{format_pair(new_p["cold_actual"]["strict"])} 个百分点。冷图使用同一训练和开发轨迹，'
       "实际作答及签名逐值相同。冷图本身也能学得高分。真实回执头保持冻结，只切感觉循环或只抹情境输入时，"
       "表中闭环成绩下降，分别检验了跨拍感知和情境通道对行为的贡献。"),
      "",
      "    ./.venv/Scripts/python.exe -m experiments.pulse_game.legacy_grounded_release --step replay",
      "",
      "[冻结回放包](legacy_grounded-release.pt) · [逐世界配对区间](legacy_grounded-paired.json) · "
      "[精确重跑核对](legacy_grounded-replay.json) · [原始实际回执学习证据](legacy_grounded-outcomes.json)",
      "",
    ])
    return "\n".join(lines)


def main():
    torch.set_num_threads(1)
    folder = audit.OUT
    gathered=gather(folder)
    path=folder/"legacy_grounded-paired.json"
    path.write_text(json.dumps(gathered,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    note=folder/"legacy_grounded-report.md"
    note.write_text(report_text(gathered),encoding="utf-8")
    print(path)
    print(note)


if __name__ == "__main__":
    main()
