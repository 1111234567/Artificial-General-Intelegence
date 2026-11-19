"""Report strict-pair transfer and the randomized rotation stress checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..spike_analysis import estimates


def load(base, part):
    return json.loads((base / f"symmetry-spike-{part}.json").read_text(encoding="utf-8"))


def aggregate(base):
    voltage, opponent, matrix, linear, dose = [load(base, key) for key in (
        "voltage-confirm", "opponent-confirm", "matrix-voltage-confirm", "matrix-linear-confirm", "exposure-confirm")]
    if voltage["seeds"] != opponent["seeds"] or matrix["world_seeds"] != linear["world_seeds"]:
        raise ValueError("Architecture comparison did not reuse paired worlds")
    splits = [set(voltage["seeds"]), set(matrix["world_seeds"]), set(dose["world_seeds"])]
    if any(splits[i] & splits[j] for j in range(3) for i in range(j)):
        raise ValueError("A held world leaked into another protocol")
    for blob in (voltage, opponent, matrix, linear):
        for row in blob["rows"]:
            if not all(row["same_a_observed_waveform"]) or not all(row["both_contact"]):
                raise ValueError("A claimed paired waveform or contact differs")
    direction = {}
    for name, scene in (("voltage", voltage), ("opponent", opponent)):
        okay = [r for r in scene["rows"] if r["name"] == "native"]
        gain = [np.asarray([r["state"][i]["direction_at_first_probe"] for r in okay]) for i in (0, 1)]
        direction[name] = {"family_positive_gain": estimates(gain[0].mean(0)),
                           "family_negative_gain": estimates(gain[1].mean(0)),
                           "opposite_signs_in_mirror_lives": estimates(((gain[0] * gain[1]) < 0).mean(0)),
                           "strict_first_score": scene["summary"]["native"]["all_correct"]}
    current = Path(__file__).resolve().parents[1]
    manifests = {}
    for key in ("voltage", "opponent", "matrix-voltage", "matrix-linear", "exposure"):
        x = load(base, key + "-frozen")
        marks = x.get("code", x.get("source"))
        manifests[key] = {name: hashlib.sha256((current / name).read_bytes()).hexdigest() == digest for name, digest in marks.items()}
    return {"scope": {"strict_worlds": len(voltage["seeds"]), "rotation_worlds": len(matrix["world_seeds"]),
                      "dose_worlds": len(dose["world_seeds"]), "births": voltage["births"], "source_checks": manifests},
            "strict": {"voltage": voltage["summary"], "opponent": opponent["summary"]},
            "learned_sign": direction, "rotation": {"voltage": matrix["summary"], "linear": linear["summary"]},
            "rotation_paired": {"voltage": matrix["matrix_matches"], "linear": linear["matrix_matches"]},
            "dose": dose["summary"]}


def pct(x):
    return f"{100*x['mean']:.2f}%"


def points(x):
    return f"{100*x['mean']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]"


def readout(data):
    v, h = (data["strict"][label] for label in ("voltage", "opponent"))
    rot = data["rotation"]["voltage"]
    lines = ["# 相同回执流下的局部元可塑性", "",
             "原世界 12 对气味、每 4 对一组。A 段只把当前生命的气味关系在正负两类之间交换。两条生命的实际 A 逐拍回执完全相同，"
             "B 的第一次真实试吃后，下一次选择关闭奖励。以下都按实际接触计分，每个世界的两种隐关系同时测试。", "",
             "## 单个方向突触", "",
             "32 节点稀疏 LIF 从公开闪光产生当前膜电位与活动迹，保留单个气味地址到真实接触。两套衰减关联权重由公开背景门控，"
             "接触后用触前回忆与实际回执的乘积调整一个共享方向突触。两侧比较取第 5 拍，后续脉冲实际推动身体。"
             "模型的适应参数不含事件表或家族标签。下表汇总 384 个新世界、4 个新出生。", "",
             "| 条件 | B 首次实际选对 | 同世界正反双对 |", "|---|---:|---:|"]
    counter = (("无镜像膜/迹地址", v["native"]), ("推挽 LIF 参考", h["native"]),
               ("只读脉冲迹", v["single_spike_trace_readout"]), ("切递归", v["cut_recurrence"]),
               ("固定方向 (+1)", v["fixed_positive"]), ("乱配 A 回执", v["permute_signs_a"]),
               ("只在首个 B 探针反转学到的方向", v["invert_gain_at_first_b"]),
               ("解析三次角核参考", v["generic_cubic_reference"]))
    for name, r in counter:
        lines.append(f"| {name} | {pct(r['all_correct'])} | {pct(r['both_correct'])} |")
    lines.extend(["", f"整段反转 A 的符号保留同样的首选成绩 {pct(v['invert_a']['all_correct'])}。这符合乘积局部规则下，同一对经历中的两个已见回执同时反转仍提供同样关系证据。"
                  f"共用两个背景的同一关联权组，首选降至 {pct(v['shared_habitat_weights']['all_correct'])}。膜/迹混合地址和真实方向更新在这个方案中有作用，切掉储层递归的收益变化区间覆盖零。", "",
                  "## 小型元突触矩阵", "",
                  "矩阵从零开始。当前已尝气味地址与触前分布式回忆的外积受签名回执调制，矩阵改变下一次回忆的读取方向。"
                  "额外压力测试只改变配对气味的物理变换，噪声、实际 A 回执和运动接口保留。仍是 384 个另外的新世界。", "",
                  "| 配对气味几何 | 全矩阵 | 对角限制 | 同地址标量 |", "|---|---:|---:|---:|"])
    for geometry, cn in (("opposite", "原反足"), ("near-identical", "近同向"), ("quarter-turn", "每生随机正交九十度")):
        labels = ("matrix_native", "matrix_diagonal", "scalar_same_encoding")
        nums = " | ".join(pct(rot[f"{geometry}-{key}"]["all_correct"]) for key in labels)
        lines.append(f"| {cn} | {nums} |")
    stats = data["rotation_paired"]["voltage"]["quarter-turn"]
    lines.extend(["", f"九十度条件下全矩阵比同地址标量高 {points(stats['scalar_same_encoding']['all_correct'])} 个百分点，比对角高 {points(stats['matrix_diagonal']['all_correct'])} 点。"
                  "原反足和近同向使用相同样本数时，满矩阵落后于标量。无镜像标量有两组 32 个关联权重和一项方向；矩阵版还存每个背景的第二时间尺度权组和 1,024 个方向参数。", "",
                  "## 完整嗅觉历史下的回执剂量", "",
                  "在相同 24 对 A 物理经历中，只公开最后 0、4、12 或 24 对真回执。后续 B 探针逐条件固定。以下每行是完整矩阵，256 个新世界。", "",
                  "| 气味几何 | 0 对 | 4 对 | 12 对 | 24 对 | 24 对标量 |", "|---|---:|---:|---:|---:|---:|"])
    for key, label in (("quarter-turn", "随机九十度"), ("haar-SO4", "每生随机正交旋转")):
        scene = data["dose"][key]
        nums = " | ".join(pct(scene[arm]["mean_pair"]) for arm in ("matrix-0", "matrix-4", "matrix-12", "matrix-24", "scalar-full"))
        lines.append(f"| {label} | {nums} |")
    lines.extend(["", "[逐世界汇总与配对区间](symmetry-spike-summary.json) · [无镜像严格协议](symmetry-spike-voltage-frozen.json) · [矩阵冻结协议](symmetry-spike-matrix-voltage-frozen.json)"])
    return "\n".join(lines) + "\n"


def figure(data, save):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    green, purple, brown = "#226b64", "#7a61a1", "#985c42"
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    ax = axs[0]
    row = data["strict"]["voltage"]
    items = (("native", "Learned gain"), ("cut_recurrence", "Recurrence cut"), ("single_spike_trace_readout", "Spikes only"),
             ("permute_signs_a", "Mismatched A"), ("fixed_positive", "Fixed gain"), ("invert_gain_at_first_b", "Reverse learned gain"))
    for i, (key, label) in enumerate(items):
        v = row[key]["all_correct"]
        m, lo, hi = v["mean"], *v["ci95"]
        ax.errorbar(m, i, xerr=[[m-lo], [hi-m]], color=green if i == 0 else brown, marker="o", capsize=2)
    ax.set_yticks(range(len(items)), [label for _, label in items], fontsize=8)
    ax.set_ylim(len(items)-.5,-.5)
    ax.set_xlim(0, 1)
    ax.axvline(.5, color="#888888", ls=":", lw=.8)
    ax.set_title("A   Identical A reward waveforms", loc="left", fontweight="bold", fontsize=11)
    ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.set_xlabel("Correct unrewarded B first contact")

    ax = axs[1]
    names = ("opposite", "near-identical", "quarter-turn")
    for index, (arm, label, color, mark) in enumerate((("matrix_native", "Full matrix", green, "o"),
                                                      ("matrix_diagonal", "Diagonal only", purple, "s"),
                                                      ("scalar_same_encoding", "Scalar", brown, "^"))):
        for pos, geom in enumerate(names):
            v = data["rotation"]["voltage"][f"{geom}-{arm}"]["all_correct"]
            m, lo, hi = v["mean"], *v["ci95"]
            ax.errorbar(pos + .14*(index-1), m, yerr=[[m-lo], [hi-m]], color=color, marker=mark,
                        capsize=2, label=label if not pos else None)
    ax.set_xticks((0, 1, 2), ("Antipodal", "Near same", "Quarter turn"), fontsize=8)
    ax.set_ylim(.33, 1)
    ax.axhline(.5, color="#888888", ls=":", lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.set_title("B   Geometry changes between lives", loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="lower left", frameon=True, edgecolor="none", fontsize=8)
    ax.set_ylabel("Correct unrewarded B first contact")

    ax = axs[2]
    for orientation, label, color, marker in (("quarter-turn", "Quarter turn", green, "o"),
                                               ("haar-SO4", "Random rotation", purple, "s")):
        curves = data["dose"][orientation]
        p = (0, 4, 12, 24)
        points_ = [curves[f"matrix-{i}"]["mean_pair"] for i in p]
        values = np.array([v["mean"] for v in points_])
        interval = np.array([v["ci95"] for v in points_])
        ax.errorbar(p, values, yerr=[values-interval[:,0], interval[:,1]-values], color=color,
                    marker=marker, capsize=2, label=label, lw=1.4)
        g = curves["scalar-full"]["mean_pair"]["mean"]
        ax.scatter(24, g, marker="x", color=color, s=20)
    ax.axhline(.5, color="#888888", ls=":", lw=.8)
    ax.set_xticks((0, 4, 12, 24))
    ax.set_ylim(.38, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.set_title("C   Same odors, more receipts exposed", loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel("A odor pairs with true displayed taste")
    ax.set_ylabel("Correct unrewarded B first contact")
    ax.legend(loc="upper left", frameon=True, edgecolor="none", fontsize=8)
    ax.text(.02, .055, "× scalar at full dose", ha="left", transform=ax.transAxes, fontsize=8)
    for panel in axs:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y" if panel is not axs[0] else "x", lw=.55, color="#dddddd")
    fig.savefig(save, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    base = args.directory
    stats = aggregate(base)
    (base / "symmetry-spike-summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (base / "symmetry-spike-summary.md").write_text(readout(stats), encoding="utf-8")
    figure(stats, base / "symmetry-spike-results.png")


if __name__ == "__main__":
    main()
