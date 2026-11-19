"""Draw natural decision accuracy and controlled state decodability side by side."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/content-timing"
OUTPUT = ROOT / "docs/evidence/content-timing-2026-11-03"


def oracle_version_payload(scores, target):
    # This oracle slice is a diagnostic; the policy never sees the true version.
    start = 4 * (target // 4)
    return int(np.argmax(scores[start:start+4])) == target % 4


def load_curves():
    natural_path = DATA / "natural-v1/summary.json"
    probe_path = DATA / "probe-v1/summary.json"
    natural = json.loads(natural_path.read_text(encoding="utf-8"))["per_seed"]
    probes = json.loads(probe_path.read_text(encoding="utf-8"))["seeds"]
    transfer_path = DATA / "transfer-v1/summary.json"
    calibration_path = DATA / "natural-probe-v1/summary.json"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))["per_seed"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["seeds"]
    curves = []
    files = [natural_path, probe_path, transfer_path, calibration_path]
    for seed in range(3):
        n = next(r for r in natural if r["seed"] == seed and r["condition"] == "long_both")
        # The same matured requests must have all three phases, including the cue.
        patterns = {format(i, "03b"): n["three_stage_correct_" + format(i, "03b")] for i in range(8)}
        total = sum(patterns.values())
        assert total == n["complete_three_stage"] and total > 0
        natural_curve = [sum(v for bits, v in patterns.items() if bits[p] == "1") / total for p in range(3)]
        probe = next(r for r in probes if r["seed"] == seed)
        matched_path = DATA / f"matched-v1/seed_{seed:03d}/records.pt"
        files.append(matched_path)
        record = torch.load(matched_path, map_location="cpu", weights_only=False)
        by_stage = []
        for phase in ("cue", "midpoint", "first_legal"):
            selected = [r for r in record["rows"] if r["stage"] == phase and r["duration"] == 8]
            by_stage.append(sum(r["conditional_choice8"] % 4 == r["mapped_payload"] for r in selected) / len(selected))
        requests = DATA / f"natural-v1/seed_{seed:03d}/requests.jsonl"
        files.append(requests)
        natural_rows = []
        with requests.open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if r["condition"] == "long_both" and r["matured"] and r["stage_status"] == "complete_three_stage":
                    natural_rows.append(r)
        assert len(natural_rows) == total
        known_natural = [sum(oracle_version_payload(r[p]["finish_scores"], r[p]["target"]) for r in natural_rows)
                         for p in ("cue", "processing_mid", "first_legal")]
        known_matched = []
        for p in ("cue", "midpoint", "first_legal"):
            chosen = [i for i, r in enumerate(record["rows"]) if r["stage"] == p and r["duration"] == 8]
            correct = sum(oracle_version_payload(record["finish_scores8"][i].numpy(), record["rows"][i]["target8"])
                          for i in chosen)
            known_matched.append(dict(correct=correct, total=len(chosen)))
        values = dict(seed=seed, natural_joint=natural_curve, natural_requests=total, original_payload=by_stage,
                      natural_known_version_correct=known_natural, matched_known_version=known_matched)
        for label in ("raw", "mapped"):
            values[label] = [next(r["accuracy"] for r in probe["pooled"] if r["duration"] == 8 and
                r["label"] == label and r["train_phase"] == r["test_phase"] == phase) for phase in ("cue", "mid", "first_legal")]
        real_phases = ("cue", "processing_mid", "first_legal")
        values["transfer_raw"] = [next(r["accuracy"] for r in transfer if r["seed"] == seed and r["phase"] == p
            and r["population"] == "complete_three_stage" and r["label"] == "raw" and r["condition"] == "long_both") for p in real_phases]
        in_domain = next(r["summary"] for r in calibration if r["seed"] == seed)
        values["natural_raw"] = [next(r["accuracy"] for r in in_domain if r["phase"] == p and
            r["label"] == "raw" and r["condition"] == "long_both") for p in real_phases]
        curves.append(values)
    bindings = {}
    for path in files:
        with path.open("rb") as handle:
            bindings[str(path.relative_to(ROOT))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(seeds=curves, source_sha256=bindings)


def draw(data):
    plt.rcParams.update({"font.family":"Microsoft YaHei", "font.size":10, "axes.unicode_minus":False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    fig.subplots_adjust(left=.052, right=.98, bottom=.27, top=.79, wspace=.24)
    labels = ("提示", "处理中点", "首次合法")
    fig.text(.045, .938, "结束动作选错时，内部状态里还能读出什么？", fontsize=16, weight="bold")
    colors = ("#186a72", "#a9511d", "#6e59a0")
    seeds = data["seeds"]
    a, b, c = axes
    for color, row in zip(colors, seeds):
        a.plot(labels, np.array(row["natural_joint"])*100, "o-", color=color, label=f"种子 {row['seed']}")
    a.set_title("自由行动  ·  实际读出的版本与内容（八选一）", fontsize=11)
    a.set_ylabel("正确率（%）")
    a.set_ylim(0, max(30, max(100*max(r["natural_joint"]) for r in seeds)+3))
    a.axhline(12.5, ls=":", lw=.9, color="#767676")
    a.legend(loc="upper center", bbox_to_anchor=(.5,-.14), ncol=3, fontsize=9, frameon=False)
    for color, (key, name) in zip(colors, (("raw","探针 · 原始符号"), ("mapped","探针 · 规则输出"),
                                         ("original_payload","原动作头 · 规则输出"))):
        vals = np.array([row[key] for row in seeds])*100
        b.plot(labels, vals.mean(axis=0), "o-", color=color, label=name)
        b.fill_between(labels, vals.min(axis=0), vals.max(axis=0), color=color, alpha=.09)
    b.set_ylim(0, 105)
    b.set_title("固定动作配对  ·  只看内容（四选一）", fontsize=11)
    b.axhline(25, ls=":", lw=.9, color="#767676")
    b.legend(loc="upper center", bbox_to_anchor=(.5,-.14), ncol=2, fontsize=8, frameon=False)
    c.set_title("真实长任务  ·  探针读原始符号（四选一）", fontsize=11)
    for color, (key, name) in zip(("#186a72","#825c25"), (("natural_raw","在自由轨迹上拟合"),("transfer_raw","在固定动作配对上拟合"))):
        vals = np.array([row[key] for row in seeds])*100
        c.plot(labels, vals.mean(axis=0), "o-", color=color, label=name)
        c.fill_between(labels, vals.min(axis=0), vals.max(axis=0), color=color, alpha=.09)
    c.set_ylim(0, 105)
    c.axhline(25, ls=":", lw=.9, color="#767676")
    c.legend(loc="upper center", bbox_to_anchor=(.5,-.14), fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.23)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.045, .038, "左、右图跟踪同一批自然长任务。中图固定处理时长为 8。中、右图实线为三种子均值，阴影为范围。虚线为均匀猜测。", color="#40505c", fontsize=9)
    fig.savefig(OUTPUT/"content-by-stage.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_curves()
    (OUTPUT/"figure-values.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    draw(data)
