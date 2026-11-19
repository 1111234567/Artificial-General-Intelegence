"""Collect held-out behavior, sign readouts, and per-birth mark-law performance."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "data/revisit-choice-20261119"
OUT = REPO / "docs/evidence/revisit-choice-2026-11-19"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def paired(values):
    x = np.asarray(values)
    if len(x) != 6:
        raise ValueError("Expected all six paired births")
    mu = x.mean()
    half = 2.57058184 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(mu), ci95=[mu-half, mu+half], by_birth=x.tolist())


def chart(probe, roots, marks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.3, 3.7), layout="constrained")
    signals = np.array(probe["mean"]["accuracy"]) * 100
    for i, label, color in ((0, "反馈是正还是负", "#087c85"), (1, "首次哪一侧有利", "#b35228")):
        left.plot(range(5), signals[:, i], "o-", label=label, color=color)
    left.set_xticks(range(5), ["品尝前", "品尝后", "编辑提交后", "等待四拍", "再次闻到"])
    left.set_ylabel("独立读出器正确率（%）")
    left.set_title("陌生气味任务中的内部活动")
    left.legend(loc="upper right", frameon=False, fontsize=9)
    for key, label, marker, color in (("law1", "甲标记有利", "o", "#087c85"),
                                      ("law-1", "乙标记有利", "s", "#b35228")):
        right.scatter(range(6), 100*np.array(marks[key]["by_birth"]), s=50, marker=marker,
                      color=color, label=label)
    right.set_xticks(range(6), roots)
    right.set_xlabel("独立出生编号")
    right.set_ylabel("模型实际选择正确率（%）")
    right.set_title("标记任务，每个生命重新确定哪一种有利")
    right.legend(loc="center", ncols=2, frameon=False, fontsize=9)
    for ax in (left, right):
        ax.axhline(50, color="#9ca39f", lw=1, zorder=0)
        ax.set_ylim(42, 104)
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT / "behavior.png", dpi=170)
    plt.close(fig)


def main():
    plan = read(RUN / "run.json")
    ev = read(RUN / "eval.json")
    probe = read(RUN / "probe-summary.json")
    value = read(RUN / "value/longitudinal/summary.json")
    roots = plan["births"]
    stages = {}
    for step in (4000, 8000, 12000):
        stages[step] = {}
        for arm, records in value.items():
            rows = [records[str(s)][str(step)] for s in roots]
            values = {k: paired([r[k] for r in rows]) for k in rows[0]}
            values["better_law"] = paired([max(r["law1"],r["law-1"]) for r in rows])
            values["good_minus_yoked"] = paired([r["own"]-r["yoked"] for r in rows])
            stages[step][arm] = values
    main_scores = {}
    for arm, step_data in ev["groups"].items():
        main_scores[arm] = {
            step: {delay: {feedback: g["behavior"][feedback]["natural_first"]["natural"]["actual"]
                           for feedback in ("true","zero","yoked","flipped")}
                   for delay, g in data["same"].items()} for step, data in step_data.items()
        }
    check = {}
    for birth in roots:
        initial = None
        for arm in plan["arms"]:
            for step in plan["checkpoints"]:
                path = RUN / f"birth_{birth}/{arm}/checkpoint_{step}.npz"
                with np.load(path) as g:
                    w, b, m = (g[key] for key in ("weights","bias","present"))
                    if not np.isfinite(w).all() or not np.isfinite(b).all() or np.any(w[~m] != 0):
                        raise ValueError(f"Invalid numeric state in {path}")
                    if step == 0:
                        if initial is None:
                            initial = (w,b,m)
                        elif not all(np.array_equal(x,y) for x,y in zip(initial,(w,b,m),strict=True)):
                            raise ValueError(f"Birth differs between arms: {birth}")
                check[path.relative_to(RUN).as_posix()] = sha256(path.read_bytes()).hexdigest()
    sources = [RUN / filename for filename in ("run.json","eval.json","eval-trials.npz",
               "probe-summary.json","probe-scores.npz","value/protocol.json","value/summary.json",
               "value/main-probe.json","value/longitudinal/protocol.json","value/longitudinal/summary.json",
               "value/reverse-transfer.json","value/exchange-4000.json","value/longitudinal/exchange-12000.json")]
    summary = dict(births=roots, autonomous_first_choice=main_scores, linear_readouts=probe["mean"],
                   variable_law_continuation=stages, checkpoint_sha256=check,
                   source_artifact_sha256={p.relative_to(RUN).as_posix(): sha256(p.read_bytes()).hexdigest()
                                           for p in sources})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary,indent=1,allow_nan=False)+"\n", encoding="utf8")
    chart(probe, roots, stages[12000]["variable"])


if __name__ == "__main__":
    main()
