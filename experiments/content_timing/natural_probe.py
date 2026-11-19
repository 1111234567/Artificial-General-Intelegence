"""Within-policy calibration for interpreting transfer across different histories."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.content_timing import probe

OUT = probe.DATA / "natural-probe-v1"
PHASES = ("cue", "processing_mid", "first_legal")
PLAN = "事后对照，选在已知固定回执探针结果、未知迁移和本对照成绩时。只用三阶段齐全的真实成熟请求。每种子、难度、任务槽、阶段分别在自然数据上拟合同一种四类岭回归，以 actor 4/5 和 6/7 整组互换训练、测试。只使用本阶段24个非输入节点，拟合量取相应自然训练侧全部请求，无筛选或调参。真实标签为原始符号或映射后内容。保存四类混淆矩阵、准确率、宏平均召回与训练侧多数类基线。比较诊断读出的样本域，训练样本量不匹配。所有模型实际动作仍由原策略产生。"


def sources():
    paths = [Path(__file__), Path(probe.__file__), probe.DATA/"natural-v1/protocol.json",
             probe.DATA/"transfer-v1/protocol.json", probe.DATA/"transfer-v1/replay_correction.json"]
    paths.extend((probe.DATA/"natural-v1"/f"seed_{s:03d}"/"requests.jsonl") for s in range(3))
    return {str(p.relative_to(probe.ROOT)):probe.sha(p) for p in paths}


def run(seed):
    base = probe.DATA/"natural-v1"/f"seed_{seed:03d}"
    features = probe.DATA/"transfer-v1"/f"seed_{seed:03d}"
    replay = json.loads((features/"replay.json").read_text(encoding="utf-8"))
    for filename, expected in replay["artifacts"].items():
        if probe.sha(Path(filename)) != expected:
            raise ValueError("natural continuous feature archive changed")
    records = json.loads((base/"result.json").read_text(encoding="utf-8"))["records"]
    rows = [r for r in map(json.loads,(base/"requests.jsonl").read_text(encoding="utf-8").splitlines())
            if r["matured"] and r["stage_status"] == "complete_three_stage"]
    folds = []
    summary = []
    for condition in ("short", "long_both"):
        chosen = [r for r in rows if r["condition"] == condition]
        cache = {}
        for r in records:
            if r["condition"] == condition:
                family = r["family"]
                arr = torch.load(features/f"{condition}_family_{family}.features.pt", map_location="cpu",
                                 weights_only=False)["x24"].numpy()
                cache[family] = (r["record"]["fork_tick"] + 1, arr)
        jobs = np.array([r["job"] for r in chosen])
        lanes = np.array([r["lane"] for r in chosen])
        for phase in PHASES:
            x = np.array([cache[r["fork_family"]][1][r[phase]["tick"]-cache[r["fork_family"]][0], r["lane"]]
                          for r in chosen])
            assert x.shape == (len(chosen), 24) and x.dtype == np.float64
            for label in ("raw", "mapped"):
                y = np.array([r["raw_payload"] if label == "raw" else r[phase]["target"] % 4 for r in chosen])
                total = np.zeros((4, 4), dtype=int)
                baseline = 0
                for job in range(2):
                    for held in ((4, 5), (6, 7)):
                        test = (jobs == job) & np.isin(lanes, held)
                        train = (jobs == job) & ~test
                        scores = probe.fit_scores(x[train], y[train], x[test])
                        pred = scores.argmax(-1)
                        cm = np.bincount(4*y[test]+pred, minlength=16).reshape(4,4)
                        most = int(np.bincount(y[train], minlength=4).argmax())
                        count = int((y[test] == most).sum())
                        folds.append(dict(condition=condition, phase=phase, label=label, job=job, heldout=list(held),
                            n_train=int(train.sum()), train_majority=most, baseline_correct=count, confusion=cm.tolist()))
                        total += cm
                        baseline += count
                assert total.sum() == len(chosen) and total.sum(axis=1).min() > 0
                summary.append(dict(condition=condition, phase=phase, label=label, correct=int(total.trace()),
                    n=len(chosen), accuracy=total.trace()/len(chosen), macro=float(np.mean(total.diagonal()/total.sum(1))),
                    majority_baseline=baseline/len(chosen), confusion=total.tolist()))
    return dict(seed=seed, replay_sha256=probe.sha(features/"replay.json"), folds=folds, summary=summary)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    plan = OUT/"protocol.json"
    if sys.argv[1:] == ["declare"]:
        with plan.open("x", encoding="utf-8") as f:
            json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN, inputs=sources()),
                      f, ensure_ascii=False, indent=2)
    elif not sys.argv[1:]:
        assert sources() == json.loads(plan.read_text(encoding="utf-8"))["inputs"]
        result = dict(protocol_sha256=probe.sha(plan), seeds=[run(seed) for seed in range(3)])
        with (OUT/"summary.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    else:
        raise SystemExit("usage: python -m experiments.content_timing.natural_probe [declare]")
