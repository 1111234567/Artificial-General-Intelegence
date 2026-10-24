# agi/store.py — 结果登记。JSONL，一条运行一行。
#
# 为什么必须要有：2026-10-24 我连「哪些组合跑过、结果如何」都只能靠回忆。
# 后果：同一个错重犯 M 次（5 轮补丁 + 3 个对照错误），每次都以为是新的。
# 有了登记，三件事变成可查的：
#   ① 这个配置跑过吗（按 fingerprint）
#   ② 这个配置的对照跑齐了吗（按指纹差集）
#   ③ 历史上最优的是什么（按指标排序）

import io
import json
import os
import time

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "results.jsonl")


def append(record, path=DEFAULT):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    rec = dict(record)
    rec.setdefault("at", time.strftime("%Y-%m-%d %H:%M:%S"))
    with io.open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def load(path=DEFAULT):
    if not os.path.exists(path):
        return []
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def seen(fp, path=DEFAULT):
    return [r for r in load(path) if r.get("fingerprint") == fp]


def summary(path=DEFAULT):
    rows = load(path)
    if not rows:
        print("（空）")
        return
    print("%-28s %-14s %5s %8s %8s %8s %8s" % (
        "config", "fingerprint", "seed", "succ", "basis", "outfire", "bias"))
    print("-" * 100)
    for r in sorted(rows, key=lambda x: -(x.get("succ") or 0)):
        print("%-28s %-14s %5s %8.3f %8.3f %8.3f %8.3f" % (
            r.get("config", "?")[:28], r.get("fingerprint", "?"),
            r.get("seed", "?"), r.get("succ", 0), r.get("basis", 0),
            r.get("out_fire", 0), r.get("out_bias", 0)))
    print("-" * 100)
    print("共 %d 条；不同配置指纹 %d 个" % (
        len(rows), len(set(r.get("fingerprint") for r in rows))))
