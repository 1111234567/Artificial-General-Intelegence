# test/run.py — 唯一入口。
#
# 用法：
#   python test/run.py list                      列出所有实验（含指纹与开关）
#   python test/run.py show <name>               展开一份配置的全部开关
#   python test/run.py check <name>              只跑前置检查（任务可行性等）
#   python test/run.py run <name> [nseed] [nproc] 跑并登记
#   python test/run.py results                   查结果登记
#   python test/run.py gaps                      查对照组差集（哪些还没跑）
#
# 原 21 个 grow_*.py / probe_*.py 已移入 test/archived/（保留作为实验记录），
# 新实验一律走这里。

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from multiprocessing import Pool

from agi import experiments, run, append, summary, load
from agi.config import Config

ENV_ROOT_OK = True


def cmd_list():
    print("%-16s %-13s %-6s %s" % ("name", "fingerprint", "n_ep", "sove"))
    print("-" * 110)
    for c in experiments.ALL:
        flags = []
        if c.plast.weight:
            flags.append("w")
        if c.plast.topology:
            flags.append("topo")
        if c.plast.bias:
            flags.append("bias")
        print("%-16s %-13s %-6d %s" % (
            c.name, c.fingerprint(), c.n_ep,
            "+".join(flags) + "  m=%s/%s/%s%s" % (
                c.signal.source, c.signal.shape, c.signal.glob,
                "" if c.signal.gate else "(nogate)")))
        if c.note:
            print("%-16s   %s" % ("", c.note))


def cmd_show(name):
    c = experiments.BY_NAME.get(name)
    if c is None:
        print("未知实验：%s" % name)
        return
    import json
    print("name        : %s" % c.name)
    print("fingerprint : %s" % c.fingerprint())
    print("note        : %s" % c.note)
    print()
    print(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))


def cmd_check(name=None):
    targets = [experiments.BY_NAME[name]] if name else experiments.ALL
    bad = 0
    for c in targets:
        errs = c.validate()
        if errs:
            bad += 1
            print("FAIL %-16s %s" % (c.name, "; ".join(errs)))
        else:
            print("OK   %-16s %s" % (c.name, _feasible_msg(c)))
    if bad:
        print("\n%d 个实验未通过前置检查（不该跑）" % bad)


def _feasible_msg(c):
    """前置检查的**通过**说明——按 env.kind 分派（否则 2D 世界会打错口径）。"""
    if c.env.kind == "math_2d":
        n_win = c.task.t_ep // c.task.t_act
        reach = n_win * c.env.k_act * c.task.t_act
        gap = max(abs(c.env.g_hi - c.env.x0_lo), abs(c.env.g_lo - c.env.x0_hi))
        return "2D 可达 %.2f >= 差距 %.2f×1.2" % (reach, gap)
    return c.task.check_feasible(max(abs(a) for a in c.env.actions))[1]


def _one(arg):
    name, seed = arg
    c = experiments.BY_NAME[name].with_(seed=seed)
    return run(c)


def cmd_run(name, nseed=8, nproc=8):
    c = experiments.BY_NAME.get(name)
    if c is None:
        print("未知实验：%s" % name)
        return
    errs = c.validate()
    if errs:
        print("前置检查未过，不跑：")
        for e in errs:
            print("  - " + e)
        return

    print("=" * 120)
    print("%s  fp=%s  n_ep=%d  种子=%d" % (c.name, c.fingerprint(), c.n_ep, nseed))
    print("  m=%s/%s/%s%s  plast=%s  isolate=%s  task=%s/%dx%d" % (
        c.signal.source, c.signal.shape, c.signal.glob,
        "" if c.signal.gate else "(nogate)",
        "+".join(k for k in ("weight", "topology", "bias") if getattr(c.plast, k)) or "none",
        c.plast.isolate, c.task.kind, c.task.t_ep, c.task.t_act))
    print("=" * 120)

    tasks = [(name, s) for s in range(nseed)]
    with Pool(nproc) as pool:
        res = pool.map(_one, tasks)

    keys = ["succ", "basis", "progress", "out_fire", "out_bias"]
    print("%-5s " % "seed" + " ".join("%-10s" % k for k in keys) + " %s" % "E/rate/deg")
    for r in res:
        print("%-5d " % r["seed"] + " ".join("%-10.3f" % r[k] for k in keys)
              + " %d/%.2f%%/%d" % (r["E_final"], r["rate"] * 100, r["out_deg"]))

    def agg(k):
        return float(np.mean([r[k] for r in res]))

    s1 = np.mean([r["succ_first"] for r in res]) * 100
    s2 = np.mean([r["succ_last"] for r in res]) * 100
    print()
    print("  跨种子: succ %.1f%% -> %.1f%% (%+.1f pp)   全程 %.1f%%" % (
        s1, s2, s2 - s1, agg("succ") * 100))
    print("          basis %.3f   progress %.2f   out_fire %.3f   out_bias %.2f" % (
        agg("basis"), agg("progress"), agg("out_fire"), agg("out_bias")))
    print("          更新次数: w=%.0f  bias=%.0f  topo=%.0f   IO=%d  ok=%s" % (
        agg("n_upd_w"), agg("n_upd_b"), agg("n_upd_top"),
        int(agg("n_io")), all(r["io_ok"] for r in res)))

    for r in res:
        append(r)
    print("  已登记 %d 条 -> data/results.jsonl" % len(res))


def cmd_results():
    summary()


def cmd_align():
    """对齐验证（必需件）：同一配置的重复运行必须给同样的数。

    2026-10-24 把旧脚本逻辑搬进框架时行为没对齐（旧 discrete 36.2% vs 框架 6.9~8.0%），
    而我不知道差在哪里。没有这一步，框架只是个「也能跑」的东西，
    **不能用来复现任何历史结论**。
    """
    rows = load()
    by_fp = {}
    for r in rows:
        by_fp.setdefault(r.get("fingerprint"), []).append(r)
    print("%-14s %-6s %-36s %s" % ("fingerprint", "n", "succ range", "status"))
    print("-" * 110)
    bad = 0
    for fp, rs in by_fp.items():
        if len(rs) < 2:
            continue
        s = [r.get("succ", 0) for r in rs]
        spread = max(s) - min(s)
        ok = spread < 0.20
        if not ok:
            bad += 1
        print("%-14s %-6d %.3f ~ %.3f (spread %.3f)  %s" % (
            fp, len(rs), min(s), max(s), spread, "OK" if ok else "**UNSTABLE**"))
    if bad:
        print("\n%d 个配置的重复运行跨度 > 20%% -- 不是统计波动，是逻辑不一致。" % bad)
    else:
        print("\n(none or all stable)")


def cmd_gaps():
    rows = load()
    done = set(r.get("fingerprint") for r in rows)
    base = experiments.BASE
    print("基准: %s fp=%s  %s" % (
        base.name, base.fingerprint(),
        "已跑" if base.fingerprint() in done else "**未跑**"))
    print()
    print("%-18s %-14s %-8s %s" % ("变体", "fingerprint", "状态", "与基准的差"))
    print("-" * 110)
    bd = base.to_dict()
    for c in experiments.ALL:
        if c.name == base.name:
            continue
        cd = c.to_dict()
        diffs = []
        for k in cd:
            if k in ("name", "note", "seed", "n_ep"):
                continue
            if cd[k] != bd.get(k):
                if isinstance(cd[k], dict):
                    sub = [kk for kk in cd[k] if cd[k][kk] != bd[k].get(kk)]
                    diffs.append("%s{%s}" % (k, ",".join(sub)))
                else:
                    diffs.append(k)
        st = "已跑" if c.fingerprint() in done else "**未跑**"
        print("%-18s %-14s %-8s %s" % (c.name, c.fingerprint(), st, "; ".join(diffs)))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "用法：run.py list|show|check|run|results|gaps")
        print("  list / show <name> / check [name] / run <name> [nseed] [nproc] / results / gaps")
        return
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "show":
        cmd_show(sys.argv[2])
    elif cmd == "check":
        cmd_check(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "run":
        nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        cmd_run(sys.argv[2], nseed, nproc)
    elif cmd == "results":
        cmd_results()
    elif cmd == "gaps":
        cmd_gaps()
    elif cmd == "align":
        cmd_align()
    else:
        print("未知命令：%s" % cmd)


if __name__ == "__main__":
    main()
