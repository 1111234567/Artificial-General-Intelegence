# scripts/run.py — src 框架的命令行入口。
#
# 用法：
#   python scripts/run.py list                      列出所有实验（含指纹与开关）
#   python scripts/run.py show <name>               展开一份配置的全部开关
#   python scripts/run.py check <name>              只跑前置检查（任务可行性等）
#   python scripts/run.py run <name> [nseed] [nproc] 跑并登记
#   python scripts/run.py results                   查结果登记
#   python scripts/run.py gaps                      查对照组差集（哪些还没跑）
#
# 独立原型、历史实现与诊断见 experiments/README.md。
# test/run.py 仅保留旧命令兼容。

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from multiprocessing import Pool

from src import experiments, run, append, summary, load
from src.config import Config

ENV_ROOT_OK = True


def _scan_one(a):
    """模块级（Windows spawn 下局部函数不能 pickle）。"""
    from src import runner
    label, cfg, seed = a
    return (label, runner.run(cfg.with_(seed=seed)))


def cmd_scan(which="homeo", nseed=2, nproc=None):
    """参数扫描（工具）。不登记结果，只看趋势。

    which: homeo = scan_base 的网格（无 E-E，扫 W_I/D_IE/D_EI）
           calib = calib5 的网格（纯随机结构，扫 w_E/抑制比/T_REF）
    """
    from src import scan as scanmod
    from src import runner
    grid = scanmod.grid_homeo() if which == "homeo" else scanmod.grid_calib()
    tasks = [(label, cfg, s) for label, cfg in grid for s in range(nseed)]
    nproc = nproc or min(len(tasks), os.cpu_count() or 4)


    print("=" * 96)
    if which == "homeo":
        print("scan / homeo — 无 E-E 时，什么参数组合让 rate 落在 0.2~0.5%？")
        print("  （旧脚本 scan_base.py 的网格；目标区间下限 0.64%%）")
    else:
        print("scan / calib — 加不应期后，什么 (w_E, 抑制比, T_REF) 标定到 2%？")
        print("  （旧脚本 calib5.py 的网格；T_REF=50 -> 上限 2%% / T_REF=20 -> 上限 5%%）")
    print("  nseed=%d  任务数=%d  并行度=%d" % (nseed, len(tasks), nproc))
    print("=" * 96)
    with Pool(nproc) as pool:
        rs = pool.map(_scan_one, tasks)

    by = {}
    for label, r in rs:
        by.setdefault(label, []).append(r)
    rows = scanmod.summarize([(k, v) for k, v in by.items()])

    if which == "homeo":
        print("%-32s %10s %10s %8s" % ("组合", "E_rate%", "I_rate%", "活跃数"))
        print("-" * 96)
        for r in rows:
            mark = ""
            if 0.15 <= r["rate_e"] <= 0.60:
                mark = "   <-- 合适（低于目标区间）"
            elif r["rate_e"] > 0.64:
                mark = "   (已在目标区间或以上)"
            print("%-32s %10.3f %10.3f %8.0f%s" % (
                r["label"], r["rate_e"], r["rate_i"], r["n_active"], mark))
    else:
        print("%-30s %10s %10s  %s" % ("组合", "rate%", "尾部 std", "状态"))
        print("-" * 96)
        for r in rows:
            m = ""
            if 1.0 <= r["rate_e"] <= 4.0 and r["std"] < 1.0:
                m = "** 目标区 **"
            elif r["rate_e"] < 0.5:
                m = "死亡"
            elif r["rate_e"] > 50:
                m = "饱和"
            print("%-30s %10.2f %10.3f  %s" % (r["label"], r["rate_e"], r["std"], m))
    print("-" * 96)
    return 0


def cmd_bench(steps_cpu=2000, steps_gpu=500):
    """模拟器成本基准（工具）。

    ⚠️ 本命令**不用框架的 Network**：它测的就是「实现形态本身」
    （稠密 vs 稀疏、CPU vs GPU、前向 vs 迹 vs 可塑），所以必须旁开一份最简实现。
    数据基线：experiments/legacy/baseline-2026-10-23-simulator.md。
    """
    import time
    t0 = time.perf_counter()
    TAU = 20.0
    DECAY = float(np.exp(-1.0 / TAU))
    V_TH = 1.0
    TAU_E = 20.0
    DECAY_E = float(np.exp(-1.0 / TAU_E))
    LR = 1e-3

    def make(N, density, seed=0):
        rng = np.random.default_rng(seed)
        ne = max(1, int(N * N * density))
        pre = rng.integers(0, N, ne).astype(np.int64)
        post = rng.integers(0, N, ne).astype(np.int64)
        sc = 1.0 / np.sqrt(max(1.0, density * N))
        w = (rng.standard_normal(ne) * sc).astype(np.float32)
        return pre, post, w, ne

    def mk_input(steps, N, rate=0.02, seed=1):
        rng = np.random.default_rng(seed)
        return (rng.random((steps, N)) < rate).astype(np.float32)

    def fwd_sparse(N, pre, post, w, X, steps, plastic=False):
        v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
        tr = np.zeros(pre.size, np.float32)
        ww = w.copy()
        for t in range(steps):
            I = np.bincount(post, weights=ww * s[pre], minlength=N).astype(np.float32)
            v = v * DECAY + I + X[t]
            s = (v > V_TH).astype(np.float32)
            v = np.where(s > 0, 0.0, v)
            if plastic:
                tr *= DECAY_E
                tr += s[pre] * s[post]
                ww += LR * tr
                np.clip(ww, -2.0, 2.0, out=ww)

    def fwd_dense(N, W, X, steps):
        v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
        for t in range(steps):
            v = v * DECAY + (W @ s) + X[t]
            s = (v > V_TH).astype(np.float32)
            v = np.where(s > 0, 0.0, v)

    print("=" * 96)
    print("bench — LIF 模拟器成本（CPU 稀疏 vs 稠密 × 前向 vs 可塑）")
    print("  CPU 每档 %d 步；GPU 部分需要 torch+cuda，不可用则跳过" % steps_cpu)
    print("=" * 96)
    print("%-8s %8s %12s %12s %12s" % ("N", "密度", "稀疏 fwd", "稀疏 +可塑", "稠密 fwd"))
    print("-" * 96)
    for N in (1000, 2000):
        for dens in (0.005, 0.01, 0.05, 0.10):
            pre, post, w, ne = make(N, dens)
            X = mk_input(steps_cpu, N)
            a = time.perf_counter(); fwd_sparse(N, pre, post, w, X, steps_cpu); ta = time.perf_counter() - a
            b = time.perf_counter(); fwd_sparse(N, pre, post, w, X, steps_cpu, True); tb = time.perf_counter() - b
            W = np.zeros((N, N), np.float32)
            np.add.at(W, (post, pre), w)
            c = time.perf_counter(); fwd_dense(N, W, X, steps_cpu); tc = time.perf_counter() - c
            print("%-8d %8.3f %10.3fms %10.3fms %10.3fms   (边 %d)" % (
                N, dens, ta / steps_cpu * 1e3, tb / steps_cpu * 1e3,
                tc / steps_cpu * 1e3, ne))
    print("-" * 96)
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU（batch 稠密）：")
            N = 1000
            pre, post, w, ne = make(N, 0.01)
            W = np.zeros((N, N), np.float32)
            np.add.at(W, (post, pre), w)
            X = mk_input(steps_gpu, N)
            dev = torch.device("cuda")
            Wt = torch.from_numpy(W).to(dev)
            Xt = torch.from_numpy(X).to(dev)
            for B in (1, 16, 64, 256):
                v = torch.zeros(B, N, device=dev)
                s = torch.zeros(B, N, device=dev)
                t1 = time.perf_counter()
                for t in range(steps_gpu):
                    v = v * DECAY + (s @ Wt.T) + Xt[t]
                    s = (v > V_TH).float()
                    v = torch.where(s > 0, torch.zeros_like(v), v)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t1) / steps_gpu * 1e6
                print("  B=%-4d  %8.3f us/步" % (B, dt))
        else:
            print("GPU：torch 在但 cuda 不可用，跳过")
    except ImportError:
        print("GPU：未安装 torch，跳过")
    print("-" * 96)
    print("总耗时 %.1fs" % (time.perf_counter() - t0))
    return 0


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
    """前置检查的**通过**说明——按流程/环境分派（否则不适用可达性的流程会打错口径）。"""
    if c.flow == "talk":
        return "不适用（talk：无目标可达性）；提问@%d 回答@%d（保持 %d 步 / 抢答窗 %d 步）" % (
            c.task.query_at, c.task.answer_from, c.task.hold_steps, c.task.early_steps)
    if c.flow == "teach":
        return "不适用（teach：固定 trial + 末窗 argmax）；trial=%d 读出窗=%d" % (
            c.task.trial_steps, c.task.read_win)
    if c.flow == "cond":
        return "不适用（cond：变长呈现）；保持期 Gamma %.0f->%.0f（CV %.1f）" % (
            c.task.d_mean_start, c.task.d_mean_end, c.task.d_cv)
    if c.flow == "homeo":
        return "不适用（homeo：单条时间轴，按放电率双向调结构）"
    if c.flow == "hebb":
        return "不适用（hebb：结构冻结 + 三段式；判据是撤输入后的自持活动）"
    if c.flow == "world":
        return "不适用（world：单条时间轴，无目标与成败）"
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
    """按历史指纹汇总成功率跨度，供人工进一步核验。

    指纹不区分 seed、episode 预算等字段；这里的跨行跨度可能混合这些差异，
    不能用来证明逐步对齐、确定性或逻辑不一致。逐层比对见 align_layers.py。
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
            fp, len(rs), min(s), max(s), spread, "范围提示" if ok else "跨度较大，待核验"))
    if bad:
        print("\n%d 组历史指纹的跨度 > 20%%；请先核对 seed、预算和运行条件，再分析差异。" % bad)
    else:
        print("\n没有超过提示阈值的分组；这不代表通过对齐验证。")


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
        print("  align / scan [homeo|calib] [nseed] [nproc] / bench")
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
    elif cmd == "scan":
        which = sys.argv[2] if len(sys.argv) > 2 else "homeo"
        nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        cmd_scan(which, nseed, nproc)
    elif cmd == "bench":
        cmd_bench()
    else:
        print("未知命令：%s" % cmd)


if __name__ == "__main__":
    main()
