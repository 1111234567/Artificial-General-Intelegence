# scripts/align_layers.py — 框架 vs 旧 grow_agent：逐层对齐探针
#
# 起因：agi.md 下一步第 1 项。旧 discrete 36.2% vs 框架 base 6.9~8.0%（差 5 倍，未定位），
# **对齐之前框架不能复现任何历史结论**，所以第 2~8 项全卡在这里。
#
# 做法：四层逐层锁死，每层通过之前不碰下一层。
#   construct  初始结构：key / pre / post / w / age 逐位
#   forward    同一结构 + 同一输入串，N 步的 state / v / trace / theta 逐位
#   m          同一动作序列的 mv / mg / m_ep
#   update     同一梯度的 w / key
#
# 用法：
#   python scripts/align_layers.py construct [nseed]
#   python scripts/align_layers.py forward   [nseed] [nsteps]
#   python scripts/align_layers.py run       [nseed] [n_ep]
#   python scripts/align_layers.py noop      [nseed] [n_ep]
#   python scripts/align_layers.py seq       [seed]  [n_ep]
#   python scripts/align_layers.py contrib   [nseed] [n_ep]
#   python scripts/align_layers.py modes     [seed]  [n_ep]
#
# 判据：全部 seed、全部字段逐位相等，才算这一层通过。
# 不等就停下来查这一层——不要往下走（下面各层的差异会被这一层放大成噪声）。

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

import numpy as np

from src.config import Config
from src.core import Network as NewNetwork

import net as old_net
import grow_agent as old_ga


def show(tag, a, b):
    """打印一项比对结果。返回是否相等。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print("  %-10s MISMATCH  shape %s vs %s" % (tag, a.shape, b.shape))
        return False
    if a.size == 0:
        print("  %-10s OK（空）" % tag)
        return True
    diff = a != b
    nbad = int(diff.sum())
    if nbad == 0:
        print("  %-10s OK       %d 项逐位相同" % (tag, a.size))
        return True
    idx = np.nonzero(diff)[0][:5]
    ex = ", ".join("#%d %r!=%r" % (i, a.flat[i], b.flat[i]) for i in idx)
    print("  %-10s **MISMATCH** %d/%d 不同  e.g. %s" % (tag, nbad, a.size, ex))
    return False


def layer_construct(seeds):
    print("=" * 104)
    print("层 1 / construct — 初始结构（同 seed，旧 grow_agent._init_net vs 框架 Network.__init__）")
    print("=" * 104)

    cfg = Config()
    n_ok = 0
    for seed in seeds:
        print("\n--- seed %d ---" % seed)

        rng_old = np.random.default_rng(seed)
        p0, p1, w0, k0, a0 = old_ga._init_net(rng_old)

        rng_new = np.random.default_rng(seed)
        netobj = NewNetwork(cfg.net, rng_new, np.asarray(cfg.readout.out_idx))

        all_eq = True
        # 常数先对一遍：常数不同的话，后面所有层都不可比
        for tag, va, vb in (
            ("N", old_net.N, netobj.N),
            ("N_E", old_net.N_E, netobj.N_E),
            ("N_IN", old_net.N_IN, netobj.N_IN),
            ("E0_INIT", old_net.E0_INIT, cfg.net.e0_init),
        ):
            same = (va == vb)
            print("  %-10s %s   %s vs %s" % (tag, "OK    " if same else "**DIFF**", va, vb))
            all_eq &= same

        print("  %-10s %s   %d vs %d" % ("边数", "OK    " if k0.size == netobj.key.size else "**DIFF**",
                                        k0.size, netobj.key.size))
        all_eq &= (k0.size == netobj.key.size)

        all_eq &= show("key", k0, netobj.key)
        all_eq &= show("pre", p0, netobj.pre)
        all_eq &= show("post", p1, netobj.post)
        all_eq &= show("w", w0, netobj.w)
        all_eq &= show("age", a0, netobj.age)

        # RNG 消耗一致性：构造完之后各 draw 3 个，相同则说明消耗量一致
        tail_old = rng_old.random(3)
        tail_new = rng_new.random(3)
        all_eq &= show("rng.tail", tail_old, tail_new)

        print("  => %s" % ("构造层通过" if all_eq else "**构造层不通过**"))
        n_ok += int(all_eq)

    print("\n" + "=" * 104)
    print("结论：%d/%d 个 seed 的初始结构逐位相同" % (n_ok, len(seeds)))
    print("=" * 104)
    return n_ok == len(seeds)


def _load_old_step():
    """从 grow_agent.py **原文**抽取 net_step（不手抄，防止抄错）。

    只做一处语法替换：`nonlocal` -> `global`（因为要 exec 到模块级 dict）。
    逻辑一行未改——这是本探针可信的前提。
    """
    p = os.path.join(ROOT, "test", "grow_agent.py")
    with open(p, encoding="utf-8") as f:
        src = f.read()
    i = src.index("    def net_step(")
    j = src.index("    def encode(", i)
    body = src[i:j]
    body = body.replace("nonlocal v, s, ref, trace, theta, ema_rate",
                        "global v, s, ref, trace, theta, ema_rate")
    body = "\n".join(ln[4:] if ln.startswith("    ") else ln
                     for ln in body.split("\n"))
    # 基底用 grow_agent 自己的模块级命名空间——net_step 里引用的所有常量（TRACE_DECAY /
    # SIDE_INH / T_REF_OUT / OUT_IDX …）都在里面，漏一个就会 NameError。
    # 为什么不只挑用到的：挑漏过一次（TRACE_DECAY），而漏报的形态是运行时才显形。
    G = {k: v for k, v in vars(old_ga).items() if not k.startswith("_")}
    G["np"] = np
    exec(compile(body, "<grow_agent.net_step>", "exec"), G)
    return G["net_step"], G


def layer_forward(seeds, nsteps=200):
    """层 2：同一结构 + 同一输入串，逐位比对前向。

    前向层不过，说明 core.step 与旧 net_step 不等价——后面所有层的差异都会被它污染。
    """
    print("=" * 104)
    print("层 2 / forward — 前向（原文抽取的 grow_agent.net_step vs 框架 Network.step）")
    print("=" * 104)

    net_step_old, G = _load_old_step()
    cfg = Config()
    N = cfg.net.N
    n_in = cfg.net.n_in
    n_ok = 0

    for seed in seeds:
        print("\n--- seed %d（%d 步 inh=True + 50 步 inh=False）---" % (seed, nsteps))

        rng = np.random.default_rng(seed)
        netobj = NewNetwork(cfg.net, rng, np.asarray(cfg.readout.out_idx))

        # 旧侧：结构用同一份（构造层已证明逐位相同），状态全部归零
        G["pre"] = netobj.pre.copy()
        G["post"] = netobj.post.copy()
        G["w"] = netobj.w.copy()
        G["v"] = np.zeros(N, np.float32)
        G["s"] = np.zeros(N, np.float32)
        G["ref"] = np.zeros(N, np.int32)
        G["trace"] = np.zeros(N, np.float32)
        G["theta"] = np.zeros(N, np.float32)
        G["bias"] = np.zeros(N, np.float32)
        G["ema_rate"] = 0.02

        rng_in = np.random.default_rng(1000 + seed)
        all_eq = True
        first_bad = None

        for t in range(nsteps + 50):
            inh = (t < nsteps)
            xin = np.zeros(N, np.float32)
            xin[:n_in] = (rng_in.random(n_in) < 0.02).astype(np.float32)

            n1 = net_step_old(xin, inh)
            n2 = netobj.step(xin, inh)

            checks = (("nsp", n1, n2), ("s", G["s"], netobj.state),
                      ("v", G["v"], netobj.v), ("ref", G["ref"], netobj.ref),
                      ("trace", G["trace"], netobj.trace),
                      ("theta", G["theta"], netobj.theta),
                      ("ema", G["ema_rate"], netobj.ema_rate))
            for tag, a, b in checks:
                a = np.asarray(a); b = np.asarray(b)
                if a.shape != b.shape or (a != b).any():
                    all_eq = False
                    if first_bad is None:
                        first_bad = (t, tag, a, b)
                    break
            if not all_eq:
                break

        if all_eq:
            print("  %d 步（含 50 步无侧抑制）全部逐位相同 -> 前向层通过" % (nsteps + 50))
        else:
            t, tag, a, b = first_bad
            a = np.asarray(a); b = np.asarray(b)
            if a.shape != b.shape:
                print("  **第 %d 步 %s 形状不同：%s vs %s**" % (t, tag, a.shape, b.shape))
            else:
                idx = np.nonzero(a != b)[0][:5]
                ex = ", ".join("#%d %r!=%r" % (i, a.flat[i], b.flat[i]) for i in idx)
                print("  **第 %d 步首次出现差异：%s  %d 处  e.g. %s**" % (
                    t, tag, int((a != b).sum()), ex))
        n_ok += int(all_eq)

    print("\n" + "=" * 104)
    print("结论：%d/%d 个 seed 的前向逐位相同" % (n_ok, len(seeds)))
    print("=" * 104)
    return n_ok == len(seeds)


def layer_run(seeds, n_ep=50, cfg_name="base"):
    """层 3：编排层端到端对拍（同一 seed、同一 episode 数，两边各跑一次）。

    本体层已逐位相同，前向层已逐位相同 -> 这里剩下的差异全部来自**编排**：
      m 的组成 / 更新时机 / 动作集 / 读出决胜 / 输入缓冲。
    本层的输出不是「通过/不通过」，是**差异量化**——这些数字就是待消除的差集。

    cfg_name="aligned" 时对的是「全部对齐开关打开」的配置（验证差集是否完整）。
    """
    from src import runner as new_runner
    from src import experiments
    from src.core import Network as _Net

    # 框架侧没有 born 计数（apply_topology 只返回删除数）——包装 _grow 计数。
    _born = {"n": 0}
    _orig_grow = _Net._grow

    def _grow_counted(self, npw, nqw, nw, _orig=_orig_grow, _b=_born):
        _b["n"] += int(npw.size)
        return _orig(self, npw, nqw, nw)

    _Net._grow = _grow_counted

    cfg0 = experiments.BY_NAME[cfg_name]
    print("=" * 104)
    print("层 3 / run — 编排端到端对拍（旧 grow_agent mode=discrete  vs  框架 %s）" % cfg0.name)
    print("  n_ep=%d  同一 seed、同一任务、同一环境根" % n_ep)
    print("=" * 104)

    # (显示名, 旧字段, 新字段, 是否百分比)
    FIELDS = [
        ("succ", "succ", "succ", True),
        ("succ_first", "succ_first", "succ_first", True),
        ("succ_last", "succ_last", "succ_last", True),
        ("progress", "progress", "progress", False),
        ("basis", "basis", "basis", False),
        ("out_fire", "out_rate", "out_fire", False),
        ("E", "E", "E_final", False),
        ("n_inh", "n_inh", "n_inh", False),
        ("rate%", "rate", "rate", True),
        ("out_deg", "out_deg", "out_deg", False),
        ("out_bias", "out_bias", "out_bias", False),
    ]

    for seed in seeds:
        print("\n--- seed %d ---" % seed)
        r_old = old_ga.run_one((seed, "discrete", n_ep))
        cfg = cfg0.with_(seed=seed, n_ep=n_ep)
        _born["n"] = 0
        r_new = new_runner.run(cfg)

        print("  %-12s %12s %12s %12s" % ("field", "old", "new", "delta"))
        for tag, ko, kn, pct in FIELDS:
            vo = r_old.get(ko, float("nan"))
            vn = r_new.get(kn, float("nan"))
            if pct:
                vo *= 100.0
                vn *= 100.0
                fmt = "%11.2f%% %11.2f%% %+11.2f"
            else:
                fmt = "%12.4f %12.4f %+12.4f"
            print("  %-12s " % tag + fmt % (vo, vn, vn - vo))

        print("  [旧] 结构动态: born %d / drop %d / m_mean %.4f / detour %.3f"
              % (r_old.get("born", 0), r_old.get("drop", 0),
                 r_old.get("m_mean", float("nan")), r_old.get("detour", float("nan"))))
        print("  [新] 结构动态: born %d / drop %d" % (_born["n"], r_new.get("n_upd_top", 0)))
        print("  [新] 更新次数: w=%d  bias=%d   (旧无计数)"
              % (r_new.get("n_upd_w", 0), r_new.get("n_upd_b", 0)))

    print("\n" + "=" * 104)
    print("本层不判通过/不通过——它输出的是待消除的差集。下一层：逐项打开旧行为，量各自贡献。")
    print("=" * 104)
    return True


def layer_noop(seeds, n_ep=20):
    """层 3b：noop 分支的实测确认（静态比对漏掉的一条）。

    静态比对时我以为动作集只是「少两个动作」。实际不是：
    框架里 `a_idx = win_i if win_i < n_actions else -1`（n_actions=2）
    -> win_i ∈ {2,3} 变成 **noop**；而旧代码 a_idx 永远是 0..3，从不 noop。

    本层量三件事：① noop 占多少；② noop 时 `m_glob` 是不是非零（= 在默默削弱全局权重）；
    ③ 对应的成功率。
    """
    from src import runner as new_runner
    from src import experiments
    from src import signal as sigmod

    print("=" * 104)
    print("层 3b / noop — 框架独有的一条分支（旧代码不存在 noop）")
    print("  n_ep=%d" % n_ep)
    print("=" * 104)

    for seed in seeds:
        st = {"noop": 0, "act": 0, "mg_noop": [], "mg_act": []}
        orig = sigmod.Signal.on_action

        def patched(self, a_idx, moved, want, _orig=orig, _st=st):
            mv, mg = _orig(self, a_idx, moved, want)
            if a_idx < 0:
                _st["noop"] += 1
                _st["mg_noop"].append(0.0 if mg is None else float(mg))
            else:
                _st["act"] += 1
                _st["mg_act"].append(0.0 if mg is None else float(mg))
            return mv, mg

        sigmod.Signal.on_action = patched
        try:
            cfg = experiments.BASE.with_(seed=seed, n_ep=n_ep)
            r = new_runner.run(cfg)
        finally:
            sigmod.Signal.on_action = orig

        n_noop = st["noop"]
        n_act = st["act"]
        tot = max(n_noop + n_act, 1)
        mg_n = np.asarray(st["mg_noop"], np.float64)
        mg_a = np.asarray(st["mg_act"], np.float64)
        nz = float((np.abs(mg_n) > 1e-9).mean()) if mg_n.size else float("nan")
        print("\n--- seed %d  succ %.1f%% ---" % (seed, r.get("succ", 0) * 100))
        print("  动作窗口：有动作 %d / noop %d（%.1f%%）" % (n_act, n_noop, 100.0 * n_noop / tot))
        if mg_n.size:
            print("  noop 时 m_glob：均值 %+.4f  中位 %+.4f  非零占比 %.1f%%"
                  % (mg_n.mean(), float(np.median(mg_n)), 100.0 * nz))
        if mg_a.size:
            print("  有动作时 m_glob：均值 %+.4f  非零占比 %.1f%%"
                  % (mg_a.mean(), 100.0 * float((np.abs(mg_a) > 1e-9).mean())))
        print("  权重更新次数 %d（= 动作窗口总数 %d，因为 mg 几乎从不为 None）"
              % (r.get("n_upd_w", 0), tot))

    print("\n" + "=" * 104)
    print("判据：若 noop 占比显著且 noop 的 m_glob 系统性非零 -> 框架在 «什么都没做» 时也在改权重。")
    print("=" * 104)
    return True


def _up_locals(name, depth=10):
    """从当前帧往上找第一个「numpy 数组型」的局部变量 name。

    为什么必须限定类型：拦截函数的参数自己就叫 v（state 的值，int），
    不限定会拿到它（第一版就踩了这个）。
    """
    f = sys._getframe(2)
    for _ in range(depth):
        if f is None:
            return None
        if name in f.f_locals:
            val = f.f_locals[name]
            if isinstance(val, np.ndarray):
                return val
        f = f.f_back
    return None


def _seq_of(mode, cfg_name, seed, n_ep, diag=False):
    """跑一边旧一边新，返回 (seq_old, seq_new, dia_old, dia_new, r_old, r_new)。

    diag=True 时额外记录每个动作窗口时刻的 (max v[out], mean bias[out])。
    """
    from src import runner as new_runner
    from src import experiments
    from src.env import LocalFilesEnv
    from src.core import Network as _NetCls

    _last = {"net": None}
    _orig_step = _NetCls.step

    def _step_rec(self, xin, apply_side_inh=True, _s=_orig_step, _l=_last):
        _l["net"] = self
        return _s(self, xin, apply_side_inh)

    _NetCls.step = _step_rec

    seq_old, seq_new = [], []
    dia_old, dia_new = [], []
    ow = old_ga.env_write
    nw = LocalFilesEnv._write

    def pw(d, name, v, _ow=ow):
        if name == "state.txt":
            seq_old.append(int(v))
            if diag:
                vv = _up_locals("v")
                bb = _up_locals("bias")
                oi = old_ga.OUT_IDX
                dia_old.append((
                    float(np.max(vv[oi])) if vv is not None else float("nan"),
                    float(np.mean(bb[oi])) if bb is not None else float("nan")))
        return _ow(d, name, v)

    def pn(self, name, v, _nw=nw, _l=_last):
        if name == "state.txt":
            seq_new.append(int(v))
            if diag:
                nt = _l["net"]
                if nt is not None:
                    dia_new.append((float(np.max(nt.v[nt.out_idx])),
                                    float(np.mean(nt.bias[nt.out_idx]))))
                else:
                    dia_new.append((float("nan"), float("nan")))
        return _nw(self, name, v)

    old_ga.env_write = pw
    LocalFilesEnv._write = pn
    try:
        r_old = old_ga.run_one((seed, mode, n_ep))
        cfg = experiments.BY_NAME[cfg_name].with_(seed=seed, n_ep=n_ep)
        r_new = new_runner.run(cfg)
    finally:
        old_ga.env_write = ow
        LocalFilesEnv._write = nw
        _NetCls.step = _orig_step

    return seq_old, seq_new, dia_old, dia_new, r_old, r_new


def _pool_map(fn, args, nproc=None):
    """统一的多进程入口。

    2026-10-24 教训（AGENTS「并行度看进程数，不看核数」）：
    批量跑的并行度 = **任务数**，不是按组串行。8 个移植项串行 = 只占 1 个核（CPU 12.5%）。
    """
    if nproc is None:
        nproc = min(len(args), os.cpu_count() or 1)
    nproc = max(1, int(nproc))
    if nproc == 1 or len(args) <= 1:
        return [fn(a) for a in args]
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        return pool.map(fn, args)


def _verify_one(arg):
    mode, cfg_name, seed, n_ep = arg
    so, sn, _do, _dn, r_old, r_new = _seq_of(mode, cfg_name, seed, n_ep)
    n = min(len(so), len(sn))
    first = next((i for i in range(n) if so[i] != sn[i]), -1)
    return dict(mode=mode, cfg=cfg_name, n_old=len(so), n_new=len(sn),
                succ_old=r_old.get("succ", 0), succ_new=r_new.get("succ", 0),
                first=first)


def layer_verify(seed=0, n_ep=60, nproc=None):
    """层 5 / verify — 全部移植项的动作序列验证（移植的唯一判据）。"""
    pairs = [
        ("discrete", "ga_discrete"),
        ("discrete_none", "ga_none"),
        ("discrete_bias", "ga_bias"),
        ("discrete_w", "ga_w"),
        ("discrete_topo", "ga_topo"),
        ("frozen", "ga_frozen"),
        ("base", "ga_base"),
        ("recall", "ga_recall"),
        ("random", "ga_random"),
    ]
    print("=" * 104)
    print("层 5 / verify — 移植验证（判据 = 动作序列逐项相同）")
    print("  seed=%d  n_ep=%d  并行=%s" % (seed, n_ep, nproc or "cpu_count"))
    print("=" * 104)

    args = [(m, c, seed, n_ep) for m, c in pairs]
    rs = _pool_map(_verify_one, args, nproc)

    print("%-18s %-14s %9s %9s %9s %9s %s" % (
        "old mode", "new config", "写入数", "old succ", "new succ", "首次分叉", "判定"))
    print("-" * 104)
    n_ok = 0
    for r in rs:
        if r["first"] < 0 and r["n_old"] == r["n_new"]:
            verdict, fd = "OK", "—"
            n_ok += 1
        elif r["first"] < 0:
            verdict, fd = "**长度不同**", str(min(r["n_old"], r["n_new"]))
        else:
            verdict, fd = "**MISMATCH**", str(r["first"])
        print("%-18s %-14s %4d/%-4d %8.1f%% %8.1f%% %9s %s" % (
            r["mode"], r["cfg"], r["n_old"], r["n_new"],
            r["succ_old"] * 100, r["succ_new"] * 100, fd, verdict))

    print("-" * 104)
    print("结论：%d/%d 个移植项逐项相同" % (n_ok, len(pairs)))
    print("=" * 104)
    return n_ok == len(pairs)


def _goal_one(arg):
    """层 6：grow_goal 移植对齐——逐字段比较（两边统计量必须逐位相同）。

    grow_goal 没有环境文件可拦，所以判据是**结构 + 统计量的逐位一致**：
    16 个字段全等；只要世界轨迹或更新时机差一步，这些量就不可能全等。
    """
    mode, cfg_name, seed, n_ep = arg
    import grow_goal as old_g
    from src import runner as new_runner
    from src import experiments

    r_old = old_g.run_one((seed, mode, n_ep))
    r_new = new_runner.run(experiments.BY_NAME[cfg_name].with_(seed=seed, n_ep=n_ep))

    FIELDS = [
        ("E", "E", "E_final"),
        ("born", "born", "born"),
        ("drop", "drop", "n_upd_top"),
        ("rate", "rate", "rate"),
        ("succ_first", "succ_first", "succ_first"),
        ("succ_last", "succ_last", "succ_last"),
        ("step_first", "step_first", "step_first"),
        ("step_last", "step_last", "step_last"),
        ("align", "align", "align"),
        ("a_mean", "a_mean", "a_mean"),
        ("a_switch", "a_switch", "a_switch"),
        ("m_steady", "m_steady", "m_steady"),
        ("m_value", "m_value", "m_value"),
        ("m_mean", "m_mean", "m_mean"),
        ("theta", "theta_mean", "theta_mean"),
        ("out_deg", "out_deg", "out_deg"),
    ]
    bad = []
    for tag, ko, kn in FIELDS:
        vo, vn = r_old.get(ko), r_new.get(kn)
        if vo is None or vn is None:
            bad.append((tag, vo, vn))
        elif float(vo) != float(vn):
            bad.append((tag, vo, vn))
    return dict(mode=mode, cfg=cfg_name, bad=bad,
                succ_old=r_old.get("succ_last", 0.0),
                succ_new=r_new.get("succ_last", 0.0))


def layer_goal(seed=0, n_ep=60, nproc=None):
    """层 6 / goal — grow_goal 的移植对齐。"""
    pairs = [
        ("A", "ga_goal_A"),
        ("A_frozen", "ga_goal_frozen"),
        ("A_fixed", "ga_goal_fixed"),
        ("A_probe", "ga_goal_probe"),
        ("T_seq", "ga_goal_t_seq"),
        ("T_fresh", "ga_goal_t_fresh"),
    ]
    print("=" * 104)
    print("层 6 / goal — grow_goal 移植验证（判据 = 16 个字段逐位相同）")
    print("  seed=%d  n_ep=%d" % (seed, n_ep))
    print("=" * 104)
    print("%-12s %-18s %14s %14s  %s" % (
        "old mode", "new config", "succ_last 旧", "succ_last 新", "判定"))
    print("-" * 104)

    args = [(m, c, seed, n_ep) for m, c in pairs]
    rs = _pool_map(_goal_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（16/16）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join("%s(%r≠%r)" % b for b in r["bad"][:3]))
        print("%-12s %-18s %13.1f%% %13.1f%%  %s" % (
            r["mode"], r["cfg"], r["succ_old"] * 100, r["succ_new"] * 100, verdict))

    print("-" * 104)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(pairs)))
    print("=" * 104)
    return n_ok == len(pairs)


def _world_one(arg):
    """层 7：grow_world 移植对齐——逐字段比较（结构 + 动力学统计量）。"""
    mode, cfg_name, seed, steps = arg
    import grow_world as old_w
    from src import runner as new_runner
    from src import experiments

    r_old = old_w.run_one((seed, mode, steps))
    r_new = new_runner.run(experiments.BY_NAME[cfg_name].with_(seed=seed, steps=steps))

    # (旧名, 新名)
    FIELDS = [
        ("E", "E_final"),
        ("born", "born"),
        ("drop", "n_upd_top"),
        ("rate", "rate"),
        ("rate_std", "rate_std"),
        ("deg_e", "deg_e"),
        ("deg_i", "deg_i"),
        ("w_abs", "w_abs"),
        ("n_inh", "n_inh"),
        ("tov_last", "tov_last"),
        ("a_std", "a_std"),
        ("a_sat", "a_sat"),
        ("a_obs_corr", "a_obs_corr"),
        ("out_deg0", "out_deg0"),
        ("out_deg1", "out_deg1"),
        ("out_r0", "out_r0"),
        ("out_r1", "out_r1"),
        ("m_mean", "m_mean"),
        ("elig_mean", "elig_mean"),
    ]
    bad = []
    for ko, kn in FIELDS:
        vo, vn = r_old.get(ko), r_new.get(kn)
        if vo is None or vn is None:
            bad.append((ko, vo, vn))
            continue
        try:
            if float(vo) != float(vn):
                bad.append((ko, vo, vn))
        except (TypeError, ValueError):
            if vo != vn:
                bad.append((ko, vo, vn))
    return dict(mode=mode, cfg=cfg_name, bad=bad,
                corr_old=r_old.get("a_obs_corr", 0.0),
                corr_new=r_new.get("a_obs_corr", 0.0),
                rate_old=r_old.get("rate", 0.0), rate_new=r_new.get("rate", 0.0))


def layer_world(seed=0, steps=5000, nproc=None):
    """层 7 / world — grow_world 的移植对齐。"""
    pairs = [
        ("closed", "ga_world_closed"),
        ("open", "ga_world_open"),
        ("randact", "ga_world_randact"),
        ("noise_in", "ga_world_noise_in"),
        ("frozen", "ga_world_frozen"),
    ]
    print("=" * 108)
    print("层 7 / world — grow_world 移植验证（判据 = 19 个字段逐位相同）")
    print("  seed=%d  steps=%d" % (seed, steps))
    print("=" * 108)
    print("%-10s %-18s %9s %9s %9s %9s  %s" % (
        "old mode", "new config", "rate 旧", "rate 新", "a-o 旧", "a-o 新", "判定"))
    print("-" * 108)

    args = [(m, c, seed, steps) for m, c in pairs]
    rs = _pool_map(_world_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（19/19）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join("%s(%r≠%r)" % b for b in r["bad"][:2]))
        print("%-10s %-18s %8.3f%% %8.3f%% %9.4f %9.4f  %s" % (
            r["mode"], r["cfg"], r["rate_old"] * 100, r["rate_new"] * 100,
            r["corr_old"], r["corr_new"], verdict))

    print("-" * 108)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(pairs)))
    print("=" * 108)
    return n_ok == len(pairs)


def layer_seq(seed, n_ep=2, diag=False, mode="discrete", cfg_name="ga_discrete"):
    """层 3c：动作序列逐项对拍——找「第几个动作开始分叉」。

    拦截两边的「写 state.txt」：旧代码走模块级 env_write，框架走 LocalFilesEnv._write。
    写进去的值序列就是动作序列。
    这是移植的**判据**：旧 mode ↔ 新 config 的动作序列必须逐项相同。
    """
    from src import runner as new_runner
    from src import experiments
    from src.env import LocalFilesEnv
    from src.core import Network as _NetCls

    # 拿框架侧当前实例：帧栈查找不可靠（不同调用路径深度不同）——改成在 step 里存引用。
    _last = {"net": None}
    _orig_step = _NetCls.step

    def _step_rec(self, xin, apply_side_inh=True, _s=_orig_step, _l=_last):
        _l["net"] = self
        return _s(self, xin, apply_side_inh)

    _NetCls.step = _step_rec

    print("=" * 104)
    print("层 3c / seq — 动作序列对拍（旧 mode=%s  vs  框架 %s）" % (mode, cfg_name))
    print("  seed=%d  n_ep=%d" % (seed, n_ep))
    print("=" * 104)

    seq_old, seq_new = [], []
    dia_old, dia_new = [], []
    ow = old_ga.env_write
    nw = LocalFilesEnv._write

    def pw(d, name, v, _ow=ow):
        if name == "state.txt":
            seq_old.append(int(v))
            if diag:
                vv = _up_locals("v")
                bb = _up_locals("bias")
                oi = old_ga.OUT_IDX
                dia_old.append((
                    float(np.max(vv[oi])) if vv is not None else float("nan"),
                    float(np.mean(bb[oi])) if bb is not None else float("nan")))
        return _ow(d, name, v)

    def pn(self, name, v, _nw=nw, _l=_last):
        if name == "state.txt":
            seq_new.append(int(v))
            if diag:
                nt = _l["net"]
                if nt is not None:
                    dia_new.append((float(np.max(nt.v[nt.out_idx])),
                                    float(np.mean(nt.bias[nt.out_idx]))))
                else:
                    dia_new.append((float("nan"), float("nan")))
        return _nw(self, name, v)

    old_ga.env_write = pw
    LocalFilesEnv._write = pn
    try:
        r_old = old_ga.run_one((seed, mode, n_ep))
        cfg = experiments.BY_NAME[cfg_name].with_(seed=seed, n_ep=n_ep)
        r_new = new_runner.run(cfg)
    finally:
        old_ga.env_write = ow
        LocalFilesEnv._write = nw
        _NetCls.step = _orig_step

    print("  旧：%d 次 state 写入（含初始化/set_state），succ %.1f%%"
          % (len(seq_old), r_old.get("succ", 0) * 100))
    print("  新：%d 次，succ %.1f%%" % (len(seq_new), r_new.get("succ", 0) * 100))

    n = min(len(seq_old), len(seq_new))
    first = -1
    for i in range(n):
        if seq_old[i] != seq_new[i]:
            first = i
            break
    if first < 0 and len(seq_old) == len(seq_new):
        print("\n  **两个动作序列逐项相同**（长度 %d）——本期编排已完全一致" % n)
    else:
        if first < 0:
            first = n
        print("\n  **首次分叉：第 %d 个 state 写入**" % first)
        lo = max(0, first - 6)
        hi = min(n, first + 8)
        print("  %-6s %-8s %-8s" % ("idx", "old", "new"))
        for i in range(lo, hi):
            mark = "  <<<" if i == first else ""
            extra = ""
            if diag and i < len(dia_old) and i < len(dia_new):
                extra = "   vmax %.4f/%.4f  bias %.4f/%.4f" % (
                    dia_old[i][0], dia_new[i][0], dia_old[i][1], dia_new[i][1])
            print("  %-6d %-8d %-8d%s%s" % (i, seq_old[i], seq_new[i], mark, extra))

    print("\n" + "=" * 104)
    print("判据：首次分叉前完全一致 -> 差异只发生在分叉之后；分叉点可用于定位机制。")
    print("=" * 104)
    return first < 0


def _contrib_one(arg):
    tag, cfg, seed, n_ep = arg
    from src import runner as new_runner
    from src.core import Network as _Net
    born = {"n": 0}
    _og = _Net._grow

    def _gc(self, npw, nqw, nw, _o=_og, _b=born):
        _b["n"] += int(npw.size)
        return _o(self, npw, nqw, nw)

    _Net._grow = _gc
    try:
        r = new_runner.run(cfg.with_(seed=seed, n_ep=n_ep))
    finally:
        _Net._grow = _og
    return (tag, float(r["succ"]), int(born["n"]))


def layer_contrib(seeds, n_ep=200, nproc=None):
    """层 4：单变量消融——从 ga_discrete 出发，每次只关掉一项，看成功率掉多少。

    为什么从已对齐的基准往下关（而不是从 base 往上开）：各项之间有交互
    （实测：单独加 topo_skip_first 时被 win_i 的 bug 掩盖，看不出贡献）。
    从「已对上的基准」往下拆，每一项的边际贡献才是干净的。
    """
    from dataclasses import replace
    from src import experiments
    from src.config import EnvSpec

    A = experiments.BY_NAME["ga_discrete"]

    def v_off(name, **kw):
        return (name, A.with_(name=name, **kw))

    variants = [
        ("aligned（全开）", A),
        v_off("-topo_gate", plast=replace(A.plast, topo_gate=True)),
        v_off("-topo_skip_first", plast=replace(A.plast, topo_skip_first=False)),
        v_off("-win_i 重置", reset_readout_per_ep=True),
        v_off("-tie_break", readout=replace(A.readout, tie_break="first")),
        v_off("-silence_update_w", signal=replace(A.signal, silence_update_w=False)),
        v_off("-bridge", signal=replace(A.signal, bridge=0.0)),
        v_off("-steady_gate_ep", signal=replace(A.signal, steady_gate_ep=False)),
        v_off("-u_buf_rows", u_buf_rows=16384),
        v_off("-actions（仅±1）", env=EnvSpec(actions=(1, -1))),
    ]

    # 并行度 = 变体数 × 种子数（不是按变体串行）
    tasks = [(tag, cfg, s, n_ep) for tag, cfg in variants for s in seeds]
    print("=" * 104)
    print("层 4 / contrib — 单变量消融（aligned 为基准，每次只关一项）")
    print("  %d 变体 × %d seeds × %d ep   并行=%s"
          % (len(variants), len(seeds), n_ep, nproc or "cpu_count"))
    print("=" * 104)

    rs = _pool_map(_contrib_one, tasks, nproc)
    per = {}
    for tag, succ, born in rs:
        per.setdefault(tag, []).append((succ, born))

    print("%-22s %10s %14s %6s %8s" % ("variant", "succ", "Δ vs aligned", "seeds", "born"))
    print("-" * 104)
    base_succ = None
    for tag, _cfg in variants:
        v = per[tag]
        m = float(np.mean([x[0] for x in v]))
        b = int(round(np.mean([x[1] for x in v])))
        if base_succ is None:
            base_succ = m
            d = "—"
        else:
            d = "%+.1f pp" % ((m - base_succ) * 100)
        print("%-22s %9.1f%% %14s %6d %8d" % (tag, m * 100, d, len(v), b))

    print("\n" + "=" * 104)
    print("读法：Δ 为该开关的边际贡献（从 aligned 关掉后的损失）。")
    print("=" * 104)
    return True


def layer_modes(seed=0, n_ep=20):
    """层 0：grow_agent 每个 mode 实际开了哪几层——**代码解析 + 实测交叉**。

    为什么必须先查：旧脚本里那段 `_MODE_PARTS` 解析很绕（`discrete_w` 其实含 bias、
    `no_gate` 其实只学 bias），只看名字会搞错移植目标。
    判据：解析结果必须被实测打中（bias 关 -> out_bias==0；拓扑关 -> born==0 且 E 不降）。
    """
    print("=" * 110)
    print("层 0 / modes — grow_agent 每个 mode 实际开了哪几层")
    print("  seed=%d  n_ep=%d" % (seed, n_ep))
    print("=" * 110)

    names = ['discrete', 'base', 'discrete_none', 'discrete_bias', 'discrete_w',
             'discrete_topo', 'discrete_bias_w', 'discrete_all',
             'frozen', 'no_gate', 'no_bridge', 'no_isolation', 'recall']

    # 「拓扑关」的判据不是 E0_INIT（那是采样数 30000），而是**去重后的初始边数**。
    _rng = np.random.default_rng(seed)
    _p0, _p1, _w0, _k0, _a0 = old_ga._init_net(_rng)
    E_INIT_REAL = int(_k0.size)
    print("  初始边数（去重后）= %d（采样 %d）" % (E_INIT_REAL, old_net.E0_INIT))

    print("%-18s %-14s %8s %8s %8s %8s %7s" % (
        "mode", "解析(b/w/topo)", "out_bias", "born", "E", "rate%", "succ"))
    print("-" * 110)
    rows = []
    for mode in names:
        # —— 与 grow_agent.py L304-311 逐字等价的解析 ——
        parts = mode.replace('discrete_', '').split('_')
        do_bias = (mode != 'frozen') and ('none' not in parts)
        do_w = (('topo' not in parts) and
                ('bias' in parts or 'all' in parts or 'w' in parts)) or mode == 'discrete'
        do_topo = ('topo' in parts or 'all' in parts or mode == 'discrete')
        if 'none' in parts:
            do_bias = do_w = do_topo = False
        if 'bias' in parts and len(parts) == 1:
            do_w = do_topo = False

        r = old_ga.run_one((seed, mode, n_ep))
        rows.append((mode, do_bias, do_w, do_topo, r))
        print("%-18s %-14s %8.4f %8d %8d %7.2f %7.1f%%" % (
            mode, "%d/%d/%d" % (do_bias, do_w, do_topo),
            r.get("out_bias", 0.0), r.get("born", 0), r.get("E", 0),
            r.get("rate", 0.0) * 100, r.get("succ", 0.0) * 100))

    print("\n  —— 解析 vs 实测的交叉校验 ——")
    bad = 0
    for mode, do_bias, do_w, do_topo, r in rows:
        issues = []
        if not do_bias and abs(r.get("out_bias", 0.0)) > 1e-9:
            issues.append("声明 bias 关但 out_bias=%.4f" % r["out_bias"])
        if not do_topo and r.get("born", 0) > 0:
            issues.append("声明 topo 关但 born=%d" % r["born"])
        if not do_topo and r.get("E", 0) != E_INIT_REAL:
            issues.append("声明 topo 关但 E=%d（初始 %d）= 被淘汰过" % (r["E"], E_INIT_REAL))
        if issues:
            bad += 1
            print("  **%s：%s**" % (mode, "; ".join(issues)))
    if not bad:
        print("  全部 mode 的解析与实测一致（%d 个）" % len(rows))

    # 同一配置的「不同 mode」会给出逐位相同的结果——这是「对照组名不副实」的直接证据。
    # 但反向推论需要前提：**跑得够长**。短跑下不同配置也会逐位相同（都还没分化）。
    print("\n  —— 行为指纹分组（同组 = 实际是同一个配置）——")
    if n_ep < 10:
        print("  **警告：n_ep=%d 太短——不同配置也会因为「都还没学到东西」而逐位相同，"
              "分组不可信。用 n_ep>=20 复核。**" % n_ep)
    groups = {}
    for mode, _db, _dw, _dt, r in rows:
        sig = (round(r.get("out_bias", 0.0), 6), r.get("born", 0),
               r.get("E", 0), round(r.get("succ", 0.0), 6))
        groups.setdefault(sig, []).append(mode)
    for sig, ms in groups.items():
        if len(ms) > 1:
            print("  **%s  —— 结果逐位相同**" % " / ".join(ms))

    print("\n  提醒：分组的依据是「结果逐位相同 + 三层开关相同」。")
    print("        短跑（n_ep<10）下很多组合会因「都没学到」而相同——用 n_ep>=20 复核。")

    print("\n" + "=" * 110)
    return bad == 0


def _talk_one(arg):
    """层 10：grow_talk 移植对齐——10 个标量 + 9 个序列逐位比较。"""
    mode, cfg_name, seed, n_present = arg
    import grow_talk as old_t
    from src import runner as new_runner
    from src import experiments

    r_old = old_t.run_one((seed, mode, n_present, old_t.ANSWER_FROM,
                           old_t.CONT_INPUT, old_t.USE_STD, old_t.USE_SCALE))
    cfg = experiments.BY_NAME[cfg_name].with_(seed=seed, n_ep=n_present)
    r_new = new_runner.run(cfg)

    SCALARS = [("E", "E_final"), ("born", "born"), ("drop", "drop"),
               ("rate", "rate"), ("n_inh", "n_inh"), ("deg_out", "out_deg"),
               ("sigma", "sigma"), ("rate_burst", "rate_burst"),
               ("rate_tail", "rate_tail"), ("burst_fire", "burst_fire")]
    SEQS = [("win", "win"), ("win_sil", "win_sil"), ("win_early", "win_early"),
            ("win_lat", "win_lat"), ("rate_win", "rate_win"),
            ("buck_n", "buck_n"), ("buck_acc", "buck_acc"),
            ("bias", "out_bias"), ("theta_out", "theta_out")]

    def eq(a, b):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))

    bad = []
    for ko, kn in SCALARS:
        if not eq(r_old.get(ko), r_new.get(kn)):
            bad.append((ko, r_old.get(ko), r_new.get(kn)))
    for ko, kn in SEQS:
        a, b = r_old.get(ko), r_new.get(kn)
        if a is None or b is None or len(a) != len(b) or not all(
                eq(p, q) for p, q in zip(a, b)):
            bad.append((ko, "len=%s" % (len(a) if a is not None else None),
                        "len=%s" % (len(b) if b is not None else None)))
    return dict(mode=mode, cfg=cfg_name, bad=bad,
                acc_old=r_old.get("acc_last", float("nan")),
                acc_new=r_new.get("acc_last", float("nan")),
                rate_old=r_old.get("rate", 0.0), rate_new=r_new.get("rate", 0.0))


def layer_talk(seed=0, n_present=100, nproc=None):
    """层 10 / talk — grow_talk 的移植对齐（结构 + 端到端 19 项）。"""
    pairs = [("teach", "ta_teach"), ("rand_m", "ta_rand_m"),
             ("no_teach", "ta_no_teach"), ("frozen", "ta_frozen")]
    print("=" * 108)
    print("层 10 / talk — grow_talk 移植验证（判据 = 10 标量 + 9 序列逐位相同）")
    print("  seed=%d  n_present=%d" % (seed, n_present))
    print("=" * 108)
    print("%-10s %-16s %9s %9s %9s %9s  %s" % (
        "old mode", "new config", "rate 旧", "rate 新", "acc 旧", "acc 新", "判定"))
    print("-" * 108)

    args = [(m, c, seed, n_present) for m, c in pairs]
    rs = _pool_map(_talk_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（19/19）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join(str(b[0]) for b in r["bad"][:3]))
        print("%-10s %-16s %8.3f%% %8.3f%% %9.3f %9.3f  %s" % (
            r["mode"], r["cfg"], r["rate_old"] * 100, r["rate_new"] * 100,
            r["acc_old"], r["acc_new"], verdict))

    print("-" * 108)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(pairs)))
    print("=" * 108)
    return n_ok == len(pairs)


def _homeo_one(arg):
    """层 11：grow_homeo 移植对齐——hist 逐行（6 列）+ 末态 E-E 边数。"""
    seed, steps, cfg_name = arg
    import grow_homeo as old_h
    from src import runner as new_runner
    from src import experiments

    r_old = old_h.grow((seed, steps, steps // 2))
    cfg = experiments.BY_NAME[cfg_name].with_(seed=seed, steps=steps)
    r_new = new_runner.run(cfg)

    ho, hn = r_old["hist"], r_new["hist"]
    bad = []
    if len(ho) != len(hn):
        bad.append(("hist_len", len(ho), len(hn)))
    for i, (a, b) in enumerate(zip(ho, hn)):
        for j in range(min(len(a), len(b))):
            try:
                ok = abs(float(a[j]) - float(b[j])) <= 1e-9 * max(1.0, abs(float(a[j])))
            except (TypeError, ValueError):
                ok = a[j] == b[j]
            if not ok:
                bad.append(("hist[%d][%d]" % (i, j), a[j], b[j]))
                break
        if bad:
            break
    if r_old["n_ee"] != r_new["n_ee"]:
        bad.append(("n_ee", r_old["n_ee"], r_new["n_ee"]))
    last = hn[-1] if hn else (0,) * 6
    return dict(seed=seed, steps=steps, cfg=cfg_name, bad=bad,
                n_ee_o=r_old["n_ee"], n_ee_n=r_new["n_ee"],
                rate=last[2], over=int(last[3]), under=int(last[4]), alive=int(last[5]))


def layer_homeo(seed=0, steps=6000, nproc=None):
    """层 11 / homeo — grow_homeo 的移植对齐（hist 逐行）。"""
    pairs = [("ho_grow", 6000), ("ho_long", 40000)]
    print("=" * 108)
    print("层 11 / homeo — grow_homeo 移植验证（判据 = hist 逐行逐列 + 末态 E-E 边数）")
    print("  seed=%d" % seed)
    print("=" * 108)
    print("%-10s %8s %9s %9s %8s %8s %6s  %s" % (
        "config", "steps", "n_ee 旧", "n_ee 新", "rate%", "超标", "不达", "判定"))
    print("-" * 108)
    args = [(seed, st, cn) for cn, st in pairs]
    rs = _pool_map(_homeo_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join(str(b[0]) for b in r["bad"][:3]))
        print("%-10s %8d %9d %9d %8.3f %8d %6d  %s" % (
            r["cfg"], r["steps"], r["n_ee_o"], r["n_ee_n"],
            r["rate"] * 100, r["over"], r["under"], verdict))
    print("-" * 108)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(pairs)))
    print("=" * 108)
    return n_ok == len(pairs)


def _homeoprobe_one(arg):
    """层 12：grow_homeo --probe 移植对齐——4 个 cue 的两两相似度 + 总放电。"""
    seed, warm, cfg_name = arg
    import grow_homeo as old_h
    from src import runner as new_runner
    from src import experiments

    r_old = old_h.probe((seed, 4, warm))
    cfg = experiments.BY_NAME[cfg_name].with_(seed=seed)
    r_new = new_runner._probe_homeo(cfg, 4, warm)

    bad = []
    if r_old["n_ee"] != r_new["n_ee"]:
        bad.append(("n_ee", r_old["n_ee"], r_new["n_ee"]))
    for i, (a, b) in enumerate(zip(r_old["sims"], r_new["sims"])):
        if abs(a - b) > 1e-9:
            bad.append(("sims[%d]" % i, a, b))
    for i, (a, b) in enumerate(zip(r_old["totals"], r_new["totals"])):
        if abs(a - b) > 1e-6:
            bad.append(("totals[%d]" % i, a, b))
    import numpy as np
    m = float(np.mean([x for x in r_new["sims"] if x == x])) if r_new["sims"] else float("nan")
    return dict(seed=seed, cfg=cfg_name, bad=bad, n_ee=r_new["n_ee"], sim_mean=m)


def layer_homeoprobe(seed=0, warm=30000, nproc=None):
    """层 12 / homeoprobe — grow_homeo --probe 的移植对齐。"""
    print("=" * 108)
    print("层 12 / homeoprobe — grow_homeo --probe（4 cue 两两相似度 + 总放电）")
    print("  warm=%d" % warm)
    print("=" * 108)
    print("%-5s %9s %12s  %s" % ("seed", "n_ee", "平均相似度", "判定"))
    print("-" * 108)
    args = [(s, warm, "ho_warm") for s in range(max(2, seed + 1))]
    rs = _pool_map(_homeoprobe_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（n_ee + 6 sims + 4 totals）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join(str(b[0]) for b in r["bad"][:3]))
        print("%-5d %9d %12.3f  %s" % (r["seed"], r["n_ee"], r["sim_mean"], verdict))
    print("-" * 108)
    print("结论：%d/%d 个种子逐位相同" % (n_ok, len(rs)))
    print("=" * 108)
    return n_ok == len(rs)


def _cond_one(arg):
    """层 13：grow_cond 移植对齐——14 个标量 + 8 个序列逐位比较。"""
    seed, n_pres, punish = arg
    import grow_cond as old_c
    from src import runner as new_runner
    from src import experiments

    r_old = old_c.run_one((seed, "teach", n_pres, punish))
    name = "co_punish" if punish else "co_teach"
    cfg = experiments.BY_NAME[name].with_(seed=seed, n_ep=n_pres)
    r_new = new_runner.run(cfg)

    SCALARS = [("E", "E_final"), ("born", "born"), ("drop", "drop"),
               ("rate", "rate"), ("n_inh", "n_inh"), ("sigma", "sigma"),
               ("deg_out", "out_deg"), ("acc_first", "acc_first"),
               ("acc_last", "acc_last"), ("early_first", "early_first"),
               ("early_last", "early_last"), ("sil_last", "sil_last"),
               ("delay_first", "delay_first"), ("delay_last", "delay_last")]
    SEQS = [("win", "win"), ("win_sil", "win_sil"), ("win_early", "win_early"),
            ("win_delay", "win_delay"), ("win_rate", "win_rate"),
            ("buck_n", "buck_n"), ("buck_acc", "buck_acc"), ("bias", "out_bias")]

    def eq(a, b):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))

    bad = []
    for ko, kn in SCALARS:
        if not eq(r_old.get(ko), r_new.get(kn)):
            bad.append((ko, r_old.get(ko), r_new.get(kn)))
    for ko, kn in SEQS:
        a, b = r_old.get(ko), r_new.get(kn)
        if a is None or b is None or len(a) != len(b) or not all(
                eq(p, q) for p, q in zip(a, b)):
            bad.append((ko, "len=%s" % (len(a) if a is not None else None),
                        "len=%s" % (len(b) if b is not None else None)))
    return dict(seed=seed, cfg=name, bad=bad,
                acc_o=r_old.get("acc_last", float("nan")),
                acc_n=r_new.get("acc_last", float("nan")),
                early_o=r_old.get("early_last", float("nan")),
                early_n=r_new.get("early_last", float("nan")),
                rate_o=r_old.get("rate", 0.0), rate_n=r_new.get("rate", 0.0))


def layer_cond(seed=0, n_pres=400, nproc=None):
    """层 13 / cond — grow_cond 的移植对齐。"""
    print("=" * 108)
    print("层 13 / cond — grow_cond 移植验证（判据 = 14 标量 + 8 序列逐位相同）")
    print("  seed=%d  n_pres=%d" % (seed, n_pres))
    print("=" * 108)
    print("%-10s %9s %9s %9s %9s %8s  %s" % (
        "config", "acc 旧", "acc 新", "抢答旧", "抢答新", "rate%", "判定"))
    print("-" * 108)
    args = [(seed, n_pres, False), (seed, n_pres, True)]
    rs = _pool_map(_cond_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（22/22）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join(str(b[0]) for b in r["bad"][:3]))
        print("%-10s %9.4f %9.4f %9.3f %9.3f %8.3f  %s" % (
            r["cfg"], r["acc_o"], r["acc_n"], r["early_o"], r["early_n"],
            r["rate_n"] * 100, verdict))
    print("-" * 108)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(args)))
    print("=" * 108)
    return n_ok == len(args)


def _teach_one(arg):
    """层 14：grow_teach 移植对齐——10 个标量 + win 序列 + conf_right/conf_wrong。"""
    seed, n_tr, mode = arg
    import grow_teach as old_t
    from src import runner as new_runner
    from src import experiments

    r_old = old_t.run_one((seed, mode, n_tr))
    cfg = experiments.BY_NAME["te_" + mode].with_(seed=seed, n_ep=n_tr)
    r_new = new_runner.run(cfg)

    SCALARS = [("E", "E_final"), ("born", "born"), ("drop", "drop"),
               ("rate", "rate"), ("n_inh", "n_inh"), ("deg_out", "out_deg"),
               ("conf_right", "conf_right"), ("conf_wrong", "conf_wrong"),
               ("acc_first", "acc_first"), ("acc_last", "acc_last")]

    def eq(a, b):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))

    bad = []
    for ko, kn in SCALARS:
        if not eq(r_old.get(ko), r_new.get(kn)):
            bad.append((ko, r_old.get(ko), r_new.get(kn)))
    a, b = r_old.get("win"), r_new.get("win")
    if a is None or b is None or len(a) != len(b) or not all(eq(x, y) for x, y in zip(a, b)):
        bad.append(("win", len(a) if a else None, len(b) if b else None))
    return dict(seed=seed, mode=mode, bad=bad,
                acc_o=r_old.get("acc_last", float("nan")),
                acc_n=r_new.get("acc_last", float("nan")),
                rate_o=r_old.get("rate", 0.0), rate_n=r_new.get("rate", 0.0))


def layer_teach(seed=0, n_tr=600, nproc=None):
    """层 14 / teach — grow_teach 的移植对齐。"""
    modes = ["teach", "rand_m", "no_teach", "frozen"]
    print("=" * 108)
    print("层 14 / teach — grow_teach 移植验证（判据 = 10 标量 + win 序列逐位相同）")
    print("  seed=%d  n_trials=%d" % (seed, n_tr))
    print("=" * 108)
    print("%-10s %9s %9s %9s %9s  %s" % (
        "mode", "acc 旧", "acc 新", "rate 旧", "rate 新", "判定"))
    print("-" * 108)
    args = [(seed, n_tr, m) for m in modes]
    rs = _pool_map(_teach_one, args, nproc)
    n_ok = 0
    for r in rs:
        if not r["bad"]:
            verdict = "OK（11/11）"
            n_ok += 1
        else:
            verdict = "**%d 项不同**：%s" % (
                len(r["bad"]), ", ".join(str(b[0]) for b in r["bad"][:3]))
        print("%-10s %9.4f %9.4f %9.4f %9.4f  %s" % (
            r["mode"], r["acc_o"], r["acc_n"], r["rate_o"], r["rate_n"], verdict))
    print("-" * 108)
    print("结论：%d/%d 个移植项逐位相同" % (n_ok, len(args)))
    print("=" * 108)
    return n_ok == len(args)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "construct"
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = list(range(nseed))
    if cmd == "construct":
        ok = layer_construct(seeds)
    elif cmd == "forward":
        nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        ok = layer_forward(seeds, nsteps)
    elif cmd == "run":
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        cfg_name = sys.argv[4] if len(sys.argv) > 4 else "base"
        ok = layer_run(seeds, n_ep, cfg_name)
    elif cmd == "noop":
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        ok = layer_noop(seeds, n_ep)
    elif cmd == "seq":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        mode = sys.argv[4] if len(sys.argv) > 4 else "discrete"
        cfg_name = sys.argv[5] if len(sys.argv) > 5 else "ga_discrete"
        ok = layer_seq(seed, n_ep, diag=("--diag" in sys.argv),
                       mode=mode, cfg_name=cfg_name)
    elif cmd == "contrib":
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        ok = layer_contrib(seeds, n_ep)
    elif cmd == "modes":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        ok = layer_modes(seed, n_ep)
    elif cmd == "verify":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_verify(seed, n_ep, nproc)
    elif cmd == "goal":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_goal(seed, n_ep, nproc)
    elif cmd == "world":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_world(seed, steps, nproc)
    elif cmd == "talk":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_present = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_talk(seed, n_present, nproc)
    elif cmd == "homeo":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_homeo(seed, 6000, nproc)
    elif cmd == "homeoprobe":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        warm = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_homeoprobe(seed, warm, nproc)
    elif cmd == "cond":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_pres = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_cond(seed, n_pres, nproc)
    elif cmd == "teach":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n_tr = int(sys.argv[3]) if len(sys.argv) > 3 else 600
        nproc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = layer_teach(seed, n_tr, nproc)
    else:
        print("未知层：%s（construct / forward / run / noop / seq / contrib / modes / verify / goal / world）" % cmd)
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
