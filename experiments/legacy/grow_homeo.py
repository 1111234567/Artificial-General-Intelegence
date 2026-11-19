# grow_homeo.py - 阶段 0：让结构在放电率稳态下自己长
#
# 形态来自 PLOS Comput Biol 2022：
#   "Homeostatic control of synaptic rewiring in recurrent networks induces
#    the formation of stable memory engrams"
#
# 关键三句原文：
#   ① "We initialize a network with random connections to and from inhibitory neurons
#      and **no excitatory-to-excitatory connections, whatsoever**. The latter are then
#      grown under the control of homeostatic structural plasticity."
#   ② "The controlled variable is the firing rate of excitatory neurons, the target rate
#      is set to ν = 8 Hz"
#   ③ "After a long-enough growth time, the network structure has reached its equilibrium
#      and all neurons fire at their target rate"
#
# 与之前所有实现（grow_teach / grow_talk / grow_cond）的根本区别：
#   之前：结构层 = 强化 + 随机新建 + **边数超预算才淘汰最弱的**（被动、单向、按权重）
#   现在：结构层 = **按每个神经元的放电率偏差双向调连接数**（闭环、双向、按率）
#
# 本文件只做一件事：看这样长出来的网络，有没有长出「多个稳定态」。
# 不涉及任务、不涉及教师、不涉及奖励。
#
# 判据（先写死）：
#   ① 平衡：E-E 边数收敛、rate 收到目标附近
#   ② 多吸引子：不同 cue → 关掉输入后的活动模式两两相似度 < 0.5
#      （旧实现里相似度 0.997，即只有一个全局态）

import sys
import time
import numpy as np
from multiprocessing import Pool

# ---- 网络 ----
N = 1000
N_E = 800
N_I = N - N_E
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50              # 兴奋权重
W_I = -1.0              # 抑制权重
# ⚠️ 原为 -2.5（旧标定值，当时是为了「有完整 E-E 网络」调的）。
# 无 E-E 时抑制完全主导：200 个抑制神经元每个扇出 140 个兴奋，
# 每步 3 个抑制放电 × 140 = 240 个兴奋被 -2.5 打击，而只有 10.5 个兴奋得到 +3.0。
# → rate 被钉在 0.07%，BG 从 0.004 到 0.015（3.75 倍）完全无效。
# 扫描（experiments/legacy/scan_base.py）：W_I=-1.0 / D_IE=0.05 时 E_rate=0.41%，692/800 活跃。
W_EE_INIT = 0.25        # 新生 E-E 边的初始强度

# ---- 初始结构（无 E-E）----
N_IN = 100
N_IN_FANOUT = 50        # 每个输入神经元 → 下游
IN_SPIKE = 3.0
D_EI = 0.20             # E→I 连接概率
D_IE = 0.05             # I→E 连接概率（原 0.20；扫描见上）
D_II = 0.10             # I→I

# ---- 稳态结构可塑性（核心）----
N_CUE = 4               # cue 的个数（输入区分成 4 块）
R_TARGET = 0.008        # 目标放电率 0.8%（PLOS CB 的 8 Hz）
TOL = 0.20              # ±20% 内不动
TAU_RATE = 500.0        # 放电率估计的时间常数（步）
T_STRUCT = 50           # 结构更新周期
ENC_STEPS = 20          # 编码步数
ENC_GAP = 150           # 两次 cue 的间隔
P_DEL = 0.30            # 超标神经元每周期删掉的入边比例
P_NEW = 0.30            # 不达标神经元每周期新建的入边比例
# ⚠️ 原为 0.06。实测：rate 0.51% 时 rel=-0.375，533 个不达标 →
# 533 × 0.375 × 0.06 = 12 条/50 步 → 长到入度 25（目标所需）要 5.5 万步。
# 加到 0.30 后约 60 条/50 步，且删除同步加快，两头才能动态平衡。
W_MAX_EE = 1.0

# ---- 背景输入 ----
# ⚠️ 必须弱到不够独立维持目标率，否则 E-E 边永远不会长。
# PLOS CB 2022 里背景输入只给一个底，靠 E-E 连接把率拉到 8 Hz。
BG_RATE = 0.015         # 每神经元每步背景脉冲概率
# 标定记录（N=1000, 700 个兴奋, W_I=-2.5, D_IE=0.2, 无 E-E）：
#   0.004 → rate 0.06%（太低：源不活跃 → 边长也没用 → 永远起不来）
#   0.010 → rate 0.07%（同上；抑制把驱动抵掉了）
#   0.020 → rate 0.73%（已在目标区间 0.64~0.96 内，不需要长）
# 所以取中间的 0.015，预期 rate ≈ 0.55%（低于区间下限 → 有生长空间）
IN_LO = BG_RATE

# ---- 活动稳态（已有的两个）----
TAU_ADAPT = 100.0
D_THETA = 0.20
TH_MIN = 1.0
TH_MAX = 3.0
USE_STD = True
TAU_REC = 100.0
U_STD = 0.30


def _ee_mask(pre, post):
    """真正的 E-E 边：**排除输入区**。

    ⚠️ 输入扇出的 pre 是 0..99（< N_E），post 也落在 100..799 里，
    所以 `(pre < N_E) & (post < N_E)` 会把输入扇出当成 E-E 边（实测 3797 条）。
    那是「感觉输入」，绝不能被稳态裁剪掉。
    """
    return (pre >= N_IN) & (pre < N_E) & (post < N_E)


def build(rng):
    """初始结构：**无 E-E**。只有 E→I、I→E、I→I、输入扇出。"""
    pre, post = [], []

    n = int(N_E * N_I * D_EI)
    pre.append(rng.integers(0, N_E, n).astype(np.int32))
    post.append(rng.integers(N_E, N, n).astype(np.int32))

    n = int(N_E * N_I * D_IE)
    pre.append(rng.integers(N_E, N, n).astype(np.int32))
    post.append(rng.integers(0, N_E, n).astype(np.int32))

    n = int(N_I * N_I * D_II)
    a = rng.integers(N_E, N, n).astype(np.int32)
    b = rng.integers(N_E, N, n).astype(np.int32)
    ok = a != b
    pre.append(a[ok]); post.append(b[ok])

    # 输入扇出：输入区 → 全体（兴奋 + 抑制）
    src = np.repeat(np.arange(N_IN, dtype=np.int32), N_IN_FANOUT)
    dst = rng.integers(N_IN, N, N_IN * N_IN_FANOUT).astype(np.int32)
    pre.append(src); post.append(dst)

    pre = np.concatenate(pre)
    post = np.concatenate(post)

    w = np.where(pre >= N_E, W_I, W_E).astype(np.float32)
    return pre, post, w


def _sort(pre, post, w):
    key = post.astype(np.int64) * N + pre
    o = np.argsort(key, kind='stable')
    key, pre, post, w = key[o], pre[o], post[o], w[o]
    if key.size > 1:
        u = np.empty(key.size, bool); u[0] = True
        np.not_equal(key[1:], key[:-1], out=u[1:])
        if not u.all():
            key, pre, post, w = key[u], pre[u], post[u], w[u]
    return (np.ascontiguousarray(pre), np.ascontiguousarray(post),
            np.ascontiguousarray(w), np.ascontiguousarray(key))


def simulate(pre, post, w, key, rng, steps, rate_ema=None, start=None,
             cue=None):
    """跑 steps 步。cue = (块号, 编码步数) 时，前几步给输入脉冲。

    ⚠️ rate_ema 必须由调用方传入并返回。
    旧版每次调用都重新 `np.full(N, R_TARGET)`，于是 hist 里打印的 0.73%
    实际上是「0.8% 衰减 50 步」的假象（0.008 × 0.998^50 = 0.0072）。
    """
    if rate_ema is None:
        rate_ema = np.zeros(N, np.float64)
    if start is None:
        v = np.zeros(N, np.float32)
        s = np.zeros(N, np.float32)
        ref = np.zeros(N, np.int32)
        theta = np.full(N, V_TH, np.float32)
        x_rec = np.ones(N, np.float32)
    else:
        v = start['v'].copy(); s = start['s'].copy(); ref = start['ref'].copy()
        theta = start['theta'].copy(); x_rec = start['x_rec'].copy()
    nsp_hist = []
    trace = np.zeros(N, np.float32)

    for t in range(steps):
        xin = np.zeros(N, np.float32)
        xin[N_IN:] = (rng.random(N - N_IN) < BG_RATE).astype(np.float32) * IN_SPIKE

        # 编码：cue 的块在前几拍给脉冲（对应 PLOS CB 的 stimulation cycle）
        if cue is not None:
            blk, enc = cue
            B = N_IN // N_CUE
            if t == 0 or t == max(1, enc - 2):
                xin[blk * B:(blk + 1) * B] = IN_SPIKE

        if USE_STD:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = T_REF

        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
        theta += (V_TH - theta) / TAU_ADAPT

        if USE_STD:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

        a = 1.0 / TAU_RATE
        rate_ema *= (1 - a)
        rate_ema += a * s

        trace *= 0.95
        trace += s

        nsp_hist.append(float(s.sum()))

    return dict(v=v, s=s, ref=ref, theta=theta, x_rec=x_rec, rate_ema=rate_ema,
                trace=trace, nsp_hist=nsp_hist)


def grow(args):
    seed, steps, cue_after = args
    rng = np.random.default_rng(seed)
    pre, post, w = build(rng)
    pre, post, w, key = _sort(pre, post, w)

    hist = []
    t0 = time.perf_counter()
    n_chunk = T_STRUCT
    done = 0
    r_ema = None
    st = None
    cue_plan = []
    tt = cue_after
    while tt < steps:
        cue_plan.append((tt, int(rng.integers(0, N_CUE))))
        tt += ENC_GAP
    ci = 0
    while done < steps:
        ch = min(n_chunk, steps - done)
        cue = None
        if ci < len(cue_plan) and cue_plan[ci][0] < done + ch:
            cue = (cue_plan[ci][1], ENC_STEPS)
            ci += 1
        st = simulate(pre, post, w, key, rng, ch, rate_ema=r_ema, cue=cue)
        r_ema = st['rate_ema'].copy()
        done += ch

        eemask = _ee_mask(pre, post)
        n_ee = int(eemask.sum())
        rr = st['rate_ema']
        hist.append((done, n_ee, float(rr[:N_E].mean()),
                     float((rr[:N_E] > R_TARGET * (1 + TOL)).sum()),
                     float((rr[:N_E] < R_TARGET * (1 - TOL)).sum()),
                     float((rr[:N_E] > R_TARGET * 0.1).sum())))

        # ===== 结构可塑性：按每个神经元的放电率偏差，双向调 E-E 入边 =====
        dev = st['rate_ema'][:N_E] - R_TARGET
        rel = dev / R_TARGET

        # ① 裁剪：超标神经元，按偏差概率删掉入边（只删 E-E）
        if eemask.any():
            idx = np.nonzero(eemask)[0]
            d_ = rel[post[idx]]
            pdel = np.clip(d_, 0, 1) * P_DEL
            keep = rng.random(idx.size) >= pdel
            drop = idx[~keep]
            if drop.size:
                m = np.ones(pre.size, bool); m[drop] = False
                pre, post, w, key = pre[m], post[m], w[m], key[m]

        # ② 生长：不达标神经元，按偏差概率新建入边（从活跃的兴奋神经元）
        # ② 生长：**按共激活选取源**（不是随机）
        # 关键。旧版从「所有活跃神经元」均匀随机挑 -> 只保证放电率，
        # 不含选择性 -> 永远长不出「某 cue -> 某一群」的映射（实测相似度 0.996）。
        # PLOS CB 的 engram 是反复共激活长出来的。
        tw = st['trace'][:N_E].astype(np.float64)
        if tw.sum() > 0:
            starve = np.nonzero(rel < -TOL)[0]
            if starve.size:
                pnew = np.clip(-rel[starve], 0, 1) * P_NEW
                want = starve[rng.random(starve.size) < pnew]
                if want.size:
                    p_src = tw / tw.sum()
                    np_pre = rng.choice(N_E, want.size, p=p_src).astype(np.int32)
                    np_post = want.astype(np.int32)
                    ok = np_pre != np_post
                    np_pre, np_post = np_pre[ok], np_post[ok]
                    if np_pre.size:
                        npw = np.full(np_pre.size, W_EE_INIT, np.float32)
                        pre, post, w, key = _sort(
                            np.concatenate([pre, np_pre]),
                            np.concatenate([post, np_post]),
                            np.concatenate([w, npw]))

    return dict(seed=seed, hist=hist, n_ee=int(_ee_mask(pre, post).sum()),
                elapsed=time.perf_counter() - t0,
                pre=pre, post=post, w=w)


def probe(args):
    """长好之后：不同 cue → 关掉输入 → 模式是否不同？"""
    seed, n_cue, warm = args
    g = grow((seed, warm, warm // 2))
    pre, post, w, key = _sort(g['pre'], g['post'], g['w'])
    rng = np.random.default_rng(seed + 99999)

    N_CUE = 100
    BLK = N_IN // n_cue
    TOTAL = np.zeros((n_cue, N), np.float64)

    for cu in range(n_cue):
        # 从同一个预热态出发（先跑 500 步背景）
        st = simulate(pre, post, w, key, rng, 500)
        v = st['v'].copy(); s = st['s'].copy(); ref = st['ref'].copy()
        theta = st['theta'].copy(); x_rec = st['x_rec'].copy()
        rate_ema = st['rate_ema'].copy()

        # 给 cue：编码 20 步
        for t in range(20):
            xin = np.zeros(N, np.float32)
            if t == 0 or t == 18:
                xin[cu * BLK:(cu + 1) * BLK] = IN_SPIKE
            xin[N_IN:] = 0.0   # probe 阶段完全关掉背景（否则模式被背景主导）
            if USE_STD:
                w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
            else:
                w_eff = w
            I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
            v = v * DECAY_V + I + xin
            fire = (v > theta) & (ref <= 0)
            s = fire.astype(np.float32)
            v = np.where(s > 0, 0.0, v)
            ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
            if fire.any():
                theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
            theta += (V_TH - theta) / TAU_ADAPT
            if USE_STD:
                x_rec += (1.0 - x_rec) / TAU_REC
                ii = np.nonzero(s[N_E:] > 0)[0] + N_E
                if ii.size:
                    x_rec[ii] *= (1.0 - U_STD)

        # 关掉 cue，只留背景，统计 300 步
        for t in range(300):
            xin = np.zeros(N, np.float32)
            xin[N_IN:] = 0.0   # probe 阶段完全关掉背景（否则模式被背景主导）
            if USE_STD:
                w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
            else:
                w_eff = w
            I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
            v = v * DECAY_V + I + xin
            fire = (v > theta) & (ref <= 0)
            s = fire.astype(np.float32)
            v = np.where(s > 0, 0.0, v)
            ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
            if fire.any():
                theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
            theta += (V_TH - theta) / TAU_ADAPT
            if USE_STD:
                x_rec += (1.0 - x_rec) / TAU_REC
                ii = np.nonzero(s[N_E:] > 0)[0] + N_E
                if ii.size:
                    x_rec[ii] *= (1.0 - U_STD)
            TOTAL[cu] += s

    sims = []
    for i in range(n_cue):
        for j in range(i + 1, n_cue):
            a, b = TOTAL[i], TOTAL[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float('nan'))
    return dict(seed=seed, n_ee=g['n_ee'], sims=sims,
                totals=[float(x.sum()) for x in TOTAL], elapsed=g['elapsed'])


def main():
    if '--probe' in sys.argv:
        nseed = 2
        warm = 30000
        print('=' * 110)
        print('阶段 0 + 探针：长完之后，不同 cue 是否留下不同模式？')
        print('  目标率 %.3f%%  生长 %d 步  4 个 cue' % (R_TARGET * 100, warm))
        print('=' * 110)
        with Pool(2) as pool:
            res = pool.map(probe, [(s, 4, warm) for s in range(nseed)])
        for r in res:
            print('\nseed %d: E-E 边数 %d  生长耗时 %.1fs' % (r['seed'], r['n_ee'], r['elapsed']))
            print('  各 cue 期间总放电: %s' % ' '.join('%.0f' % x for x in r['totals']))
            print('  两两相似度: %s' % ' '.join('%.3f' % x for x in r['sims']))
            A = np.array([x for x in r['sims'] if x == x])
            if A.size:
                print('  → 平均 %.3f  判定: %s' % (
                    A.mean(), '无记忆（单一态）' if A.mean() > 0.9 else
                    ('有区分（多吸引子）' if A.mean() < 0.5 else '部分区分')))
        return

    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    print('=' * 120)
    print('阶段 0：放电率稳态下的结构自生长（PLOS CB 2022 形态）')
    print('  初始无 E-E 连接  目标率 %.2f%%  ±%.0f%%  τ_rate=%.0f 结构周期 %d 步' % (
        R_TARGET * 100, TOL * 100, TAU_RATE, T_STRUCT))
    print('  P_DEL=%.2f  P_NEW=%.2f  背景输入 %.1f%%' % (P_DEL, P_NEW, BG_RATE * 100))
    print('=' * 120)
    with Pool(nseed) as pool:
        res = pool.map(grow, [(s, steps, steps // 2) for s in range(nseed)])

    for r in res:
        print('\nseed %d  E-E 边数 %d  耗时 %.1fs' % (r['seed'], r['n_ee'], r['elapsed']))
        print('  %8s %8s %9s %8s %8s %8s' % ('步', 'E-E边数', 'E均值率%', '超标数', '不达标', '活跃数'))
        for row in r['hist'][::max(1, len(r['hist']) // 12)]:
            t, nee, mr, hi, lo, de = row
            print('  %8d %8d %9.2f %8.0f %8.0f %8.1f' % (t, nee, mr * 100, hi, lo, de))


if __name__ == '__main__':
    main()
