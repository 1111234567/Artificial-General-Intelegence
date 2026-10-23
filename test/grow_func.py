# grow_func.py - 测功能：预测误差 / 动作增益 / 前史因果
#
# 动机（2026-10-24，experiments.md §16.7 第 2 项）：
#   §16.6 只证明了「管道通了」（rate 正常 + 动作与观测相关 0.51），
#   没有任何指标显示「学」发生了。本版加三项测量，**不改动力学**。
#
# 三项（对应 §16.7 第 2 项的 ①②③）：
#   ① 预测误差是否下降（主判据）
#      读出层用 s[t] 预测 o[t+1]，前一半 step 在线学习，后一半 **冻结**。
#      冻结后误差若仍下降 → 网络状态本身变得更可预测（不是读出层在学）。
#      frozen 组（结构冻结）是对照：它的误差不该降。
#   ② 动作增益（能动性）
#      a[t] 能多大程度解释 o[t+1]-o[t] 的方差（分段线性 R^2，含基线扣除）。
#   ③ 前史因果（排除性检验，--div）
#      同结构、同输入，只把初始膜电位加 1e-3 扰动 → 输出放电是否仍不同。
#      它只能排除「输出仅由当前输入决定」，**通过它不等于学到**（§13.1 第 9 条）。
#
# 动力学与 grow_world.py 完全一致（v5），只加测量。

import sys
import time
import numpy as np
from multiprocessing import Pool

# ---- 网络（沿用 grow_v2 / calib5 标定值）----
N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50
W_I_RATIO = 5.0
ETA_POT = 0.02
W_MAX_E = 1.0
W_MAX_I = 2.5
W_NEW_E = 0.15
W_NEW_I = 0.75
BUDGET = 15000
E0_INIT = 30000

T_S = 20
TRACE_DECAY = 0.95
P_NEW = 0.004
P_GLOBAL = 4
T_GLOBAL = 2000
MAX_ACTIVE = 48
MIN_AGE = 25

# ---- 接口 ----
N_IN = 100
IN_RATE = 0.02
GAIN = 0.8
OUT_IDX = np.array([500, 501])
A_TAU = 20.0
R_BASE = 0.02
R_SCALE = 0.03

# ---- 世界（两个漏积分变量 + 外源周期驱动 + 动作推动）----
PERIOD_FAST = 20.0
PERIOD_SLOW = 500.0
TAU_W_F = 50.0
TAU_W_S = 500.0
AMP_F = 0.10
AMP_S = 0.10
K_PHASE = 0.05
W_NOISE = 0.01

# ---- 功能测量（v6 新增）----
PRED_LR = 0.02          # 归一化 LMS 学习率（读出层）
OSEG = 100              # 输出放电分段的步长（给扰动对比用）
PERTURB = 1e-3          # 前史扰动的幅度（初始膜电位）


def _sort(pre, post, w, age):
    key = post.astype(np.int64) * N + pre
    o = np.argsort(key, kind='stable')
    key, pre, post, w, age = key[o], pre[o], post[o], w[o], age[o]
    if key.size > 1:
        u = np.empty(key.size, bool); u[0] = True
        np.not_equal(key[1:], key[:-1], out=u[1:])
        if not u.all():
            key, pre, post, w, age = key[u], pre[u], post[u], w[u], age[u]
    return (np.ascontiguousarray(pre), np.ascontiguousarray(post),
            np.ascontiguousarray(w), np.ascontiguousarray(key),
            np.ascontiguousarray(age))


def find(key, q):
    if key.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    i = np.searchsorted(key, q)
    np.clip(i, 0, key.size - 1, out=i)
    return key[i] == q, i


def steady_gate(r):
    """活动率落在 0.5%~5% 之外时把可塑性压下去（软门）。"""
    if r < 0.005:
        return max(0.0, r / 0.005)
    if r > 0.05:
        return max(0.0, 0.05 / r)
    return 1.0


def run_one(args):
    seed, mode, steps, perturb = args
    rng = np.random.default_rng(seed)

    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    age = np.full(pre.size, 100, np.int16)
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    xin = np.zeros(N, np.float32)
    trace = np.zeros(N, np.float32)

    # ---- 功能测量状态（v6）----
    nseg = max(2, steps // max(1, steps // 10))   # 10 段
    nseg = 10
    SSEG = max(1, steps // 10)
    nout = max(2, steps // OSEG)
    Wp = np.zeros((2, N), np.float32)     # s[t] -> o[t+1]
    s_prev = np.zeros(N, np.float32)
    o_hat = np.zeros(2, np.float32)
    o_prev = np.zeros(2, np.float64)
    a_prev = np.zeros(2, np.float64)
    d_prev = np.zeros(2, np.float64)
    freeze_at = steps // 2
    pred_err = np.zeros(nseg, np.float64)     # 冻结期误差（测试集）
    pred_n = np.zeros(nseg, np.int64)
    base_err = np.zeros(nseg, np.float64)     # 零模型：「猜不变」的误差（下界）
    base_n = np.zeros(nseg, np.int64)
    # 离线最优读出：正规方程累加（只在前半段），用来区分「没信息」与「在线没学好」
    XTX = np.zeros((N, N), np.float64)
    XTy = np.zeros((2, N), np.float64)
    Wls = None
    ls_err = np.zeros(nseg, np.float64)
    ls_n = np.zeros(nseg, np.int64)
    train_err = np.zeros(nseg, np.float64)    # 训练期误差（看读出层收敛）
    train_n = np.zeros(nseg, np.int64)
    # 动作增益累积（d = 相邻观测差，解释方 a）
    ga = np.zeros(nseg); ga2 = np.zeros(nseg)
    gd = np.zeros(nseg); gd2 = np.zeros(nseg)
    gad = np.zeros(nseg); gn = np.zeros(nseg, np.int64)
    out_seg = np.zeros(nout, np.float64)

    if perturb:
        v[0] = perturb          # 只动初始膜电位，不动 rng 序列

    w_f = 2.0 * np.pi / PERIOD_FAST
    w_s = 2.0 * np.pi / PERIOD_SLOW
    kf = 1.0 - 1.0 / TAU_W_F
    ks = 1.0 - 1.0 / TAU_W_S
    xf = 0.0
    xs = 0.0

    r_out = np.full(2, R_BASE, np.float64)
    a_all = np.zeros((steps, 2), np.float64)
    o_all = np.zeros((steps, 2), np.float32)

    u_in = rng.random((steps, N_IN), dtype=np.float32)
    u_wht = rng.random((steps, 2), dtype=np.float32)
    noise = rng.normal(0.0, W_NOISE, size=(steps, 2))
    u_rand = rng.uniform(-1.0, 1.0, size=(steps, 2)) if mode == 'randact' else None

    use_plast = mode != 'frozen'
    closed = mode != 'open'

    ema_rate = 0.02
    born = drop = 0
    prev = None
    tovs = []
    tail = []
    out_acc = np.zeros(2, np.float64)
    m_acc = 0.0
    m_cnt = 0
    elig_mean = 0.0
    tov_every = max(2000, steps // 5)
    t0 = time.perf_counter()

    for t in range(steps):
        # ---- 1) 观测 ----
        if mode == 'noise_in':
            o0 = float(u_wht[t, 0]) * 2.0 - 1.0
            o1 = float(u_wht[t, 1]) * 2.0 - 1.0
        else:
            o0 = float(np.tanh(xf))
            o1 = float(np.tanh(xs))
        o_all[t, 0] = o0
        o_all[t, 1] = o1

        # ---- 1b) 功能测量：用「上一步的预测」对「这一步的实测」打分（v6）----
        if t > 0:
            si = min(t // SSEG, nseg - 1)
            d0 = o0 - o_prev[0]
            d1 = o1 - o_prev[1]
            e0 = d0 - float(o_hat[0])
            e1 = d1 - float(o_hat[1])
            mse = 0.5 * (e0 * e0 + e1 * e1)
            base_err[si] += 0.5 * (d0 * d0 + d1 * d1)   # 零模型：预测「与上一步相同」
            base_n[si] += 1
            if t < freeze_at:
                train_err[si] += mse
                train_n[si] += 1
                ss = float(s_prev @ s_prev)
                if ss > 0:
                    upd = (PRED_LR * np.array([e0, e1]) / ss).astype(np.float32)
                    Wp += upd[:, None] * s_prev[None, :]
                    # 正规方程累加：用 d 与 s_prev
                    nz = np.nonzero(s_prev)[0]
                    if nz.size:
                        sv = s_prev[nz].astype(np.float64)
                        XTX[np.ix_(nz, nz)] += np.outer(sv, sv)
                        XTy[0, nz] += sv * d0
                        XTy[1, nz] += sv * d1
            else:
                pred_err[si] += mse
                pred_n[si] += 1
                if Wls is not None:
                    e2 = np.array([d0, d1]) - Wls @ s_prev
                    ls_err[si] += 0.5 * float(e2[0] * e2[0] + e2[1] * e2[1])
                    ls_n[si] += 1
            # 动作增益累积：a[t-1] 解释 d = o[t]-o[t-1]
            ga[si] += a_prev[0] + a_prev[1]
            ga2[si] += a_prev[0] ** 2 + a_prev[1] ** 2
            gd[si] += d0 + d1
            gd2[si] += d0 * d0 + d1 * d1
            gad[si] += a_prev[0] * d0 + a_prev[1] * d1
            gn[si] += 2
        o_prev[0] = o0
        o_prev[1] = o1

        # ---- 1c) 训练期结束时解一次离线最优读出（v6）----
        if t == freeze_at - 1:
            lam = 1e-2 * float(np.trace(XTX)) / N + 1e-6
            A = XTX + lam * np.eye(N)
            try:
                Wls = np.linalg.solve(A, XTy.T).T.astype(np.float32)
            except np.linalg.LinAlgError:
                Wls = None

        # ---- 2) 输入脉冲 ----
        p0 = IN_RATE * (1.0 + GAIN * o0)
        p1 = IN_RATE * (1.0 + GAIN * o1)
        xin.fill(0.0)
        xin[0:50] = u_in[t, 0:50] < p0
        xin[50:100] = u_in[t, 50:100] < p1

        # ---- 3) 网络 ----
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > V_TH) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = T_REF
        nsp = int(s.sum())
        ema_rate = 0.99 * ema_rate + 0.01 * (nsp / N)
        trace *= TRACE_DECAY
        trace += s

        # ---- 3b) 读出层预测下一步观测（v6）----
        o_hat = Wp @ s
        s_prev[:] = s
        out_seg[min(t // OSEG, nout - 1)] += float(s[OUT_IDX].sum())

        # ---- 4) 动作 ----
        r_out *= (1.0 - 1.0 / A_TAU)
        r_out += s[OUT_IDX] / A_TAU
        a = np.tanh((r_out - R_BASE) / R_SCALE)
        if mode == 'randact':
            a = u_rand[t].copy()
        a_all[t] = a
        a_prev[0] = float(a[0])
        a_prev[1] = float(a[1])

        # ---- 5) 世界 ----
        uf = AMP_F * float(np.cos(w_f * t))
        us = AMP_S * float(np.cos(w_s * t))
        if closed:
            uf += K_PHASE * a[0]
            us += K_PHASE * a[1]
        xf = kf * xf + uf + noise[t, 0]
        xs = ks * xs + us + noise[t, 1]

        # ---- 6) 结构可塑性 ----
        if use_plast and t > 0 and t % T_S == 0:
            lo = max(0, t - 200)
            aw = a_all[lo:t]
            agency = float(np.clip(max(aw[:, 0].std(), aw[:, 1].std()) / 0.2, 0.0, 1.0))
            m = steady_gate(ema_rate) * (0.2 + 0.8 * agency)
            m_acc += m
            m_cnt += 1

            if key.size:
                el = trace[pre] * trace[post]
                emax = float(el.max())
                if emax > 1e-9:
                    el /= emax
                    elig_mean += float(el.mean())
                    dw = ETA_POT * m * el
                    pos = w > 0
                    w[pos] = np.minimum(w[pos] + dw[pos] * (W_MAX_E - w[pos]), W_MAX_E)
                    neg = ~pos
                    w[neg] = np.maximum(w[neg] - dw[neg] * (W_MAX_I + w[neg]), -W_MAX_I)

            act = np.nonzero(trace > 0.5)[0]
            if act.size > MAX_ACTIVE:
                act = rng.choice(act, MAX_ACTIVE, replace=False)
            if act.size >= 2:
                na = act.size
                q_pre = np.repeat(act, na)
                q_post = np.tile(act, na)
                okk = q_pre != q_post
                q_pre, q_post = q_pre[okk], q_post[okk]
                qk = q_post.astype(np.int64) * N + q_pre
                found, _ = find(key, qk)
                npw, nqw = q_pre[~found], q_post[~found]
                if npw.size:
                    pick = rng.random(npw.size) < P_NEW
                    if pick.any():
                        npw, nqw = npw[pick], nqw[pick]
                        nw = np.where(npw < N_E, W_NEW_E, -W_NEW_I).astype(np.float32)
                        na_new = np.zeros(npw.size, np.int16)
                        pre, post, w, key, age = _sort(
                            np.concatenate([pre, npw.astype(np.int32)]),
                            np.concatenate([post, nqw.astype(np.int32)]),
                            np.concatenate([w, nw]),
                            np.concatenate([age, na_new]))
                        born += int(npw.size)

            if t % T_GLOBAL == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                okk = gi != gj
                gi, gj = gi[okk], gj[okk]
                if gi.size:
                    nw = np.where(gi >= N_E, -W_NEW_I, W_NEW_E).astype(np.float32)
                    na_new = np.zeros(gi.size, np.int16)
                    pre, post, w, key, age = _sort(
                        np.concatenate([pre, gi.astype(np.int32)]),
                        np.concatenate([post, gj.astype(np.int32)]),
                        np.concatenate([w, nw]),
                        np.concatenate([age, na_new]))
                    born += gi.size

            age += 1
            if key.size > BUDGET:
                d = key.size - BUDGET
                inh_m = pre >= N_E
                scale = np.abs(w) / np.where(inh_m, W_MAX_I, W_MAX_E)
                cand = age >= MIN_AGE
                if not cand.any():
                    cand[:] = True
                idx_c = np.nonzero(cand)[0]
                d = min(d, idx_c.size)
                if d > 0:
                    sub = idx_c[np.argpartition(scale[idx_c], d - 1)[:d]]
                    mm = np.ones(key.size, bool); mm[sub] = False
                    pre, post, w, key, age = pre[mm], post[mm], w[mm], key[mm], age[mm]
                    drop += int(d)

        # ---- 7) 记录 ----
        if t > 0 and t % tov_every == 0:
            if prev is not None:
                inter = np.intersect1d(prev, key, assume_unique=True).size
                tovs.append(1.0 - inter / max(prev.size + key.size - inter, 1))
            prev = key.copy()
        if t >= steps - 2000:
            tail.append(nsp)
            out_acc += s[OUT_IDX]

    deg_in = np.bincount(post, minlength=N)
    aw = a_all[max(0, steps - 2000):steps]
    ow = o_all[max(0, steps - 2000):steps]
    ca = 0.0
    for j in range(2):
        sj = float(aw[:, j].std())
        so = float(ow[:, j].std())
        if sj > 1e-9 and so > 1e-9:
            ca = max(ca, abs(float(np.corrcoef(aw[:, j], ow[:, j])[0, 1])))

    # ---- 功能测量汇总（v6）----
    pe = np.where(pred_n > 0, pred_err / np.maximum(pred_n, 1), np.nan)
    te = np.where(train_n > 0, train_err / np.maximum(train_n, 1), np.nan)
    base = np.where(base_n > 0, base_err / np.maximum(base_n, 1), np.nan)
    ls = np.where(ls_n > 0, ls_err / np.maximum(ls_n, 1), np.nan)
    # 动作增益：R^2 = (cov(a,d))^2 / (var(a) * var(d))，按段
    r2 = np.zeros(nseg)
    for k in range(nseg):
        n = gn[k]
        if n < 10:
            r2[k] = np.nan
            continue
        va = ga2[k] / n - (ga[k] / n) ** 2
        vd = gd2[k] / n - (gd[k] / n) ** 2
        cv = gad[k] / n - (ga[k] / n) * (gd[k] / n)
        r2[k] = (cv * cv) / (va * vd) if va > 1e-12 and vd > 1e-12 else 0.0

    ntail = max(len(tail), 1)

    # ---- 因果方向：a[t] 与 o[t+k] 的互相关（k>0 = 动作领先观测 = a 驱动 o）----
    LAG = 20
    m = min(20000, steps)
    sa = a_all[steps - m:steps]
    so = o_all[steps - m:steps]
    xc = np.zeros(2 * LAG + 1, np.float64)
    for k in range(-LAG, LAG + 1):
        if k >= 0:
            xa = sa[:m - k]
            xo = so[k:]
        else:
            xa = sa[-k:]
            xo = so[:m + k]
        c = 0.0
        for j in range(2):
            sja = float(xa[:, j].std())
            sjo = float(xo[:, j].std())
            if sja > 1e-9 and sjo > 1e-9:
                c = max(c, abs(float(np.corrcoef(xa[:, j], xo[:, j])[0, 1])))
        xc[k + LAG] = c

    return dict(
        seed=seed, mode=mode, perturb=perturb, E=int(key.size), born=born, drop=drop,
        rate=float(np.mean(tail) / N) if tail else 0.0,
        rate_std=float(np.std(tail) / N) if tail else 0.0,
        deg_e=float(deg_in[:N_E].mean()), deg_i=float(deg_in[N_E:].mean()),
        w_abs=float(np.abs(w).mean()), n_inh=int((pre >= N_E).sum()),
        tov_last=float(tovs[-1]) if tovs else float('nan'),
        a_std=float(max(aw[:, 0].std(), aw[:, 1].std())),
        a_obs_corr=ca,
        out_r0=float(out_acc[0] / ntail), out_r1=float(out_acc[1] / ntail),
        out_seg=out_seg, pred=pe, train=te, r2=r2, base=base, xc=xc, pred_ls=ls,
        elapsed=time.perf_counter() - t0)


def _fmt_curve(a):
    return ' '.join('nan' if (x != x) else ('%.4f' % x) for x in a)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    steps = int(argv[0]) if len(argv) > 0 else 50000
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['closed', 'open', 'frozen']

    if '--diag' in flags:
        # 因果方向：动作是驱动观测，还是反被观测驱动？
        print('=' * 110)
        print('因果方向诊断  N=%d  steps=%d  种子=%d  （a[t] 与 o[t+k] 的互相关，k>0 = 动作领先）' % (N, steps, nseed))
        print('=' * 110)
        with Pool(nproc) as pool:
            rc = pool.map(run_one, [(s, 'closed', steps, 0.0) for s in range(nseed)])
            ro = pool.map(run_one, [(s, 'open', steps, 0.0) for s in range(nseed)])
        XC = np.array([r['xc'] for r in rc])
        XO = np.array([r['xc'] for r in ro])
        LAG = 20
        mc = XC.mean(axis=0)
        mo = XO.mean(axis=0)
        print('%-5s %9s %9s' % ('k', 'closed', 'open'))
        for i, k in enumerate(range(-LAG, LAG + 1)):
            mark = ''
            if k == 0:
                mark = '  <- 同期'
            print('%-5d %9.4f %9.4f%s' % (k, mc[i], mo[i], mark))
        ic = int(np.argmax(mc)); io_ = int(np.argmax(mo))
        print('  closed 峰值 k=%d 值 %.4f  |  open 峰值 k=%d 值 %.4f' % (
            ic - LAG, mc[ic], io_ - LAG, mo[io_]))
        print('  负延迟侧均值(k<0) %.4f  正延迟侧均值(k>0) %.4f' % (
            mc[:LAG].mean(), mc[LAG + 1:].mean()))
        print('  判据：峰值在 k>0 且正侧均值 > 负侧 → a 驱动 o（闭环因果成立）；')
        print('        峰值在 k<0 或两侧对称 → 主要是 o 驱动 a（闭环只建立了反方向）')
        print()
        return

    if '--div' in flags:
        # 前史因果：同 seed / 同输入，只改初始膜电位
        print('=' * 110)
        print('前史因果  N=%d  预算=%d  steps=%d  种子=%d  扰动=%g' % (N, BUDGET, steps, nseed, PERTURB))
        print('=' * 110)
        with Pool(nproc) as pool:
            base = pool.map(run_one, [(s, 'closed', steps, 0.0) for s in range(nseed)])
            pert = pool.map(run_one, [(s, 'closed', steps, PERTURB) for s in range(nseed)])
        print('%-5s %8s %8s %8s %8s %8s' % ('seed', 'd(0-1k)', 'd(1-5k)', 'd(5-10k)', 'base_r', 'pert_r'))
        keep = []
        for b, p in zip(base, pert):
            d = p['out_seg'] - b['out_seg']
            half = max(1, len(d) // 2)
            q1 = max(1, min(5, len(d)))
            print('%-5d %8.1f %8.1f %8.1f %8.3f %8.3f' % (
                b['seed'], np.abs(d[:10]).sum(), np.abs(d[:q1 * 10]).sum(), np.abs(d).sum(),
                b['out_r0'], p['out_r0']))
            keep.append(np.abs(d).sum() / max(1, len(d)))
        K = np.array(keep)
        print('  跨种子：平均每段偏差 %.2f  |  偏差为 0 的种子数 %d/%d' % (
            K.mean(), int((K < 1e-9).sum()), len(K)))
        print('  判据：偏差长期不为 0 → 内部状态对行为有因果（**只排除「无记忆」，不等于学到**）')
        print('  对照阅读：base/pert 的 out_r0 应接近（只改初始 1e-3，量的差异应在动态上而非均值上）')
        print()
        return

    heads = '%-5s %6s %6s %7s %7s %7s %8s %8s' % ('seed', 'E', 'rate%', 'std%', 'a-o', 'out_r0', 'pred_h1', 'pred_h2')
    for mode in modes:
        print('=' * 130)
        print('grow_func v6  mode=%-9s  N=%d  E0=%d->预算=%d  steps=%d  种子=%d  (前 %d 步训练读出层，之后冻结)' % (
            mode, N, E0_INIT, BUDGET, steps, nseed, steps // 2))
        print('=' * 130)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, steps, 0.0) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print(heads)
        for r in res:
            p = r['pred']
            h = len(p) // 2
            h1 = np.nanmean(p[:h]) if h else np.nan
            h2 = np.nanmean(p[h:]) if h else np.nan
            print('%-5d %6d %6.2f %7.2f %7.3f %7.3f %8.5f %8.5f' % (
                r['seed'], r['E'], r['rate'] * 100, r['rate_std'] * 100,
                r['a_obs_corr'], r['out_r0'], h1, h2))

        P = np.array([r['pred'] for r in res])
        T = np.array([r['train'] for r in res])
        B = np.array([r['base'] for r in res])
        LS = np.array([r['pred_ls'] for r in res])
        R2 = np.array([r['r2'] for r in res])
        with np.errstate(invalid='ignore'):
            pm = np.nanmean(P, axis=0)
            tm = np.nanmean(T, axis=0)
            bm = np.nanmean(B, axis=0)
            lm = np.nanmean(LS, axis=0)
            rm = np.nanmean(R2, axis=0)
        print('  预测误差(冻结期,逐段): %s' % _fmt_curve(pm))
        print('  零模型(猜不变,逐段): %s' % _fmt_curve(bm))
        print('  比值 pred/base(逐段): %s' % _fmt_curve(np.where(bm > 0, pm / bm, np.nan)))
        print('  离线最优读出(逐段)  : %s' % _fmt_curve(lm))
        print('  比值 ls/base(逐段)  : %s' % _fmt_curve(np.where(bm > 0, lm / bm, np.nan)))
        print('  预测误差(训练期,逐段): %s' % _fmt_curve(tm))
        print('  动作增益 R^2 (逐段)  : %s' % _fmt_curve(rm))
        h = len(pm) // 2
        with np.errstate(invalid='ignore'):
            f1 = np.nanmean(pm[:h]); f2 = np.nanmean(pm[h:])
            q1 = np.nanmean(pm[:max(1, h // 2)]); q2 = np.nanmean(pm[-max(1, h // 2):])
        print('  冻结期前半 %.5f  后半 %.5f  比值 %.3f   |  最前 1/4 %.5f  最后 1/4 %.5f  比值 %.3f' % (
            f1, f2, f2 / f1 if f1 else float('nan'), q1, q2, q2 / q1 if q1 else float('nan')))
        with np.errstate(invalid='ignore'):
            print('  动作增益：前四段 %.4f  后四段 %.4f' % (np.nanmean(rm[:4]), np.nanmean(rm[-4:])))
        print('  耗时 %.1fs' % wall)
        print()


if __name__ == '__main__':
    main()
