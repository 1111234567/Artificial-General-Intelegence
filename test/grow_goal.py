# grow_goal.py - 闭环：目标 / 成败 / 三因子规则（按 design.md §7.3 实现）
#
# 形态：本体 = 连接 + 一条公式；环境 = 位置+目标（有成败）；闭环 = 做 -> 看结果 -> 改连接。
#
# ── v2（本版，2026-10-24）：删 4 个机制 + 补 1 项 ──
#
# 删（每个都有数据证明它不产生信息，见 CONSTANTS.md §5）：
#   1) 2x2 线性预测器（PRED_K / PRED_LR / Wp）
#      证据：pred 比在**完全不学的 frozen** 组也降 0.065，与 C_full 的 0.087 同量级
#      -> 下降来自预测器自己在线学，不是网络在学；而且它是外挂，不是世界模型
#   2) 能动性 corr(a, dx)
#      证据：A_noagency（能动性恒 1）9.5->9.7 vs A_dense 9.3->8.1，无差异
#   3) 距离桥接（dense vs sparse）
#      证据：9.3->8.1 vs 10.2->10.0，无差异
#   4) 全局随机边（P_GLOBAL / T_GLOBAL）
#      证据：与 P_NEW 功能重叠，从未单独验证
#
# 补（设计里明确要、v1 漏掉的一项）：
#   `(a_post - θ)`——BCM 滑动阈值。实现：每个神经元一个慢 EMA `theta`，
#   `Δw = η · m · trace[pre] · (trace[post] - theta[post])`。可正可负 -> 双向可塑。
#   v1 是 `trace[pre] * trace[post]`（恒 >= 0）-> 只增强不削弱，只能靠预算淘汰来缩。
#
# 同时删掉一个无依据的基底：v1 的 `m = m_steady * (0.2 + 0.8 * m_agency * m_value)`，
# 那个 0.2 是「失败时也弱强化」的拍值。现在 `m = m_steady * m_value`（门控就该是 0 或正）。
#
# 参数：37（v0）-> 15（只消重复）-> **14**（本版：删 PRED_K/PRED_LR，加 TAU_THETA）。
# 机制：6 -> 3（强化 / 预算 / 新建）+ 1 项按设计补上（BCM 双向）。
#
# 模式（mode）：
#   A          主体：成败当 m（成功 -> m=1，否则 0）
#   A_frozen   对照：结构不动（不强化、不生长）
#   A_fixed    诊断：目标固定为 (0.5,-0.5)（跨象限，排除「恒定动作碰巧命中」）
#   A_probe    诊断：从状态线性读出 (g-x)，看信息在不在
#   T_seq      步 3：先 task A（右上）再 task B（左下）——测第二次快多少
#   T_fresh    步 3 对照：直接跑 task B
#
# 已知局限（不要把结论放大）：
#   ① 环境是人工世界，不是 design.md §3.6 的「本机命令 3 秒 / 测试 10 秒」。
#   ② m 极其稀疏（成功率 ~10%，每个成功只在 T_S 步内给信号）——这是设计的本意（§3.1：
#      延迟 = 任务结束才有），不是缺陷；要缓解得靠「桥接刺激」（§14.6），而试过的距离桥接无效。
#   ③ 动作是 4 个离散，测的是「选不选对方向」，不是「控得准」。

import sys
import time
import numpy as np
from multiprocessing import Pool

from net import (N, N_E, N_IN, DECAY_V, V_TH, T_REF,
                 W_E, W_I_RATIO, W_MAX_E, W_MAX_I, W_NEW_E, W_NEW_I,
                 BUDGET, E0_INIT, T_S, TRACE_DECAY, ETA_POT, P_NEW,
                 MAX_ACTIVE, MIN_AGE, _sort, find, steady_gate, cheb)

# ---- 接口（本实验特有）----
N_CH = N_IN // 4          # [派生] 4 通道：x0 / x1 / g0 / g1
IN_RATE = 0.02            # [拍] 输入放电率基准
GAIN = 0.8                # [拍] 观测 -> 输入率的调制深度
OUT_IDX = np.array([500, 501])   # [拍] 输出神经元（回路里的普通节点）
A_TAU = 20.0              # [拍] 输出放电率 EMA
BASE_TAU = 100.0          # [拍] 动作基线的慢 EMA（防动作饱和）
TAU_THETA = 200.0         # [拍] BCM 滑动阈值的时间常数（比迹慢 10 倍，§8.2「不能响应过快」）

# ---- 世界与任务（环境设计，不是机制参数）----
T_ACT = 20                # [需求] 动作窗口（行为是一个事件窗口，design.md §14.6）
T_EP = 120                # [需求] episode 上限（120 x 0.02 = 2.4 位移能力）
K_ACT = 0.02              # [需求] 每步动作推力
TOL = 0.08                # [需求] 成功判据（切比雪夫距离）
AMP_D = 0.01              # [需求] 外源扰动（只有动作的一半）
PERIOD_D = 40.0           # [需求] 扰动周期
W_NOISE = 0.002           # [需求] 世界噪声


def run_one(args):
    seed, mode, n_ep = args
    rng = np.random.default_rng(seed)

    # ---- 结构初始化（过度生产 -> 硬预算裁剪）----
    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    # 输入区不接受反向边（grow_talk 缺陷 5：扇出边连回输入区 -> 输入自发放电）
    m_in = post < N_IN
    pre, post = pre[~m_in], post[~m_in]
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    age = np.full(pre.size, 100, np.int16)
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    xin = np.zeros(N, np.float32)
    trace = np.zeros(N, np.float32)
    theta = np.zeros(N, np.float32)      # BCM 滑动阈值（§7.3 的 (a_post - θ)）

    r_out = np.full(2, 0.0, np.float64)
    base = np.full(2, 0.0, np.float64)
    a = np.zeros(2, np.float64)

    ema_rate = 0.02
    align_acc = 0.0; align_cnt = 0
    a_sum = np.zeros(2)
    a_switch = 0; a_switch_n = 0
    x_end_sum = np.zeros(2)
    probe_S = [] if mode == 'A_probe' else None
    probe_D = [] if mode == 'A_probe' else None
    m_steady_acc = m_value_acc = m_acc = 0.0
    m_cnt = 0
    th_mean_acc = 0.0

    succ_recent = 0        # 刚成功时的倒计时（只在成功窗口附近给信号）

    ep_succ = np.zeros(n_ep, np.float32)
    ep_steps = np.zeros(n_ep, np.float32)
    born = drop = 0
    prev_keys = None
    tov = []

    tot = n_ep * T_EP + 8
    u_in = rng.random((tot, N_IN), dtype=np.float32)
    noise = rng.normal(0.0, W_NOISE, size=(tot, 2))
    x0_all = rng.uniform(-0.6, 0.6, size=(n_ep, 2))
    g_all = rng.uniform(-0.8, 0.8, size=(n_ep, 2))

    use_plast = mode != 'A_frozen'
    t_glob = 0
    t0 = time.perf_counter()

    # task 划分（步 3）：前一半 task A，后一半 task B
    if mode.startswith('T_'):
        half = n_ep // 2
        g_all[:half] = g_all[:half] * 0.5 + np.array([0.5, 0.5])     # task A：右上
        g_all[half:] = g_all[half:] * 0.5 + np.array([-0.5, -0.5])   # task B：左下
        if mode == 'T_fresh':
            g_all[:half] = g_all[half:].copy()   # 直接跑 B（对照）
    elif mode == 'A_fixed':
        g_all[:] = np.array([0.5, -0.5])

    for ep in range(n_ep):
        x = x0_all[ep].copy()
        g = g_all[ep].copy()
        succ = 0
        succ_recent = 0
        ep_step_done = T_EP

        for k in range(T_EP):
            t = t_glob
            # ---- 1) 输入脉冲（4 通道：x0 / x1 / g0 / g1）----
            p0 = IN_RATE * (1.0 + GAIN * x[0])
            p1 = IN_RATE * (1.0 + GAIN * x[1])
            q0 = IN_RATE * (1.0 + GAIN * g[0])
            q1 = IN_RATE * (1.0 + GAIN * g[1])
            xin.fill(0.0)
            xin[0:N_CH] = u_in[t, 0:N_CH] < p0
            xin[N_CH:2 * N_CH] = u_in[t, N_CH:2 * N_CH] < p1
            xin[2 * N_CH:3 * N_CH] = u_in[t, 2 * N_CH:3 * N_CH] < q0
            xin[3 * N_CH:4 * N_CH] = u_in[t, 3 * N_CH:4 * N_CH] < q1

            # ---- 2) 网络 ----
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
            # BCM 滑动阈值（慢 EMA，§8.2：不能响应过快）
            # 注意：慢平均的对象是 **trace**（迹，τ=20）而不是 s（单步放电）。
            # 取 s 会得到一个量级小 20 倍的 theta（0.005 vs 0.1），`trace - theta` 几乎恒正
            # -> BCM 项等于没起作用（2026-10-24 实测：theta 均值 0.006，trace 稳态 0.1）。
            theta *= (1.0 - 1.0 / TAU_THETA)
            theta += trace / TAU_THETA

            # ---- 3) 输出与动作（每 T_ACT 步采一次符号）----
            r_out *= (1.0 - 1.0 / A_TAU)
            r_out += s[OUT_IDX] / A_TAU
            base *= (1.0 - 1.0 / BASE_TAU)
            base += r_out / BASE_TAU
            if k % T_ACT == 0:
                a_new = np.where(r_out > base, 1.0, -1.0)
                if k > 0:
                    a_switch_n += 1
                    if not np.array_equal(a_new, a):
                        a_switch += 1
                a = a_new
                a_sum += a
                sgn = np.sign(g - x)
                sgn[sgn == 0.0] = 1.0
                align_acc += float(np.dot(a, sgn))
                align_cnt += 1

            # ---- 4) 世界推进 ----
            drift = AMP_D * float(np.cos(2.0 * np.pi * t_glob / PERIOD_D))
            x = x + K_ACT * a + drift + noise[t_glob]
            x = np.clip(x, -1.0, 1.0)

            # ---- 5) 任务判定 ----
            d = cheb(x, g)
            if mode == 'A_probe' and k % 5 == 0:
                probe_S.append(s.copy())
                probe_D.append(g - x)
            if d < TOL and succ == 0:
                succ = 1
                ep_step_done = k + 1
                succ_recent = T_S

            # ---- 6) 结构可塑性（每 T_S 步）——淘汰始终执行，只关「强化 + 生长」----
            if t_glob > 0 and t_glob % T_S == 0:
                # 价值：只在这个成功窗口附近给信号（不是整个 episode 都算）
                m_value = 1.0 if succ_recent > 0 else 0.0
                succ_recent = max(0, succ_recent - T_S)

                m_steady = steady_gate(ema_rate)
                m = m_steady * m_value
                if not use_plast:
                    m = 0.0

                m_steady_acc += m_steady
                m_value_acc += m_value
                m_acc += m
                m_cnt += 1
                th_mean_acc += float(theta.mean())

                if use_plast and m > 0 and key.size:
                    # Δw = η · m · a_pre · (a_post − θ)   —— design.md §7.3
                    el = trace[pre] * (trace[post] - theta[post])
                    emax = float(np.abs(el).max())
                    if emax > 1e-9:
                        el /= emax
                        dw = (ETA_POT * m * el).astype(np.float32)
                        cap = np.where(w > 0, W_MAX_E, W_MAX_I)
                        w = w + dw * (cap - np.abs(w))
                        np.clip(w, -W_MAX_I, W_MAX_E, out=w)

                # 生长（活跃区，固定小权重 + 年龄保护）——frozen 不生长
                act = np.nonzero(trace > 0.5)[0] if use_plast else np.zeros(0, np.int64)
                if act.size > MAX_ACTIVE:
                    act = rng.choice(act, MAX_ACTIVE, replace=False)
                if act.size >= 2:
                    na = act.size
                    q_pre = np.repeat(act, na)
                    q_post = np.tile(act, na)
                    okk = (q_pre != q_post) & (q_post >= N_IN)
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

                # 硬预算淘汰（接口边受保护）
                age += 1
                if key.size > BUDGET:
                    dd = key.size - BUDGET
                    ist_out = (post == OUT_IDX[0]) | (post == OUT_IDX[1])
                    scale = np.abs(w) / np.where(pre >= N_E, W_MAX_I, W_MAX_E)
                    cand = (age >= MIN_AGE) & (~ist_out)
                    if not cand.any():
                        cand = age >= MIN_AGE
                    if cand.any():
                        idx_c = np.nonzero(cand)[0]
                        dd = min(dd, idx_c.size)
                        if dd > 0:
                            sub = idx_c[np.argpartition(scale[idx_c], dd - 1)[:dd]]
                            mm = np.ones(key.size, bool); mm[sub] = False
                            pre, post, w, key, age = pre[mm], post[mm], w[mm], key[mm], age[mm]
                            drop += int(dd)

            t_glob += 1

        ep_succ[ep] = succ
        ep_steps[ep] = ep_step_done if succ else T_EP
        x_end_sum += x

        if ep % 50 == 0:
            if prev_keys is not None:
                inter = np.intersect1d(prev_keys, key, assume_unique=True).size
                tov.append(1.0 - inter / max(prev_keys.size + key.size - inter, 1))
            prev_keys = key.copy()

    # 步 3 的统计口径：phase2 = 「第二个任务」的前 1/6
    k6 = max(1, n_ep // 6)
    if mode == 'T_seq':
        p2 = slice(n_ep // 2, n_ep // 2 + k6)
    elif mode == 'T_fresh':
        p2 = slice(0, k6)
    else:
        p2 = slice(n_ep - k6, n_ep)

    deg_in = np.bincount(post, minlength=N)
    r2 = float('nan'); r2s = float('nan')
    if mode == 'A_probe' and probe_S is not None and len(probe_S) > 200:
        S = np.asarray(probe_S, np.float64)
        D = np.asarray(probe_D, np.float64)
        S1 = np.hstack([S, np.ones((S.shape[0], 1))])
        Wt, *_ = np.linalg.lstsq(S1, D, rcond=None)
        P = S1 @ Wt
        ss_res = ((D - P) ** 2).sum(axis=0)
        ss_tot = ((D - D.mean(axis=0)) ** 2).sum(axis=0)
        r2s = float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))
        Sg = (D > 0).astype(np.float64)
        Ws, *_ = np.linalg.lstsq(S1, Sg, rcond=None)
        Ps = S1 @ Ws
        pred_sign = (Ps > 0.5).astype(np.float64)
        r2 = float(np.mean((pred_sign == Sg).mean(axis=0)))

    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        succ_first=float(np.mean(ep_succ[:max(1, n_ep // 6)])),
        succ_last=float(np.mean(ep_succ[-max(1, n_ep // 6):])),
        step_first=float(np.mean(ep_steps[:max(1, n_ep // 6)])),
        step_last=float(np.mean(ep_steps[-max(1, n_ep // 6):])),
        phase2_succ=float(np.mean(ep_succ[p2])),
        phase2_step=float(np.mean(ep_steps[p2])),
        m_steady=float(m_steady_acc / max(m_cnt, 1)),
        m_value=float(m_value_acc / max(m_cnt, 1)),
        m_mean=float(m_acc / max(m_cnt, 1)),
        theta_mean=float(th_mean_acc / max(m_cnt, 1)),
        tov_last=float(tov[-1]) if tov else float('nan'),
        out_deg=int(deg_in[OUT_IDX].mean()),
        align=float(align_acc / max(align_cnt, 1) / 2.0),
        a_mean=float(np.abs(a_sum / max(align_cnt, 1)).mean()),
        a_switch=float(a_switch / max(a_switch_n, 1)),
        x_end=float(np.abs(x_end_sum / max(n_ep, 1)).mean()),
        r2_val=r2s, r2_sign=r2,
        elapsed=time.perf_counter() - t0)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 200
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['A']

    for mode in modes:
        print('=' * 132)
        print('grow_goal v2  mode=%-10s  N=%d  预算=%d  episode=%d×%d步  种子=%d' % (
            mode, N, BUDGET, n_ep, T_EP, nseed))
        print('=' * 132)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, n_ep) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %8s %7s %7s' % (
            'seed', 'E', 'rate%', 'n_inh', 'succ早', 'succ晚', 'step早', 'step晚',
            'out_deg', 'align'))
        for r in res:
            print('%-5d %6d %7.2f %6d %6.1f%% %6.1f%% %8.1f %8.1f %7d %7.3f' % (
                r['seed'], r['E'], r['rate'] * 100, r['n_inh'],
                r['succ_first'] * 100, r['succ_last'] * 100,
                r['step_first'], r['step_last'], r['out_deg'], r['align']))

        S1 = np.array([r['succ_first'] for r in res]) * 100
        S2 = np.array([r['succ_last'] for r in res]) * 100
        T1 = np.array([r['step_first'] for r in res])
        T2 = np.array([r['step_last'] for r in res])
        M = np.array([[r['m_steady'], r['m_value'], r['m_mean']] for r in res])
        AL = np.array([r['align'] for r in res])
        print('  跨种子: 成功率 %.1f%% -> %.1f%% (%+.1f pp)   步数 %.1f -> %.1f (%+.1f)' % (
            S1.mean(), S2.mean(), S2.mean() - S1.mean(), T1.mean(), T2.mean(), T2.mean() - T1.mean()))
        print('          m: 稳态 %.2f / 价值 %.2f / 合成 %.3f   theta均值 %.4f' % (
            M[:, 0].mean(), M[:, 1].mean(), M[:, 2].mean(),
            np.mean([r['theta_mean'] for r in res])))
        print('          align（动作朝目标）%.3f ± %.3f   |a|均值 %.3f  切换率 %.3f  |x_end| %.3f' % (
            AL.mean(), AL.std(),
            np.mean([r['a_mean'] for r in res]),
            np.mean([r['a_switch'] for r in res]),
            np.mean([r['x_end'] for r in res])))
        if mode == 'A_probe':
            RV = np.array([r['r2_val'] for r in res])
            RS = np.array([r['r2_sign'] for r in res])
            print('          探针：状态线性读出 (g-x) 的 R² = %.3f ± %.3f；符号准确率 = %.3f ± %.3f（0.5=无信息）' % (
                np.nanmean(RV), np.nanstd(RV), np.nanmean(RS), np.nanstd(RS)))
        if mode.startswith('T_'):
            P2S = np.array([r['phase2_succ'] for r in res]) * 100
            P2T = np.array([r['phase2_step'] for r in res])
            print('          步 3：phase2 前 1/6 成功率 %.1f%%（中位 %.1f%%±%.1f）、步数 %.1f' % (
                P2S.mean(), np.median(P2S), P2S.std(), P2T.mean()))
        print('          耗时 %.1fs' % wall)
        print()


if __name__ == '__main__':
    main()
