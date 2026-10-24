# grow_goal.py - 闭环：目标 / 成败 / 三层 m（步 1-3 一起做）
#
# 起因（2026-10-24 场次四）：`grow_world.py` 已建成闭环管道，但三个量为 0：
#   m（外部信号）/ 压力（环境后果）/ 隔离（写入局部性）。本文件补前两个。
#
# 与 grow_world 的形态差异（所以是新文件，不是改旧文件）：
#
#   | 项     | grow_world                    | grow_goal                          |
#   |--------|-------------------------------|------------------------------------|
#   | 世界   | 两个漏积分变量 + 纯周期驱动      | 位置 + 目标 + 扰动（有成败）          |
#   | 任务   | 无                             | episode：随机初态 + 随机目标           |
#   | 动作   | 2 维连续（tanh）                | 4 个离散（2 输出的符号对）             |
#   | m      | 稳态门 × 动作方差                | 稳态门 × 能动性 × 价值                |
#   | 判据   | rate / turnover / a-o 相关      | 预测比 / 成功率 / 第二次快多少          |
#
# 三层 m（design.md §14.6）：
#   m = 稳态门 × (0.2 + 0.8 × 能动性 × 价值)
#   稳态门：rate 落在 0.5%~5% 之外时压下去（沿用 grow_world 的 steady_gate）
#   能动性：corr(a, Δx) —— 动作与世界变化的耦合（**不是动作方差**；方差只说明动得多）
#   价值：步 1 = 预测误差的相对改善（C 打底）；步 2 = 任务成败（A 定方向）
#
# 三个 stage（`mode` 前缀区分）：
#   C_*  步 1：m 用预测误差。判据 = pred_ratio 随时间下降（世界模型有没有学起来）
#        C_full / C_novalue / C_shuffle / C_frozen
#   A_*  步 2：m 用任务成败。判据 = 成功率（前 100 ep → 后 100 ep）
#        A_dense（距离桥接）/ A_sparse（只成败）/ A_noagency（能动性恒 1）/ A_frozen
#   T_*  步 3：两个 task 顺序跑 vs 直接跑 task B。判据 =「第二次快多少」
#        T_seq / T_fresh
#
# 已知局限（写在这里，防止后人把结论放大）：
#   ① 环境是**人工的世界**，不是 design.md §3.6 说的本机环境。压力是真的（成败由客观判据给），
#      但它不是「本机命令 3 秒 / 测试 10 秒」那种可逆环境。
#   ② 预测器是 2×2 线性，能力很弱；pred_ratio 不降时先怀疑它，再怀疑「状态里没信息」。
#   ③ 动作是 4 个离散，不是连续控制；测的是「能不能选对方向」，不是「控得准」。
#   ④ 目标与位置都进输入（各 2 个通道），所以任务**不需要记忆**——它测的是「会不会用当前信息」。

import sys
import time
import numpy as np
from multiprocessing import Pool

from net import (N, N_E, N_IN, DECAY_V, V_TH, T_REF,
                 W_E, W_I_RATIO, W_MAX_E, W_MAX_I, W_NEW_E, W_NEW_I,
                 BUDGET, E0_INIT, T_S, TRACE_DECAY, ETA_POT, P_NEW,
                 P_GLOBAL, T_GLOBAL,
                 MAX_ACTIVE, MIN_AGE, _sort, find, steady_gate, cheb)

# ---- 接口（本实验特有）----
N_CH = N_IN // 4          # [派生] 4 通道：x0 / x1 / g0 / g1
IN_RATE = 0.02            # [拍] 输入放电率基准
GAIN = 0.8                # [拍] 观测 -> 输入率的调制深度
OUT_IDX = np.array([500, 501])   # [拍] 输出神经元（回路里的普通节点）
A_TAU = 20.0              # [拍] 输出放电率 EMA
BASE_TAU = 100.0          # [拍] 动作基线的慢 EMA（防动作饱和）

# ---- 世界与任务（环境设计，不是机制参数）----
T_ACT = 20                # [需求] 动作窗口（行为是一个事件窗口，design.md §14.6）
T_EP = 120                # [需求] episode 上限（120 x 0.02 = 2.4 位移能力）
K_ACT = 0.02              # [需求] 每步动作推力
TOL = 0.08                # [需求] 成功判据（切比雪夫距离）
AMP_D = 0.01              # [需求] 外源扰动（只有动作的一半）
PERIOD_D = 40.0           # [需求] 扰动周期
W_NOISE = 0.002           # [需求] 世界噪声

# ---- 预测器（C 步用；**已知是外挂**——2x2 线性，不代表世界模型，见 CONSTANTS.md）----
PRED_K = T_ACT            # [派生] 预测 T_ACT 步后
PRED_LR = 0.02            # [拍] 在线 LMS 学习率


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

    r_out = np.full(2, 0.0, np.float64)
    base = np.full(2, 0.0, np.float64)
    a = np.zeros(2, np.float64)

    ema_rate = 0.02
    # 能动性累积（每个动作窗口结束时结算）
    s_adx = 0.0; s_aa = 0.0; s_dxdx = 0.0
    align_acc = 0.0; align_cnt = 0
    a_sum = np.zeros(2)          # 诊断：动作是否饱和
    a_switch = 0; a_switch_n = 0
    x_end_sum = np.zeros(2)      # 诊断：episode 末尾的位置
    probe_S = [] if mode == 'A_probe' else None   # 诊断：状态能否线性读出 (g-x)
    probe_D = [] if mode == 'A_probe' else None
    m_steady_acc = m_agency_acc = m_value_acc = m_acc = 0.0
    m_cnt = 0

    # 预测器
    Wp = np.zeros((2, 2))
    e_ema = 0.0; eb_ema = 0.0

    d_ts_prev = 1.0        # 上一个结构更新时刻的 d（A_dense 用：比「过去 20 步的净位移」）
    succ_recent = 0        # 刚成功时的倒计时（A_sparse 用：只在成功附近给信号）

    # 统计
    ep_succ = np.zeros(n_ep, np.float32)
    ep_steps = np.zeros(n_ep, np.float32)
    pred_ratio_hist = np.zeros(n_ep, np.float32)   # 每个 episode 的 e/e_base 均值
    born = drop = 0
    prev_keys = None
    tov = []
    tail_nsp = []

    # 预生成随机数（spawn 安全；也避免每步调 rng）
    tot = n_ep * T_EP + 8
    u_in = rng.random((tot, N_IN), dtype=np.float32)
    noise = rng.normal(0.0, W_NOISE, size=(tot, 2))
    o_hist = np.zeros((tot, 2), np.float32)      # 世界状态历史（预测器要用 t-PRED_K）
    x0_all = rng.uniform(-0.6, 0.6, size=(n_ep, 2))
    g_all = rng.uniform(-0.8, 0.8, size=(n_ep, 2))
    if mode == 'C_shuffle' or mode == 'A_shuffle':
        # m 打乱对照：预生成一串 [0,1] 的随机数，按步取
        m_rand = rng.random(tot)
    else:
        m_rand = None

    use_plast = not (mode.endswith('_frozen') or mode == 'C_frozen' or mode == 'A_frozen')
    t_glob = 0
    t0 = time.perf_counter()

    # task 划分（步 3）：前一半 task A，后一半 task B
    if mode.startswith('T_'):
        half = n_ep // 2
        g_all[:half] = g_all[:half] * 0.5 + np.array([0.5, 0.5])   # task A：右上
        g_all[half:] = g_all[half:] * 0.5 + np.array([-0.5, -0.5])  # task B：左下
        if mode == 'T_fresh':
            g_all[:half] = g_all[half:].copy()   # 直接跑 B（对照）
    elif mode == 'A_fixed':
        # 定位实验：目标固定且跨象限（“恒定向右上一路推”碰巧命中的概率最低）
        g_all[:] = np.array([0.5, -0.5])

    for ep in range(n_ep):
        x = x0_all[ep].copy()
        g = g_all[ep].copy()
        d_prev = cheb(x, g)
        d_ts_prev = d_prev
        succ = 0
        succ_recent = 0
        ep_step_done = T_EP
        e_sum = 0.0; eb_sum = 0.0; e_cnt = 0

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
                # 诊断：动作是否朝目标（a · sign(g-x)）；正 = 会用目标信息
                sgn = np.sign(g - x)
                sgn[sgn == 0.0] = 1.0
                align_acc += float(np.dot(a, sgn))
                align_cnt += 1

            # ---- 4) 世界推进 ----
            drift = AMP_D * float(np.cos(2.0 * np.pi * t_glob / PERIOD_D))
            x_prev = x.copy()
            x = x + K_ACT * a + drift + noise[t_glob]
            x = np.clip(x, -1.0, 1.0)
            dx = x - x_prev

            # 能动性累积（动作与位移的相关）
            s_adx += float(a[0] * dx[0] + a[1] * dx[1])
            s_aa += float(a[0] ** 2 + a[1] ** 2)
            s_dxdx += float(dx[0] ** 2 + dx[1] ** 2)

            # ---- 5) 预测器（步 1 的价值来源）----
            if t_glob >= PRED_K:
                o_prev = o_hist[t_glob - PRED_K]
                o_now = x
                o_hat = Wp @ o_prev
                err = o_hat - o_now
                Wp -= PRED_LR * np.outer(err, o_prev)
                e = float(np.mean(err ** 2))
                eb = float(np.mean((o_prev - o_now) ** 2))
                e_ema = 0.98 * e_ema + 0.02 * e
                eb_ema = 0.98 * eb_ema + 0.02 * eb
                e_sum += e; eb_sum += eb; e_cnt += 1
            o_hist[t_glob] = x

            # ---- 6) 任务判定 ----
            d = cheb(x, g)
            if mode == 'A_probe' and k % 5 == 0:
                probe_S.append(s.copy())
                probe_D.append(g - x)
            if d < TOL and succ == 0:
                succ = 1
                ep_step_done = k + 1
                succ_recent = T_S

            # ---- 7) 结构可塑性（每 T_S 步）——淘汰始终执行，只关「强化 + 生长」----
            if t_glob > 0 and t_glob % T_S == 0:
                # 能动性 = corr(a, dx)（0~1）；不是动作方差
                if s_aa > 1e-9 and s_dxdx > 1e-9:
                    m_agency = float(np.clip(s_adx / np.sqrt(s_aa * s_dxdx), 0.0, 1.0))
                else:
                    m_agency = 0.0
                s_adx = s_aa = s_dxdx = 0.0
                if mode == 'A_noagency':
                    m_agency = 1.0

                # 价值
                if mode.startswith('C'):
                    if e_cnt > 0 and eb_ema > 1e-12:
                        ratio = max(e_ema / eb_ema, 1e-9)
                        m_value = float(np.clip(1.5 - ratio, 0.0, 1.0))
                    else:
                        m_value = 0.0
                    if mode == 'C_novalue':
                        m_value = 1.0
                elif mode.startswith('A') or mode.startswith('T'):
                    if mode.endswith('sparse'):
                        # 只在这个成功窗口附近给信号（不是整个 episode 都算）
                        m_value = 1.0 if succ_recent > 0 else 0.0
                        succ_recent = max(0, succ_recent - T_S)
                    else:
                        # 距离桥接：过去 T_S 步净位移靠近目标就奖励（不是单步比较——噪声太大）
                        m_value = 1.0 if d < d_ts_prev else 0.0
                    d_ts_prev = d
                else:
                    m_value = 1.0
                if mode in ('C_novalue', 'C_frozen'):
                    m_value = 1.0

                m_steady = steady_gate(ema_rate)
                m = m_steady * (0.2 + 0.8 * m_agency * m_value)
                if m_rand is not None:
                    m = m_rand[t_glob]
                if not use_plast:
                    m = 0.0

                m_steady_acc += m_steady; m_agency_acc += m_agency
                m_value_acc += m_value; m_acc += m; m_cnt += 1

                if use_plast and m > 0 and key.size:
                    el = trace[pre] * trace[post]
                    emax = float(el.max())
                    if emax > 1e-9:
                        el /= emax
                        dw = ETA_POT * m * el
                        pos = w > 0
                        w[pos] = np.minimum(w[pos] + dw[pos] * (W_MAX_E - w[pos]), W_MAX_E)
                        neg = ~pos
                        w[neg] = np.maximum(w[neg] - dw[neg] * (W_MAX_I + w[neg]), -W_MAX_I)

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

                if use_plast and t_glob % T_GLOBAL == 0:
                    gi = rng.integers(0, N, P_GLOBAL)
                    gj = rng.integers(0, N, P_GLOBAL)
                    okk = (gi != gj) & (gj >= N_IN)
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

            d_prev = d
            t_glob += 1

        ep_succ[ep] = succ
        ep_steps[ep] = ep_step_done if succ else T_EP
        pred_ratio_hist[ep] = (e_sum / max(eb_sum, 1e-12)) if e_cnt > 0 else np.nan
        x_end_sum += x

        # 采样：结构快照用于 turnover
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
    phase2_succ = float(np.mean(ep_succ[p2]))
    phase2_step = float(np.mean(ep_steps[p2]))

    # 尾段 rate
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
        # 符号读出：目标只是「该往哪走」
        Sg = (D > 0).astype(np.float64)
        Ws, *_ = np.linalg.lstsq(S1, Sg, rcond=None)
        Ps = S1 @ Ws
        pred_sign = (Ps > 0.5).astype(np.float64)
        r2 = float(np.mean((pred_sign == Sg).mean(axis=0)))
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        succ=ep_succ, steps=ep_steps, pred_ratio=pred_ratio_hist,
        succ_first=float(np.mean(ep_succ[:max(1, n_ep // 6)])),
        succ_last=float(np.mean(ep_succ[-max(1, n_ep // 6):])),
        step_first=float(np.mean(ep_steps[:max(1, n_ep // 6)])),
        step_last=float(np.mean(ep_steps[-max(1, n_ep // 6):])),
        pr_first=float(np.nanmean(pred_ratio_hist[:max(1, n_ep // 6)])),
        pr_last=float(np.nanmean(pred_ratio_hist[-max(1, n_ep // 6):])),
        phase2_succ=phase2_succ, phase2_step=phase2_step,
        m_steady=float(m_steady_acc / max(m_cnt, 1)),
        m_agency=float(m_agency_acc / max(m_cnt, 1)),
        m_value=float(m_value_acc / max(m_cnt, 1)),
        m_mean=float(m_acc / max(m_cnt, 1)),
        align=float(align_acc / max(align_cnt, 1) / 2.0),
        a_mean=float(np.abs(a_sum / max(align_cnt, 1)).mean()),
        a_switch=float(a_switch / max(a_switch_n, 1)),
        x_end=float(np.abs(x_end_sum / max(n_ep, 1)).mean()),
        r2_val=r2s, r2_sign=r2,
        tov_last=float(tov[-1]) if tov else float('nan'),
        out_deg=int(deg_in[OUT_IDX].mean()),
        elapsed=time.perf_counter() - t0)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 200
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['C_full']

    for mode in modes:
        print('=' * 132)
        print('grow_goal  mode=%-12s  N=%d  预算=%d  episode=%d×%d步  种子=%d' % (
            mode, N, BUDGET, n_ep, T_EP, nseed))
        print('=' * 132)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, n_ep) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %8s %8s %7s %7s %7s' % (
            'seed', 'E', 'rate%', 'n_inh', 'succ早', 'succ晚', 'step早', 'step晚',
            'pred早', 'pred晚', 'out_deg', 'align'))
        for r in res:
            print('%-5d %6d %7.2f %6d %6.1f%% %6.1f%% %8.1f %8.1f %8.3f %7.3f %7d %7.3f' % (
                r['seed'], r['E'], r['rate'] * 100, r['n_inh'],
                r['succ_first'] * 100, r['succ_last'] * 100,
                r['step_first'], r['step_last'], r['pr_first'], r['pr_last'],
                r['out_deg'], r['align']))

        S1 = np.array([r['succ_first'] for r in res]) * 100
        S2 = np.array([r['succ_last'] for r in res]) * 100
        P1 = np.array([r['pr_first'] for r in res])
        P2 = np.array([r['pr_last'] for r in res])
        M = np.array([[r['m_steady'], r['m_agency'], r['m_value'], r['m_mean']] for r in res])
        T = np.array([r['step_first'] for r in res])
        T2 = np.array([r['step_last'] for r in res])
        print('  跨种子: 成功率 %.1f%% -> %.1f%% (%+.1f pp)   步数 %.1f -> %.1f (%+.1f)' % (
            S1.mean(), S2.mean(), S2.mean() - S1.mean(), T.mean(), T2.mean(), T2.mean() - T.mean()))
        print('          pred比 %.3f -> %.3f (%+.3f)   m: 稳态 %.2f / 能动 %.2f / 价值 %.2f / 合成 %.3f' % (
            np.nanmean(P1), np.nanmean(P2), np.nanmean(P2) - np.nanmean(P1),
            M[:, 0].mean(), M[:, 1].mean(), M[:, 2].mean(), M[:, 3].mean()))
        AL = np.array([r['align'] for r in res])
        print('          align（动作朝目标）%.3f ± %.3f   （0 = 与目标无关，1 = 总是朝目标）' % (
            AL.mean(), AL.std()))
        AM = np.array([r['a_mean'] for r in res])
        AS = np.array([r['a_switch'] for r in res])
        XE = np.array([r['x_end'] for r in res])
        print('          诊断：|a|均值 %.3f（1=饱和/恒定推力）  切换率 %.3f  |x_end| %.3f' % (
            AM.mean(), AS.mean(), XE.mean()))
        if mode == 'A_probe':
            RV = np.array([r['r2_val'] for r in res])
            RS = np.array([r['r2_sign'] for r in res])
            print('          探针：状态线性读出 (g-x) 的 R² = %.3f ± %.3f；符号准确率 = %.3f ± %.3f（0.5=无信息）' % (
                np.nanmean(RV), np.nanstd(RV), np.nanmean(RS), np.nanstd(RS)))
        if mode.startswith('T_'):
            P2S = np.array([r['phase2_succ'] for r in res]) * 100
            P2T = np.array([r['phase2_step'] for r in res])
            print('          步 3：phase2 前 1/6 成功率 %.1f%%（%.1f%%±%.1f）、步数 %.1f' % (
                P2S.mean(), np.median(P2S), P2S.std(), P2T.mean()))
        print('          耗时 %.1fs' % wall)
        print()


if __name__ == '__main__':
    main()
