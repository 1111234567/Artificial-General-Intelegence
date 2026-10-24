# agi/runner.py — 一个配置跑一次，返回一条结构化结果。
#
# 设计原则：结果里必须同时有「做了什么」与「结果是什么」，且都可比对。
# 2026-10-24 的教训：之前每个脚本 print 一堆表，但没有统一的结果格式，
# 所以「哪些组合跑过 / 结果如何」只能靠回忆——那正是我反复丢东西的根源。

import time
import numpy as np

from .core import Network, find_edges
from .env import make_env
from .signal import Signal, steady_gate


def _first_spike(state, out_idx, win_i, v=None, tie_break="first"):
    if win_i >= 0:
        return win_i
    of = state[out_idx] > 0
    if not of.any():
        return -1
    cand = np.nonzero(of)[0]
    if tie_break == "argmax_v" and cand.size > 1:
        return int(cand[np.argmax(v[out_idx][cand])])
    return int(cand[0])


def src_is_episode(cfg):
    return cfg.signal.source == 'episode'


def _run_random(cfg, rng):
    """移植自 grow_agent 的 mode=random：**不建网、不学**，只随机选动作。

    这是「不学网络」的下界基线——没有它，脉冲网络的绝对成功率读不出方向
    （§16.16：`none` 5.9% 比 `random` 20.6% 还差 3.5 倍，而当时差点把 5.9% 读成「网络废了」）。

    必须不消耗 RNG 建网：旧代码在 `_init_net` 之前就 return，第一个 draw 就是选动作。
    """
    t0 = time.perf_counter()
    n_out = cfg.readout.n_out
    env = make_env(cfg.env, '%s_s%d' % (cfg.name, cfg.seed))
    n_ep = cfg.n_ep
    t_ep, t_act = cfg.task.t_ep, cfg.task.t_act

    ep_succ = np.zeros(n_ep, np.float32)
    ep_prog = np.zeros(n_ep, np.float32)
    ep_basis = np.zeros(n_ep, np.float32)
    ep_dist = np.zeros(n_ep, np.float32)

    n_io = 0
    for ep in range(n_ep):
        if cfg.task.kind == 'discrete':
            st0, goal = cfg.task.cases[ep % len(cfg.task.cases)]
        else:
            st0 = int(rng.integers(cfg.task.obs_lo, cfg.task.obs_hi))
            goal = int(rng.integers(cfg.task.obs_lo, cfg.task.obs_hi))
        env.set_target(goal)
        env.set_state(st0)
        st = st0
        start = st
        min_move = abs(goal - start)
        total = 0
        n_act = n_ok = 0
        for k in range(t_ep):
            if k % t_act == 0 and k > 0:
                a_idx = int(rng.integers(0, n_out))
                new_st, moved, _a = env.act(a_idx)
                total += abs(moved)
                want = int(np.sign(goal - st))
                if want != 0 and np.sign(moved) != 0:
                    n_act += 1
                    if np.sign(moved) == want:
                        n_ok += 1
                st = new_st
                if st == goal:
                    break
        ep_succ[ep] = 1.0 if st == goal else 0.0
        ep_prog[ep] = total / max(min_move, 1)
        ep_basis[ep] = n_ok / max(n_act, 1)
        ep_dist[ep] = abs(goal - st)

    n_io = env.n_io
    env.close()

    k6 = max(1, n_ep // 6)
    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': n_ep,
        'succ': float(ep_succ.mean()),
        'succ_first': float(ep_succ[:k6].mean()),
        'succ_last': float(ep_succ[-k6:].mean()),
        'progress': float(ep_prog.mean()),
        'basis': float(ep_basis.mean()),
        'dist_end': float(ep_dist.mean()),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': 0, 'n_inh': 0, 'rate': 0.0, 'out_deg': 0, 'out_bias': 0.0,
        'n_upd_w': 0, 'n_upd_b': 0, 'n_upd_top': 0,
        'io_ok': bool(env.io_ok), 'n_io': int(n_io),
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def _run_world(cfg):
    """移植自 grow_world.py：2D 振荡器世界 + 单条连续时间轴（无 episode）。

    与另外两个流程的差异（全部在 Config 里显式）：
      - 没有 episode / 没有重置；一条时间轴跑 steps 步
      - 观测 = tanh(漏积分变量)，动作 = tanh((r_out - base)/scale) 连续值
      - m = 稳态门 × (0.2 + 0.8 × 动作方差代理)
      - el 不减 θ（恒 ≥ 0）+ **非对称**权重更新 + 生长不过滤输入区 + 全局随机边 + 淘汰不保护接口边
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta
    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    steps = cfg.steps or (cfg.n_ep * cfg.task.t_ep)
    t_s = cfg.plast.t_s
    closed = (cfg.flow == "world") and cfg.signal.gate      # gate 作「闭环」开关
    use_plast = bool(cfg.plast.weight or cfg.plast.topology)

    # ---- RNG 顺序必须与旧脚本一致 ----
    u_in = rng.random((steps, N_IN), dtype=np.float32)
    u_wht = rng.random((steps, 2), dtype=np.float32)
    noise = rng.normal(0.0, cfg.env.w_noise, size=(steps, 2))
    u_rand = (rng.uniform(-1.0, 1.0, size=(steps, 2))
              if cfg.readout.kind == "randact" else None)

    xin = np.zeros(N, np.float32)
    w_f = 2.0 * np.pi / cfg.env.period_d
    w_s = 2.0 * np.pi / cfg.env.period_d_slow
    kf = 1.0 - 1.0 / cfg.env.tau_w_f
    ks = 1.0 - 1.0 / cfg.env.tau_w_s
    xf = xs = 0.0

    # ---- 功能测量套件（grow_func）：纯观测，不改 v/s/w，不消耗 RNG ----
    meas = bool(cfg.readout.func_measure)
    if meas:
        nseg = 10
        SSEG = max(1, steps // 10)
        nout = max(2, steps // max(1, cfg.readout.oseg))
        Wp = np.zeros((2, N), np.float32)      # s[t] -> o[t+1]
        s_prev = np.zeros(N, np.float32)
        o_hat = np.zeros(2, np.float32)
        o_prev = np.zeros(2, np.float64)
        a_prev = np.zeros(2, np.float64)
        freeze_at = steps // 2
        pred_err = np.zeros(nseg, np.float64); pred_n = np.zeros(nseg, np.int64)
        base_err = np.zeros(nseg, np.float64); base_n = np.zeros(nseg, np.int64)
        train_err = np.zeros(nseg, np.float64); train_n = np.zeros(nseg, np.int64)
        ls_err = np.zeros(nseg, np.float64); ls_n = np.zeros(nseg, np.int64)
        XTX = np.zeros((N, N), np.float64)
        XTy = np.zeros((2, N), np.float64)
        Wls = None
        ga = np.zeros(nseg); ga2 = np.zeros(nseg)
        gd = np.zeros(nseg); gd2 = np.zeros(nseg)
        gad = np.zeros(nseg); gn = np.zeros(nseg, np.int64)
        out_seg = np.zeros(nout, np.float64)
        if cfg.readout.perturb:
            net.v[0] = cfg.readout.perturb   # 前史因果：只动初始膜电位，不动 rng

    r_out = np.full(2, cfg.readout.r_base, np.float64)
    a_all = np.zeros((steps, 2), np.float64)
    o_all = np.zeros((steps, 2), np.float32)

    born = drop = 0
    m_acc = 0.0
    m_cnt = 0
    elig_mean = 0.0
    prev = None
    tovs = []
    tail = []
    out_acc = np.zeros(2, np.float64)
    tov_every = max(2000, steps // 5)
    n_upd_w = 0
    n_upd_top = 0

    for t in range(steps):
        # ---- 1) 观测 ----
        if cfg.signal.source == "noise":
            o0 = float(u_wht[t, 0]) * 2.0 - 1.0
            o1 = float(u_wht[t, 1]) * 2.0 - 1.0
        else:
            o0 = float(np.tanh(xf))
            o1 = float(np.tanh(xs))
        o_all[t, 0] = o0
        o_all[t, 1] = o1

        # ---- 1b) 功能测量：用「上一步的预测」对「这一步的实测」打分 ----
        if meas and t > 0:
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
                    upd = (cfg.readout.pred_lr * np.array([e0, e1]) / ss).astype(np.float32)
                    Wp += upd[:, None] * s_prev[None, :]
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
            ga[si] += a_prev[0] + a_prev[1]
            ga2[si] += a_prev[0] ** 2 + a_prev[1] ** 2
            gd[si] += d0 + d1
            gd2[si] += d0 * d0 + d1 * d1
            gad[si] += a_prev[0] * d0 + a_prev[1] * d1
            gn[si] += 2
        if meas:
            o_prev[0] = o0
            o_prev[1] = o1
            # 1c) 训练期结束时解一次离线最优读出
            if t == freeze_at - 1:
                lam = 1e-2 * float(np.trace(XTX)) / N + 1e-6
                Am = XTX + lam * np.eye(N)
                try:
                    Wls = np.linalg.solve(Am, XTy.T).T.astype(np.float32)
                except np.linalg.LinAlgError:
                    Wls = None

        # ---- 2) 输入脉冲 ----
        h = N_IN // 2
        p0 = 0.02 * (1.0 + 0.8 * o0)
        p1 = 0.02 * (1.0 + 0.8 * o1)
        xin.fill(0.0)
        xin[0:h] = u_in[t, 0:h] < p0
        xin[h:N_IN] = u_in[t, h:N_IN] < p1

        # ---- 3) 网络 ----
        net.step(xin, False)
        s = net.state
        if meas:
            o_hat = Wp @ s
            s_prev[:] = s
            out_seg[min(t // cfg.readout.oseg, nout - 1)] += float(s[out_idx].sum())

        # ---- 4) 动作 ----
        r_out *= (1.0 - 1.0 / cfg.readout.a_tau)
        r_out += s[out_idx] / cfg.readout.a_tau
        a = np.tanh((r_out - cfg.readout.r_base) / cfg.readout.r_scale)
        if u_rand is not None:
            a = u_rand[t].copy()
        a_all[t] = a
        if meas:
            a_prev[0] = float(a[0])
            a_prev[1] = float(a[1])

        # ---- 5) 世界 ----
        uf = cfg.env.amp_d * float(np.cos(w_f * t))
        us = cfg.env.amp_d_slow * float(np.cos(w_s * t))
        if closed:
            uf += cfg.env.k_phase * a[0]
            us += cfg.env.k_phase * a[1]
        xf = kf * xf + uf + noise[t, 0]
        xs = ks * xs + us + noise[t, 1]

        # ---- 6) 结构可塑性 ----
        if use_plast and t > 0 and t % t_s == 0:
            lo = max(0, t - 200)
            aw = a_all[lo:t]
            agency = float(np.clip(max(aw[:, 0].std(), aw[:, 1].std())
                                   / cfg.signal.agency_scale, 0.0, 1.0))
            m = steady_gate(net.ema_rate) * (cfg.signal.base + (1.0 - cfg.signal.base) * agency)
            m_acc += m
            m_cnt += 1

            if cfg.plast.weight and net.key.size:
                elig_mean += net.apply_weight_asym(m, cfg.plast.eta_w, use_theta=False)
                n_upd_w += 1

            if cfg.plast.topology:
                born += net.grow_edges(cfg.plast.topo_thresh, cfg.net.p_new, cfg.net.max_active)
                if cfg.net.p_global > 0 and t % cfg.net.t_global == 0:
                    born += net.grow_global(cfg.net.p_global)
                drop += net.prune(cfg.net.budget, cfg.net.min_age)
                n_upd_top += 1

        # ---- 7) 记录 ----
        if t > 0 and t % tov_every == 0:
            if prev is not None:
                inter = np.intersect1d(prev, net.key, assume_unique=True).size
                tovs.append(1.0 - inter / max(prev.size + net.key.size - inter, 1))
            prev = net.key.copy()
        if t >= steps - 2000:
            tail.append(int(s.sum()))
            out_acc += s[out_idx]

    deg = net.in_degree()
    aw = a_all[max(0, steps - 2000):]
    ow = o_all[max(0, steps - 2000):]
    ca = 0.0
    for j in range(2):
        sj = float(aw[:, j].std())
        so = float(ow[:, j].std())
        if sj > 1e-9 and so > 1e-9:
            ca = max(ca, abs(float(np.corrcoef(aw[:, j], ow[:, j])[0, 1])))

    if meas:
        _pe = np.where(pred_n > 0, pred_err / np.maximum(pred_n, 1), np.nan)
        _te = np.where(train_n > 0, train_err / np.maximum(train_n, 1), np.nan)
        _ba = np.where(base_n > 0, base_err / np.maximum(base_n, 1), np.nan)
        _ls = np.where(ls_n > 0, ls_err / np.maximum(ls_n, 1), np.nan)
        _r2 = np.zeros(nseg)
        for k in range(nseg):
            nn = gn[k]
            if nn < 10:
                _r2[k] = np.nan
                continue
            va = ga2[k] / nn - (ga[k] / nn) ** 2
            vd = gd2[k] / nn - (gd[k] / nn) ** 2
            cv = gad[k] / nn - (ga[k] / nn) * (gd[k] / nn)
            _r2[k] = (cv * cv) / (va * vd) if va > 1e-12 and vd > 1e-12 else 0.0
        LAG = 20
        mm = min(20000, steps)
        sa = a_all[steps - mm:steps]
        so = o_all[steps - mm:steps]
        xc = np.zeros(2 * LAG + 1, np.float64)
        for k in range(-LAG, LAG + 1):
            if k >= 0:
                xa = sa[:mm - k]
                xo = so[k:]
            else:
                xa = sa[-k:]
                xo = so[:mm + k]
            cc = 0.0
            for j in range(2):
                sja = float(xa[:, j].std())
                sjo = float(xo[:, j].std())
                if sja > 1e-9 and sjo > 1e-9:
                    cc = max(cc, abs(float(np.corrcoef(xa[:, j], xo[:, j])[0, 1])))
            xc[k + LAG] = cc
    ntail = max(len(tail), 1)

    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': 0, 'steps': steps,
        'succ': float('nan'), 'succ_first': float('nan'), 'succ_last': float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': float(out_acc[0] / ntail),
        'fixdelay': None, 'react': None,
        'E_final': int(net.key.size),
        'born': int(born), 'n_upd_top': int(drop),
        'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(np.mean(tail) / N) if tail else 0.0,
        'rate_std': float(np.std(tail) / N) if tail else 0.0,
        'deg_e': float(deg[:N_E].mean()), 'deg_i': float(deg[N_E:].mean()),
        'w_abs': float(np.abs(net.w).mean()),
        'tov_last': float(tovs[-1]) if tovs else float('nan'),
        'tov_all': tovs,
        'a_std': float(max(aw[:, 0].std(), aw[:, 1].std())),
        'a_sat': float((np.abs(aw) >= 0.95).mean()),
        'a_obs_corr': ca,
        'out_deg': int(deg[out_idx].mean()),
        'out_deg0': int(deg[out_idx[0]]),
        'out_deg1': int(deg[out_idx[1]]),
        'out_bias': 0.0,
        'out_r0': float(out_acc[0] / ntail),
        'out_r1': float(out_acc[1] / ntail),
        'm_mean': float(m_acc / m_cnt) if m_cnt else 0.0,
        'elig_mean': float(elig_mean / m_cnt) if m_cnt else 0.0,
        'n_upd_w': int(n_upd_w), 'n_upd_b': 0,
        'io_ok': True, 'n_io': 0,
        'elapsed': round(time.perf_counter() - t0, 2),
        **(dict(pred=_pe, train=_te, base=_ba, pred_ls=_ls, r2=_r2,
                xc=xc, out_seg=out_seg, perturb=cfg.readout.perturb) if meas else {}),
    }


def _run_goal(cfg):
    """移植自 grow_goal.py：2D 连续世界 + 2 维符号动作 + 每 t_s 步结构可塑性。

    与 agent 流程的差异（**全部在 Config 里显式声明**，不是隐含分支）：
      - 世界是内存数学变量（math_2d），不是文件
      - 动作是 2 维独立符号（sign_2d），不是离散索引
      - 无 bias、无侧抑制
      - m = 稳态门 × 价值（只在刚成功的 window 步内为 1）
      - 可塑性每 t_s 步触发，不是每动作；淘汰**无条件**执行
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta

    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    n_ch = N_IN // 4
    n_out = cfg.readout.n_out
    t_act, t_ep, n_ep = cfg.task.t_act, cfg.task.t_ep, cfg.n_ep
    tol, k_act = cfg.env.tol, cfg.env.k_act
    t_s = cfg.plast.t_s

    # ---- RNG 顺序必须与旧脚本一致（结构之后：u_in / noise / x0 / g）----
    tot = n_ep * t_ep + 8
    u_in = rng.random((tot, N_IN), dtype=np.float32)
    noise = rng.normal(0.0, cfg.env.w_noise, size=(tot, 2))
    x0_all = rng.uniform(cfg.env.x0_lo, cfg.env.x0_hi, size=(n_ep, 2))
    g_all = rng.uniform(cfg.env.g_lo, cfg.env.g_hi, size=(n_ep, 2))

    if cfg.task.goal_mode == "fixed":
        g_all[:] = np.asarray(cfg.task.fixed_goal, np.float64)
    elif cfg.task.goal_mode == "two_phase":
        half = n_ep // 2
        g_all[:half] = g_all[:half] * 0.5 + np.array([0.5, 0.5])
        g_all[half:] = g_all[half:] * 0.5 + np.array([-0.5, -0.5])
        if cfg.task.phase == "fresh":
            g_all[:half] = g_all[half:].copy()

    xin = np.zeros(N, np.float32)
    r_out = np.zeros(n_out, np.float64)
    base = np.zeros(n_out, np.float64)
    a = np.zeros(n_out, np.float64)

    succ_recent = 0
    ep_succ = np.zeros(n_ep, np.float32)
    ep_steps = np.zeros(n_ep, np.float32)
    born = 0
    m_acc = m_steady_acc = m_value_acc = 0.0
    m_cnt = 0
    th_acc = 0.0
    align_acc = 0.0
    align_cnt = 0
    a_sum = np.zeros(n_out)
    a_switch = 0
    a_switch_n = 0
    x_end_sum = np.zeros(2)
    probe_S = [] if cfg.readout.probe else None
    probe_D = [] if cfg.readout.probe else None
    t_glob = 0
    n_upd_w = 0
    n_upd_top = 0

    use_plast = bool(cfg.plast.weight or cfg.plast.topology)

    for ep in range(n_ep):
        x = x0_all[ep].copy()
        g = g_all[ep].copy()
        succ = 0
        succ_recent = 0
        ep_step_done = t_ep

        for k in range(t_ep):
            t = t_glob
            # ---- 1) 输入脉冲（4 通道：x0 / x1 / g0 / g1）----
            p0 = 0.02 * (1.0 + 0.8 * x[0])
            p1 = 0.02 * (1.0 + 0.8 * x[1])
            q0 = 0.02 * (1.0 + 0.8 * g[0])
            q1 = 0.02 * (1.0 + 0.8 * g[1])
            xin.fill(0.0)
            xin[0:n_ch] = u_in[t, 0:n_ch] < p0
            xin[n_ch:2 * n_ch] = u_in[t, n_ch:2 * n_ch] < p1
            xin[2 * n_ch:3 * n_ch] = u_in[t, 2 * n_ch:3 * n_ch] < q0
            xin[3 * n_ch:4 * n_ch] = u_in[t, 3 * n_ch:4 * n_ch] < q1

            # ---- 2) 网络（无侧抑制；bias 全 0）----
            net.step(xin, False)
            s = net.state

            # ---- 3) 输出与动作（每 t_act 步采一次符号）----
            r_out *= (1.0 - 1.0 / cfg.readout.a_tau)
            r_out += s[out_idx] / cfg.readout.a_tau
            base *= (1.0 - 1.0 / cfg.readout.base_tau)
            base += r_out / cfg.readout.base_tau
            if k % t_act == 0:
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
            drift = cfg.env.amp_d * float(np.cos(2.0 * np.pi * t_glob / cfg.env.period_d))
            x = x + k_act * a + drift + noise[t_glob]
            x = np.clip(x, -1.0, 1.0)

            # ---- 5) 任务判定 ----
            d = float(max(abs(x[0] - g[0]), abs(x[1] - g[1])))
            if probe_S is not None and k % 5 == 0:
                probe_S.append(s.copy())
                probe_D.append(g - x)
            if d < tol and succ == 0:
                succ = 1
                ep_step_done = k + 1
                succ_recent = t_s

            # ---- 6) 结构可塑性（每 t_s 步）——淘汰始终执行，只关「强化 + 生长」----
            if t_glob > 0 and t_glob % t_s == 0:
                m_value = 1.0 if succ_recent > 0 else 0.0
                succ_recent = max(0, succ_recent - t_s)
                m_steady = steady_gate(net.ema_rate)
                m = m_steady * m_value
                if not use_plast:
                    m = 0.0
                m_steady_acc += m_steady
                m_value_acc += m_value
                m_acc += m
                m_cnt += 1
                th_acc += float(net.theta.mean())

                if cfg.plast.weight and use_plast and m > 0:
                    net.apply_weight(None, m, cfg.plast.eta_w, None, norm="before_m")
                    n_upd_w += 1

                # 淘汰**无条件**（grow_goal：frozen 只关「强化 + 生长」）
                n_upd_top += net.apply_topology(
                    cfg.plast.topo_thresh, cfg.net.budget, cfg.net.p_new,
                    cfg.net.max_active, cfg.net.min_age,
                    grow=(use_plast and cfg.plast.topology)) or 0
                born += net.last_born

            t_glob += 1

        ep_succ[ep] = succ
        ep_steps[ep] = ep_step_done if succ else t_ep
        x_end_sum += x

    k6 = max(1, n_ep // 6)
    if cfg.task.phase == "seq" and cfg.task.goal_mode == "two_phase":
        p2 = slice(n_ep // 2, n_ep // 2 + k6)
    elif cfg.task.goal_mode == "two_phase":
        p2 = slice(0, k6)
    else:
        p2 = slice(n_ep - k6, n_ep)

    deg = net.in_degree()
    r2 = r2s = None
    if probe_S and len(probe_S) > 200:
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

    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': n_ep,
        'succ': float(ep_succ.mean()),
        'succ_first': float(ep_succ[:k6].mean()),
        'succ_last': float(ep_succ[-k6:].mean()),
        'phase2_succ': float(ep_succ[p2].mean()),
        'step_first': float(ep_steps[:k6].mean()),
        'step_last': float(ep_steps[-k6:].mean()),
        'phase2_step': float(ep_steps[p2].mean()),
        'progress': float(ep_steps.mean() / max(t_ep, 1)),   # 归一化到 0~1（越小越好；与 agent 流程的可比）
        'basis': float(align_acc / max(align_cnt, 1) / 2.0),
        'dist_end': float(np.abs(x_end_sum / max(n_ep, 1)).mean()),
        'out_fire': 0.0,
        'fixdelay': None, 'react': None,
        'E_final': int(net.key.size), 'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(net.ema_rate),
        'out_deg': int(deg[out_idx].mean()), 'out_bias': 0.0,
        'n_upd_w': int(n_upd_w), 'n_upd_b': 0, 'n_upd_top': int(n_upd_top),
        'born': int(born),
        'm_steady': float(m_steady_acc / max(m_cnt, 1)),
        'm_value': float(m_value_acc / max(m_cnt, 1)),
        'm_mean': float(m_acc / max(m_cnt, 1)),
        'theta_mean': float(th_acc / max(m_cnt, 1)),
        'align': float(align_acc / max(align_cnt, 1) / 2.0),
        'a_mean': float(np.abs(a_sum / max(align_cnt, 1)).mean()),
        'a_switch': float(a_switch / max(a_switch_n, 1)),
        'r2_val': r2s, 'r2_sign': r2,
        'io_ok': True, 'n_io': 0,
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def _run_teach(cfg):
    """移植自 grow_teach.py：阶段 1/2 v1（持续速率编码 + 末窗 argmax + 只更接口边）。

    与 talk / cond 的差异（全部在 Config 里显式）：
      - trial 固定长度（`task.trial_steps`），每 trial **清空 trace**
      - 输入是**持续速率编码**（`in_hi` / `in_lo`），不是脉冲爆发
      - 读出 = trial 最后 `read_win` 步的**放电计数 argmax**（不是 first-spike）
      - 只更新「突触后是输出神经元」的边（`plast.only_out_edges`）；m_glob 不参与权重
      - 结构更新在 **trial 末**（不是每 20 步），且随机新建无活跃区限制

    ⚠️ RNG 顺序：`_init_net` -> `u = rng.random((trial, n_in))`（一次性）-> 逐步
    `u[t] = rng.random(n_in)` 滚动替换。不能换顺序。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta

    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    N_WORD = cfg.readout.n_out
    N_OUT = N_WORD
    TRIAL = cfg.task.trial_steps
    READ_WIN = cfg.task.read_win
    IN_HI = cfg.task.in_hi
    IN_LO = cfg.task.in_lo
    ETA_W = cfg.plast.eta_w
    ETA_BIAS = cfg.plast.eta_bias
    BIAS_MIN = cfg.plast.bias_min
    BIAS_MAX = cfg.plast.bias_max
    W_MAX_E = cfg.net.w_max_e
    M_UP = cfg.signal.m_up
    M_DN = cfg.signal.m_dn
    mode = cfg.signal.source
    n_trials = cfg.n_ep
    use_plast = bool(cfg.plast.weight or cfg.plast.topology)
    use_bias = bool(cfg.plast.bias)

    # 输入掩码：4 个词各占 25 个输入神经元
    n_blk = N_IN // N_WORD
    masks = np.zeros((N_WORD, N_IN), np.float32)
    for c in range(N_WORD):
        masks[c, c * n_blk:(c + 1) * n_blk] = IN_HI
    masks[masks == 0] = IN_LO

    bias = np.zeros(N, np.float32)
    bias[out_idx] = 0.05
    ist_out = np.isin(net.post, out_idx)
    xin = np.zeros(N, np.float32)
    born = drop = 0
    conf_right, conf_wrong = [], []
    win = []
    n_corr = n_tot = 0
    n_upd_w = n_upd_b = n_upd_top = 0

    u = rng.random((TRIAL, N_IN), dtype=np.float32)

    for it in range(n_trials):
        c = int(rng.integers(0, N_WORD))
        net.trace[:] = 0.0          # 每 trial 清空迹（旧实现的局部变量语义）
        cnt = np.zeros(N_OUT, np.int64)
        for t in range(TRIAL):
            xin.fill(0.0)
            xin[0:N_IN] = u[t] < masks[c]
            xin[out_idx] += 0.05     # BIAS_OUT：先诱导说
            net.step(xin, False, add_bias=False)
            s = net.state
            if t >= TRIAL - READ_WIN:
                cnt += s[out_idx].astype(np.int64)
            u[t] = rng.random(N_IN, dtype=np.float32)

        word = int(np.argmax(cnt))
        corr = (word == c)
        n_tot += 1
        if corr:
            n_corr += 1
        strength = float(cnt.sum())
        (conf_right if corr else conf_wrong).append(strength)

        # 教师信号：按神经元分配的误差方向（只学错的地方）
        if mode == "random":
            m_out = np.where(rng.random(N_OUT) < 0.5, M_UP, M_DN)
        elif mode == "constant":
            m_out = np.ones(N_OUT, np.float32) * M_UP
        else:
            tgt = np.zeros(N_OUT, np.float32)
            tgt[c] = 1.0
            got = (cnt > 0).astype(np.float32)
            m_out = np.where(tgt > got, M_UP, np.where(got > tgt, M_DN, 0.0))

        if use_plast and net.key.size:
            gate = steady_gate(net.ema_rate)
            el = net.trace[net.pre] * net.trace[net.post]
            mp = np.zeros(N, np.float32)
            mp[out_idx] = m_out.astype(np.float32)
            sel = mp[net.post.astype(np.int64)]
            if not cfg.plast.only_out_edges and np.any(m_out):
                # 非 only_out_edges：把全局 0 填给内部边（旧实现不做，保留开关）
                sel = np.where(ist_out, sel, np.float32(0.0)).astype(np.float32)
            nz = sel != 0
            if nz.any() and el[nz].max() > 1e-9:
                el_n = el[nz] / float(el[nz].max())
                dw = ETA_W * gate * el_n * sel[nz]
                sl = np.nonzero(nz)[0]
                pos = net.w[sl] > 0
                net.w[sl] = np.where(
                    pos,
                    np.minimum(net.w[sl] + dw * (W_MAX_E - net.w[sl]), W_MAX_E),
                    net.w[sl])
                net.w[sl] = np.maximum(net.w[sl], 0.0)
            n_upd_w += 1

            if use_bias:
                nb_ = bias[out_idx] + ETA_BIAS * m_out.astype(np.float32)
                bias[out_idx] = np.clip(nb_, BIAS_MIN, BIAS_MAX)
                n_upd_b += 1

            # 结构更新（每 trial 一次）
            if cfg.plast.topology:
                act = np.nonzero(net.trace > cfg.plast.topo_thresh)[0]
                if act.size > cfg.net.max_active:
                    act = rng.choice(act, cfg.net.max_active, replace=False)
                if act.size >= 2:
                    na = act.size
                    q_pre = np.repeat(act, na)
                    q_post = np.tile(act, na)
                    okk = q_pre != q_post
                    q_pre, q_post = q_pre[okk], q_post[okk]
                    qk = q_post.astype(np.int64) * N + q_pre
                    found, _ = find_edges(net.key, qk)
                    npw, nqw = q_pre[~found], q_post[~found]
                    if npw.size:
                        pick = rng.random(npw.size) < cfg.net.p_new
                        if pick.any():
                            npw, nqw = npw[pick], nqw[pick]
                            nw = np.where(npw < N_E, cfg.net.w_new_e,
                                          -cfg.net.w_new_e * cfg.net.w_i_ratio).astype(np.float32)
                            net._grow(npw.astype(np.int32), nqw.astype(np.int32), nw)
                            born += int(npw.size)
                if cfg.net.p_global > 0 and it % cfg.net.t_global == 0:
                    born += net.grow_global(cfg.net.p_global)
                drop += net.prune(cfg.net.budget, cfg.net.min_age)
                ist_out = np.isin(net.post, out_idx)
                n_upd_top += 1

        if (it + 1) % 200 == 0:
            win.append(n_corr / max(n_tot, 1))
            n_corr = n_tot = 0

    deg_in = net.in_degree()
    cr = float(np.mean(conf_right)) if conf_right else 0.0
    cw = float(np.mean(conf_wrong)) if conf_wrong else 0.0
    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': n_trials, 'steps': n_trials * TRIAL,
        'succ': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'succ_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'succ_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': int(net.key.size), 'born': int(born), 'drop': int(drop),
        'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(net.ema_rate),
        'out_deg': int(deg_in[out_idx].mean()),
        'out_bias': tuple(round(float(x), 4) for x in bias[out_idx]),
        'n_upd_w': int(n_upd_w), 'n_upd_b': int(n_upd_b), 'n_upd_top': int(n_upd_top),
        'io_ok': True, 'n_io': 0,
        'win': win, 'conf_right': cr, 'conf_wrong': cw,
        'acc_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'acc_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def _main_period(bins, bin_size, lo=2):
    """从分格序列里估主周期（自相关第一个正峰）。太静默则返回 0。"""
    a = np.asarray(bins, float)
    if a.size < 8 or a.sum() <= 0:
        return 0.0, 0.0
    a = a - a.mean()
    if np.allclose(a, 0):
        return 0.0, 0.0
    ac = np.correlate(a, a, mode='full')[a.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    for k in range(lo, ac.size // 2):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2:
            return float(k * bin_size), float(ac[k])
    return 0.0, 0.0


def _run_hebb(cfg):
    """移植自 grow_hebb.py：结构冻结 + Hebbian 慢变量 A + 三段（warm / encode / observe）。

    与其它流程的根本区别：
      - 结构不是自己长（先跑一次 homeo），而是**冻结**——隔离变量
      - 权重 = `w_struct + A`，A 是一条与边同型的慢变量（PRX 14, 021001 的低秩形态）
      - 判据不是分类准确率，而是「撤掉输入后活动能不能自持」+ 主周期

    ⚠️ 两处容易抄错的对齐点（旧实现就是这么写的）：
      ① 放电数只算**兴奋神经元**（`s[:N_E].sum()`），不是全体
      ② `cn = c[:N_E] / mean(c[:N_E])`；pre/post 里含抑制神经元（≥N_E）——
         必须先按 mask 取子集再索引，否则越界
    """
    t0 = time.perf_counter()
    sp = cfg.net
    N = sp.N
    N_E = int(round(N * sp.e_ratio))
    N_IN = sp.n_in

    # ---- 1) 结构：homeo 长出来，然后冻结 ----
    gres = _run_homeo(cfg.with_(steps=cfg.task.grow_steps), return_net=True)
    net = gres['_net']
    pre, post = net.pre, net.post
    w_struct = net.w
    del gres

    # ---- 2) 状态重置（旧实现不用 grow 结束时的状态）----
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    theta = np.full(N, sp.v_th, np.float32)
    x_rec = np.ones(N, np.float32)
    c = np.zeros(N, np.float32)
    A = np.zeros(pre.size, np.float32)
    mask = (pre >= N_IN) & (pre < N_E) & (post < N_E)
    im_all = np.nonzero(mask)[0]

    rng = np.random.default_rng(cfg.seed + 31337)
    r = cfg.plast.hebb_r
    tau_syn = cfg.plast.tau_syn
    tau_c = cfg.plast.tau_c
    lim = cfg.plast.a_max_rel * r * sp.w_ee_init * 4.0
    use_std = bool(sp.use_std)
    in_spike = cfg.task.in_spike
    bg = cfg.task.bg_rate
    zero = np.zeros(N, np.float32)

    def step(xin, plastic):
        nonlocal v, s, ref, theta, x_rec, c, A
        w_eff = w_struct + A
        if use_std:
            w_eff = np.where(pre >= N_E, w_eff * x_rec[pre], w_eff)
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * np.float32(np.exp(-1.0 / sp.tau_v)) + I + xin
        fire = (v > theta) & (ref <= 0)
        s_new = fire.astype(np.float32)
        v = np.where(s_new > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s_new > 0] = sp.t_ref
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + sp.d_theta, sp.th_max)
        theta += (sp.v_th - theta) / sp.tau_adapt
        if use_std:
            x_rec += (1.0 - x_rec) / sp.tau_rec
            ii = np.nonzero(s_new[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - sp.u_std)
        c += (s_new - c) / tau_c
        if plastic and r > 0.0:
            m = c[:N_E].mean()
            if m > 1e-9:
                cn = c[:N_E] / m
                target = np.zeros_like(A)
                if im_all.size:
                    target[im_all] = r * sp.w_ee_init * cn[pre[im_all]] * cn[post[im_all]]
                A += (target - A) / tau_syn
                np.clip(A, -lim, lim, out=A)
                A[~mask] = 0.0
        s = s_new
        return float(s_new[:N_E].sum())

    # ---- 3) warm（背景预热）----
    for _ in range(cfg.task.warm_steps):
        xin = zero.copy()
        xin[N_IN:] = (rng.random(N - N_IN) < bg).astype(np.float32) * in_spike
        step(xin, True)

    if cfg.readout.a_shuffle:
        A = A[rng.permutation(A.size)]

    # ---- 4) encode ----
    B = N_IN // max(1, cfg.task.n_cue)
    blk = 0
    enc_tot = 0.0
    if cfg.task.input_mode == "pulse":
        for t in range(cfg.task.enc_steps):
            xin = zero.copy()
            if t == 0 or t == max(1, cfg.task.enc_steps - 2):
                xin[blk * B:(blk + 1) * B] = in_spike
            enc_tot += step(xin, True)
    else:
        phase = rng.random(N_IN) * 2 * np.pi
        for t in range(cfg.task.osc_steps):
            drive = np.cos(2 * np.pi * t / cfg.task.osc_period + phase) * cfg.task.osc_amp
            drive = np.maximum(drive, 0.0)
            xin = zero.copy()
            xin[:N_IN] = (drive.astype(np.float32)
                          * (rng.random(N_IN) < 0.9).astype(np.float32))
            enc_tot += step(xin, True)

    # ---- 5) observe（撤掉全部输入）----
    bins = []
    acc = 0.0
    bs = cfg.task.bin_size
    plastic_obs = bool(cfg.plast.weight)
    for t in range(cfg.task.obs_steps):
        acc += step(zero.copy(), plastic_obs)
        if (t + 1) % bs == 0:
            bins.append(acc)
            acc = 0.0

    per, ac = _main_period(bins, bs)
    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': 0, 'steps': int(cfg.task.warm_steps + cfg.task.enc_steps
                               + cfg.task.obs_steps),
        'succ': float('nan'), 'succ_first': float('nan'), 'succ_last': float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': int(pre.size), 'n_ee': int(net_ee(pre, post, N_IN, N_E)),
        'born': 0, 'drop': 0, 'n_inh': int((pre >= N_E).sum()),
        'rate': 0.0,
        'out_deg': 0, 'out_bias': 0.0,
        'n_upd_w': 0, 'n_upd_b': 0, 'n_upd_top': 0,
        'io_ok': True, 'n_io': 0,
        # ---- hebb 专有 ----
        'r': r, 'tau_syn': tau_syn, 'mode': cfg.task.input_mode,
        'plastic': plastic_obs, 'shuffle': bool(cfg.readout.a_shuffle),
        'enc': float(enc_tot), 'bins': bins, 'head': bins[:6],
        'tail_sum': float(np.sum(bins[6:])), 'total': float(np.sum(bins)),
        'period': per, 'ac': ac, 'a_abs': float(np.abs(A).mean()),
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def net_ee(pre, post, n_in, n_e):
    m = (pre >= n_in) & (pre < n_e) & (post < n_e)
    return int(m.sum())


def sample_delay(cfg, progress, rng):
    """可变间隔（VI）：Gamma 采样。均值随进度上升，CV 固定 → 方差随之上升。"""
    m = cfg.task.d_mean_start + (cfg.task.d_mean_end - cfg.task.d_mean_start) * float(
        np.clip(progress, 0.0, 1.0))
    shape = 1.0 / (cfg.task.d_cv * cfg.task.d_cv)
    scale = m / shape
    d = int(rng.gamma(shape, scale))
    return max(cfg.task.d_min, min(cfg.task.d_max, d))


def _run_cond(cfg):
    """移植自 grow_cond.py：经典条件作用 + 可变间隔课程（阶段 1/2 v3）。

    与 talk 流程的四处差异（全部在 Config 里显式）：
      ① **变长呈现**：每个 = burst + 保持期(从 Gamma 采样) + answer_win
      ② **只奖励正确**（`signal.punish=False`）：错/抢答/沉默都给 0，不产生更新
      ③ **抢答不终止本呈现**（`readout.early_terminate=False`）：只记录并继续跑
      ④ 分桶按**保持期长度**（hold // 50，8 桶），不是按回答延迟

    ⚠️ 抢答不终止的后果（写在 §16.14）：抢答率 100% 时正确率测不到，但
    「说什么」这条链可以测——这是当时故意接受的代价，不是缺陷。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta

    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    N_WORD = cfg.readout.n_out
    N_OUT = N_WORD
    BLK = cfg.net.blk
    N_Q = cfg.net.n_q
    Q_IDX = np.arange(N_IN, N_IN + N_Q)
    BURST = cfg.task.burst
    QBURST = cfg.task.qburst
    AW = cfg.task.answer_win
    IN_SPIKE = cfg.task.in_spike
    IN_LO = cfg.task.in_lo
    ETA_W = cfg.plast.eta_w
    ETA_BIAS = cfg.plast.eta_bias
    BIAS_MIN = cfg.plast.bias_min
    BIAS_MAX = cfg.plast.bias_max
    W_MAX_E = cfg.net.w_max_e
    M_DECAY = float(np.exp(-1.0 / cfg.signal.tau_m))
    K_UPDATE = cfg.signal.k_update
    WARMUP = cfg.signal.warmup
    PUNISH = cfg.signal.punish
    TOPO_EVERY = cfg.plast.t_s
    nb = 8

    n_pres = cfg.n_ep
    mode = cfg.signal.source
    teach = mode in ("reward_only", "teacher_word")
    use_plast = bool(cfg.plast.weight or cfg.plast.topology)
    use_bias = bool(cfg.plast.bias)

    bias = np.zeros(N, np.float32)
    bias[out_idx] = 0.05
    m_out = np.zeros(N_OUT, np.float64)
    m_glob = 0.0
    ist_out = np.isin(net.post, out_idx)

    buck_n = np.zeros(nb, np.int64)
    buck_ok = np.zeros(nb, np.int64)
    win, win_delay, win_early, win_sil, win_rate = [], [], [], [], []
    n_corr = n_tot = n_sil = n_early = 0
    nsp_win = 0
    n_win = 0
    born = drop = 0
    s_a1a0 = s_a0a0 = 0.0
    a_prev = 0.0
    n_upd_w = n_upd_b = n_upd_top = 0

    t = 0
    pres = 0
    p_start = 0
    p_delay = 0
    p_end = 0
    c = 0
    answered = False
    early_flag = False
    lim = n_pres * cfg.task.d_mean_end * 2

    while t < lim and pres < n_pres:
        # ===== 呈现边界 =====
        if t >= p_end:
            if pres > 0 and not answered:
                n_tot += 1
                n_sil += 1
                if PUNISH:
                    m_out[c] -= 1.0
                    m_glob = -1.0
            if pres > 0 and (pres % 100) == 0:
                win.append(n_corr / max(n_tot, 1))
                win_sil.append(n_sil / max(n_tot, 1))
                win_early.append(n_early / max(n_tot, 1))
                win_delay.append(p_delay)
                if n_win > 0:
                    win_rate.append(nsp_win / n_win / N)
                    nsp_win = n_win = 0
                n_corr = n_tot = n_sil = n_early = 0
            progress = pres / max(n_pres - 1, 1)
            p_start = t
            p_delay = sample_delay(cfg, progress, rng)
            p_end = p_start + p_delay + AW
            c = int(rng.integers(0, N_WORD))
            answered = False
            early_flag = False
            pres += 1

        phase = t - p_start
        xin = np.zeros(N, np.float32)
        if phase < BURST:
            if phase == 0 or phase == BURST - 2:
                xin[c * BLK:(c + 1) * BLK] = IN_SPIKE
        if p_delay <= phase < p_delay + QBURST:
            qq = phase - p_delay
            if qq == 0 or qq == QBURST // 2:
                xin[Q_IDX] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin[Q_IDX] += (rng.random(N_Q) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        nsp = net.step(xin, False, add_bias=False)
        s = net.state
        v = net.v

        nsp_win += nsp
        n_win += 1
        if t > 2000:
            s_a1a0 += float(nsp) * a_prev
            s_a0a0 += a_prev * a_prev
        a_prev = float(nsp)

        # ===== 抢答：只记录，**不终止本呈现** =====
        if (not answered) and BURST <= phase < p_delay:
            of = s[out_idx] > 0
            if of.any():
                if not early_flag:
                    n_early += 1
                    n_tot += 1
                    early_flag = True
                if PUNISH:
                    cand = np.nonzero(of)[0]
                    o = int(cand[np.argmax(v[out_idx][cand])])
                    m_out[o] -= 1.0
                    m_glob = -1.0

        # ===== 回答期 =====
        if (not answered) and p_delay <= phase < p_delay + AW:
            of = s[out_idx] > 0
            if of.any():
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[out_idx][cand])])
                answered = True
                ok = (o == c)
                n_tot += 1
                if ok:
                    n_corr += 1
                    m_out[o] += 1.0
                    m_glob = 1.0
                else:
                    if PUNISH:
                        m_out[o] -= 1.0
                        m_glob = -1.0
                hold = p_delay - BURST
                b = int(min(nb - 1, hold // 50))
                buck_n[b] += 1
                if ok:
                    buck_ok[b] += 1

        m_out *= M_DECAY
        m_glob *= M_DECAY

        # ===== 可塑性（每 k_update 步）=====
        if (t + 1) % K_UPDATE == 0 and t >= WARMUP:
            has_m = bool(np.abs(m_out).max() > 1e-6) or abs(m_glob) > 1e-6
            if use_plast and net.key.size and has_m:
                gate = steady_gate(net.ema_rate)
                el = net.trace[net.pre] * net.trace[net.post]
                mp = np.zeros(N, np.float32)
                mp[out_idx] = m_out.astype(np.float32)
                sel = mp[net.post.astype(np.int64)]
                if abs(m_glob) > 1e-9:
                    sel = np.where(ist_out, sel, np.float32(m_glob)).astype(np.float32)
                nz = sel != 0
                if nz.any() and el[nz].max() > 1e-9:
                    el_n = el[nz] / float(el[nz].max())
                    dw = ETA_W * gate * el_n * sel[nz]
                    sl = np.nonzero(nz)[0]
                    pos = net.w[sl] > 0
                    net.w[sl] = np.where(
                        pos,
                        np.minimum(net.w[sl] + dw * (W_MAX_E - net.w[sl]), W_MAX_E),
                        net.w[sl])
                    net.w[sl] = np.maximum(net.w[sl], 0.0)
                n_upd_w += 1
            if use_bias:
                nb_ = bias[out_idx] + ETA_BIAS * m_out.astype(np.float32)
                bias[out_idx] = np.clip(nb_, BIAS_MIN, BIAS_MAX)
                n_upd_b += 1

        # ===== 结构更新（每 t_s 步）=====
        if use_plast and t % TOPO_EVERY == 0 and t > 0:
            born += net.grow_edges(cfg.plast.topo_thresh, cfg.net.p_new, cfg.net.max_active)
            if cfg.net.p_global > 0 and t % cfg.net.t_global == 0:
                born += net.grow_global(cfg.net.p_global)
            ist_out = np.isin(net.post, out_idx)
            drop += net.prune(cfg.net.budget, cfg.net.min_age)
            ist_out = np.isin(net.post, out_idx)
            n_upd_top += 1

        t += 1

    deg_in = net.in_degree()
    bk = np.where(buck_n > 0, buck_ok / np.maximum(buck_n, 1), np.nan)
    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': pres, 'steps': t,
        'succ': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'succ_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'succ_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': int(net.key.size), 'born': int(born), 'drop': int(drop),
        'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(net.ema_rate),
        'sigma': float(s_a1a0 / s_a0a0) if s_a0a0 > 0 else float('nan'),
        'out_deg': int(deg_in[out_idx].mean()),
        'out_bias': tuple(round(float(x), 4) for x in bias[out_idx]),
        'n_upd_w': int(n_upd_w), 'n_upd_b': int(n_upd_b), 'n_upd_top': int(n_upd_top),
        'io_ok': True, 'n_io': 0,
        'win': win, 'win_sil': win_sil, 'win_early': win_early,
        'win_delay': win_delay, 'win_rate': win_rate,
        'buck_n': buck_n, 'buck_acc': bk,
        'acc_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'acc_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'early_first': float(np.mean(win_early[:2])) if len(win_early) >= 2 else float('nan'),
        'early_last': float(np.mean(win_early[-2:])) if len(win_early) >= 2 else float('nan'),
        'sil_last': float(np.mean(win_sil[-2:])) if len(win_sil) >= 2 else float('nan'),
        'delay_first': float(np.mean(win_delay[:2])) if len(win_delay) >= 2 else float('nan'),
        'delay_last': float(np.mean(win_delay[-2:])) if len(win_delay) >= 2 else float('nan'),
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def _run_homeo(cfg, return_net=False):
    """移植自 grow_homeo.py：放电率稳态下的结构自生长（PLOS CB 2022 形态）。

    与另外四个流程的差异（全部在 Config 里显式）：
      - 初始**无 E-E**（只有 E→I / I→E / I→I / 输入扇出）
      - 结构可塑性 = **按每个兴奋神经元的放电率偏差双向调 E-E 入边**
      - 有 STD（抑制突触短时抑郁）与阈值适应两个自组织机制
      - 周期性 cue 编码（每 enc_gap 步注入）+ 背景脉冲（只给 n_in 之后的神经元）

    ⚠️ RNG 消耗顺序：build -> cue_plan（全部先生成）-> 逐步背景脉冲。不能换。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta

    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    N_CUE = cfg.task.n_cue
    BLK = N_IN // N_CUE
    ENC = cfg.task.enc_steps
    BG = cfg.task.bg_rate
    IN_SPIKE = cfg.task.in_spike
    steps = cfg.steps
    t_struct = cfg.plast.t_s
    r_target = cfg.plast.r_target
    tol = cfg.plast.tol

    hist = []
    done = 0
    cue_plan = []
    tt = cfg.task.cue_after or (steps // 2)
    while tt < steps:
        cue_plan.append((tt, int(rng.integers(0, N_CUE))))
        tt += cfg.task.enc_gap
    ci = 0

    xin = np.zeros(N, np.float32)
    n_drop = n_born = 0

    while done < steps:
        ch = min(t_struct, steps - done)
        cue = None
        if ci < len(cue_plan) and cue_plan[ci][0] < done + ch:
            cue = (cue_plan[ci][1], ENC)
            ci += 1

        # ⚠️ 迹必须在**每个 chunk 开头清零**：旧 `simulate()` 的 trace 是函数内局部变量，
        # 每次调用（= 每个 50 步的 chunk）从 0 重新累积。
        # ⚠️ 若 `reset_state_per_chunk`：旧 grow 调 simulate 时**从不传 start=**，
        # 所以每个 chunk 的 v/s/ref/θ/x_rec 都从零开始，只保留 rate_ema 与结构。
        # 这是旧实现的缺陷（状态并不连续），为逐位对齐而保留。
        net.trace[:] = 0.0
        if cfg.plast.reset_state_per_chunk:
            net.v[:] = 0.0
            net.state[:] = 0.0
            net.ref[:] = 0
            net.theta[:] = cfg.net.v_th
            net.x_rec[:] = 1.0
        for t in range(ch):
            xin.fill(0.0)
            xin[N_IN:] = (rng.random(N - N_IN) < BG).astype(np.float32) * IN_SPIKE
            if cue is not None:
                blk, enc = cue
                if t == 0 or t == max(1, enc - 2):
                    xin[blk * BLK:(blk + 1) * BLK] = IN_SPIKE
            net.step(xin, False, False)

        done += ch
        eemask = ((net.pre >= N_IN) & (net.pre < N_E) & (net.post < N_E))
        rr = net.rate_ema
        hist.append((done, int(eemask.sum()), float(rr[:N_E].mean()),
                     float((rr[:N_E] > r_target * (1 + tol)).sum()),
                     float((rr[:N_E] < r_target * (1 - tol)).sum()),
                     float((rr[:N_E] > r_target * 0.1).sum())))
        if cfg.plast.topology:
            _, d_, b_ = net.structural_homeo(
                rng, r_target, tol, cfg.plast.p_del, cfg.plast.p_new, cfg.net.w_ee_init)
            n_drop += d_
            n_born += b_

    eemask = ((net.pre >= N_IN) & (net.pre < N_E) & (net.post < N_E))
    n_ee = int(eemask.sum())
    last = hist[-1] if hist else (0, 0, 0.0, 0.0, 0.0, 0.0)

    res = {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': 0, 'steps': steps,
        'succ': float('nan'), 'succ_first': float('nan'), 'succ_last': float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': int(net.key.size), 'n_ee': n_ee,
        'born': int(n_born), 'drop': int(n_drop),
        'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(last[2]), 'rate_mean': float(last[2]),
        'n_over': int(last[3]), 'n_under': int(last[4]), 'n_alive': int(last[5]),
        'hist': hist,
        'out_deg': 0, 'out_bias': 0.0,
        'n_upd_w': 0, 'n_upd_b': 0, 'n_upd_top': len(hist),
        'io_ok': True, 'n_io': 0,
        'elapsed': round(time.perf_counter() - t0, 2),
    }
    if return_net:
        res['_net'] = net
    return res


def _probe_homeo(cfg, n_cue=4, warm=30000):
    """移植自 grow_homeo.probe：长好之后，不同 cue 是否留下不同模式？

    做法：先用 warm 步长到平衡 → 对每个 cue 从同一个预热态出发、编码 20 步
    → **完全关掉背景**跑 300 步 → 累加放电 → 两两余弦相似度。

    ⚠️ 判据的边界（§16.12 已记）：4 个 cue 打的是 4 块**不同输入神经元**，
    下游本来就会不同——这是**通路差异**，不是记忆。低相似度不能单独当多吸引子的证据。
    """
    g = _run_homeo(cfg.with_(steps=warm), return_net=True)
    net = g['_net']
    rng = np.random.default_rng(cfg.seed + 99999)
    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    IN_SPIKE = cfg.task.in_spike
    BLK = N_IN // n_cue

    TOTAL = np.zeros((n_cue, N), np.float64)
    xin = np.zeros(N, np.float32)

    for cu in range(n_cue):
        # 从同一个预热态出发：**每个 cue 前重置状态**，再跑 500 步背景
        # （旧 probe 每轮调 simulate 不传 start，等于无条件清空）
        net.v[:] = 0.0
        net.state[:] = 0.0
        net.ref[:] = 0
        net.theta[:] = cfg.net.v_th
        net.x_rec[:] = 1.0
        net.trace[:] = 0.0
        for t in range(500):
            xin.fill(0.0)
            xin[N_IN:] = (rng.random(N - N_IN) < cfg.task.bg_rate).astype(np.float32) * IN_SPIKE
            net.step(xin, False, False)

        # 给 cue：编码 20 步（背景全关）
        for t in range(20):
            xin.fill(0.0)
            if t == 0 or t == 18:
                xin[cu * BLK:(cu + 1) * BLK] = IN_SPIKE
            net.step(xin, False, False)

        # 关掉 cue 与背景，统计 300 步
        for t in range(300):
            xin.fill(0.0)
            net.step(xin, False, False)
            TOTAL[cu] += net.state

    sims = []
    for i in range(n_cue):
        for j in range(i + 1, n_cue):
            a, b = TOTAL[i], TOTAL[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float('nan'))
    return dict(seed=cfg.seed, n_ee=g['n_ee'], sims=sims,
                totals=[float(x.sum()) for x in TOTAL], elapsed=g['elapsed'])


def _run_talk(cfg):
    """移植自 grow_talk.py：脉冲刺激 + 事件发射 + 连续时间轴（阶段 1/2 v2）。

    与另外三个流程的差异（**全部在 Config 里显式声明**）：
      - 时间轴是「呈现」为单位的三段式：编码（脉冲）→ 保持 → 提问 → 回答
      - 状态永不清空（v / state / ref / trace / theta 全部跨呈现连续）
      - 发射是事件：第一个放电的输出神经元 = 它说的词（first-spike，平手取 v 最大）
      - m = 按神经元分配的 (tgt - got) 方向，经 tau_m 衰减，每 k_update 步写一次
      - 有「抢答」与「沉默」两个独立错误类型（不与「答错」混计）

    ⚠️ 未移植：STD（抑制短时抑郁）与突触缩放——旧脚本里两者默认关，不影响默认行为。
    ⚠️ 可塑性更新**不用** Network.apply_weight：旧实现对 `el[nz]` 子集用 `max`（非 abs），
       与框架的全局 abs 归一不同。为逐位一致，这里照搬旧算式。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)
    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.trace_decay = cfg.plast.trace_decay
    net.tau_theta = cfg.plast.tau_theta

    N = cfg.net.N
    N_E = net.N_E
    N_IN = cfg.net.n_in
    N_WORD = cfg.readout.n_out
    N_OUT = N_WORD
    BLK = cfg.net.blk
    N_Q = cfg.net.n_q
    Q_IDX = np.arange(N_IN, N_IN + N_Q)
    T_PRESENT = cfg.task.t_present
    BURST = cfg.task.burst
    NBURST = cfg.task.nburst
    DELAY = cfg.task.query_at
    QBURST = cfg.task.qburst
    ANSWER_FROM = cfg.task.answer_from
    T_MAX = cfg.task.t_max
    IN_SPIKE = cfg.task.in_spike
    IN_LO = cfg.task.in_lo
    ETA_W = cfg.plast.eta_w
    ETA_BIAS = cfg.plast.eta_bias
    BIAS_MIN = cfg.plast.bias_min
    BIAS_MAX = cfg.plast.bias_max
    W_MAX_E = cfg.net.w_max_e
    TAU_M = cfg.signal.tau_m
    K_UPDATE = cfg.signal.k_update
    M_DECAY = float(np.exp(-1.0 / TAU_M))
    BL_TAU = cfg.signal.bl_tau
    BL_DECAY = float(np.exp(-1.0 / BL_TAU))
    M_UP = cfg.signal.m_up
    M_DN = cfg.signal.m_dn
    WARMUP = cfg.signal.warmup
    THRESH = cfg.plast.topo_thresh
    TOPO_EVERY = cfg.plast.t_s

    n_present = cfg.n_ep
    steps = n_present * T_PRESENT
    mode = cfg.signal.source          # teacher_word / random / none（对照由 Config 决定）
    teach = (mode == "teacher_word")
    use_plast = bool(cfg.plast.weight or cfg.plast.topology)
    use_bias = bool(cfg.plast.bias)

    # ---- 状态（全部在 Network 里，跨呈现不清空）----
    bias = np.zeros(N, np.float32)
    bias[out_idx] = 0.05              # 初始：先诱导说
    m_out = np.zeros(N_OUT, np.float64)
    m_glob = 0.0
    baseline = 0.0

    # ---- 统计 ----
    nb = 6
    buck_n = np.zeros(nb, np.int64)
    buck_ok = np.zeros(nb, np.int64)
    win, win_sil, win_early, win_lat = [], [], [], []
    n_corr = n_tot = n_sil = n_early = 0
    n_burst_fire = 0
    lat_sum = 0
    nsp_tot = 0
    nsp_burst = nsp_tail = 0
    n_burst = n_tail = 0
    s_a1a0 = 0.0
    s_a0a0 = 0.0
    a_prev = 0.0
    rate_win = []
    nsp_win = 0
    n_win = 0
    born = drop = 0
    n_upd_w = n_upd_b = n_upd_top = 0
    bias_end = None

    c = 0
    answered = False
    lat = 0
    burst_at = np.zeros(BURST, np.int32)

    ist_out = np.isin(net.post, out_idx)

    for t in range(steps):
        phase = t % T_PRESENT
        if phase == 0:
            c = int(rng.integers(0, N_WORD))
            answered = False
            lat = 0
            burst_at = np.zeros(NBURST, np.int32)
            if NBURST > 1:
                burst_at = np.linspace(0, BURST - 2, NBURST).astype(np.int32)

        # ===== 输入：三段式 =====
        xin = np.zeros(N, np.float32)
        if phase < BURST and phase in burst_at:
            base = c * BLK
            xin[base:base + BLK] = IN_SPIKE
        if DELAY <= phase < DELAY + QBURST:
            qq = phase - DELAY
            if qq == 0 or qq == QBURST // 2:
                xin[Q_IDX] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin[Q_IDX] += (rng.random(N_Q) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        # ===== 网络一步（bias 已加进 xin；侧抑制已在显式边里）=====
        nsp = net.step(xin, False, add_bias=False)
        s = net.state
        v = net.v

        nsp_tot += nsp
        nsp_win += nsp
        n_win += 1
        if t > steps // 5:
            s_a1a0 += float(nsp) * a_prev
            s_a0a0 += a_prev * a_prev
        a_prev = float(nsp)
        if phase < BURST:
            nsp_burst += nsp
            n_burst += 1
        elif phase < BURST + 100:
            nsp_tail += nsp
            n_tail += 1

        # ===== 编码期的放电：不算抢答，但单独计数 =====
        if (not answered) and phase < BURST:
            if s[out_idx].any():
                n_burst_fire += 1

        # ===== 抢答（编码结束到提问之前开口）=====
        if (not answered) and BURST <= phase < ANSWER_FROM:
            of = s[out_idx] > 0
            if of.any():
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[out_idx][cand])])
                answered = True
                n_early += 1
                n_tot += 1
                lat = -1
                if teach:
                    got = np.zeros(N_OUT)
                    got[o] = 1.0
                    tgt = np.zeros(N_OUT)
                    tgt[c] = 1.0
                    raw = tgt - got
                    m_out += np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0))
                    r = -1.0
                    m_glob = (r - baseline)
                    m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                    baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r
                elif mode == "random":
                    m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
                elif mode == "constant":
                    m_out += 1.0

        # ===== 发射事件（first-spike）=====
        if (not answered) and ANSWER_FROM <= phase < T_MAX:
            of = s[out_idx] > 0
            if of.any():
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[out_idx][cand])])
                word = o
                answered = True
                lat = phase
                ok = (word == c)
                n_tot += 1
                if ok:
                    n_corr += 1
                lat_sum += lat
                b = min(nb - 1, max(0, (lat - ANSWER_FROM) // 20))
                buck_n[b] += 1
                if ok:
                    buck_ok[b] += 1
                if teach:
                    got = np.zeros(N_OUT)
                    got[o] = 1.0
                    tgt = np.zeros(N_OUT)
                    tgt[c] = 1.0
                    raw = tgt - got
                    m_out += np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0))
                    r = 1.0 if ok else -1.0
                    m_glob = (r - baseline)
                    m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                    baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r
                elif mode == "random":
                    m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
                elif mode == "constant":
                    m_out += 1.0

        # ===== 沉默也是一次作答 =====
        if (not answered) and phase == T_MAX - 1:
            answered = True
            n_tot += 1
            n_sil += 1
            if teach:
                tgt = np.zeros(N_OUT)
                tgt[c] = 1.0
                raw = tgt
                m_out += np.where(raw > 0, 1.0, 0.0)
                r = -1.0
                m_glob = (r - baseline)
                m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r
            elif mode == "random":
                m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
            elif mode == "constant":
                m_out += 1.0

        # ===== m 衰减 =====
        m_out *= M_DECAY
        # ===== 每 K 步写一次（不是每呈现）=====
        if (t + 1) % K_UPDATE == 0 and t >= WARMUP:
            has_m = bool(np.abs(m_out).max() > 1e-6) or abs(m_glob) > 1e-6
            if use_plast and net.key.size and has_m:
                gate = steady_gate(net.ema_rate)
                el = net.trace[net.pre] * net.trace[net.post]
                mp = np.zeros(N, np.float32)
                mp[out_idx] = m_out.astype(np.float32)
                sel = mp[net.post.astype(np.int64)]
                if abs(m_glob) > 1e-9:
                    sel = np.where(ist_out, sel, np.float32(m_glob)).astype(np.float32)
                nz = sel != 0
                if nz.any() and el[nz].max() > 1e-9:
                    el_n = el[nz] / float(el[nz].max())
                    dw = ETA_W * gate * el_n * sel[nz]
                    sl = np.nonzero(nz)[0]
                    pos = net.w[sl] > 0
                    net.w[sl] = np.where(
                        pos,
                        np.minimum(net.w[sl] + dw * (W_MAX_E - net.w[sl]), W_MAX_E),
                        net.w[sl])
                    net.w[sl] = np.maximum(net.w[sl], 0.0)
                n_upd_w += 1

            if use_bias:
                nb_ = bias[out_idx] + ETA_BIAS * m_out.astype(np.float32)
                bias[out_idx] = np.clip(nb_, BIAS_MIN, BIAS_MAX)
                n_upd_b += 1

        # ===== 结构更新（每 TOPO_EVERY 步）=====
        if use_plast and t % TOPO_EVERY == 0 and t > 0:
            born += net.grow_edges(THRESH, cfg.net.p_new, cfg.net.max_active)
            if cfg.net.p_global > 0 and t % cfg.net.t_global == 0:
                born += net.grow_global(cfg.net.p_global)
            ist_out = np.isin(net.post, out_idx)
            drop += net.prune(cfg.net.budget, cfg.net.min_age)
            ist_out = np.isin(net.post, out_idx)
            n_upd_top += 1

        # ===== 统计窗口（每呈现一次）=====
        if phase == T_PRESENT - 1:
            win.append(n_corr / max(n_tot, 1))
            win_sil.append(n_sil / max(n_tot, 1))
            win_early.append(n_early / max(n_tot, 1))
            win_lat.append(lat_sum / max(n_tot - n_sil - n_early, 1)
                           if n_tot > n_sil + n_early else 0.0)
            n_corr = n_tot = n_sil = n_early = lat_sum = 0
            if n_win > 0:
                rate_win.append(nsp_win / n_win / N)
                nsp_win = 0
                n_win = 0
            if len(win) == max(4, n_present // 4):
                bias_end = bias[out_idx].copy()

    deg_in = net.in_degree()
    bk = np.where(buck_n > 0, buck_ok / np.maximum(buck_n, 1), np.nan)
    return {
        'config': cfg.name, 'fingerprint': cfg.fingerprint(), 'seed': cfg.seed,
        'n_ep': n_present,
        'succ': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'succ_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'succ_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'progress': float('nan'), 'basis': float('nan'), 'dist_end': float('nan'),
        'out_fire': 0.0, 'fixdelay': None, 'react': None,
        'E_final': int(net.key.size),
        'born': int(born), 'drop': int(drop),
        'n_inh': int((net.pre >= N_E).sum()),
        'rate': float(net.ema_rate),
        'out_deg': int(deg_in[out_idx].mean()),
        'out_bias': tuple(round(float(x), 4) for x in bias[out_idx]),
        'theta_out': tuple(round(float(x), 3) for x in net.theta[out_idx]),
        'n_upd_w': int(n_upd_w), 'n_upd_b': int(n_upd_b), 'n_upd_top': int(n_upd_top),
        'io_ok': True, 'n_io': 0,
        # ---- talk 专有 ----
        'acc_first': float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        'acc_last': float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        'sil_first': float(np.mean(win_sil[:2])) if len(win_sil) >= 2 else float('nan'),
        'sil_last': float(np.mean(win_sil[-2:])) if len(win_sil) >= 2 else float('nan'),
        'early_first': float(np.mean(win_early[:2])) if len(win_early) >= 2 else float('nan'),
        'early_last': float(np.mean(win_early[-2:])) if len(win_early) >= 2 else float('nan'),
        'rate_burst': float(nsp_burst / max(n_burst, 1) / N),
        'rate_tail': float(nsp_tail / max(n_tail, 1) / N),
        'sigma': float(s_a1a0 / s_a0a0) if s_a0a0 > 0 else float('nan'),
        'buck_n': buck_n, 'buck_acc': bk,
        'win': win, 'win_sil': win_sil, 'win_early': win_early, 'win_lat': win_lat,
        'rate_win': rate_win,
        'burst_fire': int(n_burst_fire),
        'elapsed': round(time.perf_counter() - t0, 2),
    }


def run(cfg):
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n_out = cfg.readout.n_out
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)

    if cfg.readout.kind == 'random':
        return _run_random(cfg, rng)
    if cfg.flow == 'goal':
        return _run_goal(cfg)
    if cfg.flow == 'homeo':
        return _run_homeo(cfg)
    if cfg.flow == 'cond':
        return _run_cond(cfg)
    if cfg.flow == 'teach':
        return _run_teach(cfg)
    if cfg.flow == 'hebb':
        return _run_hebb(cfg)
    if cfg.flow == 'talk':
        return _run_talk(cfg)
    if cfg.flow == 'world':
        return _run_world(cfg)
    if cfg.flow == 'world':
        return _run_world(cfg)

    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.tau_theta = cfg.plast.tau_theta
    env = make_env(cfg.env, '%s_s%d' % (cfg.name, cfg.seed))
    sig = Signal(cfg.signal, n_out)

    n_in = cfg.net.n_in
    n_ch = n_in // 4
    u_buf = rng.random((cfg.u_buf_rows, n_in), dtype=np.float32)
    xin = np.zeros(cfg.net.N, np.float32)

    t_ep = cfg.task.t_ep
    t_act = cfg.task.t_act
    n_ep = cfg.n_ep
    ep_active = np.zeros(cfg.net.N, bool)
    recall = (cfg.task.kind == "recall")
    prev_end = None

    ep_succ = np.zeros(n_ep, np.float32)
    ep_prog = np.zeros(n_ep, np.float32)
    ep_basis = np.zeros(n_ep, np.float32)
    ep_dist = np.zeros(n_ep, np.float32)
    ep_outfire = np.zeros(n_ep, np.float32)
    ep_fixdel = np.full(n_ep, np.nan, np.float32)
    ep_react = np.full(n_ep, np.nan, np.float32)

    n_upd_w = 0
    n_upd_b = 0
    n_upd_top = 0
    t_glob = 0
    win_i = -1   # 本动作窗口内第一个放电的输出。**必须在 ep 循环外**：
    # 旧 grow_agent 只在 run_one 开头初始化一次，episode 之间不重置；
    # 跑满 240 步的 episode 末尾会把 win_i 带进下一个 ep 的前几个窗口。
    # 框架原来在 ep 开头 `win_i = -1`，那一行造成第 9 处差异（实测在 ep7 第一个动作分叉）。

    def encode(st, goal, ti):
        row = u_buf[ti % u_buf.shape[0]]
        xin.fill(0.0)
        # 归一化到 [-1, 1]（旧 grow_agent 的 obs_code）；recall 时目标**不可观测** -> 中性 0
        half = cfg.env.v_max * 0.5
        xs = st / half - 1.0
        xg = 0.0 if recall else (goal / half - 1.0)
        for ch, xv in ((0, xs), (1, xg)):
            p = 0.02 * (1.0 + 0.8 * xv)
            a, b = ch * 2 * n_ch, (ch * 2 + 1) * n_ch
            xin[a:b] = row[a:b] < p
            a2, b2 = (ch * 2 + 1) * n_ch, (ch * 2 + 2) * n_ch
            xin[a2:b2] = row[a2:b2] < p

    for ep in range(n_ep):
        if cfg.task.kind == 'discrete':
            st0, goal = cfg.task.cases[ep % len(cfg.task.cases)]
            env.set_target(goal)
            env.set_state(st0)
            st = st0
        else:
            # —— 与 grow_agent.py 同序：先（每 k_ep_task）换目标，再采 st ——
            if ep % cfg.task.k_ep_task == 0:
                env.set_target(int(rng.integers(cfg.task.obs_lo, cfg.task.obs_hi)))
            st0 = int(rng.integers(cfg.task.obs_lo, cfg.task.obs_hi))
            env.set_state(st0)
            _tg = env.get_target()
            if recall:
                # 目标不可观测：要求回到上一 episode 的终点（首个 ep 用 v_max - st）
                goal = prev_end if prev_end is not None else (cfg.env.v_max - st0)
            else:
                goal = _tg
            st = st0

        start = st
        min_move = max(abs(goal - start), 1)
        total_move = 0
        n_act = n_ok = 0
        n_out_fire = 0
        cur_err = abs(goal - st)
        first_far = -1
        fixdel = -1
        last_dir = 0
        reacted = react_n = 0
        ep_active[:] = False
        if cfg.reset_readout_per_ep:
            win_i = -1
        r_out = np.zeros(n_out)

        for k in range(t_ep):
            encode(st, goal, t_glob)
            t_glob += 1
            net.step(xin, True)
            ep_active |= (net.state > 0)

            if (net.state[out_idx] > 0).any():
                n_out_fire += 1
            r_out = 0.95 * r_out + 0.05 * net.state[out_idx]

            if cfg.readout.kind == 'first_spike':
                win_i = _first_spike(net.state, out_idx, win_i, net.v,
                                     cfg.readout.tie_break)
            else:
                win_i = int(np.argmax(r_out))

            if k % t_act != 0 or k <= 0:
                continue

            want = int(np.sign(goal - st))
            n_actions = len(cfg.env.actions)

            if win_i < 0 and cfg.readout.kind == 'first_spike':
                # ---- 沉默 ----
                # 注：这里只学 bias。历史上（grow_agent）还同时调了 upd_weights，
                # 2026-10-24 试过对齐，结果更差（5.0% vs 6.9%），所以保持只学 bias。
                # **这不是已验证的结论**——它只是当前取值；真正的结论要等对齐验证完成后。
                mv = sig.on_silence(want)
                if mv is not None and cfg.plast.bias:
                    net.apply_bias(mv, cfg.plast.eta_bias,
                                   cfg.plast.bias_min, cfg.plast.bias_max)
                    n_upd_b += 1
                if mv is not None and cfg.plast.weight and cfg.signal.silence_update_w:
                    mask = None
                    if cfg.plast.isolate:
                        mask = (ep_active[net.pre] &
                                (ep_active[net.post] | (net.post >= net.N_E)))
                    net.apply_weight(mv, 0.0, cfg.plast.eta_w, mask)
                    n_upd_w += 1
                win_i = -1
                continue

            a_idx = win_i if win_i < n_actions else -1
            win_i = -1
            new_st, moved, act_v = env.act(a_idx)

            total_move += abs(moved)
            if want != 0 and moved != 0:
                n_act += 1
                if np.sign(moved) == want:
                    n_ok += 1
            new_err = abs(goal - new_st)
            if new_err > cur_err and first_far < 0:
                first_far = k
            if first_far >= 0 and new_err < cur_err and fixdel < 0:
                fixdel = k - first_far
            d_now = int(np.sign(moved))
            if last_dir != 0 and d_now != 0 and d_now != last_dir:
                reacted += 1
            if last_dir != 0 and d_now != 0:
                react_n += 1
            last_dir = d_now
            cur_err = new_err
            st = new_st

            mv, mg = sig.on_action(a_idx, moved, want)
            if cfg.plast.bias and mv is not None:
                net.apply_bias(mv, cfg.plast.eta_bias,
                               cfg.plast.bias_min, cfg.plast.bias_max)
                n_upd_b += 1
            if cfg.plast.weight and (mv is not None or mg is not None):
                mask = None
                if cfg.plast.isolate:
                    mask = (ep_active[net.pre] &
                            (ep_active[net.post] | (net.post >= net.N_E)))
                net.apply_weight(mv, mg, cfg.plast.eta_w, mask)
                n_upd_w += 1

            if cur_err == 0:
                break

        succ = (cur_err == 0)
        ep_succ[ep] = 1.0 if succ else 0.0
        ep_prog[ep] = total_move / min_move
        ep_basis[ep] = n_ok / max(n_act, 1)
        ep_dist[ep] = cur_err
        ep_outfire[ep] = n_out_fire / max(t_ep, 1)
        if fixdel >= 0:
            ep_fixdel[ep] = fixdel
        if react_n > 0:
            ep_react[ep] = reacted / react_n

        m_ep, mv_ep, mg_ep = sig.on_episode(succ, ep_prog[ep], 1.0, env.io_ok)
        if cfg.signal.steady_gate_ep:
            m_ep = m_ep * steady_gate(net.ema_rate)
        if cfg.plast.topology and (m_ep > 0 or not cfg.plast.topo_gate) \
                and (ep > 0 or not cfg.plast.topo_skip_first):
            n_upd_top += net.apply_topology(
                cfg.plast.isolate_thresh, cfg.net.budget, cfg.net.p_new,
                cfg.net.max_active, cfg.net.min_age) or 0
        if cfg.plast.weight and src_is_episode(cfg) and (mv_ep is not None or mg_ep is not None):
            mask = None
            if cfg.plast.isolate:
                mask = (ep_active[net.pre] &
                        (ep_active[net.post] | (net.post >= net.N_E)))
            net.apply_weight(mv_ep, mg_ep, cfg.plast.eta_w, mask)
            n_upd_w += 1

        prev_end = st

        for _ in range(cfg.task.t_gap):
            encode(50, 50, t_glob)
            t_glob += 1
            net.step(xin, False)

    env.close()

    k6 = max(1, n_ep // 6)
    deg = net.in_degree()
    return {
        'config': cfg.name,
        'fingerprint': cfg.fingerprint(),
        'seed': cfg.seed,
        'n_ep': n_ep,
        'succ': float(ep_succ.mean()),
        'succ_first': float(ep_succ[:k6].mean()),
        'succ_last': float(ep_succ[-k6:].mean()),
        'progress': float(ep_prog.mean()),
        'basis': float(ep_basis.mean()),
        'dist_end': float(ep_dist.mean()),
        'out_fire': float(ep_outfire.mean()),
        'fixdelay': float(np.nanmean(ep_fixdel)) if np.any(~np.isnan(ep_fixdel)) else None,
        'react': float(np.nanmean(ep_react)) if np.any(~np.isnan(ep_react)) else None,
        'E_final': int(net.key.size),
        'n_inh': int((net.pre >= net.N_E).sum()),
        'rate': float(net.ema_rate),
        'out_deg': int(deg[out_idx].mean()),
        'out_bias': float(net.bias[out_idx].mean()),
        'n_upd_w': int(n_upd_w),
        'n_upd_b': int(n_upd_b),
        'n_upd_top': int(n_upd_top),
        'io_ok': bool(env.io_ok),
        'n_io': int(env.n_io),
        'elapsed': round(time.perf_counter() - t0, 2),
    }
