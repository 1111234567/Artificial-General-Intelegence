# agi/runner.py — 一个配置跑一次，返回一条结构化结果。
#
# 设计原则：结果里必须同时有「做了什么」与「结果是什么」，且都可比对。
# 2026-10-24 的教训：之前每个脚本 print 一堆表，但没有统一的结果格式，
# 所以「哪些组合跑过 / 结果如何」只能靠回忆——那正是我反复丢东西的根源。

import time
import numpy as np

from .core import Network
from .env import make_env
from .signal import Signal


def _first_spike(state, out_idx, win_i):
    if win_i >= 0:
        return win_i
    of = state[out_idx] > 0
    if not of.any():
        return -1
    return int(np.nonzero(of)[0][0])


def src_is_episode(cfg):
    return cfg.signal.source == 'episode'


def run(cfg):
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n_out = cfg.readout.n_out
    out_idx = np.asarray(cfg.readout.out_idx, np.int64)

    net = Network(cfg.net, rng, out_idx)
    net.tau_trace = cfg.plast.tau_trace
    net.tau_theta = cfg.plast.tau_theta
    env = make_env(cfg.env, '%s_s%d' % (cfg.name, cfg.seed))
    sig = Signal(cfg.signal, n_out)

    n_in = cfg.net.n_in
    n_ch = n_in // 4
    u_buf = rng.random((16384, n_in), dtype=np.float32)
    xin = np.zeros(cfg.net.N, np.float32)

    t_ep = cfg.task.t_ep
    t_act = cfg.task.t_act
    n_ep = cfg.n_ep
    ep_active = np.zeros(cfg.net.N, bool)

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

    def encode(st, goal, ti):
        row = u_buf[ti % u_buf.shape[0]]
        xin.fill(0.0)
        for ch, val in ((0, st), (1, goal)):
            p = 0.02 * (1.0 + 0.8 * (val / (cfg.env.v_max * 0.5) - 1.0))
            a, b = ch * 2 * n_ch, (ch * 2 + 1) * n_ch
            xin[a:b] = row[a:b] < p
            a2, b2 = (ch * 2 + 1) * n_ch, (ch * 2 + 2) * n_ch
            xin[a2:b2] = row[a2:b2] < p

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
                win_i = _first_spike(net.state, out_idx, win_i)
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
        if cfg.plast.topology and m_ep > 0:
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
