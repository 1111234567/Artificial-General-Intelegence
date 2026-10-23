# grow_world.py - 闭环：输出神经元在回路里，动作送回世界
#
# 动机（2026-10-24，Jerry）：之前失败有两个根因，只修了一个——
#   ① 输入没有结构 -> 没有可学的东西
#   ② 没有输出、没有反馈 -> 内部状态与行为无耦合，内部「说什么都没区别」
#   输出不是外挂读出层，而是回路里的两个普通神经元（index 500/501）。
#   闭环：s(t) -> a(t) -> theta(t+1) -> 输入(t+1) -> s(t+1)
#
# 【五版逐次拓出的五个缺陷，每一个都有数据】
#   1) v1 衰减判据错：闲置 = 「本步前后同时放电」-> 每次保护 68 / 衰减 14932（1:221），
#      等价全局衰减。rate 0.11~0.14%（死网）。
#   2) v2 衰减尺度错：改成「前神经元 500 步未放电」后仍衰减 2822 条（1:141）——活动率 0.5% 时
#      绝大多数边在 20 步内不会被用到。而硬预算已经是「用进废退」的实现 -> **衰减关闭**。
#   3) v2 世界发散：纯旋转 + 恒定动作 => 幅度涨到 24.5（open 组 5.8），观测被 clip 成常数。
#      -> 改成**双相位振荡器**：幅度恒为 1，动作改相位速度（有界）。
#   4) v3 结构冻结：turnover 恒为 0。强化只作用于「当步同时放电的对」（0.45% 的边），
#      新边权 0.08 又远低于存量 -> 每次都是「新建 -> 立刻被淘汰」的空转。
#      -> 强化改成**资格迹**（design.md §14.6 的 Δw = η·e_ij·m），作用于全部边。
#   5) v4 抑制边沙溢（本次修正）：n_inh 从 40% 涨到 73%，rate 0.36%。两个原因：
#      a. 淘汰比较用「|w| / 本类均值」，而抑制上界 2.5 > 兴奋 1.0 -> 抑制边天然占优；
#         -> 改为按**类型上界**归一化（比「在本类里离饱和多远」）。
#      b. 新建边继承「本类均值」-> 强的那一类新建的也强 -> 正反馈；
#         -> 新建边固定小权重 + **年龄保护**（MIN_AGE 次更新内不参与淘汰）。
#
# 仍是第一版近似：能动性用动作方差代理；不做预测误差项；「有因果」的严格测法下一版。
# 对照：closed / open / randact / noise_in / frozen

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
W_NEW_E = 0.15          # 新建边的固定初始强度（不继承本类均值——v4 的正反馈根因）
W_NEW_I = 0.75
BUDGET = 15000
E0_INIT = 30000

T_S = 20                # 结构更新周期
TRACE_DECAY = 0.95      # 活动迹：时间常数 ~20 步
P_NEW = 0.004           # 每条「活跃对中不存在的边」新建概率（v4 是 0.02，新建量淹没了预算）
P_GLOBAL = 4
T_GLOBAL = 2000
MAX_ACTIVE = 48
MIN_AGE = 25            # 年龄保护：新建后 25 次结构更新（500 步）内不参与淘汰

# ---- 接口 ----
N_IN = 100              # 输入神经元 0..99（前 50 快模态，后 50 慢模态）
IN_RATE = 0.02
GAIN = 0.8
OUT_IDX = np.array([500, 501])   # 输出神经元：普通节点，在回路里
A_TAU = 20.0
R_BASE = 0.02
R_SCALE = 0.03

# ---- 世界（两个相位振荡器）----
PERIOD_FAST = 20.0
PERIOD_SLOW = 500.0
TAU_W_F = 50.0          # 快世界变量漏时间（动作效果持续 ~50 步，可被追踪）
TAU_W_S = 500.0
AMP_F = 0.10            # 外源周期驱动幅度
AMP_S = 0.10
K_PHASE = 0.05          # 动作对世界变量的推动
W_NOISE = 0.01


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
    seed, mode, steps = args
    rng = np.random.default_rng(seed)

    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    age = np.full(pre.size, 100, np.int16)   # 存量边视为已成熟
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    xin = np.zeros(N, np.float32)
    trace = np.zeros(N, np.float32)

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

        # ---- 4) 动作 ----
        r_out *= (1.0 - 1.0 / A_TAU)
        r_out += s[OUT_IDX] / A_TAU
        a = np.tanh((r_out - R_BASE) / R_SCALE)
        if mode == 'randact':
            a = u_rand[t].copy()
        a_all[t] = a

        # ---- 5) 世界：积分变量 + 外源周期驱动 + 动作推动（open 断开）----
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

            # -- 资格迹强化：e_ij = trace[pre] * trace[post]，作用于全部边 --
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

            # -- 生长：只在活跃区新建，固定小权重 + 年龄 0 --
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

            # -- 硬预算：按「在本类上界中的相对强度」竞争淘汰；年龄保护 --
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
    ntail = max(len(tail), 1)
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(np.mean(tail) / N) if tail else 0.0,
        rate_std=float(np.std(tail) / N) if tail else 0.0,
        deg_e=float(deg_in[:N_E].mean()), deg_i=float(deg_in[N_E:].mean()),
        w_abs=float(np.abs(w).mean()), n_inh=int((pre >= N_E).sum()),
        tov_last=float(tovs[-1]) if tovs else float('nan'),
        tov_all=tovs,
        a_std=float(max(aw[:, 0].std(), aw[:, 1].std())),
        a_sat=float((np.abs(aw) >= 0.95).mean()),
        a_obs_corr=ca,
        out_deg0=int(deg_in[OUT_IDX[0]]), out_deg1=int(deg_in[OUT_IDX[1]]),
        out_r0=float(out_acc[0] / ntail), out_r1=float(out_acc[1] / ntail),
        m_mean=float(m_acc / m_cnt) if m_cnt else 0.0,
        elig_mean=float(elig_mean / m_cnt) if m_cnt else 0.0,
        elapsed=time.perf_counter() - t0)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['closed', 'open', 'randact', 'noise_in', 'frozen']

    for mode in modes:
        print('=' * 136)
        print('grow_world v5  mode=%-9s  N=%d  E0=%d->预算=%d  steps=%d  种子=%d' % (
            mode, N, E0_INIT, BUDGET, steps, nseed))
        print('=' * 136)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, steps) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %7s %7s %7s %6s %7s' % (
            'seed', 'E', 'born', 'drop', 'rate%', 'std%', 'deg_E/I', 'n_inh', 'turnov',
            'a_std', 'a_sat', 'a-o'))
        for r in res:
            print('%-5d %6d %7d %6d %7.2f %7.2f %8s %7d %7.4f %7.3f %5.1f%% %7.3f' % (
                r['seed'], r['E'], r['born'], r['drop'], r['rate'] * 100,
                r['rate_std'] * 100, '%.1f/%.1f' % (r['deg_e'], r['deg_i']),
                r['n_inh'], r['tov_last'], r['a_std'], r['a_sat'] * 100, r['a_obs_corr']))
        R = np.array([r['rate'] for r in res]) * 100
        Rs = np.array([r['rate_std'] for r in res]) * 100
        T = np.array([r['tov_last'] for r in res])
        A = np.array([r['a_std'] for r in res])
        M = np.array([r['m_mean'] for r in res])
        I = np.array([r['n_inh'] for r in res])
        print('  跨种子: rate %.2f%%±%.2f  波动 %.2f%%±%.2f  tov %.5f±%.5f  a_std %.3f±%.3f  m %.3f±%.3f  n_inh %.0f±%.0f' % (
            R.mean(), R.std(), Rs.mean(), Rs.std(), np.nanmean(T), np.nanstd(T),
            A.mean(), A.std(), M.mean(), M.std(), I.mean(), I.std()))
        print('  迹均值 %.4f  |  输出神经元入度 %s  放电率 %s' % (
            res[0]['elig_mean'], [r['out_deg0'] for r in res[:4]],
            ['%.3f' % r['out_r0'] for r in res[:4]]))
        print('  耗时 %.1fs (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print('  turnover 轨迹(seed0): %s' % [round(x, 5) for x in res[0]['tov_all']])
        print()


if __name__ == '__main__':
    main()
