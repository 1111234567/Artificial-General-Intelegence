# grow_v2.py - 只衰减不活跃边 + 真正从过度生产开始
#
# grow_final 的两个设计偏差：
#   1) 【全局衰减】把已经很强的边磨掉了：w 0.50 -> 0.145（5万步衰减到 29%）
#      而共激活强化补偿不回来 -> 死亡螺旋（rate 0.13%）
#      正确形态：只衰减【本窗口没被激活过的】边（用进废退的本意）
#   2) 初始就是【已长好】状态：E 刚好等于预算，边强度 0.5
#      没有「过度生产 -> 修剪」可言
#      正确形态：E0 超额（2 倍预算）、初始权重很弱，让竞争淘汰掉一半

import sys
import time
import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50
W_I_RATIO = 5.0
W_NEW = 0.08
ETA_POT = 0.02
DECAY_IDLE = 0.9975     # 只对闲置边
W_MAX = 1.0
BUDGET = 15000
E0_INIT = 30000         # 过度生产：2 倍预算

T_S, T_G = 20, 2000
P_GLOBAL = 4
OUT_RATE = 0.002
MAX_ACTIVE = 48
P_NEW = 0.02


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


def find(key, q):
    if key.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    i = np.searchsorted(key, q)
    np.clip(i, 0, key.size - 1, out=i)
    return key[i] == q, i


def run_one(args):
    seed, mode, steps = args
    rng = np.random.default_rng(seed)

    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)  # 初始弱
    pre, post, w, key = _sort(pre, post, w)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    acc = 0
    born = drop = 0
    prev = None
    tovs = []
    tail = []
    t0 = time.perf_counter()

    for t in range(steps):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        fire = (v > V_TH) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
        nsp = int(s.sum())
        acc += nsp

        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0

            active = np.nonzero(s > 0)[0]
            if active.size > MAX_ACTIVE:
                active = rng.choice(active, MAX_ACTIVE, replace=False)

            fidx = np.zeros(0, np.int64)
            newp = newq = None
            if active.size >= 2:
                na = active.size
                q_pre = np.repeat(active, na)
                q_post = np.tile(active, na)
                okk = q_pre != q_post
                q_pre, q_post = q_pre[okk], q_post[okk]
                qk = q_post.astype(np.int64) * N + q_pre
                found, idx = find(key, qk)
                if found.any():
                    fidx = idx[found]
                npw, nqw = q_pre[~found], q_post[~found]
                if npw.size:
                    pick = rng.random(npw.size) < (P_NEW * temp)
                    if pick.any():
                        newp, newq = npw[pick], nqw[pick]

            # 只衰减【未被激活】的边
            if key.size:
                dm = np.ones(key.size, bool)
                dm[fidx] = False
                w[dm] *= (1.0 - (1.0 - DECAY_IDLE) * temp)
            # 活跃边强化
            if fidx.size:
                pos = w[fidx] > 0
                w[fidx] = np.where(
                    pos,
                    np.minimum(w[fidx] + ETA_POT * temp * (W_MAX - w[fidx]), W_MAX),
                    np.maximum(w[fidx] - ETA_POT * temp * (W_MAX + w[fidx]), -W_MAX))

            # 新建
            if newp is not None and newp.size:
                nw = np.where(newp >= N_E, -W_E * W_I_RATIO * 0.3, W_NEW).astype(np.float32)
                pre, post, w, key = _sort(
                    np.concatenate([pre, newp.astype(np.int32)]),
                    np.concatenate([post, newq.astype(np.int32)]),
                    np.concatenate([w, nw]))
                born += int(newp.size)

            # 全局跳跃
            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                okk = gi != gj
                gi, gj = gi[okk], gj[okk]
                if gi.size:
                    nw = np.where(gi >= N_E, -W_E * W_I_RATIO * 0.3, W_NEW).astype(np.float32)
                    pre, post, w, key = _sort(
                        np.concatenate([pre, gi.astype(np.int32)]),
                        np.concatenate([post, gj.astype(np.int32)]),
                        np.concatenate([w, nw]))
                    born += gi.size

            # 硬预算：按 |w| 竞争淘汰
            if key.size > BUDGET:
                d = key.size - BUDGET
                scale = np.abs(w).copy()
                inh_m = pre >= N_E
                if (~inh_m).any():
                    scale[~inh_m] /= max(float(np.abs(w[~inh_m]).mean()), 1e-9)
                if inh_m.any():
                    scale[inh_m] /= max(float(np.abs(w[inh_m]).mean()), 1e-9)
                di = np.argpartition(scale, d - 1)[:d]
                m = np.ones(key.size, bool); m[di] = False
                pre, post, w, key = pre[m], post[m], w[m], key[m]
                drop += int(d)

        if t > 0 and t % 10000 == 0:
            if prev is not None:
                inter = np.intersect1d(prev, key, assume_unique=True).size
                tovs.append(1.0 - inter / max(prev.size + key.size - inter, 1))
            prev = key.copy()
        if t >= steps - 2000:
            tail.append(nsp)

    deg = np.bincount(post, minlength=N)
    return dict(seed=seed, E=key.size, born=born, drop=drop,
                rate=float(np.mean(tail) / N) if tail else 0.0,
                rate_std=float(np.std(tail) / N) if tail else 0.0,
                deg_e=float(deg[:N_E].mean()), deg_i=float(deg[N_E:].mean()),
                w_abs=float(np.abs(w).mean()), n_inh=int((pre >= N_E).sum()),
                tov_last=float(tovs[-1]) if tovs else float('nan'),
                tov_all=tovs, elapsed=time.perf_counter() - t0)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['C']

    for mode in modes:
        print('=' * 116)
        print('grow_v2  mode=%s  E0=%d(过度生产) -> 预算=%d  steps=%d  种子=%d' % (
            mode, E0_INIT, BUDGET, steps, nseed))
        print('=' * 116)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, steps) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %7s %7s' % (
            'seed', 'E', 'born', 'drop', 'rate%', 'std%', 'deg_E/I', 'n_inh', 'turnov'))
        for r in res:
            print('%-5d %6d %7d %6d %7.2f %7.2f %8s %7d %7.4f' % (
                r['seed'], r['E'], r['born'], r['drop'], r['rate'] * 100,
                r['rate_std'] * 100, '%.1f/%.1f' % (r['deg_e'], r['deg_i']),
                r['n_inh'], r['tov_last']))
        R = np.array([r['rate'] for r in res]) * 100
        Rs = np.array([r['rate_std'] for r in res]) * 100
        T = np.array([r['tov_last'] for r in res])
        print('  跨种子: rate %.2f%%±%.2f  波动 %.2f%%±%.2f   tov %.5f±%.5f' % (
            R.mean(), R.std(), Rs.mean(), Rs.std(), np.nanmean(T), np.nanstd(T)))
        print('  耗时 %.1fs (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print('  turnover 轨迹(seed0): %s' % [round(x, 5) for x in res[0]['tov_all']])


if __name__ == '__main__':
    main()
