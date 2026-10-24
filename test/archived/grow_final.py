# grow_final.py - 结构可塑性实验（标定后的动力学）
#
# 前面 13 个版本的修正累积：
#   1) 性能：稀疏边列表（不碰 N²），多进程并行（8 进程 7 万步/秒）
#   2) 均质网络是双稳态（0.19% 或 100%），无中间态
#   3) 需要 E/I 平衡（80% 兴奋 / 20% 抑制）才能稳定在低放电率
#   4) 需要不应期（T_REF），否则放电率无上界，抑制比拉到 10 倍也压不住
#   5) 结构必须有硬预算 + 竞争，否则新建边无代价 -> 无限膨胀
#
# 标定结果（calib5）：w_E=0.50 / 抑制比 5.0 / T_REF=20 -> rate 1.99% 稳定
#
# 本实验要回答：在这个稳定动力学上，
#   局部（共激活）+ 全局（随机跳跃）+ 退火，能否让结构【收敛】？
#   看 turnover 是否自己降下来（= 温度是否在降）。

import sys
import time
import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800              # 前 800 兴奋，后 200 抑制
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50
W_I_RATIO = 5.0        # 抑制权重 = -W_E * 5.0
W_NEW = 0.05
ETA_POT = 0.02
DECAY_W = 0.9995
W_MAX = 1.0
BUDGET = 15000

T_S, T_G = 20, 2000
P_GLOBAL = 4
OUT_RATE = 0.002
MAX_ACTIVE = 48
P_NEW = 0.02


def build(E0, rng):
    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    w *= (1.0 + 0.15 * rng.standard_normal(w.size)).astype(np.float32)
    return _sort(pre, post, w)


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
    pre, post, w, key = build(BUDGET, rng)
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

            w *= (1.0 - (1.0 - DECAY_W) * temp)

            active = np.nonzero(s > 0)[0]
            if active.size > MAX_ACTIVE:
                active = rng.choice(active, MAX_ACTIVE, replace=False)
            if active.size >= 2:
                na = active.size
                q_pre = np.repeat(active, na)
                q_post = np.tile(active, na)
                ok = q_pre != q_post
                q_pre, q_post = q_pre[ok], q_post[ok]
                qk = q_post.astype(np.int64) * N + q_pre
                found, idx = find(key, qk)
                if found.any():
                    fi = idx[found]
                    is_pos = w[fi] > 0
                    # 只强化兴奋连接；抑制连接同向增强（更负）
                    w[fi] = np.where(
                        is_pos,
                        np.minimum(w[fi] + ETA_POT * temp * (W_MAX - w[fi]), W_MAX),
                        np.maximum(w[fi] - ETA_POT * temp * (W_MAX + w[fi]), -W_MAX))

                npw, nqw = q_pre[~found], q_post[~found]
                if npw.size:
                    pick = rng.random(npw.size) < (P_NEW * temp)
                    if pick.any():
                        npw, nqw = npw[pick], nqw[pick]
                        nw = np.where(npw >= N_E, -W_E * W_I_RATIO, W_NEW).astype(np.float32)
                        n0 = key.size
                        pre, post, w, key = _sort(
                            np.concatenate([pre, npw.astype(np.int32)]),
                            np.concatenate([post, nqw.astype(np.int32)]),
                            np.concatenate([w, nw]))
                        born += int(pick.sum())
                        if key.size > BUDGET:
                            d = key.size - BUDGET
                            di = np.argpartition(np.abs(w), d - 1)[:d]
                            m = np.ones(key.size, bool); m[di] = False
                            pre, post, w, key = pre[m], post[m], w[m], key[m]
                            drop += int(d)

            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                okk = gi != gj
                gi, gj = gi[okk], gj[okk]
                if gi.size:
                    nw = np.where(gi >= N_E, -W_E * W_I_RATIO, W_NEW).astype(np.float32)
                    pre, post, w, key = _sort(
                        np.concatenate([pre, gi.astype(np.int32)]),
                        np.concatenate([post, gj.astype(np.int32)]),
                        np.concatenate([w, nw]))
                    born += gi.size
                    if key.size > BUDGET:
                        d = key.size - BUDGET
                        di = np.argpartition(np.abs(w), d - 1)[:d]
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
    deg_e = deg[:N_E]; deg_i = deg[N_E:]
    return dict(seed=seed, E=key.size, born=born, drop=drop,
                rate=float(np.mean(tail) / N) if tail else 0.0,
                rate_std=float(np.std(tail) / N) if tail else 0.0,
                deg_mean=float(deg.mean()), deg_max=int(deg.max()),
                deg_e_mean=float(deg_e.mean()), deg_i_mean=float(deg_i.mean()),
                w_mean=float(np.abs(w).mean()),
                n_inh=int((pre >= N_E).sum()),
                tov_last=float(tovs[-1]) if tovs else float('nan'),
                tov_all=tovs, elapsed=time.perf_counter() - t0)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['C']

    for mode in modes:
        print('=' * 116)
        print('grow_final  mode=%s  N=%d(E%d/I%d)  steps=%d  种子=%d  预算=%d' % (
            mode, N, N_E, N - N_E, steps, nseed, BUDGET))
        print('动力学: w_E=%.2f  抑制比=%.1f  T_REF=%d  (标定 rate≈2%%)' % (W_E, W_I_RATIO, T_REF))
        print('=' * 116)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, steps) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %7s %7s %7s' % (
            'seed', 'E', 'born', 'drop', 'rate%', 'std%', 'deg_E/I', 'n_inh', 'w_abs', 'turnov'))
        for r in res:
            print('%-5d %6d %7d %6d %7.2f %7.2f %8s %7d %7.3f %7.4f' % (
                r['seed'], r['E'], r['born'], r['drop'], r['rate'] * 100,
                r['rate_std'] * 100,
                '%.1f/%.1f' % (r['deg_e_mean'], r['deg_i_mean']),
                r['n_inh'], r['w_mean'], r['tov_last']))
        R = np.array([r['rate'] for r in res]) * 100
        Rs = np.array([r['rate_std'] for r in res]) * 100
        T = np.array([r['tov_last'] for r in res])
        print('  跨种子: rate %.2f%%±%.2f   尾部波动 %.2f%%±%.2f   tov %.4f±%.4f' % (
            R.mean(), R.std(), Rs.mean(), Rs.std(), np.nanmean(T), np.nanstd(T)))
        print('  耗时 %.1fs (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print()
        print('  【初始种子 0 的 turnover 轨迹】: %s' % (
            [round(x, 4) for x in res[0]['tov_all']]))


if __name__ == '__main__':
    main()
