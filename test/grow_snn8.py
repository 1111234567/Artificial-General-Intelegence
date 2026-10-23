# grow_snn8.py - 改用阈值稳态（诊断 gain 为何失效）
#
# v7 的发现：gain 从 0.8 变到 9.0（10 倍），rate 纹丝不动在 39%
#   -> gain 根本没在控放电率
#   -> 原因：gain 是乘性的，饱和时它只能整体压低；
#      但当 rate 高时，突触输入 I 本身随 rate 上升，形成自洽固定点
#
# 本版改【阈值稳态】(intrinsic plasticity)：直接调放电阈值
#   这是 SOqC 综述里列的标准机制之一（阈值适应）
#   响应方向直接：rate 高 -> 抬高阈值 -> 放电减少
#
# 另加诊断：I 均值 / v 分位数，用来看饱和到底卡在哪

import sys
import time
import numpy as np
from multiprocessing import Pool

N           = 1000
TAU         = 20.0
DECAY_V     = float(np.exp(-1.0 / TAU))

W_INIT      = 0.02
W_STRONG    = 0.10
ETA_POT     = 0.02
DECAY_W     = 0.9995
W_MAX       = 0.25

BUDGET      = 15000
T_S, T_H, T_G = 20, 200, 2000
P_GLOBAL    = 4

TARGET_RATE = 0.02
K_TH        = 0.06       # 阈值稳态步长
VTH_INIT    = 1.0
VTH_MIN, VTH_MAX = 0.05, 50.0
OUT_RATE    = 0.002
MAX_ACTIVE  = 48
P_NEW       = 0.02


class Net:
    __slots__ = ('pre', 'post', 'w', 'key')

    def __init__(self, E0, rng):
        pre = rng.integers(0, N, E0).astype(np.int32)
        post = rng.integers(0, N, E0).astype(np.int32)
        ok = pre != post
        pre, post = pre[ok], post[ok]
        w = (W_STRONG * (1.0 + 0.2 * rng.standard_normal(len(pre)))).astype(np.float32)
        np.clip(w, 0.005, None, out=w)
        self._set(pre, post, w)

    def _set(self, pre, post, w):
        key = post.astype(np.int64) * N + pre
        order = np.argsort(key, kind='stable')
        key, pre, post, w = key[order], pre[order], post[order], w[order]
        if key.size > 1:
            uniq = np.empty(key.size, bool)
            uniq[0] = True
            np.not_equal(key[1:], key[:-1], out=uniq[1:])
            if not uniq.all():
                key, pre, post, w = key[uniq], pre[uniq], post[uniq], w[uniq]
        self.pre = np.ascontiguousarray(pre)
        self.post = np.ascontiguousarray(post)
        self.w = np.ascontiguousarray(w, dtype=np.float32)
        self.key = np.ascontiguousarray(key)

    def size(self):
        return self.key.size

    def find(self, qkey):
        if self.key.size == 0:
            return np.zeros(qkey.size, bool), np.zeros(qkey.size, np.int64)
        idx = np.searchsorted(self.key, qkey)
        np.clip(idx, 0, self.key.size - 1, out=idx)
        return self.key[idx] == qkey, idx

    def add_competitive(self, pre, post, budget):
        if pre.size == 0:
            return 0
        qkey = post.astype(np.int64) * N + pre
        found, _ = self.find(qkey)
        pre, post = pre[~found], post[~found]
        if pre.size == 0:
            return 0
        n0 = self.size()
        self._set(np.concatenate([self.pre, pre.astype(np.int32)]),
                  np.concatenate([self.post, post.astype(np.int32)]),
                  np.concatenate([self.w, np.full(pre.size, W_INIT, np.float32)]))
        if self.size() > budget:
            drop = self.size() - budget
            idx = np.argpartition(self.w, drop - 1)[:drop]
            mask = np.ones(self.size(), bool); mask[idx] = False
            self.pre = self.pre[mask]; self.post = self.post[mask]
            self.w = self.w[mask]; self.key = self.key[mask]
            return int(drop)
        return 0


def run_one(seed, mode, steps, E0=None):
    rng = np.random.default_rng(seed)
    net = Net(E0 or BUDGET, rng)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    vth = VTH_INIT

    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    h_acc = 0; i_acc = 0.0
    born = dropped = 0
    trail = []; prev_key = None; tovs = []
    vth_hist = []
    t0 = time.perf_counter()

    for t in range(steps):
        pre, post, w = net.pre, net.post, net.w
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > vth).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        h_acc += nsp; i_acc += float(I.mean())

        # 阈值稳态
        if t > 0 and t % T_H == 0:
            rate = h_acc / (T_H * N)
            h_acc = 0; i_acc = 0.0
            vth *= (1.0 + K_TH * (rate - TARGET_RATE) / TARGET_RATE)
            vth = float(np.clip(vth, VTH_MIN, VTH_MAX))

        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0
            net.w *= (1.0 - (1.0 - DECAY_W) * temp)

            active = np.nonzero(s > 0)[0]
            if active.size > MAX_ACTIVE:
                active = rng.choice(active, MAX_ACTIVE, replace=False)
            if active.size >= 2:
                na = active.size
                q_pre = np.repeat(active, na); q_post = np.tile(active, na)
                ok = q_pre != q_post
                q_pre, q_post = q_pre[ok], q_post[ok]
                qkey = q_post.astype(np.int64) * N + q_pre
                found, idx = net.find(qkey)
                if found.any():
                    fi = idx[found]
                    net.w[fi] = np.minimum(
                        net.w[fi] + ETA_POT * temp * (W_MAX - net.w[fi]), W_MAX)
                np_, nq_ = q_pre[~found], q_post[~found]
                if np_.size:
                    pick = rng.random(np_.size) < (P_NEW * temp)
                    if pick.any():
                        born += int(pick.sum())
                        dropped += net.add_competitive(np_[pick], nq_[pick], BUDGET)

            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL); gj = rng.integers(0, N, P_GLOBAL)
                ok2 = gi != gj
                if ok2.any():
                    born += int(ok2.sum())
                    dropped += net.add_competitive(gi[ok2], gj[ok2], BUDGET)

        if t > 0 and t % 10000 == 0:
            if prev_key is not None:
                inter = np.intersect1d(prev_key, net.key, assume_unique=True).size
                union = prev_key.size + net.key.size - inter
                tovs.append(1.0 - inter / max(union, 1))
            prev_key = net.key.copy()
            vth_hist.append((t, round(vth, 3), round(s.sum() / N * 100, 2)))

        if t >= steps - 2000:
            trail.append(nsp)

    deg = np.bincount(net.post, minlength=N)
    return dict(seed=seed, E=net.size(), born=born, dropped=dropped,
                rate=float(np.mean(trail) / N) if trail else 0.0,
                vth=vth, deg_mean=float(deg.mean()), deg_nz=int((deg > 0).sum()),
                w_mean=float(net.w.mean()), w_cv=float(net.w.std() / max(net.w.mean(), 1e-9)),
                tov_last=float(tovs[-1]) if tovs else float('nan'),
                vth_hist=vth_hist, elapsed=time.perf_counter() - t0)


def _worker(a):
    return run_one(*a)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['C']

    for mode in modes:
        print('=' * 112)
        print('grow_snn8  mode=%s  阈值稳态  N=%d  steps=%d  种子=%d  预算=%d  目标率=%.0f%%' % (
            mode, N, steps, nseed, BUDGET, TARGET_RATE * 100))
        print('=' * 112)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(_worker, [(s, mode, steps) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %7s %7s %7s %6s %6s %6s %7s' % (
            'seed', 'E', 'born', 'drop', 'rate%', 'vth', 'nz', 'w_mean', 'w_cv', 'turnov'))
        for r in res:
            print('%-5d %6d %7d %7d %7.2f %7.2f %6d %6.3f %6.3f %7.3f' % (
                r['seed'], r['E'], r['born'], r['dropped'], r['rate'] * 100,
                r['vth'], r['deg_nz'], r['w_mean'], r['w_cv'], r['tov_last']))
        R = np.array([r['rate'] for r in res]) * 100
        V = np.array([r['vth'] for r in res])
        T = np.array([r['tov_last'] for r in res])
        print('  跨种子: rate %.2f%%±%.2f   vth %.2f±%.2f   tov %.3f±%.3f' % (
            R.mean(), R.std(), V.mean(), V.std(), np.nanmean(T), np.nanstd(T)))
        print('  耗时 %.1fs (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print('\n  阈值/放电率轨迹 (seed0):')
        for h in res[0]['vth_hist']:
            print('    t=%-7d vth=%-7.3f rate=%.2f%%' % h)


if __name__ == '__main__':
    main()
