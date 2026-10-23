# grow_snn4.py - 修正 E 爆炸 + 多进程并行
#
# v3 的问题：1 万步里 E 从 8000 涨到 53258（新建概率 0.15 太高，无预算约束）
#
# 本版：
#   1) 新建概率 0.15 -> 0.015（慢过程，稀有事）
#   2) E_MAX 预算：超过就不新建（稀缺 -> 竞争）
#   3) 多进程并行跑多个种子（而不是一个一个跑）
#      —— 单线程 CPU 利用率只有 1/20，并行才是提吞吐的正路
#      —— 而且「多个种子」本来就是草案判据要求的（N>=5）

import os
import sys
import time
import numpy as np
from multiprocessing import Pool

N           = 1000
TAU         = 20.0
DECAY_V     = float(np.exp(-1.0 / TAU))
V_TH        = 1.0

W_INIT      = 0.10
ETA_POT     = 0.05
DECAY_W     = 0.995
W_MAX       = 1.0
THETA_ABS   = 0.012

T_S         = 20
T_G         = 2000
P_GLOBAL    = 6

TARGET_RATE = 0.02
K_HOME      = 0.25
OUT_RATE    = 0.004
MAX_ACTIVE  = 48

P_NEW       = 0.015
E_MAX       = 30000


class Net:
    __slots__ = ('pre', 'post', 'w', 'key')

    def __init__(self, E0, rng):
        pre = rng.integers(0, N, E0).astype(np.int32)
        post = rng.integers(0, N, E0).astype(np.int32)
        ok = pre != post
        pre, post = pre[ok], post[ok]
        w = (W_INIT * (1.0 + 0.15 * rng.standard_normal(len(pre)))).astype(np.float32)
        np.clip(w, 0.001, None, out=w)
        self._set(pre, post, w)

    def _set(self, pre, post, w):
        key = post.astype(np.int64) * N + pre
        order = np.argsort(key, kind='stable')
        self.pre = np.ascontiguousarray(pre[order])
        self.post = np.ascontiguousarray(post[order])
        self.w = np.ascontiguousarray(w[order], dtype=np.float32)
        self.key = np.ascontiguousarray(key[order])

    def size(self):
        return self.key.size

    def find(self, qkey):
        if self.key.size == 0:
            return np.zeros(qkey.size, bool), np.zeros(qkey.size, np.int64)
        idx = np.searchsorted(self.key, qkey)
        np.clip(idx, 0, self.key.size - 1, out=idx)
        return self.key[idx] == qkey, idx

    def add(self, pre, post, wv):
        pre = np.concatenate([self.pre, pre.astype(np.int32)])
        post = np.concatenate([self.post, post.astype(np.int32)])
        w = np.concatenate([self.w, np.full(pre.size - self.pre.size, wv, np.float32)])
        self._set(pre, post, w)

    def prune(self, thresh):
        m = self.w >= thresh
        if m.all():
            return 0
        n = self.size() - int(m.sum())
        self.pre = self.pre[m]
        self.post = self.post[m]
        self.w = self.w[m]
        self.key = self.key[m]
        return n


def run_one(seed, mode, steps, E0=8000, snap_every=0):
    rng = np.random.default_rng(seed)
    net = Net(E0, rng)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    gain = 1.0
    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    t0 = time.perf_counter()
    sp_acc = 0
    snaps = []
    born = 0
    died = 0

    for t in range(steps):
        pre, post, w = net.pre, net.post, net.w
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + gain * I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        sp_acc += int(s.sum())

        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0

            rate = sp_acc / (T_S * N)
            sp_acc = 0
            gain *= (TARGET_RATE / rate) ** K_HOME if rate > 1e-9 else 1.3
            gain = float(np.clip(gain, 1e-3, 1e3))

            net.w *= (1.0 - (1.0 - DECAY_W) * temp)
            died += net.prune(THETA_ABS)

            active = np.nonzero(s > 0)[0]
            if active.size > MAX_ACTIVE:
                active = rng.choice(active, MAX_ACTIVE, replace=False)
            if active.size >= 2:
                na = active.size
                q_pre = np.repeat(active, na)
                q_post = np.tile(active, na)
                ok = q_pre != q_post
                q_pre, q_post = q_pre[ok], q_post[ok]
                qkey = q_post.astype(np.int64) * N + q_pre
                found, idx = net.find(qkey)

                if found.any():
                    fi = idx[found]
                    net.w[fi] = np.minimum(
                        net.w[fi] + ETA_POT * temp * (W_MAX - net.w[fi]), W_MAX)

                if net.size() < E_MAX:
                    np_, nq_ = q_pre[~found], q_post[~found]
                    if np_.size:
                        pick = rng.random(np_.size) < (P_NEW * temp)
                        if pick.any():
                            born += int(pick.sum())
                            net.add(np_[pick], nq_[pick], W_INIT)

            if use_global and t % T_G == 0 and net.size() < E_MAX:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                ok2 = gi != gj
                net.add(gi[ok2], gj[ok2], W_INIT)
                born += int(ok2.sum())

        if snap_every and t > 0 and t % snap_every == 0:
            deg = np.bincount(net.post, minlength=N)
            snaps.append((t, net.size(), s.sum() / N, gain, deg.mean(), deg.max(),
                          int((deg > 0).sum())))

    deg = np.bincount(net.post, minlength=N)
    return dict(seed=seed, E=net.size(), born=born, died=died,
                rate=float(s.sum() / N), gain=gain,
                deg_mean=float(deg.mean()), deg_max=int(deg.max()),
                deg_nz=int((deg > 0).sum()),
                wsum=float(net.w.sum()),
                elapsed=time.perf_counter() - t0, snaps=snaps)


def _worker(args):
    return run_one(*args)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else nproc

    print('=' * 106)
    print('grow_snn4  N=%d  steps=%d  种子=%d  进程=%d  E_MAX=%d  目标率=%.0f%%' % (
        N, steps, nseed, nproc, E_MAX, TARGET_RATE * 100))
    print('=' * 106)

    tasks = [(s, 'C', steps) for s in range(nseed)]
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(_worker, tasks)
    wall = time.perf_counter() - t0

    print('\n%-6s %8s %8s %8s %8s %8s %7s %7s %8s' % (
        'seed', 'E', 'born', 'died', 'rate%', 'deg_mean', 'deg_max', 'nz', 'wsum'))
    for r in res:
        print('%-6d %8d %8d %8d %8.2f %8.2f %7d %7d %8.1f' % (
            r['seed'], r['E'], r['born'], r['died'], r['rate'] * 100,
            r['deg_mean'], r['deg_max'], r['deg_nz'], r['wsum']))

    Es = np.array([r['E'] for r in res])
    rates = np.array([r['rate'] for r in res]) * 100
    print('\n跨种子: E  mean=%.0f  std=%.0f  (%.1f%%)' % (Es.mean(), Es.std(), 100 * Es.std() / max(Es.mean(), 1)))
    print('        rate mean=%.2f%%  std=%.2f' % (rates.mean(), rates.std()))
    print('\n总耗时 %.1fs (%.0f 步/秒/进程, %.0f 步/秒总)' % (
        wall, steps / (wall / nseed), steps * nseed / wall))


if __name__ == '__main__':
    main()
