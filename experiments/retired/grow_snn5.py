# grow_snn5.py - 修正稳态与衰减时间尺度
#
# v4 的三个问题：
#   1) 放电率两极分化（0.0% / 49.9% / 58.7%），没有落在 2%
#      -> 外部输入不受 gain 控制；稳态窗口只有 20 步，估计噪声太大
#   2) rate 测的是最后一步瞬态 -> 改成尾部窗口平均
#   3) 衰减太快（0.995 / 20 步）：边活不过 8000 步，3 万步里换 1 万条
#      -> 结构是慢变量，衰减必须慢得多
#
# 修正：
#   1) gain 控制全部输入（包括外部注入）；稳态窗口 = 200 步
#   2) rate = 最后 2000 步平均
#   3) DECAY_W = 0.9998（慢），同时加【相对剪除】按竞争力删

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
DECAY_W     = 0.9998     # 慢衰减
W_MAX       = 1.0

T_S         = 20         # 结构更新间隔
T_H         = 200        # 稳态调整间隔
T_G         = 2000
P_GLOBAL    = 6

TARGET_RATE = 0.02
K_HOME      = 0.30
OUT_RATE    = 0.002
MAX_ACTIVE  = 48

P_NEW       = 0.015
E_MAX       = 30000
PRUNE_REL   = 0.40       # 低于该 post 入权重均值的 40% -> 删
THETA_ABS   = 0.003


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
        pre2 = np.concatenate([self.pre, pre.astype(np.int32)])
        post2 = np.concatenate([self.post, post.astype(np.int32)])
        w2 = np.concatenate([self.w, np.full(pre.size, wv, np.float32)])
        self._set(pre2, post2, w2)

    def prune_rel(self, rel, th_abs):
        """相对剪除：按每个 post 的入权重均值。"""
        cnt = np.bincount(self.post, minlength=N).astype(np.float32)
        ssum = np.bincount(self.post, weights=self.w, minlength=N).astype(np.float32)
        mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
        th = np.maximum(mean[self.post] * rel, th_abs)
        m = self.w >= th
        if m.all():
            return 0
        n = self.size() - int(m.sum())
        self.pre = self.pre[m]
        self.post = self.post[m]
        self.w = self.w[m]
        self.key = self.key[m]
        return n


def run_one(seed, mode, steps, E0=8000):
    rng = np.random.default_rng(seed)
    net = Net(E0, rng)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    gain = 1.0
    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    sp_acc = 0
    h_acc = 0
    born = died = 0
    trail = []
    t0 = time.perf_counter()

    for t in range(steps):
        pre, post, w = net.pre, net.post, net.w
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + gain * (I + x)      # gain 控制全部输入
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        sp_acc += nsp
        h_acc += nsp

        # 稳态（200 步窗口）
        if t > 0 and t % T_H == 0:
            rate = h_acc / (T_H * N)
            h_acc = 0
            gain *= (TARGET_RATE / rate) ** K_HOME if rate > 1e-9 else 1.5
            gain = float(np.clip(gain, 1e-4, 1e2))

        # 结构更新
        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0

            net.w *= (1.0 - (1.0 - DECAY_W) * temp)
            died += net.prune_rel(PRUNE_REL, THETA_ABS)

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

        if t >= steps - 2000:
            trail.append(nsp)

    deg = np.bincount(net.post, minlength=N)
    return dict(seed=seed, E=net.size(), born=born, died=died,
                rate=float(np.mean(trail) / N) if trail else 0.0,
                gain=gain, deg_mean=float(deg.mean()), deg_max=int(deg.max()),
                deg_nz=int((deg > 0).sum()), wsum=float(net.w.sum()),
                elapsed=time.perf_counter() - t0,
                w_cv=float(net.w.std() / max(net.w.mean(), 1e-9)))


def _worker(args):
    return run_one(*args)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['C']

    for mode in modes:
        print('=' * 108)
        print('grow_snn5  mode=%s  N=%d  steps=%d  种子=%d  进程=%d' % (
            mode, N, steps, nseed, nproc))
        print('=' * 108)
        tasks = [(s, mode, steps) for s in range(nseed)]
        t0 = time.perf_counter()
        with Pool(nproc) as pool:
            res = pool.map(_worker, tasks)
        wall = time.perf_counter() - t0

        print('%-5s %7s %8s %8s %7s %7s %8s %7s %7s' % (
            'seed', 'E', 'born', 'died', 'rate%', 'gain', 'deg_mean', 'nz', 'w_cv'))
        for r in res:
            print('%-5d %7d %8d %8d %7.2f %7.3f %8.2f %7d %7.3f' % (
                r['seed'], r['E'], r['born'], r['died'], r['rate'] * 100,
                r['gain'], r['deg_mean'], r['deg_nz'], r['w_cv']))
        E = np.array([r['E'] for r in res]); R = np.array([r['rate'] for r in res]) * 100
        print('  跨种子: E %.0f±%.0f (%.1f%%)   rate %.2f%%±%.2f   nz %.0f±%.0f' % (
            E.mean(), E.std(), 100 * E.std() / max(E.mean(), 1), R.mean(), R.std(),
            np.mean([r['deg_nz'] for r in res]), np.std([r['deg_nz'] for r in res])))
        print('  耗时 %.1fs  (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print()


if __name__ == '__main__':
    main()
