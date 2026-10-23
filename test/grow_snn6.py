# grow_snn6.py - 硬预算 + 竞争（修正结构正反馈）
#
# v5 的根本问题：结构侧没有负反馈
#   新建边无代价 -> E 从 8000 涨到 28000，died 只有个位数
#   正反馈：rate 高 -> 共激活多 -> 加边 -> 输入强 -> rate 更高
#   gain 是唯一负反馈，但它是对称乘性的，追不上结构膨胀
#
# v6：E 严格固定 = BUDGET。每新建一条，就淘汰一条最弱的。
#   -> 结构不再膨胀，而是【内部重组】
#   -> 新边初始权重低（0.02），需要反复被共激活才活下来
#   -> 强弱完全由使用决定（用进废退）
#
# 观测重点变了：E 恒定后，看【组成】是否收敛：turnover / 度分布 / 权重分布

import sys
import time
import numpy as np
from multiprocessing import Pool

N           = 1000
TAU         = 20.0
DECAY_V     = float(np.exp(-1.0 / TAU))
V_TH        = 1.0

W_INIT      = 0.02      # 新边很弱
W_STRONG    = 0.30      # 初始边的强度
ETA_POT     = 0.04
DECAY_W     = 0.9995
W_MAX       = 1.0

BUDGET      = 15000     # 硬预算（固定）

T_S         = 20
T_H         = 200
T_G         = 2000
P_GLOBAL    = 4

TARGET_RATE = 0.02
K_HOME      = 0.30
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
        np.clip(w, 0.01, None, out=w)
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

    def add_competitive(self, pre, post, budget):
        """新建，并按权重竞争淘汰等量的弱边（硬预算）。"""
        k = pre.size
        if k == 0:
            return 0
        old_pre, old_post, old_w = self.pre, self.post, self.w
        # 合并后淘汰最弱的 k 条（新边也算在内）
        all_pre = np.concatenate([old_pre, pre.astype(np.int32)])
        all_post = np.concatenate([old_post, post.astype(np.int32)])
        all_w = np.concatenate([old_w, np.full(k, W_INIT, np.float32)])
        total = all_w.size
        if total <= budget:
            self._set(all_pre, all_post, all_w)
            return 0
        drop = total - budget
        idx_drop = np.argpartition(all_w, drop - 1)[:drop]
        mask = np.ones(total, bool)
        mask[idx_drop] = False
        self._set(all_pre[mask], all_post[mask], all_w[mask])
        return int(drop)


def run_one(seed, mode, steps, E0=None):
    rng = np.random.default_rng(seed)
    net = Net(E0 or BUDGET, rng)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    gain = 1.0
    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    h_acc = 0
    born = dropped = 0
    trail = []
    prev_key = None
    tovs = []
    t0 = time.perf_counter()

    for t in range(steps):
        pre, post, w = net.pre, net.post, net.w
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + gain * (I + x)
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        h_acc += nsp

        if t > 0 and t % T_H == 0:
            rate = h_acc / (T_H * N)
            h_acc = 0
            gain *= (TARGET_RATE / rate) ** K_HOME if rate > 1e-9 else 1.5
            gain = float(np.clip(gain, 1e-4, 1e2))

        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0
            net.w *= (1.0 - (1.0 - DECAY_W) * temp)

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
                np_, nq_ = q_pre[~found], q_post[~found]
                if np_.size:
                    pick = rng.random(np_.size) < (P_NEW * temp)
                    if pick.any():
                        born += int(pick.sum())
                        dropped += net.add_competitive(np_[pick], nq_[pick], BUDGET)

            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
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

        if t >= steps - 2000:
            trail.append(nsp)

    deg = np.bincount(net.post, minlength=N)
    return dict(seed=seed, E=net.size(), born=born, dropped=dropped,
                rate=float(np.mean(trail) / N) if trail else 0.0,
                gain=gain, deg_mean=float(deg.mean()), deg_max=int(deg.max()),
                deg_nz=int((deg > 0).sum()), wsum=float(net.w.sum()),
                w_mean=float(net.w.mean()), w_max=float(net.w.max()),
                w_cv=float(net.w.std() / max(net.w.mean(), 1e-9)),
                tov_last=float(tovs[-1]) if tovs else float('nan'),
                tov_all=tovs, elapsed=time.perf_counter() - t0)


def _worker(args):
    return run_one(*args)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['C']

    for mode in modes:
        print('=' * 112)
        print('grow_snn6  mode=%s  N=%d  steps=%d  种子=%d  预算=%d  目标率=%.0f%%' % (
            mode, N, steps, nseed, BUDGET, TARGET_RATE * 100))
        print('=' * 112)
        tasks = [(s, mode, steps) for s in range(nseed)]
        t0 = time.perf_counter()
        with Pool(nproc) as pool:
            res = pool.map(_worker, tasks)
        wall = time.perf_counter() - t0

        print('%-5s %6s %7s %7s %7s %7s %8s %6s %6s %7s' % (
            'seed', 'E', 'born', 'drop', 'rate%', 'gain', 'deg_mean', 'nz', 'w_cv', 'turnov'))
        for r in res:
            print('%-5d %6d %7d %7d %7.2f %7.3f %8.2f %6d %6.3f %7.3f' % (
                r['seed'], r['E'], r['born'], r['dropped'], r['rate'] * 100,
                r['gain'], r['deg_mean'], r['deg_nz'], r['w_cv'], r['tov_last']))
        E = np.array([r['E'] for r in res]); R = np.array([r['rate'] for r in res]) * 100
        Nz = np.array([r['deg_nz'] for r in res]); T = np.array([r['tov_last'] for r in res])
        print('  跨种子: E %.0f±%.0f   rate %.2f%%±%.2f   nz %.0f±%.0f   tov %.3f±%.3f' % (
            E.mean(), E.std(), R.mean(), R.std(), Nz.mean(), Nz.std(),
            np.nanmean(T), np.nanstd(T)))
        print('  耗时 %.1fs  (总 %.0f 步/秒)' % (wall, steps * nseed / wall))
        print()


if __name__ == '__main__':
    main()
