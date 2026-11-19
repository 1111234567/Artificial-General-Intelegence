# bench_lif3.py - 补测：实际 spike rate / 事件驱动 / 10% 密度
#
# 目的：
#   1) 实际 spike rate 是多少（决定事件驱动能不能赢）
#   2) 事件驱动（CSR + 只遍历 active pre 的出边）vs 全边 bincount
#   3) 10% 密度下 dense / sparse 的交叉
#
# 正确性先行：事件驱动实现必须先与 bincount 版本结果一致（同一随机输入、同步数）。

import time
import numpy as np

TAU = 20.0
DECAY = float(np.exp(-1.0 / TAU))
V_TH = 1.0


def make_sparse(N, density, seed=0):
    rng = np.random.default_rng(seed)
    n_edges = max(1, int(N * N * density))
    pre = rng.integers(0, N, n_edges).astype(np.int64)
    post = rng.integers(0, N, n_edges).astype(np.int64)
    scale = 1.0 / np.sqrt(max(1.0, density * N))
    w = (rng.standard_normal(n_edges) * scale).astype(np.float32)
    return pre, post, w


def to_dense(N, pre, post, w):
    W = np.zeros((N, N), dtype=np.float32)
    np.add.at(W, (post, pre), w)
    return W


def make_input(steps, N, rate, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((steps, N)) < rate).astype(np.float32)


def build_csr(N, pre, post, w):
    order = np.argsort(pre, kind='stable')
    pre_s = pre[order]
    post_s = post[order]
    w_s = w[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, pre_s + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, post_s, w_s


# ---- 参考实现：全边 bincount ----
def run_bincount(N, pre, post, w, X, steps, collect_rate=False):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    n_sp = 0
    for t in range(steps):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        n_sp += int(s.sum())
    rate = n_sp / (steps * N)
    return s, rate


# ---- 事件驱动：只遍历 active pre 的出边 ----
def run_event(N, indptr, post_s, w_s, X, steps):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    n_sp = 0
    for t in range(steps):
        active = np.nonzero(s)[0]
        if active.size:
            starts = indptr[active]
            lens = indptr[active + 1] - starts
            total = int(lens.sum())
            if total:
                offs = np.concatenate(([0], np.cumsum(lens[:-1])))
                idx = np.arange(total) - np.repeat(offs - starts, lens)
                I = np.bincount(post_s[idx], weights=w_s[idx], minlength=N).astype(np.float32)
            else:
                I = np.zeros(N, dtype=np.float32)
        else:
            I = np.zeros(N, dtype=np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        n_sp += int(s.sum())
    return s, n_sp / (steps * N)


def timeit(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print('=' * 104)
    print('补测：spike rate / 事件驱动 / 10% 密度')
    print('=' * 104)

    N = 1000
    STEPS = 1000
    pre, post, w = make_sparse(N, 0.01, seed=0)
    indptr, post_s, w_s = build_csr(N, pre, post, w)

    # ---------- 1. 正确性：事件驱动 == bincount ----------
    print()
    print('[1] 正确性校验（事件驱动 vs 全边 bincount，同一输入）')
    X = make_input(STEPS, N, rate=0.02)
    s_ref, r_ref = run_bincount(N, pre, post, w, X, STEPS)
    s_evt, r_evt = run_event(N, indptr, post_s, w_s, X, STEPS)
    same = np.array_equal(s_ref, s_evt)
    print('  最终状态一致: %s   spike rate ref=%.3f%% evt=%.3f%%' % (same, r_ref * 100, r_evt * 100))
    if not same:
        print('  [warn] 实现不一致，速度对比无意义')

    # ---------- 2. 实际 spike rate 随输入强度 ----------
    print()
    print('[2] 实际 spike rate 随外部输入强度（N=1000, 1% 密度）')
    print('  %-12s %-14s %-14s' % ('input rate', 'spike rate', 'active edges/step'))
    E = pre.shape[0]
    for ir in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        Xr = make_input(STEPS, N, rate=ir, seed=7)
        _, sr = run_bincount(N, pre, post, w, Xr, STEPS)
        print('  %-12.3f %-14.3f%% %-14.0f' % (ir, sr * 100, sr * E))

    # ---------- 3. 事件驱动 vs 全边（不同输入强度） ----------
    print()
    print('[3] 速度：事件驱动 vs 全边 bincount（N=1000, 1%% 密度, %d 步）' % STEPS)
    print('  %-12s %-16s %-16s %-10s' % ('input rate', 'bincount ms/step', 'event ms/step', 'speedup'))
    for ir in (0.002, 0.01, 0.05):
        Xr = make_input(STEPS, N, rate=ir, seed=7)
        t1 = timeit(lambda: run_bincount(N, pre, post, w, Xr, STEPS))
        t2 = timeit(lambda: run_event(N, indptr, post_s, w_s, Xr, STEPS))
        a, b = t1 / STEPS, t2 / STEPS
        print('  %-12.3f %-16.4f %-16.4f %-10s' % (ir, a * 1e3, b * 1e3, '%.2fx' % (a / b) if b else 'n/a'))

    # ---------- 4. 10% 密度：dense vs sparse ----------
    print()
    print('[4] 密度扫描：dense(BLAS) vs sparse(bincount)（N=1000, %d 步）' % STEPS)
    print('  %-10s %-10s %-18s %-18s %-10s' % ('density', 'edges', 'dense ms/step', 'sparse ms/step', 'sparse/dense'))
    for dens in (0.005, 0.01, 0.02, 0.05, 0.10):
        p_, q_, w_ = make_sparse(N, dens, seed=0)
        W_ = to_dense(N, p_, q_, w_)
        Xd = make_input(400, N, rate=0.02, seed=3)
        td = timeit(lambda: run_bincount(N, p_, q_, w_, Xd, 400))

        def run_dense():
            v = np.zeros(N, dtype=np.float32)
            s = np.zeros(N, dtype=np.float32)
            for t in range(400):
                v = v * DECAY + (W_ @ s) + Xd[t]
                s = (v > V_TH).astype(np.float32)
                v = np.where(s > 0, 0.0, v)
            return s

        te = timeit(lambda: run_dense())
        a, b = te / 400, td / 400
        print('  %-10.1f%% %-10d %-18.4f %-18.4f %-10s' % (
            dens * 100, p_.shape[0], a * 1e3, b * 1e3, '%.2fx' % (b / a)))

    print()
    print('=' * 104)


if __name__ == '__main__':
    main()
