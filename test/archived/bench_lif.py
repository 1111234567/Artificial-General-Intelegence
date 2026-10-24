# bench_lif.py - LIF 模拟器成本基准（阶段 B 前置）
#
# 目的：量「单个体 10^5 步」的真实耗时，决定阶段 B 的规模上限。
# 网络：N 节点 / 1% 稀疏连接 / LIF 动力学。
# 对照：numpy-dense / numpy-bincount / torch-cpu / torch-cuda-dense / torch-cuda-sparse
#       + batch 维（阶段 A 需并行评估 200~500 个体）

import time
import numpy as np

# ---------------- 网络构造 ----------------

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


# ---------------- 输入 ----------------

def make_input(steps, N, rate=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((steps, N)) < rate).astype(np.float32)


# ---------------- LIF 常量 ----------------
TAU = 20.0
DT = 1.0
DECAY = float(np.exp(-DT / TAU))
V_TH = 1.0
V_RESET = 0.0


# ---------------- 实现 1: numpy dense ----------------

def run_np_dense(N, W, X, steps):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    in_scale = 1.0
    for t in range(steps):
        v = v * DECAY + (W @ s) * in_scale + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, V_RESET, v)
    return float(s.sum())


# ---------------- 实现 2: numpy bincount（稀疏） ----------------

def run_np_sparse(N, pre, post, w, X, steps):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        act = w * s[pre]
        I = np.bincount(post, weights=act, minlength=N).astype(np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, V_RESET, v)
    return float(s.sum())


# ---------------- torch 实现 ----------------

def torch_impl(device, mode, N, W_np, pre, post, w, X, steps, batch=1):
    import torch

    dev = torch.device(device)
    dt = torch.float32

    Xd = torch.from_numpy(X).to(dev).to(dt)          # (T, N)

    if mode == 'dense':
        W = torch.from_numpy(W_np).to(dev).to(dt)     # (N, N)
        v = torch.zeros(batch, N, device=dev, dtype=dt)
        s = torch.zeros(batch, N, device=dev, dtype=dt)
        xs = Xd
        for t in range(steps):
            if batch > 1:
                v = v * DECAY + (s @ W.t()) + xs[t].unsqueeze(0)
            else:
                v = v * DECAY + (s @ W.t()) + xs[t]
            s = (v > V_TH).to(dt)
            v = torch.where(s > 0, torch.zeros((), device=dev, dtype=dt), v)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        return float(s.sum().item())

    # sparse: index_add_
    preT = torch.from_numpy(pre).to(dev)
    postT = torch.from_numpy(post).to(dev)
    wT = torch.from_numpy(w).to(dev).to(dt)
    v = torch.zeros(batch, N, device=dev, dtype=dt)
    s = torch.zeros(batch, N, device=dev, dtype=dt)
    for t in range(steps):
        # (batch, E) = (batch, N) gather then weighted
        contrib = s.index_select(1, preT) * wT          # (batch, E)
        I = torch.zeros(batch, N, device=dev, dtype=dt)
        I.index_add_(1, postT, contrib)                 # (batch, N)
        if batch > 1:
            v = v * DECAY + I + Xd[t].unsqueeze(0)
        else:
            v = v * DECAY + I + Xd[t]
        s = (v > V_TH).to(dt)
        v = torch.where(s > 0, torch.zeros((), device=dev, dtype=dt), v)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return float(s.sum().item())


# ---------------- 计时工具 ----------------

def timeit(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def fmt_row(name, per_step_s, steps_ref, note=''):
    total = per_step_s * steps_ref
    unit = '%.4f s' % total if total < 100 else '%.1f s' % total
    print('  %-26s %10.4f ms/step   %10s for %s steps   %s' % (
        name, per_step_s * 1e3, unit, ('{0:.0e}'.format(steps_ref)), note))


# ---------------- main ----------------

def main():
    N = 1000
    DENSITY = 0.01
    STEPS = 2000
    REF = 100000  # 草案说的 10^5 步

    print('=' * 108)
    print('LIF 模拟器成本基准  N=%d  density=%.2f%%  edges=%d' % (
        N, DENSITY * 100, int(N * N * DENSITY)))
    print('step 参考量: %d 步（外推到 %.0e 步）' % (STEPS, REF))
    print('=' * 108)

    pre, post, w = make_sparse(N, DENSITY, seed=0)
    W_np = to_dense(N, pre, post, w)
    X = make_input(STEPS, N)

    results = []

    print()
    print('[1] CPU 实现')
    t, _ = timeit(lambda: run_np_dense(N, W_np, X, STEPS), warmup=1, repeat=3)
    fmt_row('numpy dense (BLAS)', t / STEPS, REF)
    results.append(('numpy dense', t / STEPS))

    t, _ = timeit(lambda: run_np_sparse(N, pre, post, w, X, STEPS), warmup=1, repeat=3)
    fmt_row('numpy sparse (bincount)', t / STEPS, REF)
    results.append(('numpy sparse', t / STEPS))

    try:
        t, _ = timeit(lambda: torch_impl('cpu', 'dense', N, W_np, pre, post, w, X, STEPS), warmup=1, repeat=2)
        fmt_row('torch cpu dense', t / STEPS, REF)
        results.append(('torch cpu dense', t / STEPS))
    except Exception as e:
        print('  torch cpu dense FAILED:', repr(e))

    print()
    print('[2] GPU 实现（batch=1，单个体）')
    try:
        t, _ = timeit(lambda: torch_impl('cuda', 'dense', N, W_np, pre, post, w, X, STEPS), warmup=3, repeat=3)
        fmt_row('cuda dense', t / STEPS, REF, note='<-- launch 开销主导')
        results.append(('cuda dense b=1', t / STEPS))
    except Exception as e:
        print('  cuda dense FAILED:', repr(e))

    try:
        t, _ = timeit(lambda: torch_impl('cuda', 'sparse', N, W_np, pre, post, w, X, STEPS), warmup=3, repeat=3)
        fmt_row('cuda sparse (index_add)', t / STEPS, REF)
        results.append(('cuda sparse b=1', t / STEPS))
    except Exception as e:
        print('  cuda sparse FAILED:', repr(e))

    print()
    print('[3] GPU batch（阶段 A：并行评估多个体）')
    for B in (64, 256, 512):
        bs = max(50, int(4000 / B))
        try:
            t, _ = timeit(lambda: torch_impl('cuda', 'dense', N, W_np, pre, post, w, X, bs, batch=B),
                          warmup=2, repeat=2)
            per_ind = t / bs / B
            print('  %-26s %10.4f ms/step/个体  B=%d  (%.1f ms/step 全批, %d 步)  -> %.1f s / 个体 / 10^5 步' % (
                'cuda dense batch', per_ind * 1e3, B, t / bs * 1e3, bs, per_ind * REF))
            results.append(('cuda dense b=%d' % B, per_ind))
        except Exception as e:
            print('  batch %d FAILED: %s' % (B, repr(e)))

    print()
    print('=' * 108)
    print('汇总（单个体 10^5 步的总耗时）')
    print('=' * 108)
    for name, ps in results:
        print('  %-26s %10.1f s' % (name, ps * REF))

    print()
    print('[4] CPU 线程数影响（numpy）')
    import os
    print('  os.cpu_count() =', os.cpu_count())


if __name__ == '__main__':
    main()
