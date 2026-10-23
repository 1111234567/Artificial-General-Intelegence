# bench_lif2.py - LIF 模拟器成本基准 v2
#
# v1 的三个缺陷，本版修正：
#   1) batch 档只跑 50 步 -> 噪声/launch 开销主导，数据不可信 -> 固定 >=500 步
#   2) 只测了裸前向 -> 阶段 B 还要可塑性（资格迹 + 权重更新）与状态记录 -> 分项计
#   3) 只测 N=1000 -> 草案上限 N=2000 -> 扫两档
#
# 所有外推都注明「实测 per-step × 参考步数」，并且报告实测步数。

import time
import numpy as np

TAU = 20.0
DT = 1.0
DECAY = float(np.exp(-DT / TAU))
V_TH = 1.0

TAU_E = 20.0          # 资格迹衰正常数
DECAY_E = float(np.exp(-DT / TAU_E))
LR = 1e-3


# ---------------- 构造 ----------------

def make_sparse(N, density, seed=0):
    rng = np.random.default_rng(seed)
    n_edges = max(1, int(N * N * density))
    pre = rng.integers(0, N, n_edges).astype(np.int64)
    post = rng.integers(0, N, n_edges).astype(np.int64)
    scale = 1.0 / np.sqrt(max(1.0, density * N))
    w = (rng.standard_normal(n_edges) * scale).astype(np.float32)
    return pre, post, w, n_edges


def to_dense(N, pre, post, w):
    W = np.zeros((N, N), dtype=np.float32)
    np.add.at(W, (post, pre), w)
    return W


def make_input(steps, N, rate=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((steps, N)) < rate).astype(np.float32)


# ---------------- CPU：前向 only ----------------

def fwd_np_sparse(N, pre, post, w, X, steps):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
    return s


def fwd_np_dense(N, W, X, steps):
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        v = v * DECAY + (W @ s) + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
    return s


# ---------------- CPU：前向 + 可塑性 ----------------

def fwd_plus_trace(N, pre, post, w, X, steps):
    """前向 + 资格迹（O(E)/步），不做权重写回。"""
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    tr = np.zeros(pre.shape[0], dtype=np.float32)
    for t in range(steps):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        tr *= DECAY_E
        tr += s[pre] * s[post]
    return s


def fwd_plus_plastic(N, pre, post, w, X, steps):
    """前向 + 资格迹 + 权重写回（O(E)/步 × 2）。"""
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    tr = np.zeros(pre.shape[0], dtype=np.float32)
    ww = w.copy()
    for t in range(steps):
        I = np.bincount(post, weights=ww * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY + I + X[t]
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        tr *= DECAY_E
        tr += s[pre] * s[post]
        ww += LR * tr
        np.clip(ww, -2.0, 2.0, out=ww)
    return s


# ---------------- GPU：批量（阶段 A / 多种子） ----------------

def gpu_batch_dense(N, W_np, X, steps, B, plasticity=False, record=False, pre=None, post=None, w_np=None):
    import torch
    dev = torch.device('cuda')
    ft = torch.float32
    W = torch.from_numpy(W_np).to(dev)
    Xd = torch.from_numpy(X).to(dev)
    v = torch.zeros(B, N, device=dev, dtype=ft)
    s = torch.zeros(B, N, device=dev, dtype=ft)
    rec = []

    if plasticity:
        preT = torch.from_numpy(pre).to(dev)
        postT = torch.from_numpy(post).to(dev)
        wT = torch.from_numpy(w_np).to(dev)
        tr = torch.zeros(B, pre.shape[0], device=dev, dtype=ft)

    for t in range(steps):
        v = v * DECAY + (s @ W.t()) + Xd[t].unsqueeze(0)
        s = (v > V_TH).to(ft)
        v = torch.where(s > 0, torch.zeros((), device=dev, dtype=ft), v)
        if plasticity:
            tr = tr * DECAY_E + s.index_select(1, preT) * s.index_select(1, postT)
            wT = wT + LR * tr.mean(0)
        if record and (t % 100 == 0):
            rec.append(v.clone())

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    out = float(s.sum().item())
    if record:
        out += sum(float(r.sum().item()) for r in rec)
    return out


# ---------------- 计时 ----------------

def timeit(fn, warmup=2, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print('=' * 112)
    print('LIF 模拟器成本基准 v2   (每档实测 >=500 步，外推 1e5 / 1e7)')
    print('=' * 112)

    for N, DENSITY in ((1000, 0.01), (2000, 0.01)):
        pre, post, w, E = make_sparse(N, DENSITY, seed=0)
        W_np = to_dense(N, pre, post, w)
        STEPS = 1000
        X = make_input(STEPS, N)

        print()
        print('-' * 112)
        print('N=%d  density=%.1f%%  edges=%d  steps_measured=%d' % (N, DENSITY * 100, E, STEPS))
        print('-' * 112)

        rows = []

        t = timeit(lambda: fwd_np_dense(N, W_np, X, STEPS))
        rows.append(('CPU numpy dense (BLAS)', t / STEPS))

        t = timeit(lambda: fwd_np_sparse(N, pre, post, w, X, STEPS))
        rows.append(('CPU numpy sparse (bincount)', t / STEPS))

        t = timeit(lambda: fwd_plus_trace(N, pre, post, w, X, STEPS))
        rows.append(('CPU sparse + eligibility trace', t / STEPS))

        t = timeit(lambda: fwd_plus_plastic(N, pre, post, w, X, STEPS))
        rows.append(('CPU sparse + trace + weight update', t / STEPS))

        for name, ps in rows:
            print('  %-38s %9.4f ms/step  |  %8.1f s /1e5  |  %8.1f min /1e7' % (
                name, ps * 1e3, ps * 1e5, ps * 1e7 / 60))

        # GPU 单个体（对照：证明 launch 开销）
        try:
            t = timeit(lambda: gpu_batch_dense(N, W_np, X, STEPS, 1), warmup=3, repeat=2)
            ps = t / STEPS
            print('  %-38s %9.4f ms/step  |  %8.1f s /1e5  |  %8.1f min /1e7   <-- B=1 单个体' % (
                'GPU cuda dense', ps * 1e3, ps * 1e5, ps * 1e7 / 60))
        except Exception as e:
            print('  GPU dense FAILED:', repr(e))

        # GPU 批量（阶段 A：并行个体 / 阶段 B：并行种子）
        print()
        print('  GPU 批量（每批全部个体共享同一步进，吞吐按个体摊）:')
        for B in (16, 64, 256, 512):
            bs = 500
            Xb = make_input(bs, N)
            try:
                t = timeit(lambda: gpu_batch_dense(N, W_np, Xb, bs, B), warmup=2, repeat=2)
                per_ind = t / bs / B
                print('    B=%-4d  全批 %8.4f ms/step  |  单个体 %10.6f ms/step  ->  %7.1f s /1e5/个体  %6.1f min /1e7/个体' % (
                    B, t / bs * 1e3, per_ind * 1e3, per_ind * 1e5, per_ind * 1e7 / 60))
            except Exception as e:
                print('    B=%d FAILED: %s' % (B, repr(e)))

        # GPU 批量 + 可塑性
        print()
        print('  GPU 批量 + 资格迹/权重更新（阶段 B 真实形态）:')
        for B in (16, 64, 256):
            bs = 500
            Xb = make_input(bs, N)
            try:
                t = timeit(lambda: gpu_batch_dense(N, W_np, Xb, bs, B, plasticity=True, pre=pre, post=post, w_np=w),
                           warmup=2, repeat=2)
                per_ind = t / bs / B
                print('    B=%-4d  全批 %8.4f ms/step  |  单个体 %10.6f ms/step  ->  %7.1f s /1e5/个体  %6.1f min /1e7/个体' % (
                    B, t / bs * 1e3, per_ind * 1e3, per_ind * 1e5, per_ind * 1e7 / 60))
            except Exception as e:
                print('    B=%d FAILED: %s' % (B, repr(e)))

    print()
    print('=' * 112)
    print('完毕')


if __name__ == '__main__':
    main()
