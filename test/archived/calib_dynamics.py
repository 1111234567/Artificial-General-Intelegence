# calib_dynamics.py - 先标定动力学，再谈结构
#
# 前面 8 版的教训：我一直在同时调 w 尺度 / vth / gain / 结构规则，
# 结果两头失控（41% 饱和 或 0.19% 死亡），从没落在 2%。
#
# 正确顺序：固定结构，只扫【权重尺度】，看什么值能维持目标放电率。
# 这一步不做，后面所有结构实验的数字都不可信。

import sys, time
import numpy as np
from multiprocessing import Pool

N = 1000
TAU = 20.0
DECAY_V = float(np.exp(-1.0 / TAU))
V_TH = 1.0
E0 = 15000
OUT_RATE = 0.002
STEPS = 4000


def trial(args):
    w_scale, seed = args
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    w = np.full(pre.size, w_scale, np.float32)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    acc = 0
    warm = STEPS // 2
    for t in range(STEPS):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        if t >= warm:
            acc += int(s.sum())
    rate = acc / ((STEPS - warm) * N)
    return (w_scale, seed, rate, pre.size)


def main():
    scales = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]
    seeds = [0, 1]
    tasks = [(w, sd) for w in scales for sd in seeds]

    print('=' * 78)
    print('动力学标定：固定结构 E=%d, N=%d, 只扫权重尺度（每档 2 个种子）' % (E0, N))
    print('=' * 78)
    with Pool(8) as p:
        t0 = time.perf_counter()
        res = p.map(trial, tasks)
        el = time.perf_counter() - t0

    by = {}
    for w, sd, r, E in res:
        by.setdefault(w, []).append(r)

    print('%-10s %-12s %-12s %s' % ('w_scale', 'rate', '每神经元输入', '状态'))
    for w in scales:
        rs = by[w]
        m = np.mean(rs) * 100
        # 全活跃时每神经元收到的输入
        inp = w * (E0 / N) * 1.0
        state = '死亡' if m < 0.1 else ('饱和' if m > 60 else ('** 目标区 **' if 0.8 <= m <= 6 else ''))
        print('%-10.3f %-12s %-12.3f %s' % (
            w, '%.2f%% / %.2f%%' % (rs[0] * 100, rs[1] * 100), inp, state))

    print('\n耗时 %.1fs' % el)
    print('\n（全活跃输入 = w x E/N = w x 15；要越过阈值 1.0，需要 I/(1-0.95) > 1，即 I > 0.05）')


if __name__ == '__main__':
    main()
