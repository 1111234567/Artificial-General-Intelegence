# calib3.py - 加上 E/I 平衡
#
# 前两轮标定的根本缺失（2026-10-23）：
#   我的网络【全是兴奋性连接】，没有抑制 -> 只有「全灭 / 全开」两态，
#   稳态只能把它从一个极端翻到另一个极端 -> 爆发-熄灭振荡。
#
#   真实皮层能在 1~5% 稳定工作，靠的是 E/I 平衡（兴奋驱动 + 抑制反馈）。
#   草案 §11.4 基因表里本来就写了「兴奋:抑制比 1:1 ~ 5:1」，实现时漏了。
#
# 本脚本：80% 兴奋 / 20% 抑制神经元，扫抑制权重比例，看能否开出中间稳态。

import sys, time
import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
TAU = 20.0
DECAY_V = float(np.exp(-1.0 / TAU))
V_TH = 1.0
E0 = 15000
OUT_RATE = 0.002
STEPS = 6000


def trial(args):
    w_e, ratio_i, seed = args
    rng = np.random.default_rng(seed)

    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]

    is_inh = pre >= N_E                       # 前 800 是兴奋，后 200 是抑制
    w = np.where(is_inh, -w_e * ratio_i, w_e).astype(np.float32)
    w *= (1.0 + 0.15 * rng.standard_normal(w.size)).astype(np.float32)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    cnt = np.zeros(N, np.float32)
    acc = 0
    warm = STEPS // 2
    hist = []

    for t in range(STEPS):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        if t >= warm:
            acc += nsp
        if t % 1000 == 0:
            hist.append(round(nsp / N * 100, 2))

    rate = acc / ((STEPS - warm) * N)
    return (w_e, ratio_i, seed, rate, hist)


def main():
    w_es = [0.10, 0.20, 0.30, 0.50]
    ratios = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tasks = [(w, r, 0) for w in w_es for r in ratios]

    print('=' * 96)
    print('E/I 平衡标定   N=%d (E=%d, I=%d)  E=%d 条边  无稳态  steps=%d' % (
        N, N_E, N - N_E, E0, STEPS))
    print('=' * 96)
    with Pool(8) as p:
        t0 = time.perf_counter()
        res = p.map(trial, tasks)
        el = time.perf_counter() - t0

    print('%-8s %-9s %-11s %s' % ('w_E', '抑制比', 'rate', '轨迹'))
    for w, r, sd, rate, hist in res:
        mark = ''
        if 0.8 <= rate * 100 <= 6.0:
            mark = '  <== 目标区'
        print('%-8.2f %-9.2f %-11s %s%s' % (w, r, '%.2f%%' % (rate * 100), hist, mark))

    print('\n耗时 %.1fs' % el)


if __name__ == '__main__':
    main()
