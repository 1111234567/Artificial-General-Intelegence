# calib2.py - 验证逐神经元稳态能否把双稳态拉出中间态
#
# 标定1 的发现：均质网络（常数 w）从 0.19% 直接跳到 100%，无中间态。
#   原因：所有神经元入度相同（~15）、权重相同 -> 全或无。
#   全局稳态（一个标量 vth）只能把「全开」翻成「全灭」。
#
# 本脚本对比三种稳态形态，看哪种能稳定在目标放电率附近：
#   A. 无稳态
#   B. 全局阈值稳态（一个标量）
#   C. 逐神经元阈值稳态（firing rate homeostasis）
#   D. 逐神经元 + 异质初始阈值

import sys, time
import numpy as np
from multiprocessing import Pool

N = 1000
TAU = 20.0
DECAY_V = float(np.exp(-1.0 / TAU))
V_TH = 1.0
E0 = 15000
OUT_RATE = 0.002
STEPS = 6000
TARGET = 0.02
T_H = 200
K = 0.25


def trial(args):
    w_scale, variant, seed = args
    rng = np.random.default_rng(seed)

    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    w = (w_scale * (1.0 + 0.2 * rng.standard_normal(pre.size))).astype(np.float32)
    np.clip(w, 0.005, None, out=w)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)

    if variant == 'A':
        vth = np.full(N, V_TH, np.float32)
    elif variant == 'B':
        vth = V_TH                    # 标量
    elif variant == 'C':
        vth = np.full(N, V_TH, np.float32)
    else:  # D
        vth = (V_TH * np.exp(0.6 * rng.standard_normal(N))).astype(np.float32)

    cnt = np.zeros(N, np.float32)
    acc = 0
    warm = STEPS // 2
    hist = []

    for t in range(STEPS):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > vth).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        cnt += s
        if t >= warm:
            acc += nsp

        if t > 0 and t % T_H == 0:
            if variant == 'B':
                r = cnt.sum() / (T_H * N)
                vth = float(np.clip(vth * (1.0 + K * (r - TARGET) / TARGET), 0.05, 50.0))
                cnt[:] = 0
            elif variant in ('C', 'D'):
                r_i = cnt / T_H
                vth = np.clip(vth * (1.0 + K * (r_i - TARGET) / TARGET), 0.02, 50.0).astype(np.float32)
                cnt[:] = 0
            if t % 1000 == 0:
                hist.append(round(nsp / N * 100, 2))

    rate = acc / ((STEPS - warm) * N)
    vth_arr = np.atleast_1d(vth)
    return (w_scale, variant, seed, rate, float(vth_arr.mean()),
            float(vth_arr.std()), hist)


def main():
    scales = [0.10, 0.20, 0.30, 0.50]
    variants = ['A', 'B', 'C', 'D']
    tasks = [(w, var, 0) for w in scales for var in variants]

    print('=' * 92)
    print('稳态形态对比  E=%d  N=%d  target=%.0f%%  steps=%d' % (E0, N, TARGET * 100, STEPS))
    print('  A=无稳态  B=全局阈值  C=逐神经元阈值  D=逐神经元+异质初值')
    print('=' * 92)
    with Pool(8) as p:
        t0 = time.perf_counter()
        res = p.map(trial, tasks)
        el = time.perf_counter() - t0

    print('%-8s %-4s %-10s %-10s %-10s %s' % ('w_scale', '变体', 'rate', 'vth均值', 'vth标准差', '轨迹(每秒采样)'))
    for w, var, sd, rate, vm, vs, hist in res:
        mark = ' <== 目标区' if 1.0 <= rate * 100 <= 4.0 else ''
        print('%-8.2f %-4s %-10s %-10.3f %-10.3f %s%s' % (
            w, var, '%.2f%%' % (rate * 100), vm, vs, hist, mark))

    print('\n耗时 %.1fs' % el)


if __name__ == '__main__':
    main()
