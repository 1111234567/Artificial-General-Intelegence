# calib4.py - 细扫目标区间
#
# calib3 的关键进展：加上抑制后，w_E=0.50 时 rate 随抑制比单调下降（100% -> 46%），
# 且轨迹【稳定】（46.6 / 46.6），不再振荡或双稳。E/I 平衡是对的。
#
# 本脚本细扫 w_E 与抑制比，找 rate ≈ 2% 的稳定工作点。

import time
import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
E0 = 15000
OUT_RATE = 0.002
STEPS = 8000


def trial(args):
    w_e, ratio_i, seed = args
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    is_inh = pre >= N_E
    w = np.where(is_inh, -w_e * ratio_i, w_e).astype(np.float32)
    w *= (1.0 + 0.15 * rng.standard_normal(w.size)).astype(np.float32)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    acc = 0
    warm = STEPS // 2
    tail = []

    for t in range(STEPS):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        if t >= warm:
            acc += nsp
        if t >= STEPS - 500:
            tail.append(nsp / N)

    rate = acc / ((STEPS - warm) * N)
    tail_std = float(np.std(tail) * 100)
    return (w_e, ratio_i, seed, rate, tail_std, float(np.mean(tail) * 100))


def main():
    w_es = [0.35, 0.40, 0.45, 0.50, 0.60]
    ratios = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    tasks = [(w, r, 0) for w in w_es for r in ratios]

    print('=' * 88)
    print('细扫  无稳态  E/I  目标 2%%')
    print('=' * 88)
    with Pool(12) as p:
        t0 = time.perf_counter()
        res = p.map(trial, tasks)
        el = time.perf_counter() - t0

    cur_w = None
    for w, r, sd, rate, tstd, tmean in res:
        if w != cur_w:
            print()
            print('  w_E = %.2f' % w)
            cur_w = w
        mark = '  <== 目标区 (且稳定)' if (1.5 <= rate * 100 <= 3.5 and tstd < 1.0) else ''
        print('    抑制比 %-5.1f  rate=%6.2f%%  尾部均值=%6.2f%%  尾部波动=%.3f  %s' % (
            r, rate * 100, tmean, tstd, mark))

    print('\n耗时 %.1fs' % el)


if __name__ == '__main__':
    main()
