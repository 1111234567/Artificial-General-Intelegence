# calib5.py - 加不应期（re refractory period）
#
# 关键缺失（2026-10-23）：我的 LIF 没有不应期。
#   神经元放电后立刻可以再放 -> 最大放电率无上界
#   -> 抑制比拉到 10 倍，rate 仍卡在 20~30% 降不下去
#   -> 实际上：放电率的上界【应该由不应期给出】，而不是靠精细调参
#
# 加 T_REF 步不应期后：单神经元最大放电率 = 1/T_REF
#   T_REF=50 -> 最大 2%（正好对齐目标）
#   T_REF=20 -> 最大 5%

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
    w_e, ratio_i, t_ref, seed = args
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
    ref = np.zeros(N, np.int32)
    acc = 0
    warm = STEPS // 2
    tail = []
    exc_max = 0

    for t in range(STEPS):
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        fire = (v > V_TH) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = t_ref
        nsp = int(s.sum())
        if t >= warm:
            acc += nsp
            if nsp > exc_max:
                exc_max = nsp
        if t >= STEPS - 500:
            tail.append(nsp / N)

    rate = acc / ((STEPS - warm) * N)
    return (w_e, ratio_i, t_ref, seed, rate,
            float(np.std(tail) * 100), float(np.mean(tail) * 100))


def main():
    w_es = [0.15, 0.25, 0.35, 0.50]
    ratios = [1.0, 2.0, 3.0, 5.0]
    trefs = [50, 20]
    tasks = [(w, r, tr, 0) for w in w_es for r in ratios for tr in trefs]

    print('=' * 92)
    print('加不应期后标定  目标 2%%   T_REF=50 -> 上限 2%% ; T_REF=20 -> 上限 5%%')
    print('=' * 92)
    with Pool(16) as p:
        t0 = time.perf_counter()
        res = p.map(trial, tasks)
        el = time.perf_counter() - t0

    print('%-8s %-8s %-7s %-10s %-11s %s' % ('w_E', '抑制比', 'T_REF', 'rate', '尾部波动', '状态'))
    for w, r, tr, sd, rate, tstd, tmean in res:
        st = ''
        if 1.0 <= rate * 100 <= 4.0 and tstd < 1.0:
            st = '** 目标区 **'
        elif rate * 100 < 0.5:
            st = '死亡'
        elif rate * 100 > 50:
            st = '饱和'
        print('%-8.2f %-8.1f %-7d %-10s %-11.3f %s' % (
            w, r, tr, '%.2f%%' % (rate * 100), tstd, st))

    print('\n耗时 %.1fs' % el)


if __name__ == '__main__':
    main()
