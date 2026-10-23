# scan_base.py - 扫出「无 E-E 时 rate 落在目标下方」的工作点
#
# 为什么要扫：grow_homeo 实测 BG 从 0.004 到 0.015（3.75 倍），
# rate 始终 0.07% —— 说明不是背景驱动不够，是**抑制在主导**：
#   200 个抑制神经元，每个扇出到 700×0.2 = 140 个兴奋神经元
#   每步 3 个抑制放电 × 140 = 240 个兴奋被 -2.5 打击
#   而只有 10.5 个兴奋得到 +3.0
#
# 扫两个参数：W_I（抑制权重）与 D_IE（抑制→兴奋连接概率）。
# 目标：无 E-E 时 rate 在 0.2~0.5%（低于目标 0.8% 的区间下限 0.64%）

import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
N_IN = 100
N_IN_FANOUT = 50
IN_SPIKE = 3.0
BG_RATE = 0.015
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20
W_E = 0.5
D_EI = 0.2
TAU_ADAPT = 100.0
D_THETA = 0.2
TH_MAX = 3.0
TAU_REC = 100.0
U_STD = 0.30


def run(args):
    seed, w_i, d_ie, d_ei, steps = args
    rng = np.random.default_rng(seed)
    pre, post = [], []

    n = int(N_E * (N - N_E) * d_ei)
    pre.append(rng.integers(0, N_E, n).astype(np.int32))
    post.append(rng.integers(N_E, N, n).astype(np.int32))

    n = int(N_E * (N - N_E) * d_ie)
    pre.append(rng.integers(N_E, N, n).astype(np.int32))
    post.append(rng.integers(0, N_E, n).astype(np.int32))

    src = np.repeat(np.arange(N_IN, dtype=np.int32), N_IN_FANOUT)
    dst = rng.integers(N_IN, N, N_IN * N_IN_FANOUT).astype(np.int32)
    pre.append(src); post.append(dst)

    pre = np.concatenate(pre)
    post = np.concatenate(post)
    w = np.where(pre >= N_E, w_i, W_E).astype(np.float32)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    theta = np.full(N, V_TH, np.float32)
    x_rec = np.ones(N, np.float32)
    rate = np.zeros(N, np.float64)

    for t in range(steps):
        xin = np.zeros(N, np.float32)
        xin[N_IN:] = (rng.random(N - N_IN) < BG_RATE).astype(np.float32) * IN_SPIKE
        w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = T_REF
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
        theta += (V_TH - theta) / TAU_ADAPT
        x_rec += (1.0 - x_rec) / TAU_REC
        ii = np.nonzero(s[N_E:] > 0)[0] + N_E
        if ii.size:
            x_rec[ii] *= (1.0 - U_STD)
        if t > steps // 2:
            rate += s / (steps - steps // 2)

    return dict(seed=seed, w_i=w_i, d_ie=d_ie, d_ei=d_ei,
                re=float(rate[:N_E].mean()), ri=float(rate[N_E:].mean()),
                nact=int((rate[:N_E] > 1e-4).sum()),
                tot=float(rate.sum()))


def main():
    nseed = 2
    steps = 4000
    combos = []
    for w_i in (-2.5, -1.5, -1.0, -0.6):
        for d_ie in (0.2, 0.1, 0.05):
            combos.append((w_i, d_ie, 0.2))
    # 额外：降 E→I
    for w_i in (-1.5, -1.0):
        combos.append((w_i, 0.1, 0.1))

    tasks = [(s, wi, di, de, steps) for (wi, di, de) in combos for s in range(nseed)]
    print('=' * 96)
    print('扫描：无 E-E 时，什么参数组合能让 rate 落在 0.2~0.5%？')
    print('  N=%d  BG=%.3f  W_E=%.2f  目标区间下限 0.64%%' % (N, BG_RATE, W_E))
    print('=' * 96)
    with Pool(8) as pool:
        res = pool.map(run, tasks)

    by = {}
    for r in res:
        by.setdefault((r['w_i'], r['d_ie'], r['d_ei']), []).append(r)

    print('%-10s %8s %10s %10s %8s %8s' % ('W_I', 'D_IE', 'D_EI', 'E_rate%', 'I_rate%', '活跃数'))
    print('-' * 96)
    for k in sorted(by, key=lambda x: (x[0], x[1], x[2])):
        rs = by[k]
        re_ = np.mean([r['re'] for r in rs]) * 100
        ri_ = np.mean([r['ri'] for r in rs]) * 100
        na = np.mean([r['nact'] for r in rs])
        mark = ''
        if 0.15 <= re_ <= 0.60:
            mark = '   <-- 合适（低于目标区间）'
        elif re_ > 0.64:
            mark = '   (已在目标区间或以上，不需要长)'
        print('%-10.2f %8.2f %10.2f %10.3f %8.3f %8.0f%s' % (k[0], k[1], k[2], re_, ri_, na, mark))


if __name__ == '__main__':
    main()
