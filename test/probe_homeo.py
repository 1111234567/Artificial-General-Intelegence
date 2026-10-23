# probe_homeo.py - 只回答一个问题：活动能不能自己稳在 2%？
#
# 背景（probe_persist 实测）：
#   关掉输入后：无 STD → 80 步归零；有 STD → 稳在 42.7（饱和）。
#   STD 单独用是**正反馈**：抑制神经元一直放电 → 抑制资源耗尽 → 抑制失效
#                          → 兴奋更强 → 抑制神经元更放电。
#
# 已有的负反馈是阈值适应（放电后 θ 上调，按 τ=100 回落）。
# 它压不住的直接原因：TH_MAX = 3.0 封顶。饱和时驱动目测 > 3。
#
# 本探针扫 TH_MAX，看能不能把自维持压到 2%。
#   能  -> 前腿立住，只是之前封顶太早
#   不能 -> 阈值适应不是足够的负反馈，需要换
#
# 同时记录 x_rec 均值（抑制资源）—— 它接近 0 就说明抑制已经完全失效。

import sys
import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20
W_E = 0.50
W_I_RATIO = 5.0
W_MAX_E = 1.0
W_MAX_I = 2.5
E0 = 20000
N_IN = 100
N_IN_FANOUT = 50
IN_SPIKE = 3.0
IN_LO = 0.001
TAU_ADAPT = 100.0
D_THETA = 0.20
TAU_REC = 100.0
U_STD = 0.30


def build(rng):
    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = (pre != post) & (post >= N_IN)
    pre, post = pre[ok], post[ok]
    src = np.repeat(np.arange(N_IN, dtype=np.int32), N_IN_FANOUT)
    dst = rng.integers(N_IN, N_E, N_IN * N_IN_FANOUT).astype(np.int32)
    pre = np.concatenate([pre, src])
    post = np.concatenate([post, dst])
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    return pre, post, w


def run(args):
    seed, th_max, use_std, warm_pres, tail = args
    rng = np.random.default_rng(seed)
    pre, post, w = build(rng)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    theta = np.full(N, V_TH, np.float32)
    x_rec = np.ones(N, np.float32)

    # warm 阶段：每 100 步给一次输入脉冲（模拟「有事发生」）
    T_WARM = warm_pres * 100
    rates = []
    th_tail = []
    xr_tail = []
    in_tail = []

    for t in range(T_WARM + tail):
        feeding = t < T_WARM
        xin = np.zeros(N, np.float32)
        if feeding and (t % 100) < 2:
            xin[0:25] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE

        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = T_REF

        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, th_max)
        theta += (V_TH - theta) / TAU_ADAPT

        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

        if not feeding:
            rates.append(float(s.sum()))
            th_tail.append(float(theta.mean()))
            xr_tail.append(float(x_rec[N_E:].mean()))
            in_tail.append(float(s[N_E:].mean()))

    rates = np.array(rates)
    seg = [float(rates[i:i + 100].mean()) for i in range(0, min(len(rates), 2000), 100)]
    return dict(seed=seed, th_max=th_max, std=use_std, seg=seg,
                th=float(np.mean(th_tail[-100:])),
                xr=float(np.mean(xr_tail[-100:])),
                inh=float(np.mean(in_tail[-100:])))


def main():
    th_list = [3.0, 6.0, 10.0, 20.0]
    nseed = 3
    tasks = [(s, th, True, 600, 2000) for th in th_list for s in range(nseed)]
    tasks += [(s, 3.0, False, 600, 2000) for s in range(nseed)]

    print('=' * 118)
    print('稳态探针：关掉输入后，活动能不能落在 2% 并且不衰减？')
    print('  N=%d  E0=%d  τ_v=20  T_REF=20  D_THETA=%.2f  τ_adapt=%.0f  U_STD=%.2f' % (
        N, E0, D_THETA, TAU_ADAPT, U_STD))
    print('  warm = 600 次脉冲（每 100 步一次）→ 然后 2000 步全关')
    print('=' * 118)
    with Pool(4) as pool:
        res = pool.map(run, tasks)

    by = {}
    for r in res:
        k = (r['th_max'], r['std'])
        by.setdefault(k, []).append(r)

    print('\n%-22s %s   %8s %8s %8s' % (
        '条件', '  '.join('步%3d-%-3d' % (i * 50, (i + 1) * 50 - 1) for i in range(8)),
        'θ均值', '抑制资源', '抑制率'))
    for k in sorted(by, key=lambda x: (not x[1], x[0])):
        rs = by[k]
        A = np.array([r['seg'] for r in rs])
        m = A.mean(axis=0)
        tag = ('TH_MAX=%-5.1f' % k[0]) if k[1] else '无STD(对照)  '
        print('%-22s %s   %8.2f %8.3f %8.4f' % (
            tag, '  '.join('%8.1f' % x for x in m),
            np.mean([r['th'] for r in rs]),
            np.mean([r['xr'] for r in rs]),
            np.mean([r['inh'] for r in rs])))

    print('\n判据：')
    print('  末段（步 350-399）折算成 rate = 末段值 / %d' % N)
    print('  目标：rate 1~3%%（即 10~30 个/步）且末段不继续下降')
    print('')
    for k in sorted(by, key=lambda x: (not x[1], x[0])):
        rs = by[k]
        A = np.array([r['seg'] for r in rs])
        m = A.mean(axis=0)
        tag = ('TH_MAX=%.1f' % k[0]) if k[1] else '无STD'
        last = m[-1]
        first = m[0]
        verdict = 'OK' if 10 <= last <= 30 else ('归零' if last < 1 else ('饱和' if last > 30 else '偏低'))
        ratio = last / first if first > 1e-9 else 0.0
        print('  %-12s 末段 %6.1f 个/步 = %.2f%%   首末比 %.2f   -> %s' % (
            tag, last, 100.0 * last / N, ratio, verdict))


if __name__ == '__main__':
    main()
