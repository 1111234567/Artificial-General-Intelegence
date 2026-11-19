# probe_sim_noise.py - 测量面校准：grow_homeo --probe 的相似度到底是不是噪声？
#
# 起因：§16.11 里 seed 0 的「两两相似度 0.405」被判为「有区分（多吸引子）」。
# 但那个数字有三处可疑，本文件逐一量出来：
#
#   ① 太稀疏：300 步 × 800 个兴奋神经元 = 24 万格子，总放电只有 150~500 次
#      （平均每神经元 0.2~0.6 次）——余弦相似度在这个稀疏度下方差很大。
#   ② pre-state 混淆：原 probe 里 4 个 cue 各自跑一遍 500 步预热，且共享同一个 rng，
#      于是「cue 之间的差异」里混进了「起始态不同 + 噪声序列不同」。
#   ③ 只有 between-cue，没有 within-cue 基线：不知道同一 cue 重跑两次会像多少。
#
# 本探针：一次预热 → 冻结 → 每个 cue 重复 3 次（各自独立 rng）
#   within  = 同一 cue、不同重复之间的相似度
#   between = 不同 cue 之间的相似度
#   判据：只有 when within 明显高于 between 时，「有区分」才成立。
#   另附置换对照（打乱神经元标签）作为纯噪声的期望值。
#
# 用法：python experiments/diagnostics/probe_sim_noise.py [seed] [grow_steps]

import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.legacy import grow_homeo as G


def _step(pre, post, w, v, s, ref, theta, x_rec, xin):
    """一步动力学。逐行与 grow_homeo.simulate 内层一致（唯一区别：xin 由调用方给）。"""
    if G.USE_STD:
        w_eff = np.where(pre >= G.N_E, w * x_rec[pre], w)
    else:
        w_eff = w
    I = np.bincount(post, weights=w_eff * s[pre], minlength=G.N).astype(np.float32)
    v = v * G.DECAY_V + I + xin
    fire = (v > theta) & (ref <= 0)
    s = fire.astype(np.float32)
    v = np.where(s > 0, 0.0, v)
    ref -= 1
    np.maximum(ref, 0, out=ref)
    ref[s > 0] = G.T_REF
    if fire.any():
        theta[fire] = np.minimum(theta[fire] + G.D_THETA, G.TH_MAX)
    theta += (G.V_TH - theta) / G.TAU_ADAPT
    if G.USE_STD:
        x_rec += (1.0 - x_rec) / G.TAU_REC
        ii = np.nonzero(s[G.N_E:] > 0)[0] + G.N_E
        if ii.size:
            x_rec[ii] *= (1.0 - G.U_STD)
    return v, s, ref, theta, x_rec


def _warm(pre, post, w, rng, steps=500):
    """一次预热：有背景、无 cue。返回冻结的起始态。"""
    v = np.zeros(G.N, np.float32)
    s = np.zeros(G.N, np.float32)
    ref = np.zeros(G.N, np.int32)
    theta = np.full(G.N, G.V_TH, np.float32)
    x_rec = np.ones(G.N, np.float32)
    for _ in range(steps):
        xin = np.zeros(G.N, np.float32)
        xin[G.N_IN:] = (rng.random(G.N - G.N_IN) < G.BG_RATE).astype(np.float32) * G.IN_SPIKE
        v, s, ref, theta, x_rec = _step(pre, post, w, v, s, ref, theta, x_rec, xin)
    return dict(v=v, s=s, ref=ref, theta=theta, x_rec=x_rec)


def _trial(pre, post, w, st0, blk, rng, enc=20, stat=300):
    """从一个冻结态出发：编码 20 步（cue 块给两拍脉冲、无背景）→ 无输入 300 步。

    返回 300 步里（兴奋神经元）的放电计数向量 + 总放电数。
    """
    v = st0['v'].copy(); s = st0['s'].copy(); ref = st0['ref'].copy()
    theta = st0['theta'].copy(); x_rec = st0['x_rec'].copy()
    B = G.N_IN // G.N_CUE
    zero = np.zeros(G.N, np.float32)
    for t in range(enc):
        xin = zero.copy()
        if t == 0 or t == enc - 2:
            if blk >= 0:
                xin[blk * B:(blk + 1) * B] = G.IN_SPIKE
        v, s, ref, theta, x_rec = _step(pre, post, w, v, s, ref, theta, x_rec, xin)
    tot = np.zeros(G.N, np.float64)
    for _ in range(stat):
        v, s, ref, theta, x_rec = _step(pre, post, w, v, s, ref, theta, x_rec, zero.copy())
        tot += s
    return tot[:G.N_E].copy(), float(tot.sum())


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float('nan')


def _jac(a, b, k=20):
    """top-k 放电神经元的 Jaccard —— 稀疏度下比余弦稳。"""
    ta = set(np.argsort(a)[-k:].tolist())
    tb = set(np.argsort(b)[-k:].tolist())
    return len(ta & tb) / len(ta | tb)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    reps = 3

    print('=' * 100)
    print('测量面校准：probe 的相似度是不是噪声？  seed=%d  生长 %d 步  每 cue 重复 %d 次' % (
        seed, steps, reps))
    print('=' * 100)

    t0 = time.perf_counter()
    g = G.grow((seed, steps, steps // 2))
    pre, post, w, key = G._sort(g['pre'], g['post'], g['w'])
    print('生长完成：E-E 边 %d  耗时 %.1fs' % (g['n_ee'], time.perf_counter() - t0))

    # 一次预热，冻结；所有 trial 从同一状态出发
    rng0 = np.random.default_rng(seed + 777)
    st0 = _warm(pre, post, w, rng0, 500)

    cues = list(range(G.N_CUE))
    vec = {}
    cnt = {}
    for cu in cues:
        for rep in range(reps):
            rng = np.random.default_rng(100000 * seed + 1000 * cu + rep)
            v_, n_ = _trial(pre, post, w, st0, cu, rng)
            vec[(cu, rep)] = v_
            cnt[(cu, rep)] = n_

    print('\n各 cue 各次的总放电（兴奋神经元，300 步）')
    for cu in cues:
        print('  cue %d: %s   均值 %.1f' % (
            cu, ' '.join('%.0f' % cnt[(cu, r)] for r in range(reps)),
            np.mean([cnt[(cu, r)] for r in range(reps)])))

    within = [_cos(vec[(cu, a)], vec[(cu, b)])
              for cu in cues for a in range(reps) for b in range(a + 1, reps)]
    between = [_cos(vec[(cu1, a)], vec[(cu2, b)])
               for i, cu1 in enumerate(cues) for cu2 in cues[i + 1:]
               for a in range(reps) for b in range(reps)]
    within_j = [_jac(vec[(cu, a)], vec[(cu, b)])
                for cu in cues for a in range(reps) for b in range(a + 1, reps)]
    between_j = [_jac(vec[(cu1, a)], vec[(cu2, b)])
                 for i, cu1 in enumerate(cues) for cu2 in cues[i + 1:]
                 for a in range(reps) for b in range(reps)]

    # 置换对照：随机打乱神经元标签 → 纯噪声下的相似度期望
    rngp = np.random.default_rng(seed + 424242)
    perm = [_cos(vec[(cu, 0)], vec[(cu2, 0)][rngp.permutation(G.N_E)])
            for i, cu in enumerate(cues) for cu2 in cues[i + 1:]]

    def stat(name, xs):
        a = np.array([x for x in xs if x == x])
        print('  %-22s n=%3d  均值 %+.3f  中位 %+.3f  p10 %+.3f  p90 %+.3f' % (
            name, a.size, a.mean(), np.median(a),
            np.percentile(a, 10), np.percentile(a, 90)))

    print('\n余弦相似度')
    stat('within-cue（同 cue）', within)
    stat('between-cue（不同 cue）', between)
    stat('置换对照（纯噪声）', perm)
    print('\ntop-20 Jaccard')
    stat('within-cue', within_j)
    stat('between-cue', between_j)

    mw = np.nanmean(within); mb = np.nanmean(between)
    print('\n差值 within − between = %+.3f' % (mw - mb))
    print('判定：%s' % ('有区分（within 明显高于 between）' if mw - mb > 0.20 else
                       '**无区分——原 probe 的相似度差异可能只是噪声**' if mw - mb < 0.10 else
                       '弱区分（差距 < 0.20，不足以判多吸引子）'))


if __name__ == '__main__':
    main()
