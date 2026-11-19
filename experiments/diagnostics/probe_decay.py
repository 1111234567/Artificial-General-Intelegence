# probe_decay.py - 「吸引子」到底存不存在？先测一件事：无输入时活动会不会归零
#
# 起因（Jerry）：「这个吸引子具体指的是什么？」
#
# 查下来发现：项目文档里从没定义过它——agi.md / §16.11 只写了判据
# 「关掉输入后的活动模式两两相似度 < 0.5」。而吸引子的严格含义是
# 「无输入时系统会收敛进去并**留在里面**的子集」。
# 如果活动在几百步内归零，那唯一的吸引子是「全静息」——0.038 只是瞬态尾巴。
#
# 本探针：cue 编码 20 步 → 无输入跑 MAXSTEPS 步，每 50 步统计一次兴奋放电。
# 输出：衰減曲线 + 「归零时间」。
#
# 用法：python experiments/diagnostics/probe_decay.py [seed] [grow_steps] [max_steps]

import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.legacy import grow_homeo as G
from experiments.diagnostics.probe_sim_noise import _step, _warm


def run_cue(pre, post, w, st0, blk, steps, win=50):
    """编码 20 步 → 无输入跑 steps 步；每 win 步统计一次兴奋放电。"""
    v = st0['v'].copy(); s = st0['s'].copy(); ref = st0['ref'].copy()
    theta = st0['theta'].copy(); x_rec = st0['x_rec'].copy()
    B = G.N_IN // G.N_CUE
    zero = np.zeros(G.N, np.float32)
    for t in range(20):
        xin = zero.copy()
        if t == 0 or t == 18:
            xin[blk * B:(blk + 1) * B] = G.IN_SPIKE
        v, s, ref, theta, x_rec = _step(pre, post, w, v, s, ref, theta, x_rec, xin)
    bins = []
    acc = 0.0
    for t in range(steps):
        v, s, ref, theta, x_rec = _step(pre, post, w, v, s, ref, theta, x_rec, zero.copy())
        acc += float(s[:G.N_E].sum())
        if (t + 1) % win == 0:
            bins.append(acc); acc = 0.0
    return bins


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    grow_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print('=' * 100)
    print('无输入时活动会归零吗？  seed=%d  生长 %d 步  观测 %d 步（每 50 步一格）' % (
        seed, grow_steps, max_steps))
    print('=' * 100)

    t0 = time.perf_counter()
    g = G.grow((seed, grow_steps, grow_steps // 2))
    pre, post, w, key = G._sort(g['pre'], g['post'], g['w'])
    print('生长完成：E-E 边 %d  耗时 %.1fs' % (g['n_ee'], time.perf_counter() - t0))

    rng0 = np.random.default_rng(seed + 777)
    st0 = _warm(pre, post, w, rng0, 500)

    # 对照：不给 cue（只给「无输入」）
    print('\n%5s %10s %10s %10s %10s %10s' % ('cue', '0-50', '50-100', '100-150', '150-200', '200-300'))
    for blk in list(range(G.N_CUE)) + [-1]:
        bins = run_cue(pre, post, w, st0, blk, max_steps)
        head = bins[:6]
        label = 'none' if blk < 0 else str(blk)
        print('%5s %10.0f %10.0f %10.0f %10.0f %10.0f   | 总 %d / 末 200 步 %d' % (
            label, head[0], head[1], head[2], head[3], head[4] + head[5],
            sum(bins), sum(bins[4:])))


if __name__ == '__main__':
    main()
