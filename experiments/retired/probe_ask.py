# probe_ask.py - 诊断：保持期到底是什么在驱动输出？
#
# 问题：抢答率恒为 100%。输出已经不接词块了（只接提问区 + 内部 40 个），
# 为什么在 20..119 步还会放电？
#
# 量三个量，分三个时段：
#   I   = 来自接口边的驱动
#   v   = 膜电位
#   bias= 偏置
# 看保持期（ 20..119 ）的 I 到底多大 —— 如果接近阈值，就说明
# 「内部残留的驱动」与「提问信号的驱动」量级重叠，加权求和分不开。

# 历史目标为三段式抢答；当前 grow_talk 已改为 first-spike，所需接口已不存在。
# 保留下面的历史诊断原文供查阅，拒绝把它用于当前实现。
raise SystemExit("不可运行的历史诊断：三段式抢答接口已删除；见 experiments/retired/README.md。")

import io
import numpy as np
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / 'legacy' / 'grow_talk.py'
src = _SOURCE.read_text(encoding='utf-8')
exec(src.split('def probe(')[0])

rng = np.random.default_rng(0)
pre, post, w, age = _init_net(rng)
pre, post, w, key, age = _sort(pre, post, w, age)

# 接口边：看输出神经元的入边构成
for k, o in enumerate(OUT_IDX):
    m = post == o
    srcs = pre[m]
    ww = w[m]
    n_q = int((srcs >= N_IN).sum())
    n_i = int((srcs < N_IN).sum())
    print('输出 %d: 入边 %d 条（提问区 %d / 内部 %d / 词块 %d）  权重均 %.3f  总 %.2f' % (
        o, m.sum(), n_q, n_i, 0,
        ww.mean() if ww.size else 0, ww.sum() if ww.size else 0))
print('')

v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
ref = np.zeros(N, np.int32); trace = np.zeros(N, np.float32)
theta = np.full(N, V_TH, np.float32)
bias = np.zeros(N, np.float32); bias[OUT_IDX] = 0.05
tref = np.full(N, T_REF, np.int32); tref[OUT_IDX] = 4

# 只跑一个呈现：0..499
rec = {'编码': [], '保持': [], '提问': []}
for t in range(T_PRESENT):
    phase = t
    xin = np.zeros(N, np.float32)
    if phase < BURST and phase in (0, BURST - 2):
        xin[0 * BLK:(0 + 1) * BLK] = IN_SPIKE
    if DELAY <= phase < DELAY + QBURST and phase in (DELAY, DELAY + QBURST // 2):
        xin[Q_IDX] = IN_SPIKE
    xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
    xin[Q_IDX] += (rng.random(N_Q) < IN_LO).astype(np.float32) * IN_SPIKE
    xin += bias

    I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
    v = v * DECAY_V + I + xin
    fire = (v > theta) & (ref <= 0)
    s = fire.astype(np.float32)
    v = np.where(s > 0, 0.0, v)
    ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = tref[s > 0]
    if fire.any():
        theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
        ob = fire[OUT_IDX]
        if ob.any():
            oi = OUT_IDX[ob]
            theta[oi] = np.maximum(theta[oi] - D_THETA + D_THETA_OUT, TH_MIN)
    theta += (V_TH - theta) / TAU_ADAPT
    trace *= TRACE_DECAY; trace += s

    if 0 <= phase < BURST:
        rec['编码'].append((float(I[OUT_IDX[0]]), float(v[OUT_IDX[0]]), float(theta[OUT_IDX[0]]), int(fire[OUT_IDX].sum())))
    elif BURST <= phase < DELAY:
        rec['保持'].append((float(I[OUT_IDX[0]]), float(v[OUT_IDX[0]]), float(theta[OUT_IDX[0]]), int(fire[OUT_IDX].sum())))
    elif DELAY <= phase < DELAY + 60:
        rec['提问'].append((float(I[OUT_IDX[0]]), float(v[OUT_IDX[0]]), float(theta[OUT_IDX[0]]), int(fire[OUT_IDX].sum())))

print('%-6s %10s %10s %10s %10s %8s' % ('时段', 'I均', 'I最大', 'v均', 'theta均', '输出放电步数'))
for k in ('编码', '保持', '提问'):
    A = np.array(rec[k])
    if A.size == 0:
        continue
    print('%-6s %10.3f %10.3f %10.3f %10.3f %8d/%d' % (
        k, A[:, 0].mean(), A[:, 0].max(), A[:, 1].mean(), A[:, 2].mean(),
        int(A[:, 3].sum()), A.shape[0]))

print('')
print('判据：保持期的 I 与提问期的 I 若同一量级 -> 加权求和无法区分「何时说」')
print('      需要的是门控（乘法），不是加权')
