# probe_out.py - 诊断：输出神经元为何不能区分四个词
#
# 问题：grow_teach 的正确率一直平在 25%（随机）。已试过的修正（m_wrong=0 /
# 按神经元分 m / 误差方向 m / 自适应阈值 / V_TH_OUT=0.5 / 接口边）都没改变。
# 先量清楚：输出神经元的放电数与「当前哪个输入块活跃」有无关系？

import numpy as np
import io
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / 'legacy' / 'grow_teach.py'
src = _SOURCE.read_text(encoding='utf-8')
exec(src.split('def run_one')[0])

rng = np.random.default_rng(0)
pre, post, w, age = _init_net(rng)
pre, post, w, key, age = _sort(pre, post, w, age)

v = np.zeros(N, np.float32)
s = np.zeros(N, np.float32)
ref = np.zeros(N, np.int32)
xin = np.zeros(N, np.float32)
vth = np.full(N, V_TH, np.float32)
vth[OUT_IDX] = V_TH_OUT
tref = np.full(N, T_REF, np.int32)
tref[OUT_IDX] = T_REF_OUT

masks = np.zeros((N_WORD, N_IN), np.float32)
for c in range(N_WORD):
    masks[c, c * 25:(c + 1) * 25] = IN_HI
masks[masks == 0] = IN_LO

u = rng.random((TRIAL, N_IN), dtype=np.float32)

iface = (post == OUT_IDX[0]) | (post == OUT_IDX[1])
print('interface edges %d  (500: %d, 501: %d)  w_mean %.3f' % (
    int(iface.sum()), int((post == OUT_IDX[0]).sum()),
    int((post == OUT_IDX[1]).sum()), float(w[iface].mean()) if iface.any() else 0.0))
print('total edges %d' % key.size)

rec = {}
for c in range(N_WORD):
    rec[c] = {'cnt': [], 'v': [], 'ie': [], 'ii': [], 'xin': [], 'rate': []}
for c in range(N_WORD):
    for it in range(40):
        cnt = np.zeros(2, np.int64)
        vs = []
        ies = []
        iis = []
        xins = []
        nsp = 0
        for t in range(TRIAL):
            xin.fill(0.0)
            xin[0:N_IN] = u[t] < masks[c]
            xin[OUT_IDX] += BIAS_OUT
            contrib = w * s[pre]
            I = np.bincount(post, weights=contrib, minlength=N).astype(np.float32)
            m0 = (post == OUT_IDX[0])
            ies.append(float(contrib[m0 & (w > 0)].sum()))
            iis.append(float(contrib[m0 & (w < 0)].sum()))
            xins.append(float(xin[OUT_IDX[0]]))
            v = v * DECAY_V + I + xin
            fire = (v > vth) & (ref <= 0)
            s = fire.astype(np.float32)
            v = np.where(s > 0, 0.0, v)
            ref -= 1
            np.maximum(ref, 0, out=ref)
            ref[s > 0] = tref[s > 0]
            vs.append(float(v[OUT_IDX[0]]))
            nsp += int(s.sum())
            if t >= TRIAL - READ_WIN:
                cnt += s[OUT_IDX].astype(np.int64)
            u[t] = rng.random(N_IN, dtype=np.float32)
        rec[c]['cnt'].append(cnt.copy())
        rec[c]['v'].append(np.mean(vs))
        rec[c]['ie'].append(np.mean(ies))
        rec[c]['ii'].append(np.mean(iis))
        rec[c]['xin'].append(np.mean(xins))
        rec[c]['rate'].append(nsp / float(N))

print('')
print('=== rate 时间轨迹（每 20 步，第 0-400 步）===')
v2 = np.zeros(N, np.float32)
s2 = np.zeros(N, np.float32)
re2 = np.zeros(N, np.int32)
x2 = np.zeros(N, np.float32)
u2 = rng.random((TRIAL, N_IN), dtype=np.float32)
trace_rate = []
for t in range(400):
    c = (t // TRIAL) % N_WORD
    tt = t % TRIAL
    x2.fill(0.0)
    x2[0:N_IN] = u2[tt] < masks[c]
    x2[OUT_IDX] += BIAS_OUT
    I = np.bincount(post, weights=w * s2[pre], minlength=N).astype(np.float32)
    v2 = v2 * DECAY_V + I + x2
    fire = (v2 > vth) & (re2 <= 0)
    s2 = fire.astype(np.float32)
    v2 = np.where(s2 > 0, 0.0, v2)
    re2 -= 1
    np.maximum(re2, 0, out=re2)
    re2[s2 > 0] = tref[s2 > 0]
    trace_rate.append(float(s2.mean()))
    u2[tt] = rng.random(N_IN, dtype=np.float32)
for k in range(0, 400, 20):
    seg = trace_rate[k:k + 20]
    print('  step %3d-%3d  rate %.2f%%  (首步 %.1f%%, 末步 %.1f%%)' % (
        k, k + 19, 100.0 * np.mean(seg), 100.0 * seg[0], 100.0 * seg[-1]))

print('')
print('=== cnt (20-step window) per word ===')
print('%-5s %16s %16s %8s' % ('word', 'cnt500 mean/med/max', 'cnt501 mean/med/max', 'rate%'))
for c in range(N_WORD):
    a = np.array(rec[c]['cnt'])
    print('%-5d %5.1f/%5.0f/%5d %8.1f/%5.0f/%5d %8.2f' % (
        c, a[:, 0].mean(), np.median(a[:, 0]), a[:, 0].max(),
        a[:, 1].mean(), np.median(a[:, 1]), a[:, 1].max(),
        float(np.mean(rec[c]['rate']))))

print('')
print('=== neuron 500 internals (trial mean) ===')
print('%-5s %8s %9s %9s %8s' % ('word', 'v', 'exc_I', 'inh_I', 'ext'))
for c in range(N_WORD):
    print('%-5d %8.3f %9.3f %9.3f %8.3f' % (
        c, float(np.mean(rec[c]['v'])), float(np.mean(rec[c]['ie'])),
        float(np.mean(rec[c]['ii'])), float(np.mean(rec[c]['xin']))))

allc = np.array([np.mean(rec[c]['cnt'], axis=0) for c in range(N_WORD)])
print('')
print('spread of per-word cnt means: %.3f   (if < 0.5 -> output carries no class info)' %
      float(np.linalg.norm(allc - allc.mean(axis=0))))
print('HI_TH=%d   mean cnt ~%.1f' % (HI_TH, float(allc.mean())))
