# probe_in.py - 最小验证：stim 能不能让输入区放电
import io, numpy as np

src = io.open('test/grow_talk.py', encoding='utf-8').read()
exec(src.split('def run_one')[0])

rng = np.random.default_rng(0)
pre, post, w, age = _init_net(rng)
pre, post, w, key, age = _sort(pre, post, w, age)

print('N_IN=%d  BLK=%d  IN_SPIKE=%g  V_TH=%g  T_REF=%d' % (N_IN, BLK, IN_SPIKE, V_TH, T_REF))
print('指向输入区的边数: %d  (应为 0)' % int((post < N_IN).sum()))
print('从输入区出的边数: %d' % int((pre < N_IN).sum()))
iface = np.isin(post, OUT_IDX)
print('接口边（→输出）: %d  权重均值 %.3f' % (int(iface.sum()), float(w[iface].mean())))
w_out = w[iface]
pre_out = pre[iface]
print('  其中来自输入区: %d 条，权重均值 %.3f' % (int((pre_out < N_IN).sum()), float(w_out[pre_out < N_IN].mean())))
print('  来自输入区的边按目标块分布:', [int(((pre_out >= b*BLK) & (pre_out < (b+1)*BLK)).sum()) for b in range(4)])

v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
ref = np.zeros(N, np.int32)
theta = np.full(N, V_TH, np.float32)
bias = np.zeros(N, np.float32); bias[OUT_IDX] = 0.05
tref = np.full(N, T_REF, np.int32); tref[OUT_IDX] = 4

# 只跑 3 步，第 0 步给词 0 的脉冲
for t in range(3):
    xin = np.zeros(N, np.float32)
    if t == 0:
        xin[0:BLK] = IN_SPIKE
    xin += bias
    I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
    v = v * DECAY_V + I + xin
    fire = (v > theta) & (ref <= 0)
    s = fire.astype(np.float32)
    v = np.where(s > 0, 0.0, v)
    ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = tref[s > 0]
    print('t=%d  xin[0:25].sum=%.1f  v[0:25].max=%.2f  s[0:100]=%d  s[500:504]=%d  I[500]=%.2f' % (
        t, float(xin[0:BLK].sum()), float(v[0:BLK].max()), int(s[0:N_IN].sum()),
        int(s[OUT_IDX].sum()), float(I[OUT_IDX[0]])))
