# probe_persist.py - 测「稳态」到底缺不缺，不加任何机制
#
# 问题：我一直假设「跨 mode 差 6 倍」就是稳态的缺陷，但没测过。
# 真正的问题应该是：**没有输入时，网络能不能自己维持一个非零的活动水平？**
#
# 测法：给一次刺激（20 步），然后关掉一切输入（包括背景噪声），
# 记录 rate 随时间。
#   20 步内抳到 0  -> 纯衰减，没有自我维持（信息只能活在膜电位里，所以跨度 = τ）
#   能维持 -> 有自我维持（那记忆不靠 τ，稳态不缺）
#
# 四种条件：
#   noise_on   背景噪声保留（现在的状态）
#   noise_off  全部输入关闭
#   noise_off+std  同上 + STD
#   noise_off+scale 同上 + 突触缩放

import io
import numpy as np

src = io.open(r'F:\agi\test\grow_talk.py', encoding='utf-8').read()
exec(src.split('def probe(')[0])


def run(args):
    seed, cond, warm = args
    rng = np.random.default_rng(seed)
    pre, post, w, age = _init_net(rng)
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32); trace = np.zeros(N, np.float32)
    theta = np.full(N, V_TH, np.float32)
    bias = np.zeros(N, np.float32); bias[OUT_IDX] = 0.05
    tref = np.full(N, T_REF, np.int32); tref[OUT_IDX] = 4
    x_rec = np.ones(N, np.float32)
    is_inh_e = pre >= N_E

    noise_on = (cond == 'noise_on')
    use_std = cond.endswith('+std')
    use_scale = cond.endswith('+scale')
    grp_int = ~np.isin(post, OUT_IDX)
    tgt_int = float(np.abs(w[grp_int]).mean()) if grp_int.any() else 1.0

    # 阶段 1：warm 步「正常喂」（带刺激与噪声），让网络进入稳态
    T_WARM = warm
    T_TAIL = 300
    c = 0; burst_at = np.zeros(1, np.int32)
    rates = []
    for t in range(T_WARM + T_TAIL):
        phase = t % T_PRESENT
        feeding = (t < T_WARM)
        if phase == 0 and feeding:
            c = int(rng.integers(0, N_WORD))
            burst_at = np.array([0], np.int32)
        xin = np.zeros(N, np.float32)
        if feeding and phase < BURST and phase in burst_at:
            xin[c * BLK:(c + 1) * BLK] = IN_SPIKE
        if noise_on or feeding:
            xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
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

        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

        trace *= TRACE_DECAY; trace += s

        if use_scale and feeding and t % 20 == 0 and t > 0:
            cur = float(np.abs(w[grp_int]).mean())
            if cur > tgt_int * 1.05:
                w[grp_int] *= np.float32(tgt_int / cur)

        # 阶段 2：记录 rate（只用前 20 步作基线，后续是「无输入下的衰减」）
        if not feeding:
            rates.append(float(s.sum()))

    rates = np.array(rates)
    # 分段平均（每 20 步一段）
    seg = [float(rates[i:i + 20].mean()) for i in range(0, min(len(rates), 200), 20)]
    return dict(seed=seed, cond=cond, seg=seg, kind=float(rates[:20].mean()))


def main():
    conds = ['noise_on', 'noise_off', 'noise_off+std', 'noise_off+scale']
    nseed = 4
    warm = 200 * T_PRESENT
    print('=' * 110)
    print('稳态诊断：关掉一切输入后，网络能不能自己维持活动？')
    print('  warm = %d 步（正常喂刺激+噪声）→ 然后 300 步全关，看 rate 轨迹' % warm)
    print('  N=%d  脉冲窗=%d  T_REF=%d  τ_v=%g' % (N, BURST, T_REF, 20.0))
    print('=' * 110)
    with Pool(4) as pool:
        res = pool.map(run, [(s, c, warm) for c in conds for s in range(nseed)])

    by = {}
    for r in res:
        by.setdefault(r['cond'], []).append(r['seg'])

    print('\n%-16s %s' % ('条件', '  '.join('步%3d-%-3d' % (i * 20, (i + 1) * 20 - 1) for i in range(10))))
    for c in conds:
        A = np.array(by[c])
        m = A.mean(axis=0)
        print('%-16s %s' % (c, '  '.join('%8.1f' % x for x in m)))

    print('\n判据：')
    print('  最后一列（步 180-199）≈ 0  -> 纯衰减，没有自我维持')
    print('  最后一列 > 第一列的 30%  -> 存在自我维持')


if __name__ == '__main__':
    from multiprocessing import Pool
    main()
