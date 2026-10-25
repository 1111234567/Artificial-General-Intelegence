# grow_lif.py — 阶段 0.5 步 3/4 重做：真的会过阈放电的神经元（差分对）
#
# 为什么重做两次：
#   第一版用「窗口里放几次电」表示值，那个窗口长度是我手选的，系统里没东西定它。
#   第二版去掉窗口，但单个神经元的膜电位被阈值截断在 [.., 1]，表示不了负值——
#     实测只能做对一半任务（4 个 case 做对 2 个），因为 h 的负半部分丢了。
#
# 这一版：一正一负两个神经元一组，表示一个可正可负的值。
#   a 为正 -> 正部放电多、负部少；a 为负 -> 反过来。
#   表示值 h ≈ (r_pos - r_neg) * g
#   代价：神经元数翻倍（SNN 表示带符号值的标准代价）。
#
# 时间只有两样：
#   1. 仿真节拍（1 节拍 = 1 个任务步，不引入新参数）
#   2. 膜时间常数 tau_m = 1/(1-rho)，由网络自己的谱半径定
#
# 两个标定值（有明确目标，不是我扫的超参）：
#   bias：使 a=0 时的基线放电率 = 目标（默认 5%）
#   g   ：使 |r_pos - r_neg| * g 与 RNN 的 |h| 同量级
#
# 运行：
#   python test/grow_lif.py --calib
#   python test/grow_lif.py 400 8

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import grow_rnn as G

N_EVAL = 40
DS = (0, 10, 20, 50, 100, 200)


# ---------------- 训练 ----------------
def train_rnn(seed, n_ep, lr=3e-3):
    rng = np.random.default_rng(seed)
    model = G.Elman2(rng)
    opt = G.Adam(model.params(), lr=lr)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(model, 'sup', c_st, c_tg, rng, opt, train=True)
    return model


def rnn_stats(model, n_ep=8):
    """RNN 两层的 |h| 均值；同时统计层间驱动量 |W h| 与 |U h| 的量级。"""
    h1_abs, h2_abs = [], []
    for ep in range(n_ep):
        st, tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        h1 = np.zeros(G.N_H)
        h2 = np.zeros(G.N_H)
        for k in range(G.T_EP):
            x = np.array([G.norm(st), G.norm(tg)])
            h1 = np.tanh(model.W1 @ x + model.U1 @ h1 + model.b1)
            h2 = np.tanh(model.W2 @ h1 + model.U2 @ h2 + model.b2)
            h1_abs.append(float(np.abs(h1).mean()))
            h2_abs.append(float(np.abs(h2).mean()))
            if k % G.T_ACT == 0 and k > 0:
                st = int(np.clip(
                    st + int(G.ACTIONS[int(np.argmax(model.W3 @ h2 + model.b3))]),
                    0, G.V_MAX))
    return dict(h1_abs=float(np.mean(h1_abs)), h2_abs=float(np.mean(h2_abs)))


# ---------------- 单个神经元的一步 ----------------
def run_layer(I, v, ref, tau_m, v_th, t_ref):
    decay = np.exp(-1.0 / tau_m)
    v = v * decay + I * (1.0 - decay)
    fire = (v >= v_th) & (ref <= 0)
    v = np.where(fire, 0.0, v)
    ref = np.where(fire, t_ref, ref)
    ref = np.maximum(ref - 1, 0)
    return v, fire.astype(np.float64), ref


def pair_step(a, vp, vn, rp, rn, tau_m, v_th, t_ref, bias):
    """a 是这一层收到的驱动量（可正可负）。返回 (vp,vn,sp,sn,rp,rn)。"""
    vp, sp, rp = run_layer(a + bias, vp, rp, tau_m, v_th, t_ref)
    vn, sn, rn = run_layer(-a + bias, vn, rn, tau_m, v_th, t_ref)
    return vp, vn, sp, sn, rp, rn


# ---------------- 标定 bias：使 a=0 时放电率 = 目标 ----------------
def calibrate_bias(tau_m, target=0.05, v_th=1.0, t_ref=2, n_iter=40):
    """a=0 时 I=bias 恒定，二分找 bias 使稳态放电率 = target。

    单个神经元 + 恒定电流，不需要网络——所以这是纯局部标定。
    """
    decay = np.exp(-1.0 / tau_m)

    def rate_at(c):
        v = 0.0
        ref = 0
        n_fire = 0
        for _ in range(400):
            v = v * decay + c * (1.0 - decay)
            if v >= v_th and ref <= 0:
                v = 0.0
                ref = t_ref
                n_fire += 1
            ref = max(ref - 1, 0)
        return n_fire / 400.0

    lo, hi = -20.0, 30.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, rate_at(c)


# ---------------- 跑任务 ----------------
def lif_run(model, st, goal, tau_m, g, bias, v_th=1.0, t_ref=2):
    """1 个任务步 = 1 个仿真节拍。"""
    n = G.N_H
    vp1 = np.zeros(n); vn1 = np.zeros(n)
    vp2 = np.zeros(n); vn2 = np.zeros(n)
    rp1 = np.full(n, t_ref); rn1 = np.full(n, t_ref)
    rp2 = np.full(n, t_ref); rn2 = np.full(n, t_ref)
    sp1 = np.zeros(n); sn1 = np.zeros(n)
    sp2 = np.zeros(n); sn2 = np.zeros(n)

    for k in range(G.T_EP):
        x = np.array([G.norm(st), G.norm(goal)], np.float64)
        h1_eff = (sp1 - sn1) * g
        h2_eff = (sp2 - sn2) * g

        a1 = model.W1 @ x + model.U1 @ h1_eff + model.b1
        vp1, vn1, sp1, sn1, rp1, rn1 = pair_step(a1, vp1, vn1, rp1, rn1,
                                                tau_m, v_th, t_ref, bias)
        h1_eff = (sp1 - sn1) * g

        a2 = model.W2 @ h1_eff + model.U2 @ h2_eff + model.b2
        vp2, vn2, sp2, sn2, rp2, rn2 = pair_step(a2, vp2, vn2, rp2, rn2,
                                                tau_m, v_th, t_ref, bias)
        h2_eff = (sp2 - sn2) * g

        if k % G.T_ACT != 0 or k == 0:
            continue
        z = model.W3 @ h2_eff + model.b3
        st = int(np.clip(st + int(G.ACTIONS[int(np.argmax(z))]), 0, G.V_MAX))
        if st == goal:
            return 1.0, k
    return 0.0, G.T_EP


# ---------------- 测记忆 ----------------
def measure_memory(model, cases, tau_m, g, bias, v_th=1.0, t_ref=2,
                   t_enc=20, ds=DS):
    """撒输入后，四个 cue 的表示值 h2_eff 两两距离随延迟的变化。

    用 h_eff（= 放电率差 * g）而不是膜电位：膜电位会被阈值截断，
    而 h_eff 就是网络内部实际在传的那个量。
    """
    out = []
    for dly in ds:
        hs = []
        for (a, b) in cases:
            n = G.N_H
            vp1 = np.zeros(n); vn1 = np.zeros(n)
            vp2 = np.zeros(n); vn2 = np.zeros(n)
            rp1 = np.full(n, t_ref); rn1 = np.full(n, t_ref)
            rp2 = np.full(n, t_ref); rn2 = np.full(n, t_ref)
            sp1 = np.zeros(n); sn1 = np.zeros(n)
            sp2 = np.zeros(n); sn2 = np.zeros(n)
            for k in range(t_enc):
                x = np.array([G.norm(a), G.norm(b)])
                h1_eff = (sp1 - sn1) * g
                h2_eff = (sp2 - sn2) * g
                a1 = model.W1 @ x + model.U1 @ h1_eff + model.b1
                vp1, vn1, sp1, sn1, rp1, rn1 = pair_step(a1, vp1, vn1, rp1, rn1,
                                                        tau_m, v_th, t_ref, bias)
                h1_eff = (sp1 - sn1) * g
                a2 = model.W2 @ h1_eff + model.U2 @ h2_eff + model.b2
                vp2, vn2, sp2, sn2, rp2, rn2 = pair_step(a2, vp2, vn2, rp2, rn2,
                                                        tau_m, v_th, t_ref, bias)
            for _ in range(dly):
                h1_eff = (sp1 - sn1) * g
                h2_eff = (sp2 - sn2) * g
                a1 = model.U1 @ h1_eff + model.b1
                vp1, vn1, sp1, sn1, rp1, rn1 = pair_step(a1, vp1, vn1, rp1, rn1,
                                                        tau_m, v_th, t_ref, bias)
                h1_eff = (sp1 - sn1) * g
                a2 = model.W2 @ h1_eff + model.U2 @ h2_eff + model.b2
                vp2, vn2, sp2, sn2, rp2, rn2 = pair_step(a2, vp2, vn2, rp2, rn2,
                                                        tau_m, v_th, t_ref, bias)
            hs.append(((sp2 - sn2) * g).copy())
        dd = [float(np.linalg.norm(hs[i] - hs[j]))
              for i in range(len(hs)) for j in range(i + 1, len(hs))]
        out.append(float(np.mean(dd)))
    return out


def tau_of(ds, dists):
    if dists[0] < 1e-12:
        return 0.0
    thr = 0.5 * dists[0]
    for k in range(len(ds)):
        if dists[k] < thr:
            if k == 0:
                return 0.0
            x0, x1 = ds[k - 1], ds[k]
            y0, y1 = dists[k - 1], dists[k]
            if abs(y1 - y0) < 1e-12:
                return float(x1)
            return float(x0 + (y0 - thr) * (x1 - x0) / (y0 - y1))
    return float(ds[-1])


# ---------------- 一个种子的全部工作 ----------------
def run_one(arg):
    seed, n_ep, target_rate = arg
    model = train_rnn(seed, n_ep)
    rho = float(model.spectral_radius()[0])
    tau_m = 1.0 / max(1e-6, 1.0 - rho)

    bias, r_base = calibrate_bias(tau_m, target=target_rate)
    st = rnn_stats(model)

    # 标定 g：跑一小段，测实际的 |r_pos - r_neg| 均值，对齐到 RNN 的 |h|
    n = G.N_H
    vp = np.zeros(n); vn = np.zeros(n)
    rp = np.full(n, 2); rn = np.full(n, 2)
    sp = np.zeros(n); sn = np.zeros(n)
    diffs = []
    for (a, b) in G.FIXED_CASES:
        for k in range(60):
            x = np.array([G.norm(a), G.norm(b)])
            a1 = model.W1 @ x + model.U1 @ ((sp - sn) * 1.0) + model.b1
            vp, vn, sp, sn, rp, rn = pair_step(a1, vp, vn, rp, rn,
                                              tau_m, 1.0, 2, bias)
            if k > 20:
                diffs.append(float(np.abs(sp - sn).mean()))
    d_avg = max(float(np.mean(diffs)), 1e-9)
    g = st['h1_abs'] / d_avg

    succ = []
    for ep in range(N_EVAL):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        sc, _ = lif_run(model, c_st, c_tg, tau_m, g, bias)
        succ.append(sc)

    dists = measure_memory(model, G.FIXED_CASES, tau_m, g, bias)

    rng = np.random.default_rng(999)
    base = [G.run_episode(model, 'greedy', G.FIXED_CASES[ep % 4][0],
                          G.FIXED_CASES[ep % 4][1], rng, None, train=False)['succ']
            for ep in range(N_EVAL)]

    return dict(seed=seed, rho=rho, tau_m=tau_m, bias=bias, r_base=r_base,
                g=g, d_avg=d_avg, succ=float(np.mean(succ)),
                base=float(np.mean(base)), dists=dists,
                tau_mem=tau_of(DS, dists))


# ---------------- 入口 ----------------
def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    target_rate = 0.05
    for a in sys.argv[1:]:
        if a.startswith('--rate='):
            target_rate = float(a.split('=', 1)[1])

    if '--calib' in sys.argv:
        model = train_rnn(0, 400)
        rho = float(model.spectral_radius()[0])
        tau_m = 1.0 / (1 - rho)
        st = rnn_stats(model)
        print('rho = %.3f   tau_m = %.1f 节拍' % (rho, tau_m))
        print('RNN 第一层 |h| 均值 %.3f；第二层 %.3f' % (st['h1_abs'], st['h2_abs']))
        for tgt in (0.02, 0.05, 0.10, 0.20):
            b, r = calibrate_bias(tau_m, target=tgt)
            print('  目标 %2.0f%% -> 偏置 %.2f，单神经元实测 %.1f%%'
                  % (tgt * 100, b, r * 100))
        b, _ = calibrate_bias(tau_m, target=target_rate)
        print('\n用 %.0f%% 基线跑一次完整任务:' % (target_rate * 100))
        res = run_one((0, 400, target_rate))
        print('  成功率 %.0f%%（连续版 %.0f%%）  g = %.1f  基线放电率 %.1f%%'
              % (res['succ'] * 100, res['base'] * 100, res['g'], res['r_base'] * 100))
        print('  撒输入后 h2 的两两距离:')
        for d, v in zip(DS, res['dists']):
            print('    延迟 %-4d 节拍   距离 %.5f' % (d, v))
        print('  距离降到一半：%.0f 节拍' % res['tau_mem'])

        # 对照：同一个 cue，只把初始膜电位扰动 1e-3，看距离怎么长。
        # 如果它也从 1e-3 涨到十几，那前面那些「不衰减」就可能是混沌放大，不是记忆。
        print('\n对照：同一 cue + 初始扰动 1e-3（分开「记忆」与「混沌放大」）:')
        model2 = model
        tau_m2, g2, b2 = tau_m, res['g'], res['bias']
        traj = []
        st0, tg0 = G.FIXED_CASES[0]
        for pert in (0.0, 1e-3):
            n = G.N_H
            vp1 = np.full(n, pert); vn1 = np.zeros(n)
            vp2 = np.full(n, pert); vn2 = np.zeros(n)
            rp1 = np.full(n, 2); rn1 = np.full(n, 2)
            rp2 = np.full(n, 2); rn2 = np.full(n, 2)
            sp1 = np.zeros(n); sn1 = np.zeros(n)
            sp2 = np.zeros(n); sn2 = np.zeros(n)
            seq = []
            for k in range(220):
                x = np.array([G.norm(st0), G.norm(tg0)])
                h1_eff = (sp1 - sn1) * g2
                h2_eff = (sp2 - sn2) * g2
                a1 = (model2.W1 @ x if k < 20 else 0.0) + model2.U1 @ h1_eff + model2.b1
                vp1, vn1, sp1, sn1, rp1, rn1 = pair_step(a1, vp1, vn1, rp1, rn1,
                                                        tau_m2, 1.0, 2, b2)
                h1_eff = (sp1 - sn1) * g2
                a2 = model2.W2 @ h1_eff + model2.U2 @ h2_eff + model2.b2
                vp2, vn2, sp2, sn2, rp2, rn2 = pair_step(a2, vp2, vn2, rp2, rn2,
                                                        tau_m2, 1.0, 2, b2)
                seq.append(((sp2 - sn2) * g2).copy())
            traj.append(seq)
        d = [float(np.linalg.norm(traj[0][k] - traj[1][k])) for k in range(220)]
        for k in (0, 10, 20, 50, 100, 200):
            print('    第 %-4d 节拍   两轨迹距离 %.6f' % (k, d[k]))
        print('    (扰动 1e-3 如果长到几十倍，说明网络是混沌的；那些「距离不衰减」就不可信)')
        return 0

    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    tasks = [(s, n_ep, target_rate) for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 108)
    print('grow_lif — 真的会过阈放电的神经元（差分对，重做步 3/4）')
    print('  时间：1 个任务步 = 1 个仿真节拍；膜时间常数 tau_m = 1/(1-rho)')
    print('  表示：一正一负两个神经元一组，h ≈ (r_pos - r_neg) * g')
    print('  基线放电率标定到 %.0f%%' % (target_rate * 100))
    print('=' * 108)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    print('\n  %-5s %-7s %-8s %-8s %-8s %-9s %s'
          % ('seed', 'rho', 'tau_m', '基线率', '成功率', '基准', '记忆(节拍)'))
    for r in res:
        print('  %-5d %-7.3f %-8.1f %-7.1f%% %-7.1f%% %-8.0f%% %.0f'
              % (r['seed'], r['rho'], r['tau_m'], r['r_base'] * 100,
                 r['succ'] * 100, r['base'] * 100, r['tau_mem']))
    print('\n  跨种子：脉冲版成功率 %.1f%%   连续版 %.0f%%   记忆 %.0f 节拍'
          % (np.mean([r['succ'] for r in res]) * 100,
             np.mean([r['base'] for r in res]) * 100,
             np.mean([r['tau_mem'] for r in res])))
    print('  标定：偏置 %.2f   缩放 g %.1f   基线放电率 %.1f%%'
          % (np.mean([r['bias'] for r in res]), np.mean([r['g'] for r in res]),
             np.mean([r['r_base'] for r in res]) * 100))
    print('=' * 108)
    return 0


if __name__ == '__main__':
    sys.exit(main())
