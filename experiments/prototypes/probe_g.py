# probe_g.py — 查清「混沌到底是 SNN 固有的，还是我放大 g 造成的」
#
# 背景：grow_lif.py 里 h_eff = (sp - sn) * g，g 标定为 4.6。
# 而 RNN 的 h 在 [-1,1]，所以单节拍 h_eff 只能是 {-4.6, 0, +4.6}——
# 比 RNN 值域大 4.6 倍，且每节拍往里灌一次。这很可能是混沌的来源。
#
# 本探针：固定同一个训好的 RNN、同一组权重，只改 g，看
#   ① 成功率
#   ② 同一 cue + 1e-3 初始扰动后的轨迹分离（混沌指标）
#   ③ 层间实际驱动的量级（和 RNN 的 W h 比）

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.prototypes import grow_rnn as G
from experiments.prototypes import grow_lif as L


def drive_magnitude(model, g, bias, tau_m, n_steps=60):
    """测第一层/第二层实际收到的驱动量、与 RNN 同量的对比。"""
    n = G.N_H
    vp1 = np.zeros(n); vn1 = np.zeros(n)
    vp2 = np.zeros(n); vn2 = np.zeros(n)
    rp1 = np.full(n, 2); rn1 = np.full(n, 2)
    rp2 = np.full(n, 2); rn2 = np.full(n, 2)
    sp1 = np.zeros(n); sn1 = np.zeros(n)
    sp2 = np.zeros(n); sn2 = np.zeros(n)
    d1, d2 = [], []
    st, tg = G.FIXED_CASES[0]
    for k in range(n_steps):
        x = np.array([G.norm(st), G.norm(tg)])
        h1_eff = (sp1 - sn1) * g
        h2_eff = (sp2 - sn2) * g
        a1 = model.W1 @ x + model.U1 @ h1_eff + model.b1
        vp1, vn1, sp1, sn1, rp1, rn1 = L.pair_step(a1, vp1, vn1, rp1, rn1,
                                                  tau_m, 1.0, 2, bias)
        h1_eff = (sp1 - sn1) * g
        a2 = model.W2 @ h1_eff + model.U2 @ h2_eff + model.b2
        vp2, vn2, sp2, sn2, rp2, rn2 = L.pair_step(a2, vp2, vn2, rp2, rn2,
                                                  tau_m, 1.0, 2, bias)
        if k > 20:
            d1.append(float(np.abs(model.U1 @ h1_eff).mean()))
            d2.append(float(np.abs(model.W2 @ h1_eff).mean()))
    return float(np.mean(d1)), float(np.mean(d2))


def chaos_test(model, g, bias, tau_m, steps=220):
    """同一 cue，初始膜电位差 1e-3。返回若干时间点的两轨迹距离。"""
    traj = []
    for pert in (0.0, 1e-3):
        n = G.N_H
        vp1 = np.full(n, pert); vn1 = np.zeros(n)
        vp2 = np.full(n, pert); vn2 = np.zeros(n)
        rp1 = np.full(n, 2); rn1 = np.full(n, 2)
        rp2 = np.full(n, 2); rn2 = np.full(n, 2)
        sp1 = np.zeros(n); sn1 = np.zeros(n)
        sp2 = np.zeros(n); sn2 = np.zeros(n)
        st, tg = G.FIXED_CASES[0]
        seq = []
        for k in range(steps):
            x = np.array([G.norm(st), G.norm(tg)])
            h1_eff = (sp1 - sn1) * g
            h2_eff = (sp2 - sn2) * g
            a1 = (model.W1 @ x if k < 20 else 0.0) + model.U1 @ h1_eff + model.b1
            vp1, vn1, sp1, sn1, rp1, rn1 = L.pair_step(a1, vp1, vn1, rp1, rn1,
                                                      tau_m, 1.0, 2, bias)
            h1_eff = (sp1 - sn1) * g
            a2 = model.W2 @ h1_eff + model.U2 @ h2_eff + model.b2
            vp2, vn2, sp2, sn2, rp2, rn2 = L.pair_step(a2, vp2, vn2, rp2, rn2,
                                                      tau_m, 1.0, 2, bias)
            seq.append(((sp2 - sn2) * g).copy())
        traj.append(seq)
    ds = {k: float(np.linalg.norm(traj[0][k] - traj[1][k]))
          for k in (0, 10, 20, 50, 100, 200)}
    return ds


def main():
    model = L.train_rnn(0, 400)
    rho = float(model.spectral_radius()[0])
    tau_m = 1.0 / (1 - rho)
    st = L.rnn_stats(model)
    bias, r_base = L.calibrate_bias(tau_m, target=0.05)

    # RNN 自己的量级（对照）
    h1 = np.zeros(G.N_H); h2 = np.zeros(G.N_H)
    rnn_d1, rnn_d2 = [], []
    for k in range(60):
        x = np.array([G.norm(G.FIXED_CASES[0][0]), G.norm(G.FIXED_CASES[0][1])])
        h1 = np.tanh(model.W1 @ x + model.U1 @ h1 + model.b1)
        h2 = np.tanh(model.W2 @ h1 + model.U2 @ h2 + model.b2)
        if k > 20:
            rnn_d1.append(float(np.abs(model.U1 @ h1).mean()))
            rnn_d2.append(float(np.abs(model.W2 @ h1).mean()))

    print('=' * 96)
    print('rho = %.3f   tau_m = %.1f 节拍   基线放电率 %.1f%%' % (rho, tau_m, r_base * 100))
    print('RNN 自己：|U1 h1| = %.4f   |W2 h1| = %.4f'
          % (np.mean(rnn_d1), np.mean(rnn_d2)))
    print('=' * 96)
    print('  %-7s %-9s %-9s %-11s %-10s %s'
          % ('g', '成功', '|U1 h|', '相对 RNN', '20 节拍扰动', '判定'))
    for g in (0.5, 1.0, 2.0, 3.0, 4.6, 8.0):
        succ = []
        for ep in range(40):
            a, b = G.FIXED_CASES[ep % 4]
            s, _ = L.lif_run(model, a, b, tau_m, g, bias)
            succ.append(s)
        d1, d2 = drive_magnitude(model, g, bias, tau_m)
        ds = chaos_test(model, g, bias, tau_m)
        ratio = d1 / max(np.mean(rnn_d1), 1e-9)
        verdict = '混沌' if ds[20] > 1.0 else ('稳定' if ds[20] < 1e-3 else '中间')
        print('  %-7.1f %-8.0f%% %-9.4f %-11.2f %-10.4f %s'
              % (g, np.mean(succ) * 100, d1, ratio, ds[20], verdict))
    print('=' * 96)
    print('读取：|U1 h| 相对 RNN 接近 1.0 才是量级匹配；远超 1 就是放大过头。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
