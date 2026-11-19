# grow_lif.py — 阶段 0.5 步 3：转 SNN（docs/experiments.md §11.6.3）
#
# 文档要求（逐字）：
#   「转 SNN（rate coding，T = 20 / 50 / 100 三档），同时记录每步量化误差 ε」
#   判据：成功率掉 < 10%
#
# 形态：
#   神经元是 LIF（膜电位积分 + 过阈放电 + 重置 + 不应期）。
#   一个「逻辑步」= T 个节拍。这一步的值 = T 个节拍内的**放电率**，不是瞬时放电。
#
#   为什么必须差分对：放电率在 [0,1]，而 RNN 的值在 [-1,1]。
#   一正一负两个神经元，值 = (r_pos - r_neg) * g。
#   两边都有基线放电（bias 标定到 5%），所以 a=0 时差为 0。
#
#   为什么 g 这次不会把网络搞爆：它乘的是**放电率**（连续、有界 0~1），
#   不是瞬时放电（二值）。（上一版就是乘瞬时值，单节拍被放大到 ±4.6。）
#
#   时间只有两样：仿真节拍 + 膜时间常数（= 1/(1-rho)，由网络自己的谱半径定）。
#   T 是**读数窗口**——数多少个节拍的放电，不改变网络本身的演化。
#
# 两个标定值（有明确目标，不是扫出来的超参）：
#   bias：使 a = 0 时单个神经元的稳态放电率 = 目标（默认 5%）
#   g   ：使 |r_pos - r_neg| * g 与 RNN 的 |h| 同量级
#
# 运行：
#   python experiments/prototypes/grow_lif.py --calib
#   python experiments/prototypes/grow_lif.py 400 8 20,50,100

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.prototypes import grow_rnn as G

N_EVAL = 40
DS = (0, 10, 20, 50, 100, 200)   # 测记忆时的延迟（单位：逻辑步）


# ---------------- 训练 ----------------
def train_rnn(seed, n_ep, lr=3e-3):
    rng = np.random.default_rng(seed)
    model = G.Elman2(rng)
    opt = G.Adam(model.params(), lr=lr)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(model, 'sup', c_st, c_tg, rng, opt, train=True)
    return model


def rnn_h_abs(model, n_ep=8):
    a1, a2 = [], []
    for ep in range(n_ep):
        st, tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        h1 = np.zeros(G.N_H); h2 = np.zeros(G.N_H)
        for k in range(G.T_EP):
            x = np.array([G.norm(st), G.norm(tg)])
            h1 = np.tanh(model.W1 @ x + model.U1 @ h1 + model.b1)
            h2 = np.tanh(model.W2 @ h1 + model.U2 @ h2 + model.b2)
            a1.append(float(np.abs(h1).mean()))
            a2.append(float(np.abs(h2).mean()))
            if k % G.T_ACT == 0 and k > 0:
                st = int(np.clip(
                    st + int(G.ACTIONS[int(np.argmax(model.W3 @ h2 + model.b3))]),
                    0, G.V_MAX))
    return float(np.mean(a1)), float(np.mean(a2))


# ---------------- LIF 单步 ----------------
def lif_tick(I, v, ref, tau_m, v_th, t_ref):
    decay = np.exp(-1.0 / tau_m)
    v = v * decay + I * (1.0 - decay)
    fire = (v >= v_th) & (ref <= 0)
    v = np.where(fire, 0.0, v)
    ref = np.where(fire, t_ref, ref)
    ref = np.maximum(ref - 1, 0)
    return v, fire.astype(np.float64), ref


def calibrate_bias(tau_m, target=0.05, v_th=1.0, t_ref=2, n_iter=40):
    """a = 0 时 I = bias 恒定，二分找 bias 使稳态放电率 = target。

    单个神经元 + 恒定电流，不需要网络——纯局部标定。
    """
    decay = np.exp(-1.0 / tau_m)

    def rate_at(c):
        v = 0.0
        ref = 0
        n = 0
        for _ in range(400):
            v = v * decay + c * (1.0 - decay)
            if v >= v_th and ref <= 0:
                v = 0.0
                ref = t_ref
                n += 1
            ref = max(ref - 1, 0)
        return n / 400.0

    lo, hi = -20.0, 30.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, rate_at(c)


# ---------------- 核心：一个逻辑步 = T 个节拍 ----------------
class LIFNet(object):
    """两层，每层每个值用一正一负两个 LIF 表示。

    step_T(T) 跑 T 个节拍，返回本逻辑步的放电率与脉冲计数。
    状态（膜电位 / 不应期）跨逻辑步连续，不清零。
    """

    def __init__(self, model, tau_m, bias, g, v_th=1.0, t_ref=2):
        self.m = model
        self.tau_m = tau_m
        self.bias = bias
        self.g = g
        self.v_th = v_th
        self.t_ref = t_ref
        n = G.N_H
        self.v1p = np.zeros(n); self.v1n = np.zeros(n)
        self.v2p = np.zeros(n); self.v2n = np.zeros(n)
        self.r1p = np.full(n, t_ref); self.r1n = np.full(n, t_ref)
        self.r2p = np.full(n, t_ref); self.r2n = np.full(n, t_ref)
        self.h1 = np.zeros(n)
        self.h2 = np.zeros(n)

    def step_T(self, x, T, drive=True):
        """跑 T 个节拍。返回 (h1_new, h2_new, 本段计数)。

        T 个节拍内，两层收到的驱动量恒定（= 上一段的放电率 × g）。
        这正是 rate coding：值在一段窗口内是一个常数。
        """
        c1p = np.zeros(G.N_H); c1n = np.zeros(G.N_H)
        c2p = np.zeros(G.N_H); c2n = np.zeros(G.N_H)
        a1 = (self.m.W1 @ x + self.m.U1 @ self.h1 + self.m.b1) if drive else \
             (self.m.U1 @ self.h1 + self.m.b1)
        a2 = self.m.W2 @ self.h1 + self.m.U2 @ self.h2 + self.m.b2
        for _ in range(T):
            self.v1p, s1p, self.r1p = lif_tick(a1 + self.bias, self.v1p, self.r1p,
                                              self.tau_m, self.v_th, self.t_ref)
            self.v1n, s1n, self.r1n = lif_tick(-a1 + self.bias, self.v1n, self.r1n,
                                              self.tau_m, self.v_th, self.t_ref)
            c1p += s1p; c1n += s1n
            self.v2p, s2p, self.r2p = lif_tick(a2 + self.bias, self.v2p, self.r2p,
                                              self.tau_m, self.v_th, self.t_ref)
            self.v2n, s2n, self.r2n = lif_tick(-a2 + self.bias, self.v2n, self.r2n,
                                              self.tau_m, self.v_th, self.t_ref)
            c2p += s2p; c2n += s2n
        r1 = (c1p - c1n) / float(T)
        r2 = (c2p - c2n) / float(T)
        self.h1 = r1 * self.g
        self.h2 = r2 * self.g
        return self.h1, self.h2, r1, r2


# ---------------- 跑任务 ----------------
def run_task(model, tau_m, bias, g, T, st, goal, v_th=1.0, t_ref=2):
    net = LIFNet(model, tau_m, bias, g, v_th, t_ref)
    for k in range(G.T_EP):
        x = np.array([G.norm(st), G.norm(goal)], np.float64)
        h1, h2, _, _ = net.step_T(x, T, drive=True)
        if k % G.T_ACT != 0 or k == 0:
            continue
        z = model.W3 @ h2 + model.b3
        st = int(np.clip(st + int(G.ACTIONS[int(np.argmax(z))]), 0, G.V_MAX))
        if st == goal:
            return 1.0, k
    return 0.0, G.T_EP


# ---------------- 测记忆（文档判据：4 cue 相似度 < 0.5）----------------
def measure_cue_sim(model, tau_m, bias, g, T, cases, delay, t_enc=20,
                    v_th=1.0, t_ref=2):
    """每个 cue 编码 t_enc 个逻辑步 -> 再做 delay 个逻辑步无输入 -> 比四个 cue。

    相似度用「四个 cue 最后这一段放电计数」的两两 cosine。
    > 0.9 = 无区分；< 0.5 = 有区分。
    """
    cnts = []
    for (a, b) in cases:
        net = LIFNet(model, tau_m, bias, g, v_th, t_ref)
        tot = np.zeros(G.N_H)
        for k in range(t_enc):
            x = np.array([G.norm(a), G.norm(b)])
            net.step_T(x, T, drive=True)
        for k in range(delay):
            _, _, r1, r2 = net.step_T(np.zeros(2), T, drive=False)
            if k == delay - 1:
                tot = r2.copy()
        cnts.append(tot)
    sims = []
    for i in range(len(cnts)):
        for j in range(i + 1, len(cnts)):
            a, b = cnts[i], cnts[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(1.0 if (na < 1e-12 or nb < 1e-12)
                        else float(a @ b / (na * nb)))
    return float(np.mean(sims))


# ---------------- 一个种子的全部工作 ----------------
def run_one(arg):
    seed, n_ep, target_rate, T_list = arg
    model = train_rnn(seed, n_ep)
    rho = float(model.spectral_radius()[0])
    tau_m = 1.0 / max(1e-6, 1.0 - rho)
    bias, r_base = calibrate_bias(tau_m, target=target_rate)
    h1_abs, h2_abs = rnn_h_abs(model)

    out = dict(seed=seed, rho=rho, tau_m=tau_m, bias=bias, r_base=r_base)

    for T in T_list:
        # 标定 g：先跑一小段，测实际的 |r1| 典型值
        net = LIFNet(model, tau_m, bias, 1.0)
        rs = []
        for (a, b) in G.FIXED_CASES:
            net2 = LIFNet(model, tau_m, bias, 1.0)
            for k in range(30):
                x = np.array([G.norm(a), G.norm(b)])
                _, _, r1, _ = net2.step_T(x, T)
                if k > 10:
                    rs.append(float(np.abs(r1).mean()))
        r_typ = max(float(np.mean(rs)), 1e-9)
        g = h1_abs / r_typ

        succ = []
        for ep in range(N_EVAL):
            a, b = G.FIXED_CASES[ep % 4]
            s, _ = run_task(model, tau_m, bias, g, T, a, b)
            succ.append(s)

        sim20 = measure_cue_sim(model, tau_m, bias, g, T, G.FIXED_CASES, 20)
        sim100 = measure_cue_sim(model, tau_m, bias, g, T, G.FIXED_CASES, 100)

        out['T%d' % T] = dict(succ=float(np.mean(succ)), g=g, r_typ=r_typ,
                              sim20=sim20, sim100=sim100)

    rng = np.random.default_rng(999)
    base = [G.run_episode(model, 'greedy', G.FIXED_CASES[ep % 4][0],
                          G.FIXED_CASES[ep % 4][1], rng, None, train=False)['succ']
            for ep in range(N_EVAL)]
    out['base'] = float(np.mean(base))
    return out


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
        h1, h2 = rnn_h_abs(model)
        print('rho = %.3f   tau_m = %.1f 节拍   RNN |h| = %.3f / %.3f' % (rho, tau_m, h1, h2))
        for tgt in (0.02, 0.05, 0.10):
            b, r = calibrate_bias(tau_m, target=tgt)
            print('  目标 %2.0f%% -> bias %.2f，单神经元实测 %.1f%%' % (tgt * 100, b, r * 100))
        r = run_one((0, 400, target_rate, [20, 50, 100]))
        print('\n一次完整对比（seed 0）:')
        print('  %-8s %-9s %-9s %-9s %-9s %s'
              % ('T', '成功率', 'g', 'r典型', 'sim@20', 'sim@100'))
        for T in (20, 50, 100):
            d = r['T%d' % T]
            print('  %-8d %-8.1f%% %-9.2f %-9.4f %-9.3f %.3f'
                  % (T, d['succ'] * 100, d['g'], d['r_typ'], d['sim20'], d['sim100']))
        print('  连续版（基准）%.0f%%' % (r['base'] * 100))
        return 0

    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    T_list = [int(x) for x in argv[2].split(',')] if len(argv) > 2 else [20, 50, 100]
    tasks = [(s, n_ep, target_rate, T_list) for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 104)
    print('grow_lif — 阶段 0.5 步 3：转 SNN（rate coding，T 个节拍数放电）')
    print('  LIF 神经元 + 差分对；一个逻辑步 = T 个节拍；值是这一段的放电率')
    print('  tau_m = 1/(1-rho)；基线放电率标定到 %.0f%%' % (target_rate * 100))
    print('  T 档：%s（§11.6.3 指定）' % T_list)
    print('=' * 104)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    base = np.mean([r['base'] for r in res]) * 100
    print('\n  连续版（基准）成功率 %.0f%%' % base)
    print('\n  %-6s %-10s %-11s %-10s %-10s %s'
          % ('T', '成功率', '相对基准', 'sim@20', 'sim@100', 'g'))
    for T in T_list:
        s = np.mean([r['T%d' % T]['succ'] for r in res]) * 100
        s20 = np.mean([r['T%d' % T]['sim20'] for r in res])
        s100 = np.mean([r['T%d' % T]['sim100'] for r in res])
        g = np.mean([r['T%d' % T]['g'] for r in res])
        print('  %-6d %-9.1f%% %-11s %-10.3f %-10.3f %.2f'
              % (T, s, '%+.1f pp' % (s - base), s20, s100, g))

    print('\n  逐种子成功率:')
    print('  %-5s %-7s %s' % ('seed', 'rho', '  '.join('T=%-6d' % T for T in T_list)))
    for r in res:
        print('  %-5d %-7.3f %s'
              % (r['seed'], r['rho'],
                 '  '.join('%-8.1f%%' % (r['T%d' % T]['succ'] * 100) for T in T_list)))

    print('\n' + '=' * 104)
    print('  判据（§11.6.3 步 3）：掉 < 10%%，且 sim < 0.5')
    for T in T_list:
        s = np.mean([r['T%d' % T]['succ'] for r in res]) * 100
        s20 = np.mean([r['T%d' % T]['sim20'] for r in res])
        drop = base - s
        print('  T=%-4d 掉 %5.1f pp  %s ；sim@20 %.3f %s'
              % (T, drop, 'PASS' if drop < 10 else 'FAIL', s20,
                 'PASS' if s20 < 0.5 else 'FAIL'))
    print('=' * 104)
    return 0


if __name__ == '__main__':
    sys.exit(main())
