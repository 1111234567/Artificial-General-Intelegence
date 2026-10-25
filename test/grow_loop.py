# grow_loop.py — 阶段 0.5 步 4（docs/experiments.md §11.6.3）：population + 闭环
#
# 步 3 的产物是一个「能承载策略」的率编码网络，但它的记忆跨度只有 1/(1-rho) 步
# （rho=0.941 -> 约 17 步），远不够 100 步。步 4 要回答两件事：
#   ① population（N 份独立副本）带来了什么？
#   ② 补上 h2 -> h1 的反馈边（「首尾相连」，形成 h1<->h2 回路）能不能把记忆拉长？代价是什么？
#
# 形态（按 §11.6.4 三条实现约束写的）：
#   - 独立存储：N 份 W1/U1/b1/W2/U2/b2 全部 copy，不共享数组
#   - 独立边 id：每份的 V 也是独立数组（后续结构可塑性按成员分别增删）
#   - 对称性破缺：jitter>0 时给 i>0 的副本加初始微扰（否则 N 份走出完全相同的轨迹）
#   - 读出用 mean（不是 L2 归一）：rate coding 的信息在幅度里（§11.6.4 末条）
#
# 本体的「τ」不是膜电位常数，是 rho 的函数：tau_eff ≈ 1/(1-rho)。
# 所以「> 100 步」等价于「有效谱半径 > 0.99」。
#
# 三个测量（全部在同一个训练好的网络上做）：
#   decay   : 编码 20 步 -> 撤全部输入 -> 看 ||p2|| 衰减到 1/e 的步数
#   sim4    : 4 个不同 cue 各自编码 -> 延迟 DLY 步 -> 4 个终态的两两余弦均值
#             （都归零时记 1.0：全零意味着区分力丢失）
#   succ    : 带反馈/population 后原任务还能不能做（策略是否还在）
#
# 运行：
#   python test/grow_loop.py --sanity
#   python test/grow_loop.py 400 8 1,4 0,0.05,0.1,0.15,0.2,0.3

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import grow_rnn as G

T_ENC = 20        # 编码步数
T_OBS = 600       # 衰减观测最长步数
DLY_LIST = (20, 100)   # 测相似度时的延迟档
N_EVAL = 40       # 成功率评估的 episode 数


def make_x(st, goal):
    return np.array([G.norm(st), G.norm(goal)], np.float64)


def rate_quant(v, T, rng, stochastic=False):
    """与 grow_snn_convert.py 同一个定义：p=(v+1)/2 -> 量化 -> 2p-1。

    stochastic=False：rint(p*T)/T。**确定性**——N 份副本输入相近时舍入结果相同，
      没有独立噪声可平均，population 就看不到收益（--noise 实测过）。
    stochastic=True：Binomial(T,p)/T。**随机**——这才是真 SNN 的发放形态，
      N 份独立采样后平均，标准差降 sqrt(N)。
    """
    p = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
    if stochastic:
        c = rng.binomial(T, p).astype(np.float64)
    else:
        c = np.rint(p * T)
    return 2.0 * (c / float(T)) - 1.0


class PopLoop(object):
    """N 份独立副本的 2 层 Elman + 可选 h2 -> h1 反馈。

    每份副本有自己的递归状态（对应 SNN 里各自的膜电位），
    但层间输入用**上一层的 population 均值**（对应「N 个神经元表示同一个值」）。
    """

    def __init__(self, base, N=1, v_scale=0.0, jitter=0.0, rng=None, quant_T=0,
                 stochastic=False):
        self.N = N
        self.quant_T = quant_T
        self.stochastic = stochastic
        self.rng = rng
        self.W1 = [base.W1.copy() for _ in range(N)]
        self.U1 = [base.U1.copy() for _ in range(N)]
        self.b1 = [base.b1.copy() for _ in range(N)]
        self.W2 = [base.W2.copy() for _ in range(N)]
        self.U2 = [base.U2.copy() for _ in range(N)]
        self.b2 = [base.b2.copy() for _ in range(N)]
        self.W3 = base.W3.copy()
        self.b3 = base.b3.copy()
        # 反馈：V = W2^T * v_scale / sqrt(N_H)。对称反馈是「首尾相连」最自然的初始形态。
        scale = v_scale / np.sqrt(G.N_H)
        self.V = [(base.W2.T * scale) for _ in range(N)]
        self.v_scale = v_scale
        if jitter > 0.0 and rng is not None and N > 1:
            for i in range(1, N):
                for arr in (self.W1[i], self.U1[i], self.W2[i], self.U2[i]):
                    arr += rng.standard_normal(arr.shape) * jitter

    def init_state(self):
        return ([np.zeros(G.N_H) for _ in range(self.N)],
                [np.zeros(G.N_H) for _ in range(self.N)])

    def step(self, x, h1, h2):
        """返回 (h1_new, h2_new, p1_new, p2_new, z)。

        反馈用**上一步的 p2**（延迟 1 步），否则就成了代数环。
        quant_T > 0 时，**每个成员各自独立做率编码**（这才是 population coding：
        N 份独立噪声平均后降 sqrt(N)）；如果先平均再量化，噪声相同，N 就白扩了。
        """
        p2_prev = np.mean(h2, axis=0)
        h1n = []
        for i in range(self.N):
            a1 = (self.W1[i] @ x + self.U1[i] @ h1[i]
                  + self.V[i] @ p2_prev + self.b1[i])
            h1n.append(np.tanh(a1))
        p1n = np.mean(h1n, axis=0)
        h2n = []
        for i in range(self.N):
            src = h1n[i]
            if self.quant_T > 0:
                src = rate_quant(src, self.quant_T, self.rng, self.stochastic)
            a2 = self.W2[i] @ src + self.U2[i] @ h2[i] + self.b2[i]
            h2n.append(np.tanh(a2))
        p2n = np.mean(h2n, axis=0)
        z = self.W3 @ p2n + self.b3
        return h1n, h2n, p1n, p2n, z


DS = (0, 2, 5, 10, 20, 50, 100, 200)


def measure_memory(model, cases, ds=DS, t_enc=T_ENC):
    """对每个延迟 d，算 N 个 cue 的 mean pairwise 欧氏距离 / 余弦。

    为什么不用「||p2|| 衰减到 1/e」：诊断（--diag）显示这个网络撤输入后
    **收敛到一个非零不动点**（||p2|| 从 3.77 降到 2.80 就卡住不动），
    永远到不了 1/e——那个测法在线性区才成立，饱和后失效。

    真正要测的是「**区分力**能维持多久」：4 个 cue 的状态两两距离相对 dly=0 的比值。
    比值降到 0.5 的 dly = 区分度半衰期（本装置的 tau）。
    """
    out_d = []
    out_c = []
    for dly in ds:
        states = []
        for (a, b) in cases:
            h1, h2 = model.init_state()
            x = make_x(a, b)
            for _ in range(t_enc):
                h1, h2, p1, p2, z = model.step(x, h1, h2)
            for _ in range(dly):
                h1, h2, p1, p2, z = model.step(np.zeros(2), h1, h2)
            states.append(p2.copy())
        dd = []
        cc = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                a, b = states[i], states[j]
                dd.append(float(np.linalg.norm(a - b)))
                na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                cc.append(float(a @ b / max(na * nb, 1e-12)))
        out_d.append(float(np.mean(dd)))
        out_c.append(float(np.mean(cc)))
    return out_d, out_c


def tau_of(ds, dists):
    """区分度降到 dly=0 的一半所需的延迟步数（线性插值）。"""
    d0 = dists[0]
    if d0 < 1e-12:
        return 0.0
    thr = 0.5 * d0
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


def measure_sim(model, cases, dly, t_enc=T_ENC):
    """4 个 cue 各自编码 -> 延迟 dly 步 -> 两两余弦均值（都归零计 1.0）。"""
    states = []
    for (st, tg) in cases:
        h1, h2 = model.init_state()
        x = make_x(st, tg)
        for _ in range(t_enc):
            h1, h2, p1, p2, z = model.step(x, h1, h2)
        for _ in range(dly):
            h1, h2, p1, p2, z = model.step(np.zeros(2), h1, h2)
        states.append(p2.copy())
    sims = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            a, b = states[i], states[j]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                sims.append(1.0)
            else:
                sims.append(float(a @ b / (na * nb)))
    return float(np.mean(sims))


def run_episode(model, st, goal):
    """用 PopLoop 跑一个 episode（动作 argmax(z)），返回 (succ, 步数)。"""
    h1, h2 = model.init_state()
    for k in range(G.T_EP):
        x = make_x(st, goal)
        h1, h2, p1, p2, z = model.step(x, h1, h2)
        if k % G.T_ACT != 0 or k == 0:
            continue
        a_idx = int(np.argmax(z))
        st = int(np.clip(st + int(G.ACTIONS[a_idx]), 0, G.V_MAX))
        if st == goal:
            return 1.0, k
    return 0.0, G.T_EP


def run_one(arg):
    seed, n_ep, lr, N_list, s_list, jitter = arg
    rng = np.random.default_rng(seed)
    base = G.Elman2(rng)
    opt = G.Adam(base.params(), lr=lr)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(base, 'sup', c_st, c_tg, rng, opt, train=True)
    rho = float(base.spectral_radius()[0])

    rows = []
    for N in N_list:
        for s in s_list:
            m = PopLoop(base, N=N, v_scale=s, jitter=jitter, rng=rng)
            dists, sims = measure_memory(m, G.FIXED_CASES)
            row = dict(seed=seed, rho=rho, N=N, s=s,
                       tau=tau_of(DS, dists),
                       d0=dists[0], dend=dists[-1],
                       sim20=sims[DS.index(20)],
                       sim100=sims[DS.index(100)])
            succ = []
            for ep in range(N_EVAL):
                c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
                sc, _ = run_episode(m, c_st, c_tg)
                succ.append(sc)
            row['succ'] = float(np.mean(succ))
            rows.append(row)
    return rows


def diag(seed=0, n_ep=400):
    """诊断：状态到底在什么区？（饱和则「||p2|| 衰减」测的不是记忆）"""
    rng = np.random.default_rng(seed)
    base = G.Elman2(rng)
    opt = G.Adam(base.params(), lr=3e-3)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(base, 'sup', c_st, c_tg, rng, opt, train=True)
    print('rho = %.3f' % base.spectral_radius()[0])

    m = PopLoop(base, N=1, v_scale=0.0)
    st, tg = G.FIXED_CASES[0]
    h1, h2 = m.init_state()
    x = make_x(st, tg)
    for _ in range(T_ENC):
        h1, h2, p1, p2, z = m.step(x, h1, h2)
    print('\n[编码期末] p1: min=%.3f max=%.3f mean=%.3f std=%.3f  |.|>0.9: %.1f%%'
          % (p1.min(), p1.max(), p1.mean(), p1.std(),
             100.0 * float(np.mean(np.abs(p1) > 0.9))))
    print('              p2: min=%.3f max=%.3f mean=%.3f std=%.3f  |.|>0.9: %.1f%%'
          % (p2.min(), p2.max(), p2.mean(), p2.std(),
             100.0 * float(np.mean(np.abs(p2) > 0.9))))

    print('\n[撤输入后的 ||p2||]')
    for k in range(0, 101):
        h1, h2, p1, p2, z = m.step(np.zeros(2), h1, h2)
        if k % 10 == 0:
            print('  t=%-4d ||p2||=%.4f  |p2|>0.9: %.0f%%'
                  % (k + 1, np.linalg.norm(p2),
                     100.0 * float(np.mean(np.abs(p2) > 0.9))))

    print('\n[4 个 cue 的可分性]')
    for dly in (0, 20, 100):
        states = []
        for (a, b) in G.FIXED_CASES:
            hh1, hh2 = m.init_state()
            xx = make_x(a, b)
            for _ in range(T_ENC):
                hh1, hh2, pp1, pp2, zz = m.step(xx, hh1, hh2)
            for _ in range(dly):
                hh1, hh2, pp1, pp2, zz = m.step(np.zeros(2), hh1, hh2)
            states.append(pp2.copy())
        ds = []
        cs = []
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = states[i], states[j]
                ds.append(float(np.linalg.norm(a - b)))
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                cs.append(float(a @ b / max(na * nb, 1e-12)))
        print('  dly=%-4d 欧氏距离均值=%.4f  余弦均值=%.4f  (逐对距离 %s)'
              % (dly, np.mean(ds), np.mean(cs),
                 ' '.join('%.2f' % v for v in ds)))

    print('\n[N>1 的破缺检查（为什么 population 没贡献）]')
    for jit in (0.0, 1e-3, 1e-2, 5e-2):
        m4 = PopLoop(base, N=4, v_scale=0.0, jitter=jit,
                     rng=np.random.default_rng(7))
        hh1, hh2 = m4.init_state()
        xx = make_x(*G.FIXED_CASES[0])
        for _ in range(T_ENC):
            hh1, hh2, pp1, pp2, zz = m4.step(xx, hh1, hh2)
        ds = [float(np.linalg.norm(hh2[i] - hh2[j]))
              for i in range(4) for j in range(i + 1, 4)]
        print('  jitter=%-8g 成员间 |h2| 平均距离=%.6f   成员范数均值=%.4f'
              % (jit, np.mean(ds),
                 np.mean([float(np.linalg.norm(h)) for h in hh2])))
    return 0


def run_noise_one(arg):
    """population 在率编码噪声下的价值：同样 T 下 N 大的成功率是否更高。

    只在 s=0（无反馈）下测，隔离出「population 降噪」这一个变量。
    jitter 默认 1e-2（1e-3 时成员间差异只有 0.68%，等于没破缺）。
    """
    seed, n_ep, lr, N_list, T_list, jitter, stochastic = arg
    rng = np.random.default_rng(seed)
    base = G.Elman2(rng)
    opt = G.Adam(base.params(), lr=lr)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(base, 'sup', c_st, c_tg, rng, opt, train=True)
    rows = []
    for N in N_list:
        for T in T_list:
            m = PopLoop(base, N=N, v_scale=0.0, jitter=jitter, rng=rng, quant_T=T,
                        stochastic=stochastic)
            succ = []
            for ep in range(N_EVAL):
                c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
                sc, _ = run_episode(m, c_st, c_tg)
                succ.append(sc)
            rows.append(dict(seed=seed, N=N, T=T, succ=float(np.mean(succ))))
    return rows


def main():
    if '--diag' in sys.argv:
        return diag()

    if '--noise' in sys.argv:
        nseed = 8
        N_list = [1, 4, 8]
        T_list = [0, 2, 3, 4, 5, 10]
        jitter = 1e-2
        stochastic = '--stoch' in sys.argv
        tasks = [(s, 400, 3e-3, N_list, T_list, jitter, stochastic) for s in range(nseed)]
        nproc = min(len(tasks), os.cpu_count() or 4)
        print('=' * 104)
        print('grow_loop --noise：population 在率编码噪声下的降噪作用（s=0 隔离变量）')
        print('  N = %s；T = %s（0 = 无噪声）；jitter = %g；%d 种子'
              % (N_list, T_list, jitter, nseed))
        print('  量化：%s   （只作用于 h1→h2 层间传递，且**逐成员独立**）'
              % ('Binomial(T,p) 随机发放' if stochastic else 'rint(p*T) 确定性舍入'))
        print('=' * 104)
        if nproc > 1:
            with Pool(nproc) as pool:
                nested = pool.map(run_noise_one, tasks)
        else:
            nested = [run_noise_one(t) for t in tasks]
        rows = [r for sub in nested for r in sub]
        print('\n  %-8s %s' % ('T', '  '.join('N=%-6d' % N for N in N_list)))
        for T in T_list:
            cells = []
            for N in N_list:
                rs = [r['succ'] for r in rows if r['N'] == N and r['T'] == T]
                cells.append('%-8.1f%%' % (np.mean(rs) * 100) if rs else '   —    ')
            print('  %-8d %s' % (T, '  '.join(cells)))
        print('\n  ' + '=' * 100)
        print('  判据：若 population 真能降噪，同一 T 下 N 越大成功率越高（主要在 T<=5 的临界区看）')
        print('=' * 104)
        return 0

    sanity = '--sanity' in sys.argv

    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    N_list = [int(x) for x in argv[2].split(',')] if len(argv) > 2 else [1, 4]
    s_list = [float(x) for x in argv[3].split(',')] if len(argv) > 3 else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    jitter = 1e-3

    if sanity:
        n_ep = 20
        nseed = 1
        N_list = [1, 4]
        s_list = [0.0, 0.1]

    tasks = [(s, n_ep, 3e-3, N_list, s_list, jitter) for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 116)
    print('grow_loop — 阶段 0.5 步 4（§11.6.3）：population + 闭环（h2 -> h1）')
    print('  源：BP 训好的 2 层 Elman RNN（sup，%d episode × %d 种子）' % (n_ep, nseed))
    print('  副本：N = %s；jitter = %g（对称性破缺）；读出用 mean（非 L2 归一）'
          % (N_list, jitter))
    print('  反馈：V = W2^T * s / sqrt(N_H)，s = %s（延迟 1 步）' % s_list)
    print('  测量：decay=||p2|| 衰减到 1/e 的步数（上限 %d）；sim=4 cue 延迟后两两余弦'
          % T_OBS)
    print('  判据：tau > 100 步；且 sim < 0.5')
    print('=' * 116)

    if nproc > 1:
        with Pool(nproc) as pool:
            nested = pool.map(run_one, tasks)
    else:
        nested = [run_one(t) for t in tasks]
    rows = [r for sub in nested for r in sub]

    rho = np.mean([r['rho'] for r in rows])
    print('\n  rho = %.3f   理论 tau = 1/(1-rho) = %.1f 步'
          % (rho, 1.0 / max(1e-9, 1.0 - rho)))

    for N in N_list:
        print('\n  --- N = %d ---' % N)
        print('  %-7s %-9s %-9s %-9s %-9s %-9s %s'
              % ('s', 'tau', 'd@0', 'd@200', 'sim@20', 'sim@100', 'succ'))
        for s in s_list:
            rs = [r for r in rows if r['N'] == N and abs(r['s'] - s) < 1e-12]
            if not rs:
                continue
            print('  %-7.2f %-9.1f %-9.4f %-9.4f %-9.3f %-9.3f %.1f%%'
                  % (s,
                     np.mean([r['tau'] for r in rs]),
                     np.mean([r['d0'] for r in rs]),
                     np.mean([r['dend'] for r in rs]),
                     np.mean([r['sim20'] for r in rs]),
                     np.mean([r['sim100'] for r in rs]),
                     np.mean([r['succ'] for r in rs]) * 100))

    print('\n' + '=' * 116)
    hits = [r for r in rows if r['tau'] > 100 and r['sim20'] < 0.5]
    print('  同时满足 tau>100 且 sim<0.5 的配置：%d / %d 行' % (len(hits), len(rows)))
    for N in N_list:
        for s in s_list:
            rs = [r for r in rows if r['N'] == N and abs(r['s'] - s) < 1e-12]
            if not rs:
                continue
            tau = np.mean([r['tau'] for r in rs])
            sim = np.mean([r['sim20'] for r in rs])
            ok = 'PASS' if (tau > 100 and sim < 0.5) else 'fail'
            print('    N=%d s=%.2f  tau=%6.1f  sim=%5.3f  -> %s' % (N, s, tau, sim, ok))
    print('=' * 116)
    return 0


if __name__ == '__main__':
    sys.exit(main())
