# rnn2snn.py — 阶段 0.5 步 3：把训好的 RNN 转成项目现有的 SNN 本体
#
# 用 core.Network（项目唯一的 LIF 实现，9 个脚本 / 201 项逐位对齐过），
# 不自己另写一套动力学。
#
# 为什么用差分对（每维两个神经元）：
#   现成 SNN 的 t_ref = 20 -> 单个神经元的放电率上限是 5%。
#   而 RNN 的 h 在 [-1,1]，rate coding 要 p=(h+1)/2（h=0 时 50%）。
#   两者不相容。所以每维拆成正/负两个神经元，各自在 5% 附近工作，
#   值 = (r_pos - r_neg)。RNN 的负权重只有一个落法：连到负部。
#
# 节点布局（固定编号，便于核对）：
#   0 .. 1            输入（norm(state), norm(goal)）
#   2 .. 33           L1 正部（32）
#   34 .. 65          L1 负部（32）
#   66 .. 97          L2 正部（32）
#   98 .. 129         L2 负部（32）
#   130 .. 161        抑制神经元（32：L1 16 + L2 16）
#   162 .. 165        输出（4，线性读出用膜电位，不是脉冲）
#   N = 166, N_E = 130 -> e_ratio = 130/166 = 0.783（接近 §11.4.0 的 0.8）
#
# 边映射（关键：让 h_eff = r_pos - r_neg 正确复现矩阵乘法）
#   RNN:  a = M @ h          h = r_pos - r_neg
#   所以 M @ h = M @ r_pos - M @ r_neg
#   对 M[i,j] > 0:  r_pos[j] -> pos[i] (w=M),  r_neg[j] -> neg[i] (w=M)
#   对 M[i,j] < 0:  r_pos[j] -> neg[i] (w=-M), r_neg[j] -> pos[i] (w=-M)
#   -> 兴奋边全正，负号由「接哪个部」承担。
#
# 运行：
#   python experiments/prototypes/rnn2snn.py --check
#   python experiments/prototypes/rnn2snn.py 400 8 20,50,100

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.prototypes import grow_rnn as G
from src.core import Network, sort_edges
from src.config import NetSpec

N_EVAL = 40

# 节点布局
N_IN = G.N_IN                 # 2
H = G.N_H                     # 32
I0 = N_IN                     # L1 正
I1 = I0 + H                   # L1 负
I2 = I1 + H                   # L2 正
I3 = I2 + H                   # L2 负
IINH = I3 + H                 # 抑制区起点
N_INH = 32                    # 16 + 16
IOUT = IINH + N_INH           # 输出起点
N_TOT = IOUT + G.N_OUT        # 166


def train_rnn(seed, n_ep, lr=3e-3):
    rng = np.random.default_rng(seed)
    m = G.Elman2(rng)
    opt = G.Adam(m.params(), lr=lr)
    for ep in range(n_ep):
        a, b = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(m, 'sup', a, b, rng, opt, train=True)
    return m


def map_matrix(M, src_pos, src_neg, dst_pos, dst_neg):
    """把一个可正可负的矩阵 M 映射成四条正权边组（见头部说明）。"""
    pre, post, w = [], [], []
    n_out, n_in = M.shape
    for i in range(n_out):
        for j in range(n_in):
            v = float(M[i, j])
            if abs(v) < 1e-8:
                continue
            if v > 0:
                pre.append(src_pos + j); post.append(dst_pos + i); w.append(v)
                pre.append(src_neg + j); post.append(dst_neg + i); w.append(v)
            else:
                pre.append(src_pos + j); post.append(dst_neg + i); w.append(-v)
                pre.append(src_neg + j); post.append(dst_pos + i); w.append(-v)
    return pre, post, w


def build_edges(m, w_scale=1.0):
    """把 RNN 的 W1/U1/W2/U2/W3 映射成稀疏边。

    w_scale ：LIF 的兴奋边必须 <= w_max_e (=1.0)。RNN 权重最大约 0.6，
              乘以缩放因子后要保证不超，否则会被削顶。
    """
    pre, post, w = [], [], []

    def add(p, q, vv):
        pre.extend(p); post.extend(q); w.extend(vv)

    # 输入 -> L1：不走神经元（输入是外部模拟电流，在 run_episode 里直接注入）。
    # 所以这里不再建输入边（否则会重复计算一次）。

    # L1 -> L2（W2 是 32 x 32）
    p, q, vv = map_matrix(m.W2, I0, I1, I2, I3)
    add(p, q, vv)

    # L1 自递归（U1）
    p, q, vv = map_matrix(m.U1, I0, I1, I0, I1)
    add(p, q, vv)

    # L2 自递归（U2）
    p, q, vv = map_matrix(m.U2, I2, I3, I2, I3)
    add(p, q, vv)

    # 抑制神经元：从同层兴奋区随机取样（4:1）+ 回授抑制
    rng = np.random.default_rng(2026)
    for k in range(N_INH):
        inh = IINH + k
        layer = 0 if k < N_INH // 2 else 1
        src = I0 if layer == 0 else I2
        dst = (I0, I1) if layer == 0 else (I2, I3)
        pick = rng.choice(np.arange(src, src + H), 16, replace=False)
        for s in pick:
            add([int(s)], [inh], [0.5])          # 兴奋 -> 抑制
        for t in range(H):
            add([inh], [dst[0] + t], [0.5 * 5.0])   # 抑制 -> 兴奋（负权重由区段决定）
            add([inh], [dst[1] + t], [0.5 * 5.0])

    pre = np.asarray(pre, np.int32)
    post = np.asarray(post, np.int32)
    w = np.asarray(w, np.float32) * w_scale
    # 抑制区出发的边权重为负（core 的约定：pre >= N_E 则 w < 0）
    w = np.where(pre >= (IOUT if False else IINH + 0) if False else pre >= IINH,
                 -np.abs(w), np.abs(w)).astype(np.float32)
    age = np.full(pre.size, 100, np.int16)
    return pre, post, w, age


def make_net(m, tau_v=20.0, t_ref=20, w_scale=1.0):
    import dataclasses
    spec = dataclasses.replace(
        NetSpec(),
        N=N_TOT,
        n_in=N_IN,
        e_ratio=IINH / float(N_TOT),   # 抑制区（IINH..）之前都是兴奋
        tau_v=tau_v,
        v_th=1.0,
        t_ref=t_ref,
        t_ref_out=t_ref,
        theta_mode="static",            # θ 恒 = v_th（不让适应把工作点拉跑）
        theta_out_adapt=False,
        use_std=False,
        side_inh=0.0,
        e0_init=100,
        block_input_edges=True,
    )

    rng = np.random.default_rng(1)
    out_idx = np.arange(IOUT, IOUT + G.N_OUT)
    net = Network(spec, rng, out_idx)
    pre, post, w, age = build_edges(m, w_scale)
    net.pre, net.post, net.w, net.key, net.age = sort_edges(pre, post, w, age, net.N)
    return net, spec


# ---------------- 运行：一个逻辑步 = T 个 tick ----------------
def run_episode(net, m, T, st, goal, bias_in=None):
    net.v[:] = 0.0
    net.state[:] = 0.0
    net.ref[:] = 0
    net.trace[:] = 0.0
    if net.bias is not None:
        net.bias[:] = 0.0

    xin = np.zeros(net.N, np.float32)
    for k in range(G.T_EP):
        # 输入直接当电流注入 L1 正/负部（不经过输入神经元）——
        # 否则输入神经元的 t_ref=20 会把信号卡成每 20 tick 一个脉冲（实测放电率 0.45%）。
        # 这也正是 rate coding 的常规做法：外部输入是模拟电流。
        x = np.array([G.norm(st), G.norm(goal)])
        drv = m.W1 @ x
        xin[:] = 0.0
        xin[I0:I0 + H] = np.maximum(drv, 0.0)
        xin[I1:I1 + H] = np.maximum(-drv, 0.0)
        for _ in range(T):
            net.step(xin)
        # 读出：L2 两部的膜电位差，乘 W3
        vL2 = net.v[I2:I2 + H] - net.v[I3:I3 + H]
        if k % G.T_ACT != 0 or k == 0:
            continue
        z = m.W3 @ vL2 + m.b3
        st = int(np.clip(st + int(G.ACTIONS[int(np.argmax(z))]), 0, G.V_MAX))
        if st == goal:
            return 1.0, k
    return 0.0, G.T_EP


def run_one(arg):
    seed, n_ep, T_list = arg
    m = train_rnn(seed, n_ep)
    rho = float(m.spectral_radius()[0])
    out = dict(seed=seed, rho=rho)
    for T in T_list:
        net, spec = make_net(m, tau_v=20.0, t_ref=20)
        succ = []
        for ep in range(N_EVAL):
            a, b = G.FIXED_CASES[ep % 4]
            s, _ = run_episode(net, m, T, a, b)
            succ.append(s)
        # 放电率
        rates = []
        net2, _ = make_net(m)
        for ep in range(4):
            a, b = G.FIXED_CASES[ep % 4]
            run_episode(net2, m, T, a, b)
            rates.append(float(net2.ema_rate))
        out['T%d' % T] = dict(succ=float(np.mean(succ)),
                              rate=float(np.mean(rates)),
                              nedge=int(net.pre.size))
    rng = np.random.default_rng(999)
    out['base'] = float(np.mean([
        G.run_episode(m, 'greedy', *G.FIXED_CASES[ep % 4], rng, None,
                      train=False)['succ'] for ep in range(N_EVAL)]))
    return out


def main():
    if '--check' in sys.argv:
        m = train_rnn(0, 400)
        net, spec = make_net(m)
        print('=' * 92)
        print('结构检查')
        print('  N = %d   N_E = %d   抑制 = %d   e_ratio = %.3f'
              % (net.N, net.N_E, net.N - net.N_E, net.N_E / net.N))
        print('  边数 = %d（兴奋 %d / 抑制 %d）'
              % (net.pre.size, int((net.pre < net.N_E).sum()),
                 int((net.pre >= net.N_E).sum())))
        print('  权重范围 %.3f ~ %.3f   t_ref = %d   theta_mode = %s'
              % (net.w.min(), net.w.max(), spec.t_ref, spec.theta_mode))
        print('=' * 92)
        print('  t_ref 扫描（核心矛盾：现成参数 t_ref=20 把放电率上限锁在 5%，')
        print('  而 rate coding 要表示 RNN 的 ±0.5 的值）')
        print('  %-8s %-10s %-10s %-12s %s'
              % ('t_ref', '放电率上限', '实测率', '成功率', '退化为'))
        for t_ref in (2, 5, 10, 20, 40):
            net, spec = make_net(m, t_ref=t_ref)
            s = []
            for ep in range(20):
                a, b = G.FIXED_CASES[ep % 4]
                v, _ = run_episode(net, m, 20, a, b)
                s.append(v)
            print('  %-8d %-9.1f%% %-9.2f%% %-11.0f%% %.1f 项对齐参数'
                  % (t_ref, 100.0 / t_ref, net.ema_rate * 100,
                     np.mean(s) * 100, (0 if t_ref == 20 else 0)))
        print('=' * 92)
        return 0

    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    T_list = [int(x) for x in argv[2].split(',')] if len(argv) > 2 else [20, 50, 100]
    tasks = [(s, n_ep, T_list) for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 100)
    print('rnn2snn — 阶段 0.5 步 3：RNN -> 现有 SNN 本体（core.Network）')
    print('  E/I = %.2f，t_ref = 20，tanh->差分对，矩阵映射保符号' % (IOUT / float(N_TOT)))
    print('  T 档 %s（文档 §11.6.3 指定）' % T_list)
    print('=' * 100)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    base = np.mean([r['base'] for r in res]) * 100
    print('\n  连续版（基准）%.0f%%' % base)
    print('\n  %-6s %-10s %-11s %-10s %s'
          % ('T', '成功率', '相对基准', '放电率', '边数'))
    for T in T_list:
        s = np.mean([r['T%d' % T]['succ'] for r in res]) * 100
        r = np.mean([r['T%d' % T]['rate'] for r in res]) * 100
        ne = res[0]['T%d' % T]['nedge']
        print('  %-6d %-9.1f%% %-11s %-9.2f%% %d'
              % (T, s, '%+.1f pp' % (s - base), r, ne))

    print('\n  逐种子:')
    print('  %-5s %-7s %s' % ('seed', 'rho', '  '.join('T=%-6d' % T for T in T_list)))
    for r in res:
        print('  %-5d %-7.3f %s'
              % (r['seed'], r['rho'],
                 '  '.join('%-8.1f%%' % (r['T%d' % T]['succ'] * 100) for T in T_list)))
    print('\n' + '=' * 100)
    for T in T_list:
        s = np.mean([r['T%d' % T]['succ'] for r in res]) * 100
        print('  T=%-4d 掉 %5.1f pp  %s'
              % (T, base - s, 'PASS' if (base - s) < 10 else 'FAIL'))
    print('=' * 100)
    return 0


if __name__ == '__main__':
    sys.exit(main())
