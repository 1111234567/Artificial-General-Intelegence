# grow_snn_convert.py — 阶段 0.5 步 3（docs/experiments.md §11.6.3）
#
# 把 grow_rnn 用 BP 训好的 2 层 Elman RNN 转成**率编码 SNN**，量转换损失。
#
# 形态（为什么这么做）：
#   RNN 单元值 h ∈ [-1,1]。SNN 的放电率 p ∈ [0,1]，所以表示取 p = (h+1)/2。
#   一个「逻辑步」= 一个 Elman 步，用 T 个 tick 表示：
#       p_hat = Binomial(T, p) / T        （T 次伯努利采样 = T 个脉冲）
#       h_hat = 2 * p_hat - 1             （重建）
#   权重与偏置**不动**（连续），因为 2016（arXiv 1601.04187）说权重离散化是最大损失源，
#   本步先把「状态离散」这一个变量单独隔离出来。
#
# ε 的定义（写清，否则会测错东西）：
#   **瞬时注入误差** = 每逻辑步、每层、每单元 |quant(h_cont) - h_cont| 的均值。
#   它**不含传播**（传播后果由成功率反映）。
#   所以判据是「ε 与掉点是否同时随 T 变化」，而不是「ε 小就一定不崩」。
#
# 为什么要扫 T 而不是只跑一档：
#   §11.6.2 的核心约束是「记忆长度 × 量化误差放大 = 常数」，放大因子 1/(1-rho)。
#   本任务 rho ~ 0.938 -> 放大 ~16 倍。所以 T 太小时瞬时误差会被放大成灾难。
#   实验的产出应该是「哪一档 T 够用」，而不是「能不能转」（后者必然过）。
#
# 四档 T（0 = 连续，作基准）：
#   T=0   不量化（同一模型、同一评估集）-> 应接近 100%
#   T=20 / 50 / 100
#
# 运行：
#   python experiments/prototypes/grow_snn_convert.py --sanity
#   python experiments/prototypes/grow_snn_convert.py 400 8 0,20,50,100

import os
import sys
import copy
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.prototypes import grow_rnn as G

N_EVAL = 100         # 每个 (seed, T) 评估的 episode 数（4 个 case 各 25 次）——粒度 1 pp


def rate_quant(v, T, rng, deterministic=True):
    """v in [-1,1] -> 率表示 p=(v+1)/2 -> T 个脉冲 -> 重建回 [-1,1]。

    deterministic=True（默认）：c = rint(p*T)。这是「给定恒定电流，T 个 tick 内放电数确定」
      的率编码理想形态；误差上界 |h_hat-h| <= 1/T，平均 ~0.5/T（可算、可校验）。
    deterministic=False：c ~ Binomial(T,p)。模拟泊松脉冲的随机性；E|p_hat-p| ~ sqrt(2p(1-p)/(pi*T))，
      比 round 大得多（T=20 时 0.089 vs 0.025）——**这是真 SNN 有而理想率编码没有的噪声**，
      单列为对照，不混在主线里。
    """
    p = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
    if deterministic:
        c = np.rint(p * T)
    else:
        c = rng.binomial(T, p).astype(np.float64)
    return 2.0 * (c / float(T)) - 1.0


def quantize_weights(model, bits):
    """逐张量对称量化：W -> rint(W/s)*s，s = max|W| / (2^(bits-1)-1)。

    2016（arXiv 1601.04187）说**权重离散化是最大损失源**。本步先把状态离散单独隔离，
    再用这个函数把「权重离散」作为一个独立变量叠上去。
    """
    q = 2 ** (bits - 1) - 1
    for k in model.keys():
        W = getattr(model, k)
        amax = float(np.abs(W).max())
        if amax < 1e-12:
            continue
        s = amax / q
        setattr(model, k, np.rint(W / s) * s)


def run_episode_snn(model, st, goal, rng, T, quant_in=False, quant_state=True,
                    deterministic=True):
    """与 G.run_episode 逐条对应，只把状态（可选输入）换成率编码的量化值。

    动作一律 argmax(z)（模型自己的策略，无教师）。
    返回 succ / basis / prog / eps（瞬时注入误差均值）。
    """
    start = st
    min_move = abs(goal - start)
    total_move = 0
    n_act = n_ok = 0
    eps_sum = 0.0
    eps_n = 0
    marg_sum = 0.0
    marg_n = 0

    h1 = np.zeros(G.N_H)
    h2 = np.zeros(G.N_H)
    succ = 0.0

    do_q = (T > 0 and quant_state)

    for k in range(G.T_EP):
        x = np.array([G.norm(st), G.norm(goal)], np.float64)
        if T > 0 and quant_in:
            xq = rate_quant(x, T, rng, deterministic)
            eps_sum += float(np.abs(xq - x).sum())
            eps_n += x.size
            x = xq

        # ---- 第 1 层 ----
        a1 = model.W1 @ x + model.U1 @ h1 + model.b1
        h1c = np.tanh(a1)
        if do_q:
            h1q = rate_quant(h1c, T, rng, deterministic)
            eps_sum += float(np.abs(h1q - h1c).sum())
            eps_n += G.N_H
        else:
            h1q = h1c

        # ---- 第 2 层 ----
        a2 = model.W2 @ h1q + model.U2 @ h2 + model.b2
        h2c = np.tanh(a2)
        if do_q:
            h2q = rate_quant(h2c, T, rng, deterministic)
            eps_sum += float(np.abs(h2q - h2c).sum())
            eps_n += G.N_H
        else:
            h2q = h2c

        z = model.W3 @ h2q + model.b3
        h1, h2 = h1q, h2q

        if k % G.T_ACT != 0 or k == 0:
            continue

        want = int(np.sign(goal - st))
        a_idx = int(np.argmax(z))
        zs = np.sort(z)[::-1]
        marg_sum += float(zs[0] - zs[1])
        marg_n += 1
        st2 = int(np.clip(st + int(G.ACTIONS[a_idx]), 0, G.V_MAX))
        move = st2 - st
        total_move += abs(move)
        ok_dir = (want != 0 and np.sign(move) != 0)
        if ok_dir:
            n_act += 1
            if np.sign(move) == want:
                n_ok += 1
        st = st2
        if st == goal:
            succ = 1.0
            break

    return dict(succ=succ,
                prog=total_move / max(min_move, 1),
                basis=n_ok / max(n_act, 1),
                margin=marg_sum / max(marg_n, 1),
                eps=eps_sum / max(eps_n, 1))


def train_rnn(seed, n_ep, lr):
    """训一个 sup 模式的 RNN（与 §16.27 步 1 同一配置）。"""
    rng = np.random.default_rng(seed)
    model = G.Elman2(rng)
    opt = G.Adam(model.params(), lr=lr)
    for ep in range(n_ep):
        c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
        G.run_episode(model, 'sup', c_st, c_tg, rng, opt, train=True)
    return model, rng


def run_one(arg):
    seed, n_ep, lr, T_list, quant_in, deterministic, bits_list = arg
    model, rng = train_rnn(seed, n_ep, lr)
    rho, r1, r2 = model.spectral_radius()

    out = dict(seed=seed, rho=float(rho), quant_in=quant_in,
               deterministic=deterministic)
    out['w3'] = float(np.linalg.norm(model.W3, 2))
    out['w3_1'] = float(np.linalg.norm(model.W3, 1))
    for bits in bits_list:
        m = copy.deepcopy(model) if bits > 0 else model
        if bits > 0:
            quantize_weights(m, bits)
        out['rho_b%d' % bits] = float(m.spectral_radius()[0])
        for T in T_list:
            succ = []
            eps = []
            basis = []
            marg = []
            for ep in range(N_EVAL):
                c_st, c_tg = G.FIXED_CASES[ep % len(G.FIXED_CASES)]
                r = run_episode_snn(m, c_st, c_tg, rng, T,
                                    quant_in=quant_in,
                                    quant_state=(T > 0),
                                    deterministic=deterministic)
                succ.append(r['succ'])
                eps.append(r['eps'])
                basis.append(r['basis'])
                marg.append(r['margin'])
            out['succ_b%d_T%d' % (bits, T)] = float(np.mean(succ))
            out['eps_b%d_T%d' % (bits, T)] = float(np.mean(eps))
            out['basis_b%d_T%d' % (bits, T)] = float(np.mean(basis))
            out['marg_b%d_T%d' % (bits, T)] = float(np.mean(marg))
    return out


def main():
    sanity = '--sanity' in sys.argv

    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    T_list = [int(x) for x in argv[2].split(',')] if len(argv) > 2 else [0, 20, 50, 100]
    lr = float(argv[3]) if len(argv) > 3 else 3e-3
    quant_in = '--quant-in' in sys.argv
    deterministic = '--stoch' not in sys.argv

    bits_list = [int(x) for x in argv[4].split(',')] if len(argv) > 4 else [0]
    if sanity:
        n_ep = 20
        nseed = 1
        T_list = [0, 20, 50]
        bits_list = [0, 8]

    tasks = [(s, n_ep, lr, T_list, quant_in, deterministic, bits_list)
             for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 112)
    print('grow_snn_convert — 阶段 0.5 步 3（§11.6.3）：率编码 SNN 的转换损失')
    print('  源：BP 训好的 2 层 Elman RNN（sup，%d episode × %d 种子，lr=%g）' % (n_ep, nseed, lr))
    print('  状态转换：h in [-1,1] -> p=(h+1)/2 -> %s -> h_hat=2p_hat-1'
          % ('rint(p*T)/T（确定性）' if deterministic else 'Binomial(T,p)/T（泊松）'))
    print('  权重：%s' % (' / '.join(('连续' if b == 0 else '%d-bit' % b) for b in bits_list)))
    print('  评估：%d episode/档，训练集内 4 个固定 case（测转换保真度，不是泛化）' % N_EVAL)
    print('  T 档：%s   （0 = 连续基准）%s' % (T_list, '  +输入也量化' if quant_in else ''))
    print('  判据：相对同 bits 的 T=0 掉点 < 10%%；且 eps 实测应与理论 0.5/T 同量级')
    print('=' * 112)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    rho = np.mean([r['rho'] for r in res])
    inv = 1.0 / max(1e-9, 1.0 - rho)
    print('\n  rho(未量化) = %.3f ± %.3f     放大因子 1/(1-rho) = %.1f'
          % (rho, np.std([r['rho'] for r in res]), inv))
    print('  ||W3||_2 = %.3f   预测点：掉点条件 ~ ||W3||*eps*sqrt(N_H) > margin'
          % np.mean([r['w3'] for r in res]))

    for bits in bits_list:
        tag = '连续权重' if bits == 0 else '%d-bit 权重' % bits
        rq = np.mean([r['rho_b%d' % bits] for r in res])
        print('\n  --- %s（rho = %.3f）---' % (tag, rq))
        print('  %-6s %-9s %-11s %-10s %-10s %-12s %s'
              % ('T', 'succ', 'vs T=0', 'eps', 'eps理论', 'eps x 放大', 'margin'))
        base = None
        if 0 in T_list:
            base = np.mean([r['succ_b%d_T0' % bits] for r in res])
        for T in T_list:
            s = np.mean([r['succ_b%d_T%d' % (bits, T)] for r in res]) * 100
            e = np.mean([r['eps_b%d_T%d' % (bits, T)] for r in res])
            mg = np.mean([r['marg_b%d_T%d' % (bits, T)] for r in res])
            th = 0.5 / T if T > 0 else 0.0
            d = '%+.1f pp' % (s - base * 100) if (base is not None and T != 0) else '—'
            print('  %-6d %-8.1f%% %-11s %-10.4f %-10.4f %-12.3f %.3f'
                  % (T, s, d, e, th, e * inv, mg))
        print('    逐种子: %s' % '  '.join(
            's%d=%.0f%%' % (r['seed'], r['succ_b%d_T%d' % (bits, T_list[-1])] * 100)
            for r in res))

    print('\n' + '=' * 112)
    b0 = bits_list[0]
    if 0 in T_list:
        base = np.mean([r['succ_b%d_T0' % b0] for r in res]) * 100
        for T in T_list:
            if T == 0:
                continue
            s = np.mean([r['succ_b%d_T%d' % (b0, T)] for r in res]) * 100
            drop = base - s
            print('  [权重%s] T=%-4d 掉 %.1f pp  ->  %s'
                  % ('连续' if b0 == 0 else '%dbit' % b0, T, drop,
                     'PASS（<10）' if drop < 10.0 else 'FAIL（>=10）'))
    print('=' * 112)
    return 0


if __name__ == '__main__':
    sys.exit(main())
