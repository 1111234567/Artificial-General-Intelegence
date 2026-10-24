# grow_rnn.py — 阶段 0.5 步 1/2（docs/experiments.md §11.6.3）
#
# BP 训一个 2 层 Elman RNN，做 §16.16 的**同一个任务**（同一个环境、同一个反馈）。
#
# 为什么先做这一步（§11.6.1）：§16.16 修正任务参数后的五个读数——
#   none 5.9% / random 20.6% / 只学 bias 42.0% / 全开 36.2% / linear 99.8%
# 指向的不是「学习规则不够好」，是**起点太差**。而 design.md §5.1 写的
# 「阶段 1 反向：强制、确定、不商量，定架构」从未被实现（§11.4 阶段 A 从随机结构长）。
#
# 判据（§11.6.3，先写死，不达就不进下一步）：
#   步 1  成功率 >= 90%
#   步 2  谱半径 rho ∈ [0.95, 0.99]
#
# 四个模式：
#   random  随机选动作（下界）
#   greedy  启发式（上界；同时充当 sup 的监督标签）
#   sup     监督：每个动作窗口直接给 greedy 的动作（阶段 1 允许「强制、不商量」）
#   bandit  只知道「这个动作朝目标了吗」（reward ±1）——
#           **与 baseline_linear 的 99.8% 信息量相同**，所以这一档可直接对比
#
# 一个必须先量出来的张力（§11.6.2）：
#   这个任务的最优策略**不需要记忆**（state 与 goal 都在输入里）
#   -> BP 会把循环权压到 0，rho_nat ~ 0，步 2 直接不过
#   所以本脚本额外做「谱归一化 + 微调」（--renorm），直接量出：
#   **把 rho 从 rho_nat 拉到 0.97，要付多少成功率。**
#   这是 §11.6.2「记忆长度 × 量化误差放大 = 常数」的实验版。
#
# 运行：
#   python test/grow_rnn.py 400 8 random,greedy,sup,bandit
#   python test/grow_rnn.py 400 8 sup --renorm
#   python test/grow_rnn.py --gradcheck            # 只做数值梯度校验

import os
import sys
import numpy as np
from multiprocessing import Pool

# ---------------- 任务常数（全部与 baseline_linear.py 逐一对应）----------------
V_MAX = 100
T_EP = 240
T_ACT = 5
ACTIONS = np.array([1, -1, 10, -10], np.int64)
FIXED_CASES = ((20, 60), (40, 70), (60, 20), (70, 30))

N_IN = 2          # [norm(state), norm(target)]
N_H = 32          # 每层隐藏单元
N_OUT = 4         # 4 个动作
RHO0 = 0.90       # 循环矩阵初始谱半径（正交初始化后缩放）


def norm(v):
    return float(v) / (V_MAX * 0.5) - 1.0


def greedy_action(st, goal):
    """不越过目标的最大步（baseline_linear 的 greedy；同时当 sup 的标签）。"""
    want = int(np.sign(goal - st))
    need = abs(goal - st)
    if want > 0:
        return 2 if need >= 10 else 0     # +10 / +1
    if want < 0:
        return 3 if need >= 10 else 1     # -10 / -1
    return 0


def orth_rho(n, rho, rng):
    """正交矩阵（满谱半径）缩放到 rho。用 QR 而不是随机高斯，初始 rho 才是精确的。"""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    return Q * rho


# ---------------- 本体：2 层 Elman RNN ----------------
class Elman2(object):
    """h1 = tanh(W1 x + U1 h1 + b1); h2 = tanh(W2 h1 + U2 h2 + b2); z = W3 h2 + b3

    两层各自有循环矩阵 U1 / U2——**这是步 2 要测的两个谱半径**。
    前向/反向手写（不引 torch：项目只用 numpy，且 torch 是 cpu 版，本任务用不上）。
    """

    def __init__(self, rng, rho0=RHO0):
        self.W1 = rng.standard_normal((N_H, N_IN)) / np.sqrt(N_IN)
        self.U1 = orth_rho(N_H, rho0, rng)
        self.b1 = np.zeros(N_H)
        self.W2 = rng.standard_normal((N_H, N_H)) / np.sqrt(N_H)
        self.U2 = orth_rho(N_H, rho0, rng)
        self.b2 = np.zeros(N_H)
        self.W3 = rng.standard_normal((N_OUT, N_H)) / np.sqrt(N_H)
        self.b3 = np.zeros(N_OUT)

    def keys(self):
        return ('W1', 'U1', 'b1', 'W2', 'U2', 'b2', 'W3', 'b3')

    def params(self):
        return [getattr(self, k) for k in self.keys()]

    def spectral_radius(self):
        """两个循环矩阵的最大谱半径（= §16.25 的 g_eff 的 RNN 版）。"""
        r1 = float(np.abs(np.linalg.eigvals(self.U1)).max())
        r2 = float(np.abs(np.linalg.eigvals(self.U2)).max())
        return max(r1, r2), r1, r2

    # ---- 单步（前向 + 记录）----
    def step(self, X, H1, H2, t):
        """写入 H1[t+1] / H2[t+1] / Z[t]（调用方先分配好缓冲）。返回 z。"""
        a1 = self.W1 @ X[t] + self.U1 @ H1[t] + self.b1
        H1[t + 1] = np.tanh(a1)
        a2 = self.W2 @ H1[t + 1] + self.U2 @ H2[t] + self.b2
        H2[t + 1] = np.tanh(a2)
        z = self.W3 @ H2[t + 1] + self.b3
        return z

    # ---- 反向：全序列 BPTT ----
    def backward(self, X, H1, H2, dZ):
        """dZ: (T, N_OUT)，只在有监督的步非零。返回 {name: grad}。

        索引约定（这里写清，因为最容易错）：
          a1[t] = W1 X[t] + U1 H1[t] + b1,  H1[t+1] = tanh(a1[t])
          a2[t] = W2 H1[t+1] + U2 H2[t] + b2, H2[t+1] = tanh(a2[t])
          Z[t]  = W3 H2[t+1] + b3
        H1[t+1] 同时进 a2[t]（经 W2）与 a1[t+1]（经 U1）。
        """
        T = X.shape[0]
        g = {k: np.zeros_like(v) for k, v in zip(self.keys(), self.params())}
        dH1 = np.zeros(N_H)      # dL/dH1[t+1]（下一轮用）
        dH2 = np.zeros(N_H)      # dL/dH2[t+1]
        for t in range(T - 1, -1, -1):
            dH2 = dH2 + self.W3.T @ dZ[t]
            da2 = dH2 * (1.0 - H2[t + 1] ** 2)
            g['W2'] += np.outer(da2, H1[t + 1])
            g['U2'] += np.outer(da2, H2[t])
            g['b2'] += da2
            dH1 = dH1 + self.W2.T @ da2
            dH2 = self.U2.T @ da2          # 赋值：下一轮的 dL/dH2[t]
            da1 = dH1 * (1.0 - H1[t + 1] ** 2)
            g['W1'] += np.outer(da1, X[t])
            g['U1'] += np.outer(da1, H1[t])
            g['b1'] += da1
            dH1 = self.U1.T @ da1          # 赋值：下一轮的 dL/dH1[t]
            g['W3'] += np.outer(dZ[t], H2[t + 1])
            g['b3'] += dZ[t]
        return g

    def renorm(self, rho_target):
        """把循环矩阵缩放到目标谱半径（不动其它参数）。返回缩放前的 rho。"""
        rho_before, r1, r2 = self.spectral_radius()
        if rho_before > 1e-12:
            s = rho_target / rho_before
            self.U1 *= s
            self.U2 *= s
        return rho_before


# ---------------- 优化器（Adam）----------------
class Adam(object):
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def update(self, model, grads, keys=None):
        if keys is None:
            keys = model.keys()
        self.t += 1
        for i, k in enumerate(model.keys()):
            if k not in keys:
                continue
            p = getattr(model, k)
            gp = grads[k]
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gp
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (gp * gp)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------- 一集 ----------------
def run_episode(model, mode, st, goal, rng, opt, train=True, keys=None, blind_at=0):
    """跑一个 episode（240 步，47 个动作窗口）。返回 dict。

    与 baseline_linear.run_episode **逐条对应**：同一个 clip、同一个动作集、
    同一个成功判据（st == goal）、同一个「朝目标」定义（sign(move) == sign(goal-st)）。

    blind_at > 0：从第 blind_at 步起**目标不再可见**（输入置 0），要求网络靠内部状态
    把目标记住。这是为了回答一个 §11.6.2 未预料到的问题：
    如果任务不需要记忆，rho 就只是个**自由参数**（训练管不着），
    那么「拉 rho 无损」就不能归因于「收缩不收费」，而是「任务不用它」。
    """
    start = st
    min_move = abs(goal - start)
    total_move = 0
    n_act = n_ok = 0
    n_upd = 0

    X = np.zeros((T_EP, N_IN))
    H1 = np.zeros((T_EP + 1, N_H))
    H2 = np.zeros((T_EP + 1, N_H))
    Z = np.zeros((T_EP, N_OUT))
    dZ = np.zeros((T_EP, N_OUT))
    n_sup = 0

    succ = 0.0
    for k in range(T_EP):
        X[k, 0] = norm(st)
        # blind_at > 0 时，从该步起目标不可见（要求靠内部状态记住它）
        X[k, 1] = norm(goal) if (blind_at <= 0 or k < blind_at) else 0.0
        z = model.step(X, H1, H2, k)
        Z[k] = z

        if k % T_ACT != 0 or k == 0:
            continue        # 只在动作窗口执行（与 baseline_linear 一致）

        want = int(np.sign(goal - st))

        if mode == 'random':
            a_idx = int(rng.integers(0, 4))
        elif mode == 'greedy':
            a_idx = greedy_action(st, goal)
        else:
            a_idx = int(np.argmax(z))

        st2 = int(np.clip(st + int(ACTIONS[a_idx]), 0, V_MAX))
        move = st2 - st
        total_move += abs(move)
        ok_dir = (want != 0 and np.sign(move) != 0)
        if ok_dir:
            n_act += 1
            if np.sign(move) == want:
                n_ok += 1

        # ---- 学习信号 ----
        if mode == 'sup':
            # 监督：直接给 greedy 的动作（阶段 1：强制、确定、不商量）
            lab = greedy_action(st, goal)
            dZ[k] = z
            dZ[k, lab] -= 1.0        # softmax + 交叉熵 的梯度
            n_sup += 1
        elif mode == 'bandit':
            # 只知道「朝目标了吗」——与 baseline_linear 的信息量相同
            if ok_dir:
                r = 1.0 if np.sign(move) == want else -1.0
                err = r - np.tanh(z[a_idx])
                dZ[k, a_idx] = -err     # dL/dz，注意取负（上升变下降）
                n_sup += 1

        st = st2
        if st == goal:
            succ = 1.0
            break

    if train and mode in ('sup', 'bandit') and n_sup > 0:
        dZ /= float(n_sup)               # 按监督步数归一
        grads = model.backward(X, H1, H2, dZ)
        opt.update(model, grads, keys=keys)
        n_upd = 1

    return dict(succ=succ,
                prog=total_move / max(min_move, 1),
                basis=n_ok / max(n_act, 1),
                n_sup=n_sup, n_upd=n_upd)


# ---------------- 跑一个 (mode, seed) ----------------
def run_one(arg):
    mode, seed, n_ep, lr, renorm, rho_target, freeze_loops, n_ep2, blind_at, rand_goal = arg
    rng = np.random.default_rng(seed)
    model = Elman2(rng)
    opt = Adam(model.params(), lr=lr)

    succ = np.zeros(n_ep)
    basis = np.zeros(n_ep)
    prog = np.zeros(n_ep)

    for ep in range(n_ep):
        if rand_goal:
            c_st = int(rng.integers(10, 91))
            c_tg = int(rng.integers(10, 91))
        else:
            c_st, c_tg = FIXED_CASES[ep % len(FIXED_CASES)]
        r = run_episode(model, mode, c_st, c_tg, rng, opt, train=True,
                        blind_at=blind_at)
        succ[ep] = r['succ']
        basis[ep] = r['basis']
        prog[ep] = r['prog']

    rho_phase1, r1, r2 = model.spectral_radius()

    # ---- 谱归一化 + 微调（只 W/b，循环权冻结）----
    succ2 = None
    rho_phase2 = None
    if renorm:
        rho_before = model.renorm(rho_target)
        rho_phase2 = model.spectral_radius()[0]
        if freeze_loops:
            keys = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
        else:
            keys = None
        opt2 = Adam(model.params(), lr=lr)
        succ2 = np.zeros(n_ep2)
        for ep in range(n_ep2):
            c_st, c_tg = FIXED_CASES[(n_ep + ep) % len(FIXED_CASES)]
            if rand_goal:
                c_st = int(rng.integers(10, 91))
                c_tg = int(rng.integers(10, 91))
            r = run_episode(model, mode, c_st, c_tg, rng, opt2, train=True, keys=keys,
                            blind_at=blind_at)
            succ2[ep] = r['succ']
        rho_phase2, _, _ = model.spectral_radius()
        del rho_before

    k6 = max(1, n_ep // 6)
    out = dict(mode=mode, seed=seed,
               blind_at=blind_at,
               succ_first=float(succ[:k6].mean()),
               succ_last=float(succ[-k6:].mean()),
               succ_all=float(succ.mean()),
               basis_last=float(basis[-k6:].mean()),
               prog=float(prog.mean()),
               rho=float(rho_phase1), rho1=float(r1), rho2=float(r2),
               succ=None)
    if succ2 is not None:
        k6b = max(1, n_ep2 // 6)
        out['renorm_before'] = float(succ[-k6:].mean())
        out['renorm_after'] = float(succ2[-k6b:].mean())
        out['rho_after'] = float(rho_phase2)
    return out


# ---------------- 数值梯度校验（装置本身必须能报错）----------------
def gradcheck(seed=0, T=4):
    """小尺寸（N_H=4）下比对解析梯度与中心差分。不过就不许跑实验。"""
    global N_H, N_IN, N_OUT
    old = (N_H, N_IN, N_OUT)
    N_H, N_IN, N_OUT = 4, 2, 3
    try:
        rng = np.random.default_rng(seed)
        m = Elman2(rng, rho0=0.8)
        X = rng.standard_normal((T, N_IN))
        H1 = np.zeros((T + 1, N_H))
        H2 = np.zeros((T + 1, N_H))
        for t in range(T):
            m.step(X, H1, H2, t)
        dZ = rng.standard_normal((T, N_OUT))

        def loss_of():
            # ⚠️ 必须从 X **重新前向**：H1/H2 本身依赖参数，固定 H 算出来的数值梯度
            # 对 W1/W2/U1/U2 恒为 0（它们只通过改变状态影响 loss）——那就等于测不到。
            # 解析 backward 已经含了状态路径（dH1/dH2 的递归），两边才是同一个量。
            H1_ = np.zeros((T + 1, N_H))
            H2_ = np.zeros((T + 1, N_H))
            L = 0.0
            for t in range(T):
                z = m.step(X, H1_, H2_, t)
                L += float(z @ dZ[t])
            return L

        # 解析梯度在「给定 H1/H2」的意义上是 dL/dParam，其中 L = Σ z_t·dZ_t
        g = m.backward(X, H1, H2, dZ)
        eps = 1e-6
        ok = True
        for k in m.keys():
            p = getattr(m, k)
            num = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'])
            while not it.finished:
                i = it.multi_index
                orig = p[i]
                p[i] = orig + eps
                lp = loss_of()
                p[i] = orig - eps
                lm = loss_of()
                p[i] = orig
                num[i] = (lp - lm) / (2 * eps)
                it.iternext()
            den = max(1e-9, float(np.abs(num).max()))
            err = float(np.abs(num - g[k]).max()) / den
            flag = 'OK  ' if err < 1e-5 else '**FAIL**'
            if err >= 1e-5:
                ok = False
            print('  %-4s rel_err = %.3e   %s' % (k, err, flag))
        return ok
    finally:
        N_H, N_IN, N_OUT = old


# ---------------- main ----------------
def main():
    if '--gradcheck' in sys.argv:
        print('=' * 92)
        print('gradcheck — 解析梯度 vs 中心差分（N_H=4, T=4）')
        print('=' * 92)
        ok = gradcheck()
        print('\n%s' % ('PASS —— 反向正确，可以跑实验' if ok
                        else 'FAIL —— 反向有错，先修，不许跑实验'))
        return 0 if ok else 1

    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 400
    nseed = int(argv[1]) if len(argv) > 1 else 8
    modes = argv[2].split(',') if len(argv) > 2 else ['random', 'greedy', 'sup', 'bandit']
    lr = float(argv[3]) if len(argv) > 3 else 3e-3
    renorm = '--renorm' in sys.argv
    freeze_loops = '--freeze-loops' in sys.argv
    rho_target = 0.97
    for a in sys.argv[1:]:
        if a.startswith('--rho='):
            rho_target = float(a.split('=', 1)[1])
    n_ep2 = max(100, n_ep // 4)
    blind_at = 0
    rand_goal = '--random-goal' in sys.argv
    for a in sys.argv[1:]:
        if a.startswith('--blind='):
            blind_at = int(a.split('=', 1)[1])

    tasks = [(m, s, n_ep, lr, renorm, rho_target, freeze_loops, n_ep2, blind_at, rand_goal)
             for m in modes for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 118)
    print('grow_rnn — 阶段 0.5 步 1/2（§11.6.3）')
    print('  任务 = 4 个固定（初态,目标）对 %s；动作 %s；%d 步 / %d 个动作窗口'
          % (list(FIXED_CASES), list(ACTIONS), T_EP, T_EP // T_ACT - 1))
    print('  2 层 Elman RNN，每层 %d 维；初始谱半径 %.2f；%d episode × %d 种子；lr=%g'
          % (N_H, RHO0, n_ep, nseed, lr))
    if blind_at > 0:
        print('  变体：第 %d 步起目标不可见（输入置 0）——要求靠内部状态记住它'
              % blind_at)
    if renorm:
        print('  谱归一化：rho -> %.2f，再训 %d episode（循环权%s）'
              % (rho_target, n_ep2, '冻结' if freeze_loops else '可调'))
    print('  判据（§11.6.3 步 1/2）：成功率 >= 90%%；rho ∈ [0.95, 0.99]')
    print('=' * 118)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    by = {}
    for r in res:
        by.setdefault(r['mode'], []).append(r)

    for m in modes:
        rs = by.get(m, [])
        if not rs:
            continue
        print('\n--- mode = %s ---' % m)
        extra = '  rho_after' if renorm else ''
        print('  %-5s %-9s %-9s %-9s %-9s %-8s %s' % (
            'seed', 'succ早', 'succ晚', 'basis晚', 'progress', 'rho',
            ('renorm后' + extra) if renorm else ''))
        for r in rs:
            line = '  %-5d %-8.1f%% %-8.1f%% %-9.3f %-9.2f %-8.3f' % (
                r['seed'], r['succ_first'] * 100, r['succ_last'] * 100,
                r['basis_last'], r['prog'], r['rho'])
            if renorm:
                line += ' %.1f%% -> %.1f%%  rho=%.3f' % (
                    r.get('renorm_before', 0) * 100,
                    r.get('renorm_after', 0) * 100,
                    r.get('rho_after', 0))
            print(line)
        S1 = np.mean([r['succ_first'] for r in rs]) * 100
        S2 = np.mean([r['succ_last'] for r in rs]) * 100
        SA = np.mean([r['succ_all'] for r in rs]) * 100
        R = np.mean([r['rho'] for r in rs])
        print('  跨种子: succ %.1f%% -> %.1f%% (%+.1f pp)   全程 %.1f%%   rho %.3f ± %.3f'
              % (S1, S2, S2 - S1, SA, R, np.std([r['rho'] for r in rs])))
        if renorm:
            A = np.mean([r['renorm_after'] for r in rs]) * 100
            RA = np.mean([r['rho_after'] for r in rs])
            print('          renorm: %.1f%% -> %.1f%% (%+.1f pp)   rho %.3f'
                  % (S2, A, A - S2, RA))

    # ---- 判据判定 ----
    print('\n' + '=' * 118)
    for m in ('sup', 'bandit'):
        rs = by.get(m, [])
        if not rs:
            continue
        s2 = np.mean([r['succ_last'] for r in rs]) * 100
        rho = np.mean([r['rho'] for r in rs])
        ok1 = s2 >= 90.0
        ok2 = 0.95 <= rho <= 0.99
        print('  %-7s 步1 成功率 %.1f%% (%s >= 90%%)   步2 rho %.3f (%s [0.95,0.99])'
              % (m, s2, 'PASS' if ok1 else 'FAIL', rho, 'PASS' if ok2 else 'FAIL'))
    print('=' * 118)
    return 0


if __name__ == '__main__':
    sys.exit(main())
