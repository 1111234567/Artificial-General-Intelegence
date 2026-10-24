# baseline_linear.py - 把网络拿掉，只学一个线性读出
#
# 目的（2026-10-24）：今天 5 轮补丁（first-spike / 不动选项 / 偏置 / 立即反馈 / 沉默信号）
# 每轮都修了一个真 bug，但 `basis`（动作朝目标的比例）始终 0.48~0.50。
# 所以先问一个更便宜的问题：**这个任务本身可学吗？信息在不在？**
#
# 三个对照组（全部不用脉冲网络）：
#   random    随机选动作（下界）
#   greedy    启发式：直接朝目标走（上界，假设“知道”往哪走）
#   linear    线性读出：y = Wx + b，用 delta rule 学（**只用脉冲网络同样的反馈**：
#             这个动作朝目标了吗）。如果这个都学不到，就是任务/编码的问题。
#
# 输入编码与 grow_agent 完全一致：x = [norm(state), norm(target)]，2 维。
# 动作也完全一致：ACTIONS = [1, -1, 10, -10]（+ 不动）。
# 成功判据也一致：到 0 差。
#
# 用固定的 4 个 case（与 grow_agent 的 discrete 相同），保证可比。

import sys
import numpy as np

V_MAX = 100
OBS_LO, OBS_HI = 10, 90
T_EP = 240
ACTIONS = np.array([1, -1, 10, -10], np.int64)
T_ACT = 5          # 与 grow_agent 一致：每 5 步一个动作
FIXED_CASES = ((20, 60), (40, 70), (60, 20), (70, 30))


def norm(v):
    return float(v) / (V_MAX * 0.5) - 1.0


def run_episode(mode, W, b, st, goal, rng, lr):
    """跑一个 episode。返回 (成功, 走过的距离/最短需要, 朝目标比例, 更新次数)。"""
    start = st
    min_move = abs(goal - start)
    total_move = 0
    n_act = 0
    n_ok = 0
    n_upd = 0

    for k in range(T_EP):
        x = np.array([norm(st), norm(goal)])
        want = np.sign(goal - st)

        if k % T_ACT != 0 or k == 0:
            continue          # 只在动作窗口执行（与 grow_agent 完全一致）

        if mode == 'random':
            a_idx = int(rng.integers(0, 4))
        elif mode == 'greedy':
            # 启发式：选朝目标方向的最大步（若距离够）或最小步
            need = abs(goal - st)
            if want > 0:
                a_idx = 2 if need >= 10 else 0     # +10 / +1
            else:
                a_idx = 3 if need >= 10 else 1     # -10 / -1
        else:   # linear
            z = W @ x + b
            a_idx = int(np.argmax(z))

        # 执行
        st2 = int(np.clip(st + int(ACTIONS[a_idx]), 0, V_MAX))
        move = st2 - st
        total_move += abs(move)
        if want != 0 and np.sign(move) != 0:
            n_act += 1
            if np.sign(move) == want:
                n_ok += 1

        # ---- 学习：与脉冲网络同样的反馈（这个动作朝目标了吗）----
        if mode == 'linear' and want != 0 and np.sign(move) != 0:
            tgt = 1.0 if np.sign(move) == want else -1.0
            # delta rule：只更新被选中的那个动作的权重
            z = W @ x + b
            z_sel = z[a_idx]
            err = tgt - np.tanh(z_sel)
            W[a_idx] += lr * err * x
            b[a_idx] += lr * err
            n_upd += 1

        st = st2
        if st == goal:
            return 1.0, total_move / max(min_move, 1), n_ok / max(n_act, 1), n_upd

    return 0.0, total_move / max(min_move, 1), n_ok / max(n_act, 1), n_upd


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    modes = sys.argv[3].split(',') if len(sys.argv) > 3 else ['random', 'greedy', 'linear']
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

    for mode in modes:
        print('=' * 110)
        print('baseline_linear  mode=%-8s  episode=%d  种子=%d  lr=%s' % (mode, n_ep, nseed, lr))
        print('  任务 = 4 个固定（初态, 目标）对 %s；动作 %s' % (list(FIXED_CASES), list(ACTIONS)))
        print('=' * 110)

        seeds = []
        for sd in range(nseed):
            rng = np.random.default_rng(sd)
            W = np.zeros((4, 2))
            b = np.zeros(4)

            succ = np.zeros(n_ep)
            prog = np.zeros(n_ep)
            basis = np.zeros(n_ep)
            n_upd = 0

            for ep in range(n_ep):
                c_st, c_tg = FIXED_CASES[ep % len(FIXED_CASES)]
                s, p, bs, nu = run_episode(mode, W, b, c_st, c_tg, rng, lr)
                succ[ep] = s
                prog[ep] = p
                basis[ep] = bs
                n_upd += nu

            k6 = max(1, n_ep // 6)
            seeds.append((succ, prog, basis, n_upd, W.copy(), b.copy()))

        S = np.array([x[0] for x in seeds])
        P = np.array([x[1] for x in seeds])
        B = np.array([x[2] for x in seeds])
        k6 = max(1, n_ep // 6)

        print('  %-6s %-10s %-10s %-10s %-10s %-10s %s' % (
            'seed', 'succ早', 'succ晚', '全程', 'basis晚', 'progress', 'updates'))
        for i, (succ, prog, basis, nu, W, b) in enumerate(seeds):
            print('  %-6d %-10.1f%% %-10.1f%% %-10.1f%% %-10.3f %-10.2f %d' % (
                i, succ[:k6].mean() * 100, succ[-k6:].mean() * 100,
                succ.mean() * 100, basis[-k6:].mean(), prog.mean(), nu))

        print('  跨种子: 成功率 早 %.1f%% -> 晚 %.1f%% (%+.1f pp)   全程 %.1f%%'
              % (S[:, :k6].mean() * 100, S[:, -k6:].mean() * 100,
                 (S[:, -k6:].mean() - S[:, :k6].mean()) * 100, S.mean() * 100))
        print('          basis 晚 %.3f   progress %.2f   （basis 0.5 = 与目标无关，1.0 = 总是朝目标）'
              % (B[:, -k6:].mean(), P.mean()))

        if mode == 'linear':
            W0 = seeds[0][4]
            b0 = seeds[0][5]
            print('          seed0 学到的 W（4 动作 x [state, target]）：')
            for i in range(4):
                print('            act %+3d : W = [%+7.3f, %+7.3f]   b = %+7.3f'
                      % (ACTIONS[i], W0[i, 0], W0[i, 1], b0[i]))
        print()


if __name__ == '__main__':
    main()
