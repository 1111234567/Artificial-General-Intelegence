# grow_hebb.py - 加一层 Hebbian 权重可塑性，看撤掉输入后活动会不会持续
#
# 依据（见 docs/reference.md §12.5「神经元–突触耦合动力学」）：
#   ① Clark & Abbott, PRX 14, 021001 (2024)
#      “For g<1, the trivial neuronal fixed point of the nonplastic network is
#       globally stable and the network is quiescent.”
#      “...if Hebbian plasticity is strong, there exist stable fixed points ...
#       their number showing exponential growth with network size.”
#   ② Wakhloo, Clark & Abbott, PNAS 123(40) e2526156123 (2026)
#      “following oscillatory stimulation, neurons continue to oscillate long after
#       the input is removed” → 持续振荡（复数 outlier 特征值）
#
# 本文件只加一样东西：一条与边同型的慢变量 A，它把“最近共激活的程度”写进权重。
#   A[e] ← A[e] + (1/τ_s)·(-A[e] + r·W_EE_INIT·ĉ_pre·ĉ_post)
#   w_eff = w_struct + A
# 其中 ĉ = 滤波后的活动率 / 全局平均活动率（均值归一，使 A 的尺度不随放电率漂移）。
#
# 与 grow_homeo.py 的区别：
#   那边：结构可塑性（增删边），**权重恒定** → 按上述分类属 nonplastic network
#   这边：结构**冻结**，只有权重变 —— 隔离变量
#
# 主判据不是“模式相似度”（论文里没这个量），而是：
#   ① 撤掉输入后总放电是否归零（活动能否自持）
#   ② 若能自持，放电时序的主周期是否对应输入周期
#   ③ 对照：r=0（无 A）/ A 冻结 / A 随机重排（保持统计、破坏低秩结构）
#
# 用法：
#   python experiments/legacy/grow_hebb.py                  # 主表：r × 输入形态
#   python experiments/legacy/grow_hebb.py --quick          # 少几组
#   python experiments/legacy/grow_hebb.py --freq           # 只跑振荡输入的频率响应

import os
import sys
import time
import numpy as np
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.legacy import grow_homeo as G

# ---- 可塑性参数（要扫的）----
R_LIST = (0.0, 0.02, 0.05, 0.15, 0.4)   # A 相对于结构权重的稳态强度
TAU_SYN_LIST = (600,)                   # 突触时间常数（步）。神经元 τ = 20 步
TAU_C = 20.0                            # 活动率滤波（与膜电位同尺度）
A_MAX_REL = 2.0                         # |A| 上界 = A_MAX_REL × r × W_EE_INIT

# ---- 输入 ----
ENC_STEPS = 20          # 脉冲型输入：两拍
OSC_STEPS = 400         # 振荡型输入：持续 400 步（= 4 个 100 步周期）
OSC_PERIOD = 100        # 振荡周期（步）
OSC_AMP = 3.0

# ---- 观测 ----
OBS_STEPS = 1200        # 撤掉输入后观测多少步
BIN = 20                # 分格粒度


def _plastic_mask(pre, post):
    """只对 E-E 边做权重可塑性。

    输入扇出不能碰（那是感觉通路）；I→E / E→I 也不能碰
    （Hebbian 强化抑制边会直接破坏 E/I 平衡）。
    """
    return (pre >= G.N_IN) & (pre < G.N_E) & (post < G.N_E)


def _sim(pre, post, w_struct, mask, r, v, s, ref, theta, x_rec, c, A,
         xin, tau_syn, plastic=True):
    """跑一步。返回新的状态 + 放电数。"""
    w_eff = w_struct + A
    if G.USE_STD:
        w_eff = np.where(pre >= G.N_E, w_eff * x_rec[pre], w_eff)
    I = np.bincount(post, weights=w_eff * s[pre], minlength=G.N).astype(np.float32)
    v = v * G.DECAY_V + I + xin
    fire = (v > theta) & (ref <= 0)
    s_new = fire.astype(np.float32)
    v = np.where(s_new > 0, 0.0, v)
    ref -= 1
    np.maximum(ref, 0, out=ref)
    ref[s_new > 0] = G.T_REF
    if fire.any():
        theta[fire] = np.minimum(theta[fire] + G.D_THETA, G.TH_MAX)
    theta += (G.V_TH - theta) / G.TAU_ADAPT
    if G.USE_STD:
        x_rec += (1.0 - x_rec) / G.TAU_REC
        ii = np.nonzero(s_new[G.N_E:] > 0)[0] + G.N_E
        if ii.size:
            x_rec[ii] *= (1.0 - G.U_STD)

    # 活动率滤波
    c += (s_new - c) / TAU_C

    # 权重可塑性（只 E-E，只正向边）
    # ⚠️ cn 只有 N_E 个元素，而 post 里含抑制神经元（800~999）——
    # 必须先按 mask 取子集再索引，否则 IndexError。
    if plastic and r > 0.0:
        m = c[:G.N_E].mean()
        if m > 1e-9:
            cn = c[:G.N_E] / m          # 均值归一 → 尺度不随放电率漂移
            target = np.zeros_like(A)
            im = np.nonzero(mask)[0]
            if im.size:
                target[im] = r * G.W_EE_INIT * cn[pre[im]] * cn[post[im]]
            A += (target - A) / tau_syn
            lim = A_MAX_REL * r * G.W_EE_INIT * 4.0
            np.clip(A, -lim, lim, out=A)
            A[~mask] = 0.0
    return v, s_new, ref, theta, x_rec, c, A, float(s_new[:G.N_E].sum())


def _encode(pre, post, w_struct, mask, r, st, tau_syn, mode, blk, rng, steps):
    """给输入。mode='pulse' 只给两拍；mode='osc' 给持续振荡。

    振荡输入：xin[输入区] = AMP·cos(2πt/T + θ)，θ 逐单元随机相位（同 PNAS）。
    只给前 N_IN 个输入神经元；它们再扇出到下游。
    注意：cos 可能为负 → 切掉负半周（脉冲神经元没有负输入）。
    """
    v, s, ref, theta, x_rec, c, A = (st['v'].copy(), st['s'].copy(), st['ref'].copy(),
                                     st['theta'].copy(), st['x_rec'].copy(),
                                     st['c'].copy(), st['A'].copy())
    B = G.N_IN // G.N_CUE
    phase = rng.random(G.N_IN) * 2 * np.pi
    zero = np.zeros(G.N, np.float32)
    tot = 0.0
    for t in range(steps):
        xin = zero.copy()
        if mode == 'pulse':
            if t == 0 or t == max(1, steps - 2):
                xin[blk * B:(blk + 1) * B] = G.IN_SPIKE
        else:  # osc
            drive = np.cos(2 * np.pi * t / OSC_PERIOD + phase) * OSC_AMP
            drive = np.maximum(drive, 0.0)      # 只要正半周
            xin[:G.N_IN] = drive.astype(np.float32) * (rng.random(G.N_IN) < 0.9)
        v, s, ref, theta, x_rec, c, A, n = _sim(
            pre, post, w_struct, mask, r, v, s, ref, theta, x_rec, c, A,
            xin, tau_syn, plastic=True)
        tot += n
    return dict(v=v, s=s, ref=ref, theta=theta, x_rec=x_rec, c=c, A=A), tot


def _observe(pre, post, w_struct, mask, r, st, tau_syn, steps, plastic):
    """撤掉全部输入（含背景），统计每 BIN 步的兴奋放电。

    plastic=True：A 继续变（在线可塑性）；False：A 冻结（只看已存的东西）。
    """
    v, s, ref, theta, x_rec, c, A = (st['v'].copy(), st['s'].copy(), st['ref'].copy(),
                                     st['theta'].copy(), st['x_rec'].copy(),
                                     st['c'].copy(), st['A'].copy())
    zero = np.zeros(G.N, np.float32)
    bins = []
    acc = 0.0
    for t in range(steps):
        v, s, ref, theta, x_rec, c, A, n = _sim(
            pre, post, w_struct, mask, r, v, s, ref, theta, x_rec, c, A,
            zero, tau_syn, plastic=plastic)
        acc += n
        if (t + 1) % BIN == 0:
            bins.append(acc); acc = 0.0
    return bins, A


def _warm(pre, post, w_struct, mask, r, rng, steps=500, tau_syn=600.0):
    v = np.zeros(G.N, np.float32)
    s = np.zeros(G.N, np.float32)
    ref = np.zeros(G.N, np.int32)
    theta = np.full(G.N, G.V_TH, np.float32)
    x_rec = np.ones(G.N, np.float32)
    c = np.zeros(G.N, np.float32)
    A = np.zeros(pre.size, np.float32)
    zero = np.zeros(G.N, np.float32)
    for _ in range(steps):
        xin = zero.copy()
        xin[G.N_IN:] = (rng.random(G.N - G.N_IN) < G.BG_RATE).astype(np.float32) * G.IN_SPIKE
        v, s, ref, theta, x_rec, c, A, _ = _sim(
            pre, post, w_struct, mask, r, v, s, ref, theta, x_rec, c, A,
            xin, tau_syn, plastic=True)
    return dict(v=v, s=s, ref=ref, theta=theta, x_rec=x_rec, c=c, A=A)


def _main_period(bins, lo=2):
    """从分格序列里估主周期（自相关第一个正峰）。太静默则返回 0。"""
    a = np.asarray(bins, float)
    if a.size < 8 or a.sum() <= 0:
        return 0.0, 0.0
    a = a - a.mean()
    if np.allclose(a, 0):
        return 0.0, 0.0
    ac = np.correlate(a, a, mode='full')[a.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    for k in range(lo, ac.size // 2):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2:
            return float(k * BIN), float(ac[k])
    return 0.0, 0.0


def run_one(args):
    """一个 (r, mode, plastic, shuffle) 组合。

    结构固定：seed0 长 30000 步得到的网络（一次算好当基线，这里再算一遍因为子进程隔离）。
    """
    seed, r, tau_syn, mode, plastic, shuffle, grow_steps = args
    t0 = time.perf_counter()

    g = G.grow((seed, grow_steps, grow_steps // 2))
    pre, post, w, key = G._sort(g['pre'], g['post'], g['w'])
    mask = _plastic_mask(pre, post)

    rng = np.random.default_rng(seed + 31337)
    st = _warm(pre, post, w, mask, r, rng, 500, tau_syn)

    # A 的随机重排对照：保持 A 的数值分布，破坏它与边上神经元的对应关系
    if shuffle:
        st['A'] = st['A'][rng.permutation(st['A'].size)]

    st, enc_tot = _encode(pre, post, w, mask, r, st, tau_syn, mode, 0, rng,
                          ENC_STEPS if mode == 'pulse' else OSC_STEPS)

    bins, A_end = _observe(pre, post, w, mask, r, st, tau_syn, OBS_STEPS, plastic)

    per, ac = _main_period(bins)
    head = bins[:6]
    tail = bins[6:]
    return dict(seed=seed, r=r, tau_syn=tau_syn, mode=mode, plastic=plastic,
                shuffle=shuffle, enc=enc_tot, bins=bins, head=head,
                tail_sum=float(np.sum(tail)), total=float(np.sum(bins)),
                period=per, ac=ac,
                a_abs=float(np.abs(A_end).mean()),
                elapsed=time.perf_counter() - t0)


def main():
    quick = '--quick' in sys.argv
    freq_only = '--freq' in sys.argv
    seed = 0
    grow_steps = 30000

    if freq_only:
        jobs = [(seed, 0.05, 600.0, 'osc', True, False, grow_steps)]
    elif quick:
        jobs = [(seed, r, 600.0, mode, pl, False, grow_steps)
                for mode in ('pulse', 'osc')
                for r in (0.0, 0.05, 0.4)
                for pl in (True,)]
    else:
        jobs = []
        for mode in ('pulse', 'osc'):
            for r in R_LIST:
                jobs.append((seed, r, 600.0, mode, True, False, grow_steps))
        # 对照：A 冻结 / A 重排（只取中档 r）
        for mode in ('pulse', 'osc'):
            jobs.append((seed, 0.05, 600.0, mode, False, False, grow_steps))
            jobs.append((seed, 0.05, 600.0, mode, True, True, grow_steps))
            jobs.append((seed, 0.4, 600.0, mode, True, True, grow_steps))

    print('=' * 116)
    print('Hebbian 权重可塑性：撤掉输入后，活动能不能自持？')
    print('  结构固定（seed %d 长 %d 步，E-E 边数见下表）  τ_c=%.0f  τ_syn=%.0f  A_MAX=%.1f×r' % (
        seed, grow_steps, TAU_C, TAU_SYN_LIST[0], A_MAX_REL))
    print('  %d 个任务一次性提交（并行度 = 任务数）' % len(jobs))
    print('=' * 116)

    with Pool(min(len(jobs), 8)) as pool:
        res = pool.map(run_one, jobs)

    print('\n%4s %6s %6s %6s %5s | %8s %8s %8s %8s | %9s %7s %8s' % (
        'r', 'mode', 'plastic', 'shuf', '编码', '0-20', '20-40', '40-60', '60-120',
        '尾段总', '周期', '自相关'))
    print('-' * 116)
    for x in res:
        print('%4.2f %6s %6s %6s %5.0f | %8.0f %8.0f %8.0f %8.0f | %9.0f %7.0f %8.2f' % (
            x['r'], x['mode'], 'on' if x['plastic'] else '冻结',
            'yes' if x['shuffle'] else '-', x['enc'],
            x['head'][0] if len(x['head']) > 0 else 0,
            x['head'][1] if len(x['head']) > 1 else 0,
            x['head'][2] if len(x['head']) > 2 else 0,
            sum(x['head'][3:6]), x['tail_sum'], x['period'], x['ac']))

    print('\n逐任务详情（尾段每 100 步一格，共 %d 格）：' % max(0, (OBS_STEPS // BIN) - 6))
    for x in res:
        b = x['bins']
        tail = b[6:]
        agg = [sum(tail[i:i + 5]) for i in range(0, len(tail), 5)]
        print('  r=%.2f %s %s%s  A_mean=%.2e  %.0fs' % (
            x['r'], x['mode'], 'on' if x['plastic'] else 'freeze',
            ' shuf' if x['shuffle'] else '', x['a_abs'], x['elapsed']))
        print('    %s' % ' '.join('%.0f' % v for v in agg))

    print('\n判定：')
    alive = [x for x in res if x['tail_sum'] > 0]
    if not alive:
        print('  **全部在 120 步内归零** —— 加这一层没有造出自持活动。')
    else:
        for x in alive:
            print('  r=%.2f %s%s 尾段仍有 %.0f 次放电（周期 %.0f 步，自相关 %.2f）' % (
                x['r'], x['mode'], ' shuf' if x['shuffle'] else '',
                x['tail_sum'], x['period'], x['ac']))


if __name__ == '__main__':
    main()
