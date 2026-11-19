# grow_stream.py — 连续事件流上的 trace conditioning（取代 grow_trace 的独立 trial 形态）
#
# 为什么重写（用户 2026-10-25 连续指出三个问题）：
#   ① 「你刺激到底有几个？」——grow_trace 每个 trial 只有 1 个刺激且状态清零，
#      那是 delayed match-to-sample 的独立分类，**不是 conditioning**。
#   ② 「噪声和真输入甚至不在一个通道上，如何模型甚至还写不会」——噪声在独立 16 通道，
#      BP 直接把那些通道权重压到 0 就绕开了，噪声贡献为零。
#   ③ 「迭代次数是多少？」——原实验对所有 L 都只做 500 次更新。
#
# 本文件的形态（用户描述）：
#
#   [刺激 S1] → 沉默(噪声) → [提醒 R1] → 说 S1
#   [刺激 S2] → 沉默(噪声) → [提醒 R2] → 说 S2
#   ... 一条长序列里有多个刺激；
#   揬入「孤立提醒」：前面没有未消费的刺激 → 提醒时必须沉默。
#
#   噪声：叠在**同一批通道**上（泊松背景，全程），刺激是在同一个通道上的额外同步脉冲。
#   状态：整条序列内连续，不重置；只在序列之间重置。
#   间隔：均值与标准差由小到大推进（VI 课程）。
#
# 实现上复用 grow_rnn 的本体：Elman2.step / backward 都是「按 t 索引一整条序列」，
# 直接能用。只换 N_IN / N_OUT。
#
# 运行：
#   python experiments/prototypes/grow_stream.py --sanity          # 小规模验证装置
#   python experiments/prototypes/grow_stream.py                   # 主跑：间隔课程 3->120
#   python experiments/prototypes/grow_stream.py --gap=80 --nseed=4 # 固定间隔，量跨度

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.prototypes import grow_rnn as G

# ---------------- 通道分配（信号与噪声**同通道**）----------------
N_CUE = 4
HINT = 4
N_NOISE = 16
N_IN = N_CUE + 1 + N_NOISE      # 21
N_OUT = N_CUE + 1               # 5：4 个词 + silent
SILENT = N_CUE

G.N_IN, G.N_H, G.N_OUT = N_IN, 32, N_OUT

CUE_STEPS = 3                    # 刺激持续 3 步（同步脉冲串，不是阶跃）
NOISE_RATE = 0.02                # **所有通道**的泊松背景
T_SEQ = 600                      # 一条序列的长度


def build_stream(rng, T, gap_mean, gap_cv, p_none, noise_rate=NOISE_RATE):
    """生成一条事件流。返回 X / y / mask。

    关键：噪声先铺满**所有 21 个通道**，刺激与提醒是**叠在同一个通道上的额外脉冲**。
    所以「有没有刺激」必须从噪声里检出，噪声是真干扰，不是可分离的旁路。
    """
    X = (rng.random((T, N_IN)) < noise_rate).astype(np.float64)
    y = np.full(T, SILENT, np.int64)
    mask = np.zeros(T, bool)          # True = 这一刻有监督

    t = 10
    while t < T - 20:
        # ---- 孤立提醒（前面没有未消费的刺激）→ 必须沉默 ----
        if rng.random() < p_none:
            X[t, HINT] += 1.0
            y[t] = SILENT
            mask[t] = True
            t += 1 + int(rng.integers(2, 8))
            continue

        # ---- 刺激 + 间隔 + 提醒 ----
        cue = int(rng.integers(0, N_CUE))
        n_steps = min(CUE_STEPS, T - t)
        if n_steps <= 0:
            break
        X[t:t + n_steps, cue] += 1.0
        t += n_steps

        if gap_cv > 1e-6 and gap_mean > 1.0:
            shape = 1.0 / (gap_cv * gap_cv)
            gap = int(rng.gamma(shape, gap_mean * gap_cv * gap_cv))
            gap = max(1, gap)
        else:
            gap = max(1, int(round(gap_mean)))
        t += gap
        if t >= T - 2:
            break

        X[t, HINT] += 1.0
        y[t] = cue
        mask[t] = True
        t += 1 + int(rng.integers(2, 8))

    return X, y, mask


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_seq(model, rng, opt, T, gap_mean, gap_cv, p_none, noise_rate,
            clip=5.0, silent_w=0.0):
    """跑一条序列。silent_w>0 时同时监督沉默帧（只训输出层，不推动状态）。"""
    X, y, mask = build_stream(rng, T, gap_mean, gap_cv, p_none, noise_rate)
    Z, H1, H2 = forward(model, X)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return dict(n=0, ok=0, acc=0.0, said=0.0, gnorm=0.0, grads=None)

    pred = Z[idx].argmax(axis=1)
    ok = int((pred == y[idx]).sum())

    dZ = np.zeros((T, N_OUT))
    state_mask = np.zeros(T, bool)
    state_mask[idx] = True         # 只有监督点的输出误差可以推动状态
    for t in idx:
        p = _softmax(Z[t])
        p[y[t]] -= 1.0
        dZ[t] = p
    if silent_w > 0:
        n_sil = T - idx.size
        if n_sil > 0:
            for t in range(T):
                if mask[t]:
                    continue
                q = _softmax(Z[t])
                q[SILENT] -= 1.0
                dZ[t] = q * silent_w / n_sil

    grads = model.backward(X, H1, H2, dZ, state_mask=state_mask)
    gnorm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
    return dict(n=int(idx.size), ok=ok, acc=ok / float(idx.size),
                said=float((pred != SILENT).mean()), gnorm=gnorm, grads=grads)


def forward(model, X):
    T = X.shape[0]
    H1 = np.zeros((T + 1, G.N_H))
    H2 = np.zeros((T + 1, G.N_H))
    Z = np.zeros((T, N_OUT))
    for t in range(T):
        Z[t] = model.step(X, H1, H2, t)
    return Z, H1, H2


def eval_gap(model, rng, gap, T, p_none, noise_rate, n_seq=4):
    """固定间隔下测：有刺激的提醒正确率 / 孤立提醒的沉默率 / 总体。"""
    ok_c = n_c = ok_s = n_s = 0
    for _ in range(n_seq):
        X, y, mask = build_stream(rng, T, gap, 0.0, p_none, noise_rate)
        Z, _, _ = forward(model, X)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        pred = Z[idx].argmax(axis=1)
        for t, p in zip(idx, pred):
            if y[t] == SILENT:
                n_s += 1
                ok_s += int(p == SILENT)
            else:
                n_c += 1
                ok_c += int(p == y[t])
    acc_c = ok_c / n_c if n_c else float('nan')
    acc_s = ok_s / n_s if n_s else float('nan')
    tot = (ok_c + ok_s) / max(1, n_c + n_s)
    return dict(acc_cue=acc_c, acc_sil=acc_s, acc=tot, n_cue=n_c, n_sil=n_s)


def run_one(arg):
    (seed, n_upd, bs, lr, clip, rho0, n_h, T, p_none, gap_max, gap_cv0,
     noise_rate, silent_w) = arg
    G.N_H = n_h
    rng = np.random.default_rng(seed)
    model = G.Elman2(rng, rho0=rho0)
    opt = G.Adam(model.params(), lr=lr)

    buf = None
    nbuf = 0
    hist = []
    gnorms = []
    n_done = 0
    upd = 0
    while upd < n_upd:
        frac = min(1.0, n_done / float(n_upd * bs))
        # 间隔课程：均值几何增长，**短间隔永久保留固定比例**（不保留就会被长间隔梯度冲掉）
        gap_mean = 3.0 * (float(gap_max) / 3.0) ** frac
        if rng.random() < 0.5:
            gap_mean = min(gap_mean, 12.0)
        r = run_seq(model, rng, opt, T, gap_mean, gap_cv0, p_none, noise_rate,
                    clip=clip, silent_w=silent_w)
        n_done += 1
        hist.append((gap_mean, r['acc'], r['n']))
        gnorms.append(r['gnorm'])
        if buf is None:
            buf = {k: v.copy() for k, v in r['grads'].items()}
        else:
            for k in buf:
                buf[k] += r['grads'][k]
        nbuf += 1
        if nbuf >= bs:
            for k in buf:
                buf[k] /= float(nbuf)
            gn = float(np.sqrt(sum(float((b * b).sum()) for b in buf.values())))
            if clip > 0 and gn > clip:
                s = clip / gn
                for k in buf:
                    buf[k] *= s
            opt.update(model, buf)
            buf = None
            nbuf = 0
            upd += 1

    rho = float(model.spectral_radius()[0])

    # 评估：扫间隔
    gaps = (5, 10, 20, 40, 80, 160)
    ev = {}
    for gp in gaps:
        ev[gp] = eval_gap(model, rng, gp, T, p_none, noise_rate, n_seq=4)

    k = max(1, len(hist) // 5)
    curve = [float(np.mean([h[1] for h in hist[i * k:(i + 1) * k]])) for i in range(5)]
    gmean = [float(np.mean([h[0] for h in hist[i * k:(i + 1) * k]])) for i in range(5)]

    return dict(seed=seed, n_h=n_h, rho=rho, upd=upd,
                gnorm=float(np.mean(gnorms)) if gnorms else 0.0,
                curve=curve, gmean=gmean, ev=ev)


def main():
    sanity = '--sanity' in sys.argv
    n_upd = 300 if sanity else 1500
    bs = 4
    nseed = 1 if sanity else 4
    lr = 1e-3
    clip = 5.0
    rho0 = 0.95
    nh_list = [32]
    T = 300 if sanity else T_SEQ
    p_none = 0.25
    gap_max = 40 if sanity else 120
    gap_cv0 = 0.5
    noise_rate = NOISE_RATE
    silent_w = 0.0
    for a in sys.argv[1:]:
        if a.startswith('--nupd='):
            n_upd = int(a.split('=', 1)[1])
        if a.startswith('--nseed='):
            nseed = int(a.split('=', 1)[1])
        if a.startswith('--nh='):
            nh_list = [int(x) for x in a.split('=', 1)[1].split(',')]
        if a.startswith('--lr='):
            lr = float(a.split('=', 1)[1])
        if a.startswith('--gap='):
            gap_max = int(a.split('=', 1)[1])
        if a.startswith('--T='):
            T = int(a.split('=', 1)[1])
        if a.startswith('--noise='):
            noise_rate = float(a.split('=', 1)[1])
        if a.startswith('--silent='):
            silent_w = float(a.split('=', 1)[1])

    tasks = [(s, n_upd, bs, lr, clip, rho0, nh, T, p_none, gap_max, gap_cv0,
              noise_rate, silent_w) for nh in nh_list for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 108)
    print('grow_stream — 连续事件流 + trace conditioning')
    print('  一条序列 %d 步，内含多个（刺激 → 间隔 → 提醒）对' % T)
    print('  噪声：%d 个通道**与刺激同批**，rate=%.3f（刺激是同通道上的额外脉冲）'
          % (N_IN, noise_rate))
    print('  孤立提醒（前面无刺激）占 %.0f%% → 必须沉默' % (p_none * 100))
    print('  间隔课程：均值 3 → %d（cv=%.2f），短间隔永久保留 50%%' % (gap_max, gap_cv0))
    print('  %d 次参数更新 × bs=%d；lr=%g；clip=%g；N_H %s'
          % (n_upd, bs, lr, clip, nh_list))
    print('=' * 108)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    for nh in nh_list:
        rs = [r for r in res if r['n_h'] == nh]
        if not rs:
            continue
        print('\n' + '-' * 108)
        print('N_H=%d   （%d 种子，%d 次更新）' % (nh, len(rs), rs[0]['upd']))
        print('  %-5s %-8s %-9s %s' % ('seed', 'rho', '梯度范数', '训练末段 acc'))
        for r in rs:
            print('  %-5d %-8.3f %-9.3g %.1f%%'
                  % (r['seed'], r['rho'], r['gnorm'], r['curve'][-1] * 100))

        print('\n  记忆曲线（有刺激的提醒正确率，序列数 4）：')
        gaps = sorted(rs[0]['ev'].keys())
        print('  %-10s' % '间隔' + ''.join('%-9d' % g for g in gaps))
        print('  %-10s' % 'acc_cue' + ''.join(
            '%-9s' % ('%.0f%%' % (np.nanmean([r['ev'][g]['acc_cue'] for r in rs]) * 100))
            for g in gaps))
        print('  %-10s' % 'acc_sil' + ''.join(
            '%-9s' % ('%.0f%%' % (np.nanmean([r['ev'][g]['acc_sil'] for r in rs]) * 100))
            for g in gaps))
        print('  逐种子 acc_cue: ')
        for r in rs:
            print('    seed %d: ' % r['seed'] + ' / '.join(
                '%.0f%%' % (r['ev'][g]['acc_cue'] * 100) for g in gaps))
        print('  学习曲线（训练内 acc）: ' + ' '.join(
            '%.0f%%' % (c * 100) for c in np.mean([r['curve'] for r in rs], axis=0)))
        print('  对应间隔均值: ' + ' '.join(
            '%.1f' % c for c in np.mean([r['gmean'] for r in rs], axis=0)))

    print('\n' + '=' * 108)
    return 0


if __name__ == '__main__':
    sys.exit(main())
