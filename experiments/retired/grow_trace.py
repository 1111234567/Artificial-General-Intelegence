# grow_trace.py — trace conditioning：用心理学范式测 RNN 的记忆跨度
#
# 形态（用户 2026-10-25 定，取代 grow_rnn 的 --blind 变体）：
#
#   [刺激 2 步] → [噪声期 L 步] → [提醒 1 步] → [尾 5 步]
#
#   要求（每一帧都有监督）：
#     · 刺激期 / 噪声期 / 尾期：沉默（silent）
#     · 提醒帧：说出刚才的刺激（4 选 1）
#     · **无刺激的试次：提醒帧也必须沉默**（反向对照——抢答在这里全错）
#
#   中间全部用噪声填补（模拟 SNN 后来要面对的背景输入）：
#     X[:, N_WORD+1:] = 泊松(noise_rate)，与 SNN 的稀疏脉冲输入同形态
#
#   提醒间隔 L：Gamma 采样，**mean 与 std 一起由小到大**（CV 固定 → std = mean·cv）
#     这是心理学里的 VI（variable interval）课程：间隔不可预测，行为才成为习惯性的。
#     L 是这一轮要测的自变量 —— 它直接就是「记忆跨度」。
#
# 为什么要换掉 --blind：
#   blind 变体里目标在 blind 之前一直可见，网络可以先把答案压进输出，
#   测到的不是「记住了多久」。trace conditioning 里刺激只出现 2 步，
#   **不靠内部状态跨过 L 就必然失败** —— 这是海马依赖的经典范式。
#
# 五类输出：0..3 = 四个词，4 = silent
# 损失：提醒帧的 CE + 沉默帧的 ΣCE / n_sil （两类等权，防「永远沉默」吃掉梯度）
#
# 复用 grow_rnn 的本体（Elman2 / Adam）—— 本体只有一份，不复制。
#
# 运行：
#   python experiments/retired/grow_trace.py --sanity                       # 小规模，确认装置 work
#   python experiments/retired/grow_trace.py                                # 主跑：8 种子
#   python experiments/retired/grow_trace.py --rho=0.90,0.95,0.98 --nh=32    # 扫初始谱半径

import os
import sys
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.prototypes import grow_rnn as G

# ---------------- 任务常数 ----------------
N_WORD = 4                 # 四个词
N_NOISE = 16               # 噪声通道数
N_IN = N_WORD + 1 + N_NOISE
N_OUT = N_WORD + 1         # 4 个词 + silent
SILENT = N_WORD
CUE_STEPS = 2              # 刺激持续 2 步
TAIL = 5                   # 提醒之后的尾（仍要求沉默）
L_MIN = 3
P_NONE = 0.25              # 训练时 25% 的试次没有刺激

EVAL_L = (5, 10, 20, 40, 50, 60, 80, 160, 320)

# 【关键】本体类在 grow_rnn 里用模块级全局决定形状，这里必须先补丁，否则 W1 会是 (N_H, 2)
G.N_IN, G.N_H, G.N_OUT = N_IN, 32, N_OUT


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_L(rng, mean, cv):
    """Gamma 采样：mean 与 std 一起变大（cv 固定时 std = mean*cv）。"""
    if mean <= 1.0:
        return L_MIN
    shape = 1.0 / (cv * cv)
    scale = mean * cv * cv
    return max(1, int(rng.gamma(shape, scale)))


def build_X(rng, L, cue, noise_rate):
    """构造一条序列的输入。T = CUE + L + 1(提醒) + TAIL。"""
    T = CUE_STEPS + L + 1 + TAIL
    X = np.zeros((T, N_IN), np.float64)
    if cue >= 0:
        X[:CUE_STEPS, cue] = 1.0          # 刺激只在开头出现 2 步
    X[CUE_STEPS + L, N_WORD] = 1.0        # 提醒帧
    if noise_rate > 0:
        X[:, N_WORD + 1:] = (rng.random((T, N_NOISE)) < noise_rate)
    return X


def forward(model, X):
    T = X.shape[0]
    H1 = np.zeros((T + 1, G.N_H))
    H2 = np.zeros((T + 1, G.N_H))
    Z = np.zeros((T, N_OUT))
    for t in range(T):
        Z[t] = model.step(X, H1, H2, t)
    return Z, H1, H2


def _clip(grads, max_norm=1.0):
    """全局范数裁剪。长序列 BPTT 不裁剪时循环权会被推到超临界（实测 rho=2.72、τ_eff 撞上限）。"""
    tot = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if max_norm > 0 and tot > max_norm:
        s = max_norm / tot
        for k in grads:
            grads[k] *= s
    return tot


def run_trial(model, rng, opt, L, cue, noise_rate, train=True, clip=1.0, hint_w=1.0,
              no_sil=False, nomask=False, upd=True):
    """一条序列。返回读数；train=True 时顺带做一次 BPTT 更新。"""
    X = build_X(rng, L, cue, noise_rate)
    Z, H1, H2 = forward(model, X)
    T = X.shape[0]
    t_hint = CUE_STEPS + L
    target = cue if cue >= 0 else SILENT

    pred = Z.argmax(axis=1)
    hint_ok = float(pred[t_hint] == target)
    sil_mask = np.ones(T, bool)
    sil_mask[t_hint] = False
    sil_ok = float((pred[sil_mask] == SILENT).mean()) if sil_mask.any() else 1.0
    said_word = float(pred[t_hint] != SILENT)

    if train:
        dZ = np.zeros((T, N_OUT))
        state_mask = np.zeros(T, bool)
        state_mask[t_hint] = True            # 只有提醒帧的输出误差可以推动状态
        p = _softmax(Z[t_hint])
        p[target] -= 1.0
        dZ[t_hint] = p * hint_w          # 提醒帧
        if not no_sil:
            n_sil = T - 1
            for t in range(T):
                if t == t_hint:
                    continue
                q = _softmax(Z[t])
                q[SILENT] -= 1.0
                dZ[t] = q / n_sil            # 沉默帧：合计权重 1，但只训输出层
        grads = model.backward(X, H1, H2, dZ,
                              state_mask=(state_mask if not nomask else None))
        gn = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
        if upd:
            _clip(grads, clip)
            opt.update(model, grads)
    else:
        gn = 0.0
        grads = None

    return dict(hint_ok=hint_ok, sil_ok=sil_ok, said_word=said_word, gnorm=gn,
                has_cue=bool(cue >= 0), grads=grads)


def eval_L(model, rng, L, n_trial, noise_rate, p_none=0.0):
    """固定 L 下测：有 cue 正确率 / 无 cue 沉默率 / 沉默帧沉默率（**分开报**）。"""
    ok_cue = n_cue = ok_none = n_none = 0
    sil_ok = sil_n = 0
    n_said = 0
    for _ in range(n_trial):
        cue = -1 if rng.random() < p_none else int(rng.integers(0, N_WORD))
        X = build_X(rng, L, cue, noise_rate)
        Z, _, _ = forward(model, X)
        T = X.shape[0]
        t_hint = CUE_STEPS + L
        pred = Z.argmax(axis=1)
        if cue >= 0:
            n_cue += 1
            ok_cue += int(pred[t_hint] == cue)
        else:
            n_none += 1
            ok_none += int(pred[t_hint] == SILENT)
        n_said += int(pred[t_hint] != SILENT)
        m = np.ones(T, bool)
        m[t_hint] = False
        sil_n += int(m.sum())
        sil_ok += int((pred[m] == SILENT).sum())
    n_ok = ok_cue + ok_none
    n_all = n_cue + n_none
    return dict(n=n_all, ok=n_ok, acc=n_ok / max(n_all, 1),
                acc_cue=ok_cue / max(n_cue, 1) if n_cue else float('nan'),
                acc_none=ok_none / max(n_none, 1) if n_none else float('nan'),
                n_cue=n_cue, n_none=n_none,
                sil=sil_ok / max(sil_n, 1), said=n_said / max(n_all, 1))


def _quick_val(model, rng, L, noise_rate, p_none, n_val=20):
    return eval_L(model, rng, L, n_val, noise_rate, p_none=p_none)['acc']


def tau_eff(model, n_steps=1200, warm=400):
    """无输入下「偏离不动点」的衰减到 1/e 的步数。

    ⚠️ 必须减掉零输入稳态的基线：bias 会把状态推到一个非零不动点，
    直接看 ‖h2‖ 就永远撞上限（实测 rho=0.99 也报 796 步，而理论值只要 100 步）。
    """
    X = np.zeros((n_steps, N_IN), np.float64)
    H1 = np.zeros((n_steps + 1, G.N_H))
    H2 = np.zeros((n_steps + 1, G.N_H))
    norms = np.zeros(n_steps)
    for t in range(n_steps):
        model.step(X, H1, H2, t)
        norms[t] = np.linalg.norm(H2[t + 1])
    base = float(norms[-max(1, n_steps // 10):].mean())

    X2 = np.zeros((n_steps, N_IN), np.float64)
    X2[warm:warm + CUE_STEPS, 0] = 1.0
    H1 = np.zeros((n_steps + 1, G.N_H))
    H2 = np.zeros((n_steps + 1, G.N_H))
    dev = np.zeros(n_steps)
    for t in range(n_steps):
        model.step(X2, H1, H2, t)
        dev[t] = abs(np.linalg.norm(H2[t + 1]) - base)
    peak_i = int(np.argmax(dev))
    peak = float(dev[peak_i])
    if peak < 1e-9:
        return 0.0
    thr = peak / np.e
    for t in range(peak_i + 1, n_steps):
        if dev[t] < thr:
            return float(t - peak_i)
    return float(n_steps - peak_i)


# ---------------- 一个 (rho0, n_h, seed) ----------------
def run_one(arg):
    (seed, n_ep, l_max, cv, lr, rho0, n_h, noise_rate, warm_frac, clip, hint_w,
     fixed_L, p_none, renorm_every, rho_target, cue_steps, no_sil, nomask,
     restore_best, bs, anchor, anchor_frac) = arg
    G.N_H = n_h
    global CUE_STEPS
    CUE_STEPS = cue_steps
    rng = np.random.default_rng(seed)
    model = G.Elman2(rng, rho0=rho0)
    opt = G.Adam(model.params(), lr=lr)

    hist = []
    gnorms = []
    best = None
    best_acc = -1.0
    best_ep = -1
    vrng = np.random.default_rng(99991 + seed)   # 验证专用 rng，不打乱训练轨迹
    buf = None
    nbuf = 0
    for ep in range(n_ep):
        frac = ep / max(1, n_ep - 1)
        if fixed_L > 0:
            mean = fixed_L
            L = fixed_L
        else:
            # 课程：上限几何增长，但**短档永久保留固定比例**。
            # 只涨上限会让短 L 被长 L 的梯度压掉 → 实测遗忘（L=5 从 97% 掉到 77%）。
            u = min(1.0, (ep + 1) / float(n_ep))
            l_cur = max(L_MIN, L_MIN * (float(l_max) / L_MIN) ** u)
            hi = min(l_cur, float(anchor)) if rng.random() < anchor_frac else l_cur
            t = rng.random()
            L = int(round(L_MIN * (hi / L_MIN) ** t))
            L = max(L_MIN, min(L, int(l_cur)))
            mean = int(l_cur)
        cue = -1 if rng.random() < p_none else int(rng.integers(0, N_WORD))
        r = run_trial(model, rng, opt, L, cue, noise_rate, train=True,
                      clip=clip, hint_w=hint_w, no_sil=no_sil, nomask=nomask, upd=False)
        hist.append((mean, L, r['hint_ok'], r['has_cue']))
        gnorms.append(r['gnorm'])
        # ---- minibatch：积 BS 个样本的梯度再更新一次（batch=1 时实测未段 87% / 最终 36%）----
        g = r['grads']
        if buf is None:
            buf = {k: v.copy() for k, v in g.items()}
        else:
            for k in buf:
                buf[k] += g[k]
        nbuf += 1
        if nbuf >= bs:
            for k in buf:
                buf[k] /= float(nbuf)
            _clip(buf, clip)
            opt.update(model, buf)
            buf = None
            nbuf = 0
        # 周期性把循环权锚回目标谱半径（**只压不拉**：往大拉会与 Adam 的 m/v 失配）
        if renorm_every > 0 and (ep + 1) % renorm_every == 0:
            r_now = model.spectral_radius()[0]
            if r_now > rho_target:
                s = rho_target / r_now
                model.U1 *= s
                model.U2 *= s
        # 最优检查点：不存的话「末段平均 73%」与「最终模型 27%」无法调和（模型在震荡）
        if (ep + 1) % 100 == 0 or ep == n_ep - 1:
            acc = _quick_val(model, vrng, fixed_L if fixed_L > 0 else int(L), noise_rate,
                             p_none, n_val=40)
            if acc > best_acc:
                best_acc = acc
                best = {k: getattr(model, k).copy() for k in model.keys()}
                best_ep = ep

    # 默认不 restore：先直接量「最终模型」，否则会把最佳检查点偏差混进来
    if best is not None and restore_best:
        for k in model.keys():
            setattr(model, k, best[k])

    # 诊断：训练结束后立即用与训练同分布测一次，与 curve[-1] 对比（两者应该接近）
    post_acc = _quick_val(model, vrng, fixed_L if fixed_L > 0 else int(l_max),
                          noise_rate, p_none, n_val=40)

    rho, r1, r2 = model.spectral_radius()
    tau = tau_eff(model)
    ev = {}
    for L in EVAL_L:
        # 评估必须与训练同分布（p_none 一致），否则「训练 73% / 评估 28%」无法解释
        ev[L] = eval_L(model, rng, L, 60, noise_rate, p_none=p_none)
    ev['_none'] = eval_L(model, rng, 60, 120, noise_rate, p_none=1.0)

    # 学习曲线：按训练进度分 5 段（有 cue / 无 cue 分开算，混着算会掩盖「总是沉默」）
    k = len(hist) // 5
    curve = [float(np.mean([h[2] for h in hist[i * k:(i + 1) * k]])) for i in range(5)]
    lmean = [float(np.mean([h[1] for h in hist[i * k:(i + 1) * k]])) for i in range(5)]
    curve_cue = []
    curve_none = []
    for i in range(5):
        seg = hist[i * k:(i + 1) * k]
        c = [h[2] for h in seg if h[3]]
        n = [h[2] for h in seg if not h[3]]
        curve_cue.append(float(np.mean(c)) if c else float('nan'))
        curve_none.append(float(np.mean(n)) if n else float('nan'))

    return dict(seed=seed, rho0=rho0, n_h=n_h, n_ep=n_ep, l_max=l_max,
                lr=lr, clip=clip, hint_w=hint_w,
                fixed_L=fixed_L, p_none=p_none, no_sil=no_sil, cue_steps=cue_steps,
                gnorm=float(np.mean(gnorms)) if gnorms else 0.0,
                best_acc=float(best_acc), best_ep=int(best_ep), post_acc=float(post_acc),
                curve_cue=curve_cue, curve_none=curve_none,
                rho=float(rho), rho1=float(r1), rho2=float(r2),
                tau=tau, curve=curve, lmean=lmean, ev=ev)


# ---------------- main ----------------
def main():
    sanity = '--sanity' in sys.argv
    n_ep = 500 if sanity else 3000
    l_max = 40 if sanity else 240
    rho_list = [0.95]
    nh_list = [32]
    nseed = 1 if sanity else 8
    lr_list = [1e-3]
    clip = 1.0
    hint_w = 1.0
    cv = 0.5
    noise_rate = 0.02
    warm_frac = 0.25
    fixed_L = 0
    fixed_list = [0]
    cue_steps = CUE_STEPS
    no_sil = False
    nomask = False
    restore_best = False
    bs = 16
    anchor = 10
    anchor_frac = 0.5
    nupd = 0          # >0 时按「参数更新次数」控训练量（n_ep = nupd * bs），长 L 才不会被欠训
    p_none = P_NONE
    renorm_every = 0
    rho_target = 0.95
    for a in sys.argv[1:]:
        if a.startswith('--nep='):
            n_ep = int(a.split('=', 1)[1])
        if a.startswith('--lmax='):
            l_max = int(a.split('=', 1)[1])
        if a.startswith('--rho='):
            rho_list = [float(x) for x in a.split('=', 1)[1].split(',')]
        if a.startswith('--nh='):
            nh_list = [int(x) for x in a.split('=', 1)[1].split(',')]
        if a.startswith('--nseed='):
            nseed = int(a.split('=', 1)[1])
        if a.startswith('--lr='):
            lr_list = [float(x) for x in a.split('=', 1)[1].split(',')]
        if a.startswith('--clip='):
            clip = float(a.split('=', 1)[1])
        if a.startswith('--hint='):
            hint_w = float(a.split('=', 1)[1])
        if a.startswith('--cv='):
            cv = float(a.split('=', 1)[1])
        if a.startswith('--noise='):
            noise_rate = float(a.split('=', 1)[1])
        if a.startswith('--fixedL='):
            fixed_list = [int(x) for x in a.split('=', 1)[1].split(',')]
        if a.startswith('--pnone='):
            p_none = float(a.split('=', 1)[1])
        if a.startswith('--renorm-every='):
            renorm_every = int(a.split('=', 1)[1])
        if a.startswith('--rho-target='):
            rho_target = float(a.split('=', 1)[1])
        if a.startswith('--cue='):
            cue_steps = int(a.split('=', 1)[1])
        if a == '--nosil':
            no_sil = True
        if a == '--nomask':
            nomask = True
        if a == '--restore-best':
            restore_best = True
        if a.startswith('--bs='):
            bs = int(a.split('=', 1)[1])
        if a.startswith('--nupd='):
            nupd = int(a.split('=', 1)[1])
        if a.startswith('--anchor='):
            anchor = int(a.split('=', 1)[1])
        if a.startswith('--anchor-frac='):
            anchor_frac = float(a.split('=', 1)[1])

    if nupd > 0:
        n_ep = nupd * bs

    tasks = [(s, n_ep, l_max, cv, lr, rho, nh, noise_rate, warm_frac, clip, hint_w,
              fl, p_none, renorm_every, rho_target, cue_steps, no_sil, nomask, restore_best, bs,
              anchor, anchor_frac)
             for rho in rho_list for nh in nh_list for lr in lr_list
             for fl in fixed_list for s in range(nseed)]
    nproc = min(len(tasks), os.cpu_count() or 4)

    print('=' * 112)
    print('grow_trace — trace conditioning 测记忆跨度')
    print('  序列 = 刺激 %d 步 → 噪声期 L → 提醒 1 步 → 尾 %d 步；中间 %d 维泊松噪声 (rate=%.3f)'
          % (CUE_STEPS, TAIL, N_NOISE, noise_rate))
    print('  五类输出（4 词 + silent）；%d%% 试次无刺激（提醒帧也必须沉默）' % int(P_NONE * 100))
    print('  L ~ Gamma(cv=%.2f)：mean 从 %d 几何增长到 %d（std 随 mean 一起涨）'
          % (cv, L_MIN, l_max))
    print('  %d trial × %d 种子；lr=%s；clip=%g；rho0 %s；N_H %s'
          % (n_ep, nseed, lr_list, clip, rho_list, nh_list))
    print('  参数更新次数 = %d（n_ep=%d / bs=%d）'
          % (n_ep // max(1, bs), n_ep, bs))
    print('=' * 112)

    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(run_one, tasks)
    else:
        res = [run_one(t) for t in tasks]

    for key in [(r, nh, fl) for r in rho_list for nh in nh_list for fl in fixed_list]:
        rs = [x for x in res if abs(x['rho0'] - key[0]) < 1e-9
              and x['n_h'] == key[1] and x['fixed_L'] == key[2]]
        if not rs:
            continue
        print('\n' + '-' * 112)
        print('rho0=%.2f  N_H=%d  lr=%s  %s   （%d 种子）'
              % (key[0], key[1], lr_list,
                 ('fixedL=%d' % key[2]) if key[2] > 0 else ('课程 L 3->%d' % l_max), len(rs)))
        print('  %-5s %-8s %-8s %-8s %-8s  %s' % (
            'seed', 'rho终', 'tau_eff', '提醒正确', '无cue沉默', '沉默期沉默'))
        for r in rs:
            print('  %-5d %-8.3f %-8.1f %-8.1f%% %-8.1f%%  %.1f%%' % (
                r['seed'], r['rho'], r['tau'],
                r['ev'][EVAL_L[0]]['acc'] * 100,
                r['ev']['_none']['acc'] * 100,
                r['ev'][EVAL_L[-1]]['sil'] * 100))

        print('\n  记忆曲线（提醒帧正确率，60 trial/档）：')
        print('  %-9s' % 'L' + ''.join('%-9d' % L for L in EVAL_L) + '%-9s' % '无cue')
        print('  %-9s' % 'acc' + ''.join(
            '%-9s' % ('%.0f%%' % (np.mean([r['ev'][L]['acc'] for r in rs]) * 100))
            for L in EVAL_L) + '%-9s' % ('%.0f%%' % (np.mean([r['ev']['_none']['acc'] for r in rs]) * 100)))
        print('  %-9s' % 'acc_cue' + ''.join(
            '%-9s' % ('%.0f%%' % (np.nanmean([r['ev'][L]['acc_cue'] for r in rs]) * 100))
            for L in EVAL_L) + '%-9s' % ('—'))
        print('  %-9s' % '训练时L' + ''.join(
            '%-9s' % '—' for L in EVAL_L) + '%-9s' % '—')
        print('  逐种子 L=5 / 20 / 80 / 320:')
        for r in rs:
            print('    seed %d: ' % r['seed'] + ' / '.join(
                '%.0f%%' % (r['ev'][L]['acc'] * 100) for L in (5, 20, 80, 320)))
        print('  学习曲线（有cue）: ' +
              ' '.join('%.0f%%' % (c * 100) for c in np.nanmean([r['curve_cue'] for r in rs], axis=0)) +
              '   无cue: ' +
              ' '.join('%.0f%%' % (c * 100) for c in np.nanmean([r['curve_none'] for r in rs], axis=0)))
        print('  学习曲线（混合）: ' +
              ' '.join('%.0f%%' % (c * 100) for c in np.mean([r['curve'] for r in rs], axis=0)) +
              '   最优检查点: %.0f%% @ep%d' % (
                  np.mean([r['best_acc'] for r in rs]) * 100,
                  int(np.mean([r['best_ep'] for r in rs]))) +
              '   训练后即时重测: %.0f%%' % (np.mean([r['post_acc'] for r in rs]) * 100))
        print('  对应的 L 均值: ' +
              ' '.join('%.0f' % m for m in np.mean([r['lmean'] for r in rs], axis=0)))
        print('  ρ 终值 %.3f ± %.3f   τ_eff %.1f ± %.1f 步   梯度范数 %.3g' % (
            np.mean([r['rho'] for r in rs]), np.std([r['rho'] for r in rs]),
            np.mean([r['tau'] for r in rs]), np.std([r['tau'] for r in rs]),
            np.mean([r['gnorm'] for r in rs])))

    print('\n' + '=' * 112)
    return 0


if __name__ == '__main__':
    sys.exit(main())
