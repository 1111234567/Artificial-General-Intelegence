# grow_cond.py - 阶段 1/2 v3：经典条件作用 + 可变间隔（VI）课程
#
# 与 grow_talk.py（v2）的区别（全部来自 2026-10-24 讨论）：
#
#   ① **只奖励正确，不惩罚错误**（用户：「先让它对一次，错不错可能都无所谓」）
#      v2 每次都给 ±1，在 75% 错误率下期望 −0.5 → 输出入度从 25 崩到 5（实测）。
#      现在：对 → m=+1；错 / 抢答 / 沉默 → m=0（不产生更新）。
#      代价：错的行为不会被抑制，靠「对的行为被强化」相对占优。
#
#   ② **保持期从分布采样，均值与方差随进度上升**（用户：「模仿 variable interval」）
#      VI 是行为主义里抗消退最强的强化程序：间隔不可预测 → 行为变成习惯性的，
#      而不是「卡在某个时刻」。
#      实现：d ~ Gamma(shape=1/CV², scale=mean·CV²)，mean 从 40 线性涨到 320，
#            CV = 0.5 固定（所以方差随均值一起涨）。
#      每档都有短有长：短的维持行为、长的推动记忆。
#
#   ③ **呈现长度可变**（因为 delay 在变）：每个呈现 = 编码 + 保持 + 回答窗。
#
#   ④ 迹的时间常数 20 → 200（`TRACE_DECAY` 0.95 → 0.995）
#      依据：Zenodo 2026「eligibility window 必须超过动作→评价的延迟」。
#      v2 是 τ=20 而保持期 100 步 —— 信息在物理上已经没了。
#
# 判据（先写死）：
#   · 正确率：4 类随机 = 25%
#   · 按保持期长度的分桶正确率：短桶高、长桶逐步追上 = 巩固在发生
#   · 抢答率：应当不升（抢答拿不到奖励 = 机会成本）
#   · 放电率：不得失控（防「高频放电凑巧合」的 hack）

import sys
import time
import numpy as np
from multiprocessing import Pool

# ---- 网络（沿用 calib5 标定值）----
N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50
W_I_RATIO = 5.0
W_MAX_E = 1.0
W_MAX_I = 2.5
W_NEW_E = 0.15
W_NEW_I = 0.75
BUDGET = 15000
E0_INIT = 30000

P_NEW = 0.004
P_GLOBAL = 4
MAX_ACTIVE = 48
MIN_AGE = 25

# ---- 接口 ----
N_IN = 100
N_WORD = 4
BLK = 25                          # 每个词占 25 个输入神经元（0..99）
N_Q = 10                          # 提问信号（100..109）
Q_IDX = np.arange(N_IN, N_IN + N_Q)
IN_SPIKE = 3.0                    # 必须 > V_TH，否则输入神经元不放电
IN_LO = 0.001
OUT_IDX = np.array([500, 501, 502, 503])
N_OUT = 4
SIDE_INH = -0.5
N_OUT_INT = 40                    # 输出从内部网络接多少
N_OUT_LINK = N_Q + N_OUT_INT
N_IN_FANOUT = 50

# ---- 时序 ----
BURST = 20                        # 编码期长度
NBURST = 2                        # 编码期两次同步爆发
QBURST = 10                       # 提问脉冲窗长
ANSWER_WIN = 150                  # 回答窗（从提问开始算）

# ---- 课程：可变间隔（VI）----
D_MEAN_START = 40.0               # 初始：保持期 20 步（< 膜电位 τ）
D_MEAN_END = 320.0                # 结束：保持期 300 步（15 × τ）
D_CV = 0.5                        # 变异系数（方差随均值一起涨）
D_MIN = 30                        # 最短 = 保持 10 步
D_MAX = 520                       # 最长 = 保持 500 步

# ---- 学习 ----
ETA_W = 0.008
ETA_BIAS = 0.004
BIAS_MAX = 0.40
BIAS_MIN = -0.10
TAU_M = 100.0
K_UPDATE = 5
M_DECAY = float(np.exp(-1.0 / TAU_M))
WARMUP = 3000
# ⚠️ 迹必须保持短（τ≈20）。
# 2026-10-24 实测：把 τ 拉到 200（0.995）后，rate 从 1.55% 崩到 0.15%。
# 原因：el = trace[pre]*trace[post] 两个迹同一时间常数 → τ 一长，
# 所有边都「刚才活跃过」→ 归一化后选择性消失 → 全部边（含抑制边）一起饱和。
# 文献说的「eligibility window 必须超过奖励延迟」指的是「要覆盖奖励延迟」，
# 不是「把可塑性的迹拉长」。长时间记忆靠连接反复累积，不靠单次迹。
TRACE_DECAY = 0.95                # τ_trace = 20 步（保持选择性）

# ---- 稳态 ----
TAU_ADAPT = 100.0
D_THETA = 0.20
TH_MIN = 1.0
TH_MAX = 3.0
D_THETA_OUT = 0.02


def _sort(pre, post, w, age):
    key = post.astype(np.int64) * N + pre
    o = np.argsort(key, kind='stable')
    key, pre, post, w, age = key[o], pre[o], post[o], w[o], age[o]
    if key.size > 1:
        u = np.empty(key.size, bool); u[0] = True
        np.not_equal(key[1:], key[:-1], out=u[1:])
        if not u.all():
            key, pre, post, w, age = key[u], pre[u], post[u], w[u], age[u]
    return (np.ascontiguousarray(pre), np.ascontiguousarray(post),
            np.ascontiguousarray(w), np.ascontiguousarray(key),
            np.ascontiguousarray(age))


def find(key, q):
    if key.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    i = np.searchsorted(key, q)
    np.clip(i, 0, key.size - 1, out=i)
    return key[i] == q, i


def steady_gate(r):
    if r < 0.005:
        return max(0.0, r / 0.005)
    if r > 0.05:
        return max(0.0, 0.05 / r)
    return 1.0


def _init_net(rng):
    """随机结构 + 接口（输出只接提问区 + 内部网络，**不接词块**）+ 输出侧抑制。"""
    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = (pre != post) & (post >= N_IN) & (~np.isin(post, OUT_IDX))
    pre, post = pre[ok], post[ok]

    # 接口：提问区 10 + 内部 40。不接词块——输出完全不看词，只能靠内部残留。
    for o in OUT_IDX:
        src_i = rng.choice(np.arange(N_IN + N_Q, N_E),
                           N_OUT_INT, replace=False).astype(np.int32)
        src = np.concatenate([Q_IDX.astype(np.int32), src_i])
        pre = np.concatenate([pre, src])
        post = np.concatenate([post, np.full(src.size, int(o), np.int32)])

    # 输入区 → 网络本体的扇出
    n_src = np.repeat(np.arange(N_IN + N_Q, dtype=np.int32), N_IN_FANOUT)
    n_dst = rng.integers(N_IN + N_Q, N_E, (N_IN + N_Q) * N_IN_FANOUT).astype(np.int32)
    bad = np.isin(n_dst, OUT_IDX)
    while bad.any():
        n_dst[bad] = rng.integers(N_IN + N_Q, N_E, int(bad.sum())).astype(np.int32)
        bad = np.isin(n_dst, OUT_IDX)
    pre = np.concatenate([pre, n_src])
    post = np.concatenate([post, n_dst])

    # 输出侧抑制
    sp, sq = [], []
    for o in OUT_IDX:
        for p in OUT_IDX:
            if o != p:
                sp.append(int(o)); sq.append(int(p))
    pre = np.concatenate([pre, np.array(sp, np.int32)])
    post = np.concatenate([post, np.array(sq, np.int32)])

    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    nside = len(sp)
    nlink = N_OUT * N_OUT_LINK
    nfan = (N_IN + N_Q) * N_IN_FANOUT
    n_rand = pre.size - nlink - nfan - nside
    w[n_rand:n_rand + nlink] = rng.uniform(0.15, 0.60, nlink).astype(np.float32)
    w[pre.size - nside:] = SIDE_INH
    age = np.full(pre.size, 100, np.int16)
    return pre, post, w, age


def sample_delay(progress, rng):
    """可变间隔：Gamma 采样。均值随进度上升，CV 固定 → 方差也随之上升。"""
    m = D_MEAN_START + (D_MEAN_END - D_MEAN_START) * float(np.clip(progress, 0.0, 1.0))
    shape = 1.0 / (D_CV * D_CV)
    scale = m / shape
    d = int(rng.gamma(shape, scale))
    return max(D_MIN, min(D_MAX, d))


def run_one(args):
    seed, mode, n_pres, punish = args
    rng = np.random.default_rng(seed)

    pre, post, w, age = _init_net(rng)
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    trace = np.zeros(N, np.float32)
    theta = np.full(N, V_TH, np.float32)
    bias = np.zeros(N, np.float32)
    bias[OUT_IDX] = 0.05
    tref = np.full(N, T_REF, np.int32)
    tref[OUT_IDX] = 4

    m_out = np.zeros(N_OUT, np.float64)
    m_glob = 0.0
    ist_out = np.isin(post, OUT_IDX)
    is_inh_e = pre >= N_E

    use_plast = mode != 'frozen'
    use_bias = mode != 'frozen'

    # 统计
    nb = 8
    buck_n = np.zeros(nb, np.int64)
    buck_ok = np.zeros(nb, np.int64)
    win = []
    win_delay = []
    win_early = []
    win_sil = []
    win_rate = []
    n_corr = n_tot = n_sil = n_early = 0
    nsp_win = 0
    n_win = 0
    ema_rate = 0.02
    born = drop = 0
    s_a1a0 = s_a0a0 = 0.0
    a_prev = 0.0

    # 呈现状态
    t = 0
    pres = 0
    p_start = 0
    p_delay = 0
    p_end = 0
    c = 0
    answered = False
    early_flag = False
    lat = 0

    t0 = time.perf_counter()
    while t < n_pres * D_MEAN_END * 2 and pres < n_pres:
        # ===== 呈现边界 =====
        if t >= p_end:
            if pres > 0 and not answered:
                # 沉默
                n_tot += 1
                n_sil += 1
                if punish:
                    m_out[c] -= 1.0
                    m_glob = -1.0
            if pres > 0 and (pres % 100) == 0:
                win.append(n_corr / max(n_tot, 1))
                win_sil.append(n_sil / max(n_tot, 1))
                win_early.append(n_early / max(n_tot, 1))
                win_delay.append(p_delay)
                if n_win > 0:
                    win_rate.append(nsp_win / n_win / N)
                    nsp_win = n_win = 0
                n_corr = n_tot = n_sil = n_early = 0
            progress = pres / max(n_pres - 1, 1)
            p_start = t
            p_delay = sample_delay(progress, rng)
            p_end = p_start + p_delay + ANSWER_WIN
            c = int(rng.integers(0, N_WORD))
            answered = False
            early_flag = False
            lat = 0
            pres += 1

        phase = t - p_start

        # ===== 输入 =====
        xin = np.zeros(N, np.float32)
        if phase < BURST:
            if phase == 0 or phase == BURST - 2:
                xin[c * BLK:(c + 1) * BLK] = IN_SPIKE
        if p_delay <= phase < p_delay + QBURST:
            qq = phase - p_delay
            if qq == 0 or qq == QBURST // 2:
                xin[Q_IDX] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin[Q_IDX] += (rng.random(N_Q) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        # ===== 网络 =====
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = tref[s > 0]

        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
            ob = fire[OUT_IDX]
            if ob.any():
                oi = OUT_IDX[ob]
                theta[oi] = np.maximum(theta[oi] - D_THETA + D_THETA_OUT, TH_MIN)
        theta += (V_TH - theta) / TAU_ADAPT

        nsp = int(s.sum())
        nsp_win += nsp
        n_win += 1
        if t > 2000:
            s_a1a0 += float(nsp) * a_prev
            s_a0a0 += a_prev * a_prev
        a_prev = float(nsp)
        ema_rate = 0.99 * ema_rate + 0.01 * (nsp / N)
        trace *= TRACE_DECAY
        trace += s

        # ===== 抢答：只记录，**不终止本呈现** =====
        # 为何不终止：抢答率 100% 时，每个呈现都被废掉 → 正确率永远测不到，
        # 「说什么」这条链也测不了。先让它继续说，把「说什么」打通。
        # 「何时说」暂时不测（这是已知的代价，写进文档）。
        if (not answered) and BURST <= phase < p_delay:
            of = s[OUT_IDX] > 0
            if of.any():
                if not early_flag:
                    n_early += 1
                    n_tot += 1
                    early_flag = True
                if punish:
                    cand = np.nonzero(of)[0]
                    o = int(cand[np.argmax(v[OUT_IDX][cand])])
                    m_out[o] -= 1.0
                    m_glob = -1.0

        # 回答期
        if (not answered) and p_delay <= phase < p_delay + ANSWER_WIN:
            of = s[OUT_IDX] > 0
            if of.any():
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[OUT_IDX][cand])])
                answered = True
                lat = phase - p_delay
                ok = (o == c)
                n_tot += 1
                if ok:
                    n_corr += 1
                    # 只奖励正确
                    m_out[o] += 1.0
                    m_glob = 1.0
                else:
                    if punish:
                        m_out[o] -= 1.0
                        m_glob = -1.0
                # 分桶：按保持期长度（p_delay - BURST）
                hold = p_delay - BURST
                b = int(min(nb - 1, hold // 50))
                buck_n[b] += 1
                if ok:
                    buck_ok[b] += 1

        m_out *= M_DECAY
        m_glob *= M_DECAY

        # ===== 可塑性 =====
        if (t + 1) % K_UPDATE == 0 and t >= WARMUP:
            has_m = bool(np.abs(m_out).max() > 1e-6) or abs(m_glob) > 1e-6
            if use_plast and key.size and has_m:
                gate = steady_gate(ema_rate)
                el = trace[pre] * trace[post]
                mp = np.zeros(N, np.float32)
                mp[OUT_IDX] = m_out.astype(np.float32)
                sel = mp[post.astype(np.int64)]
                if abs(m_glob) > 1e-9:
                    sel = np.where(ist_out, sel, m_glob).astype(np.float32)
                nz = sel != 0
                if nz.any() and el[nz].max() > 1e-9:
                    el_n = el[nz] / float(el[nz].max())
                    dw = ETA_W * gate * el_n * sel[nz]
                    sl = np.nonzero(nz)[0]
                    pos = w[sl] > 0
                    w[sl] = np.where(pos,
                                     np.minimum(w[sl] + dw * (W_MAX_E - w[sl]), W_MAX_E),
                                     w[sl])
                    w[sl] = np.maximum(w[sl], 0.0)

            if use_bias:
                nb_ = bias[OUT_IDX] + ETA_BIAS * m_out.astype(np.float32)
                bias[OUT_IDX] = np.clip(nb_, BIAS_MIN, BIAS_MAX)

        # ===== 结构更新 =====
        if use_plast and t % 20 == 0 and t > 0:
            act = np.nonzero(trace > 0.5)[0]
            if act.size > MAX_ACTIVE:
                act = rng.choice(act, MAX_ACTIVE, replace=False)
            if act.size >= 2:
                na = act.size
                q_pre = np.repeat(act, na)
                q_post = np.tile(act, na)
                okk = q_pre != q_post
                q_pre, q_post = q_pre[okk], q_post[okk]
                qk = q_post.astype(np.int64) * N + q_pre
                found, _ = find(key, qk)
                npw, nqw = q_pre[~found], q_post[~found]
                if npw.size:
                    pick = rng.random(npw.size) < P_NEW
                    if pick.any():
                        npw, nqw = npw[pick], nqw[pick]
                        nw = np.where(npw < N_E, W_NEW_E, -W_NEW_I).astype(np.float32)
                        na_new = np.zeros(npw.size, np.int16)
                        pre, post, w, key, age = _sort(
                            np.concatenate([pre, npw.astype(np.int32)]),
                            np.concatenate([post, nqw.astype(np.int32)]),
                            np.concatenate([w, nw]),
                            np.concatenate([age, na_new]))
                        born += int(npw.size)

            if t % 1000 == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                okk = gi != gj
                gi, gj = gi[okk], gj[okk]
                if gi.size:
                    nw = np.where(gi >= N_E, -W_NEW_I, W_NEW_E).astype(np.float32)
                    na_new = np.zeros(gi.size, np.int16)
                    pre, post, w, key, age = _sort(
                        np.concatenate([pre, gi.astype(np.int32)]),
                        np.concatenate([post, gj.astype(np.int32)]),
                        np.concatenate([w, nw]),
                        np.concatenate([age, na_new]))
                    born += gi.size

            age += 1
            ist_out = np.isin(post, OUT_IDX)
            is_inh_e = pre >= N_E
            if key.size > BUDGET:
                d = key.size - BUDGET
                scale = np.abs(w) / np.where(is_inh_e, W_MAX_I, W_MAX_E)
                cand = (age >= MIN_AGE) & (~ist_out)
                if not cand.any():
                    cand = age >= MIN_AGE
                if cand.any():
                    idx_c = np.nonzero(cand)[0]
                    d = min(d, idx_c.size)
                    if d > 0:
                        sub = idx_c[np.argpartition(scale[idx_c], d - 1)[:d]]
                        mm = np.ones(key.size, bool); mm[sub] = False
                        pre, post, w, key, age = pre[mm], post[mm], w[mm], key[mm], age[mm]
                        drop += int(d)
                        ist_out = np.isin(post, OUT_IDX)
                        is_inh_e = pre >= N_E

        t += 1

    deg_in = np.bincount(post, minlength=N)
    bk = np.where(buck_n > 0, buck_ok / np.maximum(buck_n, 1), np.nan)
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        sigma=float(s_a1a0 / s_a0a0) if s_a0a0 > 0 else float('nan'),
        win=win, win_sil=win_sil, win_early=win_early, win_delay=win_delay,
        win_rate=win_rate, buck_n=buck_n, buck_acc=bk,
        bias=tuple(round(float(x), 4) for x in bias[OUT_IDX]),
        deg_out=int(deg_in[OUT_IDX].mean()),
        acc_first=float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        acc_last=float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        early_first=float(np.mean(win_early[:2])) if len(win_early) >= 2 else float('nan'),
        early_last=float(np.mean(win_early[-2:])) if len(win_early) >= 2 else float('nan'),
        sil_last=float(np.mean(win_sil[-2:])) if len(win_sil) >= 2 else float('nan'),
        delay_first=float(np.mean(win_delay[:2])) if len(win_delay) >= 2 else float('nan'),
        delay_last=float(np.mean(win_delay[-2:])) if len(win_delay) >= 2 else float('nan'),
        elapsed=time.perf_counter() - t0)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_pres = int(argv[0]) if len(argv) > 0 else 1200
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['teach', 'frozen']
    punish = '--punish' in sys.argv

    tasks = [(s, m, n_pres, punish) for m in modes for s in range(nseed)]
    print('=' * 130)
    print('grow_cond v3  经典条件作用 + 可变间隔（VI）')
    print('  modes=%s  呈现=%d  种子=%d  任务=%d' % (','.join(modes), n_pres, nseed, len(tasks)))
    print('  只奖励正确=%s  迹τ=%d步  保持期均值 %.0f->%.0f  CV=%.1f' % (
        not punish, round(1 / (1 - TRACE_DECAY)), D_MEAN_START, D_MEAN_END, D_CV))
    print('  编码 %d 步 / 回答窗 %d 步 / 提问窗 %d 步' % (BURST, ANSWER_WIN, QBURST))
    print('=' * 130)
    with Pool(nproc) as pool:
        t0 = time.perf_counter()
        allres = pool.map(run_one, tasks)
        wall = time.perf_counter() - t0

    by = {}
    for r in allres:
        by.setdefault(r['mode'], []).append(r)

    for mode in modes:
        res = by.get(mode, [])
        if not res:
            continue
        print('-' * 130)
        print('mode=%s  种子=%d' % (mode, len(res)))
        print('%-5s %6s %7s %8s %8s %8s %8s %7s' % (
            'seed', 'E', 'rate%', 'acc早', 'acc晚', '抢答早', '抢答晚', 'out_deg'))
        for r in res:
            print('%-5d %6d %7.2f %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7d' % (
                r['seed'], r['E'], r['rate'] * 100, r['acc_first'] * 100,
                r['acc_last'] * 100, r['early_first'] * 100, r['early_last'] * 100,
                r['deg_out']))

        W = np.array([r['win'] for r in res])
        D = np.array([r['win_delay'] for r in res])
        E = np.array([r['win_early'] for r in res])
        with np.errstate(invalid='ignore'):
            wm = np.nanmean(W, axis=0)
            dm = np.nanmean(D, axis=0)
            em = np.nanmean(E, axis=0)
        print('  正确率(每200呈现): %s' % ' '.join('%.2f' % x for x in wm))
        print('  抢答率            : %s' % ' '.join('%.2f' % x for x in em))
        print('  保持期均值        : %s' % ' '.join('%.0f' % x for x in dm))

        BN = np.array([r['buck_n'] for r in res]).sum(axis=0)
        BO = np.array([[x * n for x, n in zip(r['buck_acc'], r['buck_n'])] for r in res]).sum(axis=0)
        print('  ── 按保持期长度分桶的正确率（这是「巩固」的直接测量）──')
        for i in range(len(BN)):
            lab = '%d-%d' % (i * 50, (i + 1) * 50 - 1)
            if BN[i] > 0:
                print('    保持 %-9s n=%-7d 正确率 %.1f%%' % (lab, BN[i], 100.0 * BO[i] / BN[i]))
            else:
                print('    保持 %-9s n=0' % lab)

        A1 = np.array([r['acc_first'] for r in res]) * 100
        A2 = np.array([r['acc_last'] for r in res]) * 100
        print('  早期 %.1f%% -> 晚期 %.1f%%   提升 %+.1f pp   （随机 = 25%%）' % (
            A1.mean(), A2.mean(), A2.mean() - A1.mean()))
        SIG = np.array([r['sigma'] for r in res])
        print('  分支比 σ = %.3f ± %.3f' % (np.nanmean(SIG), np.nanstd(SIG)))
        RR = np.array([r['rate'] for r in res]) * 100
        print('  放电率 %.2f%% ± %.2f%%   （失控 = 「高频放电凑巧合」的 hack）' % (
            RR.mean(), RR.std()))

    print('')
    print('  总耗时 %.1fs' % wall)


if __name__ == '__main__':
    main()
