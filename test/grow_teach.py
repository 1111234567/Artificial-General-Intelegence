# grow_teach.py - 阶段 1/2：教它说话（有教师信号），不是阶段 3 的自探索
#
# 路线（design.md §5.1）：
#   阶段 1 反向/强制 → 阶段 2 后训练（掌握基本 + 知道自己不会）→ 阶段 3 前向（自反馈自主学）
#   本文件做阶段 1/2。前两版（grow_world / grow_func）做的全是阶段 3 的形态
#   （无任务、无教师、无奖励），而阶段 1/2 一步没做 —— 在那种形态上测「有没有学到」
#   是必然否定的，且不说明任何机制问题。
#
# 「说话」的最小形态：4 个词，用两个输出神经元（500 / 501）的放电模式编码
#   词 0 = 低 低   词 1 = 低 高   词 2 = 高 低   词 3 = 高 高
#   教师给一个目标词 c，网络看到对应输入模式后要说 c。
#
# 教师信号（三因子里的 m，design.md §14.6 / §7.3）：
#   每个 trial 结束，比对「说出的词」与「目标词」
#   说对 → m = +1；说错 → m = -1
#   m 按**突触后神经元**分配：m[post] —— 输出神经元 500/501 各自的调制信号不同
#   （全局标量 m 无法告诉网络「哪个输出错了」；按 post 分配是局部规则，不需要反传）
#
# 判据（先写死，防事后挑）：**正确率**。4 类随机 = 25%。
#   上升 → 有教师信号就能学（阶段 1/2 成立，可进入阶段 3）
#   不上升 → 可塑性规则 / 稳态 / 接口有问题，阶段 3 无从谈起
#
# 四组对照：
#   teach     正常教师
#   rand_m    m 随机 ±1（有可塑性、有信号、但信号与对错无关）
#   frozen    可塑性关闭（结构固定，只跑动力学）
#   no_teach  m 恒 +1（只强化，不区分对错）
#
# 只有 teach 上升、其余三组不上升，才说明「学的来源是教师信号」。

import sys
import time
import numpy as np
from multiprocessing import Pool

# ---- 网络（沿用 calib5 / grow_world 标定值）----
N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20

W_E = 0.50
W_I_RATIO = 5.0
ETA_POT = 0.02
W_MAX_E = 1.0
W_MAX_I = 2.5
W_NEW_E = 0.15
W_NEW_I = 0.75
BUDGET = 15000
E0_INIT = 30000

TRACE_DECAY = 0.95
P_NEW = 0.004
P_GLOBAL = 4
MAX_ACTIVE = 48
MIN_AGE = 25

# ---- 接口 ----
N_IN = 100
N_WORD = 4
IN_HI = 0.030           # 目标类对应的输入区放电率
IN_LO = 0.001           # 非目标区
# ⚠️ 输入率是硬约束：IN_HI=0.15 × 25 神经元 ≈ 3.75 脉冲/步，是 grow_world（~1）的 4 倍
# → 网络被推过雪崩临界点，rate 从 2% 冲到 89%（全饱和）。实测：test/probe_out.py。

# 4 个词 = 4 个输出神经元（one-hot）+ 侧抑制
# ⚠️ 为何不用「2 个神经元编码 2 bit」：那要求两个神经元对 4 个输入块产生
# 「各自有区分且组合起来恰好 4 种」的响应——需要精细的权重关系，60 条随机接口边
# 学不出来（实测：3000 trial 只 +3 个百分点）。one-hot 把它拆成 4 个独立的
# 「该不该说这个词」二分类，而且更像生物（一个词 = 一组神经元）。
OUT_IDX = np.array([500, 501, 502, 503])
N_OUT = 4
SIDE_INH = -1.5         # 输出神经元之间的侧抑制（竞争：一个强就压制其他）
TRIAL = 40              # 每个 trial 的步数（> T_REF，让不应期不主导）
READ_WIN = 20           # 读出窗口（trial 最后 20 步）
V_TH_OUT = 1.0          # 输出神经元的阈值（§14.6 作用点一：读出阈值）
T_REF_OUT = 4           # 输出神经元的不应期（否则最大放电率被锁在 1/20 = 5%）
# ⚠️ 为何输出神经元必须单独的不应期：全局 T_REF=20 把最大放电率锁在 5%，
# 20 步窗口内期望只有 0.44 次放电——无法区分「高/低」，更谈不上编码。
# 不应期是「读出粒度」的约束，不是动力学的一部分（§14.6 五个作用点的第一层）。

# ---- 学习 ----
ETA_TEACH = 0.008       # 教师调制的学习率（原 0.06 太大，十几次就饱和）
M_CORRECT = 1.0
M_WRONG = -1.0          # 按神经元分别给（下面 m0/m1）——全局同一标量表达不了
                        # 「该高的高了、该低的没低」

# ---- 接口（design.md §2.10：接口是本体侧必须给的三样之一）----
N_OUT_LINK = 60         # 给每个输出神经元预置的「输入区 -> 输出」直连数
BIAS_OUT = 0.05         # 输出神经元的基底输入（§14.6 第一个作用点：先诱导说）


def _init_net(rng):
    """随机初始结构 + 预置接口：输入区（0..99）→ 输出神经元（500/501）的直连。

    为什么必须有：随机初始下输出神经元的入度只有 ~25，其中来自输入区的期望仅 2.5 条。
    没有「输入 -> 输出」的通路，教师信号无法把「看到什么」与「说什么」连起来——
    这不是作弊，是把接口（哪些节点进、哪些节点读）真的给出来。
    """
    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]

    # 接口：每个输出神经元接一批输入神经元（随机，带初始强度差异）
    for o in OUT_IDX:
        src = rng.choice(N_IN, N_OUT_LINK, replace=False).astype(np.int32)
        pre = np.concatenate([pre, src])
        post = np.concatenate([post, np.full(src.size, int(o), np.int32)])

    # 侧抑制：输出神经元两两互抑制（只建单向即可，4×3 = 12 条）
    sp = []
    sq = []
    for o in OUT_IDX:
        for p in OUT_IDX:
            if o != p:
                sp.append(int(o))
                sq.append(int(p))
    pre = np.concatenate([pre, np.array(sp, np.int32)])
    post = np.concatenate([post, np.array(sq, np.int32)])

    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    n_tail = N_OUT * N_OUT_LINK + len(sp)
    # 接口边：**随机初始强度**（0.05~0.35），不是统一 0.15
    # 为何：若全部相同，四个输入块对输出的驱动完全对称——没有可放大的差异，
    # 教师信号无处可施（实测：四个词的 cnt 差异范数 0.33，低于 0.5 的读出阈值）。
    w[-n_tail:-len(sp)] = rng.uniform(0.05, 0.35, N_OUT * N_OUT_LINK).astype(np.float32)
    w[-len(sp):] = SIDE_INH
    age = np.full(pre.size, 100, np.int16)
    return pre, post, w, age


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
    """活动率落在 0.5%~5% 之外时把可塑性压下去（软门）。"""
    if r < 0.005:
        return max(0.0, r / 0.005)
    if r > 0.05:
        return max(0.0, 0.05 / r)
    return 1.0


def run_one(args):
    seed, mode, ntrials = args
    rng = np.random.default_rng(seed)

    pre, post, w, age = _init_net(rng)
    pre, post, w, key, age = _sort(pre, post, w, age)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    xin = np.zeros(N, np.float32)
    trace = np.zeros(N, np.float32)
    vth = np.full(N, V_TH, np.float32)
    vth[OUT_IDX] = V_TH_OUT
    tref = np.full(N, T_REF, np.int32)
    tref[OUT_IDX] = T_REF_OUT

    # 输入掩码：4 个词各占 25 个输入神经元
    masks = np.zeros((N_WORD, N_IN), np.float32)
    for c in range(N_WORD):
        masks[c, c * 25:(c + 1) * 25] = IN_HI
    masks[masks == 0] = IN_LO
    use_plast = mode != 'frozen'

    ema_rate = 0.02
    born = drop = 0
    correct_win = []
    n_corr = 0
    n_tot = 0
    conf_right = []
    conf_wrong = []
    t0 = time.perf_counter()

    u = rng.random((TRIAL, N_IN), dtype=np.float32)

    for it in range(ntrials):
        c = int(rng.integers(0, N_WORD))     # 教师给的目标词
        trace[:] = 0.0
        cnt = np.zeros(N_OUT, np.int64)
        for t in range(TRIAL):
            xin.fill(0.0)
            xin[0:N_IN] = u[t] < masks[c]
            xin[OUT_IDX] += BIAS_OUT        # 偏置：先诱导说（§14.6 作用点一）
            I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
            v = v * DECAY_V + I + xin
            fire = (v > vth) & (ref <= 0)
            s = fire.astype(np.float32)
            v = np.where(s > 0, 0.0, v)
            ref -= 1
            np.maximum(ref, 0, out=ref)
            ref[s > 0] = tref[s > 0]
            ema_rate = 0.99 * ema_rate + 0.01 * (float(s.sum()) / N)
            trace *= TRACE_DECAY
            trace += s
            if t >= TRIAL - READ_WIN:
                cnt += s[OUT_IDX].astype(np.int64)
            u[t] = rng.random(N_IN, dtype=np.float32)

        # ---- 读出：词 = argmax(cnt)（one-hot，带侧抑制竞争）----
        word = int(np.argmax(cnt))
        corr = (word == c)
        n_tot += 1
        if corr:
            n_corr += 1
        strength = float(cnt.sum())
        (conf_right if corr else conf_wrong).append(strength)

        # ---- 教师信号（三因子里的 m）：**误差方向**，按每个输出神经元分配 ----
        # 只学错的地方：
        #   该说没说（tgt=1, got=0）-> +1（强化）
        #   不该说却说了（tgt=0, got=1）-> -1（削弱）
        #   已经对 -> 0（不动）
        # ⚠️ 若「对也给 +1 / 错就给 -1」，则「全部不说」也能拿 75% 的分，
        # 梯度会把它推向永远不说（实测：死亡螺旋，入度崩到 5）。
        gate = steady_gate(ema_rate)
        if mode == 'rand_m':
            m_out = np.where(rng.random(N_OUT) < 0.5, M_CORRECT, M_WRONG)
        elif mode == 'no_teach':
            m_out = np.ones(N_OUT, np.float32) * M_CORRECT
        else:
            tgt = np.zeros(N_OUT, np.float32)
            tgt[c] = 1.0
            got = (cnt > 0).astype(np.float32)      # 这个神经元有没有放电
            m_out = np.where(tgt > got, M_CORRECT, np.where(got > tgt, M_WRONG, 0.0))
        m_post = np.zeros(N, np.float32)
        m_post[OUT_IDX] = m_out

        # ---- 结构更新（每个 trial 一次）：Δw = η · e_ij · m[post] ----
        # 只更新「突触后是输出神经元」的边（接口边）——内部边不动，
        # 否则背景强化会把接口边的相对强度涏没（实测：输出边被优先淘汰）。
        if use_plast and key.size:
            el = trace[pre] * trace[post]
            mp = m_post[post.astype(np.int64)]
            sel = mp != 0
            if sel.any() and el[sel].max() > 1e-9:
                el_sel = el[sel] / float(el[sel].max())
                dw = ETA_TEACH * gate * el_sel * mp[sel]
                sl = np.nonzero(sel)[0]
                # 正/负分开：正向不能越过上界，负向不能越过 0（不是抑制）
                pos = w[sl] > 0
                w[sl] = np.where(pos,
                                 np.minimum(w[sl] + dw * (W_MAX_E - w[sl]), W_MAX_E),
                                 w[sl])
                w[sl] = np.maximum(w[sl], 0.0)

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

            if it % 25 == 0:
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
            if key.size > BUDGET:
                d = key.size - BUDGET
                inh_m = pre >= N_E
                scale = np.abs(w) / np.where(inh_m, W_MAX_I, W_MAX_E)
                prot = np.isin(post, OUT_IDX)
                cand = (age >= MIN_AGE) & (~prot)
                if not cand.any():
                    cand[:] = True
                idx_c = np.nonzero(cand)[0]
                d = min(d, idx_c.size)
                if d > 0:
                    sub = idx_c[np.argpartition(scale[idx_c], d - 1)[:d]]
                    mm = np.ones(key.size, bool); mm[sub] = False
                    pre, post, w, key, age = pre[mm], post[mm], w[mm], key[mm], age[mm]
                    drop += int(d)

        if (it + 1) % 200 == 0:
            correct_win.append(n_corr / max(n_tot, 1))
            n_corr = 0
            n_tot = 0

    deg_in = np.bincount(post, minlength=N)
    cr = float(np.mean(conf_right)) if conf_right else 0.0
    cw = float(np.mean(conf_wrong)) if conf_wrong else 0.0
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        win=correct_win,
        acc_first=float(np.mean(correct_win[:2])) if len(correct_win) >= 2 else float('nan'),
        acc_last=float(np.mean(correct_win[-2:])) if len(correct_win) >= 2 else float('nan'),
        conf_right=cr, conf_wrong=cw,
        deg_out=int(deg_in[OUT_IDX].mean()),
        elapsed=time.perf_counter() - t0)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    ntrials = int(argv[0]) if len(argv) > 0 else 6000
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['teach', 'rand_m', 'no_teach', 'frozen']

    for mode in modes:
        print('=' * 126)
        print('grow_teach（阶段 1/2：教说话）  mode=%-9s  N=%d  E0=%d->预算=%d  trials=%d  种子=%d' % (
            mode, N, E0_INIT, BUDGET, ntrials, nseed))
        print('=' * 126)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, ntrials) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %8s %8s %9s %9s %7s' % (
            'seed', 'E', 'rate%', 'acc早', 'acc晚', 'conf对', 'conf错', 'out_deg'))
        for r in res:
            print('%-5d %6d %7.2f %7.1f%% %7.1f%% %9.2f %9.2f %7s' % (
                r['seed'], r['E'], r['rate'] * 100, r['acc_first'] * 100,
                r['acc_last'] * 100, r['conf_right'], r['conf_wrong'],
                '%d' % r['deg_out']))
        A1 = np.array([r['acc_first'] for r in res]) * 100
        A2 = np.array([r['acc_last'] for r in res]) * 100
        W = np.array([r['win'] for r in res])
        with np.errstate(invalid='ignore'):
            wm = np.nanmean(W, axis=0)
        print('  正确率曲线(每200 trial): %s' % ' '.join('%.2f' % x for x in wm))
        print('  早期 %.1f%% -> 晚期 %.1f%%   提升 %+.1f 个百分点   （随机水平 25%%）' % (
            A1.mean(), A2.mean(), A2.mean() - A1.mean()))
        print('  conf 对/错: %.2f / %.2f  （阶段 2「知道自己不会」的雏形：对时更强 = 有区分）' % (
            np.mean([r['conf_right'] for r in res]), np.mean([r['conf_wrong'] for r in res])))
        print('  耗时 %.1fs' % wall)
        print()


if __name__ == '__main__':
    main()
