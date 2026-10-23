# grow_talk.py - 阶段 1/2（v2）：脉冲刺激 + 事件发射 + 连续时间轴
#
# 与 grow_teach.py 的三个结构性区别（用户 2026-10-24 指出）：
#   ① 刺激是**脉冲**，不是持续速率编码。一次呈现 = 10 步内 25 个脉冲，然后消失。
#      → 网络必须在刺激到达时接住、靠内部状态维持、以后再回答（工作记忆）。
#      旧形态：输入一直亮着 → 任何时候抬头就能看到答案 → 内部状态完全没用。
#   ② 状态**永不清空**（v / s / ref / trace / theta 全部跨呈现连续）。
#      → 「清理短期记忆」由阈值适应（theta 上调 + 回落）自己完成，不是我手动清。
#   ③ **发射是事件**：第一个放电的输出神经元就是它说的词（first-spike），
#      不是「固定 20 步窗口的 argmax」。说得越晚，状态衰减越多——**代价是物理的**，
#      不是我加的惩罚项。
#
# 设计依据：
#   design.md §11.4 阶段 B：一条不间断的时间轴 / 连续事件流 / 不强制输出
#   design.md §14.6：奖励是事件不是流；五个作用点（本轮用三个：读出阈值 / 节点偏置 / 连接权重）
#   design.md §2.10：时间只存在两处——状态衰减、巩固窗口
#   experiments.md §11.4.0：输入必须带结构（偶发事件 = 脉冲）
#
# 四组对照：teach / rand_m / no_teach / frozen

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

TRACE_DECAY = 0.95
P_NEW = 0.004
P_GLOBAL = 4
MAX_ACTIVE = 48
MIN_AGE = 25

# ---- 接口 ----
N_IN = 100
N_WORD = 4
IN_SPIKE = 3.0                    # 输入脉冲幅度
# ⚠️ 必须 > V_TH（1.0）。写 1.0 时 `v > theta` 不成立——输入神经元根本不放电，
# 刺激从没进过网络（实测：刺激期 rate 与刺激后无差别，1.65% vs 1.51%）。
# 而「持续输入」对照能学到，靠的是额外噪声偶尔把 xin 推到 2.0 —— 差异不在
# 「脉冲 vs 持续」，而在「输入神经元放不放电」。
BLK = 25                          # 每个词占 25 个输入神经元
IN_LO = 0.001                     # 背景噪声（每神经元每步）
OUT_IDX = np.array([500, 501, 502, 503])
N_OUT = 4
SIDE_INH = -0.5                   # 输出神经元之间的侧抑制（竞争）
# ⚠️ 原值 -1.5 。四个输出神经元同时自发活跃时，侧抑制合起来 -4.5，
# 再叠上阈值适应顶到 2.9 → 输出需要 7.4 的驱动才能响应刺激，而
# 接口边只能给 ~3.0 → 刺激永远驱动不了输出（实测差异范数 0.0073）。
N_OUT_LINK = 25                   # 每个输出神经元的入边数 = 本块全连（BLK=25）
# ⚠️ 不给「其他块」的边。原值 60（本块 25 + 其他 35）时区分度被抹平：
# I[out] = [8.89, 6.45, 3.96, 5.53]，四个都在阈值 1.0 以上 → 四个词下四个输出全放。
# 降到 5 条其他块后 I = [9.58, 2.08, 0.50, 1.05] —— 仍有两个超过阈值。
# 接口的语义就是「哪些节点读」：读自己的块就够，跨块的泛化交给学习。
N_IN_FANOUT = 50                  # 每个输入神经元连多少个下游兴奋神经元

# ---- 时间轴（一条连续的，没有 trial）----
T_PRESENT = 300                   # 教师每 300 步发一次刺激
BURST = 20                        # 脉冲窗长（刺激只在这 20 步里）
NBURST = 2                        # 脉冲窗内几次同步爆发
REPEAT = 1                        # （保留位）每个神经元每次爆发放几次
ANSWER_FROM = 100                 # 回答开端（强制延迟期：必须跨过 100-20=80 步空窗）
T_MAX = 200                       # 回答截止（刺激后 200 步内没说 = 沉默）
COOL = 30                         # 说完后的冷却（防止重复回答）

# ---- 学习 ----
ETA_W = 0.008                     # 权重学习率
ETA_BIAS = 0.004                  # 偏置学习率（慢——它控制行为频率，§14.6 作用点二）
BIAS_MAX = 0.40
BIAS_MIN = -0.10
TAU_M = 100.0                     # 奖励信号的衰减（= 巩固窗口，不是 trial 边界）
K_UPDATE = 5                      # 每 K 步写一次（不是每 trial）
M_DECAY = float(np.exp(-1.0 / TAU_M))
BL_TAU = 40.0                     # RPE 基线的时间常数（奖励预期）
BL_DECAY = float(np.exp(-1.0 / BL_TAU))
USE_RPE = True                    # 用 r - baseline 而不是 r（否则基线 25% 时会系统性削弱）
CONT_INPUT = False                # 对照：True = 刺激持续到回答期（速率编码，旧形态）
CONT_INPUT = False                # 对照：True = 刺激持续到回答期（速率编码，旧形态）
M_UP = 1.0                        # 正信号幅度
M_DN = -0.5                       # 负信号幅度（不对称：削弱只有一半，防早期死亡螺旋）
WARMUP = 3000                     # 前 N 步不开可塑性（让 RPE 基线先收敛）

# ---- 稳态（§15.2：自组织稳态之一）----
TAU_ADAPT = 100.0                 # 阈值回落
D_THETA = 0.20                    # 每次放电阈值上调（→ 自己清理短期）
TH_MIN = 1.0
TH_MAX = 3.0
D_THETA_OUT = 0.02                # 输出神经元的阈值适应幅度（极小）
# ⚠️ 输出神经元不该被阈值锁死：它们是「读出」，不是「自己清理短期」的那批。
# 用 0.20 时阈值很快撞顶 2.9，输出对刺激彻底不敏感。


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
    """随机结构 + 接口（输入区 → 输出神经元直连）+ 输出侧抑制。"""
    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    # ⚠️ 两条过滤都必需：
    #   ① post >= N_IN ：输入区是「感觉器官」，不接受网络内部的反向边。
    #      否则随机网络里的抑制神经元直接压死输入神经元（实测：25 个里只有 3.9 个能放）。
    #   ② ~isin(post, OUT_IDX) ：输出神经元的入边只应该是接口，不该混进随机边。
    #      否则随机边（权重 0.5）比接口边（0.385）还大，稀释信号（实测：接口边 359 条而非 240）。
    ok = (pre != post) & (post >= N_IN) & (~np.isin(post, OUT_IDX))
    pre, post = pre[ok], post[ok]

    # 接口：每个输出神经元接一批输入神经元。
    # ⚠️ 必须带拓扑偏置：2/3 来自「自己的块」，1/3 来自全输入区。
    # 均匀随机会让每个输出神经元从 4 个块各收 ~60 条边（实测 [61,69,59,62]），
    # 四词对输出完全对称 → 差异范数 0.0077，学习信号无处可施。
    n_own = min(BLK, N_OUT_LINK)      # 本块全连（BLK=25 = N_OUT_LINK）
    n_other = N_OUT_LINK - n_own
    for i, o in enumerate(OUT_IDX):
        own = rng.choice(BLK, n_own, replace=False).astype(np.int32) + i * BLK
        others = np.setdiff1d(np.arange(N_IN, dtype=np.int32),
                              np.arange(i * BLK, (i + 1) * BLK, dtype=np.int32))
        rest = rng.choice(others, N_OUT_LINK - n_own, replace=False).astype(np.int32)
        src = np.concatenate([own, rest])
        pre = np.concatenate([pre, src])
        post = np.concatenate([post, np.full(src.size, int(o), np.int32)])

    # 接口的另一半：输入区 → 网络本体的扇出。
    # ⚠️ 必需。随机 1% 密度下，100 个输入神经元平均只能碰到 ~100 个下游，
    # 而 w=0.5 需要 2 个同步输入才到阈值——刺激根本传不进去
    # （实测：刺激期 rate 0.051%，全局 0.011%，网络基本是死的）。
    n_src = np.repeat(np.arange(N_IN, dtype=np.int32), N_IN_FANOUT)
    n_dst = rng.integers(N_IN, N_E, N_IN * N_IN_FANOUT).astype(np.int32)
    # ⚠️ 从 N_IN 开始：写 0 会把输入区本身也连上 → 反馈 → 输入神经元自发持续放电
    # （实测：658 条指向输入区的边）。
    pre = np.concatenate([pre, n_src])
    post = np.concatenate([post, n_dst])

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
    nfan = N_IN * N_IN_FANOUT
    n_rand = pre.size - nlink - nfan - nside      # 随机初始边的条数
    # ⚠️ 偏移必须从尾往前算：加入「输入扇出」后，输出直连不再是最后一段。
    # 写错位置会让接口边拿不到权重、反而把扇出边改小（实测：差异范数 0.0073）。
    w[n_rand:n_rand + nlink] = rng.uniform(0.15, 0.60, nlink).astype(np.float32)
    w[pre.size - nside:] = SIDE_INH
    age = np.full(pre.size, 100, np.int16)
    return pre, post, w, age


def run_one(args):
    # ⚠️ 参数必须通过 args 传。Windows 是 spawn 模式：子进程会重新 import 模块，
    # 父进程对模块级变量的修改（--delay / --cont）**子进程看不到**。
    # 实测代价：--delay=20 与 --cont 两次对照跑的是同一份代码（逐字相同的输出）。
    seed, mode, n_present, answer_from, cont_input = args
    rng = np.random.default_rng(seed)

    pre, post, w, age = _init_net(rng)
    pre, post, w, key, age = _sort(pre, post, w, age)

    steps = n_present * T_PRESENT

    # ---- 状态：永不清空 ----
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    trace = np.zeros(N, np.float32)
    theta = np.full(N, V_TH, np.float32)       # 阈值（可适应）
    th_ref = np.full(N, V_TH, np.float32)      # 回落目标
    th_ref[OUT_IDX] = V_TH
    bias = np.zeros(N, np.float32)             # 节点偏置（可学）
    bias[OUT_IDX] = 0.05                       # 初始：先诱导说

    tref = np.full(N, T_REF, np.int32)
    tref[OUT_IDX] = 4                          # 输出不应期短，否则最大放电率 5%

    m_out = np.zeros(N_OUT, np.float64)        # 奖励信号（衰减）
    m_glob = 0.0                               # 全局调制（三因子里的 m，给内部边用）
    baseline = 0.0                             # RPE 基线（奖励预期）

    use_plast = mode != 'frozen'
    use_bias = mode != 'frozen'

    # ---- 统计 ----
    nb = 6
    buck_n = np.zeros(nb, np.int64)
    buck_ok = np.zeros(nb, np.int64)
    win = []          # 每 200 次呈现：正确率
    win_sil = []      # 每 200 次呈现：沉默率
    win_lat = []      # 每 200 次呈现：平均延迟
    n_corr = n_tot = n_sil = 0
    lat_sum = 0
    nsp_tot = 0
    nsp_burst = nsp_tail = 0
    n_burst = n_tail = 0
    ema_rate = 0.02
    born = drop = 0
    bias_end = None
    w_end = None

    # 本呈现的状态
    c = 0
    answered = False
    lat = 0
    burst_at = np.zeros(BLK, np.int32)
    burst_dirty = True

    t0 = time.perf_counter()

    for t in range(steps):
        phase = t % T_PRESENT

        # ===== 教师发刺激（不通知网络：无 episode 信号）=====
        if phase == 0:
            c = int(rng.integers(0, N_WORD))
            answered = False
            lat = 0
            burst_at = np.zeros(NBURST, np.int32)
            if NBURST > 1:
                burst_at = np.linspace(0, BURST - 2, NBURST).astype(np.int32)

        # ===== 输入：脉冲窗里才亮，之后只剩背景 =====
        xin = np.zeros(N, np.float32)
        if cont_input:
            # 对照：刺激持续到回答期（旧形态）。用来把「脉冲 vs 持续」单独拿出来。
            if phase < T_MAX:
                base = c * BLK
                xin[base:base + BLK] = IN_SPIKE
        elif phase < BURST and phase in burst_at:
            base = c * BLK
            xin[base:base + BLK] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        # ===== 网络一步 =====
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = tref[s > 0]

        # 阈值适应：放过的上调，全体向 th_ref 回落（自己清理短期记忆）
        # 输出神经元用极小的适应幅度（它们是读出，不该被阈值锁死）
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
            ob = fire[OUT_IDX]
            if ob.any():
                oi = OUT_IDX[ob]
                theta[oi] = np.maximum(theta[oi] - D_THETA + D_THETA_OUT, TH_MIN)
        theta += (th_ref - theta) / TAU_ADAPT

        nsp = int(s.sum())
        nsp_tot += nsp
        if phase < BURST:
            nsp_burst += nsp
            n_burst += 1
        elif phase < BURST + 100:
            nsp_tail += nsp
            n_tail += 1
        ema_rate = 0.99 * ema_rate + 0.01 * (nsp / N)
        trace *= TRACE_DECAY
        trace += s

        # ===== 发射事件（first-spike）=====
        if not answered and answer_from <= phase < T_MAX:
            of = s[OUT_IDX] > 0
            if of.any():
                # 多个同时放电时取膜电位最高的
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[OUT_IDX][cand])])
                word = o
                answered = True
                lat = phase
                ok = (word == c)
                n_tot += 1
                if ok:
                    n_corr += 1
                lat_sum += lat
                b = min(nb - 1, max(0, (lat - answer_from) // 20))
                buck_n[b] += 1
                if ok:
                    buck_ok[b] += 1
                if mode == 'rand_m':
                    m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
                elif mode == 'no_teach':
                    m_out += 1.0
                else:
                    # 每个输出神经元各自判断「该说 vs 实际说了」：
                    #   tgt[c]=1, 其余 0； got=这个神经元这次放没放电
                    #   该说没说 -> +1（强化）；不该说却说了 -> -1（削弱）；已对 -> 0
                    got = np.zeros(N_OUT)
                    got[o] = 1.0
                    tgt = np.zeros(N_OUT)
                    tgt[c] = 1.0
                    m_out += np.where(tgt > got, 1.0, np.where(got > tgt, -1.0, 0.0))
                    # 全局调制（给内部边）：RPE = r - baseline。
                    # ⚠️ 直接用 r 会在正确率 25% 时系统性削弱（均值 -0.5）→ 死亡螺旋。
                    r = 1.0 if ok else -1.0
                    m_glob = (r - baseline) if USE_RPE else r
                    m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                    baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r

        # ===== 沉默也是一次作答 =====
        if (not answered) and phase == T_MAX - 1:
            answered = True
            n_tot += 1
            n_sil += 1
            if mode == 'rand_m':
                m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
            elif mode == 'no_teach':
                m_out += 1.0
            else:
                # 沉默 = 一个都没说 → 只有「该说的那个」挨罚，而且方向是
                # **提高它的兴奋性**（+1），不是降低。
                # ⚠️ 写 -1 会让偏置无界下压（实测：-5.7 → 永远沉默的死网）。
                got = np.zeros(N_OUT)
                tgt = np.zeros(N_OUT)
                tgt[c] = 1.0
                m_out += np.where(tgt > got, 1.0, 0.0)
                r = -1.0
                m_glob = (r - baseline) if USE_RPE else r
                m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r

        # ===== m 衰减（巩固窗口）=====
        m_out *= M_DECAY

        # ===== 每 K 步写一次（不是每 trial）=====
        if (t + 1) % K_UPDATE == 0 and t >= WARMUP:
            if use_plast and key.size:
                gate = steady_gate(ema_rate)
                el = trace[pre] * trace[post]
                # 三因子：Δw = η · e_ij · m
                #   e_ij = 资格迹（区分「谁活跃过」）→ 提供选择性
                #   m    = 调制（区分「这次结果好不好」）→ 提供方向
                # 内部边：用全局 m（输出层 m 传不进去，那是信用分配问题，
                #          三因子规则本身就是粗糙归因）；
                # 输出边：用自己的 m_out（可以逐神经元算）
                mp = np.zeros(N, np.float32)
                mp[OUT_IDX] = m_out.astype(np.float32)
                sel = mp[post.astype(np.int64)]
                if abs(m_glob) > 1e-9:
                    ist_out = np.isin(post, OUT_IDX)
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
                # ⚠️ 花式索引返回的是**副本**：bias[OUT_IDX] += x 不会写回，
                # clip(out=bias[OUT_IDX]) 也不生效（实测：偏置无界跌到 -5.7，网络死）。
                nb_ = bias[OUT_IDX] + ETA_BIAS * m_out.astype(np.float32)
                bias[OUT_IDX] = np.clip(nb_, BIAS_MIN, BIAS_MAX)

        # ===== 结构更新（每 20 步，保留生长/淘汰）=====
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
            if key.size > BUDGET:
                d = key.size - BUDGET
                inh_m = pre >= N_E
                scale = np.abs(w) / np.where(inh_m, W_MAX_I, W_MAX_E)
                prot = np.isin(post, OUT_IDX)
                cand = (age >= MIN_AGE) & (~prot)
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

        # ===== 统计窗口 =====
        if phase == T_PRESENT - 1:
            win.append(n_corr / max(n_tot, 1))
            win_sil.append(n_sil / max(n_tot, 1))
            win_lat.append(lat_sum / max(n_tot - n_sil, 1) if n_tot > n_sil else 0.0)
            n_corr = n_tot = n_sil = lat_sum = 0
            if len(win) == max(4, n_present // 4):
                bias_end = bias[OUT_IDX].copy()
            if t > steps - T_PRESENT * 20:
                w_end = np.abs(w[key > 0]).mean() if key.size else 0.0

    deg_in = np.bincount(post, minlength=N)
    bk = np.where(buck_n > 0, buck_ok / np.maximum(buck_n, 1), np.nan)
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        win=win, win_sil=win_sil, win_lat=win_lat,
        buck_n=buck_n, buck_acc=bk,
        bias=tuple(round(float(x), 4) for x in bias[OUT_IDX]),
        theta_out=tuple(round(float(x), 3) for x in theta[OUT_IDX]),
        rate_burst=float(nsp_burst / max(n_burst, 1) / N),
        rate_tail=float(nsp_tail / max(n_tail, 1) / N),
        deg_out=int(deg_in[OUT_IDX].mean()),
        acc_first=float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        acc_last=float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        sil_first=float(np.mean(win_sil[:2])) if len(win_sil) >= 2 else float('nan'),
        sil_last=float(np.mean(win_sil[-2:])) if len(win_sil) >= 2 else float('nan'),
        elapsed=time.perf_counter() - t0)


def probe(args):
    """诊断：冻结权重，四个词各跑若干次，看输出神经元放电是否与词相关。

    回答一个问题：「刺激到底有没有传到输出？」
    不相关 -> 通路断了（信号强度 / 输入扇出 / 噪声淹没），不是学习规则的事。
    """
    seed, n_present, answer_from, cont_input = args
    rng = np.random.default_rng(seed)
    pre, post, w, age = _init_net(rng)
    pre, post, w, key, age = _sort(pre, post, w, age)

    v = np.zeros(N, np.float32); s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32); trace = np.zeros(N, np.float32)
    theta = np.full(N, V_TH, np.float32)
    bias = np.zeros(N, np.float32); bias[OUT_IDX] = 0.05
    tref = np.full(N, T_REF, np.int32); tref[OUT_IDX] = 4

    acc = np.zeros((N_WORD, N_OUT), np.float64)
    cnt = np.zeros(N_WORD, np.int64)
    fire_cnt = np.zeros(N_WORD, np.float64)
    step_cnt = np.zeros(N_WORD, np.int64)
    in_fire = 0.0
    in_step = 0
    drv = np.zeros(8, np.float64)

    T = n_present * T_PRESENT
    c = 0; burst_at = np.zeros(NBURST, np.int32)
    for t in range(T):
        phase = t % T_PRESENT
        if phase == 0:
            c = int(rng.integers(0, N_WORD))
            burst_at = np.linspace(0, BURST - 2, NBURST).astype(np.int32) if NBURST > 1 else np.zeros(1, np.int32)
        xin = np.zeros(N, np.float32)
        if cont_input:
            if phase < T_MAX:
                xin[c * BLK:(c + 1) * BLK] = IN_SPIKE
        elif phase < BURST and phase in burst_at:
            xin[c * BLK:(c + 1) * BLK] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = tref[s > 0]
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, TH_MAX)
            ob = fire[OUT_IDX]
            if ob.any():
                oi = OUT_IDX[ob]
                theta[oi] = np.maximum(theta[oi] - D_THETA + D_THETA_OUT, TH_MIN)
        theta += (V_TH - theta) / TAU_ADAPT
        trace *= TRACE_DECAY; trace += s
        if answer_from <= phase < T_MAX:
            acc[c] += s[OUT_IDX]
            fire_cnt[c] += float(s.sum())
            step_cnt[c] += 1
            cnt[c] += 1
        # 细粒度：输入区在刺激期的放电
        if phase < BURST:
            in_fire += float(s[0:N_IN].sum())
            in_step += 1
        # 输出神经元在刺激前几步的驱动
        if phase in (0, 1, 2):
            drv[phase] += float(I[OUT_IDX[0]])
            drv[3] += float(xin[OUT_IDX[0]])
            drv[4] += float(v[OUT_IDX[0]])
            drv[5] += 1.0

    print('%-6s %12s %12s %12s %12s %10s' % ('word', 'out0', 'out1', 'out2', 'out3', 'net_rate%'))
    for i in range(N_WORD):
        d = max(cnt[i], 1)
        print('%-6d %12.3f %12.3f %12.3f %12.3f %10.3f' % (
            i, acc[i, 0] / d, acc[i, 1] / d, acc[i, 2] / d, acc[i, 3] / d,
            100.0 * fire_cnt[i] / max(step_cnt[i], 1) / N))
    M = acc / np.maximum(cnt, 1)[:, None]
    print('  四词输出均值矩阵的差异范数 %.4f  （< 0.5 = 输出对词不敏感）' %
          float(np.linalg.norm(M - M.mean(axis=0))))
    print('  刺激期输入区放电 %.2f 个/步（期望 ~25；为 0 则刺激没进网络）' %
          (in_fire / max(in_step, 1)))
    ns = max(drv[5], 1.0)
    print('  out0 在 phase 0/1/2: I=%.2f  xin=%.2f  v=%.2f  （驱动到不了 1.0 就不放）' %
          (drv[0] / ns, drv[3] / ns, drv[4] / ns))


def main():
    global ANSWER_FROM, CONT_INPUT
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    for a in sys.argv[1:]:
        if a.startswith('--delay='):
            ANSWER_FROM = int(a.split('=')[1])
        if a == '--cont':
            CONT_INPUT = True
    if '--probe' in sys.argv:
        print('=== 诊断：冻结权重，看输出是否与词相关 ===')
        with Pool(1) as pool:
            pool.map(probe, [(0, 300, ANSWER_FROM, CONT_INPUT)])
        return
    n_present = int(argv[0]) if len(argv) > 0 else 2000
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['teach', 'rand_m', 'no_teach', 'frozen']

    for mode in modes:
        print('=' * 132)
        print('grow_talk v2  mode=%-9s  N=%d  预算=%d  呈现=%d  步=%d  种子=%d' % (
            mode, N, BUDGET, n_present, n_present * T_PRESENT, nseed))
        print('  脉冲窗 %d 步 / 回答窗 %d..%d / TAU_M=%g / K=%d / 阈值适应 %.2f@%g' % (
            BURST, ANSWER_FROM, T_MAX, TAU_M, K_UPDATE, D_THETA, TAU_ADAPT))
        print('=' * 132)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, n_present, ANSWER_FROM, CONT_INPUT) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %8s %8s %8s %8s %7s %s' % (
            'seed', 'E', 'rate%', 'acc早', 'acc晚', '沉默早', '沉默晚', 'out_deg', 'bias'))
        for r in res:
            print('%-5d %6d %7.2f %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7d %s' % (
                r['seed'], r['E'], r['rate'] * 100, r['acc_first'] * 100,
                r['acc_last'] * 100, r['sil_first'] * 100, r['sil_last'] * 100,
                r['deg_out'], r['bias']))

        W = np.array([r['win'] for r in res])
        S = np.array([r['win_sil'] for r in res])
        L = np.array([r['win_lat'] for r in res])
        BN = np.array([r['buck_n'] for r in res]).sum(axis=0)
        BO = np.array([[x * n for x, n in zip(r['buck_acc'], r['buck_n'])] for r in res]).sum(axis=0)
        with np.errstate(invalid='ignore'):
            wm = np.nanmean(W, axis=0)
            sm = np.nanmean(S, axis=0)
            lm = np.nanmean(L, axis=0)
        print('  正确率(每200呈现): %s' % ' '.join('%.2f' % x for x in list(wm[:3]) + list(wm[-3:])))
        print('  沉默率(每200呈现): %s' % ' '.join('%.2f' % x for x in list(sm[:3]) + list(sm[-3:])))
        print('  平均延迟        : %s' % ' '.join('%.0f' % x for x in list(lm[:3]) + list(lm[-3:])))
        print('  ── 延迟-正确率曲线（这是「工作记忆」的直接测量）──')
        labels = ['%d-%d' % (ANSWER_FROM + i * 20, ANSWER_FROM + (i + 1) * 20 - 1) for i in range(6)]
        for i in range(6):
            if BN[i] > 0:
                print('    延迟 %-9s n=%-7d 正确率 %.1f%%' % (labels[i], BN[i], 100.0 * BO[i] / BN[i]))
            else:
                print('    延迟 %-9s n=0' % labels[i])
        A1 = np.array([r['acc_first'] for r in res]) * 100
        A2 = np.array([r['acc_last'] for r in res]) * 100
        print('  早期 %.1f%% -> 晚期 %.1f%%   提升 %+.1f pp   （随机 = 25%%）' % (
            A1.mean(), A2.mean(), A2.mean() - A1.mean()))
        print('  偏置终值: %s   阈值: %s' % (
            res[0]['bias'], res[0]['theta_out']))
        print('  rate: 刺激期 %.3f%%  刺激后100步 %.3f%%  （全局 %.3f%%）' % (
            np.mean([r['rate_burst'] for r in res]) * 100,
            np.mean([r['rate_tail'] for r in res]) * 100,
            np.mean([r['rate'] for r in res]) * 100))
        print('  耗时 %.1fs' % wall)
        print()


if __name__ == '__main__':
    main()
