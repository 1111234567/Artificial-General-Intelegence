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
BLK = 25                          # 每个词占 25 个输入神经元（词 0..3 占 0..99）
N_Q = 10                          # 「提问」信号占 10 个神经元（100..109）
Q_IDX = np.arange(N_IN, N_IN + N_Q)   # 提问区
IN_LO = 0.001                     # 背景噪声（每神经元每步）—— 模拟自发活动
OUT_IDX = np.array([500, 501, 502, 503])
N_OUT = 4
SIDE_INH = -0.5                   # 输出神经元之间的侧抑制（竞争）
# ⚠️ 原值 -1.5 。四个输出神经元同时自发活跃时，侧抑制合起来 -4.5，
# 再叠上阈值适应顶到 2.9 → 输出需要 7.4 的驱动才能响应刺激，而
# 接口边只能给 ~3.0 → 刺激永远驱动不了输出（实测差异范数 0.0073）。
N_OUT_INT = 40                    # 输出神经元从内部网络接多少个
N_OUT_LINK = N_Q + N_OUT_INT      # 每个输出神经元的入边数（提问区 10 + 内部 40）
N_IN_FANOUT = 50                  # 每个输入神经元连多少个下游兴奋神经元

# ---- 时间轴：三段（编码 → 保持 → 提问/回答）----
# 用户 2026-10-24：「一个输入，是需要被记住的，一个刺激信号，告诉网络应该说话了。最后是输出。
#                 现在的学习模式，它根本不知道什么时候该说。」
# 旧形态是「输入 → 干等 → 它想说就说」——网络根本不知道何时该开口，
# 所以延迟全挤在刚允许说的那一刻，那是碰运气。
#
#   t=0        20             DELAY          DELAY+Q        T_MAX
#   |--编码期---|--保持期------|--提问期--------|--回答期-----|
#      词脉冲       什么都不给      提问脉冲        等它说
#
# 保持期长度 = DELAY - BURST，这是真正要扫的记忆跨度（不是「延迟」）。
T_PRESENT = 500                   # 一次呈现的总长（要 > T_MAX）
BURST = 20                        # 内容脉冲窗长
NBURST = 2                        # 内容窗内几次同步爆发
DELAY = 120                       # 提问时刻（= 编码后保持 100 步 = 5 个 τ）
QBURST = 10                       # 提问脉冲窗长
ANSWER_FROM = DELAY               # 提问一出现就可以回答
T_MAX = 300                       # 回答截止（过 = 沉默）
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

# ---- 稳态（§15.2 的两个自组织机制）----
# ① 阈值适应（TH）：放电后阈值上调、按 tau 回落 → 控放电率
TAU_ADAPT = 100.0                 # 阈值回落
D_THETA = 0.20                    # 每次放电阈值上调（→ 自己清理短期）
TH_MIN = 1.0
TH_MAX = 3.0
D_THETA_OUT = 0.02                # 输出神经元的阈值适应幅度（极小）
# ⚠️ 输出神经元不该被阈值锁死：它们是「读出」，不是「自己清理短期」的那批。
# 用 0.20 时阈值很快撞顶 2.9，输出对刺激彻底不敏感。

# ② 抑制性突触短时抑郁（STD）：抑制突触在持续传递后减弱 → 自动 E/I 平衡
# 依据：Phys Rev Research 2, 012042 (2020) —— 这两个机制就能让网络悬在临界点附近，
# 同时给出 E/I 平衡与异步不规则放电，**不需要外部调参**。
# 实测动机：同一网络、同一 seed，只改 m 的符号，rate 就差 6 倍
# （teach 1.66% vs rand_m 0.25%）—— 内部状态被学习信号推着漂，没有自己的工作点。
# 没有稳定工作点，「维持活动」就无从谈起，记忆跨度只能等于 τ。
USE_STD = False                   # 用 --std 开（默认关，留对照）
TAU_REC = 100.0                   # 抑制资源的恢复时间常数（步）
U_STD = 0.30                      # 每次传递耗掉的比例

# ③ 突触缩放（synaptic scaling，`design.md` §8.1）：学习只允许重新分配强度，
#   不允许改变总强度。这是唯一能直接锁住「工作点」的机制。
# 实测动机（2026-10-24）：同一网络、同一 seed，只改 m 的符号，rate 就差 6 倍
# （teach 1.677% / frozen 1.597% / rand_m 0.280% / no_teach 0.295%）。
# 阈值适应只能补偿 3 倍（TH 1.0~3.0），STD 反而把网络推得更开。
# 分区缩放：**接口边与内部边各自保持平均强度**，否则接口边学到的强度
# 会被 15000 条内部边稀释（比例 100:15000）。
USE_SCALE = False                 # 用 --scale 开
SCALE_EVERY = 20                  # 每 N 步缩放一次（与结构更新同周期）


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

    # 接口：输出神经元的驱动**只来自两处**：
    #   ① 提问区（10 个）—— 告诉它「现在该说了」
    #   ② 内部网络（随机 40 个）—— 提供「说什么」（从残留里读）
    # ⚠️ **不接词块**。这不是原来的「拓扑偏置」（那个是「输出 i 只连词 i」，
    # 等于把答案焊上去）。这里是不接——输出完全不看词，只能靠内部网络的残留。
    # 为何必需：接口均匀随机时，词块 25 个神经元在编码期直接驱动输出，
    # 输出被推过阈值 → 抢答率恒为 100%（实测），它无法区分
    # 「输入期」与「提问期」。
    for o in OUT_IDX:
        src_i = rng.choice(np.arange(N_IN + N_Q, N_E),
                           N_OUT_INT, replace=False).astype(np.int32)
        src = np.concatenate([Q_IDX.astype(np.int32), src_i])
        pre = np.concatenate([pre, src])
        post = np.concatenate([post, np.full(src.size, int(o), np.int32)])

    # 接口的另一半：输入区 → 网络本体的扇出。
    # ⚠️ 必需。随机 1% 密度下，100 个输入神经元平均只能碰到 ~100 个下游，
    # 而 w=0.5 需要 2 个同步输入才到阈值——刺激根本传不进去
    # （实测：刺激期 rate 0.051%，全局 0.011%，网络基本是死的）。
    n_src = np.repeat(np.arange(N_IN + N_Q, dtype=np.int32), N_IN_FANOUT)
    n_dst = rng.integers(N_IN + N_Q, N_E, (N_IN + N_Q) * N_IN_FANOUT).astype(np.int32)
    # ⚠️ 两个必须的排除：
    #   ① 从 N_IN 开始：写 0 会把输入区本身也连上 → 反馈 → 输入神经元自发持续放电。
    #   ② 排除 OUT_IDX：500..503 在 110..800 内，不排就会漏到输出上。
    #      实测：词块 25 个神经元同步爆发，通过漏进来的 7 条边直接给输出 I=14，
    #      导致编码期必然抢答。输出的入边只应该是接口那 50 条。
    bad = np.isin(n_dst, OUT_IDX)
    while bad.any():
        n_dst[bad] = rng.integers(N_IN + N_Q, N_E, int(bad.sum())).astype(np.int32)
        bad = np.isin(n_dst, OUT_IDX)
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
    seed, mode, n_present, answer_from, cont_input, use_std, use_scale = args
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
    x_rec = np.ones(N, np.float32)             # 抑制资源（STD 用）
    bias = np.zeros(N, np.float32)             # 节点偏置（可学）
    bias[OUT_IDX] = 0.05                       # 初始：先诱导说

    tref = np.full(N, T_REF, np.int32)
    tref[OUT_IDX] = 4                          # 输出不应期短，否则最大放电率 5%

    m_out = np.zeros(N_OUT, np.float64)        # 奖励信号（衰减）
    # ⚠️ 每个输出神经元一个基线：m = (tgt - got) - baseline。
    # 为何必需：「该说没说 +1 / 不该说却说了 −1」在 got 随机时期望是 −0.25
    # （4 类里 3 类是「不该说」）→ 系统性削弱 → 和突触缩放天然打架
    # （缩放要求净变化为零）。减基线后学习才真正零和。
    bl_out = np.zeros(N_OUT, np.float64)
    BL_OUT_DECAY = float(np.exp(-1.0 / 200.0))
    # 突触缩放的目标：接口边 / 内部边各自的初始平均强度
    grp_face = np.isin(post, OUT_IDX)
    tgt_face = float(np.abs(w[grp_face]).mean()) if grp_face.any() else 1.0
    tgt_int = float(np.abs(w[~grp_face]).mean()) if (~grp_face).any() else 1.0
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
    win_early = []    # 每 200 次呈现：抢答率
    win_lat = []      # 每 200 次呈现：平均延迟
    n_corr = n_tot = n_sil = n_early = 0
    n_burst_fire = 0
    lat_sum = 0
    nsp_tot = 0
    nsp_burst = nsp_tail = 0
    n_burst = n_tail = 0
    # 临界性：分支比 σ = <A(t+1)·A(t)> / <A(t)²>。σ≈1 是临界。
    # 依据：Beggs & Plenz 2003；§15.2 的自组织机制应当把网络留在 σ≈1 附近。
    s_a1a0 = 0.0
    s_a0a0 = 0.0
    a_prev = 0.0
    # rate 稳定性：按窗口记录
    rate_win = []
    nsp_win = 0
    n_win = 0
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

    # 预计算（只在结构变化时刷新）—— 避开每步/每隔几步的 np.isin(15000)
    ist_out = np.isin(post, OUT_IDX)
    is_inh_e = pre >= N_E
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

        # ===== 输入：三段式 =====
        xin = np.zeros(N, np.float32)
        # ① 编码期：词的 25 个神经元放脉冲（这是要被记住的）
        if cont_input:
            # 对照：内容持续到回答期（旧形态）
            if phase < T_MAX:
                base = c * BLK
                xin[base:base + BLK] = IN_SPIKE
        elif phase < BURST and phase in burst_at:
            base = c * BLK
            xin[base:base + BLK] = IN_SPIKE
        # ② 提问期：另一组神经元放脉冲（告诉它「现在说」）
        #    ⚠️ 这是「何时说」的唯一信号。之前没有它，网络只能碰运气。
        if not cont_input and DELAY <= phase < DELAY + QBURST:
            qq = phase - DELAY
            if qq == 0 or qq == QBURST // 2:
                xin[Q_IDX] = IN_SPIKE
        # 背景噪声（两段输入区都给）—— 模拟自发活动
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        xin[Q_IDX] += (rng.random(N_Q) < IN_LO).astype(np.float32) * IN_SPIKE
        xin += bias

        # ===== 网络一步 =====
        # STD：抑制边的有效权重 = w × 可用资源（兴奋边不变）
        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
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

        # STD 资源更新：消耗（放电）与恢复
        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

        nsp = int(s.sum())
        nsp_tot += nsp
        nsp_win += nsp
        n_win += 1
        # 分支比累加（跳过前 20% 的瞬态）
        if t > steps // 5:
            s_a1a0 += float(nsp) * a_prev
            s_a0a0 += a_prev * a_prev
        a_prev = float(nsp)
        if phase < BURST:
            nsp_burst += nsp
            n_burst += 1
        elif phase < BURST + 100:
            nsp_tail += nsp
            n_tail += 1
        ema_rate = 0.99 * ema_rate + 0.01 * (nsp / N)
        trace *= TRACE_DECAY
        trace += s

        # ===== 编码期的放电：不算抢答，但单独计数 =====
        # 为何不算：接口是均匀随机的，词块的 25 个神经元直接连着输出——
        # 强输入一进来就会把输出推过阈值，这是物理必然，不是「学不会何时说」。
        # 实测：把它算成抢答时抢答率恒为 100%。
        # 但这肯定了一个事实：**当前接口让输入直接驱动输出**，所以输出无法区分
        # 「输入期」与「提问期」。要真测「何时说」，接口应该只从提问区 + 内部网络来。
        if (not answered) and phase < BURST:
            if s[OUT_IDX].any():
                n_burst_fire += 1

        # ===== 抢答：编码结束到提问之前开口（独立的错误类型）=====
        # 用户 2026-10-24：「它根本不知道什么时候该说」。
        # 抢答不需要新的学习规则：说的那个 tgt=0/got=1 → 自然吃 −1；
        # 该说的那个 tgt=1/got=0 → 自然吃 +1。但要单独统计，否则会把它
        # 混进「答错」里，看不出「何时说」到底学没学会。
        if (not answered) and BURST <= phase < answer_from:
            of = s[OUT_IDX] > 0
            if of.any():
                cand = np.nonzero(of)[0]
                o = int(cand[np.argmax(v[OUT_IDX][cand])])
                answered = True
                n_early += 1
                n_tot += 1
                lat = -1
                if mode == 'rand_m':
                    m_out += np.where(rng.random(N_OUT) < 0.5, 1.0, -1.0)
                elif mode == 'no_teach':
                    m_out += 1.0
                else:
                    got = np.zeros(N_OUT)
                    got[o] = 1.0
                    tgt = np.zeros(N_OUT)
                    tgt[c] = 1.0
                    raw = tgt - got
                    m_out += np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0))
                    r = -1.0
                    m_glob = (r - baseline) if USE_RPE else r
                    m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                    baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r

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
                    got = np.zeros(N_OUT)
                    got[o] = 1.0
                    tgt = np.zeros(N_OUT)
                    tgt[c] = 1.0
                    raw = tgt - got
                    # ⚠️ 不能用 raw - bl_out：baseline 追踪的是 raw 的均值（≈0.25，因为 tgt 稀疏），
                    # 于是「该说且说了」（raw=0）变成 −0.25 —— 答对的那一个反而被削弱。
                    # 实测：teach 从 75% 掉到 25%。回到二值误差方向；零和问题交给
                    # 「缩放只作用于内部边」去处理（接口边不缩放）。
                    m_out += np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0))
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
                raw = tgt - got
                m_out += np.where(raw > 0, 1.0, 0.0)
                r = -1.0
                m_glob = (r - baseline) if USE_RPE else r
                m_glob = M_UP * m_glob if m_glob > 0 else M_DN * (-m_glob)
                baseline = BL_DECAY * baseline + (1 - BL_DECAY) * r

        # ===== m 衰减（巩固窗口）=====
        m_out *= M_DECAY

        # ===== 每 K 步写一次（不是每 trial）=====
        # ⚠️ 只在「有信号」时写。m 在等待期已衰减到 0，无条件全量更新
        # 会白白花掉 110 us/次 × 1/5 —— 实测这是单进程最大热点（与 bincount 同量级）。
        # 语义上也更正确：没有奖励信号就不该改权重。
        if (t + 1) % K_UPDATE == 0 and t >= WARMUP:
            has_m = bool(np.abs(m_out).max() > 1e-6) or abs(m_glob) > 1e-6
            if use_plast and key.size and has_m:
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
            # ⚠️ 结构可能在上面几步里变了（新建 / 全局新建），缓存必须**无条件**刷新。
            # 放进淘汰分支里会漏：边数变了但未超预算时，ist_out 会对不上长度（IndexError）。
            ist_out = np.isin(post, OUT_IDX)
            is_inh_e = pre >= N_E
            # 突触缩放：**只对内部边**，且**只在过强时压（只压不拉）**。
            # ⚠️ 双向缩放会把权重分布拉平（强边压低、弱边抬高）→ 推离临界：
            # 实测 σ 从 0.92 掉到 0.60，且 teach 从 81.2% 降到 62.5%。
            # 「只压不拉」保留异质性（§2.10：异质性有代价，但也带来函数生成能力）。
            if use_scale:
                mi = ~ist_out
                if mi.any():
                    cur = float(np.abs(w[mi]).mean())
                    if cur > tgt_int * 1.05:
                        w[mi] *= np.float32(tgt_int / cur)
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
                # 结构变了，刷新预计算
                ist_out = np.isin(post, OUT_IDX)
                is_inh_e = pre >= N_E

        # ===== 统计窗口 =====
        if phase == T_PRESENT - 1:
            win.append(n_corr / max(n_tot, 1))
            win_sil.append(n_sil / max(n_tot, 1))
            win_early.append(n_early / max(n_tot, 1))
            win_lat.append(lat_sum / max(n_tot - n_sil - n_early, 1) if n_tot > n_sil + n_early else 0.0)
            n_corr = n_tot = n_sil = n_early = lat_sum = 0
            if n_win > 0:
                rate_win.append(nsp_win / n_win / N)
                nsp_win = 0
                n_win = 0
            if len(win) == max(4, n_present // 4):
                bias_end = bias[OUT_IDX].copy()
            if t > steps - T_PRESENT * 20:
                w_end = np.abs(w[key > 0]).mean() if key.size else 0.0

    deg_in = np.bincount(post, minlength=N)
    bk = np.where(buck_n > 0, buck_ok / np.maximum(buck_n, 1), np.nan)
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        win=win, win_sil=win_sil, win_early=win_early, win_lat=win_lat,
        buck_n=buck_n, buck_acc=bk,
        bias=tuple(round(float(x), 4) for x in bias[OUT_IDX]),
        theta_out=tuple(round(float(x), 3) for x in theta[OUT_IDX]),
        rate_burst=float(nsp_burst / max(n_burst, 1) / N),
        rate_tail=float(nsp_tail / max(n_tail, 1) / N),
        sigma=float(s_a1a0 / s_a0a0) if s_a0a0 > 0 else float('nan'),
        rate_win=rate_win,
        deg_out=int(deg_in[OUT_IDX].mean()),
        acc_first=float(np.mean(win[:2])) if len(win) >= 2 else float('nan'),
        acc_last=float(np.mean(win[-2:])) if len(win) >= 2 else float('nan'),
        sil_first=float(np.mean(win_sil[:2])) if len(win_sil) >= 2 else float('nan'),
        sil_last=float(np.mean(win_sil[-2:])) if len(win_sil) >= 2 else float('nan'),
        early_first=float(np.mean(win_early[:2])) if len(win_early) >= 2 else float('nan'),
        early_last=float(np.mean(win_early[-2:])) if len(win_early) >= 2 else float('nan'),
        burst_fire=n_burst_fire,
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
    global ANSWER_FROM, CONT_INPUT, USE_STD, USE_SCALE
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    for a in sys.argv[1:]:
        if a.startswith('--delay='):
            ANSWER_FROM = int(a.split('=')[1])
        if a == '--cont':
            CONT_INPUT = True
        if a == '--std':
            USE_STD = True
        if a == '--scale':
            USE_SCALE = True
    if '--probe' in sys.argv:
        print('=== 诊断：冻结权重，看输出是否与词相关 ===')
        with Pool(1) as pool:
            pool.map(probe, [(0, 300, ANSWER_FROM, CONT_INPUT)])
        return
    n_present = int(argv[0]) if len(argv) > 0 else 2000
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['teach', 'rand_m', 'no_teach', 'frozen']

    # ⚠️ 一次性提交所有 (mode, seed)：并行度 = nmodes × nseed。
    # 逐 mode 串行时（原写法）并行度只有 nseed，8 核机器上 CPU 只占 nseed/8
    # （实测 nseed=2 → 25%，用户指出）。nseed=8 × 4 mode = 32 任务可跑满。
    tasks = [(s, m, n_present, ANSWER_FROM, CONT_INPUT, USE_STD, USE_SCALE)
             for m in modes for s in range(nseed)]
    print('=' * 132)
    print('grow_talk v2  modes=%s  N=%d  预算=%d  呈现=%d  步=%d  种子=%d  任务=%d' % (
        ','.join(modes), N, BUDGET, n_present, n_present * T_PRESENT, nseed, len(tasks)))
    print('  脉冲窗 %d 步 / 回答窗 %d..%d / TAU_M=%g / K=%d / 阈值适应 %.2f@%g / STD=%s / 缩放=%s' % (
        BURST, ANSWER_FROM, T_MAX, TAU_M, K_UPDATE, D_THETA, TAU_ADAPT,
        USE_STD, USE_SCALE))
    print('=' * 132)
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
        print('-' * 132)
        print('mode=%s   种子=%d' % (mode, len(res)))
        print('%-5s %6s %7s %8s %8s %8s %8s %7s %s' % (
            'seed', 'E', 'rate%', 'acc早', 'acc晚', '抢答早', '抢答晚', 'out_deg', 'bias'))
        for r in res:
            print('%-5d %6d %7.2f %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7d %s' % (
                r['seed'], r['E'], r['rate'] * 100, r['acc_first'] * 100,
                r['acc_last'] * 100, r['early_first'] * 100, r['early_last'] * 100,
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
        print('  正确率(每200呈现,首3末3): %s ... %s' % (
            ' '.join('%.2f' % x for x in wm[:3]), ' '.join('%.2f' % x for x in wm[-3:])))
        print('  沉默率(首3末3)          : %s ... %s' % (
            ' '.join('%.2f' % x for x in sm[:3]), ' '.join('%.2f' % x for x in sm[-3:])))
        print('  ── 延迟-正确率曲线（工作记忆的直接测量）──')
        labels = ['%d-%d' % (ANSWER_FROM + i * 20, ANSWER_FROM + (i + 1) * 20 - 1) for i in range(6)]
        for i in range(6):
            if BN[i] > 0:
                print('    延迟 %-9s n=%-7d 正确率 %.1f%%' % (labels[i], BN[i], 100.0 * BO[i] / BN[i]))
            else:
                print('    延迟 %-9s n=0' % labels[i])
        A1 = np.array([r['acc_first'] for r in res]) * 100
        A2 = np.array([r['acc_last'] for r in res]) * 100
        E1 = np.array([r['early_first'] for r in res]) * 100
        E2 = np.array([r['early_last'] for r in res]) * 100
        S2 = np.array([r['sil_last'] for r in res]) * 100
        print('  早期 %.1f%% -> 晚期 %.1f%%   提升 %+.1f pp   （随机 = 25%%）' % (
            A1.mean(), A2.mean(), A2.mean() - A1.mean()))
        print('  抢答率 %.1f%% -> %.1f%%   沉默率(晚) %.1f%%   （抢答下降 = 学会了「何时说」）' % (
            E1.mean(), E2.mean(), S2.mean()))
        print('  rate: 刺激期 %.3f%%  刺激后100步 %.3f%%  （全局 %.3f%%）' % (
            np.mean([r['rate_burst'] for r in res]) * 100,
            np.mean([r['rate_tail'] for r in res]) * 100,
            np.mean([r['rate'] for r in res]) * 100))
        SIG = np.array([r['sigma'] for r in res])
        print('  分支比 σ = %.3f ± %.3f   （≈1 = 临界）' % (np.nanmean(SIG), np.nanstd(SIG)))
        cvs = []
        for r in res:
            a = np.array(r['rate_win'])
            if a.size >= 6 and a.mean() > 0:
                tail_ = a[-max(4, a.size // 3):]
                cvs.append(tail_.std() / tail_.mean())
        if cvs:
            print('  rate 稳定性（末 1/3 的变异系数）= %.1f%%' % (np.mean(cvs) * 100))

    if len(by) > 1:
        print('')
        print('=' * 132)
        print('跨 mode 工作点一致性（这是「稳态」的判据）')
        print('%-10s %10s %10s %10s' % ('mode', 'rate%', 'σ', 'acc晚'))
        rr = []
        for mode in modes:
            res = by.get(mode, [])
            if not res:
                continue
            r_ = float(np.mean([x['rate'] for x in res])) * 100
            sg = float(np.nanmean([x['sigma'] for x in res]))
            ac = float(np.mean([x['acc_last'] for x in res])) * 100
            rr.append(r_)
            print('%-10s %10.3f %10.3f %9.1f%%' % (mode, r_, sg, ac))
        if rr:
            print('  rate 极差 / 均值 = %.2f   （1.0 = 完全一致；原基线是 6.0）' %
                  (max(rr) / max(min(rr), 1e-9)))

    print('')
    print('  总耗时 %.1fs' % wall)


if __name__ == '__main__':
    main()
