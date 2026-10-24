# grow_agent.py - 智能体：本机真实环境 + 三闸 + 隔离 + 四层反馈 + 四度量
#
# 本文件是为了把 `memory/diary/2026-10-24.md` 场次四提出的 **12 条偏离**全部处理而写的。
# 每条偏离 -> 本文件里对应的实现（逐条可核）：
#
#   ① 不发明新参数（§0.3-5）       -> 全部常数从 net.py 取并带 [来源] 标签；本文件新增的逐个登记在 CONSTANTS.md
#   ② 三因子公式缺 (a_post−θ)      -> Δw = η·m·trace[pre]·(trace[post]−θ)
#   ③ 门控只需三个、能手推（§7.1） -> 准入/归因/裁决，三个 0~1 标量相乘（见 `m_gate()`）
#   ④ 多吸引子当目标               -> 不测它（已是作废项）
#   ⑤ 更新频率 = 反馈频率（§5.4） -> 结构更新由**次数**触发（每 episode 一次 = 每任务一次反馈），
#                                    迹 τ 跨过反馈延迟；五个时间常数减到三个，且每个能说出对应的反馈
#   ⑥ 三要素：结构/压力/隔离（§9.2） -> 压力 = 本机真实文件后果；**隔离 = 只写入本 episode 活跃过的边**
#   ⑦ 环境是锚不是梯度（§14.8）     -> 环境层只给「成/不成 + 真实值」，不当梯度；内部层只填空隙
#   ⑧ 四层反馈（§3.3）              -> 环境/参照/内部/过程，参照层显式标「未实现」并说明为什么
#   ⑨ 过程四可测量（§4.2）          -> 弯路率 / 修正延迟 / 依据密度 / 遇意外反应，四个都算
#   ⑩ 长期记忆只有次数没有时间（§2.10）-> `T_S`（每 20 步）删除；结构更新 = 每个 episode 一次
#   ⑪ 两类可塑性是不是同一套规则两端 -> `--plast=w / topo / both` 三组对照
#   ⑫ 「短期」两个含义（§2.10）      -> episode 之间插入 `T_GAP` 步无关输入，逼退隐状态（τ=20），
#                                    目标持续 `K_EP_TASK` 个 episode -> 记住它只能靠连接
#
# **环境是真的：** 观测来自 `data/env/<seed>/state.txt` 的真实字节，动作是真实写文件（原子写 tmp+replace），
# 成败由真实读回的值判定。可逆：目录可删、可重跑。不是 2D 数学世界。
#
# 接口换成 4 输出 one-hot + 侧抑制（`grow_talk` 的形态，§16.10 已证明它能学 75%）。
# 理由：§16.13 的探针证明「信息在状态里（符号准确率 73.5%）但 2 输符号接口表达不出」。
#
# 对照（mode）：
#   base       全开
#   frozen     不学（结构冻结）
#   no_gate    m 恒等于稳态门（去掉准入/归因/裁决）
#   no_bridge  去掉桥接刺激（只靠成败）
#   no_isolation 全局更新（不限制在本 episode 活跃过的边）
#   plast_w    只有强度可塑（拓扑冻结）
#   plast_topo 只有拓扑（权重固定）
#   recall     跨 episode 回忆：目标不可观测，要求回到上一 episode 的终点

import io
import os
import shutil
import sys
import time
import numpy as np
from multiprocessing import Pool

from net import (N, N_E, N_IN, DECAY_V, V_TH, T_REF,
                 W_E, W_I_RATIO, W_MAX_E, W_MAX_I, W_NEW_E, W_NEW_I,
                 BUDGET, E0_INIT, TRACE_DECAY, ETA_POT, P_NEW,
                 MAX_ACTIVE, MIN_AGE, _sort, find, steady_gate)

ENV_ROOT = r"F:\agi\data\env"

# ---- 接口（4 输出 one-hot + 侧抑制）----
# [需求] 4 个动作 = 4 个输出神经元；§16.10 已证明这个形态能学
N_OUT = 4
OUT_IDX = np.arange(500, 504)
N_CH = N_IN // 4                 # [派生] 4 通道 × 25：state / target / （2 个备用通道）
IN_RATE = 0.02                   # [拍] 输入放电率基准
GAIN = 0.8                       # [拍] 观测 -> 输入率的调制深度
SIDE_INH = 1.5                   # [拍] 输出间侧抑制（grow_talk 的值）
T_REF_OUT = 4                    # [需求] 输出神经元的不应期（§16.10 缺陷 6）：
#                                  用 T_REF=20 时最大放电率 5%，而动作窗口才 20 步 ->
#                                  窗口内期望 0.3 次放电 -> argmax 就是噪声，选动作纯随机。
A_TAU = 20.0                     # [拍] 输出放电率 EMA（对应步级反馈）

# ---- 环境（本机真实文件）----
T_ACT = 20                       # [需求] 动作窗口（§14.6：行为是一个事件窗口）
T_EP = 120                       # [需求] 每个 episode 的步数
T_GAP = 200                      # [需求] episode 间间隙：逼退隐状态（τ_v=20），迫使依赖连接
K_EP_TASK = 5                    # [需求] 同一目标持续的 episode 数
V_MAX = 100                      # [需求] 值的取值范围 [0, V_MAX]
OBS_LO, OBS_HI = 10, 90          # [需求] state / target 的采样区间
ACTIONS = np.array([1, -1, 10, -10], np.int64)   # [需求] 4 个离散动作

# ---- m 的三闸（§7.1）----
SURPRISE_K = 10.0                # [拍] 归因门的尺度：预测误差 / K，clip 到 [0,1]
BRIDGE = 0.15                    # [拍] 桥接强度（中性事件 = 值发生变化）

# ---- 隔离 ----
ISOLATE_THRESH = 0.3             # [拍] 「本 episode 活跃过」的判定阈值（迹）


def _env_path(d, name):
    return os.path.join(d, name)


def env_write(d, name, v):
    """原子写（tmp + replace）——真实文件系统。"""
    p = _env_path(d, name)
    tmp = p + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(v)))
    os.replace(tmp, p)


def env_read(d, name):
    """真读回。"""
    with io.open(_env_path(d, name), "r", encoding="utf-8") as f:
        return int(f.read().strip())


def obs_code(v):
    """把 [0, V_MAX] 的值归一化到 [-1, 1]（观测编码）。"""
    return float(v) / (V_MAX * 0.5) - 1.0


class Env(object):
    """本机真实环境：一个目录 + state.txt / target.txt。

    只做三件事：读状态 / 写状态 / 判成败。不做任何「模拟」。
    """

    def __init__(self, root, seed):
        self.d = os.path.join(root, str(seed))
        self.io_ok = True
        self.n_io = 0
        if os.path.isdir(self.d):
            shutil.rmtree(self.d, ignore_errors=True)
        os.makedirs(self.d, exist_ok=True)
        env_write(self.d, "state.txt", 50)
        env_write(self.d, "target.txt", 50)

    def set_state(self, v):
        env_write(self.d, "state.txt", v)
        self.n_io += 1

    def set_target(self, v):
        env_write(self.d, "target.txt", v)
        self.n_io += 1

    def get_state(self):
        try:
            v = env_read(self.d, "state.txt")
            self.n_io += 1
            return v
        except (IOError, OSError, ValueError):
            self.io_ok = False
            return 0

    def get_target(self):
        try:
            v = env_read(self.d, "target.txt")
            self.n_io += 1
            return v
        except (IOError, OSError, ValueError):
            self.io_ok = False
            return 0

    def step(self, a_idx):
        """真实执行一个动作：读当前值 -> 写新值 -> 真读回。返回（新值, 是否成功写）。"""
        try:
            s = env_read(self.d, "state.txt")
            s2 = int(np.clip(s + int(ACTIONS[a_idx]), 0, V_MAX))
            env_write(self.d, "state.txt", s2)
            self.n_io += 2
            return s2, True
        except (IOError, OSError, ValueError):
            self.io_ok = False
            return 0, False

    def cleanup(self):
        shutil.rmtree(self.d, ignore_errors=True)


def m_gate(admit, correct, consistent):
    """§7.1 三闸：准入 × 裁决（= 正确 × 一致）。每项都是 0~1 标量，可手推。"""
    return float(admit) * float(correct) * float(consistent)


def _init_net(rng):
    pre = rng.integers(0, N, E0_INIT).astype(np.int32)
    post = rng.integers(0, N, E0_INIT).astype(np.int32)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    keep = post >= N_IN          # 输入区不接受反向边（§16.10 缺陷 5）
    pre, post = pre[keep], post[keep]
    w = np.where(pre >= N_E, -W_E * W_I_RATIO, W_E).astype(np.float32)
    age = np.full(pre.size, 100, np.int16)
    return _sort(pre, post, w, age)


def run_one(args):
    seed, mode, n_ep = args
    rng = np.random.default_rng(seed)
    tag = "%s_%d" % (mode, seed)

    env = Env(ENV_ROOT, tag)
    pre, post, w, key, age = _init_net(rng)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    xin = np.zeros(N, np.float32)
    trace = np.zeros(N, np.float32)
    theta = np.zeros(N, np.float32)        # BCM 滑动阈值（§7.3 的 (a_post − θ)）

    r_out = np.zeros(N_OUT, np.float64)
    ema_rate = 0.02

    # 归因门用的「世界模型」：预测下一步 state。1 参数线性（state += k·a）
    k_pred = 0.0
    pred_err_ema = 1.0

    # 隔离：本 episode 活跃过的神经元
    ep_active = np.zeros(N, bool)

    do_plast_w = mode not in ('frozen', 'plast_topo')
    do_plast_topo = mode not in ('frozen', 'plast_w')
    use_gate = mode != 'no_gate'
    use_bridge = mode != 'no_bridge'
    use_isolation = mode != 'no_isolation'
    recall = (mode == 'recall')

    ep_succ = np.zeros(n_ep, np.float32)
    ep_progress = np.zeros(n_ep, np.float32)   # 走过的距离 / 最短需要（<1 = 没走够，>1 = 走多）
    ep_detour = np.full(n_ep, np.nan, np.float32)  # 弯路率：只对成功的 episode 定义（>1 才好比）
    ep_fixdelay = np.zeros(n_ep, np.float32)
    ep_basis = np.zeros(n_ep, np.float32)
    ep_react = np.zeros(n_ep, np.float32)
    ep_consistent = np.zeros(n_ep, np.float32)
    ep_io_ok = np.zeros(n_ep, np.float32)

    m_acc = m_cnt = 0.0
    born = drop = 0
    t0 = time.perf_counter()

    def net_step(xin_vec, apply_inh):
        nonlocal v, s, ref, trace, theta, ema_rate
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        if apply_inh:
            # 输出间侧抑制（只作用于 4 个输出）
            ro = s[OUT_IDX]
            if ro.any():
                I[OUT_IDX] -= SIDE_INH * ro
        v = v * DECAY_V + I + xin_vec
        fire = (v > V_TH) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        ref[s > 0] = T_REF
        # 输出神经元单独的不应期（§16.10 缺陷 6）——否则窗口内放电太少，argmax 是噪声
        of = s[OUT_IDX] > 0
        if of.any():
            ref[OUT_IDX[of]] = T_REF_OUT
        nsp = int(s.sum())
        ema_rate = 0.99 * ema_rate + 0.01 * (nsp / N)
        trace *= TRACE_DECAY
        trace += s
        theta *= (1.0 - 1.0 / 200.0)
        theta += trace / 200.0
        return nsp

    def encode(st, tg, ti):
        """把观测编码成输入脉冲。**ti 必须每步递增**——否则每步用同一行随机数，
        输入就变成了一个固定模式（2026-10-24 实测：rate 被压到 0.00%）。"""
        row = u_buf[ti % u_buf.shape[0]]
        xin.fill(0.0)
        xs = obs_code(st)
        xg = obs_code(tg) if not recall else 0.0
        p_s = IN_RATE * (1.0 + GAIN * xs)
        p_g = IN_RATE * (1.0 + GAIN * xg)
        xin[0:N_CH] = row[0:N_CH] < p_s
        xin[N_CH:2 * N_CH] = row[N_CH:2 * N_CH] < p_s
        xin[2 * N_CH:3 * N_CH] = row[2 * N_CH:3 * N_CH] < p_g
        xin[3 * N_CH:4 * N_CH] = row[3 * N_CH:4 * N_CH] < p_g

    u_buf = rng.random((65536, N_IN), dtype=np.float32)
    t_glob = 0

    prev_end = None      # recall 模式用：上一个 episode 的终点

    for ep in range(n_ep):
        # ---- 任务：目标每 K_EP_TASK 个 episode 换一次（跨 episode 稳定）----
        if ep % K_EP_TASK == 0:
            env.set_target(int(rng.integers(OBS_LO, OBS_HI)))
        st = int(rng.integers(OBS_LO, OBS_HI))
        env.set_state(st)
        tg = env.get_target()

        if recall:
            # 目标不可观测；要求回到上一 episode 的终点（跨 episode 记忆）
            goal = prev_end if prev_end is not None else (V_MAX - st)
        else:
            goal = tg

        start = st
        cur_err = abs(goal - st)
        ep_active[:] = False
        n_act = 0
        n_ok_act = 0
        act_hist = np.zeros(N_OUT, np.float64)   # 每个动作被选了几次
        act_good = np.zeros(N_OUT, np.float64)   # 其中朝目标的次数
        total_move = 0
        min_move = abs(goal - start)
        fix_delay = -1
        first_far = -1
        reacted = 0
        react_n = 0
        last_dir = 0
        consistent_sum = 0.0
        consistent_n = 0

        for k in range(T_EP):
            encode(st, goal, t_glob)
            t_glob += 1
            net_step(xin, True)
            ep_active |= (s > 0)

            r_out *= (1.0 - 1.0 / A_TAU)
            r_out += s[OUT_IDX] / A_TAU

            if k % T_ACT == 0 and k > 0:
                a_idx = int(np.argmax(r_out))
                # 归因门：预测下一步 state（1 参数：state += k_pred * ACTION）
                pred = st + k_pred * float(ACTIONS[a_idx])
                new_st, ok = env.step(a_idx)
                if not ok:
                    break
                perr = abs(new_st - pred)
                pred_err_ema = 0.9 * pred_err_ema + 0.1 * perr
                k_pred += 0.05 * (new_st - st) / float(ACTIONS[a_idx])
                consistent = float(np.clip(1.0 - perr / SURPRISE_K, 0.0, 1.0))
                consistent_sum += consistent
                consistent_n += 1

                # 四度量
                move = new_st - st
                total_move += abs(move)
                want = np.sign(goal - st)
                if want != 0 and np.sign(move) != 0:
                    n_act += 1
                    act_hist[a_idx] += 1
                    if np.sign(move) == want:
                        n_ok_act += 1
                        act_good[a_idx] += 1
                new_err = abs(goal - new_st)
                if new_err > cur_err and first_far < 0:
                    first_far = k
                if first_far >= 0 and new_err < cur_err and fix_delay < 0:
                    fix_delay = k - first_far
                d_now = int(np.sign(move))
                if last_dir != 0 and d_now != 0 and d_now != last_dir:
                    reacted += 1
                if last_dir != 0 and d_now != 0:
                    react_n += 1
                last_dir = d_now
                cur_err = new_err
                st = new_st
                if cur_err == 0:
                    break

        succ = (cur_err == 0)
        ep_succ[ep] = 1.0 if succ else 0.0
        prog = total_move / max(min_move, 1)
        ep_progress[ep] = prog
        if succ:
            ep_detour[ep] = prog - 1.0      # 只有成功的才有「绕了多少」可比
        ep_fixdelay[ep] = float(fix_delay) if fix_delay >= 0 else float(T_EP)
        ep_basis[ep] = float(n_ok_act) / max(n_act, 1)
        ep_react[ep] = float(reacted) / max(react_n, 1)
        ep_consistent[ep] = consistent_sum / max(consistent_n, 1)
        ep_io_ok[ep] = 1.0 if env.io_ok else 0.0

        # ---- 结构更新：**每个 episode 一次**（长期记忆只有次数，没有时间 —— §2.10）----
        if ep > 0:
            admit = 1.0 if env.io_ok else 0.0
            moved = 1.0 if total_move > 0 else 0.0
            if use_bridge:
                correct = max(1.0 if succ else 0.0, BRIDGE * moved)
            else:
                correct = 1.0 if succ else 0.0
            consistent = float(ep_consistent[ep])
            if use_gate:
                m = m_gate(admit, correct, consistent) * steady_gate(ema_rate)
            else:
                m = steady_gate(ema_rate)
            m_acc += m
            m_cnt += 1

            # ** m 必须按神经元分配，不能是标量**（§16.9 缺陷 3、§16.10 缺陷 3 同一个错）：
            # 标量 m 让所有边同等更新 -> 学不到「该选哪个动作」。
            # 这里把标量门控再用「该动作朝目标的比例」分配到底四个输出神经元上。
            m_vec = np.zeros(N, np.float32)
            for i in range(N_OUT):
                if act_hist[i] > 0:
                    m_vec[OUT_IDX[i]] = float(act_good[i] / act_hist[i])

            if m > 0 and key.size:
                el = trace[pre] * (trace[post] - theta[post])
                if use_isolation:
                    # 隔离：只写入「本 episode 活跃过」的两端（§9.2 / §2.4 第 2 条）
                    mask = ep_active[pre] & (ep_active[post] | (post >= N_E))
                    el = np.where(mask, el, 0.0)
                # 接口边（突触后是输出神经元）用分配的 m；其余边用标量门控
                ist_o = (post == OUT_IDX[0]) | (post == OUT_IDX[1]) | \
                        (post == OUT_IDX[2]) | (post == OUT_IDX[3])
                sel = np.where(ist_o, m_vec[post], m).astype(np.float32)
                el = el * sel
                emax = float(np.abs(el).max())
                if emax > 1e-9:
                    el = el / emax
                    if do_plast_w:
                        dw = (ETA_POT * el).astype(np.float32)
                        cap = np.where(w > 0, W_MAX_E, W_MAX_I)
                        w = w + dw * (cap - np.abs(w))
                        np.clip(w, -W_MAX_I, W_MAX_E, out=w)

            if do_plast_topo:
                act = np.nonzero(trace > ISOLATE_THRESH)[0]
                if act.size > MAX_ACTIVE:
                    act = rng.choice(act, MAX_ACTIVE, replace=False)
                if act.size >= 2:
                    na = act.size
                    q_pre = np.repeat(act, na)
                    q_post = np.tile(act, na)
                    okk = (q_pre != q_post) & (q_post >= N_IN)
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

                age += 1
                if key.size > BUDGET:
                    dd = key.size - BUDGET
                    ist_out = (post == OUT_IDX[0]) | (post == OUT_IDX[1]) | \
                              (post == OUT_IDX[2]) | (post == OUT_IDX[3])
                    scale = np.abs(w) / np.where(pre >= N_E, W_MAX_I, W_MAX_E)
                    cand = (age >= MIN_AGE) & (~ist_out)
                    if not cand.any():
                        cand = age >= MIN_AGE
                    if cand.any():
                        idx_c = np.nonzero(cand)[0]
                        dd = min(dd, idx_c.size)
                        if dd > 0:
                            sub = idx_c[np.argpartition(scale[idx_c], dd - 1)[:dd]]
                            mm = np.ones(key.size, bool); mm[sub] = False
                            pre, post, w, key, age = pre[mm], post[mm], w[mm], key[mm], age[mm]
                            drop += int(dd)

        prev_end = st

        # ---- 间隙：逼退隐状态（τ_v = 20），迫使跨 episode 只能靠连接 ----
        for _ in range(T_GAP):
            encode(50, 50, t_glob)
            t_glob += 1
            net_step(xin, False)

    env.cleanup()

    deg_in = np.bincount(post, minlength=N)
    k6 = max(1, n_ep // 6)
    return dict(
        seed=seed, mode=mode, E=int(key.size), born=born, drop=drop,
        rate=float(ema_rate), n_inh=int((pre >= N_E).sum()),
        succ=float(np.mean(ep_succ)),
        succ_first=float(np.mean(ep_succ[:k6])),
        succ_last=float(np.mean(ep_succ[-k6:])),
        detour=float(np.nanmean(ep_detour)) if np.any(~np.isnan(ep_detour)) else float('nan'),
        n_detour=int(np.sum(~np.isnan(ep_detour))),
        progress=float(np.nanmean(ep_progress)),
        detour_first=float(np.nanmean(ep_detour[:k6])) if np.any(~np.isnan(ep_detour[:k6])) else float('nan'),
        detour_last=float(np.nanmean(ep_detour[-k6:])) if np.any(~np.isnan(ep_detour[-k6:])) else float('nan'),
        fixdelay=float(np.nanmean(ep_fixdelay)),
        basis=float(np.nanmean(ep_basis)),
        react=float(np.nanmean(ep_react)),
        consistent=float(np.nanmean(ep_consistent)),
        io_ok=float(np.mean(ep_io_ok)),
        m_mean=float(m_acc / max(m_cnt, 1)),
        n_io=env.n_io,
        out_deg=int(deg_in[OUT_IDX].mean()),
        elapsed=time.perf_counter() - t0)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_ep = int(argv[0]) if len(argv) > 0 else 100
    nproc = int(argv[1]) if len(argv) > 1 else 8
    nseed = int(argv[2]) if len(argv) > 2 else 8
    modes = argv[3].split(',') if len(argv) > 3 else ['base']

    for mode in modes:
        print('=' * 138)
        print('grow_agent  mode=%-12s  N=%d  预算=%d  episode=%d×%d步  种子=%d'
              % (mode, N, BUDGET, n_ep, T_EP, nseed))
        print('  环境 = 本机真实文件 %s\\<mode>_<seed>\\  动作 = %s  间隙 = %d 步'
              % (ENV_ROOT, list(ACTIONS), T_GAP))
        print('=' * 138)
        with Pool(nproc) as pool:
            t0 = time.perf_counter()
            res = pool.map(run_one, [(s, mode, n_ep) for s in range(nseed)])
            wall = time.perf_counter() - t0

        print('%-5s %6s %7s %6s %7s %7s %8s %8s %8s %7s %6s %6s'
              % ('seed', 'E', 'rate%', 'n_inh', 'succ', 'prog', 'fixdelay',
                 'basis', 'react', 'consist', 'm', 'odeg'))
        for r in res:
            print('%-5d %6d %7.2f %6d %7.1f%% %7.2f %8.1f %8.3f %8.3f %7.3f %6.3f %6d'
                  % (r['seed'], r['E'], r['rate'] * 100, r['n_inh'],
                     r['succ'] * 100, r['progress'], r['fixdelay'],
                     r['basis'], r['react'], r['consistent'], r['m_mean'],
                     r['out_deg']))

        def agg(k):
            return float(np.mean([r[k] for r in res]))

        s1 = np.mean([r['succ_first'] for r in res]) * 100
        s2 = np.mean([r['succ_last'] for r in res]) * 100
        print('  跨种子: 成功率 早 %.1f%% -> 晚 %.1f%% (%+.1f pp)   全程 %.1f%%'
              % (s1, s2, s2 - s1, agg('succ') * 100))
        print('  过程四度量: 弯路率 %.2f（仅成功 n=%.0f，0=最短）  '
              '修正延迟 %.1f 步  依据密度 %.3f  遇意外反应 %.3f'
              % (agg('detour'), agg('n_detour'), agg('fixdelay'), agg('basis'), agg('react')))
        print('               progress %.2f（走过距离/最短需要，<1 = 没走够）' % agg('progress'))
        print('  四层反馈: 环境层（IO 成功 %.3f）  参照层（**未实现**）  '
              '内部层（一致 %.3f）  过程层 ↑' % (agg('io_ok'), agg('consistent')))
        print('  m 均值 %.4f   IO 次数 %.0f   E %.0f±%.0f   耗时 %.1fs'
              % (agg('m_mean'), agg('n_io'), agg('E'),
                 float(np.std([r['E'] for r in res])), wall))
        print()


if __name__ == '__main__':
    main()
