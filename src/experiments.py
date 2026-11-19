# agi/experiments.py — 共享 LIF 框架的实验配置与历史复现配置。
# 相同 flow 的变体在此声明，独立原型见 experiments/prototypes/。
# 配置名字和 note 不是机制正确性的证明；已知对照限制见 docs/experiments.md。
# 本轮整理保留数值配置和执行行为，只纠正具有误导性的说明。

from dataclasses import replace

from .config import (Config, Plasticity, SignalSpec, TaskSpec, EnvSpec,
                     ReadoutSpec, NetSpec)

BASE = Config(
    name="base",
    note="全开：动作级 m + 三闸 + 三层可塑性 + 真实环境",
    n_ep=200,
)


# ---------------- 可塑性层（三层各自独立）----------------
# 2026-10-24：这三组就是为了回答「哪一层在起作用」。之前 frozen 名不副实导致结论反了。

PLAST_NONE = BASE.with_(name="plast_none", note="什么都不学（真基线）",
                        plast=Plasticity(weight=False, topology=False, bias=False))
PLAST_BIAS = BASE.with_(name="plast_bias", note="只学 bias（§14.6 作用点二）",
                        plast=Plasticity(weight=False, topology=False, bias=True))
PLAST_W = BASE.with_(name="plast_w", note="只学权重",
                     plast=Plasticity(weight=True, topology=False, bias=False))
PLAST_TOPO = BASE.with_(name="plast_topo", note="只学拓扑",
                        plast=Plasticity(weight=False, topology=True, bias=False))
PLAST_W_BIAS = BASE.with_(name="plast_w_bias", note="bias + 权重（无拓扑）",
                          plast=Plasticity(weight=True, topology=False, bias=True))


# ---------------- 信号源（学习方式本身）----------------

SIG_NONE = BASE.with_(name="sig_none", note="m 恒 0（比 plast_none 更彻底：连偏置都不动）",
                      signal=SignalSpec(source="none"),
                      plast=Plasticity(weight=False, topology=False, bias=False))
SIG_RANDOM = BASE.with_(name="sig_random", note="m 随机（检验「有 m 就行吗」）",
                        signal=SignalSpec(source="random"))
SIG_EPISODE = BASE.with_(name="sig_episode", note="历史配置：任务末权重信号丢失，沉默时仍学 bias；不可作纯延迟反馈对照",
                         signal=SignalSpec(source="episode"))
SIG_SCALAR = BASE.with_(name="sig_scalar", note="动作反馈为标量，但沉默提示仍为向量；不可作纯标量反馈对照",
                        signal=SignalSpec(source="action", shape="scalar"))
SIG_NORPE = BASE.with_(name="sig_norpe", note="m 不用 RPE（直接 r）",
                       signal=SignalSpec(source="action", glob="raw"))
SIG_NOGATE = BASE.with_(name="sig_nogate", note="不用三闸（m = 正确性）",
                        signal=SignalSpec(source="action", gate=False))


# ---------------- 隔离 ----------------
NONISO = BASE.with_(name="no_isolate", note="写入不做局部化（§9.2 隔离 = 0）",
                    plast=Plasticity(weight=True, topology=True, bias=True, isolate=False))


# ---------------- 任务 ----------------
# 2026-10-24：任务不可完成（6 个动作 vs 差距 40）曾使全部读数无意义。
# 前置检查现在会自动拦这种配置——下面两个都因此调整过：
#   task_sampled：obs 范围 80 -> 40，使最大差距=40 能被 48 个动作盖住
#   readout_argmax：argmax 只能管「输出数 = 动作数」的情形，所以 n_out=2
TASK_SAMPLED = BASE.with_(
    name="task_sampled", note="初态/目标随机采样（样本几乎不重复）",
    task=TaskSpec(kind="sampled", obs_lo=30, obs_hi=70))


# ---------------- 读数 ----------------
ARGMAX = BASE.with_(
    name="readout_argmax", note="argmax 读数（永远选一个，无沉默）",
    readout=ReadoutSpec(kind="argmax", n_out=2, out_idx=(500, 501),
                        allow_noop=False))


# ---------------- 对齐（不是实验）----------------
# 2026-10-24：只为回答「框架为什么差 5 倍」。每个开关对应旧 grow_agent(discrete) 的一处行为：
#   actions=(1,-1,10,-10)   旧 ACTION=[1,-1,10,-10,0]（第 5 个从不被索引到）
#   tie_break=argmax_v      旧：cand[argmax(v[OUT_IDX][cand])]
#   topo_gate=False         旧：每个 ep>0 都跑拓扑（不看 m_ep）
#   bridge=0.15             旧：correct = max(succ, 0.15×moved)
#   silence_update_w=True   旧：沉默时 upd_weights(mv, 0.0)
#   steady_gate_ep=True     旧：m = m_gate(...) × steady_gate(ema_rate)
#   u_buf_rows=65536        旧：u_buf = rng.random((65536, N_IN))
# 若 aligned 能回到 37%，说明差集完整；再逐项关掉量贡献。
# （上述开关已全部内联到下面的 ga_* 系列；ALIGNED 本身已由 ga_discrete 取代。）


# ---------------- 从旧 grow_agent 移植（2026-10-24）----------------
# 每个 Config 对应 grow_agent.py 的一个 mode，目标是**行为逐项一致**（用 scripts/align_layers.py seq 验证）。
# 旧代码的 mode 解析很容易让人误解（`discrete_w` 含 bias、`no_gate` 只学 bias），
# 所以下面每个 name 后面标的是**代码解析出的真实三层开关**（b/w/topo），不是 mode 名字。
_GA = dict(
    env=EnvSpec(actions=(1, -1, 10, -10)),
    readout=ReadoutSpec(kind="first_spike", n_out=4,
                        out_idx=(500, 501, 502, 503),
                        allow_noop=True, tie_break="argmax_v"),
    signal=SignalSpec(source="action", shape="per_neuron", glob="rpe", gate=True,
                      bridge=0.15, silence_update_w=True, steady_gate_ep=True),
    u_buf_rows=65536,
    reset_readout_per_ep=False,
)
_DISCRETE_TASK = TaskSpec(kind="discrete")
_SAMPLED_TASK = TaskSpec(kind="sampled", obs_lo=10, obs_hi=90, k_ep_task=5)
_RECALL_TASK = TaskSpec(kind="recall", obs_lo=10, obs_hi=90, k_ep_task=5)

GA_DISCRETE = Config(name="ga_discrete", note="grow_agent mode=discrete（b/w/topo 全开）",
                     plast=Plasticity(topo_gate=False, topo_skip_first=True), **_GA)
GA_NONE = Config(name="ga_none", note="grow_agent discrete_none（0/0/0，discrete 任务）",
                 plast=Plasticity(weight=False, topology=False, bias=False,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_BIAS = Config(name="ga_bias", note="grow_agent discrete_bias（1/0/0）",
                 plast=Plasticity(weight=False, topology=False, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_W = Config(name="ga_w", note="grow_agent discrete_w（1/1/0：名字叫 w，实际含 bias）",
              plast=Plasticity(weight=True, topology=False, bias=True,
                               topo_gate=False, topo_skip_first=True), **_GA)
GA_TOPO = Config(name="ga_topo", note="grow_agent discrete_topo（1/0/1）",
                 plast=Plasticity(weight=False, topology=True, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_FROZEN = Config(name="ga_frozen", note="grow_agent frozen（0/0/0，**sampled 任务**）",
                   task=_SAMPLED_TASK,
                   plast=Plasticity(weight=False, topology=False, bias=False,
                                    topo_gate=False, topo_skip_first=True), **_GA)
GA_BASE = Config(name="ga_base", note="grow_agent base（1/0/0，sampled 任务）",
                 task=_SAMPLED_TASK,
                 plast=Plasticity(weight=False, topology=False, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_RECALL = Config(name="ga_recall", note="grow_agent recall（1/0/0，目标不可观测）",
                   task=_RECALL_TASK,
                   plast=Plasticity(weight=False, topology=False, bias=True,
                                    topo_gate=False, topo_skip_first=True), **_GA)
_GA_NO_READOUT = {k: v for k, v in _GA.items() if k != "readout"}
GA_RANDOM = Config(name="ga_random",
                   note="grow_agent mode=random——不建网、不学，只随机选动作（下界基线）",
                   readout=ReadoutSpec(kind="random", n_out=4,
                                       out_idx=(500, 501, 502, 503),
                                       allow_noop=False),
                   **_GA_NO_READOUT)


# ---------------- 从旧 grow_goal 移植（2026-10-24）----------------
# 形态差异全部靠 Config 字段表达：flow=goal / env.kind=math_2d / readout.kind=sign_2d /
# plast.trigger=step / signal.source=value。本体（core.py）与原子操作与 agent 流程共用。
_GOAL = dict(
    flow="goal",
    net=NetSpec(side_inh=0.0, t_ref_out=20),   # grow_goal 无侧抑制、无输出专用不应期
    env=EnvSpec(kind="math_2d"),
    task=TaskSpec(kind="sampled", t_act=20, t_ep=120, t_gap=0),
    readout=ReadoutSpec(kind="sign_2d", n_out=2, out_idx=(500, 501),
                        a_tau=20.0, base_tau=100.0),
    plast=Plasticity(weight=True, topology=True, bias=False,
                     trigger="step", t_s=20, topo_thresh=0.5),
    signal=SignalSpec(source="value"),
)

GA_GOAL_A = Config(name="ga_goal_A", note="grow_goal A（主体：成败当 m，每 20 步更新）",
                   n_ep=200, **_GOAL)
GA_GOAL_FROZEN = Config(name="ga_goal_frozen",
                        note="grow_goal A_frozen（不强化、不生长；淘汰仍执行）",
                        n_ep=200,
                        **dict(_GOAL, plast=Plasticity(
                            weight=False, topology=False, bias=False,
                            trigger="step", t_s=20, topo_thresh=0.5)))
GA_GOAL_FIXED = Config(name="ga_goal_fixed",
                       note="grow_goal A_fixed（目标固定为 (0.5,-0.5)，排除「恒定动作碰巧命中」）",
                       n_ep=300,
                       **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                   t_gap=0, goal_mode="fixed")))
GA_GOAL_PROBE = Config(name="ga_goal_probe",
                       note="grow_goal A_probe（诊断：从状态线性读出 (g-x)）",
                       n_ep=150,
                       **dict(_GOAL, readout=ReadoutSpec(
                           kind="sign_2d", n_out=2, out_idx=(500, 501),
                           a_tau=20.0, base_tau=100.0, probe=True)))
GA_GOAL_T_SEQ = Config(name="ga_goal_t_seq",
                       note="grow_goal T_seq（先 task A 右上 再 task B 左下）",
                       n_ep=400,
                       **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                   t_gap=0, goal_mode="two_phase",
                                                   phase="seq")))
GA_GOAL_T_FRESH = Config(name="ga_goal_t_fresh",
                         note="grow_goal T_fresh（直接跑 task B）",
                         n_ep=400,
                         **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                     t_gap=0, goal_mode="two_phase",
                                                     phase="fresh")))


# ---------------- 从旧 grow_world 移植（2026-10-24）----------------
# 形态：单条连续时间轴（无 episode）+ 2D 振荡器世界 + 连续 tanh 动作。
# m = 稳态门 × (0.2 + 0.8×动作方差代理)；el 不减 θ；非对称权重更新。
# 生长**不过滤输入区**、有全局随机边、淘汰**不保护**接口边——这三条都与 grow_agent 相反。
_WORLD = dict(
    flow="world",
    steps=50000,
    net=NetSpec(side_inh=0.0, t_ref_out=20, w_i_ratio=5.0,
                block_input_edges=False, protect_out_edges=False,
                p_global=4, t_global=2000),
    env=EnvSpec(kind="oscillator", period_d=20.0, period_d_slow=500.0,
                tau_w_f=50.0, tau_w_s=500.0, amp_d=0.10, amp_d_slow=0.10,
                k_phase=0.05, w_noise=0.01),
    task=TaskSpec(kind="none", t_ep=1, t_act=1, t_gap=0),
    readout=ReadoutSpec(kind="tanh_2d", n_out=2, out_idx=(500, 501),
                        a_tau=20.0, r_base=0.02, r_scale=0.03),
    plast=Plasticity(weight=True, topology=True, bias=False,
                     trigger="step", t_s=20, topo_thresh=0.5,
                     tau_trace=20.0, trace_decay=0.95),
    signal=SignalSpec(source="agency", gate=True, base=0.2, agency_scale=0.2),
)

GA_WORLD_CLOSED = Config(name="ga_world_closed",
                         note="grow_world closed（闭环：动作推世界，世界回输入）", **_WORLD)
GA_WORLD_OPEN = Config(name="ga_world_open",
                       note="grow_world open（断开：动作不推世界）",
                       **dict(_WORLD, signal=SignalSpec(source="agency", gate=False,
                                                        base=0.2, agency_scale=0.2)))
GA_WORLD_RANDACT = Config(name="ga_world_randact",
                          note="grow_world randact（动作随机化）",
                          **dict(_WORLD, readout=ReadoutSpec(
                              kind="randact", n_out=2, out_idx=(500, 501),
                              a_tau=20.0, r_base=0.02, r_scale=0.03)))
GA_WORLD_NOISE = Config(name="ga_world_noise_in",
                        note="grow_world noise_in（输入白噪声，无世界结构）",
                        **dict(_WORLD, signal=SignalSpec(source="noise", gate=True,
                                                          base=0.2, agency_scale=0.2)))
GA_WORLD_FROZEN = Config(name="ga_world_frozen",
                         note="grow_world frozen（结构冻结：可塑性整块关闭）",
                         **dict(_WORLD, plast=Plasticity(
                             weight=False, topology=False, bias=False,
                             trigger="step", t_s=20, topo_thresh=0.5,
                             tau_trace=20.0, trace_decay=0.95)))


# ---------------- 从旧 grow_talk 移植（2026-10-24）----------------
# 形态：三段式时间轴（编码脉冲 → 保持 → 提问 → 回答）+ 事件发射（first-spike）
#       + 状态永不清空 + 阈值适应 + 教师 m（按神经元分配的 tgt-got）。
# 接口：输出只接「提问区 10 + 内部随机 40」——**不接词块**（不把答案焊上去）。
# ⚠️ 未移植：STD / 突触缩放（旧脚本默认关，不影响默认行为）。
_TALK = dict(
    flow="talk",
    n_ep=800,                      # 呈现数（旧脚本命令行第 1 个参数）
    net=NetSpec(
        init_kind="talk", side_inh_edges=True, side_inh=0.5,
        block_input_edges=False, protect_out_edges=True,
        p_global=4, t_global=1000,
        theta_mode="fire_adapt", tau_adapt=100.0, d_theta=0.2,
        th_min=1.0, th_max=3.0, d_theta_out=0.02,
        n_q=10, blk=25, n_out_int=40, n_in_fanout=50,
        talk_legacy_offset=True,   # 旧偏移：保持与 grow_talk 逐位一致（对齐用）
    ),
    task=TaskSpec(kind="none", t_present=500, burst=20, nburst=2, query_at=120,
                  qburst=10, answer_from=120, t_max=200, cool=30,
                  in_spike=3.0, in_lo=0.001),
    readout=ReadoutSpec(kind="first_spike", n_out=4,
                        out_idx=(500, 501, 502, 503),
                        allow_noop=True, tie_break="argmax_v"),
    plast=Plasticity(weight=True, topology=True, bias=True,
                     eta_w=0.008, eta_bias=0.004,
                     bias_min=-0.10, bias_max=0.40,
                     trace_decay=0.95, topo_thresh=0.5, t_s=20),
    signal=SignalSpec(source="teacher_word", tau_m=100.0, k_update=5,
                      m_up=1.0, m_dn=-0.5, warmup=3000, bl_tau=40.0),
)
_TALK_SIG = dict(tau_m=100.0, k_update=5, m_up=1.0, m_dn=-0.5,
                 warmup=3000, bl_tau=40.0)
_TALK_PLAST_OFF = Plasticity(weight=False, topology=False, bias=False,
                             eta_w=0.008, eta_bias=0.004,
                             bias_min=-0.10, bias_max=0.40,
                             trace_decay=0.95, topo_thresh=0.5, t_s=20)

TA_TEACH = Config(name="ta_teach", note="grow_talk teach（教师信号；基准）", **_TALK)
TA_RAND_M = Config(name="ta_rand_m", note="grow_talk rand_m（m 随机 ±1）",
                   **dict(_TALK, signal=SignalSpec(source="random", **_TALK_SIG)))
TA_NO_TEACH = Config(name="ta_no_teach", note="grow_talk no_teach（m 恒 +1）",
                     **dict(_TALK, signal=SignalSpec(source="constant", **_TALK_SIG)))
TA_FROZEN = Config(name="ta_frozen", note="grow_talk frozen（三层可塑性全关）",
                   **dict(_TALK, plast=_TALK_PLAST_OFF))

# ---- 修正接口权重偏移后的同一组（2026-10-24）----
# 与 ta_* 的唯一差异：`talk_legacy_offset=False`——接口边真正拿到 uniform(0.15~0.60)。
# ta_* 保留旧偏移（逐位可复现 / 与 §16.10 历史读数可比），两组并存。
_TALK_FIX = dict(_TALK, net=replace(_TALK["net"], talk_legacy_offset=False))

TK_TEACH = Config(name="tk_teach", note="grow_talk teach + 修正接口权重偏移", **_TALK_FIX)
TK_RAND_M = Config(name="tk_rand_m", note="grow_talk rand_m + 修正接口权重偏移",
                   **dict(_TALK_FIX, signal=SignalSpec(source="random", **_TALK_SIG)))
TK_NO_TEACH = Config(name="tk_no_teach", note="grow_talk no_teach + 修正接口权重偏移",
                     **dict(_TALK_FIX, signal=SignalSpec(source="constant", **_TALK_SIG)))
TK_FROZEN = Config(name="tk_frozen", note="grow_talk frozen + 修正接口权重偏移",
                   **dict(_TALK_FIX, plast=_TALK_PLAST_OFF))


# ---------------- 从旧 grow_homeo 移植（2026-10-24）----------------
# 形态：PLOS CB 2022 —— 初始**无 E-E**，结构按放电率偏差**双向**调 E-E 入边。
# 两个自组织机制同时开：阈值适应（fire_adapt）+ STD（抑制短时抑郁）。
# 目标率 0.8% / ±20% 容忍区间 / 率估计 τ=500 / 结构周期 50 步。
_HOMEO_READOUT = ReadoutSpec(kind="first_spike", n_out=2, out_idx=(500, 501))
_HOMEO = dict(
    flow="homeo",
    steps=6000,
    net=NetSpec(
        init_kind="homeo", theta_mode="fire_adapt",
        tau_adapt=100.0, d_theta=0.20, th_min=1.0, th_max=3.0, d_theta_out=0.0,
        theta_out_adapt=False, t_ref_out=20,   # homeo：out_idx 只是普通神经元，全体一视同仁
        w_i_ratio=2.0,          # w_i = -(0.50 × 2.0) = -1.0（旧扫描出的工作点）
        w_ee_init=0.25,
        d_ei=0.20, d_ie=0.05, d_ii=0.10,
        n_in_fanout=50,
        use_std=True, tau_rec=100.0, u_std=0.30,
        tau_rate=500.0,
    ),
    task=TaskSpec(kind="none", n_cue=4, enc_steps=20, enc_gap=150,
                  cue_after=0, bg_rate=0.015, in_spike=3.0),
    plast=Plasticity(weight=False, topology=True, bias=False,
                     trigger="step", t_s=50,
                     struct_kind="homeo", r_target=0.008, tol=0.20,
                     p_del=0.30, p_new=0.30, trace_decay=0.95,
                     reset_state_per_chunk=True),   # 旧行为：每 chunk 清空网络状态
    signal=SignalSpec(source="homeo"),
    n_ep=0,
)

HO_GROW = Config(name="ho_grow", note="grow_homeo 主表（6000 步；对应旧默认参数）", **_HOMEO)
HO_LONG = Config(name="ho_long", note="grow_homeo 长跑（40000 步；§16.11 那张表的规模）",
                 **dict(_HOMEO, steps=40000))
HO_WARM = Config(name="ho_warm", note="grow_homeo probe 的生长阶段（30000 步）",
                 **dict(_HOMEO, steps=30000))
HO_FROZEN = Config(name="ho_frozen", note="历史配置：同时关闭结构可塑性与 chunk 状态重置；非单变量对照",
                   **dict(_HOMEO, plast=Plasticity(
                       weight=False, topology=False, bias=False,
                       trigger="step", t_s=50, struct_kind="homeo",
                       r_target=0.008, tol=0.20, p_del=0.30, p_new=0.30,
                       trace_decay=0.95)))


# ---------------- 从旧 grow_cond 移植（2026-10-24）----------------
# 阶段 1/2 v3：经典条件作用 + 可变间隔（VI）课程。与 talk 的四处差异：
#   ① 变长呈现（burst + Gamma 采样的保持期 + answer_win）
#   ② **只奖励正确**（punish=False：错/抢答/沉默都给 0）
#   ③ **抢答不终止呈现**（只记录；否则抢答率 100% 时正确率永远测不到）
#   ④ 分桶按保持期长度（hold // 50，8 桶）
# 接口与 talk **修正版**相同（它的 nfan 用 (n_in+n_q)*fanout，本来就是对的）。
_COND_NET = _TALK["net"].__class__(**{
    **{f: getattr(_TALK["net"], f) for f in _TALK["net"].__dataclass_fields__},
    "talk_legacy_offset": False,
})
_COND = dict(
    flow="cond",
    n_ep=1200,                     # 呈现数（旧命令行第 1 个参数）
    net=_COND_NET,
    task=TaskSpec(kind="none", var_delay=True, burst=20, nburst=2,
                  qburst=10, answer_win=150,
                  d_mean_start=40.0, d_mean_end=320.0, d_cv=0.5,
                  d_min=30, d_max=520, in_spike=3.0, in_lo=0.001),
    readout=ReadoutSpec(kind="first_spike", n_out=4,
                        out_idx=(500, 501, 502, 503),
                        allow_noop=True, tie_break="argmax_v",
                        early_terminate=False),
    plast=Plasticity(weight=True, topology=True, bias=True,
                     eta_w=0.008, eta_bias=0.004,
                     bias_min=-0.10, bias_max=0.40,
                     trace_decay=0.95, topo_thresh=0.5, t_s=20),
    signal=SignalSpec(source="reward_only", tau_m=100.0, k_update=5,
                      warmup=3000, punish=False),
)

CO_TEACH = Config(name="co_teach", note="grow_cond teach（只奖励正确；基准）", **_COND)
CO_PUNISH = Config(name="co_punish", note="grow_cond --punish（错/抢答/沉默给 -1）",
                   **dict(_COND, signal=SignalSpec(source="reward_only", tau_m=100.0,
                                                  k_update=5, warmup=3000, punish=True)))
CO_FROZEN = Config(name="co_frozen", note="grow_cond frozen（三层可塑性全关）",
                   **dict(_COND, plast=Plasticity(
                       weight=False, topology=False, bias=False,
                       eta_w=0.008, eta_bias=0.004,
                       bias_min=-0.10, bias_max=0.40,
                       trace_decay=0.95, topo_thresh=0.5, t_s=20)))


# ---------------- 从旧 grow_teach 移植（2026-10-24）----------------
# 阶段 1/2 v1：持续速率编码 + 固定 40 步 trial + 末窗 argmax 读出。
# 与 talk / cond 的差异：
#   ① 输入是**持续速率**（in_hi / in_lo），不是脉冲爆发
#   ② 读出 = trial 末 read_win 步的**放电计数 argmax**（不是 first-spike）
#   ③ **只更新接口边**（only_out_edges）；结构更新在 trial 末
#   ④ 接口直连**输入区**（60 条，权重 0.05~0.35），且随机边**不过滤** post
_TEACH_NET = NetSpec(
    init_kind="teach", theta_mode="static",   # grow_teach 无阈值适应（vth 从不更新）
    tau_adapt=100.0, d_theta=0.20, th_min=1.0, th_max=3.0,
    theta_out_adapt=True, d_theta_out=0.02, t_ref_out=4,
    side_inh=1.5, n_out_link=60,
    block_input_edges=False, protect_out_edges=True,
    p_global=4, t_global=25, max_active=48, min_age=25, p_new=0.004,
    budget=15000, e0_init=30000,
)
_TEACH_PLAST = Plasticity(
    weight=True, topology=True, bias=True,
    eta_w=0.008, eta_bias=0.004, bias_min=-0.10, bias_max=0.40,
    trace_decay=0.95, topo_thresh=0.5, t_s=20, only_out_edges=True,
)
_TEACH = dict(
    flow="teach",
    n_ep=10000,                    # trial 数（旧默认 6000，§16.9 用的是 10000）
    net=_TEACH_NET,
    task=TaskSpec(kind="none", trial_steps=40, read_win=20,
                  in_hi=0.030, in_lo=0.001),
    readout=ReadoutSpec(kind="argmax", n_out=4,
                        out_idx=(500, 501, 502, 503),
                        allow_noop=False),
    plast=_TEACH_PLAST,
    signal=SignalSpec(source="teacher_word", m_up=1.0, m_dn=-1.0),
)
_TEACH_OFF = Plasticity(
    weight=False, topology=False, bias=False,
    eta_w=0.008, eta_bias=0.004, bias_min=-0.10, bias_max=0.40,
    trace_decay=0.95, topo_thresh=0.5, t_s=20, only_out_edges=True,
)

TE_TEACH = Config(name="te_teach", note="grow_teach teach（正常教师；基准）", **_TEACH)
TE_RAND_M = Config(name="te_rand_m", note="grow_teach rand_m（m 随机 ±1）",
                   **dict(_TEACH, signal=SignalSpec(source="random", m_up=1.0, m_dn=-1.0)))
TE_NO_TEACH = Config(name="te_no_teach", note="grow_teach no_teach（m 恒 +1）",
                     **dict(_TEACH, signal=SignalSpec(source="constant", m_up=1.0, m_dn=-1.0)))
TE_FROZEN = Config(name="te_frozen", note="grow_teach frozen（三层可塑性全关）",
                   **dict(_TEACH, plast=_TEACH_OFF))


# ---------------- 从旧 grow_func 移植（2026-10-24）----------------
# 动力学与 grow_world **完全一致**，只加功能测量（原文件头注自己写了这句）。
# 三项：① 预测误差（在线 LMS + 离线最优读出，前半训练后半冻结）
#       ② 动作增益 R²；③ 前史因果（perturb 初始膜电位）
_FUNC = dict(_WORLD, readout=ReadoutSpec(
    kind="tanh_2d", n_out=2, out_idx=(500, 501),
    a_tau=20.0, r_base=0.02, r_scale=0.03,
    func_measure=True, pred_lr=0.02, oseg=100))

FU_CLOSED = Config(name="fu_closed", note="grow_func closed（闭环 + 功能测量）",
                   **_FUNC)
FU_OPEN = Config(name="fu_open", note="grow_func open（断开动作对世界的影响）",
                 **dict(_FUNC, signal=SignalSpec(source="agency", gate=False,
                                                base=0.2, agency_scale=0.2)))
FU_FROZEN = Config(name="fu_frozen", note="grow_func frozen（结构冻结）",
                   **dict(_FUNC, plast=Plasticity(
                       weight=False, topology=False, bias=False,
                       trigger="step", t_s=20, topo_thresh=0.5,
                       tau_trace=20.0, trace_decay=0.95)))
FU_DIV = Config(name="fu_div", note="grow_func --div（前史因果：初始膜电位 +1e-3）",
                **dict(_FUNC, readout=ReadoutSpec(
                    kind="tanh_2d", n_out=2, out_idx=(500, 501),
                    a_tau=20.0, r_base=0.02, r_scale=0.03,
                    func_measure=True, pred_lr=0.02, oseg=100, perturb=1e-3)))


# ---------------- 从旧 grow_hebb 移植（2026-10-24）----------------
# 结构冻结（先用 homeo 长 30000 步）+ E-E 边上一条 Hebbian 慢变量 A。
# 依据：Clark & Abbott PRX 14, 021001 (2024) / Wakhloo et al. PNAS 2026。
# 三段：warm(500 背景) -> encode(pulse 20 / osc 400) -> observe(1200 无输入)
# 判据：撤掉输入后活动能不能自持 + 主周期。本体的 net/homeo 参数直接复用 _HOMEO。
_HEBB_TASK = TaskSpec(kind="none", n_cue=4, in_spike=3.0, bg_rate=0.015,
                      grow_steps=30000, warm_steps=500, input_mode="pulse",
                      enc_steps=20, osc_steps=400, osc_period=100.0, osc_amp=3.0,
                      obs_steps=1200, bin_size=20)
_HEBB_PLAST = Plasticity(weight=True, topology=True, bias=False,
                         trigger="step", t_s=50, struct_kind="hebb",
                         r_target=0.008, tol=0.20, p_del=0.30, p_new=0.30,
                         trace_decay=0.95, reset_state_per_chunk=True,
                         hebb_r=0.05, tau_syn=600.0, tau_c=20.0, a_max_rel=2.0)
_HEBB = dict(_HOMEO, flow="hebb", task=_HEBB_TASK, plast=_HEBB_PLAST,
             signal=SignalSpec(source="none"), n_ep=0, steps=0)

HB_PULSE = Config(name="hb_pulse", note="grow_hebb r=0.05 pulse（基准）", **_HEBB)
HB_OSC = Config(name="hb_osc", note="grow_hebb r=0.05 osc（振荡输入）",
                **dict(_HEBB, task=replace(_HEBB_TASK, input_mode="osc")))
HB_R0 = Config(name="hb_r0", note="grow_hebb r=0（无 A；对照）",
               **dict(_HEBB, plast=replace(_HEBB_PLAST, hebb_r=0.0)))
HB_SHUF = Config(name="hb_shuf", note="grow_hebb r=0.05 pulse + A 随机重排",
                 **dict(_HEBB, readout=replace(_HOMEO_READOUT, a_shuffle=True)))
# 根因①修正：cn 中心化 + tanh（可正可负）。旧实现的 cn 恒 ≥ 0 -> A 是纯偏置。
HB_CENTER = Config(name="hb_center",
                  note="grow_hebb r=0.40 pulse + cn 中心化（根因①修正）",
                  **dict(_HEBB, plast=replace(_HEBB_PLAST, hebb_r=0.40,
                                                       hebb_centered=True)))
HB_CENTER_SHUF = Config(name="hb_center_shuf",
                       note="grow_hebb r=0.40 pulse + cn 中心化 + A 重排",
                       **dict(_HEBB, plast=replace(_HEBB_PLAST, hebb_r=0.40,
                                                            hebb_centered=True),
                                readout=replace(_HOMEO_READOUT, a_shuffle=True)))
# 组合：亚临界（g_eff~0.80）+ 中心化 A（可正可负）+ 振荡输入
# 问：不过临界时，涨落成分能不能自己记住输入的频率？
HB_SUB_OSC = Config(name="hb_sub_osc",
                    note="grow_hebb w_e=0.15 + 中心化 + osc（亚临界 + 涨落）",
                    **dict(_HEBB, task=replace(_HEBB_TASK, input_mode="osc"),
                             net=replace(_HEBB["net"], w_ee_init=0.15),
                             plast=replace(_HEBB_PLAST, hebb_centered=True)))
HB_SUB_OSC_SHUF = Config(name="hb_sub_osc_shuf",
                         note="grow_hebb 亚临界 + 中心化 + osc + A 重排",
                         **dict(_HEBB, task=replace(_HEBB_TASK, input_mode="osc"),
                                  net=replace(_HEBB["net"], w_ee_init=0.15),
                                  plast=replace(_HEBB_PLAST, hebb_centered=True),
                                  readout=replace(_HOMEO_READOUT, a_shuffle=True)))

HB_CENTER_OSC = Config(name="hb_center_osc",
                       note="grow_hebb r=0.05 osc + cn 中心化（根因①修正）",
                       **dict(_HEBB, task=replace(_HEBB_TASK, input_mode="osc"),
                                plast=replace(_HEBB_PLAST, hebb_centered=True)))

HB_P040 = Config(name="hb_p040", note="grow_hebb r=0.40 pulse（§16.12 主档）",
                 **dict(_HEBB, plast=replace(_HEBB_PLAST, hebb_r=0.40)))
HB_P040_SHUF = Config(name="hb_p040_shuf",
                      note="grow_hebb r=0.40 pulse + A 随机重排（§16.12 对照）",
                      **dict(_HEBB, plast=replace(_HEBB_PLAST, hebb_r=0.40),
                               readout=replace(_HOMEO_READOUT, a_shuffle=True)))
HB_OSC_SHUF = Config(name="hb_osc_shuf",
                     note="grow_hebb r=0.05 osc + A 随机重排（§16.12 对照）",
                     **dict(_HEBB, task=replace(_HEBB_TASK, input_mode="osc"),
                              readout=replace(_HOMEO_READOUT, a_shuffle=True)))
HB_FREEZE = Config(name="hb_freeze",
                   note="grow_hebb r=0.05 pulse + observe 期 A 冻结",
                   **dict(_HEBB, plast=replace(_HEBB_PLAST, weight=False)))


ALL = [
    BASE,
    PLAST_NONE, PLAST_BIAS, PLAST_W, PLAST_TOPO, PLAST_W_BIAS,
    SIG_NONE, SIG_RANDOM, SIG_EPISODE, SIG_SCALAR, SIG_NORPE, SIG_NOGATE,
    NONISO, TASK_SAMPLED, ARGMAX,
    GA_DISCRETE, GA_NONE, GA_BIAS, GA_W, GA_TOPO, GA_FROZEN, GA_BASE, GA_RECALL,
    GA_RANDOM,
    GA_GOAL_A, GA_GOAL_FROZEN, GA_GOAL_FIXED, GA_GOAL_PROBE,
    GA_GOAL_T_SEQ, GA_GOAL_T_FRESH,
    GA_WORLD_CLOSED, GA_WORLD_OPEN, GA_WORLD_RANDACT, GA_WORLD_NOISE, GA_WORLD_FROZEN,
    TA_TEACH, TA_RAND_M, TA_NO_TEACH, TA_FROZEN,
    TK_TEACH, TK_RAND_M, TK_NO_TEACH, TK_FROZEN,
    HO_GROW, HO_LONG, HO_WARM, HO_FROZEN,
    CO_TEACH, CO_PUNISH, CO_FROZEN,
    TE_TEACH, TE_RAND_M, TE_NO_TEACH, TE_FROZEN,
    FU_CLOSED, FU_OPEN, FU_FROZEN, FU_DIV,
    HB_PULSE, HB_OSC, HB_R0, HB_SHUF, HB_FREEZE,
    HB_P040, HB_P040_SHUF, HB_OSC_SHUF,
    HB_CENTER, HB_CENTER_SHUF, HB_CENTER_OSC,
    HB_SUB_OSC, HB_SUB_OSC_SHUF,
]

BY_NAME = {c.name: c for c in ALL}
