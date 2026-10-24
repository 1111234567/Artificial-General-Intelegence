# agi/config.py — 实验的完整声明
#
# 设计原则（2026-10-24，起因：test/ 下 21 个脚本各实现一遍本体，每个丢点漏点）：
#
#   1. **一个实验 = 一个 Config 对象**。没有任何隐式默认——每一样都必须写出来。
#      看到一份 Config，就知道这个实验开了什么、关了什么。
#   2. **开关是枚举/元组，不是字符串魔法**。`Plasticity` 每个目标独立开关。
#   3. **可序列化**。Config -> dict 是稳定的，用来当结果登记的主键（去重 / 查重跑）。
#   4. **不可变**。写错了就重写一个，不要在半路上改参数。
#
# 为什么不用命令行参数：命令行能表达「开什么」，但表达不了「关了什么」。
# 而今天的教训正是「我不知道哪个版本到底有什么」——那是关项丢掉的典型后果。

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from typing import Tuple, Optional


# ---------------------------------------------------------------- 本体

@dataclass(frozen=True)
class NetSpec:
    """网络本体（LIF + 稀疏连接）。默认值全部来自恒定标定（calib5 / §11.4.0）。"""
    N: int = 1000
    e_ratio: float = 0.8
    n_in: int = 100
    tau_v: float = 20.0
    v_th: float = 1.0
    t_ref: int = 20
    t_ref_out: int = 4          # 输出神经元单独的不应期（§16.10 缺陷 6）
    w_e: float = 0.50
    w_i_ratio: float = 5.0
    w_max_e: float = 1.0
    w_new_e: float = 0.15
    budget: int = 15000
    e0_init: int = 30000        # 过度生产 -> 修剪
    max_active: int = 48
    min_age: int = 25
    p_new: float = 0.004
    side_inh: float = 1.5
    # 拓扑细节（旧两个脚本不同，必须显式）
    block_input_edges: bool = True   # 新建边是否要求 post >= n_in（grow_agent 要 / grow_world 不要）
    protect_out_edges: bool = True   # 淘汰时是否保护接口边（grow_agent 要 / grow_world 不要）
    p_global: int = 0                # 每 t_global 步加的全局随机边数（grow_world=4）
    t_global: int = 2000
    # ---- 结构初始化形态（grow_talk 需要一套专门的接口）----
    init_kind: str = "random"       # random（三个旧流程）/ talk（输入区→扇出→内部 + 输出接口 + 侧抑制边）
    side_inh_edges: bool = False    # True = 侧抑制由**显式边**实现（step 不再内联；grow_talk）
    # ---- 阈值形态 ----
    # trace_bcm（θ = trace 慢平均，BCM）/ fire_adapt（θ = 放电阈值，放电后上调）
    # static（θ 恒 = v_th，**从不更新**——grow_teach 的形态：D_THETA 等常数定义了但没用）
    theta_mode: str = "trace_bcm"
    tau_adapt: float = 100.0        # fire_adapt：阈值回落时间常数
    d_theta: float = 0.2            # fire_adapt：每次放电阈值上调幅度
    th_min: float = 1.0             # fire_adapt：阈值下界（回落目标）
    th_max: float = 3.0             # fire_adapt：阈值上界
    d_theta_out: float = 0.02       # fire_adapt：输出神经元的适应幅度（读出不该被锁死）
    # 是否对「读出神经元」做特殊的阈值/不应期处理。
    #   True  = talk（输出是读出层，不应期 4、阈值适应极小）
    #   False = homeo（out_idx 只是普通内部神经元，全体一视同仁）
    theta_out_adapt: bool = True
    # ---- talk 接口参数（init_kind="talk" 时生效）----
    n_q: int = 10                   # 提问区神经元数（紧跟输入区的 N_IN..N_IN+n_q-1）
    blk: int = 25                   # 每个词占多少输入神经元
    n_out_int: int = 40             # 每个输出从内部网络接多少条
    n_in_fanout: int = 50           # 每个输入区神经元连多少个下游
    n_out_link: int = 60            # teach：每个输出神经元直连多少个**输入区**神经元（0.05~0.35）
    # 接口权重偏移：旧实现算 `nfan = n_in * n_in_fanout`，而扇出实际插了
    # `(n_in+n_q) * n_in_fanout` 条 → 少算 n_q*n_in_fanout 条，`uniform(0.15~0.60)`
    # 整体后移、从未落到接口边上（实测接口权重全为 w_e=0.5）。
    # True = 旧偏移（对齐旧脚本 / 复现 §16.10 用）；False = 修正（接口权重真正随机）。
    talk_legacy_offset: bool = False
    # ---- homeo 形态（grow_homeo）：初始无 E-E，结构按放电率偏差双向调 ----
    d_ei: float = 0.20              # E→I 连接概率
    d_ie: float = 0.05              # I→E 连接概率（原 0.20；扫描后取 0.05，见 grow_homeo 头注）
    d_ii: float = 0.10              # I→I 连接概率
    w_ee_init: float = 0.25         # 新生 E-E 边的初始强度
    # ---- 抑制性突触短时抑郁（STD，Phys Rev Research 2, 012042 的两个机制之一）----
    use_std: bool = False
    tau_rec: float = 100.0          # 抑制资源的恢复时间常数（步）
    u_std: float = 0.30             # 每次传递耗掉的比例
    # ---- 逐神经元放电率估计（homeo 的受控变量）----
    tau_rate: float = 500.0


# ---------------------------------------------------------------- 环境

@dataclass(frozen=True)
class EnvSpec:
    """环境。`kind` 必须显式。

    local_files : 真实读写 data/env/...（state/target 为标量）
    math_2d     : 内存 2D 连续世界（x ∈ [-1,1]²，动作推力 + 漂移 + 噪声）——grow_goal 形态
    """
    kind: str = "local_files"       # local_files = 真实读写 data/env/...
    root: str = "data/env"
    v_max: int = 100
    actions: Tuple[int, ...] = (1, -1)   # 2026-10-24：±10 无用（linear 基线 W 恒 0）
    # ---- math_2d 的世界参数（环境设计，不是机制参数）----
    k_act: float = 0.02            # 每步动作推力
    tol: float = 0.08              # 成功判据（切比雪夫距离）
    amp_d: float = 0.01            # 外源扰动幅度（只有动作的一半）
    period_d: float = 40.0         # 扰动周期
    w_noise: float = 0.002         # 世界噪声
    x0_lo: float = -0.6            # 初态采样区间
    x0_hi: float = 0.6
    g_lo: float = -0.8             # 目标采样区间
    g_hi: float = 0.8
    # ---- oscillator 世界（grow_world）：两个漏积分变量 + 周期驱动 ----
    period_d_slow: float = 500.0
    tau_w_f: float = 50.0          # 快世界变量的漏时间（动作效果持续 ~50 步）
    tau_w_s: float = 500.0
    amp_d_slow: float = 0.10
    k_phase: float = 0.05          # 动作对世界变量的推动


# ---------------------------------------------------------------- 任务

@dataclass(frozen=True)
class TaskSpec:
    """任务。`kind` 决定状态怎么来。

    discrete: 固定的（初态, 目标）对循环 ---- 每个状态重复 n_ep/len(cases) 次
    sampled : 初态与目标随机采样 ---- 样本几乎不重复
    recall  : 目标不可观测，要求回到上一 episode 的终点（跟 episode 记忆）
    """
    kind: str = "discrete"
    cases: Tuple[Tuple[int, int], ...] = ((20, 60), (40, 70), (60, 20), (70, 30))
    obs_lo: int = 10
    obs_hi: int = 90
    t_ep: int = 240
    t_act: int = 5              # 动作窗口（必须满足 t_ep/t_act * step >= 最大差距）
    t_gap: int = 200            # episode 间隙（逼退隐状态）
    k_ep_task: int = 5          # sampled/recall：目标每 K 个 episode 换一次（跟 episode 稳定）
    k_ep_task: int = 5          # sampled/recall：目标每 K 个 episode 换一次（跟 episode 稳定）

    goal_mode: str = "random"       # random / fixed / two_phase（两阶段任务，grow_goal 的 T_*）
    fixed_goal: Tuple[float, float] = (0.5, -0.5)
    phase: str = "seq"              # two_phase：seq（先 A 后 B）/ fresh（直接 B）

    # ---- talk 流程（grow_talk）：三段式时间轴（编码 → 保持 → 提问/回答）----
    #
    # 两个时刻必须分开（2026-10-24 修正）：旧脚本的 `--delay=` 只改「何时允许回答」，
    # 不改「何时提问」（DELAY 是独立常数）——名字与实际含义不符，导致 `--delay=20`
    # 变成「回答窗早于提问」（不跨保持期）。现在它们是两个独立、名字准确的参数。
    #
    #   t=0      burst        query_at           t_max      t_present
    #   |-- 编码 --|-- 保持 -----|-- 提问/回答 ------|-- 尾巴 ----|
    #              ^ answer_from 可以早于 query_at（= 不需要跨保持期，旧 75% 就是这种）
    t_present: int = 500        # 一次呈现的总长（必须 > t_max）
    burst: int = 20             # 内容脉冲窗长
    nburst: int = 2             # 内容窗内几次同步爆发
    query_at: int = 120         # **何时提问**（提问脉冲的时刻）
    qburst: int = 10            # 提问脉冲窗长
    answer_from: int = 120      # **何时允许回答**（回答窗起点；无抢答窗 = 与 query_at 相等）
    t_max: int = 200            # 回答截止（过 = 沉默）
    cool: int = 30              # 说完后的冷却
    in_spike: float = 3.0       # 输入脉冲幅度（必须 > v_th，见 §16.10 缺陷 3）
    in_lo: float = 0.001        # 背景噪声率

    # ---- 派生量（不参与指纹，避免同一件事两种写法）----
    @property
    def hold_steps(self):
        """保持期长度 = 提问时刻 − 编码窗尾。这是真正要扫的记忆跨度。"""
        return self.query_at - self.burst

    @property
    def early_steps(self):
        """抢答窗长度 = 回答起点 − 编码窗尾。> 0 才可能出现「抢答」。"""
        return self.answer_from - self.burst

    @property
    def answer_before_query(self):
        """回答窗是否早于提问（= 不需要跨保持期；旧脚本 `--delay=20` 的形态）。"""
        return self.answer_from < self.query_at

    # ---- homeo 流程（grow_homeo）：周期性 cue 编码 + 背景输入 ----
    n_cue: int = 4              # cue 个数（输入区分块）
    enc_steps: int = 20         # 编码步数（cue 在第 0 与 enc_steps-2 拍注入）
    enc_gap: int = 150          # 两次 cue 的间隔
    cue_after: int = 0          # 首次 cue 的时刻（0 = 用 steps//2）
    bg_rate: float = 0.015      # 背景脉冲概率（只给 N_IN 之后的神经元）

    # ---- cond 流程（grow_cond）：变长呈现 + VI 课程（保持期从 Gamma 采样）----
    var_delay: bool = False
    d_mean_start: float = 40.0   # 保持期均值从 40 线性涨到 320
    d_mean_end: float = 320.0
    d_cv: float = 0.5            # 变异系数（固定 → 方差随均值一起涨）
    d_min: int = 30              # 最短（= 保持 10 步）
    d_max: int = 520             # 最长（= 保持 500 步）
    answer_win: int = 150        # 回答窗（从提问开始算）

    # ---- teach 流程（grow_teach）：固定长度 trial + 持续速率编码 + 末窗 argmax ----
    trial_steps: int = 40        # 一个 trial 的步数（> t_ref，让不应期不主导）
    read_win: int = 20           # 读出窗口（trial 最后 N 步做 argmax 计数）
    in_hi: float = 0.030         # 目标类对应的输入区放电率
    # 注：in_lo 复用上面的 0.001（非目标区）

    def check_feasible(self, steps_per_action: int) -> Tuple[bool, str]:
        """**前置检查**：最多能走多远 vs 目标差距。

        2026-10-24 的教训：t_ep=120 / t_act=20 = 6 个动作，而差距 40 -> 任务不可完成，
        而我用 5 轮「机制修补」去解释一个算术问题。任何实验先过这一关。
        """
        n_act = self.t_ep // self.t_act
        reach = n_act * steps_per_action
        if self.kind == "discrete":
            gap = max(abs(g - s) for s, g in self.cases)
        else:
            gap = self.obs_hi - self.obs_lo
        if reach < gap * 1.2:
            return False, "最多移动 %d < 差距 %d × 1.2（需要 %d）" % (reach, gap, int(gap * 1.2))
        return True, "最多移动 %d >= 差距 %d × 1.2" % (reach, gap)


# ---------------------------------------------------------------- 可塑性

@dataclass(frozen=True)
class Plasticity:
    """**三层独立开关**。每层单独声明，不能靠名字猜。

    2026-10-24 的教训：`frozen` 只关了权重+拓扑，bias 还在学
    -> 它实际是「只学 bias」，而真正的「什么都不学」根本不存在
    -> 「frozen 更好」被读成「不学更好」，结论完全相反。
    """
    weight: bool = True         # 连接权重（Δw = η·m·trace[pre]·(trace[post]−θ)）
    topology: bool = True       # 新建 / 淘汰边
    bias: bool = True           # 输出神经元的节点偏置（§14.6 作用点二）
    eta_w: float = 0.02
    eta_bias: float = 0.05
    bias_min: float = -0.3
    bias_max: float = 2.0
    tau_trace: float = 100.0    # 迹必须跨过「动作 → 反馈」的延迟
    trace_decay: float = 0.0    # >0 时**直接指定**每步衰减（不按 exp(-1/τ）——grow_world 用 0.95
    tau_theta: float = 200.0    # BCM 滑动阈值
    isolate: bool = True        # 只写入本 episode 活跃过的边（§9.2 隔离）
    isolate_thresh: float = 0.3
    # ---- 对齐开关（只为复现旧实现的行为，不是设计选择）----
    # 2026-10-24：旧 grow_agent 无条件每 episode 更一次拓扑（m 只累加统计）；
    # 框架改成「m_ep > 0 才更」（= 只在成功时）。两者结构更新频率差一个量级。
    topo_gate: bool = True      # True = 现行（m_ep>0）；False = 旧（无条件）
    topo_skip_first: bool = False   # True = 旧：ep0 结束不更新拓扑
    # 可塑性触发时机：action（每次动作后，grow_agent）/ step（每 t_s 步，grow_goal）
    trigger: str = "action"
    t_s: int = 20               # step 触发的周期
    topo_thresh: float = 0.3    # 选「活跃神经元」的迹阈值（grow_agent 0.3 / grow_goal 0.5）
    # ---- 结构可塑性的形态（2026-10-24：homeo 需要另一套）----
    #   budget : 强化 + 随机新建 + 超预算按权重淘汰（agent / goal / world / talk）
    #   homeo  : **按每个神经元的放电率偏差双向调 E-E 入边数**（grow_homeo / PLOS CB 2022）
    # True = 只更新「突触后是输出神经元」的边（grow_teach：内部边不动，
    #        否则背景强化会把接口边的相对强度汴没——旧实现实测输出边被优先淘汰）
    only_out_edges: bool = False
    struct_kind: str = "budget"
    r_target: float = 0.008     # homeo：目标放电率（0.8%）
    tol: float = 0.20           # homeo：±20% 内不动
    p_del: float = 0.30         # homeo：超标神经元每周期删掉入边的比例上限
    p_new: float = 0.30         # homeo：不达标神经元每周期新建入边的比例上限
    # ⚠️ homeo 的旧实现每个 chunk 都从**零状态**开始（grow 调 simulate 时不传 start），
    # 只保留 rate_ema 与结构——等于每 50 步把 v/s/ref/θ/x_rec 清空。
    # True = 旧行为（对齐 grow_homeo / 复现 §16.11 用）；False = 状态连续（设计要的）。
    reset_state_per_chunk: bool = False


# ---------------------------------------------------------------- 信号 m

@dataclass(frozen=True)
class SignalSpec:
    """`m` 的来源与形态——这是「学习方式」本身，必须显式。

    source:
        none       m ≡ 0（真基线）
        random     m 随机（检验「有 m 就行吗」）
        action     每动作：朝目标 +1 / 背目标 −1（延迟 0，§5.4 环境事实）
        episode    每 episode：成功 +1 / 失败 0（延迟 = episode 长）
    shape:
        scalar        m 是 1 个标量，广播到所有边
        per_neuron    m 是 N 维（接口边用逐输出值，其余用全局）
    glob:
        raw         直接 r
        rpe         r − 基线（可正可负）
    gate:
        三闸（§7.1）：admit × correct × consistent；关掉就是 1
    """
    source: str = "action"
    shape: str = "per_neuron"
    glob: str = "rpe"
    gate: bool = True
    bridge: float = 0.0          # 桥接强度（0 = 不用）
    silence_hint: float = 0.2    # 沉默时给「朝目标侧」的偏置增量（0 = 不提示）
    # ---- 对齐开关（只为复现旧实现的行为）----
    silence_update_w: bool = False   # 沉默时是否也更新权重（旧 grow_agent 做：upd_weights(mv, 0.0)）
    steady_gate_ep: bool = False     # episode 级 m 是否乘稳态门（旧做：m_gate × steady_gate(ema_rate)）
    # value 源（grow_goal 形态）：只在「刚成功」的 window 步内 m=1，否则 0
    value_window: int = 20
    # agency 源（grow_world 形态）：m = 稳态门 × (base + (1-base)×动作方差代理)
    agency_scale: float = 0.2
    base: float = 0.2
    # ---- teacher_word 源（grow_talk 形态）：按神经元分配的 (tgt - got) 方向 ----
    tau_m: float = 100.0        # m 的衰减（= 巩固窗口，不是 trial 边界）
    k_update: int = 5           # 每 K 步写一次（不是每 trial）
    m_up: float = 1.0           # 正信号幅度
    m_dn: float = -0.5          # 负信号幅度（不对称：削弱只有一半，防早期死亡螺旋）
    warmup: int = 3000          # 前 N 步不开可塑性（让 RPE 基线先收敛）
    bl_tau: float = 40.0        # RPE 基线的时间常数
    # ---- reward_only 源（grow_cond）：**只奖励正确，不惩罚错误** ----
    # False = 错/抢答/沉默都给 0（不产生更新，靠「对的被强化」相对占优）
    # True  = 错/抢答/沉默给 -1（grow_cond 的 --punish 对照）
    punish: bool = False


# ---------------------------------------------------------------- 读数

@dataclass(frozen=True)
class ReadoutSpec:
    """输出怎么变成动作。"""
    kind: str = "first_spike"   # first_spike（谁先放选谁；都不放=不说） / argmax / random
    n_out: int = 4
    out_idx: Tuple[int, ...] = (500, 501, 502, 503)
    allow_noop: bool = True      # 「不说」是不是一个合法动作
    ensemble: str = "action"     # argmax 是 action；first_spike 不必
    # 同一 tick 多个输出放电时选谁：first = 索引最小（现行）；argmax_v = 膜电位最高（旧）
    tie_break: str = "first"
    # sign_2d（grow_goal 形态）：每个输出维度算 sign(r_out - base)，动作是 ±1 向量
    a_tau: float = 20.0          # 输出放电率 EMA
    base_tau: float = 100.0      # 动作基线的慢 EMA（防动作饱和）
    # tanh_2d（grow_world 形态）：a = tanh((r_out - r_base)/r_scale)
    r_base: float = 0.02
    r_scale: float = 0.03
    probe: bool = False          # 是否记录状态快照做线性探针（grow_goal 的 A_probe）
    # ---- 功能测量套件（grow_func）：**不改动力学**，只加测量 ----
    func_measure: bool = False   # 在线 LMS 预测误差 + 离线最优读出 + 动作增益 R² + 因果方向
    pred_lr: float = 0.02        # 归一化 LMS 学习率（读出层 Wp: s[t] -> o[t+1]）
    oseg: int = 100              # 输出放电分段的步长（前史因果对比用）
    perturb: float = 0.0         # 初始膜电位扰动幅度（前史因果诊断；0 = 不扰动）
    # 抢答是否终止本呈现。talk 终止（抢答=答错）；cond **不终止**（只记录）。
    early_terminate: bool = True


# ---------------------------------------------------------------- 完整配置

@dataclass(frozen=True)
class Config:
    name: str = "unnamed"
    note: str = ""
    net: NetSpec = field(default_factory=NetSpec)
    env: EnvSpec = field(default_factory=EnvSpec)
    task: TaskSpec = field(default_factory=TaskSpec)
    plast: Plasticity = field(default_factory=Plasticity)
    signal: SignalSpec = field(default_factory=SignalSpec)
    readout: ReadoutSpec = field(default_factory=ReadoutSpec)
    n_ep: int = 200
    seed: int = 0
    # 输入随机缓冲行数。旧 grow_agent = 65536，框架原为 16384。
    # 它影响回绕点：88000 步的跑法下两者分叉，所以必须显式（它对行为是真的有影响）。
    u_buf_rows: int = 16384
    # 读出寄存器（win_i）是否在每个 episode 开头重置。
    # 旧 grow_agent：不重置（只在 run_one 开头初始化一次）——跑满的 ep 末尾会带进下一个 ep。
    # 2026-10-24 实测：这一条单独就是 10pp（是 9 处差异里第二大的一项）。
    reset_readout_per_ep: bool = False
    # 实验形态：流程不同，但**本体与可塑性原子操作共用**（core.py）。
    #   agent  = 离散动作 + 文件环境 + 每动作更新（grow_agent）
    #   goal   = 2D 连续动作 + 数学世界 + 每 t_s 步更新（grow_goal）
    #   world  = 2D 振荡器世界 + 单条连续时间轴 + 每 t_s 步（grow_world）
    flow: str = "agent"
    steps: int = 0          # >0 时用「单条时间轴」而不是「n_ep × t_ep」（world 流程）

    # ---- 序列化 / 指纹 ----
    def to_dict(self) -> dict:
        d = asdict(self)
        # tuple -> list，保证 json 可写
        def conv(x):
            if isinstance(x, tuple):
                return [conv(i) for i in x]
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x
        return conv(d)

    def fingerprint(self) -> str:
        """配置指纹：**只含影响行为的字段**（不含 name / note / seed / n_ep）。

        用途：查「这个配置跑过吗」。名字可以变，指纹不变。
        """
        d = self.to_dict()
        for k in ("name", "note", "seed", "n_ep"):
            d.pop(k, None)
        s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

    def validate(self) -> list:
        """硬检查。任何一条不过就不该跑。"""
        errs = []
        if self.flow == "talk":
            if self.readout.kind != "first_spike":
                errs.append("talk 流程要求 readout.kind == first_spike，实际 %s" % self.readout.kind)
            if self.signal.source not in ("teacher_word", "random", "constant", "none"):
                errs.append("talk 流程要求 signal.source ∈ {teacher_word, random, constant, none}，实际 %s"
                            % self.signal.source)
            if self.task.t_max >= self.task.t_present:
                errs.append("t_max(%d) 必须 < t_present(%d)" % (self.task.t_max, self.task.t_present))
            if self.task.query_at < self.task.burst:
                errs.append("query_at(%d) 必须 >= burst(%d)（提问不能在编码期内）" % (
                    self.task.query_at, self.task.burst))
            if self.task.answer_from < self.task.burst:
                errs.append("answer_from(%d) 必须 >= burst(%d)" % (self.task.answer_from, self.task.burst))
            if self.readout.n_out * self.net.blk > self.net.n_in:
                errs.append("n_out(%d) × blk(%d) 超过输入区 n_in(%d)" % (
                    self.readout.n_out, self.net.blk, self.net.n_in))
            return errs
        if self.flow == "teach":
            if self.readout.kind != "argmax":
                errs.append("teach 流程要求 readout.kind == argmax，实际 %s" % self.readout.kind)
            if self.signal.source not in ("teacher_word", "random", "constant", "none"):
                errs.append("teach 流程要求 signal.source ∈ {teacher_word, random, constant, none}，实际 %s"
                            % self.signal.source)
            if self.task.read_win <= 0 or self.task.read_win > self.task.trial_steps:
                errs.append("read_win(%d) 必须在 1..trial_steps(%d)" % (
                    self.task.read_win, self.task.trial_steps))
            if self.task.in_hi <= 0:
                errs.append("in_hi 必须 > 0")
            if self.readout.n_out * (self.net.n_in // self.readout.n_out) > self.net.n_in:
                errs.append("n_out(%d) 超过输入区块数" % self.readout.n_out)
            return errs
        if self.flow == "cond":
            if self.readout.kind != "first_spike":
                errs.append("cond 流程要求 readout.kind == first_spike，实际 %s" % self.readout.kind)
            if self.signal.source not in ("reward_only", "none"):
                errs.append("cond 流程要求 signal.source ∈ {reward_only, none}，实际 %s" % self.signal.source)
            if not self.task.var_delay:
                errs.append("cond 流程要求 task.var_delay == True")
            if self.task.answer_win <= 0:
                errs.append("cond 流程要求 answer_win > 0")
            if self.task.d_min < self.task.burst:
                errs.append("d_min(%d) 必须 >= burst(%d)" % (self.task.d_min, self.task.burst))
            if self.readout.n_out * self.net.blk > self.net.n_in:
                errs.append("n_out(%d) × blk(%d) 超过输入区 n_in(%d)" % (
                    self.readout.n_out, self.net.blk, self.net.n_in))
            return errs
        if self.flow == "homeo":
            if self.signal.source not in ("homeo", "none"):
                errs.append("homeo 流程要求 signal.source ∈ {homeo, none}，实际 %s" % self.signal.source)
            if self.steps <= 0:
                errs.append("homeo 流程必须显式给 steps")
            if self.plast.struct_kind != "homeo":
                errs.append("homeo 流程要求 plast.struct_kind == 'homeo'，实际 %s"
                            % self.plast.struct_kind)
            if self.task.enc_steps >= self.task.enc_gap:
                errs.append("enc_steps(%d) 必须 < enc_gap(%d)" % (
                    self.task.enc_steps, self.task.enc_gap))
            if self.task.n_cue * (self.net.n_in // self.task.n_cue) > self.net.n_in:
                errs.append("n_cue(%d) 超过输入区 n_in(%d)" % (
                    self.task.n_cue, self.net.n_in))
            return errs
        if self.flow == "world":
            # 单条时间轴，无任务、无成败 -> 没有「可达性」判据
            if self.signal.source not in ("agency", "noise"):
                errs.append("world 流程要求 signal.source ∈ {agency, noise}，实际 %s" % self.signal.source)
            if self.readout.kind not in ("tanh_2d", "randact"):
                errs.append("world 流程要求 readout.kind ∈ {tanh_2d, randact}，实际 %s" % self.readout.kind)
            if self.steps <= 0:
                errs.append("world 流程必须显式给 steps")
            return errs
        if self.readout.kind == "random":
            # random 不建网、不学——只用作「不学网络」的下界基线
            step = max(abs(a) for a in self.env.actions)
            ok, msg = self.task.check_feasible(step)
            if not ok:
                errs.append("任务不可完成：" + msg)
            return errs
        if self.env.kind == "math_2d":
            n_win = self.task.t_ep // self.task.t_act
            reach = n_win * self.env.k_act * self.task.t_act
            gap = max(abs(self.env.g_hi - self.env.x0_lo),
                      abs(self.env.g_lo - self.env.x0_hi))
            if reach < gap * 1.2:
                errs.append("2D 世界不可完成：可达 %.2f < 差距 %.2f × 1.2" % (reach, gap))
            return errs
        step = max(abs(a) for a in self.env.actions)
        ok, msg = self.task.check_feasible(step)
        if not ok:
            errs.append("任务不可完成：" + msg)
        if self.signal.source not in ("none", "random", "action", "episode", "value"):
            errs.append("未知 signal.source: %s" % self.signal.source)
        if self.signal.shape not in ("scalar", "per_neuron"):
            errs.append("未知 signal.shape: %s" % self.signal.shape)
        if self.readout.n_out != len(self.readout.out_idx):
            errs.append("n_out 与 out_idx 长度不一致")
        if len(self.env.actions) < self.readout.n_out and not self.readout.allow_noop:
            errs.append("动作数少于输出数且不允许 noop -> 有些输出永远无法执行")
        return errs

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
