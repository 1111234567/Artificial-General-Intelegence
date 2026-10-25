# test/archived — 已退役的实验脚本

> 2026-10-24 从 `test/` 根移入（`grow_trace.py` 于 10-25 追加，共 18 个）。**不是删除**——代码在 git 历史里，这里保留的是「哪一版试过什么」的轨迹。
> 当前在用的脚本仍在 `test/` 根（见文末表）。

## 为什么退役

这些是**同一实验的中间版本**，或**目标代码已不存在的诊断**。它们的结论已写进 `docs/experiments.md` §16：

| 组 | 文件 | 退役理由 |
|---|---|---|
| 自生长 15 版的版本轨迹 | `grow_snn.py` ~ `grow_snn8.py`（8）、`grow_final.py`、`grow_v2.py` | 被 `grow_world.py` 取代；那 15 版的死网（rate 0.08%）已定位为五个实现缺陷（§16.6） |
| 动力学标定的前四版 | `calib_dynamics.py`、`calib2.py`、`calib3.py`、`calib4.py` | 逐次加机制的中间态；标定结果与判据见 §11.4.0，当前入口 `calib5.py` |
| 成本实测的前后版本 | `bench_lif.py`、`bench_lif3.py` | 结论见 §16.1 与 `baseline-2026-10-23-simulator.md`，当前入口 `bench_lif2.py` |
| 目标形态已消失的诊断 | `probe_ask.py` | 它从 `grow_talk.py` 里 `exec` 代码、诊断的是「三段式抢答」；`grow_talk` 改成 first-spike 后该形态已不存在 |
| **实验形态错了的那一版** | **`grow_trace.py`** | **独立 trial 形态**（每 trial 1 个刺激 + 状态清零）——那是 delayed match-to-sample 的独立分类，不是 conditioning；且噪声放在独立 16 通道（可被权重绕过）。被 **`grow_stream.py`**（连续事件流 + 噪声与刺激同通道）取代。三个自加机制（`clip=1.0` / 每 50 步缩放 U / silent loss）也在里面，已证会把 L≤40 从 100% 压到 50~84%。§16.28 |

## 当前在 `test/` 根的脚本

> **2026-10-24 更新：** 新实验一律走框架（`test/run.py` + `src/agi/`）；
> 下表的 `grow_*` / `probe_*` 是**实验记录**，不再新增。

### 框架（新的入口）

| 文件 | 用途 |
|---|---|
| **`run.py`** | **唯一入口**：`list / show / check / run / results / gaps / align / scan / bench`（§16.17 / §16.26） |
| **`grow_rnn.py`** | **阶段 0.5 步 1/2**：BP 训 2 层 Elman RNN（含 `--gradcheck` 反向校验 / `--random-goal` / `--blind=N` / `--renorm`），§16.27。**2026-10-25 加 `backward(state_mask=)` 参数（默认 None = 原行为，gradcheck 仍 1e-10）** |
| **`grow_stream.py`** | **连续事件流上的 trace conditioning**（§16.28）：一条序列内含多个「刺激 → 间隔 → 提醒」对；**噪声与刺激同一批通道**；孤立提醒须沉默；间隔课程（短间隔永久保留固定比例）。**⚠️ 记忆跨度那条线已撤**——读数不作记忆跨度引用，四条形态约定保留 |
| **`grow_snn_convert.py`** | 阶段 0.5 步 3 第一版（tanh 值上量化）：**§16.29 / §16.32 已撤**；只保留两条与时间切分无关的数学结论（量化误差 `0.5/T`；判据要写成 `‖W3‖·ε·√N_H ≳ margin`） |
| **`grow_loop.py`** | 阶段 0.5 步 4（首尾相连 + population）：**§16.30 已撤**；只保留定性结论（环路会延长状态差异、population 只在发放随机时有用） |
| **`grow_lif.py`** | 自己造的 LIF 本体：**不再使用**（项目真正的本体是 `src/core.py`） |
| **`rnn2snn.py`** | RNN → `core.Network` 的结构映射（矩阵拆正/负部）：映射对，差在抑制比例标定 |
| **`probe_g.py`** | **可复用**：扫一个参数看「某个差异是真性质还是放大过头」——靠它撤回了「混沌」判定（§16.32） |
| `baseline_linear.py` | **非脉冲基线**：random / greedy / linear 三档，判「任务可学性 + 上界」（§16.16） |
| `CONSTANTS.md` | 常数台账（来源分类）+ 已删的机制 + 场次四 12 条核对表 |
| `net.py` | 旧共享参数（框架已收敛进 `src/agi/config.py`；仍被 `grow_agent` / `grow_goal` import） |

### 旧实验脚本（记录，不再新增）

| 文件 | 用途 |
|---|---|
| `grow_agent.py` | 智能体：本机真环境 + 三闸 + 隔离 + 四层反馈 + 四度量（§16.15） |
| `grow_goal.py` | 闭环三步 C/A/T：目标 + 成败 + 探针（§16.13） |
| `grow_world.py` | 闭环：输出神经元 → 动作 → 世界 → 输入（§16.6） |
| `grow_teach.py` | 阶段 1/2 v1（持续输入 + 固定窗口，§16.9） |
| `grow_talk.py` | 阶段 1/2 v2（脉冲 + 事件发射 + 连续状态，§16.10） |
| `grow_cond.py` | 阶段 1/2 v3（只奖励正确 + VI 课程 + 迹 τ=200）——已跑，形态不成立（§16.14） |
| `grow_func.py` | 功能测量（预测误差 / 动作增益 / 前史因果，§16.8） |
| `grow_homeo.py` | 按放电率的双向结构可塑性（PLOS CB 2022 形态，§16.11） |
| `grow_hebb.py` | Hebbian 权重可塑性（§16.12，首轮阴性） |
| `bench_lif2.py` | 模拟器成本基线（§16.1） |
| `calib5.py` | 动力学标定（§11.4.0） |
| `scan_base.py` | 参数扫描：找「无 E-E 时率落在目标下方」的工作点 |
| `probe_out.py` / `probe_in.py` | 读出能力 / 输入驱动的最小验证 |
| `probe_decay.py` / `probe_persist.py` / `probe_mem.py` | 无输入衰减 / 自我维持 / 自维持活动是否携带输入 |
| `probe_homeo.py` / `probe_sim_noise.py` | 稳态阈值扫描 / 测量面校准 |
| `probe_env.py` | 环境探测（python / numpy / torch / CUDA 版本与冒烟） |
| `baseline-2026-10-23-simulator.md` | 模拟器成本基线数据 |
