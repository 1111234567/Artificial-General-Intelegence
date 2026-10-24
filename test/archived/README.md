# test/archived — 已退役的实验脚本

> 2026-10-24 从 `test/` 根移入。**不是删除**——代码在 git 历史里，这里保留的是「哪一版试过什么」的轨迹。
> 当前在用的脚本仍在 `test/` 根（见文末表）。

## 为什么退役

这些是**同一实验的中间版本**，或**目标代码已不存在的诊断**。它们的结论已写进 `docs/experiments.md` §16：

| 组 | 文件 | 退役理由 |
|---|---|---|
| 自生长 15 版的版本轨迹 | `grow_snn.py` ~ `grow_snn8.py`（8）、`grow_final.py`、`grow_v2.py` | 被 `grow_world.py` 取代；那 15 版的死网（rate 0.08%）已定位为五个实现缺陷（§16.6） |
| 动力学标定的前四版 | `calib_dynamics.py`、`calib2.py`、`calib3.py`、`calib4.py` | 逐次加机制的中间态；标定结果与判据见 §11.4.0，当前入口 `calib5.py` |
| 成本实测的前后版本 | `bench_lif.py`、`bench_lif3.py` | 结论见 §16.1 与 `baseline-2026-10-23-simulator.md`，当前入口 `bench_lif2.py` |
| 目标形态已消失的诊断 | `probe_ask.py` | 它从 `grow_talk.py` 里 `exec` 代码、诊断的是「三段式抢答」；`grow_talk` 改成 first-spike 后该形态已不存在 |

## 当前在 `test/` 根的脚本

> **2026-10-24 更新：** 新实验一律走框架（`test/run.py` + `src/agi/`）；
> 下表的 `grow_*` / `probe_*` 是**实验记录**，不再新增。

### 框架（新的入口）

| 文件 | 用途 |
|---|---|
| **`run.py`** | **唯一入口**：`list / show / check / run / results / gaps / align`（§16.17） |
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
