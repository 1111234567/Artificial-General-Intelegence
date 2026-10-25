# agi 框架 — 为什么要它

> 2026-10-24 建立。此前 `test/` 下有 **21 个手写脚本**，每个都实现一遍本体，
> 每个丢掉前一个的一些东西，且没有任何地方能回答：
> ① 哪些组合跑过？② 结果是什么？③ 对照组齐不齐？④ 哪个版本到底有什么？

## 结构

```
src/
  config.py       声明（一个实验 = 一个 Config，没有隐式默认）
  core.py         本体（LIF 网络）——**唯一实现**
  env.py          环境（本机真实文件）
  signal.py       m 的来源与形态（= 学习方式本身）
  runner.py       跑一个配置，返回一条结构化结果（八个流程见下）
  store.py        结果登记（JSONL，按指纹查）
  experiments.py  **唯一声明实验的地方**（71 个）
  scan.py         参数网格（`run.py scan` 用；只生成网格，不登记结果）

test/run.py       唯一入口（九个子命令）
scripts/align_layers.py  移植对齐工具（17 层）
```

## 九个命令

| 命令 | 回答什么 |
|---|---|
| `list` | 现在有哪些实验，每个开了什么（含指纹） |
| `show <name>` | 展开一份配置的**全部开关**（json） |
| `check [name]` | **前置检查**：任务可行吗（动作数×步长 vs 差距） |
| `run <name> [nseed] [nproc]` | 跑并登记 |
| `results` | 历史结果排序 |
| `gaps` | **对照差集**：基准与每个变体差哪个字段、哪些还没跑 |
| `align` | **对齐验证**：同一配置重复运行的跨度 |
| `scan [homeo|calib] [nseed] [nproc]` | 参数扫描（工具；不登记；用框架的 `_run_homeo`） |
| `bench` | 模拟器成本基准（约 30 s；**不用**框架的 `Network`——它测的就是实现形态本身） |

## 四个「不能再犯」对应的机制

| 今天的教训 | 框架里的对策 |
|---|---|
| 任务不可完成（6 个动作 vs 差距 40），5 轮补丁解释一个算术问题 | `TaskSpec.check_feasible()`（含 `math_2d` 分支）+ `run.py check` 硬拦 |
| `frozen` 名不副实（只关了权重和拓扑，bias 还在学）→ 结论反了 | `Plasticity` **三层独立开关**；`gaps` 自动列出差异字段 |
| 21 个脚本各实现一遍本体，改一个参数要改 10~25 处 | `core.py` 唯一实现；实验只能在 `experiments.py` 加 Config |
| **四个“不同 mode”实际是同一配置**（`base`/`no_gate`/`no_bridge`/`no_isolation` 都只学 bias）| **行为指纹验证**：`align_layers.py modes` 把结果逐位相同的 mode 分组；`fingerprint()` 去重 |

## 移植覆盖度（2026-10-24）

> **已与旧脚本对齐：9 个脚本 / 201 项，行为逐位一致**（`grow_agent` 9/9、`grow_goal` 6/6、`grow_world` 5/5、`grow_talk` 19/19、`grow_homeo` 12/12、`grow_cond` 22/22、`grow_teach` 11/11、`grow_func` 21/21、`grow_hebb` 96/96）。
> 验证方式与全套跑法见 `docs/experiments.md` §16.18–§16.26；工具 `scripts/align_layers.py`。
>
> **已无待移植项**：9 个 grow 脚本全部对齐；三个工具类已子命令化（`scan homeo` / `scan calib` / `bench`）；
> `probe_*` 约定不移植（一次性诊断，随用随写）；`baseline_linear.py` 不依赖 `Network`，不是移植对象。

## 八个流程（`Config.flow`）

本体（`core.py`）与可塑性原子操作八个流程共用；区别只在「什么时候调哪个原子操作」。

| flow | 时间结构 | 动作 | 环境 | 更新时机 |
|---|---|---|---|---|
| `agent` | episode（n_ep × t_ep + 间隙）| 离散索引（`first_spike`）| 本机真实文件 | 每次动作后 |
| `goal` | episode | 2 维独立符号（`sign_2d`）| 2D 数学世界 | 每 `t_s` 步 |
| `world` | **单条连续时间轴**（`steps`）| 2 维连续（`tanh_2d`）| 2D 振荡器 | 每 `t_s` 步 |
| `talk` | **呈现**（n_ep × t_present，三段式：编码脉冲 → 保持 → 提问 → 回答）| 4 个离散（`first_spike`，平手取 v 最大）| 无环境（程序生成输入）| 每 `k_update` 步 |
| `homeo` | **单条时间轴**（`steps`，内部按 `t_s` 分 chunk）| 无（不读动作）| 周期性 cue 编码 + 背景脉冲 | 每 `t_s` 步（按率偏差双向调 E-E 边）|
| `cond` | **变长呈现**（burst + Gamma 采样的保持期 + answer_win）| 4 个离散（抢答不终止）| 无环境（程序生成输入）| 每 `k_update` 步 |
| `teach` | **固定 trial**（n_ep × trial_steps）+ 末 `read_win` 步 argmax | 4 个离散（计数 argmax）| 无环境（持续速率编码）| 每 trial 一次（只更接口边）|
| `hebb` | **一次性三段**（warm → encode → observe，不是循环）| 无（不读动作）| 无环境（pulse / osc 两种输入）| observe 期 A 冻结；结构在 warm 之前由 `_run_homeo` 长好 |

**这些脚本的可塑性规则确实不同**（不是同一件事的不同写法）：`agent` 对称 + BCM；`goal` 对称 + BCM；
`world` **非对称 + 无 BCM**；`talk` / `cond` / `teach` 用旧算式（`el[nz]` 子集按 `max` 归一，不是全局 `abs`）——所以它们不经 `apply_weight`；`hebb` 用的是 Hebbian 慢变量（`w_eff = w_struct + A`，结构冻结）。硬合并成一套，任何一个方向都会移植失败。

## 约定

1. 新增实验 = 在 `experiments.py` 加一个 `Config`。**不在 test/ 下写新脚本**。
2. 每个 Config 的 `note` 必须写「它在哪些维度上与 BASE 不同」。
3. 任何新常数必须带来源标签（`[标定]/[约定]/[派生]/[需求]/[拍]`）。
4. 每次 `run` 自动登记；`gaps` 告诉你对照组缺哪些。
5. **移植旧脚本时，先跑 `align_layers.py` 的对应层，逐位相同才算完成**——「配置抄过去了」不算。
