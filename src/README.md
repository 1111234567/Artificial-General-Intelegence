# 共享 LIF 实验框架

`src/` 集中已有 LIF/SNN 实验的配置、网络、执行与结果登记，减少同一逻辑在历史脚本间漂移。它支持历史复现和后续可比较实验；架构与机制的适用性仍由实验验证。项目状态见 [主文档](../agi.md)，当前证据与已知问题见 [实验文档](../docs/experiments.md)。

## 模块与入口

给新任务选模型时，从[神经元与训练模块](neurons/README.md)按类型导入。该入口同时收录 GRU、速率网络、线性感觉图、快突触更新和有适用条件的种群后端。下面的表列出早期共享 LIF 框架的配置与执行工具。

| 文件 | 职责 |
|---|---|
| [config.py](config.py) | 不可变 dataclass 配置、序列化、配置指纹、已实现的前置检查 |
| [core.py](core.py) | 框架共享的稀疏 LIF `Network`、状态推进与可塑性操作 |
| [env.py](env.py) | `agent` 使用的本机文件环境 |
| [signal.py](signal.py) | `agent` 的动作级、沉默与任务级反馈；其他 flow 的部分反馈在 runner 内计算 |
| [runner.py](runner.py) | 按 flow 执行配置，返回结构化结果；数学环境与各流程专用更新也在这里 |
| [store.py](store.py) | JSONL 结果登记与查询，默认路径为仓库的 `data/results.jsonl` |
| [experiments.py](experiments.py) | 共享框架的实验配置与历史复现配置，现有列表以 `list` 输出为准 |
| [scan.py](scan.py) | `homeo` / `calib` 参数网格与摘要 |
| [../scripts/run.py](../scripts/run.py) | 主命令行入口 |
| [../test/run.py](../test/run.py) | 兼容入口 |

独立原型在 [experiments/prototypes/](../experiments/prototypes/)，历史脚本在 [experiments/legacy/](../experiments/legacy/)，诊断、基线与退役脚本分别在 [diagnostics/](../experiments/diagnostics/)、[baselines/](../experiments/baselines/)、[retired/](../experiments/retired/)。已有 flow 的变体优先在 `experiments.py` 声明；独立研究装置按其用途放置，并说明与框架的关系。

## 命令

从仓库根目录运行，前缀统一为 `python scripts/run.py`。

| 命令 | 用途与限制 |
|---|---|
| `list` | 列出配置、指纹和部分开关；详细配置用 `show` |
| `show <name>` | 展开包含 dataclass 默认值的完整配置 JSON |
| `check [name]` | 检查已实现的约束；部分任务有可达性检查，其他 flow 检查自身条件 |
| `run <name> [nseed] [nproc]` | 校验后运行，默认 8 个种子、8 个进程，结果追加到 JSONL |
| `results` | 读取结果登记并输出摘要 |
| `gaps` | 将配置与 `BASE` 比较，按指纹提示是否有记录；不检查对照是否有效或种子是否齐全 |
| `align` | 按指纹汇总已有结果的成功率跨度；未按种子、运行长度配对，只是粗略诊断 |
| `scan [homeo\|calib] [nseed] [nproc]` | 扫描历史参数网格，输出趋势，不登记结果 |
| `bench` | 模拟器实现成本比较，使用单独的最简实现 |

```bash
python scripts/run.py show base
python scripts/run.py check base
python scripts/run.py run base 8 8
python scripts/run.py results
```

文件环境会写入 `data/`，`run` 会追加结果。`check` 通过仅表示未触发已有检查，反馈通路、任务设计、更新开关和测量仍需审核。当前 `align` 使用成功率跨度阈值，不能检验逐步确定性或替代历史移植对齐。

## 配置默认与结果指纹

`NetSpec`、`TaskSpec`、`Plasticity`、`SignalSpec`、`ReadoutSpec` 与 `Config` 都有默认值；`BASE` 及其他配置会继承其中一部分。`show` / `to_dict()` 可以展开实际取值。新增或修改配置时，应核对继承值和 runner 中真正使用的分支。

例如，`NetSpec` 默认 N=1000、兴奋比例 0.8、`tau_v=20`、`t_ref=20`、`w_e=0.50`、抑制权重比 5.0。这些来自历史装置与标定；约 1.99% 的放电率也是对应运行条件下的测量值。它们应随具体任务和动力学要求重新评估。时间常数逐单元变化也可以作为独立假说比较。

`fingerprint()` 对序列化配置去掉 `name`、`note`、`seed`、`n_ep` 后计算 12 位 SHA-1 摘要。因此：

- 同一指纹可以覆盖不同种子和不同 episode 数；`steps` 等保留字段仍参与指纹。
- 指纹不含代码版本、依赖版本、环境内容或运行硬件，不能单独标识一次可复现实验。
- 相同指纹表示参与计算的配置字段相同；不同指纹也可能仅差在该 flow 未使用的字段。行为是否改变需要检查实际路径。

比较结果时，应同时保留完整配置、代码版本、种子、运行长度、依赖、环境条件和指标定义。`gaps` 中“已跑”仅表示该指纹已有记录。

## 八类流程

这些流程共用框架网络，但时间轴、反馈、状态重置与更新公式各有历史差异。`readout.kind=random` 会直接走无网络基线。

| `Config.flow` | 时间与输入 | 读出或动作 | 主要更新与边界 |
|---|---|---|---|
| `agent` | episode 与间隙，文件状态和目标编码 | 离散动作，通常 first-spike | 动作、沉默与任务末各有分支；读出寄存器重置受配置控制 |
| `goal` | episode，2D 数学任务 | 两维符号动作 | 每 `t_s` 步的流程专用反馈与更新 |
| `world` | 单条时间轴，2D 振荡器 | 两维连续 `tanh` 动作 | 每 `t_s` 步更新；`func_*` 配置在此增加功能测量 |
| `talk` | 编码脉冲、保持、提问、回答 | 四分类 first-spike | 每 `k_update` 步；历史接口偏移与修正配置并存 |
| `homeo` | 背景脉冲与周期 cue，按 chunk 推进 | 不读动作 | 按放电率偏差调整 E-E 结构；是否每 chunk 清状态由配置决定 |
| `cond` | burst、Gamma 采样保持期、回答窗 | 四分类，抢答不终止呈现 | 每 `k_update` 步；奖励与惩罚配置决定反馈 |
| `teach` | 持续速率输入，固定 trial 与末读出窗 | 四分类放电计数 argmax | 每 trial 清 trace，更新接口边，trial 末处理结构 |
| `hebb` | 先生长结构，再 warm、encode、observe | 不读动作 | 结构冻结，Hebbian 慢变量 `A` 在 observe 期冻结 |

`agent` / `goal` 使用对称更新及 BCM，`world` 使用非对称更新；`talk` / `cond` / `teach` 保留历史的局部归一化算式，`hebb` 使用 `w_eff = w_struct + A`。比较 flow 时需要把这些差异列入实验条件。

## 已知对照问题

以下问题保留在当前执行行为中，本轮整理没有修改科学行为：

| 配置或工具 | 实际限制 | 对结果的影响 |
|---|---|---|
| `sig_episode` | `Signal.on_action()` 对 `source=episode` 清空 `_last`，任务末得到的权重反馈仍为空；沉默时仍根据目标给向量提示并更新 bias | 不能作为“只在任务末学习”的延迟反馈对照 |
| `sig_scalar` | 动作反馈去掉向量部分，`on_silence()` 仍产生目标相关的向量提示 | 不能作为纯标量反馈对照 |
| `ho_frozen` | 相比 `ho_grow` 同时关闭拓扑更新，并让 `reset_state_per_chunk` 回到默认 `False` | 结构与状态连续性同时变化，不能作结构学习的单变量消融 |
| 配置名称与 `note` | 保存了历史意图，可能与真正执行的更新路径有差异 | 名义上的 `frozen`、`scalar`、`episode` 等标签需要逐项验证 |

后续修复需要保留历史配置与新实验的版本边界，验证反馈到达位置、更新次数和状态边界，再重跑相应对照。

## 历史复现边界

历史记录包含 9 个旧脚本、201 项行为逐位对齐，明细见 [2026-10-25 实验快照](../docs/archived/experiments-2026-10-25.md)。[align_layers.py](../scripts/align_layers.py) 仍用于历史移植对齐；它验证指定配置、种子与长度下的行为一致性。

行为一致可以支持“移植保留了原行为”，也会保留原实现中的缺陷。它不能证明任务可完成、对照有效、反馈无泄漏、指标能测到目标能力，或结果能够泛化。运行历史代码之前应先读 [当前实验文档](../docs/experiments.md) 与 [本轮评审](../docs/review-2026-10-26.md)。

[scripts/historical/](../scripts/historical/) 保留旧三卷结构与拆分核验工具；现行文档使用 `python scripts/check_docs.py` 检查链接、路径与编码。文档检查和代码行为检查各自只覆盖其明确实现的范围。
