# 既有循环结构审计与非线性循环场扩展

本目录接续 [credit](../credit/README.md)，保留旧模型、训练入口和存档。**已完成 540 条既有模型诊断、9 组继续学习和 75 条最终评估；两个严格汇总通过。** 交互组新条件 78.13%，同参数平方组 65.84%，原模型 76.49%；交互组旧条件降到 48.05%，更远等待仍仅 2.92%。完整定义、分种子差异与解释边界见[实验记录](../../docs/loops-experiment-2026-10-28.md)。

## 旧模型已经有回路

原场为 `dh/dt = -h + W tanh(h)`，`W[i,j]` 表示神经元 j 对 i 的作用。稠密 W 的非零支持已有自连接、双向连接和经过多个不同神经元的有向回路。二元边不等于只有二节点回路。

[structure.py](structure.py)分开记录精确非零图、描述性权重阈值图、强连通分量、二/三节点简单回路及保证存在的简单回路长度。`tr(M^k)` 则加总带起点的加权闭合游走，允许重复节点和重复短回路；它不是长度 k 的简单回路数，也不是记忆保留 k 个 tick 的证据。

状态诊断使用 `D=diag(1-tanh(h)^2)`、`R=WD`、`K=offdiag(R)`、`J=-I+R`。这些是保存时刻的冻结线性化，没有推进时间；R/K 的幂不能代替实际 Heun 切向传播。

## 现有模型的冻结诊断

[审计协议](../../docs/evidence/loops-2026-10-28/audit-protocol-v1.json)固定种子 0、1、2，各取 A、`bp64`、`bp_replay` 终点，共 9 个已有检查点。mask 种子固定为 11、29、47，不按结果重抽。每个分叉在 CPU 继续 1,024 tick，`dt=.25`，条件为 `iid`、`long_sparse`、`longer`。

| 参数副本 | 操作 |
|---|---|
| `none` | 原 W |
| `diagonal_only` | 仅保留自连接 |
| `dag_cut` | 按固定随机节点次序保留单向非对角边，去掉不同节点间的回路 |
| `random_cut` | 每个接收神经元保留与 DAG 相同数量的非对角边，随机选来源 |
| `dag_norm_matched` / `random_norm_matched` | 每行缩小到共同较小的非对角 L2 范数，不放大权重 |
| `dag_quarter` / `random_quarter` | `0.75 W + 0.25 W_matched` |

稠密图的 DAG 端点删掉一半非对角连接；匹配行范数后，出度、谱、带符号输入和状态分布仍可能不同。随机图也可能无环，协议没有为强制有环而重抽。quarter 减轻扰动，但保留原有回路，插值后的行范数未必匹配。端点低分或落到地板，不能单独归因于去环。

第一个 mask 对每个来源运行全部 8 个副本，后两个各运行 6 个；合计 27 个任务、540 个分叉。每个来源另保存前 16 个 actor 在原边界的状态诊断，不把状态样本当作训练重复。

统计先在训练种子内平均三个固定 mask，再在三个训练种子间算均值和样本标准差。`none`、`diagonal_only` 每种子只运行一次；配对差异复用相同 `none`。任何 mask 分母未定义时，该种子的 mask 均值也未定义。

## 两种等参数的非线性扩展

[model.py](model.py)只改变自主向量场。Heun 积分后仍注入一次编码输入，再由原动作头读出；持续状态仍为 128 维。

| 分支 | 自主场，`r=tanh(h)` | 新增参数 | 总参数 | 每次场计算的稠密矩阵乘法 |
|---|---|---:|---:|---:|
| `baseline` | `-h + W r` | 0 | 34,348 | 1 |
| `source_square` | `-h + W r + V(r*r)` | 16,384 | 50,732 | 2 |
| `interaction` | `-h + W r + r*(V r)` | 16,384 | 50,732 | 2 |

source-square 的新增项取决于来源活动平方；interaction 的第 i 项为 `r_i sum_j V_ij r_j`，同时取决于接收者和来源。对 i≠j，其瞬时混合二阶偏导可为 `V_ij(1-r_i²)(1-r_j²)`；原场和 source-square 的对应偏导为零。原网络经过多步演化本来就能产生跨神经元依赖，这里比较的是瞬时场的表达形式。

V 在分叉时精确置零，不消耗随机数。旧 Parameter 对象、W/编码/读出、Adam、h、世界、时钟、合同、回执、旧经历库和 RNG 完整继承。V 加入相同超参数的新 Adam 参数组，初始没有历史矩。起点函数相同，但 V 的梯度参与联合裁剪，因此第一次更新起旧参数就可能变化不同。

两个扩展臂匹配新增参数与主要稠密矩阵运算，特征分布、梯度、条件性和训练总开销仍可不同。两者都会破坏原场的奇对称性。所有分支保留原有完整有向支持；本轮尚未实现结构出生、剪枝或学习路由。

## 继续学习、评估与恢复

[extension_study.py](extension_study.py)在既有审计完整通过后冻结有限三臂协议。三个种子各分三个分支，共 9 组，每组继续 65,536 tick。未来 NEW 等待范围为 4–8，最终合同截止至下次 NEW 的空档为 4–12；空档不是相邻 NEW 的完整间隔。已有合同、预定事件和当前状态不重建。

[train.py](train.py)复用旧 CreditTrainer 的完整 64-tick BPTT，每块一次联合裁剪和继承 Adam 更新，不读旧块学习，不改变信用规则。块边界只截断图，保留 h 数值。时间步从源存档继承；已核验全部三个种子的 `train_dts=[.25]`，不能从构造器默认值推断本轮循环多个时间步。

最终检查点统一在 CPU 冻结评估，`dt=.25`，五环境各继续 2,048 tick；`longer`、`long_sparse` 额外做最大积分子步 `.125` 的检查，两个扩展臂另做 V 置零反事实。计划 45 个自主分叉、18 个积分细化分叉、12 个 V 消融分叉。主分母只含分叉后新环境产生且完整窗口已成熟的最终有效请求；继承、旧合同和右删失分别报告。

启动与完成清单绑定协议、来源、分支；恢复接续同一分支的完整检查点。源码和存档字节哈希复核；CPU/GPU 语义比较只规范化设备元数据，保留参数、Adam、世界和 RNG。评估作用于副本。

| 文件 | 职责 |
|---|---|
| [prepare_audit.py](prepare_audit.py)、[audit_run.py](audit_run.py) | 冻结并执行既有模型诊断 |
| [structure.py](structure.py)、[interventions.py](interventions.py) | 只读图/状态统计与参数干预 |
| [report.py](report.py) | 完整性、分母、外生配对和种子级汇总 |
| [model.py](model.py)、[train.py](train.py) | 零初始化扩展、继承训练与恢复 |
| [extension_study.py](extension_study.py) | 三臂协议、来源绑定、训练和冻结评估 |
| [extension_report.py](extension_report.py) | 九个完整存档与 75 个评估分叉的严格汇总 |
| [plot_results.py](plot_results.py) | 从通过审计的汇总生成 PNG/SVG 科学图 |

实现校验覆盖零初始化函数与参数身份保持、实际 Heun 梯度、瞬时混合偏导、旧 BP64 多步数值一致性、CPU/CUDA 完整恢复。这些校验约束实现，任务性能仍由正式评估决定。

证据入口：[既有模型汇总](../../docs/evidence/loops-2026-10-28/audit-aggregate-v1/report.md)、[新增结构汇总](../../docs/evidence/loops-2026-10-28/extension-aggregate-v1/report.json)、[冻结训练协议](../../docs/evidence/loops-2026-10-28/extension-protocol-v1.json)、[完成清单](../../docs/evidence/loops-2026-10-28/completion.json)。原始状态保存在 `data/loops/`，没有用紧凑表格代替完整存档。

## 运行入口

在仓库根目录使用 `.venv\Scripts\python.exe`。正式协议和原始结果已各自保存；重做分析请写新输出目录，尤其不要覆盖已被下一阶段协议绑定的审计汇总。

```powershell
# 只读复核已有 540 条诊断；新目录避免改变原审计汇总
.venv\Scripts\python.exe -m experiments.loops.report --protocol docs/evidence/loops-2026-10-28/audit-protocol-v1.json --root data/loops/audit-v1 --output data/loops/recheck-audit

# 只读复核九个训练存档与 75 条评估
.venv\Scripts\python.exe -m experiments.loops.extension_report --protocol docs/evidence/loops-2026-10-28/extension-protocol-v1.json --root data/loops/extension-v1 --output data/loops/recheck-extension

# 若有限队列被打断，按原清单恢复；不要手工重建世界或挑选重新开始的种子
.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind batch --plan docs/evidence/loops-2026-10-28/extension-training-commands-v1.json --status data/loops/extension-v1/training-execution.json --workers 6
.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind evaluation --plan docs/evidence/loops-2026-10-28/extension-evaluation-commands-v1.json --status data/loops/extension-v1/evaluation-execution.json --training-status data/loops/extension-v1/training-execution.json --workers 4

.venv\Scripts\python.exe -m pytest -q experiments/loops
```

`prepare_audit` 和 `extension_study prepare` 用于生成新的有限协议，拒绝覆盖已有协议；它们不是每次恢复时要再跑的步骤。新科学问题或改变源码后应使用新版本和新目录，不修改旧协议使之接受新实现。
